"""Modality ablation: train an architecture N times, each time dropping one modality.

Tells you which sensors are essential — useful for wearable-paper reviewer
question "do you really need all 6 sensors?".

For each modality m in [emg, ecg, eda, temp, acc, ppg], drops ALL features
whose name starts with m_ and trains the chosen architecture on the remaining
features. Reuses configs/splits.csv from the LightGBM baseline.

Usage:
    python scripts/ablate_modalities.py --arch tcn --top-k 30 --leakage-safe
    python scripts/ablate_modalities.py --arch cnn_lstm  # uses all features

References:
- Saeb et al. 2017 — subject-wise CV
- Guyon & Elisseeff 2003 — feature ablation methodology
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.datasets import WindowFeatureDataset
from src.eval.feature_analysis import select_features_within_fold
from src.training.cv import load_or_generate_splits
from src.training.loop import TrainConfig, train_one_fold
from scripts._common import (
    ARCH_REGISTRY, find_window_feature_files, make_model_factory,
)


# Modality prefix → human label. Adjust if your feature naming differs.
MODALITY_PREFIXES = {
    'emg':  ('emg_',),
    'ecg':  ('ecg_', 'hr_', 'hrv_', 'rmssd', 'rr_'),
    'eda':  ('eda_', 'scr_', 'scl_'),
    'temp': ('temp_',),
    'acc':  ('acc_', 'ax_', 'ay_', 'az_', 'jerk_'),
    'ppg':  ('ppg_',),
}


def features_excluding_modality(all_features: List[str], modality: str) -> List[str]:
    """Return features that do NOT start with any prefix of the dropped modality."""
    prefixes = MODALITY_PREFIXES.get(modality)
    if not prefixes:
        raise ValueError(f"Unknown modality: {modality}. "
                          f"Known: {list(MODALITY_PREFIXES)}")
    kept = [f for f in all_features
            if not any(f.lower().startswith(p) for p in prefixes)]
    if len(kept) == len(all_features):
        print(f"[ablate] WARNING: dropping {modality} removed 0 features. "
              f"Check MODALITY_PREFIXES in the script.")
    return kept


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=list(ARCH_REGISTRY), required=True)
    p.add_argument('--top-k', type=int, default=None,
                    help='If set, after dropping a modality, also do top-K '
                         'leakage-safe selection on remaining features')
    p.add_argument('--leakage-safe', action='store_true',
                    help='Combine with --top-k for per-fold selection')
    p.add_argument('--modalities', type=str, nargs='+',
                    default=list(MODALITY_PREFIXES.keys()),
                    help='Which modalities to drop (default: all 6)')
    p.add_argument('--run-slug', type=str, default=None)
    p.add_argument('--labeled-root', type=Path, default=Path('data/labeled'))
    p.add_argument('--runs-root', type=Path, default=Path('runs'))
    p.add_argument('--splits', type=Path, default=Path('configs/splits.csv'))
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 1337, 7])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    return p.parse_args()


def train_one_ablation(
    dataset_full: WindowFeatureDataset,
    feature_subset: List[str],
    arch_name: str,
    out_dir: Path,
    folds,
    cfg: TrainConfig,
    seeds,
    win_paths,
    device,
    leakage_safe: bool = False,
    top_k: int = None,
):
    """Train arch_name on the given feature subset across folds × seeds."""
    if leakage_safe and top_k is not None:
        # Per-fold further selection within the subset
        for seed in seeds:
            for fold in folds:
                fold_id = fold['fold']
                train_idx = fold['train_idx']
                X_sub = dataset_full.X.numpy()[train_idx][:, [
                    dataset_full.feature_cols.index(f) for f in feature_subset
                ]]
                y_dict = {
                    'exercise': dataset_full.t_exercise[train_idx].numpy().astype(np.int64),
                    'phase':    dataset_full.t_phase[train_idx].numpy().astype(np.int64),
                    'fatigue':  np.where(
                        dataset_full.m_fatigue[train_idx].numpy(),
                        dataset_full.t_fatigue[train_idx].numpy().astype(float),
                        np.nan),
                    'reps':     np.where(
                        dataset_full.m_reps[train_idx].numpy(),
                        dataset_full.t_reps[train_idx].numpy().astype(float),
                        np.nan),
                }
                names, _ = select_features_within_fold(
                    X_sub, y_dict, feature_subset, top_k=top_k,
                )
                ds = WindowFeatureDataset(
                    window_parquets=win_paths, active_only=True,
                    feature_cols=names,
                    exercise_encoder=dataset_full.exercise_encoder,
                    phase_encoder=dataset_full.phase_encoder,
                    verbose=False,
                )
                factory = make_model_factory(
                    arch_name=arch_name,
                    n_features=ds.n_features,
                    n_exercise=ds.n_exercise,
                    n_phase=ds.n_phase,
                )
                fold_dir = out_dir / arch_name / f"seed_{seed}" / f"fold_{fold_id}"
                train_one_fold(
                    model_factory=factory, dataset=ds,
                    train_idx=fold['train_idx'], test_idx=fold['test_idx'],
                    cfg=cfg, device=device, out_dir=fold_dir,
                    n_exercise=ds.n_exercise, n_phase=ds.n_phase,
                )
    else:
        # Use the full subset directly
        ds = WindowFeatureDataset(
            window_parquets=win_paths, active_only=True,
            feature_cols=feature_subset,
            exercise_encoder=dataset_full.exercise_encoder,
            phase_encoder=dataset_full.phase_encoder,
            verbose=False,
        )
        factory = make_model_factory(
            arch_name=arch_name,
            n_features=ds.n_features,
            n_exercise=ds.n_exercise,
            n_phase=ds.n_phase,
        )
        for seed in seeds:
            for fold in folds:
                fold_id = fold['fold']
                fold_dir = out_dir / arch_name / f"seed_{seed}" / f"fold_{fold_id}"
                train_one_fold(
                    model_factory=factory, dataset=ds,
                    train_idx=fold['train_idx'], test_idx=fold['test_idx'],
                    cfg=cfg, device=device, out_dir=fold_dir,
                    n_exercise=ds.n_exercise, n_phase=ds.n_phase,
                )


def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    slug = args.run_slug or f"ablate_{args.arch}"
    parent_dir = args.runs_root / f"{timestamp}_{slug}"
    parent_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ablate] Output: {parent_dir}")

    win_paths = find_window_feature_files(args.labeled_root)
    full_dataset = WindowFeatureDataset(window_parquets=win_paths,
                                         active_only=True)
    all_features = full_dataset.feature_cols
    subject_ids = np.array(full_dataset.subject_ids)
    folds = load_or_generate_splits(subject_ids, splits_path=args.splits)

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                       lr=args.lr, weight_decay=1e-4, grad_clip=1.0,
                       patience=8, mixed_precision=True, num_workers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Save the modality manifest
    manifest = {
        'arch': args.arch,
        'modalities_dropped': args.modalities,
        'features_per_dropped_modality': {},
        'seeds': list(args.seeds),
        'leakage_safe_topk': args.top_k if args.leakage_safe else None,
    }

    for modality in args.modalities:
        kept = features_excluding_modality(all_features, modality)
        n_dropped = len(all_features) - len(kept)
        print(f"\n[ablate] === Drop '{modality}' "
              f"({n_dropped} features removed, {len(kept)} kept) ===")
        manifest['features_per_dropped_modality'][modality] = {
            'n_dropped': n_dropped,
            'n_kept': len(kept),
        }

        out_dir = parent_dir / f"no_{modality}"
        out_dir.mkdir(parents=True, exist_ok=True)
        train_one_ablation(
            dataset_full=full_dataset,
            feature_subset=kept,
            arch_name=args.arch,
            out_dir=out_dir,
            folds=folds,
            cfg=cfg,
            seeds=args.seeds,
            win_paths=win_paths,
            device=device,
            leakage_safe=args.leakage_safe,
            top_k=args.top_k,
        )

    with open(parent_dir / 'ablation_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[ablate] Done. Output: {parent_dir}")
    print(f"[ablate] Use scripts/compare_all.py with --ablation-runs to "
          f"include in master comparison.")


if __name__ == '__main__':
    main()
