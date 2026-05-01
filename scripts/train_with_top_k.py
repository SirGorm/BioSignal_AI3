"""Train a multi-task NN on a top-K feature subset and compare against the
full-feature run.

Two modes:

1. Lazy mode (--feature-list path/to/top_30_features.json):
   Use the precomputed list from scripts/analyze_features.py. WARNING: this
   is leakage-prone because the list was computed on the full dataset
   (Saeb 2017). OK for exploratory sanity checks; do not trust the metrics
   for publication.

2. Strict mode (--leakage-safe):
   Re-run feature selection INSIDE each CV fold's training set, never seeing
   the test fold. This is the publishable approach. Slightly slower.

Run:
    # Lazy:
    python scripts/train_with_top_k.py --arch tcn --top-k 30 \\
        --feature-list runs/<ts>_feature-analysis/top_30_features.json

    # Strict (recommended for final results):
    python scripts/train_with_top_k.py --arch tcn --top-k 30 --leakage-safe

References:
- Saeb et al. 2017 — leakage-aware CV
- Guyon & Elisseeff 2003 — feature selection survey
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path

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
    ARCH_REGISTRY, find_window_feature_files, make_model_factory
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=list(ARCH_REGISTRY), required=True)
    p.add_argument('--top-k', type=int, default=30)
    p.add_argument('--feature-list', type=Path, default=None,
                    help='Path to top_K_features.json from analyze_features.py')
    p.add_argument('--leakage-safe', action='store_true',
                    help='Re-run feature selection per fold (slower, publishable)')
    p.add_argument('--run-slug', type=str, default=None)
    p.add_argument('--labeled-root', type=Path, default=Path('data/labeled'))
    p.add_argument('--runs-root', type=Path, default=Path('runs'))
    p.add_argument('--splits', type=Path, default=Path('configs/splits.csv'))
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 1337, 7])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.leakage_safe and args.feature_list is None:
        raise SystemExit("Specify --feature-list <path> OR --leakage-safe.")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode = 'safe' if args.leakage_safe else 'lazy'
    slug = args.run_slug or f"top{args.top_k}_{args.arch}_{mode}"
    run_dir = args.runs_root / f"{timestamp}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[top-k] Output: {run_dir}  (mode={mode})")

    win_paths = find_window_feature_files(args.labeled_root)
    full_dataset = WindowFeatureDataset(window_parquets=win_paths,
                                         active_only=True)
    print(f"[top-k] Full dataset: {len(full_dataset)} windows × "
          f"{full_dataset.n_features} features")

    subject_ids = np.array(full_dataset.subject_ids)
    folds = load_or_generate_splits(subject_ids, splits_path=args.splits)

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                       lr=args.lr, weight_decay=1e-4, grad_clip=1.0,
                       patience=8, mixed_precision=True, num_workers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.leakage_safe:
        # Lazy mode: load shortlist once, subset dataset, train normally
        shortlist = json.loads(args.feature_list.read_text())['features']
        sub_ds = WindowFeatureDataset(
            window_parquets=win_paths, active_only=True,
            feature_cols=shortlist[:args.top_k],
            exercise_encoder=full_dataset.exercise_encoder,
            phase_encoder=full_dataset.phase_encoder,
        )
        print(f"[top-k] Subsetted to {sub_ds.n_features} features (lazy)")

        factory = make_model_factory(
            arch_name=args.arch,
            n_features=sub_ds.n_features,
            n_exercise=sub_ds.n_exercise,
            n_phase=sub_ds.n_phase,
        )

        for seed in args.seeds:
            for fold in folds:
                fold_dir = (run_dir / args.arch / f"seed_{seed}"
                             / f"fold_{fold['fold']}")
                print(f"\n[top-k] seed={seed} fold={fold['fold']} (lazy)")
                train_one_fold(
                    model_factory=factory, dataset=sub_ds,
                    train_idx=fold['train_idx'], test_idx=fold['test_idx'],
                    cfg=cfg, device=device, out_dir=fold_dir,
                    n_exercise=sub_ds.n_exercise, n_phase=sub_ds.n_phase,
                )
        return

    # Leakage-safe mode: per fold, select features on training subset only
    X_full = full_dataset.X.numpy()
    feature_names = full_dataset.feature_cols

    selected_per_fold = {}
    for fold in folds:
        train_idx = fold['train_idx']
        y_dict = {
            'exercise': full_dataset.t_exercise[train_idx].numpy().astype(np.int64),
            'phase':    full_dataset.t_phase[train_idx].numpy().astype(np.int64),
            'fatigue':  np.where(
                full_dataset.m_fatigue[train_idx].numpy(),
                full_dataset.t_fatigue[train_idx].numpy().astype(float),
                np.nan),
            'reps':     np.where(
                full_dataset.m_reps[train_idx].numpy(),
                full_dataset.t_reps[train_idx].numpy().astype(float),
                np.nan),
        }
        names, idxs = select_features_within_fold(
            X_full[train_idx], y_dict, feature_names, top_k=args.top_k,
        )
        selected_per_fold[fold['fold']] = {'names': names, 'idxs': idxs.tolist()}
        print(f"[top-k] fold {fold['fold']}: top {args.top_k} features = "
              f"{names[:5]}... (showing first 5)")

    # Save the per-fold selections for reproducibility
    with open(run_dir / 'selected_features_per_fold.json', 'w') as f:
        json.dump(selected_per_fold, f, indent=2)

    # Train each fold with its own feature subset
    for seed in args.seeds:
        for fold in folds:
            fold_id = fold['fold']
            sel = selected_per_fold[fold_id]
            print(f"\n[top-k] seed={seed} fold={fold_id} (safe, "
                  f"{len(sel['names'])} features)")

            sub_ds = WindowFeatureDataset(
                window_parquets=win_paths, active_only=True,
                feature_cols=sel['names'],
                exercise_encoder=full_dataset.exercise_encoder,
                phase_encoder=full_dataset.phase_encoder,
                verbose=False,
            )
            factory = make_model_factory(
                arch_name=args.arch,
                n_features=sub_ds.n_features,
                n_exercise=sub_ds.n_exercise,
                n_phase=sub_ds.n_phase,
            )
            fold_dir = run_dir / args.arch / f"seed_{seed}" / f"fold_{fold_id}"
            train_one_fold(
                model_factory=factory, dataset=sub_ds,
                train_idx=fold['train_idx'], test_idx=fold['test_idx'],
                cfg=cfg, device=device, out_dir=fold_dir,
                n_exercise=sub_ds.n_exercise, n_phase=sub_ds.n_phase,
            )

    print(f"\n[top-k] Complete. Compare to full-feature run via "
          f"scripts/compare_architectures.py.")


if __name__ == '__main__':
    main()
