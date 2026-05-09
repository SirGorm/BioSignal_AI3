"""Phase 2-only refit using HPs from an existing Optuna run.

Loads ``best_hps.json`` from ``--src-run-dir``, builds the dataset with the
requested ``active_only`` setting, and runs a 3-seed × N-fold phase-2 eval.
Writes outputs under ``--out-run-dir/phase2/``.

Used to re-evaluate v9 winners with rest windows included (``--include-rest``)
without re-running the full Optuna search.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import optuna
import torch

from src.data.datasets import WindowFeatureDataset
from src.data.phase_whitelist import load_phase_whitelist
from src.data.raw_window_dataset import RawMultimodalWindowDataset
from src.training.cv import load_or_generate_splits

# Reuse helpers so model factory + run_one_cv stay in sync.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_optuna import run_one_cv, suggest_hps  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True)
    p.add_argument('--variant', choices=['features', 'raw'], required=True)
    p.add_argument('--src-run-dir', type=Path, required=True,
                    help='Existing Optuna run dir to copy best_hps.json from')
    p.add_argument('--out-run-dir', type=Path, required=True,
                    help='Output dir; phase2/ is written here')
    p.add_argument('--include-rest', action='store_true',
                    help='Include rest windows (active_only=False)')
    p.add_argument('--phase2-epochs', type=int, default=150)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--phase2-seeds', type=int, nargs='+', default=[42, 1337, 7],
                    help='Default 3 seeds for variance estimation.')
    p.add_argument('--tasks', nargs='+',
                    default=['exercise', 'phase', 'fatigue', 'reps'])
    p.add_argument('--reps-mode', default='soft_window',
                    choices=['hard', 'soft_window', 'soft_overlap'])
    p.add_argument('--phase-mode', default='soft', choices=['hard', 'soft'])
    p.add_argument('--window-s', type=float, default=2.0)
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--exclude-recordings', nargs='*', default=[])
    p.add_argument('--phase-whitelist', type=Path, default=None)
    p.add_argument('--feature-prefixes', nargs='*', default=None,
                    help='Restrict features-variant to columns starting with '
                         'one of these prefixes (modality ablation).')
    p.add_argument('--raw-channels', nargs='*', default=None,
                    help='Restrict raw-variant input to a subset of '
                         '[emg, ppg_green, acc_mag, temp] (modality ablation).')
    p.add_argument('--no-uncertainty', action='store_true')
    p.add_argument('--exercise-aggregation',
                   choices=['per_window', 'per_set', 'both'],
                   default='per_window',
                   help='Exercise eval granularity. per_set aggregates '
                        'per-window predictions to one per (recording, set) '
                        'via mean-softmax — mirrors RPE supervision.')
    p.add_argument('--no-tight-hps', dest='tight_hps', action='store_false',
                    help='Use wide HP search space (default: tight, must '
                         'match how the source run was trained).')
    p.set_defaults(tight_hps=True)
    p.add_argument('--labeled-root', type=Path, default=Path('data/labeled'))
    p.add_argument('--splits', type=Path,
                    default=Path('configs/splits_loso.csv'),
                    help='Default LOSO (10 folds). Pass configs/splits.csv for 5-fold.')
    args = p.parse_args()

    if args.num_workers is None:
        args.num_workers = 2 if args.variant == 'features' else 8
    use_uncertainty = not args.no_uncertainty
    active_only = not args.include_rest

    # ---- Load HPs from source run ------------------------------------------
    src_hp = args.src_run_dir / 'best_hps.json'
    if not src_hp.exists():
        raise FileNotFoundError(f"Missing: {src_hp}")
    src = json.loads(src_hp.read_text())
    best_hps = src['best_hps']
    # Use best_hps verbatim — they already contain every HP the model and
    # training loop need (lr, weight_decay, batch_size, dropout, repr_dim,
    # hidden_dim, ...). Skipping the FixedTrial validation against the
    # current search space lets us re-evaluate older runs whose HPs
    # (e.g. repr_dim=8) fall outside today's search categorical.
    best_hps_full = dict(best_hps)
    best_hps_full['_patience'] = args.patience
    print(f"[phase2-only] HPs from {src_hp}")
    print(f"[phase2-only] {best_hps_full}")
    print(f"[phase2-only] include_rest={args.include_rest}  "
          f"active_only={active_only}")

    # ---- Build dataset -----------------------------------------------------
    if args.variant == 'features':
        files = sorted(args.labeled_root.rglob('window_features.parquet'))
    else:
        files = sorted(args.labeled_root.rglob('aligned_features.parquet'))
    if args.exclude_recordings:
        files = [pp for pp in files
                 if not any(ex in str(pp) for ex in args.exclude_recordings)]
    target_modes = {'reps': args.reps_mode, 'phase': args.phase_mode}
    phase_wl = load_phase_whitelist(args.phase_whitelist)

    if args.variant == 'features':
        feat_cols = None
        if args.feature_prefixes:
            import pandas as pd
            cols = pd.read_parquet(files[0]).columns
            feat_cols = sorted(c for c in cols
                                if any(c.startswith(pp) for pp in args.feature_prefixes))
        dataset = WindowFeatureDataset(
            window_parquets=files, active_only=active_only,
            phase_whitelist=phase_wl, feature_cols=feat_cols,
            target_modes=target_modes, window_s=args.window_s)
    else:
        dataset = RawMultimodalWindowDataset(
            parquet_paths=files, active_only=active_only,
            phase_whitelist=phase_wl, target_modes=target_modes,
            window_s=args.window_s, channels=args.raw_channels)
    print(f"[phase2-only] dataset: {len(dataset)} windows  "
          f"phase_classes={list(dataset.phase_encoder.classes_)}")

    if torch.cuda.is_available():
        t0 = time.time()
        dataset.materialize_to_device("cuda")
        print(f"[phase2-only] dataset on cuda in {time.time()-t0:.1f}s "
              f"(gpu_resident={dataset.gpu_resident})")

    subject_ids = np.array(dataset.subject_ids)
    folds = load_or_generate_splits(subject_ids, splits_path=args.splits)
    print(f"[phase2-only] {len(folds)} folds, "
          f"{len(np.unique(subject_ids))} subjects")

    # ---- Output dir + cache check ------------------------------------------
    out_dir = args.out_run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    p2_dir = out_dir / 'phase2'
    p2_dir.mkdir(exist_ok=True)
    p2_done = next(iter(p2_dir.rglob('cv_summary.json')), None)
    if p2_done is not None:
        print(f"[phase2-only] Already complete (cached): {p2_done}")
        return

    meta = {
        'arch': args.arch, 'variant': args.variant,
        'src_run_dir': str(args.src_run_dir),
        'include_rest': args.include_rest, 'active_only': active_only,
        'window_s': args.window_s,
        'exercise_classes': list(dataset.exercise_encoder.classes_),
        'phase_classes': list(dataset.phase_encoder.classes_),
    }
    if args.variant == 'features':
        meta['feature_cols'] = list(dataset.feature_cols)
        meta['n_features'] = dataset.n_features
    else:
        meta['n_channels'] = dataset.n_channels
        meta['n_timesteps'] = dataset.n_timesteps
    (out_dir / 'dataset_meta.json').write_text(json.dumps(meta, indent=2))
    (p2_dir / 'dataset_meta.json').write_text(json.dumps(meta, indent=2))
    (out_dir / 'best_hps.json').write_text(json.dumps({
        'best_hps': best_hps_full,
        'src_run_dir': str(args.src_run_dir),
        'include_rest': args.include_rest,
    }, indent=2))

    # ---- Run phase 2 -------------------------------------------------------
    print(f"\n=== PHASE 2: {len(args.phase2_seeds)} seeds × "
          f"{len(folds)} folds ===")
    t0 = time.time()
    summary, _ = run_one_cv(
        args.arch, args.variant, dataset, folds,
        seeds=args.phase2_seeds, hps=best_hps_full,
        epochs=args.phase2_epochs, num_workers=args.num_workers,
        out_dir=p2_dir, use_uncertainty=use_uncertainty,
        enabled_tasks=args.tasks, target_modes=target_modes,
        exercise_aggregation=args.exercise_aggregation,
    )
    elapsed = time.time() - t0
    print(f"[phase2-only] DONE in {elapsed/60:.1f} min")
    print(json.dumps(summary, indent=2))

    # ---- Pick best fold and copy its checkpoint to top-level out_dir ----
    import shutil
    best_fold = None
    best_score = float('inf')
    for fold_dir in p2_dir.rglob('fold_*'):
        m_path = fold_dir / 'metrics.json'
        ckpt = fold_dir / 'checkpoint_best.pt'
        if not (m_path.exists() and ckpt.exists()):
            continue
        m = json.loads(m_path.read_text())
        score = m.get('val_total')
        if score is None:
            continue
        if score < best_score:
            best_score = float(score)
            best_fold = (fold_dir, m, ckpt)

    if best_fold is not None:
        fold_dir, m, ckpt = best_fold
        target = out_dir / 'best_model.pt'
        shutil.copy2(ckpt, target)
        meta = {
            'best_fold_dir': str(fold_dir.relative_to(out_dir)),
            'val_total': best_score,
            'test_subjects': m.get('test_subjects', []),
            'metrics': {k: m.get(k) for k in ('exercise', 'phase', 'fatigue',
                                               'reps') if k in m},
            'arch': args.arch, 'variant': args.variant,
            'window_s': args.window_s,
            'tight_hps': args.tight_hps,
            'best_hps': best_hps_full,
        }
        (out_dir / 'best_model_meta.json').write_text(json.dumps(meta, indent=2,
                                                                   default=str))
        print(f"[phase2-only] Best fold: {fold_dir.relative_to(out_dir)}  "
              f"val_total={best_score:.4f}  "
              f"test={m.get('test_subjects', '?')}")
        print(f"[phase2-only] Wrote {target} + best_model_meta.json")


if __name__ == '__main__':
    main()
