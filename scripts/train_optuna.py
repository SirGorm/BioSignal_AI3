"""Optuna HP search + final 3-seed eval for one architecture.

Phase 1: N Optuna trials, 1 seed x 5 folds x phase1_epochs (early stop).
         Score = mean rank across the 4 task metrics.
Phase 2: best HPs, 3 seeds x 5 folds x phase2_epochs.

Supports both feature-input ('mlp', 'cnn1d', 'lstm', ...) and raw-input
('cnn1d_raw', 'lstm_raw', 'cnn_lstm_raw', 'tcn_raw') architectures.

Run:
    python scripts/train_optuna.py --arch mlp        --variant features --n-trials 20
    python scripts/train_optuna.py --arch tcn_raw    --variant raw      --n-trials 25
    python scripts/train_optuna.py --arch cnn_lstm_raw --variant raw    --n-trials 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import optuna

from src.data.datasets import WindowFeatureDataset
from src.data.phase_whitelist import load_phase_whitelist
from src.data.raw_window_dataset import RawMultimodalWindowDataset
from src.models.cnn1d import CNN1DMultiTask
from src.models.lstm import LSTMMultiTask
from src.models.cnn_lstm import CNNLSTMMultiTask
from src.models.tcn import TCNMultiTask
from src.models.mlp import MLPMultiTask
from src.models.raw.cnn1d_raw import CNN1DRawMultiTask
from src.models.raw.lstm_raw import LSTMRawMultiTask
from src.models.raw.cnn_lstm_raw import CNNLSTMRawMultiTask
from src.models.raw.tcn_raw import TCNRawMultiTask
from src.training.cv import load_or_generate_splits
from src.training.loop import TrainConfig, run_cv

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEAT_REGISTRY = {
    'mlp':      MLPMultiTask,
    'cnn1d':    CNN1DMultiTask,
    'lstm':     LSTMMultiTask,
    'cnn_lstm': CNNLSTMMultiTask,
    'tcn':      TCNMultiTask,
}
RAW_REGISTRY = {
    'cnn1d_raw':    CNN1DRawMultiTask,
    'lstm_raw':     LSTMRawMultiTask,
    'cnn_lstm_raw': CNNLSTMRawMultiTask,
    'tcn_raw':      TCNRawMultiTask,
}


def suggest_hps(trial: optuna.Trial, arch: str) -> Dict:
    hps = {
        'lr':           trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size':   trial.suggest_categorical('batch_size', [64, 128, 256]),
        'dropout':      trial.suggest_float('dropout', 0.1, 0.5),
        'repr_dim':     trial.suggest_categorical('repr_dim', [64, 128, 256]),
    }
    if arch == 'mlp':
        hps['hidden_dim'] = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    if arch in ('lstm', 'lstm_raw', 'cnn_lstm', 'cnn_lstm_raw'):
        hps['lstm_hidden'] = trial.suggest_categorical('lstm_hidden', [32, 64, 128])
        hps['lstm_layers'] = trial.suggest_int('lstm_layers', 1, 2)
    if arch in ('tcn', 'tcn_raw'):
        hps['tcn_kernel'] = trial.suggest_categorical('tcn_kernel', [3, 5, 7])
    return hps


def build_factory(cls, variant: str, ds, hps: Dict):
    common = {
        'n_exercise': ds.n_exercise,
        'n_phase':    ds.n_phase,
        'repr_dim':   hps['repr_dim'],
        'dropout':    hps['dropout'],
    }
    if variant == 'features':
        common['n_features'] = ds.n_features
        if 'hidden_dim' in hps:
            common['hidden_dim'] = hps['hidden_dim']
    else:
        common['n_channels'] = ds.n_channels
        common['n_timesteps'] = ds.n_timesteps

    # Optional architecture-specific kwargs (only pass if the model accepts them)
    import inspect
    sig = inspect.signature(cls.__init__).parameters
    if 'lstm_hidden' in hps:
        if 'hidden' in sig:        common['hidden'] = hps['lstm_hidden']
        elif 'lstm_hidden' in sig: common['lstm_hidden'] = hps['lstm_hidden']
    if 'lstm_layers' in hps:
        if 'n_layers' in sig:        common['n_layers'] = hps['lstm_layers']
        elif 'n_lstm_layers' in sig: common['n_lstm_layers'] = hps['lstm_layers']
    if 'tcn_kernel' in hps and 'kernel_size' in sig:
        common['kernel_size'] = hps['tcn_kernel']

    def factory():
        return cls(**common)
    return factory


def score_summary(summary: Dict) -> float:
    """Lower is better. Combines 4 tasks via z-rank-style aggregation."""
    parts = []
    for task, metric, sign in [
        ('exercise', 'f1_macro', -1),  # higher F1 = lower score
        ('phase',    'f1_macro', -1),
        ('fatigue',  'mae',      +1),
        ('reps',     'mae',      +1),
    ]:
        v = summary.get(task, {}).get(metric, {}).get('mean')
        if v is not None and not np.isnan(v):
            parts.append(sign * float(v))
    return float(np.mean(parts)) if parts else float('inf')


def run_one_cv(arch: str, variant: str, dataset, folds, seeds, hps,
                epochs, num_workers, out_dir: Path,
                use_uncertainty: bool):
    """One pass of run_cv with given HPs."""
    cls = (RAW_REGISTRY if variant == 'raw' else FEAT_REGISTRY)[arch]
    factory = build_factory(cls, variant, dataset, hps)

    cfg = TrainConfig(
        epochs=epochs, batch_size=hps['batch_size'], lr=hps['lr'],
        weight_decay=hps['weight_decay'], grad_clip=1.0, patience=5,
        mixed_precision=True, num_workers=num_workers,
        use_uncertainty_weighting=use_uncertainty,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'train_config.json').write_text(json.dumps(cfg.__dict__, indent=2))
    (out_dir / 'hps.json').write_text(json.dumps(hps, indent=2))

    summary, all_results = run_cv(
        dataset=dataset, model_factory=factory, arch_name=arch,
        cfg=cfg, splits=folds, out_root=out_dir, seeds=tuple(seeds),
    )
    return summary, all_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True)
    p.add_argument('--variant', choices=['features', 'raw'], required=True)
    p.add_argument('--n-trials', type=int, default=20)
    p.add_argument('--phase1-epochs', type=int, default=30)
    p.add_argument('--phase2-epochs', type=int, default=50)
    p.add_argument('--phase2-seeds', type=int, nargs='+', default=[42, 1337, 7])
    p.add_argument('--num-workers', type=int, default=None,
                   help='Default: 2 (features) or 8 (raw)')
    p.add_argument('--exclude-recordings', nargs='*', default=[],
                   help='Recording-id substrings to exclude (e.g. recording_003)')
    p.add_argument('--phase-whitelist', type=Path, default=None,
                   help='CSV of (recording_id, set_number) pairs whose '
                        'phase_label is clean enough to train on.')
    p.add_argument('--feature-prefixes', nargs='*', default=None,
                   help='Modality prefixes to keep in features dataset '
                        '(e.g. emg_ acc_ ppg_ temp_). Default: all.')
    p.add_argument('--no-uncertainty', action='store_true',
                   help='Disable uncertainty weighting (default: ON)')
    p.add_argument('--skip-phase2', action='store_true')
    p.add_argument('--labeled-root', type=Path, default=Path('data/labeled'))
    p.add_argument('--runs-root', type=Path, default=Path('runs'))
    p.add_argument('--splits', type=Path, default=Path('configs/splits.csv'))
    p.add_argument('--run-dir', type=Path, default=None,
                   help='Reuse this exact run dir (enables resume). If absent, '
                        'a fresh timestamped dir is created.')
    args = p.parse_args()

    if args.num_workers is None:
        args.num_workers = 2 if args.variant == 'features' else 8
    use_uncertainty = not args.no_uncertainty

    # Load dataset once
    if args.variant == 'features':
        files = sorted(args.labeled_root.rglob('window_features.parquet'))
    else:
        files = sorted(args.labeled_root.rglob('aligned_features.parquet'))
    if args.exclude_recordings:
        before = len(files)
        files = [p for p in files
                  if not any(ex in str(p) for ex in args.exclude_recordings)]
        print(f"[optuna] Excluded {before - len(files)} parquet(s) "
              f"matching {args.exclude_recordings}")
    phase_wl = load_phase_whitelist(args.phase_whitelist)
    if phase_wl is not None:
        print(f"[optuna] Phase whitelist: {args.phase_whitelist} "
              f"({len(phase_wl)} (recording, set) pairs)")
    if args.variant == 'features':
        feat_cols = None
        if args.feature_prefixes:
            import pandas as pd
            cols = pd.read_parquet(files[0]).columns
            feat_cols = sorted(c for c in cols
                                if any(c.startswith(p) for p in args.feature_prefixes))
            print(f"[optuna] feature_prefixes={args.feature_prefixes} -> "
                  f"{len(feat_cols)} cols")
        dataset = WindowFeatureDataset(window_parquets=files, active_only=True,
                                         phase_whitelist=phase_wl,
                                         feature_cols=feat_cols)
    else:
        dataset = RawMultimodalWindowDataset(parquet_paths=files, active_only=True,
                                               phase_whitelist=phase_wl)
    print(f"[optuna] {args.variant} dataset: {len(dataset)} windows")

    subject_ids = np.array(dataset.subject_ids)
    folds = load_or_generate_splits(subject_ids, splits_path=args.splits)
    print(f"[optuna] {len(folds)} CV folds, {len(np.unique(subject_ids))} subjects")

    if args.run_dir is not None:
        run_dir = args.run_dir
        is_resume = run_dir.exists() and (run_dir / 'config.json').exists()
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = args.runs_root / f"{timestamp}_optuna-{args.variant}-{args.arch}"
        is_resume = False
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[optuna] Output: {run_dir}  (resume={is_resume})")

    config_dump = {
        'arch': args.arch, 'variant': args.variant,
        'n_trials': args.n_trials, 'phase1_epochs': args.phase1_epochs,
        'phase2_epochs': args.phase2_epochs, 'phase2_seeds': args.phase2_seeds,
        'num_workers': args.num_workers, 'use_uncertainty': use_uncertainty,
    }
    if not is_resume:
        (run_dir / 'config.json').write_text(json.dumps(config_dump, indent=2))

    # Always write dataset_meta.json so plotting can label confusion matrices.
    meta = {
        'arch': args.arch,
        'variant': args.variant,
        'exercise_classes': list(dataset.exercise_encoder.classes_),
        'phase_classes':    list(dataset.phase_encoder.classes_),
    }
    if args.variant == 'features':
        meta['feature_cols'] = list(dataset.feature_cols)
        meta['n_features']   = dataset.n_features
    else:
        meta['n_channels']  = dataset.n_channels
        meta['n_timesteps'] = dataset.n_timesteps
    (run_dir / 'dataset_meta.json').write_text(json.dumps(meta, indent=2))
    # Also drop a copy under phase2/ so plot_confusion_matrices_for_run finds it.
    (run_dir / 'phase2').mkdir(exist_ok=True)
    (run_dir / 'phase2' / 'dataset_meta.json').write_text(json.dumps(meta, indent=2))

    # ------ Phase 1: Optuna search (1 seed) — RESUME-ABLE via SQLite ------
    trial_log = []
    if (run_dir / 'phase1_log.json').exists():
        try:
            trial_log = json.loads((run_dir / 'phase1_log.json').read_text())
        except Exception:
            trial_log = []

    def objective(trial: optuna.Trial) -> float:
        hps = suggest_hps(trial, args.arch)
        trial_dir = run_dir / 'phase1' / f"trial_{trial.number:03d}"
        # Per-trial cache: if cv_summary already exists, reuse the score.
        cached = next(iter(trial_dir.rglob('cv_summary.json')), None)
        if cached is not None:
            try:
                cached_summary = json.loads(cached.read_text()).get('summary', {})
                score = score_summary(cached_summary)
                print(f"[optuna] trial {trial.number} CACHED: score={score:.4f}")
                return score
            except Exception:
                pass  # fall through to fresh run
        try:
            t0 = time.time()
            summary, _ = run_one_cv(
                args.arch, args.variant, dataset, folds, seeds=[42],
                hps=hps, epochs=args.phase1_epochs,
                num_workers=args.num_workers, out_dir=trial_dir,
                use_uncertainty=use_uncertainty,
            )
            score = score_summary(summary)
            elapsed = time.time() - t0
            trial_log.append({
                'trial': trial.number, 'hps': hps, 'score': score,
                'summary': summary, 'elapsed_s': elapsed,
            })
            (run_dir / 'phase1_log.json').write_text(json.dumps(trial_log, indent=2))
            print(f"[optuna] trial {trial.number}: score={score:.4f} "
                  f"({elapsed:.0f}s)  hps={hps}")
            return score
        except Exception as e:
            print(f"[optuna] trial {trial.number} FAILED: {e}")
            return float('inf')

    print(f"\n=== PHASE 1: {args.n_trials} Optuna trials ===")
    t_start_p1 = time.time()
    storage_url = f"sqlite:///{run_dir.as_posix()}/optuna.db"
    study = optuna.create_study(
        storage=storage_url,
        study_name=args.arch,
        load_if_exists=True,
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    # Clean up trials that were RUNNING when the previous process died.
    for t in study.get_trials(deepcopy=False,
                                states=[optuna.trial.TrialState.RUNNING]):
        try:
            study._storage.set_trial_state_values(
                t._trial_id, optuna.trial.TrialState.FAIL)
        except Exception:
            pass
    n_done = len([t for t in study.get_trials(deepcopy=False)
                   if t.state in (optuna.trial.TrialState.COMPLETE,
                                    optuna.trial.TrialState.PRUNED,
                                    optuna.trial.TrialState.FAIL)])
    n_remaining = max(0, args.n_trials - n_done)
    print(f"[optuna] Resume: {n_done} trials already in study, "
          f"running {n_remaining} more")
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, show_progress_bar=False)
    p1_elapsed = time.time() - t_start_p1
    print(f"[optuna] Phase 1 complete in {p1_elapsed/60:.1f} min")

    best_hps = study.best_params
    # Re-merge any architecture defaults that weren't sampled in this study
    sample = suggest_hps(optuna.trial.FixedTrial({**best_hps}), args.arch)
    best_hps_full = {**sample, **best_hps}

    print(f"[optuna] BEST: {study.best_value:.4f} with {best_hps_full}")
    (run_dir / 'best_hps.json').write_text(json.dumps({
        'best_score': study.best_value,
        'best_hps': best_hps_full,
        'phase1_elapsed_s': p1_elapsed,
    }, indent=2))

    # ------ Phase 2: refit best HPs with multi-seed ------
    if args.skip_phase2:
        print("[optuna] Skipping phase 2 (--skip-phase2)")
        return

    p2_dir = run_dir / 'phase2'
    p2_done = next(iter(p2_dir.rglob('cv_summary.json')), None)
    if p2_done is not None:
        print(f"[optuna] Phase 2 already complete (cached at {p2_done}), skipping")
        return

    print(f"\n=== PHASE 2: 3-seed refit on best HPs ===")
    t_start_p2 = time.time()
    summary, _ = run_one_cv(
        args.arch, args.variant, dataset, folds, seeds=args.phase2_seeds,
        hps=best_hps_full, epochs=args.phase2_epochs,
        num_workers=args.num_workers, out_dir=p2_dir,
        use_uncertainty=use_uncertainty,
    )
    p2_elapsed = time.time() - t_start_p2
    print(f"[optuna] Phase 2 complete in {p2_elapsed/60:.1f} min")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
