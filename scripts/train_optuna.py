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
import torch

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


def suggest_hps(trial: optuna.Trial, arch: str, tight: bool = False,
                wide_arch: bool = False,
                repr_dim_choices=None) -> Dict:
    """Optuna search space.

    By default only training HPs are searched (lr, weight_decay, batch_size,
    repr_dim ∈ [16, 32, 64]). Architecture HPs (dropout, lstm_hidden, etc.)
    use the model class defaults — keeps param count comparable across
    architectures.

    `wide_arch=True` opens the architecture HP search:
      - repr_dim ∈ [32, 64, 128, 256]    (up from [16,32,64])
      - dropout  ∈ [0.1, 0.5] (uniform)  (otherwise model default 0.3)
      - lr ceiling raised to 5e-3 even in tight mode (raw CNN often needs it)
    Use this for raw-variant sweeps where repr_dim=16 is too small for the
    encoder backbone.

    `tight=True` narrows the lr/weight_decay ranges for the low-N regime
    (~95 effective sets, ~10 subjects); the wider mode is for sanity-check
    sweeps.
    """
    if wide_arch:
        rd_choices = list(repr_dim_choices) if repr_dim_choices else [32, 64, 128, 256]
        out = {
            'lr':           trial.suggest_float('lr', 1e-4, 5e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'batch_size':   trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'repr_dim':     trial.suggest_categorical('repr_dim', rd_choices),
            'dropout':      trial.suggest_float('dropout', 0.1, 0.5),
        }
        # LSTM-specific tuning of stacking depth (causal, bidirectional=False).
        if arch in ('lstm', 'lstm_raw'):
            out['n_layers'] = trial.suggest_categorical('n_layers', [1, 2])
        return out
    rd_choices = list(repr_dim_choices) if repr_dim_choices else [16, 32, 64]
    if tight:
        return {
            'lr':           trial.suggest_float('lr', 1e-4, 3e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
            'batch_size':   trial.suggest_categorical('batch_size', [128, 256]),
            'repr_dim':     trial.suggest_categorical('repr_dim', rd_choices),
        }
    return {
        'lr':           trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size':   trial.suggest_categorical('batch_size', [64, 128, 256]),
        'repr_dim':     trial.suggest_categorical('repr_dim', rd_choices),
    }


def build_factory(cls, variant: str, ds, hps: Dict):
    """Build a model factory that uses each class's own default architecture HPs.

    Optuna only tunes training HPs — architecture is fixed. We pass dataset
    shape (n_features OR n_channels+n_timesteps) and task heads, nothing else.
    """
    common = {
        'n_exercise': ds.n_exercise,
        'n_phase':    ds.n_phase,
    }
    if 'repr_dim' in hps:
        common['repr_dim'] = hps['repr_dim']
    if 'dropout' in hps:
        common['dropout'] = hps['dropout']
    if 'n_layers' in hps:
        common['n_layers'] = hps['n_layers']
    if variant == 'features':
        common['n_features'] = ds.n_features
    else:
        common['n_channels'] = ds.n_channels
        common['n_timesteps'] = ds.n_timesteps

    def factory():
        return cls(**common)
    return factory


# Each metric is rescaled into a [0,1] error contribution before averaging,
# so all enabled tasks (heads) contribute equally regardless of native scale.
# Bounds are fixed (not per-trial) so trial scores stay comparable across runs.
TASK_NORMALIZERS = [
    # (task, metric, fn(v) → [0,1] normalized error; lower = better)
    ('exercise', 'f1_macro',  lambda v: 1.0 - v),                # F1 → 1-F1
    ('phase',    'f1_macro',  lambda v: 1.0 - v),
    ('fatigue',  'mae',        lambda v: min(v / 3.0, 1.0)),     # 3 RPE-pts ≈ trivial-mean baseline
    ('fatigue',  'pearson_r',  lambda v: (1.0 - v) / 2.0),       # r=1 → 0, r=-1 → 1
    ('reps',     'mae',        lambda v: min(v / 1.0, 1.0)),     # soft_overlap ∈ [0,1]
]
# Within a single task, blend its metrics. Fatigue uses MAE+r so the optimizer
# isn't fooled by "regression-to-mean" predictions that minimize MAE without
# learning rep-by-rep variation.
TASK_METRIC_WEIGHTS = {
    'fatigue': {'mae': 0.5, 'pearson_r': 0.5},
}


def score_summary(summary: Dict, enabled_tasks: list = None) -> float:
    """Lower is better.

    Returns the mean uncertainty-weighted val_total across CV folds. This is
    the same multi-task loss the model trains on (with learnable per-task
    sigma_k from MultiTaskLoss), so Optuna optimizes the same objective
    that training does — no separate composite scoring is needed.

    Falls back to the legacy normalized-metric composite if val_total is
    missing (e.g., cv_summary written before this change).

    The ``enabled_tasks`` argument is kept for API compatibility but ignored:
    the uncertainty-weighted total already reflects only the enabled tasks
    (untrained heads contribute zero loss inside MultiTaskLoss).
    """
    vt = summary.get('val_total')
    if isinstance(vt, dict) and 'mean' in vt:
        m = vt.get('mean')
        if m is not None and not (isinstance(m, float) and np.isnan(m)):
            return float(m)

    # ---- Legacy composite fallback (for old cached cv_summary files) ----
    by_task: Dict[str, list] = {}
    for task, metric, fn in TASK_NORMALIZERS:
        if enabled_tasks is not None and task not in enabled_tasks:
            continue
        v = summary.get(task, {}).get(metric, {}).get('mean')
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        by_task.setdefault(task, []).append((metric, fn(float(v))))
    parts = []
    for task, metric_vals in by_task.items():
        weights = TASK_METRIC_WEIGHTS.get(task)
        if weights:
            num = sum(weights.get(m, 0.0) * v for m, v in metric_vals)
            den = sum(weights.get(m, 0.0) for m, _ in metric_vals)
            parts.append(num / den if den > 0 else float('inf'))
        else:
            parts.append(float(np.mean([v for _, v in metric_vals])))
    return float(np.mean(parts)) if parts else float('inf')


def run_one_cv(arch: str, variant: str, dataset, folds, seeds, hps,
                epochs, num_workers, out_dir: Path,
                use_uncertainty: bool,
                enabled_tasks: list = None,
                target_modes: dict = None,
                save_checkpoint: bool = True,
                exercise_aggregation: str = 'per_window'):
    """One pass of run_cv with given HPs."""
    cls = (RAW_REGISTRY if variant == 'raw' else FEAT_REGISTRY)[arch]
    factory = build_factory(cls, variant, dataset, hps)

    cfg_kwargs = dict(
        epochs=epochs, batch_size=hps['batch_size'], lr=hps['lr'],
        weight_decay=hps['weight_decay'], grad_clip=1.0,
        patience=hps.get('_patience', 5),
        mixed_precision=True, num_workers=num_workers,
        use_uncertainty_weighting=use_uncertainty,
        save_checkpoint=save_checkpoint,
        exercise_aggregation=exercise_aggregation,
    )
    if enabled_tasks is not None:
        cfg_kwargs['enabled_tasks'] = list(enabled_tasks)
    if target_modes is not None:
        cfg_kwargs['target_modes'] = dict(target_modes)
    cfg = TrainConfig(**cfg_kwargs)
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
    p.add_argument('--n-trials', type=int, default=50)
    p.add_argument('--phase1-epochs', type=int, default=50)
    p.add_argument('--phase2-epochs', type=int, default=150)
    p.add_argument('--phase2-seeds', type=int, nargs='+', default=[42, 1337, 7],
                   help='Default 3 seeds for variance estimation. LOSO with '
                        '1 seed is faster but loses seed-variance info.')
    p.add_argument('--patience', type=int, default=10,
                   help='Early-stopping patience (epochs without val improvement). '
                        'Phase 1 uses this; Phase 2 inherits it from the same flag.')
    p.add_argument('--tasks', nargs='+',
                   choices=['exercise', 'phase', 'fatigue', 'reps'],
                   default=['exercise', 'phase', 'fatigue', 'reps'],
                   help='Tasks contributing to the loss. Default: all 4 (multi-task). '
                        'Use --tasks fatigue for a fatigue-only model. Other heads '
                        'are still computed for evaluation but do not influence '
                        'gradients.')
    p.add_argument('--reps-mode',
                   choices=['hard', 'soft_window', 'soft_overlap'],
                   default='soft_window',
                   help='Rep-counting target representation. Default: '
                        'soft_window (ρ(t)=1/Δt_k integrated over window; '
                        'integer count recovered at eval via '
                        'src/eval/rep_aggregation.py). soft_overlap = '
                        'Wang et al. 2026 overlap-fraction labels (requires '
                        'scripts/add_soft_overlap_reps.py to have been run).')
    p.add_argument('--phase-mode',
                   choices=['hard', 'soft'],
                   default='soft',
                   help='Phase classification target representation. Default '
                        'soft (KL-div on per-window phase distribution).')
    p.add_argument('--window-s', type=float, default=2.0,
                   help='Window length in seconds. Hop = window/2 (50%% overlap). '
                        'soft_overlap reps column auto-selected to match. '
                        'Default 2.0 (matches raw_window_dataset legacy WINDOW_SIZE=200).')
    p.add_argument('--num-workers', type=int, default=None,
                   help='Default: 2 (features) or 8 (raw)')
    p.add_argument('--exclude-recordings', nargs='*', default=[],
                   help='Recording-id substrings to exclude (e.g. recording_003)')
    p.add_argument('--phase-whitelist', type=Path, default=None,
                   help='CSV of (recording_id, set_number) pairs whose '
                        'phase_label is clean enough to train on.')
    p.add_argument('--feature-prefixes', nargs='*', default=None,
                   help='Modality prefixes to keep in features dataset '
                        '(e.g. emg_ acc_ ppg_ temp_). Default: all. '
                        'Note: matches via startswith — `emg_dimitrov` '
                        'will also pick up `emg_dimitrov_rel`. Use '
                        '--feature-cols for exact-name matching.')
    p.add_argument('--feature-cols', nargs='*', default=None,
                   help='Exact column names to keep in features dataset. '
                        'Mutually exclusive with --feature-prefixes; takes '
                        'precedence if both are given.')
    p.add_argument('--no-uncertainty', action='store_true',
                   help='Disable uncertainty weighting (default: ON)')
    p.add_argument('--no-tight-hps', dest='tight_hps', action='store_false',
                   help='Use wider lr/weight_decay ranges. Architecture HPs '
                        '(dropout, repr_dim, channels, ...) are NEVER tuned '
                        '— model class defaults always win, so param count '
                        'stays constant across trials. Only lr, weight_decay, '
                        'and batch_size are searched in either mode.')
    p.set_defaults(tight_hps=True)
    p.add_argument('--seed-hps-from', type=Path, default=None,
                   help='Path to a best_hps.json from a prior Optuna run. '
                        'Loads its best_hps and enqueues them as Optuna '
                        'trial 0 — TPE then explores around them. Useful '
                        'for fine-tuning at a different window length '
                        'while reusing prior HP knowledge.')
    p.add_argument('--repr-dim-choices', type=int, nargs='+', default=None,
                   help='Restrict the Optuna repr_dim categorical to this '
                        'exact list (e.g. --repr-dim-choices 64 128). '
                        'Default: depends on --wide-arch-search.')
    p.add_argument('--wide-arch-search', action='store_true',
                   help='Open the architecture HP search: repr_dim '
                        '∈ {32,64,128,256} and dropout ∈ [0.1, 0.5]. '
                        'Default off — model class defaults used. Use this '
                        'for raw-variant sweeps where the default '
                        'repr_dim={16,32,64} ladder is too narrow for the '
                        'encoder backbone.')
    p.add_argument('--norm-mode',
                   choices=['baseline', 'robust', 'percentile'],
                   default='baseline',
                   help='Per-recording normalization mode. Applies to BOTH '
                        'raw and features paths. "baseline" = mean/std on '
                        'first 90s rest (legacy); "robust" = median+MAD on '
                        'full recording; "percentile" = center on baseline '
                        'mean, scale by 99th percentile of |x - center| '
                        '(equalizes max activation across subjects, '
                        'addresses exercise overfitting).')
    p.add_argument('--skip-phase2', action='store_true')
    p.add_argument('--include-rest', action='store_true',
                   help='Include rest windows (active_only=False).')
    p.add_argument('--exercise-aggregation',
                   choices=['per_window', 'per_set', 'both'],
                   default='per_window',
                   help='Exercise eval granularity. per_set aggregates '
                        'per-window predictions to one per (recording, set) '
                        'via mean-softmax — mirrors RPE supervision.')
    p.add_argument('--labeled-root', type=Path, default=Path('data/labeled'))
    p.add_argument('--runs-root', type=Path, default=Path('runs'))
    p.add_argument('--splits', type=Path,
                   default=Path('configs/splits_loso.csv'),
                   help='CV splits CSV. Default: LOSO (10 folds, 1 subject per '
                        'fold). Pass configs/splits.csv for the older 5-fold.')
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
    target_modes = {'reps': args.reps_mode, 'phase': args.phase_mode}
    if args.variant == 'features':
        feat_cols = None
        if args.feature_cols:
            import pandas as pd
            cols = set(pd.read_parquet(files[0]).columns)
            missing = [c for c in args.feature_cols if c not in cols]
            if missing:
                raise KeyError(
                    f"--feature-cols references missing columns: {missing}"
                )
            feat_cols = list(args.feature_cols)
            print(f"[optuna] feature_cols (exact)={feat_cols}")
        elif args.feature_prefixes:
            import pandas as pd
            cols = pd.read_parquet(files[0]).columns
            feat_cols = sorted(c for c in cols
                                if any(c.startswith(p) for p in args.feature_prefixes))
            print(f"[optuna] feature_prefixes={args.feature_prefixes} -> "
                  f"{len(feat_cols)} cols")
        dataset = WindowFeatureDataset(window_parquets=files,
                                         active_only=not args.include_rest,
                                         phase_whitelist=phase_wl,
                                         feature_cols=feat_cols,
                                         target_modes=target_modes,
                                         window_s=args.window_s,
                                         norm_mode=args.norm_mode)
    else:
        dataset = RawMultimodalWindowDataset(parquet_paths=files,
                                               active_only=not args.include_rest,
                                               phase_whitelist=phase_wl,
                                               target_modes=target_modes,
                                               window_s=args.window_s,
                                               norm_mode=args.norm_mode)
    print(f"[optuna] window_s={args.window_s}  target_modes={target_modes}  "
          f"enabled_tasks={args.tasks}")
    print(f"[optuna] {args.variant} dataset: {len(dataset)} windows")

    # Move dataset to GPU once and bypass DataLoader for the rest of the run.
    # Eliminates Windows worker-spawn overhead and per-batch CPU↔GPU transfer
    # — empirically 5-10× faster on this dataset's small size (~50 MB).
    if torch.cuda.is_available():
        import time as _time
        _t0 = _time.time()
        dataset.materialize_to_device("cuda")
        print(f"[optuna] dataset materialized to cuda in {_time.time()-_t0:.1f}s "
              f"(gpu_resident={dataset.gpu_resident})")

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
        hps = suggest_hps(trial, args.arch, tight=args.tight_hps,
                            wide_arch=args.wide_arch_search,
                            repr_dim_choices=args.repr_dim_choices)
        hps['_patience'] = args.patience
        trial_dir = run_dir / 'phase1' / f"trial_{trial.number:03d}"
        # Per-trial cache: if cv_summary already exists, reuse the score.
        cached = next(iter(trial_dir.rglob('cv_summary.json')), None)
        if cached is not None:
            try:
                cached_summary = json.loads(cached.read_text()).get('summary', {})
                score = score_summary(cached_summary, enabled_tasks=args.tasks)
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
                enabled_tasks=args.tasks, target_modes=target_modes,
                save_checkpoint=False,
                exercise_aggregation=args.exercise_aggregation,
            )
            score = score_summary(summary, enabled_tasks=args.tasks)
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
    # Warm-start: load best_hps from a prior run and enqueue as next trial.
    # TPE then explores around them. Only enqueue once (skip if any prior
    # trials already exist — assume they were seeded earlier).
    if args.seed_hps_from is not None and n_done == 0 and n_remaining > 0:
        try:
            seed_hps = json.loads(Path(args.seed_hps_from).read_text())
            seed_params = seed_hps.get('best_hps', seed_hps)
            study.enqueue_trial(seed_params)
            print(f"[optuna] Seeded trial 0 from {args.seed_hps_from}: "
                  f"{seed_params}")
        except Exception as e:
            print(f"[optuna] WARN: could not load seed HPs from "
                  f"{args.seed_hps_from}: {e}")
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, show_progress_bar=False)
    p1_elapsed = time.time() - t_start_p1
    print(f"[optuna] Phase 1 complete in {p1_elapsed/60:.1f} min")

    best_hps = study.best_params
    # Re-merge any architecture defaults that weren't sampled in this study
    sample = suggest_hps(optuna.trial.FixedTrial({**best_hps}), args.arch,
                          tight=args.tight_hps,
                          wide_arch=args.wide_arch_search,
                          repr_dim_choices=args.repr_dim_choices)
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
    best_hps_full['_patience'] = args.patience
    summary, _ = run_one_cv(
        args.arch, args.variant, dataset, folds, seeds=args.phase2_seeds,
        hps=best_hps_full, epochs=args.phase2_epochs,
        num_workers=args.num_workers, out_dir=p2_dir,
        use_uncertainty=use_uncertainty,
        enabled_tasks=args.tasks, target_modes=target_modes,
        exercise_aggregation=args.exercise_aggregation,
    )
    p2_elapsed = time.time() - t_start_p2
    print(f"[optuna] Phase 2 complete in {p2_elapsed/60:.1f} min")
    print(json.dumps(summary, indent=2))

    # ---- Pick best fold (lowest val_total) and copy checkpoint to top-level
    # so a single deployable model is available without digging through the
    # phase2/<arch>/seed_*/fold_* tree. Includes metadata about which subject
    # was held out and per-task scores at that fold.
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
        target = run_dir / 'best_model.pt'
        shutil.copy2(ckpt, target)
        meta = {
            'best_fold_dir': str(fold_dir.relative_to(run_dir)),
            'val_total': best_score,
            'test_subjects': m.get('test_subjects', []),
            'metrics': {k: m.get(k) for k in ('exercise', 'phase', 'fatigue',
                                               'reps') if k in m},
            'arch': args.arch, 'variant': args.variant,
            'window_s': args.window_s,
            'tight_hps': args.tight_hps,
            'best_hps': best_hps_full,
        }
        (run_dir / 'best_model_meta.json').write_text(
            json.dumps(meta, indent=2, default=_jsonable))
        print(f"[optuna] Best fold: {fold_dir.relative_to(run_dir)}  "
              f"val_total={best_score:.4f}  "
              f"test={m.get('test_subjects', '?')}")
        print(f"[optuna] Wrote {target} + best_model_meta.json")


def _jsonable(o):
    import numpy as _np
    if isinstance(o, (_np.floating, _np.integer)):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    return str(o)


if __name__ == '__main__':
    main()
