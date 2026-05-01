"""Full-depth GPU training entrypoint — features-only variant.

User request 2026-04-27: "bare kjør full økt på features ikke raw data"
- Variant A (engineered features) only; no raw-signal pathway.
- GPU via torch_device() -> RTX 5070 Ti (CUDA).
- No subsampling (subsample=False) — full data per fold.
- Epochs P1=50, P2=50, batch_size=256, Optuna 30 trials.
- Phase 1: seeds=[42], 4 archs.
- Phase 2: top 2-3 from P1, seeds=[42, 1337, 7].
- Baseline: runs/20260427_110653_default/metrics.json

Strategy to avoid CPU_* constants leaking in:
  We override run_variant's subsample kwarg to False at every call site, and
  pass explicit epochs/batch_size/patience kwargs instead of relying on the
  module-level CPU defaults. tune_hyperparams receives the device kwarg.
  The per_subject_metrics helper in train_nn.py still uses CPU_SUBSAMPLE_PER_FOLD
  internally — we re-implement it inline here (25 epochs, full data, GPU) to keep
  results credible.

References:
- Caruana 1997 — hard parameter sharing (shared encoder + 4 task heads)
- Ruder 2017 — soft vs hard sharing survey
- Bai et al. 2018 — TCN
- Hochreiter & Schmidhuber 1997 — LSTM
- Loshchilov & Hutter 2019 — AdamW optimizer
- Goodfellow et al. 2016 — regularization (dropout, batch norm)
- Saeb et al. 2017 — subject-wise cross-validation
- Akiba et al. 2019 — Optuna hyperparameter optimization
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings('ignore', category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.device import torch_device
from src.data.datasets import WindowFeatureDataset
from src.eval.metrics import compute_all_metrics
from src.training.losses import MultiTaskLoss
from src.pipeline.train_nn import (
    FilteredWindowDataset,
    load_baseline_splits,
    make_factory,
    train_one_fold,
    tune_hyperparams,
    latency_benchmark,
    run_soft_sharing_ablation,
    rank_variants,
    build_comparison,
    _aggregate,
    load_json,
    save_json,
    set_seed,
    FeatureNormalizer,
    FoldNormalizer,
    LOSS_WEIGHTS,
)

# ---- Paths -------------------------------------------------------------------
RUN_DIR = ROOT / 'runs/20260427_121303_nn_features_full'
WINDOW_FEATURES_PATH = ROOT / 'runs/20260427_110653_default/window_features.parquet'
LGBM_METRICS_PATH = ROOT / 'runs/20260427_110653_default/metrics.json'

# ---- Full-depth GPU knobs (no CPU compromises) -------------------------------
GPU_EPOCHS_P1 = 50
GPU_EPOCHS_P2 = 50
GPU_BATCH_SIZE = 256      # RTX 5070 Ti 16 GB can handle 512 on features MLP,
                          # but 256 is safer with TCN conv layers
GPU_PATIENCE = 8
GPU_OPTUNA_TRIALS = 30

ARCHS = ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']
SEEDS_P1 = [42]
SEEDS_P2 = [42, 1337, 7]
LOSS_WEIGHTS_LOCAL = LOSS_WEIGHTS  # {'exercise':1.0,'phase':1.0,'fatigue':1.0,'reps':0.5}


# =============================================================================
# Override: run_variant_gpu — wraps train_nn.run_variant but forces GPU knobs
# =============================================================================

def run_variant_gpu(
    variant_name: str,
    arch: str,
    dataset: FilteredWindowDataset,
    folds: List[Dict],
    seeds: List[int],
    epochs: int,
    device: torch.device,
    run_optuna: bool = True,
    optuna_trials: int = GPU_OPTUNA_TRIALS,
) -> Dict:
    """Run one arch variant, GPU-adapted (no subsampling, proper batch size).

    subsample=False is passed to run_variant; we also override CPU_OPTUNA_TRIALS
    by calling tune_hyperparams directly with optuna_trials override.
    """
    from src.pipeline import train_nn as _tn

    # Temporarily override the Optuna trial count used inside run_variant
    # by patching the module constant. This is cleaner than duplicating
    # the entire run_variant function.
    _original_trials = _tn.CPU_OPTUNA_TRIALS
    _tn.CPU_OPTUNA_TRIALS = optuna_trials

    variant_dir = RUN_DIR / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  VARIANT: {variant_name}  arch={arch}  seeds={seeds}")
    print(f"  subsample=False  epochs={epochs}  device={device}")
    print(f"  optuna_trials={optuna_trials}")
    print(f"{'='*60}")

    # HP tuning on fold 0 (largest training set)
    if run_optuna:
        print(f"  [HP tuning] Running {optuna_trials} Optuna trials on fold 0...")
        best_hp = tune_hyperparams(
            arch, dataset, folds[0],
            n_trials=optuna_trials,
            device=device,
        )
    else:
        # Load from Phase 1 results
        hp_path = RUN_DIR / variant_name.replace('_p2', '') / 'best_hp.json'
        best_hp = load_json(hp_path) if hp_path.exists() else {}
        print(f"  [HP] Reusing Phase 1 hp: {best_hp}")

    save_json(best_hp, variant_dir / 'best_hp.json')

    all_results = []
    for seed in seeds:
        set_seed(seed)
        for fold in folds:
            fold_id = fold['fold']
            fold_dir = variant_dir / f"seed_{seed}" / f"fold_{fold_id}"
            print(f"\n  seed={seed}  fold={fold_id}  "
                  f"test={fold['test_subjects']}  "
                  f"train_n={len(fold['train_idx'])}  test_n={len(fold['test_idx'])}")

            # NO subsampling — use full training split
            train_idx = fold['train_idx']

            factory = make_factory(arch, dataset.n_features, dataset.n_exercise,
                                    dataset.n_phase, best_hp)
            history, metrics = train_one_fold(
                factory, dataset, train_idx, fold['test_idx'],
                n_exercise=dataset.n_exercise, n_phase=dataset.n_phase,
                epochs=epochs,
                batch_size=GPU_BATCH_SIZE,
                lr=best_hp.get('lr', 1e-3),
                patience=GPU_PATIENCE,
                device=device,
                out_dir=fold_dir,
                verbose=True,
            )
            all_results.append({
                'seed': seed, 'fold': fold_id,
                'test_subjects': fold['test_subjects'],
                'metrics': metrics,
            })
            # Sanity check
            if len(history) > 1:
                first_loss = history[0]['train_loss']
                last_loss = history[-1]['train_loss']
                if last_loss >= first_loss * 0.99:
                    print(f"  WARNING: train loss did not decrease "
                          f"({first_loss:.4f} -> {last_loss:.4f})")

    summary = _aggregate(all_results)
    summary['variant'] = variant_name
    summary['arch'] = arch
    summary['best_hp'] = best_hp
    summary['n_folds'] = len(folds)
    summary['seeds'] = seeds
    summary['subsampled'] = False
    summary['subsample_n'] = None

    save_json({'summary': summary, 'all_results': all_results},
               variant_dir / 'cv_summary.json')
    print(f"\n  {variant_name} DONE.")
    print(f"  exercise F1={summary['exercise']['f1_macro']['mean']:.3f} "
          f"(std={summary['exercise']['f1_macro']['std']:.3f})")
    print(f"  phase F1  ={summary['phase']['f1_macro']['mean']:.3f} "
          f"(std={summary['phase']['f1_macro']['std']:.3f})")
    print(f"  fatigue MAE={summary['fatigue']['mae']['mean']:.3f} "
          f"(std={summary['fatigue']['mae']['std']:.3f})")
    print(f"  reps MAE   ={summary['reps']['mae']['mean']:.3f} "
          f"(std={summary['reps']['mae']['std']:.3f})")

    # Restore
    _tn.CPU_OPTUNA_TRIALS = _original_trials
    return summary


# =============================================================================
# Per-subject breakdown — GPU-adapted (full train data, 25 epochs)
# =============================================================================

def per_subject_metrics_gpu(
    arch: str,
    dataset: FilteredWindowDataset,
    fold: Dict,
    best_hp: Dict,
    device: torch.device,
    epochs: int = 25,
) -> pd.DataFrame:
    """Train on fold train split (full, no subsampling), evaluate per-subject."""
    train_idx = fold['train_idx']   # full split — no CPU_SUBSAMPLE_PER_FOLD

    feat_norm = FeatureNormalizer(dataset, train_idx)
    tgt_norm = FoldNormalizer(dataset, train_idx)

    factory = make_factory(arch, dataset.n_features, dataset.n_exercise,
                            dataset.n_phase, best_hp)
    model = factory().to(device)
    set_seed(42)
    opt = torch.optim.AdamW(model.parameters(),
                              lr=best_hp.get('lr', 1e-3),
                              weight_decay=1e-4)
    loss_fn = MultiTaskLoss(
        w_exercise=LOSS_WEIGHTS_LOCAL['exercise'],
        w_phase=LOSS_WEIGHTS_LOCAL['phase'],
        w_fatigue=LOSS_WEIGHTS_LOCAL['fatigue'],
        w_reps=LOSS_WEIGHTS_LOCAL['reps'],
    ).to(device)
    loader = DataLoader(Subset(dataset, train_idx),
                         batch_size=GPU_BATCH_SIZE,
                         shuffle=True, num_workers=0, drop_last=True)

    model.train()
    for ep in range(epochs):
        for batch in loader:
            x = feat_norm.transform(batch['x'].to(device))
            tgt = tgt_norm.normalize_targets(
                {k: v.to(device) for k, v in batch['targets'].items()})
            msk = {k: v.to(device) for k, v in batch['masks'].items()}
            opt.zero_grad(set_to_none=True)
            preds = model(x)
            total, _ = loss_fn(preds, tgt, msk)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if (ep + 1) % 5 == 0:
            print(f"    per-subject training: epoch {ep+1}/{epochs}")

    # Evaluate per-subject
    test_loader = DataLoader(Subset(dataset, fold['test_idx']),
                              batch_size=512, shuffle=False, num_workers=0)
    subject_ids_arr = np.array(dataset.subject_ids)[fold['test_idx']]

    all_p = {'exercise': [], 'phase': [], 'fatigue': [], 'reps': []}
    all_t = {k: [] for k in all_p}
    all_m = {k: [] for k in all_p}
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x = feat_norm.transform(batch['x'].to(device))
            preds = model(x)
            preds_d = dict(preds)
            preds_d['fatigue'] = torch.clamp(
                preds['fatigue'] * tgt_norm.fat_std + tgt_norm.fat_mean, 0, 15)
            preds_d['reps'] = torch.clamp(
                preds['reps'] * tgt_norm.rep_std + tgt_norm.rep_mean, 0, 40)
            for k in all_p:
                all_p[k].append(preds_d[k].cpu())
                all_t[k].append(batch['targets'][k])
                all_m[k].append(batch['masks'][k])

    cat_p = {k: torch.cat(v) for k, v in all_p.items()}
    cat_t = {k: torch.cat(v) for k, v in all_t.items()}
    cat_m = {k: torch.cat(v) for k, v in all_m.items()}
    for k in ('fatigue', 'reps'):
        bad = ~torch.isfinite(cat_p[k])
        if bad.any():
            cat_p[k] = torch.where(bad, torch.zeros_like(cat_p[k]), cat_p[k])
            cat_m[k] = torch.where(bad, torch.zeros_like(cat_m[k], dtype=torch.bool), cat_m[k])

    rows = []
    for s in np.unique(subject_ids_arr):
        mask = subject_ids_arr == s
        sp = {k: v[mask] for k, v in cat_p.items()}
        st = {k: v[mask] for k, v in cat_t.items()}
        sm = {k: v[mask] for k, v in cat_m.items()}
        m = compute_all_metrics(sp, st, sm,
                                 n_exercise=dataset.n_exercise,
                                 n_phase=dataset.n_phase)
        rows.append({
            'subject_id': s,
            'exercise_f1': m['exercise']['f1_macro'],
            'phase_f1': m['phase']['f1_macro'],
            'fatigue_mae': m['fatigue']['mae'],
            'reps_mae': m['reps']['mae'],
        })
    return pd.DataFrame(rows)


# =============================================================================
# Latency: GPU + CPU (for deployment table)
# =============================================================================

def latency_both_devices(
    dataset: FilteredWindowDataset,
    gpu_device: torch.device,
) -> Dict:
    """Measure p50/p95/p99/mean on both GPU and CPU for all 4 archs."""
    latency = {}
    for arch in ARCHS:
        hp_path = RUN_DIR / f"features_{arch}" / 'best_hp.json'
        hp = load_json(hp_path) if hp_path.exists() else {}
        factory = make_factory(arch, dataset.n_features, dataset.n_exercise,
                                dataset.n_phase, hp)
        gpu_lat = latency_benchmark(factory, dataset.n_features,
                                     n_warmup=20, n_runs=200,
                                     device=gpu_device)
        cpu_lat = latency_benchmark(factory, dataset.n_features,
                                     n_warmup=20, n_runs=200,
                                     device=torch.device('cpu'))
        latency[arch] = {'gpu': gpu_lat, 'cpu': cpu_lat}
        print(f"  {arch}: GPU p99={gpu_lat['p99_ms']:.2f}ms  "
              f"CPU p99={cpu_lat['p99_ms']:.2f}ms")
    return latency


# =============================================================================
# Main
# =============================================================================

def main():
    t_start = time.time()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    device_str = torch_device()
    device = torch.device(device_str)
    print(f"\n{'='*70}")
    print(f"run_train_nn_features_full — GPU FULL-DEPTH RUN")
    print(f"  Device: {device_str}")
    print(f"  Run dir: {RUN_DIR}")
    print(f"  Window features: {WINDOW_FEATURES_PATH}")
    print(f"  LightGBM baseline: {LGBM_METRICS_PATH}")
    print(f"  Epochs P1={GPU_EPOCHS_P1} P2={GPU_EPOCHS_P2}  "
          f"batch={GPU_BATCH_SIZE}  patience={GPU_PATIENCE}")
    print(f"  Optuna trials: {GPU_OPTUNA_TRIALS}")
    print(f"  Subsample: FALSE (full data)")
    print(f"{'='*70}\n")

    # ---- Verify preconditions ------------------------------------------------
    if not WINDOW_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Window features not found: {WINDOW_FEATURES_PATH}\n"
            "Run /train first to produce the LightGBM baseline."
        )
    if not LGBM_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"LightGBM metrics not found: {LGBM_METRICS_PATH}\n"
            "Run /train first."
        )

    # ---- Load dataset --------------------------------------------------------
    print("Loading window features...")
    base_ds = WindowFeatureDataset(
        [WINDOW_FEATURES_PATH],
        active_only=False,
        verbose=True,
    )
    dataset = FilteredWindowDataset(base_ds)
    print(f"Dataset: {len(dataset)} windows, {dataset.n_features} features, "
          f"{dataset.n_exercise} exercise classes, "
          f"{dataset.n_phase} phase classes")
    print(f"Phase classes (known only): {dataset.known_phase_classes}")

    # ---- Load splits ---------------------------------------------------------
    folds = load_baseline_splits(dataset)

    # =========================================================================
    # PHASE 1: Screening — 4 archs, features-only, 1 seed, full data
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: Screening — 4 archs × features, 1 seed, GPU, NO subsampling")
    print(f"  epochs={GPU_EPOCHS_P1}  optuna_trials={GPU_OPTUNA_TRIALS}")
    print("="*70)

    phase1_results = {}
    for arch in ARCHS:
        vname = f"features_{arch}"
        summary = run_variant_gpu(
            variant_name=vname,
            arch=arch,
            dataset=dataset,
            folds=folds,
            seeds=SEEDS_P1,
            epochs=GPU_EPOCHS_P1,
            device=device,
            run_optuna=True,
            optuna_trials=GPU_OPTUNA_TRIALS,
        )
        phase1_results[vname] = summary

    ranking = rank_variants(phase1_results)
    save_json({'phase1_results': phase1_results, 'ranking': ranking},
               RUN_DIR / 'phase1_summary.json')
    top_variants = [r['variant'] for r in ranking[:3]]
    print(f"\nPhase 1 complete. Top-3 variants: {top_variants}")
    for r in ranking:
        print(f"  {r['variant']:25s}  mean_rank={r['mean_rank']:.2f}  "
              f"ex_F1={r.get('exercise_f1_macro', float('nan')):.3f}  "
              f"ph_F1={r.get('phase_f1_macro', float('nan')):.3f}  "
              f"fat_MAE={r.get('fatigue_mae', float('nan')):.3f}  "
              f"rep_MAE={r.get('reps_mae', float('nan')):.3f}")

    t_p1 = time.time() - t_start
    print(f"\nPhase 1 wall-clock: {t_p1/3600:.2f} h")

    # =========================================================================
    # PHASE 2: Final depth — top 2-3 variants, 3 seeds, GPU, NO subsampling
    # =========================================================================
    print("\n" + "="*70)
    print(f"PHASE 2: Final depth — {top_variants}, 3 seeds, GPU, NO subsampling")
    print(f"  epochs={GPU_EPOCHS_P2}")
    print("="*70)

    phase2_results = {}
    for vname in top_variants:
        arch = vname.split('_', 1)[1]
        p2name = f"{vname}_p2"
        summary = run_variant_gpu(
            variant_name=p2name,
            arch=arch,
            dataset=dataset,
            folds=folds,
            seeds=SEEDS_P2,
            epochs=GPU_EPOCHS_P2,
            device=device,
            run_optuna=False,   # reuse Phase 1 HP
            optuna_trials=GPU_OPTUNA_TRIALS,
        )
        phase2_results[vname] = summary

    save_json(phase2_results, RUN_DIR / 'phase2_summary.json')

    t_p2 = time.time() - t_start
    print(f"\nPhase 2 wall-clock: {(t_p2 - t_p1)/3600:.2f} h")

    # =========================================================================
    # LATENCY BENCHMARKS — GPU + CPU
    # =========================================================================
    print("\nRunning latency benchmarks (GPU + CPU)...")
    latency = latency_both_devices(dataset, device)
    save_json(latency, RUN_DIR / 'latency.json')

    # =========================================================================
    # SOFT-SHARING ABLATION — winner from Phase 2
    # =========================================================================
    winner_variant = top_variants[0]
    winner_arch = winner_variant.split('_', 1)[1]
    winner_hp = load_json(RUN_DIR / winner_variant / 'best_hp.json')
    print(f"\nSoft-sharing ablation on winner: {winner_variant} ({winner_arch})")
    ablation = run_soft_sharing_ablation(
        arch=winner_arch,
        dataset=dataset,
        folds=folds[:2],   # 2 folds is enough for ablation signal
        run_dir=RUN_DIR,
        device=device,
        best_hp=winner_hp,
    )

    # =========================================================================
    # PER-SUBJECT BREAKDOWN — winner, fold 0
    # =========================================================================
    print(f"\nPer-subject breakdown for {winner_variant} (fold 0)...")
    per_sub_df = per_subject_metrics_gpu(
        arch=winner_arch,
        dataset=dataset,
        fold=folds[0],
        best_hp=winner_hp,
        device=device,
        epochs=25,
    )
    per_sub_df.to_csv(RUN_DIR / 'per_subject_breakdown.csv', index=False)
    print("Per-subject results:")
    print(per_sub_df.to_string(index=False))

    # Sanity: check for catastrophic failures (exercise F1 < 0.1 when median > 0.3)
    ex_f1_vals = per_sub_df['exercise_f1'].dropna()
    if len(ex_f1_vals) > 0:
        median_ex = ex_f1_vals.median()
        catastrophic = per_sub_df[per_sub_df['exercise_f1'] < 0.1]
        if median_ex > 0.3 and len(catastrophic) > 0:
            print(f"  WARNING: catastrophic per-subject failure for: "
                  f"{catastrophic['subject_id'].tolist()}")

    # =========================================================================
    # FINAL METRICS + COMPARISON
    # =========================================================================
    lgbm_metrics = load_json(LGBM_METRICS_PATH)
    # build_comparison expects lgbm_metrics in the format from train_nn:
    # {exercise: {mean:...}, phase: {mean:...}, fatigue: {mean:...}, reps: {mean:...}}
    lgbm_compat = {
        'exercise': {'mean': lgbm_metrics['exercise']['f1_mean']},
        'phase':    {'mean': lgbm_metrics['phase']['ml_f1_mean']},
        'fatigue':  {'mean': lgbm_metrics['fatigue']['mae_mean']},
        'reps':     {'mean': lgbm_metrics['reps']['ml_mae_mean']},
    }
    comparison = build_comparison(phase2_results, lgbm_compat, latency)
    save_json(comparison, RUN_DIR / 'final_metrics.json')

    # All results bundle
    save_json({
        'phase1': phase1_results,
        'phase2': phase2_results,
        'ranking': ranking,
        'latency': latency,
        'comparison': comparison,
        'ablation': ablation,
        'winner': winner_variant,
        'top_variants': top_variants,
        'lgbm_baseline': lgbm_compat,
        'device': device_str,
        'gpu_knobs': {
            'epochs_p1': GPU_EPOCHS_P1,
            'epochs_p2': GPU_EPOCHS_P2,
            'batch_size': GPU_BATCH_SIZE,
            'patience': GPU_PATIENCE,
            'optuna_trials': GPU_OPTUNA_TRIALS,
            'subsampled': False,
        },
    }, RUN_DIR / 'all_results.json')

    t_total = time.time() - t_start
    print(f"\nTotal wall-clock: {t_total/3600:.2f} h")

    return {
        'run_dir': str(RUN_DIR),
        'winner': winner_variant,
        'top_variants': top_variants,
        'phase1': phase1_results,
        'phase2': phase2_results,
        'latency': latency,
        'comparison': comparison,
        'ablation': ablation,
        'wall_clock_h': t_total / 3600,
    }


if __name__ == '__main__':
    results = main()
    print("\n\nAll done. Run directory:", RUN_DIR)
    print("Winner:", results['winner'])
    print("Top variants:", results['top_variants'])
