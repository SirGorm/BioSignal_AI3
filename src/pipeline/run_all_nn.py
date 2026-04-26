"""Complete /train-nn pipeline.

Runs Phase 1 (4 archs × features, 5-fold, 1 seed) then Phase 2 (top-3, 3 seeds),
soft-sharing ablation, latency benchmarks, per-subject breakdown.

All input features are z-score normalized with IQR-based std on train split to
handle acc_rms/acc_jerk_rms overflow values (~1e38) from the feature extractor.
Regression targets (fatigue/reps) are normalized to unit scale on each fold.

References:
- Caruana 1997 — hard parameter sharing
- Saeb et al. 2017 — subject-wise CV
- Goodfellow et al. 2016 — feature normalization for neural networks
- Loshchilov & Hutter 2019 — AdamW optimizer
- Bai et al. 2018 — TCN
- Hochreiter & Schmidhuber 1997 — LSTM
"""
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pipeline.train_nn import (
    FilteredWindowDataset, load_baseline_splits,
    make_factory, train_one_fold, subsample_train_idx,
    run_variant, _aggregate, rank_variants, latency_benchmark,
    per_subject_metrics, build_comparison, run_soft_sharing_ablation,
    load_json, save_json, set_seed,
    CPU_SUBSAMPLE_PER_FOLD, CPU_EPOCHS_P1, CPU_EPOCHS_P2,
    BATCH_SIZE, PATIENCE, tune_hyperparams
)
from src.data.datasets import WindowFeatureDataset
import torch, numpy as np

RUN_DIR = ROOT / 'runs/20260426_160754_nn_comparison'
device = torch.device('cpu')

print(f"GPU: not available — CPU-only PyTorch")
print(f"CPU-adapted: subsample={CPU_SUBSAMPLE_PER_FOLD}, "
      f"epochs_p1={CPU_EPOCHS_P1}, epochs_p2={CPU_EPOCHS_P2}")
print()

# ---- Load dataset
print("Loading dataset...")
base_ds = WindowFeatureDataset(
    [ROOT / 'runs/20260426_154705_default/features/window_features.parquet'],
    active_only=True, verbose=True,
)
dataset = FilteredWindowDataset(base_ds)
folds = load_baseline_splits(dataset)
print(f"Dataset: {len(dataset)} windows, {dataset.n_features} features, "
      f"{dataset.n_exercise} exercise classes, "
      f"{dataset.n_phase} phase classes (unknown excluded)")

# ===================================================================
# PHASE 1: Screen all 4 architectures × features input
# ===================================================================
print("\n" + "="*70)
print("PHASE 1: Screening — 4 archs × features, 5 folds, 1 seed")
print("="*70)

ARCHS = ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']
phase1_results = {}
for arch in ARCHS:
    vname = f'features_{arch}'
    summary = run_variant(
        variant_name=vname, arch=arch, dataset=dataset, folds=folds,
        out_root=RUN_DIR, seeds=[42], epochs=CPU_EPOCHS_P1,
        subsample=True, device=device, run_optuna=True,
    )
    phase1_results[vname] = summary

# ---- Rank and identify top 3
ranking = rank_variants(phase1_results)
save_json({'phase1_results': phase1_results, 'ranking': ranking},
           RUN_DIR / 'phase1_summary.json')
top_variants = [r['variant'] for r in ranking[:3]]

print(f"\nPhase 1 ranking:")
for i, r in enumerate(ranking):
    print(f"  {i+1}. {r['variant']}: "
          f"exF1={r.get('exercise_f1_macro', float('nan')):.3f}  "
          f"phF1={r.get('phase_f1_macro', float('nan')):.3f}  "
          f"fatMAE={r.get('fatigue_mae', float('nan')):.3f}  "
          f"repMAE={r.get('reps_mae', float('nan')):.3f}  "
          f"mean_rank={r.get('mean_rank', float('nan')):.2f}")
print(f"Top-3: {top_variants}")

# ===================================================================
# PHASE 2: Final depth — top 3 variants, 3 seeds
# ===================================================================
print("\n" + "="*70)
print(f"PHASE 2: Final depth on {top_variants}, 3 seeds")
print("="*70)

phase2_results = {}
for vname in top_variants:
    arch = vname.split('_', 1)[1]
    best_hp = load_json(RUN_DIR / vname / 'best_hp.json')
    summary = run_variant(
        variant_name=f"{vname}_p2", arch=arch, dataset=dataset, folds=folds,
        out_root=RUN_DIR, seeds=[42, 1337, 7], epochs=CPU_EPOCHS_P2,
        subsample=True, device=device, run_optuna=False,
    )
    phase2_results[vname] = summary

# ===================================================================
# SOFT-SHARING ABLATION (winner only)
# ===================================================================
winner_variant = top_variants[0]
winner_arch = winner_variant.split('_', 1)[1]
winner_hp = load_json(RUN_DIR / winner_variant / 'best_hp.json')
print(f"\n=== Soft-sharing ablation: {winner_variant} ===")
ablation = run_soft_sharing_ablation(
    arch=winner_arch, dataset=dataset, folds=folds[:2],
    run_dir=RUN_DIR, device=device, best_hp=winner_hp,
)

# ===================================================================
# LATENCY BENCHMARKS
# ===================================================================
print("\n=== Latency benchmarks ===")
latency = {}
for arch in ARCHS:
    hp = load_json(RUN_DIR / f'features_{arch}' / 'best_hp.json')
    factory = make_factory(arch, dataset.n_features, dataset.n_exercise,
                            dataset.n_phase, hp)
    lt = latency_benchmark(factory, dataset.n_features, device=device)
    latency[arch] = lt
    print(f"  {arch}: p50={lt['p50_ms']:.1f}  p95={lt['p95_ms']:.1f}  "
          f"p99={lt['p99_ms']:.1f}  mean={lt['mean_ms']:.1f} ms")
save_json(latency, RUN_DIR / 'latency.json')

# ===================================================================
# PER-SUBJECT BREAKDOWN (winner, fold 0)
# ===================================================================
winner_hp['arch'] = winner_arch
print(f"\n=== Per-subject breakdown: {winner_variant}, fold 0 ===")
per_sub_df = per_subject_metrics(None, dataset, folds[0], winner_hp, device)
per_sub_df.to_csv(RUN_DIR / 'per_subject_breakdown.csv', index=False)
print(per_sub_df.to_string(index=False))

# ===================================================================
# FINAL COMPARISON
# ===================================================================
lgbm_metrics = load_json(ROOT / 'runs/20260426_154705_default/metrics.json')
comparison = build_comparison(phase2_results, lgbm_metrics, latency)
save_json(comparison, RUN_DIR / 'final_metrics.json')

# Save all results bundle
save_json({
    'phase1': phase1_results, 'phase2': phase2_results,
    'ranking': ranking, 'latency': latency,
    'ablation': ablation, 'winner': winner_variant,
    'top_variants': top_variants,
}, RUN_DIR / 'all_results.json')

print("\n\nRUN COMPLETE. Directory:", RUN_DIR)
print("Winner:", winner_variant)
print("\nPhase 2 summary:")
for vname, s in phase2_results.items():
    print(f"  {vname}_p2: exF1={s['exercise']['f1_macro']['mean']:.3f}±{s['exercise']['f1_macro']['std']:.3f}  "
          f"phF1={s['phase']['f1_macro']['mean']:.3f}  "
          f"fatMAE={s['fatigue']['mae']['mean']:.3f}  "
          f"repMAE={s['reps']['mae']['mean']:.3f}")
