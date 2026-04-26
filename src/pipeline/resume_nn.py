"""Resume script: runs TCN folds 1-4, then Phase 2 and remaining artifacts."""
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
    BATCH_SIZE, PATIENCE
)
from src.data.datasets import WindowFeatureDataset
import torch, numpy as np

RUN_DIR = ROOT / 'runs/20260426_160754_nn_comparison'
device = torch.device('cpu')

# ---- Load dataset
print("Loading dataset...")
base_ds = WindowFeatureDataset(
    [ROOT / 'runs/20260426_154705_default/features/window_features.parquet'],
    active_only=True, verbose=False,
)
dataset = FilteredWindowDataset(base_ds)
folds = load_baseline_splits(dataset)
print(f"Dataset: {len(dataset)} windows, {dataset.n_features} features")

# ---- Complete TCN Phase 1 (folds 1-4 missing)
print("\n=== Completing TCN Phase 1 ===")
tcn_dir = RUN_DIR / 'features_tcn'
tcn_hp = load_json(tcn_dir / 'best_hp.json')
print(f"TCN best HP: {tcn_hp}")

# Load fold_0 result
all_results_tcn = []
fold0_dir = tcn_dir / 'seed_42' / 'fold_0'
if (fold0_dir / 'metrics.json').exists():
    m0 = load_json(fold0_dir / 'metrics.json')
    all_results_tcn.append({'seed': 42, 'fold': 0,
                              'test_subjects': folds[0]['test_subjects'],
                              'metrics': m0})
    print(f"  Fold 0 loaded from disk: ex_F1={m0['exercise']['f1_macro']:.3f}")

for fold in folds[1:]:  # folds 1-4
    fold_id = fold['fold']
    fold_dir = tcn_dir / 'seed_42' / f'fold_{fold_id}'
    print(f"\n  TCN fold {fold_id} test={fold['test_subjects']}")
    set_seed(42)
    train_idx = subsample_train_idx(fold['train_idx'], dataset,
                                     CPU_SUBSAMPLE_PER_FOLD, seed=42)
    print(f"    Subsampled train: {len(train_idx)}")
    factory = make_factory('tcn', dataset.n_features, dataset.n_exercise,
                            dataset.n_phase, tcn_hp)
    history, metrics = train_one_fold(
        factory, dataset, train_idx, fold['test_idx'],
        n_exercise=dataset.n_exercise, n_phase=dataset.n_phase,
        epochs=CPU_EPOCHS_P1, batch_size=BATCH_SIZE,
        lr=tcn_hp.get('lr', 1e-3),
        patience=PATIENCE, device=device,
        out_dir=fold_dir, verbose=True,
    )
    all_results_tcn.append({'seed': 42, 'fold': fold_id,
                              'test_subjects': fold['test_subjects'],
                              'metrics': metrics})

summary_tcn = _aggregate(all_results_tcn)
summary_tcn.update({'variant': 'features_tcn', 'arch': 'tcn',
                     'best_hp': tcn_hp, 'n_folds': 5, 'seeds': [42],
                     'subsampled': True, 'subsample_n': CPU_SUBSAMPLE_PER_FOLD})
save_json({'summary': summary_tcn, 'all_results': all_results_tcn},
           tcn_dir / 'cv_summary.json')
print(f"\n  TCN DONE: ex_F1={summary_tcn['exercise']['f1_macro']['mean']:.3f}  "
      f"ph_F1={summary_tcn['phase']['f1_macro']['mean']:.3f}  "
      f"fat_MAE={summary_tcn['fatigue']['mae']['mean']:.3f}  "
      f"rep_MAE={summary_tcn['reps']['mae']['mean']:.3f}")

# ---- Compile Phase 1 results
phase1_results = {}
for arch in ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']:
    vname = f'features_{arch}'
    summary = load_json(RUN_DIR / vname / 'cv_summary.json').get('summary', {})
    phase1_results[vname] = summary

ranking = rank_variants(phase1_results)
save_json({'phase1_results': phase1_results, 'ranking': ranking},
           RUN_DIR / 'phase1_summary.json')
top_variants = [r['variant'] for r in ranking[:3]]
print(f"\nPhase 1 ranking:")
for i, r in enumerate(ranking):
    print(f"  {i+1}. {r['variant']}: ex={r.get('exercise_f1_macro', 'N/A'):.3f} "
          f"ph={r.get('phase_f1_macro', 'N/A'):.3f} "
          f"fat={r.get('fatigue_mae', 'N/A'):.3f} "
          f"rep={r.get('reps_mae', 'N/A'):.3f}  "
          f"mean_rank={r.get('mean_rank', 'N/A'):.2f}")
print(f"Top-3: {top_variants}")

# ---- Phase 2: 3 seeds on top-3 variants
print(f"\n=== Phase 2: {top_variants} ===")
phase2_results = {}
for vname in top_variants:
    arch = vname.split('_', 1)[1]
    # Reuse phase 1 HP
    best_hp = load_json(RUN_DIR / vname / 'best_hp.json')
    summary = run_variant(
        variant_name=f"{vname}_p2",
        arch=arch,
        dataset=dataset,
        folds=folds,
        out_root=RUN_DIR,
        seeds=[42, 1337, 7],
        epochs=CPU_EPOCHS_P2,
        subsample=True,
        device=device,
        run_optuna=False,
    )
    phase2_results[vname] = summary

# ---- Soft-sharing ablation (winner)
winner_variant = top_variants[0]
winner_arch = winner_variant.split('_', 1)[1]
winner_hp = load_json(RUN_DIR / winner_variant / 'best_hp.json')
print(f"\n=== Soft-sharing ablation on {winner_variant} ===")
ablation = run_soft_sharing_ablation(
    arch=winner_arch,
    dataset=dataset,
    folds=folds[:2],
    run_dir=RUN_DIR,
    device=device,
    best_hp=winner_hp,
)

# ---- Latency benchmarks
print("\n=== Latency benchmarks ===")
latency = {}
ARCHS = ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']
for arch in ARCHS:
    hp = load_json(RUN_DIR / f'features_{arch}' / 'best_hp.json')
    factory = make_factory(arch, dataset.n_features, dataset.n_exercise,
                            dataset.n_phase, hp)
    lt = latency_benchmark(factory, dataset.n_features, device=device)
    latency[arch] = lt
    print(f"  {arch}: p99={lt['p99_ms']:.2f} ms  mean={lt['mean_ms']:.2f} ms")

save_json(latency, RUN_DIR / 'latency.json')

# ---- Per-subject breakdown
winner_hp['arch'] = winner_arch
print(f"\n=== Per-subject breakdown ({winner_variant}) ===")
per_sub_df = per_subject_metrics(None, dataset, folds[0], winner_hp, device)
per_sub_df.to_csv(RUN_DIR / 'per_subject_breakdown.csv', index=False)
print(per_sub_df.to_string(index=False))

# ---- Final comparison
lgbm_metrics = load_json(ROOT / 'runs/20260426_154705_default/metrics.json')
comparison = build_comparison(phase2_results, lgbm_metrics, latency)
save_json(comparison, RUN_DIR / 'final_metrics.json')
save_json({'phase1': phase1_results, 'phase2': phase2_results,
            'ranking': ranking, 'latency': latency,
            'ablation': ablation,
            'winner': winner_variant},
           RUN_DIR / 'all_results.json')

print("\n\nRESUME COMPLETE. Run directory:", RUN_DIR)
