"""Generate final artifacts: per-subject breakdown, comparison files, model cards.

Run after all Phase 1 and Phase 2 training is complete.
"""
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pipeline.train_nn import (
    FilteredWindowDataset, load_baseline_splits,
    make_factory, train_one_fold, subsample_train_idx,
    _aggregate, rank_variants, latency_benchmark,
    build_comparison, run_soft_sharing_ablation,
    load_json, save_json, set_seed,
    CPU_SUBSAMPLE_PER_FOLD, CPU_EPOCHS_P2, BATCH_SIZE, PATIENCE,
    FeatureNormalizer, FoldNormalizer, LOSS_WEIGHTS,
    MultiTaskLoss
)
from src.data.datasets import WindowFeatureDataset
from src.eval.metrics import compute_all_metrics
import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Subset

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

# ---- Load ablation and latency from log (already computed)
ablation_dir = RUN_DIR / 'ablation_soft_sharing_tcn'
if ablation_dir.exists() and (ablation_dir / 'ablation_results.json').exists():
    ablation = load_json(ablation_dir / 'ablation_results.json')
    print("Ablation loaded:", ablation.get('verdict', 'N/A'))
else:
    print("Ablation not found, running...")
    winner_hp = load_json(RUN_DIR / 'features_tcn' / 'best_hp.json')
    ablation = run_soft_sharing_ablation(
        arch='tcn', dataset=dataset, folds=folds[:2],
        run_dir=RUN_DIR, device=device, best_hp=winner_hp,
    )

# ---- Latency (already computed, reload if exists)
if (RUN_DIR / 'latency.json').exists():
    latency = load_json(RUN_DIR / 'latency.json')
    print("Latency loaded:", {k: '%.1f ms p99' % v['p99_ms'] for k, v in latency.items()})
else:
    print("Running latency benchmarks...")
    latency = {}
    for arch in ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']:
        hp = load_json(RUN_DIR / f'features_{arch}' / 'best_hp.json')
        factory = make_factory(arch, dataset.n_features, dataset.n_exercise, dataset.n_phase, hp)
        lt = latency_benchmark(factory, dataset.n_features, device=device)
        latency[arch] = lt
        print(f"  {arch}: p99={lt['p99_ms']:.2f} ms")
    save_json(latency, RUN_DIR / 'latency.json')

# ---- Per-subject breakdown (TCN winner, fold 0, with proper normalization)
print("\nPer-subject breakdown (TCN, fold 0)...")
winner_hp = load_json(RUN_DIR / 'features_tcn' / 'best_hp.json')
winner_hp['arch'] = 'tcn'

fold = folds[0]
train_idx = subsample_train_idx(fold['train_idx'], dataset, 20_000, seed=42)
feat_norm = FeatureNormalizer(dataset, train_idx)
tgt_norm = FoldNormalizer(dataset, train_idx)

factory = make_factory('tcn', dataset.n_features, dataset.n_exercise, dataset.n_phase, winner_hp)
model = factory().to(device)
set_seed(42)
opt = torch.optim.AdamW(model.parameters(), lr=winner_hp.get('lr', 1e-3), weight_decay=1e-4)
loss_fn = MultiTaskLoss(w_exercise=1.0, w_phase=1.0, w_fatigue=1.0, w_reps=0.5).to(device)
loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE,
                     shuffle=True, num_workers=0, drop_last=True)

print("  Quick training (15 epochs)...")
for ep in range(15):
    model.train()
    for batch in loader:
        x = feat_norm.transform(batch['x'].to(device))
        tgt = tgt_norm.normalize_targets({k: v.to(device) for k, v in batch['targets'].items()})
        msk = {k: v.to(device) for k, v in batch['masks'].items()}
        opt.zero_grad(set_to_none=True)
        preds = model(x)
        total, _ = loss_fn(preds, tgt, msk)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

# Evaluate per-subject
test_loader = DataLoader(Subset(dataset, fold['test_idx']),
                          batch_size=512, shuffle=False, num_workers=0)
subject_ids_test = np.array(dataset.subject_ids)[fold['test_idx']]
all_p = {'exercise': [], 'phase': [], 'fatigue': [], 'reps': []}
all_t = {k: [] for k in all_p}
all_m = {k: [] for k in all_p}
model.eval()
with torch.no_grad():
    for batch in test_loader:
        x = feat_norm.transform(batch['x'].to(device))
        preds = model(x)
        # Denormalize regression
        preds_d = dict(preds)
        preds_d['fatigue'] = torch.clamp(preds['fatigue'] * tgt_norm.fat_std + tgt_norm.fat_mean, 0, 15)
        preds_d['reps'] = torch.clamp(preds['reps'] * tgt_norm.rep_std + tgt_norm.rep_mean, 0, 40)
        for k in all_p:
            all_p[k].append(preds_d[k].cpu())
            all_t[k].append(batch['targets'][k])
            all_m[k].append(batch['masks'][k])

cat_p = {k: torch.cat(v) for k, v in all_p.items()}
cat_t = {k: torch.cat(v) for k, v in all_t.items()}
cat_m = {k: torch.cat(v) for k, v in all_m.items()}

# NaN-safe
for k in ('fatigue', 'reps'):
    p = cat_p[k]
    bad = ~torch.isfinite(p)
    if bad.any():
        cat_p[k] = torch.where(bad, torch.zeros_like(p), p)
        cat_m[k] = torch.where(bad, torch.zeros_like(cat_m[k], dtype=torch.bool), cat_m[k])

rows = []
for s in np.unique(subject_ids_test):
    mask = subject_ids_test == s
    sp = {k: v[mask] for k, v in cat_p.items()}
    st = {k: v[mask] for k, v in cat_t.items()}
    sm = {k: v[mask] for k, v in cat_m.items()}
    m = compute_all_metrics(sp, st, sm, n_exercise=dataset.n_exercise, n_phase=dataset.n_phase)
    rows.append({
        'subject_id': s,
        'exercise_f1': m['exercise']['f1_macro'],
        'phase_f1': m['phase']['f1_macro'],
        'fatigue_mae': m['fatigue']['mae'],
        'reps_mae': m['reps']['mae'],
    })
per_sub_df = pd.DataFrame(rows)
per_sub_df.to_csv(RUN_DIR / 'per_subject_breakdown.csv', index=False)
print("Per-subject breakdown (fold 0, test=Gorm):")
print(per_sub_df.to_string(index=False))

# ---- LGBM per-subject for comparison
lgbm_metrics = load_json(ROOT / 'runs/20260426_154705_default/metrics.json')
lgbm_per_sub = {k: v for k, v in lgbm_metrics['exercise']['per_subject'].items()}
print("\nLGBM exercise F1 per subject (Gorm):", lgbm_metrics['exercise']['per_subject'].get('Gorm', 'N/A'))

# ---- Build phase2 results
phase2_results = {}
for vname in ['features_tcn', 'features_cnn1d', 'features_cnn_lstm']:
    p2name = f'{vname}_p2'
    p2_path = RUN_DIR / p2name / 'cv_summary.json'
    if p2_path.exists():
        phase2_results[vname] = load_json(p2_path)['summary']

# ---- Final comparison
comparison = build_comparison(phase2_results, lgbm_metrics, latency)
save_json(comparison, RUN_DIR / 'final_metrics.json')

# ---- Save all_results bundle
phase1_results = {}
for arch in ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']:
    vname = f'features_{arch}'
    p = RUN_DIR / vname / 'cv_summary.json'
    if p.exists():
        phase1_results[vname] = load_json(p)['summary']

ranking = load_json(RUN_DIR / 'phase1_summary.json').get('ranking', [])
top_variants = [r['variant'] for r in ranking[:3]]

save_json({
    'phase1': phase1_results, 'phase2': phase2_results,
    'ranking': ranking, 'latency': latency,
    'ablation': ablation, 'winner': 'features_tcn',
    'top_variants': top_variants,
}, RUN_DIR / 'all_results.json')

print("\n\nAll artifacts complete. Run directory:", RUN_DIR)
print("Files produced:")
for f in sorted(RUN_DIR.glob('*.json')) | sorted(RUN_DIR.glob('*.csv')):
    print(" ", f.name)
