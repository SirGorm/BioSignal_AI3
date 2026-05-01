---
description: Full multi-stage NN training and comparison — all features → top-K → modality-ablation → comparison with significance tests, curves, and confusion matrices. Run AFTER /train and /feature-pipeline.
allowed-tools: Bash, Read, Write, Edit
argument-hint: [run_slug]
---

# Train + Compare All NN Models (with all extras)

Full pipeline:

1. **Stage A** — train 4 NN architectures on **all features**
2. **Stage B** — train same 4 architectures on **top-K features** (LDA + ANOVA + MI consensus, leakage-safe)
3. **Stage C** — modality-ablation on the **best-performing arch from A** (drop each of 6 modalities one at a time)
4. **Stage D** — generate plots: training curves, confusion matrices, fatigue calibration
5. **Stage E** — comprehensive comparison report with paired statistical significance vs LightGBM AND XGBoost

**Note: this command does NOT retrain LightGBM or XGBoost on top-K features.** The baselines from `/train` are reused as-is.

## Preconditions

- `/train` completed (LightGBM + XGBoost baselines in `runs/<lgbm_xgb_run>/`)
- `/feature-pipeline` completed (`top_K_features.json` exists; K-sweep recommended a value)
- `configs/splits.csv` exists from `/train` (NN runs reuse identical folds)
- GPU recommended; on CPU expect many hours per stage
- Tests pass: `pytest tests/ -x`

## Steps

### Stage 0 — Confirm scope and compute

Print plan with estimated time:

```
Stage A (all features):     4 archs × N folds × 3 seeds       ~Hours_A GPU-h
Stage B (top-K features):   4 archs × N folds × 3 seeds       ~Hours_B GPU-h
Stage C (modality ablation): 6 modalities × N folds × 3 seeds  ~Hours_C GPU-h
Stage D (plots):             aggregation                        ~5 min
Stage E (comparison):        aggregation + significance tests   ~1 min

Total estimated: ~Hours_total GPU-hours
```

Ask: "Proceed with all stages? Smoke-test first?"

### Stage A — All features (4 NN architectures)

```bash
python scripts/train_cnn1d.py    --run-slug nn-full-cnn1d
python scripts/train_lstm.py     --run-slug nn-full-lstm
python scripts/train_cnn_lstm.py --run-slug nn-full-cnn-lstm
python scripts/train_tcn.py      --run-slug nn-full-tcn
```

Halt on first crash and report.

### Stage B — Top-K features (leakage-safe, all 4 architectures)

Identify K from `runs/<ts>_sweep_k_*/sweep_summary.json` (`recommended_k`).
Default K=30 if not found.

```bash
for arch in cnn1d lstm cnn_lstm tcn; do
  python scripts/train_with_top_k.py --arch $arch --top-k <K> --leakage-safe \
      --run-slug nn-top<K>-$arch --seeds 42 1337 7
done
```

### Stage C — Modality ablation (best arch from Stage A)

After Stage A completes, identify the architecture with highest mean rank
across the 4 tasks (e.g., TCN). Run ablation on it:

```bash
python scripts/ablate_modalities.py --arch <best_arch> \
    --run-slug ablate-<best_arch> --seeds 42 1337 7
```

This trains 6 variants (no_emg, no_ecg, no_eda, no_temp, no_acc, no_ppg).
Tells reviewers "do you really need all 6 sensors?" — answer per task.

### Stage D — Plots

For every run dir produced by Stages A, B, C, generate:

```bash
python scripts/generate_plots.py --runs \
    runs/<ts>_nn-full-cnn1d \
    runs/<ts>_nn-full-lstm \
    runs/<ts>_nn-full-cnn-lstm \
    runs/<ts>_nn-full-tcn \
    runs/<ts>_nn-top<K>-cnn1d \
    runs/<ts>_nn-top<K>-lstm \
    runs/<ts>_nn-top<K>-cnn-lstm \
    runs/<ts>_nn-top<K>-tcn \
    runs/<ts>_ablate-<best_arch>/no_*
```

Per run dir, this writes:
- `<arch>/seed_*/fold_*/training_curves.png` — per-fold curves
- `training_curves_aggregated.png` — mean ± SD across all folds × seeds
- `confusion_matrix_exercise.png`, `confusion_matrix_phase.png`
- `fatigue_calibration.png`

### Stage E — Comprehensive comparison

```bash
python scripts/compare_all.py \
    --baseline-run runs/<lgbm_xgb_run>/ \
    --full-feature-runs <Stage A run dirs> \
    --topk-runs <Stage B run dirs> \
    --ablation-runs <Stage C run dirs> \
    --top-k <K> \
    --output-slug full-comparison
```

Produces `runs/<ts>_full-comparison/comparison.md` with:

- **Master table**: baselines + Stage A + Stage B + Stage C × 4 tasks
- **Best per task**: deployment recommendation
- **Significance tests** (paired t-test or Wilcoxon, Bonferroni-corrected) vs
  LightGBM AND XGBoost
- **Effect sizes** (Cohen's d) alongside p-values
- **Modality importance**: per-task degradation when each modality dropped
- **References section** with full bibliography (Demšar 2006, Wilcoxon 1945,
  etc.)

Copy aggregated plots from Stage D into `runs/<ts>_full-comparison/plots/`.

## Sanity checks before declaring success

Halt and flag if:
- Fewer than 4 architectures completed in Stage A or Stage B
- Top-K runs use different splits than baseline
- Any subject has F1 < 0.5 on its held-out fold (catastrophic failure)
- `model_card.md` files missing `## References` (verify-references hook)
- LightGBM and XGBoost differ by >0.05 on any task (suggests tuning issue
  in baselines)
- Modality ablation shows no degradation when emg dropped (suggests EMG
  channel was bad and model wasn't using it — flag in report)

## Output summary in chat (~20 lines)

```
Full comparison complete: runs/<ts>_full-comparison/

Stage A (all features):       4 archs ✓
Stage B (top-<K>):            4 archs ✓ (leakage-safe selection)
Stage C (modality ablation):  6 modalities × <best_arch> ✓
Stage D (plots):              training curves + confusion + calibration ✓
Stage E (comparison):         14 models × 4 tasks ✓

Best per task:
  Exercise:  TCN_full       F1=0.89 ± 0.03  vs LGBM 0.86  p<0.01 ✓ (Bonf)
  Phase:     CNN-LSTM_top30 F1=0.93 ± 0.02  vs XGB 0.92   p=0.18 –
  Fatigue:   XGBoost        MAE=0.81 ± 0.10 (no NN improvement)
  Reps:      state-machine  MAE=0.31

Top-K vs full features:
  TCN: top-30 within 0.01 of full (52 features) — ship the simpler model
  CNN-LSTM: top-30 better than full — regularization effect

Modality ablation (TCN, F1-macro on Exercise):
  Drop EMG  → 0.89 → 0.71  (-0.18, sensor critical)
  Drop ECG  → 0.89 → 0.86  (-0.03, modest)
  Drop EDA  → 0.89 → 0.88  (-0.01, marginal)
  Drop Temp → 0.89 → 0.89  (no effect)
  Drop Acc  → 0.89 → 0.42  (-0.47, sensor essential)
  Drop PPG  → 0.89 → 0.87  (-0.02, marginal)

Latency p99: TCN 12 ms · 1D-CNN 8 ms · CNN-LSTM 19 ms · LSTM 18 ms (all OK)

Open runs/<ts>_full-comparison/comparison.md for tables, significance tests,
modality breakdown, and per-subject metrics. Plots in plots/ subdirectory.
```

## Hard rules

- **All NN runs MUST reuse `configs/splits.csv`** from /train baseline
- **Top-K Stage B MUST use --leakage-safe**
- **Always run statistical significance** in Stage E
- **Always cite literature** in comparison.md from `literature-references` skill
- **Never** hide a per-subject failure in aggregate statistics
- **Never** retrain LightGBM/XGBoost in this command — they're frozen baselines
