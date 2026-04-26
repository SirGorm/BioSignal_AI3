---
description: Train and compare neural network architectures (1D-CNN, LSTM, CNN-LSTM, TCN) against the LightGBM baseline. Run AFTER /train.
allowed-tools: Bash, Read, Write, Edit
argument-hint: [run_slug]
---

# Train Neural Networks

Trains and compares 4 neural architectures against the LightGBM baseline. **Requires `/train` to have been run first** so a baseline exists.

## Preconditions

- `runs/<previous_run>/model_card.md` exists from `/train`
- `runs/<previous_run>/metrics.json` exists with LightGBM numbers
- `configs/splits.csv` exists (will be REUSED for fair comparison)
- GPU available (or willingness to wait many hours on CPU)
- Tests pass: `pytest tests/ -x`

## Steps

1. **Confirm compute budget with user.** Print estimate based on:
   - 4 architectures × 2 input variants = 8 model variants
   - Phase 1: GroupKFold(5) + 1 seed = ~20 GPU-hours
   - Phase 2: LOSO + 3 seeds on top 2-3 winners = ~50 GPU-hours
   - Total ~70 GPU-hours (vs 290 if running everything at full depth)

   Ask: "Run two-phase comparison (recommended) or full LOSO + 3 seeds on all 8?"

2. **Create run directory**: `runs/<YYYYMMDD_HHMMSS>_nn_<slug>/`. Slug from argument or "comparison".

3. **Pin the config**: copy `configs/nn.yaml` to run dir as `nn_config.yaml`. Reference the LightGBM baseline run path in metadata.

4. **Verify split-reuse**: confirm `configs/splits.csv` from baseline is identical to what will be used. Halt if missing.

5. **Run no-leakage test**: `pytest tests/test_no_leakage.py -v`. Halt on failure.

6. **Phase 1 — Screening (8 variants):**
   - For each combination in `{1D-CNN, LSTM, CNN-LSTM, TCN} × {features, raw}`:
     - Hard-sharing multi-task structure (single encoder + 4 heads)
     - Hyperparameter tuning with Optuna (~30 trials, inner GroupKFold(3))
     - 5-fold GroupKFold + 1 seed for screening
     - Save per-variant model_card.md with `## References`
   - Identify top 2-3 variants by aggregated metric (mean rank across 4 tasks)

7. **Phase 2 — Final depth on winners:**
   - LOSO + 3 seeds on top 2-3 variants from Phase 1
   - Soft-sharing ablation on the overall winner only
   - Per-subject breakdown for each variant
   - Latency benchmark (p99, batch 1, 2 s window)

8. **Generate comparison artifacts**:
   - `runs/<slug>/comparison.md`: 8-variant table + LightGBM baseline + state-machine baselines (where applicable per task)
   - `runs/<slug>/comparison.png`: grouped bar chart with error bars (rows = tasks, groups = arch, colors = input variant)
   - `runs/<slug>/latency_table.md`: p50/p95/p99 per variant
   - `runs/<slug>/multitask_ablation.md`: hard vs soft sharing on the winner
   - `runs/<slug>/SUMMARY.md`: deployment recommendation with rationale

9. **Sanity checks before declaring success**:
   - All 8 Phase-1 variants completed
   - Top 2-3 progressed to Phase 2 with multi-seed
   - Each variant beat its respective dummy baseline
   - At least one variant matches or exceeds LightGBM on at least one task
   - Per-subject breakdowns produced
   - Soft-sharing ablation completed on winner
   - All `model_card.md` files have `## References` (verify-references hook enforces)

10. **Print summary** to chat (~12 lines):
```
Neural network comparison complete: runs/<slug>/

Phase 1 (8 variants screening): top 2 = TCN_raw, CNN-LSTM_raw
Phase 2 (LOSO + 3 seeds on top 2):

Best per task vs LightGBM:
- Exercise:  TCN_raw  F1=0.89 ± 0.03  vs LGBM 0.86 — TCN_raw wins
- Phase:     TCN_raw  Frame-F1=0.94   vs state-machine 0.91 — TCN_raw wins
- Fatigue:   LGBM     MAE=0.82        vs best NN (CNN-LSTM_raw) 0.91 — LGBM wins
- Reps:      state-m. MAE=0.31        vs best NN 0.45 — state-machine wins

Multi-task ablation: hard sharing sufficient (no negative transfer detected).
Latency p99: TCN_raw 12 ms · CNN-LSTM_raw 19 ms (both within 100 ms budget).

Deployment recommendation:
  exercise + phase → TCN_raw (causal, fast, accurate)
  fatigue          → LGBM (low-data regime favors classical)
  reps             → state-machine (interpretable, near-zero latency)

Open runs/<slug>/comparison.md for full results.
```
