---
description: Run the full training pipeline - feature extraction + 4-task model training + evaluation + latency benchmark
allowed-tools: Bash, Read, Write, Edit
argument-hint: [run_slug]
---

# Train Models

Run the full training pipeline. Requires `/label` to have been completed first.

## Preconditions

- `data/labeled/<subject>/<session>/aligned_features.parquet` exists for all subjects
- `tests/test_no_leakage.py` and `tests/test_label_alignment.py` pass

## Steps

1. **Create run directory**: `runs/<YYYYMMDD_HHMMSS>_<slug>/`. Slug from argument or "default".

2. **Pin the config**: copy `configs/default.yaml` to `runs/.../config.yaml`. Never edit it after this point.

3. **Define splits**: Generate subject-wise CV splits, save to `configs/splits.csv` (overwriting prior). Run `pytest tests/test_no_leakage.py` — halt on failure.

4. **Invoke biosignal-feature-extractor subagent**:
   - Reads labeled parquet files
   - Computes per-window and per-set features for all 6 modalities
   - Runs feature parity test (offline vs streaming) — halt on failure
   - Outputs `window_features.parquet` and `set_features.parquet`

5. **Run baselines** for all 4 tasks (DummyClassifier, DummyRegressor, naive rep-counter, state-machine phase). Save to `runs/.../metrics_baseline.json`.

6. **Invoke ml-expert subagent** to:
   - Train fatigue regressor (per-set, RPE 1-10)
   - Train exercise classifier (per-window)
   - Try state-machine for phase + reps; train ML fallback only if state-machine F1 < 0.85
   - Tune with Optuna (50-100 trials per task)
   - Per-subject reporting via `multi-task-evaluation` skill
   - SHAP for interpretation

7. **Latency benchmark**: replay 10 minutes of data through the streaming pipeline. Halt-with-FAIL-status if p99 > 100 ms.

8. **Write artifacts** per multi-task-evaluation skill specification:
   - `model_card.md` (the human summary)
   - `metrics.json` (all numbers per-task per-subject)
   - Plots: confusion matrices, fatigue calibration, SHAP summary, latency histogram
   - `models/` directory with joblib files
   - `feature_importance.json`

9. **Sanity check** before declaring success:
   - All 4 tasks beat their respective baselines
   - p99 latency under budget
   - No subject has fatigue MAE > 3× the median MAE
   - State-machine phase F1 reported (not just ML)

## Output

```
Run: runs/<timestamp>_<slug>/
- Fatigue MAE:   X.XX ± X.XX (baseline X.XX) — PASS|FAIL
- Exercise F1:   0.XX ± 0.XX (baseline 0.XX) — PASS|FAIL
- Phase F1:      0.XX (state-machine | ML)
- Rep MAE:       X.XX (state-machine | ML)
- Latency p99:   XX ms — PASS|FAIL
Open runs/<timestamp>_<slug>/model_card.md for details.
```
