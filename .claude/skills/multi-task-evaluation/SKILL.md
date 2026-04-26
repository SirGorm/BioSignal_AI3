---
name: multi-task-evaluation
description: Use when evaluating models across the four tasks (fatigue regression, exercise classification, phase segmentation, rep counting) together. Defines per-task metrics, per-subject reporting, latency benchmarking, and the standard model_card.md format for this project.
---

# Multi-Task Evaluation

You have four tasks with different metrics. Reporting only aggregate accuracy is misleading. Always produce a per-subject breakdown.

## Per-task metrics (mandatory minimum)

### Fatigue (regression)
- **MAE** (Mean Absolute Error) — primary
- **Pearson r** between predicted and true within-subject (correlation matters more than absolute scale)
- **Spearman ρ** if target is ordinal RPE
- **Per-subject MAE distribution** — report median, IQR, and worst subject
- **Calibration plot** — predicted vs. true binned

```python
from scipy.stats import pearsonr, spearmanr
def fatigue_metrics(y_true, y_pred, subjects):
    overall_mae = np.mean(np.abs(y_true - y_pred))
    per_subj = []
    for s in np.unique(subjects):
        mask = subjects == s
        per_subj.append({
            'subject': s,
            'mae': np.mean(np.abs(y_true[mask] - y_pred[mask])),
            'pearson_r': pearsonr(y_true[mask], y_pred[mask])[0]
                          if mask.sum() > 2 else np.nan,
        })
    return {'overall_mae': overall_mae, 'per_subject': per_subj}
```

### Exercise (multi-class classification)
- **F1-macro** — primary (handles class imbalance)
- **Balanced accuracy**
- **Confusion matrix** — visualize, save as PNG
- **Per-class precision/recall** — flag any class with recall < 0.7

### Phase (per-timestep classification)
- **Frame-wise F1-macro**
- **Edge F1** — F1 of phase-transition timestamps within ±100 ms tolerance (this is what users actually feel)
- **Mean phase-duration error** per phase

```python
def edge_f1(true_edges, pred_edges, tolerance_samples=10):
    """true/pred_edges: arrays of sample indices where phase changes."""
    matched_true = set()
    matched_pred = set()
    for i, te in enumerate(true_edges):
        for j, pe in enumerate(pred_edges):
            if j in matched_pred: continue
            if abs(te - pe) <= tolerance_samples:
                matched_true.add(i); matched_pred.add(j); break
    tp = len(matched_pred)
    fp = len(pred_edges) - tp
    fn = len(true_edges) - len(matched_true)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-9)
```

### Rep counting
- **MAE on rep count per set** — primary
- **% sets with exact count match**
- **% sets within ±1 rep**
- **Per-exercise breakdown** — push-ups will likely be worse than barbell lifts

```python
def rep_metrics(true_counts, pred_counts, exercises):
    return {
        'mae': np.mean(np.abs(true_counts - pred_counts)),
        'exact_match': np.mean(true_counts == pred_counts),
        'within_one': np.mean(np.abs(true_counts - pred_counts) <= 1),
        'per_exercise_mae': {e: np.mean(np.abs(true_counts[exercises == e] -
                                                pred_counts[exercises == e]))
                             for e in np.unique(exercises)},
    }
```

## Latency benchmark (real-time deployment requirement)

```python
import time

def latency_benchmark(pipeline, replay_data, chunk_samples=10):
    """Replay recorded data through the pipeline at chunk granularity.
    Report p50, p95, p99 latency per inference."""
    latencies = []
    for chunk in chunked_iter(replay_data, chunk_samples):
        t0 = time.perf_counter()
        _ = pipeline.step(chunk)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms
    return {
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'max_ms': max(latencies),
    }
```

A model that hits 0.85 F1 but exceeds the 100 ms p99 budget is unusable. Report it as failed.

## Required artifacts per experiment

Every training run must produce a directory `runs/<timestamp>_<slug>/` containing:

```
runs/20260426_153012_lgbm_baseline/
├── model_card.md             # the human-readable summary
├── config.yaml               # exact config used (so it's reproducible)
├── metrics.json              # all numbers, per-task, per-subject
├── confusion_matrix_exercise.png
├── confusion_matrix_phase.png
├── fatigue_calibration.png
├── per_subject_breakdown.csv
├── latency_report.json
├── feature_importance.json   # SHAP for each task
├── models/                   # joblib files per task
└── logs/
    └── train.log
```

## model_card.md template

```markdown
# Run: <timestamp>_<slug>

## Dataset
- N subjects: 24 (split: 18 train, 6 test via LOSO outer fold)
- N sets: 412
- N reps total (ground truth): 4,832
- Class balance (exercise): squat 31%, bench 28%, deadlift 18%, row 15%, pushup 8%

## Pipeline
- Window: 500 ms (fatigue: 5 s)
- Hop: 100 ms
- Filtering: causal Butterworth, sosfilt with persisted state
- Normalization: per-subject baseline (first 30 s)

## Models
- Exercise: LightGBM (n_estimators=300, max_depth=6)
- Fatigue:  LightGBM regressor
- Phase:    state machine on velocity (no ML)
- Reps:     state machine on velocity (no ML)

## Results

### Exercise classification
- F1-macro: 0.86 ± 0.07 (LOSO across 6 test subjects)
- Worst subject: S012 (F1=0.71) — bad EMG channel quality, flagged

### Fatigue regression
- MAE: 0.94 RPE points (target: < 1.5)
- Pearson r per-subject: median 0.78, IQR [0.65, 0.84]
- Two subjects with r < 0.4 (S007, S019) — investigation note: both had no EMG decline within fatiguing sets, suggesting electrode issues

### Phase segmentation
- Frame F1: 0.92
- Edge F1 (±100 ms): 0.88

### Rep counting
- MAE: 0.31 reps per set
- Exact match: 78%
- Within ±1: 96%
- Push-ups worst: MAE 0.68 (low-velocity reps near threshold)

## Latency
- p50: 18 ms
- p95: 47 ms
- p99: 82 ms ✓ (under 100 ms budget)

## Limitations / Known Issues
- 2/24 subjects with poor fatigue prediction (likely data quality)
- Push-up rep counting needs lower velocity threshold
- Tempo training (>3s eccentric) not yet validated
- All training data from one gym; cross-environment validation pending
```

## Sanity checks (run automatically via Stop hook)

Before declaring a run successful, verify:

1. **No leakage**: subjects in test fold are absent from train fold (run `tests/test_no_leakage.py`).
2. **Beats baselines**: model F1/MAE beats DummyClassifier/DummyRegressor by meaningful margin.
3. **Latency under budget**: p99 < 100 ms.
4. **Per-subject worst case**: no subject has F1 < 0.5 or MAE > 3× median.

If any check fails, mark the run as `status: FAILED` in metrics.json and explain in model_card.md. Do not silently report aggregate metrics that hide failures.

## References

For evaluation metrics, CV, and reporting standards:

- **Saeb et al. 2017** — subject-wise CV (the foundational leakage paper)
- **Little et al. 2017** — practical CV for biomedical signals
- **Lundberg & Lee 2017** — SHAP values for per-task feature importance
- **Caruana 1997** — multi-task learning foundational reference (cite when evaluating shared-encoder vs separate models)

Full entries in `literature-references` skill. Never invent.
