---
name: rpe-fatigue-modeling
description: Use when training the fatigue model that predicts RPE (1-10) from biosignal features. Covers per-set granularity, baseline normalization, monotonic constraints, and how to handle the "RPE labels arrive at end of set" supervision pattern.
---

# RPE Fatigue Modeling

You have RPE labels per set, not per window. The model should predict end-of-set RPE from features collected during the set. At inference, you continuously estimate "current fatigue" from features-so-far.

## Granularity decision

**Train at per-set granularity. Inference at any granularity.**

- **Training data**: one row per set. Features = aggregations over the whole set (mean MNF slope, end-of-set MNF/baseline ratio, motion intensity, EDA ramp, time-since-session-start, set_number, exercise). Target = RPE 1–10.
- **Inference**: feed features computed up to the current sample to the same trained model. Model interprets partial-set features as "current effort". Fine because the same features, computed on a partial set, still correlate with final RPE.

Why this works: the heavily-fatigue-correlated features (MNF slope, MPV-loss, HR drift) accumulate within the set. A model trained on full-set features will give a low estimate early in the set and a high estimate late in the set — exactly what you want for real-time fatigue estimation.

## Feature selection for fatigue

From the labeled feature parquet, the per-set features that empirically work best:

```python
PER_SET_FATIGUE_FEATURES = [
    # EMG (most important per literature)
    'emg_mnf_slope',           # negative slope = more fatigue
    'emg_mdf_slope',
    'emg_dimitrov_slope',      # positive slope = more fatigue
    'emg_mnf_endset_rel',      # MNF at end / baseline
    'emg_rms_endset_rel',

    # Cardiac
    'hr_endset_rel',           # HR at end / baseline
    'hr_recovery_proxy',       # HR drop in first 10s of rest after — only post-set
    'rmssd_pre_set',           # HRV before this set started

    # Motion (effort proxy)
    'acc_rms_mean',            # average motion intensity
    'rep_count_predicted',     # from rep counter
    'set_duration_s',

    # EDA
    'eda_scl_endset_rel',
    'eda_scr_count_per_min',

    # Temp
    'temp_slope',

    # Context
    'set_number',              # 1, 2, or 3
    'sets_done_in_session',    # cumulative
    'time_into_session_s',
    'exercise_id',             # one-hot or embedding
]
```

`hr_recovery_proxy` is computed from the first 10 s post-set; available at training time but only retroactively at inference. Build two model variants:
1. **Real-time** (no recovery feature): for online prediction during the set
2. **End-of-set** (with recovery): for the post-set logged value, more accurate

## Model

```python
from lightgbm import LGBMRegressor

model = LGBMRegressor(
    objective='regression_l1',     # MAE, robust to outliers
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=10,          # critical with small N (per-set rows are few)
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)
```

With ~24 subjects × 3 exercises × 3 sets = 216 training rows in a typical study. **You're in low-data territory** — don't tune n_estimators above 500, don't go deeper than max_depth=6, and use heavy regularization. Cross-validation noise is more dangerous than bias here.

## Cross-validation

```python
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold

# Use LOSO if N <= 24, else GroupKFold(5)
groups = X['subject_id'].values
y = X.pop('rpe').values
X = X.drop(columns=['subject_id', 'session_id', 'set_number'])

cv = LeaveOneGroupOut() if len(np.unique(groups)) <= 24 else GroupKFold(5)
fold_metrics = []
for train_idx, test_idx in cv.split(X, y, groups):
    model.fit(X.iloc[train_idx], y[train_idx])
    pred = model.predict(X.iloc[test_idx])
    fold_metrics.append({
        'mae': mean_absolute_error(y[test_idx], pred),
        'pearson_r': pearsonr(y[test_idx], pred)[0] if len(test_idx) > 2 else np.nan,
    })
```

## Per-subject report (mandatory)

A model with average MAE 0.8 RPE points sounds great until you discover MAE is 0.3 for 22 subjects and 4.5 for two subjects. For each held-out subject, report:

```python
def per_subject_fatigue_report(y_true, y_pred, subjects):
    rows = []
    for s in np.unique(subjects):
        mask = subjects == s
        if mask.sum() < 2: continue
        rows.append({
            'subject': s,
            'n_sets': mask.sum(),
            'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
            'pearson_r': pearsonr(y_true[mask], y_pred[mask])[0]
                         if mask.sum() > 2 else np.nan,
            'true_range': (y_true[mask].min(), y_true[mask].max()),
            'pred_range': (y_pred[mask].min(), y_pred[mask].max()),
        })
    return pd.DataFrame(rows)
```

## Sanity check: does Pearson r per subject exceed 0.6?

If yes → model captures fatigue *trend* per subject (it might be off in absolute scale but tracks intra-subject change). Calibrate with a per-subject offset and you have a useful tool.

If no → fatigue features aren't capturing fatigue in your data. Before iterating on the model: plot MNF over time within fatiguing sets per subject. If MNF doesn't trend down within sets, the EMG channel isn't on a fatiguing muscle (placement issue) or fatigue isn't being induced (RPE labels are noisy). **Fix data, not model.**

## Subject-specific calibration (optional but powerful)

If absolute RPE is off but trend is good, add a per-subject offset:

```python
def calibrate_per_subject(y_pred, subject_first_set_pred, subject_first_set_true):
    """y_pred: model output. Calibrate by anchoring to first set per subject."""
    offset = subject_first_set_true - subject_first_set_pred
    return y_pred + offset
```

Requires one ground-truth RPE per subject for calibration — get it from the first set of session 1.

## Real-time inference: what to predict and when

During an active set:
- Every 1 s, recompute the per-set features using "set so far" data.
- Run the trained regressor. Output is "current fatigue estimate".
- Track trajectory; if trajectory is monotonically increasing within set, that's expected. If it decreases mid-set, suspect transient artifact.

After set ends (set boundary detected):
- Compute final per-set features including `hr_recovery_proxy` after 10 s of rest.
- Run the end-of-set model variant; this is the "official" RPE estimate logged.

The two estimates may differ; the end-of-set is more accurate. Display the real-time one as a live indicator, log the end-of-set one as the official value.

## Hard rules

- **Never** train on per-window data with the per-set RPE replicated to every window. That's data leakage of the target through repetition.
- **Always** use group-based CV on subject_id.
- **Always** report per-subject MAE; mean alone hides catastrophic failures.
- **Always** include the dummy baseline (median RPE).
- If MAE > 1.5 RPE points, the model is not useful — investigate features before iterating.

## References

For RPE modeling and CV decisions, cite from:

- **Borg 1982** — foundational RPE scale
- **Robertson et al. 2003** — OMNI-RES validation for resistance training (cite when project uses 1–10 RPE in strength context)
- **Helms et al. 2016** — reps-in-reserve RPE scale (relevant for near-failure predictions)
- **Day et al. 2004** — session-RPE for training-load monitoring
- **Saeb et al. 2017** — subject-wise CV motivation (the leakage problem)
- **Little et al. 2017** — practical CV guidance for biomedical data
- **Ke et al. 2017** — LightGBM
- **Lundberg & Lee 2017** — SHAP for interpretation
- **Akiba et al. 2019** — Optuna for tuning

Full entries in `literature-references` skill. Never invent.
