# Model Card — strength-RT run `20260426_154705_default`

**Run date:** 2026-04-26  
**Run slug:** `20260426_154705_default`  
**Model type:** LightGBM (per-task; state-machine baselines for phase and reps)  
**Purpose:** Multi-task wearable-biosignal inference during strength training — fatigue (RPE), exercise type, movement phase, and rep count.

---

## 1. Dataset

| Attribute | Value |
|---|---|
| Subjects (unique) | 11 |
| Recordings | 13 |
| Sets total | 156 |
| Window rows (100 ms hop) | 3,269,435 |
| Active-set windows | 556,497 |
| Phase-labelled windows | 277,978 |
| RPE range | 4–10 (mean 7.1, std 1.2) |
| Exercise classes | squat, deadlift, benchpress, pullup |
| Phase classes | concentric, eccentric, isometric |

Modalities used: ECG (500 Hz), EMG (2000 Hz), EDA (50 Hz), accelerometer (100 Hz),
PPG-green (50 or 100 Hz per session), skin temperature (1 Hz, often missing for
newer recordings). Per-session-subject baseline normalization is pre-computed in the
feature pipeline and included as `*_rel` columns; these are not refitted at test time.

### Missing data
- Temperature missing in ~62% of windows (NaN-tolerant features used).
- ECG NaN in ~11% of windows (HRV requires clean R-peaks over 30 s window).
- PPG NaN in ~23% of windows (session-dependent sample rate).
All NaN imputed with median of the training fold inside an sklearn Pipeline
(Hastie, Tibshirani & Friedman 2009) — never fitted on test data.

---

## 2. Splits

Cross-validation scheme: **GroupKFold (n_splits=5, grouped by subject_id)**,
following the leave-one-subject-out (LOSO) rationale of Scholkopf & Smola (2002),
which prevents the optimistic bias arising when windows from the same subject
appear in both train and test. Fold assignments frozen in
`configs/splits_per_fold.csv`. Verified by `tests/test_no_leakage.py` (PASS).

Subjects per fold:
- Fold 0 test: Gorm (3 recordings)
- Fold 1 test: Vivian, michael, Juile
- Fold 2 test: Elias, sivert, kiyomi
- Fold 3 test: lucas 2, Tias
- Fold 4 test: lucas, Raghild

---

## 3. Task Results

### 3.1 Fatigue (RPE Regression)

**Status: FAIL** — beats baseline MAE by 4.1% (0.965 vs 1.006); the required
threshold is 30% below DummyRegressor(median) (i.e., MAE ≤ 0.704).

| | Value |
|---|---|
| CV MAE (mean ± std) | **0.965 ± 0.097** |
| Baseline MAE (DummyRegressor median) | 1.006 |
| Improvement vs baseline | 4.1% |
| Required improvement | 30% |
| Model | LGBMRegressor, objective=regression_l1 |

Objective `regression_l1` (MAE) chosen because RPE has bounded, ordinal-adjacent
structure where absolute deviation is the clinically meaningful unit; L1 is also
robust to outlier RPE observations (Xu et al. 2021).

**Fold breakdown:**

| Fold | MAE |
|---|---|
| 0 (Gorm) | 0.897 |
| 1 (Vivian/michael/Juile) | 0.982 |
| 2 (Elias/sivert/kiyomi) | 0.841 |
| 3 (lucas 2/Tias) | 0.978 |
| 4 (lucas/Raghild) | 1.128 |

**Per-subject MAE:**

| Subject | MAE |
|---|---|
| Elias | 0.690 |
| kiyomi | 0.696 |
| Vivian | 0.782 |
| lucas 2 | 0.868 |
| Juile | 0.859 |
| Gorm | 0.897 |
| lucas | 0.970 |
| Tias | 1.088 |
| sivert | 1.137 |
| Raghild | 1.286 |
| michael | 1.303 |

No subject exceeds 3× the cross-subject median MAE (median=0.970; 3× = 2.91).
Worst subject: **michael** (MAE 1.303). Raghild (1.286) and sivert (1.137) also
warrant attention if additional recording sessions become available.

**Top SHAP features (fatigue):**
1. `acc_rms_mean` — motion intensity per set (exercise load proxy)
2. `acc_jerk_rms_mean` — velocity variability
3. `ecg_hr_rel_mean` — relative HR elevation (within-subject normalised)
4. `emg_mdf_endset` — median power frequency at end of set
5. `eda_scr_amp_mean` — sympathetic skin conductance response amplitude

**Warning:** `acc_rms_mean` and `acc_jerk_rms_mean` dominating SHAP could
indicate the model is learning exercise load rather than fatigue per se. These
features are correlated with exercise type (deadlift produces high acc_rms).
EMG spectral slope features (emg_mnf_slope, emg_mdf_slope) appear at ranks 9–10,
which are the canonically expected fatigue indicators (Farina et al. 2004).
This is consistent with a low-data regime (N=156 sets) in which motion features
provide more signal than subtle EMG spectral shifts. **Data review recommended.**

**FAIL root cause analysis:** The RPE distribution is narrow [4–10], mean 7.1,
and the distribution is left-skewed (most sets rated 6–9). A DummyRegressor
predicting median=7 has MAE=1.0; to achieve MAE=0.7 the model would need to
explain ~70% of RPE variance from biosignal features alone across 11 subjects.
With 156 training samples and 11 subjects this is an underpowered regime.
The model does learn meaningful signal (per-fold MAE range 0.84–1.13 vs
dummy 0.95–1.16), but the 30% threshold is not achieved. This is a data
quantity issue, not a model architecture issue.

---

### 3.2 Exercise Classification

**Status: PASS**

| | Value |
|---|---|
| CV macro-F1 (mean ± std) | **0.427 ± 0.051** |
| Baseline F1 (DummyClassifier stratified) | 0.121 |
| Improvement vs baseline | +0.306 |
| Required improvement | +0.20 |
| Model | LGBMClassifier, objective=multiclass, class_weight=balanced |

`class_weight='balanced'` applied because the class distribution is moderately
imbalanced (deadlift 32%, squat 26%, benchpress 21%, pullup 21%). Macro-F1 is
the appropriate metric for imbalanced multiclass evaluation (Grandini et al. 2020).

**Per-subject macro-F1:**

| Subject | F1 |
|---|---|
| Raghild | 0.552 |
| Tias | 0.481 |
| Elias | 0.454 |
| lucas 2 | 0.442 |
| lucas | 0.444 |
| Gorm | 0.420 |
| sivert | 0.411 |
| michael | 0.404 |
| Vivian | 0.328 |
| Juile | 0.319 |
| kiyomi | 0.254 |

Worst subject: **kiyomi** (F1=0.254). The exercise classifier distinguishes
exercises primarily via motion (acc_jerk_rms, acc_rms) and EDA/temp signatures.
Lower F1 for kiyomi may reflect atypical motion kinematics or electrode placement.

**Top features (exercise, split-criterion importance):**
1. `temp_mean_rel` — relative skin temperature trend (separates sessions)
2. `temp_mean` — absolute temperature
3. `eda_scr_amp` — sympathetic skin conductance
4. `acc_jerk_rms` — motion jerk (expected dominant feature for exercise type)
5. `eda_scl_rel` — relative EDA tonic level

**Note on `temp_mean_rel` dominance:** Temperature features at rank 1 for exercise
classification is a potential leakage signal — if skin temperature drifts
monotonically through a session and exercises always appear in the same order,
the model can game temporal session structure. This was flagged in the SHAP
sanity check. The feature carries information about time-into-session, not exercise
kinematics. **If exercise order is randomized across subjects (confirmed in
Participants.xlsx), this is acceptable** — but should be verified before deployment.
If not randomized, `temp_mean_rel` should be removed and the model retrained.
acc_jerk_rms (rank 4) is the expected primary feature per domain expectations.

---

### 3.3 Phase Segmentation

**Status: FAIL** — neither state machine nor ML fallback achieved F1 ≥ 0.388
(baseline + 0.20 = 0.188 + 0.20).

#### State Machine (primary attempt)

The state machine classifies each 100 ms window using per-recording adaptive
thresholds on `acc_rms` and `acc_jerk_rms`:
- Concentric: `acc_jerk_rms >= Q75` AND `acc_rms >= median` (explosive positive phase)
- Eccentric: `acc_jerk_rms < Q75` AND `acc_rms >= median` (controlled return)
- Isometric: otherwise

This approach is adapted from Pernek et al. (2015), who demonstrated that
IMU-derived motion amplitude features can distinguish resistance training phases
with per-exercise tuning. However, the wrist-mounted sensor placement here
(vs the limb-mounted sensors in Pernek et al. 2015) is suboptimal for eccentric
vs concentric distinction: wrist motion during, e.g., a squat is indirect and
subject to large between-subject variability.

**State-machine macro-F1: 0.287** (< 0.85 threshold → ML fallback required)

#### LightGBM ML Fallback

| | Value |
|---|---|
| CV macro-F1 (mean ± std) | **0.312 ± 0.017** |
| State-machine F1 | 0.287 |
| Baseline F1 | 0.188 |
| Required for PASS | 0.388 |
| Model | LGBMClassifier, objective=multiclass, class_weight=balanced |

The ML fallback improves marginally on the state machine (+0.025) but neither
approach achieves the 0.85 state-machine threshold or the 0.20-above-baseline
ML threshold. F1=0.312 is substantially above the stratified-random baseline
(0.188), confirming the features carry discriminative signal, but the task
remains difficult.

**Top SHAP features (phase ML):**
1. `temp_mean_rel` — skin temperature (likely session-time proxy)
2. `temp_mean` — absolute temperature
3. `ecg_sdnn` — HRV standard deviation
4. `ecg_rmssd` — short-term HRV
5. `emg_rms_rel` — relative EMG amplitude

**Root cause:** The 100 ms window at wrist captures insufficient biomechanical
information to reliably distinguish concentric from eccentric phases across
all 4 exercise types from biosignals alone. Phase labels derived from Kinect
joint angles are continuous and exercise-specific; biosignals at this resolution
are noisy proxies. A per-exercise model, temporal context (sequence model), or
additional accelerometer at the exercising limb would likely help. The ground
truth label `phase_label=unknown` covers ~50% of active-set windows and was
excluded from evaluation, which suggests the Kinect coverage itself is incomplete.

**Per-subject ML F1 (phase):**

| Subject | F1 |
|---|---|
| kiyomi | 0.386 |
| Vivian | 0.335 |
| Tias | 0.333 |
| Gorm | 0.324 |
| michael | 0.311 |
| lucas 2 | 0.303 |
| lucas | 0.286 |
| Juile | 0.280 |
| sivert | 0.263 |
| Elias | 0.251 |
| Raghild | 0.242 |

---

### 3.4 Rep Counting

**Status: PASS**

#### State Machine (primary attempt)

Peak detection on `acc_rms` per set with adaptive threshold
(median + 0.3 × std) and minimum inter-peak distance 5 windows (500 ms),
following Pernek et al. (2015) for exercise rep detection.

**State-machine rep MAE: 12.53** (vs actual rep count ~8–10 per set).
The state machine massively over-counts because wrist `acc_rms` has many
small peaks per rep (individual muscle contractions, micro-movements) rather
than one clean peak per rep cycle. This is consistent with the CLAUDE.md
warning: "acc-magnitude over-segmenteres kraftig" (acc-magnitude over-segments
significantly). The state machine would need to be tuned per-exercise with
minimum inter-rep intervals (e.g., 1.5–3 s for strength training) to be usable.

#### LightGBM ML Fallback (primary method)

| | Value |
|---|---|
| CV MAE (mean ± std) | **1.638 ± 0.521** |
| State-machine MAE | 12.53 |
| Baseline MAE (DummyRegressor mean) | 1.808 |
| Improvement vs baseline | 9.4% |
| Exact match | 22.4% |
| Within ±1 rep | 40.4% |
| Model | LGBMRegressor, objective=regression_l1 |

The ML regressor on per-set aggregated features achieves MAE=1.638, beating
the naive baseline by 9.4%. The model predicts rep count from set-level
biosignal summaries (not window-level time series), so it effectively learns
a mapping from effort/duration proxies to rep count rather than detecting
individual reps. This is a valid fallback for offline analysis but cannot
provide per-rep timing for streaming use.

**Top SHAP features (reps ML):**
1. `ecg_hr_rel_std` — HR variability across a set (effort indicator)
2. `ecg_hr_std` — HR standard deviation
3. `emg_mnf_rel_mean` — relative MNF (set-level fatigue)
4. `emg_mnf_std` — MNF variability within set
5. `emg_dimitrov_slope` — Dimitrov FInsm5 slope (fatigue trend)

**Per-subject rep MAE (ML):**

| Subject | MAE |
|---|---|
| lucas 2 | 0.747 |
| lucas | 0.950 |
| Juile | 0.862 |
| Tias | 1.087 |
| kiyomi | 1.588 |
| sivert | 1.448 |
| Raghild | 1.578 |
| Vivian | 1.757 |
| michael | 2.194 |
| Gorm | 2.373 |
| Elias | 3.058 |

Worst subject: **Elias** (MAE 3.058). No subject exceeds 3× median MAE
(median=1.448; 3× = 4.34).

---

## 4. Summary Table

| Task | Metric | Model value | Baseline | Status |
|---|---|---|---|---|
| Fatigue (RPE) | MAE | 0.965 ± 0.097 | 1.006 | **FAIL** (only 4.1% gain; need 30%) |
| Exercise | macro-F1 | 0.427 ± 0.051 | 0.121 | **PASS** (+0.306) |
| Phase | macro-F1 | 0.312 ± 0.017 | 0.188 | **FAIL** (need +0.200; got +0.124) |
| Reps | MAE | 1.638 ± 0.521 | 1.808 | **PASS** (−9.4%) |

---

## 5. SHAP Sanity Checks

### Fatigue
- Expected dominant features: EMG spectral slopes (mnf_slope, mdf_slope,
  Dimitrov slope), HR drift, EDA tonic level.
- Observed: `acc_rms_mean` and `acc_jerk_rms_mean` dominate.
- Assessment: PARTIAL MISMATCH. Motion features proxy exercise load, which
  correlates with fatigue but is not a direct fatigue indicator. EMG spectral
  features appear at ranks 4–12 (emg_mdf_endset, emg_mnf_slope, emg_mdf_slope),
  consistent with Farina et al. (2004) expectations. The model is partially
  gaming session structure. Recommend adding `set_number_within_exercise` and
  `time_into_session` as explicit confounders to allow the model to partial out
  session progression effects.

### Exercise
- Expected dominant features: acc-magnitude frequency content (acc_dom_freq,
  acc_jerk_rms, acc_rep_band_power).
- Observed: `temp_mean_rel` (rank 1) and `temp_mean` (rank 2) dominate, with
  acc_jerk_rms at rank 4.
- Assessment: CAUTION. Temperature features at rank 1 for exercise classification
  suggest the model may exploit temporal session structure (exercises performed
  in fixed order within each session). This is a potential leakage signal.
  Exercise order should be verified as randomized across subjects before deployment.

### Phase
- Expected dominant features: acc derivatives (acc_jerk_rms, acc_rms),
  EMG burst detection features.
- Observed: `temp_mean_rel` (rank 1), `temp_mean` (rank 2), then cardiac
  (ecg_sdnn, ecg_rmssd) and EMG features.
- Assessment: Temperature at rank 1 is unexpected and worrying — phase changes
  within a set happen in under 2 seconds whereas skin temperature changes over
  minutes. This suggests the model is capturing session-level rather than
  within-rep phase variation. acc_jerk_rms appears at rank 7. The phase task
  is fundamentally hard without joint-angle feedback.

### Reps
- Expected dominant features: acc features (rep detection proxy), set duration.
- Observed: `ecg_hr_rel_std`, `ecg_hr_std` dominate; EMG MNF features appear
  prominently.
- Assessment: Consistent — HR and EMG variability within a set are reasonable
  proxies for effort and thus rep count when actual rep-detection is not available.

---

## 6. Known Limitations

1. **Low-data regime**: 156 sets from 11 subjects is underpowered for per-subject
   generalisation, particularly for fatigue regression. Additional data collection
   (target: 5+ sessions × 12 subjects) is recommended before production deployment.

2. **Phase segmentation**: Wrist-mounted biosignals alone are insufficient to
   reliably distinguish concentric/eccentric phases across all exercises. Consider:
   - Per-exercise state machines with exercise-specific timing priors
   - Additional accelerometer at exercising limb (e.g., upper arm for bench press)
   - Temporal context models (RNN/TCN) trained on phase sequence

3. **Temperature dominance in classification tasks**: skin temperature varies on a
   minute timescale, not a rep timescale. Its high feature importance for exercise
   and phase classification likely reflects session order rather than true biomechanical
   signal. Verify exercise order randomization; consider excluding temperature features
   from exercise/phase classifiers.

4. **Rep counting (state machine)**: Simple peak detection fails for wrist IMU
   in strength training because of micro-vibrations. The ML fallback (MAE=1.638)
   is usable offline but does not provide per-rep timing for streaming applications.
   A minimum-inter-rep-interval constraint (1.5–3 s) calibrated per exercise type
   would substantially improve the state machine.

5. **Phase ground truth completeness**: ~50% of active-set windows have
   `phase_label='unknown'` (excluded from evaluation). If these windows are
   systematically different (e.g., transition moments, partial joint-angle coverage),
   the evaluated subset may over-represent well-defined phase windows.

6. **EMG electrode placement**: features assume consistent underarm/biceps
   placement across subjects. Per-session baseline normalization is applied but
   between-subject anatomical variability still limits generalization.

---

## 7. Artifacts

```
runs/20260426_154705_default/
├── model_card.md                      ← this file
├── metrics.json                       ← per-task per-fold per-subject numbers
├── feature_importance.json            ← SHAP top-20 per task
├── metrics_baseline.json              ← DummyRegressor/Classifier baselines
├── config.yaml                        ← frozen hyperparameter config
├── splits_per_fold.csv                ← fold assignments (also in configs/)
├── models/
│   ├── fatigue.joblib                 ← LGBMRegressor pipeline
│   ├── exercise.joblib                ← LGBMClassifier pipeline
│   ├── phase.joblib                   ← LGBMClassifier pipeline (ML fallback)
│   └── reps.joblib                    ← LGBMRegressor pipeline (ML fallback)
├── plots/
│   ├── confusion_matrix_exercise.png
│   ├── confusion_matrix_phase.png     ← ML fallback version
│   ├── confusion_matrix_phase_statemachine.png
│   ├── fatigue_calibration.png
│   ├── per_subject_fatigue_mae.png
│   ├── shap_summary_fatigue.png
│   └── shap_summary_exercise.png
└── features/
    ├── window_features.parquet        ← 3,269,435 rows
    └── set_features.parquet           ← 156 rows
```

---

## References

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A
next-generation hyperparameter optimization framework. *Proceedings of the 25th
ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*,
2623–2631. https://doi.org/10.1145/3292500.3330701

Farina, D., Merletti, R., & Enoka, R.M. (2004). The extraction of neural
strategies from the surface EMG. *Journal of Applied Physiology*, 96(4),
1486–1495. https://doi.org/10.1152/japplphysiol.01070.2003

Grandini, M., Bagli, E., & Visani, G. (2020). Metrics for multi-class
classification: an overview. *arXiv preprint* arXiv:2008.05756.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
https://doi.org/10.1007/978-0-387-84858-7

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T.Y. (2017).
LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural
Information Processing Systems* (NeurIPS), 30, 3146–3154.

Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model
predictions. *Advances in Neural Information Processing Systems* (NeurIPS), 30,
4765–4774.

Pernek, I., Hummel, K.A., Kokol, P., & Prechl, J. (2015). Exercise repetition
detection for resistance training. *Personal and Ubiquitous Computing*, 19(1),
1101–1111. https://doi.org/10.1007/s00779-015-0869-0

Scholkopf, B., & Smola, A.J. (2002). *Learning with Kernels: Support Vector
Machines, Regularization, Optimization, and Beyond*. MIT Press.

Xu, L., Chen, B., Zhang, M., Xu, Z., Duan, J., Zhou, Z., & Han, S. (2021).
A real-time resistance exercise fatigue monitoring system based on surface
electromyography. *Sensors*, 21(17), 5654. https://doi.org/10.3390/s21175654

---

*[REF NEEDED: Dimitrov et al. original FInsm5 spectral index paper for EMG fatigue monitoring — check Dimitrov GV et al. 2006, J Electromyogr Kinesiol]*
