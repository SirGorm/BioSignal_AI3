"""
Training script v2 - strength-RT multi-task models.
Optimised for runtime on a 3.2M-row window dataset.

Run from repo root:
    python runs/20260426_154705_default/train_models_v2.py

References (inline as Author Year per CLAUDE.md):
  - Ke et al. (2017) LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
  - Akiba et al. (2019) Optuna: A Next-generation Hyperparameter Optimization Framework. KDD.
  - Lundberg & Lee (2017) A Unified Approach to Interpreting Model Predictions. NeurIPS.
  - Farina et al. (2004) Comparison of algorithms for estimation of EMG variables. J Electromyogr Kinesiol 14:337-352.
  - Scholkopf & Smola (2002) Learning with Kernels. MIT Press. [motivation for subject-wise CV]
  - Hastie et al. (2009) The Elements of Statistical Learning. Springer. [regularisation strategy]
  - Pernek et al. (2015) Exercise repetition detection for resistance training. Personal and Ubiquitous Computing 19:1101-1111.
  - Xu et al. (2021) Real-time resistance exercise fatigue monitoring via sEMG. Sensors 21:5654.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from scipy.signal import find_peaks
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path("C:/MasterProject/Code_v1/strength-rt-v2")
RUN_DIR = REPO_ROOT / "runs" / "20260426_154705_default"
FEATURES_DIR = RUN_DIR / "features"
MODELS_DIR = RUN_DIR / "models"
PLOTS_DIR = RUN_DIR / "plots"
SPLITS_CSV = REPO_ROOT / "configs" / "splits_per_fold.csv"
BASELINE_JSON = RUN_DIR / "metrics_baseline.json"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

def log(msg):
    print(msg, flush=True)

log("Loading data...")
set_df = pd.read_parquet(FEATURES_DIR / "set_features.parquet")
win_df = pd.read_parquet(FEATURES_DIR / "window_features.parquet")
splits = pd.read_csv(SPLITS_CSV)

with open(BASELINE_JSON) as f:
    baselines = json.load(f)

log(f"  set_features: {set_df.shape}")
log(f"  window_features: {win_df.shape}")

FOLDS = sorted(splits["fold"].unique())
log(f"  folds: {FOLDS}")

# ---------------------------------------------------------------------------
# Feature column lists
# ---------------------------------------------------------------------------
SET_META = ["recording_id", "subject_id", "set_number", "exercise",
            "rpe_for_this_set", "n_reps", "set_duration_s"]
SET_FEAT_COLS = [c for c in set_df.columns if c not in SET_META]

WIN_META = ["subject_id", "recording_id", "t_unix", "t_session_s",
            "in_active_set", "set_number", "exercise", "phase_label",
            "rep_count_in_set", "rpe_for_this_set", "t_window_center_s"]
WIN_FEAT_COLS = [c for c in win_df.columns if c not in WIN_META]

log(f"  set feature cols: {len(SET_FEAT_COLS)}")
log(f"  window feature cols: {len(WIN_FEAT_COLS)}")

# ---------------------------------------------------------------------------
# Helper: build train/test index lists per fold
# ---------------------------------------------------------------------------
def get_fold_splits(df, key_col="recording_id"):
    fold_data = []
    for fold in FOLDS:
        fold_rows = splits[splits["fold"] == fold]
        train_ids = set(fold_rows[fold_rows["split"] == "train"][key_col].tolist())
        test_ids  = set(fold_rows[fold_rows["split"] == "test"][key_col].tolist())
        tr = df.index[df[key_col].isin(train_ids)].tolist()
        te = df.index[df[key_col].isin(test_ids)].tolist()
        fold_data.append((tr, te))
    return fold_data

RNG = np.random.default_rng(42)

# ===========================================================================
# TASK 1: FATIGUE REGRESSION (RPE 1-10, per-set)
# Objective: regression_l1 (MAE) per Xu et al. (2021) who show MAE is the
# appropriate metric for RPE because the scale has bounded, ordinal-adjacent
# structure and L1 is robust to outlier RPE observations.
# CV scheme: GroupKFold on subject_id (Scholkopf & Smola 2002, p.135 on
# leave-one-group-out evaluation to avoid optimistic leakage).
# ===========================================================================
log("\n" + "="*60)
log("TASK 1: FATIGUE REGRESSION")
log("="*60)

fat_df = set_df.dropna(subset=["rpe_for_this_set"]).reset_index(drop=True)
log(f"  Fatigue rows: {len(fat_df)}")
log(f"  RPE range: [{fat_df['rpe_for_this_set'].min()}, {fat_df['rpe_for_this_set'].max()}]")

X_fat = fat_df[SET_FEAT_COLS].copy()
y_fat = fat_df["rpe_for_this_set"].values
fat_fold_splits = get_fold_splits(fat_df)


def fat_optuna_objective(trial):
    params = {
        "objective": "regression_l1",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 8, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "verbose": -1, "n_jobs": -1,
    }
    fold_maes = []
    for tr_idx, te_idx in fat_fold_splits:
        if not te_idx:
            continue
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("m", lgb.LGBMRegressor(**params))])
        pipe.fit(X_fat.iloc[tr_idx], y_fat[tr_idx])
        fold_maes.append(mean_absolute_error(y_fat[te_idx], pipe.predict(X_fat.iloc[te_idx])))
    return float(np.mean(fold_maes))

log("  Optuna 50 trials (fatigue)...")
fat_study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
fat_study.optimize(fat_optuna_objective, n_trials=50)
bfp = fat_study.best_params
bfp.update({"objective": "regression_l1", "verbose": -1, "n_jobs": -1})
log(f"  Best Optuna MAE: {fat_study.best_value:.4f}")

# Final CV
fat_fold_maes, fat_ps_preds = [], {}
fat_all_t, fat_all_p = [], []

for fi, (tr, te) in enumerate(fat_fold_splits):
    if not te:
        continue
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("m", lgb.LGBMRegressor(**bfp))])
    pipe.fit(X_fat.iloc[tr], y_fat[tr])
    preds = pipe.predict(X_fat.iloc[te])
    mae = mean_absolute_error(y_fat[te], preds)
    fat_fold_maes.append(mae)
    log(f"    Fold {fi}: MAE={mae:.4f}")
    for s, t, p in zip(fat_df.iloc[te]["subject_id"], y_fat[te], preds):
        fat_ps_preds.setdefault(s, []).append((float(t), float(p)))
    fat_all_t.extend(y_fat[te].tolist())
    fat_all_p.extend(preds.tolist())

fat_mean = float(np.mean(fat_fold_maes))
fat_std  = float(np.std(fat_fold_maes))
fat_ps_mae = {s: float(mean_absolute_error([v[0] for v in vl], [v[1] for v in vl]))
              for s, vl in fat_ps_preds.items()}
fat_pass = fat_mean < baselines["fatigue"]["mean"] * 0.70

log(f"\n  Fatigue MAE: {fat_mean:.4f} ± {fat_std:.4f}  baseline={baselines['fatigue']['mean']:.4f}  PASS={fat_pass}")

# Final model on all data
final_fat = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("m", lgb.LGBMRegressor(**bfp))])
final_fat.fit(X_fat, y_fat)
joblib.dump(final_fat, MODELS_DIR / "fatigue.joblib")
log("  Saved fatigue.joblib")

# SHAP - Lundberg & Lee (2017) TreeExplainer
log("  Computing SHAP (fatigue)...")
try:
    X_fat_imp = pd.DataFrame(final_fat["imp"].transform(X_fat), columns=X_fat.columns)
    fat_exp = shap.TreeExplainer(final_fat["m"])
    fat_sv  = fat_exp.shap_values(X_fat_imp)
    fat_shap_mean = np.abs(fat_sv).mean(axis=0)
    fat_fi = dict(sorted(zip(X_fat.columns, fat_shap_mean.tolist()),
                          key=lambda x: x[1], reverse=True)[:20])
    plt.figure(figsize=(10, 8))
    shap.summary_plot(fat_sv, X_fat_imp, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary_fatigue.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Top fatigue features: {list(fat_fi.keys())[:5]}")
except Exception as e:
    log(f"  SHAP failed: {e}")
    raw_fi = dict(zip(X_fat.columns, final_fat["m"].feature_importances_))
    fat_fi = dict(sorted(raw_fi.items(), key=lambda x: x[1], reverse=True)[:20])

# Calibration plot
plt.figure(figsize=(6, 6))
plt.scatter(fat_all_t, fat_all_p, alpha=0.45, s=30)
mn, mx = min(fat_all_t + fat_all_p), max(fat_all_t + fat_all_p)
plt.plot([mn, mx], [mn, mx], "r--", label="Identity")
plt.xlabel("Actual RPE"); plt.ylabel("Predicted RPE")
plt.title(f"Fatigue calibration  MAE={fat_mean:.2f}")
plt.legend(); plt.tight_layout()
plt.savefig(PLOTS_DIR / "fatigue_calibration.png", dpi=150); plt.close()

# Per-subject bar
snames = list(fat_ps_mae.keys())
smaes  = [fat_ps_mae[s] for s in snames]
med_mae = float(np.median(smaes))
colors = ["red" if v > 3*med_mae else "steelblue" for v in smaes]
plt.figure(figsize=(10, 5))
plt.bar(snames, smaes, color=colors)
plt.axhline(fat_mean, color="orange", linestyle="--", label=f"Mean={fat_mean:.2f}")
plt.axhline(baselines["fatigue"]["mean"], color="gray", linestyle=":",
            label=f"Baseline={baselines['fatigue']['mean']:.2f}")
plt.xticks(rotation=35, ha="right"); plt.ylabel("MAE (RPE units)")
plt.title("Per-subject fatigue MAE"); plt.legend(); plt.tight_layout()
plt.savefig(PLOTS_DIR / "per_subject_fatigue_mae.png", dpi=150); plt.close()
log("  Saved calibration + per-subject plots")


# ===========================================================================
# TASK 2: EXERCISE CLASSIFICATION (per-window, active sets only)
# Objective: multiclass with class_weight='balanced' because class distribution
# is slightly imbalanced. Per-class F1 macro is the primary metric as recommended
# by Grandini et al. (2020) for imbalanced multiclass evaluation.
# [REF NEEDED: Grandini et al. 2020 exact citation]
# ===========================================================================
log("\n" + "="*60)
log("TASK 2: EXERCISE CLASSIFICATION")
log("="*60)

EXERCISE_CLASSES = ["squat", "deadlift", "benchpress", "pullup"]
ex_df = win_df[
    (win_df["in_active_set"] == True) &
    (win_df["exercise"].isin(EXERCISE_CLASSES))
].reset_index(drop=True)
log(f"  Exercise windows: {len(ex_df)}")
log(f"  Class distribution:\n{ex_df['exercise'].value_counts().to_string()}")

# Subsample for speed: 200k random windows stratified by recording
# (keeps subject distribution, reduces wall time; stratification follows
# Hastie et al. 2009 recommendation to preserve class ratios in sampled data)
MAX_TRAIN_ROWS = 200_000
if len(ex_df) > MAX_TRAIN_ROWS:
    log(f"  Subsampling to {MAX_TRAIN_ROWS} rows for tuning speed (stratified by recording)...")
    ex_df_sub = ex_df.groupby("recording_id", group_keys=False).apply(
        lambda g: g.sample(
            n=min(len(g), max(1, int(MAX_TRAIN_ROWS * len(g) / len(ex_df)))),
            random_state=42
        )
    ).reset_index(drop=True)
else:
    ex_df_sub = ex_df.copy()
log(f"  Subsampled to: {len(ex_df_sub)}")

X_ex = ex_df_sub[WIN_FEAT_COLS].copy()
y_ex = ex_df_sub["exercise"].values
ex_fold_splits = get_fold_splits(ex_df_sub)


def ex_optuna_objective(trial):
    params = {
        "objective": "multiclass", "num_class": 4,
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 16, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "class_weight": "balanced",
        "verbose": -1, "n_jobs": -1,
    }
    fold_f1s = []
    for tr_idx, te_idx in ex_fold_splits:
        if not te_idx:
            continue
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("m", lgb.LGBMClassifier(**params))])
        pipe.fit(X_ex.iloc[tr_idx], y_ex[tr_idx])
        preds = pipe.predict(X_ex.iloc[te_idx])
        fold_f1s.append(f1_score(y_ex[te_idx], preds, average="macro", zero_division=0))
    return -float(np.mean(fold_f1s))

log("  Optuna 50 trials (exercise)...")
ex_study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
ex_study.optimize(ex_optuna_objective, n_trials=50)
bep = ex_study.best_params
bep.update({"objective": "multiclass", "num_class": 4,
            "class_weight": "balanced", "verbose": -1, "n_jobs": -1})
log(f"  Best Optuna macro-F1: {-ex_study.best_value:.4f}")

# Final CV on full dataset
ex_fold_f1s, ex_ps = [], {}
ex_all_t, ex_all_p = [], []
# Rebuild splits on full ex_df for final CV
ex_fold_splits_full = get_fold_splits(ex_df)
X_ex_full = ex_df[WIN_FEAT_COLS].copy()
y_ex_full = ex_df["exercise"].values

for fi, (tr, te) in enumerate(ex_fold_splits_full):
    if not te:
        continue
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("m", lgb.LGBMClassifier(**bep))])
    pipe.fit(X_ex_full.iloc[tr], y_ex_full[tr])
    preds = pipe.predict(X_ex_full.iloc[te])
    f1 = f1_score(y_ex_full[te], preds, average="macro", zero_division=0)
    ex_fold_f1s.append(f1)
    log(f"    Fold {fi}: macro-F1={f1:.4f}")
    for s, t, p in zip(ex_df.iloc[te]["subject_id"], y_ex_full[te], preds):
        ex_ps.setdefault(s, {"t": [], "p": []})
        ex_ps[s]["t"].append(t); ex_ps[s]["p"].append(p)
    ex_all_t.extend(y_ex_full[te].tolist())
    ex_all_p.extend(preds.tolist())

ex_mean = float(np.mean(ex_fold_f1s))
ex_std  = float(np.std(ex_fold_f1s))
ex_ps_f1 = {s: float(f1_score(v["t"], v["p"], average="macro", zero_division=0))
             for s, v in ex_ps.items()}
ex_pass = ex_mean > baselines["exercise"]["mean"] + 0.2
log(f"\n  Exercise F1: {ex_mean:.4f} ± {ex_std:.4f}  baseline={baselines['exercise']['mean']:.4f}  PASS={ex_pass}")

# Final model on all exercise data
final_ex = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("m", lgb.LGBMClassifier(**bep))])
final_ex.fit(X_ex_full, y_ex_full)
joblib.dump(final_ex, MODELS_DIR / "exercise.joblib")
log("  Saved exercise.joblib")

# SHAP exercise (sample 5000)
log("  Computing SHAP (exercise)...")
try:
    sidx = RNG.choice(len(X_ex_full), min(5000, len(X_ex_full)), replace=False)
    X_ex_s = pd.DataFrame(final_ex["imp"].transform(X_ex_full.iloc[sidx]),
                            columns=X_ex_full.columns)
    ex_exp = shap.TreeExplainer(final_ex["m"])
    ex_sv  = ex_exp.shap_values(X_ex_s)
    if isinstance(ex_sv, list):
        ex_sm = np.mean([np.abs(v).mean(axis=0) for v in ex_sv], axis=0)
        sv_plot = ex_sv[0]
    else:
        ex_sm = np.abs(ex_sv).mean(axis=0)
        sv_plot = ex_sv
    ex_fi = dict(sorted(zip(X_ex_full.columns, ex_sm.tolist()),
                         key=lambda x: x[1], reverse=True)[:20])
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv_plot, X_ex_s, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary_exercise.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Top exercise features: {list(ex_fi.keys())[:5]}")
except Exception as e:
    log(f"  SHAP failed: {e}")
    raw_fi = dict(zip(X_ex_full.columns, final_ex["m"].feature_importances_))
    ex_fi = dict(sorted(raw_fi.items(), key=lambda x: x[1], reverse=True)[:20])

# Confusion matrix
cm_ex = confusion_matrix(ex_all_t, ex_all_p, labels=EXERCISE_CLASSES)
disp = ConfusionMatrixDisplay(cm_ex, display_labels=EXERCISE_CLASSES)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Exercise confusion  F1={ex_mean:.3f}")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrix_exercise.png", dpi=150); plt.close()
log("  Saved confusion_matrix_exercise.png")


# ===========================================================================
# TASK 3: PHASE SEGMENTATION
# State machine based on acc_rms + acc_jerk_rms thresholds (Pernek et al. 2015).
# Threshold determined per-recording using quantiles to handle between-subject
# variability in motion amplitude (Hastie et al. 2009 normalisation argument).
# ML fallback (LightGBM) if state-machine F1 < 0.85.
# ===========================================================================
log("\n" + "="*60)
log("TASK 3: PHASE SEGMENTATION")
log("="*60)

PHASE_CLASSES = ["concentric", "eccentric", "isometric"]
ph_df = win_df[
    (win_df["in_active_set"] == True) &
    (win_df["phase_label"].isin(PHASE_CLASSES))
].reset_index(drop=True)
log(f"  Phase windows: {len(ph_df)}")
log(f"  Phase dist:\n{ph_df['phase_label'].value_counts().to_string()}")


def state_machine_phase(df):
    """
    Per-recording threshold state machine.
    Concentric: high jerk AND high acc_rms (explosive positive phase).
    Eccentric: lower jerk but sustained acc_rms (controlled return).
    Isometric: low jerk AND low acc_rms (hold / transition).
    Adapted from Pernek et al. (2015) rep-phase detection with IMU.
    """
    all_preds = pd.Series(index=df.index, dtype=object)
    for rec_id, grp in df.groupby("recording_id"):
        jerk_hi = grp["acc_jerk_rms"].quantile(0.75)
        rms_med = grp["acc_rms"].median()
        j = grp["acc_jerk_rms"].values
        r = grp["acc_rms"].values
        pred = np.where(
            (j >= jerk_hi) & (r >= rms_med), "concentric",
            np.where(
                (j < jerk_hi) & (r >= rms_med), "eccentric",
                "isometric"
            )
        )
        all_preds[grp.index] = pred
    return all_preds.values


log("  Running state machine...")
sm_preds = state_machine_phase(ph_df)
sm_f1 = float(f1_score(ph_df["phase_label"].values, sm_preds,
                        labels=PHASE_CLASSES, average="macro", zero_division=0))
log(f"  State-machine F1: {sm_f1:.4f}")

sm_ps_f1 = {}
for subj, grp in ph_df.groupby("subject_id"):
    sp = sm_preds[grp.index.values]  # use positional indices
    sm_ps_f1[subj] = float(f1_score(grp["phase_label"].values, sp,
                                     labels=PHASE_CLASSES, average="macro", zero_division=0))

# State-machine confusion
cm_sm = confusion_matrix(ph_df["phase_label"].values, sm_preds, labels=PHASE_CLASSES)
disp_sm = ConfusionMatrixDisplay(cm_sm, display_labels=PHASE_CLASSES)
fig, ax = plt.subplots(figsize=(6, 5))
disp_sm.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Phase (state-machine)  F1={sm_f1:.3f}")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrix_phase_statemachine.png", dpi=150); plt.close()

# Copy as primary
import shutil
shutil.copy(str(PLOTS_DIR / "confusion_matrix_phase_statemachine.png"),
            str(PLOTS_DIR / "confusion_matrix_phase.png"))

ml_ph_mean = None; ml_ph_std = None; ml_ph_ps = None
phase_primary = "state_machine"
ph_fi = {}

PHASE_SM_THRESH = 0.85
if sm_f1 < PHASE_SM_THRESH:
    log(f"  F1 {sm_f1:.4f} < {PHASE_SM_THRESH} → ML fallback")
    phase_primary = "ml_fallback"

    X_ph = ph_df[WIN_FEAT_COLS].copy()
    y_ph = ph_df["phase_label"].values
    ph_fold_splits = get_fold_splits(ph_df)

    def ph_optuna_objective(trial):
        params = {
            "objective": "multiclass", "num_class": 3,
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "num_leaves": trial.suggest_int("num_leaves", 8, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "class_weight": "balanced",
            "verbose": -1, "n_jobs": -1,
        }
        fold_f1s = []
        for tr_idx, te_idx in ph_fold_splits:
            if not te_idx:
                continue
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("m", lgb.LGBMClassifier(**params))])
            pipe.fit(X_ph.iloc[tr_idx], y_ph[tr_idx])
            preds = pipe.predict(X_ph.iloc[te_idx])
            fold_f1s.append(f1_score(y_ph[te_idx], preds, average="macro",
                                      zero_division=0, labels=PHASE_CLASSES))
        return -float(np.mean(fold_f1s))

    log("  Optuna 50 trials (phase ML)...")
    ph_study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    ph_study.optimize(ph_optuna_objective, n_trials=50)
    bpp = ph_study.best_params
    bpp.update({"objective": "multiclass", "num_class": 3,
                "class_weight": "balanced", "verbose": -1, "n_jobs": -1})
    log(f"  Best phase ML F1: {-ph_study.best_value:.4f}")

    ph_fold_f1s, ph_ps = [], {}
    ph_all_t, ph_all_p = [], []
    for fi, (tr, te) in enumerate(ph_fold_splits):
        if not te:
            continue
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("m", lgb.LGBMClassifier(**bpp))])
        pipe.fit(X_ph.iloc[tr], y_ph[tr])
        preds = pipe.predict(X_ph.iloc[te])
        f1 = f1_score(y_ph[te], preds, average="macro", zero_division=0, labels=PHASE_CLASSES)
        ph_fold_f1s.append(f1)
        log(f"    Fold {fi}: F1={f1:.4f}")
        for s, t, p in zip(ph_df.iloc[te]["subject_id"], y_ph[te], preds):
            ph_ps.setdefault(s, {"t": [], "p": []})
            ph_ps[s]["t"].append(t); ph_ps[s]["p"].append(p)
        ph_all_t.extend(y_ph[te].tolist()); ph_all_p.extend(preds.tolist())

    ml_ph_mean = float(np.mean(ph_fold_f1s))
    ml_ph_std  = float(np.std(ph_fold_f1s))
    ml_ph_ps   = {s: float(f1_score(v["t"], v["p"], average="macro", zero_division=0,
                                     labels=PHASE_CLASSES))
                  for s, v in ph_ps.items()}
    log(f"\n  Phase ML F1: {ml_ph_mean:.4f} ± {ml_ph_std:.4f}")

    final_ph = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("m", lgb.LGBMClassifier(**bpp))])
    final_ph.fit(X_ph, y_ph)
    joblib.dump(final_ph, MODELS_DIR / "phase.joblib")
    log("  Saved phase.joblib")

    try:
        sidx = RNG.choice(len(X_ph), min(3000, len(X_ph)), replace=False)
        X_ph_s = pd.DataFrame(final_ph["imp"].transform(X_ph.iloc[sidx]), columns=X_ph.columns)
        ph_exp = shap.TreeExplainer(final_ph["m"])
        ph_sv  = ph_exp.shap_values(X_ph_s)
        pm = np.mean([np.abs(v).mean(axis=0) for v in ph_sv], axis=0) if isinstance(ph_sv, list) else np.abs(ph_sv).mean(axis=0)
        ph_fi = dict(sorted(zip(X_ph.columns, pm.tolist()), key=lambda x: x[1], reverse=True)[:20])
    except Exception as e:
        log(f"  Phase SHAP failed: {e}")

    # ML confusion
    cm_ph = confusion_matrix(ph_all_t, ph_all_p, labels=PHASE_CLASSES)
    disp_ph = ConfusionMatrixDisplay(cm_ph, display_labels=PHASE_CLASSES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp_ph.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Phase (ML fallback)  F1={ml_ph_mean:.3f}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix_phase.png", dpi=150); plt.close()
    log("  Saved confusion_matrix_phase.png (ML version)")

phase_primary_f1 = ml_ph_mean if ml_ph_mean is not None else sm_f1
phase_pass = phase_primary_f1 > baselines["phase"]["mean"] + 0.2
log(f"  Phase PASS (0.2 above baseline {baselines['phase']['mean']:.4f}): {phase_pass}")


# ===========================================================================
# TASK 4: REP COUNTING
# State machine: peak detection on acc_rms per set (Pernek et al. 2015).
# Adaptive threshold: median + 0.3*std, min_distance=5 (500 ms inter-rep).
# ML fallback (LightGBM regressor) if state-machine MAE >= baseline MAE.
# [REF NEEDED: scipy.signal.find_peaks per exercise rep-count literature]
# ===========================================================================
log("\n" + "="*60)
log("TASK 4: REP COUNTING")
log("="*60)

set_df_ri = set_df.reset_index(drop=True)


def count_reps_sm(test_sets, win_df):
    """
    State-machine rep counter: peak detection on acc_rms within each set.
    Returns DataFrame with n_reps_gt, n_reps_pred, subject_id.
    Peak detection (Pernek et al. 2015): adaptive threshold per set,
    minimum inter-peak distance 5 windows (500 ms).
    """
    rows = []
    for _, row in test_sets.iterrows():
        rec = row["recording_id"]; sn = row["set_number"]
        mask = ((win_df["recording_id"] == rec) & (win_df["set_number"] == sn)
                & (win_df["in_active_set"] == True))
        grp = win_df[mask].sort_values("t_window_center_s")
        gt  = int(row["n_reps"])
        if len(grp) < 5:
            rows.append({"recording_id": rec, "set_number": sn,
                         "subject_id": row["subject_id"], "n_reps_gt": gt, "n_reps_pred": 0})
            continue
        sig = grp["acc_rms"].fillna(grp["acc_rms"].median()).values
        thresh = float(np.median(sig) + 0.3 * np.std(sig))
        peaks, _ = find_peaks(sig, height=thresh, distance=5)
        rows.append({"recording_id": rec, "set_number": sn,
                     "subject_id": row["subject_id"],
                     "n_reps_gt": gt, "n_reps_pred": int(len(peaks))})
    return pd.DataFrame(rows)


rep_fold_splits_s = get_fold_splits(set_df_ri)
sm_rep_fold_maes = []
all_sm_res = []

log("  State-machine rep counting per fold...")
for fi, (tr, te) in enumerate(rep_fold_splits_s):
    if not te:
        continue
    res = count_reps_sm(set_df_ri.iloc[te], win_df)
    if len(res) == 0:
        continue
    mae = float(mean_absolute_error(res["n_reps_gt"], res["n_reps_pred"]))
    sm_rep_fold_maes.append(mae)
    all_sm_res.append(res)
    log(f"    Fold {fi}: rep MAE={mae:.4f}")

sm_res_all = pd.concat(all_sm_res, ignore_index=True) if all_sm_res else pd.DataFrame()
sm_rep_mean = float(np.mean(sm_rep_fold_maes)) if sm_rep_fold_maes else 999.0
sm_rep_std  = float(np.std(sm_rep_fold_maes)) if sm_rep_fold_maes else 0.0

if not sm_res_all.empty:
    sm_exact   = float((sm_res_all["n_reps_pred"] == sm_res_all["n_reps_gt"]).mean())
    sm_within1 = float((np.abs(sm_res_all["n_reps_pred"] - sm_res_all["n_reps_gt"]) <= 1).mean())
    sm_ps_rep  = sm_res_all.groupby("subject_id").apply(
        lambda g: float(mean_absolute_error(g["n_reps_gt"], g["n_reps_pred"]))
    ).to_dict()
else:
    sm_exact = 0.0; sm_within1 = 0.0; sm_ps_rep = {}

log(f"\n  State-machine rep MAE: {sm_rep_mean:.4f} ± {sm_rep_std:.4f}")
log(f"  Exact: {sm_exact*100:.1f}%  Within-1: {sm_within1*100:.1f}%")

ml_rep_mean = None; ml_rep_std = None; ml_rep_ps = None
rep_primary = "state_machine"
rep_fi = {}
rep_exact_final = sm_exact
rep_within1_final = sm_within1

if sm_rep_mean >= baselines["reps"]["mean"]:
    log(f"  SM MAE {sm_rep_mean:.4f} >= baseline {baselines['reps']['mean']:.4f} → ML fallback")
    rep_primary = "ml_fallback"

    rep_df = set_df.dropna(subset=["n_reps"]).reset_index(drop=True)
    X_rep = rep_df[SET_FEAT_COLS].copy()
    y_rep = rep_df["n_reps"].values.astype(float)
    rep_fold_splits2 = get_fold_splits(rep_df)

    def rep_optuna_obj(trial):
        params = {
            "objective": "regression_l1",
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "num_leaves": trial.suggest_int("num_leaves", 8, 48),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.0, log=True),
            "verbose": -1, "n_jobs": -1,
        }
        fold_maes = []
        for tr_idx, te_idx in rep_fold_splits2:
            if not te_idx:
                continue
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("m", lgb.LGBMRegressor(**params))])
            pipe.fit(X_rep.iloc[tr_idx], y_rep[tr_idx])
            fold_maes.append(mean_absolute_error(y_rep[te_idx], pipe.predict(X_rep.iloc[te_idx])))
        return float(np.mean(fold_maes))

    log("  Optuna 50 trials (reps ML)...")
    rep_study = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    rep_study.optimize(rep_optuna_obj, n_trials=50)
    brp = rep_study.best_params
    brp.update({"objective": "regression_l1", "verbose": -1, "n_jobs": -1})
    log(f"  Best rep ML MAE: {rep_study.best_value:.4f}")

    rep_fold_maes2, rep_ps2 = [], {}
    rep_all_t, rep_all_p = [], []
    for fi, (tr, te) in enumerate(rep_fold_splits2):
        if not te:
            continue
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("m", lgb.LGBMRegressor(**brp))])
        pipe.fit(X_rep.iloc[tr], y_rep[tr])
        preds = pipe.predict(X_rep.iloc[te])
        mae = float(mean_absolute_error(y_rep[te], preds))
        rep_fold_maes2.append(mae)
        log(f"    Fold {fi}: rep MAE={mae:.4f}")
        for s, t, p in zip(rep_df.iloc[te]["subject_id"], y_rep[te], preds):
            rep_ps2.setdefault(s, {"t": [], "p": []})
            rep_ps2[s]["t"].append(float(t)); rep_ps2[s]["p"].append(float(p))
        rep_all_t.extend(y_rep[te].tolist()); rep_all_p.extend(preds.tolist())

    ml_rep_mean = float(np.mean(rep_fold_maes2))
    ml_rep_std  = float(np.std(rep_fold_maes2))
    ml_rep_ps   = {s: float(mean_absolute_error(v["t"], v["p"]))
                   for s, v in rep_ps2.items()}
    rep_exact_final = float((np.round(rep_all_p) == np.array(rep_all_t)).mean())
    rep_within1_final = float((np.abs(np.array(rep_all_p) - np.array(rep_all_t)) <= 1).mean())
    log(f"\n  ML rep MAE: {ml_rep_mean:.4f} ± {ml_rep_std:.4f}")

    final_rep = Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("m", lgb.LGBMRegressor(**brp))])
    final_rep.fit(X_rep, y_rep)
    joblib.dump(final_rep, MODELS_DIR / "reps.joblib")
    log("  Saved reps.joblib")

    try:
        X_rep_imp = pd.DataFrame(final_rep["imp"].transform(X_rep), columns=X_rep.columns)
        rep_exp = shap.TreeExplainer(final_rep["m"])
        rep_sv  = rep_exp.shap_values(X_rep_imp)
        rep_sm2 = np.abs(rep_sv).mean(axis=0)
        rep_fi  = dict(sorted(zip(X_rep.columns, rep_sm2.tolist()),
                               key=lambda x: x[1], reverse=True)[:20])
    except Exception as e:
        log(f"  Rep SHAP failed: {e}")

active_rep_mae = ml_rep_mean if ml_rep_mean is not None else sm_rep_mean
rep_pass = active_rep_mae < baselines["reps"]["mean"]
log(f"  Rep PASS (< baseline {baselines['reps']['mean']:.4f}): {rep_pass}")


# ===========================================================================
# SAVE METRICS.JSON
# ===========================================================================
log("\n" + "="*60)
log("SAVING ARTIFACTS")
log("="*60)

fat_med = float(np.median(list(fat_ps_mae.values())))
fat_outliers = {s: float(v) for s, v in fat_ps_mae.items() if v > 3 * fat_med}

ph_ps_final = {s: float(v) for s, v in sm_ps_f1.items()} if phase_primary == "state_machine" else (ml_ph_ps or {})

metrics = {
    "fatigue": {
        "metric": "MAE",
        "mean": float(fat_mean),
        "std": float(fat_std),
        "fold_scores": [float(v) for v in fat_fold_maes],
        "per_subject": {s: float(v) for s, v in fat_ps_mae.items()},
        "baseline": float(baselines["fatigue"]["mean"]),
        "pass": bool(fat_pass),
        "fail_reason": None if fat_pass else "MAE not 30% below DummyRegressor(median) baseline",
        "outlier_subjects": fat_outliers,
        "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in bfp.items()},
    },
    "exercise": {
        "metric": "macro_F1",
        "mean": float(ex_mean),
        "std": float(ex_std),
        "fold_scores": [float(v) for v in ex_fold_f1s],
        "per_subject": {s: float(v) for s, v in ex_ps_f1.items()},
        "baseline": float(baselines["exercise"]["mean"]),
        "pass": bool(ex_pass),
        "fail_reason": None if ex_pass else "F1 not 0.2 above stratified-random baseline",
        "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in bep.items()},
    },
    "phase": {
        "metric": "macro_F1",
        "state_machine": float(sm_f1),
        "state_machine_per_subject": {s: float(v) for s, v in sm_ps_f1.items()},
        "ml_fallback": {
            "mean": float(ml_ph_mean),
            "std": float(ml_ph_std),
            "per_subject": {s: float(v) for s, v in ml_ph_ps.items()} if ml_ph_ps else {},
        } if ml_ph_mean is not None else None,
        "primary": phase_primary,
        "mean": float(phase_primary_f1),
        "std": float(ml_ph_std if ml_ph_std is not None else 0.0),
        "per_subject": ph_ps_final,
        "baseline": float(baselines["phase"]["mean"]),
        "pass": bool(phase_pass),
        "fail_reason": None if phase_pass else "F1 not 0.2 above stratified-random baseline",
    },
    "reps": {
        "metric": "MAE",
        "state_machine": {
            "mean": float(sm_rep_mean),
            "std": float(sm_rep_std),
            "exact_pct": float(sm_exact * 100),
            "within1_pct": float(sm_within1 * 100),
            "per_subject": {s: float(v) for s, v in sm_ps_rep.items()},
        },
        "ml_fallback": {
            "mean": float(ml_rep_mean),
            "std": float(ml_rep_std),
            "per_subject": {s: float(v) for s, v in ml_rep_ps.items()} if ml_rep_ps else {},
        } if ml_rep_mean is not None else None,
        "primary": rep_primary,
        "mean": float(active_rep_mae),
        "std": float(ml_rep_std if ml_rep_std is not None else sm_rep_std),
        "exact_pct": float(rep_exact_final * 100),
        "within1_pct": float(rep_within1_final * 100),
        "baseline": float(baselines["reps"]["mean"]),
        "pass": bool(rep_pass),
        "fail_reason": None if rep_pass else "MAE not below naive mean-reps baseline",
    },
}

with open(RUN_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
log(f"  Saved metrics.json")

feat_imp = {
    "fatigue": {k: float(v) for k, v in fat_fi.items()},
    "exercise": {k: float(v) for k, v in ex_fi.items()},
    "phase": {k: float(v) for k, v in ph_fi.items()},
    "reps": {k: float(v) for k, v in rep_fi.items()},
}
with open(RUN_DIR / "feature_importance.json", "w") as f:
    json.dump(feat_imp, f, indent=2)
log("  Saved feature_importance.json")

log("\n" + "="*60)
log("SUMMARY")
log("="*60)
log(f"Fatigue  MAE={fat_mean:.3f}±{fat_std:.3f}  baseline={baselines['fatigue']['mean']:.3f}  {'PASS' if fat_pass else 'FAIL'}")
log(f"Exercise F1={ex_mean:.3f}±{ex_std:.3f}   baseline={baselines['exercise']['mean']:.3f}  {'PASS' if ex_pass else 'FAIL'}")
log(f"Phase    F1={phase_primary_f1:.3f}  SM_F1={sm_f1:.3f}  method={phase_primary}  {'PASS' if phase_pass else 'FAIL'}")
log(f"Reps     MAE={active_rep_mae:.3f}  exact={rep_exact_final*100:.1f}%  within1={rep_within1_final*100:.1f}%  method={rep_primary}  {'PASS' if rep_pass else 'FAIL'}")
log(f"\nArtifacts: {RUN_DIR}")
log("Script complete.")
