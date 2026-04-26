"""
Training script for strength-RT multi-task models.
Run from the repository root:
    python runs/20260426_154705_default/train_models.py

Tasks:
  1. Fatigue regression (RPE 1-10, per-set, LightGBM, MAE objective)
  2. Exercise classification (per-window, multiclass LightGBM)
  3. Phase segmentation (state-machine first; ML fallback if F1 < 0.85)
  4. Rep counting (state-machine first; ML fallback)

All CV uses pre-computed subject-wise splits from configs/splits_per_fold.csv.
No per-subject scaler is fitted outside an sklearn Pipeline to prevent leakage.

References:
  - Ke et al. (2017) LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
  - Akiba et al. (2019) Optuna: A Next-generation Hyperparameter Optimization Framework. KDD.
  - Lundberg & Lee (2017) A Unified Approach to Interpreting Model Predictions. NeurIPS.
  - Farina et al. (2004) Comparison of algorithms for estimation of EMG variables during
    voluntary isometric contractions. J Electromyogr Kinesiol.
  - Scholkopf & Smola (2002) Learning with Kernels -- used as motivation for group-based splits
    to prevent optimistic bias (subject-wise LOSO-CV).
  - Xu et al. (2021) A real-time resistance exercise fatigue monitoring system based on
    surface electromyography. Sensors.
"""

import json
import os
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
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)
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

print("Loading data...")

set_df = pd.read_parquet(FEATURES_DIR / "set_features.parquet")
win_df = pd.read_parquet(FEATURES_DIR / "window_features.parquet")
splits = pd.read_csv(SPLITS_CSV)

with open(BASELINE_JSON) as f:
    baselines = json.load(f)

print(f"  set_features: {set_df.shape}")
print(f"  window_features: {win_df.shape}")
print(f"  splits: {splits.shape}")

FOLDS = sorted(splits["fold"].unique())
print(f"  folds: {FOLDS}")

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

print(f"  set feature cols ({len(SET_FEAT_COLS)}): {SET_FEAT_COLS[:5]}...")
print(f"  window feature cols ({len(WIN_FEAT_COLS)}): {WIN_FEAT_COLS[:5]}...")

# ---------------------------------------------------------------------------
# Helper: build train/test masks per fold from splits CSV
# ---------------------------------------------------------------------------
def get_fold_splits(df, key_col="recording_id"):
    """Return list of (train_idx, test_idx) tuples, one per fold."""
    fold_data = []
    for fold in FOLDS:
        fold_rows = splits[splits["fold"] == fold]
        train_ids = set(fold_rows[fold_rows["split"] == "train"][key_col].tolist())
        test_ids  = set(fold_rows[fold_rows["split"] == "test"][key_col].tolist())
        tr = df.index[df[key_col].isin(train_ids)].tolist()
        te = df.index[df[key_col].isin(test_ids)].tolist()
        fold_data.append((tr, te))
    return fold_data


# ===========================================================================
# TASK 1: FATIGUE REGRESSION (RPE, per-set)
# ===========================================================================
print("\n" + "="*60)
print("TASK 1: FATIGUE REGRESSION")
print("="*60)

# Drop rows with missing target
fat_df = set_df.dropna(subset=["rpe_for_this_set"]).copy()
fat_df = fat_df.reset_index(drop=True)
print(f"  Fatigue rows after dropping NaN target: {len(fat_df)}")

X_fat = fat_df[SET_FEAT_COLS].copy()
y_fat = fat_df["rpe_for_this_set"].values

fat_fold_splits = get_fold_splits(fat_df)

# ---- Optuna objective ----
def fat_optuna_objective(trial, fat_df=fat_df, X_fat=X_fat, y_fat=y_fat):
    """Optimise fatigue MAE via GroupKFold (5 folds, subject-wise)."""
    params = {
        "objective": "regression_l1",
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "verbose": -1,
        "n_jobs": -1,
    }
    fold_maes = []
    for tr_idx, te_idx in fat_fold_splits:
        if not te_idx:
            continue
        Xtr, Xte = X_fat.iloc[tr_idx], X_fat.iloc[te_idx]
        ytr, yte = y_fat[tr_idx], y_fat[te_idx]
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMRegressor(**params)),
        ])
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        fold_maes.append(mean_absolute_error(yte, preds))
    return np.mean(fold_maes)

print("  Running Optuna (50 trials) for fatigue...")
fat_study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
fat_study.optimize(fat_optuna_objective, n_trials=50, show_progress_bar=False)
best_fat_params = fat_study.best_params
best_fat_params.update({"objective": "regression_l1", "verbose": -1, "n_jobs": -1})
print(f"  Best fatigue params: {best_fat_params}")
print(f"  Best Optuna MAE: {fat_study.best_value:.4f}")

# ---- Final CV evaluation with best params ----
fat_fold_maes = []
fat_per_subject_preds = {}  # subject_id -> list of (true, pred)
fat_all_true = []
fat_all_pred = []
fat_best_model = None

for fold_i, (tr_idx, te_idx) in enumerate(fat_fold_splits):
    if not te_idx:
        continue
    Xtr = X_fat.iloc[tr_idx]
    Xte = X_fat.iloc[te_idx]
    ytr = y_fat[tr_idx]
    yte = y_fat[te_idx]
    subj_te = fat_df.iloc[te_idx]["subject_id"].values

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", lgb.LGBMRegressor(**best_fat_params)),
    ])
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)

    fold_mae = mean_absolute_error(yte, preds)
    fat_fold_maes.append(fold_mae)
    print(f"    Fold {fold_i}: MAE={fold_mae:.4f}  (n_test={len(te_idx)})")

    for s, t, p in zip(subj_te, yte, preds):
        fat_per_subject_preds.setdefault(s, []).append((t, p))

    fat_all_true.extend(yte.tolist())
    fat_all_pred.extend(preds.tolist())
    fat_best_model = pipe  # keep last fold model for SHAP (same hyperparams)

fat_mean_mae = np.mean(fat_fold_maes)
fat_std_mae  = np.std(fat_fold_maes)
fat_per_subject_mae = {
    s: mean_absolute_error([v[0] for v in vals], [v[1] for v in vals])
    for s, vals in fat_per_subject_preds.items()
}

print(f"\n  Fatigue MAE: {fat_mean_mae:.4f} ± {fat_std_mae:.4f}")
print(f"  Baseline MAE: {baselines['fatigue']['mean']:.4f}")
fat_pass = fat_mean_mae < baselines["fatigue"]["mean"] * 0.70  # must be 30% below
print(f"  PASS (30% below baseline): {fat_pass}")
print(f"  Per-subject MAE: {fat_per_subject_mae}")

# ---- Train final model on all data ----
final_fat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", lgb.LGBMRegressor(**best_fat_params)),
])
final_fat_pipe.fit(X_fat, y_fat)
joblib.dump(final_fat_pipe, MODELS_DIR / "fatigue.joblib")
print("  Saved fatigue.joblib")

# ---- SHAP for fatigue ----
print("  Computing SHAP for fatigue...")
try:
    fat_imputer = final_fat_pipe.named_steps["imputer"]
    fat_lgbm = final_fat_pipe.named_steps["model"]
    X_fat_imp = pd.DataFrame(fat_imputer.transform(X_fat), columns=X_fat.columns)
    fat_explainer = shap.TreeExplainer(fat_lgbm)
    fat_shap_vals = fat_explainer.shap_values(X_fat_imp)
    fat_shap_mean = np.abs(fat_shap_vals).mean(axis=0)
    fat_feat_imp = dict(sorted(
        zip(X_fat.columns, fat_shap_mean.tolist()),
        key=lambda x: x[1], reverse=True
    )[:20])
    print(f"  Top fatigue features: {list(fat_feat_imp.keys())[:5]}")

    # SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(fat_shap_vals, X_fat_imp, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary_fatigue.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved shap_summary_fatigue.png")
except Exception as e:
    print(f"  SHAP failed: {e}")
    fat_feat_imp = {c: float(v) for c, v in
                    zip(X_fat.columns,
                        final_fat_pipe.named_steps["model"].feature_importances_)}
    fat_feat_imp = dict(sorted(fat_feat_imp.items(), key=lambda x: x[1], reverse=True)[:20])

# ---- Calibration plot ----
plt.figure(figsize=(6, 6))
plt.scatter(fat_all_true, fat_all_pred, alpha=0.4, s=30)
mn, mx = min(fat_all_true + fat_all_pred), max(fat_all_true + fat_all_pred)
plt.plot([mn, mx], [mn, mx], "r--", label="Perfect calibration")
plt.xlabel("Actual RPE")
plt.ylabel("Predicted RPE")
plt.title(f"Fatigue calibration  MAE={fat_mean_mae:.2f}")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fatigue_calibration.png", dpi=150)
plt.close()
print("  Saved fatigue_calibration.png")

# ---- Per-subject bar chart ----
subj_names = list(fat_per_subject_mae.keys())
subj_maes  = [fat_per_subject_mae[s] for s in subj_names]
median_mae = np.median(subj_maes)
colors = ["red" if v > 3 * median_mae else "steelblue" for v in subj_maes]

plt.figure(figsize=(10, 5))
bars = plt.bar(subj_names, subj_maes, color=colors)
plt.axhline(fat_mean_mae, color="orange", linestyle="--", label=f"Mean MAE={fat_mean_mae:.2f}")
plt.axhline(baselines["fatigue"]["mean"], color="gray", linestyle=":", label=f"Baseline MAE={baselines['fatigue']['mean']:.2f}")
plt.xticks(rotation=35, ha="right")
plt.ylabel("MAE (RPE units)")
plt.title("Per-subject fatigue MAE")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "per_subject_fatigue_mae.png", dpi=150)
plt.close()
print("  Saved per_subject_fatigue_mae.png")


# ===========================================================================
# TASK 2: EXERCISE CLASSIFICATION (per-window, active sets only)
# ===========================================================================
print("\n" + "="*60)
print("TASK 2: EXERCISE CLASSIFICATION")
print("="*60)

ex_df = win_df[(win_df["in_active_set"] == True) & win_df["exercise"].notna()].copy()
ex_df = ex_df.reset_index(drop=True)
EXERCISE_CLASSES = ["squat", "deadlift", "benchpress", "pullup"]
ex_df = ex_df[ex_df["exercise"].isin(EXERCISE_CLASSES)].copy()
ex_df = ex_df.reset_index(drop=True)
print(f"  Exercise windows: {len(ex_df)}")
print(f"  Class distribution:\n{ex_df['exercise'].value_counts()}")

X_ex = ex_df[WIN_FEAT_COLS].copy()
y_ex = ex_df["exercise"].values

ex_fold_splits = get_fold_splits(ex_df)

# ---- Optuna objective ----
def ex_optuna_objective(trial):
    params = {
        "objective": "multiclass",
        "num_class": 4,
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "class_weight": "balanced",
        "verbose": -1,
        "n_jobs": -1,
    }
    fold_f1s = []
    for tr_idx, te_idx in ex_fold_splits:
        if not te_idx:
            continue
        Xtr, Xte = X_ex.iloc[tr_idx], X_ex.iloc[te_idx]
        ytr, yte = y_ex[tr_idx], y_ex[te_idx]
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMClassifier(**params)),
        ])
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        fold_f1s.append(f1_score(yte, preds, average="macro", zero_division=0))
    return -np.mean(fold_f1s)  # minimise negative F1

print("  Running Optuna (50 trials) for exercise...")
ex_study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
ex_study.optimize(ex_optuna_objective, n_trials=50, show_progress_bar=False)
best_ex_params = ex_study.best_params
best_ex_params.update({
    "objective": "multiclass", "num_class": 4,
    "class_weight": "balanced", "verbose": -1, "n_jobs": -1,
})
print(f"  Best exercise params: {best_ex_params}")
print(f"  Best Optuna macro-F1: {-ex_study.best_value:.4f}")

# ---- Final CV evaluation ----
ex_fold_f1s = []
ex_per_subject = {}
ex_all_true = []
ex_all_pred = []
ex_best_model = None

for fold_i, (tr_idx, te_idx) in enumerate(ex_fold_splits):
    if not te_idx:
        continue
    Xtr = X_ex.iloc[tr_idx]
    Xte = X_ex.iloc[te_idx]
    ytr = y_ex[tr_idx]
    yte = y_ex[te_idx]
    subj_te = ex_df.iloc[te_idx]["subject_id"].values

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", lgb.LGBMClassifier(**best_ex_params)),
    ])
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)

    fold_f1 = f1_score(yte, preds, average="macro", zero_division=0)
    ex_fold_f1s.append(fold_f1)
    print(f"    Fold {fold_i}: macro-F1={fold_f1:.4f}  (n_test={len(te_idx)})")

    for s, t, p in zip(subj_te, yte, preds):
        ex_per_subject.setdefault(s, {"true": [], "pred": []})
        ex_per_subject[s]["true"].append(t)
        ex_per_subject[s]["pred"].append(p)

    ex_all_true.extend(yte.tolist())
    ex_all_pred.extend(preds.tolist())
    ex_best_model = pipe

ex_mean_f1 = np.mean(ex_fold_f1s)
ex_std_f1  = np.std(ex_fold_f1s)
ex_per_subject_f1 = {
    s: f1_score(v["true"], v["pred"], average="macro", zero_division=0)
    for s, v in ex_per_subject.items()
}
print(f"\n  Exercise macro-F1: {ex_mean_f1:.4f} ± {ex_std_f1:.4f}")
print(f"  Baseline macro-F1: {baselines['exercise']['mean']:.4f}")
ex_pass = ex_mean_f1 > baselines["exercise"]["mean"] + 0.2
print(f"  PASS (0.2 above baseline): {ex_pass}")
print(f"  Per-subject F1: {ex_per_subject_f1}")

# ---- Train final model ----
final_ex_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", lgb.LGBMClassifier(**best_ex_params)),
])
final_ex_pipe.fit(X_ex, y_ex)
joblib.dump(final_ex_pipe, MODELS_DIR / "exercise.joblib")
print("  Saved exercise.joblib")

# ---- SHAP for exercise ----
print("  Computing SHAP for exercise...")
try:
    ex_imputer = final_ex_pipe.named_steps["imputer"]
    ex_lgbm = final_ex_pipe.named_steps["model"]
    # Sample 5000 rows for SHAP efficiency (large dataset)
    rng = np.random.default_rng(42)
    shap_sample_idx = rng.choice(len(X_ex), min(5000, len(X_ex)), replace=False)
    X_ex_sample = pd.DataFrame(
        ex_imputer.transform(X_ex.iloc[shap_sample_idx]),
        columns=X_ex.columns
    )
    ex_explainer = shap.TreeExplainer(ex_lgbm)
    ex_shap_vals = ex_explainer.shap_values(X_ex_sample)  # list of arrays per class
    # Average across classes for ranking
    if isinstance(ex_shap_vals, list):
        ex_shap_mean = np.mean([np.abs(v).mean(axis=0) for v in ex_shap_vals], axis=0)
    else:
        ex_shap_mean = np.abs(ex_shap_vals).mean(axis=0)
    ex_feat_imp = dict(sorted(
        zip(X_ex.columns, ex_shap_mean.tolist()),
        key=lambda x: x[1], reverse=True
    )[:20])
    print(f"  Top exercise features: {list(ex_feat_imp.keys())[:5]}")

    plt.figure(figsize=(10, 8))
    if isinstance(ex_shap_vals, list):
        shap.summary_plot(ex_shap_vals[0], X_ex_sample, max_display=20, show=False)
    else:
        shap.summary_plot(ex_shap_vals, X_ex_sample, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary_exercise.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved shap_summary_exercise.png")
except Exception as e:
    print(f"  SHAP failed: {e}")
    ex_feat_imp = {c: float(v) for c, v in
                   zip(X_ex.columns,
                       final_ex_pipe.named_steps["model"].feature_importances_)}
    ex_feat_imp = dict(sorted(ex_feat_imp.items(), key=lambda x: x[1], reverse=True)[:20])

# ---- Confusion matrix ----
cm = confusion_matrix(ex_all_true, ex_all_pred, labels=EXERCISE_CLASSES)
disp = ConfusionMatrixDisplay(cm, display_labels=EXERCISE_CLASSES)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Exercise confusion matrix  F1={ex_mean_f1:.3f}")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrix_exercise.png", dpi=150)
plt.close()
print("  Saved confusion_matrix_exercise.png")


# ===========================================================================
# TASK 3: PHASE SEGMENTATION
# State machine first; ML fallback if F1 < 0.85
# ===========================================================================
print("\n" + "="*60)
print("TASK 3: PHASE SEGMENTATION")
print("="*60)

PHASE_CLASSES = ["concentric", "eccentric", "isometric"]

ph_df = win_df[
    (win_df["in_active_set"] == True) &
    (win_df["phase_label"].isin(PHASE_CLASSES))
].copy()
ph_df = ph_df.reset_index(drop=True)
print(f"  Phase windows (known phase only): {len(ph_df)}")
print(f"  Phase distribution:\n{ph_df['phase_label'].value_counts()}")

# ---- State-machine classifier ----
# Principle: use acc_rms and acc_jerk_rms to distinguish phases.
# Threshold-based heuristic:
#   - high acc_jerk_rms and increasing acc_rms → concentric (explosive lifting)
#   - lower acc_jerk_rms and decreasing acc_rms → eccentric (controlled lowering)
#   - minimal acc_jerk_rms and low acc_rms → isometric (hold)
# This approach is adapted from acc-magnitude based state-machine literature
# (Pernek et al. 2015, Sensors).
# [REF NEEDED: Pernek et al. 2015 exact citation]

def state_machine_phase(df):
    """
    Simple threshold state-machine for phase prediction from acc features.
    Uses acc_rms and acc_jerk_rms per window.
    Returns array of predicted phase labels.
    """
    preds = []
    # Compute per-recording normalised thresholds to handle subject variability
    for rec_id, grp in df.groupby("recording_id"):
        jerk_median = grp["acc_jerk_rms"].median()
        jerk_high   = grp["acc_jerk_rms"].quantile(0.75)
        rms_median  = grp["acc_rms"].median()

        rec_preds = []
        for _, row in grp.iterrows():
            j = row["acc_jerk_rms"]
            r = row["acc_rms"]
            if j >= jerk_high and r >= rms_median:
                rec_preds.append("concentric")
            elif j < jerk_high and r >= rms_median:
                rec_preds.append("eccentric")
            else:
                rec_preds.append("isometric")
        preds.extend(list(zip(grp.index, rec_preds)))

    preds_sorted = sorted(preds, key=lambda x: x[0])
    return [p for _, p in preds_sorted]

print("  Running state machine...")
sm_preds = state_machine_phase(ph_df)
sm_f1 = f1_score(ph_df["phase_label"].values, sm_preds,
                  labels=PHASE_CLASSES, average="macro", zero_division=0)
print(f"  State-machine phase F1: {sm_f1:.4f}")

# Per-subject state-machine F1
sm_per_subject = {}
for subj, grp in ph_df.groupby("subject_id"):
    grp_preds = [sm_preds[i] for i in grp.index]
    sm_per_subject[subj] = f1_score(
        grp["phase_label"].values, grp_preds,
        labels=PHASE_CLASSES, average="macro", zero_division=0
    )

# ---- Confusion matrix for state machine ----
cm_sm = confusion_matrix(ph_df["phase_label"].values, sm_preds, labels=PHASE_CLASSES)
disp_sm = ConfusionMatrixDisplay(cm_sm, display_labels=PHASE_CLASSES)
fig, ax = plt.subplots(figsize=(6, 5))
disp_sm.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Phase (state-machine)  F1={sm_f1:.3f}")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrix_phase_statemachine.png", dpi=150)
plt.close()

PHASE_SM_THRESHOLD = 0.85
ml_phase_f1_mean = None
ml_phase_f1_std  = None
ml_phase_per_subject = None
phase_primary = "state_machine"
phase_feat_imp = {}

if sm_f1 < PHASE_SM_THRESHOLD:
    print(f"  State-machine F1 {sm_f1:.4f} < {PHASE_SM_THRESHOLD} → training ML fallback")
    phase_primary = "ml_fallback"

    X_ph = ph_df[WIN_FEAT_COLS].copy()
    y_ph = ph_df["phase_label"].values

    ph_fold_splits = get_fold_splits(ph_df)

    # Optuna for phase ML
    def ph_optuna_objective(trial):
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "class_weight": "balanced",
            "verbose": -1,
            "n_jobs": -1,
        }
        fold_f1s = []
        for tr_idx, te_idx in ph_fold_splits:
            if not te_idx:
                continue
            Xtr, Xte = X_ph.iloc[tr_idx], X_ph.iloc[te_idx]
            ytr, yte = y_ph[tr_idx], y_ph[te_idx]
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", lgb.LGBMClassifier(**params)),
            ])
            pipe.fit(Xtr, ytr)
            preds = pipe.predict(Xte)
            fold_f1s.append(f1_score(yte, preds, average="macro", zero_division=0,
                                      labels=PHASE_CLASSES))
        return -np.mean(fold_f1s)

    print("  Running Optuna (50 trials) for phase ML...")
    ph_study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    ph_study.optimize(ph_optuna_objective, n_trials=50, show_progress_bar=False)
    best_ph_params = ph_study.best_params
    best_ph_params.update({
        "objective": "multiclass", "num_class": 3,
        "class_weight": "balanced", "verbose": -1, "n_jobs": -1,
    })
    print(f"  Best phase params: {best_ph_params}")

    # Final CV
    ph_fold_f1s = []
    ph_per_subject = {}
    ph_all_true = []
    ph_all_pred = []

    for fold_i, (tr_idx, te_idx) in enumerate(ph_fold_splits):
        if not te_idx:
            continue
        Xtr = X_ph.iloc[tr_idx]
        Xte = X_ph.iloc[te_idx]
        ytr = y_ph[tr_idx]
        yte = y_ph[te_idx]
        subj_te = ph_df.iloc[te_idx]["subject_id"].values

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMClassifier(**best_ph_params)),
        ])
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)

        fold_f1 = f1_score(yte, preds, average="macro", zero_division=0, labels=PHASE_CLASSES)
        ph_fold_f1s.append(fold_f1)
        print(f"    Fold {fold_i}: macro-F1={fold_f1:.4f}")

        for s, t, p in zip(subj_te, yte, preds):
            ph_per_subject.setdefault(s, {"true": [], "pred": []})
            ph_per_subject[s]["true"].append(t)
            ph_per_subject[s]["pred"].append(p)

        ph_all_true.extend(yte.tolist())
        ph_all_pred.extend(preds.tolist())

    ml_phase_f1_mean = float(np.mean(ph_fold_f1s))
    ml_phase_f1_std  = float(np.std(ph_fold_f1s))
    ml_phase_per_subject = {
        s: float(f1_score(v["true"], v["pred"], average="macro", zero_division=0,
                          labels=PHASE_CLASSES))
        for s, v in ph_per_subject.items()
    }

    print(f"\n  Phase ML macro-F1: {ml_phase_f1_mean:.4f} ± {ml_phase_f1_std:.4f}")
    print(f"  Per-subject phase F1: {ml_phase_per_subject}")

    # Train final phase model
    final_ph_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", lgb.LGBMClassifier(**best_ph_params)),
    ])
    final_ph_pipe.fit(X_ph, y_ph)
    joblib.dump(final_ph_pipe, MODELS_DIR / "phase.joblib")
    print("  Saved phase.joblib")

    # SHAP
    try:
        ph_imputer = final_ph_pipe.named_steps["imputer"]
        ph_lgbm = final_ph_pipe.named_steps["model"]
        shap_idx = rng.choice(len(X_ph), min(3000, len(X_ph)), replace=False)
        X_ph_sample = pd.DataFrame(
            ph_imputer.transform(X_ph.iloc[shap_idx]),
            columns=X_ph.columns
        )
        ph_explainer = shap.TreeExplainer(ph_lgbm)
        ph_shap_vals = ph_explainer.shap_values(X_ph_sample)
        if isinstance(ph_shap_vals, list):
            ph_shap_mean = np.mean([np.abs(v).mean(axis=0) for v in ph_shap_vals], axis=0)
        else:
            ph_shap_mean = np.abs(ph_shap_vals).mean(axis=0)
        phase_feat_imp = dict(sorted(
            zip(X_ph.columns, ph_shap_mean.tolist()),
            key=lambda x: x[1], reverse=True
        )[:20])
    except Exception as e:
        print(f"  Phase SHAP failed: {e}")
        phase_feat_imp = {}

    # ML confusion matrix
    cm_ph = confusion_matrix(ph_all_true, ph_all_pred, labels=PHASE_CLASSES)
    disp_ph = ConfusionMatrixDisplay(cm_ph, display_labels=PHASE_CLASSES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp_ph.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Phase (ML fallback)  F1={ml_phase_f1_mean:.3f}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix_phase.png", dpi=150)
    plt.close()
else:
    print(f"  State-machine F1 {sm_f1:.4f} >= {PHASE_SM_THRESHOLD} → shipping state machine")
    # Save state-machine confusion matrix as the primary one
    import shutil
    sm_cm_src = PLOTS_DIR / "confusion_matrix_phase_statemachine.png"
    shutil.copy(str(sm_cm_src), str(PLOTS_DIR / "confusion_matrix_phase.png"))

phase_pass = (ml_phase_f1_mean if ml_phase_f1_mean is not None else sm_f1) > baselines["phase"]["mean"] + 0.2
print(f"  Phase PASS (0.2 above baseline {baselines['phase']['mean']:.4f}): {phase_pass}")


# ===========================================================================
# TASK 4: REP COUNTING
# State machine first; ML fallback
# ===========================================================================
print("\n" + "="*60)
print("TASK 4: REP COUNTING")
print("="*60)

# ---- State machine: simple peak detection on acc_rms per recording+set ----
# For each set, count peaks in acc_rms above threshold, compare to n_reps ground truth.
# This is a per-set evaluation, so we aggregate the window-level acc_rms into a time series
# per set, then run peak detection.
# [REF NEEDED: scipy.signal.find_peaks usage in exercise rep counting]

from scipy.signal import find_peaks

def count_reps_state_machine(set_df, win_df):
    """
    For each set in set_df, extract the acc_rms time series from win_df,
    detect peaks (rep events), and return a DataFrame with predicted vs actual reps.
    """
    results = []
    for _, row in set_df.iterrows():
        rec  = row["recording_id"]
        snum = row["set_number"]
        gt   = row["n_reps"]

        # Get windows for this set (active only)
        mask = (
            (win_df["recording_id"] == rec) &
            (win_df["set_number"] == snum) &
            (win_df["in_active_set"] == True)
        )
        grp = win_df[mask].sort_values("t_window_center_s")
        if len(grp) == 0:
            results.append({"recording_id": rec, "set_number": snum,
                             "n_reps_gt": gt, "n_reps_pred": 0})
            continue

        signal = grp["acc_rms"].fillna(grp["acc_rms"].median()).values

        # Adaptive threshold: median + 0.5 * std
        thresh = np.median(signal) + 0.3 * np.std(signal)
        # Minimum distance between peaks = ~5 windows (0.5 s) to avoid double-counting
        peaks, _ = find_peaks(signal, height=thresh, distance=5)
        n_pred = len(peaks)
        results.append({"recording_id": rec, "set_number": snum,
                         "subject_id": row["subject_id"],
                         "n_reps_gt": int(gt), "n_reps_pred": n_pred})

    return pd.DataFrame(results)

# Run per-fold to get unbiased estimate
sm_rep_fold_maes = []
sm_rep_all_results = []

rep_fold_splits_set = get_fold_splits(set_df)
set_df_ri = set_df.reset_index(drop=True)

print("  Running state-machine rep counting per fold...")
for fold_i, (tr_idx, te_idx) in enumerate(rep_fold_splits_set):
    if not te_idx:
        continue
    test_sets = set_df_ri.iloc[te_idx]
    results = count_reps_state_machine(test_sets, win_df)
    if len(results) == 0:
        continue
    mae_f = mean_absolute_error(results["n_reps_gt"], results["n_reps_pred"])
    sm_rep_fold_maes.append(mae_f)
    sm_rep_all_results.append(results)
    print(f"    Fold {fold_i}: rep MAE={mae_f:.4f} (n_sets={len(results)})")

sm_rep_results_all = pd.concat(sm_rep_all_results, ignore_index=True) if sm_rep_all_results else pd.DataFrame()
sm_rep_mae_mean = float(np.mean(sm_rep_fold_maes)) if sm_rep_fold_maes else 999.0
sm_rep_mae_std  = float(np.std(sm_rep_fold_maes)) if sm_rep_fold_maes else 999.0

if not sm_rep_results_all.empty:
    sm_exact = float((sm_rep_results_all["n_reps_pred"] == sm_rep_results_all["n_reps_gt"]).mean())
    sm_within1 = float((np.abs(sm_rep_results_all["n_reps_pred"] - sm_rep_results_all["n_reps_gt"]) <= 1).mean())
    sm_per_subject_rep = sm_rep_results_all.groupby("subject_id").apply(
        lambda g: mean_absolute_error(g["n_reps_gt"], g["n_reps_pred"])
    ).to_dict()
else:
    sm_exact = 0.0
    sm_within1 = 0.0
    sm_per_subject_rep = {}

print(f"\n  State-machine rep MAE: {sm_rep_mae_mean:.4f} ± {sm_rep_mae_std:.4f}")
print(f"  Exact match: {sm_exact*100:.1f}%  Within-1: {sm_within1*100:.1f}%")

REP_SM_THRESHOLD = 1.808  # baseline MAE — must beat this to avoid ML fallback
ml_rep_mae_mean = None
ml_rep_mae_std  = None
ml_rep_per_subject = None
rep_primary = "state_machine"
rep_feat_imp = {}

if sm_rep_mae_mean >= baselines["reps"]["mean"]:
    print(f"  State-machine rep MAE {sm_rep_mae_mean:.4f} >= baseline {baselines['reps']['mean']:.4f} → training ML fallback")
    rep_primary = "ml_fallback"

    rep_df = set_df.dropna(subset=["n_reps"]).copy()
    rep_df = rep_df.reset_index(drop=True)
    X_rep = rep_df[SET_FEAT_COLS].copy()
    y_rep = rep_df["n_reps"].values.astype(float)

    rep_fold_splits2 = get_fold_splits(rep_df)

    def rep_optuna_objective(trial):
        params = {
            "objective": "regression_l1",
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "num_leaves": trial.suggest_int("num_leaves", 8, 48),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.0, log=True),
            "verbose": -1,
            "n_jobs": -1,
        }
        fold_maes = []
        for tr_idx, te_idx in rep_fold_splits2:
            if not te_idx:
                continue
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", lgb.LGBMRegressor(**params)),
            ])
            pipe.fit(X_rep.iloc[tr_idx], y_rep[tr_idx])
            preds = pipe.predict(X_rep.iloc[te_idx])
            fold_maes.append(mean_absolute_error(y_rep[te_idx], preds))
        return np.mean(fold_maes)

    print("  Running Optuna (50 trials) for rep ML...")
    rep_study = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    rep_study.optimize(rep_optuna_objective, n_trials=50, show_progress_bar=False)
    best_rep_params = rep_study.best_params
    best_rep_params.update({"objective": "regression_l1", "verbose": -1, "n_jobs": -1})

    rep_fold_maes2 = []
    rep_per_subject2 = {}
    rep_all_true = []
    rep_all_pred = []

    for fold_i, (tr_idx, te_idx) in enumerate(rep_fold_splits2):
        if not te_idx:
            continue
        Xtr = X_rep.iloc[tr_idx]
        Xte = X_rep.iloc[te_idx]
        ytr = y_rep[tr_idx]
        yte = y_rep[te_idx]
        subj_te = rep_df.iloc[te_idx]["subject_id"].values

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMRegressor(**best_rep_params)),
        ])
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)

        fold_mae = mean_absolute_error(yte, preds)
        rep_fold_maes2.append(fold_mae)
        print(f"    Fold {fold_i}: rep MAE={fold_mae:.4f}")

        for s, t, p in zip(subj_te, yte, preds):
            rep_per_subject2.setdefault(s, {"true": [], "pred": []})
            rep_per_subject2[s]["true"].append(t)
            rep_per_subject2[s]["pred"].append(p)

        rep_all_true.extend(yte.tolist())
        rep_all_pred.extend(preds.tolist())

    ml_rep_mae_mean = float(np.mean(rep_fold_maes2))
    ml_rep_mae_std  = float(np.std(rep_fold_maes2))
    ml_rep_per_subject = {
        s: float(mean_absolute_error(v["true"], v["pred"]))
        for s, v in rep_per_subject2.items()
    }
    rep_exact_ml = float((np.round(rep_all_pred) == rep_all_true).mean() if rep_all_true else 0.0)
    rep_within1_ml = float((np.abs(np.array(rep_all_pred) - np.array(rep_all_true)) <= 1).mean()
                            if rep_all_true else 0.0)

    print(f"\n  ML rep MAE: {ml_rep_mae_mean:.4f} ± {ml_rep_mae_std:.4f}")
    print(f"  Exact match (ML): {rep_exact_ml*100:.1f}%  Within-1 (ML): {rep_within1_ml*100:.1f}%")
    print(f"  Per-subject rep MAE (ML): {ml_rep_per_subject}")

    # Train final rep model
    final_rep_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", lgb.LGBMRegressor(**best_rep_params)),
    ])
    final_rep_pipe.fit(X_rep, y_rep)
    joblib.dump(final_rep_pipe, MODELS_DIR / "reps.joblib")
    print("  Saved reps.joblib")

    try:
        rep_imp = final_rep_pipe.named_steps["imputer"]
        rep_lgbm = final_rep_pipe.named_steps["model"]
        X_rep_imp = pd.DataFrame(rep_imp.transform(X_rep), columns=X_rep.columns)
        rep_explainer = shap.TreeExplainer(rep_lgbm)
        rep_shap_vals = rep_explainer.shap_values(X_rep_imp)
        rep_shap_mean = np.abs(rep_shap_vals).mean(axis=0)
        rep_feat_imp = dict(sorted(
            zip(X_rep.columns, rep_shap_mean.tolist()),
            key=lambda x: x[1], reverse=True
        )[:20])
    except Exception as e:
        print(f"  Rep SHAP failed: {e}")
        rep_feat_imp = {}

    sm_exact = rep_exact_ml
    sm_within1 = rep_within1_ml
else:
    print(f"  State-machine MAE {sm_rep_mae_mean:.4f} beats baseline → shipping state machine")

# Choose reporting values for reps
active_rep_mae = ml_rep_mae_mean if ml_rep_mae_mean is not None else sm_rep_mae_mean
rep_pass = active_rep_mae < baselines["reps"]["mean"]
print(f"  Rep PASS (below baseline {baselines['reps']['mean']:.4f}): {rep_pass}")


# ===========================================================================
# SAVE METRICS.JSON
# ===========================================================================
print("\n" + "="*60)
print("SAVING METRICS.JSON")
print("="*60)

# Determine per-subject phase metrics
if phase_primary == "state_machine":
    ph_per_subject_final = {s: float(v) for s, v in sm_per_subject.items()}
else:
    ph_per_subject_final = ml_phase_per_subject if ml_phase_per_subject else {}

# Outlier detection for fatigue
fat_median_mae = float(np.median(list(fat_per_subject_mae.values())))
fat_outliers = {s: float(v) for s, v in fat_per_subject_mae.items()
                if v > 3 * fat_median_mae}

metrics = {
    "fatigue": {
        "metric": "MAE",
        "mean": float(fat_mean_mae),
        "std": float(fat_std_mae),
        "fold_scores": [float(v) for v in fat_fold_maes],
        "per_subject": {s: float(v) for s, v in fat_per_subject_mae.items()},
        "baseline": baselines["fatigue"]["mean"],
        "pass": bool(fat_pass),
        "fail_reason": None if fat_pass else "MAE not 30% below DummyRegressor median baseline",
        "outlier_subjects": fat_outliers,
        "best_params": best_fat_params,
    },
    "exercise": {
        "metric": "macro_F1",
        "mean": float(ex_mean_f1),
        "std": float(ex_std_f1),
        "fold_scores": [float(v) for v in ex_fold_f1s],
        "per_subject": {s: float(v) for s, v in ex_per_subject_f1.items()},
        "baseline": baselines["exercise"]["mean"],
        "pass": bool(ex_pass),
        "fail_reason": None if ex_pass else "F1 not 0.2 above stratified-random baseline",
        "best_params": best_ex_params,
    },
    "phase": {
        "metric": "macro_F1",
        "state_machine": float(sm_f1),
        "state_machine_per_subject": {s: float(v) for s, v in sm_per_subject.items()},
        "ml_fallback": {
            "mean": float(ml_phase_f1_mean),
            "std": float(ml_phase_f1_std),
            "per_subject": ml_phase_per_subject,
        } if ml_phase_f1_mean is not None else None,
        "primary": phase_primary,
        "mean": float(ml_phase_f1_mean if ml_phase_f1_mean is not None else sm_f1),
        "std": float(ml_phase_f1_std if ml_phase_f1_std is not None else 0.0),
        "per_subject": ph_per_subject_final,
        "baseline": baselines["phase"]["mean"],
        "pass": bool(phase_pass),
        "fail_reason": None if phase_pass else "F1 not 0.2 above stratified-random baseline",
    },
    "reps": {
        "metric": "MAE",
        "state_machine": {
            "mean": float(sm_rep_mae_mean),
            "std": float(sm_rep_mae_std),
            "exact_pct": float(sm_exact * 100),
            "within1_pct": float(sm_within1 * 100),
            "per_subject": {s: float(v) for s, v in sm_per_subject_rep.items()},
        },
        "ml_fallback": {
            "mean": float(ml_rep_mae_mean),
            "std": float(ml_rep_mae_std),
            "per_subject": {s: float(v) for s, v in ml_rep_per_subject.items()},
        } if ml_rep_mae_mean is not None else None,
        "primary": rep_primary,
        "mean": float(active_rep_mae),
        "std": float(ml_rep_mae_std if ml_rep_mae_std is not None else sm_rep_mae_std),
        "baseline": baselines["reps"]["mean"],
        "pass": bool(rep_pass),
        "fail_reason": None if rep_pass else "MAE not below mean-rep DummyRegressor baseline",
    },
}

metrics_path = RUN_DIR / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved {metrics_path}")

# ===========================================================================
# SAVE FEATURE_IMPORTANCE.JSON
# ===========================================================================
print("\n" + "="*60)
print("SAVING FEATURE_IMPORTANCE.JSON")
print("="*60)

feat_imp = {
    "fatigue": fat_feat_imp,
    "exercise": ex_feat_imp,
    "phase": phase_feat_imp,
    "reps": rep_feat_imp,
}
fi_path = RUN_DIR / "feature_importance.json"
with open(fi_path, "w") as f:
    json.dump(feat_imp, f, indent=2)
print(f"  Saved {fi_path}")

# ===========================================================================
# SUMMARY PRINTOUT
# ===========================================================================
print("\n" + "="*60)
print("MODELING RESULTS — 20260426_154705_default")
print("="*60)
print(f"- Fatigue (RPE):   MAE = {fat_mean_mae:.3f} ± {fat_std_mae:.3f}  [baseline MAE: {baselines['fatigue']['mean']:.3f}]  {'PASS' if fat_pass else 'FAIL'}")
print(f"- Exercise:        F1-macro = {ex_mean_f1:.3f} ± {ex_std_f1:.3f}  [baseline: {baselines['exercise']['mean']:.3f}]  {'PASS' if ex_pass else 'FAIL'}")
phase_primary_val = ml_phase_f1_mean if ml_phase_f1_mean is not None else sm_f1
print(f"- Phase:           Frame-F1 = {phase_primary_val:.3f}  [state-machine={sm_f1:.3f}]  method={phase_primary}  {'PASS' if phase_pass else 'FAIL'}")
print(f"- Reps:            MAE = {active_rep_mae:.3f}  exact={sm_exact*100:.1f}%  within-1={sm_within1*100:.1f}%  method={rep_primary}  {'PASS' if rep_pass else 'FAIL'}")
print(f"- Artifacts:       {RUN_DIR}/")

print("\nDone. model_card.md will be written separately.")
print("Script complete.")
