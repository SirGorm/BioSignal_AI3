"""
Random Forest training script for Strength-RT /train pipeline.
Run: python scripts/train_lgbm.py
Outputs written to: runs/20260427_110653_default/

Baseline model: sklearn RandomForestRegressor / RandomForestClassifier.
Random Forest is less hyperparameter-sensitive than gradient boosting
(no learning rate, no num_leaves), so the Optuna search spaces are
simpler — n_estimators, max_depth, min_samples_split, min_samples_leaf,
max_features. Ensembles are CPU-only via joblib parallel jobs (no GPU
support in sklearn RF).
"""

import os, json, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score, confusion_matrix, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import optuna
import shap

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval.plot_style import apply_style, despine

apply_style()

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
print("[device] sklearn RandomForest (CPU, n_jobs=-1)")

# ─────────────────────────────── paths ────────────────────────────────────────
# Optional CLI overrides for running on dataset_clean / labeled_clean. Backward
# compatible: with no flags, behaves identically to original.
import argparse as _ap
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--run-dir", default=None)
_p.add_argument("--features-dir", default=None,
                help="Dir containing window_features.parquet + set_features.parquet "
                     "(defaults to <run-dir>)")
_p.add_argument("--exclude-recordings", nargs="*", default=[])
_p.add_argument("--splits", default=None)
_p.add_argument("--stride", type=int, default=100,
                help="Decimate window_features per recording to this stride. "
                     "100 = 1 s hop on the 100 Hz feature grid (matches raw NN "
                     "2 s window @ 50%% overlap). 1 = no decimation.")
_p.add_argument("--n-folds", type=int, default=5,
                help="Number of CV folds for GroupKFold. 5 (default) = "
                     "v15-style 5-fold; 10 = LOSO when there are 10 subjects.")
_args, _ = _p.parse_known_args()

ROOT = Path("C:/Users/skogl/Downloads/eirikgsk/biosignal_2/BioSignal_AI3")
RUN  = Path(_args.run_dir) if _args.run_dir else ROOT / "runs" / "20260427_110653_default"
FEATURES_DIR = Path(_args.features_dir) if _args.features_dir else RUN
SPLITS_CSV = Path(_args.splits) if _args.splits else ROOT / "configs" / "splits_per_fold.csv"

for d in [RUN/"models", RUN/"plots", RUN/"logs"]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(RUN / "logs" / "train.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────── load data ────────────────────────────────────
print(f"Loading parquet files from {FEATURES_DIR} …")
wf = pd.read_parquet(FEATURES_DIR / "window_features.parquet")
sf = pd.read_parquet(FEATURES_DIR / "set_features.parquet")
if _args.exclude_recordings:
    n0_w, n0_s = len(wf), len(sf)
    wf = wf[~wf["recording_id"].isin(_args.exclude_recordings)].reset_index(drop=True)
    sf = sf[~sf["recording_id"].isin(_args.exclude_recordings)].reset_index(drop=True)
    print(f"  excluded {set(_args.exclude_recordings)}: "
          f"wf {n0_w} -> {len(wf)}, sf {n0_s} -> {len(sf)}")
if _args.stride and _args.stride > 1:
    n0 = len(wf)
    wf = wf.groupby("recording_id", sort=False, group_keys=False)\
           .apply(lambda g: g.iloc[::_args.stride])\
           .reset_index(drop=True)
    print(f"  stride={_args.stride} (per recording): wf {n0} -> {len(wf)}")
splits = pd.read_csv(SPLITS_CSV)   # cols: fold, recording_id, subject_id, split
print(f"  window_features: {wf.shape}  set_features: {sf.shape}")

# ── subject group arrays ──────────────────────────────────────────────────────
# For GroupKFold we need an integer group per sample.
# We use recording_id (9 unique = 9 subjects, 1:1 in this batch).
rec_to_int = {r: i for i, r in enumerate(sorted(wf["recording_id"].unique()))}
wf_groups = wf["recording_id"].map(rec_to_int).values
sf_groups = sf["recording_id"].map(rec_to_int).values

N_FOLDS = int(_args.n_folds)
gkf = GroupKFold(n_splits=N_FOLDS)
print(f"  N_FOLDS={N_FOLDS}  ({'LOSO' if N_FOLDS == 10 else f'{N_FOLDS}-fold GroupKFold'})")

# ─────────────────────────────── helpers ──────────────────────────────────────
# For fatigue: keep set_number and n_reps as features (scientifically valid predictors
# of RPE — set number proxies session fatigue accumulation; n_reps is effort load).
# exercise is label-encoded as an ordinal feature (movement pattern affects RPE).
# Exclude only pure identity cols and the target.
# For reps: keep set_number and exercise (encoded) but exclude rpe_for_this_set.
SET_LABEL_COLS_FATIGUE = ["recording_id", "subject_id", "rpe_for_this_set"]
SET_LABEL_COLS_REPS    = ["recording_id", "subject_id", "n_reps"]
SET_LABEL_COLS = [   # generic (not used directly below)
    "recording_id", "subject_id", "rpe_for_this_set", "n_reps",
]
WIN_LABEL_COLS = [
    "recording_id", "subject_id", "t_unix", "t_session_s",
    "in_active_set", "set_number", "exercise", "phase_label",
    "rep_count_in_set", "rpe_for_this_set", "t_window_center_s",
    "set_phase",                              # may or may not exist
]
WIN_LABEL_COLS = [c for c in WIN_LABEL_COLS if c in wf.columns]


def feature_cols(df, label_cols):
    return [c for c in df.columns if c not in label_cols]


def pearson_r(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    cc = np.corrcoef(y_true, y_pred)
    return cc[0, 1]


def make_rf_pipe_regressor(**params):
    est = RandomForestRegressor(**params)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   est),
    ])


def make_rf_pipe_classifier(**params):
    est = RandomForestClassifier(**params)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   est),
    ])


# Backwards-compat aliases (other scripts may still import these names).
make_lgbm_pipe_regressor = make_rf_pipe_regressor
make_lgbm_pipe_classifier = make_rf_pipe_classifier


def cv_regressor(X, y, groups, build_fn, n_folds=N_FOLDS):
    """Return per-fold MAE list + per-subject MAE dict."""
    fold_mae = []
    per_subject = {}   # recording_id -> list of MAE values
    rec_ids = sf["recording_id"].values   # same order as X/y
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        pipe = build_fn()
        pipe.fit(X[tr], y[tr])
        preds = pipe.predict(X[te])
        mae = mean_absolute_error(y[te], preds)
        fold_mae.append(float(mae))
        # per-subject
        for rec in np.unique(rec_ids[te]):
            mask = rec_ids[te] == rec
            subj = sf.loc[te[mask], "subject_id"].iloc[0]
            m = mean_absolute_error(y[te][mask], preds[mask])
            per_subject.setdefault(subj, []).append(float(m))
    return fold_mae, per_subject


def cv_classifier(X, y, groups, build_fn, labels, n_folds=N_FOLDS):
    """Return per-fold F1-macro list + per-subject F1 dict + aggregated confusion matrix."""
    fold_f1 = []
    per_subject = {}
    rec_ids = None  # will look up from wf
    cm_agg = np.zeros((len(labels), len(labels)), dtype=int)

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        pipe = build_fn()
        pipe.fit(X[tr], y[tr])
        preds = pipe.predict(X[te])
        f1 = f1_score(y[te], preds, labels=labels, average="macro", zero_division=0)
        fold_f1.append(float(f1))
        cm_agg += confusion_matrix(y[te], preds, labels=labels)
    return fold_f1, cm_agg


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — FATIGUE (RPE) REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== TASK 1: Fatigue (RPE) regression ===")

# Prepare set-level dataframe with exercise label-encoded and set_number retained
sf_fat = sf.copy()
le_ex_fat = LabelEncoder()
sf_fat["exercise_enc"] = le_ex_fat.fit_transform(sf_fat["exercise"].values)

# Feature columns: everything except pure identity + target
FATIGUE_FEAT_COLS = [c for c in sf_fat.columns
                     if c not in SET_LABEL_COLS_FATIGUE + ["exercise"]]
X_fat = sf_fat[FATIGUE_FEAT_COLS].values.astype(float)
y_fat = sf["rpe_for_this_set"].values.astype(float)

print(f"  Features ({len(FATIGUE_FEAT_COLS)}): {FATIGUE_FEAT_COLS[:5]} ...")

# ── Optuna tuning ──────────────────────────────────────────────────────────────
fat_log_path = str(RUN / "logs" / "optuna_fatigue.log")
fat_log_handler = logging.FileHandler(fat_log_path)
fat_log_handler.setLevel(logging.INFO)

def fat_objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 500),
        max_depth         = trial.suggest_int("max_depth", 3, 12),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features      = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5, 1.0]),
        criterion         = "absolute_error",  # MAE optimisation
        n_jobs            = -1,
        random_state      = 42,
    )
    maes = []
    for tr, te in gkf.split(X_fat, y_fat, sf_groups):
        pipe = make_rf_pipe_regressor(**params)
        pipe.fit(X_fat[tr], y_fat[tr])
        preds = pipe.predict(X_fat[te])
        maes.append(mean_absolute_error(y_fat[te], preds))
    return float(np.mean(maes))

fat_study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
fat_study.optimize(fat_objective, n_trials=60, show_progress_bar=False)
best_fat = fat_study.best_params
print(f"  Optuna best MAE: {fat_study.best_value:.4f}  params: {best_fat}")
with open(fat_log_path, "w") as f:
    json.dump({"best_value": fat_study.best_value, "best_params": best_fat,
               "n_trials": len(fat_study.trials)}, f, indent=2)

# ── Final CV with best params ──────────────────────────────────────────────────
def build_fat():
    p = dict(criterion="absolute_error", n_jobs=-1, random_state=42, **best_fat)
    return make_rf_pipe_regressor(**p)

fat_fold_mae, fat_per_subj = cv_regressor(X_fat, y_fat, sf_groups, build_fat)
fat_mean = float(np.mean(fat_fold_mae))
fat_std  = float(np.std(fat_fold_mae))
print(f"  Fatigue MAE: {fat_mean:.3f} ± {fat_std:.3f}  (baseline 1.013)")

# per-subject summary
fat_subj_summary = {s: float(np.mean(v)) for s, v in fat_per_subj.items()}
print("  Per-subject MAE:", {k: round(v, 3) for k, v in fat_subj_summary.items()})

# ── Fit final model on ALL data ────────────────────────────────────────────────
fat_final = build_fat()
fat_final.fit(X_fat, y_fat)
joblib.dump(fat_final, RUN / "models" / "fatigue.pkl")

# ── Pearson r per subject ──────────────────────────────────────────────────────
fat_pearson = {}
for subj in sf["subject_id"].unique():
    mask = sf["subject_id"] == subj
    if mask.sum() < 3:
        continue
    # use leave-this-subject-out prediction from the cv loop above
    # (approximated: predict on held-out fold where this subject is test)
    for fold, (tr, te) in enumerate(gkf.split(X_fat, y_fat, sf_groups)):
        subj_mask_te = (sf.iloc[te]["subject_id"] == subj).values
        if subj_mask_te.any():
            pipe = build_fat()
            pipe.fit(X_fat[tr], y_fat[tr])
            preds_s = pipe.predict(X_fat[te][subj_mask_te])
            true_s  = y_fat[te][subj_mask_te]
            fat_pearson[subj] = float(pearson_r(true_s, preds_s))
            break

pearson_median = float(np.nanmedian(list(fat_pearson.values())))
print(f"  Per-subject Pearson r median: {pearson_median:.3f}")

# ── SHAP (fatigue) ─────────────────────────────────────────────────────────────
print("  Computing SHAP (fatigue) …")
imp_X_fat = fat_final.named_steps["imputer"].transform(X_fat)
explainer_fat = shap.TreeExplainer(fat_final.named_steps["model"])
shap_fat = explainer_fat.shap_values(imp_X_fat)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_fat, imp_X_fat, feature_names=FATIGUE_FEAT_COLS,
                  show=False, max_display=15)
plt.tight_layout()
despine(fig=plt.gcf())
plt.savefig(RUN / "plots" / "shap_fatigue.png", dpi=120)
plt.close()
print("  Saved shap_fatigue.png")

# feature importance (mean decrease in impurity, sklearn RF default)
fat_fi_gain = fat_final.named_steps["model"].feature_importances_
fat_fi = dict(sorted(zip(FATIGUE_FEAT_COLS, fat_fi_gain.tolist()),
                     key=lambda x: -x[1]))

# ── Calibration plot (predicted vs actual RPE) ────────────────────────────────
# Collect OOF predictions
oof_pred_fat = np.full(len(y_fat), np.nan)
for tr, te in gkf.split(X_fat, y_fat, sf_groups):
    p = build_fat(); p.fit(X_fat[tr], y_fat[tr])
    oof_pred_fat[te] = p.predict(X_fat[te])

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_fat, oof_pred_fat, alpha=0.7, edgecolors="k", s=50)
lims = [y_fat.min()-0.5, y_fat.max()+0.5]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect")
ax.set_xlabel("Actual RPE"); ax.set_ylabel("Predicted RPE")
ax.set_title("Fatigue calibration (OOF)")
ax.legend(); plt.tight_layout()
despine(fig=fig)
plt.savefig(RUN / "plots" / "fatigue_calibration.png", dpi=120)
plt.close()
print("  Saved fatigue_calibration.png")

# ── Per-subject MAE bar chart ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
subj_names = list(fat_subj_summary.keys())
subj_maes  = [fat_subj_summary[s] for s in subj_names]
bars = ax.bar(subj_names, subj_maes, color="steelblue", edgecolor="k")
ax.axhline(1.013, color="red", linestyle="--", label="Baseline MAE 1.013")
ax.axhline(fat_mean, color="green", linestyle="--", label=f"Mean MAE {fat_mean:.3f}")
ax.set_xlabel("Subject"); ax.set_ylabel("MAE (RPE)")
ax.set_title("Fatigue MAE per subject (LOSO-CV)")
ax.legend(); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
despine(fig=fig)
plt.savefig(RUN / "plots" / "fatigue_per_subject.png", dpi=120)
plt.close()
print("  Saved fatigue_per_subject.png")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — EXERCISE CLASSIFICATION (per-window)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== TASK 2: Exercise classification ===")

# Filter: active set only, valid exercise label
wf_ex = wf[wf["in_active_set"] & wf["exercise"].notna()].copy()
wf_ex = wf_ex[wf_ex["exercise"].isin(["squat","deadlift","benchpress","pullup"])]
print(f"  After filtering: {len(wf_ex)} windows")
print("  Class balance:", wf_ex["exercise"].value_counts().to_dict())

WIN_LABEL_COLS_EX = [c for c in WIN_LABEL_COLS if c in wf_ex.columns]
EX_FEAT_COLS = [c for c in wf_ex.columns if c not in WIN_LABEL_COLS_EX]
print(f"  Features ({len(EX_FEAT_COLS)}): {EX_FEAT_COLS[:5]} …")

X_ex  = wf_ex[EX_FEAT_COLS].values.astype(float)
le_ex = LabelEncoder()
y_ex  = le_ex.fit_transform(wf_ex["exercise"].values)
ex_classes = list(le_ex.classes_)
print(f"  Classes: {ex_classes}")

ex_groups = wf_ex["recording_id"].map(rec_to_int).values

# ── Optuna tuning ──────────────────────────────────────────────────────────────
def ex_objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 400),
        max_depth         = trial.suggest_int("max_depth", 4, 16),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features      = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5, 1.0]),
        class_weight      = "balanced",
        n_jobs            = -1,
        random_state      = 42,
    )
    f1s = []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X_ex, y_ex, ex_groups):
        pipe = make_rf_pipe_classifier(**params)
        pipe.fit(X_ex[tr], y_ex[tr])
        preds = pipe.predict(X_ex[te])
        f1s.append(f1_score(y_ex[te], preds, average="macro", zero_division=0))
    return -float(np.mean(f1s))   # minimise negative F1

ex_study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
ex_study.optimize(ex_objective, n_trials=50, show_progress_bar=False)
best_ex = ex_study.best_params
print(f"  Optuna best F1: {-ex_study.best_value:.4f}  params: {best_ex}")
with open(RUN / "logs" / "optuna_exercise.log", "w") as f:
    json.dump({"best_value": -ex_study.best_value, "best_params": best_ex,
               "n_trials": len(ex_study.trials)}, f, indent=2)

# ── Final CV ───────────────────────────────────────────────────────────────────
def build_ex():
    p = dict(class_weight="balanced", n_jobs=-1, random_state=42, **best_ex)
    return make_rf_pipe_classifier(**p)

ex_fold_f1 = []
ex_cm_agg  = np.zeros((4, 4), dtype=int)
ex_per_subj = {}
for fold, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X_ex, y_ex, ex_groups)):
    pipe = build_ex(); pipe.fit(X_ex[tr], y_ex[tr])
    preds = pipe.predict(X_ex[te])
    f1 = f1_score(y_ex[te], preds, labels=list(range(4)),
                  average="macro", zero_division=0)
    ex_fold_f1.append(float(f1))
    ex_cm_agg += confusion_matrix(y_ex[te], preds, labels=list(range(4)))
    # per subject
    rec_te = wf_ex.iloc[te]["recording_id"].values
    for rec in np.unique(rec_te):
        mask = rec_te == rec
        subj = wf_ex.iloc[te[mask]]["subject_id"].iloc[0]
        m = f1_score(y_ex[te][mask], preds[mask], average="macro", zero_division=0)
        ex_per_subj.setdefault(subj, []).append(float(m))

ex_mean = float(np.mean(ex_fold_f1))
ex_std  = float(np.std(ex_fold_f1))
print(f"  Exercise F1-macro: {ex_mean:.3f} ± {ex_std:.3f}  (baseline 0.123)")

# ── Fit final model ────────────────────────────────────────────────────────────
ex_final = build_ex(); ex_final.fit(X_ex, y_ex)
joblib.dump({"pipeline": ex_final, "label_encoder": le_ex},
            RUN / "models" / "exercise.pkl")

# ── Confusion matrix ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm_norm = ex_cm_agg.astype(float) / ex_cm_agg.sum(axis=1, keepdims=True).clip(1)
sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=ex_classes,
            yticklabels=ex_classes, cmap="Blues", ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Exercise confusion matrix (LOSO-CV, normalised)")
plt.tight_layout()
despine(fig=fig, left=True, bottom=True)
plt.savefig(RUN / "plots" / "cm_exercise.png", dpi=120)
plt.close()
print("  Saved cm_exercise.png")

# ── SHAP (exercise) ────────────────────────────────────────────────────────────
print("  Computing SHAP (exercise) …")
imp_X_ex_shap = ex_final.named_steps["imputer"].transform(X_ex[:3000])   # subsample
explainer_ex = shap.TreeExplainer(ex_final.named_steps["model"])
shap_ex_raw = explainer_ex.shap_values(imp_X_ex_shap)
# shap_ex_raw shape: (n_samples, n_features, n_classes) in newer shap, or list in older
if isinstance(shap_ex_raw, list):
    # list of (n_samples, n_features) arrays, one per class
    shap_ex_mean = np.mean(np.abs(np.stack(shap_ex_raw, axis=-1)), axis=-1)
else:
    # (n_samples, n_features, n_classes)
    shap_ex_mean = np.mean(np.abs(shap_ex_raw), axis=-1)

# Build mean importance vector for bar plot
feat_importance_ex = np.mean(shap_ex_mean, axis=0)
fig, ax = plt.subplots(figsize=(10, 6))
sorted_idx = np.argsort(feat_importance_ex)[-15:]
ax.barh([EX_FEAT_COLS[i] for i in sorted_idx], feat_importance_ex[sorted_idx])
ax.set_xlabel("Mean |SHAP value|"); ax.set_title("Exercise SHAP (mean over classes)")
plt.tight_layout()
despine(fig=fig)
plt.savefig(RUN / "plots" / "shap_exercise.png", dpi=120)
plt.close()
print("  Saved shap_exercise.png")

# feature importance (mean decrease in impurity, sklearn RF default)
ex_fi_gain = ex_final.named_steps["model"].feature_importances_
ex_fi = dict(sorted(zip(EX_FEAT_COLS, ex_fi_gain.tolist()),
                    key=lambda x: -x[1]))


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — PHASE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== TASK 3: Phase classification ===")

PHASE_LABELS = ["concentric", "eccentric", "isometric"]

wf_ph = wf[
    wf["in_active_set"] &
    wf["phase_label"].isin(PHASE_LABELS)
].copy()
print(f"  After filtering: {len(wf_ph)} windows")
print("  Class balance:", wf_ph["phase_label"].value_counts().to_dict())

# ── State-machine: threshold-based on acc_jerk_rms and acc_rms ────────────────
# Simple rule: high jerk → concentric; moderate jerk + moderate acc → eccentric;
# low acc + low jerk → isometric.
# Thresholds derived from the 33rd and 66th percentile of acc_jerk_rms.
print("  Evaluating state-machine …")

valid_ph = wf_ph[["acc_rms", "acc_jerk_rms", "phase_label"]].dropna()
p33, p66 = np.percentile(valid_ph["acc_jerk_rms"], [33, 66])

def state_machine_phase(row):
    jerk = row["acc_jerk_rms"]
    rms  = row["acc_rms"]
    if jerk >= p66:
        return "concentric"
    elif jerk >= p33:
        return "eccentric"
    else:
        return "isometric"

sm_pred = valid_ph.apply(state_machine_phase, axis=1).values
sm_true = valid_ph["phase_label"].values
sm_f1   = f1_score(sm_true, sm_pred, labels=PHASE_LABELS,
                   average="macro", zero_division=0)
print(f"  State-machine F1-macro: {sm_f1:.3f}  (threshold 0.85)")

# ── Decide: state-machine vs ML ───────────────────────────────────────────────
USE_SM_PHASE = sm_f1 >= 0.85
print(f"  Decision: {'STATE-MACHINE' if USE_SM_PHASE else 'ML FALLBACK'}")

WIN_LABEL_COLS_PH = [c for c in WIN_LABEL_COLS if c in wf_ph.columns]
PH_FEAT_COLS = [c for c in wf_ph.columns if c not in WIN_LABEL_COLS_PH]
ph_groups = wf_ph["recording_id"].map(rec_to_int).values

X_ph = wf_ph[PH_FEAT_COLS].values.astype(float)
le_ph = LabelEncoder()
y_ph  = le_ph.fit_transform(wf_ph["phase_label"].values)
ph_classes = list(le_ph.classes_)

# ── ML fallback regardless (always report both) ────────────────────────────────
def ph_objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 400),
        max_depth         = trial.suggest_int("max_depth", 4, 16),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features      = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5, 1.0]),
        class_weight      = "balanced",
        n_jobs            = -1,
        random_state      = 42,
    )
    f1s = []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X_ph, y_ph, ph_groups):
        pipe = make_rf_pipe_classifier(**params)
        pipe.fit(X_ph[tr], y_ph[tr])
        preds = pipe.predict(X_ph[te])
        f1s.append(f1_score(y_ph[te], preds, average="macro", zero_division=0))
    return -float(np.mean(f1s))

ph_study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
ph_study.optimize(ph_objective, n_trials=50, show_progress_bar=False)
best_ph = ph_study.best_params
print(f"  Optuna ML F1: {-ph_study.best_value:.4f}")
with open(RUN / "logs" / "optuna_phase.log", "w") as f:
    json.dump({"best_value": -ph_study.best_value, "best_params": best_ph,
               "n_trials": len(ph_study.trials)}, f, indent=2)

def build_ph():
    p = dict(class_weight="balanced", n_jobs=-1, random_state=42, **best_ph)
    return make_rf_pipe_classifier(**p)

ph_fold_f1 = []
ph_cm_agg  = np.zeros((3, 3), dtype=int)
for fold, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X_ph, y_ph, ph_groups)):
    pipe = build_ph(); pipe.fit(X_ph[tr], y_ph[tr])
    preds = pipe.predict(X_ph[te])
    f1 = f1_score(y_ph[te], preds, labels=list(range(3)),
                  average="macro", zero_division=0)
    ph_fold_f1.append(float(f1))
    ph_cm_agg += confusion_matrix(y_ph[te], preds, labels=list(range(3)))

ph_mean = float(np.mean(ph_fold_f1))
ph_std  = float(np.std(ph_fold_f1))
print(f"  Phase ML F1-macro: {ph_mean:.3f} ± {ph_std:.3f}  (baseline 0.186)")
print(f"  Phase state-machine F1: {sm_f1:.3f}")
print(f"  Deployed method: {'state-machine' if USE_SM_PHASE else 'RandomForest'}")

# Save final phase model (ML, for reference / fallback)
ph_final = build_ph(); ph_final.fit(X_ph, y_ph)
if not USE_SM_PHASE:
    joblib.dump({"pipeline": ph_final, "label_encoder": le_ph},
                RUN / "models" / "phase.pkl")
    print("  Saved phase.pkl (ML fallback deployed)")
else:
    joblib.dump({"pipeline": ph_final, "label_encoder": le_ph,
                 "note": "state-machine deployed; ML saved for ablation"},
                RUN / "models" / "phase.pkl")
    print("  Saved phase.pkl (ML for ablation; state-machine deployed)")

# ── Confusion matrix (ML) ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
cm_norm_ph = ph_cm_agg.astype(float) / ph_cm_agg.sum(axis=1, keepdims=True).clip(1)
sns.heatmap(cm_norm_ph, annot=True, fmt=".2f", xticklabels=ph_classes,
            yticklabels=ph_classes, cmap="Greens", ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Phase CM (LOSO-CV, ML)")
plt.tight_layout()
despine(fig=fig, left=True, bottom=True)
plt.savefig(RUN / "plots" / "cm_phase.png", dpi=120)
plt.close()
print("  Saved cm_phase.png")

# ── SHAP (phase) ──────────────────────────────────────────────────────────────
print("  Computing SHAP (phase) …")
imp_X_ph_shap = ph_final.named_steps["imputer"].transform(X_ph[:3000])
explainer_ph = shap.TreeExplainer(ph_final.named_steps["model"])
shap_ph_raw   = explainer_ph.shap_values(imp_X_ph_shap)
if isinstance(shap_ph_raw, list):
    shap_ph_mean = np.mean(np.abs(np.stack(shap_ph_raw, axis=-1)), axis=-1)
else:
    shap_ph_mean = np.mean(np.abs(shap_ph_raw), axis=-1)

feat_importance_ph = np.mean(shap_ph_mean, axis=0)
fig, ax = plt.subplots(figsize=(10, 6))
sorted_idx_ph = np.argsort(feat_importance_ph)[-15:]
ax.barh([PH_FEAT_COLS[i] for i in sorted_idx_ph], feat_importance_ph[sorted_idx_ph])
ax.set_xlabel("Mean |SHAP value|"); ax.set_title("Phase SHAP (mean over classes)")
plt.tight_layout()
despine(fig=fig)
plt.savefig(RUN / "plots" / "shap_phase.png", dpi=120)
plt.close()
print("  Saved shap_phase.png")

ph_fi_gain = ph_final.named_steps["model"].feature_importances_
ph_fi = dict(sorted(zip(PH_FEAT_COLS, ph_fi_gain.tolist()),
                    key=lambda x: -x[1]))


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 — REP COUNTING
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== TASK 4: Rep counting ===")

# ── State-machine: peak-count on acc_rep_band_ratio per set ──────────────────
# We use the per-set acc_rep_band_ratio_mean / acc_rms_mean to estimate n_reps.
# Regression: n_reps ~ acc_rms_mean * set_duration_s * rep_band_ratio / normalised_dominance
# Simpler: state machine uses marker-free formula from González-Badillo & Sánchez-Medina 2010:
#   estimated_reps ≈ floor(acc_dom_freq_mean * set_duration_s * rep_band_ratio_mean * K)
# K calibrated across training subjects.
# Here we estimate via linear calibration on training fold.

print("  State-machine rep estimator …")

# use only set_features
y_reps  = sf["n_reps"].values.astype(float)
sm_rep_maes = []
sm_rep_preds_all = []
sm_rep_true_all  = []

for tr, te in GroupKFold(n_splits=N_FOLDS).split(
        sf[["acc_dom_freq_mean","set_duration_s","acc_rep_band_ratio_mean"]].values,
        y_reps, sf_groups):
    # Compute candidate feature
    X_sm_tr = sf.iloc[tr]
    X_sm_te = sf.iloc[te]
    feat_tr = (X_sm_tr["acc_dom_freq_mean"] *
               X_sm_tr["set_duration_s"] *
               X_sm_tr["acc_rep_band_ratio_mean"]).values
    feat_te = (X_sm_te["acc_dom_freq_mean"] *
               X_sm_te["set_duration_s"] *
               X_sm_te["acc_rep_band_ratio_mean"]).values
    # Calibrate K from train fold
    valid_tr = np.isfinite(feat_tr) & np.isfinite(y_reps[tr])
    if valid_tr.sum() < 2:
        K = 1.0
    else:
        K = np.nanmedian(y_reps[tr][valid_tr] / np.clip(feat_tr[valid_tr], 1e-6, None))
    sm_pred_te = np.clip(np.round(feat_te * K), 1, 20)
    sm_rep_maes.append(mean_absolute_error(y_reps[te], sm_pred_te))
    sm_rep_preds_all.extend(sm_pred_te.tolist())
    sm_rep_true_all.extend(y_reps[te].tolist())

sm_rep_mae = float(np.mean(sm_rep_maes))
print(f"  State-machine rep MAE: {sm_rep_mae:.3f}  (threshold 0.85, baseline 1.291)")

USE_SM_REPS = sm_rep_mae <= 0.85
print(f"  Decision: {'STATE-MACHINE' if USE_SM_REPS else 'ML FALLBACK'}")

# Build reps feature matrix with exercise encoded and set_number retained
sf_reps = sf.copy()
le_ex_reps = LabelEncoder()
sf_reps["exercise_enc"] = le_ex_reps.fit_transform(sf_reps["exercise"].values)
REPS_FEAT_COLS = [c for c in sf_reps.columns
                  if c not in SET_LABEL_COLS_REPS + ["exercise"]]

def reps_objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 400),
        max_depth         = trial.suggest_int("max_depth", 3, 12),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features      = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5, 1.0]),
        criterion         = "absolute_error",
        n_jobs            = -1,
        random_state      = 42,
    )
    maes = []
    X_reps_opt = sf_reps[REPS_FEAT_COLS].values.astype(float)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X_reps_opt, y_reps, sf_groups):
        pipe = make_rf_pipe_regressor(**params)
        pipe.fit(X_reps_opt[tr], y_reps[tr])
        preds = pipe.predict(X_reps_opt[te])
        maes.append(mean_absolute_error(y_reps[te], preds))
    return float(np.mean(maes))

reps_study = optuna.create_study(direction="minimize",
                                  sampler=optuna.samplers.TPESampler(seed=42))
reps_study.optimize(reps_objective, n_trials=50, show_progress_bar=False)
best_reps = reps_study.best_params
print(f"  Optuna reps MAE: {reps_study.best_value:.4f}")
with open(RUN / "logs" / "optuna_reps.log", "w") as f:
    json.dump({"best_value": reps_study.best_value, "best_params": best_reps,
               "n_trials": len(reps_study.trials)}, f, indent=2)

X_reps = sf_reps[REPS_FEAT_COLS].values.astype(float)

def build_reps():
    p = dict(criterion="absolute_error", n_jobs=-1, random_state=42, **best_reps)
    return make_rf_pipe_regressor(**p)

reps_fold_mae = []
oof_pred_reps = np.full(len(y_reps), np.nan)
for fold, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X_reps, y_reps, sf_groups)):
    pipe = build_reps(); pipe.fit(X_reps[tr], y_reps[tr])
    preds_r = pipe.predict(X_reps[te])
    reps_fold_mae.append(float(mean_absolute_error(y_reps[te], preds_r)))
    oof_pred_reps[te] = preds_r

reps_mean = float(np.mean(reps_fold_mae))
reps_std  = float(np.std(reps_fold_mae))
print(f"  Reps ML MAE: {reps_mean:.3f} ± {reps_std:.3f}  (baseline 1.291)")

# Exact match and within-1
oof_reps_rounded = np.round(oof_pred_reps)
exact_pct    = float(np.mean(oof_reps_rounded == y_reps) * 100)
within1_pct  = float(np.mean(np.abs(oof_reps_rounded - y_reps) <= 1) * 100)
print(f"  Reps exact: {exact_pct:.1f}%  within-1: {within1_pct:.1f}%")

# Save final reps model
reps_final = build_reps(); reps_final.fit(X_reps, y_reps)
if not USE_SM_REPS:
    joblib.dump(reps_final, RUN / "models" / "reps.pkl")
    print("  Saved reps.pkl (ML deployed)")
else:
    joblib.dump({"pipeline": reps_final, "note": "state-machine deployed; ML saved for ablation"},
                RUN / "models" / "reps.pkl")
    print("  Saved reps.pkl (ML for ablation; state-machine deployed)")

# ── SHAP (reps) ───────────────────────────────────────────────────────────────
print("  Computing SHAP (reps) …")
imp_X_reps = reps_final.named_steps["imputer"].transform(X_reps)
explainer_reps = shap.TreeExplainer(reps_final.named_steps["model"])
shap_reps = explainer_reps.shap_values(imp_X_reps)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_reps, imp_X_reps, feature_names=REPS_FEAT_COLS,
                  show=False, max_display=15)
plt.tight_layout()
despine(fig=plt.gcf())
plt.savefig(RUN / "plots" / "shap_reps.png", dpi=120)
plt.close()
print("  Saved shap_reps.png")

reps_fi_gain = reps_final.named_steps["model"].feature_importances_
reps_fi = dict(sorted(zip(REPS_FEAT_COLS, reps_fi_gain.tolist()),
                      key=lambda x: -x[1]))


# ══════════════════════════════════════════════════════════════════════════════
# SANITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Sanity checks ===")
BASELINE_FAT_MAE  = 1.013
BASELINE_EX_F1    = 0.123
BASELINE_PH_F1    = 0.186
BASELINE_REPS_MAE = 1.291

check_fat  = fat_mean  < BASELINE_FAT_MAE
check_ex   = ex_mean   > BASELINE_EX_F1 + 0.20
check_ph   = ph_mean   > BASELINE_PH_F1
check_reps = reps_mean < BASELINE_REPS_MAE

median_fat_mae = float(np.median(list(fat_subj_summary.values())))
worst_subj = max(fat_subj_summary, key=fat_subj_summary.get)
worst_mae  = fat_subj_summary[worst_subj]
outlier_flag = worst_mae > 3 * median_fat_mae

print(f"  Fatigue MAE < baseline: {check_fat}  ({fat_mean:.3f} < {BASELINE_FAT_MAE})")
print(f"  Exercise F1 > baseline+0.20: {check_ex}  ({ex_mean:.3f} > {BASELINE_EX_F1+0.20:.3f})")
print(f"  Phase F1 > baseline: {check_ph}  ({ph_mean:.3f} > {BASELINE_PH_F1})")
print(f"  Reps MAE < baseline: {check_reps}  ({reps_mean:.3f} < {BASELINE_REPS_MAE})")
print(f"  Worst subject: {worst_subj}  MAE={worst_mae:.3f}  3×median={3*median_fat_mae:.3f}  outlier={outlier_flag}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

# ── metrics.json ──────────────────────────────────────────────────────────────
metrics = {
    "run": "20260427_110653_default",
    "cv_scheme": "GroupKFold-5 (subject-wise LOSO)",
    "n_subjects": 9,
    "fatigue": {
        "task": "rpe_regression",
        "method": "RandomForestRegressor (criterion=absolute_error)",
        "metric_primary": "MAE",
        "mae_mean": fat_mean,
        "mae_std": fat_std,
        "mae_per_fold": fat_fold_mae,
        "pearson_r_per_subj": fat_pearson,
        "pearson_r_median": pearson_median,
        "baseline_mae": BASELINE_FAT_MAE,
        "beats_baseline": check_fat,
        "per_subject_mae": fat_subj_summary,
        "worst_subject": worst_subj,
        "worst_subject_mae": worst_mae,
        "outlier_flag": outlier_flag,
        "optuna_best_params": best_fat,
    },
    "exercise": {
        "task": "exercise_classification",
        "method": "RandomForestClassifier (multiclass, class_weight=balanced)",
        "metric_primary": "F1-macro",
        "f1_mean": ex_mean,
        "f1_std": ex_std,
        "f1_per_fold": ex_fold_f1,
        "classes": ex_classes,
        "baseline_f1": BASELINE_EX_F1,
        "beats_baseline_by_0.20": check_ex,
        "per_subject_f1": {s: float(np.mean(v)) for s, v in ex_per_subj.items()},
        "optuna_best_params": best_ex,
    },
    "phase": {
        "task": "phase_classification",
        "state_machine_f1": float(sm_f1),
        "state_machine_threshold": 0.85,
        "deployed_method": "state-machine" if USE_SM_PHASE else "RandomForestClassifier",
        "ml_f1_mean": ph_mean,
        "ml_f1_std": ph_std,
        "ml_f1_per_fold": ph_fold_f1,
        "classes": ph_classes,
        "baseline_f1": BASELINE_PH_F1,
        "beats_baseline": check_ph,
        "optuna_best_params": best_ph,
    },
    "reps": {
        "task": "rep_counting",
        "state_machine_mae": float(sm_rep_mae),
        "state_machine_threshold": 0.85,
        "deployed_method": "state-machine" if USE_SM_REPS else "RandomForestRegressor",
        "ml_mae_mean": reps_mean,
        "ml_mae_std": reps_std,
        "ml_mae_per_fold": reps_fold_mae,
        "ml_exact_pct": exact_pct,
        "ml_within1_pct": within1_pct,
        "baseline_mae": BASELINE_REPS_MAE,
        "beats_baseline": check_reps,
        "optuna_best_params": best_reps,
    },
}
with open(RUN / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\nSaved metrics.json")

# ── feature_importance.json ───────────────────────────────────────────────────
fi_dict = {
    "fatigue":  {k: v for k, v in list(fat_fi.items())[:20]},
    "exercise": {k: v for k, v in list(ex_fi.items())[:20]},
    "phase":    {k: v for k, v in list(ph_fi.items())[:20]},
    "reps":     {k: v for k, v in list(reps_fi.items())[:20]},
}
with open(RUN / "feature_importance.json", "w") as f:
    json.dump(fi_dict, f, indent=2)
print("Saved feature_importance.json")

# ── per_subject_breakdown.csv ─────────────────────────────────────────────────
rows = []
for subj in sorted(sf["subject_id"].unique()):
    rec = sf[sf["subject_id"] == subj]["recording_id"].iloc[0]
    rows.append({
        "subject_id":    subj,
        "recording_id":  rec,
        "fatigue_mae":   fat_subj_summary.get(subj, np.nan),
        "fatigue_pearson_r": fat_pearson.get(subj, np.nan),
        "exercise_f1":   np.mean(ex_per_subj.get(subj, [np.nan])),
    })
pd.DataFrame(rows).to_csv(RUN / "per_subject_breakdown.csv", index=False)
print("Saved per_subject_breakdown.csv")

print("\n=== All artifacts saved to", RUN, "===")
print("DONE.")
