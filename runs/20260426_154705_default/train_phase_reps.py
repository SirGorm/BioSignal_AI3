"""
Phase segmentation (ML fallback) + rep counting (state machine) training.
Tasks 1 and 2 results are taken from v1 output (already saved).
Run from repo root.

References:
  - Ke et al. (2017) LightGBM. NeurIPS.
  - Akiba et al. (2019) Optuna. KDD.
  - Lundberg & Lee (2017) SHAP. NeurIPS.
  - Pernek et al. (2015) Exercise rep detection with IMU. Personal Ubiquitous Comput 19:1101-1111.
  - Farina et al. (2004) EMG variables estimation. J Electromyogr Kinesiol 14:337-352.
  - Xu et al. (2021) Real-time fatigue monitoring via sEMG. Sensors 21:5654.
  - Hastie, Tibshirani & Friedman (2009) Elements of Statistical Learning. Springer.
  - Grandini et al. (2020) Metrics for multi-class classification. arXiv:2008.05756.
"""

import json
import os
import shutil
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

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

FOLDS = sorted(splits["fold"].unique())
log(f"  folds: {FOLDS}")

SET_META = ["recording_id", "subject_id", "set_number", "exercise",
            "rpe_for_this_set", "n_reps", "set_duration_s"]
SET_FEAT_COLS = [c for c in set_df.columns if c not in SET_META]

WIN_META = ["subject_id", "recording_id", "t_unix", "t_session_s",
            "in_active_set", "set_number", "exercise", "phase_label",
            "rep_count_in_set", "rpe_for_this_set", "t_window_center_s"]
WIN_FEAT_COLS = [c for c in win_df.columns if c not in WIN_META]

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

# ---------------------------------------------------------------------------
# Known results from v1 (tasks 1 and 2 already saved)
# ---------------------------------------------------------------------------
FAT_MEAN = 0.9650
FAT_STD  = 0.0969
FAT_FOLDS = [0.8968, 0.9815, 0.8412, 0.9777, 1.1278]
FAT_PS_MAE = {
    "Gorm": 0.8968, "Vivian": 0.7822, "michael": 1.3031, "Juile": 0.8593,
    "Elias": 0.6904, "sivert": 1.1372, "kiyomi": 0.6962, "lucas 2": 0.8676,
    "Tias": 1.0879, "lucas": 0.9702, "Raghild": 1.2855,
}
FAT_FI = {
    "acc_rms_mean": None, "acc_jerk_rms_mean": None, "ecg_hr_rel_mean": None,
    "emg_mdf_endset": None, "eda_scr_amp_mean": None,
}  # placeholder; will reload from saved model

EX_MEAN = 0.4270
EX_STD  = 0.0508
EX_FOLDS = [0.4203, 0.3615, 0.3863, 0.4677, 0.4994]
EX_PS_F1 = {
    "Gorm": 0.4203, "Vivian": 0.3279, "michael": 0.4039, "Juile": 0.3189,
    "Elias": 0.4544, "sivert": 0.4110, "kiyomi": 0.2538, "lucas 2": 0.4421,
    "Tias": 0.4805, "lucas": 0.4439, "Raghild": 0.5516,
}

fat_pass = FAT_MEAN < baselines["fatigue"]["mean"] * 0.70  # False but flag as such
ex_pass  = EX_MEAN > baselines["exercise"]["mean"] + 0.2   # True

log(f"Task 1 (fatigue): MAE={FAT_MEAN:.4f} baseline={baselines['fatigue']['mean']:.4f} PASS={fat_pass}")
log(f"Task 2 (exercise): F1={EX_MEAN:.4f} baseline={baselines['exercise']['mean']:.4f} PASS={ex_pass}")

# Reload SHAP feature importance from saved model
log("Reloading fatigue model for SHAP...")
try:
    final_fat = joblib.load(MODELS_DIR / "fatigue.joblib")
    fat_df_loc = set_df.dropna(subset=["rpe_for_this_set"]).reset_index(drop=True)
    X_fat = fat_df_loc[SET_FEAT_COLS].copy()
    X_fat_imp = pd.DataFrame(final_fat["imp"].transform(X_fat), columns=X_fat.columns)
    fat_exp = shap.TreeExplainer(final_fat["m"])
    fat_sv  = fat_exp.shap_values(X_fat_imp)
    fat_shap_mean = np.abs(fat_sv).mean(axis=0)
    fat_fi = dict(sorted(zip(X_fat.columns, fat_shap_mean.tolist()),
                          key=lambda x: x[1], reverse=True)[:20])
    log(f"  Top fatigue features: {list(fat_fi.keys())[:5]}")
    # Save SHAP plot if not already there
    if not (PLOTS_DIR / "shap_summary_fatigue.png").exists():
        plt.figure(figsize=(10, 8))
        shap.summary_plot(fat_sv, X_fat_imp, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_summary_fatigue.png", dpi=150, bbox_inches="tight")
        plt.close()
except Exception as e:
    log(f"  Fatigue SHAP reload failed: {e}")
    fat_fi = {k: 0.0 for k in list(FAT_FI.keys())}

log("Reloading exercise model for SHAP...")
try:
    final_ex = joblib.load(MODELS_DIR / "exercise.joblib")
    EXERCISE_CLASSES = ["squat", "deadlift", "benchpress", "pullup"]
    ex_df_loc = win_df[
        (win_df["in_active_set"] == True) &
        (win_df["exercise"].isin(EXERCISE_CLASSES))
    ].reset_index(drop=True)
    X_ex_full = ex_df_loc[WIN_FEAT_COLS].copy()
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
    log(f"  Top exercise features: {list(ex_fi.keys())[:5]}")
    if not (PLOTS_DIR / "shap_summary_exercise.png").exists():
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv_plot, X_ex_s, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_summary_exercise.png", dpi=150, bbox_inches="tight")
        plt.close()
except Exception as e:
    log(f"  Exercise SHAP reload failed: {e}")
    ex_fi = {}

# ===========================================================================
# TASK 3: PHASE SEGMENTATION
# State machine F1 from v1: 0.2870 (< 0.85 threshold) -- ML fallback required.
# State machine approach: threshold on acc_rms and acc_jerk_rms per recording
# (Pernek et al. 2015). Since SM F1 < threshold, train LightGBM.
# ===========================================================================
log("\n" + "="*60)
log("TASK 3: PHASE SEGMENTATION (ML fallback -- SM F1=0.287)")
log("="*60)

PHASE_CLASSES = ["concentric", "eccentric", "isometric"]
SM_F1 = 0.2870  # measured in v1

ph_df = win_df[
    (win_df["in_active_set"] == True) &
    (win_df["phase_label"].isin(PHASE_CLASSES))
].reset_index(drop=True)
log(f"  Phase windows: {len(ph_df)}")

# State machine per-subject F1 (recompute)
def state_machine_phase_vec(df):
    """Vectorised per-recording threshold state machine for phase (Pernek et al. 2015)."""
    all_preds = pd.Series(index=df.index, dtype=object)
    for rec_id, grp in df.groupby("recording_id"):
        jerk_hi = grp["acc_jerk_rms"].quantile(0.75)
        rms_med = grp["acc_rms"].median()
        j = grp["acc_jerk_rms"].values
        r = grp["acc_rms"].values
        pred = np.where(
            (j >= jerk_hi) & (r >= rms_med), "concentric",
            np.where((j < jerk_hi) & (r >= rms_med), "eccentric", "isometric")
        )
        all_preds[grp.index] = pred
    return all_preds.values

sm_preds = state_machine_phase_vec(ph_df)
sm_f1_check = float(f1_score(ph_df["phase_label"].values, sm_preds,
                               labels=PHASE_CLASSES, average="macro", zero_division=0))
log(f"  State-machine F1 (recomputed): {sm_f1_check:.4f}")

sm_ps_f1 = {}
for subj, grp in ph_df.groupby("subject_id"):
    sp = sm_preds[grp.index.values]
    sm_ps_f1[subj] = float(f1_score(grp["phase_label"].values, sp,
                                     labels=PHASE_CLASSES, average="macro", zero_division=0))

# State machine confusion
cm_sm = confusion_matrix(ph_df["phase_label"].values, sm_preds, labels=PHASE_CLASSES)
disp_sm = ConfusionMatrixDisplay(cm_sm, display_labels=PHASE_CLASSES)
fig, ax = plt.subplots(figsize=(6, 5))
disp_sm.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Phase state-machine  F1={sm_f1_check:.3f}")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrix_phase_statemachine.png", dpi=150)
plt.close()

# ML fallback
X_ph = ph_df[WIN_FEAT_COLS].copy()
y_ph = ph_df["phase_label"].values
ph_fold_splits = get_fold_splits(ph_df)

# Subsample for Optuna speed
MAX_PH_ROWS = 100_000
if len(ph_df) > MAX_PH_ROWS:
    ph_df_sub = ph_df.groupby("recording_id", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), max(1, int(MAX_PH_ROWS * len(g) / len(ph_df)))),
                            random_state=42)
    ).reset_index(drop=True)
else:
    ph_df_sub = ph_df
log(f"  Phase Optuna subsample: {len(ph_df_sub)}")

X_ph_sub = ph_df_sub[WIN_FEAT_COLS].copy()
y_ph_sub = ph_df_sub["phase_label"].values
ph_fold_splits_sub = get_fold_splits(ph_df_sub)


def ph_optuna_obj(trial):
    params = {
        "objective": "multiclass", "num_class": 3,
        "n_estimators": trial.suggest_int("n_estimators", 80, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "num_leaves": trial.suggest_int("num_leaves", 8, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "class_weight": "balanced",
        "verbose": -1, "n_jobs": -1,
    }
    fold_f1s = []
    for tr_idx, te_idx in ph_fold_splits_sub:
        if not te_idx:
            continue
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("m", lgb.LGBMClassifier(**params))])
        pipe.fit(X_ph_sub.iloc[tr_idx], y_ph_sub[tr_idx])
        preds = pipe.predict(X_ph_sub.iloc[te_idx])
        fold_f1s.append(f1_score(y_ph_sub[te_idx], preds, average="macro",
                                  zero_division=0, labels=PHASE_CLASSES))
    return -float(np.mean(fold_f1s)) if fold_f1s else 0.0

log("  Optuna 40 trials (phase ML)...")
ph_study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
ph_study.optimize(ph_optuna_obj, n_trials=40)
bpp = ph_study.best_params
bpp.update({"objective": "multiclass", "num_class": 3,
            "class_weight": "balanced", "verbose": -1, "n_jobs": -1})
log(f"  Best phase ML Optuna F1: {-ph_study.best_value:.4f}")

# Final CV on full phase dataset
ph_fold_f1s, ph_ps = [], {}
ph_all_t, ph_all_p = [], []
for fi, (tr, te) in enumerate(ph_fold_splits):
    if not te:
        continue
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("m", lgb.LGBMClassifier(**bpp))])
    pipe.fit(X_ph.iloc[tr], y_ph[tr])
    preds = pipe.predict(X_ph.iloc[te])
    f1 = float(f1_score(y_ph[te], preds, average="macro", zero_division=0, labels=PHASE_CLASSES))
    ph_fold_f1s.append(f1)
    log(f"    Fold {fi}: F1={f1:.4f}")
    for s, t, p in zip(ph_df.iloc[te]["subject_id"], y_ph[te], preds):
        ph_ps.setdefault(s, {"t": [], "p": []})
        ph_ps[s]["t"].append(t); ph_ps[s]["p"].append(p)
    ph_all_t.extend(y_ph[te].tolist())
    ph_all_p.extend(preds.tolist())

ml_ph_mean = float(np.mean(ph_fold_f1s)) if ph_fold_f1s else 0.0
ml_ph_std  = float(np.std(ph_fold_f1s)) if ph_fold_f1s else 0.0
ml_ph_ps   = {s: float(f1_score(v["t"], v["p"], average="macro", zero_division=0,
                                 labels=PHASE_CLASSES))
              for s, v in ph_ps.items()}
log(f"\n  Phase ML F1: {ml_ph_mean:.4f} +/- {ml_ph_std:.4f}")
log(f"  Per-subject: {ml_ph_ps}")

# Final model
final_ph = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("m", lgb.LGBMClassifier(**bpp))])
final_ph.fit(X_ph, y_ph)
joblib.dump(final_ph, MODELS_DIR / "phase.joblib")
log("  Saved phase.joblib")

# SHAP
ph_fi = {}
try:
    sidx = RNG.choice(len(X_ph), min(3000, len(X_ph)), replace=False)
    X_ph_s = pd.DataFrame(final_ph["imp"].transform(X_ph.iloc[sidx]), columns=X_ph.columns)
    ph_exp = shap.TreeExplainer(final_ph["m"])
    ph_sv  = ph_exp.shap_values(X_ph_s)
    if isinstance(ph_sv, list):
        pm = np.mean([np.abs(v).mean(axis=0) for v in ph_sv], axis=0)
    else:
        pm = np.abs(ph_sv).mean(axis=0)
    ph_fi = dict(sorted(zip(X_ph.columns, pm.tolist()), key=lambda x: x[1], reverse=True)[:20])
    log(f"  Top phase features: {list(ph_fi.keys())[:5]}")
except Exception as e:
    log(f"  Phase SHAP failed: {e}")

# ML confusion matrix
cm_ph = confusion_matrix(ph_all_t, ph_all_p, labels=PHASE_CLASSES)
disp_ph = ConfusionMatrixDisplay(cm_ph, display_labels=PHASE_CLASSES)
fig, ax = plt.subplots(figsize=(6, 5))
disp_ph.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Phase ML fallback  F1={ml_ph_mean:.3f}")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrix_phase.png", dpi=150)
plt.close()
log("  Saved confusion_matrix_phase.png")

phase_primary_f1 = ml_ph_mean
phase_primary    = "ml_fallback"
phase_pass = phase_primary_f1 > baselines["phase"]["mean"] + 0.2
log(f"  Phase PASS (> baseline+0.2={baselines['phase']['mean']+0.2:.4f}): {phase_pass}")


# ===========================================================================
# TASK 4: REP COUNTING (state machine)
# Peak detection on acc_rms per set (Pernek et al. 2015).
# Adaptive threshold = median + 0.3*std, min inter-peak distance = 5 windows (500 ms).
# ===========================================================================
log("\n" + "="*60)
log("TASK 4: REP COUNTING (state machine)")
log("="*60)

set_df_ri = set_df.reset_index(drop=True)


def count_reps_sm(test_sets, win_df):
    rows = []
    for _, row in test_sets.iterrows():
        rec  = row["recording_id"]
        sn   = row["set_number"]
        gt   = int(row["n_reps"])
        mask = ((win_df["recording_id"] == rec) & (win_df["set_number"] == sn)
                & (win_df["in_active_set"] == True))
        grp  = win_df[mask].sort_values("t_window_center_s")
        if len(grp) < 5:
            rows.append({"recording_id": rec, "set_number": sn,
                         "subject_id": row["subject_id"], "n_reps_gt": gt, "n_reps_pred": 0})
            continue
        sig    = grp["acc_rms"].fillna(grp["acc_rms"].median()).values
        thresh = float(np.median(sig) + 0.3 * np.std(sig))
        peaks, _ = find_peaks(sig, height=thresh, distance=5)
        rows.append({"recording_id": rec, "set_number": sn,
                     "subject_id": row["subject_id"],
                     "n_reps_gt": gt, "n_reps_pred": int(len(peaks))})
    return pd.DataFrame(rows)


rep_fold_splits_s = get_fold_splits(set_df_ri)
sm_rep_fold_maes  = []
all_sm_res        = []

for fi, (tr, te) in enumerate(rep_fold_splits_s):
    if not te:
        continue
    res = count_reps_sm(set_df_ri.iloc[te], win_df)
    if len(res) == 0:
        continue
    mae = float(mean_absolute_error(res["n_reps_gt"], res["n_reps_pred"]))
    sm_rep_fold_maes.append(mae)
    all_sm_res.append(res)
    log(f"    Fold {fi}: rep SM MAE={mae:.4f}")

sm_res_all  = pd.concat(all_sm_res, ignore_index=True) if all_sm_res else pd.DataFrame()
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

log(f"\n  State-machine rep MAE: {sm_rep_mean:.4f} +/- {sm_rep_std:.4f}")
log(f"  Exact: {sm_exact*100:.1f}%  Within-1: {sm_within1*100:.1f}%")
log(f"  Per-subject: {sm_ps_rep}")

ml_rep_mean = None; ml_rep_std = None; ml_rep_ps = None
rep_primary = "state_machine"
rep_fi = {}
rep_exact_final   = sm_exact
rep_within1_final = sm_within1
active_rep_mae    = sm_rep_mean

if sm_rep_mean >= baselines["reps"]["mean"]:
    log(f"  SM MAE {sm_rep_mean:.4f} >= baseline {baselines['reps']['mean']:.4f} -- training ML fallback")
    rep_primary = "ml_fallback"

    rep_df  = set_df.dropna(subset=["n_reps"]).reset_index(drop=True)
    X_rep   = rep_df[SET_FEAT_COLS].copy()
    y_rep   = rep_df["n_reps"].values.astype(float)
    rep_folds = get_fold_splits(rep_df)

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
        for tr_idx, te_idx in rep_folds:
            if not te_idx:
                continue
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("m", lgb.LGBMRegressor(**params))])
            pipe.fit(X_rep.iloc[tr_idx], y_rep[tr_idx])
            fold_maes.append(float(mean_absolute_error(y_rep[te_idx],
                                                         pipe.predict(X_rep.iloc[te_idx]))))
        return float(np.mean(fold_maes)) if fold_maes else 999.0

    log("  Optuna 40 trials (rep ML)...")
    rep_study = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    rep_study.optimize(rep_optuna_obj, n_trials=40)
    brp = rep_study.best_params
    brp.update({"objective": "regression_l1", "verbose": -1, "n_jobs": -1})

    rep_fold_maes2, rep_ps2 = [], {}
    rep_all_t, rep_all_p  = [], []
    for fi, (tr, te) in enumerate(rep_folds):
        if not te:
            continue
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("m", lgb.LGBMRegressor(**brp))])
        pipe.fit(X_rep.iloc[tr], y_rep[tr])
        preds = pipe.predict(X_rep.iloc[te])
        mae   = float(mean_absolute_error(y_rep[te], preds))
        rep_fold_maes2.append(mae)
        log(f"    Fold {fi}: rep ML MAE={mae:.4f}")
        for s, t, p in zip(rep_df.iloc[te]["subject_id"], y_rep[te], preds):
            rep_ps2.setdefault(s, {"t": [], "p": []})
            rep_ps2[s]["t"].append(float(t)); rep_ps2[s]["p"].append(float(p))
        rep_all_t.extend(y_rep[te].tolist()); rep_all_p.extend(preds.tolist())

    ml_rep_mean = float(np.mean(rep_fold_maes2))
    ml_rep_std  = float(np.std(rep_fold_maes2))
    ml_rep_ps   = {s: float(mean_absolute_error(v["t"], v["p"])) for s, v in rep_ps2.items()}
    rep_exact_final   = float((np.round(rep_all_p) == np.array(rep_all_t)).mean())
    rep_within1_final = float((np.abs(np.array(rep_all_p) - np.array(rep_all_t)) <= 1).mean())
    active_rep_mae    = ml_rep_mean
    log(f"\n  ML rep MAE: {ml_rep_mean:.4f} +/- {ml_rep_std:.4f}")

    final_rep = Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("m", lgb.LGBMRegressor(**brp))])
    final_rep.fit(X_rep, y_rep)
    joblib.dump(final_rep, MODELS_DIR / "reps.joblib")
    log("  Saved reps.joblib")

    try:
        X_rep_imp = pd.DataFrame(final_rep["imp"].transform(X_rep), columns=X_rep.columns)
        rep_exp   = shap.TreeExplainer(final_rep["m"])
        rep_sv    = rep_exp.shap_values(X_rep_imp)
        rep_sm2   = np.abs(rep_sv).mean(axis=0)
        rep_fi    = dict(sorted(zip(X_rep.columns, rep_sm2.tolist()),
                                 key=lambda x: x[1], reverse=True)[:20])
        log(f"  Top rep features: {list(rep_fi.keys())[:5]}")
    except Exception as e:
        log(f"  Rep SHAP failed: {e}")

rep_pass = active_rep_mae < baselines["reps"]["mean"]
log(f"  Rep PASS (< baseline {baselines['reps']['mean']:.4f}): {rep_pass}")

# ===========================================================================
# SAVE METRICS.JSON
# ===========================================================================
log("\n" + "="*60)
log("SAVING METRICS.JSON")
log("="*60)

fat_med     = float(np.median(list(FAT_PS_MAE.values())))
fat_outliers = {s: float(v) for s, v in FAT_PS_MAE.items() if v > 3 * fat_med}

ph_ps_final = ml_ph_ps  # always ML fallback for phase

metrics = {
    "fatigue": {
        "metric": "MAE",
        "mean": float(FAT_MEAN),
        "std": float(FAT_STD),
        "fold_scores": [float(v) for v in FAT_FOLDS],
        "per_subject": {s: float(v) for s, v in FAT_PS_MAE.items()},
        "baseline": float(baselines["fatigue"]["mean"]),
        "pass": bool(fat_pass),
        "fail_reason": (
            None if fat_pass else
            "Model beats baseline by only 4.1% (0.965 vs 1.006); the 30%-below threshold"
            " requires MAE <= 0.704. Low-data regime (N=156 sets, 11 subjects) and narrow"
            " RPE range [4-10] with mean 7.1 limits regression headroom. Data review recommended."
        ),
        "outlier_subjects": fat_outliers,
    },
    "exercise": {
        "metric": "macro_F1",
        "mean": float(EX_MEAN),
        "std": float(EX_STD),
        "fold_scores": [float(v) for v in EX_FOLDS],
        "per_subject": {s: float(v) for s, v in EX_PS_F1.items()},
        "baseline": float(baselines["exercise"]["mean"]),
        "pass": bool(ex_pass),
        "fail_reason": None,
    },
    "phase": {
        "metric": "macro_F1",
        "state_machine": float(sm_f1_check),
        "state_machine_per_subject": {s: float(v) for s, v in sm_ps_f1.items()},
        "ml_fallback": {
            "mean": float(ml_ph_mean),
            "std": float(ml_ph_std),
            "per_subject": {s: float(v) for s, v in ml_ph_ps.items()},
        },
        "primary": phase_primary,
        "mean": float(phase_primary_f1),
        "std": float(ml_ph_std),
        "per_subject": {s: float(v) for s, v in ph_ps_final.items()},
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
            "per_subject": {s: float(v) for s, v in ml_rep_ps.items()},
        } if ml_rep_mean is not None else None,
        "primary": rep_primary,
        "mean": float(active_rep_mae),
        "std": float(ml_rep_std if ml_rep_std is not None else sm_rep_std),
        "exact_pct": float(rep_exact_final * 100),
        "within1_pct": float(rep_within1_final * 100),
        "baseline": float(baselines["reps"]["mean"]),
        "pass": bool(rep_pass),
        "fail_reason": None if rep_pass else "MAE >= naive mean-reps baseline",
    },
}

with open(RUN_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
log(f"  Saved: {RUN_DIR}/metrics.json")

# ===========================================================================
# SAVE FEATURE_IMPORTANCE.JSON
# ===========================================================================
feat_imp = {
    "fatigue": {k: float(v) for k, v in fat_fi.items()},
    "exercise": {k: float(v) for k, v in ex_fi.items()},
    "phase": {k: float(v) for k, v in ph_fi.items()},
    "reps": {k: float(v) for k, v in rep_fi.items()},
}
with open(RUN_DIR / "feature_importance.json", "w") as f:
    json.dump(feat_imp, f, indent=2)
log(f"  Saved: {RUN_DIR}/feature_importance.json")

# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
log("\n" + "="*60)
log("MODELING RESULTS -- 20260426_154705_default")
log("="*60)
log(f"- Fatigue (RPE):  MAE = {FAT_MEAN:.3f} +/- {FAT_STD:.3f}  [baseline MAE: {baselines['fatigue']['mean']:.3f}]  {'PASS' if fat_pass else 'FAIL'}")
log(f"- Exercise:       F1-macro = {EX_MEAN:.3f} +/- {EX_STD:.3f}  [baseline: {baselines['exercise']['mean']:.3f}]  {'PASS' if ex_pass else 'FAIL'}")
log(f"- Phase:          F1 = {phase_primary_f1:.3f} +/- {ml_ph_std:.3f}  [SM={sm_f1_check:.3f}]  method=ml_fallback  {'PASS' if phase_pass else 'FAIL'}")
log(f"- Reps:           MAE = {active_rep_mae:.3f}  exact={rep_exact_final*100:.1f}%  within-1={rep_within1_final*100:.1f}%  method={rep_primary}  {'PASS' if rep_pass else 'FAIL'}")
log(f"- Artifacts:      {RUN_DIR}")
log("Script complete.")
