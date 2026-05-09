"""Train the tuned RF on the optimal 8-feature subset and emit the standard
artefacts (confusion matrix, classification report, per-fold/per-subject metrics).

Reuses the cached features.parquet from train_rf_extras_exercise.py so no
re-extraction is needed.

Optimal config found by scripts/tune_rf_extras.py + tune_rf_phase_fg.py:
  RandomForest(n_estimators=800, max_depth=8, class_weight='balanced')
  Features (8): emg_x_wamp, emg_mfl, emg_msr, acc_rms, acc_jerk_rms,
                acc_dom_freq, acc_rep_band_power, acc_rep_band_ratio
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
FEAT_PATH = ROOT / "runs/20260508_170518_extras_exercise_emg_acc_k5/features.parquet"

FINAL_FEATURES = [
    "emg_x_wamp", "emg_mfl", "emg_msr",
    "acc_rms", "acc_jerk_rms", "acc_dom_freq",
    "acc_rep_band_power", "acc_rep_band_ratio",
]
RF_CFG = dict(
    n_estimators=800, max_depth=8, class_weight="balanced",
    random_state=42, n_jobs=-1,
)


def main():
    full = pd.read_parquet(FEAT_PATH)
    missing = [c for c in FINAL_FEATURES if c not in full.columns]
    if missing:
        raise ValueError(f"features.parquet is missing: {missing}")

    full = full.dropna(subset=FINAL_FEATURES + ["exercise"]).reset_index(drop=True)
    X = full[FINAL_FEATURES].values
    y = full["exercise"].values
    groups = full["subject_id"].values
    classes = sorted(np.unique(y).tolist())

    print(f"Final RF: {len(full):,} rows x {len(FINAL_FEATURES)} features "
          f"({full['subject_id'].nunique()} subjects)")
    print(f"Config: {RF_CFG}")
    print(f"Features: {FINAL_FEATURES}\n")

    out_dir = ROOT / "runs" / (
        f"{datetime.now():%Y%m%d_%H%M%S}_extras_exercise_FINAL_emg_acc_k5"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    splitter = GroupKFold(n_splits=5)
    fold_metrics = []
    all_pred = np.empty_like(y)
    per_subject_f1 = {}

    print("GroupKFold (k=5, subject-wise):")
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        held = sorted(set(full["subject_id"].iloc[te].tolist()))
        clf = RandomForestClassifier(**RF_CFG)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        all_pred[te] = pred
        f1 = f1_score(y[te], pred, average="macro", labels=classes, zero_division=0)
        acc = accuracy_score(y[te], pred)
        for subj in held:
            mask = full["subject_id"].iloc[te].values == subj
            if mask.any():
                subj_f1 = f1_score(y[te][mask], pred[mask], average="macro",
                                    labels=classes, zero_division=0)
                per_subject_f1[subj] = float(subj_f1)
        fold_metrics.append({
            "fold": fold, "held_out_subjects": held, "n_test": int(len(te)),
            "macro_f1": float(f1), "accuracy": float(acc),
        })
        print(f"  fold {fold}: subjs=[{', '.join(held):20s}]  "
              f"n={len(te):>6d}  macro-F1={f1:.4f}  acc={acc:.4f}")

    overall_f1 = f1_score(y, all_pred, average="macro", labels=classes, zero_division=0)
    overall_acc = accuracy_score(y, all_pred)
    mean_f1 = float(np.mean([m["macro_f1"] for m in fold_metrics]))
    std_f1 = float(np.std([m["macro_f1"] for m in fold_metrics]))
    print(f"\nOverall (concat): macro-F1={overall_f1:.4f}  acc={overall_acc:.4f}")
    print(f"Mean fold:        macro-F1={mean_f1:.4f} +/- {std_f1:.4f}")

    print(f"\nPer-subject macro-F1:")
    for subj, f1 in sorted(per_subject_f1.items(), key=lambda kv: -kv[1]):
        print(f"  {subj:>12s}  {f1:.4f}")

    cm = confusion_matrix(y, all_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    print(f"\nConfusion matrix:\n{cm_df}")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm_df = pd.DataFrame(cm_norm.round(3), index=classes, columns=classes)
    cm_norm_df.to_csv(out_dir / "confusion_matrix_normalized.csv")
    print(f"\nConfusion matrix (row-normalized):\n{cm_norm_df}")

    report = classification_report(y, all_pred, labels=classes, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    print(f"\n{report}")

    clf_full = RandomForestClassifier(**RF_CFG)
    clf_full.fit(X, y)
    importances = sorted(zip(FINAL_FEATURES, clf_full.feature_importances_),
                          key=lambda kv: -kv[1])
    print("Final feature importances (Gini):")
    for n, v in importances:
        print(f"  {n:>22s}  {v:.4f}")

    metrics = {
        "task": "exercise_classification",
        "cv": "GroupKFold (k=5, subject-wise)",
        "model": f"RandomForestClassifier({RF_CFG})",
        "features": FINAL_FEATURES,
        "n_features": len(FINAL_FEATURES),
        "classes": classes,
        "overall_macro_f1": float(overall_f1),
        "overall_accuracy": float(overall_acc),
        "mean_fold_macro_f1": mean_f1,
        "std_fold_macro_f1": std_f1,
        "folds": fold_metrics,
        "per_subject_macro_f1": per_subject_f1,
        "feature_importances": [
            {"feature": n, "importance": float(v)} for n, v in importances
        ],
        "n_windows": int(len(full)),
        "n_subjects": int(full["subject_id"].nunique()),
        "n_recordings": int(full["recording_id"].nunique()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                          encoding="utf-8")
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
