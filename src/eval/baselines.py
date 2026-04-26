"""Baseline metrics for the 4 tasks. Subject-wise CV.

Baselines:
- Fatigue regression (per-set, RPE 1-10): DummyRegressor with mean (Pedregosa 2011)
- Exercise classification (per-window): DummyClassifier most_frequent (Pedregosa 2011)
- Phase segmentation (per-window): DummyClassifier most_frequent
- Rep counting (per-set): naive predictor = mean reps per set

These are floor performances; trained models must beat them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold


def fatigue_baseline(set_df: pd.DataFrame, n_splits: int = 5) -> dict:
    df = set_df.dropna(subset=["rpe_for_this_set"]).reset_index(drop=True)
    y = df["rpe_for_this_set"].astype(float).values
    groups = df["subject_id"].values
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    fold_maes = []
    for tr, te in gkf.split(df, groups=groups):
        clf = DummyRegressor(strategy="mean").fit(df.iloc[tr], y[tr])
        pred = clf.predict(df.iloc[te])
        fold_maes.append(mean_absolute_error(y[te], pred))
    return {
        "task": "fatigue_rpe",
        "metric": "MAE",
        "n_samples": int(len(df)),
        "fold_scores": [float(x) for x in fold_maes],
        "mean": float(np.mean(fold_maes)),
        "std": float(np.std(fold_maes)),
    }


def exercise_baseline(window_df: pd.DataFrame, n_splits: int = 5) -> dict:
    df = window_df[window_df["in_active_set"] & window_df["exercise"].notna()].reset_index(drop=True)
    y = df["exercise"].values
    groups = df["subject_id"].values
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    fold_f1 = []
    for tr, te in gkf.split(df, groups=groups):
        clf = DummyClassifier(strategy="most_frequent").fit(df.iloc[tr], y[tr])
        pred = clf.predict(df.iloc[te])
        fold_f1.append(f1_score(y[te], pred, average="macro"))
    return {
        "task": "exercise",
        "metric": "macro_F1",
        "n_samples": int(len(df)),
        "fold_scores": [float(x) for x in fold_f1],
        "mean": float(np.mean(fold_f1)),
        "std": float(np.std(fold_f1)),
    }


def phase_baseline(window_df: pd.DataFrame, n_splits: int = 5) -> dict:
    df = window_df[window_df["in_active_set"] & window_df["phase_label"].isin(["concentric", "eccentric", "isometric"])].reset_index(drop=True)
    if len(df) == 0:
        return {"task": "phase", "metric": "macro_F1", "n_samples": 0, "fold_scores": [], "mean": float("nan"), "std": float("nan")}
    y = df["phase_label"].values
    groups = df["subject_id"].values
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    fold_f1 = []
    for tr, te in gkf.split(df, groups=groups):
        clf = DummyClassifier(strategy="most_frequent").fit(df.iloc[tr], y[tr])
        pred = clf.predict(df.iloc[te])
        fold_f1.append(f1_score(y[te], pred, average="macro"))
    return {
        "task": "phase",
        "metric": "macro_F1",
        "n_samples": int(len(df)),
        "fold_scores": [float(x) for x in fold_f1],
        "mean": float(np.mean(fold_f1)),
        "std": float(np.std(fold_f1)),
    }


def rep_baseline(set_df: pd.DataFrame, n_splits: int = 5) -> dict:
    df = set_df.dropna(subset=["n_reps"]).reset_index(drop=True) if "n_reps" in set_df.columns else None
    if df is None or len(df) == 0:
        return {"task": "reps", "metric": "MAE", "n_samples": 0, "fold_scores": [], "mean": float("nan"), "std": float("nan")}
    y = df["n_reps"].astype(float).values
    groups = df["subject_id"].values
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    fold_maes = []
    for tr, te in gkf.split(df, groups=groups):
        clf = DummyRegressor(strategy="mean").fit(df.iloc[tr], y[tr])
        pred = clf.predict(df.iloc[te])
        fold_maes.append(mean_absolute_error(y[te], pred))
    return {
        "task": "reps",
        "metric": "MAE",
        "n_samples": int(len(df)),
        "fold_scores": [float(x) for x in fold_maes],
        "mean": float(np.mean(fold_maes)),
        "std": float(np.std(fold_maes)),
    }


def main(run_dir: Path) -> dict:
    feat_dir = run_dir / "features"
    window = pd.read_parquet(feat_dir / "window_features.parquet")
    sets = pd.read_parquet(feat_dir / "set_features.parquet")

    out = {
        "fatigue": fatigue_baseline(sets),
        "exercise": exercise_baseline(window),
        "phase": phase_baseline(window),
        "reps": rep_baseline(sets),
    }
    out_path = run_dir / "metrics_baseline.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    for k, v in out.items():
        print(f"  {k:10s} {v['metric']:10s} {v['mean']:.3f} ± {v['std']:.3f}  (n={v['n_samples']})")
    return out


if __name__ == "__main__":
    import sys
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/20260426_154705_default")
    main(run_dir)
