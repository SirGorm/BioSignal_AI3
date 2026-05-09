"""
Train RF for exercise classification using ONLY 5 new time-domain features
(WAMP, MFL, MSR, LS, LS4) on EMG (2000 Hz) and acc-magnitude (100 Hz).

LOSO-CV grouped by subject_id (Name from Participants.xlsx).
No fatigue / no phase / no rep prediction — exercise label only.

Output:
  runs/<timestamp>_extras_exercise/
    metrics.json           per-subject + overall macro-F1, accuracy
    classification_report.txt
    confusion_matrix.csv
    features.parquet       cached feature table (subject_id, exercise, 10 features)

References
----------
- Phinyomark et al. 2012, 2018 (feature definitions; see src/features/extra_features.py)
- Saeb et al. 2017 (subject-wise CV mandatory to avoid leakage)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.extra_features import extras_window, baseline_threshold
from src.features.emg_features import emg_window_features
from src.features.acc_features import acc_mag_window_features

FS_EMG = 2000
FS_ACC = 100
EMG_WIN_MS = 5000
ACC_WIN_MS = 5000
HOP_MS = 200
BASELINE_S = 90.0

DATASET_ALIGNED = ROOT / "dataset_aligned"
LABELED = ROOT / "data" / "labeled"


def _nanfill(x: np.ndarray) -> np.ndarray:
    out = x.astype(float).copy()
    mask = np.isnan(out)
    if not mask.any():
        return out
    idx = np.arange(len(out))
    if (~mask).sum() == 0:
        return out
    fp = np.maximum.accumulate(np.where(~mask, idx, 0))
    out[mask] = out[fp[mask]]
    m2 = np.isnan(out)
    if m2.any():
        bp = np.minimum.accumulate(np.where(~m2, idx, len(out) - 1)[::-1])[::-1]
        out[m2] = out[bp[m2]]
    return out


def _filter_emg(x: np.ndarray) -> np.ndarray:
    x = _nanfill(x)
    sos = butter(4, [20.0, 450.0], btype="band", fs=FS_EMG, output="sos")
    y = sosfiltfilt(sos, x)
    b, a = iirnotch(50.0, Q=30, fs=FS_EMG)
    return filtfilt(b, a, y)


def _filter_acc(mag: np.ndarray) -> np.ndarray:
    mag = _nanfill(mag)
    sos = butter(4, [0.5, 20.0], btype="band", fs=FS_ACC, output="sos")
    return sosfiltfilt(sos, mag)


def _slide(signal: np.ndarray, t_unix: np.ndarray, fs: int,
           win_ms: int, hop_ms: int, thr: float, prefix: str) -> pd.DataFrame:
    """Slide window-by-window and compute BOTH:
      - canonical features from src/features/{emg,acc}_features.py
      - the 5 extras (wamp/mfl/msr/ls/ls4) with prefix '<m>_x_' to avoid
        column-name collisions (extras_window uses adaptive WAMP threshold
        and Phinyomark 2018 LS definitions, so they are genuinely different
        features from the canonical ones).
    """
    win = int(win_ms * fs / 1000)
    hop = max(1, int(hop_ms * fs / 1000))
    rows = []
    n = len(signal)
    pos = 0
    while pos + win <= n:
        w = signal[pos: pos + win]
        feats = extras_window(w, thr, f"{prefix}_x")
        if prefix == "emg":
            feats.update(emg_window_features(w, fs=fs))
        elif prefix == "acc":
            feats.update(acc_mag_window_features(w, fs=fs))
        feats["t_unix"] = float(t_unix[pos + win // 2])
        rows.append(feats)
        pos += hop
    return pd.DataFrame(rows)


def _load_csv(path: Path, col: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", col])
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    return df


def process_recording(rec_dir: Path, labeled_dir: Path,
                      modalities: tuple[str, ...]) -> pd.DataFrame:
    rec_id = rec_dir.name
    parquet = labeled_dir / rec_id / "aligned_features.parquet"
    if not parquet.exists():
        print(f"  SKIP {rec_id}: no aligned_features.parquet", file=sys.stderr)
        return pd.DataFrame()
    print(f"  Processing {rec_id}...", flush=True)
    labeled = pd.read_parquet(parquet, columns=[
        "t_unix", "subject_id", "recording_id", "in_active_set", "exercise",
    ])

    t0 = float(labeled["t_unix"].min())
    baseline_end = t0 + BASELINE_S

    grid = labeled[["t_unix", "subject_id", "recording_id",
                    "in_active_set", "exercise"]].sort_values("t_unix")
    merged = grid.copy()

    if "emg" in modalities:
        emg_raw = _load_csv(rec_dir / "emg.csv", "emg")
        emg_filt = _filter_emg(emg_raw["emg"].values)
        emg_t = emg_raw["timestamp"].values
        emg_baseline = emg_filt[emg_t <= baseline_end]
        emg_thr = baseline_threshold(emg_baseline, k=0.1)
        emg_feats = _slide(emg_filt, emg_t, FS_EMG, EMG_WIN_MS, HOP_MS,
                           emg_thr, "emg").sort_values("t_unix")
        merged = pd.merge_asof(merged, emg_feats, on="t_unix",
                               direction="backward", tolerance=2.0)

    if "acc" in modalities:
        ax = _load_csv(rec_dir / "ax.csv", "ax")
        ay = _load_csv(rec_dir / "ay.csv", "ay")
        az = _load_csv(rec_dir / "az.csv", "az")
        n_acc = min(len(ax), len(ay), len(az))
        acc_t = ax["timestamp"].values[:n_acc]
        acc_mag = np.sqrt(
            ax["ax"].values[:n_acc] ** 2
            + ay["ay"].values[:n_acc] ** 2
            + az["az"].values[:n_acc] ** 2
        )
        acc_filt = _filter_acc(acc_mag)
        acc_baseline = acc_filt[acc_t <= baseline_end]
        acc_thr = baseline_threshold(acc_baseline, k=0.1)
        acc_feats = _slide(acc_filt, acc_t, FS_ACC, ACC_WIN_MS, HOP_MS,
                           acc_thr, "acc").sort_values("t_unix")
        merged = pd.merge_asof(merged, acc_feats, on="t_unix",
                               direction="backward", tolerance=4.0)

    # All feature columns added by the merge_asof rounds — pick everything
    # except the meta columns. Includes canonical emg_/acc_ features plus
    # extras with '_x_' prefix.
    meta_cols = {"t_unix", "subject_id", "recording_id",
                 "in_active_set", "exercise"}
    feat_cols = [c for c in merged.columns if c not in meta_cols]
    merged = merged[merged["in_active_set"].astype(bool)].copy()
    merged = merged.dropna(subset=feat_cols + ["exercise"])

    # Decimate to ~one row per hop (200 ms). Labeled grid is 100 Hz → keep every 20th.
    merged = merged.iloc[::20].reset_index(drop=True)
    return merged[["subject_id", "recording_id", "exercise"] + feat_cols]


def main(cv_mode: str, k_folds: int, modalities: tuple[str, ...]) -> None:
    cv_suffix = "loso" if cv_mode == "loso" else f"k{k_folds}"
    mod_suffix = "_".join(modalities)
    out_dir = ROOT / "runs" / (
        f"{datetime.now():%Y%m%d_%H%M%S}_extras_exercise_{mod_suffix}_{cv_suffix}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Modalities: {', '.join(modalities)}")
    print(f"CV mode: {cv_mode}" + (f" (k={k_folds})" if cv_mode == "kfold" else ""))

    rec_dirs = sorted([p for p in DATASET_ALIGNED.iterdir()
                       if p.is_dir() and p.name.startswith("recording_")])
    print(f"Found {len(rec_dirs)} recordings.")

    parts = []
    for rec in rec_dirs:
        try:
            df = process_recording(rec, LABELED, modalities)
            if not df.empty:
                parts.append(df)
        except Exception as exc:
            print(f"  ERROR {rec.name}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    if not parts:
        print("No features produced.", file=sys.stderr)
        sys.exit(1)

    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(out_dir / "features.parquet", index=False)
    feat_cols = [c for c in full.columns
                 if c not in ("subject_id", "recording_id", "exercise")]
    print(f"\nTotal windows: {len(full)} | features: {len(feat_cols)} | "
          f"subjects: {full['subject_id'].nunique()}")
    print(f"Per-exercise counts:\n{full['exercise'].value_counts()}")
    print(f"Per-subject counts:\n{full['subject_id'].value_counts()}")

    X = full[feat_cols].values
    y = full["exercise"].values
    groups = full["subject_id"].values

    n_subjects = full["subject_id"].nunique()
    if cv_mode == "loso":
        splitter = LeaveOneGroupOut()
        n_splits = n_subjects
        cv_label = "LOSO (LeaveOneGroupOut)"
    else:
        n_splits = min(k_folds, n_subjects)
        splitter = GroupKFold(n_splits=n_splits)
        cv_label = f"GroupKFold (k={n_splits}, subject-wise)"

    fold_metrics = []
    all_pred = np.empty_like(y)
    classes = sorted(np.unique(y).tolist())

    print(f"\n{cv_label}: {n_splits} folds.")
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        held_subjs = sorted(set(full["subject_id"].iloc[te].tolist()))
        clf = RandomForestClassifier(
            n_estimators=400, n_jobs=-1, random_state=42,
            class_weight="balanced",
        )
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        all_pred[te] = pred
        f1 = f1_score(y[te], pred, average="macro", labels=classes, zero_division=0)
        acc = accuracy_score(y[te], pred)
        fold_metrics.append({
            "fold": fold,
            "held_out_subjects": held_subjs,
            "n_test": int(len(te)),
            "macro_f1": float(f1),
            "accuracy": float(acc),
        })
        subj_str = ", ".join(held_subjs)
        print(f"  fold {fold}: subjs=[{subj_str}]  "
              f"n={len(te):>6d}  macro-F1={f1:.3f}  acc={acc:.3f}")

    overall_f1 = f1_score(y, all_pred, average="macro",
                          labels=classes, zero_division=0)
    overall_acc = accuracy_score(y, all_pred)
    print(f"\nOverall (concat across folds):  macro-F1={overall_f1:.3f}  "
          f"accuracy={overall_acc:.3f}")
    mean_f1 = float(np.mean([m["macro_f1"] for m in fold_metrics]))
    std_f1 = float(np.std([m["macro_f1"] for m in fold_metrics]))
    print(f"Mean fold macro-F1:     {mean_f1:.3f} ± {std_f1:.3f}")

    cm = confusion_matrix(y, all_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    print(f"\nConfusion matrix:\n{cm_df}")

    report = classification_report(y, all_pred, labels=classes, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    # Permutation feature importance via fold-mean Gini importance from final fit
    clf_full = RandomForestClassifier(
        n_estimators=400, n_jobs=-1, random_state=42, class_weight="balanced",
    )
    clf_full.fit(X, y)
    importances = sorted(
        zip(feat_cols, clf_full.feature_importances_),
        key=lambda kv: -kv[1],
    )
    print("\nFeature importances (Gini, RF on full data):")
    for name, imp in importances:
        print(f"  {name:>12s}  {imp:.4f}")

    metrics = {
        "task": "exercise_classification",
        "cv": cv_label,
        "model": "RandomForestClassifier(n_estimators=400, balanced)",
        "modalities": list(modalities),
        "features": feat_cols,
        "classes": classes,
        "overall_macro_f1": overall_f1,
        "overall_accuracy": overall_acc,
        "mean_fold_macro_f1": mean_f1,
        "std_fold_macro_f1": std_f1,
        "folds": fold_metrics,
        "feature_importances": [
            {"feature": n, "importance": float(v)} for n, v in importances
        ],
        "n_windows": int(len(full)),
        "n_subjects": int(full["subject_id"].nunique()),
        "n_recordings": int(full["recording_id"].nunique()),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RF on extras-only features for exercise classification."
    )
    parser.add_argument(
        "--cv", choices=["loso", "kfold"], default="kfold",
        help="Cross-validation: 'loso' (one subject out) or 'kfold' (default, k=5).",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of folds when --cv=kfold (default 5).",
    )
    parser.add_argument(
        "--modalities", choices=["emg", "acc", "both"], default="both",
        help="Which signal(s) to extract features from (default both).",
    )
    args = parser.parse_args()
    mods = ("emg", "acc") if args.modalities == "both" else (args.modalities,)
    main(cv_mode=args.cv, k_folds=args.k, modalities=mods)
