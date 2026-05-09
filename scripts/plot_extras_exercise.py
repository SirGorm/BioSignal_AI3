"""
Plot overfit (train vs test macro-F1 per LOSO fold) and confusion matrix
for the extras-only exercise RF run.

Reads cached features.parquet from a run directory and re-trains LOSO with
train-set scoring captured per fold. No new feature extraction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def latest_run() -> Path:
    runs = sorted((ROOT / "runs").glob("*_extras_exercise*"))
    if not runs:
        raise SystemExit("No extras_exercise run found.")
    return runs[-1]


def main(run_dir: Path, cv_mode: str, k_folds: int) -> None:
    feats = pd.read_parquet(run_dir / "features.parquet")
    feat_cols = [c for c in feats.columns
                 if c not in ("subject_id", "recording_id", "exercise")]
    X = feats[feat_cols].values
    y = feats["exercise"].values
    groups = feats["subject_id"].values
    classes = sorted(np.unique(y).tolist())

    n_subjects = feats["subject_id"].nunique()
    if cv_mode == "loso":
        splitter = LeaveOneGroupOut()
        cv_label = "LOSO"
    else:
        n_splits = min(k_folds, n_subjects)
        splitter = GroupKFold(n_splits=n_splits)
        cv_label = f"GroupKFold k={n_splits}"
    rows = []
    all_pred = np.empty_like(y)
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        held = sorted(set(feats["subject_id"].iloc[te].tolist()))
        clf = RandomForestClassifier(
            n_estimators=400, n_jobs=-1, random_state=42,
            class_weight="balanced",
        )
        clf.fit(X[tr], y[tr])
        train_pred = clf.predict(X[tr])
        test_pred = clf.predict(X[te])
        all_pred[te] = test_pred
        rows.append({
            "fold": fold,
            "subjects": ", ".join(held),
            "train_f1": f1_score(y[tr], train_pred, average="macro",
                                 labels=classes, zero_division=0),
            "test_f1":  f1_score(y[te], test_pred, average="macro",
                                 labels=classes, zero_division=0),
            "n_train": len(tr),
            "n_test":  len(te),
        })
        print(f"  fold {fold} [{', '.join(held)}]  "
              f"train={rows[-1]['train_f1']:.3f}  test={rows[-1]['test_f1']:.3f}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "overfit_per_fold.csv", index=False)

    # --- Plot 1: train vs test per fold (overfit) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df["train_f1"], width=w, label="Train", color="#3b82f6")
    ax.bar(x + w / 2, df["test_f1"],  width=w, label="Test (held-out)",
           color="#ef4444")
    for xi, (tr_v, te_v) in enumerate(zip(df["train_f1"], df["test_f1"])):
        ax.annotate(f"{tr_v:.2f}", (xi - w / 2, tr_v),
                    ha="center", va="bottom", fontsize=8)
        ax.annotate(f"{te_v:.2f}", (xi + w / 2, te_v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["subjects"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Overfit per fold ({cv_label}, {len(df)} folds) — RF on extras (EMG + Acc) "
                 f"| mean train={df['train_f1'].mean():.2f}, "
                 f"test={df['test_f1'].mean():.2f}, "
                 f"gap={df['train_f1'].mean() - df['test_f1'].mean():.2f}")
    ax.axhline(0.25, ls="--", color="grey", lw=0.8, label="Chance (4 classes)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "overfit_per_fold.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: confusion matrix (LOSO concat) ---
    cm = confusion_matrix(y, all_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title, fmt in [
        (axes[0], cm,      "Counts",          "d"),
        (axes[1], cm_norm, "Row-normalised (recall)", ".2f"),
    ]:
        im = ax.imshow(mat, cmap="Blues",
                       vmin=0, vmax=mat.max() if fmt == "d" else 1.0)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        thresh = mat.max() / 2.0 if fmt == "d" else 0.5
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(mat[i, j], fmt),
                        ha="center", va="center",
                        color="white" if mat[i, j] > thresh else "black",
                        fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle(
        f"Confusion matrix (LOSO concat) — macro-F1 = "
        f"{f1_score(y, all_pred, average='macro', labels=classes, zero_division=0):.3f}"
    )
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote:")
    print(f"  {run_dir / 'overfit_per_fold.png'}")
    print(f"  {run_dir / 'confusion_matrix.png'}")
    print(f"  {run_dir / 'overfit_per_fold.csv'}")
    print(f"\nSummary: train={df['train_f1'].mean():.3f}  "
          f"test={df['test_f1'].mean():.3f}  "
          f"gap={df['train_f1'].mean() - df['test_f1'].mean():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=None,
                   help="Run directory; default = latest *_extras_exercise*/")
    p.add_argument("--cv", choices=["loso", "kfold"], default="kfold",
                   help="Re-evaluate with this CV (default kfold).")
    p.add_argument("--k", type=int, default=5,
                   help="Number of folds when --cv=kfold (default 5).")
    args = p.parse_args()
    rd = Path(args.run_dir) if args.run_dir else latest_run()
    print(f"Run: {rd}")
    main(rd, cv_mode=args.cv, k_folds=args.k)
