"""Confusion matrices + fatigue plots for the v4-best models.

For each model+task pair, picks the appropriate one:
  - Exercise CM:  RF (clear winner, F1=0.43) — already saved, just copy.
  - Phase CM:     RF (F1=0.494) — already saved, just copy.
  - Fatigue:      feat-MLP (r=+0.19, best NN). Aggregate per-fold test_preds.pt
                  and draw scatter + per-subject MAE.
  - Reps:         RF (only model under MAE=2). Per-subject scatter from RF run.

Output: runs/comparison_v4/best_models/
"""
from __future__ import annotations
import json, sys, shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix as sk_cm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "comparison_v4" / "best_models"
OUT.mkdir(parents=True, exist_ok=True)

# Subjects in fold order (from configs/splits_clean.csv)
FOLD_SUBJECTS = {
    0: ["Juile", "lucas 2"],
    1: ["Hytten", "kiyomi"],
    2: ["Vivian"],
    3: ["Tias"],
    4: ["Raghild"],
}

# 1. Copy RF's existing CMs for context
RF_PLOTS = ROOT / "runs" / "optuna_clean_v4-rf" / "plots"
for fname in ("cm_exercise.png", "cm_phase.png", "fatigue_calibration.png", "fatigue_per_subject.png"):
    src = RF_PLOTS / fname
    if src.exists():
        shutil.copy2(src, OUT / f"rf_{fname}")
        print(f"copied rf_{fname}")


# 2. feat-MLP fatigue scatter + per-subject MAE
MLP_DIR = ROOT / "runs" / "optuna_clean_v4-features-mlp" / "phase2" / "mlp"


def load_mlp_predictions():
    """Return (subject, actual_rpe, predicted_rpe) tuples across all 3 seeds × 5 folds."""
    rows = []
    for seed_dir in sorted(MLP_DIR.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[1])
        for fold_dir in sorted(seed_dir.glob("fold_*")):
            fold = int(fold_dir.name.split("_")[1])
            preds_path = fold_dir / "test_preds.pt"
            if not preds_path.exists():
                continue
            try:
                d = torch.load(preds_path, weights_only=False, map_location="cpu")
            except Exception as e:
                print(f"  skip {preds_path}: {e}"); continue
            preds = d["preds"]
            targets = d["targets"]
            masks = d["masks"]
            fat_mask = masks["fatigue"].cpu().numpy().astype(bool) if "fatigue" in masks else None
            fat_pred = preds["fatigue"].cpu().numpy() if "fatigue" in preds else None
            fat_true = targets["fatigue"].cpu().numpy() if "fatigue" in targets else None
            if fat_pred is None or fat_true is None or fat_mask is None:
                continue
            fold_subjs = FOLD_SUBJECTS[fold]
            # Per-window predictions; collapse to per-set by averaging across windows that share a (subject, set) pair
            # We don't have set_number directly here, so use per-window for the scatter.
            for s in fold_subjs:
                # Cannot split per-subject when fold has 2 subjects without set_number;
                # fall back to per-fold aggregation.
                pass
            # Per-window points (large, but ok for scatter)
            valid = fat_mask
            for tp, pp in zip(fat_true[valid], fat_pred[valid]):
                rows.append((seed, fold, ",".join(fold_subjs), float(tp), float(pp)))
    return rows


def plot_mlp_fatigue():
    rows = load_mlp_predictions()
    if not rows:
        print("No feat-MLP fatigue predictions found"); return
    seeds = np.array([r[0] for r in rows])
    folds = np.array([r[1] for r in rows])
    subj_strs = np.array([r[2] for r in rows])
    actual = np.array([r[3] for r in rows])
    pred = np.array([r[4] for r in rows])

    # Per-window scatter is dense; aggregate to per-set median (one point per
    # (seed, fold, target-RPE) triple — within a fold all windows in one set
    # share the same target RPE, so this groups correctly).
    set_groups = defaultdict(list)
    for s, f, ss, t, p in rows:
        set_groups[(s, f, t)].append(p)
    set_actual = []; set_pred = []; set_fold = []; set_subj = []
    for (s, f, t), preds_list in set_groups.items():
        set_actual.append(t); set_pred.append(np.mean(preds_list))
        set_fold.append(f); set_subj.append(",".join(FOLD_SUBJECTS[f]))
    set_actual = np.array(set_actual); set_pred = np.array(set_pred)
    set_fold = np.array(set_fold); set_subj = np.array(set_subj)

    # === Scatter: predicted vs actual RPE per set ===
    fig, ax = plt.subplots(figsize=(8, 7))
    fold_colors = {0: "#27ae60", 1: "#16a085", 2: "#3498db", 3: "#9b59b6", 4: "#e67e22"}
    for f in sorted(np.unique(set_fold)):
        sel = set_fold == f
        ax.scatter(set_actual[sel], set_pred[sel], alpha=0.6, s=70, color=fold_colors[f],
                    edgecolor="black", linewidth=0.5,
                    label=f"Fold {f}: {','.join(FOLD_SUBJECTS[f])}")
    lo, hi = set_actual.min() - 0.5, set_actual.max() + 0.5
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    # Best-fit line
    slope, intercept = np.polyfit(set_actual, set_pred, 1)
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, slope * xs + intercept, "k:", linewidth=1.5,
             label=f"Best fit: y={slope:.2f}x+{intercept:.2f}")
    # Stats
    pearson_r = np.corrcoef(set_actual, set_pred)[0, 1]
    mae = float(np.mean(np.abs(set_actual - set_pred)))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual RPE (1-10)")
    ax.set_ylabel("Predicted RPE (3-seed mean per set)")
    ax.set_title(f"feat-MLP fatigue calibration (3 seeds × 5 folds, n={len(set_actual)} sets)\n"
                  f"Pearson r = {pearson_r:+.3f}  |  MAE = {mae:.3f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = OUT / "featmlp_fatigue_scatter.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # === Per-fold MAE bar ===
    fig, ax = plt.subplots(figsize=(8, 5))
    fold_mae = {f: float(np.mean(np.abs(set_actual[set_fold == f] - set_pred[set_fold == f])))
                 for f in sorted(np.unique(set_fold))}
    fold_r = {f: float(np.corrcoef(set_actual[set_fold == f], set_pred[set_fold == f])[0, 1])
               for f in sorted(np.unique(set_fold))}
    xs = list(fold_mae.keys())
    ys = [fold_mae[k] for k in xs]
    bars = ax.bar(xs, ys, color=[fold_colors[k] for k in xs],
                   edgecolor="black", linewidth=0.5)
    # Annotate Pearson r above each bar
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.02, f"MAE={y:.2f}\nr={fold_r[x]:+.2f}",
                 ha="center", va="bottom", fontsize=9)
    ax.axhline(1.013, color="grey", linestyle="--", linewidth=1, label="Baseline MAE 1.013")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Fold {x}\n({','.join(FOLD_SUBJECTS[x])})" for x in xs])
    ax.set_ylabel("MAE per fold")
    ax.set_title("feat-MLP fatigue per-fold MAE + Pearson r")
    ax.set_ylim(0, max(ys) * 1.4)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = OUT / "featmlp_fatigue_per_fold.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# 3. feat-MLP exercise + phase confusion matrices (per-window)
def plot_mlp_classification():
    """Aggregate per-window predictions across all seeds×folds and draw CM."""
    all_ex_pred, all_ex_true = [], []
    all_ph_pred, all_ph_true = [], []
    for seed_dir in sorted(MLP_DIR.glob("seed_*")):
        for fold_dir in sorted(seed_dir.glob("fold_*")):
            preds_path = fold_dir / "test_preds.pt"
            if not preds_path.exists(): continue
            try:
                d = torch.load(preds_path, weights_only=False, map_location="cpu")
            except Exception:
                continue
            for task in ("exercise", "phase"):
                pred = d["preds"].get(task)
                true = d["targets"].get(task)
                mask = d["masks"].get(task)
                if pred is None or true is None or mask is None:
                    continue
                pred = pred.cpu().numpy()
                true = true.cpu().numpy()
                mask = mask.cpu().numpy().astype(bool)
                # logits → argmax (skip if already integer)
                if pred.ndim == 2:
                    pred = pred.argmax(axis=1)
                if task == "exercise":
                    all_ex_pred.append(pred[mask]); all_ex_true.append(true[mask])
                else:
                    all_ph_pred.append(pred[mask]); all_ph_true.append(true[mask])

    # Get class names from saved encoders if available — fallback to indices
    meta = json.loads((MLP_DIR / "../../dataset_meta.json").read_text()) if (MLP_DIR / "../../dataset_meta.json").exists() else {}
    ex_classes = meta.get("exercise_classes") or ["benchpress", "deadlift", "pullup", "squat"]
    ph_classes = meta.get("phase_classes") or ["concentric", "eccentric", "isometric"]

    if all_ex_pred:
        yp = np.concatenate(all_ex_pred); yt = np.concatenate(all_ex_true)
        cm = sk_cm(yt, yp, labels=range(len(ex_classes)))
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=cm_norm.max())
        for i in range(len(ex_classes)):
            for j in range(len(ex_classes)):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                         ha="center", va="center",
                         color="white" if cm_norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(ex_classes))); ax.set_xticklabels(ex_classes, rotation=30, ha="right")
        ax.set_yticks(range(len(ex_classes))); ax.set_yticklabels(ex_classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"feat-MLP — Exercise CM (3 seeds × 5 folds, n={len(yt)} windows, normalised)")
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        out = OUT / "featmlp_cm_exercise.png"
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"Wrote {out}")

    if all_ph_pred:
        yp = np.concatenate(all_ph_pred); yt = np.concatenate(all_ph_true)
        # Drop classes with zero support
        present = sorted(set(yt.tolist()) | set(yp.tolist()))
        labels = [ph_classes[i] if i < len(ph_classes) else f"cls_{i}" for i in present]
        cm = sk_cm(yt, yp, labels=present)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm, cmap="Greens", vmin=0, vmax=cm_norm.max())
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                         ha="center", va="center",
                         color="white" if cm_norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"feat-MLP — Phase CM (3 seeds × 5 folds, n={len(yt)} windows, normalised)")
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        out = OUT / "featmlp_cm_phase.png"
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"Wrote {out}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    plot_mlp_fatigue()
    plot_mlp_classification()
    print(f"\nAll plots in {OUT}")


if __name__ == "__main__":
    main()
