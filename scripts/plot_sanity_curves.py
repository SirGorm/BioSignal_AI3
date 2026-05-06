"""Plot per-task learning curves from the baseline-norm sanity runs.

For each of the two sanity runs (raw-TCN and feat-MLP @ 2s), aggregate
train/val loss per epoch across the 7 LOSO folds (single seed only) and
plot 4 tasks × 2 cols (train, val), plus an overlay panel that compares
feat-MLP sanity vs the full v12 feat-MLP @ 2s curves so the user can
visually verify whether the exercise val-loss divergence has been
reduced.

Outputs to runs/sanity_baseline_norm-w2s-multi-feat-mlp/plots/:
  curves_per_task.png             — feat-MLP sanity, 4 tasks × (train, val)
  curves_per_task_overlay_v12.png — same, with v12 (no-norm) overlaid
And to runs/sanity_baseline_norm-w2s-multi-raw-tcn/plots/:
  curves_per_task.png             — raw-TCN sanity, 4 tasks × (train, val)
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TASKS = ["exercise", "phase", "fatigue", "reps"]


def collect(run_dir: Path):
    p2 = run_dir / "phase2"
    by_epoch = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for hp in p2.rglob("history.json"):
        try:
            hist = json.loads(hp.read_text())
        except Exception:
            continue
        for entry in hist:
            ep = entry["epoch"]
            for task in TASKS:
                tr = entry.get("train", {}).get(task)
                if tr is not None:
                    by_epoch[task]["train"][ep].append(tr)
                vl = entry.get("val_loss", {}).get(task)
                if vl is not None:
                    by_epoch[task]["val"][ep].append(vl)
    out = {}
    for task in TASKS:
        out[task] = {}
        for stat in ("train", "val"):
            d = by_epoch[task][stat]
            if not d:
                out[task][stat] = (np.array([]), np.array([]), np.array([]))
                continue
            eps = sorted(d.keys())
            mean = np.array([float(np.mean(d[e])) for e in eps])
            std = np.array([float(np.std(d[e])) for e in eps])
            out[task][stat] = (np.array(eps), mean, std)
    return out


def plot_panel(ax, curves, task, stat, color, label, with_band=True):
    x, y, s = curves[task][stat]
    if not len(x):
        return
    ax.plot(x, y, color=color, linewidth=1.8, label=label)
    if with_band and len(s):
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.15)


def render_run(run_dir: Path, label: str, out_png: Path, color="#1f77b4"):
    curves = collect(run_dir)
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    for row, task in enumerate(TASKS):
        plot_panel(axes[row, 0], curves, task, "train", color, label)
        plot_panel(axes[row, 1], curves, task, "val",   color, label)
        axes[row, 0].set_title(f"{task} — train", fontsize=11)
        axes[row, 1].set_title(f"{task} — val",   fontsize=11)
        for c in (0, 1):
            axes[row, c].set_xlabel("Epoch")
            axes[row, c].set_ylabel(f"{task} loss")
            axes[row, c].grid(linestyle=":", alpha=0.4)
    axes[0, 1].legend(loc="best", fontsize=9)
    fig.suptitle(f"Sanity — {label} @ 2 s, 7 folds × 1 seed (band = ±1 σ across folds)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    return curves


def render_overlay(sanity_curves, v12_curves, out_png: Path):
    """Overlay sanity (with band) vs v12 reference (line only)."""
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    for row, task in enumerate(TASKS):
        plot_panel(axes[row, 0], sanity_curves, task, "train",
                   "#27ae60", "feat-MLP + baseline-norm (sanity)")
        plot_panel(axes[row, 1], sanity_curves, task, "val",
                   "#27ae60", "feat-MLP + baseline-norm (sanity)")
        plot_panel(axes[row, 0], v12_curves, task, "train",
                   "#e74c3c", "feat-MLP v12 (no norm)", with_band=False)
        plot_panel(axes[row, 1], v12_curves, task, "val",
                   "#e74c3c", "feat-MLP v12 (no norm)", with_band=False)
        axes[row, 0].set_title(f"{task} — train", fontsize=11)
        axes[row, 1].set_title(f"{task} — val",   fontsize=11)
        for c in (0, 1):
            axes[row, c].set_xlabel("Epoch")
            axes[row, c].set_ylabel(f"{task} loss")
            axes[row, c].grid(linestyle=":", alpha=0.4)
    axes[0, 1].legend(loc="best", fontsize=9)
    fig.suptitle("Sanity vs full v12 — feat-MLP @ 2 s. "
                 "Green = baseline-norm sanity (band = ±1 σ across 7 folds), "
                 "red = v12 mean across 21 fold-runs (no norm).",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    feat_dir = ROOT / "runs" / "sanity_baseline_norm-w2s-multi-feat-mlp"
    tcn_dir  = ROOT / "runs" / "sanity_baseline_norm-w2s-multi-raw-tcn"
    v12_feat_dir = ROOT / "runs" / "optuna_clean_v12eqw-w2s-multi-feat-mlp"

    feat_curves = render_run(
        feat_dir, "feat-MLP + baseline-norm",
        feat_dir / "plots" / "curves_per_task.png", color="#27ae60")
    render_run(
        tcn_dir, "raw-TCN (raw path; norm unaffected)",
        tcn_dir / "plots" / "curves_per_task.png", color="#8c564b")

    if v12_feat_dir.exists():
        v12_curves = collect(v12_feat_dir)
        render_overlay(
            feat_curves, v12_curves,
            feat_dir / "plots" / "curves_per_task_overlay_v12.png")
    else:
        print(f"Skipping overlay — v12 reference not found at {v12_feat_dir}")


if __name__ == "__main__":
    main()
