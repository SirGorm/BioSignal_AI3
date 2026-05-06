"""V13 single-task learning curves — same idea as plot_v12_all_curves.py
(per-task train + val loss curves, mean across 21 fold-runs) but for the
single-task v13 runs.

Layout reflects what was actually trained in v13:
  exercise-only: archs={feat-mlp, raw-tcn},   windows={2s, 5s}
  phase-only:    archs={feat-mlp, feat-lstm}, windows={1s, 2s}
  reps-only:     archs={feat-mlp, feat-lstm}, windows={1s, 2s}

Outputs to runs/comparison_v13/:
  curves_singletask_per_task.png  — 3 rows (exercise/phase/reps) × 2 cols (train/val)
  curves_singletask_<task>.png    — one per task (train + val side by side)
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
OUT = ROOT / "runs" / "comparison_v13"
OUT.mkdir(parents=True, exist_ok=True)

# Same color palette as plot_v12_all_curves.py for cross-figure consistency.
ARCH_INFO = {
    "feat-mlp":  ("feat-MLP",  "#1f77b4"),
    "feat-lstm": ("feat-LSTM", "#ff7f0e"),
    "raw-tcn":   ("raw-TCN",   "#8c564b"),
}
LINESTYLES = {"1s": "-", "2s": "--", "5s": ":"}

TASK_GRIDS = {
    "exercise": {
        "archs": ["feat-mlp", "raw-tcn"],
        "windows": ["2s", "5s"],
        "slug_fmt": "optuna_clean_v13single-exercise-only-w{win}-{arch}",
    },
    "phase": {
        "archs": ["feat-mlp", "feat-lstm"],
        "windows": ["1s", "2s"],
        "slug_fmt": "optuna_clean_v13single-phase-only-w{win}-{arch}",
    },
    "reps": {
        "archs": ["feat-mlp", "feat-lstm"],
        "windows": ["1s", "2s"],
        "slug_fmt": "optuna_clean_v13single-reps-only-w{win}-{arch}",
    },
}


def collect_run_curves(run_dir: Path, task: str):
    """Mean train/val loss per epoch across all (seed, fold) histories."""
    p2 = run_dir / "phase2"
    if not p2.exists():
        return None
    by_epoch = {"train": defaultdict(list), "val": defaultdict(list)}
    n = 0
    for hp in p2.rglob("history.json"):
        try:
            hist = json.loads(hp.read_text())
        except Exception:
            continue
        n += 1
        for entry in hist:
            ep = entry["epoch"]
            tr = entry.get("train", {}).get(task)
            if tr is not None:
                by_epoch["train"][ep].append(tr)
            vl = entry.get("val_loss", {}).get(task)
            if vl is not None:
                by_epoch["val"][ep].append(vl)
    if n == 0:
        return None
    out = {}
    for stat in ("train", "val"):
        d = by_epoch[stat]
        if not d:
            out[stat] = (np.array([]), np.array([]))
            continue
        eps = sorted(d.keys())
        means = np.array([float(np.mean(d[e])) for e in eps])
        out[stat] = (np.array(eps), means)
    return out


def load_task_curves(task: str):
    """Return dict[(arch, window)] = {'train': (x,y), 'val': (x,y)}."""
    g = TASK_GRIDS[task]
    out = {}
    for arch in g["archs"]:
        for w in g["windows"]:
            slug = g["slug_fmt"].format(win=w, arch=arch)
            run_dir = ROOT / "runs" / slug
            curves = collect_run_curves(run_dir, task)
            if curves is not None:
                out[(arch, w)] = curves
    return out


def plot_panel(ax, task_curves, task: str, stat: str, title: str):
    for (arch, w), curves in task_curves.items():
        label, color = ARCH_INFO[arch]
        x, y = curves[stat]
        if not len(x):
            continue
        ax.plot(x, y, color=color, linestyle=LINESTYLES[w], linewidth=1.6, alpha=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{task} {stat} loss")
    ax.set_title(title, fontsize=11)
    ax.grid(linestyle=":", alpha=0.4)


def add_legend(ax, task: str):
    g = TASK_GRIDS[task]
    arch_handles = [plt.Line2D([], [], color=ARCH_INFO[a][1], linewidth=2,
                                label=ARCH_INFO[a][0]) for a in g["archs"]]
    win_handles = [plt.Line2D([], [], color="black", linewidth=2,
                                linestyle=LINESTYLES[w], label=f"{w} window")
                    for w in g["windows"]]
    leg1 = ax.legend(handles=arch_handles, loc="upper left",
                     bbox_to_anchor=(1.01, 1.0), fontsize=9, title="Architecture")
    ax.add_artist(leg1)
    ax.legend(handles=win_handles, loc="upper left",
              bbox_to_anchor=(1.01, 0.4), fontsize=9, title="Window")


def render_single_task(task: str, task_curves: dict):
    n_runs = len(task_curves)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    plot_panel(axes[0], task_curves, task, "train", f"{task} — train")
    plot_panel(axes[1], task_curves, task, "val", f"{task} — val")
    add_legend(axes[1], task)
    fig.suptitle(f"Single-task {task} ({n_runs} runs) — loss per epoch "
                 "(mean across 21 fold-runs)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.86, 0.94))
    out = OUT / f"curves_singletask_{task}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def render_combined(all_task_curves: dict):
    tasks = list(TASK_GRIDS.keys())
    fig, axes = plt.subplots(len(tasks), 2, figsize=(15, 13))
    for row, task in enumerate(tasks):
        plot_panel(axes[row, 0], all_task_curves[task], task, "train",
                   f"{task} — train")
        plot_panel(axes[row, 1], all_task_curves[task], task, "val",
                   f"{task} — val")
    # Single combined legend on top-right panel — use the union of archs
    union_archs = []
    for task in tasks:
        for a in TASK_GRIDS[task]["archs"]:
            if a not in union_archs:
                union_archs.append(a)
    arch_handles = [plt.Line2D([], [], color=ARCH_INFO[a][1], linewidth=2,
                                label=ARCH_INFO[a][0]) for a in union_archs]
    win_handles = [plt.Line2D([], [], color="black", linewidth=2,
                                linestyle=LINESTYLES[w], label=f"{w} window")
                    for w in ["1s", "2s", "5s"]]
    leg1 = axes[0, 1].legend(handles=arch_handles, loc="upper left",
                             bbox_to_anchor=(1.01, 1.0), fontsize=9, title="Architecture")
    axes[0, 1].add_artist(leg1)
    axes[0, 1].legend(handles=win_handles, loc="upper left",
                      bbox_to_anchor=(1.01, 0.4), fontsize=9, title="Window")
    n_runs = sum(len(c) for c in all_task_curves.values())
    fig.suptitle(f"Single-task ({n_runs} runs) — per-task loss per epoch "
                 "(mean across 21 fold-runs)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.88, 0.97))
    out = OUT / "curves_singletask_per_task.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    all_task_curves = {}
    for task in TASK_GRIDS:
        all_task_curves[task] = load_task_curves(task)
        render_single_task(task, all_task_curves[task])
    render_combined(all_task_curves)


if __name__ == "__main__":
    main()
