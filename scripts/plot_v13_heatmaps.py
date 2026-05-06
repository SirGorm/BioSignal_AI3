"""V13 single-task heatmaps — same look-and-feel as plot_v12_comparison.py
heatmaps but with the per-task arch/window grid that was actually trained
in v13. v13 is single-task only, so each task gets its own matrix.

Trained combos (verified by globbing phase2/*/cv_summary.json):
  exercise: archs={feat-mlp, raw-tcn},   windows={2s, 5s}
  phase:    archs={feat-mlp, feat-lstm}, windows={1s, 2s}
  reps:     archs={feat-mlp, feat-lstm}, windows={1s, 2s}

Outputs to runs/comparison_v13/:
  heatmap_exercise_f1.png, heatmap_exercise_balacc.png
  heatmap_phase_f1.png,    heatmap_phase_balacc.png
  heatmap_reps_mae.png
  heatmap_v13_combined.png  (3-panel summary: exercise F1, phase F1, reps MAE)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "comparison_v13"
OUT.mkdir(parents=True, exist_ok=True)

TASK_GRIDS = {
    "exercise": {
        "archs": ["feat-mlp", "raw-tcn"],
        "arch_labels": ["feat-MLP", "raw-TCN"],
        "windows": ["2s", "5s"],
        "win_labels": ["2.0", "5.0"],
        "slug_fmt": "optuna_clean_v13single-exercise-only-w{win}-{arch}",
    },
    "phase": {
        "archs": ["feat-mlp", "feat-lstm"],
        "arch_labels": ["feat-MLP", "feat-LSTM"],
        "windows": ["1s", "2s"],
        "win_labels": ["1.0", "2.0"],
        "slug_fmt": "optuna_clean_v13single-phase-only-w{win}-{arch}",
    },
    "reps": {
        "archs": ["feat-mlp", "feat-lstm"],
        "arch_labels": ["feat-MLP", "feat-LSTM"],
        "windows": ["1s", "2s"],
        "win_labels": ["1.0", "2.0"],
        "slug_fmt": "optuna_clean_v13single-reps-only-w{win}-{arch}",
    },
}


def load_cv_summary(run_dir: Path):
    cv = next(iter((run_dir / "phase2").rglob("cv_summary.json")), None) if run_dir.exists() else None
    if cv is None:
        return None
    return json.loads(cv.read_text())["summary"]


def build_matrix(task: str, metric_path: tuple[str, ...]):
    """Return (mean_matrix, std_matrix) shaped (n_archs, n_windows)."""
    g = TASK_GRIDS[task]
    n_a, n_w = len(g["archs"]), len(g["windows"])
    means = np.full((n_a, n_w), np.nan)
    stds = np.full((n_a, n_w), np.nan)
    for i, arch in enumerate(g["archs"]):
        for j, win in enumerate(g["windows"]):
            slug = g["slug_fmt"].format(win=win, arch=arch)
            summary = load_cv_summary(ROOT / "runs" / slug)
            if summary is None:
                continue
            cur = summary
            for k in (task, *metric_path):
                if not isinstance(cur, dict) or k not in cur:
                    cur = None
                    break
                cur = cur[k]
            if not isinstance(cur, dict):
                continue
            means[i, j] = cur.get("mean", np.nan)
            stds[i, j] = cur.get("std", np.nan)
    return means, stds


def heatmap(ax, matrix, row_labels, col_labels, title, value_fmt="{:.3f}",
            cmap="viridis", highlight_best="max", std_matrix=None):
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Window (s)")
    if not np.all(np.isnan(matrix)):
        if highlight_best == "max":
            best_idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)
        else:
            best_idx = np.unravel_index(np.nanargmin(matrix), matrix.shape)
        ax.add_patch(plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                                   fill=False, edgecolor="gold", linewidth=3))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", color="grey")
                continue
            txt = value_fmt.format(v)
            if std_matrix is not None and not np.isnan(std_matrix[i, j]):
                txt = f"{txt}\n± {std_matrix[i, j]:.3f}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9)
    plt.colorbar(im, ax=ax)


def write_single(task, metric_label, metric_path, value_fmt, cmap, best, fname):
    means, stds = build_matrix(task, metric_path)
    g = TASK_GRIDS[task]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    heatmap(ax, means, g["arch_labels"], g["win_labels"],
            f"{task.capitalize()} — {metric_label} (single-task)",
            value_fmt, cmap, best, std_matrix=stds)
    fig.tight_layout()
    out = OUT / fname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return means, stds


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    ex_f1, ex_f1_s = write_single(
        "exercise", "macro-F1", ("f1_macro",), "{:.3f}", "Blues", "max",
        "heatmap_exercise_f1.png")
    write_single(
        "exercise", "balanced accuracy", ("balanced_accuracy",),
        "{:.3f}", "Blues", "max", "heatmap_exercise_balacc.png")

    ph_f1, ph_f1_s = write_single(
        "phase", "macro-F1", ("f1_macro",), "{:.3f}", "Greens", "max",
        "heatmap_phase_f1.png")
    write_single(
        "phase", "balanced accuracy", ("balanced_accuracy",),
        "{:.3f}", "Greens", "max", "heatmap_phase_balacc.png")

    rp_mae, rp_mae_s = write_single(
        "reps", "MAE", ("mae",), "{:.3f}", "YlOrRd_r", "min",
        "heatmap_reps_mae.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    panels = [
        (axes[0], ex_f1, ex_f1_s, "exercise", "Exercise macro-F1", "{:.3f}", "Blues", "max"),
        (axes[1], ph_f1, ph_f1_s, "phase",    "Phase macro-F1",    "{:.3f}", "Greens", "max"),
        (axes[2], rp_mae, rp_mae_s, "reps",   "Reps MAE",          "{:.3f}", "YlOrRd_r", "min"),
    ]
    for ax, mat, std, task, title, fmt, cmap, best in panels:
        g = TASK_GRIDS[task]
        heatmap(ax, mat, g["arch_labels"], g["win_labels"], title, fmt, cmap, best, std_matrix=std)
    fig.suptitle("Single-task — arch × window per task (3 seeds × 7 folds)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "heatmap_v13_combined.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"\nAll heatmaps in {OUT}")


if __name__ == "__main__":
    main()
