"""V12 line plots — metric vs window length, one line per architecture, per
task head. Mirrors the existing fatigue_r_vs_window.png but extends to all
four task heads.

Outputs to runs/comparison_v12/:
  exercise_f1_vs_window.png
  phase_f1_vs_window.png
  fatigue_mae_vs_window.png
  reps_mae_vs_window.png
  metric_vs_window_combined.png   (2x2 panel of all four)
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
OUT = ROOT / "runs" / "comparison_v12"
OUT.mkdir(parents=True, exist_ok=True)

WINDOWS = ["1s", "2s", "5s"]
WIN_X = [1.0, 2.0, 5.0]

ARCHS = [
    "multi-feat-mlp", "multi-feat-lstm",
    "multi-raw-cnn1d", "multi-raw-lstm", "multi-raw-cnn_lstm", "multi-raw-tcn",
]
ARCH_LABELS = ["feat-MLP", "feat-LSTM", "raw-cnn1d", "raw-lstm",
               "raw-cnn-lstm", "raw-tcn"]

FATIGUE_ARCHS = ["fatigue-raw-tcn", "fatigue-raw-lstm"]
FATIGUE_ARCH_LABELS = ["fatigue-tcn (single)", "fatigue-lstm (single)"]


def load_results(arch_slugs):
    out = {}
    for slug in arch_slugs:
        out[slug] = {}
        for w in WINDOWS:
            rd = ROOT / "runs" / f"optuna_clean_v12eqw-w{w}-{slug}"
            cv = next(iter((rd / "phase2").rglob("cv_summary.json")), None) if rd.exists() else None
            if cv is None:
                continue
            out[slug][w] = json.loads(cv.read_text())["summary"]
    return out


def get_metric(summary, task, metric):
    """Returns (mean, std) or (nan, nan) if missing/untrained."""
    t = summary.get(task, {}) if summary else {}
    if not isinstance(t, dict) or t.get("untrained"):
        return np.nan, np.nan
    m = t.get(metric, {})
    if not isinstance(m, dict):
        return np.nan, np.nan
    return m.get("mean", np.nan), m.get("std", np.nan)


def plot_metric(ax, results, archs, labels, task, metric, ylabel, title,
                add_zero_line=False, invert_y=False, include_fatigue_only=False):
    """Plot one metric as line-vs-window, one line per arch.

    Single-task fatigue runs are only added if include_fatigue_only=True
    (since they have no exercise/phase/reps heads).
    """
    if include_fatigue_only:
        merged_archs = list(archs) + list(FATIGUE_ARCHS)
        merged_labels = list(labels) + list(FATIGUE_ARCH_LABELS)
    else:
        merged_archs = list(archs)
        merged_labels = list(labels)

    for slug, label in zip(merged_archs, merged_labels):
        means, stds = [], []
        for w in WINDOWS:
            m, s = get_metric(results.get(slug, {}).get(w, {}), task, metric)
            means.append(m); stds.append(s)
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)
        ls = "--" if "fatigue-" in slug else "-"
        ax.errorbar(WIN_X, means, yerr=stds, fmt=f"{ls}o", label=label,
                    linewidth=1.5, capsize=3, alpha=0.85)

    if add_zero_line:
        ax.axhline(0, color="black", linewidth=0.5)
    if invert_y:
        ax.invert_yaxis()
    ax.set_xlabel("Window length (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(WIN_X)
    ax.grid(linestyle=":", alpha=0.5)


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    results = load_results(ARCHS)
    fatigue_results = load_results(FATIGUE_ARCHS)
    merged = {**results, **fatigue_results}

    panels = [
        # (task, metric, ylabel, title, zero_line, include_fatigue_only)
        ("exercise", "f1_macro", "Exercise macro-F1",
         "Exercise F1 vs window length", False, False),
        ("phase", "f1_macro", "Phase macro-F1",
         "Phase F1 vs window length", False, False),
        ("fatigue", "mae", "Fatigue MAE (RPE)",
         "Fatigue MAE vs window length", False, True),
        ("reps", "mae", "Reps MAE (soft-overlap)",
         "Reps MAE vs window length", False, False),
    ]

    # Standalone PNGs
    for task, metric, ylabel, title, zero, include_fat in panels:
        fig, ax = plt.subplots(figsize=(11, 5))
        plot_metric(ax, merged, ARCHS, ARCH_LABELS, task, metric, ylabel, title,
                    add_zero_line=zero, include_fatigue_only=include_fat)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        fig.tight_layout()
        out = OUT / f"{task}_{metric}_vs_window.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")

    # 2x2 combined panel for thesis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (task, metric, ylabel, title, zero, include_fat) in zip(axes.flat, panels):
        plot_metric(ax, merged, ARCHS, ARCH_LABELS, task, metric, ylabel, title,
                    add_zero_line=zero, include_fatigue_only=include_fat)
    # Single legend at top right of figure
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fat_handles, fat_labs = axes[1, 0].get_legend_handles_labels()
    seen = set(labs)
    for h, l in zip(fat_handles, fat_labs):
        if l not in seen:
            handles.append(h); labs.append(l); seen.add(l)
    fig.legend(handles, labs, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("V12 — per-task metric vs window length (mean ± std over 21 fold-seed runs)",
                 fontsize=12, y=0.94)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT / "metric_vs_window_combined.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"\nAll plots in {OUT}")


if __name__ == "__main__":
    main()
