"""Generate per-task comparison bars across all 7 v4 models.

Output:
  runs/comparison_v4/comparison_all_tasks.png   — 4-panel (one per task)
  runs/comparison_v4/per_task_<task>.png        — individual task PNGs
  runs/comparison_v4/per_subject_fatigue.png    — per-fold fatigue r
  runs/comparison_v4/v4_vs_v3_delta.png         — delta from v3
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
OUT = ROOT / "runs" / "comparison_v4"
OUT.mkdir(parents=True, exist_ok=True)

NN_MODELS = {
    "feat-MLP":     ROOT / "runs" / "optuna_clean_v4-features-mlp",
    "feat-LSTM":    ROOT / "runs" / "optuna_clean_v4-features-lstm",
    "raw-cnn1d":    ROOT / "runs" / "optuna_clean_v4-raw-cnn1d_raw",
    "raw-lstm":     ROOT / "runs" / "optuna_clean_v4-raw-lstm_raw",
    "raw-cnn_lstm": ROOT / "runs" / "optuna_clean_v4-raw-cnn_lstm_raw",
    "raw-tcn":      ROOT / "runs" / "optuna_clean_v4-raw-tcn_raw",
}
RF_PATH = ROOT / "runs" / "optuna_clean_v4-rf" / "metrics.json"

PALETTE = {
    "RF":           "#2c3e50",
    "feat-MLP":     "#27ae60",
    "feat-LSTM":    "#16a085",
    "raw-cnn1d":    "#3498db",
    "raw-lstm":     "#9b59b6",
    "raw-cnn_lstm": "#e67e22",
    "raw-tcn":      "#e74c3c",
}

BASELINE = {
    "exercise_f1": 0.123,
    "phase_f1":    0.186,
    "fatigue_mae": 1.013,
    "reps_mae":    1.291,
}


def load_results():
    """Return dict[model_name] -> dict[task_metric] = (mean, std)."""
    out = {}
    rf = json.loads(RF_PATH.read_text())
    out["RF"] = {
        "exercise_f1":  (rf["exercise"]["f1_mean"], rf["exercise"]["f1_std"]),
        "phase_f1":     (rf["phase"]["ml_f1_mean"], rf["phase"]["ml_f1_std"]),
        "fatigue_mae":  (rf["fatigue"]["mae_mean"], rf["fatigue"]["mae_std"]),
        "fatigue_r":    (rf["fatigue"]["pearson_r_median"], 0.0),
        "reps_mae":     (rf["reps"]["ml_mae_mean"], rf["reps"]["ml_mae_std"]),
    }
    for name, rd in NN_MODELS.items():
        cv = next(iter((rd / "phase2").rglob("cv_summary.json")), None)
        if cv is None:
            continue
        s = json.loads(cv.read_text())["summary"]
        out[name] = {
            "exercise_f1":  (s["exercise"]["f1_macro"]["mean"],     s["exercise"]["f1_macro"]["std"]),
            "phase_f1":     (s["phase"]["f1_macro"]["mean"],        s["phase"]["f1_macro"]["std"]),
            "fatigue_mae":  (s["fatigue"]["mae"]["mean"],           s["fatigue"]["mae"]["std"]),
            "fatigue_r":    (s["fatigue"]["pearson_r"]["mean"],     s["fatigue"]["pearson_r"]["std"]),
            "reps_mae":     (s["reps"]["mae"]["mean"],              s["reps"]["mae"]["std"]),
        }
    return out


def plot_task(ax, results, key, title, ylabel, baseline=None,
              higher_is_better=True, sort_desc=None):
    if sort_desc is None:
        sort_desc = higher_is_better
    items = [(name, *results[name][key]) for name in results]
    items.sort(key=lambda t: t[1], reverse=sort_desc)
    names = [t[0] for t in items]
    means = [t[1] for t in items]
    stds = [t[2] for t in items]
    colors = [PALETTE.get(n, "#7f8c8d") for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4,
                   edgecolor="black", linewidth=0.5, alpha=0.9)
    if baseline is not None:
        ax.axhline(baseline, color="grey", linestyle="--", linewidth=1, label=f"baseline = {baseline:.3f}")
        ax.legend(loc="lower right" if higher_is_better else "upper right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    # Highlight winner
    winner_idx = 0  # already sorted by sort_desc to put the best first
    bars[winner_idx].set_edgecolor("gold")
    bars[winner_idx].set_linewidth(2.5)
    # Annotate values
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + (s if s > 0 else 0) * 1.05, f"{m:.3f}",
                ha="center", va="bottom", fontsize=8)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    results = load_results()

    # 1. Combined 2x2 panel
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_task(axes[0, 0], results, "exercise_f1",
              "Exercise classification (F1-macro, n=15)", "F1-macro ↑",
              baseline=BASELINE["exercise_f1"], higher_is_better=True)
    plot_task(axes[0, 1], results, "phase_f1",
              "Phase classification (F1-macro, n=15)", "F1-macro ↑",
              baseline=BASELINE["phase_f1"], higher_is_better=True)
    plot_task(axes[1, 0], results, "fatigue_mae",
              "Fatigue (RPE) — MAE", "MAE (lower=better) ↓",
              baseline=BASELINE["fatigue_mae"], higher_is_better=False)
    plot_task(axes[1, 1], results, "fatigue_r",
              "Fatigue (RPE) — Pearson r", "Pearson r ↑",
              baseline=0.0, higher_is_better=True)
    fig.suptitle("Strength-RT v4 — model comparison (3 seeds × 5 folds, 7 subjects)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = OUT / "comparison_all_tasks.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}")

    # 2. Reps in its own figure (RF dominates by 2x)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_task(ax, results, "reps_mae",
              "Rep counting — MAE (n=15 for NN, n=5 for RF)", "MAE ↓",
              baseline=BASELINE["reps_mae"], higher_is_better=False)
    fig.tight_layout()
    p = OUT / "per_task_reps.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}")

    # 3. Individual task figures
    for key, label, baseline, hib in [
        ("exercise_f1", "Exercise F1-macro", BASELINE["exercise_f1"], True),
        ("phase_f1",    "Phase F1-macro",    BASELINE["phase_f1"],    True),
        ("fatigue_mae", "Fatigue MAE",       BASELINE["fatigue_mae"], False),
        ("fatigue_r",   "Fatigue Pearson r", 0.0,                      True),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_task(ax, results, key, label, label, baseline=baseline, higher_is_better=hib)
        fig.tight_layout()
        p = OUT / f"per_task_{key}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {p}")

    # 4. Per-fold fatigue Pearson r for top-3 fatigue models
    fig, ax = plt.subplots(figsize=(11, 5))
    folds_x = np.arange(5)
    fold_subjects = ["Juile, lucas2", "Hytten, kiyomi", "Vivian", "Tias", "Raghild"]
    width = 0.22
    for i, name in enumerate(["feat-MLP", "raw-cnn1d", "raw-tcn"]):
        rd = NN_MODELS[name] if name in NN_MODELS else None
        if rd is None:
            continue
        cv = json.loads(next(iter((rd / "phase2").rglob("cv_summary.json"))).read_text())
        from collections import defaultdict
        fold_r = defaultdict(list)
        for r in cv.get("all_results", []):
            fk = r.get("fold")
            if "metrics" in r and "fatigue" in r["metrics"]:
                fold_r[fk].append(r["metrics"]["fatigue"].get("pearson_r", 0))
        means = [np.mean(fold_r[f]) if fold_r[f] else 0 for f in range(5)]
        ax.bar(folds_x + (i - 1) * width, means, width=width, label=name,
                color=PALETTE.get(name, "#7f8c8d"), edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(folds_x)
    ax.set_xticklabels([f"Fold {f}\n({s})" for f, s in enumerate(fold_subjects)], fontsize=9)
    ax.set_ylabel("Pearson r per fold (3-seed mean)")
    ax.set_title("Fatigue Pearson r per fold — top-3 NN models")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    p = OUT / "per_fold_fatigue_r.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}")

    # 5. V3 → V4 delta plot
    v3_results = {}
    for name, p in [
        ("feat-MLP",     "runs/optuna_clean_v3-features-mlp"),
        ("feat-LSTM",    "runs/optuna_clean_v3-features-lstm"),
        ("raw-cnn1d",    "runs/optuna_clean_v3-raw-cnn1d_raw"),
        ("raw-lstm",     "runs/optuna_clean_v3-raw-lstm_raw"),
        ("raw-cnn_lstm", "runs/optuna_clean_v3-raw-cnn_lstm_raw"),
        ("raw-tcn",      "runs/optuna_clean_v3-raw-tcn_raw"),
    ]:
        cv = next(iter((Path(p) / "phase2").rglob("cv_summary.json")), None)
        if cv is None:
            continue
        s = json.loads(cv.read_text())["summary"]
        v3_results[name] = {
            "exercise_f1": s["exercise"]["f1_macro"]["mean"],
            "phase_f1":    s["phase"]["f1_macro"]["mean"],
            "fatigue_mae": s["fatigue"]["mae"]["mean"],
            "fatigue_r":   s["fatigue"]["pearson_r"]["mean"],
        }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = [("exercise_f1", "Exercise F1-macro Δ", True),
                ("phase_f1",    "Phase F1-macro Δ",    True),
                ("fatigue_mae", "Fatigue MAE Δ (negative=better)", False),
                ("fatigue_r",   "Fatigue Pearson r Δ", True)]
    for ax, (key, title, hib) in zip(axes.flat, metrics):
        names = list(v3_results.keys())
        deltas = [results[n][key][0] - v3_results[n][key] for n in names]
        colors = ["#27ae60" if (d > 0) == hib else "#e74c3c" for d in deltas]
        x = np.arange(len(names))
        ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        for i, d in enumerate(deltas):
            ax.text(i, d + (0.005 if d > 0 else -0.005),
                    f"{d:+.3f}", ha="center",
                    va="bottom" if d > 0 else "top", fontsize=8)
    fig.suptitle("V3 → V4 metric delta (100 trials, 200 epochs, patience=10)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = OUT / "v4_vs_v3_delta.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}")

    print(f"\nAll plots in {OUT}")


if __name__ == "__main__":
    main()
