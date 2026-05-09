"""Correlation plot — EMG MNF/RMS slope vs self-reported RPE across all sets.

Aggregates JASA points from every recording in data/labeled/ and produces a
2-panel scatter:

    Panel 1: x = RPE, y = MNF slope (%/s).  Expected fatigue direction = NEGATIVE.
    Panel 2: x = RPE, y = RMS slope (%/s).  Expected fatigue direction = POSITIVE.

Each set is one point. Color = exercise. Regression line + Pearson/Spearman
correlation statistics are annotated. The expected-fatigue-direction half-plane
is lightly shaded (red below zero on MNF, red above zero on RMS) so a viewer
can immediately see whether the cloud sits in the "fatigue" region.

Optional --rpe-jitter spreads points sideways within each integer RPE for
readability since RPE is discrete and many sets share the same value.

Usage:
    python scripts/plot_rpe_correlation.py
    python scripts/plot_rpe_correlation.py --rpe-jitter 0.18
    python scripts/plot_rpe_correlation.py --clip-pct 90
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.plot_style import apply_style, despine
from scripts.plot_set_emg_acc_rms import EXERCISE_COLORS, load_window_features

apply_style()


def _set_slope_pct(set_rows: pd.DataFrame, col: str) -> float:
    t = set_rows["t_session_s"].to_numpy()
    y = set_rows[col].to_numpy()
    m = ~np.isnan(y)
    if m.sum() < 3:
        return float("nan")
    slope, intercept = np.polyfit(t[m], y[m], 1)
    y_start = intercept + slope * t[m][0]
    return 100.0 * slope / y_start if y_start else float("nan")


def _gather_summary() -> pd.DataFrame:
    rows = []
    for rec_dir in sorted(Path("data/labeled").glob("recording_*")):
        rid = rec_dir.name.split("_")[-1]
        try:
            df = load_window_features(rid)
        except FileNotFoundError:
            continue
        active = df[df["in_active_set"] == True]  # noqa: E712
        for set_n in sorted(int(n) for n in active["set_number"].dropna().unique()):
            s = active[active["set_number"] == float(set_n)]
            rows.append({
                "recording_id": rid,
                "set_number": set_n,
                "exercise": s["exercise"].iloc[0],
                "rpe": int(s["rpe_for_this_set"].iloc[0]),
                "mnf_slope_pct": _set_slope_pct(s, "emg_mnf"),
                "rms_slope_pct": _set_slope_pct(s, "emg_rms"),
            })
    return pd.DataFrame(rows).dropna(subset=["mnf_slope_pct", "rms_slope_pct"])


def _draw_panel(ax, df: pd.DataFrame, y_col: str, ylabel: str,
                expected_dir: str, jitter: float, y_clip: float) -> None:
    """expected_dir: 'down' (FATIGUE = y<0) or 'up' (FATIGUE = y>0)."""
    rng = np.random.default_rng(42)
    x = df["rpe"].to_numpy(dtype=float)
    if jitter > 0:
        x = x + rng.uniform(-jitter, jitter, size=len(x))
    y = df[y_col].to_numpy()

    # Shade the expected-fatigue half-plane
    xmin, xmax = float(min(x)) - 0.6, float(max(x)) + 0.6
    if expected_dir == "down":
        ax.axhspan(-y_clip, 0, xmin=0, xmax=1, color="#E45756", alpha=0.07, zorder=0)
        ax.text(xmax - 0.05, -y_clip * 0.92,
                "expected fatigue direction\n(MNF slope < 0)",
                ha="right", va="bottom", fontsize=9, color="#B71C1C", style="italic")
    else:
        ax.axhspan(0, y_clip, xmin=0, xmax=1, color="#E45756", alpha=0.07, zorder=0)
        ax.text(xmax - 0.05, y_clip * 0.92,
                "expected fatigue direction\n(RMS slope > 0)",
                ha="right", va="top", fontsize=9, color="#B71C1C", style="italic")

    ax.axhline(0, color="#222", linewidth=0.8, zorder=1)

    # Per-exercise scatter
    for ex in ["squat", "deadlift", "benchpress", "pullup"]:
        m = (df["exercise"] == ex).to_numpy()
        if not m.any():
            continue
        ax.scatter(x[m], y[m], s=42, color=EXERCISE_COLORS[ex],
                   edgecolor="black", linewidth=0.5, alpha=0.75,
                   label=f"{ex} (n={m.sum()})", zorder=3)

    # Overall linear regression line (on un-jittered RPE)
    rpe_int = df["rpe"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(rpe_int, df[y_col].to_numpy(), 1)
    xs = np.linspace(rpe_int.min() - 0.3, rpe_int.max() + 0.3, 100)
    ax.plot(xs, slope * xs + intercept, color="#222", linewidth=1.6,
            linestyle="--", zorder=4,
            label=f"OLS fit (slope={slope:+.3f})")

    # Stats
    r_p, p_p = pearsonr(rpe_int, df[y_col])
    r_s, p_s = spearmanr(rpe_int, df[y_col])
    expected = "negative" if expected_dir == "down" else "positive"
    matches = (r_p < 0 if expected_dir == "down" else r_p > 0)
    box_color = "#2E7D32" if matches else "#B71C1C"
    txt = (f"Pearson r = {r_p:+.3f}  (p={p_p:.3f})\n"
           f"Spearman ρ = {r_s:+.3f}  (p={p_s:.3f})\n"
           f"expected sign: {expected}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=10, fontweight="bold",
            color=box_color,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=box_color, linewidth=1.2, alpha=0.92))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-y_clip, y_clip)
    ax.set_xlabel("Self-reported RPE (1–10)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(df["rpe"].unique()))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.3)


def plot_rpe_correlation(jitter: float, clip_pct: float, out_path: Path) -> Path:
    df = _gather_summary()

    y_clip_mnf = max(float(np.percentile(df["mnf_slope_pct"].abs(), clip_pct)) * 1.2, 0.5)
    y_clip_rms = max(float(np.percentile(df["rms_slope_pct"].abs(), clip_pct)) * 1.2, 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    _draw_panel(axes[0], df, "mnf_slope_pct",
                ylabel="EMG MNF slope (%/s)  —  drop = spectral compression",
                expected_dir="down", jitter=jitter, y_clip=y_clip_mnf)
    _draw_panel(axes[1], df, "rms_slope_pct",
                ylabel="EMG RMS slope (%/s)  —  rise = motor unit recruitment",
                expected_dir="up", jitter=jitter, y_clip=y_clip_rms)

    n_recs = df["recording_id"].nunique()
    fig.suptitle(
        f"EMG fatigue slopes vs self-reported RPE  —  all recordings "
        f"(n={n_recs}, {len(df)} sets)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    despine(fig=fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")

    print(f"\nSummary stats (n={len(df)}):")
    for col, expected in [("mnf_slope_pct", "negative"), ("rms_slope_pct", "positive")]:
        rp, pp = pearsonr(df["rpe"], df[col])
        rs, ps = spearmanr(df["rpe"], df[col])
        print(f"  {col:>16}  Pearson r={rp:+.3f} (p={pp:.3f})  "
              f"Spearman rho={rs:+.3f} (p={ps:.3f})  expected: {expected}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rpe-jitter", type=float, default=0.18,
                        help="horizontal jitter on integer RPE (default 0.18)")
    parser.add_argument("--clip-pct", type=float, default=90.0,
                        help="y-axis clip percentile of |slope| (default 90)")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "inspections/rpe_emg_correlation.png")
    args = parser.parse_args()
    plot_rpe_correlation(args.rpe_jitter, args.clip_pct, args.out)


if __name__ == "__main__":
    main()
