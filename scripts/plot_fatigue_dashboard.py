"""Compact fatigue dashboard — one or all sets of an exercise.

Three stacked panels per set, sharing time axis:
  1. EMG MNF (Hz)         — drops with localized muscle fatigue (De Luca 1984; Cifrek 2009)
  2. EMG MDF (Hz)          — drops alongside MNF; less sensitive to noise (Cifrek 2009)
  3. Acc RMS (g)          — drops as bar/limb velocity drops near failure (González-Badillo 2010)

Each panel: raw + light smoothing, linear trend over the active set, and a
Δ% box from start to end of the set. Rep markers shown as vertical dashed lines.

Single set:
    python scripts/plot_fatigue_dashboard.py --recording 014 --set 1

All sets of one exercise (fatigue progression across sets):
    python scripts/plot_fatigue_dashboard.py --recording 014 --exercise pullup

JASA scatter (Luttmann et al. 1996) — all sets in one figure, classifies each set into
fatigue / recovery / force↑ / force↓ quadrants based on MNF-slope vs RMS-slope:
    python scripts/plot_fatigue_dashboard.py --recording 014 --jasa

Add --rel to use baseline-normalized EMG (emg_mnf_rel, emg_rms_rel).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.plot_style import apply_style, despine
from scripts.plot_set_emg_acc_rms import (
    EXERCISE_COLORS,
    _draw_trend,
    load_rep_times,
    load_window_features,
)

apply_style()


def _smooth(y: np.ndarray, window: int = 11) -> np.ndarray:
    return pd.Series(y).rolling(window=window, min_periods=1, center=True).median().to_numpy()


def _metric_specs(use_rel: bool) -> list[dict]:
    """Three fatigue panels. EMG RMS dropped — MNF/MDF carry the fatigue signal."""
    return [
        {
            "col": "emg_mnf_rel" if use_rel else "emg_mnf",
            "label": "EMG MNF" + (" (rel)" if use_rel else " (Hz)"),
            "color": "#1F4E79",
            "unit": "" if use_rel else "Hz",
            "expect_dir": "down",  # drop = fatigue
        },
        {
            "col": "emg_mdf_rel" if use_rel else "emg_mdf",
            "label": "EMG MDF" + (" (rel)" if use_rel else " (Hz)"),
            "color": "#7B3F99",
            "unit": "" if use_rel else "Hz",
            "expect_dir": "down",  # drop = fatigue
        },
        {
            "col": "acc_rms",
            "label": "Acc RMS (g)",
            "color": "#54A24B",
            "unit": "g",
            "expect_dir": "down",  # drop = velocity loss
        },
    ]


def _draw_panel(ax, t_full, y_full, t_active, y_active, spec) -> dict:
    ax.plot(t_full, y_full, color=spec["color"], linewidth=0.8, alpha=0.35)
    ax.plot(t_full, _smooth(y_full, window=11), color=spec["color"],
            linewidth=1.5, alpha=0.95)

    mask = ~np.isnan(y_active)
    fit = {"slope": np.nan, "pct": np.nan}
    if mask.sum() >= 3:
        ta, ya = t_active[mask], y_active[mask]
        slope, intercept = np.polyfit(ta, ya, 1)
        y_start = intercept + slope * ta[0]
        y_end = intercept + slope * ta[-1]
        pct = 100.0 * (y_end - y_start) / y_start if y_start else np.nan
        matches = (
            (spec["expect_dir"] == "down" and slope < 0) or
            (spec["expect_dir"] == "up" and slope > 0)
        )
        trend_color = "#2E7D32" if matches else "#B71C1C"
        ax.plot([ta[0], ta[-1]], [y_start, y_end],
                color=trend_color, linewidth=2.0, linestyle="--", zorder=4)
        ax.annotate(
            f"Δ {pct:+.1f}%",
            xy=(0.98, 0.94), xycoords="axes fraction",
            ha="right", va="top", fontsize=16, fontweight="bold",
            color=trend_color,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=trend_color, linewidth=1.2, alpha=0.92),
        )
        fit = {"slope": slope, "pct": pct}

    ax.tick_params(axis="y", labelsize=11)
    ax.grid(False)
    return fit


def _set_window(df: pd.DataFrame, set_number: int, pad_s: float):
    """Returns (set_rows, win_rows, t_start, t_end, exercise, rpe)."""
    set_rows = df[df["set_number"] == float(set_number)]
    if set_rows.empty:
        avail = sorted(int(n) for n in df["set_number"].dropna().unique())
        raise ValueError(f"set {set_number} not found (available: {avail})")
    t_start = float(set_rows["t_session_s"].min())
    t_end = float(set_rows["t_session_s"].max())
    win = df[(df["t_session_s"] >= t_start - pad_s)
             & (df["t_session_s"] <= t_end + pad_s)]
    return (
        set_rows, win, t_start, t_end,
        str(set_rows["exercise"].iloc[0]),
        int(set_rows["rpe_for_this_set"].iloc[0]),
    )


def plot_dashboard_set(recording_id: str, set_number: int,
                       pad_s: float = 2.0, use_rel: bool = False) -> Path:
    df = load_window_features(recording_id)
    set_rows, win, t_start, t_end, exercise, _ = _set_window(df, set_number, pad_s)
    specs = _metric_specs(use_rel)

    n_panels = len(specs)
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 2.5 * n_panels + 0.5),
                             sharex=True, gridspec_kw={"hspace": 0.10})

    rep_times = load_rep_times(recording_id, set_number)
    fits = []
    for i, (ax, spec) in enumerate(zip(axes, specs)):
        ax.axvspan(t_start, t_end, color="#4C78A8", alpha=0.10, zorder=0)
        for rt in rep_times:
            ax.axvline(rt, color="#888", linestyle="--", linewidth=0.7, alpha=0.55)
        fit = _draw_panel(
            ax,
            t_full=win["t_session_s"].to_numpy(),
            y_full=win[spec["col"]].to_numpy(),
            t_active=set_rows["t_session_s"].to_numpy(),
            y_active=set_rows[spec["col"]].to_numpy(),
            spec=spec,
        )
        ax.set_ylabel(spec["label"], fontsize=14, fontweight="bold",
                      color=spec["color"])
        if i == len(axes) - 1:
            ax.set_xlabel("Tid (s)", fontsize=14)
            ax.tick_params(axis="x", labelsize=12)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        fits.append(fit)

    fig.suptitle(f"Recording {recording_id} — {exercise}", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    despine(fig=fig)

    out_dir = Path(f"inspections/recording_{recording_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_rel" if use_rel else ""
    out_path = out_dir / f"set{set_number:02d}_fatigue_dashboard{suffix}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_dashboard_exercise(recording_id: str, exercise: str,
                            pad_s: float = 2.0, use_rel: bool = False) -> Path:
    df = load_window_features(recording_id)
    ex_rows = df[df["exercise"] == exercise]
    if ex_rows.empty:
        avail = sorted(df["exercise"].dropna().unique().tolist())
        raise ValueError(f"exercise {exercise!r} not found (available: {avail})")

    set_numbers = sorted(int(n) for n in ex_rows["set_number"].dropna().unique())
    n = len(set_numbers)
    specs = _metric_specs(use_rel)

    n_rows = len(specs)
    fig, axes = plt.subplots(n_rows, n, figsize=(4.6 * n, 2.4 * n_rows + 1.2),
                             sharey="row",
                             gridspec_kw={"hspace": 0.10, "wspace": 0.06})
    if n == 1:
        axes = axes.reshape(n_rows, 1)

    for col_idx, set_n in enumerate(set_numbers):
        set_rows, win, t_start, t_end, _, rpe = _set_window(df, set_n, pad_s)
        rep_times = load_rep_times(recording_id, set_n)

        for row_idx, spec in enumerate(specs):
            ax = axes[row_idx, col_idx]
            ax.axvspan(t_start, t_end, color="#4C78A8", alpha=0.10, zorder=0)
            for rt in rep_times:
                ax.axvline(rt, color="#888", linestyle="--", linewidth=0.6, alpha=0.5)
            _draw_panel(
                ax,
                t_full=win["t_session_s"].to_numpy(),
                y_full=win[spec["col"]].to_numpy(),
                t_active=set_rows["t_session_s"].to_numpy(),
                y_active=set_rows[spec["col"]].to_numpy(),
                spec=spec,
            )
            ax.set_xlim(t_start - pad_s, t_end + pad_s)
            if row_idx == 0:
                ax.set_title(f"RPE {rpe}", fontsize=18, fontweight="bold",
                             color=EXERCISE_COLORS.get(exercise, "#222"))
            if col_idx == 0:
                ax.set_ylabel(spec["label"], fontsize=14, fontweight="bold",
                              color=spec["color"])
            if row_idx == n_rows - 1:
                ax.set_xlabel("Tid (s)", fontsize=14)
                ax.tick_params(axis="x", labelsize=12)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

    fig.suptitle(f"Recording {recording_id} — {exercise}", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    despine(fig=fig)

    out_dir = Path(f"inspections/recording_{recording_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_rel" if use_rel else ""
    out_path = out_dir / f"{exercise}_fatigue_dashboard{suffix}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def _set_slopes_pct(set_rows: pd.DataFrame, col: str) -> float:
    """Linfit slope of `col` over active set, expressed as %/s relative to the set's start value.
    Returns NaN if not enough points or start value is zero."""
    t = set_rows["t_session_s"].to_numpy()
    y = set_rows[col].to_numpy()
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return np.nan
    slope, intercept = np.polyfit(t[mask], y[mask], 1)
    y_start = intercept + slope * t[mask][0]
    if not y_start:
        return np.nan
    return 100.0 * slope / y_start  # %/s


def plot_jasa(recording_id: str) -> Path:
    """JASA (Joint Analysis of Spectrum and Amplitude) — Luttmann et al. 1996.
    One point per set; x = MNF slope (%/s), y = RMS slope (%/s).
    Quadrant labels classify each set: fatigue / recovery / force↑ / force↓."""
    df = load_window_features(recording_id)
    active = df[df["in_active_set"] == True]  # noqa: E712
    set_numbers = sorted(int(n) for n in active["set_number"].dropna().unique())

    rows = []
    for set_n in set_numbers:
        s = active[active["set_number"] == float(set_n)]
        rows.append({
            "set_number": set_n,
            "exercise": s["exercise"].iloc[0],
            "rpe": int(s["rpe_for_this_set"].iloc[0]),
            "mnf_slope_pct": _set_slopes_pct(s, "emg_mnf"),
            "rms_slope_pct": _set_slopes_pct(s, "emg_rms"),
        })
    summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 8))

    x_max = max(abs(summary["mnf_slope_pct"].max()), abs(summary["mnf_slope_pct"].min())) * 1.25
    y_max = max(abs(summary["rms_slope_pct"].max()), abs(summary["rms_slope_pct"].min())) * 1.25

    # Quadrant shading (Luttmann fatigue diagram)
    ax.axhspan(0, y_max, xmin=0, xmax=0.5, color="#E45756", alpha=0.10, zorder=0)   # MNF↓ RMS↑ = fatigue
    ax.axhspan(-y_max, 0, xmin=0.5, xmax=1.0, color="#54A24B", alpha=0.10, zorder=0) # MNF↑ RMS↓ = recovery
    ax.axhspan(0, y_max, xmin=0.5, xmax=1.0, color="#F58518", alpha=0.06, zorder=0)  # force↑
    ax.axhspan(-y_max, 0, xmin=0, xmax=0.5, color="#888", alpha=0.06, zorder=0)      # force↓

    ax.axhline(0, color="#222", linewidth=0.8)
    ax.axvline(0, color="#222", linewidth=0.8)

    # Quadrant labels
    pad_x, pad_y = x_max * 0.04, y_max * 0.04
    ax.text(-x_max + pad_x, y_max - pad_y, "FATIGUE\n(MNF↓ RMS↑)",
            fontsize=10, fontweight="bold", color="#B71C1C", va="top", ha="left")
    ax.text(x_max - pad_x, -y_max + pad_y, "RECOVERY\n(MNF↑ RMS↓)",
            fontsize=10, fontweight="bold", color="#1E6B2F", va="bottom", ha="right")
    ax.text(x_max - pad_x, y_max - pad_y, "force↑\n(MNF↑ RMS↑)",
            fontsize=9, color="#666", va="top", ha="right", style="italic")
    ax.text(-x_max + pad_x, -y_max + pad_y, "force↓\n(MNF↓ RMS↓)",
            fontsize=9, color="#666", va="bottom", ha="left", style="italic")

    # Plot one point per set; size = RPE; color = exercise
    for _, row in summary.iterrows():
        size = 60 + row["rpe"] * 35  # rpe 1 ≈ 95, rpe 10 ≈ 410
        color = EXERCISE_COLORS.get(row["exercise"], "#888")
        ax.scatter(row["mnf_slope_pct"], row["rms_slope_pct"],
                   s=size, color=color, edgecolor="black", linewidth=1.0,
                   alpha=0.85, zorder=4)
        ax.annotate(
            f"S{row['set_number']}",
            xy=(row["mnf_slope_pct"], row["rms_slope_pct"]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=9, fontweight="bold",
        )

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel("EMG MNF slope (%/s)  —  drop = spectral compression / fatigue")
    ax.set_ylabel("EMG RMS slope (%/s)  —  rise = motor unit recruitment")
    ax.set_title(
        f"Recording {recording_id} — JASA fatigue diagram (Luttmann et al. 1996)\n"
        f"one point per set; size scales with RPE; color = exercise"
    )
    ax.grid(alpha=0.3)

    # Exercise legend (color) + RPE size legend
    used = summary["exercise"].unique()
    ex_handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=10,
                   markerfacecolor=EXERCISE_COLORS[e], markeredgecolor="black",
                   label=e)
        for e in ["pullup", "squat", "deadlift", "benchpress"] if e in used
    ]
    rpe_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#666",
                   markersize=np.sqrt(60 + r * 35),
                   markerfacecolor="#ccc", markeredgecolor="black",
                   label=f"RPE {r}")
        for r in (3, 6, 9)
    ]
    leg1 = ax.legend(handles=ex_handles, loc="upper right", fontsize=8,
                     framealpha=0.9, title="exercise")
    ax.add_artist(leg1)
    ax.legend(handles=rpe_handles, loc="lower right", fontsize=8,
              framealpha=0.9, title="RPE")

    fig.tight_layout()
    despine(fig=fig)

    out_dir = Path(f"inspections/recording_{recording_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "jasa_fatigue_diagram.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    # Print quadrant counts
    n_fat = ((summary["mnf_slope_pct"] < 0) & (summary["rms_slope_pct"] > 0)).sum()
    n_rec = ((summary["mnf_slope_pct"] > 0) & (summary["rms_slope_pct"] < 0)).sum()
    n_fup = ((summary["mnf_slope_pct"] > 0) & (summary["rms_slope_pct"] > 0)).sum()
    n_fdn = ((summary["mnf_slope_pct"] < 0) & (summary["rms_slope_pct"] < 0)).sum()
    print(f"Saved: {out_path}")
    print(f"Quadrant counts: fatigue={n_fat}, recovery={n_rec}, force_up={n_fup}, force_down={n_fdn}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", required=True, help="recording id e.g. 014")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--set", dest="set_number", type=int, help="single set 1..12")
    group.add_argument("--exercise",
                       choices=["pullup", "squat", "deadlift", "benchpress"],
                       help="all sets of this exercise side-by-side")
    group.add_argument("--jasa", action="store_true",
                       help="JASA fatigue scatter — all sets in one figure")
    parser.add_argument("--pad", type=float, default=2.0,
                        help="seconds of context before/after each set (default 2)")
    parser.add_argument("--rel", action="store_true",
                        help="use baseline-normalized EMG (emg_mnf_rel, emg_rms_rel)")
    args = parser.parse_args()
    rid = args.recording.zfill(3)
    if args.jasa:
        plot_jasa(rid)
    elif args.exercise:
        plot_dashboard_exercise(rid, args.exercise, pad_s=args.pad, use_rel=args.rel)
    else:
        plot_dashboard_set(rid, args.set_number, pad_s=args.pad, use_rel=args.rel)


if __name__ == "__main__":
    main()
