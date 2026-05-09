"""Compact fatigue dashboard — one or all sets of an exercise.

Four stacked panels per set, sharing time axis:
  1. EMG MNF (Hz)         — drops with localized muscle fatigue (De Luca 1984; Cifrek 2009)
  2. EMG MDF (Hz)          — drops alongside MNF; less sensitive to noise (Cifrek 2009)
  3. EMG RMS (a.u.)        — rises with motor unit recruitment near failure (Luttmann et al. 1996)
  4. Acc RMS (g)          — drops as bar/limb velocity drops near failure (González-Badillo 2010)

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
    """Four fatigue panels: MNF/MDF (spectral compression), EMG RMS (recruitment),
    Acc RMS (velocity loss)."""
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
            "col": "emg_rms_rel" if use_rel else "emg_rms",
            "label": "EMG RMS" + (" (rel)" if use_rel else " (a.u.)"),
            "color": "#E45756",
            "unit": "" if use_rel else "a.u.",
            "expect_dir": "up",  # rise = motor unit recruitment / fatigue (Luttmann 1996)
            "show_peak": True,
        },
        {
            "col": "acc_rms",
            "label": "Acc RMS (g)",
            "color": "#54A24B",
            "unit": "g",
            "expect_dir": "down",  # drop = velocity loss
            "show_peak_valley": True,
        },
    ]


def _peak_valley_per_rep(t_active: np.ndarray, y_active: np.ndarray,
                         rep_times: np.ndarray, win_s: float = 0.5
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per rep marker: local peak (max within +/- win_s).
    Per gap between consecutive reps: local valley (min in that gap)."""
    rep_times = np.asarray(rep_times, dtype=float)
    if len(t_active) == 0:
        return (np.array([]),) * 4
    rep_times = rep_times[(rep_times >= t_active[0]) & (rep_times <= t_active[-1])]
    pt, py, vt, vy = [], [], [], []
    for r in rep_times:
        m = (t_active >= r - win_s) & (t_active <= r + win_s)
        ys = y_active[m]
        if ys.size and not np.all(np.isnan(ys)):
            i = np.nanargmax(ys)
            pt.append(t_active[m][i]); py.append(ys[i])
    for k in range(len(rep_times) - 1):
        r0, r1 = rep_times[k], rep_times[k + 1]
        m = (t_active > r0 + win_s) & (t_active < r1 - win_s)
        ys = y_active[m]
        if ys.size and not np.all(np.isnan(ys)):
            i = np.nanargmin(ys)
            vt.append(t_active[m][i]); vy.append(ys[i])
    return np.array(pt), np.array(py), np.array(vt), np.array(vy)


def _linfit_pct(t: np.ndarray, y: np.ndarray
                ) -> tuple[float, float, float, float]:
    """Returns (slope, y_start, y_end, pct). NaNs if not enough points."""
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan
    ta, ya = t[mask], y[mask]
    if len(ta) == 2:
        slope = (ya[1] - ya[0]) / (ta[1] - ta[0]) if ta[1] != ta[0] else np.nan
        intercept = ya[0] - slope * ta[0]
    else:
        slope, intercept = np.polyfit(ta, ya, 1)
    y_start = intercept + slope * ta[0]
    y_end = intercept + slope * ta[-1]
    pct = 100.0 * (y_end - y_start) / y_start if y_start else np.nan
    return float(slope), float(y_start), float(y_end), float(pct)


def _trend_color(slope: float, expect_dir: str) -> str:
    matches = ((expect_dir == "down" and slope < 0) or
               (expect_dir == "up" and slope > 0))
    return "#2E7D32" if matches else "#B71C1C"


def _annotate_delta(ax, label: str, pct: float, color: str,
                    y_frac: float = 0.94, x_frac: float = 0.98,
                    ha: str = "right", va: str = "top",
                    fontsize: int = 14) -> None:
    ax.annotate(
        f"{label} {pct:+.1f}%",
        xy=(x_frac, y_frac), xycoords="axes fraction",
        ha=ha, va=va, fontsize=fontsize, fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor=color, linewidth=1.0, alpha=0.92),
    )


def _draw_panel(ax, t_full, y_full, t_active, y_active, spec,
                rep_times=None) -> dict:
    """Δ% is computed from first rep to last rep (active-set window narrowed
    to that range). For panels with show_peak_valley, two trends are drawn:
    peak (max near each rep) and valley (min between consecutive reps)."""
    ax.plot(t_full, y_full, color="#888", linewidth=0.8, alpha=0.45)
    ax.plot(t_full, _smooth(y_full, window=11), color="#111",
            linewidth=1.5, alpha=0.95)

    fit = {"slope": np.nan, "pct": np.nan}

    # Narrow the active window to [first rep, last rep].
    if rep_times is not None and len(rep_times) >= 2 and len(t_active):
        rt = np.asarray(rep_times, dtype=float)
        rt = rt[(rt >= float(t_active[0])) & (rt <= float(t_active[-1]))]
        if len(rt) >= 2:
            r0, r1 = float(rt[0]), float(rt[-1])
            clip = (t_active >= r0) & (t_active <= r1)
            t_active = t_active[clip]
            y_active = y_active[clip]

    if spec.get("show_peak_valley") and rep_times is not None and len(rep_times) >= 2:
        pt, py, vt, vy = _peak_valley_per_rep(t_active, y_active,
                                              np.asarray(rep_times, dtype=float))
        if len(pt) >= 2:
            slope_p, ys_p, ye_p, pct_p = _linfit_pct(pt, py)
            cp = _trend_color(slope_p, spec["expect_dir"])
            ax.plot([pt[0], pt[-1]], [ys_p, ye_p], color=cp,
                    linewidth=2.0, linestyle="--", zorder=4)
            ax.scatter(pt, py, s=44, marker="^", color=cp,
                       edgecolor="white", linewidth=0.7, zorder=5)
            _annotate_delta(ax, "peak", pct_p, cp,
                            x_frac=0.98, y_frac=0.94, ha="right", va="top")
            fit = {"slope": slope_p, "pct": pct_p}
        if len(vt) >= 2:
            slope_v, ys_v, ye_v, pct_v = _linfit_pct(vt, vy)
            cv = _trend_color(slope_v, spec["expect_dir"])
            ax.plot([vt[0], vt[-1]], [ys_v, ye_v], color=cv,
                    linewidth=2.0, linestyle=":", zorder=4)
            ax.scatter(vt, vy, s=36, marker="v", color=cv,
                       edgecolor="white", linewidth=0.7, zorder=5)
            _annotate_delta(ax, "valley", pct_v, cv,
                            x_frac=0.98, y_frac=0.06, ha="right", va="bottom")
    elif spec.get("show_peak") and rep_times is not None and len(rep_times) >= 2:
        pt, py, _, _ = _peak_valley_per_rep(t_active, y_active,
                                            np.asarray(rep_times, dtype=float))
        if len(pt) >= 2:
            slope_p, ys_p, ye_p, pct_p = _linfit_pct(pt, py)
            cp = _trend_color(slope_p, spec["expect_dir"])
            ax.plot([pt[0], pt[-1]], [ys_p, ye_p], color=cp,
                    linewidth=2.0, linestyle="--", zorder=4)
            ax.scatter(pt, py, s=44, marker="^", color=cp,
                       edgecolor="white", linewidth=0.7, zorder=5)
            _annotate_delta(ax, "peak", pct_p, cp,
                            x_frac=0.98, y_frac=0.94, ha="right", va="top")
            fit = {"slope": slope_p, "pct": pct_p}
    else:
        mask = ~np.isnan(y_active)
        if mask.sum() >= 3:
            ta, ya = t_active[mask], y_active[mask]
            slope, intercept = np.polyfit(ta, ya, 1)
            y_start = intercept + slope * ta[0]
            y_end = intercept + slope * ta[-1]
            pct = 100.0 * (y_end - y_start) / y_start if y_start else np.nan
            trend_color = _trend_color(slope, spec["expect_dir"])
            ax.plot([ta[0], ta[-1]], [y_start, y_end],
                    color=trend_color, linewidth=2.0, linestyle="--", zorder=4)

            if rep_times is not None and len(rep_times) > 0:
                rep_t = np.asarray(rep_times, dtype=float)
                rep_t = rep_t[(rep_t >= ta[0]) & (rep_t <= ta[-1])]
                if rep_t.size > 0:
                    rep_y = np.interp(rep_t, ta, ya)
                    ax.plot(rep_t, rep_y, color=trend_color, linewidth=1.0,
                            alpha=0.85, zorder=4)
                    ax.scatter(rep_t, rep_y, s=42, color=trend_color,
                               edgecolor="white", linewidth=0.7, zorder=5)

            _annotate_delta(ax, "Δ", pct, trend_color, y_frac=0.94, fontsize=16)
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
            rep_times=rep_times,
        )
        ax.set_ylabel(spec["label"], fontsize=14, fontweight="bold",
                      color="#111")
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
                rep_times=rep_times,
            )
            ax.set_xlim(t_start - pad_s, t_end + pad_s)
            if row_idx == 0:
                ax.set_title(f"RPE {rpe}", fontsize=18, fontweight="bold",
                             color=EXERCISE_COLORS.get(exercise, "#222"))
            if col_idx == 0:
                ax.set_ylabel(spec["label"], fontsize=14, fontweight="bold",
                              color="#111")
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

    # Plot one point per set; size = RPE; color = exercise.
    # RPE is rendered inside each circle; set number is annotated outside.
    for _, row in summary.iterrows():
        size = 180 + row["rpe"] * 60  # bumped so the RPE digit fits inside
        color = EXERCISE_COLORS.get(row["exercise"], "#888")
        ax.scatter(row["mnf_slope_pct"], row["rms_slope_pct"],
                   s=size, color=color, edgecolor="black", linewidth=1.0,
                   alpha=0.85, zorder=4)
        ax.text(
            row["mnf_slope_pct"], row["rms_slope_pct"],
            f"{row['rpe']}",
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="white", zorder=5,
        )
        ax.annotate(
            f"S{row['set_number']}",
            xy=(row["mnf_slope_pct"], row["rms_slope_pct"]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=9, fontweight="bold",
        )

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel("EMG MNF slope (%/s)  —  drop = spectral compression / fatigue")
    ax.set_ylabel("EMG RMS slope (%/s)  —  rise = motor unit recruitment")
    ax.set_title(f"Recording {recording_id} — JASA")
    ax.grid(alpha=0.3)

    used = summary["exercise"].unique()
    ex_handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=14,
                   markerfacecolor=EXERCISE_COLORS[e], markeredgecolor="black",
                   label=e)
        for e in ["pullup", "squat", "deadlift", "benchpress"] if e in used
    ]
    ax.legend(handles=ex_handles, loc="upper right", fontsize=14,
              framealpha=0.9, title="exercise", title_fontsize=14)

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


def plot_jasa_all(label_points: bool = False, clip_pct: float = 90.0,
                  exercise_filter: str | None = None) -> Path:
    """JASA scatter pooled across every recording in data/labeled/.

    All sets of all recordings on one figure: ~108 points for 9 recordings ×
    12 sets. Color = exercise; size = RPE; RPE digit inside each circle. Set
    labels are off by default (would clutter at this density); pass
    --label-points to turn them on.

    `clip_pct` clips the axis range to the given percentile of |slope| so a
    single outlier set (e.g. squat S6 in rec 011) does not flatten everyone
    else against the origin. Outlier points are still plotted but may sit
    outside the axis frame.

    If `exercise_filter` is set, only that exercise's sets are kept and the
    output filename / title reflect the filter.
    """
    labeled_dir = Path("data/labeled")
    rec_dirs = sorted(p for p in labeled_dir.glob("recording_*") if p.is_dir())

    rows = []
    for rec_dir in rec_dirs:
        rid = rec_dir.name.split("_")[-1]
        try:
            df = load_window_features(rid)
        except FileNotFoundError:
            continue
        active = df[df["in_active_set"] == True]  # noqa: E712
        if active.empty:
            continue
        for set_n in sorted(int(n) for n in active["set_number"].dropna().unique()):
            s = active[active["set_number"] == float(set_n)]
            rows.append({
                "recording_id": rid,
                "set_number": set_n,
                "exercise": s["exercise"].iloc[0],
                "rpe": int(s["rpe_for_this_set"].iloc[0]),
                "mnf_slope_pct": _set_slopes_pct(s, "emg_mnf"),
                "rms_slope_pct": _set_slopes_pct(s, "emg_rms"),
            })
    summary = pd.DataFrame(rows).dropna(subset=["mnf_slope_pct", "rms_slope_pct"])
    if exercise_filter is not None:
        summary = summary[summary["exercise"] == exercise_filter].reset_index(drop=True)
        if summary.empty:
            raise ValueError(f"No sets found for exercise '{exercise_filter}'")

    fig, ax = plt.subplots(figsize=(12, 10))

    x_max = float(np.percentile(summary["mnf_slope_pct"].abs(), clip_pct)) * 1.15
    y_max = float(np.percentile(summary["rms_slope_pct"].abs(), clip_pct)) * 1.15
    x_max = max(x_max, 0.5)
    y_max = max(y_max, 0.5)

    ax.axhspan(0, y_max, xmin=0, xmax=0.5, color="#E45756", alpha=0.10, zorder=0)
    ax.axhspan(-y_max, 0, xmin=0.5, xmax=1.0, color="#54A24B", alpha=0.10, zorder=0)
    ax.axhspan(0, y_max, xmin=0.5, xmax=1.0, color="#F58518", alpha=0.06, zorder=0)
    ax.axhspan(-y_max, 0, xmin=0, xmax=0.5, color="#888", alpha=0.06, zorder=0)
    ax.axhline(0, color="#222", linewidth=0.8)
    ax.axvline(0, color="#222", linewidth=0.8)

    pad_x, pad_y = x_max * 0.04, y_max * 0.04
    ax.text(-x_max + pad_x, y_max - pad_y, "FATIGUE\n(MNF↓ RMS↑)",
            fontsize=11, fontweight="bold", color="#B71C1C", va="top", ha="left")
    ax.text(x_max - pad_x, -y_max + pad_y, "RECOVERY\n(MNF↑ RMS↓)",
            fontsize=11, fontweight="bold", color="#1E6B2F", va="bottom", ha="right")
    ax.text(x_max - pad_x, y_max - pad_y, "force↑\n(MNF↑ RMS↑)",
            fontsize=10, color="#666", va="top", ha="right", style="italic")
    ax.text(-x_max + pad_x, -y_max + pad_y, "force↓\n(MNF↓ RMS↓)",
            fontsize=10, color="#666", va="bottom", ha="left", style="italic")

    for _, row in summary.iterrows():
        size = 180 + row["rpe"] * 60
        color = EXERCISE_COLORS.get(row["exercise"], "#888")
        ax.scatter(row["mnf_slope_pct"], row["rms_slope_pct"],
                   s=size, color=color, edgecolor="black", linewidth=0.9,
                   alpha=0.80, zorder=4)
        ax.text(row["mnf_slope_pct"], row["rms_slope_pct"],
                f"{row['rpe']}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=5)
        if label_points:
            ax.annotate(
                f"R{row['recording_id']}.S{row['set_number']}",
                xy=(row["mnf_slope_pct"], row["rms_slope_pct"]),
                xytext=(7, 7), textcoords="offset points",
                fontsize=7, color="#333",
            )

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel("EMG MNF slope (%/s)  —  drop = spectral compression / fatigue")
    ax.set_ylabel("EMG RMS slope (%/s)  —  rise = motor unit recruitment")

    n_total = len(summary)
    n_recs = summary["recording_id"].nunique()
    n_fat = ((summary["mnf_slope_pct"] < 0) & (summary["rms_slope_pct"] > 0)).sum()
    n_rec = ((summary["mnf_slope_pct"] > 0) & (summary["rms_slope_pct"] < 0)).sum()
    n_fup = ((summary["mnf_slope_pct"] > 0) & (summary["rms_slope_pct"] > 0)).sum()
    n_fdn = ((summary["mnf_slope_pct"] < 0) & (summary["rms_slope_pct"] < 0)).sum()

    if exercise_filter is None:
        title_lead = f"JASA — all recordings (n={n_recs}, {n_total} sets)"
    else:
        title_lead = (f"JASA — {exercise_filter} (n={n_recs} recordings, "
                      f"{n_total} sets)")
    ax.set_title(
        f"{title_lead}  |  "
        f"fatigue={n_fat}  recovery={n_rec}  force↑={n_fup}  force↓={n_fdn}"
    )
    ax.grid(alpha=0.3)

    used = summary["exercise"].unique()
    ex_handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=14,
                   markerfacecolor=EXERCISE_COLORS[e], markeredgecolor="black",
                   label=e)
        for e in ["pullup", "squat", "deadlift", "benchpress"] if e in used
    ]
    ax.legend(handles=ex_handles, loc="upper right", fontsize=12,
              framealpha=0.9, title="exercise", title_fontsize=12)

    fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.08)
    despine(fig=fig)

    out_dir = Path("inspections")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / f"jasa_{exercise_filter}.png"
                if exercise_filter else out_dir / "jasa_all_recordings.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"n_recordings={n_recs}, n_sets={n_total}")
    print(f"Quadrant counts: fatigue={n_fat}, recovery={n_rec}, force_up={n_fup}, force_down={n_fdn}")

    off_x = summary["mnf_slope_pct"].abs() > x_max
    off_y = summary["rms_slope_pct"].abs() > y_max
    off = summary[off_x | off_y]
    if not off.empty:
        print(f"\nOutliers outside axis frame ({len(off)}/{n_total} sets):")
        for _, r in off.iterrows():
            print(f"  R{r['recording_id']}.S{r['set_number']:02d} {r['exercise']:>10}  "
                  f"RPE {r['rpe']}  MNF {r['mnf_slope_pct']:+7.2f}%/s  "
                  f"RMS {r['rms_slope_pct']:+8.2f}%/s")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", help="recording id e.g. 014 (omit for --jasa-all)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--set", dest="set_number", type=int, help="single set 1..12")
    group.add_argument("--exercise",
                       choices=["pullup", "squat", "deadlift", "benchpress"],
                       help="all sets of this exercise side-by-side")
    group.add_argument("--jasa", action="store_true",
                       help="JASA fatigue scatter — all sets of one recording")
    group.add_argument("--jasa-all", action="store_true",
                       help="JASA scatter pooled across every recording in data/labeled/")
    group.add_argument("--jasa-per-exercise", action="store_true",
                       help="One JASA scatter per exercise (4 PNGs in inspections/)")
    parser.add_argument("--pad", type=float, default=2.0,
                        help="seconds of context before/after each set (default 2)")
    parser.add_argument("--rel", action="store_true",
                        help="use baseline-normalized EMG (emg_mnf_rel, emg_rms_rel)")
    parser.add_argument("--label-points", action="store_true",
                        help="(--jasa-all only) annotate each point with R{rec}.S{set}")
    parser.add_argument("--clip-pct", type=float, default=90.0,
                        help="(--jasa-all only) axis clip percentile of |slope| (default 90)")
    args = parser.parse_args()
    if args.jasa_all:
        plot_jasa_all(label_points=args.label_points, clip_pct=args.clip_pct)
        return
    if args.jasa_per_exercise:
        for ex in ["pullup", "squat", "deadlift", "benchpress"]:
            plot_jasa_all(label_points=args.label_points, clip_pct=args.clip_pct,
                          exercise_filter=ex)
        return
    if not args.recording:
        parser.error("--recording is required unless using --jasa-all")
    rid = args.recording.zfill(3)
    if args.jasa:
        plot_jasa(rid)
    elif args.exercise:
        plot_dashboard_exercise(rid, args.exercise, pad_s=args.pad, use_rel=args.rel)
    else:
        plot_dashboard_set(rid, args.set_number, pad_s=args.pad, use_rel=args.rel)


if __name__ == "__main__":
    main()
