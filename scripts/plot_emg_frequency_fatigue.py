"""Plot EMG frequency (MNF/MDF) over a session with RPE fatigue overlay.

Full session:
    python scripts/plot_emg_frequency_fatigue.py --recording 014
    python scripts/plot_emg_frequency_fatigue.py --recording 014 --all

Zoom on individual sets:
    python scripts/plot_emg_frequency_fatigue.py --recording 014 --sets 1
    python scripts/plot_emg_frequency_fatigue.py --recording 014 --sets 1,2,3
    python scripts/plot_emg_frequency_fatigue.py --recording 014 --sets 1,4,7,10  # one set per exercise
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.plot_style import apply_style, despine

apply_style()

EXERCISE_COLORS = {
    "pullup":     "#4C78A8",
    "squat":      "#F58518",
    "deadlift":   "#54A24B",
    "benchpress": "#E45756",
}


def smooth(series: pd.Series, window: int = 11) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).median()


def load_session(recording_id: str) -> pd.DataFrame:
    path = Path(f"data/labeled/recording_{recording_id}/window_features.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run /label first")
    return pd.read_parquet(path)


def per_set_table(df: pd.DataFrame) -> pd.DataFrame:
    active = df[df["in_active_set"] == True]  # noqa: E712
    return (
        active.groupby("set_number")
        .agg(
            t_start=("t_session_s", "min"),
            t_end=("t_session_s", "max"),
            exercise=("exercise", "first"),
            rpe=("rpe_for_this_set", "first"),
            mnf_med=("emg_mnf", "median"),
            mdf_med=("emg_mdf", "median"),
            mnf_rel_med=("emg_mnf_rel", "median"),
            mdf_rel_med=("emg_mdf_rel", "median"),
        )
        .reset_index()
        .sort_values("set_number")
    )


def plot(recording_id: str, metric: str = "mnf", use_rel: bool = False) -> Path:
    df = load_session(recording_id)
    sets = per_set_table(df)

    col = f"emg_{metric}_rel" if use_rel else f"emg_{metric}"
    set_med_col = f"{metric}_rel_med" if use_rel else f"{metric}_med"
    ylabel = (
        f"EMG {metric.upper()} (relative to baseline)"
        if use_rel
        else f"EMG {metric.upper()} (Hz)"
    )
    title = (
        f"Recording {recording_id} — EMG {metric.upper()}"
        f"{' (baseline-normalized)' if use_rel else ''} with RPE fatigue overlay"
    )

    # MNF outside active sets is unreliable — mask to active sets only
    active_mask = df["in_active_set"] == True  # noqa: E712
    t = df["t_session_s"].to_numpy()
    y_full = df[col].where(active_mask).to_numpy()
    y_smooth = smooth(pd.Series(y_full), window=11).to_numpy()

    fig, ax = plt.subplots(figsize=(15, 6))
    ax2 = ax.twinx()

    # Set-region shading + set/exercise/RPE annotation
    used_exercises = set()
    for _, row in sets.iterrows():
        color = EXERCISE_COLORS.get(row["exercise"], "#888")
        ax.axvspan(row["t_start"], row["t_end"], color=color, alpha=0.18, zorder=0)
        used_exercises.add(row["exercise"])
        ax.annotate(
            f"S{int(row['set_number'])}\nRPE {int(row['rpe'])}",
            xy=((row["t_start"] + row["t_end"]) / 2, 1.02),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            fontweight="bold",
        )

    # EMG frequency trace (active-set only)
    ax.plot(t, y_full, color="#888", alpha=0.35, linewidth=0.6, label=f"{metric.upper()} (raw)")
    ax.plot(t, y_smooth, color="#222", linewidth=1.4, label=f"{metric.upper()} (rolling median)")

    # Per-set median as larger marker — connects fatigue trajectory across sets
    ax.scatter(
        (sets["t_start"] + sets["t_end"]) / 2,
        sets[set_med_col],
        s=70,
        color="black",
        zorder=5,
        marker="D",
        label=f"per-set median {metric.upper()}",
    )

    # RPE overlay on right axis
    rpe_x = (sets["t_start"] + sets["t_end"]) / 2
    ax2.plot(
        rpe_x,
        sets["rpe"],
        color="#C0392B",
        linewidth=2.0,
        marker="o",
        markersize=9,
        label="RPE (per set)",
        zorder=4,
    )
    for _, row in sets.iterrows():
        ax2.annotate(
            f"{int(row['rpe'])}",
            xy=((row["t_start"] + row["t_end"]) / 2, row["rpe"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color="#C0392B",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("Session time (s)")
    ax.set_ylabel(ylabel)
    ax2.set_ylabel("RPE (1–10)", color="#C0392B")
    ax2.tick_params(axis="y", labelcolor="#C0392B")
    ax2.set_ylim(0, 10.5)

    ax.set_title(title)

    # Legends
    ex_handles = [
        mpatches.Patch(color=EXERCISE_COLORS[e], alpha=0.5, label=e)
        for e in ["pullup", "squat", "deadlift", "benchpress"]
        if e in used_exercises
    ]
    leg1 = ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    leg2 = ax.legend(handles=ex_handles, loc="upper right", fontsize=8, title="exercise", framealpha=0.9)
    ax.add_artist(leg1)
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    despine(fig=fig)

    out_dir = Path(f"inspections/recording_{recording_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{metric}{'_rel' if use_rel else ''}"
    out_path = out_dir / f"emg_frequency_fatigue_{suffix}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    # Pearson correlation (per-set median EMG-freq vs RPE) as a quick fatigue check
    corr = np.corrcoef(sets[set_med_col].to_numpy(), sets["rpe"].to_numpy())[0, 1]
    print(f"Saved: {out_path}")
    print(f"Per-set Pearson r (median {metric.upper()} vs RPE) = {corr:+.3f}")
    print(f"  (negative r expected — frequency drops as fatigue rises)")
    return out_path


def plot_zoom(
    recording_id: str,
    set_numbers: list[int],
    metric: str = "mnf",
    use_rel: bool = False,
    pad_s: float = 6.0,
) -> Path:
    """Zoomed view focused on selected sets, with within-set linear regression."""
    df = load_session(recording_id)
    sets = per_set_table(df)
    sets_dict = {int(r.set_number): r for r in sets.itertuples(index=False)}

    missing = [n for n in set_numbers if n not in sets_dict]
    if missing:
        raise ValueError(f"sets {missing} not found in recording {recording_id} "
                         f"(available: {sorted(sets_dict.keys())})")

    col = f"emg_{metric}_rel" if use_rel else f"emg_{metric}"
    ylabel = (
        f"EMG {metric.upper()} (relative to baseline)"
        if use_rel
        else f"EMG {metric.upper()} (Hz)"
    )

    n = len(set_numbers)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.2), sharey=True, squeeze=False)
    axes = axes[0]

    title_bits = []
    for ax, set_n in zip(axes, set_numbers):
        s = sets_dict[set_n]
        color = EXERCISE_COLORS.get(s.exercise, "#888")
        t_lo = s.t_start - pad_s
        t_hi = s.t_end + pad_s

        win = df[(df["t_session_s"] >= t_lo) & (df["t_session_s"] <= t_hi)]
        active = win[win["in_active_set"] == True]  # noqa: E712

        # Active-set shading
        ax.axvspan(s.t_start, s.t_end, color=color, alpha=0.18, zorder=0)

        # Raw + smoothed EMG-freq
        t_all = win["t_session_s"].to_numpy()
        y_all = win[col].where(win["in_active_set"] == True).to_numpy()  # noqa: E712
        y_smooth = smooth(pd.Series(y_all), window=7).to_numpy()
        ax.plot(t_all, y_all, color="#888", alpha=0.45, linewidth=0.8, label=f"{metric.upper()} (raw)")
        ax.plot(t_all, y_smooth, color="#222", linewidth=1.6, label=f"{metric.upper()} (smoothed)")

        # Within-set linear regression — fatigue signature is a downward slope
        slope = np.nan
        slope_pct_per_s = np.nan
        if len(active) >= 3:
            ta = active["t_session_s"].to_numpy()
            ya = active[col].to_numpy()
            mask = ~np.isnan(ya)
            if mask.sum() >= 3:
                slope, intercept = np.polyfit(ta[mask], ya[mask], 1)
                ax.plot(
                    [s.t_start, s.t_end],
                    [intercept + slope * s.t_start, intercept + slope * s.t_end],
                    color="#1F77B4", linewidth=2.2, linestyle="--",
                    label=f"linfit slope={slope:+.3f}/s",
                    zorder=4,
                )
                # Fatigue-relevant: % change over the set duration relative to start value
                start_val = intercept + slope * s.t_start
                if start_val and not np.isnan(start_val):
                    pct = 100.0 * slope * (s.t_end - s.t_start) / start_val
                    slope_pct_per_s = pct

        # RPE label box top-right
        ax.annotate(
            f"Set {set_n} • {s.exercise}\nRPE {int(s.rpe)}",
            xy=(0.98, 0.98), xycoords="axes fraction",
            ha="right", va="top",
            fontsize=10, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=color, linewidth=1.2, alpha=0.92),
        )

        # Slope annotation (within-set fatigue indicator)
        if not np.isnan(slope_pct_per_s):
            ax.annotate(
                f"Δ over set: {slope_pct_per_s:+.1f}%",
                xy=(0.02, 0.02), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=9, color="#1F77B4",
                fontweight="bold",
            )

        ax.set_xlabel("Session time (s)")
        ax.set_xlim(t_lo, t_hi)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.25)
        title_bits.append(f"S{set_n}({s.exercise},RPE{int(s.rpe)})")

    axes[0].set_ylabel(ylabel)
    fig.suptitle(
        f"Recording {recording_id} — zoom on sets: " + ", ".join(title_bits)
        + (" (baseline-normalized)" if use_rel else ""),
        fontsize=11,
    )
    fig.tight_layout()
    despine(fig=fig)

    out_dir = Path(f"inspections/recording_{recording_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sets_tag = "-".join(str(n) for n in set_numbers)
    suffix = f"{metric}{'_rel' if use_rel else ''}"
    out_path = out_dir / f"emg_frequency_fatigue_zoom_sets{sets_tag}_{suffix}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def parse_sets(arg: str) -> list[int]:
    return [int(x) for x in arg.split(",") if x.strip()]


def plot_exercise_trend(
    recording_id: str,
    exercise: str,
    bin_s: float = 2.0,
    use_rel: bool = False,
    stitched: bool = True,
) -> Path:
    """For one exercise, plot MNF + MDF binned to `bin_s` windows across all
    its sets, plus acc_rms as its own subplot below sharing the x-axis.
    Default: stitched — sets placed back-to-back with a vertical separator,
    rest periods fully removed. Pass stitched=False for real session-time."""
    df = load_session(recording_id)
    sets = per_set_table(df)
    ex_sets = sets[sets["exercise"] == exercise].sort_values("set_number")
    if ex_sets.empty:
        raise ValueError(f"exercise {exercise!r} not in recording {recording_id} "
                         f"(available: {sorted(sets['exercise'].unique())})")

    mnf_col = "emg_mnf_rel" if use_rel else "emg_mnf"
    mdf_col = "emg_mdf_rel" if use_rel else "emg_mdf"
    rel_tag = " (baseline-normalized)" if use_rel else ""

    fig, (ax, ax_acc) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.08},
        constrained_layout=True,
    )
    color = EXERCISE_COLORS.get(exercise, "#888")

    cursor = 0.0  # for stitched mode
    set_centers: list[tuple[float, int, float]] = []  # (x_center, set_number, rpe)

    for _, s in ex_sets.iterrows():
        seg = df[(df["t_session_s"] >= s["t_start"])
                 & (df["t_session_s"] <= s["t_end"])
                 & (df["in_active_set"] == True)].copy()  # noqa: E712
        if seg.empty:
            continue

        # Bin to bin_s seconds: median of MNF/MDF per bin (the "2-s window")
        t0 = seg["t_session_s"].min()
        seg["_bin"] = ((seg["t_session_s"] - t0) // bin_s).astype(int)
        binned = (
            seg.groupby("_bin")
            .agg(
                t_center=("t_session_s", "mean"),
                mnf=(mnf_col, "median"),
                mdf=(mdf_col, "median"),
            )
            .reset_index(drop=True)
        )
        if binned.empty:
            continue

        if stitched:
            x = cursor + (binned["t_center"] - t0).to_numpy()
            x_acc = cursor + (seg["t_session_s"].to_numpy() - t0)
            x_lo = cursor
            x_hi = cursor + (s["t_end"] - s["t_start"])
            cursor = x_hi
        else:
            x = binned["t_center"].to_numpy()
            x_acc = seg["t_session_s"].to_numpy()
            x_lo = s["t_start"]
            x_hi = s["t_end"]

        first = int(s["set_number"]) == int(ex_sets["set_number"].iloc[0])

        # --- Top panel: MNF + MDF ---
        ax.axvspan(x_lo, x_hi, color=color, alpha=0.10, zorder=0)
        ax.plot(x, binned["mnf"], "-o", color="#1F4E79", markersize=4, linewidth=1.5,
                label="MNF" if first else None)
        ax.plot(x, binned["mdf"], "-s", color="#C0392B", markersize=4, linewidth=1.5,
                label="MDF" if first else None)

        # --- Bottom panel: acc_rms at native window resolution (10 ms) ---
        ax_acc.axvspan(x_lo, x_hi, color=color, alpha=0.10, zorder=0)
        ax_acc.plot(x_acc, seg["acc_rms"].to_numpy(), color="#444",
                    linewidth=0.8, alpha=0.9,
                    label="acc_rms" if first else None)

        # Clear vertical separator between sets (not after the final set)
        if stitched and int(s["set_number"]) != int(ex_sets["set_number"].iloc[-1]):
            for a in (ax, ax_acc):
                a.axvline(x_hi, color="black", linewidth=1.6, linestyle="-",
                          alpha=0.85, zorder=3)

        set_centers.append(((x_lo + x_hi) / 2, int(s["set_number"]), float(s["rpe"])))

    # Set labels above the top panel
    for x_c, set_n, rpe in set_centers:
        ax.annotate(
            f"Set {set_n} • RPE {int(rpe)}",
            xy=(x_c, 1.01), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=10, color=color, fontweight="bold",
        )

    ax_acc.set_xlabel("Stitched time (s)" if stitched else "Session time (s)")
    ylabel = f"EMG frequency{rel_tag}" if use_rel else "EMG frequency (Hz)"
    ax.set_ylabel(ylabel)
    ax_acc.set_ylabel("acc_rms (g)")
    ax.set_title(
        f"Recording {recording_id} — {exercise}: MNF & MDF trends "
        f"({bin_s:g}-s windows){rel_tag}  •  acc_rms below"
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax_acc.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax_acc.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), visible=False)

    despine(fig=fig)

    out_dir = Path(f"inspections/recording_{recording_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{exercise}_{int(bin_s)}s{'_rel' if use_rel else ''}{'_stitched' if stitched else ''}"
    out_path = out_dir / f"emg_freq_trend_{suffix}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, help="recording id e.g. 014")
    parser.add_argument("--metric", choices=["mnf", "mdf"], default="mnf",
                        help="MNF (mean) or MDF (median) frequency")
    parser.add_argument("--rel", action="store_true",
                        help="use baseline-normalized variant (emg_mnf_rel)")
    parser.add_argument("--sets", type=parse_sets, default=None,
                        help="comma-separated set numbers to zoom on, e.g. 1,2,3")
    parser.add_argument("--pad", type=float, default=6.0,
                        help="seconds of context to show before/after each set in zoom mode")
    parser.add_argument("--exercise", choices=list(EXERCISE_COLORS.keys()), default=None,
                        help="plot MNF+MDF trend across all sets of one exercise")
    parser.add_argument("--bin-s", type=float, default=2.0,
                        help="bin width in seconds for the per-window aggregation (default 2 s)")
    parser.add_argument("--no-stitched", dest="stitched", action="store_false",
                        help="use real session time on x-axis instead of stitched")
    parser.set_defaults(stitched=True)
    parser.add_argument("--all", action="store_true",
                        help="render all 4 variants (mnf, mdf, mnf_rel, mdf_rel)")
    args = parser.parse_args()

    rid = args.recording.zfill(3)
    if args.exercise:
        if args.all:
            for rel in (False, True):
                plot_exercise_trend(rid, args.exercise, bin_s=args.bin_s,
                                    use_rel=rel, stitched=args.stitched)
        else:
            plot_exercise_trend(rid, args.exercise, bin_s=args.bin_s,
                                use_rel=args.rel, stitched=args.stitched)
    elif args.sets:
        if args.all:
            for m in ("mnf", "mdf"):
                for rel in (False, True):
                    plot_zoom(rid, args.sets, metric=m, use_rel=rel, pad_s=args.pad)
        else:
            plot_zoom(rid, args.sets, metric=args.metric, use_rel=args.rel, pad_s=args.pad)
    elif args.all:
        for m in ("mnf", "mdf"):
            for rel in (False, True):
                plot(rid, metric=m, use_rel=rel)
    else:
        plot(rid, metric=args.metric, use_rel=args.rel)


if __name__ == "__main__":
    main()
