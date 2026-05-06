"""Plot emg_rms and acc_rms through a single set, all sets of one exercise, or session means.

Single set:
    python scripts/plot_set_emg_acc_rms.py --recording 014 --set 1

All 3 sets of one exercise (fatigue progression across sets):
    python scripts/plot_set_emg_acc_rms.py --recording 014 --exercise pullup
    python scripts/plot_set_emg_acc_rms.py --recording 014 --exercise squat

Mean RMS per set across the full session (all 12 sets):
    python scripts/plot_set_emg_acc_rms.py --recording 014 --session-means
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.plot_style import apply_style, despine

apply_style()


def load_window_features(recording_id: str) -> pd.DataFrame:
    path = Path(f"data/labeled/recording_{recording_id}/window_features.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run /label and /train first")
    return pd.read_parquet(path)


def _draw_trend(ax, t: np.ndarray, y: np.ndarray, unit: str) -> dict:
    """Linear regression over the active-set portion. Slope sign + Δ% indicate fatigue trend.

    Returns the fit summary so callers can also report Δ% across sets.
    """
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return {"slope": np.nan, "pct": np.nan, "y_start": np.nan, "y_end": np.nan}
    ta, ya = t[mask], y[mask]
    slope, intercept = np.polyfit(ta, ya, 1)
    y_start = intercept + slope * ta[0]
    y_end = intercept + slope * ta[-1]
    pct = 100.0 * (y_end - y_start) / y_start if y_start else np.nan
    arrow = "↑" if slope > 0 else "↓"
    color = "#2E7D32" if slope > 0 else "#B71C1C"
    ax.plot([ta[0], ta[-1]], [y_start, y_end],
            color=color, linewidth=2.2, linestyle="--",
            label=f"trend {arrow} Δ={pct:+.1f}% (slope={slope:+.2e} {unit}/s)",
            zorder=4)
    return {"slope": slope, "pct": pct, "y_start": y_start, "y_end": y_end}


def load_rep_times(recording_id: str, set_number: int) -> list[float]:
    """Per-rep session-time markers from markers.json for the given set."""
    markers_path = Path(f"dataset_aligned/recording_{recording_id}/markers.json")
    if not markers_path.exists():
        return []
    payload = json.loads(markers_path.read_text())
    markers = payload["markers"] if isinstance(payload, dict) else payload
    tag = f"Set:{set_number}_Rep:"
    return [float(m["time"]) for m in markers if m.get("label", "").startswith(tag)]


def plot_set(recording_id: str, set_number: int, pad_s: float = 2.0) -> Path:
    df = load_window_features(recording_id)
    set_rows = df[df["set_number"] == set_number]
    if set_rows.empty:
        avail = sorted(df["set_number"].dropna().unique().astype(int).tolist())
        raise ValueError(f"set {set_number} not found in recording {recording_id} (available: {avail})")

    t_start = float(set_rows["t_session_s"].min())
    t_end = float(set_rows["t_session_s"].max())
    exercise = str(set_rows["exercise"].iloc[0])
    rpe = int(set_rows["rpe_for_this_set"].iloc[0])

    win = df[(df["t_session_s"] >= t_start - pad_s) & (df["t_session_s"] <= t_end + pad_s)]
    t = win["t_session_s"].to_numpy()
    emg = win["emg_rms"].to_numpy()
    acc = win["acc_rms"].to_numpy()

    t_active = set_rows["t_session_s"].to_numpy()
    emg_active = set_rows["emg_rms"].to_numpy()
    acc_active = set_rows["acc_rms"].to_numpy()

    fig, (ax_emg, ax_acc) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    for ax in (ax_emg, ax_acc):
        ax.axvspan(t_start, t_end, color="#4C78A8", alpha=0.12, zorder=0,
                   label="active set")

    ax_emg.plot(t, emg, color="#C0392B", linewidth=1.2, alpha=0.85)
    _draw_trend(ax_emg, t_active, emg_active, unit="V")
    ax_emg.set_ylabel("EMG RMS (V)")

    ax_acc.plot(t, acc, color="#1F4E79", linewidth=1.2, alpha=0.85)
    _draw_trend(ax_acc, t_active, acc_active, unit="g")
    ax_acc.set_ylabel("Acc RMS (g)")
    ax_acc.set_xlabel("Session time (s)")

    rep_times = load_rep_times(recording_id, set_number)
    for rt in rep_times:
        for ax in (ax_emg, ax_acc):
            ax.axvline(rt, color="#888", linestyle="--", linewidth=0.8, alpha=0.7)
    if rep_times:
        ax_emg.plot([], [], color="#888", linestyle="--", linewidth=0.8,
                    label=f"rep markers (n={len(rep_times)})")

    ax_emg.set_title(
        f"Recording {recording_id} — Set {set_number} ({exercise}, RPE {rpe}): "
        f"EMG RMS & Acc RMS"
    )
    ax_emg.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_acc.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_emg.grid(alpha=0.3)
    ax_acc.grid(alpha=0.3)
    plt.setp(ax_emg.get_xticklabels(), visible=False)

    fig.tight_layout()
    despine(fig=fig)

    out_dir = Path(f"inspections/recording_{recording_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"set{set_number:02d}_emg_acc_rms.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_exercise_sets(recording_id: str, exercise: str, pad_s: float = 2.0) -> Path:
    """Side-by-side panels for all sets of one exercise. Per-set linear trend +
    Δ%-of-start lets you read fatigue progression across sets at a glance.
    """
    df = load_window_features(recording_id)
    ex_rows = df[df["exercise"] == exercise]
    if ex_rows.empty:
        avail = sorted(df["exercise"].dropna().unique().tolist())
        raise ValueError(f"exercise {exercise!r} not found in recording {recording_id} (available: {avail})")

    set_numbers = sorted(int(n) for n in ex_rows["set_number"].dropna().unique())
    n = len(set_numbers)

    fig, axes = plt.subplots(
        2, n, figsize=(4.6 * n, 6.2),
        sharey="row",
        gridspec_kw={"hspace": 0.10, "wspace": 0.06},
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    summary = []  # collected per-set Δ% for the suptitle line
    for col, set_n in enumerate(set_numbers):
        ax_emg = axes[0, col]
        ax_acc = axes[1, col]
        set_rows = df[df["set_number"] == float(set_n)]

        t_start = float(set_rows["t_session_s"].min())
        t_end = float(set_rows["t_session_s"].max())
        rpe = int(set_rows["rpe_for_this_set"].iloc[0])

        win = df[(df["t_session_s"] >= t_start - pad_s) & (df["t_session_s"] <= t_end + pad_s)]
        t = win["t_session_s"].to_numpy()
        emg_full = win["emg_rms"].to_numpy()
        acc_full = win["acc_rms"].to_numpy()

        t_active = set_rows["t_session_s"].to_numpy()
        emg_active = set_rows["emg_rms"].to_numpy()
        acc_active = set_rows["acc_rms"].to_numpy()

        for ax in (ax_emg, ax_acc):
            ax.axvspan(t_start, t_end, color="#4C78A8", alpha=0.12, zorder=0)
            ax.set_xlim(t_start - pad_s, t_end + pad_s)
            ax.grid(alpha=0.3)

        ax_emg.plot(t, emg_full, color="#C0392B", linewidth=1.0, alpha=0.85)
        emg_fit = _draw_trend(ax_emg, t_active, emg_active, unit="V")

        ax_acc.plot(t, acc_full, color="#1F4E79", linewidth=1.0, alpha=0.85)
        acc_fit = _draw_trend(ax_acc, t_active, acc_active, unit="g")
        ax_acc.set_xlabel("Session time (s)")

        rep_times = load_rep_times(recording_id, set_n)
        for rt in rep_times:
            for ax in (ax_emg, ax_acc):
                ax.axvline(rt, color="#888", linestyle="--", linewidth=0.7, alpha=0.6)

        ax_emg.set_title(f"Set {set_n} • RPE {rpe} • {len(rep_times)} reps", fontsize=10)
        ax_emg.legend(loc="upper left", fontsize=7, framealpha=0.9)
        ax_acc.legend(loc="upper left", fontsize=7, framealpha=0.9)
        plt.setp(ax_emg.get_xticklabels(), visible=False)

        summary.append((set_n, rpe, emg_fit["pct"], acc_fit["pct"]))

    axes[0, 0].set_ylabel("EMG RMS (V)")
    axes[1, 0].set_ylabel("Acc RMS (g)")

    summary_str = "  |  ".join(
        f"S{sn}(RPE{rpe}): EMG {ep:+.1f}% / Acc {ap:+.1f}%"
        for sn, rpe, ep, ap in summary
    )
    fig.suptitle(
        f"Recording {recording_id} — {exercise}: EMG RMS & Acc RMS across {n} sets\n"
        f"per-set Δ% (start→end):  {summary_str}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    despine(fig=fig)

    out_dir = Path(f"inspections/recording_{recording_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{exercise}_all_sets_emg_acc_rms.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Per-set delta-% (start to end): "
          + summary_str.replace("Δ", "d").replace("→", "->"))
    return out_path


EXERCISE_COLORS = {
    "pullup":     "#4C78A8",
    "squat":      "#F58518",
    "deadlift":   "#54A24B",
    "benchpress": "#E45756",
}


def plot_session_means(recording_id: str) -> Path:
    """Mean EMG RMS and Acc RMS per set across the full session (all 12 sets).
    Points colored by exercise, RPE annotated next to each point."""
    df = load_window_features(recording_id)
    active = df[df["in_active_set"] == True]  # noqa: E712
    per_set = (
        active.groupby("set_number")
        .agg(
            exercise=("exercise", "first"),
            rpe=("rpe_for_this_set", "first"),
            emg_mean=("emg_rms", "mean"),
            emg_std=("emg_rms", "std"),
            acc_mean=("acc_rms", "mean"),
            acc_std=("acc_rms", "std"),
        )
        .reset_index()
        .sort_values("set_number")
    )

    fig, (ax_emg, ax_acc) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True,
        gridspec_kw={"hspace": 0.10},
    )

    x = per_set["set_number"].to_numpy()

    for ax, mean_col, std_col, color_main, label, ylabel in (
        (ax_emg, "emg_mean", "emg_std", "#C0392B", "EMG RMS", "EMG RMS (V)"),
        (ax_acc, "acc_mean", "acc_std", "#1F4E79", "Acc RMS", "Acc RMS (g)"),
    ):
        # Connecting line shows the session-level trend
        ax.plot(x, per_set[mean_col], color=color_main, linewidth=1.4,
                alpha=0.6, zorder=2, label=f"mean {label} per set")
        # Std as light error bars
        ax.errorbar(x, per_set[mean_col], yerr=per_set[std_col],
                    fmt="none", ecolor=color_main, alpha=0.35,
                    capsize=3, zorder=2)
        # Per-set marker colored by exercise
        for _, row in per_set.iterrows():
            ax.scatter(row["set_number"], row[mean_col],
                       s=90, color=EXERCISE_COLORS.get(row["exercise"], "#888"),
                       edgecolor="black", linewidth=0.8, zorder=4)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    # RPE annotated above each EMG point
    for _, row in per_set.iterrows():
        ax_emg.annotate(
            f"RPE {int(row['rpe'])}",
            xy=(row["set_number"], row["emg_mean"]),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=8, color="#333",
        )

    ax_acc.set_xlabel("Set number")
    ax_acc.set_xticks(range(1, int(per_set["set_number"].max()) + 1))

    # Exercise legend
    used = per_set["exercise"].unique()
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                   markerfacecolor=EXERCISE_COLORS[e], markeredgecolor="black",
                   label=e)
        for e in ["pullup", "squat", "deadlift", "benchpress"] if e in used
    ]
    ax_emg.legend(handles=handles, loc="upper left", fontsize=8,
                  framealpha=0.9, title="exercise")
    ax_emg.set_title(
        f"Recording {recording_id} — mean EMG RMS & Acc RMS per set "
        f"(all {len(per_set)} sets)"
    )
    plt.setp(ax_emg.get_xticklabels(), visible=False)

    fig.tight_layout()
    despine(fig=fig)

    out_dir = Path(f"inspections/recording_{recording_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "session_mean_emg_acc_rms.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", required=True, help="recording id e.g. 014")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--set", dest="set_number", type=int,
                       help="single set number 1..12")
    group.add_argument("--exercise",
                       choices=["pullup", "squat", "deadlift", "benchpress"],
                       help="all sets of this exercise side-by-side")
    group.add_argument("--session-means", action="store_true",
                       help="mean RMS per set across all 12 sets")
    parser.add_argument("--pad", type=float, default=2.0,
                        help="seconds of context before/after each set (default 2)")
    args = parser.parse_args()
    rid = args.recording.zfill(3)
    if args.session_means:
        plot_session_means(rid)
    elif args.exercise:
        plot_exercise_sets(rid, args.exercise, pad_s=args.pad)
    else:
        plot_set(rid, args.set_number, pad_s=args.pad)


if __name__ == "__main__":
    main()
