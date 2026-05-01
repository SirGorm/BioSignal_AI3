"""Joint-angle rep/phase QC visualizer.

For each canonical set in a recording, produces a panel showing:
- Primary joint-angle trace (degrees vs. seconds since set start)
- Phase background bands (concentric / eccentric / isometric / unknown)
  derived by `src.labeling.joint_angles.label_phase` — the same function the
  offline labeling pipeline uses
- Rep markers from `markers.json` (`Set:N_Rep:M`) as solid green vertical lines
- Rep markers from `count_reps_from_angles` (joint-angle peak/valley detector,
  same function the offline pipeline uses) as dashed orange vertical lines

This is a QC tool to compare manually-annotated rep timing against the
joint-angle-derived rep timing the labeling pipeline would fall back on
when markers are missing.

Outputs (per recording, under inspections/joint_rep_qc/recording_NNN/):
  set_NN_<exercise>_rep_fase.png  — one PNG per canonical set
  all_sets_rep_fase.png            — single 4x3 multi-panel figure

Benchpress sets: Kinect cannot see the elbow under the lifter's torso, so
joint-angle data is mostly NaN. Those panels still render with the marker
reps and whatever (sparse) angle samples exist; the joint-angle rep
detector typically returns 0 reps as expected.

Usage
-----
    python scripts/visualize_joint_reps.py                # all 9 recordings
    python scripts/visualize_joint_reps.py --recording 014
    python scripts/visualize_joint_reps.py --recording 014 \
        --output-dir inspections/joint_rep_qc/recording_014

References
----------
- Tao et al. (2012). Gait analysis using wearable sensors. Sensors 12(2),
  2255-2283. [peak/valley rep counting from kinematic signals]
- González-Badillo & Sánchez-Medina (2010). Movement velocity as loading-
  intensity measure. Int J Sports Med 31(5), 347-352. [phase definition]
- Bulling et al. (2014). Tutorial on human activity recognition. ACM
  Comput. Surv. 46(3), 33:1-33:33. [annotation-based ground truth as
  preferred reference for set/rep labels]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# Seaborn style for cleaner, presentation-ready panels.
sns.set_theme(style="white", context="notebook")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import load_metadata, load_imu
from src.data.participants import load_participants, get_recording_info
from src.labeling.markers import parse_markers, select_canonical_sets
from src.labeling.joint_angles import (
    load_joint_angles_for_set,
    label_phase,
    label_phase_from_acc,
    count_reps_from_angles,
    count_reps_from_acc,
    compute_wrist_vertical_velocity,
    smooth_angles_for_rep_detection,
    get_rep_detection_params,
)


# Bench uses acc instead of joint angle (Kinect occlusion)
_ACC_REP_EXERCISES = {"benchpress"}


DATASET_ALIGNED_DIR = REPO_ROOT / "dataset_aligned"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "inspections" / "joint_rep_qc"
PARTICIPANTS_XLSX = Path(
    "C:/Users/skogl/Downloads/eirikgsk/BioSignal_AI/dataset/Participants/Participants.xlsx"
)

# Phase colors. Only concentric and eccentric are drawn — isometric and
# unknown are intentionally omitted (no background shading).
PHASE_COLORS = {
    "concentric": "#bcecc4",   # light green
    "eccentric": "#bcd6ec",    # light blue
}


# ---------------------------------------------------------------------------
# Per-set computation
# ---------------------------------------------------------------------------

def compute_set_panel_data(
    rec_dir: Path,
    set_marker,
    exercise: str,
    imu_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Compute everything needed to render one set's panel.

    Returns a dict with keys:
        t_rel:           seconds since set start (np.ndarray)
        angle:           primary joint angle in degrees (np.ndarray, may be NaN)
        phase:           per-frame phase label (np.ndarray of str)
        marker_rep_t:    list of seconds-since-start of marker reps
        joint_rep_t:     list of seconds-since-start of joint-detected reps
        n_marker_reps:   int
        n_joint_reps:    int
    """
    sm = set_marker
    exer = (exercise or "unknown").lower()
    marker_rep_t = [r.unix_time - sm.start_unix for r in sm.rep_markers]

    # ---------------------------------------------------------------
    # Benchpress: use wrist IMU instead of joint angle (Kinect can't
    # see the elbow under the lifter's torso).
    # ---------------------------------------------------------------
    if exer in _ACC_REP_EXERCISES and imu_df is not None:
        return _compute_acc_panel(sm, exer, imu_df, marker_rep_t)

    jdf = load_joint_angles_for_set(
        rec_dir,
        sm.set_num,
        exer if exer != "unknown" else "squat",  # placeholder for triplet lookup
        sm.start_unix,
        set_end_unix=sm.end_unix,
    ) if exer in {"squat", "deadlift", "benchpress", "pullup"} else None

    if jdf is None or len(jdf) == 0:
        return {
            "t_rel": np.array([]),
            "angle": np.array([]),
            "angle_smoothed": np.array([]),
            "lp_hz": float("nan"),
            "source": "joint",
            "phase": np.array([], dtype=object),
            "marker_rep_t": marker_rep_t,
            "joint_rep_t": [],
            "n_marker_reps": len(marker_rep_t),
            "n_joint_reps": 0,
            "available": False,
        }

    jdf = jdf[
        (jdf["t_unix"] >= sm.start_unix) & (jdf["t_unix"] <= sm.end_unix)
    ].copy()
    jdf = jdf.reset_index(drop=True)

    n = len(jdf)
    t_rel = (jdf["t_unix"].to_numpy(dtype=float) - sm.start_unix)
    angle = jdf["primary_joint_angle_deg"].to_numpy(dtype=float)

    # Smoothed trace exactly as count_reps_from_angles sees it
    if exer in {"squat", "deadlift", "benchpress", "pullup"}:
        angle_smoothed = smooth_angles_for_rep_detection(
            jdf["primary_joint_angle_deg"], jdf["t_unix"], exer,
        )
        lp_hz = get_rep_detection_params(exer)["lp_hz"]
    else:
        angle_smoothed = angle.copy()
        lp_hz = float("nan")

    # Phase via the same function the labeling pipeline uses
    if n >= 30 and exer in {"squat", "deadlift", "benchpress", "pullup"}:
        phase_series = label_phase(
            jdf["primary_joint_angle_deg"],
            jdf["t_unix"],
            exer,
            target_n_reps=sm.n_reps if sm.n_reps > 0 else None,
        )
        phase = np.asarray(phase_series.values, dtype=object)
    else:
        phase = np.full(n, "unknown", dtype=object)

    # Joint-angle rep detection (anchored to marker count when available)
    if exer in {"squat", "deadlift", "benchpress", "pullup"} and n >= 10:
        rep_series = count_reps_from_angles(
            jdf["primary_joint_angle_deg"],
            jdf["t_unix"],
            exer,
            target_n_reps=sm.n_reps if sm.n_reps > 0 else None,
        )
        rep_counts = rep_series.to_numpy(dtype=int)
        # Rep boundaries = indices where the count first increments
        diffs = np.diff(rep_counts, prepend=0)
        rep_idx = np.where(diffs > 0)[0]
        joint_rep_t = list(t_rel[rep_idx])
    else:
        joint_rep_t = []

    return {
        "t_rel": t_rel,
        "angle": angle,
        "angle_smoothed": angle_smoothed,
        "lp_hz": lp_hz,
        "source": "joint",
        "phase": phase,
        "marker_rep_t": marker_rep_t,
        "joint_rep_t": joint_rep_t,
        "n_marker_reps": len(marker_rep_t),
        "n_joint_reps": len(joint_rep_t),
        "available": True,
    }


def _compute_acc_panel(sm, exer, imu_df, marker_rep_t) -> dict:
    """Build a panel for an exercise that uses wrist IMU instead of joint
    angle (currently: benchpress). The Y-axis trace becomes signed wrist
    vertical velocity; rep markers come from negative-velocity peaks.
    """
    mask = ((imu_df["timestamp"] >= sm.start_unix)
            & (imu_df["timestamp"] <= sm.end_unix))
    imu_set = imu_df.loc[mask].sort_values("timestamp")
    n = len(imu_set)
    if n < 50:
        return {
            "t_rel": np.array([]),
            "angle": np.array([]),
            "angle_smoothed": np.array([]),
            "lp_hz": float("nan"),
            "source": "acc",
            "phase": np.array([], dtype=object),
            "marker_rep_t": marker_rep_t,
            "joint_rep_t": [],
            "n_marker_reps": len(marker_rep_t),
            "n_joint_reps": 0,
            "available": False,
        }

    ax = imu_set["ax"].to_numpy(dtype=float)
    ay = imu_set["ay"].to_numpy(dtype=float)
    az = imu_set["az"].to_numpy(dtype=float)
    ts = imu_set["timestamp"].to_numpy(dtype=float)
    t_rel = ts - sm.start_unix

    velocity = compute_wrist_vertical_velocity(ax, ay, az, fs=100.0)

    # Phase from acc — same function the labeling pipeline uses
    phase = label_phase_from_acc(
        ax, ay, az, ts, exer, fs=100.0,
        target_n_reps=sm.n_reps if sm.n_reps > 0 else None,
    )

    # Rep timing from acc — anchored to marker count
    rep_series = count_reps_from_acc(
        ax, ay, az, ts, fs=100.0,
        target_n_reps=sm.n_reps if sm.n_reps > 0 else None,
    )
    rep_counts = rep_series.to_numpy(dtype=int)
    diffs = np.diff(rep_counts, prepend=0)
    rep_idx = np.where(diffs > 0)[0]
    joint_rep_t = list(t_rel[rep_idx])

    return {
        "t_rel": t_rel,
        "angle": velocity,            # vertical velocity (m/s surrogate)
        "angle_smoothed": velocity,   # same — already smoothed by the helper
        "lp_hz": 0.7,
        "source": "acc",
        "phase": np.asarray(phase, dtype=object),
        "marker_rep_t": marker_rep_t,
        "joint_rep_t": joint_rep_t,
        "n_marker_reps": len(marker_rep_t),
        "n_joint_reps": len(joint_rep_t),
        "available": True,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_phase_bands(ax, t_rel: np.ndarray, phase: np.ndarray, ymin: float,
                      ymax: float) -> None:
    """Shade contiguous concentric/eccentric blocks across the panel
    background. Isometric and unknown intervals are left unshaded.
    """
    if len(t_rel) == 0:
        return
    n = len(phase)
    i = 0
    while i < n:
        j = i
        cur = str(phase[i])
        while j + 1 < n and str(phase[j + 1]) == cur:
            j += 1
        if cur in PHASE_COLORS:
            t_left = t_rel[i] if i == 0 else 0.5 * (t_rel[i - 1] + t_rel[i])
            t_right = t_rel[j] if j == n - 1 else 0.5 * (t_rel[j] + t_rel[j + 1])
            ax.axvspan(t_left, t_right, color=PHASE_COLORS[cur], alpha=0.55,
                       zorder=0, linewidth=0)
        i = j + 1


def render_set_panel(ax, panel: dict, set_num: int, exercise: str,
                     duration_s: float) -> None:
    """Render one set's panel onto a matplotlib Axes."""
    t_rel = panel["t_rel"]
    angle = panel["angle"]
    phase = panel["phase"]
    source = panel.get("source", "joint")
    is_acc = source == "acc"

    if panel["available"] and np.any(np.isfinite(angle)):
        finite = np.isfinite(angle)
        ymin = float(np.nanmin(angle[finite]))
        ymax = float(np.nanmax(angle[finite]))
        pad = 0.1 if is_acc else 5.0
        ymin -= pad
        ymax += pad
        if (ymax - ymin) < (0.2 if is_acc else 10.0):
            ymax = ymin + (0.2 if is_acc else 10.0)
    else:
        ymin, ymax = 0.0, 1.0

    _draw_phase_bands(ax, t_rel, phase, ymin, ymax)

    if is_acc:
        # Acc-source panel: trace = signed wrist vertical velocity
        if len(t_rel) > 0:
            ax.plot(t_rel, angle, color="#d62728", linewidth=1.4,
                    alpha=0.95, label="vertical velocity (m/s)", zorder=2.5)
            ax.axhline(0.0, color="#444444", linewidth=0.6, alpha=0.6,
                       zorder=1)
    else:
        # Joint-source panel: raw + smoothed angle
        if len(t_rel) > 0:
            ax.plot(t_rel, angle, color="#888888", linewidth=0.9, alpha=0.7,
                    label="raw joint angle", zorder=2)
        angle_smoothed = panel.get("angle_smoothed")
        if angle_smoothed is not None and len(angle_smoothed) > 0:
            ax.plot(t_rel, angle_smoothed, color="#d62728", linewidth=1.4,
                    alpha=0.95, label="smoothed (detector view)", zorder=2.5)

    # Detected reps (orange dashed) — joint-angle or acc, depending on source
    for tr in panel["joint_rep_t"]:
        ax.axvline(tr, color="#ff7f0e", linewidth=1.4, alpha=0.9,
                   linestyle="--", zorder=3)

    ax.set_xlim(0, max(duration_s, 1.0))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("time within set (s)", fontsize=8)
    ax.set_ylabel("vertical velocity" if is_acc else "angle (deg)",
                  fontsize=8)
    title = f"set {set_num} | {exercise} | reps={panel['n_joint_reps']}"
    if not panel["available"]:
        title += "  [no data]"
    ax.set_title(title, fontsize=9)

    # Hide y-axis numeric tick labels + tick marks; keep the axis label
    ax.set_yticklabels([])
    ax.tick_params(axis="y", left=False, right=False)
    ax.tick_params(axis="x", labelsize=7)

    # Despine: drop top + both sides, keep only the bottom (x-axis)
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)


def _legend_handles() -> list:
    return [
        Patch(facecolor=PHASE_COLORS["concentric"], alpha=0.55,
              label="concentric"),
        Patch(facecolor=PHASE_COLORS["eccentric"], alpha=0.55,
              label="eccentric"),
        plt.Line2D([0], [0], color="#888888", linewidth=0.9, alpha=0.7,
                   label="raw angle"),
        plt.Line2D([0], [0], color="#d62728", linewidth=1.4,
                   label="smoothed angle"),
        plt.Line2D([0], [0], color="#ff7f0e", linewidth=1.4, linestyle="--",
                   label="detected rep"),
    ]


# ---------------------------------------------------------------------------
# Per-recording driver
# ---------------------------------------------------------------------------

def process_recording(
    recording_num: int,
    participants_data: dict,
    output_root: Path,
) -> dict:
    """Render all panels for one recording. Returns a summary dict."""
    rec_id = f"recording_{recording_num:03d}"
    rec_dir = DATASET_ALIGNED_DIR / rec_id

    summary = {
        "recording_id": rec_id,
        "n_sets": 0,
        "panels_written": 0,
        "rep_diff_flags": [],
        "errors": [],
    }

    if not rec_dir.exists():
        summary["errors"].append(f"missing {rec_dir}")
        return summary

    info = get_recording_info(participants_data, recording_num)
    if info is None:
        summary["errors"].append(
            f"recording {recording_num} not in Participants.xlsx"
        )
        return summary
    exercises = info["exercises"]

    try:
        markers = parse_markers(rec_dir / "markers.json")
    except Exception as e:
        summary["errors"].append(f"markers parse failed: {e}")
        return summary

    canonical = select_canonical_sets(markers, expected_n=12)
    summary["n_sets"] = len(canonical)
    if not canonical:
        summary["errors"].append("no canonical sets")
        return summary

    # Load IMU once per recording — needed for bench acc-based rep detection
    imu_df: Optional[pd.DataFrame] = None
    if any((exercises[i] if i < len(exercises) else "").lower() in _ACC_REP_EXERCISES
           for i in range(len(canonical))):
        try:
            imu_df = load_imu(rec_dir)
        except Exception as e:
            summary["errors"].append(f"IMU load failed (bench panels degraded): {e}")

    out_dir = output_root / rec_id
    out_dir.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[int, str, dict, float]] = []

    for pos_idx, sm in enumerate(canonical):
        exer = exercises[pos_idx] if pos_idx < len(exercises) else "unknown"
        panel = compute_set_panel_data(rec_dir, sm, exer, imu_df=imu_df)
        panels.append((sm.set_num, exer, panel, sm.duration_s))

        # Per-set PNG
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        render_set_panel(ax, panel, sm.set_num, exer, sm.duration_s)
        ax.legend(handles=_legend_handles(), loc="upper right",
                  fontsize=7, framealpha=0.85)
        fig.suptitle(f"{rec_id} — set {sm.set_num} ({exer})",
                     fontsize=11, y=0.995)
        fig.tight_layout()

        per_set_path = out_dir / f"set_{sm.set_num:02d}_{exer}_rep_fase.png"
        fig.savefig(per_set_path, dpi=110)
        plt.close(fig)
        summary["panels_written"] += 1

        if abs(panel["n_marker_reps"] - panel["n_joint_reps"]) > 1 \
                and panel["available"]:
            summary["rep_diff_flags"].append(
                f"set {sm.set_num} ({exer}): "
                f"markers={panel['n_marker_reps']} "
                f"joint={panel['n_joint_reps']}"
            )

    # Multi-panel figure (4x3 grid for 12 sets; adapt if fewer)
    n_panels = len(panels)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows),
                             squeeze=False)
    for idx, (set_num, exer, panel, dur) in enumerate(panels):
        r, c = divmod(idx, ncols)
        render_set_panel(axes[r][c], panel, set_num, exer, dur)
    # Blank any unused axes
    for idx in range(n_panels, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.legend(handles=_legend_handles(), loc="lower center",
               ncol=4, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        f"{rec_id} — joint-angle phase & rep QC "
        f"(subject={info.get('subject_id', '?')})",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))

    multi_path = out_dir / "all_sets_rep_fase.png"
    fig.savefig(multi_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--recording", type=int, default=None,
        help="Recording number (e.g. 14). Default: process all recordings "
             "found in dataset_aligned/.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (default: "
             "inspections/joint_rep_qc/recording_NNN/).",
    )
    args = parser.parse_args(argv)

    if not PARTICIPANTS_XLSX.exists():
        print(f"ERROR: Participants.xlsx not found at {PARTICIPANTS_XLSX}",
              file=sys.stderr)
        return 1

    print(f"Loading Participants.xlsx ...")
    participants_data = load_participants(PARTICIPANTS_XLSX)
    print(f"  {len(participants_data)} recordings in Participants.xlsx")

    if args.recording is not None:
        rec_nums = [args.recording]
    else:
        rec_nums = sorted(
            int(p.name.split("_")[1])
            for p in DATASET_ALIGNED_DIR.glob("recording_*")
            if p.is_dir() and p.name.split("_")[1].isdigit()
        )

    print(f"Processing {len(rec_nums)} recording(s): {rec_nums}")

    output_root = args.output_dir.parent if (
        args.recording is not None and args.output_dir is not None
    ) else DEFAULT_OUTPUT_ROOT

    all_flags = []
    for rn in rec_nums:
        print(f"\n=== recording_{rn:03d} ===")
        if args.recording is not None and args.output_dir is not None:
            # User passed an explicit output dir — write into it directly
            out_root = args.output_dir.parent
        else:
            out_root = DEFAULT_OUTPUT_ROOT
        summary = process_recording(rn, participants_data, out_root)
        print(f"  panels written: {summary['panels_written']}/{summary['n_sets']}")
        if summary["errors"]:
            for e in summary["errors"]:
                print(f"  ERROR: {e}")
        for f in summary["rep_diff_flags"]:
            print(f"  [rep diff] {f}")
            all_flags.append(f"{summary['recording_id']}: {f}")

    print(f"\nDone. {len(all_flags)} sets with |marker - joint| > 1 reps.")
    if all_flags:
        print("Flagged sets:")
        for f in all_flags:
            print(f"  - {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
