"""Segmentation QC visualizer.

For a chosen recording, dump:
- overview.png         — full-session stack (6 modalities + joint angle + exercise/RPE band)
- interactive.html     — same as overview.png but Plotly (zoomable)
- per_set/set_NN_<ex>.png — per-set zoom (set window ± 10 s padding)
- report.md            — sanity-check table comparing parquet labels vs. raw
                         markers.json + metadata.json (set boundaries, rep counts)

Filtering is OFFLINE (filtfilt) — the goal is clean signals to verify that
sets, reps, and phases are correctly aligned to what the model gets fed.
This is a QC tool; never used at inference.

Sources of truth that get overlaid:
- aligned_features.parquet                    — what the model sees (labels)
- dataset_aligned/<rec>/metadata.json         — canonical set start/end (Kinect)
- dataset_aligned/<rec>/markers.json          — canonical rep timing (manual)

If parquet-derived labels disagree with raw metadata/markers, the report
flags it and the plots show both (parquet shading vs. raw vertical lines).

Usage
-----
    python scripts/visualize_segmentation.py --recording 012
    python scripts/visualize_segmentation.py --recording 012 --no-per-set
    python scripts/visualize_segmentation.py --recording 012 \
        --output-dir inspections/segmentation_qc/recording_012

References
----------
- González-Badillo & Sánchez-Medina (2010). Movement velocity as loading-
  intensity measure. Int J Sports Med, 31(5), 347-352. [phase definition]
- Bonomi et al. (2009). Detection of activity from acc magnitude. MSSE,
  41(9), 1770-1777. [acc-magnitude segmentation]
- De Luca (1997). Surface EMG in biomechanics. J Appl Biomech, 13(2),
  135-163. [EMG envelope and band-pass]
- Oppenheim & Schafer (2010). Discrete-Time Signal Processing (3rd ed.).
  Pearson. [filtfilt zero-phase offline filtering]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.plot_style import apply_style, despine

apply_style()


# ---------------------------------------------------------------------------
# Filter spec — offline (zero-phase filtfilt) at each modality's NATIVE
# sample rate (loaded directly from dataset_aligned/recording_NNN/<mod>.csv).
# This is what the LightGBM feature pipeline operates on (window_features.py
# loads native CSVs); for the parquet-based NN pipeline, signals are at 100 Hz
# but visual QC is more accurate at native rates — ECG QRS detail (500 Hz)
# and EMG spectral content (2000 Hz envelope) need their full bandwidth.
# ---------------------------------------------------------------------------

FS = 100.0  # parquet unified rate (kept for backward-compat in older code paths)

# Native sample rate per modality (verified via inspections/recording_012)
NATIVE_FS = {
    "ecg":       500.0,   # 500 Hz brest electrodes
    "emg":       2000.0,  # 2000 Hz forearm/biceps
    "eda":       50.0,    # 50 Hz wrist
    "ppg_green": 100.0,   # 100 Hz wrist
    "acc_mag":   100.0,   # 100 Hz wrist
    "temp":      1.0,     # 1 Hz skin
}

FILTER_SPEC = {
    "ecg":       {"type": "bp", "low": 0.5,  "high": 40.0, "order": 4,
                   "notch_hz": 50.0},
    "ppg_green": {"type": "bp", "low": 0.5,  "high": 8.0,  "order": 4},
    "acc_mag":   {"type": "bp", "low": 0.5,  "high": 20.0, "order": 4},
    "eda":       {"type": "lp", "cutoff": 5.0,             "order": 4},
    "temp":      {"type": "lp", "cutoff": 0.1,             "order": 2},
    # EMG: 20-450 Hz BP + 50 Hz notch + 100 ms RMS envelope (only feasible
    # at native 2000 Hz; at 100 Hz the BP upper edge would be Nyquist-clipped)
    "emg":       {"type": "emg_envelope", "low": 20.0, "high": 450.0,
                   "notch_hz": 50.0, "rms_window_s": 0.1},
}

# Phase colours — matches the joint_rep_qc visualizer for consistency.
# Only concentric and eccentric are drawn; isometric / unknown / rest
# intervals are intentionally left unshaded.
PHASE_COLORS = {
    "concentric": "#bcecc4",   # light green
    "eccentric":  "#bcd6ec",   # light blue
}

# Blue vertical line for each detected rep (joint-angle peak/valley or, on
# bench, acc vertical-velocity extremum — populated into rep_count_in_set
# by the labeling pipeline).
DETECTED_REP_COLOR = "#1f4e79"

EXERCISE_PALETTE = {
    "squat":      "#3498db",
    "deadlift":   "#9b59b6",
    "benchpress": "#1abc9c",
    "pullup":     "#e67e22",
    "rest":       "#ecf0f1",
    "unknown":    "#bdc3c7",
}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _apply_filter(x: np.ndarray, spec: dict, fs: float) -> np.ndarray:
    """Apply offline (filtfilt) filter per spec at the given fs.
    NaN-safe: NaNs preserved at original positions."""
    from scipy.signal import iirnotch, tf2sos, sosfiltfilt

    valid = np.isfinite(x)
    if valid.sum() < 50:
        return x.copy()

    y = x.copy()
    xv = x[valid].astype(float)

    if spec["type"] == "bp":
        b, a = butter(spec["order"], [spec["low"], spec["high"]],
                      btype="band", fs=fs)
        yv = filtfilt(b, a, xv)
        # Optional notch (line interference)
        if spec.get("notch_hz"):
            b_n, a_n = iirnotch(spec["notch_hz"], 30.0, fs=fs)
            yv = filtfilt(b_n, a_n, yv)
    elif spec["type"] == "lp":
        b, a = butter(spec["order"], spec["cutoff"], btype="low", fs=fs)
        yv = filtfilt(b, a, xv)
    elif spec["type"] == "emg_envelope":
        # 20-450 Hz BP + 50 Hz notch + RMS envelope (De Luca 1997)
        # At 2000 Hz native fs, the 450 Hz upper edge fits comfortably
        # below Nyquist (1000 Hz). At lower fs (e.g. 100 Hz parquet),
        # we clip the upper edge to 0.95 * Nyquist.
        nyq = fs / 2.0
        upper = min(spec.get("high", 450.0), 0.95 * nyq)
        b, a = butter(4, [spec["low"], upper], btype="band", fs=fs)
        bp = filtfilt(b, a, xv)
        if spec.get("notch_hz") and spec["notch_hz"] < nyq:
            b_n, a_n = iirnotch(spec["notch_hz"], 30.0, fs=fs)
            bp = filtfilt(b_n, a_n, bp)
        win = max(1, int(round(spec["rms_window_s"] * fs)))
        sq = bp ** 2
        kernel = np.ones(win) / win
        rms = np.sqrt(np.convolve(sq, kernel, mode="same"))
        yv = rms
    else:
        raise ValueError(f"unknown filter type: {spec['type']}")

    y[valid] = yv
    return y


def load_native_modality(
    aligned_dir: Path, modality: str, t_lo: float, t_hi: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single modality from its native CSV in dataset_aligned/.

    Returns (timestamps, values) trimmed to [t_lo, t_hi] Unix range.
    """
    if modality == "acc_mag":
        # Compute from ax/ay/az
        dfs = {}
        for axis in ("ax", "ay", "az"):
            p = aligned_dir / f"{axis}.csv"
            d = pd.read_csv(p)
            dfs[axis] = d
        # Use ax timestamps as reference
        t = dfs["ax"]["timestamp"].to_numpy(dtype=float)
        a_mag = np.sqrt(
            dfs["ax"]["ax"].to_numpy(dtype=float) ** 2
            + dfs["ay"]["ay"].to_numpy(dtype=float) ** 2
            + dfs["az"]["az"].to_numpy(dtype=float) ** 2
        )
        return t, a_mag
    elif modality == "temp":
        p = aligned_dir / "temperature.csv"
        if not p.exists():
            return np.array([]), np.array([])
        d = pd.read_csv(p)
        if d.empty or "temperature" not in d.columns:
            return np.array([]), np.array([])
        t = d["timestamp"].to_numpy(dtype=float)
        return t, d["temperature"].to_numpy(dtype=float)
    else:
        col_map = {"ecg": "ecg", "emg": "emg", "eda": "eda",
                    "ppg_green": "ppg_green"}
        col = col_map.get(modality, modality)
        p = aligned_dir / f"{modality}.csv"
        if not p.exists():
            return np.array([]), np.array([])
        d = pd.read_csv(p)
        t = d["timestamp"].to_numpy(dtype=float)
        return t, d[col].to_numpy(dtype=float)


def filter_all_modalities_native(aligned_dir: Path, t_lo: float, t_hi: float
                                   ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load and filter each modality from its native CSV at native fs.

    Returns dict mapping modality name → (timestamps, filtered_values).
    Trimmed to [t_lo, t_hi] for plotting context.

    Also adds a synthetic key `wrist_vz` containing signed wrist vertical
    velocity at 100 Hz (computed by `compute_wrist_vertical_velocity` from
    raw ax/ay/az). This is used by bench panels in place of joint angle.
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for mod in NATIVE_FS:
        spec = FILTER_SPEC[mod]
        fs = NATIVE_FS[mod]
        t, v = load_native_modality(aligned_dir, mod, t_lo, t_hi)
        if len(v) < 10:
            out[mod] = (np.array([]), np.array([]))
            continue
        # Trim to padded range (so filter has settling room outside view)
        pad = max(2.0, 5.0 / max(fs, 1.0))  # at least 2 s padding
        keep = (t >= t_lo - pad) & (t <= t_hi + pad)
        t = t[keep]
        v = v[keep]
        if len(v) < 50:
            out[mod] = (t, v)
            continue
        try:
            v_filt = _apply_filter(v, spec, fs)
        except Exception:
            v_filt = v
        out[mod] = (t, v_filt)

    # Compute wrist vertical velocity for the whole recording from raw IMU.
    # Used by bench panels as the joint-panel substitute (Kinect can't see
    # the elbow under the lifter's torso, so the joint-angle trace is NaN
    # there; wrist vz comes from the same pipeline that drives bench rep
    # detection in count_reps_from_acc).
    try:
        from src.labeling.joint_angles import compute_wrist_vertical_velocity
        ax_df = pd.read_csv(aligned_dir / "ax.csv")
        ay_df = pd.read_csv(aligned_dir / "ay.csv")
        az_df = pd.read_csv(aligned_dir / "az.csv")
        t_imu = ax_df["timestamp"].to_numpy(dtype=float)
        ax_arr = ax_df["ax"].to_numpy(dtype=float)
        ay_arr = ay_df["ay"].to_numpy(dtype=float)
        az_arr = az_df["az"].to_numpy(dtype=float)
        # filtfilt cannot handle NaN — drop trailing/leading NaN samples
        # (typically a few post-recording rows) so the LP/HP cascade
        # doesn't produce all-NaN output.
        good = (np.isfinite(ax_arr) & np.isfinite(ay_arr)
                & np.isfinite(az_arr) & np.isfinite(t_imu))
        t_imu = t_imu[good]
        ax_arr = ax_arr[good]
        ay_arr = ay_arr[good]
        az_arr = az_arr[good]
        if len(t_imu) >= 50:
            vz = compute_wrist_vertical_velocity(ax_arr, ay_arr, az_arr,
                                                   fs=100.0)
            out["wrist_vz"] = (t_imu, vz)
        else:
            out["wrist_vz"] = (np.array([]), np.array([]))
    except Exception:
        out["wrist_vz"] = (np.array([]), np.array([]))

    return out


# Legacy parquet-based filtering (kept as fallback)
def filter_all_modalities(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Filter signals from the 100 Hz parquet — fallback when native CSVs
    aren't accessible. Returns dict of np.ndarray indexed by parquet rows."""
    out = {}
    for col, spec in FILTER_SPEC.items():
        if col not in df.columns:
            out[col] = None
            continue
        x = df[col].to_numpy(dtype=float)
        if not np.any(np.isfinite(x)):
            out[col] = None
            continue
        out[col] = _apply_filter(x, spec, FS)
    return out


# ---------------------------------------------------------------------------
# Raw-source loading (for cross-check overlay)
# ---------------------------------------------------------------------------

def load_raw_metadata(rec_dir: Path) -> Tuple[List[Dict], float]:
    """Return (kinect_sets list, data_start_unix)."""
    with (rec_dir / "metadata.json").open() as f:
        m = json.load(f)
    return m.get("kinect_sets", []), float(m.get("data_start_unix_time", 0.0))


def load_raw_markers(rec_dir: Path) -> List[Dict]:
    with (rec_dir / "markers.json").open() as f:
        mk = json.load(f)
    return mk.get("markers", [])


def parse_rep_markers(markers: List[Dict]) -> Dict[int, List[float]]:
    """Return {set_num: [unix_time of each rep]}."""
    out: Dict[int, List[float]] = {}
    for entry in markers:
        lbl = entry.get("label", "")
        if "_Rep:" in lbl and lbl.startswith("Set:"):
            try:
                set_part, rep_part = lbl.split("_Rep:")
                set_num = int(set_part.replace("Set:", ""))
                out.setdefault(set_num, []).append(float(entry["unix_time"]))
            except (ValueError, KeyError):
                continue
    for k in out:
        out[k].sort()
    return out


# ---------------------------------------------------------------------------
# Sanity checks: parquet labels vs. raw sources
# ---------------------------------------------------------------------------

def cross_check(df: pd.DataFrame, kinect_sets: List[Dict],
                rep_markers: Dict[int, List[float]]) -> List[Dict]:
    """Compare parquet's derived labels against raw metadata/markers.

    Important: parquet's `set_number` column uses CANONICAL positional
    indexing (1..12), not the original metadata set_number (which can be
    1..15 when the recording has aborted/restarted attempts). We therefore
    match each parquet set to its corresponding metadata entry BY TIME,
    not by index.

    Returns list of {level, msg}.
    """
    flags: List[Dict] = []

    parquet_sets = sorted(df["set_number"].dropna().unique().astype(int))

    # Build canonical mapping: parquet_set_num → metadata kinect_set entry
    # by closest start-time match (within 2 s tolerance — set windows are
    # tens of seconds long so a 2 s misalignment would be unambiguous).
    parquet_to_meta: Dict[int, Dict] = {}
    unmatched: List[int] = []
    for sn in parquet_sets:
        sub = df[df["set_number"] == sn]
        p_start = float(sub["t_unix"].min())
        best = min(kinect_sets, key=lambda s: abs(s["start_unix_time"] - p_start))
        if abs(best["start_unix_time"] - p_start) <= 2.0:
            parquet_to_meta[sn] = best
        else:
            unmatched.append(sn)

    # 1. Set-count and matching
    n_extras = len(kinect_sets) - len(parquet_sets)
    if unmatched:
        flags.append({"level": "warn",
                      "msg": f"Could not match parquet sets {unmatched} to "
                             f"any metadata kinect_set within 2 s — labeling "
                             f"may be misaligned"})
    else:
        if n_extras > 0:
            flags.append({"level": "ok",
                          "msg": f"All {len(parquet_sets)} canonical sets matched "
                                 f"to metadata; {n_extras} extra metadata set(s) "
                                 f"correctly excluded by canonical selector"})
        else:
            flags.append({"level": "ok",
                          "msg": f"All {len(parquet_sets)} sets matched between "
                                 f"parquet and metadata.json"})

    # 2. Per-set start/end alignment (by canonical match)
    for sn, meta in parquet_to_meta.items():
        sub = df[df["set_number"] == sn]
        p_start = float(sub["t_unix"].min())
        p_end = float(sub["t_unix"].max())
        d_start = abs(p_start - meta["start_unix_time"])
        d_end = abs(p_end - meta["end_unix_time"])
        if d_start > 0.05 or d_end > 0.05:  # 50 ms tolerance at 100 Hz
            flags.append({"level": "warn",
                          "msg": f"Canonical set {sn} (orig {meta['set_number']}): "
                                 f"start drift {d_start*1000:.1f} ms, "
                                 f"end drift {d_end*1000:.1f} ms vs metadata"})

    # 3. Per-set rep count: parquet rep_count_in_set max vs. raw markers count
    #    using the matched original set_number for rep_markers lookup.
    for sn, meta in parquet_to_meta.items():
        sub = df[df["set_number"] == sn]
        max_rep = sub["rep_count_in_set"].max()
        parquet_n = int(max_rep) if pd.notna(max_rep) else 0
        orig_sn = meta["set_number"]
        raw_n = len(rep_markers.get(orig_sn, []))
        if parquet_n != raw_n:
            flags.append({"level": "warn",
                          "msg": f"Canonical set {sn} (orig {orig_sn}): "
                                 f"parquet says {parquet_n} reps, markers.json "
                                 f"says {raw_n}"})

    # 4. Phase coverage inside active sets
    inside = df[df["in_active_set"]]
    if len(inside) > 0:
        unk = (inside["phase_label"] == "unknown").mean()
        if unk > 0.3:
            flags.append({"level": "warn",
                          "msg": f"{unk:.0%} of active-set samples have "
                                 f"phase_label='unknown' (joint-angle gaps)"})
        else:
            flags.append({"level": "ok",
                          "msg": f"Phase coverage inside sets: "
                                 f"{(1-unk):.0%} labeled"})

    # 5. EDA usability flag from parquet
    if "eda_status" in df.columns:
        status = str(df["eda_status"].iloc[0])
        if status != "ok":
            flags.append({"level": "warn",
                          "msg": f"EDA status: {status} — channel will be "
                                 f"NaN in plots"})

    # 6. Temperature gaps
    if "temp" in df.columns:
        temp_nan = df["temp"].isna().mean()
        if temp_nan > 0.05:
            flags.append({"level": "warn",
                          "msg": f"Temperature {temp_nan:.0%} NaN — sensor "
                                 f"dropout"})

    if not any(f["level"] == "warn" for f in flags):
        flags.append({"level": "ok",
                      "msg": "No segmentation/sync warnings raised"})
    return flags


# ---------------------------------------------------------------------------
# PNG: full-session overview
# ---------------------------------------------------------------------------

PANEL_ORDER = [
    ("joint",     "Joint angle"),
    ("ecg",       "ECG"),
    ("emg",       "EMG"),
    ("eda",       "EDA"),
    ("ppg_green", "PPG"),
    ("acc_mag",   "Acc"),
    ("temp",      "Temp"),
]

SET_START_COLOR = "red"
SET_END_COLOR = "green"


def _draw_phase_bg(ax, t_rel: np.ndarray, phase: np.ndarray,
                    in_active: np.ndarray):
    """Shade contiguous concentric/eccentric blocks. Other phases
    (isometric, unknown, rest) are intentionally left unshaded.
    """
    if len(t_rel) == 0:
        return
    cur = phase[0]
    cur_start = t_rel[0]
    for i in range(1, len(phase)):
        if phase[i] != cur:
            if cur in PHASE_COLORS:
                ax.axvspan(cur_start, t_rel[i], facecolor=PHASE_COLORS[cur],
                           alpha=0.45, zorder=0, lw=0)
            cur = phase[i]
            cur_start = t_rel[i]
    if cur in PHASE_COLORS:
        ax.axvspan(cur_start, t_rel[-1], facecolor=PHASE_COLORS[cur],
                   alpha=0.45, zorder=0, lw=0)


def _draw_set_lines(ax, kinect_sets: List[Dict], t0: float):
    """Vertical lines at set start (red, solid) and end (green, dashed)."""
    for s in kinect_sets:
        ax.axvline(s["start_unix_time"] - t0, color=SET_START_COLOR, lw=0.9,
                   ls="-", alpha=0.7, zorder=1)
        ax.axvline(s["end_unix_time"] - t0, color=SET_END_COLOR, lw=0.9,
                   ls="--", alpha=0.7, zorder=1)


def _detected_rep_times_session_s(df: pd.DataFrame) -> np.ndarray:
    """Return session-relative times (s) of every rep increment in the
    parquet's `rep_count_in_set` column. Increments are populated by the
    labeling pipeline using joint-angle peaks (squat/deadlift/pullup) or
    wrist vertical-velocity peaks (benchpress).
    """
    if "rep_count_in_set" not in df.columns:
        return np.array([])
    rc = df["rep_count_in_set"].fillna(0).to_numpy()
    diffs = np.diff(rc, prepend=rc[0] if len(rc) else 0.0)
    inc_idx = np.where(diffs > 0)[0]
    if len(inc_idx) == 0:
        return np.array([])
    return df["t_session_s"].to_numpy()[inc_idx]


def _draw_detected_rep_lines(ax, rep_times_s: np.ndarray,
                               t_lo_s: Optional[float] = None,
                               t_hi_s: Optional[float] = None):
    """Vertical blue line at each detected rep (joint-angle / acc-derived
    timing from rep_count_in_set increments)."""
    if len(rep_times_s) == 0:
        return
    for rt in rep_times_s:
        if t_lo_s is not None and rt < t_lo_s:
            continue
        if t_hi_s is not None and rt > t_hi_s:
            continue
        ax.axvline(rt, color=DETECTED_REP_COLOR, lw=1.0, alpha=0.85,
                   zorder=3)


def plot_overview_png(df: pd.DataFrame,
                       sigs_native: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       kinect_sets: List[Dict], rep_markers: Dict[int, List[float]],
                       title: str, out_path: Path):
    n_panels = len(PANEL_ORDER) + 1  # + exercise/RPE band
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(16, 1.6 * len(PANEL_ORDER) + 1.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0] * len(PANEL_ORDER) + [0.7]},
    )

    t0 = float(df["t_unix"].iloc[0])
    t_rel = df["t_session_s"].to_numpy()
    phase = df["phase_label"].astype(str).to_numpy()
    in_active = df["in_active_set"].to_numpy(dtype=bool)

    for ax, (key, ylabel) in zip(axes[:len(PANEL_ORDER)], PANEL_ORDER):
        if key == "joint":
            y_t = t_rel
            y = df["primary_joint_angle_deg"].to_numpy()
            _draw_phase_bg(ax, t_rel, phase, in_active)
        else:
            t_native, v_native = sigs_native.get(key, (np.array([]), np.array([])))
            if len(v_native) > 0:
                y_t = t_native - t0
                y = v_native
            else:
                y_t, y = None, None

        if y is None or not np.any(np.isfinite(y)):
            ax.text(0.5, 0.5, f"{key} unavailable",
                    transform=ax.transAxes, ha="center", va="center",
                    color="red", fontsize=10)
            ax.set_yticks([])
        else:
            # For very high-rate signals, downsample for plotting (keeps file size sane).
            # ECG @500Hz × 2400s = 1.2M points; downsample to ~50k for overview.
            if len(y) > 100_000:
                stride = max(1, len(y) // 100_000)
                y_t_p = y_t[::stride]
                y_p = y[::stride]
            else:
                y_t_p = y_t
                y_p = y
            ax.plot(y_t_p, y_p, color="#2c3e50", lw=0.4, zorder=2)
            if key != "joint":
                ylim = ax.get_ylim()
                ax.fill_between(t_rel, ylim[0], ylim[1], where=in_active,
                                 color="#3498db", alpha=0.06, zorder=0)
                ax.set_ylim(ylim)

        _draw_set_lines(ax, kinect_sets, t0)
        ax.set_ylabel(ylabel, fontsize=10, rotation=0, ha="right", va="center",
                      labelpad=10)
        ax.tick_params(left=False, labelleft=False)
        ax.grid(alpha=0.15)

    # Detected reps (joint-angle / acc) — blue vertical lines on every panel
    detected_reps_s = _detected_rep_times_session_s(df)
    for ax in axes[:len(PANEL_ORDER)]:
        _draw_detected_rep_lines(ax, detected_reps_s)

    # Exercise / RPE band
    ex_ax = axes[-1]
    for s in kinect_sets:
        sn = s["set_number"]
        sub = df[df["set_number"] == sn]
        if len(sub) == 0:
            continue
        exer = str(sub["exercise"].dropna().iloc[0]) if sub["exercise"].notna().any() else "unknown"
        rpe_vals = sub["rpe_for_this_set"].dropna()
        rpe = float(rpe_vals.iloc[0]) if len(rpe_vals) else float("nan")
        x0 = s["start_unix_time"] - t0
        x1 = s["end_unix_time"] - t0
        ex_ax.axvspan(x0, x1, facecolor=EXERCISE_PALETTE.get(exer, "#bdc3c7"),
                      alpha=0.85, lw=0)
        mid = (x0 + x1) / 2
        ex_ax.text(mid, 0.5, f"S{int(sn)} {exer}\nRPE {rpe:.0f}",
                   ha="center", va="center", fontsize=7)
    ex_ax.set_yticks([])
    ex_ax.set_ylabel("Set", fontsize=8)
    ex_ax.set_ylim(0, 1)
    ex_ax.set_xlabel("Time (s, session-relative)", fontsize=10)

    # Legend: phases + detected reps + set lines
    legend = [
        Patch(facecolor=PHASE_COLORS["concentric"], alpha=0.45, label="concentric"),
        Patch(facecolor=PHASE_COLORS["eccentric"], alpha=0.45, label="eccentric"),
        plt.Line2D([0], [0], color=DETECTED_REP_COLOR, lw=1.0,
                   label="detected rep"),
        plt.Line2D([0], [0], color=SET_START_COLOR, lw=1.2, ls="-",
                   label="set start"),
        plt.Line2D([0], [0], color=SET_END_COLOR, lw=1.2, ls="--",
                   label="set end"),
    ]
    axes[0].legend(handles=legend, loc="upper right", fontsize=7,
                   ncol=len(legend), framealpha=0.9)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    despine(fig=fig, left=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-set zoom PNGs
# ---------------------------------------------------------------------------

def plot_set_zoom_png(df: pd.DataFrame,
                       sigs_native: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       kinect_set: Dict, rep_times: List[float],
                       out_path: Path, pad_s: float = 8.0):
    sn = kinect_set["set_number"]
    s_start = kinect_set["start_unix_time"]
    s_end = kinect_set["end_unix_time"]
    t_lo = s_start - pad_s
    t_hi = s_end + pad_s

    mask = (df["t_unix"] >= t_lo) & (df["t_unix"] <= t_hi)
    sub = df[mask]
    if len(sub) == 0:
        return
    idx = np.where(mask.to_numpy())[0]

    t0 = float(df["t_unix"].iloc[0])
    t_rel = sub["t_session_s"].to_numpy()
    phase = sub["phase_label"].astype(str).to_numpy()
    in_active = sub["in_active_set"].to_numpy(dtype=bool)

    # Detected rep times (joint-angle / acc) within the zoom window
    detected_reps_s = _detected_rep_times_session_s(sub)
    n_detected = len(detected_reps_s)

    exer_vals = sub["exercise"].dropna()
    exer = str(exer_vals.iloc[0]) if len(exer_vals) else "unknown"
    rpe_vals = sub["rpe_for_this_set"].dropna()
    rpe = float(rpe_vals.iloc[0]) if len(rpe_vals) else float("nan")

    fig, axes = plt.subplots(
        len(PANEL_ORDER), 1,
        figsize=(14, 1.5 * len(PANEL_ORDER)),
        sharex=True,
    )

    is_bench = exer.lower() == "benchpress"

    for ax, (key, ylabel) in zip(axes, PANEL_ORDER):
        if key == "joint":
            if is_bench:
                # Bench: substitute joint angle with wrist vertical velocity
                # (same trace shown in joint_rep_qc bench panels — Kinect
                # cannot see the elbow under the lifter's torso).
                t_vz, v_vz = sigs_native.get("wrist_vz",
                                              (np.array([]), np.array([])))
                if len(v_vz) > 0:
                    m = (t_vz >= t_lo) & (t_vz <= t_hi)
                    y_t = t_vz[m] - t0
                    y = v_vz[m]
                else:
                    y_t, y = np.array([]), np.array([])
                ylabel = "Wrist v_z"
            else:
                y_full = df["primary_joint_angle_deg"].to_numpy()
                y = y_full[idx]
                y_t = t_rel
            _draw_phase_bg(ax, t_rel, phase, in_active)
        else:
            t_native, v_native = sigs_native.get(key, (np.array([]), np.array([])))
            if len(v_native) > 0:
                # Slice to set-window range for zoom panel
                m = (t_native >= t_lo) & (t_native <= t_hi)
                y_t = t_native[m] - t0
                y = v_native[m]
            else:
                y_t, y = np.array([]), np.array([])

        if y is None or len(y) == 0 or not np.any(np.isfinite(y)):
            ax.text(0.5, 0.5, f"{key} unavailable",
                    transform=ax.transAxes, ha="center", va="center",
                    color="red")
        else:
            ax.plot(y_t, y, color="#2c3e50", lw=0.6)

        # Set start (red, solid) / end (green, dashed)
        ax.axvline(s_start - t0, color=SET_START_COLOR, lw=1.2, ls="-", alpha=0.8)
        ax.axvline(s_end - t0, color=SET_END_COLOR, lw=1.2, ls="--", alpha=0.8)

        # Active-set faint shading
        if key != "joint":
            ylim = ax.get_ylim()
            ax.fill_between(t_rel, ylim[0], ylim[1], where=in_active,
                             color="#3498db", alpha=0.06, zorder=0)
            ax.set_ylim(ylim)

        # Detected reps (joint-angle / acc) — blue vertical lines
        _draw_detected_rep_lines(ax, detected_reps_s)

        ax.set_ylabel(ylabel, fontsize=10, rotation=0, ha="right", va="center",
                      labelpad=10)
        ax.tick_params(left=False, labelleft=False)
        ax.grid(alpha=0.15)

    axes[-1].set_xlabel("Time (s, session-relative)", fontsize=10)
    fig.suptitle(f"Set {int(sn)} — {exer} (RPE {rpe:.0f}) — "
                 f"{int(s_end - s_start)}s, {n_detected} reps",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    despine(fig=fig, left=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interactive HTML (Plotly)
# ---------------------------------------------------------------------------

def plot_overview_html(df: pd.DataFrame,
                        sigs_native: Dict[str, Tuple[np.ndarray, np.ndarray]],
                        kinect_sets: List[Dict],
                        rep_markers: Dict[int, List[float]],
                        title: str, out_path: Path):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        out_path.write_text(
            f"<html><body><h1>{title}</h1>"
            f"<p>Plotly not installed: <code>pip install plotly</code></p>"
            f"</body></html>"
        )
        print("[viz]   Plotly not installed — wrote stub HTML")
        return

    t0 = float(df["t_unix"].iloc[0])
    t_rel = df["t_session_s"].to_numpy()

    n_panels = len(PANEL_ORDER) + 1
    fig = make_subplots(
        rows=n_panels, cols=1, shared_xaxes=True,
        row_heights=[2.0] * len(PANEL_ORDER) + [0.7],
        vertical_spacing=0.012,
        subplot_titles=[lbl for _, lbl in PANEL_ORDER] + ["Set / Exercise / RPE"],
    )

    for i, (key, _) in enumerate(PANEL_ORDER, start=1):
        if key == "joint":
            y = df["primary_joint_angle_deg"].to_numpy()
            x_native = t_rel
        else:
            t_native, v_native = sigs_native.get(key, (np.array([]), np.array([])))
            if len(v_native) == 0:
                fig.add_annotation(text=f"{key} unavailable",
                                   row=i, col=1, showarrow=False,
                                   font=dict(color="red"))
                continue
            x_native = t_native - t0
            y = v_native
        if y is None or not np.any(np.isfinite(y)):
            fig.add_annotation(text=f"{key} unavailable",
                               row=i, col=1, showarrow=False,
                               font=dict(color="red"))
            continue
        # Cap each trace at ~150k points for HTML responsiveness
        stride = max(1, len(y) // 150_000)
        fig.add_trace(
            go.Scatter(x=x_native[::stride], y=y[::stride], mode="lines",
                       name=key, line=dict(width=0.8, color="#2c3e50"),
                       hovertemplate=f"{key}: %{{y:.3g}}<br>t=%{{x:.1f}}s"
                                      f"<extra></extra>"),
            row=i, col=1,
        )

    # Set boundaries (vrects across all panels) — red start, green end
    for s in kinect_sets:
        x0 = s["start_unix_time"] - t0
        x1 = s["end_unix_time"] - t0
        for i in range(1, len(PANEL_ORDER) + 1):
            fig.add_vline(x=x0, line_color=SET_START_COLOR, line_width=1,
                          line_dash="solid", opacity=0.7, row=i, col=1)
            fig.add_vline(x=x1, line_color=SET_END_COLOR, line_width=1,
                          line_dash="dash", opacity=0.7, row=i, col=1)

    # Phase bands disabled in HTML overview (joint panel removed)

    # Exercise band as colored rects on bottom row
    band_row = n_panels
    for s in kinect_sets:
        sn = s["set_number"]
        sub = df[df["set_number"] == sn]
        if len(sub) == 0:
            continue
        exer = str(sub["exercise"].dropna().iloc[0]) if sub["exercise"].notna().any() else "unknown"
        rpe_vals = sub["rpe_for_this_set"].dropna()
        rpe = float(rpe_vals.iloc[0]) if len(rpe_vals) else float("nan")
        x0 = s["start_unix_time"] - t0
        x1 = s["end_unix_time"] - t0
        fig.add_vrect(x0=x0, x1=x1, fillcolor=EXERCISE_PALETTE.get(exer, "#bdc3c7"),
                      opacity=0.85, line_width=0, row=band_row, col=1)
        fig.add_annotation(x=(x0 + x1) / 2, y=0.5,
                           text=f"S{int(sn)} {exer}<br>RPE {rpe:.0f}",
                           showarrow=False, font=dict(size=9),
                           xref=f"x{band_row}", yref=f"y{band_row}",
                           row=band_row, col=1)
    fig.update_yaxes(visible=False, row=band_row, col=1, range=[0, 1])

    # Rep markers as scatter on acc_mag panel
    acc_row = next(i for i, (k, _) in enumerate(PANEL_ORDER, start=1)
                   if k == "acc_mag")
    rep_x, rep_text = [], []
    for sn, times in rep_markers.items():
        for ri, rt in enumerate(times, start=1):
            rep_x.append(rt - t0)
            rep_text.append(f"S{sn} r{ri}")
    if rep_x:
        _, acc_v = sigs_native.get("acc_mag", (np.array([]), np.array([])))
        if len(acc_v) > 0 and np.any(np.isfinite(acc_v)):
            finite_max = float(np.nanmax(acc_v[np.isfinite(acc_v)]))
        else:
            finite_max = 1.0
        fig.add_trace(
            go.Scatter(x=rep_x, y=[finite_max] * len(rep_x),
                       mode="markers", name="rep marker",
                       marker=dict(symbol="triangle-down", color="#e67e22",
                                   size=8),
                       text=rep_text, hovertemplate="%{text}<br>t=%{x:.2f}s"
                                                     "<extra></extra>"),
            row=acc_row, col=1,
        )

    fig.update_layout(
        title=title,
        height=180 * len(PANEL_ORDER) + 120,
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=70, r=20, t=60, b=40),
    )
    fig.update_xaxes(title_text="Time (s, session-relative)", row=n_panels, col=1)
    fig.write_html(out_path, include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(rec_id: str, df: pd.DataFrame, kinect_sets: List[Dict],
                  rep_markers: Dict[int, List[float]], flags: List[Dict],
                  out_dir: Path, png_name: str, html_name: str,
                  per_set_files: List[str]):
    subj = str(df["subject_id"].iloc[0]) if "subject_id" in df.columns else "?"
    eda_status = str(df["eda_status"].iloc[0]) if "eda_status" in df.columns else "?"

    lines = [
        f"# Segmentation QC — recording_{rec_id}",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Subject: {subj}",
        f"EDA status: {eda_status}",
        f"Duration: {df['t_session_s'].max():.1f} s",
        f"Sample rate: 100 Hz (parquet unified grid)",
        "",
        "## Plots",
        "",
        f"- Overview PNG: [{png_name}]({png_name})",
        f"- Interactive HTML: [{html_name}]({html_name})",
    ]
    if per_set_files:
        lines.append(f"- Per-set zooms: `per_set/`")
        for p in per_set_files:
            lines.append(f"  - [{p}](per_set/{p})")
    lines += ["", "## Sanity checks (parquet vs. raw markers/metadata)", ""]
    for f in flags:
        sym = "WARN" if f["level"] == "warn" else "OK"
        lines.append(f"- **{sym}**: {f['msg']}")

    # Per-set summary table — iterate over parquet canonical positions
    # (1..12) and look up their matched metadata entry by time.
    lines += ["", "## Per-set summary", "",
              "| Canon pos | Orig set | Exercise | Duration (s) | Reps (parquet / markers) | RPE | Phase coverage |",
              "|-----------|----------|----------|--------------|--------------------------|-----|----------------|"]
    parquet_sets = sorted(df["set_number"].dropna().unique().astype(int))
    for sn in parquet_sets:
        sub = df[df["set_number"] == sn]
        p_start = float(sub["t_unix"].min())
        meta = min(kinect_sets, key=lambda s: abs(s["start_unix_time"] - p_start))
        if abs(meta["start_unix_time"] - p_start) > 2.0:
            meta = None  # no good match
        exer = str(sub["exercise"].dropna().iloc[0]) if sub["exercise"].notna().any() else "?"
        dur = float(sub["t_unix"].max() - sub["t_unix"].min())
        max_rep = sub["rep_count_in_set"].max()
        parquet_n = int(max_rep) if pd.notna(max_rep) else 0
        if meta is not None:
            orig_sn = meta["set_number"]
            marker_n = len(rep_markers.get(orig_sn, []))
            orig_str = str(orig_sn)
        else:
            marker_n = 0
            orig_str = "?"
        rpe_vals = sub["rpe_for_this_set"].dropna()
        rpe = f"{rpe_vals.iloc[0]:.0f}" if len(rpe_vals) else "—"
        labeled = (sub["phase_label"].isin(["eccentric", "concentric",
                                              "isometric"])).mean()
        lines.append(f"| {sn} | {orig_str} | {exer} | {dur:.1f} | "
                     f"{parquet_n} / {marker_n} | {rpe} | "
                     f"{labeled:.0%} |")

    lines += [
        "",
        "## How to interpret",
        "",
        "- **Red solid lines** = set start (from `metadata.json[kinect_sets][...].start_unix_time`)",
        "- **Green dashed lines** = set end",
        "- **Orange triangles** = rep markers (from `markers.json`)",
        "- **Light blue shading** = `in_active_set` mask (parquet-derived)",
        "- **Red/green/grey background on joint panel** = phase label (parquet-derived from joint angles)",
        "- **Bottom band** = exercise + RPE per set (from Participants.xlsx)",
        "",
        "If solid set-start lines do NOT line up with the start of acc-mag bursts, "
        "or if rep triangles do NOT line up with peaks in acc-mag/joint angle, "
        "labeling is misaligned and `/label` needs to be re-run after fixing the source.",
        "",
        "## References",
        "",
        "- González-Badillo, J. J., & Sánchez-Medina, L. (2010). Movement velocity as "
        "a measure of loading intensity in resistance training. "
        "*International Journal of Sports Medicine*, 31(5), 347–352.",
        "- Bonomi, A. G., Goris, A. H. C., Yin, B., & Westerterp, K. R. (2009). "
        "Detection of type, duration, and intensity of physical activity "
        "using an accelerometer. *MSSE*, 41(9), 1770–1777.",
        "- De Luca, C. J. (1997). The use of surface electromyography in "
        "biomechanics. *J. Appl. Biomech.*, 13(2), 135–163.",
        "- Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal "
        "Processing* (3rd ed.). Pearson.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", "-r", required=True,
                     help="recording id, e.g. 012")
    ap.add_argument("--labeled-root", default="data/labeled", type=Path)
    ap.add_argument("--aligned-root", default="dataset_aligned", type=Path)
    ap.add_argument("--output-dir", default=None, type=Path)
    ap.add_argument("--no-per-set", action="store_true",
                     help="Skip per-set zoom PNGs (faster)")
    ap.add_argument("--no-html", action="store_true",
                     help="Skip the (slow) interactive.html — useful when "
                          "only PNG plots need refresh")
    ap.add_argument("--report-only", action="store_true",
                     help="Skip ALL plotting; only update report.md "
                          "(for fast cross-check verification)")
    args = ap.parse_args()

    rec_id = args.recording.lstrip("0").zfill(3)  # normalize to 3 digits
    rec_name = f"recording_{rec_id}"

    parquet_path = args.labeled_root / rec_name / "aligned_features.parquet"
    aligned_dir = args.aligned_root / rec_name
    if not parquet_path.exists():
        raise SystemExit(f"Parquet not found: {parquet_path}. Run /label first.")
    if not aligned_dir.exists():
        raise SystemExit(f"Aligned dir not found: {aligned_dir}.")

    out_dir = args.output_dir or Path("inspections/segmentation_qc") / rec_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_set").mkdir(exist_ok=True)

    print(f"[viz] Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"[viz]   {len(df)} samples, "
          f"{df['t_session_s'].max():.0f} s @ {FS:.0f} Hz")

    print(f"[viz] Loading raw markers + metadata")
    kinect_sets, _ = load_raw_metadata(aligned_dir)
    raw_markers = load_raw_markers(aligned_dir)
    rep_markers = parse_rep_markers(raw_markers)
    print(f"[viz]   {len(kinect_sets)} kinect_sets, "
          f"{sum(len(v) for v in rep_markers.values())} rep markers")

    if args.report_only:
        sigs_native = {}
    else:
        t_lo = float(df["t_unix"].min())
        t_hi = float(df["t_unix"].max())
        print(f"[viz] Loading + filtering 6 modalities at NATIVE sample rates "
              f"(ECG 500Hz, EMG 2000Hz, EDA 50Hz, PPG 100Hz, Acc 100Hz, Temp 1Hz)")
        sigs_native = filter_all_modalities_native(aligned_dir, t_lo, t_hi)

    print(f"[viz] Cross-checking parquet labels vs raw sources")
    flags = cross_check(df, kinect_sets, rep_markers)
    n_warn = sum(1 for f in flags if f["level"] == "warn")
    print(f"[viz]   {n_warn} warning(s)")

    title = f"recording_{rec_id} — segmentation QC"
    png_name = "overview.png"
    html_name = "interactive.html"

    if not args.report_only:
        print(f"[viz] Writing overview PNG")
        plot_overview_png(df, sigs_native, kinect_sets, rep_markers, title,
                          out_dir / png_name)

        if not args.no_html:
            print(f"[viz] Writing interactive HTML")
            plot_overview_html(df, sigs_native, kinect_sets, rep_markers, title,
                                out_dir / html_name)

    per_set_files = []
    if args.report_only:
        # Scan existing per_set folder for previously-generated zooms
        ps_dir = out_dir / "per_set"
        if ps_dir.exists():
            per_set_files = sorted(p.name for p in ps_dir.glob("*.png"))
    elif not args.no_per_set:
        print(f"[viz] Writing per-set zoom PNGs")
        for s in kinect_sets:
            sn = s["set_number"]
            sub = df[df["set_number"] == sn]
            exer = (str(sub["exercise"].dropna().iloc[0])
                    if len(sub) and sub["exercise"].notna().any() else "unknown")
            fname = f"set_{int(sn):02d}_{exer}.png"
            plot_set_zoom_png(df, sigs_native, s, rep_markers.get(sn, []),
                              out_dir / "per_set" / fname)
            per_set_files.append(fname)
            print(f"[viz]   {fname}")

    write_report(rec_id, df, kinect_sets, rep_markers, flags, out_dir,
                 png_name, html_name, per_set_files)

    print(f"\n[viz] Done. Open: {out_dir / 'report.md'}")
    if n_warn:
        print(f"[viz] {n_warn} warning(s) — see report.md")


if __name__ == "__main__":
    main()
