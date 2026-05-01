"""
Resample all biosignals and labels to a unified 100 Hz grid.

Strategy
--------
1. The biosignal with the highest sample rate is 2000 Hz (EMG). We resample
   everything to 100 Hz (the ACC/PPG native rate) to avoid large file sizes
   while preserving all clinically relevant information (EMG features will be
   extracted from the native 2000 Hz series by the feature extractor, but the
   aligned parquet stores the downsampled version as a compact reference).

2. Resampling method per signal type:
   - Continuous biosignals (ECG, EMG, EDA, PPG-green): linear interpolation
     onto the 100 Hz grid. This is acceptable for offline alignment; causal
     feature extraction will operate on the raw files.
   - IMU (ax, ay, az, acc_mag): linear interpolation (already at 100 Hz —
     no-op in practice, just timestamp alignment).
   - Temperature: forward-fill (very slow signal, 1 Hz; step-interpolation
     is appropriate).
   - Label columns (phase_label, exercise, set_number, etc.): nearest-neighbour
     propagation (categorical).
   - rep_count_in_set: nearest-neighbour (integer, monotonically
     non-decreasing).

3. The unified grid is generated from bio_t0 to bio_tend in steps of 0.01 s.

References
----------
- Resampling biosignals to a common grid via linear interpolation for
  multi-modal alignment is standard practice in wearable computing
  (Bulling et al. 2014, IEEE Pervasive Computing 13(2), 62-75).
- Forward-fill for temperature (1 Hz sensor) avoids introducing spurious
  sub-Hz variation: consistent with Maeda et al. (2011) recommendation
  for low-rate physiological signals.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def make_100hz_grid(t_start: float, t_end: float) -> np.ndarray:
    """Generate a 100 Hz Unix-epoch timestamp grid.

    Parameters
    ----------
    t_start: First timestamp (Unix epoch s).
    t_end:   Last timestamp (Unix epoch s).

    Returns
    -------
    numpy array of float64 timestamps at 0.01 s intervals.
    """
    # Use integer-safe linspace
    n = int(round((t_end - t_start) * 100.0)) + 1
    return np.linspace(t_start, t_start + (n - 1) / 100.0, n)


# ---------------------------------------------------------------------------
# Per-modality resamplers
# ---------------------------------------------------------------------------

def _resample_linear(
    src_t: np.ndarray,
    src_v: np.ndarray,
    grid_t: np.ndarray,
) -> np.ndarray:
    """Linear interpolation; extrapolated values become NaN."""
    return np.interp(grid_t, src_t, src_v, left=np.nan, right=np.nan)


def _resample_ffill(
    src_t: np.ndarray,
    src_v: np.ndarray,
    grid_t: np.ndarray,
) -> np.ndarray:
    """Nearest lower-bound (forward-fill) interpolation for step signals."""
    idx = np.searchsorted(src_t, grid_t, side="right") - 1
    result = np.full(len(grid_t), np.nan)
    valid = (idx >= 0) & (idx < len(src_v))
    result[valid] = src_v[idx[valid]]
    return result


def _resample_nearest(
    src_t: np.ndarray,
    src_v: np.ndarray,
    grid_t: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour interpolation for categorical/integer signals."""
    idx = np.searchsorted(src_t, grid_t, side="left")
    idx = np.clip(idx, 0, len(src_t) - 1)
    # Among left and right candidates, pick the closer one
    idx_left = np.maximum(idx - 1, 0)
    d_right = np.abs(src_t[idx] - grid_t)
    d_left = np.abs(src_t[idx_left] - grid_t)
    chosen = np.where(d_left <= d_right, idx_left, idx)
    return src_v[chosen]


# ---------------------------------------------------------------------------
# Main alignment function
# ---------------------------------------------------------------------------

def build_aligned_dataframe(
    grid_t: np.ndarray,
    bio_t0: float,
    # --- biosignals ---
    ecg_df: pd.DataFrame,
    emg_df: pd.DataFrame,
    eda_df: pd.DataFrame,
    ppg_green_df: pd.DataFrame,
    imu_df: pd.DataFrame,
    temp_df: Optional[pd.DataFrame],
    # --- per-set label arrays (built by run.py) ---
    set_info: pd.DataFrame,
    # --- joint angle arrays (one row per 100 Hz grid sample or None) ---
    joint_angle_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build the final aligned_features DataFrame at 100 Hz.

    Parameters
    ----------
    grid_t:       100 Hz Unix epoch timestamp array.
    bio_t0:       Session start Unix time (for t_session_s column).
    ecg_df:       DataFrame with ['timestamp', 'ecg'].
    emg_df:       DataFrame with ['timestamp', 'emg'].
    eda_df:       DataFrame with ['timestamp', 'eda'].
    ppg_green_df: DataFrame with ['timestamp', 'ppg_green'].
    imu_df:       DataFrame with ['timestamp', 'ax', 'ay', 'az', 'acc_mag'].
    temp_df:      DataFrame with ['timestamp', 'temperature'] or None/empty.
    set_info:     DataFrame with one row per 100 Hz grid sample containing:
                    'in_active_set', 'set_number', 'exercise',
                    'rpe_for_this_set', 'set_phase',
                    'rep_count_in_set', 'rep_index'
                  (aligned to grid_t, same length).
    joint_angle_df: DataFrame with ['t_unix', 'primary_joint_angle_deg',
                    'joint_velocity_deg_s', 'joint_accel_deg_s2',
                    'phase_label'] sampled at Kinect 30 Hz; will be
                    resampled to grid. None if no joint data available.

    Returns
    -------
    DataFrame with all required columns at 100 Hz.
    """
    n = len(grid_t)

    df = pd.DataFrame()
    df["t_unix"] = grid_t
    df["t"] = grid_t  # alias expected by tests
    df["t_session_s"] = grid_t - bio_t0

    # -----------------------------------------------------------------------
    # Biosignals — linear interpolation
    # -----------------------------------------------------------------------
    for src_df, col in [
        (ecg_df, "ecg"),
        (emg_df, "emg"),
        (eda_df, "eda"),
        (ppg_green_df, "ppg_green"),
    ]:
        t = src_df["timestamp"].to_numpy(dtype=float)
        v = src_df[col].to_numpy(dtype=float)
        df[col] = _resample_linear(t, v, grid_t)

    # IMU
    for col in ["ax", "ay", "az", "acc_mag"]:
        t = imu_df["timestamp"].to_numpy(dtype=float)
        v = imu_df[col].to_numpy(dtype=float)
        df[col] = _resample_linear(t, v, grid_t)

    # Temperature — forward fill; column name is 'temp' (required by hook)
    if temp_df is not None and len(temp_df) > 1:
        t = temp_df["timestamp"].to_numpy(dtype=float)
        v = temp_df["temperature"].to_numpy(dtype=float)
        df["temp"] = _resample_ffill(t, v, grid_t)
    else:
        df["temp"] = np.nan

    # -----------------------------------------------------------------------
    # Per-set labels (already on the 100 Hz grid from build_set_info_array)
    # -----------------------------------------------------------------------
    assert len(set_info) == n, (
        f"set_info length {len(set_info)} != grid length {n}"
    )
    for col in set_info.columns:
        df[col] = set_info[col].values

    # -----------------------------------------------------------------------
    # Joint angles — nearest-neighbour from 30 Hz Kinect frames
    # -----------------------------------------------------------------------
    if joint_angle_df is not None and len(joint_angle_df) > 0:
        jt = joint_angle_df["t_unix"].to_numpy(dtype=float)
        # Primary joint angle
        ja = joint_angle_df["primary_joint_angle_deg"].to_numpy(dtype=float)
        df["primary_joint_angle_deg"] = _resample_linear(jt, ja, grid_t)

        # Phase label: nearest-neighbour for categorical
        if "phase_label" in joint_angle_df.columns:
            phases = joint_angle_df["phase_label"].to_numpy(dtype=object)
            resampled_phases = _resample_nearest(jt, phases.astype(object), grid_t)
            # Convert NaN floats to "unknown" so all values are strings
            phase_labels = []
            for p in resampled_phases:
                if isinstance(p, float) and np.isnan(p):
                    phase_labels.append("unknown")
                elif p is None:
                    phase_labels.append("unknown")
                else:
                    phase_labels.append(str(p))
            df["phase_label"] = phase_labels
            # Overwrite rest periods (not in active set) with 'rest'
            if "in_active_set" in df.columns:
                outside = ~df["in_active_set"].fillna(False)
                df.loc[outside, "phase_label"] = "rest"
        else:
            # Inside active sets: unknown; outside: rest
            if "in_active_set" in df.columns:
                df["phase_label"] = np.where(df["in_active_set"].fillna(False), "unknown", "rest")
            else:
                df["phase_label"] = "rest"
    else:
        df["primary_joint_angle_deg"] = np.nan
        if "in_active_set" in df.columns:
            df["phase_label"] = np.where(df["in_active_set"].fillna(False), "unknown", "rest")
        else:
            df["phase_label"] = "rest"

    # Ensure phase_label is 'rest' outside active sets (hard rule)
    if "in_active_set" in df.columns:
        outside = ~df["in_active_set"].fillna(False)
        df.loc[outside, "phase_label"] = "rest"
        # Also nullify rep_count and rep_index outside sets
        for col in ["rep_count_in_set", "rep_index", "rpe_for_this_set",
                    "set_number", "exercise"]:
            if col in df.columns:
                # Convert to object dtype to allow NaN assignment
                df[col] = df[col].where(df["in_active_set"])

    return df


# ---------------------------------------------------------------------------
# Set-info array builder
# ---------------------------------------------------------------------------

def _rep_boundaries_from_joint(
    jdf: pd.DataFrame,
) -> list[float]:
    """Extract rep-end timestamps from a joint-angle DataFrame.

    `rep_count_in_set` is monotonic non-decreasing; each step (count k-1 -> k)
    marks the completion time of rep k. Returns the list of completion times.
    """
    if jdf is None or len(jdf) == 0 or "rep_count_in_set" not in jdf.columns:
        return []
    jr = jdf["rep_count_in_set"].to_numpy(dtype=float)
    jt = jdf["t_unix"].to_numpy(dtype=float)
    boundaries: list[float] = []
    prev_count = 0
    for i in range(len(jr)):
        c = jr[i]
        if not np.isfinite(c):
            continue
        ci = int(c)
        if ci > prev_count:
            boundaries.append(float(jt[i]))
            prev_count = ci
    return boundaries


def _fill_rep_density_hz(
    grid_t: np.ndarray,
    indices: np.ndarray,
    set_start: float,
    rep_boundaries: list[float],
    out: np.ndarray,
) -> None:
    """Write per-sample rep-density (Hz) into `out` for samples in `indices`.

    rep_boundaries[k] is the END time of rep k (1-indexed). Rep k spans
        [prev, rep_boundaries[k]],  prev = set_start for k=1, else rep_boundaries[k-1].
    Density during rep k = 1 / (boundary_k - prev). Outside any rep: 0.

    Integrating density over a window therefore yields the (fractional) number
    of reps captured in that window — the soft rep-detection target used by
    the model. Marker convention is 'end-of-rep'; if you switch to start-of-rep
    markers, shift boundaries before calling.
    """
    if not rep_boundaries:
        return
    t_in_set = grid_t[indices]
    prev = float(set_start)
    for boundary in rep_boundaries:
        dur = boundary - prev
        if dur <= 0:
            prev = boundary
            continue
        d_hz = 1.0 / dur
        local_mask = (t_in_set >= prev) & (t_in_set <= boundary)
        if local_mask.any():
            out[indices[local_mask]] = d_hz
        prev = boundary


def build_set_info_array(
    grid_t: np.ndarray,
    canonical_sets: list,          # list of SetMarker
    exercises: list[str | None],   # length 12, indexed by set position
    rpe_list: list[int | None],    # length 12, indexed by set position
    joint_angle_dfs: dict[int, pd.DataFrame | None],  # set_num -> angle df
    exercise_for_set: dict[int, str],  # set_num -> exercise name
) -> pd.DataFrame:
    """Build the per-100Hz-sample label array for one session.

    Parameters
    ----------
    grid_t:         100 Hz timestamp array.
    canonical_sets: List of SetMarker objects (canonical 12 sets).
    exercises:      From Participants.xlsx, exercise per set (list of 12).
    rpe_list:       From Participants.xlsx, RPE per set (list of 12).
    joint_angle_dfs: Dict mapping set_num to the joint angle DataFrame
                     (or None if unavailable).
    exercise_for_set: Maps canonical set_num to exercise string.

    Returns
    -------
    DataFrame with columns:
        in_active_set, set_number, exercise, rpe_for_this_set,
        set_phase, rep_count_in_set, rep_index, rep_density_hz
    """
    n = len(grid_t)

    in_active = np.zeros(n, dtype=bool)
    set_number = np.full(n, np.nan, dtype=object)
    exercise_arr = np.full(n, np.nan, dtype=object)
    rpe_arr = np.full(n, np.nan, dtype=object)
    set_phase_arr = np.full(n, "rest", dtype=object)
    rep_count_arr = np.full(n, np.nan, dtype=float)
    rep_index_arr = np.full(n, np.nan, dtype=float)
    rep_density_arr = np.zeros(n, dtype=float)

    for pos_idx, sm in enumerate(canonical_sets):
        # Position in canonical list (0-based) maps to Participants.xlsx set slot
        if pos_idx < len(exercises):
            exer = exercises[pos_idx]
            rpe = rpe_list[pos_idx] if pos_idx < len(rpe_list) else None
        else:
            exer = None
            rpe = None

        # Find grid indices within [start, end]
        mask = (grid_t >= sm.start_unix) & (grid_t <= sm.end_unix)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue

        in_active[indices] = True
        set_number[indices] = pos_idx + 1   # canonical 1-based index
        exercise_arr[indices] = exer if exer else "unknown"
        rpe_arr[indices] = float(rpe) if rpe is not None else np.nan

        # Set phase: start / middle / end (thirds of the set duration)
        t_within = grid_t[indices] - sm.start_unix
        dur = sm.duration_s
        thirds = dur / 3.0
        for ii, gi in enumerate(indices):
            tw = t_within[ii]
            if tw < thirds:
                set_phase_arr[gi] = "start"
            elif tw < 2 * thirds:
                set_phase_arr[gi] = "middle"
            else:
                set_phase_arr[gi] = "end"

        # Rep count: prefer joint-angle timing (precise per-rep extrema)
        # over marker timing (manual button-presses, count-accurate but
        # timing-imprecise). count_reps_from_angles in run.py was called
        # with target_n_reps=sm.n_reps so when joint data is usable, the
        # max of jdf['rep_count_in_set'] equals sm.n_reps. When the joint
        # detector returns all-zeros (signal poverty / Kinect tracking
        # failed / acc-fallback case), max is 0 and we fall back to
        # markers, then to acc-based phase if even markers are missing.
        jdf = joint_angle_dfs.get(sm.set_num)
        joint_max = 0
        if jdf is not None and len(jdf) > 0 and "rep_count_in_set" in jdf.columns:
            jr_full = jdf["rep_count_in_set"].to_numpy(dtype=float)
            if len(jr_full) > 0 and np.isfinite(jr_full).any():
                joint_max = int(np.nanmax(jr_full))

        if joint_max > 0:
            # Joint-angle primary: rep timing follows the actual movement
            jt = jdf["t_unix"].to_numpy(dtype=float)
            jr = jdf["rep_count_in_set"].to_numpy(dtype=float)
            resampled = _resample_nearest(jt, jr, grid_t[indices])
            rep_count_arr[indices] = resampled
            rep_index_arr[indices] = np.where(resampled > 0, resampled, np.nan)
            rep_boundaries = _rep_boundaries_from_joint(jdf)
            _fill_rep_density_hz(grid_t, indices, sm.start_unix,
                                  rep_boundaries, rep_density_arr)
        elif sm.rep_markers:
            # Marker fallback: manual rep timing (precise count, ±1s timing)
            rep_times = sorted([r.unix_time for r in sm.rep_markers])
            for gi in indices:
                t = grid_t[gi]
                passed = sum(1 for rt in rep_times if t >= rt)
                rep_count_arr[gi] = passed
                rep_idx_val = passed
                rep_index_arr[gi] = rep_idx_val if rep_idx_val > 0 else np.nan
            # marker_position='end' convention: boundary k = marker k unix_time
            _fill_rep_density_hz(grid_t, indices, sm.start_unix,
                                  rep_times, rep_density_arr)

    df = pd.DataFrame({
        "in_active_set": in_active,
        "set_number": set_number,
        "exercise": exercise_arr,
        "rpe_for_this_set": rpe_arr,
        "set_phase": set_phase_arr,
        "rep_count_in_set": rep_count_arr,
        "rep_index": rep_index_arr,
        "rep_density_hz": rep_density_arr,
    })

    return df
