"""
Resample all biosignals and labels to a unified 100 Hz grid.

Strategy
--------
1. The biosignal with the highest sample rate is 2000 Hz (EMG). The aligned
   parquet stores all signals at 100 Hz so the raw NN dataset reads a single
   coherent grid.

2. Resampling method per signal type:
   - EMG (2000 Hz native): bandpass 20-450 Hz + 50 Hz notch at native rate,
     then RMS envelope (50 ms moving window), then linear interpolation onto
     the 100 Hz grid. The RMS window doubles as the anti-aliasing filter for
     the 20:1 decimation (Oppenheim & Schafer 2010). The envelope preserves
     EMG amplitude information (Konrad 2005, Merletti & Parker 2004); spectral
     fatigue indicators (MNF, MDF, Dimitrov FInsm5) are still computed from
     the native 2000 Hz signal by src/features/emg_features.py for the
     feature-based pipeline.
   - ECG, EDA, PPG-green: linear interpolation onto the 100 Hz grid.
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
- RMS envelope as standard amplitude estimator for sEMG: Konrad (2005)
  recommends a 50-100 ms RMS window; Merletti & Parker (2004, ch. 6)
  describe smoothed-rectification amplitude estimation. The RMS window
  also acts as the anti-aliasing low-pass before decimation (Oppenheim
  & Schafer 2010, ch. 4.6).
- De Luca (1997) — bandpass 20-450 Hz for surface EMG.
- IEC 60601-2-40 / Merletti & Parker (2004) — 50 Hz notch for power-line
  interference (Norway/EU mains frequency).
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt


# ---------------------------------------------------------------------------
# EMG envelope parameters
# ---------------------------------------------------------------------------
FS_EMG_NATIVE = 2000      # Hz — verified against metadata.json on every recording
EMG_BAND_LOW_HZ = 20.0    # De Luca 1997
EMG_BAND_HIGH_HZ = 450.0  # De Luca 1997
EMG_NOTCH_HZ = 50.0       # Norway mains frequency
EMG_NOTCH_Q = 30
EMG_RMS_WINDOW_MS = 50.0  # Konrad 2005 — 50 ms gives ~20 Hz LP, safe for 100 Hz grid


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
# EMG envelope (filter + RMS) — runs at native 2000 Hz before grid alignment
# ---------------------------------------------------------------------------

def _nanfill(signal: np.ndarray) -> np.ndarray:
    """Forward + backward fill NaN values. Required before IIR filtering
    because filtfilt/sosfiltfilt propagate NaN through the whole signal."""
    out = signal.copy().astype(float)
    mask = np.isnan(out)
    if not mask.any():
        return out
    if mask.all():
        return out
    idx = np.arange(len(out))
    fp = np.maximum.accumulate(np.where(~mask, idx, 0))
    out[mask] = out[fp[mask]]
    mask2 = np.isnan(out)
    if mask2.any():
        rev_idx = np.where(~mask2, idx, len(out) - 1)[::-1]
        bp = np.minimum.accumulate(rev_idx)[::-1]
        out[mask2] = out[bp[mask2]]
    return out


def emg_envelope(
    signal: np.ndarray,
    fs: int = FS_EMG_NATIVE,
    band_hz: tuple[float, float] = (EMG_BAND_LOW_HZ, EMG_BAND_HIGH_HZ),
    notch_hz: float = EMG_NOTCH_HZ,
    notch_q: float = EMG_NOTCH_Q,
    rms_window_ms: float = EMG_RMS_WINDOW_MS,
) -> np.ndarray:
    """Bandpass + notch + RMS-envelope at native rate.

    Pipeline (offline, zero-phase):
        nan-fill → Butterworth bandpass 20-450 Hz (sosfiltfilt)
        → IIR notch 50 Hz (filtfilt) → squared signal
        → centered moving-average over `rms_window_ms` → sqrt → envelope.

    The RMS window is the anti-aliasing low-pass before decimation: a 50 ms
    boxcar has its first zero at 20 Hz, so frequencies above ~20 Hz in the
    squared signal are attenuated below the 50 Hz Nyquist of the 100 Hz grid
    (Oppenheim & Schafer 2010; Konrad 2005).

    Parameters
    ----------
    signal       : 1D raw EMG samples at native fs.
    fs           : Native sample rate (Hz).
    band_hz      : Bandpass cutoffs (low, high).
    notch_hz     : Power-line notch frequency.
    notch_q      : Notch quality factor.
    rms_window_ms: RMS smoothing window in milliseconds.

    Returns
    -------
    Envelope at the same length and rate as the input.
    """
    if len(signal) == 0:
        return signal.astype(float).copy()

    x = _nanfill(signal)

    sos = butter(4, list(band_hz), btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, x)

    b_notch, a_notch = iirnotch(notch_hz, Q=notch_q, fs=fs)
    x = filtfilt(b_notch, a_notch, x)

    win_samp = max(1, int(round(rms_window_ms * fs / 1000.0)))
    if win_samp > 1:
        kernel = np.ones(win_samp, dtype=float) / win_samp
        x = np.convolve(x ** 2, kernel, mode="same")
        x = np.sqrt(np.maximum(x, 0.0))
    else:
        x = np.abs(x)

    nan_mask = np.isnan(signal)
    if nan_mask.any():
        x = x.astype(float)
        x[nan_mask] = np.nan
    return x


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
    # Biosignals — linear interpolation (ECG, EDA, PPG-green)
    # -----------------------------------------------------------------------
    for src_df, col in [
        (ecg_df, "ecg"),
        (eda_df, "eda"),
        (ppg_green_df, "ppg_green"),
    ]:
        t = src_df["timestamp"].to_numpy(dtype=float)
        v = src_df[col].to_numpy(dtype=float)
        df[col] = _resample_linear(t, v, grid_t)

    # EMG — bandpass + notch + RMS envelope at native 2000 Hz, then resample
    # the envelope to the 100 Hz grid. The RMS window is the anti-aliasing
    # low-pass for the 20:1 decimation, so the raw NN models read a clean
    # amplitude representation instead of an aliased version of the raw
    # 2000 Hz signal. Spectral fatigue features (MNF/MDF/Dimitrov) are still
    # computed on the native 2000 Hz CSV by src/features/emg_features.py.
    emg_t = emg_df["timestamp"].to_numpy(dtype=float)
    emg_v = emg_df["emg"].to_numpy(dtype=float)
    emg_env = emg_envelope(emg_v, fs=FS_EMG_NATIVE)
    df["emg"] = _resample_linear(emg_t, emg_env, grid_t)

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
    exercises: list[str | None],   # canonical-aligned (one per canonical set)
    rpe_list: list[int | None],    # canonical-aligned (one per canonical set)
    joint_angle_dfs: dict[int, pd.DataFrame | None],  # set_num -> angle df
    exercise_for_set: dict[int, str],  # set_num -> exercise name
) -> pd.DataFrame:
    """Build the per-100Hz-sample label array for one session.

    Parameters
    ----------
    grid_t:         100 Hz timestamp array.
    canonical_sets: List of SetMarker objects (canonical 12 sets).
    exercises:      Exercise per canonical set, already mapped from
                    Participants.xlsx using orig marker number (sm.set_num).
                    Same length as canonical_sets.
    rpe_list:       RPE per canonical set, same alignment as `exercises`.
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

        # Sets where markers say 0 reps are almost always wrong recordings
        # (aborted attempts, button never pressed, mis-numbered sets). Mark
        # them inactive so they neither train nor evaluate any task. The
        # qc_report still shows the set boundaries from kinect_sets so the
        # exclusion is visible.
        if sm.n_reps == 0:
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

        # Rep timing: prefer joint-angle extrema (precise per-rep peak/valley
        # times) over marker button-presses (count-accurate but ±0.5-1s timing
        # imprecise). The joint detector is called with target_n_reps=sm.n_reps
        # in run.py, so when it succeeds it returns EXACTLY sm.n_reps reps
        # anchored at the actual extrema. We use those times for both
        # rep_count_in_set transitions and rep_density_hz boundaries.
        #
        # Fallback to marker times only when joint detection failed
        # (NaN frames > 50%, no extrema found, or returned-count != sm.n_reps).
        # The marker fallback preserves the exact rep count but with looser
        # timing (still acceptable for soft_window rep targets, but biases
        # phase boundaries by ~0.5-1s).
        jdf = joint_angle_dfs.get(sm.set_num)
        joint_count = 0
        if jdf is not None and len(jdf) > 0 and "rep_count_in_set" in jdf.columns:
            jr_full = jdf["rep_count_in_set"].to_numpy(dtype=float)
            if len(jr_full) > 0 and np.isfinite(jr_full).any():
                joint_count = int(np.nanmax(jr_full))

        if joint_count == sm.n_reps:
            # Joint detector produced the correct count → use its extrema
            # times for rep boundaries (precise timing).
            jt = jdf["t_unix"].to_numpy(dtype=float)
            jr = jdf["rep_count_in_set"].to_numpy(dtype=float)
            resampled = _resample_nearest(jt, jr, grid_t[indices])
            rep_count_arr[indices] = resampled
            rep_index_arr[indices] = np.where(resampled > 0,
                                                resampled, np.nan)
            rep_boundaries = _rep_boundaries_from_joint(jdf)
            _fill_rep_density_hz(grid_t, indices, sm.start_unix,
                                  rep_boundaries, rep_density_arr)
        else:
            # Joint detection failed or returned wrong count → fall back to
            # marker times. Count is preserved (markers is truth); timing
            # may be ±0.5-1 s off the actual extrema.
            if sm.rep_markers:
                rep_times = sorted([r.unix_time for r in sm.rep_markers])
            else:
                step = (sm.end_unix - sm.start_unix) / sm.n_reps
                rep_times = [sm.start_unix + (k + 1) * step
                              for k in range(sm.n_reps)]
            for gi in indices:
                t = grid_t[gi]
                passed = sum(1 for rt in rep_times if t >= rt)
                rep_count_arr[gi] = passed
                rep_index_arr[gi] = passed if passed > 0 else np.nan
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
