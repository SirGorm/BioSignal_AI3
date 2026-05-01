"""
Compute primary joint angles from Azure Kinect K4ABT skeleton frames.

Azure Kinect K4ABT joint indices (verified against recording_012/recording_01_joints.json):
  0  PELVIS
  1  SPINE_NAVEL
  2  SPINE_CHEST
  3  NECK
  4  CLAVICLE_LEFT
  5  SHOULDER_LEFT
  6  ELBOW_LEFT
  7  WRIST_LEFT
  8  HAND_LEFT
  9  HANDTIP_LEFT
  10 THUMB_LEFT
  11 CLAVICLE_RIGHT
  12 SHOULDER_RIGHT
  13 ELBOW_RIGHT
  14 WRIST_RIGHT
  15 HAND_RIGHT
  16 HANDTIP_RIGHT
  17 THUMB_RIGHT
  18 HIP_LEFT
  19 KNEE_LEFT
  20 ANKLE_LEFT
  21 FOOT_LEFT
  22 HIP_RIGHT
  23 KNEE_RIGHT
  24 ANKLE_RIGHT
  25 FOOT_RIGHT
  26 HEAD
  27 NOSE
  28 EYE_LEFT
  29 EAR_LEFT
  30 EYE_RIGHT
  31 EAR_RIGHT

Joint angle computation uses the vector dot-product formula:
    angle = arccos( (v1 · v2) / (|v1| * |v2|) )
where v1 and v2 are the two bone vectors meeting at the vertex joint.

We average LEFT and RIGHT side angles to reduce noise and handle partial
tracking failures (Fukuchi et al. 2018, Sensors).

For phase labeling (concentric/eccentric/isometric), we compute the
first-order temporal derivative of the angle trace and threshold it.
The isometric threshold is 5 deg/s — below this angular velocity the
joint is considered not moving meaningfully (De Luca et al. 1997;
Cram & Kasman 1998 — though their thresholds apply to EMG, the kinematic
threshold of 5 deg/s is widely used in biomechanics software such as
Visual3D and C-Motion).

References
----------
- Fukuchi et al. 2018 — bilateral averaging for lower-limb kinematics:
  Fukuchi CA et al. (2018). "A public dataset of overground and treadmill
  walking kinematics and kinetics in healthy individuals." PeerJ 6:e4640.
- De Luca CJ (1997). "The use of surface electromyography in biomechanics."
  J. Appl. Biomech. 13(2), 135-163.
- Cram JR & Kasman GS (1998). Introduction to surface electromyography.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Joint index constants (verified from recordings)
# ---------------------------------------------------------------------------

_JOINT_NAMES = [
    "PELVIS", "SPINE_NAVEL", "SPINE_CHEST", "NECK",
    "CLAVICLE_LEFT", "SHOULDER_LEFT", "ELBOW_LEFT", "WRIST_LEFT",
    "HAND_LEFT", "HANDTIP_LEFT", "THUMB_LEFT",
    "CLAVICLE_RIGHT", "SHOULDER_RIGHT", "ELBOW_RIGHT", "WRIST_RIGHT",
    "HAND_RIGHT", "HANDTIP_RIGHT", "THUMB_RIGHT",
    "HIP_LEFT", "KNEE_LEFT", "ANKLE_LEFT", "FOOT_LEFT",
    "HIP_RIGHT", "KNEE_RIGHT", "ANKLE_RIGHT", "FOOT_RIGHT",
    "HEAD", "NOSE", "EYE_LEFT", "EAR_LEFT", "EYE_RIGHT", "EAR_RIGHT",
]

# Index lookup
_IDX = {name: i for i, name in enumerate(_JOINT_NAMES)}

# Per-exercise angle triplets: (proximal, vertex, distal)
# Averaged over left and right sides.
_EXERCISE_TRIPLETS = {
    "squat": [
        # knee angle: HIP -> KNEE -> ANKLE
        ("HIP_LEFT", "KNEE_LEFT", "ANKLE_LEFT"),
        ("HIP_RIGHT", "KNEE_RIGHT", "ANKLE_RIGHT"),
    ],
    "deadlift": [
        # hip angle: PELVIS -> HIP -> KNEE
        ("PELVIS", "HIP_LEFT", "KNEE_LEFT"),
        ("PELVIS", "HIP_RIGHT", "KNEE_RIGHT"),
    ],
    "benchpress": [
        # elbow angle: SHOULDER -> ELBOW -> WRIST
        ("SHOULDER_LEFT", "ELBOW_LEFT", "WRIST_LEFT"),
        ("SHOULDER_RIGHT", "ELBOW_RIGHT", "WRIST_RIGHT"),
    ],
    "pullup": [
        # elbow angle (same as benchpress; phase direction is inverted in labeling)
        ("SHOULDER_LEFT", "ELBOW_LEFT", "WRIST_LEFT"),
        ("SHOULDER_RIGHT", "ELBOW_RIGHT", "WRIST_RIGHT"),
    ],
}

# Isometric velocity threshold (degrees/second)
_ISOMETRIC_THRESHOLD_DEG_S = 5.0
# Kinect frame rate
_KINECT_FPS = 30.0


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

def _angle_from_triplet(
    positions: list,
    prox_idx: int,
    vertex_idx: int,
    distal_idx: int,
) -> float:
    """Return the angle in degrees at the vertex joint.

    positions: list of 32 [x, y, z] values.
    """
    p = np.array(positions[prox_idx], dtype=np.float64)
    v = np.array(positions[vertex_idx], dtype=np.float64)
    d = np.array(positions[distal_idx], dtype=np.float64)

    v1 = p - v
    v2 = d - v

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return float("nan")

    cos_a = np.dot(v1, v2) / (norm1 * norm2)
    # Clamp to [-1, 1] to avoid arccos domain errors
    cos_a = max(-1.0, min(1.0, cos_a))
    return float(np.degrees(np.arccos(cos_a)))


# ---------------------------------------------------------------------------
# Frame-level extraction
# ---------------------------------------------------------------------------

def extract_angles_from_frames(
    frames: list[dict],
    exercise: str,
    set_start_unix: float,
    set_end_unix: Optional[float] = None,
    kinect_fps: float = _KINECT_FPS,
) -> pd.DataFrame:
    """Extract primary joint angle time series from a list of Kinect frames.

    Frame Unix-time anchoring (per CLAUDE.md):
      - If `set_end_unix` is provided: spread frames linearly across
        [set_start_unix, set_end_unix]. Effective fps is derived from the
        actual frame count and the known set duration. This is the
        authoritative method — Azure Kinect's effective rate in this
        dataset is ~21 fps, not the nominal 30 fps, so a fixed-fps anchor
        leaves the last ~30% of every set without joint frames.
      - If `set_end_unix` is None (legacy): fall back to fixed-fps anchor
        `t = set_start_unix + frame_id / kinect_fps`.

    The internal `timestamp_usec` field is always 0 in this dataset, so we
    cannot rely on per-frame device timestamps.

    Parameters
    ----------
    frames:          List of frame dicts from *_joints.json['frames'].
    exercise:        One of 'squat', 'deadlift', 'benchpress', 'pullup'.
    set_start_unix:  Unix epoch start time from markers.json Set:N_Start.
    set_end_unix:    Unix epoch end time from markers.json Set_N_End (or
                     metadata.kinect_sets[N-1].end_unix_time). Strongly
                     recommended.
    kinect_fps:      Fallback frame rate when set_end_unix is None.

    Returns
    -------
    DataFrame with columns:
        't_unix'           — Unix epoch float64
        'primary_joint_angle_deg' — angle in degrees (bilateral average)
    """
    exercise = exercise.lower()
    if exercise not in _EXERCISE_TRIPLETS:
        raise ValueError(
            f"Unknown exercise '{exercise}'. "
            f"Valid: {list(_EXERCISE_TRIPLETS.keys())}"
        )

    triplets = _EXERCISE_TRIPLETS[exercise]
    indices = [
        (_IDX[prox], _IDX[vert], _IDX[dist])
        for prox, vert, dist in triplets
    ]

    n_frames = len(frames)
    use_linear = (
        set_end_unix is not None
        and n_frames >= 2
        and set_end_unix > set_start_unix
    )
    if use_linear:
        dt_lin = (set_end_unix - set_start_unix) / (n_frames - 1)

    rows = []
    for i, frame in enumerate(frames):
        if use_linear:
            t_unix = set_start_unix + i * dt_lin
        else:
            frame_id = int(frame["frame_id"])
            t_unix = set_start_unix + frame_id / kinect_fps

        bodies = frame.get("bodies", [])
        if not bodies:
            rows.append({"t_unix": t_unix, "primary_joint_angle_deg": float("nan")})
            continue

        # Use first body (single-person recording)
        positions = bodies[0]["joint_positions"]

        angles = []
        for prox_i, vert_i, dist_i in indices:
            a = _angle_from_triplet(positions, prox_i, vert_i, dist_i)
            if not np.isnan(a):
                angles.append(a)

        if angles:
            avg_angle = float(np.mean(angles))
        else:
            avg_angle = float("nan")

        rows.append({"t_unix": t_unix, "primary_joint_angle_deg": avg_angle})

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Joint angle derivative and phase labeling
# ---------------------------------------------------------------------------

def compute_angle_derivatives(
    angle_series: pd.Series,
    t_unix: pd.Series,
    window: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Compute angular velocity and acceleration.

    Uses central differences over a window of ±window frames to smooth
    the inherently noisy finite-difference estimates at 30 Hz.

    Parameters
    ----------
    angle_series: Primary joint angle in degrees.
    t_unix:       Corresponding Unix timestamps.
    window:       Half-window size for central differences (frames).

    Returns
    -------
    (velocity_deg_s, acceleration_deg_s2) as pd.Series.
    """
    angles = angle_series.to_numpy(dtype=float)
    times = t_unix.to_numpy(dtype=float)

    n = len(angles)
    velocity = np.full(n, np.nan)
    acceleration = np.full(n, np.nan)

    for i in range(window, n - window):
        dt = times[i + window] - times[i - window]
        if dt < 1e-9:
            continue
        velocity[i] = (angles[i + window] - angles[i - window]) / dt

    for i in range(window, n - window):
        dt = times[i + window] - times[i - window]
        if dt < 1e-9:
            continue
        dv = velocity[i + window] - velocity[i - window]
        if np.isnan(dv):
            continue
        acceleration[i] = dv / dt

    return pd.Series(velocity, index=angle_series.index), pd.Series(
        acceleration, index=angle_series.index
    )


def label_phase(
    angle_series: pd.Series,
    t_unix: pd.Series,
    exercise: str,
    isometric_threshold_deg_s: float = _ISOMETRIC_THRESHOLD_DEG_S,
    velocity_lp_hz: float = 2.0,
    min_block_s: float = 0.3,
    extrema_anchored: bool = True,
    target_n_reps: Optional[int] = None,
) -> pd.Series:
    """Label each frame as 'concentric', 'eccentric', or 'isometric'.

    Phase direction convention (from configs/exercises.yaml):
    - squat/deadlift/benchpress:
        concentric = angle increasing (moving toward top of motion, angle larger)
        eccentric  = angle decreasing (moving toward bottom, angle smaller)
    - pullup (phase_inverted=True):
        concentric = angle decreasing (arms flexing, angle smaller)
        eccentric  = angle increasing (arms extending, angle larger)

    Smoothing
    ---------
    Raw Kinect joint-position estimates jitter by ~1–3 deg even when the
    joint is essentially still, which produces many spurious zero-crossings
    in the velocity trace. We low-pass-filter the velocity at
    `velocity_lp_hz` (default 1.0 Hz — well below the ~0.3 Hz rep
    frequency) and then merge any phase block shorter than `min_block_s`
    (default 0.7 s) into its longer neighbour. This yields one clean
    ECC + CON pair per rep instead of 4–8 sub-rep flicker.

    Isometric: |smoothed angular velocity| < isometric_threshold_deg_s.

    NaN angle → phase = NaN.

    Parameters
    ----------
    angle_series: Primary joint angle in degrees.
    t_unix:       Unix timestamps.
    exercise:     Exercise name for phase inversion.
    isometric_threshold_deg_s: Threshold for isometric detection.
    velocity_lp_hz: Low-pass cutoff for the velocity signal. Set to 0
        to disable smoothing.
    min_block_s: Minimum phase block duration after thresholding. Set
        to 0 to disable merging.

    Returns
    -------
    pd.Series of str ('concentric', 'eccentric', 'isometric') or NaN.
    """
    phase_inverted = exercise.lower() == "pullup"
    angles = angle_series.to_numpy(dtype=float)
    times = t_unix.to_numpy(dtype=float)
    if len(times) >= 2:
        dt_med = float(np.median(np.diff(times)))
        fs = 1.0 / dt_med if dt_med > 1e-6 else 30.0
    else:
        fs = 30.0

    # ----------------------------------------------------------------------
    # Extrema-anchored labeling (preferred — guarantees N reps → N ECC + N CON)
    # ----------------------------------------------------------------------
    # Detect peaks and valleys of the smoothed angle. A rep cycle is
    # always valley → peak → valley. Each cycle contains exactly one
    # ECC half and one CON half (with optional ISO at the extrema).
    # This sidesteps the velocity-sign jitter problem entirely.
    if extrema_anchored:
        n = len(angles)
        if n < 30:
            return pd.Series(np.full(n, "unknown", dtype=object),
                              index=angle_series.index)

        # Reject sets with too much joint-data dropout — interpolating
        # >50% NaN produces phantom peaks from noise.
        finite = np.isfinite(angles)
        nan_frac = 1.0 - finite.sum() / n
        if nan_frac > 0.50:
            return pd.Series(np.full(n, "unknown", dtype=object),
                              index=angle_series.index)
        if finite.sum() < 30:
            return pd.Series(np.full(n, "unknown", dtype=object),
                              index=angle_series.index)

        # Smooth the angle (not the velocity) for peak detection
        from scipy.signal import butter, filtfilt, find_peaks
        a_smooth = angles.copy()
        if not finite.all():
            xp = np.where(finite)[0]
            fp = angles[finite]
            a_smooth = np.interp(np.arange(n), xp, fp)
        if velocity_lp_hz > 0 and fs > 2 * velocity_lp_hz:
            try:
                b, a = butter(4, velocity_lp_hz, btype="low", fs=fs)
                a_smooth = filtfilt(b, a, a_smooth)
            except ValueError:
                pass

        # Peak/valley detection: prominence ≥ 25% of full angle range
        # rejects sub-rep wobble; min_distance ≥ 0.8 s prevents double-
        # counting at top/bottom of motion (typical strength rep ≥ 1.5 s).
        sig_range = float(np.nanmax(a_smooth) - np.nanmin(a_smooth))
        if sig_range < 5.0:
            return pd.Series(np.full(n, "isometric", dtype=object),
                              index=angle_series.index)
        # Per-exercise tuning — chosen by observed outlier patterns
        # across all 9 recordings (see segmentation_qc/SUMMARY.md):
        #
        #   squat: knee angle spans ~80-130°. Lifters tend to wobble at
        #     bottom (sticking point) — needs strict prominence (40%) and
        #     longer min-distance (1.2 s) to ignore sub-rep oscillation.
        #
        #   deadlift: hip angle spans ~30-50° in clean sets, ~8-15° when
        #     Kinect mis-tracks. Lower prominence (22%) preserves real
        #     peaks; reject sets with range < 15° as "unknown" (tracking
        #     failed — e.g., rec_007 sets 7-9, rec_014 sets 7-8 all had
        #     8-13° range with 7-14 reps, an impossible combination).
        #
        #   pullup: elbow angle spans ~50-150°. Default 35% works.
        ex_lower = exercise.lower()
        if ex_lower == "squat":
            prom_frac, prom_floor, min_dist_s = 0.40, 12.0, 1.2
            min_range = 30.0
        elif ex_lower == "deadlift":
            prom_frac, prom_floor, min_dist_s = 0.22, 5.0, 1.0
            min_range = 10.0
        elif ex_lower == "pullup":
            prom_frac, prom_floor, min_dist_s = 0.35, 10.0, 1.0
            min_range = 20.0
        else:  # benchpress falls back to acc-based path before reaching here
            prom_frac, prom_floor, min_dist_s = 0.35, 10.0, 1.0
            min_range = 20.0

        if sig_range < min_range:
            # Kinect tracking failed for this exercise — return "unknown"
            # rather than fabricate phases from noise.
            return pd.Series(np.full(n, "unknown", dtype=object),
                              index=angle_series.index)
        min_dist = max(3, int(round(min_dist_s * fs)))

        # Find valleys (= bottom of each rep cycle for non-inverted, or
        # top of pullup cycle for inverted). Adaptively pick N most
        # prominent valleys when target_n_reps is provided.
        if phase_inverted:
            search_signal = a_smooth   # for pullup, reps are between peaks
        else:
            search_signal = -a_smooth  # for squat/dead/bench, between valleys

        if target_n_reps is not None and target_n_reps > 0:
            # Find candidates at low prominence threshold, then keep N most prominent
            cands, props = find_peaks(search_signal, prominence=1.0,
                                        distance=min_dist)
            if len(cands) >= target_n_reps:
                keep = np.argsort(props["prominences"])[-target_n_reps:]
                anchors = np.sort(cands[keep])
            elif len(cands) > 0:
                # Fewer candidates than target — keep what we have
                anchors = cands
            else:
                anchors = np.array([], dtype=int)
        else:
            anchors = find_peaks(search_signal,
                                  prominence=max(sig_range * prom_frac, prom_floor),
                                  distance=min_dist)[0]

        if len(anchors) == 0:
            return pd.Series(np.full(n, "unknown", dtype=object),
                              index=angle_series.index)

        # Each rep cycle is bounded by consecutive anchors (valleys for
        # non-inverted, peaks for inverted). Inside each cycle, we split
        # at the local extremum of the OPPOSITE type — for non-inverted,
        # the peak in between two valleys; for inverted, the valley
        # between two peaks.
        opposite_signal = -search_signal
        phases = np.full(n, "unknown", dtype=object)

        # Phase before the first anchor: in a non-inverted exercise, we
        # start at the top of motion → going down to first valley = ecc.
        # In an inverted (pullup) we start at the bottom → going up to
        # first peak = concentric.
        if not phase_inverted:
            lbl_before, lbl_after = "eccentric", "concentric"
            # Non-inverted: between valleys, first half = CON (up to peak),
            # second half = ECC (down to next valley)
            in_cycle_first, in_cycle_second = "concentric", "eccentric"
        else:
            lbl_before, lbl_after = "concentric", "eccentric"
            # Pullup: between peaks (= top of pullup), first half = ECC
            # (extending down to valley), second half = CON (flexing up
            # to next peak)
            in_cycle_first, in_cycle_second = "eccentric", "concentric"

        # Pre-first-anchor: ramp toward first anchor
        if anchors[0] > 0:
            phases[:anchors[0]] = lbl_before

        for i in range(len(anchors) - 1):
            s = anchors[i]
            e = anchors[i + 1]
            # Find the opposite-type extremum between s and e (highest
            # peak between two valleys, or deepest valley between two peaks)
            seg = opposite_signal[s:e]
            if len(seg) > 2:
                mid = s + int(np.argmax(seg))
            else:
                mid = s + (e - s) // 2  # degenerate fallback
            phases[s:mid] = in_cycle_first
            phases[mid:e] = in_cycle_second

        # Post-last-anchor: ramp away (one half-cycle)
        if anchors[-1] < n:
            phases[anchors[-1]:] = lbl_after

        # Note: we deliberately do NOT mark NaN-angle samples as "unknown"
        # here. The angle was already interpolated through gaps before
        # peak detection (line ~388), so phase labels are continuous on
        # the smoothed signal. Inserting "unknown" at NaN indices would
        # fragment a single ECC/CON block into multiple blocks separated
        # by "unknown", inflating the block count for sets with frequent
        # Kinect dropouts (e.g. recording_006 squat sets had 48-70% NaN
        # producing 20+ blocks for 8 reps before this fix).

        return pd.Series(phases, index=angle_series.index)

    # ----------------------------------------------------------------------
    # Fallback: velocity-sign threshold + smoothing + merge
    # (kept for backward-compat / debugging)
    # ----------------------------------------------------------------------
    velocity, _ = compute_angle_derivatives(angle_series, t_unix)

    vel_arr = velocity.to_numpy(dtype=float).copy()

    # Low-pass smooth the velocity signal to suppress sub-rep jitter.
    # Need fs > 2 * lp_hz. With Kinect ~30 Hz this is comfortable.
    if velocity_lp_hz > 0 and fs > 2 * velocity_lp_hz:
        from scipy.signal import butter, filtfilt
        finite = np.isfinite(vel_arr)
        if finite.sum() >= 30:  # filtfilt needs ~3*order samples
            b, a = butter(4, velocity_lp_hz, btype="low", fs=fs)
            vv = vel_arr[finite]
            try:
                vv_smoothed = filtfilt(b, a, vv)
                vel_arr[finite] = vv_smoothed
            except ValueError:
                pass  # too few valid samples — skip smoothing

    angles = angle_series.to_numpy(dtype=float)
    phases = np.empty(len(vel_arr), dtype=object)
    for i, (vel, ang) in enumerate(zip(vel_arr, angles)):
        if np.isnan(ang) or np.isnan(vel):
            phases[i] = "unknown"
            continue
        if abs(vel) < isometric_threshold_deg_s:
            phases[i] = "isometric"
        elif vel > 0:
            phases[i] = "eccentric" if phase_inverted else "concentric"
        else:
            phases[i] = "concentric" if phase_inverted else "eccentric"

    # Merge sub-rep flicker. _merge_short_phase_blocks is defined later
    # in this module (forward reference is fine — both live at module top).
    if min_block_s > 0:
        phases = _merge_short_phase_blocks(phases, fs, min_block_s)

    # Restore NaN sentinels where original angle was NaN
    out = []
    for p, ang in zip(phases, angles):
        if np.isnan(ang):
            out.append(float("nan"))
        else:
            out.append(p)
    return pd.Series(out, index=angle_series.index)


# ---------------------------------------------------------------------------
# Acc-based phase fallback (used when joint angles are unavailable, e.g.
# benchpress where Kinect cannot see the elbow under the lifter's torso)
# ---------------------------------------------------------------------------

def _merge_short_phase_blocks(
    phases: np.ndarray, fs: float, min_block_s: float
) -> np.ndarray:
    """Merge any contiguous phase block shorter than `min_block_s` into the
    longer of its two neighbouring blocks. Boundary singletons are merged
    into the only neighbour they have. Iterates until stable.
    """
    min_n = max(1, int(round(min_block_s * fs)))
    out = phases.copy()
    while True:
        # Identify run starts/ends
        n = len(out)
        if n == 0:
            return out
        starts = [0]
        for i in range(1, n):
            if out[i] != out[i - 1]:
                starts.append(i)
        starts.append(n)
        # Find shortest sub-min block; merge it into longer neighbour
        runs = []
        for r in range(len(starts) - 1):
            runs.append((starts[r], starts[r + 1], out[starts[r]]))
        # Find first run shorter than min_n
        target = None
        for s, e, lbl in runs:
            if e - s < min_n:
                target = (s, e, lbl)
                break
        if target is None:
            return out
        s, e, lbl = target
        # Identify neighbours
        prev_lbl = out[s - 1] if s > 0 else None
        next_lbl = out[e] if e < n else None
        if prev_lbl is None and next_lbl is None:
            # Whole array is one block of unknown — bail out
            return out
        if prev_lbl is None:
            new_lbl = next_lbl
        elif next_lbl is None:
            new_lbl = prev_lbl
        else:
            # Merge with longer neighbour to favor stability
            prev_len = 0
            for ss, ee, ll in runs:
                if ee == s:
                    prev_len = ee - ss
                    break
            next_len = 0
            for ss, ee, ll in runs:
                if ss == e:
                    next_len = ee - ss
                    break
            new_lbl = prev_lbl if prev_len >= next_len else next_lbl
        out[s:e] = new_lbl


def label_phase_from_acc(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    t_unix: np.ndarray,
    exercise: str,
    fs: float = 100.0,
    velocity_threshold_m_s: float = 0.05,
    velocity_lp_hz: float = 0.7,
    min_block_s: float = 0.7,
    target_n_reps: Optional[int] = None,
) -> np.ndarray:
    """Derive phase (concentric/eccentric/isometric) from a wrist-worn
    3-axis accelerometer when joint-angle data is unavailable.

    Approach
    --------
    1. Estimate gravity vector by low-passing each axis at 0.3 Hz (gravity
       is the only quasi-DC component during exercise; Karantonis et al.
       2006 used 0.25 Hz for the same purpose).
    2. Project the dynamic acceleration on the gravity unit vector to get
       signed vertical acceleration (positive = upward, opposite gravity).
    3. Band-pass-filter (0.3–3 Hz) to keep the rep-frequency band.
    4. Integrate to vertical velocity, then high-pass at 0.3 Hz to remove
       integration drift.
    5. Threshold the signed velocity:
         |v| < velocity_threshold     → "isometric"
         v > +velocity_threshold      → "concentric" (wrist moving up)
         v < -velocity_threshold      → "eccentric"  (wrist moving down)

    Phase convention agrees with `label_phase()` for all four exercises
    in this project: in every case the wrist moves upward during the
    concentric portion (lifting against gravity) and downward during the
    eccentric portion. For pullup the body rises while the wrist is
    fixed-ish on the bar, but the wrist still translates upward as the
    body comes up — so the upward = concentric rule still holds.

    Parameters
    ----------
    ax, ay, az : 3-axis accel in g or m/s² (units cancel out — only sign
        matters for phase). Same length, same timestamps.
    t_unix     : Unix-epoch timestamps. Used only for length validation.
    exercise   : Exercise name (currently unused — kept for API parity
        with `label_phase()` in case future per-exercise tuning is needed).
    fs         : Sample rate in Hz.
    velocity_threshold_m_s : Magnitude threshold for "isometric" call.
        Note: with `cumsum / fs` integration of band-passed accel, the
        units of the resulting "velocity" are not strictly m/s but a
        dimensionally-consistent surrogate; the threshold is empirically
        tuned for ~0.05 m/s wrist motion at typical strength tempos.

    Returns
    -------
    np.ndarray of dtype object with values "concentric" / "eccentric" /
    "isometric" / "unknown". "unknown" is returned only when input length
    is too short to filter (< 50 samples).

    References
    ----------
    - Karantonis DM et al. (2006). Implementation of a real-time human
      movement classifier using a triaxial accelerometer for ambulatory
      monitoring. *IEEE Trans. Inf. Technol. Biomed.* 10(1), 156–167.
      [low-pass cutoff for gravity estimation from wrist accel]
    - Bonomi AG et al. (2009). Detection of type, duration, and intensity
      of physical activity using an accelerometer. *MSSE* 41(9), 1770–1777.
      [rep-frequency band for strength training motion]
    - González-Badillo & Sánchez-Medina (2010). Movement velocity as
      loading-intensity measure. *Int J Sports Med* 31(5), 347–352.
      [phase-from-velocity-sign convention used throughout this codebase]
    """
    from scipy.signal import butter, filtfilt

    n = len(ax)
    if n < 50:
        return np.full(n, "unknown", dtype=object)

    # 1. Gravity = LP 0.3 Hz of each axis
    b_g, a_g = butter(2, 0.3, btype="low", fs=fs)
    gx = filtfilt(b_g, a_g, ax)
    gy = filtfilt(b_g, a_g, ay)
    gz = filtfilt(b_g, a_g, az)

    g_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    g_mag = np.maximum(g_mag, 1e-6)
    gxn, gyn, gzn = gx / g_mag, gy / g_mag, gz / g_mag

    # 2. Signed vertical accel (positive = upward, opposite to gravity)
    dx, dy, dz = ax - gx, ay - gy, az - gz
    a_vert = -(dx * gxn + dy * gyn + dz * gzn)

    # 3. Band-pass to rep band (lower cutoff = HP for drift, upper = LP
    # to suppress jitter that would cause spurious phase flips)
    b_bp, a_bp = butter(4, [0.3, max(velocity_lp_hz, 0.5)],
                         btype="band", fs=fs)
    a_vert_bp = filtfilt(b_bp, a_bp, a_vert)

    # 4. Integrate to vertical velocity, HP to remove drift, then LP again
    # to clean up the velocity signal itself (Karantonis 2006 §III)
    v = np.cumsum(a_vert_bp) / fs
    b_hp, a_hp = butter(4, 0.3, btype="high", fs=fs)
    v = filtfilt(b_hp, a_hp, v)
    b_lp, a_lp = butter(4, velocity_lp_hz, btype="low", fs=fs)
    v = filtfilt(b_lp, a_lp, v)

    # 5+6. Anchor-based labeling using velocity peaks/valleys.
    # When target_n_reps is given, pick the N most prominent positive
    # peaks and N most prominent negative peaks (valleys) of the velocity
    # signal; between consecutive valleys lies one CON segment, between
    # consecutive peaks lies one ECC segment. This matches the joint-
    # angle anchor approach and guarantees N ECC + N CON blocks exactly.
    from scipy.signal import find_peaks
    if target_n_reps is not None and target_n_reps > 0:
        # CON anchors = positive velocity peaks (wrist moving up — middle
        # of concentric phase). ECC anchors = negative velocity peaks
        # (wrist moving down — middle of eccentric phase).
        min_dist_samples = max(1, int(round(0.8 * fs)))  # min 0.8 s between same-type peaks
        pos_idx, pos_props = find_peaks(v, distance=min_dist_samples)
        neg_idx, neg_props = find_peaks(-v, distance=min_dist_samples)

        if len(pos_idx) >= target_n_reps and len(neg_idx) >= target_n_reps:
            # Keep N most prominent of each
            pos_keep = np.argsort(pos_props.get("prominences", v[pos_idx]))[-target_n_reps:]
            neg_keep = np.argsort(neg_props.get("prominences", -v[neg_idx]))[-target_n_reps:]
            con_centers = np.sort(pos_idx[pos_keep])
            ecc_centers = np.sort(neg_idx[neg_keep])

            # Build phases by interleaving: each rep cycle alternates
            # ECC (down) and CON (up). Sort all centers by index and
            # assign segments based on which center we're between.
            all_centers = sorted(
                [(c, "concentric") for c in con_centers]
                + [(c, "eccentric") for c in ecc_centers]
            )
            phases = np.full(n, "isometric", dtype=object)
            # Pre-region: before first center, label = first center type
            first_idx, first_lbl = all_centers[0]
            phases[:first_idx] = first_lbl
            # Each segment between centers: split halfway, first half =
            # type of left center, second half = type of right center
            for i in range(len(all_centers) - 1):
                s, sl = all_centers[i]
                e, el = all_centers[i + 1]
                if sl == el:
                    # same type both sides — fill with that label
                    phases[s:e] = sl
                else:
                    mid = (s + e) // 2
                    phases[s:mid] = sl
                    phases[mid:e] = el
            # Post-region
            last_idx, last_lbl = all_centers[-1]
            phases[last_idx:] = last_lbl
            return phases

    # Fallback (no target, or insufficient peaks): velocity-sign threshold + merge
    phases = np.full(n, "isometric", dtype=object)
    phases[v > velocity_threshold_m_s] = "concentric"
    phases[v < -velocity_threshold_m_s] = "eccentric"
    if min_block_s > 0:
        phases = _merge_short_phase_blocks(phases, fs, min_block_s)
    return phases


# ---------------------------------------------------------------------------
# Wrist-IMU vertical velocity (shared by acc-based phase + acc-based rep
# detection so the two stay consistent — both see the same velocity trace).
# ---------------------------------------------------------------------------

def compute_wrist_vertical_velocity(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    fs: float = 100.0,
    velocity_lp_hz: float = 0.7,
) -> np.ndarray:
    """Return signed vertical wrist velocity (positive = upward).

    Same pipeline as `label_phase_from_acc` (Karantonis 2006 §III):
        1. LP 0.3 Hz to estimate gravity per-axis
        2. Project dynamic accel onto gravity unit vector → signed vertical
        3. Band-pass 0.3-velocity_lp_hz to keep rep frequency band
        4. Integrate, HP, LP to clean up
    Returns np.zeros if input is shorter than 50 samples.
    """
    from scipy.signal import butter, filtfilt

    n = len(ax)
    if n < 50:
        return np.zeros(n, dtype=float)

    b_g, a_g = butter(2, 0.3, btype="low", fs=fs)
    gx = filtfilt(b_g, a_g, ax)
    gy = filtfilt(b_g, a_g, ay)
    gz = filtfilt(b_g, a_g, az)

    g_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    g_mag = np.maximum(g_mag, 1e-6)
    gxn, gyn, gzn = gx / g_mag, gy / g_mag, gz / g_mag

    dx, dy, dz = ax - gx, ay - gy, az - gz
    a_vert = -(dx * gxn + dy * gyn + dz * gzn)

    b_bp, a_bp = butter(4, [0.3, max(velocity_lp_hz, 0.5)],
                         btype="band", fs=fs)
    a_vert_bp = filtfilt(b_bp, a_bp, a_vert)

    v = np.cumsum(a_vert_bp) / fs
    b_hp, a_hp = butter(4, 0.3, btype="high", fs=fs)
    v = filtfilt(b_hp, a_hp, v)
    b_lp, a_lp = butter(4, velocity_lp_hz, btype="low", fs=fs)
    v = filtfilt(b_lp, a_lp, v)
    return v


def count_reps_from_acc(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    t_unix: np.ndarray,
    fs: float = 100.0,
    velocity_lp_hz: float = 0.7,
    min_rep_duration_s: float = 0.8,
    target_n_reps: Optional[int] = None,
) -> pd.Series:
    """Count reps from wrist-IMU vertical velocity.

    A rep = one full descent + ascent of the wrist. We anchor reps at
    NEGATIVE velocity peaks (= middle of eccentric phase, when the wrist
    is moving downward fastest). Equivalently this is the most reliable
    landmark of bottom-of-motion: at the position minimum velocity is 0,
    but the negative-velocity peak just before it is a sharp extremum
    that find_peaks resolves more reliably than a zero crossing.

    Same convention as the joint-angle anchored detector: each negative
    velocity peak = one rep event, so `max(rep_count) = N` exactly when
    `target_n_reps=N`.

    Used for benchpress where Kinect cannot see the elbow under the
    lifter's torso. González-Badillo & Sánchez-Medina (2010); Karantonis
    et al. (2006).

    Parameters
    ----------
    ax, ay, az : 3-axis wrist accel (units cancel — only sign matters).
    t_unix     : Unix-epoch timestamps. Must be same length as ax.
    fs         : Sample rate in Hz (default 100, matching dataset_aligned).
    velocity_lp_hz : Velocity LP cutoff (default 0.7 Hz, matches
        label_phase_from_acc).
    min_rep_duration_s : Minimum allowed rep period (rejects spurious
        adjacent peaks). Default 0.8 s.
    target_n_reps : When provided, return exactly N reps by selecting the
        N most prominent negative-velocity peaks. Returns all-zeros if
        fewer than N candidates exist (caller falls back to markers).

    Returns
    -------
    pd.Series of int (monotonically non-decreasing within the set), length
    matching ax. All-zeros if input is too short or insufficient peaks.
    """
    from scipy.signal import find_peaks

    n = len(ax)
    if n < 50:
        return pd.Series(np.zeros(n, dtype=int))

    v = compute_wrist_vertical_velocity(ax, ay, az, fs=fs,
                                         velocity_lp_hz=velocity_lp_hz)

    min_dist_samples = max(1, int(round(min_rep_duration_s * fs)))

    if target_n_reps is not None and target_n_reps > 0:
        # Pick the N most prominent negative velocity peaks. Fall back to
        # smaller distance constraints if not enough candidates.
        valleys = None
        for trial_dist in (min_dist_samples,
                            max(1, min_dist_samples // 2), 1):
            cands, props = find_peaks(-v, distance=trial_dist)
            if len(cands) >= target_n_reps:
                keep = np.argsort(
                    props.get("prominences", -v[cands])
                )[-target_n_reps:]
                valleys = np.sort(cands[keep])
                break
        if valleys is None:
            return pd.Series(np.zeros(n, dtype=int))
    else:
        # Free mode: every prominent negative velocity peak counts as a rep.
        # Prominence threshold = 0.05 (m/s surrogate units, matches the
        # "isometric" velocity_threshold in label_phase_from_acc).
        valleys, _ = find_peaks(-v, prominence=0.05,
                                distance=min_dist_samples)

    valley_times = [t_unix[v_idx] for v_idx in valleys]
    rep_count = np.zeros(n, dtype=int)
    for i in range(n):
        rep_count[i] = sum(1 for vt in valley_times if t_unix[i] >= vt)
    return pd.Series(rep_count)


# ---------------------------------------------------------------------------
# Rep counting from joint angles
# ---------------------------------------------------------------------------

# Per-exercise tuning for rep detection. Centralised so the QC visualizer
# (scripts/visualize_joint_reps.py) can render the exact same smoothed
# trace the detector operates on.
_REP_DETECTION_PARAMS = {
    "squat":      {"prom_frac": 0.40, "prom_floor": 12.0, "min_dist_s": 0.8,
                    "min_range": 30.0, "lp_hz": 1.5},
    "deadlift":   {"prom_frac": 0.22, "prom_floor": 5.0,  "min_dist_s": 1.0,
                    "min_range": 7.0, "lp_hz": 2.0},
    "pullup":     {"prom_frac": 0.35, "prom_floor": 10.0, "min_dist_s": 1.0,
                    "min_range": 20.0, "lp_hz": 1.5},
    "benchpress": {"prom_frac": 0.35, "prom_floor": 10.0, "min_dist_s": 1.0,
                    "min_range": 20.0, "lp_hz": 1.0},
}
_REP_DETECTION_DEFAULT = _REP_DETECTION_PARAMS["benchpress"]


def get_rep_detection_params(exercise: str) -> dict:
    """Return the per-exercise tuning dict used by the rep detector."""
    return _REP_DETECTION_PARAMS.get(exercise.lower(), _REP_DETECTION_DEFAULT)


def smooth_angles_for_rep_detection(
    angle_series: pd.Series,
    t_unix: pd.Series,
    exercise: str,
) -> np.ndarray:
    """Apply the same two-stage smoothing the rep detector uses internally.

    Stage 1: 3-frame median filter (kills isolated Kinect spikes).
    Stage 2: Per-exercise zero-phase Butterworth LP (suppresses jitter).

    Returns an np.ndarray the same length as `angle_series`. NaNs in the
    input are linearly interpolated before filtering. Use this from QC
    code to overlay the detector's view on top of the raw angle trace.
    """
    from scipy.signal import medfilt, butter, filtfilt

    angles = angle_series.ffill().bfill().to_numpy(dtype=float)
    times = t_unix.to_numpy(dtype=float)
    n = len(angles)
    if n < 3:
        return angles

    dt = float(np.median(np.diff(times)))
    if dt <= 0 or not np.isfinite(dt):
        return angles
    fs = 1.0 / dt

    lp_hz = get_rep_detection_params(exercise)["lp_hz"]

    a_smooth = angles.copy()
    finite = np.isfinite(angles)
    if not finite.all() and finite.any():
        xp = np.where(finite)[0]
        fp = angles[finite]
        a_smooth = np.interp(np.arange(n), xp, fp)

    a_smooth = medfilt(a_smooth, kernel_size=3)
    if fs > 2 * lp_hz:
        try:
            b, a = butter(4, lp_hz, btype="low", fs=fs)
            a_smooth = filtfilt(b, a, a_smooth)
        except ValueError:
            pass
    return a_smooth


def count_reps_from_angles(
    angle_series: pd.Series,
    t_unix: pd.Series,
    exercise: str,
    min_rep_duration_s: float = 0.8,
    target_n_reps: Optional[int] = None,
) -> pd.Series:
    """Assign a monotonically non-decreasing rep_count_in_set to each frame.

    Rep detection strategy (peak/valley based):
    - For non-inverted exercises (squat, deadlift, benchpress): a full rep is
      a valley → peak → valley cycle (angle dips to bottom, rises to top,
      returns to bottom). Rep is credited at the second valley.
    - For pullup (inverted): a full rep is a peak → valley → peak cycle.

    Convention: with N true valleys in the trace the function returns
    `max(rep_count) = N - 1`. The first valley is treated as the lifter
    descending into starting position; reps are credited at each
    subsequent valley.

    Anchored mode (`target_n_reps` set)
    -----------------------------------
    When `target_n_reps=N` is given, the detector loosens prominence to 1.0
    and selects the **N+1 most prominent valleys** (so the convention above
    yields `max(rep_count) = N` exactly). This is the right mode when an
    independent count-of-reps source (e.g. markers.json) is available and
    the goal is precise per-rep timing rather than count discovery.

    Per-exercise thresholds (free mode, no `target_n_reps`) match those used
    by `label_phase` (see lines ~420-432 of this module).

    Reference: Tao W et al. (2012). "Gait analysis using wearable sensors."
    Sensors 12(2), 2255-2283. González-Badillo & Sánchez-Medina (2010) for
    rep-cycle definition.

    Parameters
    ----------
    angle_series: Primary joint angle in degrees.
    t_unix:       Unix timestamps.
    exercise:     Exercise name (controls phase inversion + thresholds).
    min_rep_duration_s: Minimum allowed rep duration (reject spurious peaks).
    target_n_reps: When provided, force exactly this rep count by selecting
        the N+1 most prominent valleys. Returns all-zeros if fewer than
        N+1 candidate valleys can be found at the lowest prominence.

    Returns
    -------
    pd.Series of int (monotonically non-decreasing within the set). Returns
    all-zeros to signal "could not detect" — callers fall back to markers.
    """
    from scipy.signal import find_peaks

    angles = angle_series.ffill().bfill().to_numpy(dtype=float)
    times = t_unix.to_numpy(dtype=float)
    n = len(angles)

    if n < 10:
        return pd.Series(np.zeros(n, dtype=int), index=angle_series.index)

    dt = np.median(np.diff(times))
    if dt <= 0 or not np.isfinite(dt):
        return pd.Series(np.zeros(n, dtype=int), index=angle_series.index)

    sig_range = float(np.nanmax(angles) - np.nanmin(angles))
    if sig_range < 5.0:
        # Insufficient range of motion — cannot reliably detect reps
        return pd.Series(np.zeros(n, dtype=int), index=angle_series.index)

    # Per-exercise thresholds sourced from _REP_DETECTION_PARAMS at top of
    # this section. Edit that dict to tune behaviour — both the detector
    # (here) and the QC visualizer's smoothed overlay read from it, so
    # they stay in sync.
    ex_lower = exercise.lower()
    params = get_rep_detection_params(exercise)
    prom_frac = params["prom_frac"]
    prom_floor = params["prom_floor"]
    min_dist_s = params["min_dist_s"]
    min_range = params["min_range"]

    if sig_range < min_range:
        # Tracking failed — refuse to fabricate reps from noise
        return pd.Series(np.zeros(n, dtype=int), index=angle_series.index)

    # Smoothing handled by the shared helper so the QC visualizer can
    # render the exact same trace this detector operates on.
    fs = 1.0 / dt
    a_smooth = smooth_angles_for_rep_detection(angle_series, t_unix, exercise)

    # Search-signal: valleys for non-inverted, peaks-as-valleys-of-negated
    # for inverted. Both flow through find_peaks(-search_signal_orig) below.
    # Match label_phase semantics exactly: pullup uses +a_smooth, others -a_smooth.
    phase_inverted = ex_lower == "pullup"
    if phase_inverted:
        search_signal = a_smooth
    else:
        search_signal = -a_smooth

    min_dist = max(3, int(round(min_dist_s * fs)))

    if target_n_reps is not None and target_n_reps > 0:
        # Anchored mode: find candidates at very low prominence, keep the
        # N most prominent. Each valley = one rep event (no -1 trick).
        # Sets without an "approach valley" (e.g. deadlift starting from
        # the bottom position) would have failed the older N+1 convention
        # because find_peaks would only return N candidates at all.
        n_anchors = target_n_reps

        # Try progressively looser thresholds until we have enough candidates.
        # Order: per-exercise min_dist → half min_dist → no distance constraint.
        valleys = None
        for trial_dist in (min_dist, max(3, min_dist // 2), 1):
            cands, props = find_peaks(search_signal, prominence=1.0,
                                      distance=trial_dist)
            if len(cands) >= n_anchors:
                keep = np.argsort(props["prominences"])[-n_anchors:]
                valleys = np.sort(cands[keep])
                break

        if valleys is None:
            # Even at distance=1, fewer than N candidates exist — the signal
            # is genuinely too flat to support the target rep count. Caller
            # falls back to markers.
            return pd.Series(np.zeros(n, dtype=int), index=angle_series.index)

        # Anchored mode counts each valley as one rep event.
        valley_times = [times[v] for v in valleys]
        rep_count = np.zeros(n, dtype=int)
        for i in range(n):
            rep_count[i] = sum(1 for vt in valley_times if times[i] >= vt)
        return pd.Series(rep_count, index=angle_series.index)

    # Free mode: per-exercise prominence threshold. Keep the legacy
    # `passed - 1` convention because we don't know N — assuming an
    # "approach valley" is the safer default for free-running detection.
    prominence = max(sig_range * prom_frac, prom_floor)
    valleys, _ = find_peaks(
        search_signal,
        prominence=prominence,
        distance=min_dist,
    )

    valley_times = [times[v] for v in valleys]
    rep_count = np.zeros(n, dtype=int)
    for i in range(n):
        passed = sum(1 for vt in valley_times if times[i] >= vt)
        rep_count[i] = max(0, passed - 1)

    return pd.Series(rep_count, index=angle_series.index)


# ---------------------------------------------------------------------------
# Top-level: load joint file and return angle DataFrame
# ---------------------------------------------------------------------------

def load_joint_angles_for_set(
    rec_dir: Path,
    set_num: int,
    exercise: str,
    set_start_unix: float,
    set_end_unix: Optional[float] = None,
    kinect_fps: float = _KINECT_FPS,
) -> Optional[pd.DataFrame]:
    """Load the joints JSON for one set and return an angle DataFrame.

    Attempts multiple joint filename patterns:
      recording_{set_num:02d}_joints.json   (rec_003..rec_014 standard)
      recording_{set_num}_joints.json        (rec_002 without zero-padding)
      recording_{set_num:02d}_joints.json    (with offset, for rec_001)

    For rec_001 the joint files are numbered 05-16 (offset +4 relative to
    set number). We detect this by scanning the available files.

    Parameters
    ----------
    rec_dir:         Path to the recording directory.
    set_num:         1-indexed set number.
    exercise:        Exercise name for angle computation.
    set_start_unix:  Unix epoch start from markers.json.
    set_end_unix:    Unix epoch end from markers.json. When provided, frames
                     are linearly spread across [start, end] (preferred —
                     Azure Kinect effective fps is ~21, not 30, in this
                     dataset).
    kinect_fps:      Fallback frame rate when set_end_unix is None.

    Returns
    -------
    DataFrame with columns ['t_unix', 'primary_joint_angle_deg'] or None
    if no matching file found.
    """
    joint_file = _find_joint_file(rec_dir, set_num)
    if joint_file is None:
        return None

    with open(joint_file) as fh:
        data = json.load(fh)

    frames = data.get("frames", [])
    if not frames:
        return None

    return extract_angles_from_frames(
        frames,
        exercise,
        set_start_unix,
        set_end_unix=set_end_unix,
        kinect_fps=kinect_fps,
    )


def _find_joint_file(rec_dir: Path, set_num: int) -> Optional[Path]:
    """Find the joint file for a given set number, handling naming variations."""
    # Candidates in priority order
    candidates = [
        rec_dir / f"recording_{set_num:02d}_joints.json",
        rec_dir / f"recording_{set_num}_joints.json",
    ]

    for c in candidates:
        if c.exists():
            return c

    # Scan for any file whose numeric part matches set_num
    # (handles rec_001 where files are offset by 4: set1 -> recording_05)
    import re as _re
    pattern = _re.compile(r"recording_0*(\d+)_joints\.json$")
    all_joint_files = sorted(rec_dir.glob("recording_*_joints.json"))
    # Filter out 'feil' (Norwegian for 'error/wrong') files
    all_joint_files = [f for f in all_joint_files if "feil" not in f.name.lower()]

    if not all_joint_files:
        return None

    # Extract numeric indices from available joint files
    available_nums = []
    for f in all_joint_files:
        m = pattern.match(f.name)
        if m:
            available_nums.append((int(m.group(1)), f))

    available_nums.sort(key=lambda x: x[0])

    if len(available_nums) == 0:
        return None

    # If count matches: direct index (1-based set_num maps to i-th sorted file)
    if 1 <= set_num <= len(available_nums):
        return available_nums[set_num - 1][1]

    return None
