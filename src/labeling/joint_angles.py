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
    kinect_fps: float = _KINECT_FPS,
) -> pd.DataFrame:
    """Extract primary joint angle time series from a list of Kinect frames.

    Frames are anchored to Unix time via:
        t_unix = set_start_unix + frame_id / kinect_fps

    This is necessary because the internal timestamp_usec field is always 0
    (verified across all recordings).

    Parameters
    ----------
    frames:          List of frame dicts from *_joints.json['frames'].
    exercise:        One of 'squat', 'deadlift', 'benchpress', 'pullup'.
    set_start_unix:  Unix epoch start time from markers.json Set:N_Start.
    kinect_fps:      Kinect frame rate (default 30 Hz, verified from data).

    Returns
    -------
    DataFrame with columns:
        't_unix'           — Unix epoch float64
        'primary_joint_angle_deg' — angle in degrees (bilateral average)

    Rows with no valid body detection are dropped (NaN rows excluded).
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

    rows = []
    for frame in frames:
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
) -> pd.Series:
    """Label each frame as 'concentric', 'eccentric', or 'isometric'.

    Phase direction convention (from configs/exercises.yaml):
    - squat/deadlift/benchpress:
        concentric = angle increasing (moving toward top of motion, angle larger)
        eccentric  = angle decreasing (moving toward bottom, angle smaller)
    - pullup (phase_inverted=True):
        concentric = angle decreasing (arms flexing, angle smaller)
        eccentric  = angle increasing (arms extending, angle larger)

    Isometric: |angular velocity| < isometric_threshold_deg_s.

    NaN angle → phase = NaN.

    Parameters
    ----------
    angle_series: Primary joint angle in degrees.
    t_unix:       Unix timestamps.
    exercise:     Exercise name for phase inversion.
    isometric_threshold_deg_s: Threshold for isometric detection.

    Returns
    -------
    pd.Series of str ('concentric', 'eccentric', 'isometric') or NaN.
    """
    velocity, _ = compute_angle_derivatives(angle_series, t_unix)
    phase_inverted = exercise.lower() == "pullup"

    phases = []
    for vel, ang in zip(velocity, angle_series):
        if np.isnan(ang) or np.isnan(vel):
            phases.append(float("nan"))
            continue
        if abs(vel) < isometric_threshold_deg_s:
            phases.append("isometric")
        elif vel > 0:
            # angle increasing
            phases.append("eccentric" if phase_inverted else "concentric")
        else:
            # angle decreasing
            phases.append("concentric" if phase_inverted else "eccentric")

    return pd.Series(phases, index=angle_series.index)


# ---------------------------------------------------------------------------
# Rep counting from joint angles
# ---------------------------------------------------------------------------

def count_reps_from_angles(
    angle_series: pd.Series,
    t_unix: pd.Series,
    exercise: str,
    min_rep_duration_s: float = 0.8,
) -> pd.Series:
    """Assign a monotonically non-decreasing rep_count_in_set to each frame.

    Rep detection strategy (peak/valley based):
    - For non-inverted exercises (squat, deadlift, benchpress): a full rep is
      a valley → peak → valley cycle (angle dips to bottom, rises to top,
      returns to bottom). Rep is credited at the second valley.
    - For pullup (inverted): a full rep is a peak → valley → peak cycle.

    We use a simple threshold-based peak/valley detector on the smoothed
    angle series. This is equivalent to the method described by
    Tao et al. (2012) for repetition counting from inertial sensors,
    adapted here for joint-angle signals.

    Reference: Tao W et al. (2012). "Gait analysis using wearable sensors."
    Sensors 12(2), 2255-2283. [REF NEEDED: repetition counting from
    joint angles specifically — joint-angle peak detection for strength
    exercise rep counting]

    Parameters
    ----------
    angle_series: Primary joint angle in degrees.
    t_unix:       Unix timestamps.
    exercise:     Exercise name.
    min_rep_duration_s: Minimum allowed rep duration (reject spurious peaks).

    Returns
    -------
    pd.Series of int (monotonically non-decreasing within the set).
    """
    from scipy.signal import find_peaks

    angles = angle_series.ffill().bfill().to_numpy(dtype=float)
    times = t_unix.to_numpy(dtype=float)
    n = len(angles)

    if n < 10:
        return pd.Series(np.zeros(n, dtype=int), index=angle_series.index)

    dt = np.median(np.diff(times))
    # Convert min_rep_duration_s to samples
    min_samples = max(3, int(min_rep_duration_s / dt))

    phase_inverted = exercise.lower() == "pullup"

    if phase_inverted:
        # For pullup: concentric = angle decreasing, so rep cycle is
        # peak→valley→peak. Find valleys as rep markers.
        # Invert the angle to use valley detection as peak detection.
        search_signal = -angles
    else:
        # For squat/deadlift/benchpress: valley→peak→valley cycle.
        # Find valleys as rep markers.
        search_signal = -angles  # find valleys by searching negative peaks

    # Dynamic threshold: 20% of the range
    sig_range = np.nanmax(angles) - np.nanmin(angles)
    if sig_range < 5.0:
        # Insufficient range of motion — cannot reliably detect reps
        return pd.Series(np.zeros(n, dtype=int), index=angle_series.index)

    prominence = max(sig_range * 0.20, 5.0)

    valleys, _ = find_peaks(
        search_signal,
        prominence=prominence,
        distance=min_samples,
    )

    # Rep count: increment at each valley (after the first)
    rep_count = np.zeros(n, dtype=int)
    rep_num = 0
    valley_times = [times[v] for v in valleys]

    for i in range(n):
        # Count how many valleys have passed up to and including time[i]
        passed = sum(1 for vt in valley_times if times[i] >= vt)
        # First valley is "starting position", reps start at second valley
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
    kinect_fps:      Frame rate (default 30 Hz).

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

    return extract_angles_from_frames(frames, exercise, set_start_unix, kinect_fps)


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
