---
name: joint-angle-labeling
description: Use to derive phase (concentric/eccentric/isometric) and rep-count ground-truth labels from joint-angle data. Handles per-exercise primary-joint selection, peak/valley detection on angle traces, and alignment with biosignal timestamps. Used ONLY in offline labeling, never in real-time inference.
---

# Joint-Angle Labeling

You have joint-angle data only during active sets, and you use it to create ground-truth labels for phase and rep count. The trained model later predicts these from biosignals alone.

## Per-exercise joint mapping

`configs/exercises.yaml`:
```yaml
squat:
  primary_joint: knee_angle
  bottom_value_deg: 60      # smallest angle at deepest position
  top_value_deg: 175        # largest angle at standing
  rom_min_deg: 70           # rep must traverse at least this much

bench_press:
  primary_joint: elbow_angle
  bottom_value_deg: 80
  top_value_deg: 175
  rom_min_deg: 60

pushup:
  primary_joint: elbow_angle
  bottom_value_deg: 80
  top_value_deg: 175
  rom_min_deg: 60

deadlift:
  primary_joint: hip_angle
  bottom_value_deg: 95
  top_value_deg: 175
  rom_min_deg: 50
```

Adjust per your subjects. Bottom and top are descriptive — what matters is the cycle.

## Algorithm: rep + phase from joint angle

Within an active set:

1. **Smooth** the angle trace: 100 ms moving average (offline, so any filter is fine).
2. **Find peaks and valleys** with `scipy.signal.find_peaks` (legitimate to use offline — we're labeling ground truth, not running real-time).
3. **Validate** each candidate cycle:
   - ROM (peak − valley) > `rom_min_deg`
   - Cycle duration between 0.8 s and 6 s (filters out noise + stalls)
4. **Pair** valley → peak → valley as one rep.
5. **Label phase per timestep** based on angular velocity:
   - `eccentric`: angular velocity moving toward bottom_value
   - `concentric`: angular velocity moving toward top_value
   - `isometric`: |angular velocity| < threshold (configurable, ~10°/s default)

## Reference implementation

```python
import numpy as np
from scipy.signal import find_peaks, savgol_filter

def label_set_from_joint(angle, t, exercise_cfg, fs):
    """
    Label one active set's worth of data.
    Returns: phase (str array), rep_index (int array), velocity (float array).
    """
    # Smooth (offline OK)
    smooth = savgol_filter(angle, window_length=int(0.1 * fs) | 1, polyorder=2)
    velocity = np.gradient(smooth, 1.0 / fs)  # deg/s

    # Determine sign convention: are peaks at top (extension) or bottom (flexion)?
    # For our convention (top_value_deg > bottom_value_deg), peaks = top, valleys = bottom
    top_v = exercise_cfg['top_value_deg']
    bot_v = exercise_cfg['bottom_value_deg']
    rom_min = exercise_cfg['rom_min_deg']

    # Find peaks (top of motion)
    peak_idx, _ = find_peaks(smooth,
                              prominence=rom_min * 0.7,
                              distance=int(0.5 * fs))
    # Find valleys (bottom of motion)
    valley_idx, _ = find_peaks(-smooth,
                                prominence=rom_min * 0.7,
                                distance=int(0.5 * fs))

    # Sort all extrema by time
    extrema = sorted([(i, 'peak') for i in peak_idx] +
                      [(i, 'valley') for i in valley_idx])

    # Build reps from valley → peak → valley sequences
    reps = []  # list of (valley_start, peak, valley_end)
    i = 0
    while i < len(extrema) - 2:
        a, b, c = extrema[i], extrema[i+1], extrema[i+2]
        if a[1] == 'valley' and b[1] == 'peak' and c[1] == 'valley':
            rom_up = smooth[b[0]] - smooth[a[0]]
            rom_down = smooth[b[0]] - smooth[c[0]]
            if rom_up > rom_min and rom_down > rom_min:
                reps.append((a[0], b[0], c[0]))
                i += 2
                continue
        i += 1

    # Assign labels
    phase = np.full(len(angle), 'isometric', dtype=object)
    rep_index = np.full(len(angle), -1, dtype=int)
    iso_thresh = 10.0  # deg/s

    for r_idx, (v_start, peak, v_end) in enumerate(reps):
        # Eccentric typically comes first (descending) — but depends on exercise
        # For squat/bench/pushup: rep starts standing/up, eccentric is first
        # For deadlift: rep starts at bottom, concentric is first (handled by valley-first)
        # Use velocity sign: positive velocity = angle increasing = moving toward top
        for k in range(v_start, v_end + 1):
            v = velocity[k]
            if abs(v) < iso_thresh:
                phase[k] = 'isometric'
            elif v > 0:
                phase[k] = 'concentric'  # moving toward top
            else:
                phase[k] = 'eccentric'   # moving toward bottom
            rep_index[k] = r_idx

    return phase, rep_index, velocity, len(reps)
```

## Alignment with biosignal timestamps

Biosignals and joint-angles likely have different sample rates. Align by interpolation onto a common time base (typically the biosignal time):

```python
from scipy.interpolate import interp1d

def align_joint_to_biosignal(joint_t, joint_angle, biosignal_t):
    """Interpolate joint angle onto biosignal timestamps. Outside joint_t range -> NaN."""
    f = interp1d(joint_t, joint_angle, bounds_error=False, fill_value=np.nan)
    return f(biosignal_t)
```

Then run the labeling on the aligned series within each active set.

## Edge cases

1. **Subject doesn't reach full ROM** (e.g., quarter squats): `rom_min_deg` filter rejects them. Fix: lower threshold OR mark set as low-quality and exclude from rep-count training.
2. **Tempo work** (very slow eccentrics): default `distance=0.5*fs` between peaks may be too short. Increase to 1 s for tempo training.
3. **Exercise has multiple joints moving** (e.g., deadlift = hip + knee): use the joint with largest ROM as primary. List both in config and pick at runtime via `max(angle_range)`.
4. **Bilateral mismatch** (subject favors one side): use the more dominant side or average bilateral. Configurable per exercise.
5. **Joint data has gaps within a set** (occlusion in optical tracking): interpolate up to 200 ms; longer gaps → split set into sub-segments.

## Validation against participants.xlsx

After deriving rep counts:
- If derived count differs from expected (8–10) by > 2, flag in quality report
- If three sets of same exercise have wildly different rep counts: subject-specific issue
- Save derived counts to `data/labeled/<subject>/<session>/derived_rep_counts.csv` for audit

## Output schema

For each active set, append to the per-timestep aligned dataset:
- `phase_label`: str in {'concentric', 'eccentric', 'isometric'}
- `rep_index`: int (0-indexed within set, -1 outside reps)
- `joint_velocity`: float (deg/s)
- `rep_count_in_set`: int (cumulative count up to this timestep)

Outside active sets: phase_label='rest', rep_index=-1, joint columns NaN.

## References

For joint-angle-derived phase and rep labels:

- **González-Badillo & Sánchez-Medina 2010** — VBT framework for rep cycles
- **Sánchez-Medina & González-Badillo 2011** — concentric vs eccentric phase definitions in resistance training

Full entries in `literature-references` skill. Never invent.
