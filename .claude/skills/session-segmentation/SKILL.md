---
name: session-segmentation
description: Use to detect active sets vs rest periods in continuous biosignal recordings. Uses acc-magnitude as the primary signal. Outputs a list of (start, end, type) segments and validates against participants.xlsx expected set count.
---

# Session Segmentation

The 2-minute rest periods make this nearly trivial: acc-magnitude is essentially zero during rest and high during active sets. The challenge is robustness against fidgeting, equipment adjustment, and warm-up reps that aren't part of the official set.

## Algorithm

1. Compute acc-magnitude (already in labeled data as `acc_mag`).
2. Subtract gravity bias: `mag_centered = acc_mag - 9.81` (use median of first 30 s as baseline if needed).
3. Take absolute value, then smooth with a 1-second moving average (causal-OK for offline use; `np.convolve` with mode='valid' or scipy's `uniform_filter1d`).
4. Threshold-based segmentation: above `threshold_g=0.3` for at least `min_active_duration_s=20` = active.
5. Coalesce active segments separated by < 5 s of "rest" (handles brief mid-set pauses).
6. Cross-validate against expected count from participants.xlsx.

## Reference implementation (offline)

```python
import numpy as np
from scipy.ndimage import uniform_filter1d

def segment_session(acc_mag, t, fs=100, threshold_g=0.3,
                    min_active_s=20, merge_gap_s=5):
    """
    Returns list of dicts: [{'type': 'rest' | 'active', 't_start': ..., 't_end': ...}, ...]
    """
    centered = np.abs(acc_mag - np.median(acc_mag[:int(30 * fs)]))
    smooth = uniform_filter1d(centered, size=int(1.0 * fs))

    is_active = smooth > threshold_g

    # Find runs
    diff = np.diff(is_active.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if is_active[0]: starts = np.r_[0, starts]
    if is_active[-1]: ends = np.r_[ends, len(is_active)]

    # Filter by min duration
    active = [(s, e) for s, e in zip(starts, ends) if (e - s) / fs >= min_active_s]

    # Merge close segments
    merged = []
    for s, e in active:
        if merged and (s - merged[-1][1]) / fs < merge_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # Build full segment list including rest periods
    segments = []
    cursor = 0
    for s, e in merged:
        if s > cursor:
            segments.append({'type': 'rest', 't_start': t[cursor], 't_end': t[s]})
        segments.append({'type': 'active', 't_start': t[s], 't_end': t[e - 1]})
        cursor = e
    if cursor < len(t):
        segments.append({'type': 'rest', 't_start': t[cursor], 't_end': t[-1]})

    return segments
```

## Cross-validation against spreadsheet

```python
def cross_validate_segments(segments, participants_df, tolerance_s=5):
    detected_active = [s for s in segments if s['type'] == 'active']
    expected = participants_df.sort_values('set_start_time').reset_index(drop=True)

    n_det, n_exp = len(detected_active), len(expected)
    issues = []
    if n_det != n_exp:
        issues.append(f"Set count mismatch: detected {n_det}, expected {n_exp} from spreadsheet")

    matched = []
    for i, exp_row in expected.iterrows():
        # Find detected segment overlapping or near the spreadsheet window
        best = None; best_overlap = -1
        for j, seg in enumerate(detected_active):
            ovl_start = max(seg['t_start'], exp_row['set_start_time'])
            ovl_end = min(seg['t_end'], exp_row['set_end_time'])
            ovl = max(0, ovl_end - ovl_start)
            if ovl > best_overlap:
                best_overlap = ovl; best = j
        if best_overlap > 0:
            matched.append((i, best, best_overlap))
        else:
            issues.append(f"Spreadsheet set {i} ({exp_row['exercise']} #{exp_row['set_number']}) "
                          f"has no matching detected segment within tolerance")

    return matched, issues
```

## What to do on mismatch

- **Detected fewer than expected**: subject was fidgeting or doing partial reps below threshold. Either lower `threshold_g` (cautiously, will let in noise) or trust the spreadsheet times and label active = `[set_start_time, set_end_time]` from spreadsheet, ignoring detected segments.
- **Detected more than expected**: warm-up reps, unracking the bar, or adjusting equipment. Trust the spreadsheet — only label sets that match within tolerance.
- **Always**: emit warning to `quality_report.md`, never silently align.

## When acc threshold doesn't work

Some setups have residual hand motion during rest (subject scrolls phone, drinks water). If false-active rate is high:

1. Use HR signal as a secondary feature: HR during set > HR during rest
2. Use EMG amplitude as confirmation
3. Combine: active = `(acc_high) AND (emg_high OR hr_elevated)`

But first, check whether the spreadsheet's `set_start_time`/`set_end_time` columns are reliable — if they are, just use them and skip detection entirely. The detection is a sanity check, not the source of truth.

## Online (streaming) variant

For sanntid (uten spreadsheet), bruk en state machine:

```python
class SetStateMachine:
    """Detects set boundaries online from acc-magnitude.
    States: REST -> ACTIVE -> REST. Emits set start/end events."""
    def __init__(self, fs=100, threshold_g=0.3, min_active_s=10, min_rest_s=15):
        self.fs = fs; self.threshold = threshold_g
        self.min_active = int(min_active_s * fs)
        self.min_rest = int(min_rest_s * fs)
        self.state = 'REST'
        self.consec_active = 0; self.consec_rest = 0
        self.set_start_t = None
        self.events = []

    def step(self, acc_mag_chunk, t_chunk):
        events = []
        for x, t in zip(acc_mag_chunk, t_chunk):
            high = abs(x - 9.81) > self.threshold
            if high:
                self.consec_active += 1; self.consec_rest = 0
            else:
                self.consec_rest += 1; self.consec_active = 0

            if self.state == 'REST' and self.consec_active >= self.min_active:
                self.state = 'ACTIVE'
                self.set_start_t = t - self.min_active / self.fs
                events.append({'type': 'set_start', 't': self.set_start_t})
            elif self.state == 'ACTIVE' and self.consec_rest >= self.min_rest:
                self.state = 'REST'
                events.append({'type': 'set_end', 't': t - self.min_rest / self.fs,
                               'duration': t - self.min_rest / self.fs - self.set_start_t})
        return events
```

This emits set_start/set_end events with low latency (`min_active_s` after the set actually starts), suitable for the real-time pipeline.

## References

When documenting active-set detection thresholds and acc-based segmentation:

- **Bonomi et al. 2009** — acc-based activity detection (canonical reference for threshold selection)
- **Mannini & Sabatini 2010** — window-based activity classification

Full entries in `literature-references` skill. Never invent.
