---
name: motion-rep-detection
description: Use for real-time rep counting and phase classification from biosignals at inference time (when joint-angle data is unavailable). State-machine on acc-magnitude is the primary approach; ML fallback only when state-machine F1 < 0.85. Joint-angle ground truth from labeling is used to validate this state machine.
---

# Real-Time Rep Detection (no joint angles)

At inference time you don't have joint angles — only biosignals. The acc-magnitude signal alone is usually sufficient for rep counting and phase, gated by exercise label. This skill provides the state machine; the joint-angle-derived ground truth (from `joint-angle-labeling`) is your validation reference.

## Strategy

1. Exercise classifier predicts the exercise on a 2 s window.
2. Per-exercise rep config (from `configs/exercises.yaml`) provides expected rep frequency band and threshold.
3. State machine on acc-magnitude detects rep boundaries.
4. Phase derived from where in the rep cycle we are.

## Per-exercise config

```yaml
squat:
  rep_freq_hz_range: [0.3, 0.8]      # ~1-3 sec per rep
  acc_peak_threshold_g: 0.4
  min_rep_duration_s: 1.0
  max_rep_duration_s: 5.0

bench_press:
  rep_freq_hz_range: [0.4, 1.0]
  acc_peak_threshold_g: 0.3
  min_rep_duration_s: 0.8
  max_rep_duration_s: 4.0

pushup:
  rep_freq_hz_range: [0.4, 1.0]
  acc_peak_threshold_g: 0.2          # lower for bodyweight
  min_rep_duration_s: 0.8
  max_rep_duration_s: 4.0

deadlift:
  rep_freq_hz_range: [0.2, 0.5]
  acc_peak_threshold_g: 0.5
  min_rep_duration_s: 1.5
  max_rep_duration_s: 6.0
```

Tune from your data: derive `rep_freq` per subject per exercise from joint-angle ground truth labels, then take the median.

## Online rep detector (acc-only)

```python
import numpy as np
from collections import deque

class RepDetectorAcc:
    """Online rep counting and phase tracking from acc-magnitude alone.
    Detects local maxima of (smoothed |acc - g|) as rep events; phase derived
    from acc sign + velocity sign."""

    def __init__(self, fs, exercise_cfg):
        self.fs = fs
        self.cfg = exercise_cfg
        self.acc_buf = deque(maxlen=int(2.0 * fs))  # 2 s lookback
        self.smooth_buf = deque(maxlen=int(fs * exercise_cfg['max_rep_duration_s']))
        self.last_rep_t = -1e9
        self.reps = 0
        self.phase = 'rest'
        self.in_set = False
        # Simple causal smoothing state
        self.alpha = 1 - np.exp(-1.0 / (0.1 * fs))  # 100 ms time constant
        self.smooth = 0.0
        self.t = 0.0

    def update(self, acc_mag_chunk):
        events = []
        for x in acc_mag_chunk:
            self.acc_buf.append(x)
            self.smooth = self.smooth + self.alpha * (abs(x - 9.81) - self.smooth)
            self.smooth_buf.append(self.smooth)
            self.t += 1.0 / self.fs

            # Phase from buffer trend
            if len(self.smooth_buf) >= 5:
                slope = self.smooth_buf[-1] - self.smooth_buf[-5]
                if self.smooth < 0.05:
                    self.phase = 'isometric' if self.in_set else 'rest'
                elif slope > 0:
                    # Acc-magnitude rising — peak of effort, often top/bottom of rep
                    self.phase = 'concentric'  # heuristic; refine per exercise
                else:
                    self.phase = 'eccentric'

            # Rep peak detection: local max in smooth_buf, above threshold,
            # and far enough from last rep
            if len(self.smooth_buf) >= 5:
                middle = self.smooth_buf[-3]
                is_peak = (middle > self.smooth_buf[-2] and
                           middle > self.smooth_buf[-4] and
                           middle > self.smooth_buf[-1] and
                           middle > self.smooth_buf[-5] and
                           middle > self.cfg['acc_peak_threshold_g'])
                if is_peak and (self.t - self.last_rep_t) >= self.cfg['min_rep_duration_s']:
                    self.reps += 1
                    self.last_rep_t = self.t - 3.0 / self.fs  # peak was 3 samples ago
                    events.append({'type': 'rep', 'rep_num': self.reps,
                                    't': self.last_rep_t})

        return events, {'reps': self.reps, 'phase': self.phase}

    def reset_set(self):
        self.reps = 0; self.last_rep_t = -1e9; self.phase = 'rest'
        self.smooth_buf.clear(); self.smooth = 0.0
```

This is intentionally simple. Sufficient accuracy for most strength exercises. For tempo work, push-ups, or unusual movement patterns, evaluate against the joint-angle ground-truth.

## Validation against joint-angle ground truth

In offline evaluation:

```python
def evaluate_rep_detector(predicted_reps, ground_truth_reps, tolerance_s=0.5):
    """Both are arrays of rep timestamps within a set.
    Returns precision, recall, count_error."""
    matched_pred = set(); matched_gt = set()
    for i, gt in enumerate(ground_truth_reps):
        for j, pr in enumerate(predicted_reps):
            if j in matched_pred: continue
            if abs(pr - gt) <= tolerance_s:
                matched_pred.add(j); matched_gt.add(i); break
    tp = len(matched_pred)
    return {
        'precision': tp / max(len(predicted_reps), 1),
        'recall': tp / max(len(ground_truth_reps), 1),
        'count_error': len(predicted_reps) - len(ground_truth_reps),
    }
```

Run on every labeled session. If recall < 0.85 for an exercise, retune `rep_freq_hz_range` and `acc_peak_threshold_g` for that exercise specifically.

## Phase classification: when state machine isn't enough

The acc-only phase heuristic is rough. Better phase labels come from EMG burst patterns + acc, but require ML.

If state-machine phase F1 < 0.7 vs joint-angle ground truth on holdout subjects:

```python
# Per-window classifier
features_for_phase = ['acc_jerk_rms', 'acc_rms_short_window',
                      'emg_rms', 'emg_burst_indicator',
                      'phase_lag_3', 'phase_lag_5']  # lagged features for context
model = LGBMClassifier(objective='multiclass', n_estimators=200)
# Train on (window features, phase_label from joint-angle ground truth)
```

But always start with the state machine — it's interpretable, fast, and often good enough.

## Set boundary detection (no joint angles)

Use the streaming `SetStateMachine` from the `session-segmentation` skill. On `set_start` event: call `RepDetectorAcc.reset_set()`. On `set_end` event: emit final rep count and reset.

## Hard rules

- Rep detector state ALWAYS resets on set boundary detection.
- `RepDetectorAcc` works on a sliding buffer of recent samples — never request samples from the future.
- Per-exercise config is loaded once at pipeline init; never look up exercise label DURING phase update (race condition with classifier).

## References

When documenting rep detection and phase classification choices, cite from these:

- **González-Badillo & Sánchez-Medina 2010** — bar/wrist velocity as load and effort indicator
- **Sánchez-Medina & González-Badillo 2011** — velocity loss as fatigue marker (anchor for the 20–30% MPV-loss thresholds)
- **Weakley et al. 2021** — practical VBT review
- **Bonomi et al. 2009** — accelerometer-based activity detection (justifies threshold + duration choices)
- **Mannini & Sabatini 2010** — window-based activity recognition

Full entries in `literature-references` skill. Never invent.
