---
name: real-time-pipeline
description: Use when writing online/streaming feature extractors, causal filters, or any code that must run in real-time. Covers window state management, causal filtering with persisted state, online normalization (Welford), latency budgeting, and the streaming pipeline pattern. Apply whenever code lives under src/streaming/ or src/pipeline/realtime*.
---

# Real-Time Streaming Pipeline

## Cardinal rule

**Causal only.** No look-ahead, no `filtfilt`, no `savgol_filter` over future samples, no FFT over the whole recording. Every operation must be expressible as: given samples up to time `t`, output features for time `t`.

## Streaming filter pattern

```python
from scipy.signal import butter, sosfilt, sosfilt_zi

class CausalBandpass:
    def __init__(self, low_hz, high_hz, fs, order=4):
        self.sos = butter(order, [low_hz, high_hz], btype='band', fs=fs, output='sos')
        self.zi = sosfilt_zi(self.sos)  # CRITICAL: persist filter state
        self.initialized = False

    def step(self, x):
        # x: 1D array of new samples (any length, including 1)
        if not self.initialized:
            # warm-start so the filter doesn't ring at session start
            self.zi = sosfilt_zi(self.sos) * x[0]
            self.initialized = True
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y
```

The `zi` carries filter memory between calls. Without it, every chunk gets a transient.

## Sliding window pattern

```python
from collections import deque
import numpy as np

class SlidingWindow:
    def __init__(self, size_samples, hop_samples):
        self.size = size_samples
        self.hop = hop_samples
        self.buffer = deque(maxlen=size_samples)
        self.samples_since_emit = 0

    def push(self, x):
        # x: 1D array of new samples
        out = []
        for v in x:
            self.buffer.append(v)
            self.samples_since_emit += 1
            if len(self.buffer) == self.size and self.samples_since_emit >= self.hop:
                out.append(np.array(self.buffer))
                self.samples_since_emit = 0
        return out  # list of full windows ready for feature extraction
```

## Online normalization (Welford)

Never z-score a streaming signal with `(x - mean) / std` computed on the whole signal — you don't have the whole signal. Use Welford's algorithm:

```python
class OnlineStats:
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0
    def update(self, x):
        for v in np.atleast_1d(x):
            self.n += 1
            d = v - self.mean
            self.mean += d / self.n
            self.M2 += d * (v - self.mean)
    @property
    def std(self):
        return (self.M2 / max(self.n - 1, 1)) ** 0.5
    def z(self, x):
        return (x - self.mean) / (self.std + 1e-8)
```

Per subject and per channel. Reset at session start. After ~30 s the running stats stabilize and z-scores are reliable.

## Forbidden operations (block via test or hook)

| Forbidden | Why | Causal alternative |
|-----------|-----|--------------------|
| `scipy.signal.filtfilt` | zero-phase = looks ahead | `sosfilt` with persisted `zi` |
| `np.fft.fft(whole_signal)` | non-causal | STFT per-window inside the sliding buffer |
| `pywt.wavedec(signal)` | non-causal | per-window CWT, accept the edge effects |
| `(x - x.mean()) / x.std()` over a session | uses future | Welford as above |
| `scipy.signal.find_peaks(whole_signal)` | sees future | online peak detector, see vbt-rep-detection skill |
| Resampling with `polyphase` over full signal | non-causal | `scipy.signal.resample_poly` per-window with overlap, or stick to native fs |

## Latency budget

Target: **< 100 ms p99** from sample-in to prediction-out. Breakdown:

- Filter step: ~1 ms
- Window emission: 50–100 ms (this is the dominant term, equal to hop size)
- Feature extraction per window: < 10 ms
- Model inference (LightGBM): < 5 ms
- Total: ~65–115 ms typical

If you exceed budget, the bottleneck is almost always feature extraction (some FFT or entropy that can be windowed smaller), not the model.

Benchmark with:

```python
import time
def benchmark(streamer, n_samples=10_000):
    latencies = []
    for chunk in stream_simulator(n_samples, chunk_size=10):
        t0 = time.perf_counter()
        streamer.step(chunk)
        latencies.append((time.perf_counter() - t0) * 1000)
    return np.percentile(latencies, [50, 95, 99])
```

## Skeleton pipeline

```python
class RealTimePipeline:
    def __init__(self, fs_per_channel, ...):
        self.filters = {ch: CausalBandpass(...) for ch in channels}
        self.windows = {ch: SlidingWindow(...) for ch in channels}
        self.norm = {ch: OnlineStats() for ch in channels}
        self.exercise_model = ...   # loaded LightGBM
        self.fatigue_model = ...
        self.rep_counter = ...      # see vbt-rep-detection skill
        self.phase_state = ...      # state machine

    def step(self, samples_per_channel):
        # samples_per_channel: dict[ch] -> ndarray of new samples
        outputs = {}
        for ch, x in samples_per_channel.items():
            x_filt = self.filters[ch].step(x)
            self.norm[ch].update(x_filt)
            for window in self.windows[ch].push(x_filt):
                feats = extract_features_causal(window, fs=self.fs[ch])
                outputs.setdefault(ch, []).append(feats)

        if all_channels_ready(outputs):
            joined = join_features_across_channels(outputs)
            return {
                'exercise': self.exercise_model.predict_proba([joined])[0],
                'fatigue':  self.fatigue_model.predict([joined])[0],
                'phase':    self.phase_state.update(joined),
                'reps':     self.rep_counter.update(joined),
            }
        return None
```

## Testing real-time code

Replay a recorded session sample-by-sample and verify:
1. **Equivalence-up-to-startup**: outputs after first 2 seconds should match an offline causal pipeline applied to the same data.
2. **No future leakage**: shuffle samples after timestep `t` and verify outputs for `t' < t` are unchanged.
3. **Latency**: p99 < 100 ms over 10 minutes of data.

Save these as `tests/test_streaming_*.py`.

## References

For causal filtering, online normalization, and streaming DSP:

- **Oppenheim & Schafer 2010** — IIR/FIR filtering, causal vs zero-phase filtering
- **Welford 1962** — online running mean/variance algorithm (used in Welford z-score normalization)
- **Welch 1967** — Welch's PSD method (applied per-window in streaming)
- **Virtanen et al. 2020** — scipy.signal documentation

Full entries in `literature-references` skill. Never invent.
