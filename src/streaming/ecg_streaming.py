"""
Causal ECG feature extractor for real-time deployment.

Produces the same features as src/features/ecg_features.py but uses only
causal operations: sosfilt with persisted zi, rolling RR buffer, and
online peak detection (Pan & Tompkins 1985).

NO filtfilt, NO savgol_filter, NO find_peaks over the whole signal.
The hook check-no-filtfilt.sh enforces this.

Features emitted per window (30 s rolling, updated on each hop):
  ecg_hr, ecg_rmssd, ecg_sdnn, ecg_pnn50, ecg_mean_rr

References
----------
- Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
  IEEE Transactions on Biomedical Engineering, BME-32(3), 230-236.
- Task Force of the European Society of Cardiology (1996). Heart rate
  variability: standards of measurement. Circulation, 93(5), 1043-1065.
- Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing
  (3rd ed.). Pearson.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.streaming.filters import CausalBandpass, CausalNotch
from src.streaming.window_buffer import SlidingWindowBuffer


FS_ECG = 500
RR_BUFFER_S = 30  # keep last 30 s of RR intervals for HRV (Shaffer & Ginsberg 2017)
HOP_SAMPLES = 50  # 100 ms at 500 Hz


class OnlineRPeakDetector:
    """Real-time R-peak detector based on Pan-Tompkins 1985.

    Uses a derivative filter + squaring + moving-window integration + adaptive
    threshold to detect QRS complexes without look-ahead.

    Pan & Tompkins 1985 — cite for any R-peak detection in streaming code.
    """

    def __init__(self, fs: int = FS_ECG) -> None:
        self._fs = fs
        self._integ_win = int(0.2 * fs)  # 200 ms integration window
        self._refractory = int(0.2 * fs)  # 200 ms refractory period
        self._integ_buffer: deque = deque(maxlen=self._integ_win)
        self._threshold = 0.0
        self._samples_since_last_peak = self._refractory + 1
        self._signal_level = 0.0
        self._noise_level = 0.0
        self._prev_sample = 0.0
        self._sample_count = 0

    def step(self, x: float) -> bool:
        """Process one sample. Returns True if an R-peak is detected here.

        Pan & Tompkins 1985 — derivative + squaring + moving average.
        """
        # Derivative (Pan & Tompkins 1985 eq.)
        diff = x - self._prev_sample
        self._prev_sample = x
        squared = diff ** 2
        self._integ_buffer.append(squared)
        integrated = float(np.mean(self._integ_buffer))

        self._sample_count += 1
        self._samples_since_last_peak += 1

        # Adaptive threshold initialisation: first 0.5 s
        if self._sample_count < int(0.5 * self._fs):
            self._signal_level = max(self._signal_level, integrated)
            self._threshold = 0.5 * self._signal_level
            return False

        is_peak = (
            integrated > self._threshold
            and self._samples_since_last_peak > self._refractory
        )

        if is_peak:
            # Update signal level and threshold adaptively (Pan & Tompkins 1985)
            self._signal_level = 0.125 * integrated + 0.875 * self._signal_level
            self._threshold = self._noise_level + 0.25 * (
                self._signal_level - self._noise_level
            )
            self._samples_since_last_peak = 0
        else:
            self._noise_level = 0.125 * integrated + 0.875 * self._noise_level
            self._threshold = self._noise_level + 0.25 * (
                self._signal_level - self._noise_level
            )

        return is_peak

    def reset(self) -> None:
        self._integ_buffer.clear()
        self._threshold = 0.0
        self._samples_since_last_peak = self._refractory + 1
        self._signal_level = 0.0
        self._noise_level = 0.0
        self._prev_sample = 0.0
        self._sample_count = 0


class StreamingECGExtractor:
    """Causal streaming ECG feature extractor.

    Usage
    -----
    extractor = StreamingECGExtractor()
    for chunk in stream:
        features = extractor.step(chunk, t_unix_chunk)
        if features:
            print(features[-1])  # most recent window features

    Features are emitted on each hop (100 ms). Empty list if no new window.
    """

    def __init__(self, fs: int = FS_ECG, hop_ms: float = 100.0) -> None:
        self._fs = fs
        self._hop_samples = max(1, int(hop_ms * fs / 1000))

        # Causal filters (Oppenheim & Schafer 2010)
        self._bp = CausalBandpass(0.5, 40.0, fs, order=4)
        self._notch = CausalNotch(50.0, fs, Q=30.0)

        # R-peak detector (Pan & Tompkins 1985)
        self._r_detector = OnlineRPeakDetector(fs)

        # Rolling RR interval buffer (last 30 s).
        # _rr_buffer and _rr_times are 1:1 — both contain only valid RR intervals.
        # _last_peak_t tracks the time of the most recent R-peak for diff calculation.
        self._rr_buffer: deque = deque()
        self._rr_times: deque = deque()  # trailing peak times for valid RR intervals
        self._last_peak_t: float | None = None
        self._rr_buffer_duration_s = float(RR_BUFFER_S)

        # Sample counter for time tracking
        self._sample_idx: int = 0
        self._t0_unix: float | None = None

        # Hop counter for emitting windows
        self._samples_since_emit: int = 0

        # Baseline HR for normalisation (locked after 60 s)
        self._baseline_hrs: list[float] = []
        self._baseline_locked: bool = False
        self._baseline_hr: float = float("nan")
        self._baseline_end_unix: float | None = None

    def set_baseline_end(self, t_unix: float) -> None:
        """Set the Unix time at which baseline capture ends."""
        self._baseline_end_unix = t_unix

    def step(self, chunk: np.ndarray, t_unix_chunk: np.ndarray) -> list[dict]:
        """Process a chunk of ECG samples.

        Parameters
        ----------
        chunk        : 1D array of raw ECG samples.
        t_unix_chunk : Corresponding Unix timestamps.

        Returns
        -------
        List of feature dicts, one per completed hop (may be empty).
        """
        chunk = np.asarray(chunk, dtype=float)
        t_unix_chunk = np.asarray(t_unix_chunk, dtype=float)

        if len(chunk) == 0:
            return []

        if self._t0_unix is None and len(t_unix_chunk) > 0:
            self._t0_unix = float(t_unix_chunk[0])

        # Causal filtering
        x_bp = self._bp.step(chunk)
        x_filt = self._notch.step(x_bp)

        results = []
        for i, (sample, t) in enumerate(zip(x_filt, t_unix_chunk)):
            is_peak = self._r_detector.step(sample)
            if is_peak:
                # Record RR interval from the previous peak
                if self._last_peak_t is not None:
                    rr_ms = (t - self._last_peak_t) * 1000.0
                    # Physiological filter: 300-2000 ms (Shaffer & Ginsberg 2017)
                    if 300.0 < rr_ms < 2000.0:
                        self._rr_buffer.append(rr_ms)
                        self._rr_times.append(t)
                        # Baseline capture
                        if (self._baseline_end_unix is None or t <= self._baseline_end_unix):
                            if not self._baseline_locked:
                                hr_inst = 60_000.0 / rr_ms
                                self._baseline_hrs.append(hr_inst)
                self._last_peak_t = t

            # Trim RR buffer to last 30 s — _rr_buffer and _rr_times are always 1:1
            while (self._rr_times
                   and self._rr_times[0] < t - self._rr_buffer_duration_s):
                self._rr_buffer.popleft()
                self._rr_times.popleft()

            self._samples_since_emit += 1
            if self._samples_since_emit >= self._hop_samples:
                self._samples_since_emit = 0
                # Check if baseline should be locked
                if (self._baseline_end_unix is not None
                        and t > self._baseline_end_unix
                        and not self._baseline_locked):
                    self._baseline_locked = True
                    if self._baseline_hrs:
                        self._baseline_hr = float(np.median(self._baseline_hrs))

                feats = self._compute_features(t)
                results.append(feats)

        return results

    def _compute_features(self, t_unix: float) -> dict:
        """Compute HRV features from current RR buffer (causal — only past RRs)."""
        rr = np.array(self._rr_buffer, dtype=float)
        rr_times = np.array(self._rr_times, dtype=float)

        # _rr_buffer is already trimmed to last 30 s by the step() loop.
        # Apply a final mask to be safe.
        if len(rr_times) > 0 and len(rr) == len(rr_times):
            mask = rr_times >= (t_unix - self._rr_buffer_duration_s)
            rr = rr[mask]

        if len(rr) < 3:
            return {
                "ecg_hr": float("nan"),
                "ecg_rmssd": float("nan"),
                "ecg_sdnn": float("nan"),
                "ecg_pnn50": float("nan"),
                "ecg_mean_rr": float("nan"),
                "ecg_hr_rel": float("nan"),
                "t_unix": t_unix,
            }

        hr = float(60_000.0 / np.mean(rr))
        rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        sdnn = float(np.std(rr, ddof=1))
        pnn50 = float(np.mean(np.abs(np.diff(rr)) > 50) * 100)
        hr_rel = hr / self._baseline_hr if not np.isnan(self._baseline_hr) else float("nan")

        return {
            "ecg_hr": hr,
            "ecg_rmssd": rmssd,
            "ecg_sdnn": sdnn,
            "ecg_pnn50": pnn50,
            "ecg_mean_rr": float(np.mean(rr)),
            "ecg_hr_rel": hr_rel,
            "t_unix": t_unix,
        }

    def reset(self) -> None:
        """Reset all state (call at session start)."""
        self._bp.reset()
        self._notch.reset()
        self._r_detector.reset()
        self._rr_buffer.clear()
        self._rr_times.clear()
        self._last_peak_t = None
        self._sample_idx = 0
        self._t0_unix = None
        self._samples_since_emit = 0
        self._baseline_hrs.clear()
        self._baseline_locked = False
        self._baseline_hr = float("nan")
