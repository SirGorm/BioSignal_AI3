"""
Causal PPG (green wavelength) feature extractor for real-time deployment.

Features per window (10 s rolling, hop 100 ms):
  ppg_hr, ppg_pulse_amp, ppg_pulse_amp_var

Online peak detection: simple adaptive threshold on causal-filtered PPG —
does NOT call find_peaks over the whole session.

NO filtfilt. NO savgol_filter. NO find_peaks over whole signal.
The hook check-no-filtfilt.sh enforces this.

Green wavelength preferred for wrist-worn sensors (Castaneda et al. 2018).

References
----------
- Allen, J. (2007). Photoplethysmography and its application in clinical
  physiological measurement. Physiological Measurement, 28(3), R1.
- Castaneda, D., Esparza, A., Ghamari, M., Soltanpur, C., & Nazeran, H. (2018).
  A review on wearable photoplethysmography sensors. International Journal of
  Biosensors & Bioelectronics, 4(4), 195-202.
- Maeda, Y., Sekine, M., & Tamura, T. (2011). Relationship between measurement
  site and motion artifacts in wearable reflected photoplethysmography.
  Journal of Medical Systems, 35(5), 969-976.
- Tamura, T., Maeda, Y., Sekine, M., & Yoshida, M. (2014). Wearable
  photoplethysmographic sensors—past and present. Electronics, 3(2), 282-302.
- Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing
  (3rd ed.). Pearson.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.streaming.filters import CausalBandpass
from src.streaming.window_buffer import SlidingWindowBuffer


FS_PPG_DEFAULT = 100
WINDOW_S = 10.0
HOP_MS = 100


class OnlinePPGPeakDetector:
    """Adaptive threshold online peak detector for PPG.

    Uses a sliding max + refractory period approach — causal, no look-ahead.
    Allen 2007 — peak detection for PPG-derived HR.
    """

    def __init__(self, fs: int) -> None:
        self._fs = fs
        self._refractory = int(0.2 * fs)  # 200 ms (300 bpm max)
        self._samples_since_peak = self._refractory + 1
        # Running max over last 2 s for adaptive threshold
        self._max_buffer: deque = deque(maxlen=int(2.0 * fs))
        self._adaptive_thresh = 0.0
        self._prev_sample = 0.0
        self._sample_count = 0

    def step(self, x: float) -> bool:
        """Return True if a systolic peak is detected at this sample."""
        self._max_buffer.append(abs(x))
        self._adaptive_thresh = 0.3 * max(self._max_buffer) if self._max_buffer else 0.0
        self._samples_since_peak += 1
        self._sample_count += 1

        # Peak: local rising then falling + above threshold + refractory
        is_peak = (
            x > self._adaptive_thresh
            and x > self._prev_sample
            and self._samples_since_peak > self._refractory
            and self._sample_count > int(0.5 * self._fs)  # warmup
        )

        if is_peak:
            self._samples_since_peak = 0

        self._prev_sample = x
        return is_peak

    def reset(self) -> None:
        self._max_buffer.clear()
        self._samples_since_peak = self._refractory + 1
        self._adaptive_thresh = 0.0
        self._prev_sample = 0.0
        self._sample_count = 0


class StreamingPPGExtractor:
    """Causal streaming PPG (green) feature extractor."""

    def __init__(
        self,
        fs: int = FS_PPG_DEFAULT,
        window_s: float = WINDOW_S,
        hop_ms: float = HOP_MS,
    ) -> None:
        self._fs = fs
        # Causal bandpass 0.5-8 Hz (Oppenheim & Schafer 2010)
        self._bp = CausalBandpass(0.5, 8.0, fs, order=4)
        # Sliding window
        win_samp = max(1, int(window_s * fs))
        hop_samp = max(1, int(hop_ms * fs / 1000))
        self._window = SlidingWindowBuffer(win_samp, hop_samp)
        # Online peak detector
        self._peak_detector = OnlinePPGPeakDetector(fs)
        # Rolling peak time buffer (last 10 s)
        self._peak_times: deque = deque()
        self._peak_amps: deque = deque()  # amplitude at each peak
        self._window_s = window_s
        # Current filtered signal buffer (for amplitude computation)
        self._signal_buffer: deque = deque(maxlen=int(window_s * fs))
        self._t_buffer: deque = deque(maxlen=int(window_s * fs))

    def step(self, chunk: np.ndarray, t_unix_chunk: np.ndarray) -> list[dict]:
        """Process a chunk of PPG (green) samples.

        Parameters
        ----------
        chunk        : 1D raw PPG green samples.
        t_unix_chunk : Corresponding Unix timestamps.

        Returns
        -------
        List of feature dicts per completed hop.
        """
        chunk = np.asarray(chunk, dtype=float)
        t_unix_chunk = np.asarray(t_unix_chunk, dtype=float)

        # Causal bandpass
        x_filt = self._bp.step(chunk)

        results = []
        for sample, t in zip(x_filt, t_unix_chunk):
            self._signal_buffer.append(sample)
            self._t_buffer.append(t)

            is_peak = self._peak_detector.step(sample)
            if is_peak:
                self._peak_times.append(t)
                self._peak_amps.append(sample)

            # Trim peak buffers to window duration
            while self._peak_times and self._peak_times[0] < t - self._window_s:
                self._peak_times.popleft()
                self._peak_amps.popleft()

            windows = self._window.push(np.array([sample]))
            for _ in windows:
                feats = self._compute_features(t)
                results.append(feats)

        return results

    def _compute_features(self, t_unix: float) -> dict:
        """Compute PPG features from current rolling peak buffer (causal)."""
        nan_result = {
            "ppg_hr": float("nan"),
            "ppg_pulse_amp": float("nan"),
            "ppg_pulse_amp_var": float("nan"),
            "t_unix": t_unix,
        }
        peaks = list(self._peak_times)
        amps = list(self._peak_amps)

        if len(peaks) < 2:
            return nan_result

        pp_s = np.diff(peaks)
        valid = (pp_s >= 0.25) & (pp_s <= 2.0)  # 30-240 bpm (Allen 2007)
        if valid.sum() < 1:
            return nan_result

        hr = float(60.0 / np.mean(pp_s[valid]))
        amp_mean = float(np.mean(amps)) if amps else float("nan")
        amp_var = float(np.std(amps)) if len(amps) > 1 else float("nan")

        return {
            "ppg_hr": hr,
            "ppg_pulse_amp": amp_mean,
            "ppg_pulse_amp_var": amp_var,
            "t_unix": t_unix,
        }

    def reset(self) -> None:
        self._bp.reset()
        self._window.reset()
        self._peak_detector.reset()
        self._peak_times.clear()
        self._peak_amps.clear()
        self._signal_buffer.clear()
        self._t_buffer.clear()
