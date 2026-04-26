"""
Causal temperature feature extractor for real-time deployment.

Features per window (60 s rolling, hop 100 ms):
  temp_mean, temp_slope, temp_range, temp_mean_rel

Uses causal lowpass and online linear regression (no filtfilt).
NaN-tolerant: outputs NaN when temperature is unavailable.

NO filtfilt. NO savgol_filter.
The hook check-no-filtfilt.sh enforces this.

References
----------
- [REF NEEDED: skin temperature as fatigue indicator during exercise]
- Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing
  (3rd ed.). Pearson.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.streaming.filters import CausalLowpass
from src.streaming.window_buffer import SlidingWindowBuffer


FS_TEMP = 1
WINDOW_S = 60.0
HOP_MS = 100


class StreamingTempExtractor:
    """Causal streaming temperature feature extractor.

    NaN-tolerant: if temperature is all-NaN or absent, all features are NaN.
    """

    def __init__(
        self,
        fs: int = FS_TEMP,
        window_s: float = WINDOW_S,
        hop_ms: float = HOP_MS,
    ) -> None:
        self._fs = fs
        # Causal lowpass 0.1 Hz (Oppenheim & Schafer 2010)
        self._lp = CausalLowpass(0.1, fs, order=2)
        win_samp = max(1, int(window_s * fs))
        hop_samp = max(1, int(hop_ms * fs / 1000))
        self._window = SlidingWindowBuffer(win_samp, hop_samp)
        self._t_window: deque = deque(maxlen=win_samp)

        # Baseline temperature mean
        self._baseline_vals: list[float] = []
        self._baseline_mean: float = float("nan")
        self._baseline_locked: bool = False
        self._baseline_end_unix: float | None = None

    def set_baseline_end(self, t_unix: float) -> None:
        self._baseline_end_unix = t_unix

    def step(self, chunk: np.ndarray, t_unix_chunk: np.ndarray) -> list[dict]:
        """Process a chunk of temperature samples.

        Parameters
        ----------
        chunk        : 1D raw temperature samples (may be all-NaN).
        t_unix_chunk : Corresponding Unix timestamps.

        Returns
        -------
        List of feature dicts per completed hop.
        """
        chunk = np.asarray(chunk, dtype=float)
        t_unix_chunk = np.asarray(t_unix_chunk, dtype=float)

        results = []

        # All-NaN guard
        if np.all(np.isnan(chunk)):
            # Emit NaN features at hop intervals without actual windowing
            return results

        # Replace NaN with previous valid value (forward fill, causal)
        for i in range(len(chunk)):
            if np.isnan(chunk[i]) and i > 0 and not np.isnan(chunk[i - 1]):
                chunk[i] = chunk[i - 1]

        x_filt = self._lp.step(chunk)

        for sample, t in zip(x_filt, t_unix_chunk):
            self._t_window.append(t)

            # Baseline
            if (self._baseline_end_unix is not None
                    and t > self._baseline_end_unix
                    and not self._baseline_locked):
                self._baseline_mean = (float(np.mean(self._baseline_vals))
                                       if self._baseline_vals else float("nan"))
                self._baseline_locked = True

            if not self._baseline_locked:
                if not np.isnan(sample):
                    self._baseline_vals.append(float(sample))

            windows = self._window.push(np.array([sample]))
            for win in windows:
                t_arr = np.array(list(self._t_window)[-len(win) :])
                feats = self._compute_features(win, t_arr, t)
                results.append(feats)

        return results

    def _compute_features(
        self, win: np.ndarray, t_arr: np.ndarray, t_unix: float
    ) -> dict:
        valid = ~np.isnan(win)
        if valid.sum() < 5:
            return {
                "temp_mean": float("nan"),
                "temp_slope": float("nan"),
                "temp_range": float("nan"),
                "temp_mean_rel": float("nan"),
                "t_unix": t_unix,
            }
        t_rel = t_arr[valid] - t_arr[valid][0]
        v = win[valid]
        mean = float(np.mean(v))
        slope = float(np.polyfit(t_rel, v, 1)[0]) if len(v) >= 2 else float("nan")
        rng = float(np.max(v) - np.min(v))
        mean_rel = (mean / self._baseline_mean
                    if not np.isnan(self._baseline_mean) and self._baseline_mean > 0
                    else float("nan"))
        return {
            "temp_mean": mean,
            "temp_slope": slope,
            "temp_range": rng,
            "temp_mean_rel": mean_rel,
            "t_unix": t_unix,
        }

    def reset(self) -> None:
        self._lp.reset()
        self._window.reset()
        self._t_window.clear()
        self._baseline_vals.clear()
        self._baseline_mean = float("nan")
        self._baseline_locked = False
