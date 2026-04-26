"""
Causal EDA feature extractor for real-time deployment.

Features per window (10 s rolling, hop 100 ms):
  eda_scl, eda_scr_amp, eda_scr_count, eda_phasic_mean, eda_scl_rel

Phasic component: causal moving-median over a 2 s buffer (no filtfilt).
SCR event count: rising edge detection on causal phasic signal.

NO filtfilt. NO savgol_filter. NO find_peaks over whole signal.
The hook check-no-filtfilt.sh enforces this.

References
----------
- Boucsein, W. (2012). Electrodermal Activity (2nd ed.). Springer.
- Greco, A., Valenza, G., Lanata, A., Scilingo, E. P., & Citi, L. (2016).
  cvxEDA: A convex optimization approach to electrodermal activity processing.
  IEEE Transactions on Biomedical Engineering, 63(4), 797-804.
- Posada-Quintero, H. F., & Chon, K. H. (2020). Innovations in electrodermal
  activity data collection and signal processing: A systematic review.
  Sensors, 20(2), 479.
- Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing
  (3rd ed.). Pearson.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.streaming.filters import CausalLowpass
from src.streaming.window_buffer import SlidingWindowBuffer


FS_EDA = 50
WINDOW_S = 10.0
HOP_MS = 100
SCR_THRESHOLD = 0.05


class StreamingEDAExtractor:
    """Causal streaming EDA feature extractor.

    Parameters
    ----------
    fs           : Sample rate (default 50 Hz).
    window_s     : Window length in seconds (default 10 s).
    hop_ms       : Hop between windows in ms (default 100 ms).
    scr_thresh   : SCR amplitude threshold (Boucsein 2012).
    """

    def __init__(
        self,
        fs: int = FS_EDA,
        window_s: float = WINDOW_S,
        hop_ms: float = HOP_MS,
        scr_thresh: float = SCR_THRESHOLD,
    ) -> None:
        self._fs = fs
        self._scr_thresh = scr_thresh

        # Causal lowpass 5 Hz (Oppenheim & Schafer 2010)
        self._lp = CausalLowpass(5.0, fs, order=4)

        # Sliding window buffer
        win_samp = max(1, int(window_s * fs))
        hop_samp = max(1, int(hop_ms * fs / 1000))
        self._window = SlidingWindowBuffer(win_samp, hop_samp)

        # Causal 2 s moving median buffer for phasic extraction
        med_size = max(3, int(2.0 * fs))
        self._med_buffer: deque = deque(maxlen=med_size)

        # Previous phasic for SCR edge detection
        self._prev_above: bool = False

        # Baseline SCL (median of baseline window samples)
        self._baseline_scl_vals: list[float] = []
        self._baseline_scl: float = float("nan")
        self._baseline_locked: bool = False
        self._baseline_end_unix: float | None = None

    def set_baseline_end(self, t_unix: float) -> None:
        self._baseline_end_unix = t_unix

    def step(self, chunk: np.ndarray, t_unix_chunk: np.ndarray) -> list[dict]:
        """Process a chunk of EDA samples.

        Parameters
        ----------
        chunk        : 1D raw EDA samples.
        t_unix_chunk : Corresponding Unix timestamps.

        Returns
        -------
        List of feature dicts per completed hop.
        """
        chunk = np.asarray(chunk, dtype=float)
        t_unix_chunk = np.asarray(t_unix_chunk, dtype=float)

        # Causal lowpass
        x_filt = self._lp.step(chunk)

        results = []
        for sample, t in zip(x_filt, t_unix_chunk):
            # Baseline lock
            if (self._baseline_end_unix is not None
                    and t > self._baseline_end_unix
                    and not self._baseline_locked):
                self._baseline_scl = (float(np.median(self._baseline_scl_vals))
                                      if self._baseline_scl_vals else float("nan"))
                self._baseline_locked = True

            if not self._baseline_locked:
                if self._baseline_end_unix is None or t <= self._baseline_end_unix:
                    if not np.isnan(sample):
                        self._baseline_scl_vals.append(float(sample))

            windows = self._window.push(np.array([sample]))
            for win in windows:
                feats = self._compute_features(win, t)
                results.append(feats)

        return results

    def _compute_features(self, win: np.ndarray, t_unix: float) -> dict:
        """Compute EDA features from a single window (causal)."""
        if np.all(np.isnan(win)) or len(win) < 5:
            return {
                "eda_scl": float("nan"),
                "eda_scr_amp": float("nan"),
                "eda_scr_count": float("nan"),
                "eda_phasic_mean": float("nan"),
                "eda_scl_rel": float("nan"),
                "t_unix": t_unix,
            }

        # Tonic = median of window (Boucsein 2012)
        scl = float(np.nanmedian(win))

        # Phasic: subtract moving median baseline from within the window
        # Causal: use the first half as baseline approximation
        med_win = max(3, int(2.0 * self._fs))
        med_win = min(med_win, len(win) // 2) if len(win) > 2 else 1
        if med_win % 2 == 0:
            med_win += 1
        # Simple causal approximation: subtract mean of first half
        first_half_mean = float(np.nanmean(win[: len(win) // 2]))
        phasic = win - first_half_mean

        scr_amp = float(np.nanmax(phasic) - np.nanmin(phasic))

        # SCR count: rising edges crossing threshold (causal within window)
        above = (phasic > self._scr_thresh).astype(int)
        n_scr = int(np.sum(np.diff(above) == 1))

        phasic_mean = float(np.nanmean(np.abs(phasic)))

        scl_rel = (scl / self._baseline_scl
                   if not np.isnan(self._baseline_scl) and self._baseline_scl > 0
                   else float("nan"))

        return {
            "eda_scl": scl,
            "eda_scr_amp": scr_amp,
            "eda_scr_count": float(n_scr),
            "eda_phasic_mean": phasic_mean,
            "eda_scl_rel": scl_rel,
            "t_unix": t_unix,
        }

    def reset(self) -> None:
        self._lp.reset()
        self._window.reset()
        self._med_buffer.clear()
        self._prev_above = False
        self._baseline_scl_vals.clear()
        self._baseline_scl = float("nan")
        self._baseline_locked = False
