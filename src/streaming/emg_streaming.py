"""
Causal EMG feature extractor for real-time deployment.

Produces the same per-window features as src/features/emg_features.py but
uses only causal operations: sosfilt with persisted zi, Welch PSD applied to
each window independently (not over the whole signal).

NO filtfilt. NO savgol_filter. NO find_peaks over whole signal.
The hook check-no-filtfilt.sh enforces this.

Features emitted per window (500 ms, hop 100 ms):
  emg_rms, emg_iemg, emg_mnf, emg_mdf, emg_dimitrov
  + baseline-normalised: emg_*_rel

References
----------
- De Luca, C. J. (1997). The use of surface electromyography in biomechanics.
  Journal of Applied Biomechanics, 13(2), 135-163.
- Dimitrov, G. V., Arabadzhiev, T. I., Mileva, K. N., et al. (2006). Muscle
  fatigue during dynamic contractions assessed by new spectral indices. Medicine
  and Science in Sports and Exercise, 38(11), 1971-1979.
- Welch, P. (1967). The use of fast Fourier transform for the estimation of
  power spectra. IEEE Transactions on Audio and Electroacoustics, 15(2), 70-73.
- Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing
  (3rd ed.). Pearson.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from src.streaming.filters import CausalBandpass, CausalNotch
from src.streaming.window_buffer import SlidingWindowBuffer


FS_EMG = 2000
WINDOW_MS = 500
HOP_MS = 100


def _emg_window_features_causal(emg_window: np.ndarray, fs: int = FS_EMG) -> dict:
    """Compute EMG features for one window (causal — window is already complete).

    Welch PSD applied per-window only (Welch 1967) — not over the whole session.
    Spectral moment formulas: MNF, MDF (De Luca 1997), FInsm5 (Dimitrov et al. 2006).
    """
    nan_result = {k: float("nan") for k in
                  ["emg_rms", "emg_iemg", "emg_mnf", "emg_mdf", "emg_dimitrov"]}

    if len(emg_window) < 16:
        return nan_result

    rms = float(np.sqrt(np.mean(emg_window ** 2)))
    iemg = float(np.sum(np.abs(emg_window)))

    nperseg = min(256, len(emg_window))
    f, pxx = welch(emg_window, fs=fs, nperseg=nperseg)

    mask = (f >= 20.0) & (f <= 450.0)
    f_phys = f[mask]
    pxx_phys = pxx[mask]

    total_power = pxx_phys.sum()
    if total_power < 1e-20:
        return nan_result

    mnf = float(np.sum(f_phys * pxx_phys) / total_power)
    cum_psd = np.cumsum(pxx_phys)
    mdf_idx = np.clip(np.searchsorted(cum_psd, cum_psd[-1] / 2.0), 0, len(f_phys) - 1)
    mdf = float(f_phys[mdf_idx])
    M_neg1 = float(np.sum(pxx_phys / np.maximum(f_phys, 1.0)))
    M_5 = float(np.sum((f_phys ** 5) * pxx_phys))
    dimitrov = M_neg1 / (M_5 + 1e-30)

    return {
        "emg_rms": rms,
        "emg_iemg": iemg,
        "emg_mnf": mnf,
        "emg_mdf": mdf,
        "emg_dimitrov": dimitrov,
    }


class StreamingEMGExtractor:
    """Causal streaming EMG feature extractor.

    Parameters
    ----------
    fs           : Sample rate (default 2000 Hz).
    window_ms    : Window length in ms (default 500 ms).
    hop_ms       : Hop between windows in ms (default 100 ms).
    baseline_s   : Duration in seconds for baseline capture (default 60 s).
    """

    def __init__(
        self,
        fs: int = FS_EMG,
        window_ms: float = WINDOW_MS,
        hop_ms: float = HOP_MS,
        baseline_s: float = 60.0,
    ) -> None:
        self._fs = fs
        self._baseline_s = baseline_s

        # Causal filters (Oppenheim & Schafer 2010)
        self._bp = CausalBandpass(20.0, 450.0, fs, order=4)
        self._notch = CausalNotch(50.0, fs, Q=30.0)

        # Sliding window buffer
        win_samp = max(1, int(window_ms * fs / 1000))
        hop_samp = max(1, int(hop_ms * fs / 1000))
        self._window = SlidingWindowBuffer(win_samp, hop_samp)

        # Baseline normalisation accumulators (per feature key)
        _keys = ("emg_mnf", "emg_mdf", "emg_rms", "emg_dimitrov", "emg_iemg")
        self._baseline_accum: dict[str, list[float]] = {k: [] for k in _keys}
        self._baseline_medians: dict[str, float] = {k: float("nan") for k in _keys}
        self._baseline_locked: bool = False
        self._baseline_end_unix: float | None = None

        # Time tracking
        self._t0_unix: float | None = None
        self._sample_idx: int = 0

    def set_baseline_end(self, t_unix: float) -> None:
        """Set the Unix time at which baseline capture ends."""
        self._baseline_end_unix = t_unix

    def step(self, chunk: np.ndarray, t_unix_chunk: np.ndarray) -> list[dict]:
        """Process a chunk of EMG samples.

        Parameters
        ----------
        chunk        : 1D raw EMG samples.
        t_unix_chunk : Corresponding Unix timestamps.

        Returns
        -------
        List of feature dicts per completed window hop.
        """
        chunk = np.asarray(chunk, dtype=float)
        t_unix_chunk = np.asarray(t_unix_chunk, dtype=float)

        if self._t0_unix is None and len(t_unix_chunk) > 0:
            self._t0_unix = float(t_unix_chunk[0])

        # Causal filtering
        x_bp = self._bp.step(chunk)
        x_filt = self._notch.step(x_bp)

        results = []
        for i, (sample, t) in enumerate(zip(x_filt, t_unix_chunk)):
            windows = self._window.push(np.array([sample]))
            for win in windows:
                # Check baseline lock
                if (self._baseline_end_unix is not None
                        and t > self._baseline_end_unix
                        and not self._baseline_locked):
                    self._lock_baseline()

                feats = _emg_window_features_causal(win, self._fs)

                # Feed baseline
                if not self._baseline_locked:
                    if self._baseline_end_unix is None or t <= self._baseline_end_unix:
                        for k in self._baseline_accum:
                            v = feats.get(k, float("nan"))
                            if not np.isnan(v):
                                self._baseline_accum[k].append(v)

                # Normalise
                feats = self._normalize(feats)
                feats["t_unix"] = t
                results.append(feats)

        return results

    def _lock_baseline(self) -> None:
        for k, vals in self._baseline_accum.items():
            self._baseline_medians[k] = float(np.median(vals)) if vals else float("nan")
        self._baseline_locked = True

    def _normalize(self, feats: dict) -> dict:
        out = dict(feats)
        for k, bmed in self._baseline_medians.items():
            v = feats.get(k, float("nan"))
            if not np.isnan(bmed) and bmed > 0:
                out[f"{k}_rel"] = v / bmed
            else:
                out[f"{k}_rel"] = float("nan")
        return out

    def reset(self) -> None:
        """Reset all state (call at session start)."""
        self._bp.reset()
        self._notch.reset()
        self._window.reset()
        for k in self._baseline_accum:
            self._baseline_accum[k].clear()
        self._baseline_locked = False
        self._t0_unix = None
        self._sample_idx = 0
