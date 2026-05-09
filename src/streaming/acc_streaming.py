"""
Causal accelerometer feature extractor for real-time deployment.

Computes acc_mag = sqrt(ax^2 + ay^2 + az^2) FIRST, then causal bandpass.
Features per window (2 s, hop 100 ms):
  acc_rms, acc_jerk_rms, acc_dom_freq, acc_rep_band_power, acc_rep_band_ratio,
  acc_lscore, acc_mfl, acc_msr, acc_wamp

Welch PSD is applied per window only (not over the whole signal).
Phinyomark et al. 2012 amplitude descriptors (lscore, mfl, msr) and the
Hudgins et al. 1993 Willison amplitude (wamp) are pure window-level
computations — no filter state is required.

NO filtfilt. NO savgol_filter. NO find_peaks over whole signal.
The hook check-no-filtfilt.sh enforces this.

References
----------
- Bonomi, A. G., Goris, A. H., Yin, B., & Westerterp, K. R. (2009). Detection
  of type, duration, and intensity of physical activity using an accelerometer.
  Medicine & Science in Sports & Exercise, 41(9), 1770-1777.
- González-Badillo, J. J., & Sánchez-Medina, L. (2010). Movement velocity as a
  measure of loading intensity in resistance training. International Journal of
  Sports Medicine, 31(05), 347-352.
- Hudgins, B., Parker, P., & Scott, R. N. (1993). A new strategy for multifunction
  myoelectric control. IEEE Transactions on Biomedical Engineering, 40(1), 82-94.
- Khan, A. M., Lee, Y. K., Lee, S. Y., & Kim, T. S. (2010). A triaxial
  accelerometer-based physical-activity recognition via augmented-signal features
  and a hierarchical recognizer. IEEE Transactions on Information Technology in
  Biomedicine, 14(5), 1166-1172.
- Mannini, A., & Sabatini, A. M. (2010). Machine learning methods for classifying
  human physical activity from on-body accelerometers. Sensors, 10(2), 1154-1175.
- Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). Feature reduction
  and selection for EMG signal classification. Expert Systems with Applications,
  39(8), 7420-7431.
- Welch, P. (1967). The use of fast Fourier transform for the estimation of power
  spectra. IEEE Transactions on Audio and Electroacoustics, 15(2), 70-73.
- Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing
  (3rd ed.). Pearson.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.signal import welch

from src.streaming.filters import CausalBandpass
from src.streaming.window_buffer import SlidingWindowBuffer


FS_ACC = 100
WINDOW_MS = 2000
HOP_MS = 100
REP_BAND_LOW = 0.3
REP_BAND_HIGH = 1.5

# Must mirror src.features.acc_features.ACC_WAMP_THRESHOLD for offline/online
# parity (Hudgins et al. 1993).
ACC_WAMP_THRESHOLD = 0.05


def _acc_features_from_window(acc_mag_win: np.ndarray, fs: int,
                                wamp_threshold: float = ACC_WAMP_THRESHOLD) -> dict:
    """Compute accelerometer features from one window (causal, Welch per-window).

    González-Badillo & Sánchez-Medina 2010 — rep-rate spectral band 0.3-1.5 Hz.
    Welch 1967 — PSD method. Phinyomark et al. 2012 / Hudgins et al. 1993
    amplitude descriptors require no filter state.
    """
    feat_keys = ["acc_rms", "acc_jerk_rms", "acc_dom_freq",
                 "acc_rep_band_power", "acc_rep_band_ratio",
                 "acc_lscore", "acc_mfl", "acc_msr", "acc_wamp"]
    nan_result = {k: float("nan") for k in feat_keys}

    if len(acc_mag_win) < 16:
        return nan_result

    abs_x = np.abs(acc_mag_win)
    rms = float(np.sqrt(np.mean(acc_mag_win ** 2)))
    jerk = np.diff(acc_mag_win - np.mean(acc_mag_win)) * fs
    jerk_rms = float(np.sqrt(np.mean(jerk ** 2))) if len(jerk) > 0 else float("nan")

    diffs = np.abs(np.diff(acc_mag_win))
    wamp = float(np.sum(diffs > wamp_threshold) / max(1, len(diffs)))
    wl_sq = float(np.sum(diffs ** 2))
    mfl = float(np.log10(np.sqrt(wl_sq) + 1e-30))
    msr = float(np.mean(np.sqrt(abs_x)) ** 2)
    lscore = float(np.exp(np.mean(np.log(np.maximum(abs_x, 1e-30)))))

    nperseg = min(256, len(acc_mag_win))
    f, pxx = welch(acc_mag_win, fs=fs, nperseg=nperseg)
    dom_freq = float(f[np.argmax(pxx)])
    rep_mask = (f >= REP_BAND_LOW) & (f <= REP_BAND_HIGH)
    total_power = float(np.sum(pxx))
    rep_band_power = float(np.sum(pxx[rep_mask]))
    rep_band_ratio = rep_band_power / (total_power + 1e-30)

    return {
        "acc_rms": rms,
        "acc_jerk_rms": jerk_rms,
        "acc_dom_freq": dom_freq,
        "acc_rep_band_power": rep_band_power,
        "acc_rep_band_ratio": rep_band_ratio,
        "acc_lscore": lscore,
        "acc_mfl": mfl,
        "acc_msr": msr,
        "acc_wamp": wamp,
    }


class StreamingAccExtractor:
    """Causal streaming accelerometer feature extractor.

    Accepts either pre-computed acc_mag or raw ax/ay/az.
    """

    def __init__(
        self,
        fs: int = FS_ACC,
        window_ms: float = WINDOW_MS,
        hop_ms: float = HOP_MS,
    ) -> None:
        self._fs = fs
        # Causal bandpass 0.5-20 Hz applied after magnitude (Oppenheim & Schafer 2010)
        self._bp = CausalBandpass(0.5, 20.0, fs, order=4)
        win_samp = max(1, int(window_ms * fs / 1000))
        hop_samp = max(1, int(hop_ms * fs / 1000))
        self._window = SlidingWindowBuffer(win_samp, hop_samp)

    def step(
        self,
        ax: np.ndarray,
        ay: np.ndarray,
        az: np.ndarray,
        t_unix_chunk: np.ndarray,
    ) -> list[dict]:
        """Process a chunk of IMU samples.

        Parameters
        ----------
        ax, ay, az   : Raw accelerometer axes.
        t_unix_chunk : Corresponding Unix timestamps.

        Returns
        -------
        List of feature dicts per completed hop.
        """
        ax = np.asarray(ax, dtype=float)
        ay = np.asarray(ay, dtype=float)
        az = np.asarray(az, dtype=float)
        t_unix_chunk = np.asarray(t_unix_chunk, dtype=float)

        # Magnitude FIRST, then bandpass (project spec)
        acc_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        acc_filt = self._bp.step(acc_mag)

        results = []
        for sample, t in zip(acc_filt, t_unix_chunk):
            windows = self._window.push(np.array([sample]))
            for win in windows:
                feats = _acc_features_from_window(win, self._fs)
                feats["t_unix"] = t
                results.append(feats)

        return results

    def step_mag(
        self,
        acc_mag: np.ndarray,
        t_unix_chunk: np.ndarray,
    ) -> list[dict]:
        """Process pre-computed acc_mag (already bandpassed externally)."""
        acc_mag = np.asarray(acc_mag, dtype=float)
        t_unix_chunk = np.asarray(t_unix_chunk, dtype=float)
        acc_filt = self._bp.step(acc_mag)

        results = []
        for sample, t in zip(acc_filt, t_unix_chunk):
            windows = self._window.push(np.array([sample]))
            for win in windows:
                feats = _acc_features_from_window(win, self._fs)
                feats["t_unix"] = t
                results.append(feats)

        return results

    def reset(self) -> None:
        self._bp.reset()
        self._window.reset()
