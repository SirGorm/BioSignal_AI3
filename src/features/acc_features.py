"""
Accelerometer feature extraction — offline (filtfilt allowed).

Computes acc_mag = sqrt(ax^2 + ay^2 + az^2) FIRST, then bandpasses 0.5-20 Hz.
This order is required by the project spec (CLAUDE.md).

Features per window (2 s, hop 100 ms):
  acc_rms            : root-mean-square of acceleration magnitude (Mannini & Sabatini 2010)
  acc_jerk_rms       : RMS of jerk signal (derivative of acc_mag) (Khan et al. 2010)
  acc_dom_freq       : dominant frequency from Welch PSD (Welch 1967)
  acc_rep_band_power : spectral power in 0.3-1.5 Hz rep-frequency band
  acc_rep_band_ratio : rep-band power / total power

The 0.3-1.5 Hz band captures typical strength-training rep rates of
~0.3-1.5 reps/s (González-Badillo & Sánchez-Medina 2010).

References
----------
- Bonomi, A. G., Goris, A. H., Yin, B., & Westerterp, K. R. (2009). Detection of
  type, duration, and intensity of physical activity using an accelerometer.
  Medicine & Science in Sports & Exercise, 41(9), 1770-1777.
- González-Badillo, J. J., & Sánchez-Medina, L. (2010). Movement velocity as a
  measure of loading intensity in resistance training. International Journal of
  Sports Medicine, 31(05), 347-352.
- Khan, A. M., Lee, Y. K., Lee, S. Y., & Kim, T. S. (2010). A triaxial
  accelerometer-based physical-activity recognition via augmented-signal features
  and a hierarchical recognizer. IEEE Transactions on Information Technology in
  Biomedicine, 14(5), 1166-1172.
- Mannini, A., & Sabatini, A. M. (2010). Machine learning methods for classifying
  human physical activity from on-body accelerometers. Sensors, 10(2), 1154-1175.
- Welch, P. (1967). The use of fast Fourier transform for the estimation of power
  spectra. IEEE Transactions on Audio and Electroacoustics, 15(2), 70-73.
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental
  algorithms for scientific computing in Python. Nature Methods, 17, 261-272.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch


FS_ACC = 100  # Hz — verified against metadata.json
WINDOW_MS = 2000  # ms — captures full rep cycle at ~0.5-1 Hz rep rate
HOP_MS = 100
REP_BAND_LOW = 0.3  # Hz — lower bound for rep-rate spectral band
REP_BAND_HIGH = 1.5  # Hz — upper bound (González-Badillo & Sánchez-Medina 2010)


def _nanfill_signal(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill NaN values before filtering."""
    out = arr.copy().astype(float)
    mask = np.isnan(out)
    if not mask.any():
        return out
    idx = np.arange(len(out))
    good = np.where(~mask)[0]
    if len(good) == 0:
        return out
    fp = np.maximum.accumulate(np.where(~mask, idx, 0))
    out[mask] = out[fp[mask]]
    mask2 = np.isnan(out)
    if mask2.any():
        bp = np.minimum.accumulate(np.where(~mask2, idx, len(out) - 1)[::-1])[::-1]
        out[mask2] = out[bp[mask2]]
    return out


def _compute_acc_mag(ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    """Compute acceleration magnitude before bandpass (project spec requirement)."""
    return np.sqrt(ax ** 2 + ay ** 2 + az ** 2)


def _filter_acc_offline(acc_mag: np.ndarray, fs: int = FS_ACC) -> np.ndarray:
    """Bandpass 0.5-20 Hz on acc_mag, zero-phase (offline — filtfilt).

    Applied AFTER magnitude computation per project spec (CLAUDE.md).
    NaN-tolerant: interpolates NaN before filtering (Virtanen et al. 2020).
    """
    if len(acc_mag) < 13:
        return acc_mag.copy()
    acc_mag = _nanfill_signal(acc_mag)
    sos = butter(4, [0.5, 20.0], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, acc_mag)


def _filter_acc_causal(acc_mag: np.ndarray, fs: int = FS_ACC) -> np.ndarray:
    """Causal bandpass 0.5-20 Hz — for parity testing only.

    Uses sosfilt (same as streaming) so offline and streaming agree on
    filter output. Not used in production offline processing.
    Oppenheim & Schafer 2010 — causal IIR with persisted state.
    """
    from scipy.signal import sosfilt, sosfilt_zi
    if len(acc_mag) < 3:
        return acc_mag.copy()
    acc_mag = _nanfill_signal(acc_mag)
    sos = butter(4, [0.5, 20.0], btype="band", fs=fs, output="sos")
    zi = sosfilt_zi(sos) * acc_mag[0]
    filtered, _ = sosfilt(sos, acc_mag, zi=zi)
    return filtered


def acc_mag_window_features(acc_mag_window: np.ndarray, fs: int = FS_ACC) -> dict:
    """Compute accelerometer features for one bandpass-filtered window.

    Parameters
    ----------
    acc_mag_window : 1D array of filtered acc_mag samples.
    fs             : Sample rate (default 100 Hz).

    Returns
    -------
    dict with keys: acc_rms, acc_jerk_rms, acc_dom_freq,
                    acc_rep_band_power, acc_rep_band_ratio
    """
    nan_result = {k: np.nan for k in
                  ["acc_rms", "acc_jerk_rms", "acc_dom_freq",
                   "acc_rep_band_power", "acc_rep_band_ratio"]}

    if len(acc_mag_window) < 16:
        return nan_result

    # RMS — motion intensity (Bonomi et al. 2009)
    rms = float(np.sqrt(np.mean(acc_mag_window ** 2)))

    # Jerk = derivative of acceleration (Khan et al. 2010)
    jerk = np.diff(acc_mag_window - np.mean(acc_mag_window)) * fs
    jerk_rms = float(np.sqrt(np.mean(jerk ** 2))) if len(jerk) > 0 else np.nan

    # Welch PSD (Welch 1967) — use at most 256 samples per segment
    nperseg = min(256, len(acc_mag_window))
    f, pxx = welch(acc_mag_window, fs=fs, nperseg=nperseg)

    # Dominant frequency (Mannini & Sabatini 2010)
    dom_freq = float(f[np.argmax(pxx)])

    # Rep-frequency band power (González-Badillo & Sánchez-Medina 2010)
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
    }


def extract_acc_features(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    t_unix: np.ndarray,
    fs: int = FS_ACC,
    window_ms: float = WINDOW_MS,
    hop_ms: float = HOP_MS,
) -> pd.DataFrame:
    """Extract accelerometer features over sliding windows (offline).

    Parameters
    ----------
    ax, ay, az : Raw accelerometer axes at native fs (100 Hz).
    t_unix     : Unix timestamps at native rate.
    fs         : Sample rate.
    window_ms  : Window length in ms (default 2000 ms).
    hop_ms     : Hop size in ms.

    Returns
    -------
    DataFrame with one row per window.
    """
    # Compute magnitude THEN filter (project spec, CLAUDE.md)
    acc_mag = _compute_acc_mag(ax, ay, az)
    filtered = _filter_acc_offline(acc_mag, fs)

    win_samp = max(1, int(window_ms * fs / 1000))
    hop_samp = max(1, int(hop_ms * fs / 1000))
    n = len(filtered)

    rows = []
    pos = 0
    while pos + win_samp <= n:
        t_center = t_unix[pos + win_samp // 2]
        feats = acc_mag_window_features(filtered[pos : pos + win_samp], fs)
        feats["t_unix"] = t_center
        rows.append(feats)
        pos += hop_samp

    return pd.DataFrame(rows)
