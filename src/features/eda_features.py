"""
EDA feature extraction — offline (full-session access, filtfilt allowed).

Features per window (10 s, hop 100 ms):
  eda_scl        : skin conductance level — tonic component (Boucsein 2012)
  eda_scr_amp    : phasic SCR peak-to-trough amplitude (Greco et al. 2016)
  eda_scr_count  : number of SCR events in window (Boucsein 2012)
  eda_phasic_mean: mean absolute phasic signal (Posada-Quintero & Chon 2020)

Tonic/phasic decomposition: moving-median baseline (2 s window) subtracted from
the lowpass-filtered signal to yield phasic. This is the standard time-domain
approach when cvxEDA (Greco et al. 2016) is too expensive for per-window use.

References
----------
- Boucsein, W. (2012). Electrodermal Activity (2nd ed.). Springer.
- Greco, A., Valenza, G., Lanata, A., Scilingo, E. P., & Citi, L. (2016). cvxEDA:
  A convex optimization approach to electrodermal activity processing. IEEE
  Transactions on Biomedical Engineering, 63(4), 797-804.
- Posada-Quintero, H. F., & Chon, K. H. (2020). Innovations in electrodermal
  activity data collection and signal processing: A systematic review. Sensors,
  20(2), 479.
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental
  algorithms for scientific computing in Python. Nature Methods, 17, 261-272.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt


FS_EDA = 50  # Hz — verified against metadata.json
WINDOW_S = 10.0  # seconds (Boucsein 2012: SCR latency ~1-5 s, window >=10 s preferred)
HOP_MS = 100
SCR_THRESHOLD = 0.05  # microsiemens threshold for SCR event detection (Boucsein 2012)


def _filter_eda_offline(signal: np.ndarray, fs: int = FS_EDA) -> np.ndarray:
    """Lowpass Butterworth 5 Hz, zero-phase (offline — filtfilt).

    EDA is a slow signal; 5 Hz LP removes motion artifacts while preserving
    SCR dynamics (Posada-Quintero & Chon 2020).
    Virtanen et al. 2020 — scipy.signal usage.
    """
    if len(signal) < 13:  # min padlen for sosfiltfilt
        return signal.copy()
    sos = butter(4, 5.0, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def _filter_eda_causal(signal: np.ndarray, fs: int = FS_EDA) -> np.ndarray:
    """Causal lowpass 5 Hz — for parity testing only.

    Uses sosfilt (same as streaming) to test state management independently
    from filtfilt vs sosfilt amplitude differences.
    Oppenheim & Schafer 2010.
    """
    from scipy.signal import sosfilt, sosfilt_zi
    if len(signal) < 3:
        return signal.copy()
    sos = butter(4, 5.0, btype="low", fs=fs, output="sos")
    zi = sosfilt_zi(sos) * signal[0]
    filtered, _ = sosfilt(sos, signal, zi=zi)
    return filtered


def eda_window_features(
    eda_window: np.ndarray,
    fs: int = FS_EDA,
    scr_thresh: float = SCR_THRESHOLD,
) -> dict:
    """Compute EDA tonic and phasic features for one pre-filtered window.

    Parameters
    ----------
    eda_window : 1D array of lowpass-filtered EDA samples.
    fs         : Sample rate (default 50 Hz).
    scr_thresh : Amplitude threshold for SCR event detection (Boucsein 2012).

    Returns
    -------
    dict with keys: eda_scl, eda_scr_amp, eda_scr_count, eda_phasic_mean
    """
    nan_result = {k: np.nan for k in
                  ["eda_scl", "eda_scr_amp", "eda_scr_count", "eda_phasic_mean"]}

    if len(eda_window) < 5 or np.all(np.isnan(eda_window)):
        return nan_result

    # Tonic level = median (robust; Boucsein 2012)
    scl = float(np.nanmedian(eda_window))

    # Phasic component: subtract a 2-s moving median baseline (Greco et al. 2016)
    med_win = max(3, int(2.0 * fs))
    med_win = min(med_win, len(eda_window))
    if med_win % 2 == 0:
        med_win += 1
    baseline = median_filter(eda_window, size=med_win)
    phasic = eda_window - baseline

    scr_amp = float(np.nanmax(phasic) - np.nanmin(phasic))

    # SCR event count: rising edges crossing threshold (Boucsein 2012)
    above = (phasic > scr_thresh).astype(int)
    n_scr = int(np.sum(np.diff(above) == 1))

    phasic_mean = float(np.nanmean(np.abs(phasic)))

    return {
        "eda_scl": scl,
        "eda_scr_amp": scr_amp,
        "eda_scr_count": n_scr,
        "eda_phasic_mean": phasic_mean,
    }


def extract_eda_features(
    eda_raw: np.ndarray,
    t_unix: np.ndarray,
    fs: int = FS_EDA,
    window_s: float = WINDOW_S,
    hop_s: float = HOP_MS / 1000.0,
    baseline_scl: float | None = None,
) -> pd.DataFrame:
    """Extract EDA features over sliding windows (offline — filtfilt allowed).

    Parameters
    ----------
    eda_raw      : Raw EDA signal at native fs (50 Hz).
    t_unix       : Unix timestamps at native rate.
    fs           : Sample rate.
    window_s     : Window length in seconds (default 10 s).
    hop_s        : Hop size in seconds (default 100 ms).
    baseline_scl : Baseline SCL median for normalisation (optional).

    Returns
    -------
    DataFrame with one row per window.
    """
    if len(eda_raw) == 0 or np.all(np.isnan(eda_raw)):
        return pd.DataFrame()

    filtered = _filter_eda_offline(eda_raw, fs)
    win_samp = max(1, int(window_s * fs))
    hop_samp = max(1, int(hop_s * fs))
    n = len(filtered)

    rows = []
    pos = 0
    while pos + win_samp <= n:
        t_center = t_unix[pos + win_samp // 2]
        feats = eda_window_features(filtered[pos : pos + win_samp], fs)
        feats["t_unix"] = t_center
        if baseline_scl is not None and not np.isnan(baseline_scl) and baseline_scl > 0:
            feats["eda_scl_rel"] = feats["eda_scl"] / baseline_scl
        else:
            feats["eda_scl_rel"] = np.nan
        rows.append(feats)
        pos += hop_samp

    return pd.DataFrame(rows)
