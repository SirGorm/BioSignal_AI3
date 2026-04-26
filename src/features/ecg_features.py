"""
ECG feature extraction — offline (full-session access, filtfilt allowed).

Features per window (30 s, hop 100 ms):
  ecg_hr      : beats per minute from RR intervals
  ecg_rmssd   : root-mean-square of successive RR differences (Task Force 1996)
  ecg_sdnn    : standard deviation of NN intervals (Task Force 1996)
  ecg_pnn50   : proportion of successive RR diffs > 50 ms (Task Force 1996)
  ecg_mean_rr : mean RR interval in ms

R-peak detection via Pan & Tompkins derivative + threshold algorithm,
implemented through NeuroKit2 (Makowski et al. 2021).

References
----------
- Task Force of the European Society of Cardiology and the North American Society
  of Pacing and Electrophysiology (1996). Heart rate variability: standards of
  measurement, physiological interpretation, and clinical use. Circulation, 93(5),
  1043-1065.
- Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. IEEE
  Transactions on Biomedical Engineering, BME-32(3), 230-236.
- Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability
  metrics and norms. Frontiers in Public Health, 5, 258.
- Makowski, D., Pham, T., Lau, Z. J., et al. (2021). NeuroKit2: A Python toolbox
  for neurophysiological signal processing. Behavior Research Methods, 53(4), 1689-1696.
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental
  algorithms for scientific computing in Python. Nature Methods, 17, 261-272.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, sosfilt


FS_ECG = 500  # Hz — verified against dataset/recording_012/metadata.json
WINDOW_S = 30.0  # seconds for HRV stability (Shaffer & Ginsberg 2017)


def _bandpass_ecg(signal: np.ndarray, fs: int = FS_ECG) -> np.ndarray:
    """Zero-phase Butterworth 0.5-40 Hz + 50 Hz notch (offline only — uses filtfilt).

    Virtanen et al. 2020 — scipy.signal usage.
    """
    sos = butter(4, [0.5, 40.0], btype="band", fs=fs, output="sos")
    filtered = filtfilt(*_sos_to_ba_approx(sos), signal)
    # 50 Hz notch
    b_notch, a_notch = iirnotch(50.0, Q=30, fs=fs)
    filtered = filtfilt(b_notch, a_notch, filtered)
    return filtered


def _sos_to_ba_approx(sos):
    """Convert SOS to (b, a) for filtfilt when b/a form is needed."""
    from scipy.signal import sosfilt
    # Use direct filtfilt on sos
    return sos, None  # placeholder — we use sos directly below


def _nanfill_signal(signal: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill NaN values before filtering."""
    out = signal.copy().astype(float)
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


def _bandpass_ecg_v2(signal: np.ndarray, fs: int = FS_ECG) -> np.ndarray:
    """Zero-phase Butterworth 0.5-40 Hz + 50 Hz notch using filtfilt on sos directly.

    NaN-tolerant: interpolates NaN before filtering (Virtanen et al. 2020).
    """
    from scipy.signal import sosfiltfilt, butter, iirnotch
    signal = _nanfill_signal(signal)
    sos = butter(4, [0.5, 40.0], btype="band", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, signal)
    b_notch, a_notch = iirnotch(50.0, Q=30, fs=fs)
    filtered = filtfilt(b_notch, a_notch, filtered)
    return filtered


def detect_r_peaks(ecg_filtered: np.ndarray, fs: int = FS_ECG) -> np.ndarray:
    """Detect R-peaks using NeuroKit2 (Makowski et al. 2021) with Pan-Tompkins
    derivative-threshold detector (Pan & Tompkins 1985).

    Falls back to a simple threshold detector if NeuroKit2 is unavailable.

    Returns array of sample indices for each R-peak.
    """
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(ecg_filtered, sampling_rate=fs, method="pantompkins1985")
        peaks = info["ECG_R_Peaks"]
        return np.asarray(peaks, dtype=int)
    except ImportError:
        # Simple fallback: threshold-based detector
        return _simple_r_detector(ecg_filtered, fs)


def _simple_r_detector(signal: np.ndarray, fs: int) -> np.ndarray:
    """Pan-Tompkins inspired derivative + threshold R-peak detector.
    Pan & Tompkins 1985.
    """
    from scipy.signal import find_peaks
    # Derivative filter approximating Pan-Tompkins
    diff = np.diff(signal)
    squared = diff ** 2
    # Moving average integration (≈200 ms window)
    win = int(0.2 * fs)
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")
    threshold = 0.3 * np.max(integrated)
    refractory = int(0.2 * fs)  # 200 ms
    peaks, _ = find_peaks(integrated, height=threshold, distance=refractory)
    return peaks


def rr_intervals_ms(r_peaks: np.ndarray, fs: int = FS_ECG) -> np.ndarray:
    """Convert R-peak sample indices to RR intervals in milliseconds."""
    if len(r_peaks) < 2:
        return np.array([], dtype=float)
    return np.diff(r_peaks) * 1000.0 / fs


def ecg_hrv_features(rr_ms: np.ndarray) -> dict:
    """Compute HRV time-domain features from RR intervals (Task Force 1996).

    Parameters
    ----------
    rr_ms : RR intervals in milliseconds within the window.

    Returns
    -------
    dict with keys: ecg_hr, ecg_rmssd, ecg_sdnn, ecg_pnn50, ecg_mean_rr
    """
    nan_result = {
        "ecg_hr": np.nan,
        "ecg_rmssd": np.nan,
        "ecg_sdnn": np.nan,
        "ecg_pnn50": np.nan,
        "ecg_mean_rr": np.nan,
    }
    rr = np.asarray(rr_ms, dtype=float)
    # Remove physiologically implausible RR (< 300 ms or > 2000 ms) as simple
    # artefact rejection (Shaffer & Ginsberg 2017)
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 3:
        return nan_result
    hr = 60_000.0 / np.mean(rr)
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    sdnn = float(np.std(rr, ddof=1))
    pnn50 = float(np.mean(np.abs(np.diff(rr)) > 50) * 100)
    return {
        "ecg_hr": float(hr),
        "ecg_rmssd": float(rmssd),
        "ecg_sdnn": float(sdnn),
        "ecg_pnn50": float(pnn50),
        "ecg_mean_rr": float(np.mean(rr)),
    }


def extract_ecg_features(
    ecg_raw: np.ndarray,
    t_unix: np.ndarray,
    fs: int = FS_ECG,
    window_s: float = WINDOW_S,
    hop_s: float = 0.1,
) -> pd.DataFrame:
    """Extract ECG features over sliding windows (offline — filtfilt allowed).

    Parameters
    ----------
    ecg_raw  : Raw ECG signal at native fs (500 Hz).
    t_unix   : Unix timestamps corresponding to each sample.
    fs       : Sample rate (default 500 Hz).
    window_s : Window length in seconds (default 30 s for HRV stability).
    hop_s    : Hop size in seconds (default 100 ms).

    Returns
    -------
    DataFrame indexed by t_window_center_s with ECG features.
    """
    filtered = _bandpass_ecg_v2(ecg_raw, fs)
    r_peaks = detect_r_peaks(filtered, fs)
    # RR intervals in ms and their midpoint times
    rr_ms = rr_intervals_ms(r_peaks, fs)
    # Time of each RR interval = time of the trailing R-peak
    if len(r_peaks) >= 2:
        rr_times = t_unix[r_peaks[1:]]  # unix time of each beat boundary
    else:
        rr_times = np.array([])

    win_samp = int(window_s * fs)
    hop_samp = max(1, int(hop_s * fs))
    n = len(ecg_raw)

    rows = []
    pos = 0
    while pos + win_samp <= n:
        t_center = t_unix[pos + win_samp // 2]
        t_start = t_unix[pos]
        t_end = t_unix[pos + win_samp - 1]
        # RR intervals whose trailing peak falls in [t_start, t_end]
        mask = (rr_times >= t_start) & (rr_times <= t_end)
        rr_window = rr_ms[mask]
        feats = ecg_hrv_features(rr_window)
        feats["t_unix"] = t_center
        rows.append(feats)
        pos += hop_samp

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
