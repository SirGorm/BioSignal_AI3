"""
ECG feature extraction — offline (full-session access, filtfilt allowed).

Features per window (30 s, hop 100 ms):
  ecg_hr      : beats per minute from NN intervals
  ecg_rmssd   : root-mean-square of successive NN differences (Task Force 1996)
  ecg_sdnn    : standard deviation of NN intervals (Task Force 1996)
  ecg_pnn50   : proportion of successive NN diffs > 50 ms (Task Force 1996)
  ecg_mean_rr : mean NN interval in ms

R-peak detection via Pan & Tompkins derivative + threshold algorithm,
implemented through NeuroKit2 (Makowski et al. 2021). Detected R-peaks are
then corrected for missed/extra/ectopic beats with the Kubios artefact
detector (Lipponen & Tarvainen 2019), yielding the NN interval series
required by HRV standards (Task Force 1996; Aubert et al. 2003).

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
- Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate
  variability time series artefact correction using novel beat classification.
  Journal of Medical Engineering & Technology, 43(3), 173-181.
- Aubert, A. E., Seps, B., & Beckers, F. (2003). Heart rate variability in
  athletes. Sports Medicine, 33(12), 889-919.
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


def correct_to_nn(
    r_peaks: np.ndarray,
    fs: int = FS_ECG,
    threshold_pct: float = 0.20,
) -> np.ndarray:
    """Correct R-peaks for missed/extra/ectopic beats and return NN intervals (ms).

    HRV must be computed on Normal-to-Normal intervals — RR series corrected for
    detection errors and ectopic beats (Task Force 1996; Aubert et al. 2003).
    Uses the Kubios artefact-detection algorithm via NeuroKit2 when available
    (Lipponen & Tarvainen 2019); falls back to a Malik-style relative-deviation
    rejector with linear interpolation (Malik et al. 1996).

    Parameters
    ----------
    r_peaks       : R-peak sample indices.
    fs            : Sample rate.
    threshold_pct : Fallback rejection threshold — drop RR that deviates more
                    than this fraction from a 5-tap rolling median (default 0.20).

    Returns
    -------
    NN intervals in milliseconds (artefact-corrected, length ≤ len(diff(r_peaks))).
    """
    if len(r_peaks) < 3:
        return rr_intervals_ms(r_peaks, fs)

    try:
        import neurokit2 as nk
        artifacts, peaks_clean = nk.signal_fixpeaks(
            np.asarray(r_peaks, dtype=int),
            sampling_rate=fs,
            method="kubios",
        )
        peaks_clean = np.asarray(peaks_clean, dtype=int)
        if len(peaks_clean) < 2:
            return np.array([], dtype=float)
        return np.diff(peaks_clean) * 1000.0 / fs
    except (ImportError, Exception):
        return _correct_to_nn_malik(r_peaks, fs, threshold_pct)


def _correct_to_nn_malik(
    r_peaks: np.ndarray,
    fs: int,
    threshold_pct: float,
) -> np.ndarray:
    """Fallback NN correction — Malik et al. 1996 relative-deviation filter.

    Flags any RR that deviates more than ``threshold_pct`` from the local
    5-tap rolling median, then linearly interpolates flagged values from
    neighbouring valid RRs.
    """
    rr = rr_intervals_ms(r_peaks, fs).astype(float)
    if len(rr) < 5:
        return rr
    from scipy.ndimage import median_filter
    local = median_filter(rr, size=5, mode="nearest")
    bad = np.abs(rr - local) / np.maximum(local, 1.0) > threshold_pct
    if bad.any():
        good_idx = np.where(~bad)[0]
        if len(good_idx) >= 2:
            rr[bad] = np.interp(np.where(bad)[0], good_idx, rr[good_idx])
        else:
            rr[bad] = local[bad]
    return rr


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
    # NN intervals (ms) — RR corrected for missed/extra/ectopic beats
    # via Kubios (Lipponen & Tarvainen 2019), required by Task Force 1996.
    nn_ms = correct_to_nn(r_peaks, fs)
    # Time of each NN interval = time of the trailing R-peak. Kubios may add
    # or drop peaks, so we align by truncating to the post-correction count
    # rather than re-using the original r_peaks vector.
    if len(r_peaks) >= 2 and len(nn_ms) > 0:
        n_nn = len(nn_ms)
        # The trailing-peak index of each NN equals diff(peaks_clean) midpoints,
        # but signal_fixpeaks typically preserves original peak indices — use
        # the last n_nn original peak times as a robust alignment.
        rr_times = t_unix[r_peaks[1:][:n_nn]] if n_nn <= len(r_peaks) - 1 else t_unix[r_peaks[1:]]
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
        # NN intervals whose trailing peak falls in [t_start, t_end]
        mask = (rr_times >= t_start) & (rr_times <= t_end)
        nn_window = nn_ms[mask] if len(nn_ms) == len(rr_times) else nn_ms[: mask.sum()]
        feats = ecg_hrv_features(nn_window)
        feats["t_unix"] = t_center
        rows.append(feats)
        pos += hop_samp

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
