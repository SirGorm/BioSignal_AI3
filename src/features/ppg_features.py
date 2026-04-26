"""
PPG (green wavelength) feature extraction — offline (filtfilt allowed).

Green wavelength only per project spec (CLAUDE.md). Sample rate read from
metadata.json per recording (50 Hz for rec_001, 100 Hz for rec_012+).

Features per window (10 s, hop 100 ms):
  ppg_hr           : heart rate in BPM from pulse intervals (Allen 2007)
  ppg_pulse_amp    : mean peak-to-trough pulse amplitude (Allen 2007)
  ppg_pulse_amp_var: standard deviation of pulse amplitude (Tamura et al. 2014)

Cross-validation: ppg_hr should agree with ecg_hr within 5 bpm during rest;
persistent large discrepancy indicates motion artifact (Maeda et al. 2011).

Green wavelength preferred for wrist sites because it provides the best
motion-artifact rejection among common PPG wavelengths (Castaneda et al. 2018).

References
----------
- Allen, J. (2007). Photoplethysmography and its application in clinical
  physiological measurement. Physiological Measurement, 28(3), R1.
- Castaneda, D., Esparza, A., Ghamari, M., Soltanpur, C., & Nazeran, H. (2018).
  A review on wearable photoplethysmography sensors and their potential future
  applications in health care. International Journal of Biosensors &
  Bioelectronics, 4(4), 195-202.
- Maeda, Y., Sekine, M., & Tamura, T. (2011). Relationship between measurement
  site and motion artifacts in wearable reflected photoplethysmography. Journal
  of Medical Systems, 35(5), 969-976.
- Tamura, T., Maeda, Y., Sekine, M., & Yoshida, M. (2014). Wearable
  photoplethysmographic sensors—past and present. Electronics, 3(2), 282-302.
- Makowski, D., Pham, T., Lau, Z. J., et al. (2021). NeuroKit2: A Python toolbox
  for neurophysiological signal processing. Behavior Research Methods, 53(4),
  1689-1696.
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental
  algorithms for scientific computing in Python. Nature Methods, 17, 261-272.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks


FS_PPG_DEFAULT = 100  # Hz — default; read from metadata.json per recording
WINDOW_S = 10.0  # seconds for stable HR estimation (Allen 2007)
HOP_MS = 100


def _filter_ppg_offline(signal: np.ndarray, fs: int = FS_PPG_DEFAULT) -> np.ndarray:
    """Zero-phase Butterworth 0.5-8 Hz bandpass (offline — filtfilt).

    0.5 Hz removes slow baseline wander; 8 Hz removes high-freq noise while
    preserving pulse morphology at up to 240 bpm (Allen 2007).
    Virtanen et al. 2020 — scipy.signal.
    """
    if len(signal) < 13:
        return signal.copy()
    sos = butter(4, [0.5, 8.0], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def detect_ppg_peaks(ppg_filtered: np.ndarray, fs: int) -> np.ndarray:
    """Detect systolic peaks in filtered PPG (offline — find_peaks over window only).

    Uses scipy.signal.find_peaks with physiological constraints:
    - minimum inter-peak distance = 200 ms (300 bpm upper limit)
    - minimum height = 0.3 * max amplitude

    Allen 2007 — peak detection for PPG-derived HR.

    Returns array of peak indices within the window.
    """
    if len(ppg_filtered) < 3:
        return np.array([], dtype=int)

    min_dist = max(1, int(0.2 * fs))  # 200 ms refractory (300 bpm max)
    height_thresh = 0.3 * np.max(np.abs(ppg_filtered))
    peaks, _ = find_peaks(ppg_filtered, distance=min_dist, height=height_thresh)
    return peaks


def ppg_window_features(
    ppg_window: np.ndarray,
    fs: int = FS_PPG_DEFAULT,
) -> dict:
    """Compute PPG features for one pre-filtered window.

    Parameters
    ----------
    ppg_window : 1D array of filtered PPG (green wavelength) samples.
    fs         : Sample rate.

    Returns
    -------
    dict with keys: ppg_hr, ppg_pulse_amp, ppg_pulse_amp_var
    """
    nan_result = {k: np.nan for k in
                  ["ppg_hr", "ppg_pulse_amp", "ppg_pulse_amp_var"]}

    if len(ppg_window) < int(0.5 * fs):
        return nan_result

    peaks = detect_ppg_peaks(ppg_window, fs)
    if len(peaks) < 2:
        return nan_result

    # HR from peak-to-peak intervals (Allen 2007)
    pp_s = np.diff(peaks) / fs
    # Physiological filter: 30-240 bpm → PP interval 0.25-2.0 s
    valid_pp = pp_s[(pp_s >= 0.25) & (pp_s <= 2.0)]
    if len(valid_pp) < 1:
        return nan_result
    hr = float(60.0 / np.mean(valid_pp))

    # Pulse amplitude: peak-to-trough around each systolic peak (Allen 2007)
    half_win = max(1, int(0.3 * fs))
    amps = []
    for p in peaks:
        a = max(0, p - half_win)
        b = min(len(ppg_window), p + half_win)
        amps.append(float(np.max(ppg_window[a:b]) - np.min(ppg_window[a:b])))

    return {
        "ppg_hr": hr,
        "ppg_pulse_amp": float(np.mean(amps)),
        "ppg_pulse_amp_var": float(np.std(amps)) if len(amps) > 1 else np.nan,
    }


def extract_ppg_features(
    ppg_raw: np.ndarray,
    t_unix: np.ndarray,
    fs: int = FS_PPG_DEFAULT,
    window_s: float = WINDOW_S,
    hop_s: float = HOP_MS / 1000.0,
) -> pd.DataFrame:
    """Extract PPG (green) features over sliding windows (offline).

    Parameters
    ----------
    ppg_raw  : Raw PPG (green) signal at native fs.
    t_unix   : Unix timestamps at native rate.
    fs       : Sample rate (read from metadata.json["sampling_rates"]["ppg"]).
    window_s : Window length in seconds.
    hop_s    : Hop size in seconds.

    Returns
    -------
    DataFrame with one row per window.
    """
    if len(ppg_raw) == 0 or np.all(np.isnan(ppg_raw)):
        return pd.DataFrame()

    filtered = _filter_ppg_offline(ppg_raw, fs)
    win_samp = max(1, int(window_s * fs))
    hop_samp = max(1, int(hop_s * fs))
    n = len(filtered)

    rows = []
    pos = 0
    while pos + win_samp <= n:
        t_center = t_unix[pos + win_samp // 2]
        feats = ppg_window_features(filtered[pos : pos + win_samp], fs)
        feats["t_unix"] = t_center
        rows.append(feats)
        pos += hop_samp

    return pd.DataFrame(rows)
