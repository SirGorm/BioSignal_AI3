"""
Temperature feature extraction — offline.

Features per window (60 s, hop 100 ms):
  temp_mean  : mean skin temperature in window
  temp_slope : linear slope in degrees/second (rising = metabolic heat)
  temp_range : max - min within window

NaN-tolerant: when temperature.csv is empty (many recordings after rec_006),
all outputs are NaN. This is expected and handled gracefully — the ML pipeline
must tolerate NaN columns for temp features.

References
----------
- [REF NEEDED: skin temperature as fatigue indicator during exercise]
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental
  algorithms for scientific computing in Python. Nature Methods, 17, 261-272.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


FS_TEMP = 1  # Hz — verified against metadata.json (declared 1 Hz)
WINDOW_S = 60.0  # seconds — very slow signal, needs long window
HOP_MS = 100


def _filter_temp_offline(signal: np.ndarray, fs: int = FS_TEMP) -> np.ndarray:
    """Lowpass 0.1 Hz to remove noise (offline — filtfilt).

    Temperature signal is very slow; LP at 0.1 Hz removes measurement noise
    while preserving true thermal trends.
    """
    if np.all(np.isnan(signal)) or len(signal) < 13:
        return signal.copy()
    # Replace NaN with forward-fill before filtering
    arr = _nanfill(signal)
    sos = butter(2, 0.1, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, arr)


def _nanfill(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill NaN values."""
    out = arr.copy()
    mask = np.isnan(out)
    if not mask.any():
        return out
    idx = np.where(~mask)[0]
    if len(idx) == 0:
        return out
    # Forward fill
    fp = np.maximum.accumulate(np.where(~mask, np.arange(len(out)), 0))
    out[mask] = out[fp[mask]]
    # Backward fill remaining leading NaN
    mask2 = np.isnan(out)
    if mask2.any():
        bp = np.minimum.accumulate(np.where(~mask2, np.arange(len(out)), len(out) - 1)[::-1])[::-1]
        out[mask2] = out[bp[mask2]]
    return out


def temp_window_features(
    temp_window: np.ndarray,
    t_window: np.ndarray,
) -> dict:
    """Compute temperature features for one window.

    Parameters
    ----------
    temp_window : 1D array of temperature samples (may contain NaN).
    t_window    : Timestamps in seconds for each sample.

    Returns
    -------
    dict with keys: temp_mean, temp_slope, temp_range
    """
    nan_result = {"temp_mean": np.nan, "temp_slope": np.nan, "temp_range": np.nan}

    valid = ~np.isnan(temp_window)
    if valid.sum() < 5:
        return nan_result

    t_valid = t_window[valid] - t_window[valid][0]
    v_valid = temp_window[valid]

    return {
        "temp_mean": float(np.mean(v_valid)),
        "temp_slope": float(np.polyfit(t_valid, v_valid, 1)[0]),
        "temp_range": float(np.max(v_valid) - np.min(v_valid)),
    }


def extract_temp_features(
    temp_raw: np.ndarray,
    t_unix: np.ndarray,
    fs: int = FS_TEMP,
    window_s: float = WINDOW_S,
    hop_s: float = HOP_MS / 1000.0,
    baseline_mean: float | None = None,
) -> pd.DataFrame:
    """Extract temperature features over sliding windows.

    Parameters
    ----------
    temp_raw     : Raw temperature signal (1 Hz). May be all-NaN.
    t_unix       : Unix timestamps at native rate.
    fs           : Sample rate (default 1 Hz).
    window_s     : Window length in seconds (default 60 s).
    hop_s        : Hop size in seconds.
    baseline_mean: Baseline temperature mean for normalisation (optional).

    Returns
    -------
    DataFrame with one row per window (all NaN when temp_raw is all-NaN).
    """
    # NaN-tolerant: return empty DataFrame if no data
    if len(temp_raw) == 0 or np.all(np.isnan(temp_raw)):
        return pd.DataFrame()

    filtered = _filter_temp_offline(temp_raw, fs)
    win_samp = max(1, int(window_s * fs))
    hop_samp = max(1, int(hop_s * fs))
    n = len(filtered)

    rows = []
    pos = 0
    while pos + win_samp <= n:
        t_center = t_unix[pos + win_samp // 2]
        t_rel = t_unix[pos : pos + win_samp]
        feats = temp_window_features(filtered[pos : pos + win_samp], t_rel)
        feats["t_unix"] = t_center
        if (baseline_mean is not None
                and not np.isnan(baseline_mean)
                and baseline_mean > 0):
            feats["temp_mean_rel"] = feats["temp_mean"] / baseline_mean
        else:
            feats["temp_mean_rel"] = np.nan
        rows.append(feats)
        pos += hop_samp

    return pd.DataFrame(rows)
