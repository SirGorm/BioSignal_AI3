"""PPG features — neurokit2 ppg_process + manual amplitude/HR fallback.

Returns per-window:
    ppg_hr_bpm        : instantaneous heart rate
    ppg_pulse_amp     : mean trough-to-peak amplitude
    ppg_pulse_amp_cv  : amplitude coefficient of variation
    ppg_hrv_sdnn_ms   : SDNN of inter-beat intervals (ms)

References
----------
- Allen 2007 — PPG and its applications in clinical physiological measurement
- Makowski et al. 2021 — NeuroKit2: a Python toolbox for biosignal processing
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks

from src3.features._common import nanfill, windows


FS_PPG = 100
WINDOW_S = 10.0
HOP_S = 0.1


def _filter_offline(sig: np.ndarray, fs: int = FS_PPG) -> np.ndarray:
    if len(sig) < 13:
        return sig.copy()
    sig = nanfill(sig)
    sos = butter(4, [0.5, 8.0], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def _features_neurokit(sig: np.ndarray, fs: int) -> dict[str, float]:
    """Use neurokit2.ppg_process when available — well-tested PPG pipeline."""
    import neurokit2 as nk
    signals, info = nk.ppg_process(sig, sampling_rate=fs)
    peaks = info.get("PPG_Peaks", [])
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 2:
        return _empty()

    rr_ms = np.diff(peaks) * 1000.0 / fs
    hr_bpm = 60_000.0 / np.mean(rr_ms)
    sdnn_ms = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else float("nan")

    # Pulse amplitude: peak - preceding trough.
    troughs, _ = find_peaks(-sig, distance=int(0.2 * fs))
    amps = []
    for p in peaks:
        before = troughs[troughs < p]
        if len(before):
            amps.append(float(sig[p] - sig[before[-1]]))
    if not amps:
        return {"ppg_hr_bpm": hr_bpm, "ppg_pulse_amp": float("nan"),
                "ppg_pulse_amp_cv": float("nan"), "ppg_hrv_sdnn_ms": sdnn_ms}
    amps = np.asarray(amps)
    return {
        "ppg_hr_bpm": float(hr_bpm),
        "ppg_pulse_amp": float(np.mean(amps)),
        "ppg_pulse_amp_cv": float(np.std(amps) / (np.mean(amps) + 1e-12)),
        "ppg_hrv_sdnn_ms": sdnn_ms,
    }


def _features_fallback(sig: np.ndarray, fs: int) -> dict[str, float]:
    """No-neurokit fallback: scipy find_peaks on the bandpassed signal."""
    if len(sig) < int(0.5 * fs):
        return _empty()
    peaks, _ = find_peaks(sig, distance=int(0.3 * fs),
                           height=float(np.percentile(sig, 70)))
    if len(peaks) < 2:
        return _empty()
    rr_ms = np.diff(peaks) * 1000.0 / fs
    troughs, _ = find_peaks(-sig, distance=int(0.2 * fs))
    amps = []
    for p in peaks:
        before = troughs[troughs < p]
        if len(before):
            amps.append(float(sig[p] - sig[before[-1]]))
    amps = np.asarray(amps) if amps else np.array([np.nan])
    return {
        "ppg_hr_bpm": float(60_000.0 / np.mean(rr_ms)),
        "ppg_pulse_amp": float(np.nanmean(amps)),
        "ppg_pulse_amp_cv": float(np.nanstd(amps) / (np.nanmean(amps) + 1e-12)),
        "ppg_hrv_sdnn_ms": float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else float("nan"),
    }


def _empty() -> dict[str, float]:
    return {"ppg_hr_bpm": float("nan"), "ppg_pulse_amp": float("nan"),
            "ppg_pulse_amp_cv": float("nan"), "ppg_hrv_sdnn_ms": float("nan")}


def window_features(sig: np.ndarray, fs: int = FS_PPG) -> dict[str, float]:
    try:
        return _features_neurokit(sig, fs)
    except Exception:
        return _features_fallback(sig, fs)


def extract_features(
    sig: np.ndarray, t_unix: np.ndarray,
    fs: int = FS_PPG, window_s: float = WINDOW_S, hop_s: float = HOP_S,
) -> pd.DataFrame:
    sig = _filter_offline(sig, fs)
    win = max(1, int(window_s * fs))
    hop = max(1, int(hop_s * fs))
    rows = []
    for s, e in windows(len(sig), win, hop):
        f = window_features(sig[s:e], fs)
        f["t_unix"] = float(t_unix[s + win // 2])
        rows.append(f)
    return pd.DataFrame(rows)
