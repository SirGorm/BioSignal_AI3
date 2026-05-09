"""EMG features — libemg FeatureExtractor + Dimitrov FInsm5 (custom).

libemg provides RMS, IEMG, MNF, MDF, WAMP, MFL, MSR, L-Score (LD) directly.
Dimitrov FInsm5 = M(-1) / M(5) is not in libemg, so we compute it from the
window's PSD (Welch) — same formula as src/features/emg_features.py.

Causal filtering (CausalBandpass + CausalNotch) lives in src3/streaming/ —
this module assumes the signal arrives already filtered (envelope or raw).

References
----------
- Eddy et al. 2024 — LibEMG: An Open-Source EMG Library
- Dimitrov et al. 2006 — FInsm5 fatigue index
- Phinyomark et al. 2012 — feature reduction for EMG
- Hudgins et al. 1993 — WAMP
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import welch, butter, sosfiltfilt, iirnotch, filtfilt

from src3.features._common import nanfill, windows


FS_EMG = 2000
WINDOW_MS = 500
HOP_MS = 100
WAMP_THRESHOLD_V = 5e-5  # 50 µV — Hudgins et al. 1993


def _filter_offline(sig: np.ndarray, fs: int = FS_EMG) -> np.ndarray:
    """20–450 Hz BP + 50 Hz notch (zero-phase, offline)."""
    sig = nanfill(sig)
    sos = butter(4, [20.0, 450.0], btype="band", fs=fs, output="sos")
    y = sosfiltfilt(sos, sig)
    b, a = iirnotch(50.0, Q=30, fs=fs)
    return filtfilt(b, a, y)


def _dimitrov_finsm5(psd: np.ndarray, freqs: np.ndarray) -> float:
    """FInsm5 = M(-1) / M(5)  with f restricted to (0, fs/2]."""
    keep = freqs > 0
    f = freqs[keep]
    p = psd[keep]
    m_minus = float(np.sum((f ** -1) * p))
    m_plus5 = float(np.sum((f ** 5) * p))
    if m_plus5 <= 0:
        return float("nan")
    return m_minus / m_plus5


def window_features(sig: np.ndarray, fs: int = FS_EMG,
                    wamp_threshold: float = WAMP_THRESHOLD_V) -> dict[str, float]:
    """Per-window EMG features. Returns dict of NaN if window too short.

    Tries libemg.FeatureExtractor for the canonical features; falls back to
    in-house numpy for the same formulas if libemg is not installed (so the
    module imports cleanly in CI without the optional dep).
    """
    keys = ("emg_rms", "emg_iemg", "emg_mnf", "emg_mdf", "emg_dimitrov",
            "emg_lscore", "emg_mfl", "emg_msr", "emg_wamp")
    if len(sig) < 16:
        return {k: float("nan") for k in keys}

    out: dict[str, float] = {}

    try:
        from libemg.feature_extractor import FeatureExtractor

        fe = FeatureExtractor()
        # libemg expects (windows, channels, samples)
        x = sig[None, None, :]
        feat_set = ["MAV", "RMS", "WL", "MFL", "MSR", "LD", "WAMP"]
        # Spectral features need a Welch PSD inside libemg — passing fs.
        spec_set = ["MNF", "MDF"]
        f1 = fe.extract_features(feat_set, x)
        f2 = fe.extract_features(spec_set, x, fs=fs)
        out["emg_rms"]   = float(f1["RMS"][0, 0])
        out["emg_mfl"]   = float(f1["MFL"][0, 0])
        out["emg_msr"]   = float(f1["MSR"][0, 0])
        out["emg_lscore"] = float(f1["LD"][0, 0])
        out["emg_iemg"]  = float(np.sum(np.abs(sig)))
        # WL is Σ|Δx|; WAMP is Σ I(|Δx|>τ). libemg's WAMP uses an internal
        # default threshold and returns a fraction; we want the project's
        # 50 µV threshold to stay comparable, so compute WAMP ourselves.
        diffs = np.abs(np.diff(sig))
        out["emg_wamp"]  = float(np.sum(diffs > wamp_threshold) / max(1, len(diffs)))
        out["emg_mnf"]   = float(f2["MNF"][0, 0])
        out["emg_mdf"]   = float(f2["MDF"][0, 0])

    except Exception:
        # In-house fallback (matches src/features/emg_features.py exactly).
        out["emg_rms"]    = float(np.sqrt(np.mean(sig ** 2)))
        out["emg_iemg"]   = float(np.sum(np.abs(sig)))
        diffs = np.abs(np.diff(sig))
        out["emg_wamp"]   = float(np.sum(diffs > wamp_threshold) / max(1, len(diffs)))
        out["emg_mfl"]    = float(np.log10(np.sqrt(float(np.sum(diffs ** 2))) + 1e-30))
        out["emg_msr"]    = float(np.mean(np.sqrt(np.abs(sig))) ** 2)
        out["emg_lscore"] = float(np.exp(np.mean(np.log(np.maximum(np.abs(sig), 1e-30)))))
        nperseg = min(256, len(sig))
        f, p = welch(sig, fs=fs, nperseg=nperseg)
        band = (f >= 20.0) & (f <= 450.0)
        f_b, p_b = f[band], p[band]
        if p_b.sum() > 0:
            out["emg_mnf"] = float(np.sum(f_b * p_b) / np.sum(p_b))
            cdf = np.cumsum(p_b)
            half = cdf[-1] / 2
            out["emg_mdf"] = float(f_b[np.searchsorted(cdf, half)])
        else:
            out["emg_mnf"] = float("nan")
            out["emg_mdf"] = float("nan")

    # Dimitrov FInsm5 is not in libemg — compute from Welch PSD over the
    # full physiological band. Same band/window as src/features/emg_features.py.
    nperseg = min(256, len(sig))
    f, p = welch(sig, fs=fs, nperseg=nperseg)
    band = (f >= 20.0) & (f <= 450.0)
    out["emg_dimitrov"] = _dimitrov_finsm5(p[band], f[band])

    return out


def extract_features(
    emg: np.ndarray,
    t_unix: np.ndarray,
    fs: int = FS_EMG,
    window_ms: float = WINDOW_MS,
    hop_ms: float = HOP_MS,
    do_filter: bool = True,
) -> pd.DataFrame:
    """Sliding-window feature extraction over a full session."""
    sig = _filter_offline(emg, fs) if do_filter else nanfill(emg)
    win = max(1, int(window_ms * fs / 1000))
    hop = max(1, int(hop_ms * fs / 1000))
    rows: list[dict] = []
    for s, e in windows(len(sig), win, hop):
        feats = window_features(sig[s:e], fs)
        feats["t_unix"] = float(t_unix[s + win // 2])
        rows.append(feats)
    return pd.DataFrame(rows)
