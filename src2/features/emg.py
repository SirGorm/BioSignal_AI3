"""EMG features via libemg.

Replaces all hand-rolled MNF / MDF / Dimitrov / RMS / IEMG / WAMP / MFL / MSR
code in `src/features/emg_features.py` with libemg's `FeatureExtractor`.

libemg.feature_extractor.FeatureExtractor exposes one-shot batch extraction
over a `(N_windows, N_channels, window_length)` ndarray. We wrap it to:
  - filter (causal-friendly Butterworth + 50 Hz notch via scipy)
  - window the raw 2000 Hz EMG signal
  - call FeatureExtractor.extract_features(['MAV', 'RMS', ...], windows)
  - return a tidy DataFrame with one row per window

References:
- Eddy et al. 2023 — libemg: an open-source library for the recognition of
  electromyographic signals (https://github.com/LibEMG/libemg)
- De Luca 1997 — surface EMG biomechanics fundamentals
- Dimitrov et al. 2006 — fatigue spectral indices
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

from src2.features._common import sliding_windows  # noqa: E402

FS_EMG_DEFAULT = 2000

# libemg feature codes — see libemg.feature_extractor.FeatureExtractor.get_feature_list().
# These cover everything the existing src/features/emg_features.py computes,
# plus a few canonical extras (Hjorth params, sample entropy via libemg).
DEFAULT_LIBEMG_FEATURES: tuple[str, ...] = (
    "MAV",       # mean absolute value         ≈ src.iemg / N
    "RMS",       # root-mean-square            = src.emg_rms
    "WL",        # waveform length
    "ZC",        # zero crossings
    "SSC",       # slope sign changes
    "WAMP",      # Willison amplitude          = src.emg_wamp
    "MFL",       # max fractal length          = src.emg_mfl
    "MSR",       # mean square root            = src.emg_msr
    "LD",        # log detector / L-Score      = src.emg_lscore
    "MNF",       # mean frequency              = src.emg_mnf
    "MDF",       # median frequency            = src.emg_mdf
)


@dataclass
class EmgFilterCfg:
    fs: int = FS_EMG_DEFAULT
    band_low_hz: float = 20.0
    band_high_hz: float = 450.0
    notch_hz: float = 50.0
    notch_q: float = 30.0
    order: int = 4


def filter_emg(signal: np.ndarray, cfg: EmgFilterCfg | None = None) -> np.ndarray:
    """Zero-phase 20–450 Hz BP + 50 Hz notch (offline). NaN-safe."""
    cfg = cfg or EmgFilterCfg()
    x = np.asarray(signal, dtype=float).copy()
    # forward-fill NaNs (filtfilt propagates NaN otherwise)
    if np.isnan(x).any():
        idx = np.arange(len(x))
        good = ~np.isnan(x)
        if good.any():
            x = np.interp(idx, idx[good], x[good])
        else:
            return np.zeros_like(x)
    sos_bp = butter(
        cfg.order, [cfg.band_low_hz, cfg.band_high_hz],
        btype="band", fs=cfg.fs, output="sos",
    )
    y = sosfiltfilt(sos_bp, x)
    b_n, a_n = iirnotch(cfg.notch_hz, Q=cfg.notch_q, fs=cfg.fs)
    sos_n = tf2sos(b_n, a_n)
    y = sosfiltfilt(sos_n, y)
    return y


def extract_window_features(
    emg_filtered: np.ndarray,
    fs: int = FS_EMG_DEFAULT,
    window_ms: float = 500.0,
    hop_ms: float = 100.0,
    feature_list: Sequence[str] = DEFAULT_LIBEMG_FEATURES,
    t_unix: np.ndarray | None = None,
) -> pd.DataFrame:
    """Extract libemg window features over one-channel filtered EMG.

    Returns a DataFrame with one row per window and columns:
        t_unix (if provided), emg_<feat_lowercase>, ...
    """
    from libemg.feature_extractor import FeatureExtractor

    win_samp = int(round(window_ms * fs / 1000.0))
    hop_samp = max(1, int(round(hop_ms * fs / 1000.0)))

    # libemg expects windows in shape (N_windows, N_channels, window_length).
    windows, starts = sliding_windows(emg_filtered, win_samp, hop_samp)
    if windows.size == 0:
        cols = [f"emg_{f.lower()}" for f in feature_list]
        return pd.DataFrame(columns=(["t_unix"] if t_unix is not None else []) + cols)
    windows = windows[:, np.newaxis, :]  # add channel dim → (N, 1, T)

    fe = FeatureExtractor()
    feats = fe.extract_features(
        list(feature_list), windows, feature_dic={"sampling_frequency": fs}
    )
    # libemg returns dict[str, ndarray of shape (N_windows, N_channels=1)]
    out = {f"emg_{k.lower()}": v.squeeze(-1) for k, v in feats.items()}
    if t_unix is not None:
        centers = starts + win_samp // 2
        centers = np.clip(centers, 0, len(t_unix) - 1)
        out = {"t_unix": np.asarray(t_unix)[centers], **out}
    return pd.DataFrame(out)


def extract_emg_pipeline(
    raw_signal: np.ndarray,
    t_unix: np.ndarray | None = None,
    fs: int = FS_EMG_DEFAULT,
    window_ms: float = 500.0,
    hop_ms: float = 100.0,
    feature_list: Sequence[str] = DEFAULT_LIBEMG_FEATURES,
) -> pd.DataFrame:
    """Filter -> window -> libemg features. One-call wrapper.

    Drop-in replacement for `src.features.emg_features.extract_emg_features`,
    minus the EmgBaselineNormalizer (handled elsewhere — apply per-recording
    median normalization downstream if desired).
    """
    filtered = filter_emg(raw_signal, EmgFilterCfg(fs=fs))
    return extract_window_features(
        filtered,
        fs=fs,
        window_ms=window_ms,
        hop_ms=hop_ms,
        feature_list=feature_list,
        t_unix=t_unix,
    )
