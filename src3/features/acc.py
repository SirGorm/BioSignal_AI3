"""Accelerometer features — antropy + scipy.signal.

Same feature set as src/features/acc_features.py:
    acc_rms, acc_jerk_rms, acc_dom_freq, acc_rep_band_power,
    acc_rep_band_ratio, acc_lscore, acc_mfl, acc_msr, acc_wamp
plus an entropy term from antropy when available.

References
----------
- Bonomi et al. 2009 — RMS for activity recognition
- González-Badillo & Sánchez-Medina 2010 — rep frequency band 0.3-1.5 Hz
- Vallat & Garon 2022 — AntroPy entropy library
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

from src3.features._common import nanfill, windows


FS_ACC = 100
WINDOW_MS = 2000
HOP_MS = 100
REP_BAND_LOW = 0.3
REP_BAND_HIGH = 1.5
WAMP_THRESHOLD = 0.05


def _acc_mag(ax, ay, az) -> np.ndarray:
    return np.sqrt(ax ** 2 + ay ** 2 + az ** 2)


def _filter_offline(mag: np.ndarray, fs: int = FS_ACC) -> np.ndarray:
    if len(mag) < 13:
        return mag.copy()
    mag = nanfill(mag)
    sos = butter(4, [0.5, 20.0], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, mag)


def window_features(win: np.ndarray, fs: int = FS_ACC,
                    wamp_threshold: float = WAMP_THRESHOLD) -> dict[str, float]:
    keys = ("acc_rms", "acc_jerk_rms", "acc_dom_freq",
            "acc_rep_band_power", "acc_rep_band_ratio",
            "acc_lscore", "acc_mfl", "acc_msr", "acc_wamp",
            "acc_perm_entropy")
    if len(win) < 16:
        return {k: float("nan") for k in keys}

    abs_x = np.abs(win)
    rms = float(np.sqrt(np.mean(win ** 2)))
    jerk = np.diff(win - np.mean(win)) * fs
    jerk_rms = float(np.sqrt(np.mean(jerk ** 2))) if len(jerk) else float("nan")

    diffs = np.abs(np.diff(win))
    wamp = float(np.sum(diffs > wamp_threshold) / max(1, len(diffs)))
    mfl = float(np.log10(np.sqrt(float(np.sum(diffs ** 2))) + 1e-30))
    msr = float(np.mean(np.sqrt(abs_x)) ** 2)
    lscore = float(np.exp(np.mean(np.log(np.maximum(abs_x, 1e-30)))))

    nperseg = min(256, len(win))
    f, p = welch(win, fs=fs, nperseg=nperseg)
    dom = float(f[np.argmax(p)])
    band = (f >= REP_BAND_LOW) & (f <= REP_BAND_HIGH)
    total = float(np.sum(p))
    rep = float(np.sum(p[band]))

    try:
        from antropy import perm_entropy
        pe = float(perm_entropy(win, normalize=True))
    except Exception:
        pe = float("nan")

    return {
        "acc_rms": rms,
        "acc_jerk_rms": jerk_rms,
        "acc_dom_freq": dom,
        "acc_rep_band_power": rep,
        "acc_rep_band_ratio": rep / (total + 1e-30),
        "acc_lscore": lscore,
        "acc_mfl": mfl,
        "acc_msr": msr,
        "acc_wamp": wamp,
        "acc_perm_entropy": pe,
    }


def extract_features(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    t_unix: np.ndarray, fs: int = FS_ACC,
    window_ms: float = WINDOW_MS, hop_ms: float = HOP_MS,
) -> pd.DataFrame:
    mag = _filter_offline(_acc_mag(ax, ay, az), fs)
    win = max(1, int(window_ms * fs / 1000))
    hop = max(1, int(hop_ms * fs / 1000))
    rows = []
    for s, e in windows(len(mag), win, hop):
        f = window_features(mag[s:e], fs)
        f["t_unix"] = float(t_unix[s + win // 2])
        rows.append(f)
    return pd.DataFrame(rows)
