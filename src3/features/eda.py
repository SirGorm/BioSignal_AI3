"""EDA features — neurokit2 eda_process when available.

EDA is excluded from the model in this project (Greco et al. 2016 — sensor
floor). This extractor is kept for diagnostic plots / audits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from src3.features._common import nanfill, windows


FS_EDA = 50
WINDOW_S = 10.0
HOP_S = 0.1


def _filter_offline(sig: np.ndarray, fs: int = FS_EDA) -> np.ndarray:
    if len(sig) < 13:
        return sig.copy()
    sig = nanfill(sig)
    sos = butter(4, 5.0, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def window_features(sig: np.ndarray, fs: int = FS_EDA) -> dict[str, float]:
    if len(sig) < 16:
        return {"eda_mean": float("nan"), "eda_std": float("nan"),
                "eda_n_scr": float("nan")}
    try:
        import neurokit2 as nk
        signals, info = nk.eda_process(sig, sampling_rate=fs)
        n_scr = float(len(info.get("SCR_Peaks", []) or []))
        return {
            "eda_mean": float(np.mean(sig)),
            "eda_std": float(np.std(sig)),
            "eda_n_scr": n_scr,
        }
    except Exception:
        return {
            "eda_mean": float(np.mean(sig)),
            "eda_std": float(np.std(sig)),
            "eda_n_scr": float("nan"),
        }


def extract_features(
    sig: np.ndarray, t_unix: np.ndarray, fs: int = FS_EDA,
    window_s: float = WINDOW_S, hop_s: float = HOP_S,
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
