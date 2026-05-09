"""Skin-temperature features. Trivial — kept thin.

The temp signal is 1 Hz native; the labeled grid stores it forward-filled
at 100 Hz. Per-window: mean, std, slope, trend over the session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src3.features._common import nanfill, windows


FS_TEMP_GRID = 100  # forward-filled to grid by labeling pipeline
WINDOW_S = 60.0
HOP_S = 1.0


def window_features(win: np.ndarray) -> dict[str, float]:
    if len(win) == 0 or np.all(np.isnan(win)):
        return {"temp_mean": float("nan"), "temp_std": float("nan"),
                "temp_slope": float("nan")}
    x = nanfill(win)
    t = np.arange(len(x), dtype=float)
    # slope via least-squares without statsmodels.
    slope = float(np.polyfit(t, x, 1)[0]) if len(x) > 1 else float("nan")
    return {
        "temp_mean": float(np.mean(x)),
        "temp_std": float(np.std(x)),
        "temp_slope": slope,
    }


def extract_features(
    sig: np.ndarray, t_unix: np.ndarray,
    fs: int = FS_TEMP_GRID, window_s: float = WINDOW_S, hop_s: float = HOP_S,
) -> pd.DataFrame:
    win = max(1, int(window_s * fs))
    hop = max(1, int(hop_s * fs))
    rows = []
    for s, e in windows(len(sig), win, hop):
        f = window_features(sig[s:e])
        f["t_unix"] = float(t_unix[s + win // 2])
        rows.append(f)
    return pd.DataFrame(rows)
