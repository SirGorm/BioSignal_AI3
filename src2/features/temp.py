"""Skin-temperature features — kept thin (stdlib + numpy)."""

from __future__ import annotations

import numpy as np
import pandas as pd

FS_TEMP_DEFAULT = 1


def extract_temp_pipeline(
    temp: np.ndarray,
    t_unix: np.ndarray | None = None,
    fs: int = FS_TEMP_DEFAULT,
    window_s: float = 60.0,
    hop_s: float = 1.0,
) -> pd.DataFrame:
    """Window mean / slope (linear regression coefficient) of skin temperature."""
    win = int(round(window_s * fs))
    hop = max(1, int(round(hop_s * fs)))
    sig = np.asarray(temp, dtype=float)
    rows: list[dict] = []
    for start in range(0, len(sig) - win + 1, hop):
        seg = sig[start : start + win]
        valid = seg[~np.isnan(seg)]
        if len(valid) < 5:
            continue
        x = np.arange(len(valid), dtype=float) / fs
        slope = float(np.polyfit(x, valid, 1)[0])
        row = {
            "temp_mean_c": float(np.mean(valid)),
            "temp_slope_c_per_s": slope,
        }
        if t_unix is not None:
            row["t_unix"] = float(
                t_unix[min(start + win // 2, len(t_unix) - 1)]
            )
        rows.append(row)
    return pd.DataFrame(rows)
