"""Accelerometer-magnitude features via antropy + scipy.

Per-window stats:
  - intensity   : RMS, std (motion magnitude)
  - dominant frequency : peak of Welch PSD in 0.3–3 Hz rep band
  - jerk RMS   : RMS of finite-difference derivative
  - sample entropy : antropy.sample_entropy (signal complexity)
  - permutation entropy : antropy.perm_entropy

References:
- Karantonis et al. 2006 — IMU activity classification
- González-Badillo & Sánchez-Medina 2010 — velocity-based training reps band
- Vallat 2022 — antropy: entropy and complexity measures
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

FS_ACC_DEFAULT = 100


def extract_acc_pipeline(
    acc_mag: np.ndarray,
    t_unix: np.ndarray | None = None,
    fs: int = FS_ACC_DEFAULT,
    window_ms: float = 2000.0,
    hop_ms: float = 100.0,
) -> pd.DataFrame:
    """Per-window acc-magnitude features."""
    import antropy as ant

    win = int(round(window_ms * fs / 1000))
    hop = max(1, int(round(hop_ms * fs / 1000)))
    sig = np.asarray(acc_mag, dtype=float)
    rows: list[dict] = []
    for start in range(0, len(sig) - win + 1, hop):
        seg = sig[start : start + win]
        seg_clean = seg[~np.isnan(seg)]
        if len(seg_clean) < 10:
            continue

        rms = float(np.sqrt(np.mean(seg_clean**2)))
        std = float(np.std(seg_clean))
        diff = np.diff(seg_clean)
        jerk_rms = float(np.sqrt(np.mean(diff**2))) if len(diff) > 0 else np.nan

        # Dominant frequency in 0.3–3 Hz rep band (Welch).
        nperseg = min(len(seg_clean), 256)
        try:
            f, p = welch(seg_clean, fs=fs, nperseg=nperseg)
            band = (f >= 0.3) & (f <= 3.0)
            dom_f = float(f[band][p[band].argmax()]) if band.any() else np.nan
        except Exception:
            dom_f = np.nan

        try:
            samp_en = float(ant.sample_entropy(seg_clean))
        except Exception:
            samp_en = np.nan
        try:
            perm_en = float(ant.perm_entropy(seg_clean, normalize=True))
        except Exception:
            perm_en = np.nan

        row = {
            "acc_rms": rms,
            "acc_std": std,
            "acc_jerk_rms": jerk_rms,
            "acc_dominant_freq_hz": dom_f,
            "acc_sample_entropy": samp_en,
            "acc_perm_entropy": perm_en,
        }
        if t_unix is not None:
            row["t_unix"] = float(
                t_unix[min(start + win // 2, len(t_unix) - 1)]
            )
        rows.append(row)
    return pd.DataFrame(rows)
