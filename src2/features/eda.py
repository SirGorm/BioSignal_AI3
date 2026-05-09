"""EDA features via NeuroKit2.

Even though EDA is excluded from model input on this dataset (sensor-floor
issue, see CLAUDE.md), keeping a NK2-based extractor here lets us swap in a
better sensor later without re-implementing decomposition.

References:
- Makowski et al. 2021 — NeuroKit2
- Greco et al. 2016 — cvxEDA (NK2's default decomposition algorithm)
- Boucsein 2012 — Electrodermal Activity (textbook)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FS_EDA_DEFAULT = 50


def extract_eda_pipeline(
    raw_signal: np.ndarray,
    t_unix: np.ndarray | None = None,
    fs: int = FS_EDA_DEFAULT,
    window_s: float = 10.0,
    hop_s: float = 1.0,
) -> pd.DataFrame:
    """Per-window tonic / phasic stats + SCR rate via NK2 cvxEDA decomposition."""
    import neurokit2 as nk

    sig = np.asarray(raw_signal, dtype=float)
    sig = np.nan_to_num(sig, nan=np.nanmean(sig) if np.isfinite(sig).any() else 0.0)
    try:
        signals, info = nk.eda_process(sig, sampling_rate=fs, method="cvxeda")
    except Exception:
        return pd.DataFrame()

    win = int(window_s * fs)
    hop = max(1, int(hop_s * fs))
    rows: list[dict] = []
    tonic = signals["EDA_Tonic"].to_numpy()
    phasic = signals["EDA_Phasic"].to_numpy()
    scr_peaks = (
        signals["SCR_Peaks"].to_numpy() if "SCR_Peaks" in signals.columns else None
    )
    for start in range(0, len(sig) - win + 1, hop):
        end = start + win
        row = {
            "eda_tonic_mean": float(np.nanmean(tonic[start:end])),
            "eda_phasic_mean": float(np.nanmean(phasic[start:end])),
            "eda_phasic_std": float(np.nanstd(phasic[start:end])),
            "eda_scr_rate_hz": (
                float(scr_peaks[start:end].sum() / window_s)
                if scr_peaks is not None
                else np.nan
            ),
        }
        if t_unix is not None:
            row["t_unix"] = float(t_unix[min(start + win // 2, len(t_unix) - 1)])
        rows.append(row)
    return pd.DataFrame(rows)
