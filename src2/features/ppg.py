"""PPG features via NeuroKit2.

Replaces hand-rolled HR / pulse-amplitude code in `src/features/ppg_features.py`
with NeuroKit2's `nk.ppg_process` + `nk.hrv_time`. NK2 handles bandpass, peak
detection (Elgendi 2013), interpolation gap-fill, and HRV computation in one
function — far more battle-tested than the project's bespoke peak picker.

References:
- Makowski et al. 2021 — NeuroKit2 (Behav Res Methods)
- Elgendi 2013 — Optimal signal processing for PPG
- Allen 2007 — PPG and its application in clinical physiological measurement
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FS_PPG_DEFAULT = 100


def extract_ppg_pipeline(
    raw_signal: np.ndarray,
    t_unix: np.ndarray | None = None,
    fs: int = FS_PPG_DEFAULT,
    window_s: float = 10.0,
    hop_s: float = 0.1,
) -> pd.DataFrame:
    """Per-window HR + HRV features from green PPG.

    Pipeline:
      1. nk.ppg_process       → cleans signal, detects peaks
      2. nk.ppg_intervalrelated over rolling windows for HRV time-domain stats
      3. instantaneous HR (bpm) at each window center

    Output columns: t_unix, ppg_hr_bpm, ppg_hrv_rmssd, ppg_hrv_sdnn, ppg_pulse_amp.
    Returns NaN-filled rows where peak detection fails (clean failure mode).
    """
    import neurokit2 as nk

    sig = np.asarray(raw_signal, dtype=float)
    sig = np.nan_to_num(sig, nan=np.nanmean(sig) if np.isfinite(sig).any() else 0.0)
    try:
        signals, info = nk.ppg_process(sig, sampling_rate=fs)
    except Exception:
        # NK2 sometimes errors on extremely flat signals — return all-NaN.
        n_windows = max(0, int((len(sig) - int(window_s * fs)) / int(hop_s * fs)) + 1)
        return pd.DataFrame(
            {
                "ppg_hr_bpm": [np.nan] * n_windows,
                "ppg_hrv_rmssd": [np.nan] * n_windows,
                "ppg_hrv_sdnn": [np.nan] * n_windows,
                "ppg_pulse_amp": [np.nan] * n_windows,
            }
        )

    win = int(window_s * fs)
    hop = max(1, int(hop_s * fs))
    rows: list[dict] = []
    for start in range(0, len(sig) - win + 1, hop):
        end = start + win
        seg = signals.iloc[start:end]
        peaks_in_seg = seg["PPG_Peaks"].to_numpy() if "PPG_Peaks" in seg.columns else None
        # HR: from PPG_Rate column (NK2 fills rolling estimate per sample).
        hr = float(np.nanmean(seg["PPG_Rate"])) if "PPG_Rate" in seg.columns else np.nan
        # Pulse amplitude: median of segment's local-max minus local-min around each peak.
        amp = float(np.nan)
        if peaks_in_seg is not None and peaks_in_seg.any():
            cleaned = seg["PPG_Clean"].to_numpy()
            peak_idx = np.where(peaks_in_seg)[0]
            if len(peak_idx) >= 2:
                amps = []
                for pi in peak_idx:
                    lo = max(0, pi - int(0.3 * fs))
                    hi = min(len(cleaned), pi + int(0.3 * fs))
                    amps.append(cleaned[pi] - cleaned[lo:hi].min())
                amp = float(np.median(amps))
        # HRV time-domain over this window's peaks (skip if too few peaks).
        rmssd = sdnn = float(np.nan)
        if peaks_in_seg is not None and peaks_in_seg.sum() >= 3:
            try:
                hrv = nk.hrv_time(np.where(peaks_in_seg)[0], sampling_rate=fs)
                rmssd = float(hrv.get("HRV_RMSSD", [np.nan])[0])
                sdnn = float(hrv.get("HRV_SDNN", [np.nan])[0])
            except Exception:
                pass

        row = {
            "ppg_hr_bpm": hr,
            "ppg_hrv_rmssd": rmssd,
            "ppg_hrv_sdnn": sdnn,
            "ppg_pulse_amp": amp,
        }
        if t_unix is not None:
            row["t_unix"] = float(t_unix[min(start + win // 2, len(t_unix) - 1)])
        rows.append(row)
    return pd.DataFrame(rows)
