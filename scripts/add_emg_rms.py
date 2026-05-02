"""Replace the aliased `emg` column in aligned_features.parquet with a
properly computed RMS envelope.

Diagnosis (2026-05-02): The current `emg` column is produced by `np.interp`
from native 2000 Hz to the 100 Hz parquet grid in src/labeling/align.py:75
without an anti-aliasing low-pass first. With raw EMG content in 20-450 Hz,
this folds 50-1000 Hz noise into the 0-50 Hz band on the 100 Hz output —
68.8% of the resulting "EMG" energy is in 20-50 Hz (verified on rec_010).

This script computes the *correct* EMG envelope at native rate and decimates
to 100 Hz with proper anti-aliasing:

    raw 2000 Hz EMG
       ↓ band-pass 20-450 Hz (Butter 4) — remove motion artifact + DC drift
       ↓ 50 Hz notch (Q=30) — remove powerline interference
       ↓ |·| (full-wave rectification)
       ↓ moving-window RMS over 100 ms (200 samples at 2000 Hz)
       ↓ low-pass anti-alias filter at 35 Hz (Butter 4)
       ↓ decimate to 100 Hz grid (np.interp on the same t_unix grid)
    EMG RMS envelope @ 100 Hz   ← physically meaningful, monotonic w/ activation

References
----------
- De Luca, C. J. (1997). The use of surface electromyography in biomechanics.
  J Appl Biomech 13(2), 135-163. — RMS envelope is standard EMG input.
- Cifrek, M., Medved, V., Tonkovic, S., & Ostojic, S. (2009). Surface EMG-
  based muscle fatigue evaluation in biomechanics. Clin Biomech 24(4),
  327-340. — RMS slope over set is the gold-standard fatigue indicator.
- Sterk et al. 2018 (Sensors): engineered EMG features (RMS, MNF) outperformed
  raw NN inputs on small-dataset RPE prediction.

Backs up the old aligned_features.parquet to *_pre_emgrms.parquet on first run.

Usage:
    python scripts/add_emg_rms.py
    python scripts/add_emg_rms.py --rms-window-ms 100 --lp-cutoff 35
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, tf2sos

ROOT = Path(__file__).resolve().parent.parent
LABELED_ROOT = ROOT / "data" / "labeled_clean"
DATASET_CLEAN = ROOT / "dataset_clean"

# EMG native sample rate per CLAUDE.md
FS_EMG = 2000.0


def _butter_bp_sos(low: float, high: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    high = min(high, 0.95 * nyq)
    return tf2sos(*butter(order, [low / nyq, high / nyq], btype="band"))


def _butter_lp_sos(cutoff: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    cutoff = min(cutoff, 0.95 * nyq)
    return tf2sos(*butter(order, cutoff / nyq, btype="low"))


def compute_rms_envelope(emg: np.ndarray, fs: float, rms_window_ms: float,
                          lp_cutoff_hz: float) -> np.ndarray:
    """Native-rate RMS envelope. Returns same length as input.

    Steps: bandpass 20-450 Hz → 50 Hz notch → rectify → RMS over rms_window_ms
    → low-pass at lp_cutoff_hz (anti-alias for downstream 100 Hz grid).
    NaN-safe: NaNs in input become NaN in output at the same indices.
    """
    if len(emg) < 100:
        return emg.copy()

    finite = np.isfinite(emg)
    out = np.full_like(emg, np.nan, dtype=np.float32)
    x = emg[finite].astype(np.float64)

    # 1) Band-pass 20-450 Hz
    bp_sos = _butter_bp_sos(20.0, 450.0, fs)
    x = sosfiltfilt(bp_sos, x)

    # 2) 50 Hz notch (line interference)
    nb, na = iirnotch(50.0, 30.0, fs=fs)
    x = filtfilt(nb, na, x)

    # 3) Rectify
    x = np.abs(x)

    # 4) Moving-window RMS over rms_window_ms
    win = max(1, int(round(rms_window_ms * 1e-3 * fs)))
    sq = x ** 2
    kernel = np.ones(win, dtype=np.float64) / win
    rms = np.sqrt(np.convolve(sq, kernel, mode="same"))

    # 5) Anti-alias LP at lp_cutoff_hz (must be < Nyquist of target grid =
    #    100 Hz / 2 = 50 Hz; 35 Hz default leaves a comfortable margin).
    lp_sos = _butter_lp_sos(lp_cutoff_hz, fs)
    rms = sosfiltfilt(lp_sos, rms)

    out[finite] = rms.astype(np.float32)
    return out


def process_recording(rec_id: str, rms_window_ms: float, lp_cutoff_hz: float
                       ) -> dict:
    """Replace `emg` in data/labeled_clean/<rec_id>/aligned_features.parquet
    with the proper RMS envelope, sampled at the parquet's existing t_unix grid.
    """
    parquet_path = LABELED_ROOT / rec_id / "aligned_features.parquet"
    src_csv = DATASET_CLEAN / rec_id / "emg.csv"
    if not parquet_path.exists() or not src_csv.exists():
        return {"rec": rec_id, "status": "missing_input"}

    # Backup parquet on first run.
    backup = parquet_path.with_name("aligned_features_pre_emgrms.parquet")
    if not backup.exists():
        shutil.copy2(parquet_path, backup)

    # Load parquet — keep all columns, we only overwrite `emg`.
    df = pd.read_parquet(parquet_path)
    if "emg" not in df.columns or "t_unix" not in df.columns:
        return {"rec": rec_id, "status": "no_emg_or_tunix_col"}
    grid_t = df["t_unix"].to_numpy(dtype=np.float64)

    # Load native 2000 Hz EMG
    emg_df = pd.read_csv(src_csv)
    if "timestamp" not in emg_df.columns or "emg" not in emg_df.columns:
        return {"rec": rec_id, "status": "bad_emg_csv"}
    src_t = emg_df["timestamp"].to_numpy(dtype=np.float64)
    src_v = emg_df["emg"].to_numpy(dtype=np.float64)

    # Compute envelope at native 2000 Hz
    env_native = compute_rms_envelope(src_v, FS_EMG, rms_window_ms, lp_cutoff_hz)

    # Resample envelope to parquet grid via linear interp. Anti-aliasing was
    # already applied in compute_rms_envelope (LP at lp_cutoff_hz < 50 Hz),
    # so this np.interp is now spectrally clean.
    new_emg = np.interp(grid_t, src_t, env_native, left=np.nan, right=np.nan)
    new_emg = new_emg.astype(np.float32)

    df["emg"] = new_emg
    df.to_parquet(parquet_path, index=False)

    # Stats for the report
    finite = np.isfinite(new_emg)
    return {
        "rec": rec_id, "status": "ok",
        "n_rows": int(len(df)),
        "n_finite": int(finite.sum()),
        "emg_min": float(np.nanmin(new_emg)),
        "emg_max": float(np.nanmax(new_emg)),
        "emg_mean": float(np.nanmean(new_emg)),
        "emg_std": float(np.nanstd(new_emg)),
    }


def propagate_to_window_features(rec_id: str) -> dict:
    """Also overwrite the `emg` column in the per-recording window_features.parquet
    if it exists (used by feature NN models). Both files share the same t_unix
    grid so a direct copy works.
    """
    aligned_p = LABELED_ROOT / rec_id / "aligned_features.parquet"
    window_p = LABELED_ROOT / rec_id / "window_features.parquet"
    if not (aligned_p.exists() and window_p.exists()):
        return {"rec": rec_id, "status": "no_window_features"}

    a = pd.read_parquet(aligned_p, columns=["t_unix", "emg"])
    w = pd.read_parquet(window_p)
    if "emg" not in w.columns:
        return {"rec": rec_id, "status": "no_emg_in_window"}
    w = w.merge(a.rename(columns={"emg": "emg_new"}), on="t_unix", how="left")
    w["emg"] = w["emg_new"].astype("float32")
    w = w.drop(columns=["emg_new"])
    w.to_parquet(window_p, index=False)
    return {"rec": rec_id, "status": "ok",
            "n_rows": int(len(w)),
            "emg_std": float(np.nanstd(w["emg"]))}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rms-window-ms", type=float, default=100.0,
                    help="RMS window length in ms (default 100)")
    ap.add_argument("--lp-cutoff", type=float, default=35.0,
                    help="Anti-alias LP cutoff in Hz before 100 Hz decimation "
                         "(default 35; must be < 50 Hz Nyquist)")
    args = ap.parse_args()

    rec_dirs = sorted(d for d in LABELED_ROOT.iterdir()
                      if d.is_dir() and d.name.startswith("recording_"))
    print(f"Processing {len(rec_dirs)} recordings  "
          f"(RMS window = {args.rms_window_ms} ms, LP = {args.lp_cutoff} Hz)\n")

    for rd in rec_dirs:
        rep = process_recording(rd.name, args.rms_window_ms, args.lp_cutoff)
        if rep["status"] == "ok":
            print(f"  {rd.name}: rows={rep['n_rows']}  "
                  f"std={rep['emg_std']:.6f}  "
                  f"range=[{rep['emg_min']:.2e}, {rep['emg_max']:.2e}]")
            wrep = propagate_to_window_features(rd.name)
            if wrep["status"] == "ok":
                print(f"     → window_features.parquet updated (rows={wrep['n_rows']})")
        else:
            print(f"  {rd.name}: SKIP — {rep['status']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
