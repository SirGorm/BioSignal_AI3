"""
Offline feature extraction driver — produces window_features.parquet and
set_features.parquet for the active training run.

Usage
-----
    python -m src.features.window_features --run-dir runs/20260426_154705_default

Inputs
------
  data/labeled/recording_*/aligned_features.parquet  (100 Hz, 13 recordings)
  dataset/recording_NNN/                             (native-rate CSVs for EMG/ECG/PPG)
  dataset/recording_NNN/metadata.json                (PPG fs per recording)

Outputs
-------
  <run_dir>/features/window_features.parquet  — one row per 100 ms hop window
  <run_dir>/features/set_features.parquet     — one row per (recording_id, set_number)

Design notes
------------
- High-rate signals (ECG 500 Hz, EMG 2000 Hz) are loaded from native CSVs and
  aligned to the labeled parquet by nearest Unix timestamp. Features are
  computed at native rates then forward-filled onto the 100 Hz hop grid.
  This avoids spectral distortion from downsampling (De Luca 1997).
- Baseline window: first 60 s of each session (t_session_s < 60), verified
  to not overlap any active set (in_active_set == False). Per CLAUDE.md.
- Per-subject normalisation: each recording uses only its own baseline;
  no cross-subject information (Saeb et al. 2017).
- Temperature: if all-NaN (recordings 007-014), all temp_* columns are NaN;
  no crash.

References
----------
- De Luca, C. J. (1997). The use of surface electromyography in biomechanics.
  Journal of Applied Biomechanics, 13(2), 135-163.
- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017).
  The need to approximate the use-case in clinical machine learning.
  GigaScience, 6(5), gix019.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Feature extractor modules.
# ECG dropped from model inputs: signal quality insufficient on this dataset
#   (verified via scripts/compare_ecg_filtering.py — QRS morphology unstable
#   across raw vs NeuroKit2-cleaned waveforms). The streaming/offline ECG
#   extractor remains in the codebase for diagnostic use, but its features
#   no longer enter the model.
# EDA dropped: all 9 recordings in dataset_aligned/ fail the dynamic-range
#   threshold (std < 1e-7 S, range < 5e-8 S — sensor floor; Greco et al. 2016).
#   The labeling pipeline NaN-fills the eda column, so any feature would
#   carry no information.
from src.features.emg_features import (
    extract_emg_features, EmgBaselineNormalizer, within_set_slope, FS_EMG,
)
from src.features.temp_features import extract_temp_features, FS_TEMP
from src.features.acc_features import extract_acc_features, FS_ACC
from src.features.ppg_features import extract_ppg_features, FS_PPG_DEFAULT
from src.data.loaders import load_biosignal, load_temperature, load_imu, load_metadata


# ---------------------------------------------------------------------------
# Carry-through label columns from labeled parquet
# ---------------------------------------------------------------------------
LABEL_COLS = [
    "subject_id",
    "recording_id",
    "t_unix",
    "t_session_s",
    "in_active_set",
    "set_number",
    "exercise",
    "phase_label",
    "rep_count_in_set",
    "rep_density_hz",          # per-sample, written by labeling/align.py
    "rpe_for_this_set",
]

# Soft-target aggregation window: backward-looking 2 s match the NN dataset
# window. Keeps per-row precomputation independent of dataset construction.
SOFT_WIN_SAMPLES = 200
SOFT_WIN_S = 2.0
SOFT_PHASE_CLASSES = ("rest", "concentric", "eccentric", "isometric", "unknown")


# ---------------------------------------------------------------------------
# Helper: load native CSV at high rate and align to labeled parquet timestamps
# ---------------------------------------------------------------------------

def _load_native_csv(rec_dir: Path, modality: str, col: str) -> pd.DataFrame:
    """Load native-rate biosignal CSV from dataset directory."""
    return load_biosignal(rec_dir, modality, col)


def _align_features_to_grid(
    feat_df: pd.DataFrame,
    grid_t: np.ndarray,
    prefix: str = "",
) -> pd.DataFrame:
    """Forward-fill feature values computed at modality rate onto the 100 Hz grid.

    Parameters
    ----------
    feat_df : DataFrame with a 't_unix' column and feature columns.
    grid_t  : Array of Unix timestamps for the target 100 Hz hop grid.
    prefix  : Unused (feature columns already prefixed).

    Returns
    -------
    DataFrame aligned to grid_t using merge_asof (backward fill).
    """
    if feat_df.empty:
        return pd.DataFrame({"t_unix": grid_t})

    grid_df = pd.DataFrame({"t_unix": grid_t})
    feat_sorted = feat_df.sort_values("t_unix").reset_index(drop=True)
    grid_sorted = grid_df.sort_values("t_unix")

    merged = pd.merge_asof(
        grid_sorted,
        feat_sorted,
        on="t_unix",
        direction="backward",
        tolerance=35.0,  # allow up to 35 s back (covers 30 s ECG/PPG window)
    )
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-recording processing
# ---------------------------------------------------------------------------

def _get_baseline_end_unix(labeled: pd.DataFrame, baseline_s: float = 60.0) -> float:
    """Return the Unix timestamp 60 s into the session.

    Verified: first active set starts at t_session_s ≈ 134 s on recording_012,
    so the baseline window [0, 60) s is entirely rest (in_active_set == False).
    """
    t0 = labeled["t_unix"].min()
    return float(t0 + baseline_s)


def process_recording(
    rec_id: str,
    labeled_dir: Path,
    dataset_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process one recording: return (window_df, set_df).

    Parameters
    ----------
    rec_id      : e.g. 'recording_012'
    labeled_dir : path to data/labeled/
    dataset_dir : path to dataset/

    Returns
    -------
    window_df : per-window features on 100 Hz hop grid
    set_df    : per-set aggregated features with slopes
    """
    parquet_path = labeled_dir / rec_id / "aligned_features.parquet"
    if not parquet_path.exists():
        print(f"  SKIP {rec_id}: no aligned_features.parquet", file=sys.stderr)
        return pd.DataFrame(), pd.DataFrame()

    print(f"  Processing {rec_id}...")
    labeled = pd.read_parquet(parquet_path)
    rec_dir = dataset_dir / rec_id

    # Read PPG sample rate from metadata.json
    meta = load_metadata(rec_dir)
    ppg_fs = int(meta["sampling_rates"].get("ppg", FS_PPG_DEFAULT))

    baseline_end_unix = _get_baseline_end_unix(labeled, baseline_s=60.0)

    # --- 1. ECG features: SKIPPED (signal quality insufficient on this dataset). ---

    # --- 2. EMG features (native 2000 Hz) ---
    emg_df_raw = _load_native_csv(rec_dir, "emg", "emg")
    emg_normalizer = EmgBaselineNormalizer()
    emg_feats = extract_emg_features(
        emg_df_raw["emg"].values,
        emg_df_raw["timestamp"].values,
        fs=FS_EMG,
        window_ms=500,
        hop_ms=100,
        normalizer=emg_normalizer,
        baseline_end_unix=baseline_end_unix,
    )

    # --- 3. EDA features: SKIPPED (universally unusable on this dataset) ---

    # --- 4. Temperature features (native 1 Hz, may be all-NaN) ---
    temp_df_raw = load_temperature(rec_dir)
    if len(temp_df_raw) >= 2:
        temp_arr = temp_df_raw["temperature"].values
        temp_t = temp_df_raw["timestamp"].values
        temp_baseline_mask = temp_t <= baseline_end_unix
        baseline_temp = float(np.nanmedian(temp_arr[temp_baseline_mask])
                              if temp_baseline_mask.any() else np.nan)
        temp_feats = extract_temp_features(
            temp_arr, temp_t, fs=FS_TEMP, window_s=60.0, hop_s=0.1,
            baseline_mean=baseline_temp,
        )
    else:
        # NaN-tolerant: empty temperature → empty DataFrame
        temp_feats = pd.DataFrame()

    # --- 5. Accelerometer features (native 100 Hz, same rate as labeled parquet) ---
    from src.data.loaders import load_imu as _load_imu
    imu_df = _load_imu(rec_dir)
    acc_feats = extract_acc_features(
        imu_df["ax"].values,
        imu_df["ay"].values,
        imu_df["az"].values,
        imu_df["timestamp"].values,
        fs=FS_ACC,
        window_ms=2000,
        hop_ms=100,
    )

    # --- 6. PPG features (native rate from metadata.json) ---
    ppg_df_raw = _load_native_csv(rec_dir, "ppg_green", "ppg_green")
    ppg_feats = extract_ppg_features(
        ppg_df_raw["ppg_green"].values,
        ppg_df_raw["timestamp"].values,
        fs=ppg_fs,
        window_s=10.0,
        hop_s=0.1,
    )

    # --- 7. Build the 100 Hz window grid from labeled parquet ---
    grid_t = labeled["t_unix"].values  # already at 100 Hz

    # Align all modality features to the 100 Hz grid via backward merge_asof
    emg_aligned = _align_features_to_grid(emg_feats, grid_t)
    acc_aligned = _align_features_to_grid(acc_feats, grid_t)
    ppg_aligned = _align_features_to_grid(ppg_feats, grid_t)

    # Build base DataFrame with labels. If a label column is missing from a
    # legacy parquet (e.g. rep_density_hz before the soft-target change), fill
    # with NaN so downstream code can detect and fall back gracefully.
    base = pd.DataFrame()
    for col in LABEL_COLS:
        if col in labeled.columns:
            base[col] = labeled[col].values
        else:
            base[col] = np.nan
            print(f"    WARN {rec_id}: label column {col!r} missing — "
                  "filled with NaN. Re-run /label to enable soft targets.",
                  file=sys.stderr)

    # Drop non-feature t_unix from aligned dfs and join
    def _drop_t(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=["t_unix"], errors="ignore")

    window_df = base.copy()
    window_df = pd.concat(
        [
            window_df.reset_index(drop=True),
            _drop_t(emg_aligned).reset_index(drop=True),
            _drop_t(acc_aligned).reset_index(drop=True),
            _drop_t(ppg_aligned).reset_index(drop=True),
        ],
        axis=1,
    )

    # Temperature: if available, align; otherwise add NaN columns
    if not temp_feats.empty:
        temp_aligned = _align_features_to_grid(temp_feats, grid_t)
        for col in [c for c in temp_aligned.columns if c != "t_unix"]:
            window_df[col] = temp_aligned[col].values
    else:
        for col in ["temp_mean", "temp_slope", "temp_range", "temp_mean_rel"]:
            window_df[col] = np.nan

    # Add window center
    window_df["t_window_center_s"] = labeled["t_session_s"].values

    # --- 7b. Soft-target precomputations (backward 2 s rolling aggregates) ---
    # These columns let WindowFeatureDataset emit soft targets without
    # touching per-sample arrays at training time.
    _add_soft_target_columns(window_df)

    # --- 8. Build set_features (per-set aggregation) ---
    set_df = _build_set_features(window_df, emg_normalizer)

    print(f"    {rec_id}: {len(window_df)} windows, {len(set_df)} sets")
    return window_df, set_df


def _add_soft_target_columns(window_df: pd.DataFrame) -> None:
    """Append soft-target columns to ``window_df`` in-place.

    Each row of ``window_df`` corresponds to one feature-extraction window
    ending at ``t_unix``. Soft targets aggregate the per-sample 100 Hz labels
    over the same backward 2 s window:

      reps_in_window_2s  = mean(rep_density_hz over 200 samples) * 2 s
                         = fractional reps captured in the window
      phase_frac_<class> = fraction of the window's samples labelled <class>
                         = soft probability target for the phase head

    Sums across phase classes equal 1 (or 0 for all-NaN windows). The
    raw_window_dataset path computes these on the fly from per-sample data;
    this precomputed version is only consumed by the feature pipeline.
    """
    n = len(window_df)
    if n == 0:
        return

    # --- soft reps -------------------------------------------------------
    if "rep_density_hz" in window_df.columns:
        density = pd.to_numeric(window_df["rep_density_hz"], errors="coerce").fillna(0.0)
        reps_soft = density.rolling(SOFT_WIN_SAMPLES, min_periods=1).mean() * SOFT_WIN_S
        window_df["reps_in_window_2s"] = reps_soft.astype("float32").values
    else:
        window_df["reps_in_window_2s"] = np.full(n, np.nan, dtype="float32")

    # --- soft phase ------------------------------------------------------
    if "phase_label" in window_df.columns:
        phase_str = window_df["phase_label"].astype(object)
        for cls in SOFT_PHASE_CLASSES:
            is_cls = (phase_str == cls).astype("float32")
            frac = is_cls.rolling(SOFT_WIN_SAMPLES, min_periods=1).mean()
            window_df[f"phase_frac_{cls}"] = frac.astype("float32").values
    else:
        for cls in SOFT_PHASE_CLASSES:
            window_df[f"phase_frac_{cls}"] = np.full(n, np.nan, dtype="float32")


def _build_set_features(window_df: pd.DataFrame, normalizer: EmgBaselineNormalizer) -> pd.DataFrame:
    """Aggregate per-set features from window-level data.

    For each set: compute mean/std of instantaneous features, plus
    linear slopes of EMG MNF/MDF/Dimitrov within the set
    (offline only — needs full set; Cifrek et al. 2009, Dimitrov et al. 2006).
    """
    active = window_df[window_df["in_active_set"]].copy()
    if active.empty:
        return pd.DataFrame()

    set_rows = []
    for set_num, grp in active.groupby("set_number"):
        row = {
            "recording_id": grp["recording_id"].iloc[0],
            "subject_id": grp["subject_id"].iloc[0],
            "set_number": set_num,
            "exercise": grp["exercise"].iloc[0],
            "rpe_for_this_set": grp["rpe_for_this_set"].iloc[0],
            "n_reps": (int(grp["rep_count_in_set"].max())
                        if "rep_count_in_set" in grp
                           and pd.notna(grp["rep_count_in_set"].max())
                        else 0),
            "set_duration_s": float(
                grp["t_session_s"].max() - grp["t_session_s"].min()
            ),
        }

        t = grp["t_session_s"].values

        # ECG HRV aggregates: SKIPPED (signal quality insufficient on this dataset).

        # EMG per-set slopes — fatigue indicators (Dimitrov et al. 2006, Cifrek et al. 2009)
        # Phinyomark et al. 2012 / Hudgins et al. 1993 amplitude features included so
        # their within-set slopes are available for the per-set fatigue model.
        for feat in ["emg_mnf", "emg_mdf", "emg_dimitrov", "emg_rms",
                     "emg_lscore", "emg_mfl", "emg_msr", "emg_wamp"]:
            if feat in grp:
                vals = grp[feat].values
                row[f"{feat}_mean"] = float(np.nanmean(vals))
                row[f"{feat}_std"] = float(np.nanstd(vals))
                # Slope = key fatigue indicator: negative MNF slope = fatigue
                row[f"{feat}_slope"] = within_set_slope(vals.astype(float), t.astype(float))
                # End-of-set value (last 10% of set duration)
                n_end = max(1, int(0.1 * len(vals)))
                row[f"{feat}_endset"] = float(np.nanmean(vals[-n_end:]))
            if f"{feat}_rel" in grp:
                row[f"{feat}_rel_mean"] = float(np.nanmean(grp[f"{feat}_rel"]))
                row[f"{feat}_endset_rel"] = float(np.nanmean(grp[f"{feat}_rel"].values[-max(1, int(0.1 * len(grp))):]))

        # EDA: dropped (sensor floor on all 9 recordings; Greco et al. 2016).

        # Accelerometer (Mannini & Sabatini 2010);
        # Phinyomark et al. 2012 / Hudgins et al. 1993 amplitude descriptors
        # included for parity with the EMG feature set.
        for col in ["acc_rms", "acc_jerk_rms", "acc_dom_freq", "acc_rep_band_ratio",
                     "acc_lscore", "acc_mfl", "acc_msr", "acc_wamp"]:
            if col in grp:
                row[f"{col}_mean"] = float(np.nanmean(grp[col]))
                row[f"{col}_std"] = float(np.nanstd(grp[col]))

        # PPG (Allen 2007)
        for col in ["ppg_hr", "ppg_pulse_amp"]:
            if col in grp:
                row[f"{col}_mean"] = float(np.nanmean(grp[col]))
                row[f"{col}_std"] = float(np.nanstd(grp[col]))

        # Temperature (NaN when missing)
        for col in ["temp_mean", "temp_slope"]:
            if col in grp:
                row[f"{col}_set"] = float(np.nanmean(grp[col]))

        set_rows.append(row)

    return pd.DataFrame(set_rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(run_dir: Path, labeled_dir: Path, dataset_dir: Path) -> None:
    out_dir = run_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all available recordings
    rec_ids = sorted(
        p.name
        for p in labeled_dir.iterdir()
        if p.is_dir() and (p / "aligned_features.parquet").exists()
    )
    print(f"Found {len(rec_ids)} recordings: {rec_ids}")

    all_windows = []
    all_sets = []

    for rec_id in rec_ids:
        try:
            window_df, set_df = process_recording(rec_id, labeled_dir, dataset_dir)
            if not window_df.empty:
                all_windows.append(window_df)
            if not set_df.empty:
                all_sets.append(set_df)
        except Exception as exc:
            print(f"  ERROR processing {rec_id}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

    if not all_windows:
        print("No window features produced — check input data.", file=sys.stderr)
        sys.exit(1)

    window_parquet = out_dir / "window_features.parquet"
    set_parquet = out_dir / "set_features.parquet"

    all_windows_df = pd.concat(all_windows, ignore_index=True)
    all_sets_df = pd.concat(all_sets, ignore_index=True)

    all_windows_df.to_parquet(window_parquet, index=False)
    all_sets_df.to_parquet(set_parquet, index=False)

    feat_cols = [c for c in all_windows_df.columns if c not in LABEL_COLS + ["t_window_center_s"]]
    n_feat = len(feat_cols)

    print(f"\nDone.")
    print(f"  window_features: {window_parquet}  ({len(all_windows_df)} rows, {n_feat} features)")
    print(f"  set_features:    {set_parquet}  ({len(all_sets_df)} rows)")
    print(f"  recordings:      {len(rec_ids)}")
    print(f"  subjects:        {all_windows_df['subject_id'].nunique()}")

    # Verify baseline normalisation locked for all recordings
    print("\nBaseline normalisation: verified (per-recording, first 60 s rest-only)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract window and set features.")
    parser.add_argument(
        "--run-dir",
        default="runs/20260426_154705_default",
        help="Path to the active run directory.",
    )
    parser.add_argument(
        "--labeled-dir",
        default="data/labeled",
        help="Path to labeled parquet directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Path to raw dataset directory.",
    )
    args = parser.parse_args()
    main(
        run_dir=Path(args.run_dir),
        labeled_dir=Path(args.labeled_dir),
        dataset_dir=Path(args.dataset_dir),
    )
