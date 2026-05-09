"""Feature-extraction orchestrator.

Reads aligned_features.parquet (the labeled, 100 Hz unified-grid output of
src/labeling/) and produces window_features.parquet — the input to the
LightGBM baselines and to src3/data/feature_dataset.py.

Joins per-modality feature DataFrames on `t_unix` (nearest match within a
small tolerance) so the label columns from the aligned grid attach
cleanly to each window.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src3.features import acc, emg, ppg, temp


JOIN_TOL_S = 0.05  # 50 ms — half a window hop


def extract_for_recording(aligned_parquet: Path, fs_grid: int = 100) -> pd.DataFrame:
    """Run all per-modality extractors on a single recording's aligned parquet."""
    df = pd.read_parquet(aligned_parquet)

    t = df["t_unix"].to_numpy()
    feats: list[pd.DataFrame] = []

    if "emg" in df.columns:
        feats.append(emg.extract_features(
            df["emg"].to_numpy(), t, fs=fs_grid, do_filter=False,
        ))
    if {"ax", "ay", "az"}.issubset(df.columns):
        feats.append(acc.extract_features(
            df["ax"].to_numpy(), df["ay"].to_numpy(), df["az"].to_numpy(),
            t, fs=fs_grid,
        ))
    if "ppg_green" in df.columns:
        feats.append(ppg.extract_features(df["ppg_green"].to_numpy(), t, fs=fs_grid))
    if "temp" in df.columns:
        feats.append(temp.extract_features(df["temp"].to_numpy(), t, fs=fs_grid))

    if not feats:
        raise ValueError(f"No modalities found in {aligned_parquet}")

    out = feats[0].sort_values("t_unix")
    for f in feats[1:]:
        f = f.sort_values("t_unix")
        out = pd.merge_asof(
            out, f, on="t_unix", direction="nearest",
            tolerance=JOIN_TOL_S,
        )

    # Attach labels + metadata via nearest-time join from the labeled grid.
    label_cols = [c for c in (
        "subject_id", "recording_id", "set_number", "in_active_set",
        "exercise", "phase_label", "rep_count_in_set", "rep_density_hz",
        "rpe_for_this_set", "has_rep_intervals",
    ) if c in df.columns]
    label_df = df[["t_unix", *label_cols]].sort_values("t_unix")
    out = pd.merge_asof(
        out, label_df, on="t_unix", direction="nearest",
        tolerance=JOIN_TOL_S,
    )
    return out


def extract_all(labeled_root: Path, out_path: Path) -> None:
    """Run extract_for_recording on every recording_NNN/aligned_features.parquet
    under labeled_root and write a single concatenated parquet to out_path."""
    parquets = sorted(Path(labeled_root).glob("recording_*/aligned_features.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No aligned_features.parquet under {labeled_root}")

    chunks = [extract_for_recording(p) for p in parquets]
    out = pd.concat(chunks, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
