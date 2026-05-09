"""Feature-extraction orchestrator: runs all 4 modality extractors and joins
their per-window outputs into one DataFrame on a shared 100 Hz grid.

Used to produce the per-window feature parquet that the LightGBM baseline
trains on. The neural networks instead consume the raw 100 Hz signals
directly via `src2.data.parquet_dataset` — feature extraction is for the
classical baseline path only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src2.features.acc import extract_acc_pipeline
from src2.features.emg import extract_emg_pipeline
from src2.features.ppg import extract_ppg_pipeline
from src2.features.temp import extract_temp_pipeline


def extract_all_modalities(
    rec_dir: Path,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Run every extractor on one recording and merge on t_unix (asof join).

    Args:
        rec_dir:  dataset_aligned/recording_NNN.
        out_path: optional parquet output path.
    """
    from src2.data.loaders import load_biosignal, load_imu, load_temperature

    emg_df = load_biosignal(rec_dir, "emg", "emg")
    ppg_df = load_biosignal(rec_dir, "ppg_green", "ppg_green")
    imu_df = load_imu(rec_dir)
    temp_df = load_temperature(rec_dir)

    feats = []
    feats.append(
        extract_emg_pipeline(
            emg_df["emg"].to_numpy(),
            t_unix=emg_df["timestamp"].to_numpy(),
            fs=2000,
        )
    )
    feats.append(
        extract_ppg_pipeline(
            ppg_df["ppg_green"].to_numpy(),
            t_unix=ppg_df["timestamp"].to_numpy(),
            fs=100,
        )
    )
    feats.append(
        extract_acc_pipeline(
            imu_df["acc_mag"].to_numpy(),
            t_unix=imu_df["timestamp"].to_numpy(),
            fs=100,
        )
    )
    if len(temp_df):
        feats.append(
            extract_temp_pipeline(
                temp_df["temperature"].to_numpy(),
                t_unix=temp_df["timestamp"].to_numpy(),
                fs=1,
            )
        )

    # asof-merge on t_unix using the densest extractor as the index (EMG @ 100 ms).
    feats = [df.sort_values("t_unix") for df in feats if len(df)]
    merged = feats[0]
    for df in feats[1:]:
        merged = pd.merge_asof(
            merged, df, on="t_unix", direction="nearest", tolerance=0.5
        )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out_path, index=False)
    return merged
