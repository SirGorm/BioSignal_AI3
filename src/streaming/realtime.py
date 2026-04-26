"""
Real-time streaming feature extraction pipeline.

Replays a recorded session from raw CSVs sample-by-sample through the causal
streaming feature extractors. Can be used for:
  1. Live inference: replace the CSV loader with a hardware SDK callback.
  2. Replay validation: compare streaming outputs to offline features.

Usage
-----
    python -m src.streaming.realtime --replay dataset/recording_012

All streaming extractors are causal — no filtfilt, no find_peaks over whole
signal. The hook check-no-filtfilt.sh enforces this for all files under
src/streaming/.

References
----------
- Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing
  (3rd ed.). Pearson. [causal filtering with persisted state]
- Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
  IEEE Transactions on Biomedical Engineering, BME-32(3), 230-236.
  [R-peak detection in streaming ECG]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.streaming.ecg_streaming import StreamingECGExtractor, FS_ECG
from src.streaming.emg_streaming import StreamingEMGExtractor, FS_EMG
from src.streaming.eda_streaming import StreamingEDAExtractor, FS_EDA
from src.streaming.acc_streaming import StreamingAccExtractor, FS_ACC
from src.streaming.ppg_streaming import StreamingPPGExtractor, FS_PPG_DEFAULT
from src.streaming.temp_streaming import StreamingTempExtractor, FS_TEMP
from src.data.loaders import (
    load_biosignal,
    load_temperature,
    load_imu,
    load_metadata,
)


CHUNK_SIZE = 100  # samples per modality call (at 100 Hz = 100 ms)
BASELINE_S = 60.0  # seconds of rest for baseline capture


class StreamingFeaturePipeline:
    """Full multi-modal streaming feature pipeline.

    One instance per session. Call step() with synchronised chunks from
    all modalities. Internally routes to per-modality extractors.

    Parameters
    ----------
    fs_ecg  : ECG sample rate (Hz).
    fs_emg  : EMG sample rate (Hz).
    fs_eda  : EDA sample rate (Hz).
    fs_acc  : Accelerometer sample rate (Hz).
    fs_ppg  : PPG sample rate (Hz) — read from metadata.json per recording.
    fs_temp : Temperature sample rate (Hz).
    """

    def __init__(
        self,
        fs_ecg: int = FS_ECG,
        fs_emg: int = FS_EMG,
        fs_eda: int = FS_EDA,
        fs_acc: int = FS_ACC,
        fs_ppg: int = FS_PPG_DEFAULT,
        fs_temp: int = FS_TEMP,
    ) -> None:
        self._ecg = StreamingECGExtractor(fs=fs_ecg)
        self._emg = StreamingEMGExtractor(fs=fs_emg)
        self._eda = StreamingEDAExtractor(fs=fs_eda)
        self._acc = StreamingAccExtractor(fs=fs_acc)
        self._ppg = StreamingPPGExtractor(fs=fs_ppg)
        self._temp = StreamingTempExtractor(fs=fs_temp)
        self._last_features: dict = {}

    def set_baseline_end(self, t_unix: float) -> None:
        """Tell all extractors when baseline period ends (unix timestamp)."""
        self._ecg.set_baseline_end(t_unix)
        self._emg.set_baseline_end(t_unix)
        self._eda.set_baseline_end(t_unix)
        self._temp.set_baseline_end(t_unix)

    def step(
        self,
        ecg_chunk: np.ndarray,
        ecg_t: np.ndarray,
        emg_chunk: np.ndarray,
        emg_t: np.ndarray,
        eda_chunk: np.ndarray,
        eda_t: np.ndarray,
        ax_chunk: np.ndarray,
        ay_chunk: np.ndarray,
        az_chunk: np.ndarray,
        acc_t: np.ndarray,
        ppg_chunk: np.ndarray,
        ppg_t: np.ndarray,
        temp_chunk: np.ndarray | None = None,
        temp_t: np.ndarray | None = None,
    ) -> list[dict]:
        """Process one multi-modal chunk and return feature dicts.

        Returns
        -------
        List of feature dicts keyed by t_unix (one per acc hop = 100 ms).
        """
        ecg_feats = self._ecg.step(ecg_chunk, ecg_t)
        emg_feats = self._emg.step(emg_chunk, emg_t)
        eda_feats = self._eda.step(eda_chunk, eda_t)
        acc_feats = self._acc.step(ax_chunk, ay_chunk, az_chunk, acc_t)
        ppg_feats = self._ppg.step(ppg_chunk, ppg_t)
        temp_feats = []
        if temp_chunk is not None and temp_t is not None:
            temp_feats = self._temp.step(temp_chunk, temp_t)

        # Merge: use acc_feats as the primary hop grid (100 Hz base)
        # Update running last-seen values for slower modalities
        def _latest(flist: list[dict]) -> dict:
            return flist[-1] if flist else {}

        latest_ecg = _latest(ecg_feats)
        latest_emg = _latest(emg_feats)
        latest_eda = _latest(eda_feats)
        latest_ppg = _latest(ppg_feats)
        latest_temp = _latest(temp_feats)

        # Update persistent last-features
        for d in [latest_ecg, latest_emg, latest_eda, latest_ppg, latest_temp]:
            for k, v in d.items():
                if k != "t_unix":
                    self._last_features[k] = v

        # Emit one merged dict per acc window
        results = []
        for af in acc_feats:
            merged = dict(self._last_features)
            merged.update(af)  # acc features are freshest
            results.append(merged)

        return results

    def reset(self) -> None:
        """Reset all state for a new session."""
        self._ecg.reset()
        self._emg.reset()
        self._eda.reset()
        self._acc.reset()
        self._ppg.reset()
        self._temp.reset()
        self._last_features.clear()


# ---------------------------------------------------------------------------
# Replay driver
# ---------------------------------------------------------------------------

def _chunked_iter(arr: np.ndarray, t_arr: np.ndarray, chunk_size: int):
    """Yield (chunk, t_chunk) pairs of fixed size."""
    n = len(arr)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield arr[start:end], t_arr[start:end]


def replay_session(rec_dir: Path, chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    """Replay a recorded session through the streaming pipeline.

    Parameters
    ----------
    rec_dir    : Path to dataset/recording_NNN/
    chunk_size : Samples per chunk at 100 Hz (default 100 = 1 s).

    Returns
    -------
    DataFrame of streaming features, one row per 100 ms hop.
    """
    meta = load_metadata(rec_dir)
    ppg_fs = int(meta["sampling_rates"].get("ppg", FS_PPG_DEFAULT))
    t_start_unix = float(meta.get("recording_start_unix_time", 0.0))

    # Load all modalities at native rates
    ecg_df = load_biosignal(rec_dir, "ecg", "ecg")
    emg_df = load_biosignal(rec_dir, "emg", "emg")
    eda_df = load_biosignal(rec_dir, "eda", "eda")
    imu_df = load_imu(rec_dir)
    ppg_df = load_biosignal(rec_dir, "ppg_green", "ppg_green")
    temp_df = load_temperature(rec_dir)

    pipeline = StreamingFeaturePipeline(
        fs_ecg=FS_ECG,
        fs_emg=FS_EMG,
        fs_eda=FS_EDA,
        fs_acc=FS_ACC,
        fs_ppg=ppg_fs,
        fs_temp=FS_TEMP,
    )

    # Baseline ends 60 s into session
    baseline_end = float(ecg_df["timestamp"].iloc[0]) + BASELINE_S
    pipeline.set_baseline_end(baseline_end)

    all_rows: list[dict] = []

    # Drive by acc (100 Hz) — other modalities chunk proportionally
    acc_n = len(imu_df)
    ecg_ratio = FS_ECG // FS_ACC  # 5
    emg_ratio = FS_EMG // FS_ACC  # 20
    eda_ratio_inv = FS_ACC // FS_EDA  # 2 (EDA is slower)

    has_temp = len(temp_df) >= 2
    if has_temp:
        temp_arr = temp_df["temperature"].values
        temp_t_arr = temp_df["timestamp"].values

    print(f"Replaying {rec_dir.name}: {acc_n} acc samples at {FS_ACC} Hz...")

    acc_chunk_samp = chunk_size  # 100 Hz chunks
    ecg_chunk_samp = acc_chunk_samp * ecg_ratio
    emg_chunk_samp = acc_chunk_samp * emg_ratio
    eda_chunk_samp = max(1, acc_chunk_samp // (FS_ACC // FS_EDA))

    for acc_start in range(0, acc_n, acc_chunk_samp):
        acc_end = min(acc_start + acc_chunk_samp, acc_n)
        acc_slice = imu_df.iloc[acc_start:acc_end]
        t_acc = acc_slice["timestamp"].values

        if len(t_acc) == 0:
            continue

        # Find corresponding chunks in other modalities by time
        t_lo, t_hi = float(t_acc[0]), float(t_acc[-1])

        def _time_slice(df: pd.DataFrame, t_col: str = "timestamp"):
            mask = (df[t_col] >= t_lo) & (df[t_col] <= t_hi + 0.1)
            return df[mask]

        ecg_sl = _time_slice(ecg_df)
        emg_sl = _time_slice(emg_df)
        eda_sl = _time_slice(eda_df)
        ppg_sl = _time_slice(ppg_df)

        if len(ecg_sl) == 0:
            ecg_sl = ecg_df.iloc[
                max(0, acc_start * ecg_ratio) : acc_end * ecg_ratio
            ]
        if len(emg_sl) == 0:
            emg_sl = emg_df.iloc[
                max(0, acc_start * emg_ratio) : acc_end * emg_ratio
            ]
        if len(eda_sl) == 0:
            eda_sl = eda_df.iloc[
                max(0, acc_start // (FS_ACC // FS_EDA)) : acc_end // (FS_ACC // FS_EDA)
            ]

        temp_chunk_arr = None
        temp_t_chunk = None
        if has_temp:
            tmask = (temp_t_arr >= t_lo) & (temp_t_arr <= t_hi + 1.0)
            if tmask.any():
                temp_chunk_arr = temp_arr[tmask]
                temp_t_chunk = temp_t_arr[tmask]

        feats_list = pipeline.step(
            ecg_chunk=ecg_sl["ecg"].values,
            ecg_t=ecg_sl["timestamp"].values,
            emg_chunk=emg_sl["emg"].values,
            emg_t=emg_sl["timestamp"].values,
            eda_chunk=eda_sl["eda"].values,
            eda_t=eda_sl["timestamp"].values,
            ax_chunk=acc_slice["ax"].values,
            ay_chunk=acc_slice["ay"].values,
            az_chunk=acc_slice["az"].values,
            acc_t=t_acc,
            ppg_chunk=ppg_sl["ppg_green"].values,
            ppg_t=ppg_sl["timestamp"].values,
            temp_chunk=temp_chunk_arr,
            temp_t=temp_t_chunk,
        )
        all_rows.extend(feats_list)

    df = pd.DataFrame(all_rows)
    print(f"  Produced {len(df)} streaming feature rows.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a recording through the streaming feature pipeline."
    )
    parser.add_argument(
        "--replay",
        required=True,
        metavar="RECORDING_DIR",
        help="Path to dataset/recording_NNN/",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save streaming features as parquet.",
    )
    args = parser.parse_args()

    rec_dir = Path(args.replay)
    if not rec_dir.exists():
        print(f"ERROR: {rec_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    df = replay_session(rec_dir)
    print(f"Streaming features shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if args.output:
        out_path = Path(args.output)
        df.to_parquet(out_path, index=False)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
