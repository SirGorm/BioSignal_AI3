"""
Robust biosignal loaders for strength-RT recordings.

Two CSV timestamp formats exist in the dataset:
  - Old (rec_001..rec_005): 'Time (s)' column with relative seconds;
    Unix start embedded in header: 'Recording Start Unix Time: <float>'
  - New (rec_006..rec_014): 'timestamp' column containing Unix epoch seconds directly.

All loaders return a DataFrame with a 'timestamp' column in Unix epoch seconds
(float64) and validate t_unix > 1e9. Temperature returns NaN when the file is
empty or absent (header-only).

References
----------
- Unix epoch timestamp validation > 1e9: standard practice — any POSIX time
  after 2001-09-09 exceeds 1e9.
- NaN-tolerant temperature handling required because many newer recordings
  have an empty temperature.csv (header only) per inspection findings.
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_csv_auto(filepath: Path) -> pd.DataFrame:
    """Read a per-modality CSV regardless of old/new timestamp format.

    Returns a DataFrame with columns ['timestamp', <signal>].
    'timestamp' is always Unix epoch float64.
    """
    # Peek at header to decide format
    with open(filepath, encoding="utf-8") as fh:
        header = fh.readline().strip()

    cols = [c.strip() for c in header.split(",")]

    if cols[0] == "timestamp":
        # New format: timestamp already Unix epoch
        df = pd.read_csv(filepath, usecols=[0, 1])
        df.columns = ["timestamp", cols[1]]
    elif cols[0].startswith("Time"):
        # Old format: relative seconds + embedded Unix start
        m = re.search(r"Recording Start Unix Time:\s*([0-9.e+]+)", header)
        if m is None:
            raise ValueError(
                f"Old-format CSV {filepath} has no 'Recording Start Unix Time' in header."
            )
        unix_start = float(m.group(1))
        signal_col = cols[1]
        df = pd.read_csv(filepath, usecols=[0, 1], skiprows=0)
        df.columns = ["timestamp", signal_col]
        df["timestamp"] = df["timestamp"] + unix_start
    else:
        raise ValueError(f"Unrecognised CSV header in {filepath}: {header!r}")

    return df


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_biosignal(rec_dir: Path, modality: str, col: str) -> pd.DataFrame:
    """Load a single biosignal CSV and return a DataFrame with Unix timestamps.

    Parameters
    ----------
    rec_dir:  Path to the recording directory (e.g. dataset/recording_012).
    modality: Filename base, e.g. 'ecg' → 'ecg.csv'.
    col:      Expected signal column name, e.g. 'ecg'.

    Returns
    -------
    DataFrame with columns ['timestamp', <col>].

    Raises
    ------
    FileNotFoundError if the CSV is absent.
    ValueError if timestamps are not valid Unix epoch (> 1e9).
    """
    filepath = rec_dir / f"{modality}.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Missing biosignal file: {filepath}")

    df = _read_csv_auto(filepath)

    # Normalise column name
    if df.columns[1] != col:
        df = df.rename(columns={df.columns[1]: col})

    # Validate Unix timestamps
    bad = df["timestamp"] <= 1e9
    if bad.any():
        raise ValueError(
            f"{filepath}: {bad.sum()} rows have timestamp <= 1e9 "
            "(not valid Unix epoch seconds)."
        )

    return df.reset_index(drop=True)


def load_temperature(rec_dir: Path) -> pd.DataFrame:
    """Load temperature.csv; returns NaN-filled DataFrame when file is empty.

    Old-format files have a relative-time first column; header encodes Unix
    start. Empty files (header only, <= 1 data row) return an empty DataFrame
    that the caller must handle by emitting NaN.

    Returns
    -------
    DataFrame with columns ['timestamp', 'temperature'], possibly empty.
    """
    filepath = rec_dir / "temperature.csv"
    if not filepath.exists():
        return pd.DataFrame(columns=["timestamp", "temperature"])

    # Check if file has data beyond the header line
    with open(filepath, encoding="utf-8") as fh:
        header = fh.readline()
        first_data = fh.readline().strip()

    if not first_data:
        return pd.DataFrame(columns=["timestamp", "temperature"])

    # File has content — use auto-reader
    df = _read_csv_auto(filepath)
    if df.columns[1] != "temperature":
        df = df.rename(columns={df.columns[1]: "temperature"})

    if len(df) < 2:
        return pd.DataFrame(columns=["timestamp", "temperature"])

    return df.reset_index(drop=True)


def load_imu(rec_dir: Path) -> pd.DataFrame:
    """Load ax, ay, az and compute acc_mag = sqrt(ax^2 + ay^2 + az^2).

    Returns
    -------
    DataFrame with columns ['timestamp', 'ax', 'ay', 'az', 'acc_mag'].
    """
    ax_df = load_biosignal(rec_dir, "ax", "ax")
    ay_df = load_biosignal(rec_dir, "ay", "ay")
    az_df = load_biosignal(rec_dir, "az", "az")

    # All three should have the same timestamps (same device, same 100 Hz clock)
    df = ax_df.copy()
    df["ay"] = ay_df["ay"].values
    df["az"] = az_df["az"].values
    df["acc_mag"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)

    return df


def load_metadata(rec_dir: Path) -> dict:
    """Load and return metadata.json as a dictionary."""
    path = rec_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {path}")
    with open(path) as fh:
        return json.load(fh)


def load_all_biosignals(rec_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all biosignal modalities for one recording.

    Returns a dict keyed by modality name:
        'ecg', 'emg', 'eda', 'temperature', 'ppg_green',
        'ax', 'ay', 'az', 'acc_mag'
    plus a combined 'imu' DataFrame with all four columns.

    'temperature' always present but may be an empty DataFrame if file is
    header-only (caller must check len(df) == 0 and fill with NaN).
    """
    modalities_map = {
        "ecg": "ecg",
        "emg": "emg",
        "eda": "eda",
        "ppg_green": "ppg_green",
    }
    result: dict[str, pd.DataFrame] = {}

    for mod, col in modalities_map.items():
        result[mod] = load_biosignal(rec_dir, mod, col)

    result["temperature"] = load_temperature(rec_dir)
    result["imu"] = load_imu(rec_dir)

    return result
