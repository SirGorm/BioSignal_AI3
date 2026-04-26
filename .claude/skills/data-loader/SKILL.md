---
name: data-loader
description: Use when reading raw session files (biosignals, joint-angle data, participants.xlsx). All timestamps are Unix epoch — synchronization is by absolute time, no offset estimation needed. Provides robust loaders that validate schema, verify Unix timestamps, and fail loudly on mismatches.
---

# Data Loader (Unix time anchored)

Files in this project use **Unix epoch timestamps** as the absolute clock. Biosignal-to-joint-angle synchronization is solved by simply matching Unix times — no signal correlation, no manual offset, no warm-up sync pulse.

## Expected file layout

```
data/raw/<subject_id>/<session_id>/
├── biosignals.<csv|parquet>          # Continuous, all 6 modalities, Unix-time
├── joint_angles.csv                  # Only during active sets, Unix-time
└── participants.xlsx                 # Per-set metadata (or project-level)
```

## Why Unix time matters

When biosignals start at Unix `1714123456.789` and joint-angles start at `1714123456.812`, you know they were recorded simultaneously to within 23 ms. **Convert both to session-relative time by subtracting the same reference (typically biosignal start), and they're aligned.** No interpolation onto a session-internal clock; the wall clock IS the session clock.

If a file has timestamps starting at `0.0` (session-relative), it's not Unix time — flag it and refuse to proceed. Manual offset estimation is brittle and a frequent source of silent labeling errors.

## participants.xlsx loader

```python
import pandas as pd
from pathlib import Path

REQUIRED_COLS = {
    'subject_id', 'session_id', 'exercise', 'set_number',
    'set_start_unix', 'set_end_unix', 'rpe',
}
RENAME_MAP = {
    'set_start': 'set_start_unix',
    'set_start_time': 'set_start_unix',
    'set_end': 'set_end_unix',
    'set_end_time': 'set_end_unix',
    'rate_of_perceived_exertion': 'rpe',
}

def load_participants(path: Path, subject_id=None, session_id=None) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"participants.xlsx missing columns: {missing}. "
            f"Found: {sorted(df.columns)}. Update RENAME_MAP if names differ."
        )

    df['set_start_unix'] = pd.to_numeric(df['set_start_unix'])
    df['set_end_unix'] = pd.to_numeric(df['set_end_unix'])
    if not (df['set_start_unix'] > 1e9).all():
        raise ValueError(
            f"set_start_unix must be Unix epoch (>1e9). "
            f"Got {df['set_start_unix'].iloc[0]} — confirm timestamp format."
        )

    df['rpe'] = pd.to_numeric(df['rpe'], errors='coerce')
    if df['rpe'].isna().any():
        raise ValueError(f"Non-numeric RPE: {df[df['rpe'].isna()]}")
    if (df['rpe'] < 1).any() or (df['rpe'] > 10).any():
        raise ValueError("RPE out of [1,10] range — confirm scale.")

    if subject_id:
        df = df[df['subject_id'].astype(str) == str(subject_id)]
    if session_id:
        df = df[df['session_id'].astype(str) == str(session_id)]
    if df.empty:
        raise ValueError(f"No rows for {subject_id}/{session_id}")
    return df.reset_index(drop=True)
```

## Biosignal loader (Unix-time-anchored)

```python
import numpy as np

EXPECTED_CHANNELS = {'ecg', 'emg', 'eda', 'temp', 'ax', 'ay', 'az'}
PPG_GREEN_ALIAS = ['ppg_green', 'ppg_g', 'ppg_550', 'ppg_2']
# Adjust PPG_GREEN_ALIAS after running /inspect — it tells you the actual column name.

def load_biosignals(path: Path) -> pd.DataFrame:
    if path.suffix == '.csv':
        df = pd.read_csv(path)
    elif path.suffix in ('.parquet', '.pq'):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    df.columns = [c.lower().strip() for c in df.columns]

    for cand in ['t_unix', 'timestamp', 'unix_time', 't', 'time']:
        if cand in df.columns:
            df = df.rename(columns={cand: 't_unix'})
            break
    else:
        raise ValueError(f"No timestamp column: {df.columns.tolist()}")

    if not (df['t_unix'] > 1e9).all():
        raise ValueError(
            f"Timestamps in {path} are not Unix epoch. "
            f"Run /inspect to verify."
        )

    rename = {
        'ecg_raw': 'ecg', 'emg_raw': 'emg', 'eda_raw': 'eda',
        'temperature': 'temp', 'skin_temp': 'temp',
        'accel_x': 'ax', 'accel_y': 'ay', 'accel_z': 'az',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    ppg_green = next((c for c in PPG_GREEN_ALIAS if c in df.columns), None)
    if ppg_green is None:
        raise ValueError(
            f"No PPG-green column. Tried {PPG_GREEN_ALIAS}. "
            f"Check inspections/<...>/findings.md and update PPG_GREEN_ALIAS."
        )
    if ppg_green != 'ppg_green':
        df = df.rename(columns={ppg_green: 'ppg_green'})

    missing = EXPECTED_CHANNELS - set(df.columns)
    if missing:
        raise ValueError(f"Missing channels: {missing}")

    # Combine 3-axis acc to magnitude (project requirement)
    df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    return df


def biosignal_sample_rate(df: pd.DataFrame, channel: str = None) -> float:
    if channel is None:
        ts = df['t_unix'].values
    else:
        valid = df[df[channel].notna()]
        ts = valid['t_unix'].values
    return 1.0 / np.median(np.diff(ts))
```

## Joint-angle loader

```python
def load_joint_angles(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Joint file: {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    for cand in ['t_unix', 'timestamp', 'unix_time', 't', 'time']:
        if cand in df.columns:
            df = df.rename(columns={cand: 't_unix'})
            break
    else:
        raise ValueError(f"No timestamp column: {df.columns.tolist()}")

    if not (df['t_unix'] > 1e9).all():
        raise ValueError(f"Joint timestamps in {path} are not Unix epoch.")

    angle_cols = [c for c in df.columns
                  if c.endswith('_angle') or c in
                  ('knee', 'hip', 'elbow', 'shoulder', 'ankle')]
    if not angle_cols:
        raise ValueError(f"No angle columns: {df.columns.tolist()}")
    return df
```

## Sync via Unix time

```python
from scipy.interpolate import interp1d

def align_joint_to_biosignal(bio_df, joint_df, joint_col='knee_angle'):
    """Interpolate joint angle onto biosignal Unix timestamps.
    Outside joint coverage -> NaN. No offset estimation; Unix time is shared clock."""
    valid = joint_df[joint_df[joint_col].notna()]
    if len(valid) < 2:
        return np.full(len(bio_df), np.nan)
    f = interp1d(
        valid['t_unix'].values, valid[joint_col].values,
        bounds_error=False, fill_value=np.nan, assume_sorted=False,
    )
    return f(bio_df['t_unix'].values)
```

That's the whole sync logic. No DTW, no cross-correlation, no manual offset.

## Validation before labeling

```python
def validate_session_files(subject_id, session_id, root=Path('data/raw')):
    sess = root / str(subject_id) / str(session_id)
    if not sess.exists():
        raise FileNotFoundError(f"Session not found: {sess}")

    bio = next(iter(sess.glob('biosignals.*')), None)
    joint = sess / 'joint_angles.csv'
    parts = sess / 'participants.xlsx'
    if not parts.exists():
        parts = root.parent / 'participants.xlsx'

    for p, name in [(bio, 'biosignals'), (joint, 'joint_angles'),
                     (parts, 'participants.xlsx')]:
        if p is None or not p.exists():
            raise FileNotFoundError(f"{name} not found for {subject_id}/{session_id}")

    # Inspection must have run at least once on this dataset
    if not Path('inspections').exists() or not any(Path('inspections').iterdir()):
        raise RuntimeError(
            "No inspections/ directory found. Run /inspect on at least one "
            "session before labeling so sample rates and channel names are confirmed."
        )
    return bio, joint, parts
```

## Output summary after loading

```
Subject: S001 / Session: 1
  Biosignals:    14:32 duration, fs={ecg:500, emg:1000, eda:32, temp:4, acc:100, ppg:64}
  Joint angles:  3 active periods, 4:18 active total
  Participants:  9 sets logged, RPE range [4, 9]
  Unix sync:     bio_start=1714123456.789, joint_start=1714123456.812 (Δ=23ms)
  Validation:    PASSED
```
