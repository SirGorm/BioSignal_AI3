"""PyTorch dataset for strength-RT — Phase 1 (per-window features only).

Reads window_features.parquet from data/labeled/<subject>/<session>/.
RPE labels come from the same file (broadcast per set; the
biosignal-feature-extractor copies set RPE onto every window of the set).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Columns that are metadata or labels — never treated as model inputs
METADATA_COLS = {
    'subject_id', 'session_id', 'set_number', 'rep_index',
    't_unix', 't_session_s', 't_window_start', 't_window_end',
    'in_active_set', 'set_phase',
}
LABEL_COLS = {
    'exercise', 'phase_label', 'rep_count_in_set',
    'rpe_for_this_set', 'rpe',
}


def _is_nan(v):
    try:
        return np.isnan(v)
    except (TypeError, ValueError):
        return False


class LabelEncoder:
    """Deterministic str -> int mapping. Persisted with the dataset."""

    def __init__(self, classes: Optional[List[str]] = None):
        self.classes_: List[str] = list(classes) if classes else []
        self._idx = {c: i for i, c in enumerate(self.classes_)}

    def fit(self, values):
        unique = sorted({str(v) for v in values
                          if v is not None and not _is_nan(v)})
        self.classes_ = unique
        self._idx = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, values):
        out = np.full(len(values), -1, dtype=np.int64)
        for i, v in enumerate(values):
            if v is None or _is_nan(v):
                out[i] = -1
            else:
                out[i] = self._idx.get(str(v), -1)
        return out

    @property
    def n_classes(self) -> int:
        return len(self.classes_)


class WindowFeatureDataset(Dataset):
    """Per-window engineered features. Phase 1 dataset.

    Returns at index i:
      x       : (n_features,) float32 tensor
      targets : {'exercise': long, 'phase': long, 'fatigue': float, 'reps': float}
      masks   : {same keys, bool}  — True where target is valid
    """

    def __init__(
        self,
        window_parquets: List[Path],
        feature_cols: Optional[List[str]] = None,
        active_only: bool = True,
        exercise_encoder: Optional[LabelEncoder] = None,
        phase_encoder: Optional[LabelEncoder] = None,
        verbose: bool = True,
    ):
        dfs = [pd.read_parquet(p) for p in window_parquets]
        df = pd.concat(dfs, ignore_index=True)

        if active_only and 'in_active_set' in df.columns:
            n_before = len(df)
            df = df[df['in_active_set'].astype(bool)].reset_index(drop=True)
            if verbose:
                print(f"[dataset] Filtered to active windows: "
                      f"{n_before} -> {len(df)}")

        rpe_col = ('rpe_for_this_set' if 'rpe_for_this_set' in df.columns
                   else 'rpe')
        for col in ('exercise', 'phase_label', 'rep_count_in_set', rpe_col):
            if col not in df.columns:
                raise ValueError(
                    f"Required label column missing from window_features.parquet: "
                    f"{col}. Columns present: {sorted(df.columns)}"
                )

        excluded = METADATA_COLS | LABEL_COLS
        if feature_cols is None:
            feature_cols = [c for c in df.columns
                             if c not in excluded
                             and pd.api.types.is_numeric_dtype(df[c])]
            if not feature_cols:
                raise ValueError(
                    "No numeric feature columns found. Verify the "
                    "biosignal-feature-extractor produced features."
                )
        else:
            missing = set(feature_cols) - set(df.columns)
            if missing:
                raise ValueError(f"Requested feature_cols missing: {missing}")
        self.feature_cols = feature_cols

        if verbose:
            print(f"[dataset] {len(feature_cols)} features, {len(df)} windows, "
                  f"{df['subject_id'].nunique()} subjects")

        self.exercise_encoder = (exercise_encoder
                                  or LabelEncoder().fit(df['exercise']))
        self.phase_encoder = (phase_encoder
                               or LabelEncoder().fit(df['phase_label']))

        x_arr = df[feature_cols].to_numpy(dtype=np.float32)
        n_nan = int(np.isnan(x_arr).sum())
        if n_nan > 0 and verbose:
            print(f"[dataset] {n_nan} NaN values in features "
                  f"({n_nan/x_arr.size:.2%}); replacing with 0.")
        x_arr = np.nan_to_num(x_arr, nan=0.0, posinf=0.0, neginf=0.0)
        self.X = torch.from_numpy(x_arr)

        ex_idx = self.exercise_encoder.transform(df['exercise'])
        ph_idx = self.phase_encoder.transform(df['phase_label'])
        rpe = df[rpe_col].to_numpy(dtype=np.float32)
        reps = df['rep_count_in_set'].to_numpy(dtype=np.float32)

        self.t_exercise = torch.from_numpy(ex_idx)
        self.t_phase = torch.from_numpy(ph_idx)
        self.m_exercise = self.t_exercise >= 0
        self.m_phase = self.t_phase >= 0

        self.m_fatigue = torch.from_numpy(~np.isnan(rpe))
        self.m_reps = torch.from_numpy(~np.isnan(reps))
        self.t_fatigue = torch.from_numpy(np.nan_to_num(rpe, nan=0.0))
        self.t_reps = torch.from_numpy(np.nan_to_num(reps, nan=0.0))

        self.subject_ids: List[str] = df['subject_id'].astype(str).tolist()

    def __len__(self):
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_exercise(self) -> int:
        return self.exercise_encoder.n_classes

    @property
    def n_phase(self) -> int:
        return self.phase_encoder.n_classes

    def __getitem__(self, idx):
        return {
            'x': self.X[idx],
            'targets': {
                'exercise': self.t_exercise[idx],
                'phase':    self.t_phase[idx],
                'fatigue':  self.t_fatigue[idx],
                'reps':     self.t_reps[idx],
            },
            'masks': {
                'exercise': self.m_exercise[idx],
                'phase':    self.m_phase[idx],
                'fatigue':  self.m_fatigue[idx],
                'reps':     self.m_reps[idx],
            },
        }
