"""Per-window engineered features dataset (Phase-1 / variant=features).

Reads `window_features.parquet` produced by `src/features/window_features.py`
or `src2/features/pipeline.py`. Returns one (n_features,) tensor per window
plus the same multi-task targets/masks as `AlignedWindowDataset`.

Compatible drop-in for the LightningDataModule when --variant=features is
selected. ECG/EDA columns are excluded per the project rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

EXCLUDED_FEATURE_PREFIXES = ("ecg_", "eda_")
METADATA_COLS = {
    "subject_id", "session_id", "recording_id", "set_number", "rep_index",
    "t_unix", "t_session_s", "t_window_start", "t_window_end",
    "in_active_set", "set_phase",
}
LABEL_COLS = {
    "exercise", "phase_label", "rep_count_in_set",
    "rpe_for_this_set", "rpe", "rep_density_hz",
    "soft_overlap_reps", "has_rep_intervals",
}
CLIP_SIGMA = 8.0


class WindowFeatureDataset(Dataset):
    """Per-window engineered-feature dataset.

    Each sample:
        x       : (n_features,) float32 — z-scored, clipped, NaN→0.
        targets : exercise (long), phase (long, hard-only here for simplicity),
                  fatigue (float), reps (float).
        masks   : bool per task.
    """

    def __init__(
        self,
        parquet_paths: Sequence[Path | str],
        active_only: bool = False,
        feature_cols: Sequence[str] | None = None,
        exercise_encoder: LabelEncoder | None = None,
        phase_encoder: LabelEncoder | None = None,
    ):
        self.paths = [Path(p) for p in parquet_paths]
        dfs = [pd.read_parquet(p) for p in self.paths]
        self._df = pd.concat(dfs, ignore_index=True)

        # Auto-pick feature columns: numeric, non-metadata, non-label, non-excluded.
        if feature_cols is None:
            cols: list[str] = []
            for c in self._df.columns:
                if c in METADATA_COLS or c in LABEL_COLS:
                    continue
                if any(c.startswith(p) for p in EXCLUDED_FEATURE_PREFIXES):
                    continue
                if not pd.api.types.is_numeric_dtype(self._df[c]):
                    continue
                cols.append(c)
            feature_cols = cols
        self.feature_cols = list(feature_cols)
        if not self.feature_cols:
            raise ValueError("No feature columns found in parquet")

        if active_only and "in_active_set" in self._df.columns:
            self._df = self._df[self._df["in_active_set"].astype(bool)].reset_index(
                drop=True
            )

        # Per-feature z-score normalisation. Robust median/IQR (Hastie 2009)
        # handles the occasional 1e30 spike from acc_rep_band_power overflow.
        X = self._df[self.feature_cols].to_numpy(dtype=np.float32)
        X = np.clip(X, -1e30, 1e30)
        self._mean = np.nanmedian(X, axis=0).astype(np.float32)
        q75, q25 = np.nanpercentile(X, [75, 25], axis=0)
        self._std = np.maximum((q75 - q25) / 1.35, 1e-6).astype(np.float32)

        all_ex = self._df.get("exercise", pd.Series(dtype=str)).dropna().astype(str)
        all_ph = (
            self._df.get("phase_label", pd.Series(dtype=str)).dropna().astype(str)
        )
        self.exercise_encoder = exercise_encoder or LabelEncoder().fit(all_ex.tolist())
        known_ph = [v for v in all_ph.tolist() if v != "unknown"]
        self.phase_encoder = phase_encoder or LabelEncoder().fit(known_ph)

        # Subject ids per row (for CV grouping by FoldDataModule).
        if "subject_id" in self._df.columns:
            self._subjects = self._df["subject_id"].astype(str).to_numpy()
        elif "recording_id" in self._df.columns:
            self._subjects = self._df["recording_id"].astype(str).to_numpy()
        else:
            self._subjects = np.array([f"row_{i}" for i in range(len(self._df))])

        # Cache the normalised feature matrix once — windows are tiny so
        # materialising in RAM is much faster than per-getitem normalisation.
        normed = (X - self._mean) / self._std
        normed = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)
        self._X = np.clip(normed, -CLIP_SIGMA, CLIP_SIGMA).astype(np.float32)

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    @property
    def n_exercise(self) -> int:
        return int(len(self.exercise_encoder.classes_))

    @property
    def n_phase(self) -> int:
        return int(len(self.phase_encoder.classes_))

    @property
    def subject_ids(self) -> np.ndarray:
        return self._subjects

    @property
    def channels(self) -> list[str]:
        return list(self.feature_cols)

    def __len__(self) -> int:
        return int(len(self._df))

    def __getitem__(self, idx: int) -> dict:
        row = self._df.iloc[idx]
        x = torch.from_numpy(self._X[idx])

        # exercise — hard CE, mask out rest windows.
        ex_val = row.get("exercise")
        if pd.notna(ex_val) and str(ex_val) in set(self.exercise_encoder.classes_):
            ex_idx = int(self.exercise_encoder.transform([str(ex_val)])[0])
            ex_mask = True
        else:
            ex_idx, ex_mask = 0, False

        # phase — hard label here (soft_phase_frac_* columns might be present
        # but feature-input training treats them as features, not targets).
        ph_str = (
            str(row.get("phase_label")) if pd.notna(row.get("phase_label")) else "unknown"
        )
        if ph_str == "unknown" or ph_str not in set(self.phase_encoder.classes_):
            ph_idx, ph_mask = 0, False
        else:
            ph_idx = int(self.phase_encoder.transform([ph_str])[0])
            ph_mask = True

        rpe = row.get("rpe_for_this_set")
        rpe_f = float(rpe) if pd.notna(rpe) else 0.0
        rpe_mask = pd.notna(rpe)

        rc = row.get("rep_count_in_set")
        rep_f = float(rc) if pd.notna(rc) else 0.0
        rep_mask = pd.notna(rc)

        return {
            "x": x,
            "targets": {
                "exercise": torch.tensor(ex_idx, dtype=torch.long),
                "phase": torch.tensor(ph_idx, dtype=torch.long),
                "fatigue": torch.tensor(rpe_f, dtype=torch.float32),
                "reps": torch.tensor(rep_f, dtype=torch.float32),
            },
            "masks": {
                "exercise": torch.tensor(bool(ex_mask)),
                "phase": torch.tensor(bool(ph_mask)),
                "fatigue": torch.tensor(bool(rpe_mask)),
                "reps": torch.tensor(bool(rep_mask)),
            },
        }
