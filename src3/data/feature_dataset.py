"""Per-window engineered-feature Dataset (Phase-1 input variant).

Reads window_features.parquet produced by src3/features/pipeline.py (or
the legacy src/features/window_features.py — schema is identical).

Returns:
    x       : (n_features,) float32
    targets : exercise(long), phase(long|soft (K,)), fatigue(float), reps(float)
    masks   : dict of bool — True where target is valid

Implementation note: __init__ materializes every column we need into
numpy arrays so __getitem__ is pure tensor indexing — no pandas .iloc per
item. This is the difference between ~10 µs/item and ~10 ms/item on a
2.1 M-row parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src3.data.encoders import LabelEncoder


METADATA_COLS = {
    "subject_id", "session_id", "recording_id", "set_number", "rep_index",
    "t_unix", "t_session_s", "t_window_start", "t_window_end",
    "in_active_set", "set_phase", "has_rep_intervals",
}
LABEL_COLS = {
    "exercise", "phase_label", "rep_count_in_set",
    "rpe_for_this_set", "rpe", "rep_density_hz",
}
EXCLUDED_FEATURE_PREFIXES = ("ecg_", "eda_")
SOFT_REPS_COL = "reps_in_window_2s"
SOFT_OVERLAP_COL = "soft_overlap_reps"
PHASE_FRAC_PREFIX = "phase_frac_"


class WindowFeatureDataset(Dataset):
    def __init__(
        self,
        parquets: Sequence[Path],
        feature_cols: Sequence[str] | None = None,
        active_only: bool = False,
        target_modes: Mapping[str, str] | None = None,
        exercise_encoder: LabelEncoder | None = None,
        phase_encoder: LabelEncoder | None = None,
        clip_sigma: float = 8.0,
        window_s: float = 2.0,
        stride: int | None = None,
    ):
        """
        window_s : Aggregation window in seconds. For feature-input models the
                   features themselves are pre-extracted at fixed per-modality
                   windows; window_s here only:
                   - decimates rows per recording to stride = window_s/2 × 100,
                     matching v17's WindowFeatureDataset stride policy
                   - selects matching soft_overlap_reps_<N>s column when
                     target_modes['reps'] == 'soft_overlap'
        stride   : Override the auto-derived stride (defaults to window_s/2 × 100).
        """
        self.target_modes = {"reps": "hard", "phase": "soft", **(target_modes or {})}
        self.clip_sigma = float(clip_sigma)
        self.window_s = float(window_s)

        df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
        if active_only and "in_active_set" in df.columns:
            df = df[df["in_active_set"].astype(bool)].reset_index(drop=True)

        # Decimate per recording_id to a fixed stride on the 100 Hz feature
        # grid — matches v17's stride = window_s/2 × 100 rule (see
        # src/data/datasets.py:120). Per recording so we don't drop entire
        # recordings; we keep contiguous coverage.
        if stride is None:
            stride = max(1, int(round(window_s / 2 * 100)))
        if stride > 1 and "recording_id" in df.columns:
            n_before = len(df)
            df = (df.groupby("recording_id", sort=False, group_keys=False)
                    .apply(lambda g: g.iloc[::stride])
                    .reset_index(drop=True))
            print(f"[WindowFeatureDataset] window_s={window_s}, stride={stride}: "
                  f"{n_before:,} -> {len(df):,} rows")
        self.stride = int(stride)

        rpe_col = ("rpe_for_this_set" if "rpe_for_this_set" in df.columns
                   else "rpe")
        for col in ("exercise", "phase_label", "rep_count_in_set", rpe_col):
            if col not in df.columns:
                raise ValueError(f"Required column missing: {col!r}")

        if feature_cols is None:
            feature_cols = [
                c for c in df.columns
                if c not in METADATA_COLS
                and c not in LABEL_COLS
                and not c.startswith(EXCLUDED_FEATURE_PREFIXES)
                and not c.startswith(PHASE_FRAC_PREFIX)
                and c != SOFT_REPS_COL
                and c != SOFT_OVERLAP_COL
                and pd.api.types.is_numeric_dtype(df[c])
            ]
        self.feature_cols = list(feature_cols)
        if not self.feature_cols:
            raise ValueError("No feature columns selected.")

        self.exercise_encoder = exercise_encoder or LabelEncoder().fit(df["exercise"])
        known_phases = [v for v in df["phase_label"].dropna().astype(str)
                        if v != "unknown"]
        self.phase_encoder = phase_encoder or LabelEncoder().fit(known_phases)

        # ---- materialise feature matrix + labels as numpy/torch up front --
        X = df[self.feature_cols].to_numpy(dtype=np.float32)
        med = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - med[None, :]), axis=0) * 1.4826
        center = med.astype(np.float32)
        scale = np.where(mad > 1e-8, mad, 1.0).astype(np.float32)
        X = (X - center[None, :]) / scale[None, :]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(X, -self.clip_sigma, self.clip_sigma, out=X)
        self._x = torch.from_numpy(X)  # (N, F)

        # exercise — long indices, mask = True where label was non-NaN
        ex_str = df["exercise"].fillna("").astype(str).to_numpy()
        ex_idx = np.full(len(df), -1, dtype=np.int64)
        for i, name in enumerate(self.exercise_encoder.classes_):
            ex_idx[ex_str == name] = i
        ex_mask = ex_idx >= 0
        self._ex = torch.from_numpy(np.where(ex_mask, ex_idx, 0))
        self._ex_mask = torch.from_numpy(ex_mask)

        # phase — soft (B, K) or hard (B,)
        if self.target_modes["phase"] == "soft":
            soft = np.zeros((len(df), self.n_phase), dtype=np.float32)
            for k, name in enumerate(self.phase_encoder.classes_):
                col = f"{PHASE_FRAC_PREFIX}{name}"
                if col in df.columns:
                    v = df[col].to_numpy(dtype=np.float32)
                    soft[:, k] = np.nan_to_num(v, nan=0.0)
            ssum = soft.sum(axis=1)
            ph_mask = ssum > 0.5
            soft = np.where(ssum[:, None] > 0, soft / np.maximum(ssum, 1e-12)[:, None], soft)
            self._ph = torch.from_numpy(soft)
            self._ph_mask = torch.from_numpy(ph_mask)
        else:
            ph_str = df["phase_label"].fillna("unknown").astype(str).to_numpy()
            ph_idx = np.full(len(df), -1, dtype=np.int64)
            for i, name in enumerate(self.phase_encoder.classes_):
                ph_idx[ph_str == name] = i
            ph_mask = ph_idx >= 0
            self._ph = torch.from_numpy(np.where(ph_mask, ph_idx, 0))
            self._ph_mask = torch.from_numpy(ph_mask)

        # fatigue (RPE)
        rpe = df[rpe_col].to_numpy(dtype=np.float32)
        rpe_mask = ~np.isnan(rpe)
        self._fat = torch.from_numpy(np.where(rpe_mask, rpe, 0.0))
        self._fat_mask = torch.from_numpy(rpe_mask)

        # reps — three modes:
        #   hard         : rep_count_in_set (cumulative integer)
        #   soft_window  : reps_in_window_2s (rolling window count)
        #   soft_overlap : soft_overlap_reps_{N}s (Wang 2026, fractional overlap)
        reps_mode = self.target_modes["reps"]
        if reps_mode == "soft_overlap":
            wlabel = f"{self.window_s:g}s".replace(".", "_")  # 2.0 -> "2s", 2.5 -> "2_5s"
            col = f"soft_overlap_reps_{wlabel}"
            if col not in df.columns:
                col = SOFT_OVERLAP_COL  # legacy alias = 2 s
            if col not in df.columns:
                raise KeyError(
                    f"target_modes['reps']='soft_overlap' needs column "
                    f"{col!r} in window_features.parquet."
                )
            rc = df[col].to_numpy(dtype=np.float32)
            in_act = df.get("in_active_set",
                             pd.Series([True] * len(df))).fillna(False).astype(bool).to_numpy()
            has_iv = df.get("has_rep_intervals",
                             pd.Series([True] * len(df))).fillna(True).astype(bool).to_numpy()
            rep_mask = in_act & has_iv & ~np.isnan(rc)
        elif reps_mode == "soft_window":
            col = f"reps_in_window_{self.window_s:g}s"
            if col not in df.columns:
                col = SOFT_REPS_COL
            if col not in df.columns:
                raise KeyError(
                    f"target_modes['reps']='soft_window' needs column "
                    f"{col!r} in window_features.parquet."
                )
            rc = df[col].to_numpy(dtype=np.float32)
            rep_mask = ~np.isnan(rc)
        else:
            rc = df["rep_count_in_set"].to_numpy(dtype=np.float32)
            rep_mask = ~np.isnan(rc)
        self._reps = torch.from_numpy(np.where(rep_mask, rc, 0.0))
        self._reps_mask = torch.from_numpy(rep_mask)

        self._subjects = (
            df["subject_id"].astype(str).tolist()
            if "subject_id" in df.columns
            else df.get("recording_id",
                        pd.Series(["subject"] * len(df))).astype(str).tolist()
        )
        self._n = len(df)
        self._device: torch.device | None = None

    # -------- device residency ----------------------------------------------

    def to(self, device) -> "WindowFeatureDataset":
        """Move all tensor state to `device` once. Subsequent __getitem__
        calls return GPU-resident slices — no per-batch host→device copy.

        Memory: 2.1 M × 23 floats + 5 × 2.1 M targets ≈ 250 MB on a 16 GB GPU
        — comfortable. Saves the entire DataLoader transfer cost per epoch.
        """
        device = torch.device(device)
        self._x = self._x.to(device, non_blocking=True)
        self._ex = self._ex.to(device, non_blocking=True)
        self._ph = self._ph.to(device, non_blocking=True)
        self._fat = self._fat.to(device, non_blocking=True)
        self._reps = self._reps.to(device, non_blocking=True)
        self._ex_mask = self._ex_mask.to(device, non_blocking=True)
        self._ph_mask = self._ph_mask.to(device, non_blocking=True)
        self._fat_mask = self._fat_mask.to(device, non_blocking=True)
        self._reps_mask = self._reps_mask.to(device, non_blocking=True)
        self._device = device
        return self

    # -------- introspection --------------------------------------------------

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    @property
    def n_exercise(self) -> int:
        return self.exercise_encoder.n_classes

    @property
    def n_phase(self) -> int:
        return self.phase_encoder.n_classes

    @property
    def subject_ids(self) -> list[str]:
        return self._subjects

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": self._x[idx],
            "targets": {
                "exercise": self._ex[idx],
                "phase":    self._ph[idx],
                "fatigue":  self._fat[idx],
                "reps":     self._reps[idx],
            },
            "masks": {
                "exercise": self._ex_mask[idx],
                "phase":    self._ph_mask[idx],
                "fatigue":  self._fat_mask[idx],
                "reps":     self._reps_mask[idx],
            },
        }
