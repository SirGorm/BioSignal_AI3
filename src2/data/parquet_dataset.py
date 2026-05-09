"""Torch Dataset over labeled aligned_features.parquet files.

Reads the same parquet schema produced by `src/labeling/run.py`. Channel order
and z-score normalization match `src/data/raw_window_dataset.py` so models
trained on src2 are directly comparable to src/ runs.

Soft-target handling: phase=soft (probability vector over phase classes,
KL-div in loss), reps=soft_window (windowed rep density × window seconds).
Both are read from columns the labeling pipeline writes; if those columns
are missing we fall back to hard targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

RAW_CHANNELS: tuple[str, ...] = ("emg", "ppg_green", "acc_mag", "temp")
FS_GRID = 100  # parquet grid Hz, fixed by src/labeling/align.py
DEFAULT_WINDOW_S = 2.0
BASELINE_S = 90.0
CLIP_SIGMA = 5.0
PHASE_UNKNOWN_MAX_FRAC = 0.5


class AlignedWindowDataset(Dataset):
    """One sample = one window of (C, T) raw biosignals + 4 task targets/masks.

    Args:
        parquet_paths: list of aligned_features.parquet files (one per recording).
        channels:      modality channel names (defaults to RAW_CHANNELS).
        window_s, hop_s: window length / hop in seconds on the 100 Hz grid.
        active_only:   if True, drop windows whose end-of-window sample is
                       outside an active set.
        target_modes:  {'phase': 'soft'|'hard', 'reps': 'soft_window'|'hard'}.
    """

    def __init__(
        self,
        parquet_paths: Sequence[Path | str],
        channels: Sequence[str] = RAW_CHANNELS,
        window_s: float = DEFAULT_WINDOW_S,
        hop_s: float | None = None,
        active_only: bool = False,
        target_modes: dict[str, str] | None = None,
        exercise_encoder: LabelEncoder | None = None,
        phase_encoder: LabelEncoder | None = None,
    ):
        self.paths = [Path(p) for p in parquet_paths]
        self.channels = list(channels)
        self.window_size = int(round(window_s * FS_GRID))
        self.window_s = float(window_s)
        self.hop_size = (
            int(round(hop_s * FS_GRID)) if hop_s is not None else self.window_size // 2
        )
        self.active_only = active_only
        self.target_modes = {"phase": "soft", "reps": "soft_window"}
        if target_modes:
            self.target_modes.update(target_modes)

        self._dfs: list[pd.DataFrame] = []
        self._chan_mean: list[np.ndarray] = []
        self._chan_std: list[np.ndarray] = []
        all_ex: list[str] = []
        all_ph: list[str] = []

        for p in self.paths:
            df = pd.read_parquet(p)
            self._dfs.append(df)
            mu, sd = self._baseline_stats(df)
            self._chan_mean.append(mu)
            self._chan_std.append(sd)
            all_ex.extend(df["exercise"].dropna().astype(str).tolist())
            all_ph.extend(df["phase_label"].dropna().astype(str).tolist())

        # Label encoders — sklearn handles unseen labels via .classes_ membership.
        self.exercise_encoder = exercise_encoder or LabelEncoder().fit(all_ex)
        known_phases = [v for v in all_ph if v != "unknown"]
        self.phase_encoder = phase_encoder or LabelEncoder().fit(known_phases)

        # Build window index list[(file_idx, start_sample)]
        self._idx: list[tuple[int, int]] = []
        self._subjects: list[str] = []
        for fi, df in enumerate(self._dfs):
            n = len(df)
            subj = (
                str(df["subject_id"].iloc[0])
                if "subject_id" in df.columns
                else f"file_{fi}"
            )
            mask = (
                df["in_active_set"].astype(bool).to_numpy()
                if active_only and "in_active_set" in df.columns
                else np.ones(n, dtype=bool)
            )
            for start in range(0, n - self.window_size + 1, self.hop_size):
                end = start + self.window_size - 1
                if not mask[end]:
                    continue
                self._idx.append((fi, start))
                self._subjects.append(subj)

    # ---- public API expected by Lightning callers --------------------------

    @property
    def n_exercise(self) -> int:
        return int(len(self.exercise_encoder.classes_))

    @property
    def n_phase(self) -> int:
        return int(len(self.phase_encoder.classes_))

    @property
    def subject_ids(self) -> np.ndarray:
        return np.asarray(self._subjects)

    def __len__(self) -> int:
        return len(self._idx)

    # ---- internals ---------------------------------------------------------

    def _baseline_stats(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n_base = int(BASELINE_S * FS_GRID)
        base = df.iloc[:n_base][self.channels].to_numpy(dtype=np.float32)
        full = df[self.channels].to_numpy(dtype=np.float32)
        mu = np.zeros(len(self.channels), dtype=np.float32)
        sd = np.ones(len(self.channels), dtype=np.float32)
        for i in range(len(self.channels)):
            b = base[:, i][~np.isnan(base[:, i])]
            f = full[:, i][~np.isnan(full[:, i])]
            ref = b if len(b) >= 10 else f
            if len(ref) > 0:
                mu[i] = float(np.mean(ref))
                sd[i] = max(float(np.std(ref)), 1e-8)
        return mu, sd

    def __getitem__(self, idx: int) -> dict:
        fi, start = self._idx[idx]
        df = self._dfs[fi]
        end = start + self.window_size  # exclusive
        win = df.iloc[start:end][self.channels].to_numpy(dtype=np.float32)
        win = (win - self._chan_mean[fi]) / self._chan_std[fi]
        win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)
        win = np.clip(win, -CLIP_SIGMA, CLIP_SIGMA)
        x = torch.from_numpy(win.T)  # (C, T)

        row = df.iloc[end - 1]

        # exercise (hard CE) — mask out rest windows.
        ex_val = row.get("exercise")
        if pd.notna(ex_val) and str(ex_val) in set(self.exercise_encoder.classes_):
            ex_idx = int(
                self.exercise_encoder.transform([str(ex_val)])[0]
            )
            ex_mask = True
        else:
            ex_idx, ex_mask = 0, False

        # phase target — soft (prob vec) or hard (long index).
        if self.target_modes["phase"] == "soft":
            ph_samples = (
                df["phase_label"].iloc[start:end].astype(str).to_numpy()
            )
            n_unknown = int((ph_samples == "unknown").sum())
            n_known = len(ph_samples) - n_unknown
            ph_target = np.zeros(self.n_phase, dtype=np.float32)
            if n_known > 0:
                for k, name in enumerate(self.phase_encoder.classes_):
                    ph_target[k] = (ph_samples == name).sum() / n_known
            ph_mask = (n_unknown / max(len(ph_samples), 1)) < PHASE_UNKNOWN_MAX_FRAC
            phase_tensor = torch.from_numpy(ph_target)
        else:
            ph_str = str(row.get("phase_label") or "unknown")
            if ph_str == "unknown" or ph_str not in set(self.phase_encoder.classes_):
                phase_tensor = torch.tensor(0, dtype=torch.long)
                ph_mask = False
            else:
                phase_tensor = torch.tensor(
                    int(self.phase_encoder.transform([ph_str])[0]),
                    dtype=torch.long,
                )
                ph_mask = True

        # fatigue (RPE per set) — regression.
        rpe = row.get("rpe_for_this_set")
        rpe_f = float(rpe) if pd.notna(rpe) else 0.0
        rpe_mask = pd.notna(rpe)

        # reps target.
        if (
            self.target_modes["reps"] == "soft_window"
            and "rep_density_hz" in df.columns
        ):
            density = (
                df["rep_density_hz"]
                .iloc[start:end]
                .to_numpy(dtype=np.float32)
            )
            density = np.nan_to_num(density, nan=0.0)
            rep_f = float(density.mean() * self.window_s)
            in_act = bool(row.get("in_active_set", False))
            rc = row.get("rep_count_in_set")
            rep_mask = in_act and pd.notna(rc)
        else:
            rc = row.get("rep_count_in_set")
            rep_f = float(rc) if pd.notna(rc) else 0.0
            rep_mask = pd.notna(rc)

        return {
            "x": x,
            "targets": {
                "exercise": torch.tensor(ex_idx, dtype=torch.long),
                "phase": phase_tensor,
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
