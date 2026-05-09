"""Per-window raw multimodal biosignals as a torch Dataset.

Slim port of src/data/raw_window_dataset.py with the custom
_GPUBatchIterator removed — Lightning's Trainer + DataLoader(pin_memory=True,
persistent_workers=True) is the standard path. Same parquet schema, same
window/hop/normalization semantics, same target masking.

Channels (input to model) — fixed order:
    [emg, ppg_green, acc_mag, temp]
ECG and EDA are excluded (see CLAUDE.md "Feature-strategi per modalitet").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src3.data.encoders import LabelEncoder


RAW_CHANNELS = ("emg", "ppg_green", "acc_mag", "temp")
FS_LABELED_GRID = 100
BASELINE_S = 90.0
CLIP_SIGMA = 5.0
PHASE_UNKNOWN_MAX_FRAC = 0.5
DEFAULT_TARGET_MODES = {"reps": "hard", "phase": "soft"}
SOFT_OVERLAP_COL = "soft_overlap_reps"


@dataclass
class WindowSpec:
    window_s: float = 2.0
    hop_s: float = 1.0
    norm_mode: str = "baseline"   # baseline | robust | percentile
    active_only: bool = False


class AlignedWindowDataset(Dataset):
    """Reads aligned_features.parquet files and yields (x, targets, masks).

    Each item:
        x       : (C, T) float32 tensor — z-scored, clipped, NaN→0
        targets : dict — exercise(long), phase(long|soft (K,)), fatigue(float),
                  reps(float)
        masks   : dict of bool — True where target is valid
    """

    def __init__(
        self,
        parquet_paths: Sequence[Path],
        channels: Sequence[str] = RAW_CHANNELS,
        spec: WindowSpec | None = None,
        exercise_encoder: LabelEncoder | None = None,
        phase_encoder: LabelEncoder | None = None,
        target_modes: Mapping[str, str] | None = None,
    ):
        self.spec = spec or WindowSpec()
        self.channels = tuple(channels)
        unknown = [c for c in self.channels if c not in RAW_CHANNELS]
        if unknown:
            raise ValueError(f"Unknown channels {unknown!r}; valid: {RAW_CHANNELS}")
        self.target_modes = {**DEFAULT_TARGET_MODES, **(target_modes or {})}

        self.window_size = int(round(self.spec.window_s * FS_LABELED_GRID))
        self.hop_size = max(1, int(round(self.spec.hop_s * FS_LABELED_GRID)))

        self._dfs: list[pd.DataFrame] = []
        self._chan_center: list[np.ndarray] = []
        self._chan_scale: list[np.ndarray] = []

        all_ex: list[str] = []
        all_ph: list[str] = []
        for path in parquet_paths:
            df = pd.read_parquet(path)
            self._dfs.append(df)
            c, s = _baseline_stats(df, self.channels, self.spec.norm_mode)
            self._chan_center.append(c)
            self._chan_scale.append(s)
            all_ex.extend(df["exercise"].dropna().astype(str).tolist())
            all_ph.extend(df["phase_label"].dropna().astype(str).tolist())

        self.exercise_encoder = exercise_encoder or LabelEncoder().fit(all_ex)
        known_phases = [v for v in all_ph if v != "unknown"]
        self.phase_encoder = phase_encoder or LabelEncoder().fit(known_phases)

        self._index: list[tuple[int, int]] = []
        self._subjects: list[str] = []
        for fi, df in enumerate(self._dfs):
            n = len(df)
            subj = (str(df["subject_id"].iloc[0])
                    if "subject_id" in df.columns else f"file_{fi}")
            if self.spec.active_only and "in_active_set" in df.columns:
                active = df["in_active_set"].astype(bool).to_numpy()
            else:
                active = np.ones(n, dtype=bool)
            for start in range(0, n - self.window_size + 1, self.hop_size):
                end = start + self.window_size - 1
                if not active[end]:
                    continue
                self._index.append((fi, start))
                self._subjects.append(subj)

    # -------- introspection --------------------------------------------------

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_timesteps(self) -> int:
        return self.window_size

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
        return len(self._index)

    # -------- item -----------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        fi, start = self._index[idx]
        df = self._dfs[fi]
        end = start + self.window_size
        center = self._chan_center[fi]
        scale = self._chan_scale[fi]

        win = df.iloc[start:end][list(self.channels)].to_numpy(dtype=np.float32)
        win = (win - center[None, :]) / scale[None, :]
        win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)
        win = np.clip(win, -CLIP_SIGMA, CLIP_SIGMA)
        x = torch.from_numpy(win.T)  # (C, T)

        row = df.iloc[end - 1]

        # --- exercise ---
        ex_val = row["exercise"]
        if pd.notna(ex_val):
            ex_idx = int(self.exercise_encoder.transform([str(ex_val)])[0])
            ex_mask = ex_idx >= 0
        else:
            ex_idx, ex_mask = 0, False

        # --- phase ---
        if self.target_modes["phase"] == "soft":
            phase_samples = (
                df["phase_label"].iloc[start:end].astype(str).to_numpy()
            )
            n_total = len(phase_samples)
            n_unknown = int((phase_samples == "unknown").sum())
            n_known = n_total - n_unknown
            ph_target = np.zeros(self.n_phase, dtype=np.float32)
            if n_known > 0:
                for k, name in enumerate(self.phase_encoder.classes_):
                    ph_target[k] = (phase_samples == name).sum() / n_known
            ph_mask = (n_unknown / max(n_total, 1)) < PHASE_UNKNOWN_MAX_FRAC
            phase_tensor = torch.from_numpy(ph_target)
        else:
            ph_str = str(row["phase_label"]) if pd.notna(row["phase_label"]) else "unknown"
            if ph_str == "unknown":
                ph_idx, ph_mask = 0, False
            else:
                ph_idx = int(self.phase_encoder.transform([ph_str])[0])
                ph_mask = ph_idx >= 0
            phase_tensor = torch.tensor(ph_idx, dtype=torch.long)

        # --- fatigue ---
        rpe = row.get("rpe_for_this_set", float("nan"))
        rpe_f = float(rpe) if pd.notna(rpe) else float("nan")
        fat_valid = not np.isnan(rpe_f)

        # --- reps ---
        if self.target_modes["reps"] == "soft_overlap":
            col = (f"soft_overlap_reps_{self.spec.window_s:g}s".replace(".", "_"))
            if col not in df.columns:
                col = SOFT_OVERLAP_COL
            if col not in df.columns:
                raise KeyError(
                    f"target_modes['reps']='soft_overlap' needs column {col!r}. "
                    "Run scripts/add_soft_overlap_reps.py first."
                )
            rep_f = float(df[col].iloc[end - 1])
            in_act = row.get("in_active_set", False)
            has_iv = row.get("has_rep_intervals", True)
            rep_valid = bool(in_act) and bool(has_iv) if pd.notna(in_act) else False
        elif self.target_modes["reps"] == "soft_window":
            density = df["rep_density_hz"].iloc[start:end].to_numpy(dtype=np.float32)
            density = np.nan_to_num(density, nan=0.0)
            rep_f = float(density.mean() * self.spec.window_s)
            in_act = row.get("in_active_set", False)
            rc_val = row.get("rep_count_in_set")
            rep_valid = (bool(in_act) and pd.notna(rc_val)
                         if pd.notna(in_act) else False)
        else:
            rc_val = row["rep_count_in_set"]
            rep_f = float(rc_val) if pd.notna(rc_val) else float("nan")
            rep_valid = not np.isnan(rep_f)

        return {
            "x": x,
            "targets": {
                "exercise": torch.tensor(ex_idx, dtype=torch.long),
                "phase":    phase_tensor,
                "fatigue":  torch.tensor(rpe_f if fat_valid else 0.0, dtype=torch.float32),
                "reps":     torch.tensor(rep_f if rep_valid else 0.0, dtype=torch.float32),
            },
            "masks": {
                "exercise": torch.tensor(ex_mask, dtype=torch.bool),
                "phase":    torch.tensor(ph_mask, dtype=torch.bool),
                "fatigue":  torch.tensor(fat_valid, dtype=torch.bool),
                "reps":     torch.tensor(rep_valid, dtype=torch.bool),
            },
        }


# --- per-recording normalization stats --------------------------------------


def _baseline_stats(
    df: pd.DataFrame,
    channels: Sequence[str],
    norm_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (center, scale) per channel for z-score normalization."""
    n_baseline = int(BASELINE_S * FS_LABELED_GRID)
    baseline = df.iloc[:n_baseline][list(channels)].to_numpy(dtype=np.float32)
    full = df[list(channels)].to_numpy(dtype=np.float32)

    center = np.zeros(len(channels), dtype=np.float32)
    scale = np.ones(len(channels), dtype=np.float32)

    for i in range(len(channels)):
        b = baseline[:, i][~np.isnan(baseline[:, i])]
        f = full[:, i][~np.isnan(full[:, i])]
        if f.size == 0:
            continue
        if norm_mode == "baseline":
            use = b if b.size >= 10 else f
            center[i] = float(np.mean(use))
            scale[i] = max(float(np.std(use)), 1e-8)
        elif norm_mode == "robust":
            med = float(np.median(f))
            mad = float(np.median(np.abs(f - med)))
            center[i] = med
            scale[i] = max(mad * 1.4826, 1e-8)
        elif norm_mode == "percentile":
            ref = b if b.size >= 10 else f
            center[i] = float(np.mean(ref))
            scale[i] = max(float(np.percentile(np.abs(f - center[i]), 99.0)), 1e-8)
        else:
            raise ValueError(f"unknown norm_mode: {norm_mode!r}")
    return center, scale
