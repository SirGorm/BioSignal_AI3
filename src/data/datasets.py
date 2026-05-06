"""PyTorch dataset for strength-RT — Phase 1 (per-window features only).

Reads window_features.parquet from data/labeled/<subject>/<session>/.
RPE labels come from the same file (broadcast per set; the
biosignal-feature-extractor copies set RPE onto every window of the set).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.phase_whitelist import whitelist_mask


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
# Modalities excluded from model inputs project-wide. ECG was deemed
# insufficient quality for this dataset (see compare_ecg_filtering output);
# EDA fails the dynamic-range threshold on all 9 recordings (sensor floor;
# Greco et al. 2016). Any column starting with one of these prefixes is
# stripped from the auto-selected feature list — defends against legacy
# parquets that still contain the columns.
EXCLUDED_FEATURE_PREFIXES = ('ecg_', 'eda_')

# Soft-target column names produced by src/features/window_features.py
# (rolling aggregates over a backward 2 s window). Listed here so that the
# dataset both treats them as labels (not features) and knows where to read
# soft targets from when target_modes != 'hard'.
SOFT_REPS_COL = 'reps_in_window_2s'
SOFT_PHASE_COL_PREFIX = 'phase_frac_'   # e.g. phase_frac_concentric

VALID_REPS_MODES = ('hard', 'soft_window', 'soft_overlap')
# soft_overlap: Wang et al. 2026 (J Appl Sci Eng 31:26031038, Eq. 2) —
#   pre-computed Σ overlap_fraction(rep_k, window) per row by
#   scripts/add_soft_overlap_reps.py.
SOFT_OVERLAP_COL = 'soft_overlap_reps'
HAS_REP_INTERVALS_COL = 'has_rep_intervals'
VALID_PHASE_MODES = ('hard', 'soft')
DEFAULT_TARGET_MODES = {'reps': 'hard', 'phase': 'soft'}


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
        active_only: bool = False,
        exercise_encoder: Optional[LabelEncoder] = None,
        phase_encoder: Optional[LabelEncoder] = None,
        phase_whitelist: Optional[Set[Tuple[str, int]]] = None,
        target_modes: Optional[Dict[str, str]] = None,
        verbose: bool = True,
        stride: Optional[int] = None,
        window_s: float = 2.0,
        norm_mode: str = "baseline",
    ):
        if norm_mode not in ("baseline", "robust", "percentile"):
            raise ValueError(f"norm_mode={norm_mode!r} not in "
                             "{'baseline', 'robust', 'percentile'}")
        self.norm_mode = norm_mode
        # If stride not given, default to window_s/2 × 100 Hz (50 % overlap on
        # the 100 Hz feature grid). Matches raw NN windowing for fairness.
        if stride is None:
            stride = max(1, int(round(window_s / 2 * 100)))
        self.window_s = float(window_s)
        # Resolve target modes (defaults preserve hard-label legacy pipeline).
        modes = dict(DEFAULT_TARGET_MODES)
        if target_modes:
            modes.update(target_modes)
        if modes['reps'] not in VALID_REPS_MODES:
            raise ValueError(f"target_modes['reps']={modes['reps']!r} not in {VALID_REPS_MODES}")
        if modes['phase'] not in VALID_PHASE_MODES:
            raise ValueError(f"target_modes['phase']={modes['phase']!r} not in {VALID_PHASE_MODES}")
        self.target_modes = modes

        dfs = [pd.read_parquet(p) for p in window_parquets]
        df = pd.concat(dfs, ignore_index=True)

        # Decimate to a fixed stride on the 100 Hz feature grid so that the
        # feature NN trains on the same window-center cadence as the raw NN
        # (which uses HOP_SIZE=100 = 1 s in raw_window_dataset). Default 100
        # = 1 s hop → matches 2 s window with 50 % overlap. Per recording so
        # we keep contiguous coverage rather than dropping whole recordings.
        if stride is not None and stride > 1:
            n_before = len(df)
            df = df.groupby('recording_id', sort=False, group_keys=False)\
                   .apply(lambda g: g.iloc[::stride])\
                   .reset_index(drop=True)
            if verbose:
                print(f"[dataset] stride={stride} (per recording): "
                      f"{n_before} -> {len(df)}")

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

        excluded = METADATA_COLS | LABEL_COLS | {SOFT_REPS_COL,
                                                    SOFT_OVERLAP_COL,
                                                    HAS_REP_INTERVALS_COL}
        # phase_frac_* columns are soft-target supervision, never features
        soft_phase_cols = [c for c in df.columns
                           if c.startswith(SOFT_PHASE_COL_PREFIX)]
        excluded = excluded | set(soft_phase_cols)
        if feature_cols is None:
            feature_cols = [c for c in df.columns
                             if c not in excluded
                             and not c.startswith(EXCLUDED_FEATURE_PREFIXES)
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
            dropped = [c for c in feature_cols
                       if c.startswith(EXCLUDED_FEATURE_PREFIXES)]
            if dropped:
                if verbose:
                    print(f"[dataset] dropping excluded modalities from "
                          f"feature_cols: {dropped}")
                feature_cols = [c for c in feature_cols if c not in dropped]
        self.feature_cols = feature_cols

        if verbose:
            print(f"[dataset] {len(feature_cols)} features, {len(df)} windows, "
                  f"{df['subject_id'].nunique()} subjects")

        self.exercise_encoder = (exercise_encoder
                                  or LabelEncoder().fit(df['exercise']))
        self.phase_encoder = (phase_encoder
                               or LabelEncoder().fit(df['phase_label']))

        x_arr = df[feature_cols].to_numpy(dtype=np.float32)

        # Per-recording feature normalization. Three modes mirror the raw
        # path (src/data/raw_window_dataset.py):
        #   "baseline"   : center=baseline mean, scale=baseline std.
        #   "robust"     : center=full-recording median,
        #                   scale=MAD × 1.4826.
        #   "percentile" : center=baseline mean,
        #                   scale=99-percentile of |feature - center|
        #                   over the full recording. Equalizes "max
        #                   activation" scale across subjects.
        # Sparse-feature fallback (< 100 valid baseline rows) drops to
        # full-recording stats for that feature.
        BASELINE_S = 90.0
        rec_ids = df['recording_id'].to_numpy()
        t_sess = df['t_session_s'].to_numpy()
        n_feat = x_arr.shape[1]
        for rec in np.unique(rec_ids):
            rec_mask = rec_ids == rec
            base_mask = rec_mask & (t_sess < BASELINE_S)
            base_rows = x_arr[base_mask]
            full_rows = x_arr[rec_mask]
            center = np.zeros(n_feat, dtype=np.float32)
            scale = np.ones(n_feat, dtype=np.float32)
            for j in range(n_feat):
                col_base = base_rows[:, j]
                col_full = full_rows[:, j]
                base_valid = col_base[~np.isnan(col_base)]
                full_valid = col_full[~np.isnan(col_full)]
                if len(full_valid) == 0:
                    continue
                if self.norm_mode == "baseline":
                    use = base_valid if len(base_valid) >= 100 else full_valid
                    center[j] = float(np.mean(use))
                    s = float(np.std(use))
                elif self.norm_mode == "robust":
                    med = float(np.median(full_valid))
                    mad = float(np.median(np.abs(full_valid - med)))
                    center[j] = med
                    s = mad * 1.4826
                else:  # percentile
                    base_ref = base_valid if len(base_valid) >= 100 else full_valid
                    center[j] = float(np.mean(base_ref))
                    s = float(np.percentile(np.abs(full_valid - center[j]), 99.0))
                scale[j] = s if s > 1e-8 else 1.0
            x_arr[rec_mask] = (x_arr[rec_mask] - center) / scale
        # Clip outliers; loose enough for bursty features under heavy sets.
        x_arr = np.clip(x_arr, -8.0, 8.0)

        n_nan = int(np.isnan(x_arr).sum())
        if n_nan > 0 and verbose:
            print(f"[dataset] {n_nan} NaN values in features "
                  f"({n_nan/x_arr.size:.2%}); replacing with 0.")
        x_arr = np.nan_to_num(x_arr, nan=0.0, posinf=0.0, neginf=0.0)
        if verbose:
            print(f"[dataset] per-recording {self.norm_mode} normalization "
                  f"({len(np.unique(rec_ids))} recordings, baseline={BASELINE_S}s)")
        self.X = torch.from_numpy(x_arr)

        ex_idx = self.exercise_encoder.transform(df['exercise'])
        ph_idx_hard = self.phase_encoder.transform(df['phase_label'])
        rpe = df[rpe_col].to_numpy(dtype=np.float32)

        self.t_exercise = torch.from_numpy(ex_idx)
        self.m_exercise = self.t_exercise >= 0

        # ---- phase target (hard or soft) -----------------------------------
        if self.target_modes['phase'] == 'soft':
            class_cols = [SOFT_PHASE_COL_PREFIX + name
                          for name in self.phase_encoder.classes_]
            missing = [c for c in class_cols if c not in df.columns]
            if missing:
                raise KeyError(
                    f"target_modes['phase']='soft' requires columns {class_cols} "
                    f"in window_features.parquet. Missing: {missing}. "
                    "Re-run feature extraction after the soft-target change."
                )
            soft_arr = df[class_cols].to_numpy(dtype=np.float32)
            soft_arr = np.nan_to_num(soft_arr, nan=0.0)
            self.t_phase = torch.from_numpy(soft_arr)
            unknown_col = SOFT_PHASE_COL_PREFIX + 'unknown'
            if unknown_col in df.columns:
                unknown_frac = df[unknown_col].to_numpy(dtype=np.float32)
                unknown_frac = np.nan_to_num(unknown_frac, nan=1.0)
                self.m_phase = torch.from_numpy(unknown_frac < 0.5)
            else:
                self.m_phase = torch.from_numpy(soft_arr.sum(axis=1) > 0)
        else:
            self.t_phase = torch.from_numpy(ph_idx_hard)
            self.m_phase = self.t_phase >= 0

        if phase_whitelist is not None:
            if 'recording_id' not in df.columns or 'set_number' not in df.columns:
                raise ValueError(
                    "phase_whitelist requires 'recording_id' and 'set_number' "
                    "columns in window_features.parquet."
                )
            wl_mask = whitelist_mask(
                df['recording_id'].to_numpy(),
                df['set_number'].to_numpy(),
                phase_whitelist,
            )
            n_kept = int(wl_mask.sum())
            n_total = int(self.m_phase.sum().item())
            self.m_phase = self.m_phase & torch.from_numpy(wl_mask)
            if verbose:
                print(f"[dataset] phase whitelist: kept {n_kept}/{n_total} "
                      f"phase-labelled windows "
                      f"({len(phase_whitelist)} (recording, set) pairs)")

        # ---- reps target: hard | soft_window | soft_overlap -----------------
        if self.target_modes['reps'] == 'soft_overlap':
            # Pick window-specific column; 2 s alias = legacy SOFT_OVERLAP_COL.
            col = (f'soft_overlap_reps_{self.window_s:g}s'
                   .replace('.', '_'))
            if col not in df.columns:
                col = SOFT_OVERLAP_COL  # legacy 2 s
            if col not in df.columns:
                raise KeyError(
                    f"target_modes['reps']='soft_overlap' requires column "
                    f"{col!r} in window_features.parquet. "
                    "Run scripts/add_soft_overlap_reps.py --window-s {self.window_s}."
                )
            reps = df[col].to_numpy(dtype=np.float32)
            in_act = (df['in_active_set'].astype(bool).to_numpy()
                      if 'in_active_set' in df.columns
                      else ~np.isnan(reps))
            has_iv = (df[HAS_REP_INTERVALS_COL].astype(bool).to_numpy()
                      if HAS_REP_INTERVALS_COL in df.columns
                      else np.ones(len(df), dtype=bool))
            self.m_reps = torch.from_numpy(in_act & has_iv & ~np.isnan(reps))
        elif self.target_modes['reps'] == 'soft_window':
            if SOFT_REPS_COL not in df.columns:
                raise KeyError(
                    f"target_modes['reps']='soft_window' requires column "
                    f"{SOFT_REPS_COL!r} in window_features.parquet. Re-run "
                    "feature extraction after the soft-target change."
                )
            reps = df[SOFT_REPS_COL].to_numpy(dtype=np.float32)
            in_act = (df['in_active_set'].astype(bool).to_numpy()
                      if 'in_active_set' in df.columns
                      else ~np.isnan(reps))
            self.m_reps = torch.from_numpy(in_act & ~np.isnan(reps))
        else:
            reps = df['rep_count_in_set'].to_numpy(dtype=np.float32)
            self.m_reps = torch.from_numpy(~np.isnan(reps))
        self.t_reps = torch.from_numpy(np.nan_to_num(reps, nan=0.0))

        # Fatigue is supervised only on active-set windows. Rest windows still
        # get a forward-pass prediction (model(x) is unconditional), but they
        # do not contribute to the loss → no gradient flow from rest periods.
        # Symmetric with the reps mask above.
        in_act_fat = (df['in_active_set'].astype(bool).to_numpy()
                      if 'in_active_set' in df.columns
                      else np.ones(len(df), dtype=bool))
        self.m_fatigue = torch.from_numpy((~np.isnan(rpe)) & in_act_fat)
        self.t_fatigue = torch.from_numpy(np.nan_to_num(rpe, nan=0.0))

        self.subject_ids: List[str] = df['subject_id'].astype(str).tolist()
        # Filled in by materialize_to_device(); flag enables GPU fast path in
        # src/training/loop.py:_make_loader.
        self.gpu_resident: bool = False

    def materialize_to_device(self, device) -> None:
        """Move all tensors to `device` and expose them under _gpu_* names so
        the training loop can index directly without DataLoader overhead.

        Idempotent: re-calling on an already-resident dataset is a no-op."""
        if self.gpu_resident:
            return
        device = torch.device(device)
        if device.type != 'cuda':
            # CPU fast path is also fine: DataLoader bypass still helps.
            pass
        self._gpu_x = self.X.to(device, non_blocking=True)
        self._gpu_targets = {
            'exercise': self.t_exercise.to(device, non_blocking=True),
            'phase':    self.t_phase.to(device, non_blocking=True),
            'fatigue':  self.t_fatigue.to(device, non_blocking=True),
            'reps':     self.t_reps.to(device, non_blocking=True),
        }
        self._gpu_masks = {
            'exercise': self.m_exercise.to(device, non_blocking=True),
            'phase':    self.m_phase.to(device, non_blocking=True),
            'fatigue':  self.m_fatigue.to(device, non_blocking=True),
            'reps':     self.m_reps.to(device, non_blocking=True),
        }
        self.gpu_resident = True

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
