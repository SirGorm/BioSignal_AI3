"""PyTorch dataset for raw multimodal biosignals — Phase 2 (raw-signal input).

Reads aligned_features.parquet files (one per recording). Each parquet contains
per-sample data at a uniform 100 Hz timeline (pragmatic simplification — all
modalities were resampled to 100 Hz during the labeling pipeline; see SUMMARY.md
"Open questions" for the hybrid-fusion alternative that preserves native rates).

Window construction: 2.0 s = 200 samples, 100 ms hop = 10 samples.
Input to models: (C=6, T=200) float32 tensors.
Channels in order: [ecg, emg, eda, ppg_green, acc_mag, temp]

Z-score normalization per channel per recording using stats from the first 90 s
(baseline rest period, per CLAUDE.md "EMG-baseline" rule). NaN -> 0 after
normalization. Clip at ±5 σ.

Active-only filter: drop windows where in_active_set is False at the end-of-window
sample (matches the features-pipeline filtering in WindowFeatureDataset).

Lazy __getitem__: window indices are pre-computed but signal data is read from the
cached DataFrame, not stored as a large 3D array. This keeps RAM bounded.

References:
- Bai et al. 2018 — TCN: temporal convolutional network for sequence modeling
- Hochreiter & Schmidhuber 1997 — LSTM
- Yang et al. 2015 — 1D-CNN for multichannel sensor data
- Saeb et al. 2017 — subject-wise CV
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.datasets import LabelEncoder
from src.data.phase_whitelist import WhitelistKey

# Channels used as input — in this fixed order.
# EDA omitted: all 9 recordings fail the dynamic-range threshold
#   (std < 1e-7 S, range < 5e-8 S — sensor floor; Greco et al. 2016).
# ECG omitted: signal quality on this dataset is insufficient for stable
#   feature extraction (verified via scripts/compare_ecg_filtering.py
#   across raw vs NeuroKit2-cleaned waveforms; QRS morphology unstable).
RAW_CHANNELS = ['emg', 'ppg_green', 'acc_mag', 'temp']
N_CHANNELS = 4
# Default window: 2.0 s @ 100 Hz with 50% overlap (1.0 s hop). Can be
# overridden per-instance via `RawMultimodalWindowDataset(window_s=...)`.
DEFAULT_WINDOW_S = 2.0
FS_LABELED_GRID = 100  # parquet grid is at 100 Hz (set in src/labeling/align.py)
BASELINE_SAMPLES = 9000  # 90 s at 100 Hz — independent of window
CLIP_SIGMA = 5.0

# Per-recording normalization modes — see _compute_baseline_stats. Selected
# at construction time via `norm_mode=`.
#   "baseline" (default): mean / std on first 90 s rest. Backward-compat.
#   "robust"  : median / (MAD × 1.4826) on full recording. Robust to occasional
#               heavy-set outliers; addresses exercise overfitting where the
#               subject's max activation dominates standard z-score.
#   "percentile": (x − baseline_mean) / 99th-percentile-of-|x − baseline_mean|.
#                 Equalizes "max activation" scale across subjects so subjects
#                 with different lift weights map to the same dynamic range.
VALID_NORM_MODES = ("baseline", "robust", "percentile")

# Backward-compat module constants for callers that don't override.
WINDOW_SIZE = int(DEFAULT_WINDOW_S * FS_LABELED_GRID)
WINDOW_DUR_S = DEFAULT_WINDOW_S
HOP_SIZE = WINDOW_SIZE // 2
HOP_DUR_S = HOP_SIZE / FS_LABELED_GRID

VALID_REPS_MODES = ('hard', 'soft_window', 'soft_overlap')
VALID_PHASE_MODES = ('hard', 'soft')
DEFAULT_TARGET_MODES = {'reps': 'hard', 'phase': 'soft'}
# soft_overlap: Wang et al. 2026 (J Appl Sci Eng 31:26031038, Eq. 2) —
#   y_cnt(window) = Σ_k overlap_fraction(rep_k, window). Pre-computed and
#   stored as `soft_overlap_reps` column by scripts/add_soft_overlap_reps.py.
SOFT_OVERLAP_COL = 'soft_overlap_reps'
PHASE_UNKNOWN_MAX_FRAC = 0.5  # mask soft-phase target if more than this is unknown


class RawMultimodalWindowDataset(Dataset):
    """Per-window raw multimodal biosignals.

    Returns at index i:
      x       : (C=4, T=200) float32 tensor — z-scored, clipped, NaN->0
      targets : {'exercise': long, 'phase': long, 'fatigue': float, 'reps': float}
      masks   : {same keys, bool} — True where target is valid
    """

    def __init__(
        self,
        parquet_paths: List[Path],
        active_only: bool = False,
        exercise_encoder: Optional[LabelEncoder] = None,
        phase_encoder: Optional[LabelEncoder] = None,
        phase_whitelist: Optional[Set[WhitelistKey]] = None,
        target_modes: Optional[Dict[str, str]] = None,
        verbose: bool = True,
        window_s: float = DEFAULT_WINDOW_S,
        hop_s: Optional[float] = None,
        channels: Optional[List[str]] = None,
        norm_mode: str = "baseline",
    ):
        if norm_mode not in VALID_NORM_MODES:
            raise ValueError(f"norm_mode={norm_mode!r} not in {VALID_NORM_MODES}")
        self.norm_mode = norm_mode
        self.parquet_paths = [Path(p) for p in parquet_paths]
        self.active_only = active_only
        self.verbose = verbose
        self.phase_whitelist = phase_whitelist
        # Per-instance channel selection — defaults to RAW_CHANNELS. Used for
        # modality ablation: pass a subset to drop or keep one modality.
        self.channels = list(channels) if channels else list(RAW_CHANNELS)
        unknown = [c for c in self.channels if c not in RAW_CHANNELS]
        if unknown:
            raise ValueError(
                f"channels={self.channels} contains unknown entries {unknown}. "
                f"Valid: {RAW_CHANNELS}")
        # Per-instance window/hop in samples on the 100 Hz labeled grid.
        # 50 % overlap default: hop = window/2.
        self.window_s = float(window_s)
        self.window_size = int(round(self.window_s * FS_LABELED_GRID))
        self.hop_s = float(hop_s) if hop_s is not None else self.window_s / 2
        self.hop_size = max(1, int(round(self.hop_s * FS_LABELED_GRID)))

        # Resolve target modes (defaults preserve hard-label legacy pipeline).
        modes = dict(DEFAULT_TARGET_MODES)
        if target_modes:
            modes.update(target_modes)
        if modes['reps'] not in VALID_REPS_MODES:
            raise ValueError(f"target_modes['reps']={modes['reps']!r} not in {VALID_REPS_MODES}")
        if modes['phase'] not in VALID_PHASE_MODES:
            raise ValueError(f"target_modes['phase']={modes['phase']!r} not in {VALID_PHASE_MODES}")
        self.target_modes = modes

        # ---- Load all DataFrames; keep as list (lazy per-window access) -----
        self._dfs: List[pd.DataFrame] = []
        # per-recording z-score stats computed from first 90 s
        self._chan_mean: List[np.ndarray] = []
        self._chan_std: List[np.ndarray] = []

        all_exercise_vals: List[str] = []
        all_phase_vals: List[str] = []

        for path in self.parquet_paths:
            df = pd.read_parquet(path)
            self._dfs.append(df)
            mean, std = self._compute_baseline_stats(df)
            self._chan_mean.append(mean)
            self._chan_std.append(std)
            all_exercise_vals.extend(df['exercise'].dropna().astype(str).tolist())
            all_phase_vals.extend(df['phase_label'].dropna().astype(str).tolist())

        # Validate that the parquet has the columns needed for soft modes.
        if self.target_modes['reps'] == 'soft_window':
            missing = [str(p) for p, df in zip(self.parquet_paths, self._dfs)
                       if 'rep_density_hz' not in df.columns]
            if missing:
                raise KeyError(
                    "target_modes['reps']='soft_window' requires 'rep_density_hz' "
                    "column in aligned_features.parquet — re-run /label after the "
                    "soft-target labeling change. Missing in: " + ", ".join(missing)
                )

        # ---- Build label encoders (fit on all data, deterministic) ----------
        self.exercise_encoder: LabelEncoder = (
            exercise_encoder or LabelEncoder().fit(all_exercise_vals)
        )
        # Phase: fit only known phases (exclude 'unknown')
        known_phases = [v for v in all_phase_vals if v != 'unknown']
        self.phase_encoder: LabelEncoder = (
            phase_encoder or LabelEncoder().fit(known_phases)
        )
        if verbose:
            print(f"[raw_dataset] exercise classes: {self.exercise_encoder.classes_}")
            print(f"[raw_dataset] phase classes:    {self.phase_encoder.classes_}")

        # ---- Build window index: list of (file_idx, start_sample_in_df) -----
        self._window_idx: List[Tuple[int, int]] = []
        self._subject_ids_per_window: List[str] = []
        # Per-window (recording, set) identifiers — populated below so that
        # per-set evaluation (Rute A) can aggregate predictions by set
        # without re-deriving them at eval time.
        self._recording_ids_per_window: List[str] = []
        self._set_numbers_per_window: List[int] = []
        # Per-window override mask for the phase head. True => phase loss/eval
        # contributes; False => masked out (set is not whitelisted).
        self._phase_wl_per_window: List[bool] = []

        n_phase_kept = 0
        n_phase_total = 0

        for file_idx, df in enumerate(self._dfs):
            n = len(df)
            subj = df['subject_id'].iloc[0] if 'subject_id' in df.columns else f"file_{file_idx}"
            # For active-only: precompute boolean mask
            if active_only and 'in_active_set' in df.columns:
                active_mask = df['in_active_set'].astype(bool).values
            else:
                active_mask = np.ones(n, dtype=bool)

            # Per-sample (recording, set) lookup — always computed (cheap),
            # used both for the optional phase whitelist and the per-window
            # set identifier exposed for per-set exercise eval.
            if 'recording_id' in df.columns:
                rid_col = df['recording_id'].astype(str).to_numpy()
            else:
                rid_col = np.array([f"file_{file_idx}"] * n, dtype=object)
            if 'set_number' in df.columns:
                set_col_f = pd.to_numeric(df['set_number'],
                                            errors='coerce').to_numpy()
            else:
                set_col_f = np.full(n, np.nan)

            if phase_whitelist is not None:
                if 'recording_id' not in df.columns or 'set_number' not in df.columns:
                    raise ValueError(
                        "phase_whitelist requires 'recording_id' and 'set_number' "
                        "columns in aligned_features.parquet."
                    )
                wl_per_sample = np.zeros(n, dtype=bool)
                for i in range(n):
                    s = set_col_f[i]
                    if s != s:  # NaN
                        continue
                    wl_per_sample[i] = (rid_col[i], int(round(s))) in phase_whitelist
            else:
                wl_per_sample = None

            for start in range(0, n - self.window_size + 1, self.hop_size):
                end = start + self.window_size - 1  # last sample (inclusive)
                if not active_mask[end]:
                    continue
                self._window_idx.append((file_idx, start))
                self._subject_ids_per_window.append(str(subj))
                self._recording_ids_per_window.append(str(rid_col[end]))
                s_end = set_col_f[end]
                self._set_numbers_per_window.append(
                    -1 if s_end != s_end else int(round(s_end))
                )
                if wl_per_sample is None:
                    self._phase_wl_per_window.append(True)
                else:
                    keep = bool(wl_per_sample[end])
                    self._phase_wl_per_window.append(keep)
                    n_phase_total += 1
                    if keep:
                        n_phase_kept += 1

        if phase_whitelist is not None and verbose:
            print(f"[raw_dataset] phase whitelist: kept {n_phase_kept}/{n_phase_total} "
                  f"active windows for phase head "
                  f"({len(phase_whitelist)} (recording, set) pairs)")

        if verbose:
            print(f"[raw_dataset] {len(self._window_idx)} active windows from "
                  f"{len(self._dfs)} recordings")
        # Per-window numpy arrays (parallel to _window_idx). Used by per-set
        # exercise eval (Rute A) — slice by test_idx, build a (rec, set) key,
        # pass to compute_all_metrics(set_keys=...).
        self.recording_ids = np.asarray(self._recording_ids_per_window,
                                          dtype=object)
        self.set_numbers = np.asarray(self._set_numbers_per_window,
                                        dtype=np.int64)
        # Filled in by materialize_to_device(); enables GPU fast path.
        self.gpu_resident: bool = False

    def _compute_baseline_stats(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-channel (center, scale) for z-score normalization.

        The output is always (center, scale) so the per-window code path is
        a single (window − center) / scale regardless of norm_mode. The mode
        only changes how center/scale are derived:

        - "baseline" : center=mean(first 90 s), scale=std(first 90 s).
                       Backward-compatible default.
        - "robust"   : center=median(full recording),
                       scale=MAD(full recording) × 1.4826.
                       Robust to heavy-lift outliers; recommended when one
                       subject dominates std due to a single high-amplitude
                       set (Huber 1981; Rousseeuw & Croux 1993).
        - "percentile": center=mean(first 90 s),
                        scale=99th-percentile(|x − center|) over full recording.
                        Equalizes "max activation" scale across subjects of
                        different absolute strength.

        Sparse-channel fallback (e.g., temp): if the chosen window has < 10
        valid samples, fall back to the full-recording equivalent.
        """
        chans = self.channels
        baseline = df.iloc[:BASELINE_SAMPLES][chans].to_numpy(dtype=np.float32)
        full = df[chans].to_numpy(dtype=np.float32)
        center = np.full(len(chans), 0.0, dtype=np.float32)
        scale = np.full(len(chans), 1.0, dtype=np.float32)

        for i in range(len(chans)):
            col_base = baseline[:, i]
            col_full = full[:, i]
            base_valid = col_base[~np.isnan(col_base)]
            full_valid = col_full[~np.isnan(col_full)]
            if len(full_valid) == 0:
                continue  # keep 0/1 defaults

            if self.norm_mode == "baseline":
                use = base_valid if len(base_valid) >= 10 else full_valid
                center[i] = float(np.mean(use))
                scale[i] = max(float(np.std(use)), 1e-8)

            elif self.norm_mode == "robust":
                med = float(np.median(full_valid))
                mad = float(np.median(np.abs(full_valid - med)))
                center[i] = med
                # MAD × 1.4826 ≈ std for normally-distributed data
                scale[i] = max(mad * 1.4826, 1e-8)

            elif self.norm_mode == "percentile":
                base_ref = base_valid if len(base_valid) >= 10 else full_valid
                center[i] = float(np.mean(base_ref))
                p99 = float(np.percentile(np.abs(full_valid - center[i]), 99.0))
                scale[i] = max(p99, 1e-8)

        return center, scale

    def __len__(self) -> int:
        return len(self._window_idx)

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
    def subject_ids(self) -> List[str]:
        return self._subject_ids_per_window

    def materialize_to_device(self, device) -> None:
        """Pre-materialize all windows + targets onto `device` and stash them
        under _gpu_* names so the training loop can bypass DataLoader.

        Iterates __getitem__ once per window — typically <30 s for ~17 k
        active windows on this dataset. Subsequent __getitem__ calls return
        from the cache, so DataLoader-based code still works.
        """
        if getattr(self, "gpu_resident", False):
            return
        device = torch.device(device)
        n = len(self)
        if n == 0:
            self.gpu_resident = False
            return
        sample = self[0]

        x_storage = torch.empty((n, *sample['x'].shape), dtype=sample['x'].dtype)
        target_storage = {}
        for k, v in sample['targets'].items():
            shape = (n,) if v.ndim == 0 else (n, *v.shape)
            target_storage[k] = torch.empty(shape, dtype=v.dtype)
        mask_storage = {k: torch.empty(n, dtype=torch.bool)
                         for k in sample['masks']}

        for i in range(n):
            item = self[i]  # uses original lazy path (gpu_resident still False)
            x_storage[i] = item['x']
            for k, v in item['targets'].items():
                target_storage[k][i] = v
            for k, v in item['masks'].items():
                mask_storage[k][i] = v

        self._gpu_x = x_storage.to(device, non_blocking=True)
        self._gpu_targets = {k: v.to(device, non_blocking=True)
                              for k, v in target_storage.items()}
        self._gpu_masks = {k: v.to(device, non_blocking=True)
                            for k, v in mask_storage.items()}
        self.gpu_resident = True

    def __getitem__(self, idx: int) -> Dict:
        # Fast path: GPU-materialized cache (used by DataLoader fallbacks; the
        # main training loop bypasses __getitem__ entirely via _GPUBatchIterator).
        if getattr(self, "gpu_resident", False):
            return {
                'x': self._gpu_x[idx],
                'targets': {k: v[idx] for k, v in self._gpu_targets.items()},
                'masks':   {k: v[idx] for k, v in self._gpu_masks.items()},
            }
        file_idx, start = self._window_idx[idx]
        df = self._dfs[file_idx]
        chan_mean = self._chan_mean[file_idx]
        chan_std = self._chan_std[file_idx]

        end_sample = start + self.window_size  # exclusive

        # ---- Extract raw signal window: (T, C) -> normalize -> (C, T) -----
        window = df.iloc[start:end_sample][self.channels].to_numpy(dtype=np.float32)
        # Z-score normalize per channel
        window = (window - chan_mean[np.newaxis, :]) / chan_std[np.newaxis, :]
        # NaN -> 0
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip at ±5 sigma
        window = np.clip(window, -CLIP_SIGMA, CLIP_SIGMA)
        # Transpose to (C, T)
        x = torch.from_numpy(window.T)  # (C=6, T=200)

        # ---- End-of-window targets -----------------------------------------
        row = df.iloc[end_sample - 1]

        # exercise: NaN (rest periods between sets) -> mask=False so the
        # window contributes to other heads (phase, etc.) but not to the
        # exercise classification loss.
        ex_val = row['exercise']
        if pd.notna(ex_val):
            try:
                ex_arr = self.exercise_encoder.transform([str(ex_val)])
                ex_idx = int(ex_arr[0])
                ex_mask = ex_idx >= 0
            except (KeyError, ValueError):
                ex_idx = 0
                ex_mask = False
        else:
            ex_idx = 0
            ex_mask = False

        # phase target: hard (long index) or soft (probability vector).
        if self.target_modes['phase'] == 'soft':
            phase_samples = (
                df['phase_label'].iloc[start:end_sample].astype(str).to_numpy()
            )
            n_total = len(phase_samples)
            n_unknown = int((phase_samples == 'unknown').sum())
            n_known = n_total - n_unknown
            ph_target = np.zeros(self.n_phase, dtype=np.float32)
            if n_known > 0:
                for k, name in enumerate(self.phase_encoder.classes_):
                    ph_target[k] = (phase_samples == name).sum() / n_known
            unknown_frac = n_unknown / max(n_total, 1)
            ph_mask = (
                unknown_frac < PHASE_UNKNOWN_MAX_FRAC
                and self._phase_wl_per_window[idx]
            )
            phase_tensor = torch.from_numpy(ph_target)
        else:
            # 'unknown' -> mask=False; valid -> remap through phase_encoder.
            # 'rest' is a valid label (align.py writes it outside active sets)
            # and gets a real class index when in the encoder vocabulary.
            ph_str = str(row['phase_label']) if pd.notna(row['phase_label']) else 'unknown'
            if ph_str == 'unknown':
                ph_idx = 0
                ph_mask = False
            else:
                try:
                    ph_arr = self.phase_encoder.transform([ph_str])
                    ph_idx = int(ph_arr[0])
                    ph_mask = ph_idx >= 0
                except (KeyError, ValueError):
                    ph_idx = 0
                    ph_mask = False
            ph_mask = ph_mask and self._phase_wl_per_window[idx]
            phase_tensor = torch.tensor(ph_idx, dtype=torch.long)

        # fatigue (rpe_for_this_set)
        rpe_col = 'rpe_for_this_set'
        rpe_val = row[rpe_col] if rpe_col in df.columns else float('nan')
        rpe_float = float(rpe_val) if pd.notna(rpe_val) else float('nan')
        fat_valid = not np.isnan(rpe_float)

        # reps target: three modes —
        #   hard         — cumulative integer count at end-of-window
        #   soft_window  — mean rep_density_hz over window × window_dur_s
        #   soft_overlap — Σ overlap_fraction(rep_k, window) (Wang 2026)
        if self.target_modes['reps'] == 'soft_overlap':
            # Pick the soft_overlap column matching this dataset's window size,
            # falling back to the default 2 s column for backward compat.
            col = (f'soft_overlap_reps_{self.window_s:g}s'
                   .replace('.', '_'))
            if col not in df.columns:
                col = SOFT_OVERLAP_COL  # legacy fallback (= 2 s)
            if col not in df.columns:
                raise KeyError(
                    f"target_modes['reps']='soft_overlap' requires column "
                    f"{col!r} in aligned_features.parquet. "
                    "Run scripts/add_soft_overlap_reps.py first."
                )
            rep_float = float(df[col].iloc[end_sample - 1])
            in_act = row.get('in_active_set', False)
            has_iv = row.get('has_rep_intervals', True)
            rep_valid = (bool(in_act) and bool(has_iv)
                          if pd.notna(in_act) else False)
        elif self.target_modes['reps'] == 'soft_window':
            density = (
                df['rep_density_hz'].iloc[start:end_sample]
                .to_numpy(dtype=np.float32)
            )
            density = np.nan_to_num(density, nan=0.0)
            rep_float = float(density.mean() * self.window_s)
            in_act = row.get('in_active_set', False)
            # Mask reps when rep_count_in_set is NaN at end-of-window. This
            # is how the per-set blacklist (src/labeling/run.py
            # _PHASE_REPS_BLACKLIST) signals "active set, but reps are
            # untrustworthy — do not contribute to the reps loss".
            rc_val = row.get('rep_count_in_set')
            rep_valid = (
                bool(in_act) and pd.notna(rc_val)
                if pd.notna(in_act) else False
            )
        else:
            rep_val = row['rep_count_in_set']
            rep_float = float(rep_val) if pd.notna(rep_val) else float('nan')
            rep_valid = not np.isnan(rep_float)

        return {
            'x': x,
            'targets': {
                'exercise': torch.tensor(ex_idx, dtype=torch.long),
                'phase':    phase_tensor,
                'fatigue':  torch.tensor(rpe_float if fat_valid else 0.0,
                                         dtype=torch.float32),
                'reps':     torch.tensor(rep_float if rep_valid else 0.0,
                                         dtype=torch.float32),
            },
            'masks': {
                'exercise': torch.tensor(ex_mask, dtype=torch.bool),
                'phase':    torch.tensor(ph_mask, dtype=torch.bool),
                'fatigue':  torch.tensor(fat_valid, dtype=torch.bool),
                'reps':     torch.tensor(rep_valid, dtype=torch.bool),
            },
        }
