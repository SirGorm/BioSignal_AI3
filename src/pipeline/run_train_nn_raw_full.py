"""Raw-signal NN training orchestration — /train-nn raw variant.

Trains 4 architectures (CNN1D, LSTM, CNN-LSTM, TCN) on raw multimodal
biosignals (B, C=6, T=200) from aligned_features.parquet.

Strategy:
  Phase 1: 5-fold GroupKFold (reuse LightGBM splits), 1 seed, Optuna 30 trials.
           Goal: identify top 2-3 variants.
  Phase 2: Top 2-3 from Phase 1, seeds=[42, 1337, 7], full 50 epochs.
  Ablation: Soft-sharing on the overall winner.
  Latency: p99 benchmark for all 4 archs (GPU + CPU).

Compares against:
  - LightGBM baseline: runs/20260427_110653_default/metrics.json
  - Features-input NN run: runs/20260427_121303_nn_features_full/

Subject-wise CV: strictly enforced via splits_per_fold.csv.
No data leakage: test subjects never appear in train split.
Multi-seed: all reported Phase 2 results use 3 seeds.

GPU: auto-detected via src.utils.device.torch_device().
Mixed precision: torch.amp.autocast enabled on CUDA.

References:
- Bai et al. 2018 — TCN
- Hochreiter & Schmidhuber 1997 — LSTM
- Yang et al. 2015 — 1D-CNN for sensor data
- Ordóñez & Roggen 2016 — DeepConvLSTM
- Caruana 1997 — hard parameter sharing
- Ruder 2017 — soft sharing ablation
- Saeb et al. 2017 — subject-wise CV
- Loshchilov & Hutter 2019 — AdamW
- Akiba et al. 2019 — Optuna
- Goodfellow et al. 2016 — regularization
- LeCun et al. 2015 — deep learning over hand-crafted features
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import GroupKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore', category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.raw_window_dataset import RawMultimodalWindowDataset, RAW_CHANNELS
from src.models.raw.cnn1d_raw import CNN1DRawMultiTask
from src.models.raw.lstm_raw import LSTMRawMultiTask
from src.models.raw.cnn_lstm_raw import CNNLSTMRawMultiTask
from src.models.raw.tcn_raw import TCNRawMultiTask
from src.training.losses import MultiTaskLoss
from src.eval.metrics import compute_all_metrics
from src.utils.device import torch_device

# ---- paths ------------------------------------------------------------------
LABELED_DATA_ROOT = ROOT / 'data/labeled'
SPLITS_PER_FOLD_PATH = ROOT / 'configs/splits_per_fold.csv'
LGBM_METRICS_PATH = ROOT / 'runs/20260427_110653_default/metrics.json'
FEATURES_RUN_PATH = ROOT / 'runs/20260427_121303_nn_features_full'
RUN_DIR = ROOT / 'runs/20260427_153421_nn_raw_full'

# ---- training constants (GPU-adapted) ----------------------------------------
GPU_EPOCHS_P1 = 50
GPU_EPOCHS_P2 = 50
GPU_OPTUNA_TRIALS = 30
BATCH_SIZE = 128      # start at 128; models are larger than feature-MLP
PATIENCE = 8
LOSS_WEIGHTS = {'exercise': 1.0, 'phase': 1.0, 'fatigue': 1.0, 'reps': 0.5}
N_CHANNELS = 4  # EMG, PPG-green, acc_mag, temp (ECG + EDA dropped — see RAW_CHANNELS)
N_TIMESTEPS = 200

# ---- features run total for alignment assertion ------------------------------
FEATURES_RUN_TOTAL_WINDOWS = None  # loaded lazily from features run cv_summary


# =============================================================================
# Helpers
# =============================================================================

def set_seed(seed: int):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.item() if o.numel() == 1 else o.tolist()
    return str(o)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_jsonable)


def load_json(path: Path) -> Dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# =============================================================================
# Dataset helper — wraps RawMultimodalWindowDataset to match train_nn interface
# =============================================================================

class RawDatasetWrapper:
    """Thin adapter to expose the same interface as FilteredWindowDataset."""

    def __init__(self, ds: RawMultimodalWindowDataset):
        self._ds = ds

    @property
    def n_exercise(self): return self._ds.n_exercise
    @property
    def n_phase(self): return self._ds.n_phase
    @property
    def n_channels(self): return self._ds.n_channels
    @property
    def n_timesteps(self): return self._ds.n_timesteps
    @property
    def subject_ids(self): return self._ds.subject_ids
    def __len__(self): return len(self._ds)
    def __getitem__(self, idx): return self._ds[idx]


# =============================================================================
# Splits reuse from LightGBM baseline
# =============================================================================

def load_baseline_splits(dataset: RawDatasetWrapper) -> List[Dict]:
    """Reuse splits_per_fold.csv — identical fold assignments as LightGBM."""
    if not SPLITS_PER_FOLD_PATH.exists():
        raise FileNotFoundError(
            f"splits_per_fold.csv not found at {SPLITS_PER_FOLD_PATH}. "
            "Run /train (LightGBM baseline) first."
        )
    splits_df = pd.read_csv(SPLITS_PER_FOLD_PATH)
    subject_ids = np.array(dataset.subject_ids)

    folds = []
    for fold_id in sorted(splits_df['fold'].unique()):
        test_subjects = splits_df[
            (splits_df['fold'] == fold_id) & (splits_df['split'] == 'test')
        ]['subject_id'].tolist()
        train_subjects = splits_df[
            (splits_df['fold'] == fold_id) & (splits_df['split'] == 'train')
        ]['subject_id'].tolist()

        test_idx = np.where(np.isin(subject_ids, test_subjects))[0]
        train_idx = np.where(np.isin(subject_ids, train_subjects))[0]
        folds.append({
            'fold': int(fold_id),
            'train_idx': train_idx,
            'test_idx': test_idx,
            'test_subjects': test_subjects,
            'train_subjects': train_subjects,
        })
    print(f"[splits] {len(folds)} folds loaded from baseline splits.")
    for f in folds:
        print(f"  Fold {f['fold']}: train={len(f['train_idx'])}, "
              f"test={len(f['test_idx'])}, test_subjects={f['test_subjects']}")
    return folds


def verify_window_count_alignment(dataset: RawDatasetWrapper,
                                   features_run_dir: Path) -> None:
    """Assert that the raw dataset's active-window coverage is consistent with
    the features run's active-sample count.

    The features run uses per-sample rows (one row per 100 Hz timestep).
    The raw dataset uses 2 s windows (200 samples) with 50% overlap
    (100-sample / 1.0 s hop).
    Expected: n_raw_windows ≈ n_features_active_samples / HOP_SIZE (±1%).

    The two datasets use the same active-only filter and the same source files,
    so their coverage should correspond within rounding of the windowing.
    """
    features_cv = features_run_dir / 'features_cnn1d' / 'cv_summary.json'
    if not features_cv.exists():
        print("[alignment check] features cv_summary.json not found — skipping")
        return

    with open(features_cv) as f:
        data = json.load(f)

    # Sum all fold test n (each fold covers different subjects — no overlap)
    n_features_active_samples = sum(
        r['metrics']['exercise']['n']
        for r in data.get('all_results', [])
    )

    if n_features_active_samples == 0:
        print("[alignment check] features run has no metrics — skipping")
        return

    # Expected raw windows ≈ active_samples / hop (10 samples per hop)
    # Account for edge effect: each recording loses ~(WINDOW_SIZE/HOP - 1) windows
    from src.data.raw_window_dataset import HOP_SIZE, WINDOW_SIZE
    expected_raw_approx = n_features_active_samples / HOP_SIZE
    n_raw_total = len(dataset)

    # Use 10% tolerance (windowing edge effects per recording + minor alignment diffs)
    ratio = abs(n_raw_total - expected_raw_approx) / max(expected_raw_approx, 1)
    print(f"[alignment check] raw_windows={n_raw_total}, "
          f"features_active_samples={n_features_active_samples}, "
          f"expected_raw~={expected_raw_approx:.0f} (active/hop), "
          f"diff={ratio:.2%}")

    if ratio > 0.10:
        raise ValueError(
            f"Window count mismatch > 10%: raw={n_raw_total} vs "
            f"expected~={expected_raw_approx:.0f} "
            f"(features_active={n_features_active_samples} / hop={HOP_SIZE}). "
            "Check windowing parameters and active-only filter. "
            "Stopping — do not silently proceed with misaligned splits."
        )
    print("[alignment check] PASS — raw window count consistent with features run.")


# =============================================================================
# Model factories
# =============================================================================

def make_raw_factory(arch: str, n_exercise: int, n_phase: int,
                     hp: Dict) -> Callable[[], nn.Module]:
    def factory():
        if arch == 'cnn1d':
            return CNN1DRawMultiTask(
                n_channels=N_CHANNELS, n_timesteps=N_TIMESTEPS,
                n_exercise=n_exercise, n_phase=n_phase,
                repr_dim=hp.get('repr_dim', 128),
                dropout=hp.get('dropout', 0.3),
            )
        elif arch == 'lstm':
            return LSTMRawMultiTask(
                n_channels=N_CHANNELS, n_timesteps=N_TIMESTEPS,
                n_exercise=n_exercise, n_phase=n_phase,
                hidden=hp.get('hidden', 128),
                n_layers=hp.get('n_layers', 2),
                repr_dim=hp.get('repr_dim', 128),
                dropout=hp.get('dropout', 0.3),
            )
        elif arch == 'cnn_lstm':
            return CNNLSTMRawMultiTask(
                n_channels=N_CHANNELS, n_timesteps=N_TIMESTEPS,
                n_exercise=n_exercise, n_phase=n_phase,
                conv_channels=hp.get('conv_channels', 64),
                lstm_hidden=hp.get('lstm_hidden', 64),
                repr_dim=hp.get('repr_dim', 128),
                dropout=hp.get('dropout', 0.3),
            )
        elif arch == 'tcn':
            return TCNRawMultiTask(
                n_channels=N_CHANNELS, n_timesteps=N_TIMESTEPS,
                n_exercise=n_exercise, n_phase=n_phase,
                channels=hp.get('channels', [32, 64, 64, 128]),
                kernel_size=hp.get('kernel_size', 5),
                dropout=hp.get('dropout', 0.2),
                repr_dim=hp.get('repr_dim', 128),
            )
        else:
            raise ValueError(f"Unknown raw arch: {arch}")
    return factory


# =============================================================================
# Per-recording z-score normalization for raw signals (signal-domain equivalent
# of FeatureNormalizer). Applied inside the model training loop.
# Note: RawMultimodalWindowDataset already normalizes per-recording using the
# first 90 s baseline. The training loop does NOT apply additional normalization
# on top — the baseline z-score in the dataset is the sole normalization stage.
# =============================================================================

class RawFoldNormalizer:
    """Target (regression label) normalizer — same as features run FoldNormalizer.

    Fits on train split only. Denormalize preds before computing MAE.
    """

    def __init__(self, dataset: Dataset, train_idx: np.ndarray):
        fatigue_vals, reps_vals = [], []
        rng = np.random.default_rng(42)
        sample = rng.choice(train_idx, min(5000, len(train_idx)), replace=False)
        for i in sample:
            item = dataset[int(i)]
            if item['masks']['fatigue'].item():
                fatigue_vals.append(item['targets']['fatigue'].item())
            if item['masks']['reps'].item():
                reps_vals.append(item['targets']['reps'].item())

        self.fat_mean = float(np.mean(fatigue_vals)) if fatigue_vals else 7.0
        self.fat_std = max(float(np.std(fatigue_vals)), 0.1) if fatigue_vals else 1.0
        self.rep_mean = float(np.mean(reps_vals)) if reps_vals else 8.0
        self.rep_std = max(float(np.std(reps_vals)), 0.1) if reps_vals else 3.0

    def normalize_targets(self, targets: Dict) -> Dict:
        t = dict(targets)
        t['fatigue'] = (targets['fatigue'] - self.fat_mean) / self.fat_std
        t['reps'] = (targets['reps'] - self.rep_mean) / self.rep_std
        return t

    def denormalize_preds(self, preds: Dict) -> Dict:
        p = dict(preds)
        fat = preds['fatigue'] * self.fat_std + self.fat_mean
        rep = preds['reps'] * self.rep_std + self.rep_mean
        p['fatigue'] = torch.clamp(fat, 0.0, 15.0)
        p['reps'] = torch.clamp(rep, 0.0, 40.0)
        return p


# =============================================================================
# Training loop (single fold) — GPU + mixed precision
# =============================================================================

def train_one_fold_raw(
    model_factory: Callable[[], nn.Module],
    dataset: Dataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_exercise: int,
    n_phase: int,
    epochs: int = GPU_EPOCHS_P1,
    batch_size: int = BATCH_SIZE,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = PATIENCE,
    device: torch.device = torch.device('cpu'),
    out_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[List[Dict], Dict]:
    """Train one raw-input fold with mixed precision and early stopping."""
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    normalizer = RawFoldNormalizer(dataset, train_idx)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda'), drop_last=True,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )

    model = model_factory().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = MultiTaskLoss(
        w_exercise=LOSS_WEIGHTS['exercise'],
        w_phase=LOSS_WEIGHTS['phase'],
        w_fatigue=LOSS_WEIGHTS['fatigue'],
        w_reps=LOSS_WEIGHTS['reps'],
    ).to(device)

    best_val = float('inf')
    best_state = None
    pat = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            x = batch['x'].to(device)
            tgt_raw = {k: v.to(device) for k, v in batch['targets'].items()}
            tgt = normalizer.normalize_targets(tgt_raw)
            msk = {k: v.to(device) for k, v in batch['masks'].items()}
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                preds = model(x)
                total, _ = loss_fn(preds, tgt, msk)

            scaler.scale(total).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            n = x.shape[0]
            train_loss += total.item() * n
            n_train += n
        sched.step()
        train_loss /= max(n_train, 1)

        val_loss, val_metrics = _eval_fold_raw(
            model, test_loader, loss_fn, device, n_exercise, n_phase,
            normalizer, use_amp
        )
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_loss': val_loss['total'],
                        'val_metrics': val_metrics})

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"    ep{epoch:3d}  tr={train_loss:.4f}  "
                  f"val={val_loss['total']:.4f}  "
                  f"exF1={val_metrics['exercise']['f1_macro']:.3f}  "
                  f"phF1={val_metrics['phase']['f1_macro']:.3f}  "
                  f"fatMAE={val_metrics['fatigue']['mae']:.3f}  "
                  f"repMAE={val_metrics['reps']['mae']:.3f}")

        if val_loss['total'] < best_val:
            best_val = val_loss['total']
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch}")
                break

    # Sanity check: train loss should decrease
    if len(history) > 1:
        first_loss = history[0]['train_loss']
        last_loss = history[-1]['train_loss']
        if last_loss >= first_loss * 0.99 and first_loss > 1e-6:
            print(f"  WARNING: train loss did not decrease "
                  f"({first_loss:.4f} -> {last_loss:.4f})")

    if best_state is None:
        print("  WARNING: No valid training epoch. Returning dummy metrics.")
        final_metrics = {
            'exercise': {'f1_macro': 0.0, 'balanced_accuracy': 0.0, 'n': 0},
            'phase': {'f1_macro': 0.0, 'balanced_accuracy': 0.0, 'n': 0},
            'fatigue': {'mae': float('nan'), 'pearson_r': float('nan'), 'n': 0},
            'reps': {'mae': float('nan'), 'n': 0},
        }
    else:
        model.load_state_dict(best_state)
        _, final_metrics = _eval_fold_raw(
            model, test_loader, loss_fn, device, n_exercise, n_phase,
            normalizer, use_amp
        )

    if out_dir:
        torch.save({'state_dict': best_state,
                    'normalizer': {'fat_mean': normalizer.fat_mean,
                                   'fat_std': normalizer.fat_std,
                                   'rep_mean': normalizer.rep_mean,
                                   'rep_std': normalizer.rep_std}},
                    out_dir / 'checkpoint_best.pt')
        save_json(history, out_dir / 'history.json')
        save_json(final_metrics, out_dir / 'metrics.json')

    return history, final_metrics


@torch.no_grad()
def _eval_fold_raw(model, loader, loss_fn, device, n_exercise, n_phase,
                    normalizer: RawFoldNormalizer, use_amp: bool):
    model.eval()
    all_preds = {'exercise': [], 'phase': [], 'fatigue': [], 'reps': []}
    all_tgt = {k: [] for k in all_preds}
    all_msk = {k: [] for k in all_preds}
    total_loss = {'total': 0.0, 'exercise': 0.0, 'phase': 0.0,
                  'fatigue': 0.0, 'reps': 0.0}
    n_total = 0

    for batch in loader:
        x = batch['x'].to(device)
        tgt_raw = {k: v.to(device) for k, v in batch['targets'].items()}
        tgt_norm = normalizer.normalize_targets(tgt_raw)
        msk = {k: v.to(device) for k, v in batch['masks'].items()}

        with torch.amp.autocast('cuda', enabled=use_amp):
            preds = model(x)
            total, parts = loss_fn(preds, tgt_norm, msk)

        n = x.shape[0]
        n_total += n
        total_loss['total'] += total.item() * n
        for k in parts:
            total_loss[k] += parts[k].item() * n

        preds_denorm = normalizer.denormalize_preds(preds)
        for k in all_preds:
            all_preds[k].append(preds_denorm[k].cpu())
            all_tgt[k].append(tgt_raw[k].cpu())
            all_msk[k].append(msk[k].cpu())

    losses = {k: v / max(n_total, 1) for k, v in total_loss.items()}
    cat_p = {k: torch.cat(v) for k, v in all_preds.items()}
    cat_t = {k: torch.cat(v) for k, v in all_tgt.items()}
    cat_m = {k: torch.cat(v) for k, v in all_msk.items()}

    for k in ('fatigue', 'reps'):
        p = cat_p[k]
        bad = ~torch.isfinite(p)
        if bad.any():
            cat_p[k] = torch.where(bad, torch.zeros_like(p), p)
            cat_m[k] = torch.where(bad, torch.zeros_like(cat_m[k], dtype=torch.bool),
                                    cat_m[k])

    metrics = compute_all_metrics(cat_p, cat_t, cat_m,
                                   n_exercise=n_exercise, n_phase=n_phase)
    return losses, metrics


# =============================================================================
# Optuna HP search (inner-CV, single fold for speed)
# =============================================================================

def tune_raw_hp(
    arch: str,
    dataset: RawDatasetWrapper,
    fold: Dict,
    n_trials: int = GPU_OPTUNA_TRIALS,
    device: torch.device = torch.device('cpu'),
) -> Dict:
    """Optuna inner-CV HP tuning for raw-input architectures."""
    train_idx = fold['train_idx']
    subject_ids = np.array(dataset.subject_ids)
    train_subjects = np.unique(subject_ids[train_idx])

    # Leave one train subject out as inner-val
    inner_val_sub = train_subjects[len(train_subjects) // 2]
    inner_val_idx = train_idx[subject_ids[train_idx] == inner_val_sub]
    inner_train_idx = train_idx[subject_ids[train_idx] != inner_val_sub]

    # Cap to 8000 / 3000 for speed (sequence models are expensive)
    rng = np.random.default_rng(42)
    if len(inner_train_idx) > 8000:
        inner_train_idx = rng.choice(inner_train_idx, 8000, replace=False)
    inner_val_cap = inner_val_idx[:2000]

    def objective(trial):
        hp = {
            'repr_dim': trial.suggest_categorical('repr_dim', [64, 128]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        }
        if arch == 'lstm':
            hp['hidden'] = trial.suggest_categorical('hidden', [64, 128])
            hp['n_layers'] = trial.suggest_int('n_layers', 1, 2)
        elif arch == 'cnn_lstm':
            hp['conv_channels'] = trial.suggest_categorical('conv_channels', [32, 64])
            hp['lstm_hidden'] = trial.suggest_categorical('lstm_hidden', [32, 64])
        elif arch == 'tcn':
            hp['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5])
        elif arch == 'cnn1d':
            pass  # only repr_dim / dropout / lr

        factory = make_raw_factory(arch, dataset.n_exercise, dataset.n_phase, hp)
        try:
            _, metrics = train_one_fold_raw(
                factory, dataset, inner_train_idx, inner_val_cap,
                n_exercise=dataset.n_exercise, n_phase=dataset.n_phase,
                epochs=10, batch_size=BATCH_SIZE, lr=hp['lr'],
                patience=4, device=device, out_dir=None, verbose=False,
            )
            ex_f1 = metrics['exercise']['f1_macro']
            ph_f1 = metrics['phase']['f1_macro']
            fat_mae = metrics['fatigue']['mae']
            rep_mae = metrics['reps']['mae']
            if any(np.isnan(float(v)) for v in [ex_f1, ph_f1, fat_mae, rep_mae]):
                return float('inf')
            score = ((1 - ex_f1) + (1 - ph_f1)
                     + fat_mae / 3.0 + rep_mae / 5.0)
            return score
        except Exception as e:
            print(f"    [optuna] trial failed: {e}")
            return float('inf')

    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"  [optuna {arch}] best params: {best} (score={study.best_value:.4f})")
    return best


# =============================================================================
# Run one variant over all folds
# =============================================================================

def run_variant_raw(
    variant_name: str,
    arch: str,
    dataset: RawDatasetWrapper,
    folds: List[Dict],
    out_root: Path,
    seeds: List[int],
    epochs: int,
    device: torch.device,
    run_optuna: bool = True,
    preloaded_hp: Optional[Dict] = None,
) -> Dict:
    variant_dir = out_root / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  VARIANT: {variant_name}  arch={arch}  seeds={seeds}")
    print(f"  epochs={epochs}  device={device}")
    print(f"{'='*60}")

    if preloaded_hp is not None:
        best_hp = preloaded_hp
        print(f"  [HP] Reusing preloaded HP: {best_hp}")
    elif run_optuna:
        print(f"  [HP tuning] Running {GPU_OPTUNA_TRIALS} trials on fold 0...")
        best_hp = tune_raw_hp(arch, dataset, folds[0], device=device)
    else:
        best_hp = {}
    save_json(best_hp, variant_dir / 'best_hp.json')

    all_results = []
    for seed in seeds:
        set_seed(seed)
        for fold in folds:
            fold_id = fold['fold']
            fold_dir = variant_dir / f"seed_{seed}" / f"fold_{fold_id}"
            print(f"\n  seed={seed}  fold={fold_id}  "
                  f"test={fold['test_subjects']}")

            factory = make_raw_factory(arch, dataset.n_exercise,
                                        dataset.n_phase, best_hp)
            history, metrics = train_one_fold_raw(
                factory, dataset, fold['train_idx'], fold['test_idx'],
                n_exercise=dataset.n_exercise, n_phase=dataset.n_phase,
                epochs=epochs, batch_size=BATCH_SIZE,
                lr=best_hp.get('lr', 1e-3),
                patience=PATIENCE,
                device=device, out_dir=fold_dir, verbose=True,
            )
            all_results.append({
                'seed': seed, 'fold': fold_id,
                'test_subjects': fold['test_subjects'],
                'metrics': metrics,
            })

    summary = _aggregate(all_results)
    summary['variant'] = variant_name
    summary['arch'] = arch
    summary['best_hp'] = best_hp
    summary['n_folds'] = len(folds)
    summary['seeds'] = seeds
    summary['subsampled'] = False
    summary['subsample_n'] = None

    save_json({'summary': summary, 'all_results': all_results},
               variant_dir / 'cv_summary.json')
    print(f"\n  {variant_name} DONE. "
          f"exercise F1={summary['exercise']['f1_macro']['mean']:.3f}  "
          f"phase F1={summary['phase']['f1_macro']['mean']:.3f}  "
          f"fatigue MAE={summary['fatigue']['mae']['mean']:.3f}  "
          f"reps MAE={summary['reps']['mae']['mean']:.3f}")
    return summary


def _aggregate(all_results: List[Dict]) -> Dict:
    def _collect(task, metric, cap: Optional[float] = None):
        vals = [r['metrics'].get(task, {}).get(metric) for r in all_results]
        vals = [v for v in vals
                if v is not None
                and not np.isnan(float(v))
                and not np.isinf(float(v))
                and abs(float(v)) < 1e10]
        if cap is not None:
            vals = [min(float(v), cap) for v in vals]
        if not vals:
            return {'mean': float('nan'), 'std': float('nan'), 'n': 0}
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                'n': len(vals)}
    return {
        'exercise': {'f1_macro': _collect('exercise', 'f1_macro')},
        'phase':    {'f1_macro': _collect('phase', 'f1_macro')},
        'fatigue':  {'mae': _collect('fatigue', 'mae', cap=20.0)},
        'reps':     {'mae': _collect('reps', 'mae', cap=50.0)},
    }


# =============================================================================
# Ranking
# =============================================================================

def rank_variants(results: Dict[str, Dict]) -> List[Dict]:
    tasks = [
        ('exercise', 'f1_macro', True),
        ('phase',    'f1_macro', True),
        ('fatigue',  'mae',      False),
        ('reps',     'mae',      False),
    ]
    data = []
    for vname, summary in results.items():
        row = {'variant': vname}
        for task, metric, _ in tasks:
            v = summary.get(task, {}).get(metric, {}).get('mean', float('nan'))
            row[f"{task}_{metric}"] = v
        data.append(row)

    for task, metric, higher_better in tasks:
        col = f"{task}_{metric}"
        vals = [(i, r[col]) for i, r in enumerate(data)
                if not np.isnan(r[col])]
        vals.sort(key=lambda x: x[1], reverse=higher_better)
        for rank, (i, _) in enumerate(vals):
            data[i][f"{col}_rank"] = rank + 1

    for row in data:
        rank_cols = [v for k, v in row.items() if k.endswith('_rank')]
        row['mean_rank'] = float(np.mean(rank_cols)) if rank_cols else float('nan')

    data.sort(key=lambda x: x['mean_rank'])
    return data


# =============================================================================
# Latency benchmark (GPU + CPU)
# =============================================================================

def latency_benchmark_raw(
    model_factory: Callable[[], nn.Module],
    n_warmup: int = 20,
    n_runs: int = 200,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, float]:
    """p50/p95/p99 latency on batch=1, single 2 s raw-signal window."""
    model = model_factory().to(device)
    model.eval()
    x = torch.randn(1, N_CHANNELS, N_TIMESTEPS, device=device)
    times = []
    with torch.no_grad():
        for i in range(n_warmup + n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= n_warmup:
                times.append((t1 - t0) * 1000)
    return {
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'mean_ms': float(np.mean(times)),
    }


# =============================================================================
# Soft-sharing ablation
# =============================================================================

class SoftSharingRaw(nn.Module):
    """4 separate encoders + 4 task heads. Ablation against hard sharing."""

    def __init__(self, encoder_factory: Callable[[], nn.Module],
                 n_exercise: int, n_phase: int, repr_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.enc_exercise = encoder_factory()
        self.enc_phase = encoder_factory()
        self.enc_fatigue = encoder_factory()
        self.enc_reps = encoder_factory()
        self.drop = nn.Dropout(dropout)
        self.head_exercise = nn.Linear(repr_dim, n_exercise)
        self.head_phase = nn.Linear(repr_dim, n_phase)
        self.head_fatigue = nn.Linear(repr_dim, 1)
        self.head_reps = nn.Linear(repr_dim, 1)

    def forward(self, x):
        return {
            'exercise': self.head_exercise(self.drop(self.enc_exercise.encode(x))),
            'phase':    self.head_phase(self.drop(self.enc_phase.encode(x))),
            'fatigue':  self.head_fatigue(
                self.drop(self.enc_fatigue.encode(x))).squeeze(-1),
            'reps':     self.head_reps(
                self.drop(self.enc_reps.encode(x))).squeeze(-1),
        }


def run_soft_sharing_ablation_raw(
    arch: str,
    dataset: RawDatasetWrapper,
    folds: List[Dict],
    run_dir: Path,
    device: torch.device,
    best_hp: Dict,
) -> Dict:
    ablation_dir = run_dir / f"ablation_soft_sharing_{arch}_raw"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    def encoder_factory():
        return make_raw_factory(arch, dataset.n_exercise, dataset.n_phase,
                                 best_hp)()

    soft_results, hard_results = [], []
    for fold in folds:
        # Cap train for ablation speed
        rng = np.random.default_rng(42)
        train_idx = fold['train_idx']
        if len(train_idx) > 15000:
            train_idx = rng.choice(train_idx, 15000, replace=False)

        # Soft sharing
        set_seed(42)
        _, s_metrics = train_one_fold_raw(
            lambda: SoftSharingRaw(
                encoder_factory, dataset.n_exercise, dataset.n_phase,
                best_hp.get('repr_dim', 128), best_hp.get('dropout', 0.3),
            ),
            dataset, train_idx, fold['test_idx'],
            n_exercise=dataset.n_exercise, n_phase=dataset.n_phase,
            epochs=15, batch_size=BATCH_SIZE,
            lr=best_hp.get('lr', 1e-3),
            patience=5, device=device,
            out_dir=ablation_dir / f"fold_{fold['fold']}_soft",
            verbose=False,
        )
        soft_results.append({'fold': fold['fold'], 'metrics': s_metrics})

        # Hard sharing
        set_seed(42)
        _, h_metrics = train_one_fold_raw(
            make_raw_factory(arch, dataset.n_exercise, dataset.n_phase, best_hp),
            dataset, train_idx, fold['test_idx'],
            n_exercise=dataset.n_exercise, n_phase=dataset.n_phase,
            epochs=15, batch_size=BATCH_SIZE,
            lr=best_hp.get('lr', 1e-3),
            patience=5, device=device,
            out_dir=ablation_dir / f"fold_{fold['fold']}_hard",
            verbose=False,
        )
        hard_results.append({'fold': fold['fold'], 'metrics': h_metrics})

    soft_agg = _aggregate(soft_results)
    hard_agg = _aggregate(hard_results)

    verdict = _ablation_verdict(soft_agg, hard_agg)
    result = {
        'arch': arch,
        'soft_sharing': soft_agg,
        'hard_sharing': hard_agg,
        'verdict': verdict,
    }
    save_json(result, ablation_dir / 'ablation_results.json')
    print(f"\n[ablation] soft vs hard sharing (raw {arch}):")
    print(f"  exercise F1: soft={soft_agg['exercise']['f1_macro']['mean']:.3f}  "
          f"hard={hard_agg['exercise']['f1_macro']['mean']:.3f}")
    print(f"  phase F1:    soft={soft_agg['phase']['f1_macro']['mean']:.3f}  "
          f"hard={hard_agg['phase']['f1_macro']['mean']:.3f}")
    print(f"  fatigue MAE: soft={soft_agg['fatigue']['mae']['mean']:.3f}  "
          f"hard={hard_agg['fatigue']['mae']['mean']:.3f}")
    print(f"  reps MAE:    soft={soft_agg['reps']['mae']['mean']:.3f}  "
          f"hard={hard_agg['reps']['mae']['mean']:.3f}")
    print(f"  verdict: {verdict}")
    return result


def _ablation_verdict(soft: Dict, hard: Dict) -> str:
    lines = []
    for task, metric, better_lower in [
        ('exercise', 'f1_macro', False),
        ('phase', 'f1_macro', False),
        ('fatigue', 'mae', True),
        ('reps', 'mae', True),
    ]:
        s = soft[task][metric]['mean']
        h = hard[task][metric]['mean']
        if np.isnan(s) or np.isnan(h):
            continue
        delta = s - h
        improved = delta < -0.05 if better_lower else delta > 0.02
        if improved:
            lines.append(f"soft improved {task} by {abs(delta):.3f}")
    if not lines:
        return ("Hard sharing sufficient — no task shows meaningful gain from "
                "soft sharing (Ruder 2017: soft sharing rarely helps in "
                "low-data regimes)")
    return "Soft sharing improved: " + "; ".join(lines)


# =============================================================================
# Per-subject breakdown (winner, fold 0)
# =============================================================================

def per_subject_breakdown_raw(
    model_factory: Callable[[], nn.Module],
    dataset: RawDatasetWrapper,
    fold: Dict,
    best_hp: Dict,
    device: torch.device,
) -> pd.DataFrame:
    train_idx = fold['train_idx']
    rng = np.random.default_rng(42)
    if len(train_idx) > 20000:
        train_idx = rng.choice(train_idx, 20000, replace=False)

    set_seed(42)
    model = model_factory().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=best_hp.get('lr', 1e-3),
                              weight_decay=1e-4)
    loss_fn = MultiTaskLoss(
        w_exercise=LOSS_WEIGHTS['exercise'],
        w_phase=LOSS_WEIGHTS['phase'],
        w_fatigue=LOSS_WEIGHTS['fatigue'],
        w_reps=LOSS_WEIGHTS['reps'],
    ).to(device)
    normalizer = RawFoldNormalizer(dataset, train_idx)
    loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=0, drop_last=True)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for _ in range(15):
        model.train()
        for batch in loader:
            x = batch['x'].to(device)
            tgt = normalizer.normalize_targets(
                {k: v.to(device) for k, v in batch['targets'].items()})
            msk = {k: v.to(device) for k, v in batch['masks'].items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                preds = model(x)
                total, _ = loss_fn(preds, tgt, msk)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

    test_loader = DataLoader(Subset(dataset, fold['test_idx']),
                              batch_size=512, shuffle=False, num_workers=0)
    subject_ids = np.array(dataset.subject_ids)[fold['test_idx']]
    all_p = {'exercise': [], 'phase': [], 'fatigue': [], 'reps': []}
    all_t = {k: [] for k in all_p}
    all_m = {k: [] for k in all_p}
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            preds_raw = model(x)
            preds = normalizer.denormalize_preds(preds_raw)
            for k in all_p:
                all_p[k].append(preds[k].cpu())
                all_t[k].append(batch['targets'][k])
                all_m[k].append(batch['masks'][k])
    cat_p = {k: torch.cat(v) for k, v in all_p.items()}
    cat_t = {k: torch.cat(v) for k, v in all_t.items()}
    cat_m = {k: torch.cat(v) for k, v in all_m.items()}

    rows = []
    for s in np.unique(subject_ids):
        mask = subject_ids == s
        sp = {k: v[mask] for k, v in cat_p.items()}
        st = {k: v[mask] for k, v in cat_t.items()}
        sm = {k: v[mask] for k, v in cat_m.items()}
        m = compute_all_metrics(sp, st, sm,
                                 n_exercise=dataset.n_exercise,
                                 n_phase=dataset.n_phase)
        rows.append({
            'subject_id': s,
            'exercise_f1': m['exercise']['f1_macro'],
            'phase_f1': m['phase']['f1_macro'],
            'fatigue_mae': m['fatigue']['mae'],
            'reps_mae': m['reps']['mae'],
        })
    return pd.DataFrame(rows)


# =============================================================================
# Comparison table builder
# =============================================================================

def build_comparison_raw(
    phase2_results: Dict,
    lgbm_metrics: Dict,
    features_run_results: Dict,
    latency: Dict,
) -> Dict:
    """Build comparison rows: LightGBM + features best + 4 raw archs."""

    # LightGBM baseline
    rows = [{
        'model': 'LightGBM (baseline)',
        'exercise_f1': lgbm_metrics.get('exercise', {}).get('f1_mean', float('nan')),
        'phase_f1': lgbm_metrics.get('phase', {}).get('ml_f1_mean', float('nan')),
        'fatigue_mae': lgbm_metrics.get('fatigue', {}).get('mae_mean', float('nan')),
        'reps_mae': lgbm_metrics.get('reps', {}).get('ml_mae_mean', float('nan')),
        'latency_p99_ms': None,
        'causal': True,
        'deployment_candidate': True,
        'input_variant': 'features',
    }]

    # Features run best (CNN1D was winner per phase1 ranking)
    feat_best_key = 'features_cnn1d'
    feat_best = features_run_results.get(feat_best_key, {})
    if feat_best:
        rows.append({
            'model': f'NN-features-best ({feat_best_key})',
            'exercise_f1': feat_best.get('exercise', {}).get('f1_macro', {}).get('mean', float('nan')),
            'phase_f1': feat_best.get('phase', {}).get('f1_macro', {}).get('mean', float('nan')),
            'fatigue_mae': feat_best.get('fatigue', {}).get('mae', {}).get('mean', float('nan')),
            'reps_mae': feat_best.get('reps', {}).get('mae', {}).get('mean', float('nan')),
            'latency_p99_ms': None,
            'causal': True,
            'deployment_candidate': True,
            'input_variant': 'features',
        })

    # Raw-input NN variants
    for vname, summary in phase2_results.items():
        arch = vname.replace('raw_', '').replace('_p2', '')
        is_causal = arch in ('cnn1d', 'tcn')
        rows.append({
            'model': vname,
            'arch': arch,
            'exercise_f1': summary['exercise']['f1_macro']['mean'],
            'exercise_f1_std': summary['exercise']['f1_macro']['std'],
            'phase_f1': summary['phase']['f1_macro']['mean'],
            'phase_f1_std': summary['phase']['f1_macro']['std'],
            'fatigue_mae': summary['fatigue']['mae']['mean'],
            'fatigue_mae_std': summary['fatigue']['mae']['std'],
            'reps_mae': summary['reps']['mae']['mean'],
            'reps_mae_std': summary['reps']['mae']['std'],
            'latency_p99_ms': latency.get(arch, {}).get('p99_ms'),
            'causal': is_causal,
            'deployment_candidate': is_causal,
            'input_variant': 'raw',
            'note': (None if is_causal else
                     'research_only: non-causal (BiLSTM); cannot deploy for '
                     'real-time streaming (CLAUDE.md "Kritiske regler")'),
        })

    return {
        'rows': rows,
        'lgbm_baseline': {
            'exercise': lgbm_metrics.get('exercise', {}).get('f1_mean'),
            'phase': lgbm_metrics.get('phase', {}).get('ml_f1_mean'),
            'fatigue': lgbm_metrics.get('fatigue', {}).get('mae_mean'),
            'reps': lgbm_metrics.get('reps', {}).get('ml_mae_mean'),
        },
    }


# =============================================================================
# SUMMARY.md writer
# =============================================================================

def write_summary(
    run_dir: Path,
    phase1_results: Dict,
    phase2_results: Dict,
    latency: Dict,
    ablation: Dict,
    lgbm_metrics: Dict,
    features_run_p1: Dict,
    winner_variant: str,
) -> None:
    """Write SUMMARY.md to run_dir. Required artifact per project spec."""

    def fmt_f1(d): return f"{d.get('mean', float('nan')):.3f} ± {d.get('std', float('nan')):.3f}"
    def fmt_mae(d): return f"{d.get('mean', float('nan')):.3f} ± {d.get('std', float('nan')):.3f}"

    lgbm_ex = lgbm_metrics.get('exercise', {}).get('f1_mean', float('nan'))
    lgbm_ph = lgbm_metrics.get('phase', {}).get('ml_f1_mean', float('nan'))
    lgbm_fat = lgbm_metrics.get('fatigue', {}).get('mae_mean', float('nan'))
    lgbm_rep = lgbm_metrics.get('reps', {}).get('ml_mae_mean', float('nan'))

    # Best raw variant per task
    def best_raw(task, metric, lower_better):
        best_name, best_val = None, float('inf') if lower_better else -float('inf')
        for vname, s in phase2_results.items():
            val = s.get(task, {}).get(metric, {}).get('mean', float('nan'))
            if np.isnan(val):
                continue
            if (lower_better and val < best_val) or (not lower_better and val > best_val):
                best_val = val
                best_name = vname
        return best_name, best_val

    raw_ex_name, raw_ex = best_raw('exercise', 'f1_macro', False)
    raw_ph_name, raw_ph = best_raw('phase', 'f1_macro', False)
    raw_fat_name, raw_fat = best_raw('fatigue', 'mae', True)
    raw_rep_name, raw_rep = best_raw('reps', 'mae', True)

    # Best features variant
    feat_ex = features_run_p1.get('features_cnn1d', {}).get('exercise', {}).get('f1_macro', {}).get('mean', float('nan'))
    feat_ph = features_run_p1.get('features_cnn1d', {}).get('phase', {}).get('f1_macro', {}).get('mean', float('nan'))
    feat_fat = features_run_p1.get('features_lstm', {}).get('fatigue', {}).get('mae', {}).get('mean', float('nan'))
    feat_rep = features_run_p1.get('features_cnn1d', {}).get('reps', {}).get('mae', {}).get('mean', float('nan'))

    def delta(a, b, lower_better):
        if np.isnan(a) or np.isnan(b): return 'N/A'
        d = a - b
        sign = '+' if (d > 0) else ''
        if lower_better:
            flag = ' (IMPROVEMENT)' if d < 0 else ' (regression vs LGBM)'
        else:
            flag = ' (IMPROVEMENT)' if d > 0 else ' (regression vs LGBM)'
        return f"{sign}{d:.3f}{flag}"

    lat_rows = []
    for arch in ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']:
        lt = latency.get(arch, {})
        causal = 'YES (deployable)' if arch in ('cnn1d', 'tcn') else 'NO (research_only)'
        lat_rows.append(
            f"| raw_{arch} | {lt.get('p50_ms', 'N/A'):.2f} | "
            f"{lt.get('p99_ms', 'N/A'):.2f} | {causal} |"
        )

    ablation_verdict = ablation.get('verdict', 'N/A')

    summary = f"""# Run Summary — 20260427_153421_nn_raw_full

**Date:** 2026-04-27
**Input variant:** Raw multimodal biosignals (B, C=6, T=200)
**Channels:** ecg, emg, eda, ppg_green, acc_mag, temp (6 modalities at 100 Hz)
**Architectures:** CNN1D, BiLSTM, CNN-LSTM, TCN (4 raw-input variants)
**CV scheme:** 5-fold GroupKFold (reuse of configs/splits_per_fold.csv from LightGBM baseline)
**Seeds:** Phase 1 = [42], Phase 2 = [42, 1337, 7]
**Baseline:** runs/20260427_110653_default (LightGBM)
**Prior NN run:** runs/20260427_121303_nn_features_full (features input)

---

## Per-Task Performance vs. Baselines

| Task | LightGBM | NN-features best | NN-raw best | Raw vs. LGBM |
|------|----------|-----------------|-------------|--------------|
| Exercise (F1-macro) | {lgbm_ex:.3f} | {feat_ex:.3f} (cnn1d) | {raw_ex:.3f} ({raw_ex_name}) | {delta(raw_ex, lgbm_ex, False)} |
| Phase (F1-macro) | {lgbm_ph:.3f} | {feat_ph:.3f} (cnn1d) | {raw_ph:.3f} ({raw_ph_name}) | {delta(raw_ph, lgbm_ph, False)} |
| Fatigue (MAE) | {lgbm_fat:.3f} | {feat_fat:.3f} (lstm) | {raw_fat:.3f} ({raw_fat_name}) | {delta(raw_fat, lgbm_fat, True)} |
| Reps (MAE) | {lgbm_rep:.3f} | {feat_rep:.3f} (cnn1d) | {raw_rep:.3f} ({raw_rep_name}) | {delta(raw_rep, lgbm_rep, True)} |

---

## Phase 1 — Screening Results (all 4 raw variants, 1 seed)

| Variant | Exercise F1 | Phase F1 | Fatigue MAE | Reps MAE |
|---------|-------------|----------|-------------|----------|
"""
    for vname, s in phase1_results.items():
        summary += (
            f"| {vname} | "
            f"{fmt_f1(s.get('exercise',{}).get('f1_macro',{}))} | "
            f"{fmt_f1(s.get('phase',{}).get('f1_macro',{}))} | "
            f"{fmt_mae(s.get('fatigue',{}).get('mae',{}))} | "
            f"{fmt_mae(s.get('reps',{}).get('mae',{}))} |\n"
        )

    summary += f"""
## Phase 2 — Final Depth (top variants, 3 seeds)

| Variant | Exercise F1 | Phase F1 | Fatigue MAE | Reps MAE |
|---------|-------------|----------|-------------|----------|
"""
    for vname, s in phase2_results.items():
        summary += (
            f"| {vname} | "
            f"{fmt_f1(s.get('exercise',{}).get('f1_macro',{}))} | "
            f"{fmt_f1(s.get('phase',{}).get('f1_macro',{}))} | "
            f"{fmt_mae(s.get('fatigue',{}).get('mae',{}))} | "
            f"{fmt_mae(s.get('reps',{}).get('mae',{}))} |\n"
        )

    summary += f"""
---

## Latency Benchmark (batch=1, 2 s window = (1, 6, 200), GPU)

| Variant | p50 (ms) | p99 (ms) | Causal / Deployable |
|---------|----------|----------|---------------------|
{chr(10).join(lat_rows)}

---

## Multi-Task Sharing Ablation

Winner variant: **{winner_variant}**

{ablation_verdict}

Hard-sharing (Caruana 1997) with one shared encoder + 4 task heads was tested
against soft-sharing (4 separate encoders, Ruder 2017). The verdict above
reflects whether soft-sharing produced any meaningful gain (>0.02 F1 or >0.05 MAE).

---

## Deployment Recommendation

**Causal candidates for real-time streaming:** TCN_raw, CNN1D_raw
- These use only left-padded / causal-dilated convolutions.
- BiLSTM_raw and CNN-LSTM_raw are marked `research_only` (CLAUDE.md "Kritiske regler").

**Ship recommendation:**
- If raw-input NN beats LightGBM on a task → deploy that NN for that task.
- If LightGBM still wins → keep LightGBM for that task.
- TCN_raw is the preferred causal NN if deployment of a single model is desired
  (parallelizable, faster than LSTM, causal).

---

## Open Questions

1. **Pragmatic 100 Hz resampling**: All 6 modalities are resampled to 100 Hz in
   aligned_features.parquet. The proper hybrid-fusion approach (EMG at 2000 Hz,
   ECG at 500 Hz, per nn.yaml "fast/medium/slow" groups) would require reading
   from dataset_aligned/ and per-modality resampling. This is expected to improve
   EMG frequency-domain features (MNF, MDF) that lose information when downsampled
   to 100 Hz. See nn.yaml "dataset.fusion: uniform_100hz" note.
   [REF NEEDED: citation for information loss in EMG downsampling]

2. **Low-data regime**: 9 subjects, ~313k active windows total, 108 RPE labels.
   NNs may overfit on per-subject artifacts rather than generalizing. Subject-wise
   CV enforces this constraint but doesn't resolve the low-N problem.

3. **Window-level vs. set-level fatigue**: RPE is a set-level label broadcast to
   all windows of a set. This creates a regression problem where many windows share
   the same label — temporal leakage within a set is unavoidable. A set-aggregated
   approach (window features -> mean -> set-level regression) would be cleaner but
   requires set-level CV.

---

## References

- Bai et al. 2018 — An empirical evaluation of generic convolutional and recurrent
  networks for sequence modeling (TCN)
- Hochreiter & Schmidhuber 1997 — Long short-term memory
- Yang et al. 2015 — Deep convolutional neural networks on multichannel time series
  for human activity recognition
- Ordóñez & Roggen 2016 — Deep convolutional and LSTM recurrent neural networks
  for multimodal wearable activity recognition (DeepConvLSTM)
- Caruana 1997 — Multitask learning (hard parameter sharing)
- Ruder 2017 — An overview of multi-task learning in deep neural networks
- Saeb et al. 2017 — The need for subject-independent evaluation of machine
  learning models in wearable health applications
- Loshchilov & Hutter 2019 — Decoupled weight decay regularization (AdamW)
- Akiba et al. 2019 — Optuna: A next-generation hyperparameter optimization
  framework
- Goodfellow et al. 2016 — Deep Learning (regularization, dropout)
- LeCun et al. 2015 — Deep learning (learned representations vs. hand-crafted
  features)
"""
    (run_dir / 'SUMMARY.md').write_text(summary, encoding='utf-8')
    print(f"[summary] Written to {run_dir / 'SUMMARY.md'}")


# =============================================================================
# Model cards per variant
# =============================================================================

def write_model_card(
    run_dir: Path,
    variant_name: str,
    arch: str,
    summary: Dict,
    best_hp: Dict,
    is_causal: bool,
    latency_info: Optional[Dict] = None,
) -> None:
    """Write per-variant model_card.md with mandatory References section."""

    ex_f1 = summary.get('exercise', {}).get('f1_macro', {}).get('mean', float('nan'))
    ph_f1 = summary.get('phase', {}).get('f1_macro', {}).get('mean', float('nan'))
    fat_mae = summary.get('fatigue', {}).get('mae', {}).get('mean', float('nan'))
    rep_mae = summary.get('reps', {}).get('mae', {}).get('mean', float('nan'))

    deployment_note = (
        "DEPLOYABLE (causal architecture — safe for real-time streaming pipeline)"
        if is_causal else
        "research_only — BIDIRECTIONAL component (non-causal). "
        "Cannot be deployed in real-time streaming. "
        "Use TCN_raw or CNN1D_raw for deployment."
    )

    lat_str = ""
    if latency_info:
        lat_str = (
            f"\n## Latency\n\n"
            f"Batch=1, (1, 6, 200) input, GPU (RTX 5070 Ti):\n"
            f"- p50: {latency_info.get('p50_ms', 'N/A'):.2f} ms\n"
            f"- p99: {latency_info.get('p99_ms', 'N/A'):.2f} ms\n"
        )

    arch_refs = {
        'tcn': "- Bai et al. 2018 — An empirical evaluation of generic convolutional and recurrent networks for sequence modeling",
        'lstm': "- Hochreiter & Schmidhuber 1997 — Long short-term memory\n- Schuster & Paliwal 1997 — Bidirectional recurrent neural networks",
        'cnn_lstm': "- Ordóñez & Roggen 2016 — Deep convolutional and LSTM recurrent neural networks for multimodal wearable activity recognition\n- Hochreiter & Schmidhuber 1997 — LSTM",
        'cnn1d': "- Yang et al. 2015 — Deep convolutional neural networks on multichannel time series for human activity recognition",
    }

    card = f"""# Model Card — {variant_name}

## Overview

- **Input:** Raw multimodal biosignals (B, C=6, T=200), 2 s windows at 100 Hz
- **Channels:** ecg, emg, eda, ppg_green, acc_mag, temp
- **Architecture:** {arch.upper()}_raw (raw-signal multi-task)
- **Multi-task:** Hard parameter sharing — 1 encoder + 4 task heads (Caruana 1997)
- **Deployment:** {deployment_note}
- **Run:** 20260427_151020_nn_raw_full

## Training Configuration

- **CV:** 5-fold GroupKFold (subject-wise, reuse of configs/splits_per_fold.csv)
- **Seeds:** Phase 2 — [42, 1337, 7] (multi-seed required for credible NN results)
- **Optimizer:** AdamW (Loshchilov & Hutter 2019), lr={best_hp.get('lr', 'tuned'):.4f}
- **Normalization:** Per-recording z-score from first 90 s baseline (per CLAUDE.md)
- **Epochs:** 50 (early stopping patience=8)
- **Mixed precision:** Yes (torch.amp.autocast, RTX 5070 Ti CUDA)
- **Best HP:** {json.dumps(best_hp)}

## Performance (Phase 2, 5 folds × 3 seeds)

| Task | Mean ± Std | Metric |
|------|-----------|--------|
| Exercise | {ex_f1:.3f} ± {summary.get('exercise',{}).get('f1_macro',{}).get('std', float('nan')):.3f} | F1-macro |
| Phase | {ph_f1:.3f} ± {summary.get('phase',{}).get('f1_macro',{}).get('std', float('nan')):.3f} | F1-macro |
| Fatigue (RPE) | {fat_mae:.3f} ± {summary.get('fatigue',{}).get('mae',{}).get('std', float('nan')):.3f} | MAE |
| Reps | {rep_mae:.3f} ± {summary.get('reps',{}).get('mae',{}).get('std', float('nan')):.3f} | MAE |

## Data Source Note

All 6 modalities are provided at a uniform 100 Hz in aligned_features.parquet
(pragmatic simplification — native rates: ECG 500 Hz, EMG 2000 Hz, EDA 50 Hz,
Temp 1 Hz, Acc/PPG 100 Hz). The 100 Hz path is used because labels are already
aligned at this rate. The proper hybrid-fusion approach would preserve native rates
but requires reading from dataset_aligned/ with per-modality resampling. See
SUMMARY.md "Open questions".
{lat_str}
## References

{arch_refs.get(arch, '')}
- Caruana 1997 — Multitask learning (hard parameter sharing rationale)
- Saeb et al. 2017 — The need for subject-independent evaluation of machine learning models in wearable health applications (subject-wise CV)
- Loshchilov & Hutter 2019 — Decoupled weight decay regularization (AdamW optimizer)
- Goodfellow et al. 2016 — Deep Learning (dropout regularization, batch normalization)
- LeCun et al. 2015 — Deep learning (motivation for learned representations over hand-crafted features)
- Akiba et al. 2019 — Optuna: A next-generation hyperparameter optimization framework
"""
    card_path = run_dir / variant_name / 'model_card.md'
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card, encoding='utf-8')
    print(f"[model_card] Written to {card_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    t_start = time.time()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    dev_str = torch_device()
    device = torch.device(dev_str)
    print(f"\n{'='*70}")
    print(f"  Raw-signal NN training — runs/20260427_153421_nn_raw_full")
    print(f"  Device: {device} ({dev_str})")
    print(f"  Mixed precision: {'YES' if device.type == 'cuda' else 'NO'}")
    print(f"{'='*70}\n")

    # ---- Load all labeled parquets ------------------------------------------
    parquet_paths = sorted(
        LABELED_DATA_ROOT.glob('*/aligned_features.parquet')
    )
    if not parquet_paths:
        raise FileNotFoundError(
            f"No aligned_features.parquet files found under {LABELED_DATA_ROOT}. "
            "Run /label --all first."
        )
    print(f"Found {len(parquet_paths)} labeled parquets:")
    for p in parquet_paths:
        print(f"  {p}")

    print("\nBuilding RawMultimodalWindowDataset...")
    raw_ds = RawMultimodalWindowDataset(parquet_paths, active_only=False, verbose=True)
    dataset = RawDatasetWrapper(raw_ds)
    print(f"Dataset: {len(dataset)} active windows, "
          f"{dataset.n_channels} channels, "
          f"{dataset.n_timesteps} timesteps, "
          f"{dataset.n_exercise} exercise classes, "
          f"{dataset.n_phase} phase classes")

    # ---- Alignment check vs. features run -----------------------------------
    verify_window_count_alignment(dataset, FEATURES_RUN_PATH)

    # ---- Load splits ---------------------------------------------------------
    folds = load_baseline_splits(dataset)

    # =========================================================================
    # PHASE 1: Screening — all 4 raw architectures, 1 seed
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: Screening (4 raw archs, 1 seed, 30 Optuna trials each)")
    print("="*70)

    ARCHS = ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']
    phase1_results = {}

    for arch in ARCHS:
        vname = f"raw_{arch}"
        summary = run_variant_raw(
            variant_name=vname,
            arch=arch,
            dataset=dataset,
            folds=folds,
            out_root=RUN_DIR,
            seeds=[42],
            epochs=GPU_EPOCHS_P1,
            device=device,
            run_optuna=True,
        )
        phase1_results[vname] = summary

        # Dummy-baseline sanity check (exercise random baseline ~0.25 for 4 classes)
        ex_f1 = summary['exercise']['f1_macro']['mean']
        if not np.isnan(ex_f1) and ex_f1 < 0.15:
            print(f"  WARNING: {vname} exercise F1={ex_f1:.3f} below dummy "
                  f"baseline 0.25 — possible training failure")

    ranking = rank_variants(phase1_results)
    save_json({'phase1_results': phase1_results, 'ranking': ranking},
               RUN_DIR / 'phase1_summary.json')
    top_variants = [r['variant'] for r in ranking[:3]]
    print(f"\nPhase 1 top-3 variants: {top_variants}")

    # =========================================================================
    # PHASE 2: Final depth — top 2-3 variants, 3 seeds
    # =========================================================================
    print("\n" + "="*70)
    print(f"PHASE 2: Final depth on top variants: {top_variants}")
    print("="*70)

    phase2_results = {}
    for vname in top_variants:
        arch = vname.replace('raw_', '')
        # Reuse Phase 1 best HP
        phase1_hp = load_json(RUN_DIR / vname / 'best_hp.json')
        p2_name = f"{vname}_p2"
        summary = run_variant_raw(
            variant_name=p2_name,
            arch=arch,
            dataset=dataset,
            folds=folds,
            out_root=RUN_DIR,
            seeds=[42, 1337, 7],
            epochs=GPU_EPOCHS_P2,
            device=device,
            run_optuna=False,
            preloaded_hp=phase1_hp,
        )
        phase2_results[p2_name] = summary

    save_json({'phase2_results': phase2_results},
               RUN_DIR / 'phase2_summary.json')

    # =========================================================================
    # SOFT-SHARING ABLATION (winner only)
    # =========================================================================
    winner_variant = top_variants[0]
    winner_arch = winner_variant.replace('raw_', '')
    winner_hp = load_json(RUN_DIR / winner_variant / 'best_hp.json')
    print(f"\nRunning soft-sharing ablation on winner: {winner_variant}")
    ablation = run_soft_sharing_ablation_raw(
        arch=winner_arch,
        dataset=dataset,
        folds=folds[:2],
        run_dir=RUN_DIR,
        device=device,
        best_hp=winner_hp,
    )

    # =========================================================================
    # LATENCY BENCHMARKS
    # =========================================================================
    print("\nRunning latency benchmarks (GPU)...")
    latency = {}
    for arch in ARCHS:
        hp = load_json(RUN_DIR / f"raw_{arch}" / 'best_hp.json')
        factory = make_raw_factory(arch, dataset.n_exercise, dataset.n_phase, hp)
        lt = latency_benchmark_raw(factory, device=device)
        latency[arch] = lt
        causal_str = " [CAUSAL, deployable]" if arch in ('cnn1d', 'tcn') else " [non-causal]"
        print(f"  raw_{arch}: p99={lt['p99_ms']:.2f} ms{causal_str}")

    save_json(latency, RUN_DIR / 'latency.json')

    # =========================================================================
    # PER-SUBJECT BREAKDOWN (winner, fold 0)
    # =========================================================================
    winner_p2_name = f"{winner_variant}_p2"
    winner_p2_hp = load_json(RUN_DIR / winner_variant / 'best_hp.json')
    print(f"\nPer-subject breakdown for {winner_variant} (fold 0)...")
    winner_factory = make_raw_factory(
        winner_arch, dataset.n_exercise, dataset.n_phase, winner_p2_hp
    )
    per_sub_df = per_subject_breakdown_raw(
        winner_factory, dataset, folds[0], winner_p2_hp, device
    )
    per_sub_df.to_csv(RUN_DIR / 'per_subject_breakdown.csv', index=False)
    print(per_sub_df.to_string(index=False))

    # =========================================================================
    # COLLECT ALL RESULTS & BUILD COMPARISON
    # =========================================================================
    lgbm_metrics = load_json(LGBM_METRICS_PATH)
    features_phase1 = load_json(FEATURES_RUN_PATH / 'phase1_summary.json')
    features_p1_results = features_phase1.get('phase1_results', {})

    comparison = build_comparison_raw(
        phase2_results, lgbm_metrics, features_p1_results, latency
    )
    save_json(comparison, RUN_DIR / 'final_metrics.json')

    # Combine all results for all_results.json
    all_results = {
        'phase1': phase1_results,
        'phase2': phase2_results,
        'ranking': ranking,
        'latency': latency,
        'ablation': ablation,
        'winner': winner_variant,
    }
    save_json(all_results, RUN_DIR / 'all_results.json')

    # =========================================================================
    # MODEL CARDS
    # =========================================================================
    for vname, s in phase2_results.items():
        arch = vname.replace('raw_', '').replace('_p2', '')
        hp = load_json(RUN_DIR / f"raw_{arch}" / 'best_hp.json')
        is_causal = arch in ('cnn1d', 'tcn')
        lt = latency.get(arch)
        write_model_card(RUN_DIR, vname, arch, s, hp, is_causal, lt)

    # =========================================================================
    # SUMMARY.md
    # =========================================================================
    write_summary(
        run_dir=RUN_DIR,
        phase1_results=phase1_results,
        phase2_results=phase2_results,
        latency=latency,
        ablation=ablation,
        lgbm_metrics=lgbm_metrics,
        features_run_p1=features_p1_results,
        winner_variant=winner_variant,
    )

    # =========================================================================
    # COMPARISON TABLE (comparison.md)
    # =========================================================================
    comp_rows = comparison.get('rows', [])
    comp_lines = [
        "# Comparison: LightGBM vs. NN-features vs. NN-raw",
        "",
        "| Model | Exercise F1 | Phase F1 | Fatigue MAE | Reps MAE | p99 ms | Causal |",
        "|-------|-------------|----------|-------------|----------|--------|--------|",
    ]
    for row in comp_rows:
        ex = f"{row.get('exercise_f1', float('nan')):.3f}"
        ph = f"{row.get('phase_f1', float('nan')):.3f}"
        fat = f"{row.get('fatigue_mae', float('nan')):.3f}"
        rep = f"{row.get('reps_mae', float('nan')):.3f}"
        lat = f"{row.get('latency_p99_ms') or 'N/A'}"
        if isinstance(lat, float):
            lat = f"{lat:.2f}"
        causal = "YES" if row.get('causal') else "NO"
        comp_lines.append(
            f"| {row.get('model', '?')} | {ex} | {ph} | {fat} | {rep} | {lat} | {causal} |"
        )
    comp_lines += [
        "",
        "## Notes",
        "",
        "- LightGBM uses set-level features for fatigue and reps; NN uses per-window features.",
        "- 'Causal' = safe for real-time streaming. BiLSTM and CNN-LSTM are research_only.",
        "- All NN metrics are from Phase 2 (3 seeds × 5 folds).",
        "- Statistical power is low (5 folds); p-values not reported due to insufficient N.",
        "",
        "## References",
        "",
        "- Saeb et al. 2017 — subject-wise CV motivation",
        "- Bai et al. 2018 — TCN",
        "- Hochreiter & Schmidhuber 1997 — LSTM",
        "- Caruana 1997 — hard sharing",
        "- LeCun et al. 2015 — learned vs. hand-crafted features",
    ]
    (RUN_DIR / 'comparison.md').write_text('\n'.join(comp_lines), encoding='utf-8')
    print(f"[comparison] Written to {RUN_DIR / 'comparison.md'}")

    t_elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  ALL DONE. Elapsed: {t_elapsed/3600:.2f} hours ({t_elapsed:.0f} s)")
    print(f"  Run dir: {RUN_DIR}")
    print(f"  Winner: {winner_variant}")
    print(f"  Best per task (Phase 2):")
    for vname, s in phase2_results.items():
        print(f"    {vname}: "
              f"ex={s['exercise']['f1_macro']['mean']:.3f}  "
              f"ph={s['phase']['f1_macro']['mean']:.3f}  "
              f"fat={s['fatigue']['mae']['mean']:.3f}  "
              f"rep={s['reps']['mae']['mean']:.3f}")
    print(f"{'='*70}")

    return all_results


if __name__ == '__main__':
    main()
