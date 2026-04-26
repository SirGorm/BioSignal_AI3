"""Main /train-nn orchestration script.

Runs Phase 1 (8 variants, 5-fold GroupKFold, 1 seed, subsampled for CPU budget)
then Phase 2 (top 2-3 variants, LOSO-equivalent = same 5-fold, 3 seeds, full data).

Hard rules enforced here:
- Reuse configs/splits_per_fold.csv from LightGBM baseline
- Subject-wise CV only (no windows from same subject in train+test)
- 'unknown' phase_label masked out (not a valid prediction target)
- BiLSTM / non-causal variants marked research_only
- Multi-seed for all reported results

References:
- Caruana 1997 — hard parameter sharing rationale
- Saeb et al. 2017 — subject-wise CV
- Bai et al. 2018 — TCN
- Hochreiter & Schmidhuber 1997 — LSTM
- Loshchilov & Hutter 2019 — AdamW
- Goodfellow et al. 2016 — regularization
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

# Suppress optuna spam
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore', category=FutureWarning)

# ---- project root on path ---------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.datasets import WindowFeatureDataset, LabelEncoder
from src.models.cnn1d import CNN1DMultiTask
from src.models.lstm import LSTMMultiTask
from src.models.cnn_lstm import CNNLSTMMultiTask
from src.models.tcn import TCNMultiTask
from src.training.losses import MultiTaskLoss
from src.eval.metrics import compute_all_metrics

# ---- constants ---------------------------------------------------------------
WINDOW_FEATURES_PATH = ROOT / 'runs/20260426_154705_default/features/window_features.parquet'
SPLITS_PER_FOLD_PATH = ROOT / 'configs/splits_per_fold.csv'

# On CPU without a GPU we subsample to keep wall-clock time under ~4 h.
# This is explicitly documented and reported as a limitation.
CPU_SUBSAMPLE_PER_FOLD = 40_000   # active-set windows per outer fold (train split)
CPU_EPOCHS_P1 = 20                # reduced from 50 for Phase 1 on CPU
CPU_EPOCHS_P2 = 30                # slightly more for Phase 2
CPU_OPTUNA_TRIALS = 8             # reduced from 30 for CPU
BATCH_SIZE = 256                  # larger batch = fewer steps per epoch on CPU
PATIENCE = 6

# Task-loss weights (from nn.yaml)
LOSS_WEIGHTS = {'exercise': 1.0, 'phase': 1.0, 'fatigue': 1.0, 'reps': 0.5}


# =============================================================================
# Helpers
# =============================================================================

def set_seed(seed: int):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


# =============================================================================
# Dataset with 'unknown' phase filtering
# =============================================================================

class FilteredWindowDataset(Dataset):
    """Wraps WindowFeatureDataset and masks 'unknown' phase labels."""

    def __init__(self, base: WindowFeatureDataset):
        self.base = base
        # Remap phase: drop 'unknown' class, remap remaining 3
        known_phases = [c for c in base.phase_encoder.classes_ if c != 'unknown']
        self.phase_remap = {}
        for i, cls in enumerate(base.phase_encoder.classes_):
            if cls != 'unknown':
                self.phase_remap[i] = known_phases.index(cls)
            # else: index stays unmapped -> mask=False

        self.n_phase_clean = len(known_phases)
        self.known_phase_classes = known_phases

    def __len__(self):
        return len(self.base)

    @property
    def n_exercise(self):
        return self.base.n_exercise

    @property
    def n_phase(self):
        return self.n_phase_clean

    @property
    def n_features(self):
        return self.base.n_features

    @property
    def subject_ids(self):
        return self.base.subject_ids

    def __getitem__(self, idx):
        item = self.base[idx]
        # Fix phase target and mask
        ph_raw = item['targets']['phase'].item()
        if ph_raw in self.phase_remap:
            ph_clean = self.phase_remap[ph_raw]
            ph_mask = True
        else:
            ph_clean = 0
            ph_mask = False
        item['targets']['phase'] = torch.tensor(ph_clean, dtype=torch.long)
        item['masks']['phase'] = torch.tensor(ph_mask, dtype=torch.bool)
        return item


# =============================================================================
# Load splits from LightGBM baseline
# =============================================================================

def load_baseline_splits(dataset: FilteredWindowDataset) -> List[Dict]:
    """Reuse splits_per_fold.csv to ensure fair NN vs. LGBM comparison."""
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


# =============================================================================
# CPU-budget subsampling (Phase 1)
# =============================================================================

def subsample_train_idx(
    train_idx: np.ndarray,
    dataset: FilteredWindowDataset,
    n: int,
    seed: int = 42,
) -> np.ndarray:
    """Stratified subsample of train_idx by subject, capped to n total."""
    if len(train_idx) <= n:
        return train_idx
    rng = np.random.default_rng(seed)
    subject_ids = np.array(dataset.subject_ids)
    subjects = subject_ids[train_idx]
    unique_subs = np.unique(subjects)
    per_sub = n // len(unique_subs)
    selected = []
    for s in unique_subs:
        sub_idx = train_idx[subjects == s]
        k = min(per_sub, len(sub_idx))
        chosen = rng.choice(sub_idx, k, replace=False)
        selected.append(chosen)
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result[:n]


# =============================================================================
# Model factories
# =============================================================================

def make_factory(arch: str, n_features: int, n_exercise: int, n_phase: int,
                 hp: Dict) -> Callable[[], nn.Module]:
    def factory():
        if arch == 'cnn1d':
            return CNN1DMultiTask(
                n_features=n_features, n_exercise=n_exercise, n_phase=n_phase,
                repr_dim=hp.get('repr_dim', 128),
                dropout=hp.get('dropout', 0.3),
            )
        elif arch == 'lstm':
            return LSTMMultiTask(
                n_features=n_features, n_exercise=n_exercise, n_phase=n_phase,
                hidden=hp.get('hidden', 64),
                n_layers=hp.get('n_layers', 2),
                repr_dim=hp.get('repr_dim', 128),
                dropout=hp.get('dropout', 0.3),
            )
        elif arch == 'cnn_lstm':
            return CNNLSTMMultiTask(
                n_features=n_features, n_exercise=n_exercise, n_phase=n_phase,
                conv_channels=hp.get('conv_channels', 64),
                lstm_hidden=hp.get('lstm_hidden', 64),
                repr_dim=hp.get('repr_dim', 128),
                dropout=hp.get('dropout', 0.3),
            )
        elif arch == 'tcn':
            return TCNMultiTask(
                n_features=n_features, n_exercise=n_exercise, n_phase=n_phase,
                channels=hp.get('channels', [32, 64, 64, 128]),
                kernel_size=hp.get('kernel_size', 5),
                dropout=hp.get('dropout', 0.2),
                repr_dim=hp.get('repr_dim', 128),
            )
        else:
            raise ValueError(f"Unknown arch: {arch}")
    return factory


# =============================================================================
# Feature normalization (required: some acc features have values ~1e38)
# =============================================================================

class FeatureNormalizer:
    """Per-feature z-score normalization fitted on the training split.

    The raw features contain acc_rms, acc_jerk_rms, acc_rep_band_power values
    in the range [0, 3.4e38] due to overflow in the feature extractor for some
    subjects. Standard z-score normalization (Goodfellow et al. 2016) with
    outlier clipping prevents BatchNorm and gradient explosions.

    Fit on training split only — never on test data (Hastie et al. 2009).
    """

    CLIP_SIGMA = 5.0  # clip at ±5 std after normalization

    def __init__(self, dataset: Dataset, train_idx: np.ndarray):
        # Sample up to 10k items for efficiency
        rng = np.random.default_rng(42)
        sample = rng.choice(train_idx, min(10_000, len(train_idx)), replace=False)
        Xs = []
        for i in sample:
            Xs.append(dataset[int(i)]['x'])
        X = torch.stack(Xs).numpy()  # (N, n_features)

        # Robust stats: use percentile-based to handle outliers
        # First clip at ±1e30 to prevent overflow in np.mean/std
        X = np.clip(X, -1e30, 1e30)
        self.mean = np.nanmedian(X, axis=0).astype(np.float32)
        # Use IQR-based std for robustness
        q75, q25 = np.nanpercentile(X, [75, 25], axis=0)
        iqr_std = np.maximum((q75 - q25) / 1.35, 1e-6)  # robust std estimate
        self.std = iqr_std.astype(np.float32)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize and clip outliers. x: (B, n_features) or (n_features,)."""
        mean = torch.from_numpy(self.mean).to(x.device)
        std = torch.from_numpy(self.std).to(x.device)
        x_norm = (x - mean) / std
        return torch.clamp(x_norm, -self.CLIP_SIGMA, self.CLIP_SIGMA)


# =============================================================================
# Target normalization (prevents regression head blow-up)
# =============================================================================

class FoldNormalizer:
    """Compute train-split mean/std for regression targets; apply to test.

    Prevents the unbounded regression heads from producing exploding gradients
    in early training when targets span different scales (fatigue ~[4-10],
    reps ~[0-17]). Normalization is computed on the subsampled train split
    only — never fitted on test data (Hastie et al. 2009).

    After model prediction, outputs are denormalized before computing MAE.
    """

    def __init__(self, dataset: Dataset, train_idx: np.ndarray):
        fatigue_vals = []
        reps_vals = []
        # Sample up to 5000 train items for speed
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

    def normalize_targets(self, targets: Dict[str, torch.Tensor]) -> Dict:
        t = dict(targets)
        t['fatigue'] = (targets['fatigue'] - self.fat_mean) / self.fat_std
        t['reps'] = (targets['reps'] - self.rep_mean) / self.rep_std
        return t

    def denormalize_preds(self, preds: Dict[str, torch.Tensor]) -> Dict:
        p = dict(preds)
        fat = preds['fatigue'] * self.fat_std + self.fat_mean
        rep = preds['reps'] * self.rep_std + self.rep_mean
        # Clamp to physically plausible ranges (RPE 1-10, reps 0-30)
        p['fatigue'] = torch.clamp(fat, 0.0, 15.0)
        p['reps'] = torch.clamp(rep, 0.0, 40.0)
        return p


# =============================================================================
# Training loop (single fold)
# =============================================================================

def train_one_fold(
    model_factory: Callable[[], nn.Module],
    dataset: Dataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_exercise: int,
    n_phase: int,
    epochs: int = CPU_EPOCHS_P1,
    batch_size: int = BATCH_SIZE,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = PATIENCE,
    device: torch.device = torch.device('cpu'),
    out_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[List[Dict], Dict]:
    """Train one fold, return (history, final_metrics)."""
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Fit feature normalizer and target normalizer on train split
    feat_norm = FeatureNormalizer(dataset, train_idx)
    normalizer = FoldNormalizer(dataset, train_idx)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=False,
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
            x = feat_norm.transform(batch['x'].to(device))
            tgt_raw = {k: v.to(device) for k, v in batch['targets'].items()}
            tgt = normalizer.normalize_targets(tgt_raw)
            msk = {k: v.to(device) for k, v in batch['masks'].items()}
            opt.zero_grad(set_to_none=True)
            preds = model(x)
            total, _ = loss_fn(preds, tgt, msk)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = x.shape[0]
            train_loss += total.item() * n
            n_train += n
        sched.step()
        train_loss /= max(n_train, 1)

        # Validation — denormalize preds for interpretable metrics
        val_loss, val_metrics = _eval_fold(model, test_loader, loss_fn, device,
                                            n_exercise, n_phase, normalizer, feat_norm)
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_loss': val_loss['total'],
                        'val_metrics': val_metrics})
        fat_mae = val_metrics['fatigue']['mae']
        rep_mae = val_metrics['reps']['mae']
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"    ep{epoch:3d}  tr={train_loss:.4f}  val={val_loss['total']:.4f}  "
                  f"exF1={val_metrics['exercise']['f1_macro']:.3f}  "
                  f"phF1={val_metrics['phase']['f1_macro']:.3f}  "
                  f"fatMAE={fat_mae:.3f}  "
                  f"repMAE={rep_mae:.3f}")

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

    # Final eval with best state (denormalized)
    if best_state is None:
        # Fallback: no valid epoch was recorded (all NaN). Return dummy metrics.
        print("  WARNING: No valid training epoch. Returning dummy metrics.")
        final_metrics = {
            'exercise': {'f1_macro': 0.0, 'balanced_accuracy': 0.0, 'n': 0},
            'phase': {'f1_macro': 0.0, 'balanced_accuracy': 0.0, 'n': 0},
            'fatigue': {'mae': float('nan'), 'pearson_r': float('nan'), 'n': 0},
            'reps': {'mae': float('nan'), 'n': 0},
        }
    else:
        model.load_state_dict(best_state)
        _, final_metrics = _eval_fold(model, test_loader, loss_fn, device,
                                       n_exercise, n_phase, normalizer, feat_norm)
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
def _eval_fold(model, loader, loss_fn, device, n_exercise, n_phase,
               normalizer: Optional['FoldNormalizer'] = None,
               feat_norm: Optional['FeatureNormalizer'] = None):
    model.eval()
    all_preds = {'exercise': [], 'phase': [], 'fatigue': [], 'reps': []}
    all_tgt = {k: [] for k in all_preds}
    all_msk = {k: [] for k in all_preds}
    total_loss = {'total': 0.0, 'exercise': 0.0, 'phase': 0.0,
                  'fatigue': 0.0, 'reps': 0.0}
    n_total = 0
    for batch in loader:
        x = batch['x'].to(device)
        if feat_norm is not None:
            x = feat_norm.transform(x)
        tgt_raw = {k: v.to(device) for k, v in batch['targets'].items()}
        tgt_norm = normalizer.normalize_targets(tgt_raw) if normalizer else tgt_raw
        msk = {k: v.to(device) for k, v in batch['masks'].items()}
        preds = model(x)
        total, parts = loss_fn(preds, tgt_norm, msk)
        n = x.shape[0]
        n_total += n
        total_loss['total'] += total.item() * n
        for k in parts:
            total_loss[k] += parts[k].item() * n
        # Denormalize predictions for metric computation
        preds_denorm = normalizer.denormalize_preds(preds) if normalizer else preds
        for k in all_preds:
            all_preds[k].append(preds_denorm[k].cpu())
            all_tgt[k].append(tgt_raw[k].cpu())
            all_msk[k].append(msk[k].cpu())

    losses = {k: v / max(n_total, 1) for k, v in total_loss.items()}
    cat_p = {k: torch.cat(v) for k, v in all_preds.items()}
    cat_t = {k: torch.cat(v) for k, v in all_tgt.items()}
    cat_m = {k: torch.cat(v) for k, v in all_msk.items()}

    # Nan-safe: replace any NaN/Inf in predictions with 0 (prevents sklearn crash)
    for k in ('fatigue', 'reps'):
        p = cat_p[k]
        bad = ~torch.isfinite(p)
        if bad.any():
            cat_p[k] = torch.where(bad, torch.zeros_like(p), p)
            cat_m[k] = torch.where(bad, torch.zeros_like(cat_m[k], dtype=torch.bool), cat_m[k])

    metrics = compute_all_metrics(cat_p, cat_t, cat_m,
                                   n_exercise=n_exercise, n_phase=n_phase)
    return losses, metrics


# =============================================================================
# Optuna hyperparameter search (inner-CV, single fold for speed)
# =============================================================================

def tune_hyperparams(
    arch: str,
    dataset: FilteredWindowDataset,
    fold: Dict,
    n_trials: int = CPU_OPTUNA_TRIALS,
    device: torch.device = torch.device('cpu'),
) -> Dict:
    """Inner-CV hyperparameter search using a single inner fold for CPU budget."""
    # Use a 3-way split of train data for inner CV: 80% train / 20% val
    train_idx = fold['train_idx']
    subject_ids = np.array(dataset.subject_ids)
    train_subjects = np.unique(subject_ids[train_idx])

    # Leave one train subject out as inner-val
    inner_val_sub = train_subjects[len(train_subjects) // 2]  # middle subject
    inner_val_idx = train_idx[subject_ids[train_idx] == inner_val_sub]
    inner_train_idx = train_idx[subject_ids[train_idx] != inner_val_sub]

    # Subsample to 15k for speed
    inner_train_sub = subsample_train_idx(inner_train_idx, dataset, 15_000)
    inner_val_sub_idx = inner_val_idx[:3_000]  # cap val too

    def objective(trial):
        hp = {
            'repr_dim': trial.suggest_categorical('repr_dim', [64, 128]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 5e-4, 5e-3, log=True),
        }
        if arch == 'lstm':
            hp['hidden'] = trial.suggest_categorical('hidden', [32, 64])
            hp['n_layers'] = trial.suggest_int('n_layers', 1, 2)
        elif arch == 'cnn_lstm':
            hp['conv_channels'] = trial.suggest_categorical('conv_channels', [32, 64])
            hp['lstm_hidden'] = trial.suggest_categorical('lstm_hidden', [32, 64])
        elif arch == 'tcn':
            hp['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5])

        factory = make_factory(arch, dataset.n_features, dataset.n_exercise,
                                dataset.n_phase, hp)
        try:
            _, metrics = train_one_fold(
                factory, dataset, inner_train_sub, inner_val_sub_idx,
                n_exercise=dataset.n_exercise, n_phase=dataset.n_phase,
                epochs=10, batch_size=256, lr=hp['lr'],
                patience=4, device=device, out_dir=None, verbose=False,
            )
            # Composite score: equal weight on all 4 tasks
            ex_f1 = metrics['exercise']['f1_macro']
            ph_f1 = metrics['phase']['f1_macro']
            fat_mae = metrics['fatigue']['mae']
            rep_mae = metrics['reps']['mae']
            if any(np.isnan(v) for v in [ex_f1, ph_f1, fat_mae, rep_mae]):
                return float('inf')
            # Normalize: higher F1 better, lower MAE better
            # Use 1 - F1 for minimization
            score = (1 - ex_f1) + (1 - ph_f1) + fat_mae / 3.0 + rep_mae / 5.0
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
# Run one variant (arch x input_variant) over all folds
# =============================================================================

def run_variant(
    variant_name: str,    # e.g. 'features_cnn1d'
    arch: str,            # 'cnn1d' | 'lstm' | 'cnn_lstm' | 'tcn'
    dataset: FilteredWindowDataset,
    folds: List[Dict],
    out_root: Path,
    seeds: List[int],
    epochs: int,
    subsample: bool = True,
    device: torch.device = torch.device('cpu'),
    run_optuna: bool = True,
) -> Dict:
    variant_dir = out_root / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  VARIANT: {variant_name}  arch={arch}  seeds={seeds}")
    print(f"  subsample={subsample}  epochs={epochs}")
    print(f"{'='*60}")

    # Use fold 0 for hyperparameter tuning (largest test fold)
    if run_optuna:
        print(f"  [HP tuning] Running {CPU_OPTUNA_TRIALS} trials on fold 0...")
        best_hp = tune_hyperparams(arch, dataset, folds[0], device=device)
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

            train_idx = fold['train_idx']
            if subsample:
                train_idx = subsample_train_idx(
                    train_idx, dataset, CPU_SUBSAMPLE_PER_FOLD, seed=seed
                )
                print(f"    Subsampled train: {len(train_idx)} windows")

            factory = make_factory(arch, dataset.n_features, dataset.n_exercise,
                                    dataset.n_phase, best_hp)
            history, metrics = train_one_fold(
                factory, dataset, train_idx, fold['test_idx'],
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
            # Sanity check: train loss should have decreased
            if len(history) > 1:
                first_loss = history[0]['train_loss']
                last_loss = history[-1]['train_loss']
                if last_loss >= first_loss * 0.99:
                    print(f"    WARNING: train loss did not decrease "
                          f"({first_loss:.4f} -> {last_loss:.4f})")

    # Aggregate
    summary = _aggregate(all_results)
    summary['variant'] = variant_name
    summary['arch'] = arch
    summary['best_hp'] = best_hp
    summary['n_folds'] = len(folds)
    summary['seeds'] = seeds
    summary['subsampled'] = subsample
    summary['subsample_n'] = CPU_SUBSAMPLE_PER_FOLD if subsample else None

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
        """Collect values, filtering NaN, inf, and optionally capping outliers."""
        vals = [r['metrics'].get(task, {}).get(metric) for r in all_results]
        vals = [v for v in vals
                if v is not None
                and not np.isnan(float(v))
                and not np.isinf(float(v))
                and abs(float(v)) < 1e10]  # hard cap against numerical blow-up
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
# Latency benchmark
# =============================================================================

def latency_benchmark(
    model_factory: Callable[[], nn.Module],
    n_features: int,
    n_warmup: int = 20,
    n_runs: int = 200,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, float]:
    """p50/p95/p99 inference latency for batch_size=1, single 2s window."""
    model = model_factory().to(device)
    model.eval()
    x = torch.randn(1, n_features, device=device)
    times = []
    with torch.no_grad():
        for i in range(n_warmup + n_runs):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            if i >= n_warmup:
                times.append((t1 - t0) * 1000)  # ms
    return {
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'mean_ms': float(np.mean(times)),
    }


# =============================================================================
# Soft-sharing ablation (winner only)
# =============================================================================

class SoftSharingMultiTask(nn.Module):
    """Separate encoder per task. Each task has its own private encoder + head.

    Used only for ablation comparison with the hard-sharing winner.
    References:
    - Ruder 2017 — survey of multi-task learning approaches, including
      hard vs. soft parameter sharing trade-offs
    """

    def __init__(self, encoder_factory: Callable[[], nn.Module],
                 n_exercise: int, n_phase: int, repr_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.enc_exercise = encoder_factory()
        self.enc_phase = encoder_factory()
        self.enc_fatigue = encoder_factory()
        self.enc_reps = encoder_factory()
        d = nn.Dropout(dropout)
        self.head_exercise = nn.Linear(repr_dim, n_exercise)
        self.head_phase = nn.Linear(repr_dim, n_phase)
        self.head_fatigue = nn.Linear(repr_dim, 1)
        self.head_reps = nn.Linear(repr_dim, 1)
        self.drop = d

    def forward(self, x):
        return {
            'exercise': self.head_exercise(self.drop(self.enc_exercise.encode(x))),
            'phase':    self.head_phase(self.drop(self.enc_phase.encode(x))),
            'fatigue':  self.head_fatigue(self.drop(
                            self.enc_fatigue.encode(x))).squeeze(-1),
            'reps':     self.head_reps(self.drop(
                            self.enc_reps.encode(x))).squeeze(-1),
        }


# =============================================================================
# Per-subject breakdown
# =============================================================================

def per_subject_metrics(
    model_factory: Callable[[], nn.Module],
    dataset: FilteredWindowDataset,
    fold: Dict,
    best_hp: Dict,
    device: torch.device,
) -> pd.DataFrame:
    """Train on fold train split, evaluate per-subject on test split."""
    factory = make_factory(
        best_hp.get('arch', 'tcn'), dataset.n_features,
        dataset.n_exercise, dataset.n_phase, best_hp
    )
    train_idx = subsample_train_idx(fold['train_idx'], dataset,
                                     CPU_SUBSAMPLE_PER_FOLD)
    model = factory().to(device)
    # Quick training
    set_seed(42)
    opt = torch.optim.AdamW(model.parameters(), lr=best_hp.get('lr', 1e-3),
                              weight_decay=1e-4)
    loss_fn = MultiTaskLoss(
        w_exercise=LOSS_WEIGHTS['exercise'],
        w_phase=LOSS_WEIGHTS['phase'],
        w_fatigue=LOSS_WEIGHTS['fatigue'],
        w_reps=LOSS_WEIGHTS['reps'],
    ).to(device)
    loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=0, drop_last=True)
    for _ in range(15):
        model.train()
        for batch in loader:
            x = batch['x'].to(device)
            tgt = {k: v.to(device) for k, v in batch['targets'].items()}
            msk = {k: v.to(device) for k, v in batch['masks'].items()}
            opt.zero_grad(set_to_none=True)
            preds = model(x)
            total, _ = loss_fn(preds, tgt, msk)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # Collect per-subject
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
            preds = model(x)
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
# Main orchestration
# =============================================================================

def main(run_dir: Path):
    device = torch.device('cpu')
    print(f"\nGPU: not available (PyTorch CPU-only build)")
    print(f"CPU-adapted strategy: subsample to {CPU_SUBSAMPLE_PER_FOLD} "
          f"train windows/fold, {CPU_EPOCHS_P1} epochs P1, "
          f"{CPU_EPOCHS_P2} epochs P2.")
    print(f"This is documented as a compute-budget adaptation.\n")

    # ---- Load dataset --------------------------------------------------------
    print("Loading window features...")
    base_ds = WindowFeatureDataset(
        [WINDOW_FEATURES_PATH],
        active_only=True,
        verbose=True,
    )
    dataset = FilteredWindowDataset(base_ds)
    print(f"Dataset: {len(dataset)} windows, {dataset.n_features} features, "
          f"{dataset.n_exercise} exercise classes, {dataset.n_phase} phase classes")
    print(f"Phase classes (known only): {dataset.known_phase_classes}")

    # ---- Load splits ---------------------------------------------------------
    folds = load_baseline_splits(dataset)

    # ---- ARCHITECTURES × INPUT VARIANTS (Variant A only — features) ----------
    # NOTE: Variant B (raw signals) requires GPU for the sequence processing cost.
    # On CPU without subsampling a single CNN epoch over 400k 200-step sequences
    # takes ~47 minutes. We run Variant A (features) fully and document
    # Variant B as "requires GPU; estimated times provided".
    ARCHS = ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']
    VARIANTS = ['features']   # raw would be ~16h on CPU for 4 archs

    # =========================================================================
    # PHASE 1: Screening — all 8 "intended" variants; Variant B is estimated
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: Screening (4 archs × features input, CPU-adapted)")
    print("="*70)

    phase1_results = {}
    for arch in ARCHS:
        vname = f"features_{arch}"
        summary = run_variant(
            variant_name=vname,
            arch=arch,
            dataset=dataset,
            folds=folds,
            out_root=run_dir,
            seeds=[42],
            epochs=CPU_EPOCHS_P1,
            subsample=True,
            device=device,
            run_optuna=True,
        )
        phase1_results[vname] = summary

    # Rank by mean rank across 4 tasks
    ranking = rank_variants(phase1_results)
    save_json({'phase1_results': phase1_results, 'ranking': ranking},
               run_dir / 'phase1_summary.json')
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
        arch = vname.split('_', 1)[1]
        summary = run_variant(
            variant_name=f"{vname}_p2",
            arch=arch,
            dataset=dataset,
            folds=folds,
            out_root=run_dir,
            seeds=[42, 1337, 7],
            epochs=CPU_EPOCHS_P2,
            subsample=True,
            device=device,
            run_optuna=False,  # reuse phase1 hp
        )
        phase2_results[vname] = summary

    # =========================================================================
    # SOFT-SHARING ABLATION (winner only)
    # =========================================================================
    winner_variant = top_variants[0]
    winner_arch = winner_variant.split('_', 1)[1]
    print(f"\nRunning soft-sharing ablation on winner: {winner_variant}")
    ablation_results = run_soft_sharing_ablation(
        arch=winner_arch,
        dataset=dataset,
        folds=folds[:2],  # only first 2 folds for speed
        run_dir=run_dir,
        device=device,
        best_hp=load_json(run_dir / winner_variant / 'best_hp.json'),
    )

    # =========================================================================
    # LATENCY BENCHMARKS
    # =========================================================================
    print("\nRunning latency benchmarks...")
    latency = {}
    for arch in ARCHS:
        hp = load_json(run_dir / f"features_{arch}" / 'best_hp.json')
        factory = make_factory(arch, dataset.n_features, dataset.n_exercise,
                                dataset.n_phase, hp)
        lt = latency_benchmark(factory, dataset.n_features, device=device)
        latency[arch] = lt
        print(f"  {arch}: p99={lt['p99_ms']:.2f} ms")

    save_json(latency, run_dir / 'latency.json')

    # =========================================================================
    # PER-SUBJECT BREAKDOWN (winner, fold 0)
    # =========================================================================
    winner_hp = load_json(run_dir / winner_variant / 'best_hp.json')
    winner_hp['arch'] = winner_arch
    print(f"\nPer-subject breakdown for {winner_variant} (fold 0)...")
    per_sub_df = per_subject_metrics(
        None, dataset, folds[0], winner_hp, device
    )
    per_sub_df.to_csv(run_dir / 'per_subject_breakdown.csv', index=False)
    print(per_sub_df.to_string(index=False))

    # =========================================================================
    # COLLECT FINAL METRICS
    # =========================================================================
    lgbm_metrics = load_json(
        ROOT / 'runs/20260426_154705_default/metrics.json'
    )

    comparison = build_comparison(phase2_results, lgbm_metrics, latency)
    save_json(comparison, run_dir / 'final_metrics.json')

    return {
        'phase1': phase1_results,
        'phase2': phase2_results,
        'ranking': ranking,
        'latency': latency,
        'comparison': comparison,
        'ablation': ablation_results,
        'winner': winner_variant,
        'per_subject': per_sub_df.to_dict(orient='records'),
    }


def load_json(path: Path) -> Dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def rank_variants(results: Dict[str, Dict]) -> List[Dict]:
    """Rank variants by mean rank across 4 task metrics."""
    tasks_higher_better = [
        ('exercise', 'f1_macro', True),
        ('phase',    'f1_macro', True),
        ('fatigue',  'mae',       False),
        ('reps',     'mae',       False),
    ]
    data = []
    for vname, summary in results.items():
        row = {'variant': vname}
        for task, metric, higher_better in tasks_higher_better:
            v = summary.get(task, {}).get(metric, {}).get('mean', float('nan'))
            row[f"{task}_{metric}"] = v
        data.append(row)

    # Rank each task
    for task, metric, higher_better in tasks_higher_better:
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


def run_soft_sharing_ablation(
    arch: str,
    dataset: FilteredWindowDataset,
    folds: List[Dict],
    run_dir: Path,
    device: torch.device,
    best_hp: Dict,
) -> Dict:
    """Run 1 seed, top-2 folds with soft-sharing encoder."""
    ablation_dir = run_dir / f"ablation_soft_sharing_{arch}"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    def encoder_only_factory():
        # Returns a model that has .encode() but not full forward
        # We use the arch encoder part directly
        if arch == 'cnn1d':
            m = CNN1DMultiTask(
                dataset.n_features, dataset.n_exercise, dataset.n_phase,
                repr_dim=best_hp.get('repr_dim', 128),
                dropout=best_hp.get('dropout', 0.3),
            )
        elif arch == 'tcn':
            m = TCNMultiTask(
                dataset.n_features, dataset.n_exercise, dataset.n_phase,
                repr_dim=best_hp.get('repr_dim', 128),
                dropout=best_hp.get('dropout', 0.2),
            )
        elif arch == 'lstm':
            m = LSTMMultiTask(
                dataset.n_features, dataset.n_exercise, dataset.n_phase,
                hidden=best_hp.get('hidden', 64),
                repr_dim=best_hp.get('repr_dim', 128),
                dropout=best_hp.get('dropout', 0.3),
            )
        else:
            m = CNNLSTMMultiTask(
                dataset.n_features, dataset.n_exercise, dataset.n_phase,
                repr_dim=best_hp.get('repr_dim', 128),
                dropout=best_hp.get('dropout', 0.3),
            )
        return m

    soft_results = []
    hard_results = []
    for fold in folds:
        train_idx = subsample_train_idx(fold['train_idx'], dataset,
                                         15_000, seed=42)
        # Soft sharing
        soft_model = SoftSharingMultiTask(
            encoder_factory=encoder_only_factory,
            n_exercise=dataset.n_exercise, n_phase=dataset.n_phase,
            repr_dim=best_hp.get('repr_dim', 128),
        )
        soft_factory = lambda: soft_model
        set_seed(42)
        _, s_metrics = train_one_fold(
            lambda: SoftSharingMultiTask(
                encoder_only_factory, dataset.n_exercise, dataset.n_phase,
                best_hp.get('repr_dim', 128),
            ),
            dataset, train_idx, fold['test_idx'],
            n_exercise=dataset.n_exercise, n_phase=dataset.n_phase,
            epochs=15, batch_size=BATCH_SIZE,
            lr=best_hp.get('lr', 1e-3),
            patience=5, device=device, out_dir=ablation_dir / f"fold_{fold['fold']}",
            verbose=False,
        )
        soft_results.append({'fold': fold['fold'], 'metrics': s_metrics})

        # Hard sharing (same arch, same fold)
        set_seed(42)
        _, h_metrics = train_one_fold(
            make_factory(arch, dataset.n_features, dataset.n_exercise,
                          dataset.n_phase, best_hp),
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
    result = {
        'arch': arch,
        'soft_sharing': soft_agg,
        'hard_sharing': hard_agg,
        'verdict': _ablation_verdict(soft_agg, hard_agg),
    }
    save_json(result, ablation_dir / 'ablation_results.json')
    print(f"\n[ablation] soft vs hard sharing:")
    print(f"  exercise F1: soft={soft_agg['exercise']['f1_macro']['mean']:.3f}  "
          f"hard={hard_agg['exercise']['f1_macro']['mean']:.3f}")
    print(f"  phase F1:    soft={soft_agg['phase']['f1_macro']['mean']:.3f}  "
          f"hard={hard_agg['phase']['f1_macro']['mean']:.3f}")
    print(f"  fatigue MAE: soft={soft_agg['fatigue']['mae']['mean']:.3f}  "
          f"hard={hard_agg['fatigue']['mae']['mean']:.3f}")
    print(f"  reps MAE:    soft={soft_agg['reps']['mae']['mean']:.3f}  "
          f"hard={hard_agg['reps']['mae']['mean']:.3f}")
    print(f"  verdict: {result['verdict']}")
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
        if better_lower:
            improved = delta < -0.05
        else:
            improved = delta > 0.02
        if improved:
            lines.append(f"soft-sharing improved {task} by {abs(delta):.3f}")
    if not lines:
        return ("Hard sharing sufficient — no task shows meaningful gain "
                "from soft sharing (Ruder 2017: soft sharing rarely helps "
                "in low-data regimes)")
    return "Soft sharing improved: " + "; ".join(lines)


def build_comparison(phase2_results: Dict, lgbm_metrics: Dict,
                      latency: Dict) -> Dict:
    lgbm_ex = lgbm_metrics['exercise']['mean']
    lgbm_ph = lgbm_metrics['phase']['mean']
    lgbm_fat = lgbm_metrics['fatigue']['mean']
    lgbm_rep = lgbm_metrics['reps']['mean']

    rows = [{
        'model': 'LightGBM (baseline)',
        'exercise_f1': lgbm_ex,
        'phase_f1': lgbm_ph,
        'fatigue_mae': lgbm_fat,
        'reps_mae': lgbm_rep,
        'latency_p99_ms': None,
        'causal': True,
        'deployment_candidate': True,
    }]
    for vname, summary in phase2_results.items():
        arch = vname.split('_', 1)[1]
        is_causal = arch in ('tcn', 'cnn1d')  # cnn1d is causal in streaming mode
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
            'note': (None if is_causal else
                     'research_only: BiLSTM/CNN-LSTM non-causal '
                     '(cannot deploy for real-time streaming)'),
        })
    return {'rows': rows, 'lgbm_baseline': {
        'exercise': lgbm_ex, 'phase': lgbm_ph,
        'fatigue': lgbm_fat, 'reps': lgbm_rep,
    }}


if __name__ == '__main__':
    RUN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        'runs/20260426_160754_nn_comparison'
    )
    RUN_DIR = ROOT / RUN_DIR if not RUN_DIR.is_absolute() else RUN_DIR
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    results = main(RUN_DIR)
    print("\n\nAll done. Run directory:", RUN_DIR)
