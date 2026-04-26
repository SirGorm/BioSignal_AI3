---
name: deep-learning-training
description: Use when training neural networks (1D-CNN, LSTM, CNN-LSTM, TCN) for the multi-task biosignal pipeline. Covers loss combination, masking for partial labels, subject-wise CV with PyTorch, mixed-precision training, gradient clipping, early stopping, and deterministic seeding. Use AFTER neural-architectures and multimodal-fusion are loaded.
---

# Deep Learning Training (multi-task, low-data, subject-wise CV)

The training loop differs from LightGBM in three important ways:
1. Multi-task loss combination (4 simultaneous objectives, different scales)
2. Masking for tasks with partial labels (phase/reps only valid in active sets; fatigue per-set)
3. Subject-wise mini-batching is impossible — train randomly across windows but EVALUATE per held-out subject

## Multi-task loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    """Weighted sum of 4 task losses. Each loss is masked where targets are NaN/invalid.

    Initial weights default to 1.0; tune per-task scale issue or use uncertainty
    weighting (Kendall et al. 2018) if losses are wildly different magnitudes."""

    def __init__(self, w_exercise=1.0, w_phase=1.0, w_fatigue=1.0, w_reps=1.0,
                 use_uncertainty_weighting=False):
        super().__init__()
        self.weights = {'exercise': w_exercise, 'phase': w_phase,
                         'fatigue': w_fatigue, 'reps': w_reps}
        self.use_uncertainty = use_uncertainty_weighting
        if use_uncertainty_weighting:
            # Learnable log(sigma^2) per task (Kendall et al. 2018)
            self.log_var = nn.Parameter(torch.zeros(4))

    def forward(self, preds, targets, masks):
        losses = {}

        # Exercise: cross-entropy where mask is True
        if masks['exercise'].any():
            losses['exercise'] = F.cross_entropy(
                preds['exercise'][masks['exercise']],
                targets['exercise'][masks['exercise']]
            )
        else:
            losses['exercise'] = torch.tensor(0.0, device=preds['exercise'].device)

        # Phase: cross-entropy on active-set windows only
        if masks['phase'].any():
            losses['phase'] = F.cross_entropy(
                preds['phase'][masks['phase']],
                targets['phase'][masks['phase']]
            )
        else:
            losses['phase'] = torch.tensor(0.0, device=preds['phase'].device)

        # Fatigue: L1 (MAE) on active-set windows only
        if masks['fatigue'].any():
            losses['fatigue'] = F.l1_loss(
                preds['fatigue'][masks['fatigue']].squeeze(-1),
                targets['fatigue'][masks['fatigue']].float()
            )
        else:
            losses['fatigue'] = torch.tensor(0.0, device=preds['fatigue'].device)

        # Reps: smooth L1 on active-set windows
        if masks['reps'].any():
            losses['reps'] = F.smooth_l1_loss(
                preds['reps'][masks['reps']].squeeze(-1),
                targets['reps'][masks['reps']].float()
            )
        else:
            losses['reps'] = torch.tensor(0.0, device=preds['reps'].device)

        # Combine
        if self.use_uncertainty:
            # Kendall et al. 2018: total = sum(0.5 * exp(-log_var) * L + 0.5 * log_var)
            keys = ['exercise', 'phase', 'fatigue', 'reps']
            total = sum(
                0.5 * torch.exp(-self.log_var[i]) * losses[k] + 0.5 * self.log_var[i]
                for i, k in enumerate(keys)
            )
        else:
            total = sum(self.weights[k] * losses[k] for k in losses)

        return total, losses
```

## Subject-wise CV with PyTorch

PyTorch doesn't have GroupKFold built in; you wire it manually. The `subject_id` for each window comes from the dataset's row in the labeled parquet.

```python
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
import numpy as np

def build_subject_cv_folds(dataset, n_splits=None):
    """dataset has .subject_ids attribute (length = len(dataset))."""
    subject_ids = np.asarray(dataset.subject_ids)
    n_unique = len(np.unique(subject_ids))
    if n_splits is None:
        cv = LeaveOneGroupOut() if n_unique <= 24 else GroupKFold(5)
    else:
        cv = GroupKFold(n_splits)
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(
            cv.split(np.zeros(len(subject_ids)), groups=subject_ids)):
        folds.append({
            'fold': fold_idx,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'test_subjects': sorted(np.unique(subject_ids[test_idx]).tolist()),
        })
    return folds
```

## Training loop

```python
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

def train_one_fold(model, dataset, fold, config, device):
    train_ds = Subset(dataset, fold['train_idx'])
    val_ds   = Subset(dataset, fold['test_idx'])

    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True,
        num_workers=config.get('num_workers', 4), pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['batch_size'] * 2, shuffle=False,
        num_workers=config.get('num_workers', 4), pin_memory=True,
    )

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(),
                             lr=config['lr'], weight_decay=config['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['epochs'])
    loss_fn = MultiTaskLoss(use_uncertainty_weighting=config.get('uncertainty_weighting', False)).to(device)
    scaler = GradScaler(enabled=config.get('mixed_precision', True))

    best_val = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_losses = {'total': 0.0, 'exercise': 0.0, 'phase': 0.0, 'fatigue': 0.0, 'reps': 0.0}
        n_batches = 0
        for batch in train_loader:
            x = move_to_device(batch['x'], device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}
            masks   = {k: v.to(device) for k, v in batch['masks'].items()}

            opt.zero_grad()
            with autocast(enabled=config.get('mixed_precision', True)):
                preds = model(x)
                total, parts = loss_fn(preds, targets, masks)

            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            train_losses['total'] += total.item()
            for k in parts:
                train_losses[k] += parts[k].item()
            n_batches += 1
        for k in train_losses:
            train_losses[k] /= max(n_batches, 1)

        # Validate
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        sched.step()

        history.append({'epoch': epoch, 'train': train_losses, 'val': val_metrics})

        # Early stopping on validation total loss
        if val_metrics['total'] < best_val:
            best_val = val_metrics['total']
            patience_counter = 0
            save_checkpoint(model, opt, fold, epoch, val_metrics, config['run_dir'])
        else:
            patience_counter += 1
            if patience_counter >= config.get('patience', 10):
                print(f"Early stopping at epoch {epoch}")
                break

    return best_val, history


def evaluate(model, loader, loss_fn, device):
    model.eval()
    all_preds = {'exercise': [], 'phase': [], 'fatigue': [], 'reps': []}
    all_targets = {k: [] for k in all_preds}
    all_masks = {k: [] for k in all_preds}
    total_loss = 0.0; n_batches = 0
    with torch.no_grad():
        for batch in loader:
            x = move_to_device(batch['x'], device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}
            masks   = {k: v.to(device) for k, v in batch['masks'].items()}
            preds = model(x)
            total, _ = loss_fn(preds, targets, masks)
            total_loss += total.item(); n_batches += 1
            for k in all_preds:
                all_preds[k].append(preds[k].cpu())
                all_targets[k].append(targets[k].cpu())
                all_masks[k].append(masks[k].cpu())
    # Concatenate and compute metrics — defer to multi-task-evaluation skill
    return {'total': total_loss / max(n_batches, 1),
            'preds': {k: torch.cat(v) for k, v in all_preds.items()},
            'targets': {k: torch.cat(v) for k, v in all_targets.items()},
            'masks': {k: torch.cat(v) for k, v in all_masks.items()}}


def move_to_device(x, device):
    """Handle both tensor and dict-of-tensors (for late/hybrid fusion)."""
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)
```

## Determinism (required for reproducibility in research)

```python
def set_deterministic(seed=42):
    import random, os
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Document the seed in `model_card.md`. Run final reported metrics with at least 3 seeds and report mean ± std — single-seed neural net results in research are not credible.

## Hyperparameter tuning

With ~84k training windows but only ~24 subjects, you tune within an inner CV loop, NOT on the held-out test fold. Optuna with nested CV:

```python
import optuna

def objective(trial, dataset, outer_fold):
    config = {
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-3),
        'weight_decay': trial.suggest_loguniform('wd', 1e-6, 1e-2),
        'batch_size': trial.suggest_categorical('bs', [32, 64, 128]),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
        'epochs': 50, 'patience': 8,
        ...
    }
    # Inner GroupKFold on outer_fold['train_idx']
    inner_subjects = np.unique(np.array(dataset.subject_ids)[outer_fold['train_idx']])
    inner_cv = GroupKFold(min(5, len(inner_subjects)))
    inner_scores = []
    for inner_train, inner_val in inner_cv.split(...):
        model = build_model(config)
        score, _ = train_one_fold(model, dataset, inner_fold, config, device)
        inner_scores.append(score)
    return np.mean(inner_scores)
```

Budget: 30-50 trials per architecture is realistic. With 4 architectures × 3 seeds × LOSO over 24 subjects, this is many GPU hours. Plan accordingly.

## Compute budget reality check

| Architecture | Time per epoch (24 subjects, 1 GPU) | Per LOSO fold (50 epochs) | Full LOSO (24 folds × 3 seeds) |
|--------------|-------------------------------------|---------------------------|--------------------------------|
| 1D-CNN       | ~30 s                               | ~25 min                   | ~30 hours                      |
| LSTM         | ~90 s                               | ~75 min                   | ~90 hours                      |
| CNN-LSTM     | ~60 s                               | ~50 min                   | ~60 hours                      |
| TCN          | ~40 s                               | ~33 min                   | ~40 hours                      |

Estimates assume single mid-range GPU (RTX 3080-class). If this is prohibitive, switch from full LOSO to GroupKFold(5) — 5× fewer folds, same statistical validity for 24 subjects.

## Saving artifacts (for model_card.md)

Per fold, save:
- `runs/<slug>/<arch>/fold_<i>/checkpoint_best.pt`
- `runs/<slug>/<arch>/fold_<i>/history.json` (train/val loss per epoch)
- `runs/<slug>/<arch>/fold_<i>/test_preds.parquet` (preds, targets, subject_id)

Aggregate across folds:
- `runs/<slug>/<arch>/cv_metrics.json` (mean ± std per task)
- `runs/<slug>/<arch>/per_subject_breakdown.csv`
- `runs/<slug>/<arch>/training_curves.png`
- `runs/<slug>/<arch>/model_card.md` (final per-architecture summary)

Top-level comparison:
- `runs/<slug>/comparison.md` (LightGBM vs all 4 NN architectures)
- `runs/<slug>/comparison.png`

## Hard rules

- **Always use deterministic seeding.** Single-seed results are not publishable.
- **Always run ≥3 seeds** for final reported numbers.
- **Always use subject-wise CV** (GroupKFold or LOSO).
- **Always tune in nested CV** — never tune on the test fold.
- **Always evaluate per-subject** in addition to mean — check for catastrophic failures.
- **Always document fold-level variance** (std across folds), not just the mean.
- **Always compare to LightGBM baseline** — if NN doesn't beat LightGBM by a meaningful margin, you must report that and explain.

## References

When documenting training-loop choices, cite from these (full entries in `literature-references` skill):

- **Kendall et al. 2018** — uncertainty-weighted multi-task loss
- **Loshchilov & Hutter 2019** — AdamW optimizer
- **Loshchilov & Hutter 2017** — cosine annealing LR schedule
- **Smith 2018** — practical NN hyperparameter tuning
- **Saeb et al. 2017** — subject-wise CV motivation
- **Akiba et al. 2019** — Optuna for hyperparameter optimization
- **Goodfellow et al. 2016** — for fundamentals (regularization, optimization, gradient clipping)

All are in the central `literature-references` skill. Never invent.
