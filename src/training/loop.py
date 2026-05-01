"""Unified multi-task training loop. All 4 architecture scripts call run_cv()."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler

from src.training.losses import MultiTaskLoss
from src.eval.metrics import compute_all_metrics


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 8
    mixed_precision: bool = True
    num_workers: int = 4
    use_uncertainty_weighting: bool = False
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'exercise': 1.0, 'phase': 1.0, 'fatigue': 1.0, 'reps': 0.5
    })
    # Per-task target representation; defaults preserve hard-label baseline.
    # See configs/nn.yaml `target_modes` and src/training/losses.py.
    target_modes: Dict[str, str] = field(default_factory=lambda: {
        'reps': 'hard', 'phase': 'hard',
    })


def set_deterministic(seed: int):
    """For reproducible NN results — required for research credibility."""
    import random, os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _move_x_to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device, non_blocking=True) for k, v in x.items()}
    return x.to(device, non_blocking=True)


def _train_one_epoch(model, loader, loss_fn, opt, scaler, device, cfg):
    model.train()
    total_n = 0
    sums = {'total': 0.0, 'exercise': 0.0, 'phase': 0.0, 'fatigue': 0.0, 'reps': 0.0}
    for batch in loader:
        x = _move_x_to_device(batch['x'], device)
        targets = {k: v.to(device, non_blocking=True)
                   for k, v in batch['targets'].items()}
        masks = {k: v.to(device, non_blocking=True)
                 for k, v in batch['masks'].items()}

        opt.zero_grad(set_to_none=True)
        with autocast("cuda",   enabled=cfg.mixed_precision):
            preds = model(x)
            total, parts = loss_fn(preds, targets, masks)

        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        n = next(iter(targets.values())).shape[0]
        total_n += n
        sums['total'] += total.item() * n
        for k in parts:
            sums[k] += parts[k].item() * n

    return {k: v / max(total_n, 1) for k, v in sums.items()}


@torch.no_grad()
def _evaluate(model, loader, loss_fn, device, cfg, n_exercise, n_phase):
    model.eval()
    all_preds = {'exercise': [], 'phase': [], 'fatigue': [], 'reps': []}
    all_targets = {k: [] for k in all_preds}
    all_masks = {k: [] for k in all_preds}
    total_n = 0
    sums = {'total': 0.0, 'exercise': 0.0, 'phase': 0.0, 'fatigue': 0.0, 'reps': 0.0}
    for batch in loader:
        x = _move_x_to_device(batch['x'], device)
        targets = {k: v.to(device) for k, v in batch['targets'].items()}
        masks = {k: v.to(device) for k, v in batch['masks'].items()}
        with autocast("cuda",   enabled=cfg.mixed_precision):
            preds = model(x)
            total, parts = loss_fn(preds, targets, masks)

        n = next(iter(targets.values())).shape[0]
        total_n += n
        sums['total'] += total.item() * n
        for k in parts:
            sums[k] += parts[k].item() * n

        for k in all_preds:
            all_preds[k].append(preds[k].cpu())
            all_targets[k].append(targets[k].cpu())
            all_masks[k].append(masks[k].cpu())

    losses = {k: v / max(total_n, 1) for k, v in sums.items()}
    cat_preds = {k: torch.cat(v) for k, v in all_preds.items()}
    cat_targets = {k: torch.cat(v) for k, v in all_targets.items()}
    cat_masks = {k: torch.cat(v) for k, v in all_masks.items()}

    metrics = compute_all_metrics(cat_preds, cat_targets, cat_masks,
                                   n_exercise=n_exercise, n_phase=n_phase)
    return losses, metrics, cat_preds, cat_targets, cat_masks


def train_one_fold(
    model_factory: Callable[[], torch.nn.Module],
    dataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
    n_exercise: int,
    n_phase: int,
):
    """Train one outer-CV fold. Returns history + final test metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Subset(dataset, train_idx)
    test_ds = Subset(dataset, test_idx)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    model = model_factory().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = MultiTaskLoss(
        **{f'w_{k}': v for k, v in cfg.loss_weights.items()},
        use_uncertainty_weighting=cfg.use_uncertainty_weighting,
        target_modes=cfg.target_modes,
    ).to(device)
    scaler = GradScaler(enabled=cfg.mixed_precision)

    best_val = float('inf')
    best_state = None
    patience = 0
    history = []

    for epoch in range(cfg.epochs):
        train_losses = _train_one_epoch(model, train_loader, loss_fn, opt, scaler,
                                          device, cfg)
        val_losses, val_metrics, *_ = _evaluate(model, test_loader, loss_fn,
                                                  device, cfg,
                                                  n_exercise, n_phase)
        sched.step()
        history.append({'epoch': epoch, 'train': train_losses,
                         'val_loss': val_losses, 'val_metrics': val_metrics})
        print(f"  Epoch {epoch:3d}  train_total={train_losses['total']:.4f}  "
              f"val_total={val_losses['total']:.4f}  "
              f"ex_F1={val_metrics['exercise']['f1_macro']:.3f}  "
              f"ph_F1={val_metrics['phase']['f1_macro']:.3f}  "
              f"fat_MAE={val_metrics['fatigue']['mae']:.3f}  "
              f"rep_MAE={val_metrics['reps']['mae']:.3f}")

        if val_losses['total'] < best_val:
            best_val = val_losses['total']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                print(f"  Early stop at epoch {epoch}")
                break

    # Restore best and evaluate final
    model.load_state_dict(best_state)
    final_losses, final_metrics, preds, targets, masks = _evaluate(
        model, test_loader, loss_fn, device, cfg, n_exercise, n_phase
    )

    # Persist
    torch.save({'state_dict': best_state, 'config': cfg.__dict__},
                out_dir / 'checkpoint_best.pt')
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2, default=_jsonable)

    # Save raw preds for later analysis
    torch.save({'preds': preds, 'targets': targets, 'masks': masks,
                 'test_idx': test_idx},
                out_dir / 'test_preds.pt')

    return history, final_metrics


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run_cv(
    dataset,
    model_factory: Callable[[], torch.nn.Module],
    arch_name: str,
    cfg: TrainConfig,
    splits: List[Dict],          # list of {'fold': i, 'train_idx': ..., 'test_idx': ...}
    out_root: Path,
    seeds: List[int] = (42, 1337, 7),
    n_exercise: Optional[int] = None,
    n_phase: Optional[int] = None,
):
    """Run subject-wise CV across folds × seeds. Aggregates metrics at the end.

    `splits` is the SAME list used by the LightGBM baseline to ensure fair
    comparison (re-read from configs/splits.csv).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[run_cv] Architecture={arch_name}  device={device}  seeds={list(seeds)}  "
          f"n_folds={len(splits)}")

    if n_exercise is None:
        n_exercise = dataset.n_exercise
    if n_phase is None:
        n_phase = dataset.n_phase

    arch_dir = out_root / arch_name
    arch_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in seeds:
        for fold in splits:
            fold_id = fold['fold']
            print(f"\n[run_cv] seed={seed}  fold={fold_id}  "
                  f"train_n={len(fold['train_idx'])}  test_n={len(fold['test_idx'])}  "
                  f"test_subjects={fold.get('test_subjects', [])}")
            set_deterministic(seed)
            fold_dir = arch_dir / f"seed_{seed}" / f"fold_{fold_id}"
            history, metrics = train_one_fold(
                model_factory=model_factory,
                dataset=dataset,
                train_idx=fold['train_idx'],
                test_idx=fold['test_idx'],
                cfg=cfg,
                device=device,
                out_dir=fold_dir,
                n_exercise=n_exercise,
                n_phase=n_phase,
            )
            all_results.append({
                'seed': seed, 'fold': fold_id,
                'test_subjects': fold.get('test_subjects', []),
                'metrics': metrics,
            })

    # Aggregate
    summary = aggregate_cv_results(all_results)
    with open(arch_dir / 'cv_summary.json', 'w') as f:
        json.dump({'arch': arch_name, 'summary': summary,
                    'all_results': all_results}, f, indent=2, default=_jsonable)
    print(f"\n[run_cv] {arch_name} complete. Summary in {arch_dir}/cv_summary.json")
    return summary, all_results


def aggregate_cv_results(all_results: List[Dict]) -> Dict:
    """Mean ± std across folds × seeds for each task metric."""
    def _collect(task: str, metric: str):
        vals = []
        for r in all_results:
            v = r['metrics'].get(task, {}).get(metric)
            if v is not None and not np.isnan(v):
                vals.append(float(v))
        if not vals:
            return {'mean': float('nan'), 'std': float('nan'), 'n': 0}
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                'n': len(vals)}

    return {
        'exercise': {
            'f1_macro':  _collect('exercise', 'f1_macro'),
            'balanced_accuracy': _collect('exercise', 'balanced_accuracy'),
        },
        'phase': {
            'f1_macro':  _collect('phase', 'f1_macro'),
            'balanced_accuracy': _collect('phase', 'balanced_accuracy'),
        },
        'fatigue': {
            'mae':       _collect('fatigue', 'mae'),
            'pearson_r': _collect('fatigue', 'pearson_r'),
        },
        'reps': {
            'mae':       _collect('reps', 'mae'),
        },
    }
