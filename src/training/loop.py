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

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

# GPU speedups: cuDNN auto-tuner picks the fastest convolution algorithm for
# the input shapes (one-time cost amortized across epochs); TF32 enables
# fast matmul on Ampere+ (RTX 30/40/50). Both produce small numerical drift
# vs. strict deterministic mode but are safe for HP search and final eval —
# variance from CV folds dominates anyway.
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass  # older torch


# ---------------------------------------------------------------------------
# GPU-resident batch iterator. Skips DataLoader entirely when dataset has
# been materialized on cuda (see Dataset.materialize_to_device). Eliminates
# CPU↔GPU transfer per batch and Windows worker-spawn overhead — empirically
# 5-10× faster than num_workers=0 + DataLoader on our small dataset.
# ---------------------------------------------------------------------------

class _GPUBatchIterator:
    """Mimics DataLoader's `for batch in loader:` interface but builds batches
    by direct GPU-tensor indexing — no DataLoader, no workers, no pin_memory."""

    __slots__ = ("dataset", "indices", "batch_size", "shuffle",
                  "drop_last", "device")

    def __init__(self, dataset, indices, batch_size, shuffle, drop_last, device):
        self.dataset = dataset
        self.indices = torch.as_tensor(np.asarray(indices), dtype=torch.long,
                                         device=device)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.device = device

    def __iter__(self):
        n = self.indices.shape[0]
        if self.shuffle:
            perm = torch.randperm(n, device=self.device)
            idx = self.indices[perm]
        else:
            idx = self.indices
        end = (n // self.batch_size) * self.batch_size if self.drop_last else n
        bs = self.batch_size
        for i in range(0, end, bs):
            bi = idx[i:i + bs]
            yield {
                "x": self.dataset._gpu_x[bi],
                "targets": {k: v[bi] for k, v in self.dataset._gpu_targets.items()},
                "masks":   {k: v[bi] for k, v in self.dataset._gpu_masks.items()},
            }

    def __len__(self):
        n = self.indices.shape[0]
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def _make_loader(dataset, indices, batch_size, shuffle, drop_last,
                 num_workers, device):
    """Return a DataLoader-compatible iterable. Uses GPU fast path when the
    dataset has been materialized on device; otherwise falls back to the
    standard DataLoader+Subset path for backward compat."""
    if getattr(dataset, "gpu_resident", False):
        return _GPUBatchIterator(dataset, indices, batch_size, shuffle,
                                  drop_last, device)
    sub = Subset(dataset, indices)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=drop_last)


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
    use_uncertainty_weighting: bool = True
    # When False, skip torch.save of checkpoint_best.pt and test_preds.pt
    # per fold. Useful for Optuna phase 1 trials (250+ disposable runs)
    # where the .pt files are never used. Phase 2 should leave this True.
    save_checkpoint: bool = True
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'exercise': 1.0, 'phase': 1.0, 'fatigue': 1.0, 'reps': 0.5
    })
    # Per-task target representation; matches configs/nn.yaml.
    # reps='soft_window'  : ρ(t)=1/Δt_k integrated over window; integer count
    #                       recovered at eval via src/eval/rep_aggregation.py.
    # phase='soft'        : KL-div on per-window phase distribution.
    target_modes: Dict[str, str] = field(default_factory=lambda: {
        'reps': 'soft_window', 'phase': 'soft',
    })
    # Tasks contributing to the total loss. Default: all 4 (multi-task).
    # Single-task example: enabled_tasks=['fatigue'] → fatigue-only model.
    enabled_tasks: List[str] = field(default_factory=lambda: [
        'exercise', 'phase', 'fatigue', 'reps',
    ])


def set_deterministic(seed: int):
    """Seed RNGs but DO NOT disable cuDNN benchmark — we trade strict
    bit-reproducibility for the 1.5-2× speedup from auto-tuned conv kernels.
    Variance from subject-wise CV folds dominates the run-to-run drift from
    benchmark, so this is safe for HP search and 3-seed final eval."""
    import random, os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark stays True (set at module load); do not flip per-seed.


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

        # Skip optimizer step when this batch has no training signal for any
        # enabled task (all masks False). Fatigue-only + active_only=False can
        # produce all-rest batches; without this guard AMP grad scaler raises
        # "No inf checks were recorded for this optimizer".
        enabled = getattr(loss_fn, 'enabled', None) or list(masks.keys())
        has_signal = any(bool(masks[k].any().item())
                         for k in enabled if k in masks)
        if has_signal:
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

    train_loader = _make_loader(dataset, train_idx, cfg.batch_size,
                                  shuffle=True, drop_last=True,
                                  num_workers=cfg.num_workers, device=device)
    test_loader = _make_loader(dataset, test_idx, cfg.batch_size * 2,
                                 shuffle=False, drop_last=False,
                                 num_workers=cfg.num_workers, device=device)

    model = model_factory().to(device)
    loss_fn = MultiTaskLoss(
        **{f'w_{k}': v for k, v in cfg.loss_weights.items()},
        use_uncertainty_weighting=cfg.use_uncertainty_weighting,
        target_modes=cfg.target_modes,
        enabled_tasks=list(cfg.enabled_tasks),
    ).to(device)
    # Kendall uncertainty weighting introduces learnable log_var parameters
    # on the loss module — they MUST be in the optimizer or they stay at 0
    # and the weighting collapses to a uniform 0.5·Σ losses. Use a separate
    # param group with weight_decay=0 since log_var is a noise scale, not a
    # weight, and should not be regularized toward zero.
    if cfg.use_uncertainty_weighting:
        opt = torch.optim.AdamW(
            [
                {'params': list(model.parameters())},
                {'params': list(loss_fn.parameters()), 'weight_decay': 0.0},
            ],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.mixed_precision)

    # TensorBoard writer — one logdir per (seed, fold) so the UI shows each
    # fold as a separate "run". Launch with:
    #   tensorboard --logdir runs/<run-dir>
    # and TB discovers all seeds × folds × archs under it.
    tb_writer = SummaryWriter(log_dir=str(out_dir / "tb"),
                              flush_secs=10) if _TB_AVAILABLE else None

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
        if tb_writer is not None:
            for k, v in train_losses.items():
                tb_writer.add_scalar(f"loss/train/{k}", v, epoch)
            for k, v in val_losses.items():
                tb_writer.add_scalar(f"loss/val/{k}", v, epoch)
            for task in ('exercise', 'phase', 'fatigue', 'reps'):
                for metric, val in val_metrics.get(task, {}).items():
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue
                    tb_writer.add_scalar(f"metrics/{task}/{metric}",
                                         float(val), epoch)
            tb_writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)
            tb_writer.flush()
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

    if tb_writer is not None:
        tb_writer.close()

    # Restore best and evaluate final
    model.load_state_dict(best_state)
    final_losses, final_metrics, preds, targets, masks = _evaluate(
        model, test_loader, loss_fn, device, cfg, n_exercise, n_phase
    )
    # Surface the uncertainty-weighted total val loss so Optuna can rank
    # trials by it directly (no separate composite score needed).
    final_metrics['val_total'] = float(final_losses['total'])
    final_metrics['val_total_per_task'] = {
        k: float(v) for k, v in final_losses.items() if k != 'total'
    }

    # Persist
    if cfg.save_checkpoint:
        torch.save({'state_dict': best_state, 'config': cfg.__dict__},
                    out_dir / 'checkpoint_best.pt')
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2, default=_jsonable)

    # Save raw preds for later analysis
    if cfg.save_checkpoint:
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

    # Aggregate — gate untrained heads so they don't contaminate comparisons
    summary = aggregate_cv_results(all_results,
                                    enabled_tasks=list(cfg.enabled_tasks))
    with open(arch_dir / 'cv_summary.json', 'w') as f:
        json.dump({'arch': arch_name, 'summary': summary,
                    'enabled_tasks': list(cfg.enabled_tasks),
                    'all_results': all_results}, f, indent=2, default=_jsonable)
    print(f"\n[run_cv] {arch_name} complete. Summary in {arch_dir}/cv_summary.json")
    return summary, all_results


def aggregate_cv_results(all_results: List[Dict],
                          enabled_tasks: Optional[List[str]] = None) -> Dict:
    """Mean ± std across folds × seeds for each task metric.

    Tasks not in ``enabled_tasks`` are marked ``{'untrained': True}`` so plots
    and tables can skip them — predictions from a non-enabled head come from
    a random-init linear layer and must not be compared to trained metrics.
    """
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

    enabled = (set(enabled_tasks)
               if enabled_tasks is not None
               else {'exercise', 'phase', 'fatigue', 'reps'})

    def _block(task: str, metrics: Dict[str, Dict]):
        if task not in enabled:
            return {'untrained': True}
        return metrics

    # Mean uncertainty-weighted val_total across folds (used by Optuna).
    val_totals = [r['metrics'].get('val_total') for r in all_results]
    val_totals = [float(v) for v in val_totals
                  if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if val_totals:
        val_total_block = {
            'mean': float(np.mean(val_totals)),
            'std': float(np.std(val_totals)),
            'n': len(val_totals),
        }
    else:
        val_total_block = {'mean': float('nan'), 'std': float('nan'), 'n': 0}

    return {
        'val_total': val_total_block,
        'exercise': _block('exercise', {
            'f1_macro':  _collect('exercise', 'f1_macro'),
            'balanced_accuracy': _collect('exercise', 'balanced_accuracy'),
        }),
        'phase': _block('phase', {
            'f1_macro':  _collect('phase', 'f1_macro'),
            'balanced_accuracy': _collect('phase', 'balanced_accuracy'),
        }),
        'fatigue': _block('fatigue', {
            'mae':       _collect('fatigue', 'mae'),
            'pearson_r': _collect('fatigue', 'pearson_r'),
        }),
        'reps': _block('reps', {
            'mae':       _collect('reps', 'mae'),
        }),
    }
