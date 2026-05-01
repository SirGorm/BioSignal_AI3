"""Plot training curves and confusion matrices from completed runs.

Reads:
- history.json (per-fold per-seed) for training/validation curves
- test_preds.pt for confusion matrices

Writes PNGs to the run directory.

References:
- Saeb et al. 2017 — per-subject reporting
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import json

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for headless servers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.eval.plot_style import apply_style, despine

apply_style()


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_history_for_fold(history_path: Path, out_path: Path):
    """Plot train vs val total loss + per-task losses across epochs."""
    history = json.loads(history_path.read_text())
    if not history:
        return

    def _scalar(entry, key):
        v = entry.get(key) or entry.get(f'{key}_loss')
        return v.get('total') if isinstance(v, dict) else v

    def _per_task(entry, key):
        v = entry.get(key) or entry.get(f'{key}_loss')
        return v if isinstance(v, dict) else {}

    epochs = [h['epoch'] for h in history]
    train_total = [_scalar(h, 'train') for h in history]
    val_total = [_scalar(h, 'val') for h in history]

    train_per_task = {
        k: [_per_task(h, 'train').get(k, np.nan) for h in history]
        for k in ('exercise', 'phase', 'fatigue', 'reps')
    }
    val_per_task = {
        k: [_per_task(h, 'val').get(k, np.nan) for h in history]
        for k in ('exercise', 'phase', 'fatigue', 'reps')
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    ax = axes[0, 0]
    ax.plot(epochs, train_total, label='train', lw=2)
    ax.plot(epochs, val_total, label='val', lw=2)
    ax.set_title('Total loss')
    ax.set_xlabel('epoch'); ax.set_ylabel('loss')
    ax.legend(); ax.grid(alpha=0.3)

    for i, k in enumerate(['exercise', 'phase', 'fatigue', 'reps']):
        ax = axes.flat[i + 1]
        ax.plot(epochs, train_per_task[k], label='train', lw=2)
        ax.plot(epochs, val_per_task[k], label='val', lw=2)
        ax.set_title(k)
        ax.set_xlabel('epoch'); ax.set_ylabel('loss')
        ax.legend(); ax.grid(alpha=0.3)

    axes[1, 2].axis('off')   # 5 subplots in 2x3 grid; hide the empty one
    fig.suptitle(out_path.parent.name)
    fig.tight_layout()
    despine(fig=fig)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_all_histories(arch_run_dir: Path):
    """Find all history.json under <arch_run_dir>/<arch>/seed_*/fold_*/ and
    write a curves PNG next to each."""
    histories = list(arch_run_dir.rglob('history.json'))
    if not histories:
        print(f"[plot] No history.json files under {arch_run_dir}")
        return
    for h in histories:
        out = h.parent / 'training_curves.png'
        try:
            plot_history_for_fold(h, out)
        except Exception as e:
            print(f"[plot] Failed for {h}: {e}")
    print(f"[plot] Wrote {len(histories)} training curves under {arch_run_dir}")


# ---------------------------------------------------------------------------
# Aggregated curves: mean ± std across folds × seeds
# ---------------------------------------------------------------------------

def plot_aggregated_history(arch_run_dir: Path, out_path: Path):
    """Aggregate train/val total loss across all folds × seeds and plot
    mean ± 1 SD band."""
    histories = []
    for h in arch_run_dir.rglob('history.json'):
        try:
            histories.append(json.loads(h.read_text()))
        except Exception:
            continue
    if not histories:
        return

    def _loss(entry, key):
        v = entry.get(key) or entry.get(f'{key}_loss')
        if isinstance(v, dict):
            return v.get('total')
        return v

    # Truncate to shortest run (early stopping varies)
    min_len = min(len(h) for h in histories)
    train = np.array([[_loss(h, 'train') for h in run[:min_len]]
                       for run in histories], dtype=float)
    val = np.array([[_loss(h, 'val') for h in run[:min_len]]
                     for run in histories], dtype=float)

    epochs = np.arange(min_len)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, arr, color in [('train', train, 'C0'), ('val', val, 'C1')]:
        m = arr.mean(axis=0)
        s = arr.std(axis=0)
        ax.plot(epochs, m, label=label, color=color, lw=2)
        ax.fill_between(epochs, m - s, m + s, alpha=0.2, color=color)
    ax.set_title(f'{arch_run_dir.name} — mean ± SD across folds × seeds')
    ax.set_xlabel('epoch'); ax.set_ylabel('total loss')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    despine(fig=fig)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[plot] Wrote {out_path}")


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                            title: str, out_path: Path, normalize: bool = True):
    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(max(5, len(class_names)),
                                      max(4, len(class_names) * 0.8)))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)

    # Text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm_norm[i, j]
            color = 'white' if v > 0.5 else 'black'
            txt = f"{v:.2f}" if normalize else f"{int(v)}"
            ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    despine(fig=fig, left=True, bottom=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_confusion_matrices_for_run(
    arch_run_dir: Path,
    exercise_classes: Optional[List[str]] = None,
    phase_classes: Optional[List[str]] = None,
):
    """Aggregate per-fold predictions and plot confusion matrices for
    classification tasks (exercise + phase)."""
    import torch

    pred_files = list(arch_run_dir.rglob('test_preds.pt'))
    if not pred_files:
        print(f"[plot] No test_preds.pt under {arch_run_dir}")
        return

    # Try loading dataset_meta.json for class names
    meta_path = arch_run_dir / 'dataset_meta.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        exercise_classes = exercise_classes or meta.get('exercise_classes')
        phase_classes = phase_classes or meta.get('phase_classes')

    all_ex_true, all_ex_pred = [], []
    all_ph_true, all_ph_pred = [], []
    for p in pred_files:
        d = torch.load(p, map_location='cpu', weights_only=False)
        preds = d['preds']
        targets = d['targets']
        masks = d['masks']
        m = masks['exercise'].numpy().astype(bool)
        if m.any():
            all_ex_true.append(targets['exercise'].numpy()[m])
            all_ex_pred.append(preds['exercise'].argmax(dim=-1).numpy()[m])
        m = masks['phase'].numpy().astype(bool)
        if m.any():
            all_ph_true.append(targets['phase'].numpy()[m])
            all_ph_pred.append(preds['phase'].argmax(dim=-1).numpy()[m])

    if all_ex_true and exercise_classes:
        y_t = np.concatenate(all_ex_true)
        y_p = np.concatenate(all_ex_pred)
        cm = confusion_matrix(y_t, y_p, labels=list(range(len(exercise_classes))))
        plot_confusion_matrix(cm, exercise_classes,
                               'Exercise (aggregated across folds × seeds)',
                               arch_run_dir / 'confusion_matrix_exercise.png')
        print(f"[plot] Wrote {arch_run_dir / 'confusion_matrix_exercise.png'}")

    if all_ph_true and phase_classes:
        y_t = np.concatenate(all_ph_true)
        y_p = np.concatenate(all_ph_pred)
        cm = confusion_matrix(y_t, y_p, labels=list(range(len(phase_classes))))
        plot_confusion_matrix(cm, phase_classes,
                               'Phase (aggregated across folds × seeds)',
                               arch_run_dir / 'confusion_matrix_phase.png')
        print(f"[plot] Wrote {arch_run_dir / 'confusion_matrix_phase.png'}")


# ---------------------------------------------------------------------------
# Calibration plot for fatigue
# ---------------------------------------------------------------------------

def plot_fatigue_calibration(arch_run_dir: Path):
    """Bin predicted RPE in 1-point bins, plot mean true vs predicted."""
    import torch

    pred_files = list(arch_run_dir.rglob('test_preds.pt'))
    if not pred_files:
        return

    all_true, all_pred = [], []
    for p in pred_files:
        d = torch.load(p, map_location='cpu', weights_only=False)
        m = d['masks']['fatigue'].numpy().astype(bool)
        if m.any():
            all_true.append(d['targets']['fatigue'].numpy()[m])
            all_pred.append(d['preds']['fatigue'].numpy()[m])
    if not all_true:
        return
    y_t = np.concatenate(all_true)
    y_p = np.concatenate(all_pred)

    bins = np.arange(1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    digitized = np.digitize(y_p, bins) - 1
    digitized = np.clip(digitized, 0, len(bin_centers) - 1)
    mean_true_per_bin = [y_t[digitized == i].mean() if (digitized == i).any()
                          else np.nan for i in range(len(bin_centers))]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([1, 10], [1, 10], 'k--', alpha=0.5, label='perfect')
    ax.plot(bin_centers, mean_true_per_bin, 'o-', lw=2, label='observed')
    ax.scatter(y_p, y_t, alpha=0.1, s=10)
    ax.set_xlim(0.5, 10.5); ax.set_ylim(0.5, 10.5)
    ax.set_xlabel('Predicted RPE'); ax.set_ylabel('True RPE')
    ax.set_title('Fatigue calibration')
    ax.legend(); ax.grid(alpha=0.3); ax.set_aspect('equal')
    out_path = arch_run_dir / 'fatigue_calibration.png'
    fig.tight_layout()
    despine(fig=fig)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[plot] Wrote {out_path}")


def plot_reps_evaluation(arch_run_dir: Path):
    """Reps eval: scatter pred vs true + tolerance accuracy bar chart."""
    import torch

    pred_files = list(arch_run_dir.rglob('test_preds.pt'))
    if not pred_files:
        return

    all_true, all_pred = [], []
    for p in pred_files:
        d = torch.load(p, map_location='cpu', weights_only=False)
        m = d['masks']['reps'].numpy().astype(bool)
        if m.any():
            all_true.append(d['targets']['reps'].numpy()[m])
            all_pred.append(d['preds']['reps'].numpy()[m])
    if not all_true:
        return
    y_t = np.concatenate(all_true)
    y_p = np.concatenate(all_pred)
    err = np.abs(y_p - y_t)

    # Tolerance accuracy: fraction of samples with |pred - true| <= N
    tols = [0, 1, 2, 3, 4, 5]
    pcts = [100.0 * float(np.mean(err <= t)) for t in tols]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: scatter pred vs true with diagonal
    ax = axes[0]
    lo = float(min(y_t.min(), y_p.min()) - 0.5)
    hi = float(max(y_t.max(), y_p.max()) + 0.5)
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='perfect')
    ax.scatter(y_p, y_t, alpha=0.15, s=10)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('Predicted reps'); ax.set_ylabel('True reps')
    ax.set_title(f'Reps calibration (n={len(y_t)})')
    ax.legend(); ax.grid(alpha=0.3); ax.set_aspect('equal')

    # Right: tolerance accuracy bar chart
    ax = axes[1]
    colors = ['#2ca02c' if t == 0 else '#1f77b4' for t in tols]
    bars = ax.bar([f"+/-{t}" for t in tols], pcts, color=colors, edgecolor='black')
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 1.5,
                f'{pct:.1f}%', ha='center', fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel('% of windows within tolerance')
    ax.set_xlabel('Tolerance (reps)')
    ax.set_title(f'Reps accuracy by tolerance band (MAE={err.mean():.2f})')
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    despine(fig=fig)
    out_path = arch_run_dir / 'reps_evaluation.png'
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[plot] Wrote {out_path}")


# ---------------------------------------------------------------------------
# Top-level convenience: plot everything for a run
# ---------------------------------------------------------------------------

def plot_everything_for_run(arch_run_dir: Path):
    """Convenience: training curves (per-fold + aggregated), confusion
    matrices, fatigue calibration, reps evaluation."""
    print(f"[plot] Processing {arch_run_dir}")
    plot_all_histories(arch_run_dir)
    plot_aggregated_history(arch_run_dir,
                              arch_run_dir / 'training_curves_aggregated.png')
    plot_confusion_matrices_for_run(arch_run_dir)
    plot_fatigue_calibration(arch_run_dir)
    plot_reps_evaluation(arch_run_dir)
