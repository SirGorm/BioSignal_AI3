"""Plot training curves + confusion matrices for the best multitask model.

Default target is tcn_raw/multi/w1s in runs/sweep_raw/ (lowest val_total in
the completed sweep, val_total=-0.275). Aggregates over all (seed, fold)
pairs found under phase2_seeds/.

Outputs to <run_dir>/plots/:
    loss_curves.png         — train+val loss per task vs epoch
    cm_exercise.png         — exercise confusion matrix (per-window argmax)
    cm_phase.png            — phase confusion matrix (per-window argmax)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def collect_histories(run_dir: Path):
    hists = []
    for hp in run_dir.rglob('history.json'):
        try:
            hists.append((hp, json.loads(hp.read_text())))
        except Exception as e:
            print(f"  skip {hp}: {e}")
    return hists


def collect_preds(run_dir: Path):
    out = []
    for pp in run_dir.rglob('test_preds.pt'):
        try:
            out.append((pp, torch.load(pp, map_location='cpu',
                                        weights_only=False)))
        except Exception as e:
            print(f"  skip {pp}: {e}")
    return out


def aggregate_curves(histories):
    """Return per-task dict of train/val arrays aligned by epoch with mean+std."""
    by_task = defaultdict(lambda: {'train': defaultdict(list),
                                    'val':   defaultdict(list)})
    for _, hist in histories:
        for entry in hist:
            ep = entry['epoch']
            for task, v in entry['train'].items():
                by_task[task]['train'][ep].append(v)
            for task, v in entry['val_loss'].items():
                by_task[task]['val'][ep].append(v)

    out = {}
    for task, splits in by_task.items():
        epochs = sorted(splits['train'].keys())
        out[task] = {
            'epochs': np.array(epochs),
            'train_mean': np.array([np.mean(splits['train'][e]) for e in epochs]),
            'train_std':  np.array([np.std (splits['train'][e]) for e in epochs]),
            'val_mean':   np.array([np.mean(splits['val'][e])   for e in epochs]),
            'val_std':    np.array([np.std (splits['val'][e])   for e in epochs]),
            'n_runs':     np.array([len(splits['train'][e])     for e in epochs]),
        }
    return out


def plot_loss_curves(curves, out_path: Path, run_label: str):
    tasks = ['total', 'exercise', 'phase', 'fatigue', 'reps']
    tasks = [t for t in tasks if t in curves]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 4),
                              sharex=False)
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        c = curves[task]
        ax.plot(c['epochs'], c['train_mean'], label='train', color='C0')
        ax.fill_between(c['epochs'],
                        c['train_mean'] - c['train_std'],
                        c['train_mean'] + c['train_std'], alpha=0.2, color='C0')
        ax.plot(c['epochs'], c['val_mean'],   label='val',   color='C1')
        ax.fill_between(c['epochs'],
                        c['val_mean'] - c['val_std'],
                        c['val_mean'] + c['val_std'], alpha=0.2, color='C1')
        ax.set_title(f"{task}  (n_runs≤{c['n_runs'].max()})")
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Training curves — {run_label}  "
                 f"({len(curves[tasks[0]]['epochs'])} epochs max, "
                 f"mean±std across {curves[tasks[0]]['n_runs'][0]} runs)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


def aggregate_preds(pred_list, task: str):
    """For each (seed, fold) pair, argmax preds and targets, return concatenated."""
    all_pred, all_true = [], []
    for _, d in pred_list:
        preds   = d['preds'].get(task)
        targets = d['targets'].get(task)
        masks   = d['masks'].get(task, None)
        if preds is None or targets is None:
            continue
        pp = preds.float().argmax(dim=-1).numpy()
        if targets.ndim == 2:           # soft label: one-hot prob → argmax
            tt = targets.float().argmax(dim=-1).numpy()
        else:
            tt = targets.long().numpy()
        if masks is not None:
            m = masks.bool().numpy()
            pp, tt = pp[m], tt[m]
        all_pred.append(pp)
        all_true.append(tt)
    return np.concatenate(all_true), np.concatenate(all_pred)


def plot_cm(y_true, y_pred, classes, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(0.9 + 0.6 * len(classes),
                                      0.9 + 0.6 * len(classes)))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i,j]}\n{cm_norm[i,j]:.2f}",
                    ha='center', va='center',
                    color='white' if cm_norm[i,j] > 0.5 else 'black',
                    fontsize=9)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(f"{title}\nN={len(y_true)}  acc={(y_true==y_pred).mean():.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path,
                     default=Path('runs/sweep_raw/tcn_raw__multi__w1s'))
    ap.add_argument('--out-dir', type=Path, default=None,
                     help='Default: <run-dir>/plots')
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or run_dir / 'plots').resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_label = run_dir.name
    print(f"[plot] run_dir = {run_dir}")
    print(f"[plot] out_dir = {out_dir}")

    # ── Loss curves
    p2_root = run_dir / 'phase2_seeds'
    histories = collect_histories(p2_root)
    print(f"[plot] found {len(histories)} history.json files")
    curves = aggregate_curves(histories)
    plot_loss_curves(curves, out_dir / 'loss_curves.png', run_label)

    # ── Class names
    meta_path = next(p2_root.rglob('dataset_meta.json'), None)
    meta = json.loads(meta_path.read_text()) if meta_path else {}
    ex_classes = meta.get('exercise_classes', ['0','1','2','3'])
    ph_classes = meta.get('phase_classes',    ['0','1','2'])
    print(f"[plot] exercise_classes = {ex_classes}")
    print(f"[plot] phase_classes    = {ph_classes}")

    # ── Confusion matrices
    pred_list = collect_preds(p2_root)
    print(f"[plot] found {len(pred_list)} test_preds.pt files")

    yt, yp = aggregate_preds(pred_list, 'exercise')
    plot_cm(yt, yp, ex_classes, out_dir / 'cm_exercise.png',
            f"Exercise — {run_label}")

    yt, yp = aggregate_preds(pred_list, 'phase')
    plot_cm(yt, yp, ph_classes, out_dir / 'cm_phase.png',
            f"Phase — {run_label}")

    print("[plot] done.")


if __name__ == '__main__':
    main()
