"""V13 plots: single-task vs multi-task comparison + per-task best.

Outputs to runs/comparison_v13/:
  bar_v13_vs_v12.png         — single-task best vs multi-task best per task
  reps_scatter_best.png      — predicted vs actual reps for v13 best reps
  reps_scatter_v12_best.png  — same for v12 multi-task best reps
  cm_phase_v13_best.png      — confusion matrix for v13 best phase
  cm_exercise_v13_best.png   — confusion matrix for v13 best exercise
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (confusion_matrix as sk_cm,
                              accuracy_score, balanced_accuracy_score, f1_score)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "comparison_v13"
OUT.mkdir(parents=True, exist_ok=True)

# Best-per-task winners.
WINNERS = {
    'phase':    ('v13', 'optuna_clean_v13single-phase-only-w2s-feat-mlp', 0.554),
    'exercise': ('v13', 'optuna_clean_v13single-exercise-only-w5s-feat-mlp', 0.398),
    'fatigue_r':('v12', 'optuna_clean_v12eqw-w2s-fatigue-raw-tcn', 0.429),
    'reps':     ('v13', 'optuna_clean_v13single-reps-only-w1s-feat-lstm', 0.005),
}
V12_BEST = {
    'phase':    ('optuna_clean_v12eqw-w2s-multi-feat-mlp', 0.501),
    'exercise': ('optuna_clean_v12eqw-w5s-multi-feat-mlp', 0.368),
    'fatigue_r':('optuna_clean_v12eqw-w5s-multi-raw-cnn_lstm', 0.363),
    'reps':     ('optuna_clean_v12eqw-w1s-multi-feat-lstm', 0.110),
}


def load_preds(run_dir: Path, task: str):
    """Aggregate (preds, targets) across all (seed, fold) test_preds.pt."""
    p2 = run_dir / 'phase2'
    yp_all, yt_all = [], []
    n_files = 0
    for fd in p2.glob('*/seed_*/fold_*'):
        try:
            d = torch.load(fd / 'test_preds.pt', weights_only=False, map_location='cpu')
        except Exception:
            continue
        n_files += 1
        m = d['masks'].get(task)
        if m is None:
            continue
        m = m.numpy().astype(bool)
        if not m.any():
            continue
        yp = d['preds'][task].numpy()
        yt = d['targets'][task].numpy()
        if task in ('exercise', 'phase') and yp.ndim == 2:
            yp = yp.argmax(axis=1)
        yp_all.append(yp[m]); yt_all.append(yt[m])
    if not yp_all:
        return None, None, n_files
    return np.concatenate(yp_all), np.concatenate(yt_all), n_files


def plot_bar_comparison():
    tasks = ['phase', 'exercise', 'fatigue_r', 'reps']
    labels = ['Phase F1', 'Exercise F1', 'Fatigue r', 'Reps MAE']
    v13_v = [WINNERS[t][2] for t in tasks]
    v12_v = [V12_BEST[t][1] for t in tasks]
    # For reps MAE we want lower-is-better — flip sign for visual consistency.
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    for ax, task, label in zip(axes, tasks, labels):
        sl_v13 = WINNERS[task][2]
        sl_v12 = V12_BEST[task][1]
        bars = ax.bar(['multi-task', 'single-task'],
                       [sl_v12, sl_v13],
                       color=['#7f8c8d', '#27ae60'], edgecolor='black')
        for b, v in zip(bars, [sl_v12, sl_v13]):
            ax.text(b.get_x()+b.get_width()/2, v,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        if 'MAE' in label:
            ax.invert_yaxis()
            ax.set_ylabel('MAE (lower = better)')
        else:
            ax.set_ylabel('higher = better')
    fig.suptitle('Single-task vs multi-task — best per task',
                  fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / 'bar_v13_vs_v12.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def plot_reps_scatter(run_dir: Path, label: str, out_path: Path,
                        color='#27ae60'):
    yp, yt, _ = load_preds(run_dir, 'reps')
    if yp is None:
        print(f'  No reps preds for {run_dir.name}')
        return
    mae = float(np.mean(np.abs(yp - yt)))
    r = float(np.corrcoef(yp, yt)[0, 1]) if np.std(yp) > 0 and np.std(yt) > 0 else 0.0
    fig, ax = plt.subplots(figsize=(7, 7))
    # Hexbin for density, since N can be ~30k+
    hb = ax.hexbin(yt, yp, gridsize=40, mincnt=1, cmap='YlGnBu')
    plt.colorbar(hb, ax=ax, label='count')
    lim_max = float(max(yt.max(), yp.max()) * 1.05) if len(yt) else 1.0
    ax.plot([0, lim_max], [0, lim_max], 'r--', alpha=0.6, label='y=x')
    ax.set_xlabel('Actual reps (soft_overlap fraction)')
    ax.set_ylabel('Predicted reps')
    ax.set_xlim(0, lim_max); ax.set_ylim(0, lim_max)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.set_title(f'{label}\nMAE={mae:.4f}  Pearson r={r:+.4f}  N={len(yt)}',
                  fontsize=11)
    ax.grid(linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}  (MAE={mae:.4f}, r={r:+.4f})')


def plot_cm(run_dir: Path, task: str, classes, label: str, out_path: Path,
             cmap='Blues'):
    yp, yt, _ = load_preds(run_dir, task)
    if yp is None:
        print(f'  No {task} preds for {run_dir.name}')
        return
    present = sorted(set(yt.tolist()) | set(yp.tolist()))
    cls_labels = [classes[i] if i < len(classes) else f'cls_{i}' for i in present]
    cm = sk_cm(yt, yp, labels=present)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    acc = accuracy_score(yt, yp)
    bal = balanced_accuracy_score(yt, yp)
    f1m = f1_score(yt, yp, labels=present, average='macro', zero_division=0)
    fig, ax = plt.subplots(figsize=(7, 6.4))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=max(cm_norm.max(), 0.01))
    for i in range(len(cls_labels)):
        for j in range(len(cls_labels)):
            ax.text(j, i, f'{cm_norm[i, j]:.2f}\n({cm[i, j]})',
                     ha='center', va='center', fontsize=9, color='black')
    ax.set_xticks(range(len(cls_labels)))
    ax.set_xticklabels(cls_labels, rotation=30, ha='right')
    ax.set_yticks(range(len(cls_labels)))
    ax.set_yticklabels(cls_labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{label}\nAcc={acc:.3f}  Bal acc={bal:.3f}  '
                 f'Macro-F1={f1m:.3f}  N={cm.sum()}', fontsize=10)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}  (acc={acc:.3f}, F1={f1m:.3f})')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    plot_bar_comparison()

    # Reps scatter — v13 best vs v12 best
    plot_reps_scatter(ROOT / 'runs' / WINNERS['reps'][1],
                       'Reps — single-task best (feat-LSTM @ 1s)',
                       OUT / 'reps_scatter_v13_best.png')
    plot_reps_scatter(ROOT / 'runs' / V12_BEST['reps'][0],
                       'Reps — multi-task best (feat-LSTM @ 1s)',
                       OUT / 'reps_scatter_v12_best.png')

    # CMs for v13 single-task winners
    PH_CLASSES = ['concentric', 'eccentric']
    EX_CLASSES = ['benchpress', 'deadlift', 'pullup', 'squat']
    plot_cm(ROOT / 'runs' / WINNERS['phase'][1], 'phase', PH_CLASSES,
             'Phase CM — single-task best (feat-MLP @ 2s)',
             OUT / 'cm_phase_v13_best.png', cmap='Greens')
    plot_cm(ROOT / 'runs' / WINNERS['exercise'][1], 'exercise', EX_CLASSES,
             'Exercise CM — single-task best (feat-MLP @ 5s)',
             OUT / 'cm_exercise_v13_best.png', cmap='Blues')


if __name__ == '__main__':
    main()
