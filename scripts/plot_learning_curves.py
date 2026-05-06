"""Plot per-epoch learning curves for any v9/v12/v13 run, aggregated across
all (seed, fold) histories. Helps diagnose convergence vs overfitting.

Usage:
    python scripts/plot_learning_curves.py --run-dir runs/optuna_clean_v12eqw-w1s-multi-feat-mlp
    python scripts/plot_learning_curves.py --run-dir runs/optuna_clean_v13single-phase-only-w2s-feat-mlp

Outputs (under <run-dir>/plots/):
    learning_curve_total.png       — train + val total loss
    learning_curve_per_task.png    — per-task train/val loss + val metric
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def collect_histories(run_dir: Path):
    """Walk phase2/.../seed_*/fold_*/history.json and group by epoch index.

    Returns dict[task_or_total][stat] -> dict[epoch] -> list of values.
    Where stat in ('train_loss', 'val_loss', 'val_metric').
    Tasks: 'total', 'exercise', 'phase', 'fatigue', 'reps'.
    """
    p2 = run_dir / 'phase2'
    if not p2.exists():
        sys.exit(f'No phase2 dir: {p2}')

    # data[task][stat][epoch] = list of values
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    n_folds = 0
    for hist_path in p2.rglob('history.json'):
        try:
            hist = json.loads(hist_path.read_text())
        except Exception:
            continue
        n_folds += 1
        for entry in hist:
            ep = entry['epoch']
            for task in ('total', 'exercise', 'phase', 'fatigue', 'reps'):
                tr = entry.get('train', {}).get(task)
                if tr is not None:
                    data[task]['train_loss'][ep].append(tr)
                vl = entry.get('val_loss', {}).get(task)
                if vl is not None:
                    data[task]['val_loss'][ep].append(vl)
            vm = entry.get('val_metrics', {})
            # val metric per task: F1 for exercise/phase, MAE for fatigue/reps,
            #                      pearson_r for fatigue (extra)
            for task, metric in [('exercise', 'f1_macro'),
                                  ('phase', 'f1_macro'),
                                  ('fatigue', 'mae'),
                                  ('reps', 'mae')]:
                v = vm.get(task, {}).get(metric)
                if v is not None:
                    data[task]['val_metric'][ep].append(v)
            # Also track fatigue r so we can plot it explicitly
            r = vm.get('fatigue', {}).get('pearson_r')
            if r is not None:
                data['fatigue']['val_r'][ep].append(r)
    return data, n_folds


def stat_curve(per_epoch: dict):
    """Convert dict[ep] -> list to (epochs, mean, std) sorted by epoch."""
    if not per_epoch:
        return np.array([]), np.array([]), np.array([])
    eps = sorted(per_epoch.keys())
    mean = np.array([np.mean(per_epoch[e]) for e in eps])
    std = np.array([np.std(per_epoch[e]) for e in eps])
    return np.array(eps), mean, std


def fill_band(ax, x, y, yerr, color, label, ls='-'):
    ax.plot(x, y, ls, color=color, label=label, linewidth=1.7)
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.15)


def plot_total(data, run_dir: Path, out: Path, n_folds: int):
    fig, ax = plt.subplots(figsize=(9, 5))
    for stat, color, label, ls in [
        ('train_loss', '#2c3e50', 'Train (total)', '-'),
        ('val_loss',   '#e74c3c', 'Val (total)',   '--'),
    ]:
        x, y, e = stat_curve(data['total'][stat])
        if len(x):
            fill_band(ax, x, y, e, color, label, ls)
    # Mark min-val epoch (best checkpoint)
    val_x, val_y, _ = stat_curve(data['total']['val_loss'])
    if len(val_x):
        best_idx = int(np.argmin(val_y))
        ax.axvline(val_x[best_idx], color='#e74c3c', linewidth=0.8,
                    linestyle=':', alpha=0.6)
        ax.text(val_x[best_idx], ax.get_ylim()[1] * 0.95,
                 f'  min val @ epoch {val_x[best_idx]}', color='#e74c3c',
                 fontsize=9, va='top')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total loss')
    ax.set_title(f'{run_dir.name}\nLearning curve (mean ± std across {n_folds} fold-runs)',
                  fontsize=11)
    ax.legend()
    ax.grid(linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def plot_per_task(data, run_dir: Path, out: Path, n_folds: int):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    tasks = ['exercise', 'phase', 'fatigue', 'reps']
    metric_label = {'exercise': 'F1-macro', 'phase': 'F1-macro',
                    'fatigue': 'MAE', 'reps': 'MAE'}
    metric_better = {'exercise': 'higher', 'phase': 'higher',
                      'fatigue': 'lower',  'reps': 'lower'}

    for col, task in enumerate(tasks):
        # Top: per-task loss (train + val)
        ax = axes[0, col]
        x, y, e = stat_curve(data[task]['train_loss'])
        if len(x):
            fill_band(ax, x, y, e, '#2c3e50', f'Train {task}', '-')
        x, y, e = stat_curve(data[task]['val_loss'])
        if len(x):
            fill_band(ax, x, y, e, '#e74c3c', f'Val {task}', '--')
        ax.set_xlabel('Epoch'); ax.set_ylabel(f'{task} loss')
        ax.set_title(f'{task} loss', fontsize=11)
        ax.grid(linestyle=':', alpha=0.5); ax.legend(fontsize=9)
        # Bottom: val metric
        ax = axes[1, col]
        x, y, e = stat_curve(data[task]['val_metric'])
        if len(x):
            fill_band(ax, x, y, e, '#27ae60',
                       f'Val {metric_label[task]}', '-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{metric_label[task]} ({metric_better[task]} = better)')
        ax.set_title(f'{task} val metric', fontsize=11)
        ax.grid(linestyle=':', alpha=0.5); ax.legend(fontsize=9)

    # Add fatigue r overlay on the fatigue val-metric subplot
    if 'val_r' in data['fatigue']:
        ax = axes[1, 2]
        x, y, e = stat_curve(data['fatigue']['val_r'])
        if len(x):
            ax2 = ax.twinx()
            fill_band(ax2, x, y, e, '#9b59b6', 'Pearson r', ':')
            ax2.axhline(0, color='#9b59b6', linewidth=0.5, alpha=0.5)
            ax2.set_ylabel('Pearson r', color='#9b59b6')
            ax2.tick_params(axis='y', labelcolor='#9b59b6')
            ax2.legend(loc='lower right', fontsize=8)

    fig.suptitle(f'{run_dir.name} — per-task learning curves '
                  f'(mean ± std, {n_folds} folds)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, default=None,
                   help='Output dir (defaults to <run-dir>/plots/)')
    args = p.parse_args()

    out_dir = args.out or (args.run_dir / 'plots')
    out_dir.mkdir(parents=True, exist_ok=True)

    data, n_folds = collect_histories(args.run_dir)
    if n_folds == 0:
        sys.exit(f'No history.json files under {args.run_dir / "phase2"}')
    print(f'Aggregated {n_folds} fold-runs from {args.run_dir.name}')

    plot_total(data, args.run_dir, out_dir / 'learning_curve_total.png',
                n_folds)
    plot_per_task(data, args.run_dir, out_dir / 'learning_curve_per_task.png',
                   n_folds)


if __name__ == '__main__':
    main()
