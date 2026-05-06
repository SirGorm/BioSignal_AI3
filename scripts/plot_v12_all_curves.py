"""V12 all-models learning curves on shared axes.

Each curve is the mean train (or val) loss per epoch averaged across the
21 fold-runs for that (arch, window) combination. Color identifies the
architecture; line style identifies the window length.

Generates one figure per architecture group:
  group=all       all 24 (multi + fatigue-only)
  group=feat      multi-feat-mlp + multi-feat-lstm
  group=multi_raw multi-raw-cnn1d / lstm / cnn_lstm / tcn
  group=fatigue   fatigue-raw-tcn + fatigue-raw-lstm

Outputs to runs/comparison_v12/:
  all_curves_<group>_total.png      — total loss, 2 panels (train, val)
  all_curves_<group>_per_task.png   — per-task loss, 4 rows × 2 cols
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "comparison_v12"
OUT.mkdir(parents=True, exist_ok=True)

WINDOWS = ['1s', '2s', '5s']
LINESTYLES = {'1s': '-', '2s': '--', '5s': ':'}
# Single source of truth: arch slug -> (label, color)
ARCH_INFO = {
    'multi-feat-mlp':     ('feat-MLP',     '#1f77b4'),
    'multi-feat-lstm':    ('feat-LSTM',    '#ff7f0e'),
    'multi-raw-cnn1d':    ('raw-cnn1d',    '#2ca02c'),
    'multi-raw-lstm':     ('raw-lstm',     '#d62728'),
    'multi-raw-cnn_lstm': ('raw-cnn-lstm', '#9467bd'),
    'multi-raw-tcn':      ('raw-tcn',      '#8c564b'),
    'fatigue-raw-tcn':    ('fatigue-tcn',  '#e377c2'),
    'fatigue-raw-lstm':   ('fatigue-lstm', '#7f7f7f'),
}
ALL_ARCHS = list(ARCH_INFO.keys())

GROUPS = {
    'all':       ALL_ARCHS,
    'feat':      ['multi-feat-mlp', 'multi-feat-lstm'],
    'multi_raw': ['multi-raw-cnn1d', 'multi-raw-lstm',
                  'multi-raw-cnn_lstm', 'multi-raw-tcn'],
    'fatigue':   ['fatigue-raw-tcn', 'fatigue-raw-lstm'],
}

TASKS = ['total', 'exercise', 'phase', 'fatigue', 'reps']


def collect_run_curves(run_dir: Path):
    """Return dict[task][stat][epoch] = mean loss across all fold-histories."""
    p2 = run_dir / 'phase2'
    if not p2.exists():
        return None
    by_epoch = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    n = 0
    for hp in p2.rglob('history.json'):
        try:
            hist = json.loads(hp.read_text())
        except Exception:
            continue
        n += 1
        for entry in hist:
            ep = entry['epoch']
            for task in TASKS:
                tr = entry.get('train', {}).get(task)
                if tr is not None:
                    by_epoch[task]['train'][ep].append(tr)
                vl = entry.get('val_loss', {}).get(task)
                if vl is not None:
                    by_epoch[task]['val'][ep].append(vl)
    if n == 0:
        return None
    out = {}
    for task in TASKS:
        out[task] = {}
        for stat in ('train', 'val'):
            d = by_epoch[task][stat]
            if not d:
                out[task][stat] = (np.array([]), np.array([]))
                continue
            eps = sorted(d.keys())
            mean = np.array([np.mean(d[e]) for e in eps])
            out[task][stat] = (np.array(eps), mean)
    return out


def plot_panel(ax, all_curves, archs, task: str, stat: str, title: str):
    for slug in archs:
        label, color = ARCH_INFO[slug]
        for w in WINDOWS:
            curve = all_curves.get((slug, w))
            if curve is None: continue
            x, y = curve[task][stat]
            if not len(x): continue
            ax.plot(x, y, color=color, linestyle=LINESTYLES[w], linewidth=1.6,
                     alpha=0.9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{task} {stat} loss')
    ax.set_title(title, fontsize=11)
    ax.grid(linestyle=':', alpha=0.4)


def add_legend(ax_for_legend, archs):
    """Two legends: arch (color) and window (line style)."""
    arch_handles = [plt.Line2D([], [], color=ARCH_INFO[s][1], linewidth=2,
                                 label=ARCH_INFO[s][0])
                     for s in archs]
    win_handles = [plt.Line2D([], [], color='black', linewidth=2,
                               linestyle=LINESTYLES[w], label=f'{w} window')
                    for w in WINDOWS]
    leg1 = ax_for_legend.legend(handles=arch_handles, loc='upper left',
                                  bbox_to_anchor=(1.01, 1.0), fontsize=9,
                                  title='Architecture')
    ax_for_legend.add_artist(leg1)
    ax_for_legend.legend(handles=win_handles, loc='upper left',
                          bbox_to_anchor=(1.01, 0.4), fontsize=9,
                          title='Window')


def render_group(group_name: str, archs: list, all_curves: dict):
    n_runs = sum(1 for s in archs for w in WINDOWS if (s, w) in all_curves)
    title_prefix = f'V12 {group_name} ({n_runs} runs)'

    # 1) Total loss: train + val
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_panel(axes[0], all_curves, archs, 'total', 'train', 'Train total loss')
    plot_panel(axes[1], all_curves, archs, 'total', 'val',   'Val total loss')
    add_legend(axes[1], archs)
    fig.suptitle(f'{title_prefix} — total loss per epoch '
                  '(mean across 21 fold-runs)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.85, 0.96))
    out = OUT / f'all_curves_{group_name}_total.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')

    # 2) Per-task: 4 tasks × (train, val) = 8 panels
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    for row, task in enumerate(['exercise', 'phase', 'fatigue', 'reps']):
        plot_panel(axes[row, 0], all_curves, archs, task, 'train',
                    f'{task} — train')
        plot_panel(axes[row, 1], all_curves, archs, task, 'val',
                    f'{task} — val')
    add_legend(axes[0, 1], archs)
    fig.suptitle(f'{title_prefix} — per-task loss per epoch '
                  '(mean across 21 fold-runs)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.88, 0.97))
    out = OUT / f'all_curves_{group_name}_per_task.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', nargs='*',
                     default=['all', 'feat', 'multi_raw', 'fatigue'],
                     choices=list(GROUPS.keys()),
                     help='Which arch groups to render (default: all 4).')
    args = ap.parse_args()

    # Collect curves once for all archs we might need
    needed_archs = set()
    for g in args.group:
        needed_archs.update(GROUPS[g])
    all_curves = {}
    for slug in needed_archs:
        for w in WINDOWS:
            rd = ROOT / 'runs' / f'optuna_clean_v12eqw-w{w}-{slug}'
            c = collect_run_curves(rd)
            if c is not None:
                all_curves[(slug, w)] = c
    print(f'Loaded {len(all_curves)} runs total\n')

    for g in args.group:
        print(f'--- group: {g} ---')
        render_group(g, GROUPS[g], all_curves)


if __name__ == '__main__':
    main()
