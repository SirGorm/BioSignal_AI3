"""For a single run, plot total + all 4 head losses (train + val) on one panel.

Each curve is the mean over all (seed, fold) histories per epoch.
Color by head, line style by train/val:
  total       black     solid=train   dashed=val
  exercise    blue
  phase       green
  fatigue     red
  reps        orange

Usage:
    python scripts/plot_all_heads_combined.py --run-dir runs/optuna_clean_v12eqw-w1s-multi-feat-mlp
    python scripts/plot_all_heads_combined.py --run-dir runs/... --out plots/myplot.png
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

HEAD_COLORS = {
    'total':    '#2c3e50',
    'exercise': '#1f77b4',
    'phase':    '#2ca02c',
    'fatigue':  '#d62728',
    'reps':     '#ff7f0e',
}
HEADS = ['total', 'exercise', 'phase', 'fatigue', 'reps']


def collect_curves(run_dir: Path):
    """dict[head][stat][epoch] = list of values across (seed, fold)."""
    p2 = run_dir / 'phase2'
    if not p2.exists():
        sys.exit(f'No phase2 dir: {p2}')
    data = {h: {'train': defaultdict(list), 'val': defaultdict(list)}
            for h in HEADS}
    n_files = 0
    for hp in p2.rglob('history.json'):
        try:
            hist = json.loads(hp.read_text())
        except Exception:
            continue
        n_files += 1
        for entry in hist:
            ep = entry['epoch']
            for h in HEADS:
                tr = entry.get('train', {}).get(h)
                vl = entry.get('val_loss', {}).get(h)
                if tr is not None: data[h]['train'][ep].append(tr)
                if vl is not None: data[h]['val'][ep].append(vl)
    return data, n_files


def stat_curve(per_epoch: dict):
    if not per_epoch:
        return np.array([]), np.array([])
    eps = sorted(per_epoch.keys())
    return np.array(eps), np.array([np.mean(per_epoch[e]) for e in eps])


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None,
                     help='Output path (defaults to <run-dir>/plots/all_heads_combined.png)')
    ap.add_argument('--logy', action='store_true',
                     help='Use log y-axis (helpful when scales differ).')
    args = ap.parse_args()

    out = args.out or (args.run_dir / 'plots' / 'all_heads_combined.png')
    out.parent.mkdir(parents=True, exist_ok=True)

    data, n_files = collect_curves(args.run_dir)
    print(f'Aggregated {n_files} fold-histories from {args.run_dir.name}')

    fig, ax = plt.subplots(figsize=(11, 6.5))
    n_drawn = 0
    for head in HEADS:
        color = HEAD_COLORS[head]
        for stat, ls, suffix in [('train', '-', 'train'),
                                   ('val',   '--', 'val')]:
            x, y = stat_curve(data[head][stat])
            if not len(x): continue
            ax.plot(x, y, color=color, linestyle=ls, linewidth=1.7,
                     alpha=0.9, label=f'{head} {suffix}')
            n_drawn += 1
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if args.logy: ax.set_yscale('log')
    ax.set_title(f'{args.run_dir.name}\n'
                  f'All heads — train (solid) vs val (dashed), '
                  f'mean across {n_files} fold-runs', fontsize=11)
    ax.grid(linestyle=':', alpha=0.4)
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left', fontsize=9,
               ncol=1)
    fig.tight_layout(rect=(0, 0, 0.85, 1.0))
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
