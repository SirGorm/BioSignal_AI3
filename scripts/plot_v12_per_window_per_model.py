"""V12 learning-curve plots organized two ways:

  per_window/<w>.png       — for each window (1s, 2s, 5s), one panel showing
                              train (solid) + val (dashed) for ALL 8 archs.
  per_model/<arch>_<w>.png — for each of the 24 (arch, window) combinations,
                              one panel showing train + val for just that
                              specific configuration.

Outputs to runs/comparison_v12/.
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
PER_WIN = OUT / "per_window"
PER_MOD = OUT / "per_model"
PER_WIN.mkdir(parents=True, exist_ok=True)
PER_MOD.mkdir(parents=True, exist_ok=True)

WINDOWS = ['1s', '2s', '5s']
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
ARCHS = list(ARCH_INFO.keys())
MULTITASK_ARCHS = [s for s in ARCHS if s.startswith('multi-')]


def collect_run_curves(run_dir: Path):
    """Mean train and val total loss per epoch across all fold-histories."""
    p2 = run_dir / 'phase2'
    if not p2.exists():
        return None, None
    by_train = defaultdict(list)
    by_val = defaultdict(list)
    for hp in p2.rglob('history.json'):
        try:
            hist = json.loads(hp.read_text())
        except Exception:
            continue
        for entry in hist:
            ep = entry['epoch']
            tr = entry.get('train', {}).get('total')
            vl = entry.get('val_loss', {}).get('total')
            if tr is not None: by_train[ep].append(tr)
            if vl is not None: by_val[ep].append(vl)
    if not by_train:
        return None, None
    eps = sorted(by_train.keys())
    train_mean = np.array([np.mean(by_train[e]) for e in eps])
    val_eps = sorted(by_val.keys())
    val_mean = np.array([np.mean(by_val[e]) for e in val_eps])
    return (np.array(eps), train_mean), (np.array(val_eps), val_mean)


def plot_per_window(window: str, all_curves: dict):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    n_runs = 0
    for slug in MULTITASK_ARCHS:
        curve = all_curves.get((slug, window))
        if curve is None: continue
        n_runs += 1
        label, color = ARCH_INFO[slug]
        (tr_x, tr_y), (vl_x, vl_y) = curve
        ax.plot(tr_x, tr_y, color=color, linestyle='-',  linewidth=1.6,
                 alpha=0.9, label=f'{label} train')
        ax.plot(vl_x, vl_y, color=color, linestyle='--', linewidth=1.6,
                 alpha=0.9, label=f'{label} val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total loss')
    ax.set_title(f'V12 window={window} — train (solid) vs val (dashed) for '
                 f'{n_runs} archs (mean across 21 fold-runs)', fontsize=11)
    ax.grid(linestyle=':', alpha=0.4)
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left', fontsize=8,
               ncol=1)
    fig.tight_layout(rect=(0, 0, 0.82, 1.0))
    out = PER_WIN / f'{window}.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def plot_per_model(slug: str, window: str, all_curves: dict):
    curve = all_curves.get((slug, window))
    if curve is None: return
    label, color = ARCH_INFO[slug]
    (tr_x, tr_y), (vl_x, vl_y) = curve
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(tr_x, tr_y, color='#2c3e50', linewidth=1.8, label='Train')
    ax.plot(vl_x, vl_y, color='#e74c3c', linewidth=1.8, linestyle='--',
             label='Val')
    # Highlight min val epoch
    if len(vl_y):
        best = int(np.argmin(vl_y))
        ax.axvline(vl_x[best], color='#e74c3c', linewidth=0.8, linestyle=':',
                    alpha=0.6)
        ax.text(vl_x[best], ax.get_ylim()[1] * 0.95,
                 f'  min val @ epoch {vl_x[best]} '
                 f'({vl_y[best]:.3f})', color='#e74c3c', fontsize=9, va='top')
    final_gap = (vl_y[-5:].mean() - tr_y[-5:].mean()) if len(tr_y) >= 5 else 0
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total loss')
    ax.set_title(f'{label} @ {window} — train vs val\n'
                 f'final gap (val-train, last 5 ep) = {final_gap:+.3f}',
                  fontsize=11)
    ax.grid(linestyle=':', alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out = PER_MOD / f'{slug}_{window}.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    all_curves = {}
    for slug in ARCHS:
        for w in WINDOWS:
            rd = ROOT / 'runs' / f'optuna_clean_v12eqw-w{w}-{slug}'
            c = collect_run_curves(rd)
            if c[0] is not None:
                all_curves[(slug, w)] = c
    print(f'Loaded {len(all_curves)} runs\n')

    print('--- per-window plots ---')
    for w in WINDOWS:
        plot_per_window(w, all_curves)

    print('\n--- per-model plots ---')
    for slug in ARCHS:
        for w in WINDOWS:
            plot_per_model(slug, w, all_curves)


if __name__ == '__main__':
    main()
