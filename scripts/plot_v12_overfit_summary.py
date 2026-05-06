"""V12 overfit summary: heatmap of train-val gap per (arch, window).

For each v12 run, computes the gap between final train loss and final val
loss (averaged across the last 5 epochs of each fold-run, then averaged
across all 21 folds). Larger gap = more overfitting.

Outputs to runs/comparison_v12/overfit_summary.png.
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
ARCHS = [
    'multi-feat-mlp', 'multi-feat-lstm', 'multi-raw-cnn1d', 'multi-raw-lstm',
    'multi-raw-cnn_lstm', 'multi-raw-tcn', 'fatigue-raw-tcn', 'fatigue-raw-lstm',
]
ARCH_LABELS = ['feat-MLP', 'feat-LSTM', 'raw-cnn1d', 'raw-lstm',
                'raw-cnn-lstm', 'raw-tcn', 'fatigue-tcn', 'fatigue-lstm']


def gap_for_run(run_dir: Path, last_n: int = 5):
    """Mean (val_loss − train_loss) over the last `last_n` epochs of each
    fold-history, averaged across all (seed, fold) histories."""
    p2 = run_dir / 'phase2'
    if not p2.exists():
        return np.nan, np.nan, np.nan
    train_finals, val_finals = [], []
    for hp in p2.rglob('history.json'):
        try:
            hist = json.loads(hp.read_text())
        except Exception:
            continue
        if len(hist) == 0: continue
        tail = hist[-last_n:]
        tr = np.mean([e['train']['total'] for e in tail
                       if 'train' in e and 'total' in e['train']])
        vl = np.mean([e['val_loss']['total'] for e in tail
                       if 'val_loss' in e and 'total' in e['val_loss']])
        train_finals.append(tr); val_finals.append(vl)
    if not train_finals:
        return np.nan, np.nan, np.nan
    return (float(np.mean(train_finals)),
             float(np.mean(val_finals)),
             float(np.mean(np.array(val_finals) - np.array(train_finals))))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    n_arch, n_win = len(ARCHS), len(WINDOWS)
    train_m = np.full((n_arch, n_win), np.nan)
    val_m = np.full((n_arch, n_win), np.nan)
    gap_m = np.full((n_arch, n_win), np.nan)

    for i, slug in enumerate(ARCHS):
        for j, w in enumerate(WINDOWS):
            rd = ROOT / 'runs' / f'optuna_clean_v12eqw-w{w}-{slug}'
            tr, vl, gap = gap_for_run(rd)
            train_m[i, j] = tr
            val_m[i, j] = vl
            gap_m[i, j] = gap

    def heatmap(ax, mat, title, cmap, fmt='{:.3f}'):
        im = ax.imshow(mat, cmap=cmap, aspect='auto')
        ax.set_xticks(range(n_win)); ax.set_xticklabels(WINDOWS)
        ax.set_yticks(range(n_arch)); ax.set_yticklabels(ARCH_LABELS)
        ax.set_xlabel('Window')
        ax.set_title(title, fontsize=11)
        for i in range(n_arch):
            for j in range(n_win):
                v = mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, '—', ha='center', va='center', color='grey')
                else:
                    ax.text(j, i, fmt.format(v), ha='center', va='center',
                             color='black', fontsize=9)
        plt.colorbar(im, ax=ax)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    heatmap(axes[0], train_m, 'Final train loss (last 5 ep)',
             'YlGnBu')
    heatmap(axes[1], val_m,   'Final val loss (last 5 ep)',
             'YlOrRd')
    heatmap(axes[2], gap_m,   'Val − Train gap (overfit indicator)',
             'RdPu')
    fig.suptitle('V12 overfit summary — averaged over 21 fold-runs each',
                  fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / 'overfit_summary.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')

    # Also dump as a table for inspection
    print('\nTrain / Val / Gap per (arch, window):')
    print(f'{"arch":<22}  {"window":<6}  {"train":>8}  {"val":>8}  {"gap":>8}')
    for i, slug in enumerate(ARCHS):
        for j, w in enumerate(WINDOWS):
            print(f'{slug:<22}  {w:<6}  {train_m[i,j]:>8.3f}  '
                  f'{val_m[i,j]:>8.3f}  {gap_m[i,j]:>+8.3f}')


if __name__ == '__main__':
    main()
