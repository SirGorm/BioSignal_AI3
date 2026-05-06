"""V9 visualization: heatmaps (window × arch per task) + per-subject fatigue
+ deployment summary plots.

Outputs to runs/comparison_v9/.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "comparison_v9"
OUT.mkdir(parents=True, exist_ok=True)

WINDOWS = ['1s', '2s', '3s', '4s', '5s']
WIN_LABELS = ['1.0', '2.0', '3.0', '4.0', '5.0']
ARCHS = [
    'multi-feat-mlp', 'multi-feat-lstm', 'multi-raw-cnn1d', 'multi-raw-lstm',
    'multi-raw-cnn_lstm', 'multi-raw-tcn', 'fatigue-raw-tcn', 'fatigue-raw-lstm',
]
ARCH_LABELS = ['feat-MLP', 'feat-LSTM', 'raw-cnn1d', 'raw-lstm',
                'raw-cnn-lstm', 'raw-tcn', 'fatigue-tcn', 'fatigue-lstm']


def load_results():
    out = {}
    for slug in ARCHS:
        out[slug] = {}
        for w in WINDOWS:
            rd = ROOT / 'runs' / f'optuna_clean_v9-w{w}-{slug}'
            cv = next(iter((rd/'phase2').rglob('cv_summary.json')), None) if rd.exists() else None
            if cv is None: continue
            out[slug][w] = json.loads(cv.read_text())['summary']
    return out


def heatmap(ax, matrix, row_labels, col_labels, title, value_fmt='{:.2f}',
             cmap='viridis', highlight_best='max', std_matrix=None):
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Window (s)')
    # Find best cell (max or min)
    if highlight_best == 'max':
        best_idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    else:
        best_idx = np.unravel_index(np.nanargmin(matrix), matrix.shape)
    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, '—', ha='center', va='center', color='grey')
                continue
            color = 'white' if v < (np.nanmin(matrix) + 0.5*(np.nanmax(matrix)-np.nanmin(matrix))) else 'black'
            txt = value_fmt.format(v)
            if std_matrix is not None and not np.isnan(std_matrix[i, j]):
                txt = f'{txt}\n± {std_matrix[i, j]:.3f}'
            ax.text(j, i, txt, ha='center', va='center',
                     color=color, fontsize=7)
    # Highlight best
    ax.add_patch(plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1,
                                  fill=False, edgecolor='gold', linewidth=3))
    plt.colorbar(im, ax=ax)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    results = load_results()

    # Build matrices for 4 tasks (rows=archs, cols=windows)
    n_archs = len(ARCHS); n_wins = len(WINDOWS)
    ex_f1 = np.full((n_archs, n_wins), np.nan)
    ph_f1 = np.full((n_archs, n_wins), np.nan)
    fat_mae = np.full((n_archs, n_wins), np.nan)
    fat_r = np.full((n_archs, n_wins), np.nan)
    reps_mae = np.full((n_archs, n_wins), np.nan)
    ex_f1_s = np.full((n_archs, n_wins), np.nan)
    ph_f1_s = np.full((n_archs, n_wins), np.nan)
    fat_mae_s = np.full((n_archs, n_wins), np.nan)
    fat_r_s = np.full((n_archs, n_wins), np.nan)
    reps_mae_s = np.full((n_archs, n_wins), np.nan)

    for i, slug in enumerate(ARCHS):
        for j, w in enumerate(WINDOWS):
            m = results[slug].get(w)
            if m is None: continue
            ex_f1[i, j] = m['exercise']['f1_macro']['mean']
            ph_f1[i, j] = m['phase']['f1_macro']['mean']
            fat_mae[i, j] = m['fatigue']['mae']['mean']
            fat_r[i, j] = m['fatigue']['pearson_r']['mean']
            reps_mae[i, j] = m['reps']['mae']['mean']
            ex_f1_s[i, j] = m['exercise']['f1_macro'].get('std', np.nan)
            ph_f1_s[i, j] = m['phase']['f1_macro'].get('std', np.nan)
            fat_mae_s[i, j] = m['fatigue']['mae'].get('std', np.nan)
            fat_r_s[i, j] = m['fatigue']['pearson_r'].get('std', np.nan)
            reps_mae_s[i, j] = m['reps']['mae'].get('std', np.nan)

    # 2×2 panel
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    heatmap(axes[0,0], ex_f1, ARCH_LABELS, WIN_LABELS,
            'Exercise F1-macro', '{:.3f}', 'Blues', 'max', std_matrix=ex_f1_s)
    heatmap(axes[0,1], ph_f1, ARCH_LABELS, WIN_LABELS,
            'Phase F1-macro', '{:.3f}', 'Greens', 'max', std_matrix=ph_f1_s)
    heatmap(axes[1,0], fat_mae, ARCH_LABELS, WIN_LABELS,
            'Fatigue MAE', '{:.3f}', 'YlOrRd_r', 'min', std_matrix=fat_mae_s)
    heatmap(axes[1,1], fat_r, ARCH_LABELS, WIN_LABELS,
            'Fatigue Pearson r', '{:+.2f}', 'RdYlGn', 'max', std_matrix=fat_r_s)
    fig.suptitle('V9 — window × arch (75 trials each, 300 P2 epochs / patience 20)',
                  fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = OUT / 'heatmap_4tasks.png'
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')

    # Reps standalone (different scale, soft_overlap fractional)
    fig, ax = plt.subplots(figsize=(9, 5))
    heatmap(ax, reps_mae, ARCH_LABELS, WIN_LABELS,
            'Reps MAE',
            '{:.3f}', 'YlOrRd_r', 'min', std_matrix=reps_mae_s)
    fig.tight_layout()
    fig.savefig(OUT / 'heatmap_reps.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT/"heatmap_reps.png"}')

    # Per-subject fatigue r for fatigue-raw-tcn @ 3s vs RF
    import torch
    from collections import defaultdict
    FOLD_SUBJECTS = {
        0: 'Vivian', 1: 'Hytten', 2: 'kiyomi', 3: 'lucas 2',
        4: 'Tias', 5: 'Juile', 6: 'Raghild',
    }
    rd = ROOT / 'runs/optuna_clean_v9-w3s-fatigue-raw-tcn/phase2'
    by_fold_r = defaultdict(list)
    for seed_dir in rd.glob('*/seed_*'):
        for fold_dir in sorted(seed_dir.glob('fold_*')):
            fk = int(fold_dir.name.split('_')[1])
            try:
                d = torch.load(fold_dir / 'test_preds.pt', weights_only=False, map_location='cpu')
            except Exception:
                continue
            mask = d['masks']['fatigue'].numpy().astype(bool)
            yp = d['preds']['fatigue'].numpy()[mask]
            yt = d['targets']['fatigue'].numpy()[mask]
            if len(yp) > 1 and np.std(yp) > 0:
                by_fold_r[fk].append(float(np.corrcoef(yp, yt)[0,1]))

    rfm = json.loads((ROOT/'runs/optuna_clean_v9-rf/metrics.json').read_text())
    rf_per_subj = rfm['fatigue']['pearson_r_per_subj']

    subjects = list(FOLD_SUBJECTS.values())
    nn_vals = [np.mean(by_fold_r[i]) if by_fold_r[i] else 0 for i in range(7)]
    rf_vals = [rf_per_subj.get(s, 0) for s in subjects]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(subjects))
    width = 0.38
    bars_nn = ax.bar(x - width/2, nn_vals, width, label='fatigue-raw-tcn @ 3s',
                      color='#27ae60', edgecolor='black')
    bars_rf = ax.bar(x + width/2, rf_vals, width, label='RF baseline',
                      color='#7f8c8d', edgecolor='black')
    for b, v in zip(bars_nn, nn_vals):
        ax.text(b.get_x()+b.get_width()/2, v + (0.02 if v>0 else -0.05),
                 f'{v:+.2f}', ha='center', fontsize=9)
    for b, v in zip(bars_rf, rf_vals):
        ax.text(b.get_x()+b.get_width()/2, v + (0.02 if v>0 else -0.05),
                 f'{v:+.2f}', ha='center', fontsize=9, color='#555')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(subjects, fontsize=10)
    ax.set_ylabel('Pearson r')
    ax.set_title('Fatigue Pearson r per subject — best NN vs RF baseline (LOSO)')
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / 'fatigue_per_subject.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT/"fatigue_per_subject.png"}')

    # Best window per arch, fatigue r
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, slug in enumerate(ARCHS):
        ys = [results[slug].get(w, {}).get('fatigue',{}).get('pearson_r',{}).get('mean', np.nan)
              for w in WINDOWS]
        ax.plot([1.0, 2.0, 3.0, 4.0, 5.0], ys, '-o', label=ARCH_LABELS[i], linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Window length (s)')
    ax.set_ylabel('Fatigue Pearson r (3 seeds × 5 folds)')
    ax.set_title('Fatigue Pearson r vs window length, per architecture')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / 'fatigue_r_vs_window.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT/"fatigue_r_vs_window.png"}')

    print(f'\nAll plots in {OUT}')


if __name__ == '__main__':
    main()
