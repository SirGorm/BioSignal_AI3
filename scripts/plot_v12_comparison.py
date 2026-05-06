"""V12 visualization: heatmaps (window × arch per task) + per-subject fatigue
+ deployment summary plots. Mirrors plot_v9_comparison.py but for v12 (3
windows: 1, 2, 5 s) trained with the new equal-weight Optuna score.

Outputs to runs/comparison_v12/.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "comparison_v12"
OUT.mkdir(parents=True, exist_ok=True)

WINDOWS = ['1s', '2s', '5s']
WIN_LABELS = ['1.0', '2.0', '5.0']
# Multi-task only — fatigue-only single-task runs are reported separately in
# the per-subject and v13 plots. They would create empty rows for exercise/
# phase/reps since those tasks aren't trained.
ARCHS = [
    'multi-feat-mlp', 'multi-feat-lstm', 'multi-raw-cnn1d', 'multi-raw-lstm',
    'multi-raw-cnn_lstm', 'multi-raw-tcn',
]
ARCH_LABELS = ['feat-MLP', 'feat-LSTM', 'raw-cnn1d', 'raw-lstm',
                'raw-cnn-lstm', 'raw-tcn']
# fatigue-only configs: kept for the per-subject and fatigue-r-vs-window plots
FATIGUE_ARCHS = ['fatigue-raw-tcn', 'fatigue-raw-lstm']
FATIGUE_ARCH_LABELS = ['fatigue-tcn', 'fatigue-lstm']


def load_results(arch_slugs=None):
    out = {}
    for slug in (arch_slugs or ARCHS):
        out[slug] = {}
        for w in WINDOWS:
            rd = ROOT / 'runs' / f'optuna_clean_v12eqw-w{w}-{slug}'
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
            txt = value_fmt.format(v)
            if std_matrix is not None and not np.isnan(std_matrix[i, j]):
                txt = f'{txt}\n± {std_matrix[i, j]:.3f}'
            ax.text(j, i, txt, ha='center', va='center',
                     color='black', fontsize=8)
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

    def _g(d, *keys):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur: return np.nan
            cur = cur[k]
        return cur if cur is not None else np.nan

    for i, slug in enumerate(ARCHS):
        for j, w in enumerate(WINDOWS):
            m = results[slug].get(w)
            if m is None: continue
            ex_f1[i, j]    = _g(m, 'exercise', 'f1_macro', 'mean')
            ph_f1[i, j]    = _g(m, 'phase',    'f1_macro', 'mean')
            fat_mae[i, j]  = _g(m, 'fatigue',  'mae',      'mean')
            fat_r[i, j]    = _g(m, 'fatigue',  'pearson_r','mean')
            reps_mae[i, j] = _g(m, 'reps',     'mae',      'mean')
            ex_f1_s[i, j]    = _g(m, 'exercise', 'f1_macro', 'std')
            ph_f1_s[i, j]    = _g(m, 'phase',    'f1_macro', 'std')
            fat_mae_s[i, j]  = _g(m, 'fatigue',  'mae',      'std')
            fat_r_s[i, j]    = _g(m, 'fatigue',  'pearson_r','std')
            reps_mae_s[i, j] = _g(m, 'reps',     'mae',      'std')

    # Combined 2×2 panel + one standalone PNG per task.
    panels = [
        ('exercise_f1',  ex_f1,    ex_f1_s,    'Exercise F1-macro',                     '{:.3f}',  'Blues',   'max'),
        ('phase_f1',     ph_f1,    ph_f1_s,    'Phase F1-macro',                        '{:.3f}',  'Greens',  'max'),
        ('fatigue_mae',  fat_mae,  fat_mae_s,  'Fatigue MAE',                           '{:.3f}',  'YlOrRd_r','min'),
        ('fatigue_r',    fat_r,    fat_r_s,    'Fatigue Pearson r',                     '{:+.2f}', 'RdYlGn',  'max'),
        ('reps_mae',     reps_mae, reps_mae_s, 'Reps MAE',                              '{:.3f}',  'YlOrRd_r','min'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (_, mat, std, title, fmt, cmap, best) in zip(axes.flat, panels[:4]):
        heatmap(ax, mat, ARCH_LABELS, WIN_LABELS, title, fmt, cmap, best, std_matrix=std)
    fig.suptitle('V12 — window × arch (50 trials, equal-weight score, 300 P2 epochs)',
                  fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / 'heatmap_4tasks.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT / "heatmap_4tasks.png"}')

    # One PNG per task — easier to drop into reports/slides individually.
    for name, mat, std, title, fmt, cmap, best in panels:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        heatmap(ax, mat, ARCH_LABELS, WIN_LABELS, title, fmt, cmap, best, std_matrix=std)
        fig.tight_layout()
        out_path = OUT / f'heatmap_{name}.png'
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {out_path}')

    # Per-subject fatigue r for fatigue-raw-tcn @ 3s vs RF
    import torch
    from collections import defaultdict
    FOLD_SUBJECTS = {
        0: 'Vivian', 1: 'Hytten', 2: 'kiyomi', 3: 'lucas 2',
        4: 'Tias', 5: 'Juile', 6: 'Raghild',
    }
    rd = ROOT / 'runs/optuna_clean_v12eqw-w2s-fatigue-raw-tcn/phase2'
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
    bars_nn = ax.bar(x - width/2, nn_vals, width, label='fatigue-raw-tcn @ 2s (v12)',
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

    # Best window per arch, fatigue r — include fatigue-only single-task too
    fat_results = load_results(arch_slugs=FATIGUE_ARCHS)
    all_archs = list(ARCHS) + list(FATIGUE_ARCHS)
    all_labels = list(ARCH_LABELS) + list(FATIGUE_ARCH_LABELS)
    merged = {**results, **fat_results}
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, slug in enumerate(all_archs):
        ys = [merged[slug].get(w, {}).get('fatigue',{}).get('pearson_r',{}).get('mean', np.nan)
              for w in WINDOWS]
        ax.plot([1.0, 2.0, 5.0], ys, '-o', label=all_labels[i], linewidth=1.5)
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
