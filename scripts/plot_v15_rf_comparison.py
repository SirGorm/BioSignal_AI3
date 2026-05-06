"""V15 RF baseline comparison plots — across 3 windows (1, 2, 5 s) per task.

Each task gets one bar-chart panel showing RF performance at each window plus
the corresponding best-NN result for context. Per-subject Pearson r and MAE
breakdowns are added for fatigue.

Outputs to runs/comparison_v15_rf/:
  rf_4task_summary.png         — 4 panels (exercise/phase/fatigue/reps)
  rf_fatigue_per_subject.png   — per-subject fatigue r across 3 windows
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "comparison_v15_rf"
OUT.mkdir(parents=True, exist_ok=True)

WINDOWS = ['1s', '2s', '5s']

# Best NN per task (from v13 single-task summary).
NN_BEST = {
    'exercise': ('feat-MLP @ 5s (v13 single-task)',  0.398),
    'phase':    ('feat-MLP @ 2s (v13 single-task)',  0.554),
    'fatigue':  ('fatigue-tcn @ 2s (v12)',           0.794),  # MAE
    'reps':     ('feat-LSTM @ 1s (v13 single-task)', 0.005),  # soft_overlap MAE — different scale
}


def load_rf(window: str):
    p = ROOT / 'runs' / f'optuna_clean_v15rf-w{window}' / 'metrics.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    rf = {w: load_rf(w) for w in WINDOWS}
    print('Loaded RF metrics:', {w: bool(d) for w, d in rf.items()})

    # ---- 4-panel summary ---------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = np.arange(len(WINDOWS))
    bw = 0.6

    # Pearson r median per window for fatigue (added to fatigue subplot title)
    fat_rs = [rf[w]['fatigue']['pearson_r_median'] for w in WINDOWS]
    fat_title = ('Fatigue MAE  (median r: '
                 + ', '.join(f'{w}={r:+.2f}' for w, r in zip(WINDOWS, fat_rs))
                 + ')')

    panels = [
        # (ax_pos, task_path_keys, value_key, std_key, ylabel, title, text_offset)
        ((0, 0), ['exercise'], 'f1_mean', 'f1_std', 'F1-macro',
         'Exercise F1', 0.005),
        ((0, 1), ['phase'], 'ml_f1_mean', 'ml_f1_std', 'F1-macro',
         'Phase F1', 0.005),
        ((1, 0), ['fatigue'], 'mae_mean', 'mae_std', 'MAE',
         fat_title, 0.02),
        ((1, 1), ['reps'], 'ml_mae_mean', 'ml_mae_std', 'MAE',
         'Reps MAE', 0.05),
    ]
    for pos, keys, vk, ek, ylabel, title, off in panels:
        ax = axes[pos]
        vals = [rf[w][keys[0]][vk] for w in WINDOWS]
        errs = [rf[w][keys[0]][ek] for w in WINDOWS]
        bars = ax.bar(x, vals, bw, yerr=errs, capsize=5,
                       color='#7f8c8d', edgecolor='black')
        for b, v, e in zip(bars, vals, errs):
            ax.text(b.get_x()+b.get_width()/2, v + e + off,
                     f'{v:.3f}', ha='center', fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(WINDOWS)
        ax.set_xlabel('Window'); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, max(max(vals) + max(errs), 0.001) * 1.2)
        ax.grid(axis='y', linestyle=':', alpha=0.5)

    fig.suptitle('Random Forest baseline', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT / 'rf_4task_summary.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')

    # ---- Per-subject fatigue r across 3 windows ----------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    SUBJ = ['Vivian', 'Hytten', 'kiyomi', 'lucas 2', 'Tias', 'Juile', 'Raghild']
    width = 0.25
    x = np.arange(len(SUBJ))
    colors = ['#3498db', '#e67e22', '#27ae60']
    for wi, (w, c) in enumerate(zip(WINDOWS, colors)):
        rs = [rf[w]['fatigue']['pearson_r_per_subj'].get(s, 0) for s in SUBJ]
        bars = ax.bar(x + (wi - 1) * width, rs, width, color=c,
                       edgecolor='black', label=f'RF @ {w}')
        for b, v in zip(bars, rs):
            ax.text(b.get_x()+b.get_width()/2, v + (0.02 if v >= 0 else -0.06),
                     f'{v:+.2f}', ha='center', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(SUBJ, fontsize=10)
    ax.set_ylabel('Pearson r')
    ax.set_title('RF fatigue Pearson r — per subject × window', fontsize=11)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    fig.tight_layout()
    out = OUT / 'rf_fatigue_per_subject.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
