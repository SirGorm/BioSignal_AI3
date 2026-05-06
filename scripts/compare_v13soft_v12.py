"""Side-by-side v12 eqw vs v13 soft comparison across the full grid.

Outputs:
  runs/comparison_v13/v12_vs_v13soft.csv     - flat numeric table
  runs/comparison_v13/v12_vs_v13soft.md      - markdown summary with deltas
  runs/comparison_v13/v12_vs_v13soft_heatmap.png  - 4-task heatmap of delta
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'runs' / 'comparison_v13'
OUT.mkdir(parents=True, exist_ok=True)

ARCHS = [
    ('multi-feat-mlp',   'feat-MLP'),
    ('multi-feat-lstm',  'feat-LSTM'),
    ('multi-raw-cnn1d',  'raw-CNN1D'),
    ('multi-raw-cnn_lstm','raw-CNN-LSTM'),
    ('multi-raw-lstm',   'raw-LSTM'),
    ('multi-raw-tcn',    'raw-TCN'),
]
WINS = ['w1s', 'w2s', 'w5s']


def get(d):
    p = next(iter((d / 'phase2').rglob('cv_summary.json')), None)
    if p is None: return None
    return json.loads(p.read_text())['summary']


def m(s, t, k):
    if s is None: return None
    return s.get(t, {}).get(k, {}).get('mean')


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    # ----------- collect -----------
    rows = []
    for w in WINS:
        for slug, label in ARCHS:
            v12 = get(ROOT / 'runs' / f'optuna_clean_v12eqw-{w}-{slug}')
            v13 = get(ROOT / 'runs' / f'optuna_clean_v13soft-{w}-{slug}')
            for camp, s in (('v12_eqw', v12), ('v13_soft', v13)):
                if s is None: continue
                rows.append({
                    'window': w, 'arch': label, 'campaign': camp,
                    'exercise_f1':  m(s, 'exercise', 'f1_macro'),
                    'phase_f1':     m(s, 'phase',    'f1_macro'),
                    'fatigue_r':    m(s, 'fatigue',  'pearson_r'),
                    'fatigue_mae':  m(s, 'fatigue',  'mae'),
                    'reps_mae':     m(s, 'reps',     'mae'),
                })

    # ----------- CSV ---------------
    csv_path = OUT / 'v12_vs_v13soft.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f'Wrote {csv_path}  ({len(rows)} rows)')

    # ----------- Markdown summary ----
    md = ['# v12 (equal weights) vs v13soft (Kendall + soft phase + soft_overlap reps)', '']
    md.append('Same 6 archs, same 3 windows, same data, same folds. Phase-2 with 3 seeds (42, 1337, 7).')
    md.append('Differences: v13soft uses (a) soft phase target -> KL-div, (b) soft_overlap reps target, (c) Kendall (2018) learnable task weights.')
    md.append('')
    md.append('Sign convention: positive delta = v13soft is BETTER for that metric (higher F1/r, lower MAE).')
    md.append('')
    md.append('| Window | Arch | dx F1 | dph F1 | dfat r | dfat MAE | dreps MAE | net |')
    md.append('|---|---|---|---|---|---|---|---|')
    summary = {'win': {'v13': 0, 'v12': 0, 'tie': 0},
               'task': {'exercise_f1': [], 'phase_f1': [],
                        'fatigue_r': [], 'fatigue_mae': [], 'reps_mae': []}}
    cells = {}
    for w in WINS:
        for slug, label in ARCHS:
            v12 = next((r for r in rows
                        if r['window']==w and r['arch']==label
                        and r['campaign']=='v12_eqw'), None)
            v13 = next((r for r in rows
                        if r['window']==w and r['arch']==label
                        and r['campaign']=='v13_soft'), None)
            if not v12 or not v13: continue
            d_ex = v13['exercise_f1'] - v12['exercise_f1']
            d_ph = v13['phase_f1']   - v12['phase_f1']
            d_fr = v13['fatigue_r']  - v12['fatigue_r']
            d_fm = v12['fatigue_mae']- v13['fatigue_mae']  # lower better
            d_re = v12['reps_mae']   - v13['reps_mae']     # lower better
            summary['task']['exercise_f1'].append(d_ex)
            summary['task']['phase_f1'].append(d_ph)
            summary['task']['fatigue_r'].append(d_fr)
            summary['task']['fatigue_mae'].append(d_fm)
            summary['task']['reps_mae'].append(d_re)
            wins13 = sum(d > 0 for d in (d_ex, d_ph, d_fr, d_fm, d_re))
            wins12 = sum(d < 0 for d in (d_ex, d_ph, d_fr, d_fm, d_re))
            if wins13 > wins12: summary['win']['v13'] += 1
            elif wins12 > wins13: summary['win']['v12'] += 1
            else: summary['win']['tie'] += 1
            net = ('v13' if wins13 > wins12 else
                   'v12' if wins12 > wins13 else 'tie')
            cells[(w, label)] = {
                'd_ex': d_ex, 'd_ph': d_ph, 'd_fr': d_fr,
                'd_fm': d_fm, 'd_re': d_re, 'net': net,
            }
            md.append(f'| {w} | {label} | {d_ex:+.4f} | {d_ph:+.4f} | '
                      f'{d_fr:+.4f} | {d_fm:+.4f} | {d_re:+.4f} | {net} |')

    md.append('')
    md.append(f'**Overall arch x window winners**: v13soft={summary["win"]["v13"]}, '
              f'v12 eqw={summary["win"]["v12"]}, tie={summary["win"]["tie"]} '
              f'(out of {sum(summary["win"].values())} pairs)')
    md.append('')
    md.append('## Per-task mean delta (positive = v13soft better)')
    md.append('| Task | Mean delta | Median | n positive | n negative |')
    md.append('|---|---|---|---|---|')
    for task in ('exercise_f1', 'phase_f1', 'fatigue_r',
                 'fatigue_mae', 'reps_mae'):
        ds = np.array(summary['task'][task])
        md.append(f'| {task} | {ds.mean():+.4f} | {np.median(ds):+.4f} | '
                  f'{(ds > 0).sum()} | {(ds < 0).sum()} |')

    (OUT / 'v12_vs_v13soft.md').write_text('\n'.join(md), encoding='utf-8')
    print(f'Wrote {OUT}/v12_vs_v13soft.md')

    # ----------- Heatmap -----------
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    arch_labels = [a[1] for a in ARCHS]
    metric_titles = [
        ('d_ex', 'exercise F1', 'higher better', 'RdBu_r'),
        ('d_ph', 'phase F1',    'higher better', 'RdBu_r'),
        ('d_fr', 'fatigue r',   'higher better', 'RdBu_r'),
        ('d_fm', 'fatigue MAE', 'lower better (sign-flipped)', 'RdBu_r'),
        ('d_re', 'reps MAE',    'lower better (sign-flipped)', 'RdBu_r'),
    ]
    for ax, (key, title, direction, cmap) in zip(axes, metric_titles):
        mat = np.full((len(WINS), len(ARCHS)), np.nan)
        for i, w in enumerate(WINS):
            for j, (slug, lab) in enumerate(ARCHS):
                v = cells.get((w, lab))
                if v: mat[i, j] = v[key]
        vmax = max(0.05, np.nanmax(np.abs(mat)))
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(ARCHS)));
        ax.set_xticklabels(arch_labels, rotation=30, ha='right', fontsize=9)
        ax.set_yticks(range(len(WINS))); ax.set_yticklabels(WINS)
        for i in range(len(WINS)):
            for j in range(len(ARCHS)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f'{mat[i, j]:+.3f}',
                            ha='center', va='center', fontsize=8,
                            color='black')
        ax.set_title(f'd {title}\n({direction})', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle('v13soft - v12 eqw  '
                 '(positive = v13soft better; red = v13soft wins, blue = v12 wins)',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = OUT / 'v12_vs_v13soft_heatmap.png'
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_png}')


if __name__ == '__main__':
    main()
