"""V13 scorecard: one row per model (v13single + v13soft), one column per
task. Best-per-task highlighted. Outputs CSV + markdown + PNG heatmap.

Outputs:
  runs/comparison_v13/v13_scorecard.csv
  runs/comparison_v13/v13_scorecard.md
  runs/comparison_v13/v13_scorecard.png
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

# Each task: (display name, key_in_summary, metric_in_summary, higher_better)
TASKS = [
    ('exercise F1',  'exercise', 'f1_macro',  True),
    ('phase F1',     'phase',    'f1_macro',  True),
    ('fatigue r',    'fatigue',  'pearson_r', True),
    ('fatigue MAE',  'fatigue',  'mae',       False),
    ('reps MAE',     'reps',     'mae',       False),
]


def parse_slug(name: str) -> dict:
    s = name.replace('optuna_clean_', '')
    if s.startswith('v13single-'):
        rest = s[len('v13single-'):]                  # exercise-only-w5s-feat-mlp
        task_focus, _only, window, *arch = rest.split('-')
        return {'campaign': 'v13single', 'window': window,
                'arch': '-'.join(arch), 'task_focus': task_focus}
    if s.startswith('v13soft-'):
        rest = s[len('v13soft-'):]                    # w1s-multi-feat-mlp
        window, _multi, *arch = rest.split('-')
        return {'campaign': 'v13soft', 'window': window,
                'arch': '-'.join(arch), 'task_focus': 'multi'}
    return None


def get_summary(d: Path):
    p = next(iter((d / 'phase2').rglob('cv_summary.json')), None)
    if p is None: return None
    return json.loads(p.read_text())['summary']


def cell(s, t, k):
    if s is None: return None
    v = s.get(t, {}).get(k, {}).get('mean')
    return v if v is not None else None


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    # ---- Collect rows --------------------------------------------------
    rows = []
    for d in sorted((ROOT / 'runs').iterdir()):
        if not d.is_dir() or 'optuna_clean_v13' not in d.name:
            continue
        info = parse_slug(d.name)
        if info is None: continue
        s = get_summary(d)
        if s is None: continue
        rows.append({
            'run':        d.name,
            'campaign':   info['campaign'],
            'task_focus': info['task_focus'],
            'window':     info['window'],
            'arch':       info['arch'],
            'exercise_f1':  cell(s, 'exercise', 'f1_macro'),
            'phase_f1':     cell(s, 'phase',    'f1_macro'),
            'fatigue_r':    cell(s, 'fatigue',  'pearson_r'),
            'fatigue_mae':  cell(s, 'fatigue',  'mae'),
            'reps_mae':     cell(s, 'reps',     'mae'),
        })
    print(f'Collected {len(rows)} rows.')

    # ---- Sort: campaign first, then arch, then window ------------------
    win_order = {'w1s': 1, 'w2s': 2, 'w5s': 5}
    rows.sort(key=lambda r: (r['campaign'], r['task_focus'],
                              r['arch'], win_order.get(r['window'], 9)))

    # ---- Best-per-task indices -----------------------------------------
    best_idx = {}
    for col, _, _, higher in TASKS:
        col_key = (col.replace(' ', '_').lower())
        valid = [(i, r[col_key]) for i, r in enumerate(rows)
                 if r[col_key] is not None]
        if not valid: continue
        if higher:
            best_idx[col_key] = max(valid, key=lambda x: x[1])[0]
        else:
            best_idx[col_key] = min(valid, key=lambda x: x[1])[0]
    print('Best per task:')
    for k, i in best_idx.items():
        print(f'  {k:14s} -> row {i}: {rows[i]["run"]} = {rows[i][k]}')

    # ---- CSV -----------------------------------------------------------
    csv_path = OUT / 'v13_scorecard.csv'
    cols = ['run', 'campaign', 'task_focus', 'window', 'arch',
            'exercise_f1', 'phase_f1', 'fatigue_r',
            'fatigue_mae', 'reps_mae']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{r[k]:.4f}' if isinstance(r[k], float)
                            else (r[k] if r[k] is not None else ''))
                        for k in cols})
    print(f'Wrote {csv_path}')

    # ---- Markdown ------------------------------------------------------
    md_path = OUT / 'v13_scorecard.md'
    md = ['# v13 scorecard - all models, all heads', '']
    md.append('Bold = best across all 30 v13 models for that task.')
    md.append('Empty cell = task was not trained for that single-task model.')
    md.append('')
    md.append('| # | campaign | focus | win | arch | exercise F1 | '
              'phase F1 | fatigue r | fatigue MAE | reps MAE |')
    md.append('|---|---|---|---|---|---|---|---|---|---|')
    for i, r in enumerate(rows):
        cells = [str(i + 1), r['campaign'], r['task_focus'],
                 r['window'], r['arch']]
        for col, _, _, _ in TASKS:
            key = col.replace(' ', '_').lower()
            v = r[key]
            if v is None:
                cells.append(' ')
            else:
                txt = f'{v:.4f}' if 'r' not in col[-2:] else f'{v:+.4f}'
                if best_idx.get(key) == i:
                    txt = f'**{txt}**'
                cells.append(txt)
        md.append('| ' + ' | '.join(cells) + ' |')

    md.append('')
    md.append('## Best per task (across all v13)')
    md.append('| Task | Best model | Score |')
    md.append('|---|---|---|')
    for col, _, _, _ in TASKS:
        key = col.replace(' ', '_').lower()
        if key in best_idx:
            r = rows[best_idx[key]]
            v = r[key]
            md.append(f'| {col} | {r["run"]} | {v:+.4f} |')
    Path(md_path).write_text('\n'.join(md), encoding='utf-8')
    print(f'Wrote {md_path}')

    # ---- PNG visual scorecard ------------------------------------------
    # 5 task columns x 30 model rows. Each cell colored by per-task rank.
    fig_h = max(8, 0.32 * len(rows) + 2)
    fig, axes = plt.subplots(1, len(TASKS), figsize=(18, fig_h),
                              sharey=True)
    y = np.arange(len(rows))
    labels = [f'{r["campaign"][3:]:<5s} {r["task_focus"]:<8s} '
              f'{r["window"]:<3s} {r["arch"]}' for r in rows]
    for ax, (col, t, m, higher) in zip(axes, TASKS):
        key = col.replace(' ', '_').lower()
        vals = np.array([r[key] if r[key] is not None else np.nan
                         for r in rows], dtype=float)
        # Map to color: use rank among defined values.
        finite_mask = np.isfinite(vals)
        order = vals.copy()
        if finite_mask.any():
            ranks = np.full_like(vals, np.nan, dtype=float)
            idx_sorted = np.argsort(vals[finite_mask])
            if not higher: idx_sorted = idx_sorted  # smaller=better -> rank 0 = best
            else:           idx_sorted = idx_sorted[::-1]  # bigger=better -> rank 0 = best
            ranks_finite = np.empty(finite_mask.sum())
            for rk, idx in enumerate(idx_sorted):
                ranks_finite[idx] = rk
            ranks[finite_mask] = ranks_finite / max(1, finite_mask.sum() - 1)
            order = ranks
        # Colors: green for best, red for worst.
        cmap = plt.cm.RdYlGn_r
        bar_colors = [cmap(o) if np.isfinite(o) else (0.92, 0.92, 0.92, 1)
                      for o in order]
        bars = ax.barh(y, np.where(finite_mask, vals, 0), color=bar_colors,
                        edgecolor='black', linewidth=0.4)
        # Mark best with a star
        if key in best_idx:
            ax.scatter(vals[best_idx[key]] * 1.01, best_idx[key],
                       marker='*', color='black', s=120, zorder=5)
        # Annotate values
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                txt = f'{v:.3f}'
                ha = 'left' if v >= 0 else 'right'
                ax.text(v, yi, ' ' + txt, va='center', ha=ha,
                        fontsize=7)
        ax.set_title(f'{col}\n({"higher" if higher else "lower"} better)',
                     fontsize=11)
        ax.grid(axis='x', linestyle=':', alpha=0.4)
        ax.axvline(0, color='black', linewidth=0.6)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=8,
                                                     family='monospace')
    axes[0].invert_yaxis()
    fig.suptitle('v13 scorecard - all models, all heads (* = best in task)',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = OUT / 'v13_scorecard.png'
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_png}')


if __name__ == '__main__':
    main()
