"""Build a single thesis-ready master CSV combining:
  - v17 features (RF + NN multi + NN single + ablation)
  - v20 raw (NN multi + NN single)

Same column schema for both pipelines so the CSV is directly comparable
(per-task mean ± std, n_folds × seeds, HPs hint).

Output: results/Final/v17_v20_thesis_master.csv
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def to_f(v):
    try: return float(v) if v not in (None, '') else None
    except (TypeError, ValueError): return None


def main():
    rows_out = []

    # ---- v17 features pipeline (incl. RF, ablation) ----
    # Find source CSV: prefer Final/ if user moved it there
    p17 = Path('results/Final/v17_master_comparison.csv')
    if not p17.exists():
        p17 = Path('results/v17_master_comparison.csv')
    if p17.exists():
        with open(p17, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rows_out.append({
                    'pipeline': 'v17_features',
                    'category': r['category'],
                    'slug': r['slug'],
                    'arch': r['arch'],
                    'variant': 'features',
                    'window_s': r['window_s'],
                    'tasks': r['tasks'],
                    'modality_dropped': r['modality_dropped'],
                    'cv_scheme': r.get('cv_scheme', 'LOSO 10-fold'),
                    'n_folds': r.get('n_folds'),
                    'n_seeds': r.get('n_seeds'),
                    'exercise_f1_mean':    r.get('exercise_f1_mean'),
                    'exercise_f1_std':     r.get('exercise_f1_std'),
                    'phase_f1_mean':       r.get('phase_f1_mean'),
                    'phase_f1_std':        r.get('phase_f1_std'),
                    'fatigue_mae_mean':    r.get('fatigue_mae_mean'),
                    'fatigue_mae_std':     r.get('fatigue_mae_std'),
                    'fatigue_pearson_mean':r.get('fatigue_pearson_mean'),
                    'fatigue_pearson_std': r.get('fatigue_pearson_std'),
                    'reps_mae_mean':       r.get('reps_mae_mean'),
                    'reps_mae_std':        r.get('reps_mae_std'),
                    'val_total_mean':      r.get('val_total_mean'),
                    'val_total_std':       r.get('val_total_std'),
                })

    # ---- v20 raw pipeline ----
    p20 = Path('results/Final/v20_all_results.csv')
    if not p20.exists():
        p20 = Path('results/v20_all_results.csv')
    if p20.exists():
        with open(p20, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                # category from variant (mt = multi-task, st = single-task)
                category = 'multi-task' if r['variant'] == 'mt' else 'single-task'
                tasks = 'all 4' if r['variant'] == 'mt' else r['target_task']
                rows_out.append({
                    'pipeline': 'v20_raw',
                    'category': category,
                    'slug': r['run_name'],
                    'arch': r['arch'],
                    'variant': 'raw',
                    'window_s': r['window_s'],
                    'tasks': tasks,
                    'modality_dropped': 'none',
                    'cv_scheme': 'LOSO 10-fold',
                    'n_folds': 10,
                    'n_seeds': int(int(r['n_fold_seeds']) / 10) if r.get('n_fold_seeds') else None,
                    'exercise_f1_mean':    r.get('exercise_f1_macro_mean'),
                    'exercise_f1_std':     r.get('exercise_f1_macro_std'),
                    'phase_f1_mean':       r.get('phase_f1_macro_mean'),
                    'phase_f1_std':        r.get('phase_f1_macro_std'),
                    'fatigue_mae_mean':    r.get('fatigue_mae_mean'),
                    'fatigue_mae_std':     r.get('fatigue_mae_std'),
                    'fatigue_pearson_mean':r.get('fatigue_pearson_r_mean'),
                    'fatigue_pearson_std': r.get('fatigue_pearson_r_std'),
                    'reps_mae_mean':       r.get('reps_mae_mean'),
                    'reps_mae_std':        r.get('reps_mae_std'),
                    'val_total_mean':      r.get('val_total_mean'),
                    'val_total_std':       r.get('val_total_std'),
                })

    out = Path('results/Final/v17_v20_thesis_master.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows_out[0].keys())
    with open(out, 'w', newline='', encoding='utf-8') as f:
        cw = csv.DictWriter(f, fieldnames=fields)
        cw.writeheader()
        for r in rows_out: cw.writerow(r)
    print(f'Wrote {len(rows_out)} rows to {out}')

    # Print best-per-task comparison
    print()
    print('=== Best per task across BOTH pipelines (excluding ablations) ===')
    print(f'{"task":<14} {"pipeline":<14} {"slug":<35} {"value":>10}')
    print('-' * 80)

    main_rows = [r for r in rows_out if r['category'] != 'ablation']

    for task, col, direction in [
        ('Exercise F1', 'exercise_f1_mean', 'max'),
        ('Phase F1',    'phase_f1_mean',    'max'),
        ('Fatigue MAE', 'fatigue_mae_mean', 'min'),
        ('Reps MAE',    'reps_mae_mean',    'min'),
    ]:
        cands = [(r['pipeline'], r['slug'], to_f(r[col])) for r in main_rows
                  if to_f(r[col]) is not None]
        if direction == 'max':
            best = max(cands, key=lambda kv: kv[2])
        else:
            best = min(cands, key=lambda kv: kv[2])
        print(f'{task:<14} {best[0]:<14} {best[1]:<35} {best[2]:>10.4f}')

    # By category breakdown
    print()
    print('=== Row count by pipeline + category ===')
    from collections import Counter
    counts = Counter((r['pipeline'], r['category']) for r in rows_out)
    for (pipe, cat), n in sorted(counts.items()):
        print(f'  {pipe:<14} {cat:<14}  {n}')


if __name__ == '__main__':
    main()
