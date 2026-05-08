"""Build CSVs of modality ablation deltas:

results/v17_modality_importance.csv: 24 rows, one per (arch, window, dropped)
    with full vs dropped metrics + delta.

results/v17_modality_importance_summary.csv: 4 rows, one per modality,
    averaged over 6 (arch, window) combos.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_master():
    rows = []
    with open('results/v17_master_comparison.csv', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return rows


def to_f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def main():
    rows = load_master()
    full = {(r['arch'], str(r['window_s'])): r
            for r in rows
            if r['category'] == 'multi-task' and r['modality_dropped'] == 'none'}
    abl = {}
    for r in rows:
        if r['category'] == 'ablation':
            abl[(r['arch'], str(r['window_s']), r['modality_dropped'])] = r

    detail = []
    for (arch, w, drop), a in sorted(abl.items()):
        f_row = full.get((arch, w))
        if not f_row: continue
        # Deltas: F1 = ablation - full (positive = better when dropped)
        # MAE  = full - ablation (positive = better when dropped, i.e. lower MAE)
        f_ex_full,  f_ex_abl  = to_f(f_row['exercise_f1_mean']),  to_f(a['exercise_f1_mean'])
        f_ph_full,  f_ph_abl  = to_f(f_row['phase_f1_mean']),     to_f(a['phase_f1_mean'])
        f_fat_full, f_fat_abl = to_f(f_row['fatigue_mae_mean']),  to_f(a['fatigue_mae_mean'])
        f_re_full,  f_re_abl  = to_f(f_row['reps_mae_mean']),     to_f(a['reps_mae_mean'])

        detail.append({
            'arch': arch, 'window_s': w, 'modality_dropped': drop,
            'full_exercise_f1':       f_ex_full,
            'dropped_exercise_f1':    f_ex_abl,
            'delta_exercise_f1':      (f_ex_abl - f_ex_full) if (f_ex_abl is not None and f_ex_full is not None) else None,
            'full_phase_f1':          f_ph_full,
            'dropped_phase_f1':       f_ph_abl,
            'delta_phase_f1':         (f_ph_abl - f_ph_full) if (f_ph_abl is not None and f_ph_full is not None) else None,
            'full_fatigue_mae':       f_fat_full,
            'dropped_fatigue_mae':    f_fat_abl,
            'delta_fatigue_mae_lower_better': (f_fat_full - f_fat_abl) if (f_fat_abl is not None and f_fat_full is not None) else None,
            'full_reps_mae':          f_re_full,
            'dropped_reps_mae':       f_re_abl,
            'delta_reps_mae_lower_better': (f_re_full - f_re_abl) if (f_re_abl is not None and f_re_full is not None) else None,
        })

    out_detail = Path('results/v17_modality_importance.csv')
    out_detail.parent.mkdir(parents=True, exist_ok=True)
    fields = list(detail[0].keys())
    with open(out_detail, 'w', newline='', encoding='utf-8') as f:
        cw = csv.DictWriter(f, fieldnames=fields)
        cw.writeheader()
        for r in detail: cw.writerow(r)
    print(f'Wrote {len(detail)} rows to {out_detail}')

    # Summary: average delta per modality across all 6 (arch, window) combos
    by_mod = defaultdict(lambda: {'exer':[], 'phase':[], 'fat':[], 'reps':[]})
    for r in detail:
        m = r['modality_dropped']
        if r['delta_exercise_f1']            is not None: by_mod[m]['exer'].append(r['delta_exercise_f1'])
        if r['delta_phase_f1']               is not None: by_mod[m]['phase'].append(r['delta_phase_f1'])
        if r['delta_fatigue_mae_lower_better']is not None: by_mod[m]['fat'].append(r['delta_fatigue_mae_lower_better'])
        if r['delta_reps_mae_lower_better']  is not None: by_mod[m]['reps'].append(r['delta_reps_mae_lower_better'])

    summary = []
    for m in ('emg','acc','ppg','temp'):
        d = by_mod[m]
        summary.append({
            'modality': m,
            'n_combos': len(d['exer']),
            'mean_delta_exercise_f1':       np.mean(d['exer']) if d['exer'] else None,
            'std_delta_exercise_f1':        np.std(d['exer']) if d['exer'] else None,
            'mean_delta_phase_f1':          np.mean(d['phase']) if d['phase'] else None,
            'std_delta_phase_f1':           np.std(d['phase']) if d['phase'] else None,
            'mean_delta_fatigue_mae_lower_better': np.mean(d['fat']) if d['fat'] else None,
            'std_delta_fatigue_mae_lower_better':  np.std(d['fat']) if d['fat'] else None,
            'mean_delta_reps_mae_lower_better':    np.mean(d['reps']) if d['reps'] else None,
            'std_delta_reps_mae_lower_better':     np.std(d['reps']) if d['reps'] else None,
        })

    out_summary = Path('results/v17_modality_importance_summary.csv')
    fields = list(summary[0].keys())
    with open(out_summary, 'w', newline='', encoding='utf-8') as f:
        cw = csv.DictWriter(f, fieldnames=fields)
        cw.writeheader()
        for r in summary: cw.writerow(r)
    print(f'Wrote {len(summary)} rows to {out_summary}')

    # Print summary
    print()
    print('=== Modality importance summary ===')
    print('Negative = modality was useful (removing it hurt the task)')
    print(f'{"modality":<10} {"d_exer":>10} {"d_phase":>10} {"d_fat":>10} {"d_reps":>10}')
    for r in summary:
        print(f"{r['modality']:<10} "
              f"{r['mean_delta_exercise_f1']:>+10.4f} "
              f"{r['mean_delta_phase_f1']:>+10.4f} "
              f"{r['mean_delta_fatigue_mae_lower_better']:>+10.4f} "
              f"{r['mean_delta_reps_mae_lower_better']:>+10.4f}")


if __name__ == '__main__':
    main()
