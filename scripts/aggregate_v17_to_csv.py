"""Aggregate all v17 training values into two CSVs.

results/v17_all_folds.csv:
    one row per (slug, seed, fold) with final per-fold metrics

results/v17_all_history.csv:
    one row per (slug, seed, fold, epoch) with per-epoch loss + val metrics
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def main():
    fold_rows = []
    hist_rows = []

    # Iterate every v17 NN run dir (skip RF — no per-epoch history)
    for run_dir in sorted(Path('runs').glob('v17*')):
        if not run_dir.is_dir():
            continue
        slug = run_dir.name
        if 'rf' in slug:
            continue

        for fold_dir in run_dir.rglob('phase2/*/seed_*/fold_*'):
            posix_path = fold_dir.as_posix()
            m = re.search(r'seed_(\d+)/fold_(\d+)', posix_path)
            if not m:
                continue
            seed = int(m.group(1))
            fold = int(m.group(2))

            # Per-fold final metrics
            m_path = fold_dir / 'metrics.json'
            if m_path.exists():
                mdata = json.loads(m_path.read_text())
                fold_rows.append({
                    'slug': slug,
                    'seed': seed,
                    'fold': fold,
                    'test_subjects': '|'.join(mdata.get('test_subjects', [])),
                    'val_total':           mdata.get('val_total'),
                    'exercise_f1':         mdata.get('exercise', {}).get('f1_macro'),
                    'exercise_balanced_acc': mdata.get('exercise', {}).get('balanced_accuracy'),
                    'exercise_n':          mdata.get('exercise', {}).get('n'),
                    'phase_f1':            mdata.get('phase', {}).get('f1_macro'),
                    'phase_balanced_acc':  mdata.get('phase', {}).get('balanced_accuracy'),
                    'phase_n':             mdata.get('phase', {}).get('n'),
                    'fatigue_mae':         mdata.get('fatigue', {}).get('mae'),
                    'fatigue_pearson':     mdata.get('fatigue', {}).get('pearson_r'),
                    'fatigue_n':           mdata.get('fatigue', {}).get('n'),
                    'reps_mae':            mdata.get('reps', {}).get('mae'),
                    'reps_n':              mdata.get('reps', {}).get('n'),
                })

            # Per-epoch history
            h_path = fold_dir / 'history.json'
            if h_path.exists():
                history = json.loads(h_path.read_text())
                for ep in history:
                    train = ep.get('train', {})
                    val = ep.get('val_loss', {})
                    vm = ep.get('val_metrics', {})
                    hist_rows.append({
                        'slug': slug, 'seed': seed, 'fold': fold,
                        'epoch':           ep.get('epoch'),
                        'train_total':     train.get('total'),
                        'train_exercise':  train.get('exercise'),
                        'train_phase':     train.get('phase'),
                        'train_fatigue':   train.get('fatigue'),
                        'train_reps':      train.get('reps'),
                        'val_total':       val.get('total'),
                        'val_exercise':    val.get('exercise'),
                        'val_phase':       val.get('phase'),
                        'val_fatigue':     val.get('fatigue'),
                        'val_reps':        val.get('reps'),
                        'val_exercise_f1': vm.get('exercise', {}).get('f1_macro'),
                        'val_phase_f1':    vm.get('phase', {}).get('f1_macro'),
                        'val_fatigue_mae': vm.get('fatigue', {}).get('mae'),
                        'val_fatigue_r':   vm.get('fatigue', {}).get('pearson_r'),
                        'val_reps_mae':    vm.get('reps', {}).get('mae'),
                    })

    out_folds = Path('results/v17_all_folds.csv')
    out_folds.parent.mkdir(parents=True, exist_ok=True)
    if fold_rows:
        with open(out_folds, 'w', newline='', encoding='utf-8') as f:
            cw = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
            cw.writeheader()
            for r in fold_rows:
                cw.writerow(r)
        print(f'Wrote {len(fold_rows)} rows to {out_folds}')

    out_hist = Path('results/v17_all_history.csv')
    if hist_rows:
        with open(out_hist, 'w', newline='', encoding='utf-8') as f:
            cw = csv.DictWriter(f, fieldnames=list(hist_rows[0].keys()))
            cw.writeheader()
            for r in hist_rows:
                cw.writerow(r)
        print(f'Wrote {len(hist_rows)} rows to {out_hist}')

    # Summary by run
    from collections import defaultdict
    by_run = defaultdict(lambda: {'folds': 0, 'epochs': 0})
    for r in fold_rows:
        by_run[r['slug']]['folds'] += 1
    for r in hist_rows:
        by_run[r['slug']]['epochs'] += 1
    print()
    print(f'=== Summary by run ({len(by_run)} runs) ===')
    for slug in sorted(by_run):
        b = by_run[slug]
        print(f'  {slug:<45} folds={b["folds"]:>3}  epochs={b["epochs"]:>5}')


if __name__ == '__main__':
    main()
