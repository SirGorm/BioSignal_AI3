"""Build a master CSV with one row per model, all metrics side-by-side.

Columns:
  category, slug, arch, variant, window_s, tasks, modality_dropped,
  cv_scheme, n_folds, n_seeds,
  exercise_f1_mean, exercise_f1_std, exercise_n,
  phase_f1_mean, phase_f1_std, phase_n,
  fatigue_mae_mean, fatigue_mae_std, fatigue_pearson_mean, fatigue_n,
  reps_mae_mean, reps_mae_std, reps_n,
  val_total_mean, val_total_std,
  hp_lr, hp_dropout, hp_repr_dim, hp_weight_decay, hp_batch_size, hp_extra
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def load_cv_summary(run_dir: Path):
    cands = list(run_dir.rglob('phase2/*/cv_summary.json'))
    if not cands:
        return None
    return json.loads(cands[0].read_text()).get('summary')


def get(d, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur if cur is not None else default


def parse_slug(slug: str):
    """Return (category, arch, window_s, modality_dropped, tasks)."""
    if 'rf' in slug:
        return ('rf_baseline', 'rf', None, 'none', 'all')
    m = re.match(r'v17multi-(\w+?)-w(\d+)s$', slug)
    if m:
        return ('multi-task', m.group(1), int(m.group(2)), 'none', 'all 4')
    m = re.match(r'v17single-(\w+?)-(\w+?)-w(\d+)s$', slug)
    if m:
        return ('single-task', m.group(2), int(m.group(3)), 'none', m.group(1))
    m = re.match(r'v17abl-drop_(\w+?)-(\w+?)-w(\d+)s$', slug)
    if m:
        return ('ablation', m.group(2), int(m.group(3)), m.group(1), 'all 4')
    return ('unknown', '?', None, '?', '?')


def main():
    rows = []

    # RF baseline (separate metrics layout)
    rf_dirs = sorted(Path('runs').glob('v17rf_*'))
    for rf_dir in rf_dirs:
        m_path = rf_dir / 'metrics.json'
        if not m_path.exists():
            continue
        rf = json.loads(m_path.read_text())
        rows.append({
            'category': 'rf_baseline',
            'slug': rf_dir.name,
            'arch': 'RandomForest',
            'variant': 'features',
            'window_s': '-',
            'tasks': 'all 4 (separate models)',
            'modality_dropped': 'none',
            'cv_scheme': rf.get('cv_scheme', 'GroupKFold-10 (LOSO)'),
            'n_folds': 10,
            'n_seeds': 1,
            'exercise_f1_mean':    rf.get('exercise', {}).get('f1_mean'),
            'exercise_f1_std':     rf.get('exercise', {}).get('f1_std'),
            'exercise_n':          '-',
            'phase_f1_mean':       rf.get('phase', {}).get('ml_f1_mean'),
            'phase_f1_std':        rf.get('phase', {}).get('ml_f1_std'),
            'phase_n':             '-',
            'fatigue_mae_mean':    rf.get('fatigue', {}).get('mae_mean'),
            'fatigue_mae_std':     rf.get('fatigue', {}).get('mae_std'),
            'fatigue_pearson_mean':rf.get('fatigue', {}).get('pearson_r_median'),
            'fatigue_pearson_std': '-',
            'fatigue_n':           '-',
            'reps_mae_mean':       rf.get('reps', {}).get('ml_mae_mean'),
            'reps_mae_std':        rf.get('reps', {}).get('ml_mae_std'),
            'reps_n':              '-',
            'val_total_mean':      '-',
            'val_total_std':       '-',
            'hp_lr':         '-',
            'hp_dropout':    '-',
            'hp_repr_dim':   '-',
            'hp_weight_decay': '-',
            'hp_batch_size': '-',
            'hp_extra':      json.dumps({k: rf.get(k, {}).get('optuna_best_params')
                                          for k in ('exercise','phase','fatigue','reps')}),
        })

    # NN runs (multi-task, single-task, ablation)
    for run_dir in sorted(Path('runs').glob('v17*')):
        if not run_dir.is_dir(): continue
        slug = run_dir.name
        if 'rf' in slug: continue

        cv = load_cv_summary(run_dir)
        if not cv: continue

        category, arch, window_s, drop, tasks = parse_slug(slug)

        # HPs from best_hps.json
        hp_path = run_dir / 'best_hps.json'
        hps = {}
        if hp_path.exists():
            try:
                hp_data = json.loads(hp_path.read_text())
                hps = hp_data.get('best_hps', hp_data)
            except Exception:
                pass

        # Count folds × seeds from cv summary
        all_results = cv.get('all_results', []) if 'all_results' in cv else []
        n_total = len(all_results) if all_results else None
        # Try to read seeds + folds count from metrics.n which is set per task
        if not n_total:
            n_total = get(cv, 'val_total', 'n')

        rows.append({
            'category': category,
            'slug': slug,
            'arch': arch,
            'variant': 'features',
            'window_s': window_s,
            'tasks': tasks,
            'modality_dropped': drop,
            'cv_scheme': 'LOSO 10-fold',
            'n_folds': 10,
            'n_seeds': max(1, (n_total or 30) // 10),
            'exercise_f1_mean':    get(cv, 'exercise', 'f1_macro', 'mean'),
            'exercise_f1_std':     get(cv, 'exercise', 'f1_macro', 'std'),
            'exercise_n':          get(cv, 'exercise', 'f1_macro', 'n'),
            'phase_f1_mean':       get(cv, 'phase', 'f1_macro', 'mean'),
            'phase_f1_std':        get(cv, 'phase', 'f1_macro', 'std'),
            'phase_n':             get(cv, 'phase', 'f1_macro', 'n'),
            'fatigue_mae_mean':    get(cv, 'fatigue', 'mae', 'mean'),
            'fatigue_mae_std':     get(cv, 'fatigue', 'mae', 'std'),
            'fatigue_pearson_mean':get(cv, 'fatigue', 'pearson_r', 'mean'),
            'fatigue_pearson_std': get(cv, 'fatigue', 'pearson_r', 'std'),
            'fatigue_n':           get(cv, 'fatigue', 'mae', 'n'),
            'reps_mae_mean':       get(cv, 'reps', 'mae', 'mean'),
            'reps_mae_std':        get(cv, 'reps', 'mae', 'std'),
            'reps_n':              get(cv, 'reps', 'mae', 'n'),
            'val_total_mean':      get(cv, 'val_total', 'mean'),
            'val_total_std':       get(cv, 'val_total', 'std'),
            'hp_lr':         hps.get('lr'),
            'hp_dropout':    hps.get('dropout'),
            'hp_repr_dim':   hps.get('repr_dim'),
            'hp_weight_decay': hps.get('weight_decay'),
            'hp_batch_size': hps.get('batch_size'),
            'hp_extra':      json.dumps({k: v for k, v in hps.items()
                                          if k not in ('lr','dropout','repr_dim',
                                                        'weight_decay','batch_size','_patience')}),
        })

    out = Path('results/v17_master_comparison.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print('No rows collected.'); return
    fields = list(rows[0].keys())
    with open(out, 'w', newline='', encoding='utf-8') as f:
        cw = csv.DictWriter(f, fieldnames=fields)
        cw.writeheader()
        for r in rows: cw.writerow(r)
    print(f'Wrote {len(rows)} rows to {out}')

    # Print summary
    from collections import Counter
    cats = Counter(r['category'] for r in rows)
    print()
    print('Rows per category:')
    for cat, n in cats.items():
        print(f'  {cat}: {n}')


if __name__ == '__main__':
    main()
