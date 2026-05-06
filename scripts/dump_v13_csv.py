"""Dump all v13 results (v13single + v13soft) into one CSV.

For each run we pull:
  - per-task mean / std for the four core metrics (with N folds)
  - phase1 progress / status for in-flight runs that have no cv_summary yet

Output: runs/comparison_v13/v13_all_results.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / 'runs' / 'comparison_v13' / 'v13_all_results.csv'
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def parse_slug(name: str) -> dict:
    s = name.replace('optuna_clean_', '')
    if s.startswith('v13single-'):
        rest = s[len('v13single-'):]                      # e.g. exercise-only-w5s-feat-mlp
        task_focus, _only, window, *arch_parts = rest.split('-')
        arch = '-'.join(arch_parts)
        return {'campaign': 'v13single', 'task_focus': task_focus,
                'window': window, 'arch': arch, 'variant': 'single-task'}
    if s.startswith('v13soft-'):
        rest = s[len('v13soft-'):]                        # e.g. w1s-multi-feat-mlp
        window, multi, *arch_parts = rest.split('-')
        arch = '-'.join(arch_parts)
        return {'campaign': 'v13soft', 'task_focus': 'multi-task',
                'window': window, 'arch': arch, 'variant': 'soft-phase+soft-reps+kendall'}
    return {'campaign': 'unknown', 'task_focus': '', 'window': '', 'arch': s,
            'variant': ''}


def get_summary(run_dir: Path):
    p = next(iter((run_dir / 'phase2').rglob('cv_summary.json')), None)
    if p is None:
        return None
    return json.loads(p.read_text()).get('summary', {})


def get_phase1_progress(run_dir: Path):
    """Return (n_done, n_target) — best-effort phase1 status."""
    cfg_p = run_dir / 'config.json'
    n_target = ''
    if cfg_p.exists():
        try:
            cfg = json.loads(cfg_p.read_text())
            n_target = cfg.get('n_trials', '')
        except Exception:
            pass
    n_done = 0
    p1 = run_dir / 'phase1'
    if p1.exists():
        n_done = sum(1 for d in p1.iterdir()
                     if d.is_dir() and d.name.startswith('trial_'))
    return n_done, n_target


def fmt(v, prec=4):
    if v is None: return ''
    try:
        if isinstance(v, float) and (v != v):  # NaN
            return ''
        return f'{v:.{prec}f}'
    except Exception:
        return str(v)


def row_for(d: Path):
    info = parse_slug(d.name)
    s = get_summary(d)
    n_done, n_target = get_phase1_progress(d)
    has_cv = s is not None
    base = {
        'run_dir':    d.name,
        'campaign':   info['campaign'],
        'variant':    info['variant'],
        'window':     info['window'],
        'arch':       info['arch'],
        'task_focus': info['task_focus'],
        'phase2_done': 'yes' if has_cv else 'no',
        'phase1_done': n_done,
        'phase1_target': n_target,
    }
    keys = [
        ('exercise_f1_mean',     ('exercise', 'f1_macro', 'mean')),
        ('exercise_f1_std',      ('exercise', 'f1_macro', 'std')),
        ('phase_f1_mean',        ('phase',    'f1_macro', 'mean')),
        ('phase_f1_std',         ('phase',    'f1_macro', 'std')),
        ('fatigue_r_mean',       ('fatigue',  'pearson_r', 'mean')),
        ('fatigue_r_std',        ('fatigue',  'pearson_r', 'std')),
        ('fatigue_mae_mean',     ('fatigue',  'mae', 'mean')),
        ('fatigue_mae_std',      ('fatigue',  'mae', 'std')),
        ('reps_mae_mean',        ('reps',     'mae', 'mean')),
        ('reps_mae_std',         ('reps',     'mae', 'std')),
        ('n_folds',              ('exercise', 'f1_macro', 'n')),
    ]
    for col, (t, m, k) in keys:
        v = None
        if s is not None:
            v = s.get(t, {}).get(m, {}).get(k)
        base[col] = fmt(v) if k != 'n' else (v if v is not None else '')
    return base


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    runs = sorted(p for p in (ROOT / 'runs').iterdir()
                  if p.is_dir() and p.name.startswith('optuna_clean_v13'))
    rows = [row_for(d) for d in runs]
    cols = list(rows[0].keys())
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f'Wrote {OUT_CSV}  ({len(rows)} rows)')
    # Stdout summary
    print(f'\n{"run":50s}  ph2  ph1  ex_F1  ph_F1  fat_r   fat_MAE  rep_MAE')
    for r in rows:
        print(f'{r["run_dir"][:50]:50s}  {r["phase2_done"]:3s}  '
              f'{r["phase1_done"]:>2}/{r["phase1_target"]:>2}  '
              f'{r["exercise_f1_mean"][:5]:5s}  {r["phase_f1_mean"][:5]:5s}  '
              f'{r["fatigue_r_mean"][:6]:6s}  {r["fatigue_mae_mean"][:6]:6s}  '
              f'{r["reps_mae_mean"][:6]:6s}')


if __name__ == '__main__':
    main()
