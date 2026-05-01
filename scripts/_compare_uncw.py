"""Quick partial-progress comparison of TCN uncertainty-weighted vs fixed-weight baseline."""
import json, glob, numpy as np, os

b_dir = 'runs/20260428_025001_nn-full-tcn/tcn'
u_dir = 'runs/20260428_115055_nn-full-tcn-uncw/tcn'

done = set()
for f in sorted(glob.glob(f'{u_dir}/seed_*/fold_*/metrics.json')):
    parts = f.replace(os.sep, '/').split('/')
    done.add((parts[-3], parts[-2]))

def collect(root, keep):
    agg = {'exercise_f1': [], 'phase_f1': [], 'fatigue_mae': [],
           'fatigue_r': [], 'reps_mae': []}
    for f in sorted(glob.glob(f'{root}/seed_*/fold_*/metrics.json')):
        parts = f.replace(os.sep, '/').split('/')
        if (parts[-3], parts[-2]) not in keep:
            continue
        m = json.load(open(f))
        agg['exercise_f1'].append(m['exercise']['f1_macro'])
        agg['phase_f1'].append(m['phase']['f1_macro'])
        agg['fatigue_mae'].append(m['fatigue']['mae'])
        agg['fatigue_r'].append(m['fatigue']['pearson_r'])
        agg['reps_mae'].append(m['reps']['mae'])
    return {k: (float(np.nanmean(v)), float(np.nanstd(v)), len(v))
            for k, v in agg.items()}

b = collect(b_dir, done)
u = collect(u_dir, done)

print(f'Matched {len(done)}/15 folds (uncertainty-weighted run still on fold_3 of seed_7)\n')
print(f'{"metric":15s} {"baseline":>20s} {"uncertainty-weighted":>26s}   {"delta (u - b)":>14s}')
print('-' * 85)
better = {'exercise_f1': '+', 'phase_f1': '+', 'fatigue_mae': '-',
          'fatigue_r': '+', 'reps_mae': '-'}
for k in b:
    bm, bs, n = b[k]
    um, us, _ = u[k]
    delta = um - bm
    arrow = ('better' if (delta > 0 and better[k] == '+') or
             (delta < 0 and better[k] == '-') else 'worse')
    print(f'{k:15s} {bm:7.4f}±{bs:.4f}    {um:7.4f}±{us:.4f}        {delta:+.4f}  {arrow}')
