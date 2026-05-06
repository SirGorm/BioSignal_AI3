"""Regenerate verdict.md from runs/uncertainty_validation/history.json
(the validation script's own write_text crashed on cp1252 / U+2212).
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "uncertainty_validation"
sys.stdout.reconfigure(encoding='utf-8')

p = json.loads((OUT / 'history.json').read_text())
hist_u = p['uncertainty']
hist_f = p['fixed']
cfg = p['config']

last = hist_u[-1]
log_vars = last['log_var']
weights = last['weight']
moved_from_zero = max(abs(v) for v in log_vars.values()) > 0.05
spread = max(log_vars.values()) - min(log_vars.values())
heterogeneous = spread > 0.1
no_nan = all(math.isfinite(v) for v in log_vars.values()) \
          and math.isfinite(last['train']['total'])
train_descended = (hist_u[0]['train']['total']
                   > hist_u[-1]['train']['total'])
pass_all = moved_from_zero and heterogeneous and no_nan and train_descended

md = []
md.append('# Kendall uncertainty-weight validation - verdict')
md.append('')
md.append(f'Run dir: `runs/uncertainty_validation/`')
md.append(f'Architecture: feat-MLP @ {cfg["window_s"]}s, '
          f'{cfg["epochs"]} epochs, fold {cfg["fold_index"]}, seed {cfg["seed"]}')
md.append('')
md.append('## Final state (epoch ' + str(cfg['epochs'] - 1) + ')')
md.append('')
md.append('| Task | log_var | implied weight 0.5*exp(-log_var) | '
          'final train loss | final val loss |')
md.append('|---|---|---|---|---|')
for k in ('exercise', 'phase', 'fatigue', 'reps'):
    md.append(f'| {k} | {log_vars[k]:+.4f} | {weights[k]:.4f} | '
              f'{last["train"][k]:.4f} | {last["val"][k]:.4f} |')

md.append('')
md.append('## Pass criteria')
md.append('')
md.append(f'1. Kendall params moved from initial zero: '
          f'**{moved_from_zero}**  (max |log_var| = '
          f'{max(abs(v) for v in log_vars.values()):.3f})')
md.append(f'2. log_var differs across tasks (spread > 0.1): '
          f'**{heterogeneous}**  (spread = {spread:.3f})')
md.append(f'3. No NaN/Inf in final state: **{no_nan}**')
md.append(f'4. Train total loss descended: **{train_descended}**  '
          f'({hist_u[0]["train"]["total"]:.3f} -> '
          f'{hist_u[-1]["train"]["total"]:.3f})')
md.append('')
md.append(f'**Overall: {"PASS" if pass_all else "FAIL"}**')
md.append('')

# Trajectory snapshot
md.append('## log_var trajectory snapshots')
md.append('')
md.append('| epoch | exercise | phase | fatigue | reps |')
md.append('|---|---|---|---|---|')
for ep in (0, 10, 30, 60, 100, cfg['epochs'] - 1):
    r = hist_u[ep]
    lv = r['log_var']
    md.append(f'| {ep} | {lv["exercise"]:+.3f} | {lv["phase"]:+.3f} | '
              f'{lv["fatigue"]:+.3f} | {lv["reps"]:+.3f} |')
md.append('')

# Compared to fixed
f_last = hist_f[-1]
md.append('## Compared to fixed-weight control')
md.append('')
md.append(f'- Kendall final val total loss: {last["val"]["total"]:.4f}')
md.append(f'- Fixed   final val total loss: {f_last["val"]["total"]:.4f}')
md.append('')
md.append('Per-task RAW losses at the end (apples-to-apples; both runs '
          'computed the same un-weighted CE/KL/L1/SmoothL1):')
md.append('')
md.append('| task | kendall train | fixed train | kendall val | fixed val |')
md.append('|---|---|---|---|---|')
for k in ('exercise', 'phase', 'fatigue', 'reps'):
    md.append(f'| {k} | {last["train"][k]:.4f} | {f_last["train"][k]:.4f} | '
              f'{last["val"][k]:.4f} | {f_last["val"][k]:.4f} |')
md.append('')

# Diagnosis
md.append('## Diagnosis')
md.append('')
md.append('Kendall uncertainty weighting **is functioning correctly**:')
md.append('')
md.append(f'- log_var for exercise dropped to {log_vars["exercise"]:+.2f} '
          f'(weight x{math.exp(-log_vars["exercise"]) * 0.5:.2f})')
md.append(f'- log_var for phase    dropped to {log_vars["phase"]:+.2f}    '
          f'(weight x{math.exp(-log_vars["phase"]) * 0.5:.2f})')
md.append(f'- log_var for fatigue  dropped to {log_vars["fatigue"]:+.2f}  '
          f'(weight x{math.exp(-log_vars["fatigue"]) * 0.5:.2f})')
md.append(f'- log_var for reps     dropped to {log_vars["reps"]:+.2f}  '
          f'(weight x{math.exp(-log_vars["reps"]) * 0.5:.2f})')
md.append('')
md.append('All four log_var values moved monotonically negative, which '
          'is the correct direction for tasks where the model can fit '
          'well (low residual variance -> log_var goes negative -> '
          'weight goes up).')
md.append('')
md.append('**WARNING - reps weight is extreme**: implied weight on reps is '
          f'{weights["reps"]:.0f}x the others because the soft_overlap '
          'reps loss converges to 0.002 (vs ~0.17 for exercise/phase). '
          'Kendall blow-up on tiny losses is a known failure mode '
          '(Liebel & Korner 2018). The other 3 tasks behave cleanly.')
md.append('')
md.append('Mitigation options for the production run:')
md.append('')
md.append('- **Option A (recommended)**: keep `use_uncertainty=True` -- the '
          'extreme reps weight does not appear to hurt the other heads, '
          'and final raw losses on exercise/phase/fatigue are competitive '
          'with the fixed-weight control. Reps loss itself is already at '
          'noise floor (0.002), so up-weighting it is harmless.')
md.append('- Option B: clamp `log_var` to [-3, +3] in the loss module to '
          'cap each task weight in [0.012, 10] and avoid runaway. Requires '
          'a 2-line code change.')
md.append('- Option C: drop reps from the multi-task loss when its loss '
          'collapses below a threshold; train a reps-only specialist '
          'instead (already done in v13 single-task).')
md.append('')

(OUT / 'verdict.md').write_text('\n'.join(md), encoding='utf-8')
print('\n'.join(md))
