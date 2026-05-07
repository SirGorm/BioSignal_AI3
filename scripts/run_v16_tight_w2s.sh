#!/usr/bin/env bash
# v16 tight w2s only: feat-mlp + raw-tcn at 2-second windows.
# Runs both in parallel. ETA ~60-90 min.

set -uo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/v16_tight_w2s_${TS}.log
mkdir -p logs

echo "== START $(date)" | tee "$LOG"

# Launch both in parallel
echo ">> launching v16tight-mlp-w2s-features  $(date)" | tee -a "$LOG"
(python -m scripts.train_optuna \
    --variant features --arch mlp --window-s 2.0 \
    --tasks exercise phase fatigue reps \
    --runs-root runs --run-dir runs/v16tight-mlp-w2s-features \
    --tight-hps \
    --n-trials 50 \
    --phase1-epochs 30 --phase2-epochs 30 \
    --patience 4 \
    > logs/v16t_mlp-w2s.log 2>&1) &
P1=$!

echo ">> launching v16tight-tcn_raw-w2s-raw  $(date)" | tee -a "$LOG"
(python -m scripts.train_optuna \
    --variant raw --arch tcn_raw --window-s 2.0 \
    --tasks exercise phase fatigue reps \
    --runs-root runs --run-dir runs/v16tight-tcn_raw-w2s-raw \
    --tight-hps \
    --n-trials 50 \
    --phase1-epochs 30 --phase2-epochs 30 \
    --patience 4 \
    > logs/v16t_tcn-w2s.log 2>&1) &
P2=$!

wait $P1 $P2
echo "== TRAINING DONE $(date)" | tee -a "$LOG"

# Aggregate results
python - <<'PY' 2>&1 | tee -a "$LOG"
import json, csv
from pathlib import Path

rows = []
for slug, arch, variant in [
    ('v16tight-mlp-w2s-features', 'mlp', 'features'),
    ('v16tight-tcn_raw-w2s-raw', 'tcn_raw', 'raw'),
]:
    cv_files = list(Path(f'runs/{slug}').rglob('phase2/*/cv_summary.json'))
    if not cv_files:
        print(f'  NO cv_summary for {slug}'); continue
    s = json.loads(cv_files[0].read_text())['summary']
    row = {
        'slug': slug, 'variant': variant, 'arch': arch, 'window_s': 2,
        'val_total_mean': s['val_total']['mean'],
        'val_total_std':  s['val_total']['std'],
        'exercise_f1_mean': s['exercise']['f1_macro']['mean'],
        'exercise_f1_std':  s['exercise']['f1_macro']['std'],
        'phase_f1_mean':    s['phase']['f1_macro']['mean'],
        'phase_f1_std':     s['phase']['f1_macro']['std'],
        'fatigue_mae_mean': s['fatigue']['mae']['mean'],
        'fatigue_mae_std':  s['fatigue']['mae']['std'],
        'fatigue_pearson_mean': s['fatigue']['pearson_r']['mean'],
        'reps_mae_mean': s['reps']['mae']['mean'],
        'reps_mae_std':  s['reps']['mae']['std'],
    }
    rows.append(row)

out = Path('results/v16tight_w2s_results.csv')
out.parent.mkdir(parents=True, exist_ok=True)
if rows:
    with open(out, 'w', newline='', encoding='utf-8') as f:
        cw = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        cw.writeheader()
        for r in rows: cw.writerow(r)
    print(f'\nWrote {len(rows)} rows to {out}')

    print('\n=== v16 tight w2s results ===')
    for r in rows:
        print(f'{r["slug"]:<35} '
              f'ex={r["exercise_f1_mean"]:.4f}±{r["exercise_f1_std"]:.3f}  '
              f'ph={r["phase_f1_mean"]:.4f}±{r["phase_f1_std"]:.3f}  '
              f'fat={r["fatigue_mae_mean"]:.4f}±{r["fatigue_mae_std"]:.3f}  '
              f'reps={r["reps_mae_mean"]:.4f}±{r["reps_mae_std"]:.3f}')

    print('\n=== Compare to baselines ===')
    print('feat-mlp w2s  baseline (v15): ex=0.5402  ph=0.3398  fat=1.2757  reps=0.1688')
    print('raw-tcn  w2s  baseline (v19): ex=0.3340  ph=0.2745  fat=1.0522  reps=0.2061')
PY

echo "== DONE $(date)" | tee -a "$LOG"
