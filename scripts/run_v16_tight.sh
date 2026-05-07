#!/usr/bin/env bash
# v16 tight-HP campaign: feat-mlp + raw-tcn × 3 windows = 6 multi-task runs
# with shrunk Optuna search space (repr_dim<=64, dropout>=0.3,
# weight_decay>=1e-4, LSTM layers=1) and tighter early stopping
# (patience=4, phase2_epochs=30) to combat overfit.
#
# 3-way parallel. ETA ~3-4 hours.
# Logs: logs/v16_tight_<TS>.log + per-campaign log under logs/v16t_<slug>.log

set -uo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/v16_tight_${TS}.log
mkdir -p logs

echo "== START $(date)" | tee "$LOG"

MAX_PAR=3
declare -a PIDS=()

throttle() {
  while true; do
    local alive=0
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done
    if [ "$alive" -lt "$MAX_PAR" ]; then return; fi
    sleep 5
  done
}

launch() {
  local variant="$1"; local arch="$2"; local w="$3"
  local slug="v16tight-${arch}-w${w}s-${variant}"
  local sublog="logs/v16t_${slug}.log"
  echo ">> launching ${slug}  $(date)" | tee -a "$LOG"
  (python -m scripts.train_optuna \
      --variant "$variant" --arch "$arch" --window-s "${w}.0" \
      --tasks exercise phase fatigue reps \
      --runs-root runs --run-dir "runs/$slug" \
      --tight-hps \
      --n-trials 50 \
      --phase1-epochs 30 \
      --phase2-epochs 30 \
      --patience 4 \
      > "$sublog" 2>&1) &
  PIDS+=("$!")
}

# 6 campaigns: feat-mlp + raw-tcn × 1s/2s/5s
for w in 1 2 5; do
  throttle
  launch features mlp "$w"
done
for w in 1 2 5; do
  throttle
  launch raw tcn_raw "$w"
done

wait
echo "== DONE $(date)" | tee -a "$LOG"

# Aggregate winners and write a summary CSV
python - <<'PY' 2>&1 | tee -a "$LOG"
import json, csv
from pathlib import Path

rows = []
for slug_dir in sorted(Path('runs').glob('v16tight-*')):
    name = slug_dir.name
    # parse arch from slug
    import re
    m = re.match(r'v16tight-(\w+?)-w(\d+)s-(\w+)', name)
    if not m: continue
    arch, w, variant = m.group(1), int(m.group(2)), m.group(3)

    cv = list(slug_dir.rglob('phase2/*/cv_summary.json'))
    if not cv: continue
    s = json.loads(cv[0].read_text())['summary']
    rows.append({
        'slug': name, 'variant': variant, 'arch': arch, 'window_s': w,
        'val_total': s['val_total']['mean'],
        'exercise_f1': s['exercise']['f1_macro']['mean'],
        'exercise_f1_std': s['exercise']['f1_macro']['std'],
        'phase_f1': s['phase']['f1_macro']['mean'],
        'phase_f1_std': s['phase']['f1_macro']['std'],
        'fatigue_mae': s['fatigue']['mae']['mean'],
        'fatigue_mae_std': s['fatigue']['mae']['std'],
        'fatigue_pearson': s['fatigue']['pearson_r']['mean'],
        'reps_mae': s['reps']['mae']['mean'],
        'reps_mae_std': s['reps']['mae']['std'],
    })

out = Path('results/v16tight_results.csv')
out.parent.mkdir(parents=True, exist_ok=True)
if rows:
    with open(out, 'w', newline='', encoding='utf-8') as f:
        cw = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        cw.writeheader()
        for r in rows: cw.writerow(r)
    print(f'\nWrote {len(rows)} rows to {out}')

    print('\n=== v16 tight-HP results ===')
    for r in rows:
        print(f'{r["slug"]:<35} ex={r["exercise_f1"]:.4f} ph={r["phase_f1"]:.4f} '
              f'fat={r["fatigue_mae"]:.4f} reps={r["reps_mae"]:.4f}')
PY

echo "== ALL DONE $(date)" | tee -a "$LOG"
