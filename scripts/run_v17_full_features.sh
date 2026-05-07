#!/usr/bin/env bash
# v17 full feature pipeline campaign:
#   1. Per-recording window_features.parquet (feature extraction)
#   2. RF (LightGBM) baseline — 4 tasks, fast classical baseline
#   3. 6 multi-task NN: feat-mlp + feat-lstm × {1, 2, 5}s windows
#   4. Pick winning (arch, window) per task from multi-task results
#   5. 4 single-task NN with the winning (arch, window) per task
#
# All NN campaigns use new defaults:
#   --tight-hps (default ON)
#   --n-trials 50, --phase1-epochs 50
#   --phase2-epochs 150, --patience 10, 3 seeds
#   --splits configs/splits_loso.csv (10 LOSO folds)
#
# 3-way parallel for NN campaigns. ETA ~9-12 hours.

set -uo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/v17_full_${TS}.log
mkdir -p logs

echo "== START $(date)" | tee "$LOG"

# ---------------------------------------------------------------------------
# Step 1: feature extraction (per-recording window_features.parquet)
# ---------------------------------------------------------------------------
echo "== feature extraction $(date)" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
from pathlib import Path
from src.features.window_features import process_recording

for rec in sorted(Path('data/labeled').glob('recording_*')):
    rec_id = rec.name
    print(f'=== {rec_id} ===', flush=True)
    win, st = process_recording(rec_id, Path('data/labeled'),
                                 Path('dataset_aligned'))
    win.to_parquet(rec / 'window_features.parquet', index=False)
    st.to_parquet(rec / 'set_features.parquet', index=False)
    print(f'  -> {len(win)} windows, {len(st)} sets', flush=True)
PY

# ---------------------------------------------------------------------------
# Step 2: RF (LightGBM) baseline
# ---------------------------------------------------------------------------
echo "== RF (LightGBM) baseline $(date)" | tee -a "$LOG"
RF_RUN=runs/v17rf_${TS}
mkdir -p "$RF_RUN"
python scripts/train_lgbm.py --run-dir "$RF_RUN" --splits configs/splits_loso.csv \
    2>&1 | tee logs/v17_rf.log

# ---------------------------------------------------------------------------
# Steps 3-5: NN campaigns (feature pipeline)
# ---------------------------------------------------------------------------
MAX_PAR=3
declare -a PIDS=()

throttle() {
  while true; do
    local alive=0
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    if [ "$alive" -lt "$MAX_PAR" ]; then return; fi
    sleep 5
  done
}

run_campaign() {
  local slug="$1"; shift
  local sublog="logs/v17_${slug}.log"
  echo ">> launching ${slug}  $(date)" | tee -a "$LOG"
  (python -m scripts.train_optuna --variant features \
      --runs-root runs --run-dir "runs/$slug" \
      "$@" \
      > "$sublog" 2>&1) &
  PIDS+=("$!")
}

# Step 3: multi-task × 2 archs × 3 windows
echo "== multi-task batch $(date)" | tee -a "$LOG"
for arch in mlp lstm; do
  for w in 1 2 5; do
    throttle
    run_campaign "v17multi-${arch}-w${w}s" \
      --arch "$arch" --window-s "${w}.0" \
      --tasks exercise phase fatigue reps
  done
done

wait
PIDS=()
echo "== multi-task batch DONE $(date)" | tee -a "$LOG"

# Step 4: pick winners and launch single-task
echo "== analyze winners $(date)" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import json, re
from pathlib import Path

results = {}
for arch in ('mlp','lstm'):
    for w in (1,2,5):
        slug = f'v17multi-{arch}-w{w}s'
        cv = list(Path(f'runs/{slug}').rglob('phase2/*/cv_summary.json'))
        if cv:
            d = json.loads(cv[0].read_text())['summary']
            results[slug] = (arch, w, d)
            print(f'  {slug}: ex={d["exercise"]["f1_macro"]["mean"]:.4f} '
                  f'ph={d["phase"]["f1_macro"]["mean"]:.4f} '
                  f'fat={d["fatigue"]["mae"]["mean"]:.4f} '
                  f'reps={d["reps"]["mae"]["mean"]:.4f}')

def best_max(t, mkey):
    return max(results.items(), key=lambda kv: kv[1][2][t][mkey]['mean'])
def best_min(t, mkey):
    return min(results.items(), key=lambda kv: kv[1][2][t][mkey]['mean'])

winners = {
    'exercise': best_max('exercise', 'f1_macro'),
    'phase':    best_max('phase', 'f1_macro'),
    'fatigue':  best_min('fatigue', 'mae'),
    'reps':     best_min('reps', 'mae'),
}
out = {}
for task, (slug, (arch, w, _)) in winners.items():
    out[task] = {'slug': slug, 'arch': arch, 'window': w}
    print(f'  {task}: best = {slug} ({arch} w{w}s)')

Path('logs').mkdir(exist_ok=True)
with open('logs/v17_winners.json','w') as f: json.dump(out, f, indent=2)
PY

# Step 5: single-task with winning (arch, window) per task — 4 in parallel
echo "== single-task batch $(date)" | tee -a "$LOG"
PIDS=()
for line in $(python -c "
import json
w = json.load(open('logs/v17_winners.json'))
for task, info in w.items():
    print(f\"{task}:{info['arch']}:{info['window']}\")
"); do
  task=$(echo "$line" | cut -d: -f1)
  arch=$(echo "$line" | cut -d: -f2)
  w=$(echo "$line" | cut -d: -f3)
  throttle
  run_campaign "v17single-${task}-${arch}-w${w}s" \
    --arch "$arch" --window-s "${w}.0" \
    --tasks "$task"
done

wait
echo "== ALL DONE $(date)" | tee -a "$LOG"

# Aggregate to CSV
python - <<'PY' 2>&1 | tee -a "$LOG"
import json, csv
from pathlib import Path

rows = []
for run_dir in sorted(Path('runs').glob('v17*')):
    if not run_dir.is_dir(): continue
    cv = list(run_dir.rglob('phase2/*/cv_summary.json'))
    if not cv: continue
    s = json.loads(cv[0].read_text()).get('summary', {})
    if not s: continue
    rows.append({
        'slug': run_dir.name,
        'val_total_mean': s.get('val_total',{}).get('mean'),
        'val_total_std':  s.get('val_total',{}).get('std'),
        'exercise_f1':    s.get('exercise',{}).get('f1_macro',{}).get('mean'),
        'exercise_f1_std':s.get('exercise',{}).get('f1_macro',{}).get('std'),
        'phase_f1':       s.get('phase',{}).get('f1_macro',{}).get('mean'),
        'phase_f1_std':   s.get('phase',{}).get('f1_macro',{}).get('std'),
        'fatigue_mae':    s.get('fatigue',{}).get('mae',{}).get('mean'),
        'fatigue_mae_std':s.get('fatigue',{}).get('mae',{}).get('std'),
        'fatigue_r':      s.get('fatigue',{}).get('pearson_r',{}).get('mean'),
        'reps_mae':       s.get('reps',{}).get('mae',{}).get('mean'),
        'reps_mae_std':   s.get('reps',{}).get('mae',{}).get('std'),
    })

out = Path('results/v17_full_features.csv')
out.parent.mkdir(parents=True, exist_ok=True)
if rows:
    with open(out, 'w', newline='', encoding='utf-8') as f:
        cw = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        cw.writeheader()
        for r in rows: cw.writerow(r)
    print(f'\nWrote {len(rows)} rows to {out}')
PY
