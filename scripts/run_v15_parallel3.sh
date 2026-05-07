#!/usr/bin/env bash
# v15 feature pipeline with 3-way parallelism.
#   1. extract per-recording window_features.parquet (sequential)
#   2. LDA + ANOVA + MI feature relevance (sequential)
#   3. multi-task × 6, max 3 concurrent
#   4. pick best (arch, window) per task
#   5. single-task × 4, max 3 concurrent
#
# ETA ~5.5-6 hours. GPU ~4.5-6 GB used (well within 16 GB).
# Logs: logs/v15_parallel3_<TS>.log + per-campaign log under logs/v15p_<slug>.log

set -uo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/v15_parallel3_${TS}.log
mkdir -p logs

echo "== START $(date)" | tee "$LOG"

# ---------------------------------------------------------------------------
# Step 1: per-recording window_features.parquet (sequential, sub-process)
# ---------------------------------------------------------------------------
echo "== feature extraction $(date)" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
from pathlib import Path
from src.features.window_features import process_recording

for rec in sorted(Path('data/labeled').glob('recording_*')):
    rec_id = rec.name
    print(f'=== feature extraction: {rec_id} ===', flush=True)
    win_df, set_df = process_recording(rec_id, Path('data/labeled'),
                                        Path('dataset_aligned'))
    win_df.to_parquet(rec / 'window_features.parquet', index=False)
    set_df.to_parquet(rec / 'set_features.parquet', index=False)
    print(f'  -> {len(win_df)} windows, {len(set_df)} sets', flush=True)
PY

# ---------------------------------------------------------------------------
# Step 2: LDA / ANOVA / MI feature relevance
# ---------------------------------------------------------------------------
echo "== LDA feature analysis $(date)" | tee -a "$LOG"
python scripts/analyze_features.py 2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------------------
# Helpers: launch one campaign in background, throttle to MAX_PAR concurrent
# ---------------------------------------------------------------------------
MAX_PAR=3
declare -a PIDS=()

# Wait until at most MAX_PAR-1 children are alive (so launching one more = MAX_PAR)
throttle() {
  while true; do
    local alive=0
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done
    if [ "$alive" -lt "$MAX_PAR" ]; then
      return
    fi
    sleep 5
  done
}

launch_multi() {
  local arch="$1"; local w="$2"
  local slug="v15multi-${arch}-w${w}s"
  local sublog="logs/v15p_${slug}.log"
  echo ">> launching ${slug} -> ${sublog} $(date)" | tee -a "$LOG"
  (python -m scripts.train_optuna \
      --variant features --arch "$arch" --window-s "${w}.0" \
      --tasks exercise phase fatigue reps \
      --runs-root runs --run-dir "runs/$slug" \
      --n-trials 50 --phase1-epochs 30 --phase2-epochs 50 \
      > "$sublog" 2>&1) &
  PIDS+=("$!")
}

launch_single() {
  local task="$1"; local arch="$2"; local w="$3"
  local slug="v15single-${task}-${arch}-w${w}s"
  local sublog="logs/v15p_${slug}.log"
  echo ">> launching ${slug} -> ${sublog} $(date)" | tee -a "$LOG"
  (python -m scripts.train_optuna \
      --variant features --arch "$arch" --window-s "${w}.0" \
      --tasks "$task" \
      --runs-root runs --run-dir "runs/$slug" \
      --n-trials 50 --phase1-epochs 30 --phase2-epochs 50 \
      > "$sublog" 2>&1) &
  PIDS+=("$!")
}

# ---------------------------------------------------------------------------
# Step 3: multi-task × 6, max 3 concurrent
# ---------------------------------------------------------------------------
echo "== multi-task batch $(date)" | tee -a "$LOG"
for arch in mlp lstm; do
  for w in 1 2 5; do
    throttle
    launch_multi "$arch" "$w"
  done
done

# Wait for ALL multi-task children to finish before single-task
wait
PIDS=()
echo "== multi-task batch DONE $(date)" | tee -a "$LOG"

# ---------------------------------------------------------------------------
# Step 4: pick winning (arch, window) per task, then launch single-task × 4
# ---------------------------------------------------------------------------
echo "== analyze winners $(date)" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import json, re
from pathlib import Path

runs_root = Path('runs')
slugs = [f'v15multi-{arch}-w{w}s' for arch in ('mlp','lstm') for w in (1,2,5)]
results = {}
for slug in slugs:
    cands = list((runs_root / slug).rglob('phase2_summary.json'))
    if not cands:
        cands = list((runs_root / slug).rglob('summary.json'))
    if cands:
        results[slug] = json.loads(cands[0].read_text())

def get_metric(d, task):
    if task in ('exercise','phase'):
        v = d.get(task,{}).get('f1_macro_mean') or d.get(task,{}).get('f1_macro')
        return v
    v = d.get(task,{}).get('mae_mean') or d.get(task,{}).get('mae')
    return v

winners = {}
for task in ('exercise','phase','fatigue','reps'):
    best_slug, best_val = None, None
    for slug, summary in results.items():
        val = get_metric(summary, task)
        if val is None: continue
        if best_val is None: best_val, best_slug = val, slug
        elif task in ('exercise','phase'):
            if val > best_val: best_val, best_slug = val, slug
        else:
            if val < best_val: best_val, best_slug = val, slug
    winners[task] = best_slug
    print(f'{task}: best multi = {best_slug} ({best_val})')

with open('logs/v15_winners.json','w') as f: json.dump(winners, f, indent=2)
PY

echo "== single-task batch $(date)" | tee -a "$LOG"
for task_arch_w in $(python - <<'PY'
import json, re
w = json.load(open('logs/v15_winners.json'))
for task, slug in w.items():
    if slug is None: continue
    m = re.match(r'v15multi-(\w+)-w(\d+)s', slug)
    if m: print(f'{task}:{m.group(1)}:{m.group(2)}')
PY
); do
  task=$(echo "$task_arch_w" | cut -d: -f1)
  arch=$(echo "$task_arch_w" | cut -d: -f2)
  w=$(echo "$task_arch_w" | cut -d: -f3)
  throttle
  launch_single "$task" "$arch" "$w"
done

wait
echo "== DONE $(date)" | tee -a "$LOG"
