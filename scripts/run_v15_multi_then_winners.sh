#!/usr/bin/env bash
# v15 feature pipeline:
#   1. extract per-recording window_features.parquet
#   2. LDA + ANOVA + MI feature relevance analysis
#   3. multi-task × 2 archs × 3 windows (6 runs)
#   4. pick best (arch, window) per task from multi-task results
#   5. single-task × 4 (one per task with the winning arch+window)
#
# ETA ~12-13 hours. Logs to logs/v15_multi_winners_<TS>.log.

set -euo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/v15_multi_winners_${TS}.log
mkdir -p logs

echo "== START $(date)" | tee "$LOG"

# ---------------------------------------------------------------------------
# Step 1: per-recording window_features.parquet
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
# Step 3: multi-task × 2 archs × 3 windows
# ---------------------------------------------------------------------------
for arch in mlp lstm; do
  for w in 1 2 5; do
    slug="v15multi-${arch}-w${w}s"
    echo "== multi-task ${slug} $(date)" | tee -a "$LOG"
    python -m scripts.train_optuna \
      --variant features --arch "$arch" --window-s "${w}.0" \
      --tasks exercise phase fatigue reps \
      --runs-root runs --run-dir "runs/$slug" \
      --n-trials 50 --phase1-epochs 30 --phase2-epochs 50 \
      2>&1 | tee -a "$LOG"
  done
done

# ---------------------------------------------------------------------------
# Step 4: pick winning (arch, window) per task from multi-task results
#         and run single-task with that (arch, window).
# Run as one Python block so subprocess is in scope.
# ---------------------------------------------------------------------------
echo "== analyze multi-task winners + run single-task $(date)" | tee -a "$LOG"
python - <<PY 2>&1 | tee -a "$LOG"
import json, re, subprocess
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
        print(f'  loaded {slug}: {cands[0]}')
    else:
        print(f'  NO summary for {slug}')

def get_metric(d, task):
    if task in ('exercise', 'phase'):
        v = d.get(task, {}).get('f1_macro_mean')
        if v is None:
            v = d.get(task, {}).get('f1_macro')
        return v
    v = d.get(task, {}).get('mae_mean')
    if v is None:
        v = d.get(task, {}).get('mae')
    return v

winners = {}
for task in ('exercise', 'phase', 'fatigue', 'reps'):
    best_slug, best_val = None, None
    for slug, summary in results.items():
        val = get_metric(summary, task)
        if val is None: continue
        if best_val is None:
            best_val, best_slug = val, slug
        elif task in ('exercise','phase'):
            if val > best_val: best_val, best_slug = val, slug
        else:
            if val < best_val: best_val, best_slug = val, slug
    winners[task] = best_slug
    print(f'{task}: best multi-task = {best_slug} (val={best_val})')

Path('logs').mkdir(exist_ok=True)
with open('logs/v15_winners.json', 'w') as f:
    json.dump(winners, f, indent=2)

# Now launch single-task runs
for task, multi_slug in winners.items():
    if multi_slug is None:
        print(f'  WARNING: no winner for {task}, skipping single-task run')
        continue
    m = re.match(r'v15multi-(\w+)-w(\d+)s', multi_slug)
    if not m:
        print(f'  parse error for {multi_slug}'); continue
    arch, window = m.group(1), m.group(2)
    slug = f'v15single-{task}-{arch}-w{window}s'
    print(f'>> launching {slug}', flush=True)
    cmd = [
        'python', '-m', 'scripts.train_optuna',
        '--variant', 'features', '--arch', arch,
        '--window-s', f'{window}.0',
        '--tasks', task,
        '--runs-root', 'runs', '--run-dir', f'runs/{slug}',
        '--n-trials', '50', '--phase1-epochs', '30', '--phase2-epochs', '50',
    ]
    subprocess.run(cmd, check=False)
PY

echo "== DONE $(date)" | tee -a "$LOG"
