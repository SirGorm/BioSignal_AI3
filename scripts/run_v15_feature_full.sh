#!/usr/bin/env bash
# Full feature-pipeline campaign: 3 single-task × 2 archs + 2 multi-task.
# Runs sequentially. ETA ~7.5-8 hours on RTX 5070 Ti.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/v15_feature_full_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs

echo "== START $(date)" | tee "$LOG"

# Step 1: regenerate per-recording window_features.parquet from the new
# aligned_features.parquet (post EMG-envelope + blacklist labeling).
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

run_optuna() {
  local slug="$1"; shift
  echo "== campaign $slug $(date)" | tee -a "$LOG"
  python -m scripts.train_optuna \
    --variant features \
    --runs-root runs \
    --run-dir "runs/$slug" \
    --n-trials 50 \
    --phase1-epochs 30 \
    --phase2-epochs 50 \
    "$@" 2>&1 | tee -a "$LOG"
}

# Single-task (matches v13single naming)
run_optuna v15single-exercise-only-w2s-feat-mlp  --arch mlp  --tasks exercise
run_optuna v15single-exercise-only-w2s-feat-lstm --arch lstm --tasks exercise
run_optuna v15single-phase-only-w2s-feat-mlp     --arch mlp  --tasks phase
run_optuna v15single-phase-only-w2s-feat-lstm    --arch lstm --tasks phase
run_optuna v15single-reps-only-w2s-feat-mlp      --arch mlp  --tasks reps
run_optuna v15single-reps-only-w2s-feat-lstm     --arch lstm --tasks reps

# Multi-task (matches v13soft naming)
run_optuna v15multi-feat-mlp  --arch mlp  --tasks exercise phase fatigue reps
run_optuna v15multi-feat-lstm --arch lstm --tasks exercise phase fatigue reps

echo "== DONE $(date)" | tee -a "$LOG"
