#!/usr/bin/env bash
# Train the two missing v17 mlp-w1s single-task models (fatigue + reps).
# Runs them in parallel with the same defaults as run_v17_full_features.sh.
set -uo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

run_one() {
  local task="$1"
  local slug="v17single-${task}-mlp-w1s"
  local log="logs/${slug}_${TS}.log"
  echo ">> launching ${slug}  $(date)"
  python -m scripts.train_optuna --variant features \
      --runs-root runs --run-dir "runs/${slug}" \
      --arch mlp --window-s 1.0 \
      --tasks "${task}" \
      > "${log}" 2>&1 &
  echo "   PID=$! log=${log}"
}

run_one fatigue
run_one reps

wait
echo "== ALL DONE $(date)"
