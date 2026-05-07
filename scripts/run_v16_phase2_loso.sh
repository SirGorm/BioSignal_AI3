#!/usr/bin/env bash
# v16 phase 2 only: best HPs from partial v16 search, 1 seed, LOSO (10 folds).
# Both feat-mlp and raw-tcn at w2s.
# ETA ~30-50 min sequential, ~20-30 min parallel.

set -uo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/v16_phase2_loso_${TS}.log
mkdir -p logs

echo "== START $(date)" | tee "$LOG"

# Run both in parallel
echo ">> launching phase2 mlp w2s LOSO  $(date)" | tee -a "$LOG"
(python -m scripts.train_phase2_only \
    --arch mlp --variant features --window-s 2.0 \
    --tasks exercise phase fatigue reps \
    --src-run-dir runs/v16tight-mlp-w2s-features \
    --out-run-dir runs/v16phase2-mlp-w2s-loso \
    --splits configs/splits_loso.csv \
    --tight-hps \
    --reps-mode soft_window --phase-mode soft \
    --phase2-seeds 42 \
    --phase2-epochs 30 --patience 4 \
    > logs/v16p2_mlp-w2s.log 2>&1) &
P1=$!

echo ">> launching phase2 tcn_raw w2s LOSO  $(date)" | tee -a "$LOG"
(python -m scripts.train_phase2_only \
    --arch tcn_raw --variant raw --window-s 2.0 \
    --tasks exercise phase fatigue reps \
    --src-run-dir runs/v16tight-tcn_raw-w2s-raw \
    --out-run-dir runs/v16phase2-tcn_raw-w2s-loso \
    --splits configs/splits_loso.csv \
    --tight-hps \
    --reps-mode soft_window --phase-mode soft \
    --phase2-seeds 42 \
    --phase2-epochs 30 --patience 4 \
    > logs/v16p2_tcn-w2s.log 2>&1) &
P2=$!

wait $P1 $P2
echo "== TRAINING DONE $(date)" | tee -a "$LOG"

# Aggregate results
python - <<'PY' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path

print('=== v16 phase 2 LOSO results (1 seed × 10 folds) ===\n')
print('Baselines for comparison (v15/v19 multi-task w2s, 3 seeds × 5 folds):')
print('  feat-mlp: ex=0.5402  ph=0.3398  fat=1.2757  reps=0.1688')
print('  raw-tcn:  ex=0.3340  ph=0.2745  fat=1.0522  reps=0.2061')
print()

for slug in ['v16phase2-mlp-w2s-loso', 'v16phase2-tcn_raw-w2s-loso']:
    cv = list(Path(f'runs/{slug}').rglob('phase2/*/cv_summary.json'))
    if not cv:
        print(f'  NO cv_summary for {slug}'); continue
    s = json.loads(cv[0].read_text())['summary']
    print(f'{slug}:')
    print(f'  exercise F1 = {s["exercise"]["f1_macro"]["mean"]:.4f} '
          f'± {s["exercise"]["f1_macro"]["std"]:.3f}')
    print(f'  phase    F1 = {s["phase"]["f1_macro"]["mean"]:.4f} '
          f'± {s["phase"]["f1_macro"]["std"]:.3f}')
    print(f'  fatigue MAE = {s["fatigue"]["mae"]["mean"]:.4f} '
          f'± {s["fatigue"]["mae"]["std"]:.3f}')
    print(f'  fatigue r   = {s["fatigue"]["pearson_r"]["mean"]:.4f} '
          f'± {s["fatigue"]["pearson_r"]["std"]:.3f}')
    print(f'  reps    MAE = {s["reps"]["mae"]["mean"]:.4f} '
          f'± {s["reps"]["mae"]["std"]:.3f}')
    print()
PY

echo "== DONE $(date)" | tee -a "$LOG"
