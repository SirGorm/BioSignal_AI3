#!/usr/bin/env bash
# v16 phase 2 with longer training: 200 epochs, patience 20.
# 1 seed, LOSO 10 folds. Uses HPs from v16-tight Optuna search (partial).
# ETA ~1.5-2.5 hours.

set -uo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/v16_phase2_long_${TS}.log
mkdir -p logs

echo "== START $(date)" | tee "$LOG"

# Run both in parallel
echo ">> launching long phase2 mlp w2s LOSO  $(date)" | tee -a "$LOG"
(python -m scripts.train_phase2_only \
    --arch mlp --variant features --window-s 2.0 \
    --tasks exercise phase fatigue reps \
    --src-run-dir runs/v16tight-mlp-w2s-features \
    --out-run-dir runs/v16phase2long-mlp-w2s-loso \
    --splits configs/splits_loso.csv \
    --tight-hps \
    --reps-mode soft_window --phase-mode soft \
    --phase2-seeds 42 \
    --phase2-epochs 200 --patience 20 \
    > logs/v16p2long_mlp-w2s.log 2>&1) &
P1=$!

echo ">> launching long phase2 tcn_raw w2s LOSO  $(date)" | tee -a "$LOG"
(python -m scripts.train_phase2_only \
    --arch tcn_raw --variant raw --window-s 2.0 \
    --tasks exercise phase fatigue reps \
    --src-run-dir runs/v16tight-tcn_raw-w2s-raw \
    --out-run-dir runs/v16phase2long-tcn_raw-w2s-loso \
    --splits configs/splits_loso.csv \
    --tight-hps \
    --reps-mode soft_window --phase-mode soft \
    --phase2-seeds 42 \
    --phase2-epochs 200 --patience 20 \
    > logs/v16p2long_tcn-w2s.log 2>&1) &
P2=$!

wait $P1 $P2
echo "== TRAINING DONE $(date)" | tee -a "$LOG"

python - <<'PY' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path

print('=== v16 LONG phase 2 LOSO results (1 seed × 10 folds, 200ep/patience=20) ===\n')
print('Baselines for comparison:')
print('  v15 mlp w2s (5fold-3seed wide-HP):  ex=0.5402  ph=0.3398  fat=1.2757  reps=0.1688')
print('  v19 tcn  w2s (5fold-3seed wide-HP):  ex=0.3340  ph=0.2745  fat=1.0522  reps=0.2061')
print('  v16 mlp short (LOSO-1seed-30ep):     ex=0.4950  ph=0.3316  fat=1.7634  reps=0.2184')
print('  v16 tcn short (LOSO-1seed-30ep):     ex=0.1571  ph=0.2244  fat=1.0371  reps=0.2185')
print()

for slug in ['v16phase2long-mlp-w2s-loso', 'v16phase2long-tcn_raw-w2s-loso']:
    cv = list(Path(f'runs/{slug}').rglob('phase2/*/cv_summary.json'))
    if not cv:
        print(f'  NO cv_summary for {slug}'); continue
    s = json.loads(cv[0].read_text())['summary']
    print(f'{slug}:')
    print(f'  exercise F1 = {s["exercise"]["f1_macro"]["mean"]:.4f} ± {s["exercise"]["f1_macro"]["std"]:.3f}')
    print(f'  phase    F1 = {s["phase"]["f1_macro"]["mean"]:.4f} ± {s["phase"]["f1_macro"]["std"]:.3f}')
    print(f'  fatigue MAE = {s["fatigue"]["mae"]["mean"]:.4f} ± {s["fatigue"]["mae"]["std"]:.3f}')
    print(f'  fatigue r   = {s["fatigue"]["pearson_r"]["mean"]:.4f} ± {s["fatigue"]["pearson_r"]["std"]:.3f}')
    print(f'  reps    MAE = {s["reps"]["mae"]["mean"]:.4f} ± {s["reps"]["mae"]["std"]:.3f}')
    print()
PY

echo "== DONE $(date)" | tee -a "$LOG"
