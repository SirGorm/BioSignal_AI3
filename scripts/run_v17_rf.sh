#!/usr/bin/env bash
# v17 RF baseline. Concats per-recording parquets into single file
# expected by train_lgbm.py, then runs it with LOSO 10-fold.

set -uo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=runs/v17rf_loso_${TS}
mkdir -p "$RUN_DIR"
LOG=logs/v17_rf_${TS}.log
mkdir -p logs

echo "== START $(date)" | tee "$LOG"

# Concat per-recording parquets into single window_features.parquet + set_features.parquet
python - <<PY 2>&1 | tee -a "$LOG"
from pathlib import Path
import pandas as pd

run_dir = Path("$RUN_DIR")
labeled = Path("data/labeled")

wins = []
sets = []
for rec in sorted(labeled.glob("recording_*")):
    wp = rec / "window_features.parquet"
    sp = rec / "set_features.parquet"
    if wp.exists():
        wins.append(pd.read_parquet(wp))
    if sp.exists():
        sets.append(pd.read_parquet(sp))

wf = pd.concat(wins, ignore_index=True)
sf = pd.concat(sets, ignore_index=True)
wf.to_parquet(run_dir / "window_features.parquet", index=False)
sf.to_parquet(run_dir / "set_features.parquet", index=False)
print(f"Concatenated: window_features {wf.shape}  set_features {sf.shape}")
print(f"  Subjects: {sorted(wf['subject_id'].unique())}")
PY

# Run train_lgbm.py with LOSO 10-fold
echo "== running train_lgbm.py (LOSO 10-fold)  $(date)" | tee -a "$LOG"
python scripts/train_lgbm.py \
    --run-dir "$RUN_DIR" \
    --features-dir "$RUN_DIR" \
    --splits configs/splits_loso.csv \
    --n-folds 10 \
    2>&1 | tee -a "$LOG"

echo "== DONE $(date)" | tee -a "$LOG"
echo "Results in $RUN_DIR" | tee -a "$LOG"

# Summary
python - <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
run = Path("$RUN_DIR")
print('=== v17 RF (LOSO) results ===')
for mc in sorted(run.rglob("model_card.md")):
    print(f'  {mc}')
for j in sorted(run.rglob("metrics.json")):
    print(f'  {j}')
    try:
        d = json.loads(j.read_text())
        print(f'    {d}')
    except Exception:
        pass
PY
