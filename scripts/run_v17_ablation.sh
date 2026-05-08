#!/usr/bin/env bash
# v17 modality ablation: for each of 6 multi-task feature models,
# drop one modality at a time and re-evaluate with phase 2 only
# (reuses best HPs from v17 Optuna search). Same LOSO + tight + 3 seeds.
#
# 6 archs × 4 modalities = 24 phase-2 runs. 3-parallel.
# ETA ~1.5-2 hours.

set -uo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/v17_ablation_${TS}.log
mkdir -p logs

echo "== START $(date)" | tee "$LOG"

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

# Feature prefixes for each modality kept (3 of 4)
# emg_, acc_, ppg_, temp_
declare -A KEEP_WITHOUT
KEEP_WITHOUT[emg]="acc_ ppg_ temp_"
KEEP_WITHOUT[acc]="emg_ ppg_ temp_"
KEEP_WITHOUT[ppg]="emg_ acc_ temp_"
KEEP_WITHOUT[temp]="emg_ acc_ ppg_"

run_ablation() {
  local arch="$1"; local w="$2"; local drop="$3"
  local src_slug="v17multi-${arch}-w${w}s"
  local out_slug="v17abl-drop_${drop}-${arch}-w${w}s"
  local sublog="logs/v17abl_${out_slug}.log"
  local kept="${KEEP_WITHOUT[$drop]}"
  echo ">> launching ${out_slug} (keep: ${kept})  $(date)" | tee -a "$LOG"
  (python -m scripts.train_phase2_only \
      --arch "$arch" --variant features --window-s "${w}.0" \
      --tasks exercise phase fatigue reps \
      --src-run-dir "runs/$src_slug" \
      --out-run-dir "runs/$out_slug" \
      --feature-prefixes ${kept} \
      --reps-mode soft_window --phase-mode soft \
      --phase2-seeds 42 1337 7 \
      --phase2-epochs 150 --patience 10 \
      --splits configs/splits_loso.csv \
      > "$sublog" 2>&1) &
  PIDS+=("$!")
}

# Sanity: verify all 6 src run-dirs have best_hps.json
for arch in mlp lstm; do
  for w in 1 2 5; do
    if [ ! -f "runs/v17multi-${arch}-w${w}s/best_hps.json" ]; then
      echo "ERROR: missing runs/v17multi-${arch}-w${w}s/best_hps.json" | tee -a "$LOG"
      exit 1
    fi
  done
done
echo "All 6 source run-dirs found." | tee -a "$LOG"

# 24 ablations: 2 archs × 3 windows × 4 modalities
for arch in mlp lstm; do
  for w in 1 2 5; do
    for drop in emg acc ppg temp; do
      throttle
      run_ablation "$arch" "$w" "$drop"
    done
  done
done

wait
echo "== ALL ABLATIONS DONE $(date)" | tee -a "$LOG"

# Aggregate to CSV
python - <<'PY' 2>&1 | tee -a "$LOG"
import json, csv, re
from pathlib import Path

rows = []

# Reference rows from v17 multi-task (no ablation = "full")
for arch in ('mlp','lstm'):
    for w in (1,2,5):
        slug = f'v17multi-{arch}-w{w}s'
        cv = list(Path(f'runs/{slug}').rglob('phase2/*/cv_summary.json'))
        if not cv: continue
        s = json.loads(cv[0].read_text())['summary']
        rows.append({
            'modality_dropped': 'none',
            'arch': arch, 'window_s': w,
            'src_slug': slug,
            'exercise_f1': s['exercise']['f1_macro']['mean'],
            'phase_f1':    s['phase']['f1_macro']['mean'],
            'fatigue_mae': s['fatigue']['mae']['mean'],
            'reps_mae':    s['reps']['mae']['mean'],
        })

# Ablation rows
for slug in sorted(Path('runs').glob('v17abl-*')):
    m = re.match(r'v17abl-drop_(\w+)-(\w+)-w(\d+)s', slug.name)
    if not m: continue
    drop, arch, w = m.group(1), m.group(2), int(m.group(3))
    cv = list(slug.rglob('phase2/*/cv_summary.json'))
    if not cv: continue
    s = json.loads(cv[0].read_text())['summary']
    rows.append({
        'modality_dropped': drop,
        'arch': arch, 'window_s': w,
        'src_slug': slug.name,
        'exercise_f1': s['exercise']['f1_macro']['mean'],
        'phase_f1':    s['phase']['f1_macro']['mean'],
        'fatigue_mae': s['fatigue']['mae']['mean'],
        'reps_mae':    s['reps']['mae']['mean'],
    })

out = Path('results/v17_ablation_results.csv')
out.parent.mkdir(parents=True, exist_ok=True)
fields = ['modality_dropped','arch','window_s','src_slug',
          'exercise_f1','phase_f1','fatigue_mae','reps_mae']
with open(out, 'w', newline='', encoding='utf-8') as f:
    cw = csv.DictWriter(f, fieldnames=fields)
    cw.writeheader()
    for r in rows: cw.writerow(r)
print(f'\nWrote {len(rows)} rows to {out}')

# Per-arch impact summary
print('\n=== Modality importance (delta from full to dropped) ===')
print(f'{"arch+w":<12} {"drop":<6} {"Δexer":>8} {"Δphase":>8} {"Δfat":>8} {"Δreps":>8}')
print('-' * 60)
for arch in ('mlp','lstm'):
    for w in (1,2,5):
        full = next((r for r in rows
                      if r['arch']==arch and r['window_s']==w
                      and r['modality_dropped']=='none'), None)
        if not full: continue
        for drop in ('emg','acc','ppg','temp'):
            ab = next((r for r in rows
                       if r['arch']==arch and r['window_s']==w
                       and r['modality_dropped']==drop), None)
            if not ab: continue
            dex = ab['exercise_f1'] - full['exercise_f1']
            dph = ab['phase_f1']    - full['phase_f1']
            df  = full['fatigue_mae'] - ab['fatigue_mae']  # MAE: lower better
            dr  = full['reps_mae']    - ab['reps_mae']
            print(f'{arch}-w{w}s     {drop:<6} {dex:>+8.3f} {dph:>+8.3f} '
                  f'{df:>+8.3f} {dr:>+8.3f}')
PY

echo "== DONE $(date)" | tee -a "$LOG"
