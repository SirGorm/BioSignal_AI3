#!/usr/bin/env bash
# Verify the labeled dataset has all expected columns after data-labeler runs.
set -euo pipefail

# Find aligned_features.parquet files modified in last 5 minutes
files=$(find data/labeled -name "aligned_features.parquet" -mmin -5 2>/dev/null || true)

if [[ -z "$files" ]]; then
  exit 0
fi

python3 - <<'PYEOF'
import sys
import os
from pathlib import Path

required_cols = {
    'subject_id', 'session_id', 't_unix', 't_session_s', 'in_active_set',
    'exercise', 'set_number', 'set_phase',
    'phase_label', 'rep_count_in_set', 'rpe_for_this_set',
    'ecg', 'emg', 'eda', 'temp', 'ax', 'ay', 'az', 'acc_mag', 'ppg_green',
}

errors = []
for line in os.popen('find data/labeled -name "aligned_features.parquet" -mmin -5').read().strip().split('\n'):
    if not line.strip():
        continue
    p = Path(line.strip())
    if not p.exists():
        continue
    try:
        import pyarrow.parquet as pq
        cols = set(pq.read_schema(p).names)
    except ImportError:
        try:
            import pandas as pd
            cols = set(pd.read_parquet(p).columns)
        except Exception as e:
            errors.append(f"{p}: cannot read: {e}")
            continue
    except Exception as e:
        errors.append(f"{p}: cannot read schema: {e}")
        continue

    missing = required_cols - cols
    if missing:
        errors.append(f"{p}: missing columns: {sorted(missing)}")

if errors:
    print("LABELED DATA VALIDATION FAILED:", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    print("", file=sys.stderr)
    print("data-labeler must produce all required columns before ml-expert runs.", file=sys.stderr)
    print("Required columns: " + ", ".join(sorted(required_cols)), file=sys.stderr)
    sys.exit(2)
PYEOF
