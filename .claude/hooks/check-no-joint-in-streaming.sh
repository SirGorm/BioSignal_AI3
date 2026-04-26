#!/usr/bin/env bash
# CRITICAL: Block any code in src/streaming/ from importing labeling/joint code.
# Joint-angle data is offline-only training-time supervision; the streaming
# pipeline must never depend on it.

set -euo pipefail

input=$(cat)
file_path=$(echo "$input" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('tool_input', {}).get('file_path', ''))" 2>/dev/null || true)

# Only check files in streaming module
if [[ ! "$file_path" =~ src/streaming/ ]]; then
  exit 0
fi

if [[ ! "$file_path" =~ \.py$ ]]; then
  exit 0
fi

if [[ ! -f "$file_path" ]]; then
  exit 0
fi

forbidden=(
  "from src\.labeling"
  "from \.\.labeling"
  "from labeling"
  "import labeling"
  "joint_angle"
  "joint_angles"
  "load_joint"
  "knee_angle"
  "elbow_angle"
  "hip_angle"
  "participants\.xlsx"
  "load_participants"
  "rpe_for_this_set"
  "phase_label"      # the offline-derived phase ground truth
  "rep_index"        # the offline-derived rep counter
)

violations=()
for pattern in "${forbidden[@]}"; do
  if grep -nE "$pattern" "$file_path" > /dev/null 2>&1; then
    matches=$(grep -nE "$pattern" "$file_path" | head -3)
    violations+=("$pattern:")
    while IFS= read -r line; do violations+=("    $line"); done <<< "$matches"
  fi
done

if [[ ${#violations[@]} -gt 0 ]]; then
  echo "BLOCKED: streaming code may not depend on offline labeling data: $file_path" >&2
  printf '%s\n' "${violations[@]}" >&2
  echo "" >&2
  echo "Joint-angle data is only available during training. The real-time pipeline" >&2
  echo "must predict phase/reps/exercise/fatigue from biosignals alone." >&2
  echo "" >&2
  echo "If you need offline labels at training time, do that work in src/labeling/" >&2
  echo "or src/features/ — never in src/streaming/." >&2
  exit 2
fi

exit 0
