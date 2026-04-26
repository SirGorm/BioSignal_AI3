#!/usr/bin/env bash
# Block non-causal operations in streaming/realtime code.
# Reads JSON from stdin (Claude Code hook payload), checks the file path.
# Exits 0 (allow) or 2 (block with stderr message).

set -euo pipefail

input=$(cat)
file_path=$(echo "$input" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('tool_input', {}).get('file_path', ''))" 2>/dev/null || true)

# Only check files in streaming or realtime modules
if [[ ! "$file_path" =~ (src/streaming/|src/pipeline/realtime|/streaming/) ]]; then
  exit 0
fi

# Only check Python files
if [[ ! "$file_path" =~ \.py$ ]]; then
  exit 0
fi

if [[ ! -f "$file_path" ]]; then
  exit 0
fi

forbidden=(
  "scipy\.signal\.filtfilt"
  "from scipy\.signal import .*filtfilt"
  "scipy\.signal\.savgol_filter"
  "np\.fft\.fft\(.*signal"
  "scipy\.signal\.find_peaks\(.*signal"
)

violations=()
for pattern in "${forbidden[@]}"; do
  if grep -nE "$pattern" "$file_path" > /dev/null 2>&1; then
    matches=$(grep -nE "$pattern" "$file_path")
    violations+=("$pattern -> $matches")
  fi
done

if [[ ${#violations[@]} -gt 0 ]]; then
  echo "BLOCKED: non-causal operation in real-time code path: $file_path" >&2
  printf '%s\n' "${violations[@]}" >&2
  echo "" >&2
  echo "Use causal alternatives (see real-time-pipeline skill):" >&2
  echo "  filtfilt -> sosfilt with persisted zi" >&2
  echo "  savgol_filter -> causal smoother" >&2
  echo "  fft over whole signal -> per-window STFT" >&2
  echo "  find_peaks over whole signal -> online state-machine peak detector" >&2
  exit 2
fi

exit 0
