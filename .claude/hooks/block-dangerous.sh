#!/usr/bin/env bash
# Block dangerous shell commands.
set -euo pipefail

input=$(cat)
cmd=$(echo "$input" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('tool_input', {}).get('command', ''))" 2>/dev/null || true)

# Patterns to block
if [[ "$cmd" =~ rm\ -rf\ / || "$cmd" =~ rm\ -rf\ \~ ]]; then
  echo "BLOCKED: refusing to run destructive recursive delete on root or home." >&2
  exit 2
fi
if [[ "$cmd" =~ git\ push.*--force ]]; then
  echo "BLOCKED: force push not allowed via agent. Run manually if intended." >&2
  exit 2
fi
if [[ "$cmd" =~ chmod\ -R.*777 ]]; then
  echo "BLOCKED: chmod 777 recursive is rarely correct." >&2
  exit 2
fi

exit 0
