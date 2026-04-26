#!/usr/bin/env bash
# Auto-format and lint Python files after Claude edits them.
set -euo pipefail

input=$(cat)
file_path=$(echo "$input" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('tool_input', {}).get('file_path', ''))" 2>/dev/null || true)

if [[ -z "$file_path" || ! -f "$file_path" ]]; then
  exit 0
fi

if [[ "$file_path" =~ \.py$ ]]; then
  if command -v ruff > /dev/null 2>&1; then
    ruff format "$file_path" 2>/dev/null || true
    ruff check --fix "$file_path" 2>/dev/null || true
  fi
fi

exit 0
