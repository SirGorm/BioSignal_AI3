#!/usr/bin/env bash
# Verify that key deliverable documents (model_card.md, findings.md, quality_report.md)
# include a ## References section. This enforces the project's citation requirement.

set -euo pipefail

input=$(cat)
file_path=$(echo "$input" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('tool_input', {}).get('file_path', ''))" 2>/dev/null || true)

if [[ -z "$file_path" || ! -f "$file_path" ]]; then
  exit 0
fi

# Only check the specific deliverable filenames
filename=$(basename "$file_path")
case "$filename" in
  model_card.md|findings.md|quality_report.md) ;;
  *) exit 0 ;;
esac

# Check for ## References section
if grep -qE '^##\s+References' "$file_path"; then
  # Found the section. Now verify it's not empty or only contains TODO.
  refs_content=$(awk '/^##\s+References/{flag=1; next} /^##\s/{flag=0} flag' "$file_path")
  meaningful_lines=$(echo "$refs_content" | grep -cE '^\s*[-*•]\s+[A-Z]' || true)
  todo_only=$(echo "$refs_content" | grep -cE '\[REF NEEDED' || true)

  if [[ "$meaningful_lines" -lt 1 ]] && [[ "$todo_only" -gt 0 ]]; then
    echo "WARNING: $file_path has only [REF NEEDED] placeholders in References." >&2
    echo "Resolve them with the user before this document is considered complete." >&2
    # Warning, not block — this is allowed mid-iteration
    exit 0
  fi

  if [[ "$meaningful_lines" -lt 1 ]]; then
    echo "BLOCKED: $file_path has '## References' but no actual citations." >&2
    echo "Add citations from the literature-references skill, or remove the section." >&2
    exit 2
  fi

  exit 0
fi

# Section is missing — block
echo "BLOCKED: $file_path is missing the required '## References' section." >&2
echo "" >&2
echo "Every deliverable that documents methodological choices must cite literature." >&2
echo "Use the literature-references skill for canonical citations." >&2
echo "If a needed reference is genuinely absent from the skill, write '[REF NEEDED: <topic>]'" >&2
echo "as a placeholder and ask the user." >&2
exit 2
