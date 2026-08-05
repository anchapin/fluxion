#!/usr/bin/env bash
# scripts/refresh_known_issues.sh
#
# Refresh the Last Updated date in docs/KNOWN_ISSUES.md to today.
# Run this after reviewing and updating the document's sections.
#
# Usage:
#   bash scripts/refresh_known_issues.sh [--dry-run]
#
# Exit codes:
#   0 — date refreshed successfully
#   1 — file not found or date already current

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

KNOWN_ISSUES_PATH="docs/KNOWN_ISSUES.md"

if [[ ! -f "$KNOWN_ISSUES_PATH" ]]; then
  echo "FAIL: $KNOWN_ISSUES_PATH not found" >&2
  exit 1
fi

TODAY=$(date '+%Y-%m-%d')

# Check current date
CURRENT_DATE=$(grep -oP '\*Last Updated: \K[0-9]{4}-[0-9]{2}-[0-9]{2}' "$KNOWN_ISSUES_PATH" || true)

if [[ -z "$CURRENT_DATE" ]]; then
  echo "WARN: Could not find '*Last Updated: YYYY-MM-DD*' in $KNOWN_ISSUES_PATH" >&2
fi

if [[ "$CURRENT_DATE" == "$TODAY" ]]; then
  echo "OK: $KNOWN_ISSUES_PATH is already current ($TODAY)"
  exit 0
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY-RUN: would update Last Updated from '$CURRENT_DATE' to '$TODAY'"
  exit 0
fi

# Replace the date using sed (BSD/macOS compatible)
if sed -i '' "s/\*Last Updated: [0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}\*/\*Last Updated: $TODAY\*/" "$KNOWN_ISSUES_PATH"; then
  echo "OK: $KNOWN_ISSUES_PATH Last Updated refreshed to $TODAY"
else
  # Fallback for GNU sed
  if sed -i "s/\*Last Updated: [0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}\*/\*Last Updated: $TODAY\*/" "$KNOWN_ISSUES_PATH"; then
    echo "OK: $KNOWN_ISSUES_PATH Last Updated refreshed to $TODAY"
  else
    echo "FAIL: could not update $KNOWN_ISSUES_PATH" >&2
    exit 1
  fi
fi
