#!/usr/bin/env bash
# scripts/verify_issues_closed.sh
#
# Verify a given issue is closed using the GitHub CLI.
# Used by wave-orchestrator sub-agents to confirm an issue was closed.
#
# Usage:
#   bash scripts/verify_issues_closed.sh <ISSUE_NUMBER>
#
# Exit codes:
#   0 — issue is closed
#   1 — issue is not closed (open, or other state)
#   2 — usage error

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <issue_number>" >&2
  exit 2
fi

ISSUE_NUMBER="$1"

# The GitHub CLI returns state in UPPERCASE (CLOSED, OPEN); lowercase it so the
# comparison is case-insensitive (issue #2488).
STATE=$(gh issue view "$ISSUE_NUMBER" --json state --jq '.state' | tr '[:upper:]' '[:lower:]')

case "$STATE" in
  closed)
    echo "OK Issue #$ISSUE_NUMBER is closed"
    exit 0
    ;;
  open)
    echo "FAIL Issue #$ISSUE_NUMBER is still open" >&2
    exit 1
    ;;
  *)
    echo "FAIL Issue #$ISSUE_NUMBER has unexpected state: $STATE" >&2
    exit 1
    ;;
esac
