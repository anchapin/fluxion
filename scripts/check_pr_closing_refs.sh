#!/usr/bin/env bash
# scripts/check_pr_closing_refs.sh
#
# Verify a PR's `closingIssuesReferences` count matches the expected value.
# Used by wave-orchestrator sub-agents after `gh pr create` to catch the
# `--fill` auto-parse issue documented in docs/orchestration/pr-body-conventions.md.
#
# Usage:
#   bash scripts/check_pr_closing_refs.sh <PR_NUMBER> <EXPECTED_COUNT>
#
# Exit codes:
#   0 — count matches expected
#   1 — count does not match (with diagnostic output)
#   2 — usage error

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <pr_number> <expected_count>" >&2
  exit 2
fi

PR_NUMBER="$1"
EXPECTED="$2"

# Get the count of closingIssuesReferences
ACTUAL=$(gh pr view "$PR_NUMBER" --json closingIssuesReferences --jq '.closingIssuesReferences | length')

# Also fetch the issue numbers for diagnostic output
NUMBERS=$(gh pr view "$PR_NUMBER" --json closingIssuesReferences \
  --jq '[.closingIssuesReferences[].number] | join(",")')

if [[ "$ACTUAL" == "$EXPECTED" ]]; then
  echo "OK PR #$PR_NUMBER closingIssuesReferences count = $ACTUAL (issues: ${NUMBERS:-(none)})"
  exit 0
else
  echo "FAIL PR #$PR_NUMBER closingIssuesReferences count = $ACTUAL, expected $EXPECTED" >&2
  echo "  referenced issues: ${NUMBERS:-(none)}" >&2
  echo "  Fix: gh pr edit $PR_NUMBER --body \"Closes #<OWN>\"" >&2
  exit 1
fi
