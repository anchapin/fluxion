#!/usr/bin/env bash
# scripts/refresh_known_issues.sh
#
# Refresh the Last Updated date in docs/KNOWN_ISSUES.md to today.
# Run this after reviewing and updating the document's sections.
#
# Usage:
#   bash scripts/refresh_known_issues.sh [--dry-run] [--file PATH]
#
# Exit codes:
#   0 — date refreshed successfully, already current, or --dry-run reported
#   1 — file not found, '*Last Updated: YYYY-MM-DD*' marker missing, or the
#       update failed (in all three cases the file is left unmodified)
#   2 — usage error (unknown argument)

set -euo pipefail

DRY_RUN=false
KNOWN_ISSUES_PATH="docs/KNOWN_ISSUES.md"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run] [--file PATH]

Refresh the '*Last Updated: YYYY-MM-DD*' marker in PATH (default:
docs/KNOWN_ISSUES.md) to today's date. Run this after reviewing and
updating the document's sections.

Options:
  --dry-run     Report the pending change without modifying the file.
  --file PATH   Operate on PATH instead of docs/KNOWN_ISSUES.md
                (used by scripts/ci/test_refresh_known_issues.py).
  -h, --help    Show this help and exit.

Exit codes:
  0  date refreshed, already current, or dry-run reported
  1  file not found / marker missing / update failed (file unmodified)
  2  usage error
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --file)
      if [[ $# -lt 2 ]]; then
        echo "FAIL: --file requires a path argument" >&2
        exit 2
      fi
      KNOWN_ISSUES_PATH="$2"
      shift 2
      ;;
    --file=*)
      KNOWN_ISSUES_PATH="${1#--file=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "FAIL: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$KNOWN_ISSUES_PATH" ]]; then
  echo "FAIL: $KNOWN_ISSUES_PATH not found" >&2
  exit 1
fi

TODAY=$(date '+%Y-%m-%d')

# Extract the current marker date. Portable BRE (works on BSD and GNU
# sed); avoids grep -P, which macOS grep does not support.
extract_date() {
  sed -n 's/.*\*Last Updated: \([0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}\).*/\1/p' "$1" | head -n 1
}

CURRENT_DATE=$(extract_date "$KNOWN_ISSUES_PATH")

if [[ -z "$CURRENT_DATE" ]]; then
  echo "FAIL: no '*Last Updated: YYYY-MM-DD*' marker in $KNOWN_ISSUES_PATH — file left unmodified" >&2
  exit 1
fi

if [[ "$CURRENT_DATE" == "$TODAY" ]]; then
  echo "OK: $KNOWN_ISSUES_PATH is already current ($TODAY)"
  exit 0
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY-RUN: would update Last Updated from '$CURRENT_DATE' to '$TODAY' (file unmodified)"
  exit 0
fi

# Replace only the date token, preserving any parenthetical review summary
# between the date and the closing '*' (e.g. the "(LIMIT-30 UPDATE ...)"
# annotations that check_known_issues_stale.py accepts). On the plain
# '*Last Updated: YYYY-MM-DD*' form this is byte-identical to the old GNU
# behavior; on the parenthetical form the old script silently no-op'd while
# reporting success — now the date is actually refreshed.
#
# Portable in-place edit: '-i.bak' works on both BSD and GNU sed, and the
# backup is removed immediately after. The old 'sed -i ""' first-try
# misparsed on GNU sed on every Linux run (the 's/.../' expression was
# taken as an input filename); its error was swallowed by the if/fallback
# sequence. This single invocation cannot hide a real failure behind a
# fallback, and the post-write date re-check below turns a silent
# non-substitution into a hard failure.
if sed -i.bak "s/\(\*Last Updated: \)[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}/\1${TODAY}/" "$KNOWN_ISSUES_PATH" \
  && rm -f "${KNOWN_ISSUES_PATH}.bak"; then
  NEW_DATE=$(extract_date "$KNOWN_ISSUES_PATH")
  if [[ "$NEW_DATE" == "$TODAY" ]]; then
    echo "OK: $KNOWN_ISSUES_PATH Last Updated refreshed to $TODAY"
    exit 0
  fi
fi

echo "FAIL: could not update $KNOWN_ISSUES_PATH" >&2
exit 1
