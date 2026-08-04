#!/usr/bin/env bash
# memory-budget-gate.sh — Warn or exit if peak RSS exceeds budget
#
# Usage: ./scripts/memory-budget-gate.sh --warn GB [--command "CMD"]
#
# Options:
#   --warn GB    Warning threshold in GB (default: 10)
#   --exit GB    Exit threshold in GB (default: unlimited)
#   --command    Command to run while monitoring memory
#
# Exit codes: 0 = OK (below warn), 1 = warning (above warn), 2 = exceeded exit

set -euo pipefail

WARN_GB=10
EXIT_GB=
COMMAND=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --warn) WARN_GB="$2"; shift 2 ;;
    --exit) EXIT_GB="$2"; shift 2 ;;
    --command) COMMAND="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

WARN_KB=$((WARN_GB * 1024 * 1024))
EXIT_KB=""
if [[ -n "$EXIT_GB" ]]; then
  EXIT_KB=$((EXIT_GB * 1024 * 1024))
fi

echo "Memory budget gate"
echo "Warning threshold: ${WARN_GB} GB"
if [[ -n "$EXIT_KB" ]]; then
  echo "Exit threshold: ${EXIT_GB} GB"
fi

MAX_RSS_KB=0

monitor_memory() {
  while true; do
    RSS=$(ps -o rss= -p $$ 2>/dev/null || echo 0)
    if [[ "$RSS" -gt "$MAX_RSS_KB" ]]; then
      MAX_RSS_KB=$RSS
    fi
    sleep 0.5
  done
}

if [[ -n "$COMMAND" ]]; then
  monitor_memory &
  MONITOR_PID=$!

  trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

  eval "$COMMAND"
  kill $MONITOR_PID 2>/dev/null || true
  wait $MONITOR_PID 2>/dev/null || true
fi

MAX_RSS_GB=$(echo "scale=2; $MAX_RSS_KB / 1024 / 1024" | bc)
echo ""
echo "Peak RSS: ${MAX_RSS_GB} GB"

if [[ -n "$EXIT_KB" ]] && [[ "$MAX_RSS_KB" -gt "$EXIT_KB" ]]; then
  echo "ERROR: Peak RSS ${MAX_RSS_GB} GB exceeds exit threshold ${EXIT_GB} GB" >&2
  exit 2
fi

if [[ "$MAX_RSS_KB" -gt "$WARN_KB" ]]; then
  echo "WARNING: Peak RSS ${MAX_RSS_GB} GB exceeds warning threshold ${WARN_GB} GB" >&2
  exit 1
fi

echo "Memory budget check passed (${MAX_RSS_GB} GB < ${WARN_GB} GB)"
exit 0
