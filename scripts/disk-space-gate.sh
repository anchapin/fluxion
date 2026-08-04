#!/usr/bin/env bash
# disk-space-gate.sh — Warn or exit if disk space is critically low
#
# Usage: ./scripts/disk-space-gate.sh [--warn GB] [--exit GB] [--path DIR]
#
# Defaults: --warn 10 GB free, --exit 5 GB free, --path .
#
# Exit codes: 0 = OK, 1 = warning, 2 = insufficient space (exits)

set -euo pipefail

WARN_GB=10
EXIT_GB=5
CHECK_PATH="."

while [[ $# -gt 0 ]]; do
  case "$1" in
    --warn) WARN_GB="$2"; shift 2 ;;
    --exit) EXIT_GB="$2"; shift 2 ;;
    --path) CHECK_PATH="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

AVAILABLE_KB=$(df -k "$CHECK_PATH" | awk 'NR==2 {print $4}')
AVAILABLE_GB=$(echo "scale=2; $AVAILABLE_KB / 1048576" | bc)

echo "Available disk space: ${AVAILABLE_GB} GB on $CHECK_PATH"

if (( $(echo "$AVAILABLE_GB < $EXIT_GB" | bc -l) )); then
  echo "ERROR: Insufficient disk space (${AVAILABLE_GB} GB < ${EXIT_GB} GB minimum)" >&2
  echo "Please free up disk space before building. Suggestions:" >&2
  echo "  - Run: ./scripts/cleanup-build.sh" >&2
  echo "  - Or:  CARGO_INCREMENTAL=0 cargo build" >&2
  exit 2
fi

if (( $(echo "$AVAILABLE_GB < $WARN_GB" | bc -l) )); then
  echo "WARNING: Low disk space (${AVAILABLE_GB} GB < ${WARN_GB} GB recommended)" >&2
  echo "Consider running: ./scripts/cleanup-build.sh" >&2
  exit 1
fi

echo "Disk space check passed (${AVAILABLE_GB} GB available)"
exit 0
