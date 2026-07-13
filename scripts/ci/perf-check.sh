#!/bin/bash
# ci/perf-check.sh — CI integration for performance regression gate
# Usage: Run in CI after cargo test passes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PERF_GATE="$PROJECT_ROOT/scripts/performance_gate.py"
FAILED=0

echo "=== Performance Gate CI Check ==="

if [[ ! -f "$PERF_GATE" ]]; then
    echo "ERROR: performance_gate.py not found at $PERF_GATE"
    exit 1
fi

if ! python3 "$PERF_GATE"; then
    echo ""
    echo "=== PERFORMANCE GATE FAILED ==="
    echo "One or more benchmarks regressed by more than 10%."
    echo "Please investigate the performance regression or update the baseline"
    echo "if the regression is expected (e.g., new feature with acceptable overhead)."
    FAILED=1
else
    echo "Performance gate passed"
fi

exit $FAILED
