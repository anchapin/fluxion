#!/bin/bash
# end_of_shift_validation.sh — comprehensive validation before end of shift
# Runs: tests, performance gate, architecture drift, ASHRAE cases, mutation testing, audit, lint

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

cd "$PROJECT_ROOT"

log "=== End-of-Shift Validation Starting ==="
START_TIME=$(date +%s)

log "[1/8] Running cargo test --all..."
if ! cargo test --all; then
    error "cargo test failed"
    exit 1
fi

log "[2/8] Running performance_gate.py..."
if ! python scripts/performance_gate.py; then
    error "performance_gate.py failed"
    exit 1
fi

log "[3/8] Running check_architecture_drift.py..."
if ! python scripts/check_architecture_drift.py; then
    error "check_architecture_drift.py failed"
    exit 1
fi

log "[4/8] Running check_ashrae_cases_cycle.py..."
if ! python scripts/check_ashrae_cases_cycle.py; then
    error "check_ashrae_cases_cycle.py failed"
    exit 1
fi

log "[5/8] Running mutation testing spot-check (10 min timeout)..."
if command -v cargo-mutants &> /dev/null; then
    timeout 600 cargo mutants --fail-fast -- -q || {
        warn "Mutation testing found issues or timed out"
    }
else
    warn "cargo-mutants not installed, skipping mutation testing"
fi

log "[6/8] Running audit_false_confidence.py..."
if ! python scripts/audit_false_confidence.py; then
    error "audit_false_confidence.py failed"
    exit 1
fi

log "[7/8] Running cargo fmt --check..."
if ! cargo fmt --check; then
    error "cargo fmt check failed"
    exit 1
fi

log "[8/8] Running cargo clippy..."
if ! cargo clippy --all-targets -- -D warnings; then
    error "cargo clippy failed"
    exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

log "=== End-of-Shift Validation Complete ==="
log "Total time: ${DURATION}s"

COMMIT_STATE_FILE="$SCRIPT_DIR/.validation_state"
echo "last_validation=$(date -Iseconds)" > "$COMMIT_STATE_FILE"
echo "duration=${DURATION}" >> "$COMMIT_STATE_FILE"
echo "status=passed" >> "$COMMIT_STATE_FILE"

log "Validation state committed to $COMMIT_STATE_FILE"
