#!/bin/bash
# Run mutation testing on specified modules
# Usage: ./scripts/run_mutation_testing.sh [modules] [toolchain]
#
# Examples:
#   ./scripts/run_mutation_testing.sh                    # Test physics and solar modules
#   ./scripts/run_mutation_testing.sh src/physics/       # Test only physics module
#   ./scripts/run_mutation_testing.sh all stable         # Test all modules with stable Rust

set -euo pipefail

MODULES="${1:-src/physics/,src/solar/}"
TOOLCHAIN="${2:-stable}"
TIMEOUT="${TIMEOUT:-300}"
JOBS="${JOBS:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/mutation_testing_results"

echo "============================================"
echo "  Fluxion Mutation Testing"
echo "============================================"
echo "Modules: $MODULES"
echo "Toolchain: $TOOLCHAIN"
echo "Results directory: $RESULTS_DIR"
echo "============================================"

cd "$PROJECT_ROOT"

echo ""
echo "[1/4] Installing cargo-mutants..."
cargo +"$TOOLCHAIN" install cargo-mutants --locked --quiet

echo ""
echo "[2/4] Running mutation tests on: $MODULES"

IFS=',' read -ra MODULE_ARRAY <<< "$MODULES"
for MODULE in "${MODULE_ARRAY[@]}"; do
    MODULE_TRIMMED=$(echo "$MODULE" | tr -d ' ')
    echo ""
    echo "--------------------------------------------"
    echo "Testing module: $MODULE_TRIMMED"
    echo "--------------------------------------------"

    MODULE_RESULTS="$RESULTS_DIR/$(basename "$MODULE_TRIMMED")"
    mkdir -p "$MODULE_RESULTS"

    cargo mutants \
        --directory . \
        --output-dir "$MODULE_RESULTS" \
        --timeout "$TIMEOUT" \
        --jobs "$JOBS" \
        -- \
        test --lib --verbose --no-default-features -- --test-threads="$JOBS"
done

echo ""
echo "[3/4] Generating summary..."

for MODULE_DIR in "$RESULTS_DIR"/*/; do
    if [ -f "${MODULE_DIR}outcomes.json" ]; then
        MODULE_NAME=$(basename "$MODULE_DIR")
        echo ""
        echo "=== $MODULE_NAME ==="

        cargo mutants --directory . show --format console 2>/dev/null | head -50 || true

        KILLED=$(grep -o '"killed":[0-9]*' "${MODULE_DIR}outcomes.json" | grep -o '[0-9]*$' | head -1 || echo "0")
        SURVIVED=$(grep -o '"survived":[0-9]*' "${MODULE_DIR}outcomes.json" | grep -o '[0-9]*$' | head -1 || echo "0")
        TIMEOUT=$(grep -o '"timeout":[0-9]*' "${MODULE_DIR}outcomes.json" | grep -o '[0-9]*$' | head -1 || echo "0")
        ERROR=$(grep -o '"error":[0-9]*' "${MODULE_DIR}outcomes.json" | grep -o '[0-9]*$' | head -1 || echo "0")
        TOTAL=$((KILLED + SURVIVED))

        if [ "$TOTAL" -gt 0 ]; then
            KILL_RATE=$(awk "BEGIN {printf \"%.1f\", ($KILLED/$TOTAL)*100}")
        else
            KILL_RATE="N/A"
        fi

        echo ""
        echo "Summary:"
        echo "  Total mutants: $TOTAL"
        echo "  Killed:        $KILLED"
        echo "  Survived:      $SURVIVED"
        echo "  Timeout:       $TIMEOUT"
        echo "  Error:         $ERROR"
        echo "  Kill rate:     ${KILL_RATE}%"

        if [ "$SURVIVED" -gt 0 ]; then
            echo ""
            echo "⚠️  WARNING: $SURVIVED surviving mutants detected!"
            echo "    Review the report and consider tightening test assertions."
        fi
    fi
done

echo ""
echo "[4/4] Results saved to: $RESULTS_DIR"
echo ""
echo "============================================"
echo "  Mutation Testing Complete"
echo "============================================"
