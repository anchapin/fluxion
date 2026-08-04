#!/bin/bash
# Check available disk space before operations
# Usage: ./scripts/disk-space-check.sh [--cleanup]
#
# Options:
#   --cleanup    Suggest cleanup actions if space is low
#
# Thresholds:
#   Minimum:     10 GB free
#   Recommended: 50 GB free
#
# Exits with:
#   0  = sufficient space
#   1  = below minimum (10 GB)
#   2  = below recommended (50 GB)

set -euo pipefail

CLEANUP=false
if [[ "${1:-}" == "--cleanup" ]]; then
    CLEANUP=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Get available space in bytes for the project root filesystem
AVAILABLE_BYTES=$(df -B1 "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
AVAILABLE_GB=$((AVAILABLE_BYTES / 1024 / 1024 / 1024))

MINIMUM_GB=10
RECOMMENDED_GB=50

echo "============================================"
echo "  Fluxion Disk Space Check"
echo "============================================"
echo "Project root: $PROJECT_ROOT"
echo "Available space: ${AVAILABLE_GB} GB"
echo "Minimum required: ${MINIMUM_GB} GB"
echo "Recommended: ${RECOMMENDED_GB} GB"
echo "============================================"

if [[ "$AVAILABLE_GB" -lt "$MINIMUM_GB" ]]; then
    echo ""
    echo "ERROR: Insufficient disk space!"
    echo "Available: ${AVAILABLE_GB} GB"
    echo "Minimum required: ${MINIMUM_GB} GB"
    echo ""
    echo "Disk space exhaustion can cause:"
    echo "  - Credential lock failures"
    echo "  - PR creation failures"
    echo "  - Git ref lock failures"
    echo "  - Build failures"
    echo ""

    if $CLEANUP; then
        echo "Cleanup suggestions:"
        echo "  - Run: cargo clean"
        echo "  - Remove target/ directories in worktrees"
        echo "  - Remove mutation_testing_results/"
        echo "  - Remove validation_artifacts.zip, crossval_logs.zip"
        echo "  - Remove test_results/ directory"
    fi

    exit 1
fi

if [[ "$AVAILABLE_GB" -lt "$RECOMMENDED_GB" ]]; then
    echo ""
    echo "WARNING: Disk space below recommended level."
    echo "Available: ${AVAILABLE_GB} GB"
    echo "Recommended: ${RECOMMENDED_GB} GB"
    echo ""
    echo "Operations may fail if disk fills up during:"
    echo "  - cargo build --release"
    echo "  - mutation testing"
    echo "  - large validation runs"
    echo ""

    if $CLEANUP; then
        echo "Cleanup suggestions:"
        echo "  - Run: cargo clean"
        echo "  - Remove old target/ directories"
        echo "  - Remove mutation_testing_results/"
    fi

    exit 2
fi

echo ""
echo "Disk space check passed."
echo "Available: ${AVAILABLE_GB} GB (${RECOMMENDED_GB} GB recommended)"
echo ""

exit 0
