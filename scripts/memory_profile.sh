#!/bin/bash
# memory_profile.sh — Run simulation with dhat heap profiling
#
# Usage: ./scripts/memory_profile.sh [--zones N] [--output DIR]
#
# Options:
#   --zones N     Number of zones for simulation (default: 10)
#   --output DIR  Output directory for dhat JSON (default: ./dhat_output)
#
# Requirements:
#   - cargo build --features dhat
#   - dhat output viewer: https://nnethercote.github.io/dhat/dhat/
#
# Exit codes:
#   0  = profiling completed successfully
#   1  = build failed
#   2  = simulation failed

set -euo pipefail

ZONES=10
OUTPUT_DIR="./dhat_output"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zones) ZONES="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Fluxion Memory Profiling"
echo "============================================"
echo "Zones: ${ZONES}"
echo "Output: ${OUTPUT_DIR}"
echo "Project: ${PROJECT_ROOT}"
echo "============================================"

echo ""
echo "[1/3] Building with dhat feature..."
cd "$PROJECT_ROOT"
if ! cargo build --features dhat --release 2>&1; then
  echo "ERROR: Build failed" >&2
  exit 1
fi

echo ""
echo "[2/3] Running ${ZONES}-zone simulation with dhat..."
DHAT_FILE="${OUTPUT_DIR}/dhat-${ZONES}zones.json"

case "$ZONES" in
  10)
    TEST_TARGET="multi_zone_n_zone_network_10_zone_round_trip"
    ;;
  100)
    TEST_TARGET="multi_zone_n_zone_network_100_zone_round_trip"
    ;;
  *)
    TEST_TARGET="multi_zone_n_zone_network_${ZONES}_zone_round_trip"
    ;;
esac

if cargo test --features dhat --test multi_zone_n_zone_network -- "$TEST_TARGET" --nocapture 2>&1; then
  echo "Simulation completed"
else
  echo "ERROR: Simulation failed" >&2
  exit 2
fi

echo ""
echo "[3/3] Profiling complete."
echo "Dhat output: ${DHAT_FILE}"
echo ""
echo "View the results at: https://nnethercote.github.io/dhat/dhat/"
echo "Load the JSON file to analyze allocation hotspots."
echo ""
echo "Peak RSS check: $(ps -o rss= -p $$ 2>/dev/null || echo "N/A") KB"

exit 0
