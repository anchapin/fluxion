#!/bin/bash
# Quick test of Case 900 cooling fix

set -e

echo "=== Building fluxion in release mode ==="
cargo build --lib --release 2>&1 | tail -5

echo ""
echo "=== Running Case 900 cooling test ==="
cargo test test_case_900_annual_cooling_within_reference_range --lib --release -- --nocapture 2>&1 | grep -A 20 "Case 900 Annual Cooling"

echo ""
echo "=== Running Case 900 heating test ==="
cargo test test_case_900_annual_heating_within_reference_range --lib --release -- --nocapture 2>&1 | grep -A 10 "Case 900 Annual Heating"

echo ""
echo "=== Running full 900-series regression test ==="
cargo test test_900_series_regression --lib --release -- --nocapture 2>&1 | tail -20
