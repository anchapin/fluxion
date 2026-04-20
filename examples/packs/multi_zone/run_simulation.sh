#!/bin/bash
# Multi Zone Pack - Simulation script

set -e

echo "=== Multi Zone Pack ==="
echo "Testing multi-zone thermal model with inter-zone coupling"
echo ""

# Build the multi_zone_demo example
echo "Building multi_zone_demo example..."
cargo build --example multi_zone_demo --release 2>/dev/null || cargo build --example multi_zone_demo

echo ""
echo "Running multi-zone simulation..."
echo ""

cargo run --example multi_zone_demo 2>/dev/null || cargo run --example multi_zone_demo

echo ""
echo "=== Multi Zone Simulation Complete ==="
echo "Check for energy conservation validation results"
