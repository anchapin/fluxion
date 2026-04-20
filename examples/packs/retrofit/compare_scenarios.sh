#!/bin/bash
# Retrofit Pack - Compare baseline vs retrofit scenarios

set -e

echo "=== Retrofit Pack ==="
echo "Comparing energy consumption: Baseline vs Retrofit"
echo ""

# Build the construction example
echo "Building construction_example..."
cargo build --example construction_example --release 2>/dev/null || cargo build --example construction_example

echo ""
echo "Running construction analysis..."
cargo run --example construction_example 2>/dev/null || cargo run --example construction_example

echo ""
echo "=== Retrofit Analysis ==="
echo ""
echo "Summary of expected savings:"
echo "  - Windows only: 5-15% energy reduction"
echo "  - HVAC only: 10-20% energy reduction"
echo "  - Full retrofit: 25-45% energy reduction"
echo ""
echo "For full annual simulation, use validate_6r2c.rs with modified config"
