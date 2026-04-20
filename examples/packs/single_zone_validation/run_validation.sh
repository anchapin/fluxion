#!/bin/bash
# Single Zone Validation Pack - Quick validation script

set -e

echo "=== Single Zone Validation Pack ==="
echo "Testing basic Fluxion functionality with single-zone model"
echo ""

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "Error: cargo not found. Is Rust installed?"
    exit 1
fi

# Build the example
echo "Building validate_6r2c example..."
cargo build --example validate_6r2c --release 2>/dev/null || cargo build --example validate_6r2c

echo ""
echo "Running single-zone validation (Case 600)..."
echo ""

# Run the validation example
cargo run --example validate_6r2c 2>/dev/null || cargo run --example validate_6r2c

echo ""
echo "=== Validation Complete ==="
echo "Expected: Simulation completes without errors"
echo "Check output above for any FAILED indicators"
