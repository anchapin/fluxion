#!/usr/bin/env bash
# Quick helper to build Python bindings (locally) and run an example
set -euo pipefail

echo "1) Activate Python venv if you have one (optional)"
echo "   python3 -m venv .venv && source .venv/bin/activate"

echo "2) Install build tool 'maturin' if not present"
python -m pip install --upgrade pip
python -m pip install 'maturin>=1.0,<2.0'

echo "3) Build and install local Python bindings"
maturin develop --release

echo "4) Run the small oracle example"
python examples/run_oracle.py
