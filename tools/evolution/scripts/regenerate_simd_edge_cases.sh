#!/usr/bin/env bash
# Regenerate the per-edge reference output values for the issue #3338
# SIMD/cache-blocked solar evolution harness.
#
# Run from repo root:
#
#   $ tools/evolution/scripts/regenerate_simd_edge_cases.sh
#
# The script is idempotent: identical inputs produce identical output
# (modulo the determinism-digest hash baked into the fixture).
#
# Issues:
#   * #3338 — solar / radiation SIMD/cache-blocked evolution
#
# See `tools/evolution/edge_cases/solar_simd.json` for the schema.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
echo "[regenerate_simd_edge_cases] cargo run --release --example regenerate_simd_edge_cases"
cargo run --release --example regenerate_simd_edge_cases
echo "[regenerate_simd_edge_cases] done. Diff:"
git diff --stat tools/evolution/edge_cases/solar_simd.json || true
