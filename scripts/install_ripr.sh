#!/bin/bash
# Install ripr — static mutation-exposure analyzer (Issue #1254).
#
# ripr reads a PR diff and reports which changed behavior the current tests
# reach but do not actually check, WITHOUT compiling or running any mutants.
# It is the cheap, per-PR advisory companion to cargo-mutants (which stays on
# the 32 GB runner for the confirmation run).
#
# Usage:
#   ./scripts/install_ripr.sh                # install latest stable ripr
#   ./scripts/install_ripr.sh 0.10.0         # pin a specific version
#   ./scripts/install_ripr.sh --verify       # install + run a smoke pilot
#
# Requires: Rust >= 1.95 (fluxion's `stable` toolchain satisfies this).
# Docs:     https://github.com/EffortlessMetrics/ripr
# See:      docs/ripr_investigation_1254.md

set -euo pipefail

VERSION="${1:-}"
VERIFY=0
if [[ "${VERSION}" == "--verify" ]]; then
  VERIFY=1
  VERSION=""
fi

echo "============================================"
echo "  Installing ripr (static mutation-exposure)"
echo "============================================"

TOOLCHAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TOOLCHAIN_DIR"

# ripr needs Rust >= 1.95; fluxion pins `stable` in rust-toolchain.toml.
INSTALL_ARGS=(--locked)
if [[ -n "${VERSION}" ]]; then
  INSTALL_ARGS+=(--version "${VERSION}")
  echo "Pinning ripr to ${VERSION}"
fi

cargo install ripr "${INSTALL_ARGS[@]}"

echo ""
echo "Installed:"
ripr --version

if [[ "$VERIFY" -eq 1 ]]; then
  echo ""
  echo "============================================"
  echo "  Smoke test: ripr pilot (advisory, static)"
  echo "============================================"
  echo "Analyzing diff vs origin/main (if present)..."
  # first-pr names the single top repairable gap; non-fatal if no diff/base.
  ripr first-pr --root . --base origin/main --head HEAD || \
    echo "(no origin/main base available — run inside a PR branch for real output)"
fi

echo ""
echo "============================================"
echo "  ripr ready."
echo "  Per-PR advisory:  ripr pilot --root ."
echo "  Confirmation run: cargo mutants  (32 GB runner, nightly)"
echo "============================================"
