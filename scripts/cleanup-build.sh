#!/usr/bin/env bash
# cleanup-build.sh — Free disk space by removing build artifacts
#
# Usage: ./scripts/cleanup-build.sh [--deep]
#
# Options:
#   --deep    Remove target/ directory entirely (full rebuild)
#   (default) Remove only incremental build cache, keep target/

set -euo pipefail

DEEP=false
if [[ "${1:-}" == "--deep" ]]; then
  DEEP=true
fi

echo "Cleaning build artifacts..."

if [[ -d "target" ]]; then
  if $DEEP; then
    echo "  Removing target/ directory (deep clean)..."
    rm -rf target/
  else
    echo "  Removing incremental build cache..."
    rm -rf target/.cache 2>/dev/null || true
    rm -rf target/.rustc_info.json 2>/dev/null || true
    # Keep the target directory itself to avoid re-creating it
  fi
fi

# Clean cargo's incremental build cache in the global cache
if [[ -d ~/.cargo/.cache ]]; then
  echo "  Cleaning cargo incremental cache..."
  rm -rf ~/.cargo/.cache 2>/dev/null || true
fi

echo "Build cleanup complete."

if $DEEP; then
  echo ""
  echo "NOTE: A full rebuild will be required on next build."
  echo "      To avoid this, run without --deep next time."
fi
