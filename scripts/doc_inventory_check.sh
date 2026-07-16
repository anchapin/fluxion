#!/usr/bin/env bash
# Doc Inventory Check for Fluxion.
#
# Verifies docs/doc-inventory.md is in sync:
#   (a) every listed file has a 7-line summary at lines 2-8
#   (b) table rows match actual file paths
#
# Usage:
#   python3 scripts/doc_inventory_check.py
#   bash scripts/doc_inventory_check.sh
#
# Exit codes:
#   0 — All checks pass
#   1 — Discrepancies found
#   2 — Script error

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

exec python3 "$REPO_ROOT/scripts/doc_inventory_check.py" "$@"
