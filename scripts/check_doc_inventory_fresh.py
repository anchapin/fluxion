#!/usr/bin/env python3
"""
Doc Inventory Freshness Check for Fluxion.

Verifies that `docs/doc-inventory.md` is in sync with what
`scripts/generate_doc_inventory.py` would produce.  Closes Issue #2765
acceptance criterion #2: a CI gate that fails when the inventory drifts.

The check is content-based (not mtime-based): we run the generator and
compare the resulting file byte-for-byte against the committed copy.
This catches drift in either direction — added/removed docs, changed
summary status (✅/❌), or hand-edits inside the auto-generated block.
Content comparison is the only reliable signal because git does not
preserve mtimes across checkouts (every file has the checkout timestamp
in a fresh clone, so an mtime-based gate would be a no-op in CI).

Fail-loud contract:
    If `generate_doc_inventory.py` itself exits non-zero (crashes,
    missing markers, IO error, ...), this gate propagates exit code 2
    and restores the pre-check snapshot.  A broken generator must
    never silently green this gate.

Usage:
    python3 scripts/check_doc_inventory_fresh.py

Exit codes:
    0 — Inventory is fresh (committed copy matches generator output).
    1 — Inventory is stale (committed copy differs from generator output).
    2 — Script error (e.g. generator crashed, markers missing).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INVENTORY_FILE = REPO_ROOT / "docs" / "doc-inventory.md"
GENERATOR = REPO_ROOT / "scripts" / "generate_doc_inventory.py"


def main() -> int:
    print("=== Fluxion Doc Inventory Freshness Check ===")
    print(f"Repo: {REPO_ROOT}")
    print(f"Inventory: {INVENTORY_FILE.relative_to(REPO_ROOT)}")
    print()

    if not INVENTORY_FILE.exists():
        print(
            f"FAIL: {INVENTORY_FILE.relative_to(REPO_ROOT)} does not exist. "
            f"Run `python3 scripts/generate_doc_inventory.py` to create it."
        )
        return 1
    if not GENERATOR.exists():
        print(f"FAIL: {GENERATOR.relative_to(REPO_ROOT)} does not exist.")
        return 2

    before = INVENTORY_FILE.read_bytes()

    # Run the generator.  It modifies INVENTORY_FILE in place when the
    # table drifts.  We capture stdout/stderr and propagate any non-zero
    # exit code as a hard failure (fail-loud: a crashed generator must
    # never silently green this gate).
    print(f"Running {GENERATOR.relative_to(REPO_ROOT)} ...")
    result = subprocess.run(
        ["python3", str(GENERATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        print()
        print(
            f"FAIL: generator exited {result.returncode} — the freshness "
            f"gate cannot verify the inventory while the generator is "
            f"broken.  Fix {GENERATOR.relative_to(REPO_ROOT)} first."
        )
        # Restore the pre-check snapshot so a crashed generator does
        # not leave the inventory in a half-written state.
        INVENTORY_FILE.write_bytes(before)
        return 2

    after = INVENTORY_FILE.read_bytes()

    if before == after:
        print()
        print("PASS: docs/doc-inventory.md is fresh.")
        return 0

    # Drift detected.  Restore the committed content so the gate does
    # not silently rewrite the working tree (the contributor must run
    # the generator explicitly to fix the failure).
    INVENTORY_FILE.write_bytes(before)
    print()
    print("FAIL: docs/doc-inventory.md is stale.")
    print()
    print(
        "The committed inventory differs from what "
        "`scripts/generate_doc_inventory.py` would produce.  This "
        "usually means a `docs/**/*.md` file was added, removed, or "
        "its 7-line summary status changed without regenerating the "
        "inventory table."
    )
    print()
    print("To fix:")
    print("  python3 scripts/generate_doc_inventory.py")
    print("  git add docs/doc-inventory.md")
    print("  git commit  # or amend the docs-touching commit")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
