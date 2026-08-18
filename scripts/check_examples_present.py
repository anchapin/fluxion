#!/usr/bin/env python3
"""
Example-presence detector for Fluxion (Issue #3125).

`examples/*.rs` are user-facing documentation. They are the canonical
"does the public API still compile for users?" surface and the entry
point most new users copy from. Issue #3125 closed the *compile* gap
(`cargo check --workspace --examples --all-targets` now runs in the
Workspace Check CI job); this gate closes the *presence* gap: a future
"let's delete examples/*.rs" cleanup must not silently shrink the
public surface.

Scope:
  Walks the repo-root ``examples/`` directory and counts every ``.rs``
  file. The count is compared against ``MIN_EXAMPLE_COUNT`` (5). The
  threshold is intentionally generous — the current tree carries 9
  examples — so a single deletion still passes, but a wholesale
  removal (``examples/`` emptied or removed) fails loud.

Usage:
  python3 scripts/check_examples_present.py

Exit codes:
  0 — example count meets the threshold (PASS).
  1 — example count below the threshold (FAIL); see stdout.
  2 — script error (e.g. ``examples/`` directory missing, IO failure).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Tunables. Keep MIN_EXAMPLE_COUNT aligned with the issue #3125 acceptance
# criterion ("ls examples/*.rs | wc -l >= 5"). The threshold is the floor
# below which the public surface has clearly shrunk enough to demand
# attention; raising it requires a deliberate decision in the issue
# thread.
# ---------------------------------------------------------------------------
MIN_EXAMPLE_COUNT = 5
EXAMPLES_DIR: str = "examples"


def list_example_files() -> list[Path]:
    """Return every ``examples/*.rs`` file under the repo root, sorted.

    The sorted order makes the diagnostic output deterministic. Files
    outside the repo-root ``examples/`` directory (e.g. example modules
    nested under a workspace sibling) are intentionally ignored — the
    issue scope is the top-level user-facing example set.
    """
    base = REPO_ROOT / EXAMPLES_DIR
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.suffix == ".rs")


def main() -> int:
    print("=== Fluxion example-presence detector (Issue #3125) ===")
    print(f"Repo:              {REPO_ROOT}")
    print(f"Examples dir:      {EXAMPLES_DIR}/")
    print(f"Minimum examples:  {MIN_EXAMPLE_COUNT}")
    print()

    examples_dir = REPO_ROOT / EXAMPLES_DIR
    if not examples_dir.is_dir():
        print(f"FAIL: examples directory missing at {examples_dir}")
        print(
            "The repo-root examples/ tree is the canonical user-facing "
            "entry point. Removing it (or moving it elsewhere) requires a "
            "deliberate decision in an issue; the gate fails loud so "
            "such a cleanup cannot ship silently."
        )
        return 2

    files = list_example_files()
    count = len(files)
    print(f"Found {count} example file(s) in {EXAMPLES_DIR}/:")
    for p in files:
        print(f"  - {p.relative_to(REPO_ROOT)}")
    print()

    if count >= MIN_EXAMPLE_COUNT:
        print(f"PASS: {count} example file(s) >= minimum {MIN_EXAMPLE_COUNT}.")
        return 0

    print(f"FAIL: {count} example file(s) below minimum {MIN_EXAMPLE_COUNT}.")
    print(
        "The repo-root examples/ tree is the canonical user-facing entry "
        "point. Removing or shrinking it below the floor requires a "
        "deliberate decision in an issue (issue #3125); the gate fails "
        "loud so such a cleanup cannot ship silently."
    )
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — top-level barrier per repo style
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
