#!/usr/bin/env python3
"""
Gate-inventory sync detector — Issue #4181.

Asserts that every `scripts/check_*.py` basename appears in:
  1. AGENTS.md §"CI Gates You Can Run Locally" table
  2. docs/scripts/README.md §"CI gate checks" table

The "one-stop lookup" promise in AGENTS.md is enforced rather than asserted.
Drift is caught in CI before it reaches the user.

Exit codes:
  0 — all check_*.py files are named in both tables
  1 — missing from one or both tables; missing basenames are listed

Usage:
  python3 scripts/check_gate_inventory_sync.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_MD = REPO_ROOT / "AGENTS.md"
DOCS_SCRIPTS_README = REPO_ROOT / "docs" / "scripts" / "README.md"
SCRIPTS_DIR = REPO_ROOT / "scripts"


def load_check_scripts() -> set[str]:
    """Return basenames of all check_*.py files in scripts/."""
    return {
        p.name
        for p in SCRIPTS_DIR.glob("check_*.py")
        if p.is_file() and not p.name.startswith("_")
    }


def extract_table_gates(markdown_path: Path, section_start: str) -> set[str]:
    """Extract check_*.py basenames from a markdown table section.

    Scans the file for lines matching the section header, then collects
    all check_*.py basenames from table rows until a separator or
    section boundary is reached.
    """
    if not markdown_path.exists():
        return set()

    content = markdown_path.read_text()
    lines = content.splitlines()

    # Find the section header
    section_line = -1
    for i, line in enumerate(lines):
        if section_start in line:
            section_line = i
            break

    if section_line == -1:
        return set()

    # Collect gate names from table rows
    gates: set[str] = set()
    in_table = False
    header_seen = False

    for line in lines[section_line + 1 :]:
        stripped = line.strip()

        # Section separator (---) ends the table
        if stripped.startswith("---"):
            break

        # Table header row (| Gate | ...) — only the FIRST one is the header;
        # later rows whose purpose text contains "Gate"/"Script" are data rows.
        if (
            stripped.startswith("|")
            and not header_seen
            and ("Gate" in stripped or "Script" in stripped)
        ):
            in_table = True
            header_seen = True
            continue

        # Table rows containing check_*.py
        if in_table and stripped.startswith("|"):
            # Match patterns like:
            # - `scripts/check_foo.py`
            # - check_foo.py (bare, as in docs/scripts/README.md)
            # Simple basename pattern: handles digits (check_criterion_2_drift.py),
            # backticks, and path prefixes uniformly.
            for match in re.findall(r"check_[A-Za-z0-9_]+\.py", stripped):
                gates.add(match)

    return gates


def main() -> int:
    check_scripts = load_check_scripts()

    # Extract gates from both tables
    agents_gates = extract_table_gates(
        AGENTS_MD, "## CI Gates You Can Run Locally"
    )
    readme_gates = extract_table_gates(
        DOCS_SCRIPTS_README, "## CI gate checks"
    )

    # Find missing gates
    missing_from_agents = check_scripts - agents_gates
    missing_from_readme = check_scripts - readme_gates
    missing_from_either = missing_from_agents | missing_from_readme

    if not missing_from_either:
        print("check_gate_inventory_sync: PASS — all check_*.py files named in AGENTS.md and docs/scripts/README.md")
        return 0

    print("check_gate_inventory_sync: FAIL", file=sys.stderr)
    print(file=sys.stderr)

    if missing_from_agents:
        print(f"  Missing from AGENTS.md §'CI Gates You Can Run Locally' ({len(missing_from_agents)}):", file=sys.stderr)
        for gate in sorted(missing_from_agents):
            print(f"    - {gate}", file=sys.stderr)

    if missing_from_readme:
        print(f"  Missing from docs/scripts/README.md §'CI gate checks' ({len(missing_from_readme)}):", file=sys.stderr)
        for gate in sorted(missing_from_readme):
            print(f"    - {gate}", file=sys.stderr)

    print(file=sys.stderr)
    print(f"Total: {len(check_scripts)} check_*.py files, {len(agents_gates)} in AGENTS.md, {len(readme_gates)} in README", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
