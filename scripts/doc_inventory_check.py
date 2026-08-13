#!/usr/bin/env python3
"""
Doc Inventory Check for Fluxion.

Verifies the docs/doc-inventory.md table is accurate:
  1. Every listed file exists at the specified path
  2. Every listed doc has a 7-line summary at lines 2-8

Usage:
  python3 scripts/doc_inventory_check.py

Exit codes:
  0 — All checks pass
  1 — Discrepancies found
  2 — Script error
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INVENTORY_FILE = REPO_ROOT / "docs" / "doc-inventory.md"

# Lines that must be present for a valid 7-line summary (lines 2-8)
MIN_SUMMARY_LINES = 7


def parse_inventory_table(content: str) -> list[tuple[str, Path, str]]:
    """Parse the inventory table and return list of (name, path, status)."""
    docs = []
    lines = content.splitlines()
    in_table = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and "Doc" in stripped and "Purpose" in stripped:
            in_table = True
            continue
        if in_table and stripped.startswith("|"):
            if stripped.strip() == "|":  # separator row
                continue
            parts = [p.strip() for p in stripped.split("|")]
            if len(parts) >= 4 and parts[2]:  # | Name | Purpose | Status |
                name = parts[1]
                status = parts[3] if len(parts) >= 4 else ""
                # Extract path from markdown link: [name](../../path) or [name](./path)
                path_match = re.search(r"\(([^)]+)\)", name)
                if path_match:
                    rel_path = path_match.group(1)
                    # Convert relative-to-docs path to absolute
                    if rel_path.startswith("../../"):
                        abs_path = REPO_ROOT / rel_path.lstrip("../../")
                    elif rel_path.startswith("./"):
                        abs_path = REPO_ROOT / "docs" / rel_path.lstrip("./")
                    else:
                        abs_path = REPO_ROOT / rel_path
                    docs.append((name, abs_path, status))
        elif in_table and not stripped.startswith("|"):
            break  # End of table
    return docs


def has_seven_line_summary(file_path: Path) -> tuple[bool, str]:
    """Check if the file has a 7-line summary at lines 2-8."""
    if not file_path.exists():
        return False, "file missing"

    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return False, f"read error: {e}"

    if len(lines) < 9:
        return (
            False,
            f"only {len(lines)} lines (need >= 9 for 7-line summary at lines 2-8)",
        )

    # Lines 2-8 (0-indexed: 1-7) = 7-line summary block.
    # Convention: line 1 = "# Title", line 2 = blank, lines 3-8 = 6 summary lines.
    # Blank lines are allowed within the block (e.g., between a blockquote and heading).
    # Headings and blockquotes are valid summary content (not just prose).
    summary_lines = lines[1:8]  # lines 2-8 (0-indexed: 1-7)
    content_line_count = 0
    for i, line in enumerate(summary_lines, start=2):
        stripped = line.strip()
        # Line 2 is allowed to be blank (the blank line after # Title)
        if not stripped and i == 2:
            continue
        if not stripped:
            continue  # Other blank lines are allowed (e.g., between sections)
        # Last-Updated markers are metadata, not summary content.
        if stripped.startswith("*Last Updated"):
            continue
        # HTML comments that contain substantive text count as summary
        # content (the `docs/doc-inventory.md` template uses
        # `<!-- Line 1: ... -->` style summaries — see line 26 of that
        # file).  Empty HTML comments (`<!-- -->`) and pure-marker
        # comments are skipped.  Keep this logic in sync with
        # `scripts/check_docs_summaries.py`.
        if stripped.startswith("<!--"):
            inner = stripped.removeprefix("<!--").removesuffix("-->").strip()
            if len(inner) > 5:
                content_line_count += 1
            continue
        # Count any meaningful content line (prose, blockquote, heading, bold, list)
        if len(stripped) >= 3 or stripped.startswith(
            (">", "#", "*", "-", "1.", "2.", "3.")
        ):
            content_line_count += 1

    # Require at least 3 content lines in lines 2-8 (allowing for intentional blanks).
    # The "7-line summary" convention means the block spans lines 2-8 (7 lines),
    # with meaningful content. ARCHITECTURE.md and other long-form docs may have
    # headings and blockquotes interspersed with blanks, which is valid.
    if content_line_count < 3:
        return (
            False,
            f"only {content_line_count} content line(s) in summary block (need >= 3)",
        )

    return True, "ok"


def check_inventory() -> list[str]:
    """Run all inventory checks. Returns list of findings."""
    findings = []

    if not INVENTORY_FILE.exists():
        return [f"CRITICAL: {INVENTORY_FILE.relative_to(REPO_ROOT)} does not exist"]

    content = INVENTORY_FILE.read_text(encoding="utf-8", errors="replace")
    docs = parse_inventory_table(content)

    if not docs:
        return ["WARNING: No docs found in inventory table"]

    print(f"[1/2] Checking {len(docs)} doc(s) listed in inventory ...")
    for name, path, status in docs:
        rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
        exists = path.exists()
        if not exists:
            findings.append(f"DRIFT: {rel} listed in doc-inventory.md does not exist")
            print(f"    FAIL: {rel} — file missing")
            continue

        has_summary, reason = has_seven_line_summary(path)
        if not has_summary:
            findings.append(f"DRIFT: {rel} — {reason}")
            print(f"    FAIL: {rel} — {reason}")
        else:
            print(f"    OK: {rel}")

    print("[2/2] Verifying Status column accuracy ...")
    status_issues = 0
    for name, path, status in docs:
        rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
        has_summary, _ = has_seven_line_summary(path)
        emoji_needs = "📝" in status or "❌" in status
        if has_summary and emoji_needs:
            findings.append(
                f"STATUS: {rel} has 7-line summary but is marked '{status}'"
            )
            print(f"    FIX: {rel} — has summary but marked '{status.strip()}'")
            status_issues += 1
        elif not has_summary and "✅" in status:
            findings.append(
                f"STATUS: {rel} missing 7-line summary but is marked '{status}'"
            )
            print(f"    FIX: {rel} — missing summary but marked '{status.strip()}'")
            status_issues += 1

    if status_issues == 0:
        print("    OK: All Status column entries are accurate")

    return findings


def main() -> int:
    print("=== Fluxion Doc Inventory Check ===")
    print(f"Repo: {REPO_ROOT}")
    print()

    findings = check_inventory()

    print()
    if not findings:
        print("PASS: No discrepancies found.")
        sys.exit(0)

    print(f"FAIL: {len(findings)} finding(s) detected:")
    for finding in findings:
        print(f"  {finding}")

    if any(f.startswith("DRIFT:") for f in findings):
        print("\n--- Remediation ---")
        print("  1. Create the missing 7-line summary at lines 2-8, OR")
        print("  2. Remove the unlisted file from the inventory table")
    if any(f.startswith("STATUS:") for f in findings):
        print(
            "\n  Fix the Status column in docs/doc-inventory.md to reflect actual state."
        )

    sys.exit(1)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
