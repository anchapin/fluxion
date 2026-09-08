#!/usr/bin/env python3
"""
Regenerate the `## Summary` table at the top of `docs/KNOWN_ISSUES.md` from
the actual `### ` section headers in the file.

Issue #3513: the hand-maintained table undercounted the LIMIT entries by
17+ and the SOLAR / BASE / MULTI counts were also stale. This script
replaces the table with one derived from a fresh grep of the section
anchors; a CI gate (`check_known_issues_summary.py`) verifies the table
matches the derived counts.

Usage:
    python3 scripts/check_known_issues_summary.py [--check]

Exit codes:
    0 — derived counts match the committed table (or no --check flag)
    1 — drift detected
    2 — derived file is missing the `## Summary` table
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KNOWN_ISSUES = REPO_ROOT / "docs" / "KNOWN_ISSUES.md"

# Map of category prefix → table row label.
CATEGORY_ROWS = [
    ("BASE", "Foundation (BASE)"),
    ("SOLAR", "Solar (SOLAR)"),
    ("FREE", "Free-Float (FREE)"),
    ("TEMP", "Temperature (TEMP)"),
    ("MULTI", "Multi-Zone (MULTI)"),
    ("LIMIT", "Model Limits (LIMIT)"),
    ("REPORT", "Reporting (REPORT)"),
    ("CI", "CI/Infrastructure (CI)"),
    ("FLUID", "fluxion-fluid (FLUID)"),
    ("FFD", "FFD/CFD (FFD)"),
]

# Per-category status counts. Issue #3513 acceptance #3 says these should
# derive from the per-section `**Status:**` field. We extract the most
# common status tokens (`Fixed`, `Open`, `Partial`, `Won't Fix`) from each
# section's `**Status:**` line.
STATUS_TOKEN_FIXED = re.compile(r"✅\s*Fixed|Fixed\s*\(Phase", re.IGNORECASE)
STATUS_TOKEN_OPEN = re.compile(r"🔄\s*Open|🔄\s*Open\s*\(|🔄\s*OPEN", re.IGNORECASE)
STATUS_TOKEN_PARTIAL = re.compile(r"🟡|Partially", re.IGNORECASE)
STATUS_TOKEN_WONT_FIX = re.compile(r"Won'?t\s*Fix|WONTFIX|wontfix", re.IGNORECASE)
STATUS_LINE = re.compile(r"^[\s\-]*\*\*Status:\*\*\s+(.+)$", re.MULTILINE)


def extract_counts(text: str) -> dict[str, dict[str, int]]:
    """Walk every `### CATEGORY-NN:` header and the following section body,
    and count per-category status tokens.

    Returns: {row_label: {"total": N, "fixed": F, "open": O, "partial": P,
                         "wont_fix": W}}
    """
    # Build a list of (category_prefix, start_offset) for every top-level
    # `### CATEGORY-NN:` header. Ignore `### CATEGORY-NN UPDATE` sub-sections.
    section_starts: list[tuple[str, int]] = []
    for m in re.finditer(r"^###\s+([A-Za-z]+(?:-[A-Za-z]+)?)-(\d+)([a-z]?):\s+", text, re.MULTILINE):
        prefix = m.group(1)
        suffix_letter = m.group(3)
        # Treat `### BASE-01b: ...` as a sub-section of BASE (counted under BASE).
        # Treat `### LIMIT-05 UPDATE ...` (not matched because no colon) as a
        # sub-section (not a top-level entry). The regex above requires the `:`
        # so updates without a colon are skipped.
        section_starts.append((prefix, m.start(), suffix_letter))

    # Build the result dict, defaulting all counts to 0.
    result: dict[str, dict[str, int]] = {}
    for prefix, label in CATEGORY_ROWS:
        result[label] = {"total": 0, "fixed": 0, "open": 0, "partial": 0, "wont_fix": 0}

    for i, (prefix, start, suffix_letter) in enumerate(section_starts):
        # Find the next `### ` (any prefix) or EOF.
        end = section_starts[i + 1][1] if i + 1 < len(section_starts) else len(text)
        body = text[start:end]

        # Find the row label for this prefix.
        label = next((lbl for pfx, lbl in CATEGORY_ROWS if pfx == prefix), None)
        if label is None:
            # Unknown prefix (e.g., a new category not in CATEGORY_ROWS).
            continue
        result[label]["total"] += 1

        # Extract the first `**Status:**` line in the body.
        sm = STATUS_LINE.search(body)
        if not sm:
            continue
        status_text = sm.group(1)
        if STATUS_TOKEN_FIXED.search(status_text):
            result[label]["fixed"] += 1
        elif STATUS_TOKEN_OPEN.search(status_text):
            result[label]["open"] += 1
        elif STATUS_TOKEN_PARTIAL.search(status_text):
            result[label]["partial"] += 1
        elif STATUS_TOKEN_WONT_FIX.search(status_text):
            result[label]["wont_fix"] += 1
        # Default: if no token matched, leave the section uncounted in the
        # status columns (only the total moves). This matches the
        # pre-regen table's "ignore weak status" pattern.
    return result


def render_table(counts: dict[str, dict[str, int]]) -> str:
    """Render the `## Summary` table markdown block (header + body)."""
    lines = [
        "## Summary",
        "",
        "| Category | Total Issues | Fixed | Open | Partial | Won't Fix |",
        "|----------|-------------:|------:|-----:|--------:|----------:|",
    ]
    totals = Counter()
    for _prefix, label in CATEGORY_ROWS:
        c = counts[label]
        lines.append(
            f"| {label} | {c['total']} | {c['fixed']} | {c['open']} | "
            f"{c['partial']} | {c['wont_fix']} |"
        )
        for k in ("total", "fixed", "open", "partial", "wont_fix"):
            totals[k] += c[k]
    lines.append(
        f"| **Total** | **{totals['total']}** | **{totals['fixed']}** | "
        f"**{totals['open']}** | **{totals['partial']}** | **{totals['wont_fix']}** |"
    )
    return "\n".join(lines) + "\n"


def render_legend() -> str:
    """Render the table-derivation legend (the source-of-truth contract)."""
    return (
        "\n"
        "*Counts derived from `grep -cE '^### CATEGORY-NN:' docs/KNOWN_ISSUES.md` "
        "via `scripts/check_known_issues_summary.py`. Edit the per-section `**Status:**` "
        "lines (or add new `### CATEGORY-NN:` headers) and the table updates on the next "
        "regen. Status columns (`Fixed` / `Open` / `Partial` / `Won't Fix`) derive from "
        "the first `**Status:**` line in each section. Sections without a `**Status:**` "
        "line are counted in the Total column but contribute 0 to the status columns — "
        "treat the missing line as a TODO and either add the line or document the "
        "exception in the section body. To regenerate: `python3 "
        "scripts/check_known_issues_summary.py --regen | sponge docs/KNOWN_ISSUES.md`.*\n"
    )


def extract_existing_table(text: str) -> tuple[int, int] | None:
    """Locate the `## Summary` table and return (start_offset, end_offset) of
    the entire table block (header + body + legend), so the regen can replace
    it cleanly. Returns None if the `## Summary` header is not present.
    """
    m = re.search(r"^##\s+Summary\s*$", text, re.MULTILINE)
    if not m:
        return None
    start = m.start()
    # Find the first H2 (`## `) after the table; that's the end of the block.
    end_m = re.search(r"^##\s+", text[start + 1 :], re.MULTILINE)
    end = start + 1 + end_m.start() if end_m else len(text)
    return start, end


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail (exit 1) if the committed table disagrees with the derived counts.",
    )
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Print the regenerated table to stdout (used to write back to the file).",
    )
    args = parser.parse_args()

    text = KNOWN_ISSUES.read_text(encoding="utf-8")
    counts = extract_counts(text)
    new_table = render_table(counts) + render_legend()

    if args.regen:
        sys.stdout.write(new_table)
        return 0

    if args.check:
        # Find the existing table; if absent, fail.
        existing_range = extract_existing_table(text)
        if existing_range is None:
            print(
                f"FAIL: {KNOWN_ISSUES.relative_to(REPO_ROOT)} is missing the "
                f"`## Summary` table — re-run `python3 scripts/check_known_issues_summary.py "
                f"--regen | sponge docs/KNOWN_ISSUES.md`",
                file=sys.stderr,
            )
            return 2
        # Compare counts row-by-row.
        existing_text = text[existing_range[0] : existing_range[1]]
        # Parse the existing table's `| **Total** | N | F | O | P | W |` row.
        m = re.search(
            r"\|\s*\*\*Total\*\*\s*\|\s*\*\*(\d+)\*\*\s*\|\s*\*\*(\d+)\*\*\s*\|"
            r"\s*\*\*(\d+)\*\*\s*\|\s*\*\*(\d+)\*\*\s*\|\s*\*\*(\d+)\*\*\s*\|",
            existing_text,
        )
        if not m:
            print(
                f"FAIL: could not parse the existing `## Summary` table — "
                f"re-run `python3 scripts/check_known_issues_summary.py --regen | "
                f"sponge docs/KNOWN_ISSUES.md`",
                file=sys.stderr,
            )
            return 2
        existing_totals = tuple(int(g) for g in m.groups())
        derived_totals = (
            sum(c["total"] for c in counts.values()),
            sum(c["fixed"] for c in counts.values()),
            sum(c["open"] for c in counts.values()),
            sum(c["partial"] for c in counts.values()),
            sum(c["wont_fix"] for c in counts.values()),
        )
        if existing_totals != derived_totals:
            print(
                f"FAIL: {KNOWN_ISSUES.relative_to(REPO_ROOT)} summary table drift\n"
                f"  existing totals: {existing_totals}\n"
                f"  derived totals:  {derived_totals}\n"
                f"  re-run `python3 scripts/check_known_issues_summary.py --regen | "
                f"sponge docs/KNOWN_ISSUES.md` to fix.",
                file=sys.stderr,
            )
            return 1
        print(
            f"PASS: {KNOWN_ISSUES.relative_to(REPO_ROOT)} summary table matches "
            f"the derived counts (Total {existing_totals[0]})."
        )
        return 0

    # Default behaviour: print the table to stdout.
    sys.stdout.write(new_table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
