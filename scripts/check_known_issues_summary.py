#!/usr/bin/env python3
"""
Regenerate the `## Summary` table at the top of `docs/KNOWN_ISSUES.md` from
the per-row catalog tables under each `## ` category section.

Issue #3513: the hand-maintained table undercounted the LIMIT entries by
17+ and the SOLAR / BASE / MULTI counts were also stale. Issue #4288 then
restructured the document body from `### CATEGORY-NN:` sections into
per-category tables whose rows look like `| **BASE-01** | ... | open | ... |`,
so this script counts each catalog row and its Status column.

Usage:
    python3 scripts/check_known_issues_summary.py [--check] [--regen]

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

# Map of category prefix -> table row label.
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
    ("REF", "Reference data (REF)"),
]

# A catalog row in a category table: `| **BASE-01** | ... |` (also matches
# suffixed IDs like `**BASE-01b**` and multi-case prefixes like
# `**PeakHeatingLimit-01**`, which are attributed to their section).
ROW_RE = re.compile(r"^\|\s*\*\*([A-Za-z]+)-(\d+)([a-z]?)\*\*\s*\|")

# A category section heading. The category token is either the
# parenthesized suffix - `## Foundation (BASE)` - or the bare heading text
# (`## CI`).
HEADING_CATEGORY_RE = re.compile(r"\(([A-Z]+)\)\s*$")

# Status column vocabulary (post-#4288). The cell is compared
# case-insensitively after whitespace stripping; anything outside this map
# leaves the row counted in the Total column only.
STATUS_MAP = {
    "resolved": "fixed",
    "fixed": "fixed",
    "open": "open",
    "tracking only": "partial",
    "partial": "partial",
    "won't fix": "wont_fix",
    "wontfix": "wont_fix",
}


def extract_counts(text: str) -> dict[str, dict[str, int]]:
    """Walk every `## ` category section and count its catalog table rows,
    classifying each row by its Status column.

    Returns: {row_label: {"total": N, "fixed": F, "open": O, "partial": P,
                         "wont_fix": W}}
    """
    # Build a list of (category_prefix, start_offset) for every top-level
    # `## ` heading that maps to a known category.
    section_starts: list[tuple[str, int]] = []
    for m in re.finditer(r"^##\s+(.+?)\s*$", text, re.MULTILINE):
        heading = m.group(1)
        if heading == "Summary":
            continue
        hm = HEADING_CATEGORY_RE.search(heading)
        prefix = hm.group(1) if hm else heading.strip()
        section_starts.append((prefix, m.start()))

    # Build the result dict, defaulting all counts to 0.
    result: dict[str, dict[str, int]] = {}
    for _prefix, label in CATEGORY_ROWS:
        result[label] = {"total": 0, "fixed": 0, "open": 0, "partial": 0, "wont_fix": 0}

    for i, (prefix, start) in enumerate(section_starts):
        # Find the next `## ` (any heading) or EOF.
        end = section_starts[i + 1][1] if i + 1 < len(section_starts) else len(text)
        body = text[start:end]

        # Find the row label for this section's category.
        label = next((lbl for pfx, lbl in CATEGORY_ROWS if pfx == prefix), None)
        if label is None:
            # Not a category section (e.g. `## How to read this document`).
            continue

        for line in body.splitlines():
            rm = ROW_RE.match(line)
            if not rm:
                continue
            result[label]["total"] += 1

            # The Status column sits between `Closes when` and `History`:
            # `| ID | Symptom | Current | Issue | Closes | Status | History |`.
            cells = [c.strip() for c in line.split("|")]
            if len(cells) < 8:
                continue
            status = STATUS_MAP.get(cells[6].lower())
            if status is not None:
                result[label][status] += 1
            # Unrecognized status text leaves the row counted in Total only.
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
        "*Counts derived from the per-row catalog tables under each category "
        "section (`| **CATEGORY-NN** | ... |`) via "
        "`scripts/check_known_issues_summary.py`. Edit a row in place (or add a "
        "new row) and the table updates on the next regen. Status columns "
        "(`Fixed` / `Open` / `Partial` / `Won't Fix`) derive from each row's "
        "Status cell: `resolved` -> Fixed, `open` -> Open, `tracking only` -> "
        "Partial, `Won't Fix` -> Won't Fix. Rows without a recognized status are "
        "counted in the Total column but contribute 0 to the status columns. To "
        "regenerate: `python3 "
        "scripts/check_known_issues_summary.py --regen | sponge docs/KNOWN_ISSUES.md`.*\n"
    )


def extract_existing_table(text: str) -> tuple[int, int] | None:
    """Locate the `## Summary` block and return (start_offset, end_offset) of
    the entire block (header + table + legend), so the regen can replace it
    cleanly. The block ends at the end of the legend line - NOT at the next
    `## ` heading - so content sitting between the legend and the following
    heading (e.g. the doc intro and `*Last Updated:*` marker, as lost in
    PR #4309) is never swallowed by a regeneration. Returns None if the
    `## Summary` header is not present.
    """
    m = re.search(r"^##\s+Summary\s*$", text, re.MULTILINE)
    if not m:
        return None
    start = m.start()
    # The generated block ends with the legend line rendered by
    # render_legend(): a single italic line starting "*Counts derived from".
    legend_m = re.search(
        r"^\*Counts derived from the per-row catalog tables.*?\*\s*$",
        text[start:],
        re.MULTILINE,
    )
    if legend_m:
        end = start + legend_m.end()
        return start, end
    # Fallback for a legacy file with no legend: the first H2 after the
    # table is the end of the block.
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
                f"`## Summary` table - re-run `python3 scripts/check_known_issues_summary.py "
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
                "FAIL: could not parse the existing `## Summary` table - "
                "re-run `python3 scripts/check_known_issues_summary.py --regen | "
                "sponge docs/KNOWN_ISSUES.md`",
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
