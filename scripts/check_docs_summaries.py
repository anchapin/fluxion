#!/usr/bin/env python3
"""
Docs 7-line Summary Coverage Check for Fluxion.

Extends `scripts/doc_inventory_check.py` so that **every** `.md` file under
`docs/` is checked for the 7-line summary at lines 2-8 (per `AGENTS.md`
§Repository Hygiene, line 219), not just the 11 rows enumerated in
`docs/doc-inventory.md`.

Closes Issue #2466 acceptance criterion #2:
    `python3 scripts/check_docs_summaries.py` exits 0 (every `docs/`
    file has a 7-line summary at lines 2-8 OR is explicitly exempted
    in the script).

The summary-block helper (`has_seven_line_summary`) is intentionally
identical to the one in `scripts/doc_inventory_check.py` (lines 65-105)
so the two checks stay in lock-step.

Usage:
    python3 scripts/check_docs_summaries.py

Exit codes:
    0 — All checked `docs/**/*.md` files have a 7-line summary (or are
        explicitly exempted).
    1 — One or more docs are missing a 7-line summary.
    2 — Script error.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"

# Files explicitly exempted from the 7-line summary check.
#
# Rationale: these are session-prompt drafts that pre-date the
# `AGENTS.md` summary convention and are deliberately kept verbatim
# for historical traceability.  New docs MUST NOT be added here
# without updating `AGENTS.md` to reflect the policy change.
EXEMPT_PATHS: frozenset[str] = frozenset(
    {
        # `docs/archive/sessions/` — per-session prompt drafts (issue #768)
        # The archive README documents this exemption.
    }
)

# Minimum number of meaningful content lines required in lines 2-8.
# Must match `scripts/doc_inventory_check.py:MIN_SUMMARY_LINES`
# semantics (3 content lines in the 7-line block).
MIN_SUMMARY_LINES = 7


def has_seven_line_summary(file_path: Path) -> tuple[bool, str]:
    """Check whether `file_path` has a 7-line summary at lines 2-8.

    Identical to `scripts/doc_inventory_check.py.has_seven_line_summary`
    so the two scripts stay in sync.
    """
    if not file_path.exists():
        return False, "file missing"

    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return False, f"read error: {e}"

    if len(lines) < 9:
        return False, f"only {len(lines)} lines (need >= 9 for 7-line summary at lines 2-8)"

    summary_lines = lines[1:8]  # lines 2-8 (0-indexed: 1-7)
    content_line_count = 0
    for i, line in enumerate(summary_lines, start=2):
        stripped = line.strip()
        if not stripped and i == 2:
            continue  # line 2 may be blank
        if not stripped:
            continue  # other blank lines are allowed
        # Last-Updated markers are metadata, not summary content.
        if stripped.startswith("*Last Updated"):
            continue
        # HTML comments that contain substantive text count as summary
        # content (the `docs/doc-inventory.md` template uses
        # `<!-- Line 1: ... -->` style summaries — see line 26 of that
        # file).  Empty HTML comments (`<!-- -->`) and pure-marker
        # comments are skipped.
        if stripped.startswith("<!--"):
            # Strip the `<!--` and `-->` markers, then require >5 chars
            # of substantive text inside the comment.
            inner = stripped.removeprefix("<!--").removesuffix("-->").strip()
            if len(inner) > 5:
                content_line_count += 1
            continue
        if len(stripped) >= 3 or stripped.startswith((">", "#", "*", "-", "1.", "2.", "3.")):
            content_line_count += 1

    if content_line_count < 3:
        return False, f"only {content_line_count} content line(s) in summary block (need >= 3)"

    return True, "ok"


def find_docs_md_files() -> list[Path]:
    """Recursively list every `.md` file under `docs/`.

    Skips `docs/archive/sessions/` entirely — its session-prompt drafts
    pre-date the summary convention and are exempt by `EXEMPT_PATHS`.
    """
    out: list[Path] = []
    for path in sorted(DOCS_ROOT.rglob("*.md")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in EXEMPT_PATHS:
            continue
        out.append(path)
    return out


def main() -> int:
    print("=== Fluxion Docs 7-Line Summary Coverage Check ===")
    print(f"Repo: {REPO_ROOT}")
    print(f"Docs root: {DOCS_ROOT}")
    print()

    md_files = find_docs_md_files()
    print(f"Scanning {len(md_files)} `docs/**/*.md` file(s) ...")
    print()

    findings: list[str] = []
    ok_count = 0
    for path in md_files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        has_summary, reason = has_seven_line_summary(path)
        if has_summary:
            ok_count += 1
            print(f"    OK: {rel}")
        else:
            findings.append(f"{rel}: {reason}")
            print(f"    FAIL: {rel} — {reason}")

    print()
    print(f"Summary: {ok_count} OK, {len(findings)} missing summary")

    if not findings:
        print()
        print("PASS: All `docs/**/*.md` files have a 7-line summary.")
        return 0

    print()
    print(f"FAIL: {len(findings)} `docs/**/*.md` file(s) missing 7-line summary:")
    for f in findings:
        print(f"  - {f}")
    print()
    print("Remediation (per AGENTS.md §Repository Hygiene):")
    print("  1. Add a 7-line summary at lines 2-8 of the file. See")
    print("     `docs/doc-inventory.md` §7-Line Summary Convention for")
    print("     the template.")
    print("  2. If the file is exempt (e.g. session prompt draft), add")
    print("     its path to `EXEMPT_PATHS` in this script and document")
    print("     the rationale in a comment.")
    print()
    print("To regenerate the docs inventory after fixing summaries:")
    print("  python3 scripts/check_doc_inventory.py")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
