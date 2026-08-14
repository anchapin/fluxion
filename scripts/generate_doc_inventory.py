#!/usr/bin/env python3
"""
Regenerate `docs/doc-inventory.md` to enumerate every `docs/**/*.md` file.

Closes Issue #2466 acceptance criterion #3:
    `docs/doc-inventory.md` table is regenerated to enumerate **all**
    `docs/**/*.md` files (no longer just 11 rows).

This script regenerates the table portion of `docs/doc-inventory.md`
between the `<!-- BEGIN AUTO-GENERATED INVENTORY -->` and
`<!-- END AUTO-GENERATED INVENTORY -->` markers.  All other content
(intro, 7-Line Summary Convention, Case 600 #[ignore] tracking,
Maintenance) is preserved verbatim.

Usage:
    python3 scripts/generate_doc_inventory.py

Exit codes:
    0 — Inventory regenerated successfully.
    1 — `docs/doc-inventory.md` missing required markers.
    2 — Script error.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"
INVENTORY_FILE = REPO_ROOT / "docs" / "doc-inventory.md"

BEGIN_MARKER = "<!-- BEGIN AUTO-GENERATED INVENTORY -->"
END_MARKER = "<!-- END AUTO-GENERATED INVENTORY -->"


def _list_tracked_docs(repo_root: Path) -> list[Path] | None:
    """Enumerate git-tracked ``docs/**/*.md`` files, or ``None`` on git failure.

    Uses ``git ls-files 'docs/*.md' 'docs/**/*.md'`` so the script's
    view matches the committed tree (Closes #2961).  Files matched by
    ``.gitignore`` (e.g. ``**/*_PLAN.md``, ``**/*_ANALYSIS.md``) are
    excluded automatically because they are never tracked.

    Two pattern globs are needed because git's pathspec semantics do
    not make ``**`` match zero intermediate directories — ``docs/**/*.md``
    alone would miss top-level files like ``docs/README.md``.  The
    combined pattern matches the recursive ``Path.rglob("*.md")``
    semantics the script used previously.

    Returns ``None`` when git is unavailable or ``repo_root`` is not
    inside a working tree (e.g. ``tests/check_*`` harness uses
    ``tmp_path`` as a synthetic repo root).  The caller is expected to
    fall back to a filesystem walk with a printed warning so the
    generator remains usable in non-git contexts.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "docs/*.md", "docs/**/*.md"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        print(
            f"WARNING: git ls-files failed ({type(e).__name__}: {e}); "
            f"falling back to filesystem walk.  The committed inventory "
            f"may diverge from the working tree when gitignored files "
            f"are present.",
            file=sys.stderr,
        )
        return None

    paths: list[Path] = []
    for line in result.stdout.splitlines():
        rel = line.strip()
        if not rel or not rel.endswith(".md"):
            # Defensive: ``git ls-files`` always emits repo-relative
            # paths, but a stray empty line or non-.md entry would
            # silently corrupt the table.  Skip explicitly.
            continue
        paths.append(repo_root / rel)
    return paths


def has_seven_line_summary(file_path: Path) -> bool:
    """Return True if the file has a 7-line summary at lines 2-8.

    Mirrors the helper in `scripts/check_docs_summaries.py` and
    `scripts/doc_inventory_check.py`.  We deliberately duplicate the
    logic here rather than importing the other scripts because:
      1. The other scripts are also in flux — keeping the
         implementation identical to all of them would require a
         shared module.
      2. The intent here is only to render a status emoji.
    """
    if not file_path.exists():
        return False
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return False
    if len(lines) < 9:
        return False
    summary_lines = lines[1:8]
    content_line_count = 0
    for i, line in enumerate(summary_lines, start=2):
        stripped = line.strip()
        if not stripped and i == 2:
            continue
        if not stripped:
            continue
        if stripped.startswith("*Last Updated"):
            continue
        if stripped.startswith("<!--"):
            inner = stripped.removeprefix("<!--").removesuffix("-->").strip()
            if len(inner) > 5:
                content_line_count += 1
            continue
        if len(stripped) >= 3 or stripped.startswith((">", "#", "*", "-", "1.", "2.", "3.")):
            content_line_count += 1
    return content_line_count >= 3


def build_inventory_table() -> str:
    """Build the markdown table that enumerates every `docs/**/*.md` file.

    Enumeration is git-tracked-only so that gitignored files
    (``**/*_PLAN.md``, ``**/*_ANALYSIS.md``, ...) cannot sneak into the
    committed inventory and break ``check_doc_inventory_fresh.py`` in
    CI.  When git is unavailable (e.g. test harness with a synthetic
    ``tmp_path`` repo root), the function falls back to a filesystem
    walk — see ``_list_tracked_docs`` for the rationale.
    """
    rows: list[str] = []
    tracked = _list_tracked_docs(REPO_ROOT)
    if tracked is None:
        md_files = sorted(p for p in DOCS_ROOT.rglob("*.md") if p.is_file())
    else:
        md_files = sorted(tracked)
    for path in md_files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        status = "✅ Has summary" if has_seven_line_summary(path) else "❌ Missing summary"
        rows.append(f"| [{rel}]({rel}) | {rel} | {status} |")
    header = (
        "| Doc | Path | Status |\n"
        "|-----|------|--------|"
    )
    return "\n".join([header, *rows])


def regenerate() -> int:
    if not INVENTORY_FILE.exists():
        print(f"ERROR: {INVENTORY_FILE} not found", file=sys.stderr)
        return 1

    content = INVENTORY_FILE.read_text(encoding="utf-8")
    if BEGIN_MARKER not in content or END_MARKER not in content:
        print(
            f"ERROR: {INVENTORY_FILE} is missing the "
            f"`{BEGIN_MARKER}` / `{END_MARKER}` markers.  "
            f"Add them around the inventory table and re-run.",
            file=sys.stderr,
        )
        return 1

    # Pattern: capture everything between the markers (exclusive).
    pattern = re.compile(
        re.escape(BEGIN_MARKER) + r"(.*?)" + re.escape(END_MARKER),
        re.DOTALL,
    )
    new_block = (
        BEGIN_MARKER
        + "\n"
        + "The table below is auto-generated by "
        + "`scripts/generate_doc_inventory.py` — do not edit by hand.  "
        + "Re-run after adding/removing files under `docs/`.\n\n"
        + build_inventory_table()
        + "\n"
        + END_MARKER
    )
    new_content = pattern.sub(new_block, content)

    if new_content == content:
        print(f"Inventory already up to date in {INVENTORY_FILE.relative_to(REPO_ROOT)}")
        return 0

    INVENTORY_FILE.write_text(new_content, encoding="utf-8")
    # Count what the table actually enumerates so the printed number
    # matches the rendered rows in either enumerated-source path
    # (git-tracked vs. filesystem walk fallback).
    tracked = _list_tracked_docs(REPO_ROOT)
    if tracked is None:
        enumerated_count = len(list(DOCS_ROOT.rglob("*.md")))
    else:
        enumerated_count = len(tracked)
    print(
        f"Regenerated inventory table in {INVENTORY_FILE.relative_to(REPO_ROOT)} "
        f"({enumerated_count} docs enumerated)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(regenerate())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
