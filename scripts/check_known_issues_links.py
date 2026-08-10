#!/usr/bin/env python3
"""
Verify that every intra-repo Markdown link in `docs/FAQ.md` and
`docs/TROUBLESHOOTING.md` resolves to a file that exists in the
repository.

Closes Issue #2541 acceptance criterion:
    A `scripts/check_known_issues_links.py` test verifies the FAQ
    links resolve.

The check walks the markdown link syntax `[label](target)` and, for
each link whose target is a relative path (no scheme, no anchor-only),
confirms the target file exists on disk. It does **not** fetch remote
URLs — only intra-repo references.

Usage:
    python3 scripts/check_known_issues_links.py

Exit codes:
    0 — All intra-repo links in the FAQ/TROUBLESHOOTING resolve.
    1 — One or more links are broken.
    2 — Script error.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"

# Documents whose links we audit. Both are the canonical
# user-facing / developer-facing troubleshooting surfaces (issue #2541).
AUDITED_FILES = (
    DOCS_ROOT / "FAQ.md",
    DOCS_ROOT / "TROUBLESHOOTING.md",
)

# Match Markdown link syntax: [label](target)
# Captures the target only. Skips code spans / fenced code blocks
# by virtue of the target needing to look like a path or URL.
LINK_RE = re.compile(r"\[(?:[^\]]+)\]\(([^)]+)\)")


def is_intra_repo(target: str) -> bool:
    """Return True if `target` is a relative intra-repo path."""
    if not target:
        return False
    # Strip an optional anchor fragment.
    path_part = target.split("#", 1)[0]
    if not path_part:
        return False  # anchor-only link like (#section)
    # Skip URLs with a scheme (http://, https://, mailto:, …).
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", path_part):
        return False
    if path_part.startswith("mailto:"):
        return False
    return True


def resolve_link(source_doc: Path, target: str) -> Path | None:
    """Resolve `target` relative to `source_doc`'s parent directory."""
    path_part = target.split("#", 1)[0]
    # Markdown links are resolved relative to the file containing them.
    resolved = (source_doc.parent / path_part).resolve()
    return resolved


def audit_file(path: Path) -> list[tuple[str, str]]:
    """Return a list of (target, reason) tuples for broken links."""
    broken: list[tuple[str, str]] = []
    if not path.exists():
        broken.append((str(path), "audited file itself is missing"))
        return broken

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        broken.append((str(path), f"read error: {e}"))
        return broken

    for match in LINK_RE.finditer(content):
        target = match.group(1).strip()
        if not is_intra_repo(target):
            continue
        resolved = resolve_link(path, target)
        if not resolved.exists():
            broken.append((target, f"-> {resolved} (does not exist)"))
    return broken


def main() -> int:
    print("=== Fluxion FAQ/TROUBLESHOOTING Link Check ===")
    print(f"Repo: {REPO_ROOT}")
    print(f"Auditing: {', '.join(p.name for p in AUDITED_FILES)}")
    print()

    total_broken = 0
    for doc in AUDITED_FILES:
        broken = audit_file(doc)
        if broken:
            total_broken += len(broken)
            print(f"FAIL: {doc.relative_to(REPO_ROOT)} — {len(broken)} broken link(s):")
            for target, reason in broken:
                print(f"  - {target}  {reason}")
        else:
            print(f"  OK: {doc.relative_to(REPO_ROOT)}")

    print()
    if total_broken == 0:
        print("PASS: All intra-repo links resolve.")
        return 0

    print(f"FAIL: {total_broken} broken intra-repo link(s).")
    print("Remediation: fix the target path, or convert to an absolute")
    print("URL if the target is external.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
