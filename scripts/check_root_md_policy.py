#!/usr/bin/env python3
"""
Root `.md` Policy Check for Fluxion.

Enforces `AGENTS.md` §Repository Hygiene: only a fixed allow-list of `.md`
files may live at the repo root.  Anything else is a transient artifact
and must be moved to `tmp/`, `docs/archive/`, or `docs/investigations/`
before committing.

Allow-list (per `AGENTS.md` lines 214-220):
    README.md, ARCHITECTURE.md, CODEBASE_MAP.md, CONTRIBUTING.md,
    RULES.md, CHANGELOG.md, AGENTS.md

Special handling:
    CLAUDE.md — auto-generated per-session by the Bernstein agent.
                 Warns but does not fail (the policy relies on
                 `.gitignore` rather than the gate to keep it out of
                 the repo).  See AGENTS.md line 218.

Usage:
    python3 scripts/check_root_md_policy.py

Exit codes:
    0 — All checks pass (only allow-listed `.md` files at root; no
        transient files detected; CLAUDE.md is either absent or
        git-ignored).
    1 — One or more transient `.md` files exist at root.
    2 — Script error.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The hard allow-list.  Anything else at the root is a violation.
# Keep in sync with AGENTS.md §Repository Hygiene (line 216).
ROOT_MD_ALLOWLIST: frozenset[str] = frozenset(
    {
        "README.md",
        "ARCHITECTURE.md",
        "CODEBASE_MAP.md",
        "CONTRIBUTING.md",
        "RULES.md",
        "CHANGELOG.md",
        "AGENTS.md",
    }
)

# Auto-generated per-session file that should never be committed.
# Warns but never fails (per AGENTS.md line 218).  Belt-and-braces
# protection lives in `.gitignore`.
ROOT_MD_WARNLIST: frozenset[str] = frozenset({"CLAUDE.md"})


def find_root_md_files() -> list[Path]:
    """List every `.md` file directly at the repo root (no recursion)."""
    return sorted(p for p in REPO_ROOT.iterdir() if p.is_file() and p.suffix == ".md")


def is_gitignored(path: Path) -> bool:
    """Return True if `path` is matched by a `.gitignore` rule.

    Uses `git check-ignore` so the answer matches what Git itself
    considers ignored.  If Git is unavailable or the path is not in a
    Git repo, falls back to a permissive answer (returns False).
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(path.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def main() -> int:
    print("=== Fluxion Root `.md` Policy Check ===")
    print(f"Repo: {REPO_ROOT}")
    print(f"Allow-list: {sorted(ROOT_MD_ALLOWLIST)}")
    print(f"Warn-list (non-blocking): {sorted(ROOT_MD_WARNLIST)}")
    print()

    root_md_files = find_root_md_files()
    transient: list[Path] = []
    warned: list[Path] = []
    tracked: list[Path] = []

    for path in root_md_files:
        name = path.name
        if name in ROOT_MD_ALLOWLIST:
            tracked.append(path)
            continue
        if name in ROOT_MD_WARNLIST:
            if is_gitignored(path):
                tracked.append(path)  # git-ignored: treat as compliant
                print(f"    OK (git-ignored): {name}")
            else:
                warned.append(path)
                print(f"    WARN: {name} present but NOT git-ignored")
            continue
        transient.append(path)
        print(f"    FAIL: {name} is not in the root allow-list")

    print()
    print(
        f"Scanned {len(root_md_files)} `.md` file(s) at repo root: "
        f"{len(tracked)} allow-listed, {len(warned)} warned, "
        f"{len(transient)} transient."
    )

    if warned:
        print()
        print(
            f"WARN: {len(warned)} warn-listed `.md` file(s) present and "
            f"NOT git-ignored (CLAUDE.md is auto-generated per AGENTS.md "
            f"line 218 and should be added to `.gitignore`):"
        )
        for p in warned:
            print(f"  - {p.name}")
        print(
            "  Add the relevant filename(s) to `.gitignore` and remove "
            "from the working tree (`git rm --cached` for tracked files)."
        )

    if not transient:
        print()
        print("PASS: No transient `.md` files at repo root.")
        return 0

    print()
    print(f"FAIL: {len(transient)} transient `.md` file(s) at repo root:")
    for p in transient:
        print(f"  - {p.name}")
    print()
    print("Remediation (per AGENTS.md §Repository Hygiene):")
    print("  1. Move the file to `tmp/`, `docs/archive/`,")
    print("     `docs/archive/planning/`, `docs/archive/sessions/`,")
    print("     `docs/investigations/`, or another appropriate location.")
    print("  2. If it is a case analysis: `docs/investigations/`")
    print("  3. If it is a session summary / batch log: `docs/archive/planning/`")
    print("  4. If it is a security report: `docs/archive/security/`")
    print("  5. Update any links in other docs to the new path.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
