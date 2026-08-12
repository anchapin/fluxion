#!/usr/bin/env python3
"""
Audit-Config Uniqueness Check for Fluxion.

Enforces a single canonical `cargo audit` config: exactly one `audit.toml`
must exist, and it must be the canonical one under `.cargo/audit.toml`
(cargo-audit's default lookup path). A stray root-level `audit.toml` is a
supply-chain hazard — it shadows or contradicts the canonical file, can
mislabel ignored advisories, and puts directives (e.g. `deny = [...]`) under
the wrong table where they silently no-op. See issue #2773.

Scanned locations:
    - Repo root, non-recursive (a stray `audit.toml` here is the historical
      trap — cargo-audit does NOT read it by default, so it is pure cruft
      that misleads reviewers).
    - `.cargo/` (the canonical path cargo-audit reads by default).

Usage:
    python3 scripts/check_audit_config_unique.py

Exit codes:
    0 — Exactly one `audit.toml` found (the canonical `.cargo/audit.toml`).
    1 — Zero or more than one `audit.toml` found.
    2 — Script error.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_REL = Path(".cargo") / "audit.toml"


def find_audit_tomls() -> list[Path]:
    """Return every `audit.toml` at the repo root (non-recursive) and
    under `.cargo/` (non-recursive inside that directory)."""
    found: list[Path] = []

    # Repo root, non-recursive.
    for p in REPO_ROOT.iterdir():
        if p.is_file() and p.name == "audit.toml":
            found.append(p)

    # `.cargo/`, non-recursive.
    cargo_dir = REPO_ROOT / ".cargo"
    if cargo_dir.is_dir():
        for p in cargo_dir.iterdir():
            if p.is_file() and p.name == "audit.toml":
                found.append(p)

    return sorted(found)


def main() -> int:
    print("=== Fluxion Audit-Config Uniqueness Check ===")
    print(f"Repo: {REPO_ROOT}")
    print(f"Canonical path: {CANONICAL_REL}")
    print()

    found = find_audit_tomls()
    canonical_abs = REPO_ROOT / CANONICAL_REL

    for p in found:
        rel = p.relative_to(REPO_ROOT)
        tag = "CANONICAL" if p == canonical_abs else "STRAY"
        print(f"  [{tag}] {rel}")

    print()
    print(f"Found {len(found)} `audit.toml` file(s).")

    if len(found) == 1 and found[0] == canonical_abs:
        print()
        print("PASS: Exactly one `audit.toml` at the canonical path.")
        return 0

    print()
    print(
        "FAIL: Expected exactly one `audit.toml` at "
        f"`{CANONICAL_REL}` (cargo-audit's default lookup path). "
        f"Found {len(found)}."
    )
    print()
    print("Remediation (issue #2773, AGENTS.md §Toolchain Quirks):")
    print("  - A root-level `audit.toml` is dead weight: cargo-audit reads")
    print("    `.cargo/audit.toml` by default, so the root file is ignored")
    print("    by the tool but still misleads reviewers and can shadow the")
    print("    canonical config in editors/searches.")
    print("  - Delete the stray file(s) with `git rm <path>`.")
    print("  - If `.cargo/audit.toml` is missing, restore it — it is the")
    print("    source of truth for advisory ignores.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
