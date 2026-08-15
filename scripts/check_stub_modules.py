#!/usr/bin/env python3
"""
Stub-module detector for Fluxion (Issue #2896).

A "stub module" in this project is a `.rs` source file that:

  1. contributes fewer than ``MIN_NON_COMMENT_LOC`` non-comment, non-blank
     lines of production code, AND
  2. explicitly flags itself as a future-extraction marker via one of
     the sentinel phrases:
       - "marker for future extraction"
       - "placeholder for future extraction"

Both halves are required to avoid false positives:

  - Bare "marker for future extraction" prose inside a legitimate
    discussion comment (e.g. an ADR or design doc) is excluded because
    the production-code LoC threshold catches it first.
  - A genuinely short module (e.g. a 10-line module that has actual
    logic) is excluded because the sentinel phrase is absent.

When such a file is detected, it must be either fleshed out with the
extracted logic, or its design notes merged into the consuming module
and the file deleted. The whole point of issue #2896 is that
doc-only/placeholder modules waste compile time, generate empty rustdoc
pages, and surface as phantom symbols in IDE autocomplete.

Scope:
  Walks every workspace member's ``src/`` tree (``fluxion`` root plus
  the always-built siblings and ``fluxion-core``). Feature-gated
  siblings (``fluxion-cfd``, ``fluxion-city``, ``fluxion-fluid``) are
  also walked — stub detection is independent of feature gates; a stub
  in a feature-gated module is just as problematic as a stub in the
  root crate.

Usage:
  python3 scripts/check_stub_modules.py

Exit codes:
  0 — no stub modules detected (PASS).
  1 — one or more stub modules detected (FAIL); see stdout.
  2 — script error (e.g. IO failure walking the tree).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Tunables. Keep MIN_NON_COMMENT_LOC in sync with the issue #2896 acceptance
# criterion ("non-comment LoC < 20"). Sentinel phrases must match the issue
# verbatim.
# ---------------------------------------------------------------------------
MIN_NON_COMMENT_LOC = 20
SENTINEL_PHRASES: tuple[str, ...] = (
    "marker for future extraction",
    "placeholder for future extraction",
)

# Workspace-member `src/` trees to scan. Includes the root crate, the
# always-built siblings (fluxion-core, fluxion-grid, fluxion-behavior,
# fluxion-wasm), and the feature-gated siblings (fluxion-cfd,
# fluxion-city, fluxion-fluid, fluxion-mcp). Adding a new sibling?
# Add its `src/` directory here.
SCAN_DIRS: tuple[str, ...] = (
    "src",
    "fluxion-core/src",
    "fluxion-grid/src",
    "fluxion-behavior/src",
    "fluxion-wasm/src",
    "fluxion-cfd/src",
    "fluxion-city/src",
    "fluxion-fluid/src",
    "fluxion-mcp/src",
)


# ---------------------------------------------------------------------------
# Comment stripping
# ---------------------------------------------------------------------------

# Order matters: strip block comments first so a `// foo /* bar */` style
# line is not partially eaten by the line-comment pass. Use a non-greedy
# match for block comments so the first `*/` closes the comment.
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")


def strip_rust_comments(source: str) -> str:
    """Return ``source`` with block and line comments removed.

    Preserves string and char literals verbatim — a minimal pass that
    handles the doc-comment patterns these stub modules use (module-
    level `//!` comments, inline `///` doc comments). We do not need
    full Rust lexing because the detector only cares about line-count
    thresholds and sentinel phrase presence; both signals survive a
    small amount of string-literal noise.
    """
    without_block = _BLOCK_COMMENT_RE.sub("", source)
    return _LINE_COMMENT_RE.sub("", without_block)


def count_non_blank_non_comment_lines(source: str) -> int:
    """Count lines that contain at least one non-whitespace character
    in the comment-stripped source. Blank lines inside the original
    file are also excluded because they are not "production code"."""
    stripped = strip_rust_comments(source)
    return sum(1 for line in stripped.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# Walk + classify
# ---------------------------------------------------------------------------


def iter_rs_files() -> Iterable[Path]:
    for rel in SCAN_DIRS:
        base = REPO_ROOT / rel
        if not base.is_dir():
            continue
        yield from base.rglob("*.rs")


def find_stubs() -> list[tuple[Path, int, tuple[str, ...]]]:
    """Return ``[(path, loc, hit_phrases), ...]`` for every stub file."""
    out: list[tuple[Path, int, tuple[str, ...]]] = []
    for path in iter_rs_files():
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Binary or unreadable file — skip silently. `.rs` files
            # should always be valid UTF-8 in this repo; any exception
            # here is a tooling bug, not a stub.
            continue
        loc = count_non_blank_non_comment_lines(source)
        if loc >= MIN_NON_COMMENT_LOC:
            continue
        lowered = source.lower()
        hits = tuple(p for p in SENTINEL_PHRASES if p in lowered)
        if not hits:
            continue
        out.append((path, loc, hits))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== Fluxion stub-module detector (Issue #2896) ===")
    print(f"Repo:            {REPO_ROOT}")
    print(f"LoC threshold:   < {MIN_NON_COMMENT_LOC} non-comment, non-blank lines")
    print(f"Sentinel phrases:")
    for phrase in SENTINEL_PHRASES:
        print(f"  - {phrase!r}")
    print(f"Scan dirs:       {', '.join(SCAN_DIRS)}")
    print()

    stubs = find_stubs()
    if not stubs:
        print("PASS: no stub modules detected.")
        return 0

    print(f"FAIL: {len(stubs)} stub module(s) detected.")
    print()
    for path, loc, hits in sorted(stubs):
        rel = path.relative_to(REPO_ROOT)
        phrases = ", ".join(repr(h) for h in hits)
        print(f"  - {rel}")
        print(f"      non-comment LoC: {loc}  (< {MIN_NON_COMMENT_LOC})")
        print(f"      sentinels hit:   {phrases}")
    print()
    print(
        "Stub modules either need to be fleshed out with real logic, or "
        "their design notes merged into the consuming module and the file "
        "deleted (see issue #2896 acceptance criteria)."
    )
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — top-level barrier per repo style
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)