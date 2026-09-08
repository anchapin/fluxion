#!/usr/bin/env python3
"""
CLI doc-stub consistency gate for Fluxion (Issue #3550).

The Fluxion `fluxion` binary ships three CLI path families that are
intentionally stubbed per issue #2947 (originally issue #2711):
`fluxion run -w ...`, `fluxion measure ...`, and the diagnostic
case-range paths (`fluxion validate-case` / `fluxion diagnose ...`).
The fail-loud contract is enforced in source by
`test_workflow_execution_is_gated_non_silent` and its siblings in
`src/bin/fluxion.rs`, and is policy-guarded by the AGENTS.md
"Physics and Validation Guardrails" bullet:

    The `fluxion` help includes intentionally stubbed paths (direct
    simulation, workflow/measure execution, and diagnostic case ranges).
    They must fail non-zero with issue `#2947`; do not turn them into
    silent success.

Issue #3534 corrected QUICKSTART.md §6 to match. Issue #3550 corrects
`docs/cli_design.md` to match. This script closes the gap so a future
docs patch cannot quietly re-introduce the contradiction: every
`fluxion run -w ...` / `fluxion measure ...` / `fluxion diagnose ...`
invocation in `docs/**/*.md` MUST carry an explicit issue `#2947`
annotation in its surrounding context, OR be inside a clearly-marked
historical / planning section that begins with the `#2947 stub-note`
sentinel.

Detection logic
---------------

A *stubbed-path reference* is a line in `docs/**/*.md` that contains
one of:

  - ``fluxion run -w``        (workflow execution; case-insensitive
                                inside the command flag)
  - ``fluxion measure``        (measure management)
  - ``fluxion diagnose``       (diagnostic case range)
  - ``fluxion validate-case``  (diagnostic case range; same #2947 gate)

For each hit the script builds a ``±WINDOW`` line context window
(default ``WINDOW = 12``) and checks that *at least one* line in that
window mentions issue ``#2947`` or the original-issue alias ``#2711``,
OR carries the literal ``stub-note`` / ``stubbed`` / ``fail-loud`` /
``SUPERSEDED`` sentinel. Hits that lack any annotation fail.

Allowed-exception patterns (not flagged):

  1. A line whose content is a *historical* / *planning* section
     inside a clearly-marked historical block whose first
     `#2947 stub-note` sentinel sits within the same window.
  2. A reference inside a fenced code block (```` ``` ````) that is
     itself preceded by a `#2947` / `stub-note` / `fail-loud` /
     `SUPERSEDED` line in the surrounding prose.

Scope
-----

Walks every `.md` file under `docs/`. The root `README.md` and
`AGENTS.md` are intentionally out of scope — both are root-policy
documents, and `AGENTS.md` is the *source* of the stub-policy
bullet, not a user-facing invocation surface. `README.md` is
covered separately by `scripts/check_root_md_policy.py`.

Usage::

    python3 scripts/check_cli_doc_stubs.py

Exit codes:

    0 — every CLI stub reference in `docs/**/*.md` carries a
        `#2947` annotation (PASS).
    1 — one or more CLI stub references lack a `#2947` annotation
        (FAIL); see stdout for `file:line` locations.
    2 — script error (e.g. IO failure walking the tree).

See issue #3550 acceptance criterion #3.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"

# ---------------------------------------------------------------------------
# Tunables. WINDOW is the ± line-context used when checking for a #2947
# annotation around each stub reference. 12 is large enough to cover a
# preceding `#2947 stub-note:` block + a fenced code block + a closing
# disclaimer (the worst-case shape in `cli_design.md` after #3550 lands),
# and small enough that an annotation in an unrelated section does not
# silently rescue a runnable-looking reference.
# ---------------------------------------------------------------------------
WINDOW = 12

# Stubbed CLI path families per AGENTS.md "Physics and Validation
# Guardrails" and src/bin/fluxion.rs gates (issues #2947 / #2711).
# Each entry is a regex that matches a *runnable-looking* invocation
# on the line — i.e. the lowercase binary name `fluxion` (the binary
# is always invoked lowercase; capitalized "Fluxion" in prose refers
# to the product name or the `FluxionMeasure` Python class, never to
# the CLI). The patterns accept any whitespace and any trailing
# argument so they also catch `fluxion  run -w foo.fwf` and
# `fluxion run -w` (no path).
STUBBED_INVOCATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bfluxion\s+run\s+-w\b"),
    re.compile(r"\bfluxion\s+measure\s+(?:--?[A-Za-z][\w-]*|update|compute_arguments|run_tests)\b"),
    re.compile(r"\bfluxion\s+diagnose\s+(?:--?[A-Za-z][\w-]*|\d)"),
    re.compile(r"\bfluxion\s+validate-case\s+(?:--?[A-Za-z][\w-]*|\d)"),
)

# An annotation may be either:
#   1. An explicit issue reference (`#2947` or the original `#2711`), or
#   2. One of the documented stub-policy sentinel words (case-insensitive).
ANNOTATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"#2947\b"),
    re.compile(r"#2711\b"),
    re.compile(r"\bstub-note\b", re.IGNORECASE),
    re.compile(r"\bstubbed\b", re.IGNORECASE),
    re.compile(r"\bfail[\s-]?loud\b", re.IGNORECASE),
    re.compile(r"\bSUPERSEDED\b"),
    # Explicit "not implemented" / "not yet implemented" prose.
    re.compile(r"\bnot\s+(?:yet\s+)?implemented\b", re.IGNORECASE),
    re.compile(r"\bnot\s+runnable\b", re.IGNORECASE),
)

# Issue link aliases — these are the ONLY references that count as
# proper annotations (the gate is policy-traceable to #2947). Anything
# else (e.g. "#1234 stubbed") is rejected so a stale-issue callout
# cannot accidentally satisfy the gate.

# ---------------------------------------------------------------------------
# File walking
# ---------------------------------------------------------------------------


def iter_docs_md_files() -> Iterable[Path]:
    if not DOCS_ROOT.is_dir():
        return
    for path in sorted(DOCS_ROOT.rglob("*.md")):
        # `docs/archive/sessions/` is a session-prompt draft archive and
        # is out of scope for the docs-hygiene contract; mirror the
        # `scripts/check_docs_summaries.py` exemption list and skip the
        # whole archive subtree here as well.
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.startswith("docs/archive/"):
            continue
        yield path


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def find_stub_invocations(lines: list[str]) -> list[int]:
    """Return 0-indexed line numbers of every stubbed-path invocation.

    Fenced code blocks (` ``` ` ... ` ``` `) are scanned too — a fenced
    block that says `fluxion run -w baseline.fwf` as a runnable example
    is exactly the contradiction issue #3550 is closing.
    """
    hits: list[int] = []
    for idx, line in enumerate(lines):
        if any(pat.search(line) for pat in STUBBED_INVOCATION_PATTERNS):
            hits.append(idx)
    return hits


def has_annotation_in_window(lines: list[str], hit_idx: int) -> bool:
    """Return True if any line in ±WINDOW around `hit_idx` carries an annotation.

    The window is bounded to the same fenced-code block as the hit when
    possible (so a fenced `#2947 stub-note` prologue to a fenced runnable
    block satisfies the gate, but a distant section's annotation does
    not). If the hit is outside any fenced block, the window is the
    full file (so a `#2947 stub-note` header line higher up still counts
    when the runnable-looking block lives below it in the same section).
    """
    block_start, block_end = enclosing_fenced_block(lines, hit_idx)
    if block_start is not None:
        window_start = block_start
        window_end = block_end
    else:
        window_start = max(0, hit_idx - WINDOW)
        window_end = min(len(lines) - 1, hit_idx + WINDOW)

    for j in range(window_start, window_end + 1):
        if any(pat.search(lines[j]) for pat in ANNOTATION_PATTERNS):
            return True
    return False


def enclosing_fenced_block(
    lines: list[str], hit_idx: int
) -> tuple[int | None, int | None]:
    r"""Return ``(start, end)`` 0-indexed inclusive bounds of the fenced
    code block containing ``hit_idx``, or ``(None, None)`` if the hit is
    not inside a fenced block.

    A "fenced block" here is a triple-backtick fence with optional
    language tag (``bash``, ``text``, ``json``, bare triple-backtick,
    etc.). We match ``\`\`\`(lang)?`` where ``lang`` is an optional
    alphanumeric+plus token at the start of the line; this is enough to
    handle every fenced block in the repo.
    """
    fence_open = re.compile(r"^\s*```[\w+-]*\s*$")
    open_idx: int | None = None
    for j in range(hit_idx + 1):
        line = lines[j]
        if fence_open.match(line):
            if open_idx is None:
                open_idx = j
            else:
                # Closing fence. If hit_idx is inside, return bounds.
                if open_idx <= hit_idx <= j:
                    return open_idx, j
                open_idx = None
    return None, None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== Fluxion CLI doc-stub consistency gate (Issue #3550) ===")
    print(f"Repo:         {REPO_ROOT}")
    print(f"Docs root:    {DOCS_ROOT}")
    print(f"Context ±:    {WINDOW} lines")
    print(
        "Stubbed CLI families: fluxion run -w ... / fluxion measure ... "
        "/ fluxion diagnose ... / fluxion validate-case ..."
    )
    print()

    findings: list[tuple[Path, int, str]] = []
    for path in iter_docs_md_files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"ERROR: could not read {path}: {exc}", file=sys.stderr)
            return 2
        lines = text.splitlines()
        for hit_idx in find_stub_invocations(lines):
            if not has_annotation_in_window(lines, hit_idx):
                rel = path.relative_to(REPO_ROOT)
                findings.append((rel, hit_idx + 1, lines[hit_idx].strip()))

    if not findings:
        print("PASS: every CLI stub reference in `docs/**/*.md` carries a #2947 annotation.")
        return 0

    print(f"FAIL: {len(findings)} CLI stub reference(s) without a #2947 annotation:")
    print()
    for rel, lineno, content in findings:
        print(f"  - {rel}:{lineno}")
        print(f"      {content}")
    print()
    print(
        "Each of these `fluxion run -w ...` / `fluxion measure ...` /\n"
        "`fluxion diagnose ...` / `fluxion validate-case ...` references\n"
        "needs an explicit `#2947` / `stub-note` / `fail-loud` /\n"
        "`SUPERSEDED` annotation within ±{} lines (issue #3550\n"
        "acceptance criterion #3; cross-reference issue #2947 and\n"
        "AGENTS.md 'Physics and Validation Guardrails').".format(WINDOW)
    )
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — top-level barrier per repo style
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)