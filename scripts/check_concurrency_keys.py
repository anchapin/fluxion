#!/usr/bin/env python3
"""
CI guard: verify that every `.github/workflows/*.yml` carries the
ADR-0015 per-`head_sha` `concurrency:` block (Issue #3366).

Companion to `scripts/update_concurrency_keys.py` — that script applies
the block, this script enforces it. Wired into `scripts-tests.yml` so
any workflow drift (e.g. an engineer adding a new workflow without the
template) fails the CI gate before it can merge.

For each `.github/workflows/*.yml`:

  1. The file MUST declare a `concurrency:` top-level key.
  2. The block MUST contain both a `group:` and a `cancel-in-progress:`
     key.
  3. The `group:` value MUST evaluate to per-`head_sha` for
     `pull_request` events — i.e. must reference
     `github.event.pull_request.head.sha` and combine it with
     `github.ref` via `||`.
  4. The `cancel-in-progress:` value MUST be conditional on
     `github.event_name == 'push'` AND on the ref being `refs/heads/main`
     or `refs/heads/develop` — i.e. must reference both
     `github.event_name == 'push'` and
     `contains('refs/heads/main,refs/heads/develop', github.ref)`.

Any file that fails any check is reported as a drift with the file
name and the specific failing invariant. Exit code is 1 on drift, 0 on
clean.

Usage:
    python3 scripts/check_concurrency_keys.py

Exit codes:
    0 — every workflow carries the ADR-0015 template.
    1 — one or more workflows are missing the template or carry a stale
        variant.
    2 — script error (e.g. `.github/workflows/` missing).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Match a top-level `concurrency:` block: the `concurrency:` line followed
# by one or more indented lines until the next 0-indent key or EOF.
_CONCURRENCY_BLOCK_RE = re.compile(
    r"^concurrency:\s*\n"
    r"((?:[ \t]+[^\n]*\n)+)"
    r"(?=\n[^\t ]|\Z)",
    re.MULTILINE,
)

# Pull the `group:` and `cancel-in-progress:` lines from a block. We only
# need the *value* side — the rest of the folded scalar (the indented
# continuation lines) is matched separately by the distinctive-marker
# tests below.
_GROUP_LINE_RE = re.compile(r"^[ \t]+group:[ \t]*([^\n]+)", re.MULTILINE)
_CANCEL_LINE_RE = re.compile(
    r"^[ \t]+cancel-in-progress:[ \t]*([^\n]+)", re.MULTILINE
)


def extract_block(text: str) -> str | None:
    """Return the raw `concurrency:` block text, or None if absent."""
    m = _CONCURRENCY_BLOCK_RE.search(text)
    return m.group(0) if m else None


def block_uses_per_sha_group(block: str) -> bool:
    """True when the `group:` folded scalar references
    ``github.event.pull_request.head.sha`` and combines it with
    ``github.ref`` via ``||`` (the ADR-0015 per-SHA shape)."""
    # Folded scalars (`>-`) put the value across multiple lines. Pull
    # the first `group:` line plus the next ~8 indented lines and check
    # both markers are present in that span.
    group_m = _GROUP_LINE_RE.search(block)
    if not group_m:
        return False
    start = group_m.start()
    # Capture the `group:` line plus up to 8 lines of folded continuation.
    fold_lines = block[start:].split("\n", 9)[:9]
    fold_text = "\n".join(fold_lines)
    has_pull_sha = "github.event.pull_request.head.sha" in fold_text
    has_or_ref = "|| github.ref" in fold_text
    return has_pull_sha and has_or_ref


def block_uses_conditional_cancel(block: str) -> bool:
    """True when the `cancel-in-progress:` folded scalar is conditional
    on ``github.event_name == 'push'`` AND on the ref being
    ``refs/heads/main`` or ``refs/heads/develop``."""
    cancel_m = _CANCEL_LINE_RE.search(block)
    if not cancel_m:
        return False
    start = cancel_m.start()
    fold_lines = block[start:].split("\n", 9)[:9]
    fold_text = "\n".join(fold_lines)
    has_push_cond = "github.event_name == 'push'" in fold_text
    has_main_develop_cond = (
        "contains('refs/heads/main,refs/heads/develop'" in fold_text
    )
    return has_push_cond and has_main_develop_cond


def check_workflow(path: Path) -> list[str]:
    """Return a list of drift findings for the workflow at ``path``.
    Empty list means the workflow is compliant.
    """
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(REPO_ROOT).as_posix()
    findings: list[str] = []

    block = extract_block(text)
    if block is None:
        findings.append(f"{rel}: missing top-level `concurrency:` block")
        return findings

    if not block_uses_per_sha_group(block):
        findings.append(
            f"{rel}: `concurrency.group` does not reference "
            "`github.event.pull_request.head.sha` and `|| github.ref` "
            "(ADR-0015 per-`head_sha` shape). Re-run "
            "`scripts/update_concurrency_keys.py`."
        )
    if not block_uses_conditional_cancel(block):
        findings.append(
            f"{rel}: `concurrency.cancel-in-progress` is not gated on "
            "`github.event_name == 'push' && "
            "contains('refs/heads/main,refs/heads/develop', github.ref)` "
            "(ADR-0015 §Decision). Re-run "
            "`scripts/update_concurrency_keys.py`."
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        help="Restrict to a single workflow file (e.g. rust-tests.yml).",
    )
    args = parser.parse_args()

    if not WORKFLOWS_DIR.is_dir():
        print(f"ERROR: {WORKFLOWS_DIR} not found", file=sys.stderr)
        return 2

    files = sorted(WORKFLOWS_DIR.glob("*.yml"))
    if args.workflow:
        files = [f for f in files if f.name == args.workflow]
        if not files:
            print(f"ERROR: workflow {args.workflow} not found", file=sys.stderr)
            return 2

    all_findings: list[str] = []
    for path in files:
        findings = check_workflow(path)
        all_findings.extend(findings)

    if all_findings:
        print("ADR-0015 concurrency drift detected:", file=sys.stderr)
        for f in all_findings:
            print(f"  - {f}", file=sys.stderr)
        print(
            f"\n{len(all_findings)} finding(s) across {len(files)} workflow(s).",
            file=sys.stderr,
        )
        print(
            "Run `python3 scripts/update_concurrency_keys.py` to apply the "
            "ADR-0015 template, then re-run this check.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: all {len(files)} workflow(s) carry the ADR-0015 "
        "per-`head_sha` `concurrency:` block."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())