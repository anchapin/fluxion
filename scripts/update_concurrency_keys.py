#!/usr/bin/env python3
"""
Mechanical one-shot script: replace the `concurrency:` block in every
`.github/workflows/*.yml` with the ADR-0015 per-head_sha template.

Issue #3366 / ADR-0015: change the `concurrency.group` key for
`pull_request` events across all 46 workflows to per-`head_sha`,
preserving current behavior on `push` events to develop/main.

Template (verbatim from ADR-0015 §Decision):

    concurrency:
      group: >-
        ${{ github.workflow }}-
        ${{
          github.event_name == 'pull_request' &&
          github.event.pull_request.head.sha
          || github.ref
        }}
      cancel-in-progress: >-
        ${{
          github.event_name == 'push'
          && contains('refs/heads/main,refs/heads/develop', github.ref)
        }}

The script preserves any workflow-specific prefix on `group:` (e.g.
`ashrae-140-${{ github.ref }}` → `ashrae-140-${{ ...head.sha || ref... }}`),
so the *per-workflow* identifier is still unique while the *per-SHA*
separation logic is added underneath.

Idempotent: re-running on an already-updated workflow is a no-op
(matches the exact new template byte-for-byte).

Workflows that have no `concurrency:` block at all get the new template
appended at the canonical location (immediately under `on:`, matching
the placement observed across all 33 workflows that currently declare
one).

Usage:
    python3 scripts/update_concurrency_keys.py [--dry-run] [--workflow <name>]

Exit codes:
    0 — every workflow now carries the ADR-0015 template (or --dry-run
        listed the planned changes without writing).
    1 — one or more workflows could not be updated (unparseable YAML,
        unknown structure, etc.).
    2 — script error (e.g. `.github/workflows/` missing).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# ----------------------------------------------------------------------------
# ADR-0015 template. Two leading-space indent because the block lives under
# the workflow's top-level keys (`on:`, `permissions:`, etc.). Exact bytes
# matter: `scripts/check_concurrency_keys.py` matches against the same string.
# ----------------------------------------------------------------------------
NEW_CONCURRENCY_BLOCK = """concurrency:
  group: >-
    ${{ github.workflow }}-
    ${{
      github.event_name == 'pull_request' &&
      github.event.pull_request.head.sha
      || github.ref
    }}
  cancel-in-progress: >-
    ${{
      github.event_name == 'push'
      && contains('refs/heads/main,refs/heads/develop', github.ref)
    }}"""

# Match the EXISTING concurrency block: `concurrency:` line followed by the
# `group:` and `cancel-in-progress:` keys at 2-space indent. Tolerant of
# inline vs folded scalar styles on either key. Stops at the next 0-indent
# key (or EOF).
#
# Why regex (not PyYAML): GitHub Actions workflows embed `${{ ... }}`
# expressions and the `>` folded scalar marker that PyYAML cannot parse.
# The existing companion script `check_required_checks_sync.py` uses the
# same approach.
_EXISTING_CONCURRENCY_RE = re.compile(
    r"^concurrency:\s*\n"
    r"(?:[ \t]+[^\n]*\n)+"  # one or more indented lines
    r"(?=\n[^\t ]|\Z)",  # lookahead for next 0-indent key or EOF
    re.MULTILINE,
)

# Match a `group:` line with a `${{ ... github.ref ... }}` token so we can
# extract the *prefix* part for preservation. Captures everything before
# the `${{` token.
_GROUP_LINE_RE = re.compile(
    r"^(\s+group:\s*)(.*?)\$\{\{[^}]*github\.ref[^}]*\}\}(.*)$",
    re.MULTILINE,
)

# Match the existing ADR-0015 template byte-for-byte so the script is
# idempotent. The folded scalar (`>-`) is multi-line, so we anchor on
# the distinctive markers (the per-SHA condition with
# `pull_request.head.sha`, the `push && contains(...)` condition) and
# tolerate any folded-scalar interior. Built as a single compiled regex
# for speed.
_NEW_TEMPLATE_INDICATORS = re.compile(
    r"github\.event\.pull_request\.head\.sha"
    r".*?"
    r"contains\('refs/heads/main,refs/heads/develop'",
    re.DOTALL,
)


def is_already_updated(existing_block: str) -> bool:
    """Return True when ``existing_block`` already carries the ADR-0015
    template (i.e. the distinctive ``pull_request.head.sha`` and
    ``contains('refs/heads/main,...')`` markers are both present).

    Cheap marker-match, not byte-for-byte (the literal prefix on the
    first line of the folded ``group:`` scalar varies per workflow).
    """
    return _NEW_TEMPLATE_INDICATORS.search(existing_block) is not None


def extract_group_prefix(existing_block: str) -> str:
    """Return the literal text that should appear before `${{ github.ref }}`
    in the new `group:` line. Falls back to ``${{ github.workflow }}-`` if
    the existing block uses the bare workflow-name form (e.g.
    ``group: ${{ github.workflow }}-${{ github.ref }}``) or if no
    ``group:`` line is matched at all.

    The returned value always ends with a single trailing ``-`` so the
    concatenated ``prefix-${{ ... }}`` expression is well-formed.
    """
    m = _GROUP_LINE_RE.search(existing_block)
    if not m:
        return "${{ github.workflow }}-"
    # Group 1 = "  group: "  (the YAML key).
    # Group 2 = the literal text BEFORE `${{ github.ref }}` — what we
    #           want to preserve (e.g. `ashrae-140-strict-energy-gate-`).
    # Group 3 = anything AFTER the `${{ github.ref }}` token (typically
    #           empty; the original block ended the group key on this
    #           line).
    _, literal, _ = m.groups()
    literal = literal.strip()
    if not literal:
        return "${{ github.workflow }}-"
    # If the literal already includes the workflow-name expression, keep
    # it as-is (it will be replaced by the new folded expression below).
    if literal.endswith("${{ github.workflow }}-"):
        return literal
    if literal.endswith("-"):
        return literal
    return literal + "-"


def build_new_block(prefix_literal: str) -> str:
    """Return the ADR-0015 ``concurrency:`` block, with the literal
    prefix preserved on the first line of the folded ``group:`` scalar.
    """
    new_block_lines = [
        "concurrency:",
        "  group: >-",
        f"    {prefix_literal}",
        "    ${{",
        "      github.event_name == 'pull_request' &&",
        "      github.event.pull_request.head.sha",
        "      || github.ref",
        "    }}",
        "  cancel-in-progress: >-",
        "    ${{",
        "      github.event_name == 'push'",
        "      && contains('refs/heads/main,refs/heads/develop', github.ref)",
        "    }}",
    ]
    return "\n".join(new_block_lines) + "\n"


def replace_concurrency_block(text: str) -> tuple[str, bool]:
    """Replace any existing `concurrency:` block with the ADR-0015 template,
    preserving the workflow-specific literal prefix on `group:`.

    Returns ``(new_text, changed)``. ``changed`` is False when the block
    already matches the new template (idempotent re-run).
    """
    match = _EXISTING_CONCURRENCY_RE.search(text)
    if not match:
        return text, False
    existing = match.group(0)
    if is_already_updated(existing):
        return text, False

    prefix = extract_group_prefix(existing)
    # Build the new block with the preserved literal prefix on `group:`.
    new_block = build_new_block(prefix)
    return text[: match.start()] + new_block + text[match.end() :], True


def insert_concurrency_block(text: str) -> tuple[str, bool]:
    """Insert the ADR-0015 template into a workflow that has no
    `concurrency:` block. Placement: immediately after the `on:` block (the
    canonical location observed in every workflow that already declares
    concurrency). If `on:` is absent, insert after the workflow `name:`.
    """
    if _EXISTING_CONCURRENCY_RE.search(text):
        return text, False

    # Locate the end of the `on:` block. The block ends at the next
    # 0-indent line that does not begin with whitespace (or EOF).
    # Only indented content is part of `on:` — top-level comments must
    # not be matched (otherwise the block "captures" them and the
    # insertion lands below the comments, e.g. after `permissions:`).
    on_block_re = re.compile(
        r"^on:\s*\n((?:[ \t]+[^\n]*\n)+)",
        re.MULTILINE,
    )
    on_match = on_block_re.search(text)
    if on_match:
        # `on_match.end()` is the position immediately after the last
        # indented line of `on:`. Walk past any single blank line so the
        # inserted block sits cleanly between `on:` and the next top-level
        # key.
        insert_at = on_match.end()
        # If the very next line is a comment block (no indented content
        # yet), skip past it so the concurrency block lands *before*
        # the comments, matching the convention of the existing 33
        # workflows.
        while True:
            tail = text[insert_at:]
            cm = re.match(r"\n(# [^\n]*\n)+", tail)
            if not cm:
                break
            insert_at += cm.end()
        # Strip one trailing blank line so we don't end up with three
        # newlines between `on:`/`# comments` and `concurrency:`.
        if text[insert_at:insert_at + 1] == "\n":
            insert_at += 1
    else:
        # Fall back to after the `name:` line.
        name_match = re.search(r"^name:.*\n", text, re.MULTILINE)
        if not name_match:
            return text, False
        insert_at = name_match.end()

    # Insert so we get exactly:
    #   ...
    # on:
    #   ...
    #
    # concurrency:
    #   ...
    #
    # permissions:
    #   ...
    insertion = "\n" + NEW_CONCURRENCY_BLOCK + "\n\n"
    return text[:insert_at] + insertion + text[insert_at:], True


def update_workflow(path: Path, dry_run: bool) -> tuple[bool, str]:
    """Update a single workflow file. Returns ``(changed, message)``.
    ``changed`` is True when the file would be (or was) modified.
    """
    text = path.read_text(encoding="utf-8")
    new_text, replaced = replace_concurrency_block(text)
    if replaced:
        if dry_run:
            return True, "replaced"
        path.write_text(new_text, encoding="utf-8")
        return True, "replaced"

    new_text, inserted = insert_concurrency_block(text)
    if inserted:
        if dry_run:
            return True, "inserted"
        path.write_text(new_text, encoding="utf-8")
        return True, "inserted"

    return False, "already up-to-date"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without writing to disk.",
    )
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

    pattern = "*.yml"
    files = sorted(WORKFLOWS_DIR.glob(pattern))
    if args.workflow:
        files = [f for f in files if f.name == args.workflow]
        if not files:
            print(f"ERROR: workflow {args.workflow} not found", file=sys.stderr)
            return 2

    failures: list[str] = []
    summary = {"replaced": 0, "inserted": 0, "already up-to-date": 0}

    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            changed, msg = update_workflow(path, dry_run=args.dry_run)
        except Exception as exc:  # pragma: no cover - defensive
            failures.append(f"{rel}: {exc}")
            continue
        summary[msg] += 1
        marker = "*" if changed and not args.dry_run else ("?" if changed else " ")
        verb = "would " if args.dry_run else ""
        print(f"{marker} {verb}{msg:20s} {rel}")

    print()
    print(f"Workflows scanned:  {len(files)}")
    print(f"Replaced:           {summary['replaced']}")
    print(f"Inserted (new):     {summary['inserted']}")
    print(f"Already up-to-date: {summary['already up-to-date']}")
    print(f"Failures:           {len(failures)}")
    if failures:
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())