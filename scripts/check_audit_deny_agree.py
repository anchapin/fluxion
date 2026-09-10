#!/usr/bin/env python3
"""
Audit↔Deny Advisory-Agreement Check for Fluxion.

Cross-checks the advisory-ignore lists of ``.cargo/audit.toml`` (cargo-audit)
and ``deny.toml`` (cargo-deny) against each other so the two supply-chain
gates cannot drift apart silently. See issue #3654 and AGENTS.md:
"Keep ``.cargo/audit.toml`` and ``deny.toml`` advisory exceptions
synchronized."

The two tools have deliberately different unmaintained scopes:

    - ``cargo audit --deny warnings`` enforces ``unmaintained`` for ALL
      crates, including transitive dependencies.
    - ``deny.toml`` sets ``unmaintained = "workspace"``, so transitive
      unmaintained advisories are out of cargo-deny's scope and an ignore
      entry for one there would only emit ``advisory-not-detected`` noise
      (the #2749 cleanup removed exactly that noise).

The gate therefore enforces a scoped agreement contract rather than naive
set equality:

    1. No duplicate RUSTSEC ids within either file's ignore list.
    2. Every ``deny.toml`` ignore MUST also be ignored in
       ``.cargo/audit.toml`` — cargo-audit is the strictly broader gate,
       so a deny-side-only ignore would not protect the cargo-audit run.
    3. Every ``.cargo/audit.toml`` ignore that has no ``deny.toml``
       counterpart MUST be annotated with the machine-readable marker
       ``deny-scope-exempt`` in its entry comment (same line or any
       comment line between the previous entry and this one) — declaring
       it deliberately out of cargo-deny's scope (transitive unmaintained).
       Unmarked audit-only ignores are drift and fail the gate.
    4. The exemption contract in rule 3 is only valid while ``deny.toml``
       declares ``unmaintained = "workspace"``. If the scope changes
       (e.g. to ``"all"``) or disappears, transitive unmaintained becomes
       cargo-deny's business again and every exempt entry fails until it
       is mirrored into ``deny.toml``.

Commented-out entries (``# "RUSTSEC-..."``) are documentation, not
enforcement, and are ignored by the parser.

Usage:
    python3 scripts/check_audit_deny_agree.py
    python3 scripts/check_audit_deny_agree.py --quiet

Exit codes:
    0 — the two advisory-ignore lists agree under the scoped contract.
    1 — drift detected (deny-only entry, unmarked audit-only entry,
        exempt entry under a non-workspace unmaintained scope, or
        duplicate ids within one file).
    2 — script error (a config file is missing or its ``ignore`` block
        cannot be located).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_TOML = REPO_ROOT / ".cargo" / "audit.toml"
DENY_TOML = REPO_ROOT / "deny.toml"

# Machine-readable marker declaring an audit.toml ignore entry as
# deliberately absent from deny.toml (transitive-unmaintained class,
# out of cargo-deny's `unmaintained = "workspace"` scope). See rule 3.
EXEMPT_MARKER = "deny-scope-exempt"

# The only deny.toml `unmaintained` scope under which rule-3 exemptions
# are meaningful. Any other value (or a missing key) invalidates them.
EXEMPT_ALLOWED_SCOPE = "workspace"

ENTRY_RE = re.compile(r'^\s*"(RUSTSEC-\d{4}-\d{4})"')


def _find_ignore_block_bounds(lines: list[str]) -> tuple[int, int] | None:
    """Return (start, end) line indices of the ``ignore = [...]`` block.

    ``start`` is the line with ``ignore = [`` (an inline ``ignore = []``
    yields start == end); ``end`` is the matching closing ``]`` line.
    Returns ``None`` when no such block exists. Only the FIRST block is
    considered — in both config files the ``[advisories]`` table is the
    only one carrying an ``ignore`` key.
    """
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*ignore\s*=\s*\[", line):
            start = i
            break
    if start is None:
        return None
    # Inline empty list: `ignore = []` closes on the same line.
    if "]" in lines[start].split("[", 1)[1]:
        return (start, start)
    for j in range(start + 1, len(lines)):
        if lines[j].strip().startswith("]"):
            return (start, j)
    return None


def parse_ignore_entries(text: str) -> list[dict]:
    """Parse the ``[advisories] ignore`` list into per-entry records.

    Each record contains:

        - ``id``: the ``RUSTSEC-YYYY-NNNN`` identifier.
        - ``line``: 1-indexed line number of the entry.
        - ``exempt``: True when the entry carries the
          ``deny-scope-exempt`` marker in its own trailing comment or in
          the comment lines between the previous entry and this one.

    Commented-out entries are skipped: only lines whose first non-space
    character is a quote count as active ignores.
    """
    lines = text.splitlines()
    bounds = _find_ignore_block_bounds(lines)
    if bounds is None:
        raise ValueError("no `ignore = [...]` block found")
    start, end = bounds

    entries: list[dict] = []
    pending_comments: list[str] = []

    for k in range(start + 1, end + 1):
        line = lines[k]
        stripped = line.strip()
        if stripped.startswith("#"):
            pending_comments.append(stripped)
            continue

        m = ENTRY_RE.match(line)
        if m:
            trailing = line.split("#", 1)[1] if "#" in line else ""
            exempt = EXEMPT_MARKER in trailing or any(
                EXEMPT_MARKER in c for c in pending_comments
            )
            entries.append(
                {
                    "id": m.group(1),
                    "line": k + 1,
                    "exempt": exempt,
                }
            )
            pending_comments = []

    return entries


def parse_unmaintained_scope(text: str) -> str | None:
    """Return deny.toml's ``unmaintained`` scope value, or ``None``.

    Accepts cargo-deny 0.14+ scope strings (``all`` | ``workspace`` |
    ``transitive`` | ``none``). A missing key returns ``None`` — the
    caller treats that as "exemptions not authorised".
    """
    m = re.search(r'^\s*unmaintained\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else None


def _duplicates(entries: list[dict], label: str) -> list[str]:
    """Return one problem string per duplicated advisory id in ``entries``."""
    seen: dict[str, int] = {}
    problems: list[str] = []
    for e in entries:
        rid = e["id"]
        if rid in seen:
            problems.append(
                f"  - {rid} is ignored more than once in {label} "
                f"(lines {seen[rid]} and {e['line']}) — duplicate entries "
                f"hide copy-paste drift between the two files."
            )
        else:
            seen[rid] = e["line"]
    return problems


def _load_entries(path: Path, label: str) -> list[dict]:
    """Read + parse one config's ignore list; exit 2 on any failure."""
    if not path.exists():
        print(f"ERROR: {label} not found at {path}", file=sys.stderr)
        sys.exit(2)
    text = path.read_text(encoding="utf-8")
    try:
        return parse_ignore_entries(text)
    except ValueError as exc:
        print(f"ERROR: cannot parse {label}: {exc}", file=sys.stderr)
        sys.exit(2)


def collect_drift(
    audit_entries: list[dict],
    deny_entries: list[dict],
    unmaintained_scope: str | None,
) -> list[str]:
    """Evaluate the four contract rules; return human-readable problems."""
    problems: list[str] = []

    problems.extend(_duplicates(audit_entries, ".cargo/audit.toml"))
    problems.extend(_duplicates(deny_entries, "deny.toml"))

    audit_by_id = {e["id"]: e for e in audit_entries}
    deny_ids = {e["id"] for e in deny_entries}

    # Rule 2: deny-side ignores must be mirrored in audit.toml.
    for e in deny_entries:
        if e["id"] not in audit_by_id:
            problems.append(
                f"  - {e['id']} is ignored in deny.toml (line {e['line']}) "
                f"but MISSING from .cargo/audit.toml — cargo-audit runs "
                f"with `--deny warnings` and would fail on it. Mirror the "
                f"entry into .cargo/audit.toml's ignore list (keep both "
                f"entries; merge the reasons) or drop it from deny.toml."
            )

    # Rule 3 + 4: audit-side ignores must be mirrored or marked exempt,
    # and exemptions require deny.toml's workspace-scoped unmaintained.
    unmarked: list[dict] = [
        e for e in audit_entries if e["id"] not in deny_ids and not e["exempt"]
    ]
    exempt: list[dict] = [
        e for e in audit_entries if e["id"] not in deny_ids and e["exempt"]
    ]

    for e in unmarked:
        problems.append(
            f"  - {e['id']} is ignored in .cargo/audit.toml "
            f"(line {e['line']}) but has no deny.toml counterpart and no "
            f"'{EXEMPT_MARKER}' marker — either mirror the entry into "
            f"deny.toml's [advisories] ignore list, or annotate its "
            f"comment with '{EXEMPT_MARKER} (transitive-unmaintained)' "
            f"if the advisory is out of cargo-deny's unmaintained scope."
        )

    if exempt and unmaintained_scope != EXEMPT_ALLOWED_SCOPE:
        shown = unmaintained_scope if unmaintained_scope is not None else "<unset>"
        ids = ", ".join(e["id"] for e in exempt)
        problems.append(
            f"  - deny.toml declares `unmaintained = \"{shown}\"` (expected "
            f"\"{EXEMPT_ALLOWED_SCOPE}\"), so transitive unmaintained "
            f"advisories are (or may be) in cargo-deny's scope — the "
            f"'{EXEMPT_MARKER}' contract no longer holds for: {ids}. "
            f"Mirror these entries into deny.toml or restore "
            f"`unmaintained = \"{EXEMPT_ALLOWED_SCOPE}\"`."
        )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-file parse summary; print only the verdict.",
    )
    args = parser.parse_args(argv)

    audit_entries = _load_entries(AUDIT_TOML, ".cargo/audit.toml")
    deny_entries = _load_entries(DENY_TOML, "deny.toml")
    deny_text = DENY_TOML.read_text(encoding="utf-8")
    unmaintained_scope = parse_unmaintained_scope(deny_text)

    if not args.quiet:
        print("=== Fluxion Audit↔Deny Advisory-Agreement Check (Issue #3654) ===")
        print(f"Repo: {REPO_ROOT}")
        print()
        print(
            f"Parsed {len(audit_entries)} ignore entries from "
            f".cargo/audit.toml "
            f"({sum(1 for e in audit_entries if e['exempt'])} "
            f"{EXEMPT_MARKER})."
        )
        print(
            f"Parsed {len(deny_entries)} ignore entries from deny.toml "
            f"(unmaintained scope: "
            f'{unmaintained_scope if unmaintained_scope is not None else "<unset>"})'
        )
        print()

    problems = collect_drift(audit_entries, deny_entries, unmaintained_scope)

    if problems:
        print(f"FAIL: {len(problems)} advisory-ignore drift problem(s):")
        for p in problems:
            print(p)
        print()
        print(
            "AGENTS.md requires `.cargo/audit.toml` and `deny.toml` "
            "advisory exceptions to stay synchronized (issue #3654). Keep "
            "BOTH entries when reconciling — never silently drop an "
            "exception."
        )
        return 1

    if not args.quiet:
        print()
    print(
        "PASS: .cargo/audit.toml and deny.toml advisory-ignore lists agree "
        "under the scoped contract (deny ⊆ audit; audit-only entries are "
        f"'{EXEMPT_MARKER}' while deny.toml scopes unmaintained to "
        f'"{EXEMPT_ALLOWED_SCOPE}").'
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
