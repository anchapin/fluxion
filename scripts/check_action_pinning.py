#!/usr/bin/env python3
"""Reject mutable GitHub Actions references in workflow files.

Issue #3530 keeps the nightly ASHRAE 140 gauge workflow and future PRs
from re-introducing floating action refs. The check is deliberately narrow:
it reports the named branch, stable, HEAD, and short major-version refs while
leaving the existing, stricter ``check_workflow_pin.py`` gate to enforce the
full SHA-pinning policy.

Usage::

    python3 scripts/check_action_pinning.py
    python3 scripts/check_action_pinning.py --self-test

Exit codes:

    0 -- no forbidden action references were found.
    1 -- one or more forbidden action references were found.
    2 -- the workflow directory could not be scanned.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# These refs are explicitly called out by the issue. Keep the set
# centralized so the scanner and its self-test cannot drift apart.
FORBIDDEN_REFS = frozenset({"stable", "main", "master", "HEAD", "v1", "v2", "v3", "v4"})
_FORBIDDEN_REFS_CASEFOLD = frozenset(ref.casefold() for ref in FORBIDDEN_REFS)
_SHORT_MAJOR_VERSION_RE = re.compile(r"v[1-4]$", re.IGNORECASE)
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

# A workflow step may be a YAML list item or a bare key:
#   - uses: owner/action@ref
#       uses: owner/action@ref
# Capture only the action value; a trailing version comment is not part of
# the resolved ref.
_USES_LINE_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<value>[^\s#]+)")


def _parse_action_reference(value: str) -> str | None:
    """Return an action-shaped value containing an external ``@ref``.

    Values without ``@`` (for example, a local ``./action`` path) are not
    external action refs and are intentionally outside this narrow check.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]

    if "@" not in value:
        return None

    action, _ = value.rsplit("@", 1)
    if "/" not in action:
        return None
    return value


def iter_uses_lines(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, action_ref)`` for active action references."""
    references: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if raw_line.lstrip().startswith("#"):
            continue
        match = _USES_LINE_RE.match(raw_line)
        if not match:
            continue
        action_ref = _parse_action_reference(match.group("value"))
        if action_ref is not None:
            references.append((line_number, action_ref))
    return references


def is_forbidden_ref(ref: str) -> bool:
    """Return whether a ref is one of the explicitly forbidden mutable forms."""
    if "@" in ref:
        ref = ref.rsplit("@", 1)[1]
    if _SHA_RE.fullmatch(ref):
        return False
    return ref.casefold() in _FORBIDDEN_REFS_CASEFOLD or bool(
        _SHORT_MAJOR_VERSION_RE.fullmatch(ref)
    )


def check_workflow(path: Path) -> list[str]:
    """Return one human-readable finding for each forbidden ref in ``path``."""
    text = path.read_text(encoding="utf-8")
    rel = _display_path(path)
    findings: list[str] = []
    for line_number, ref in iter_uses_lines(text):
        if is_forbidden_ref(ref):
            findings.append(
                f"{rel}:{line_number}: uses: {ref} -- forbidden mutable ref; "
                "use a 40-character commit SHA with a trailing version comment"
            )
    return findings


def _display_path(path: Path) -> str:
    """Prefer a repository-relative path while supporting hermetic fixtures."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _workflow_files() -> list[Path]:
    """Return every YAML workflow file, including nested workflow files."""
    if not WORKFLOWS_DIR.is_dir():
        raise FileNotFoundError(f"{WORKFLOWS_DIR} not found")
    files = [
        path
        for suffix in ("*.yml", "*.yaml")
        for path in WORKFLOWS_DIR.rglob(suffix)
        if path.is_file()
    ]
    files.sort()
    if not files:
        raise FileNotFoundError(f"no .yml or .yaml workflows found in {WORKFLOWS_DIR}")
    return files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check GitHub Actions references for forbidden mutable refs."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the deterministic parser and gate self-test without scanning",
    )
    return parser


def _self_test() -> int:
    """Exercise the ref classifier and the violation/exit-code contract."""
    sample = "\n".join(
        (
            "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1",
            "        uses: dtolnay/rust-toolchain@f8be11a05b1d4f3fcebe6410cc16743212b999b0  # 1.98.0",
            "        uses: dtolnay/rust-toolchain@stable",
            "#       uses: actions/checkout@v1",
            "        uses: ./.github/actions/local-action",
        )
    )
    references = iter_uses_lines(sample)
    expected = [
        (1, "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"),
        (2, "dtolnay/rust-toolchain@f8be11a05b1d4f3fcebe6410cc16743212b999b0"),
        (3, "dtolnay/rust-toolchain@stable"),
    ]
    if references != expected:
        print(
            f"SELF-TEST FAIL: expected {expected!r}, got {references!r}",
            file=sys.stderr,
        )
        return 2
    if not is_forbidden_ref("actions/checkout@v1"):
        print("SELF-TEST FAIL: v1 was not rejected", file=sys.stderr)
        return 2
    for ref in FORBIDDEN_REFS:
        if not is_forbidden_ref(ref):
            print(f"SELF-TEST FAIL: {ref!r} was not rejected", file=sys.stderr)
            return 2
    print("SELF-TEST PASS: action reference parsing and forbidden-ref checks passed")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Scan the repository and return the documented gate exit code."""
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else list(argv))
    if args.self_test:
        return _self_test()

    try:
        files = _workflow_files()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    total_refs = 0
    findings: list[str] = []
    for path in files:
        rel = _display_path(path)
        references = iter_uses_lines(path.read_text(encoding="utf-8"))
        total_refs += len(references)
        file_findings = check_workflow(path)
        if file_findings:
            print(f"FAIL: {rel} ({len(references)} action ref(s) checked)")
            for finding in file_findings:
                print(f"  {finding}")
            findings.extend(file_findings)
        else:
            count_label = (
                "no action references"
                if not references
                else f"{len(references)} action ref(s) checked"
            )
            print(f"PASS: {rel} ({count_label})")

    print()
    print(
        f"Scanned {len(files)} workflow file(s); checked {total_refs} action "
        f"ref(s); {len(findings)} violation(s)."
    )
    if findings:
        print("ACTION PINNING FAILED: mutable action refs are not allowed.")
        return 1

    print("ACTION PINNING PASS: no forbidden mutable action refs were found.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
