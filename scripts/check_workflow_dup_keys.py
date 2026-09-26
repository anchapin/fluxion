#!/usr/bin/env python3
"""
CI hygiene guard: fail on duplicate YAML mapping keys in
``.github/workflows/*.yml``.

Issue #4068: PR #4018's phase-gating mechanically inserted
``needs: precheck`` / ``if: needs.precheck.outputs.should_run == 'true'``
immediately after each job's ``name:`` line. In four jobs across three
workflows (``fast_math_check.yml`` ``compare`` + ``fast-math-gh``,
``ashrae_validation.yml`` ``validate-fallback``,
``performance_dashboard.yml`` ``published-crate-bench``) the job already
had a ``needs:`` / ``if:`` key later in its block, producing duplicate
mapping keys. GitHub's workflow parser rejects the whole file at
parse time with the generic banner "This run likely failed because of
a workflow file issue" — zero jobs, zero logs — so every ``push``
event to ``develop`` showed three red X entries with no diagnostics.
Every local gate (``check_concurrency_keys.py``,
``check_workflow_pin.py``, plain YAML parses) passed, because stock
PyYAML silently keeps the *last* duplicate key.

The compliance rule: no mapping in any ``.github/workflows/*.yml``
may define the same key twice. Duplicate keys are almost always a
merge/edit accident (as in #4068), never intentional.

Implementation: parse each workflow with a strict ``yaml.SafeLoader``
subclass whose mapping constructor records every duplicate key (with
both line numbers), and report file, key, and line numbers.

Usage::

    python3 scripts/check_workflow_dup_keys.py             # scan all workflows
    python3 scripts/check_workflow_dup_keys.py --self-test # deterministic self-test

Exit codes:
    0 -- no duplicate keys in any workflow
    1 -- one or more duplicate keys found (drift detected)
    2 -- script error (e.g. ``.github/workflows/`` missing, PyYAML missing)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is in scripts/requirements-test.txt
    sys.stderr.write(
        "ERROR: PyYAML is required for scripts/check_workflow_dup_keys.py. "
        "Install with `pip install pyyaml` (already in "
        "scripts/requirements-test.txt, used by the scripts-tests workflow).\n"
    )
    raise SystemExit(2)


class _StrictLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that records duplicate mapping keys."""


def _strict_mapping_constructor(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
):
    mapping: dict = {}
    first_lines: dict = {}
    for key_node, value_node in node.value:
        # NOTE: keys are hashables, not necessarily strings (YAML 1.1
        # parses ``on:`` as boolean True), so first-lines live in a
        # side table, not in the mapping itself.
        try:
            key = loader.construct_object(key_node, deep=True)
            hash(key)
        except (TypeError, yaml.YAMLError):
            # Unhashable (e.g. a sequence used as a key) or
            # unconstructible key node: fall back to the raw scalar text.
            key = key_node.value
        line = key_node.start_mark.line + 1
        if key in first_lines:
            # Collect rather than raise: one gate run should report every
            # duplicate in the file, not just the first (Issue #4068's
            # fast_math_check.yml had dupes in two different jobs).
            _DUP_FINDINGS.append((repr(key), first_lines[key], line))
        else:
            mapping[key] = loader.construct_object(value_node, deep=deep)
            first_lines[key] = line
    return mapping


# Module-level collector for _strict_mapping_constructor (parse is
# single-threaded; check_workflow drains it per file).
_DUP_FINDINGS: list[tuple[str, int, int]] = []


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _strict_mapping_constructor
)


def check_workflow(path: Path) -> list[str]:
    """Return a list of duplicate-key findings for one workflow file."""
    findings: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{path}: cannot read file: {exc}"]
    _DUP_FINDINGS.clear()
    try:
        list(yaml.load_all(text, Loader=_StrictLoader))
    except yaml.YAMLError as exc:
        findings.append(f"{path.name}: YAML parse error: {exc}")
        return findings
    for key, first_line, dup_line in _DUP_FINDINGS:
        findings.append(
            f"{path.name}: duplicate key {key} "
            f"(first at line {first_line}, redefined at line {dup_line})"
        )
    return findings


def check_all(workflows_dir: Path = WORKFLOWS_DIR) -> list[str]:
    """Scan every ``*.yml`` under ``workflows_dir``; return all findings."""
    if not workflows_dir.is_dir():
        raise SystemExit(f"ERROR: workflows directory not found: {workflows_dir}")
    findings: list[str] = []
    for path in sorted(workflows_dir.glob("*.yml")):
        findings.extend(check_workflow(path))
    return findings


def self_test() -> int:
    """Deterministic self-test: planted duplicates must be caught."""
    import tempfile

    cases = [
        # (name, yaml text, expect_findings)
        (
            "clean",
            "name: x\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n",
            False,
        ),
        (
            "dup-needs",
            "jobs:\n  compare:\n    needs: precheck\n    needs: [a, b]\n",
            True,
        ),
        (
            "dup-if",
            "jobs:\n  b:\n    if: a == 'x'\n    runs-on: ubuntu-latest\n    if: always()\n",
            True,
        ),
        (
            "dup-nested",
            "jobs:\n  b:\n    steps:\n      - name: s\n        with:\n          a: 1\n          a: 2\n",
            True,
        ),
        (
            "same-key-sibling-mappings-ok",
            "jobs:\n  a:\n    runs-on: ubuntu-latest\n  b:\n    runs-on: ubuntu-latest\n",
            False,
        ),
    ]
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for name, text, expect in cases:
            p = tmpdir / f"{name}.yml"
            p.write_text(text, encoding="utf-8")
            found = check_workflow(p)
            ok = bool(found) == expect
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: findings={found}")
            failures += 0 if ok else 1
    print(f"self-test: {'OK' if failures == 0 else f'{failures} FAILURES'}")
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if "--self-test" in args:
        return self_test()
    # NOTE: pass WORKFLOWS_DIR explicitly (not via check_all()'s default
    # arg) so tests can monkeypatch the module constant.
    findings = check_all(WORKFLOWS_DIR)
    if findings:
        print("Duplicate YAML mapping keys in .github/workflows/:")
        for finding in findings:
            print(f"  FAIL: {finding}")
        print(
            "\nDuplicate keys fail GitHub's workflow parser at startup "
            '("This run likely failed because of a workflow file issue") '
            "while stock PyYAML silently keeps the last one — see Issue #4068."
        )
        return 1
    print(f"OK: no duplicate keys in {len(list(WORKFLOWS_DIR.glob('*.yml')))} workflows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
