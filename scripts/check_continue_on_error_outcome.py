#!/usr/bin/env python3
"""Reject `continue-on-error` + `steps.<id>.outcome` re-raise anti-pattern.

Issue #4159: a step with ``continue-on-error: true`` ALWAYS reports
``steps.<id>.outcome == 'success'`` — the outcome reflects the
continue-on-error handling, not the step's real result. Only
``steps.<id>.conclusion`` carries the actual ``'success'`` / ``'failure'``.
Any downstream ``if:`` / ``run:`` expression that tests
``steps.<id>.outcome`` on such a step is dead logic: the terminal
"fail if X failed" re-raise can never fire, and the workflow goes green
while the gate it was guarding is red (the ASHRAE 140 release-gate
re-raise in ``ashrae_140_validation.yml`` was exactly this bug).

This gate fails any workflow in ``.github/workflows/*.yml`` that
references ``steps.<id>.outcome`` where the step with ``id: <id>`` has
``continue-on-error: true``. References to ``.outcome`` on steps WITHOUT
``continue-on-error`` are left alone (``.outcome`` is reliable there —
e.g. ``mutation-testing.yml``'s ``steps.mutants.outcome != 'skipped'``),
as are ``.conclusion`` references, which are the correct form.

Usage::

    python3 scripts/check_continue_on_error_outcome.py
    python3 scripts/check_continue_on_error_outcome.py --self-test

Exit codes:

    0 -- no masked-step `.outcome` references found
    1 -- one or more anti-pattern references found
    2 -- script error (e.g. ``.github/workflows/`` missing)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# `id: <name>` — the step identifier other steps reference.
_ID_RE = re.compile(r"^\s*id:\s*([A-Za-z0-9_][A-Za-z0-9_-]*)\s*$")
# `continue-on-error: true` (YAML 1.1 booleans; the repo writes `true`;
# a trailing `#` comment is allowed, e.g. pgo-nightly.yml's harness step).
_COE_RE = re.compile(r"^\s*continue-on-error:\s*true(\s+#.*)?\s*$", re.IGNORECASE)
# A step list item: `      - name: ...` / `      - uses: ...` / `      - run: ...`.
_STEP_ITEM_RE = re.compile(r"^(\s*)-\s+")
# `steps.<id>.outcome` anywhere in an expression.
_OUTCOME_REF_RE = re.compile(r"steps\.([A-Za-z0-9_][A-Za-z0-9_-]*)\.outcome\b")


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _masked_step_ids(lines: list[str]) -> set[str]:
    """Return the ids of steps that set ``continue-on-error: true``.

    For each ``continue-on-error: true`` line, the owning step is the
    nearest enclosing ``- `` list item: walk backwards to the first
    ``- `` line at a smaller indent, then collect the ``id:`` declared
    on the lines between that item and the ``continue-on-error:`` line
    (an ``id:`` key sits at the step-mapping indent, deeper than the
    item's ``- ``).
    """
    masked: set[str] = set()
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        if not _COE_RE.match(line):
            continue
        coe_indent = _indent_of(line)
        # Find the enclosing step item: nearest preceding `- ` at a
        # smaller indent.
        item_idx: int | None = None
        for back in range(idx - 1, -1, -1):
            prev = lines[back]
            if not prev.strip() or prev.lstrip().startswith("#"):
                continue
            m = _STEP_ITEM_RE.match(prev)
            if m and _indent_of(prev) < coe_indent:
                item_idx = back
                break
            if _indent_of(prev) < coe_indent:
                break  # dedented past the step without finding its item
        if item_idx is None:
            continue
        # The `id:` key lives between the item and the coe line.
        for fwd in range(item_idx, idx):
            m = _ID_RE.match(lines[fwd])
            if m:
                masked.add(m.group(1))
                break
    return masked


def check_workflow(path: Path) -> list[str]:
    """Return drift findings for the workflow at ``path`` (empty = clean)."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rel = path.relative_to(REPO_ROOT).as_posix()
    masked = _masked_step_ids(lines)
    if not masked:
        return []
    findings: list[str] = []
    for idx, line in enumerate(lines, start=1):
        if line.lstrip().startswith("#"):
            continue
        for m in _OUTCOME_REF_RE.finditer(line):
            step_id = m.group(1)
            if step_id in masked:
                findings.append(
                    f"{rel}:{idx}: `steps.{step_id}.outcome` is read but "
                    f"step `{step_id}` sets `continue-on-error: true`, so "
                    f"`.outcome` is always 'success' here — the check can "
                    f"never fire. Use `steps.{step_id}.conclusion` "
                    f"(Issue #4159)."
                )
    return findings


def _workflow_files() -> list[Path]:
    if not WORKFLOWS_DIR.is_dir():
        raise FileNotFoundError(f"{WORKFLOWS_DIR} not found")
    files = sorted(WORKFLOWS_DIR.glob("*.yml"))
    if not files:
        raise FileNotFoundError(f"no .yml workflows found in {WORKFLOWS_DIR}")
    return files


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else sys.argv[1:]
    if "--self-test" in argv:
        return _self_test()

    try:
        files = _workflow_files()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    findings: list[str] = []
    for path in files:
        findings.extend(check_workflow(path))

    if findings:
        print(f"FAIL: {len(findings)} continue-on-error `.outcome` "
              f"reference(s):")
        for f in findings:
            print(f"  {f}")
        return 1
    print(f"OK: {len(files)} workflow(s) scanned; no masked-step "
          f"`.outcome` references.")
    return 0


def _self_test() -> int:
    """Hermetic regression test for the detector."""
    import tempfile

    masked_step = (
        "jobs:\n"
        "  gate:\n"
        "    steps:\n"
        "      - name: Check\n"
        "        id: check\n"
        "        continue-on-error: true\n"
        "        run: ./check.sh\n"
        "      - name: Re-raise\n"
        "        if: steps.check.outcome == 'failure'\n"
        "        run: exit 1\n"
    )
    fixed_step = masked_step.replace(
        "steps.check.outcome == 'failure'",
        "steps.check.conclusion == 'failure'",
    )
    unmasked_step = masked_step.replace("        continue-on-error: true\n", "")
    cases = [
        ("masked .outcome fails", masked_step, True),
        ("conclusion passes", fixed_step, False),
        ("unmasked .outcome passes", unmasked_step, False),
        ("no steps clean", "name: X\non: push\njobs: {}\n", False),
    ]
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, yaml_text, expect_findings in cases:
            path = Path(tmp) / "wf.yml"
            path.write_text(yaml_text, encoding="utf-8")
            global REPO_ROOT  # noqa: PLW0603
            old_root = REPO_ROOT
            REPO_ROOT = Path(tmp)
            try:
                findings = check_workflow(path)
            finally:
                REPO_ROOT = old_root
            if bool(findings) != expect_findings:
                print(f"FAIL: case {name!r}: expected "
                      f"findings={expect_findings}, got {findings}",
                      file=sys.stderr)
                failures += 1
    if failures:
        return 1
    print(f"PASS: self-test ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
