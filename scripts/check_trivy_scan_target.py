#!/usr/bin/env python3
"""Reject Trivy scans that audit the repo filesystem instead of the image.

Issue #4186: the ``security`` job in ``.github/workflows/docker.yml`` used
to run ``aquasecurity/trivy-action`` with ``scan-type: 'fs'`` + ``scan-ref:
'.'`` — auditing the checkout's Python/JS dependencies while the shipped
Debian runtime layer went completely unscanned, with no ``exit-code`` so
even the filesystem findings could not fail the build. The workflow now
scans the built image (``input:`` tarball from ``build-and-test``) with
``exit-code: '1'``.

This gate keeps the regression from recurring: it fails any
``aquasecurity/trivy-action`` step in ``.github/workflows/*.yml`` that
sets ``scan-type: 'fs'`` (quoted or not) unless the step carries a
documented justification — a comment line containing
``trivy-fs-justified:`` between the step's first line and its ``uses:``
line, naming why a filesystem scan is the right target there.

Usage::

    python3 scripts/check_trivy_scan_target.py
    python3 scripts/check_trivy_scan_target.py --self-test

Exit codes:

    0 -- every trivy-action step scans an image (or carries justification)
    1 -- one or more unjustified ``scan-type: 'fs'`` steps found
    2 -- script error (e.g. ``.github/workflows/`` missing)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Matches the `uses:` line of the Trivy action (any pinned ref).
_TRIVY_USES_RE = re.compile(
    r"^\s*(?:-\s*)?uses:\s*aquasecurity/trivy-action@\S+"
)

# `scan-type:` inside the step's `with:` block, quoted or bare.
_SCAN_TYPE_RE = re.compile(
    r"^\s*scan-type:\s*['\"]?([A-Za-z]+)['\"]?\s*$"
)

# A step list item begins a new step; anything indented deeper belongs to
# the current step. The `uses:` line itself may be the list item.
_STEP_ITEM_RE = re.compile(r"^(\s*)-\s+(?:name|uses|run|id)\s*:")

# Justification marker: a `#` comment containing this token on any line
# between the step's first line and its trivy `uses:` line.
_JUSTIFICATION_TOKEN = "trivy-fs-justified:"


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _step_start_index(lines: list[str], uses_idx: int) -> int:
    """Return the index of the first line of the step containing ``uses_idx``.

    Finds the step's ``- `` list item by walking backwards: blank lines,
    comments, and deeper-indented body lines are skipped; a same-indent
    non-item line (e.g. an ``id:`` sibling key) is stepped over; the walk
    stops at the first list item (the step start) or when it dedents past
    the step. Afterwards the step's leading comment block is included so
    a ``trivy-fs-justified:`` comment placed above the step is in range.
    """
    uses_indent = _indent_of(lines[uses_idx])
    step_item_idx: int | None = None
    if _STEP_ITEM_RE.match(lines[uses_idx]):
        # The `uses:` line is itself the step's list item (`- uses:`).
        step_item_idx = uses_idx
    else:
        for i in range(uses_idx - 1, -1, -1):
            line = lines[i]
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = _indent_of(line)
            if indent > uses_indent:
                continue
            if _STEP_ITEM_RE.match(line):
                step_item_idx = i
                break
            if indent == uses_indent:
                continue  # sibling key inside the step (e.g. `id:`)
            break  # dedented past the step without finding its list item
    start = step_item_idx if step_item_idx is not None else uses_idx
    # Include the step's leading comment block (justification lives here).
    while start > 0:
        prev = lines[start - 1].strip()
        if not prev or prev.startswith("#"):
            start -= 1
            continue
        break
    return start


def _scan_type_of_step(lines: list[str], uses_idx: int) -> str | None:
    """Return the ``scan-type:`` value for the trivy step, or ``None``.

    Scans forward through the step body — everything until the next step
    list item or a dedent out of the step. ``with:`` sits at the same
    indent as ``uses:`` (both are keys of the step mapping), so the scan
    must NOT stop there; it stops at a new ``- `` item at or above the
    ``uses:`` indent, or any line dedented below it.
    """
    uses_indent = _indent_of(lines[uses_idx])
    for line in lines[uses_idx + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = _indent_of(line)
        if indent < uses_indent:
            break
        if indent <= uses_indent and _STEP_ITEM_RE.match(line):
            break
        m = _SCAN_TYPE_RE.match(line)
        if m:
            return m.group(1).lower()
    return None


def _has_justification(lines: list[str], start_idx: int, uses_idx: int) -> bool:
    """True if a ``trivy-fs-justified:`` comment sits above the ``uses:`` line."""
    for line in lines[start_idx:uses_idx]:
        stripped = line.strip()
        if stripped.startswith("#") and _JUSTIFICATION_TOKEN in stripped:
            return True
    return False


def check_workflow(path: Path) -> list[str]:
    """Return drift findings for the workflow at ``path`` (empty = clean)."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rel = path.relative_to(REPO_ROOT).as_posix()
    findings: list[str] = []
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        if not _TRIVY_USES_RE.match(line):
            continue
        scan_type = _scan_type_of_step(lines, idx)
        if scan_type != "fs":
            continue
        start_idx = _step_start_index(lines, idx)
        if _has_justification(lines, start_idx, idx):
            continue
        findings.append(
            f"{rel}:{idx + 1}: trivy-action uses scan-type 'fs' (scans the "
            f"repo checkout, not the shipped image) without a "
            f"`{_JUSTIFICATION_TOKEN}` justification comment. "
            f"Scan the built image via `input:` + `scan-type: 'image'` "
            f"(see .github/workflows/docker.yml), or document why a "
            f"filesystem scan is correct here."
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
        print(f"FAIL: {len(findings)} unjustified Trivy filesystem scan(s):")
        for f in findings:
            print(f"  {f}")
        return 1
    print(f"OK: {len(files)} workflow(s) scanned; no unjustified "
          f"Trivy filesystem scans.")
    return 0


def _self_test() -> int:
    """Hermetic regression test: fs-without-justification must fail."""
    import tempfile

    cases = [
        # (name, yaml, expect_findings)
        (
            "image scan passes",
            "steps:\n"
            "  - name: Run Trivy vulnerability scanner\n"
            "    uses: aquasecurity/trivy-action@abc123\n"
            "    with:\n"
            "      input: /tmp/image.tar\n"
            "      scan-type: 'image'\n"
            "      exit-code: '1'\n",
            False,
        ),
        (
            "bare fs scan fails",
            "steps:\n"
            "  - name: Run Trivy vulnerability scanner\n"
            "    uses: aquasecurity/trivy-action@abc123\n"
            "    with:\n"
            "      scan-type: 'fs'\n"
            "      scan-ref: '.'\n",
            True,
        ),
        (
            "quoted fs scan fails",
            "steps:\n"
            "  - uses: aquasecurity/trivy-action@abc123\n"
            "    with:\n"
            '      scan-type: "fs"\n',
            True,
        ),
        (
            "justified fs scan passes",
            "steps:\n"
            "  # trivy-fs-justified: lockfile audit for the docs site;\n"
            "  # the image scan covers the runtime layer separately.\n"
            "  - name: Run Trivy vulnerability scanner\n"
            "    uses: aquasecurity/trivy-action@abc123\n"
            "    with:\n"
            "      scan-type: 'fs'\n",
            False,
        ),
        (
            "no scan-type passes",
            "steps:\n"
            "  - uses: aquasecurity/trivy-action@abc123\n"
            "    with:\n"
            "      image-ref: 'myimage:tag'\n",
            False,
        ),
    ]
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, yaml_text, expect_findings in cases:
            path = Path(tmp) / "wf.yml"
            path.write_text(yaml_text, encoding="utf-8")
            # check_workflow() is relative to REPO_ROOT; bypass by
            # monkeypatching the module constant for the hermetic run.
            global REPO_ROOT  # noqa: PLW0603
            old_root = REPO_ROOT
            REPO_ROOT = Path(tmp)
            try:
                findings = check_workflow(path)
            finally:
                REPO_ROOT = old_root
            got = bool(findings)
            if got != expect_findings:
                print(f"FAIL: case {name!r}: expected findings={expect_findings}, "
                      f"got {findings}", file=sys.stderr)
                failures += 1
    if failures:
        return 1
    print(f"PASS: self-test ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
