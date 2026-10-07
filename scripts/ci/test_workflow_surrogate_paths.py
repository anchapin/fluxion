#!/usr/bin/env python3
"""Hermetic check that the surrogate path filters in CI workflows actually
match the surrogate source files (Issue #4161).

GitHub's `paths:` patterns match path *segments*: a bare ``*`` matches any
characters within one segment but never crosses ``/``; ``**`` crosses
segments. This module reimplements that semantics and asserts that

* ``.github/workflows/onnx-integrity.yml`` (pull_request.paths) matches
  ``src/ai/surrogate.rs`` and every file under ``src/ai/surrogate/``;
* ``.github/workflows/surrogate-drift.yml`` (pull_request.paths) also
  matches every file under ``src/ai/surrogate/`` (the historical
  ``**/surrogate*.rs`` patterns alone cannot, since a segment glob never
  reaches ``src/ai/surrogate/integrity.rs``).

Run: ``python3 scripts/ci/test_workflow_surrogate_paths.py`` (exit 0 = pass).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ONNX_WORKFLOW = REPO / ".github/workflows/onnx-integrity.yml"
DRIFT_WORKFLOW = REPO / ".github/workflows/surrogate-drift.yml"

# Files the surrogate loader is decomposed into (src/ai/surrogate.rs:14-17,
# plus the module file itself).
SURROGATE_FILES = [
    "src/ai/surrogate.rs",
    "src/ai/surrogate/integrity.rs",
    "src/ai/surrogate/manager.rs",
    "src/ai/surrogate/metrics.rs",
    "src/ai/surrogate/session_pool.rs",
    # the rename/addition case the acceptance criterion names
    "src/ai/surrogate/some_new_module.rs",
]


def _segment_matches(pattern: str, segment: str) -> bool:
    """GitHub semantics for one path segment: ``*`` is a within-segment glob."""
    regex = "".join(".*" if part == "*" else re.escape(part) for part in pattern.split("*"))
    return re.fullmatch(regex, segment) is not None


def path_matches(path: str, pattern: str) -> bool:
    """Match a repo-relative path against a GitHub workflow paths: pattern."""
    if "**" not in pattern:
        parts = path.split("/")
        pparts = pattern.split("/")
        if len(parts) != len(pparts):
            return False
        return all(_segment_matches(pp, seg) for pp, seg in zip(pparts, parts))
    head, _, tail = pattern.partition("**")
    if head and not path.startswith(head.rstrip("/")):
        return False
    rest = path[len(head.rstrip("/")) :] if head else path
    rest = rest.lstrip("/")
    tail = tail.lstrip("/")
    if not tail:
        return True
    # '**' matches zero or more leading segments.
    while True:
        if path_matches(rest, tail):
            return True
        if "/" not in rest:
            return False
        rest = rest.split("/", 1)[1]


def parse_paths(workflow: Path) -> list[str]:
    """Extract the pull_request.paths list from a workflow file."""
    text = workflow.read_text()
    match = re.search(r"pull_request:.*?paths:\n((?:\s+-\s+'[^']+'\n?)+)", text, re.DOTALL)
    if not match:
        raise AssertionError(f"no pull_request.paths block in {workflow}")
    return re.findall(r"-\s+'([^']+)'", match.group(1))


def filter_covers(patterns: list[str], path: str) -> bool:
    return any(path_matches(path, p) for p in patterns)


def main() -> int:
    onnx_patterns = parse_paths(ONNX_WORKFLOW)
    drift_patterns = parse_paths(DRIFT_WORKFLOW)

    failures: list[str] = []

    for f in SURROGATE_FILES:
        if not filter_covers(onnx_patterns, f):
            failures.append(f"onnx-integrity.yml paths do not match {f}")
        if not filter_covers(drift_patterns, f):
            failures.append(f"surrogate-drift.yml paths do not match {f}")

    # The old surrogate-drift patterns alone must NOT have been a false pass:
    # prove the directory file was previously uncovered.
    if filter_covers(["**/surrogate*.rs", "**/Surrogate*.rs"], "src/ai/surrogate/integrity.rs"):
        failures.append(
            "segment-glob regression: '**/surrogate*.rs' unexpectedly matches "
            "src/ai/surrogate/integrity.rs (test semantics are wrong)"
        )

    # And the guard against silent narrowing: the file-level gate entry and
    # models/reference paths must still be covered.
    for f in ["models/surrogate_zone_thermal.onnx", "models/surrogate_zone_thermal.onnx.sha256"]:
        if not filter_covers(onnx_patterns, f):
            failures.append(f"onnx-integrity.yml no longer matches {f}")
    for f in ["models/foo.onnx", "tests/reference_data/surrogate/x.csv"]:
        if not filter_covers(drift_patterns, f):
            failures.append(f"surrogate-drift.yml no longer matches {f}")

    if failures:
        for line in failures:
            print(f"FAIL: {line}")
        return 1
    print(
        "PASS: onnx-integrity.yml and surrogate-drift.yml path filters cover "
        f"{len(SURROGATE_FILES)} surrogate source files and retain their "
        "original non-surrogate coverage"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --- pytest entry points (Scripts Test Suite) -------------------------------


def test_onnx_integrity_paths_cover_surrogate_tree() -> None:
    patterns = parse_paths(ONNX_WORKFLOW)
    for f in SURROGATE_FILES:
        assert filter_covers(patterns, f), f"onnx-integrity.yml paths do not match {f}"


def test_surrogate_drift_paths_cover_surrogate_tree() -> None:
    patterns = parse_paths(DRIFT_WORKFLOW)
    for f in SURROGATE_FILES:
        assert filter_covers(patterns, f), f"surrogate-drift.yml paths do not match {f}"


def test_segment_glob_does_not_cross_directory() -> None:
    # Regression guard: the historical '**/surrogate*.rs' patterns alone can
    # never reach src/ai/surrogate/integrity.rs — only the added
    # 'src/ai/surrogate/**' entry can.
    assert not filter_covers(["**/surrogate*.rs", "**/Surrogate*.rs"], "src/ai/surrogate/integrity.rs")


def test_original_non_surrogate_coverage_retained() -> None:
    onnx_patterns = parse_paths(ONNX_WORKFLOW)
    drift_patterns = parse_paths(DRIFT_WORKFLOW)
    for f in ["models/surrogate_zone_thermal.onnx", "models/surrogate_zone_thermal.onnx.sha256"]:
        assert filter_covers(onnx_patterns, f), f
    for f in ["models/foo.onnx", "tests/reference_data/surrogate/x.csv"]:
        assert filter_covers(drift_patterns, f), f
