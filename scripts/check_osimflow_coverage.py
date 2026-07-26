#!/usr/bin/env python3
# scripts/check_osimflow_coverage.py
#
# Enforce per-file line-coverage thresholds for the OSimFlow Python
# orchestration suite (Issue #1847).
#
# This checker was extracted from the inline Python step that lived in
# ``.github/workflows/python-tests.yml`` (PR #1849).  That inline step had a
# path-normalization bug: coverage.py records filenames relative to the
# ``--cov=scripts`` source root (e.g. ``cloud_campaign_manager.py``), but the
# threshold table keyed on canonical repo paths (``scripts/cloud_campaign_manager.py``),
# so no ``<class>`` element ever matched and every target was reported as
# "no coverage data" — turning the OSimFlow pytest job red on every PR
# (Issue #1864).
#
# Exit codes:
#   0 — every target meets its threshold
#   1 — one or more targets are missing or below threshold
#
# Usage:
#   python scripts/check_osimflow_coverage.py [path/to/coverage.xml]
#
# The default coverage XML path is ``scripts/ci/coverage.xml``, matching the
# ``--cov-report=xml`` argument in ``python-tests.yml``.

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

DEFAULT_COVERAGE_XML = os.path.join("scripts", "ci", "coverage.xml")

# Canonical repository paths → minimum line-coverage percent.
TARGETS: dict[str, float] = {
    "scripts/cloud_campaign_manager.py": 60.0,
    "scripts/autonomous_parameter_sweep.py": 60.0,
    "scripts/ashrae_benchmark_harness.py": 60.0,
}


@dataclass
class CoverageResult:
    """Outcome of evaluating one coverage target."""

    path: str
    lines_valid: int
    lines_covered: int
    threshold: float
    found: bool

    @property
    def percent(self) -> float:
        if self.lines_valid == 0:
            return 0.0
        return (self.lines_covered / self.lines_valid) * 100.0

    @property
    def passed(self) -> bool:
        return self.found and self.lines_valid > 0 and self.percent >= self.threshold

    @property
    def reason(self) -> str:
        if not self.found:
            return "no coverage data"
        if self.lines_valid == 0:
            return "no measurable lines"
        return "below threshold"


def normalize_filename(filename: str) -> str:
    """Normalize a coverage.py ``filename`` attribute to a canonical repo path.

    coverage.py emits filenames relative to its ``--cov`` source root. With
    ``--cov=scripts`` the XML stores bare names such as
    ``cloud_campaign_manager.py``; with ``--cov=.`` it stores
    ``scripts/cloud_campaign_manager.py``; third-party / installed packages
    may appear as ``.../site-packages/foo/bar.py``. This helper collapses all
    of those forms onto the canonical ``scripts/<basename>`` key used by
    :data:`TARGETS`.

    The normalization is intentionally lenient about directory prefixes and
    path separators (Windows ``\\`` is tolerated) but strict about the
    *basename* — only files whose basename matches a known target are mapped.
    """

    # Strip any ``.../site-packages/`` prefix and use the trailing segment.
    tail = filename.replace("\\", "/").split("site-packages/")[-1]
    # Drop leading ``./`` that some coverage.py versions emit.
    tail = tail.lstrip("./")
    basename = tail.rsplit("/", 1)[-1]
    return f"scripts/{basename}"


def _class_line_counts(cls: ET.Element) -> tuple[int, int]:
    """Return ``(lines_valid, lines_covered)`` for one ``<class>`` element.

    Cobertura XML comes in two flavours depending on the coverage.py version:

    * **Summary attributes** — ``lines-valid`` / ``lines-covered`` on the
      ``<class>`` element (older coverage.py and some CI tools).
    * **Per-line records only** — coverage.py >= 6 omits the summary
      attributes on ``<class>`` and emits a ``<lines>`` child containing one
      ``<line number="N" hits="H"/>`` per executable line. The summary lives
      only on the root ``<coverage>`` element.

    This helper prefers the summary attributes (a single attribute read) and
    falls back to counting ``<line>`` children so both shapes work.
    """

    lv_attr = cls.get("lines-valid")
    lc_attr = cls.get("lines-covered")
    if lv_attr is not None and lc_attr is not None:
        try:
            return int(lv_attr), int(lc_attr)
        except ValueError:
            pass  # fall through to per-line counting

    lines = cls.findall("./lines/line")
    valid = len(lines)
    covered = sum(1 for ln in lines if int(ln.get("hits", "0")) > 0)
    return valid, covered


def evaluate_coverage(
    root: ET.Element, targets: dict[str, float] | None = None
) -> list[CoverageResult]:
    """Evaluate per-file coverage against ``targets`` from a parsed Cobertura XML root.

    Aggregates line counts across every ``<class>`` whose normalized filename
    matches a target, so that split-class or partial-filename reports are
    summed correctly. Works with both Cobertura summary-attribute and
    per-line-record XML shapes (see :func:`_class_line_counts`).
    """

    targets = targets if targets is not None else TARGETS
    aggregated: dict[str, tuple[int, int]] = {}
    seen: set[str] = set()

    for cls in root.findall(".//class"):
        filename = cls.get("filename", "")
        key = normalize_filename(filename)
        if key not in targets:
            continue
        seen.add(key)
        lines_valid, lines_covered = _class_line_counts(cls)
        cur_v, cur_c = aggregated.get(key, (0, 0))
        aggregated[key] = (cur_v + lines_valid, cur_c + lines_covered)

    results: list[CoverageResult] = []
    for path, threshold in targets.items():
        valid, covered = aggregated.get(path, (0, 0))
        results.append(
            CoverageResult(
                path=path,
                lines_valid=valid,
                lines_covered=covered,
                threshold=threshold,
                found=path in seen,
            )
        )
    return results


def report(results: list[CoverageResult]) -> bool:
    """Print a per-target report and return ``True`` if all targets passed."""

    all_passed = True
    for r in results:
        status = "OK  " if r.passed else "FAIL"
        print(
            f"{status} {r.path}: {r.percent:.2f}% "
            f"(threshold {r.threshold:.0f}%, "
            f"{r.lines_covered}/{r.lines_valid} lines)"
        )
        if not r.passed:
            all_passed = False

    failures = [r for r in results if not r.passed]
    if failures:
        print("\n::error::OSimFlow coverage threshold failures:")
        for r in failures:
            print(
                f"::error::  {r.path}: {r.percent:.2f}% "
                f"< {r.threshold:.0f}% — {r.reason}"
            )
    return all_passed


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    coverage_xml = argv[0] if argv else DEFAULT_COVERAGE_XML

    try:
        root = ET.parse(coverage_xml).getroot()
    except FileNotFoundError:
        print(f"::error::coverage XML not found: {coverage_xml}")
        return 1
    except ET.ParseError as exc:
        print(f"::error::failed to parse {coverage_xml}: {exc}")
        return 1

    results = evaluate_coverage(root)
    return 0 if report(results) else 1


if __name__ == "__main__":
    sys.exit(main())
