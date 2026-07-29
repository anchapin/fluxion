"""
Tests for ``scripts/check_osimflow_coverage.py`` — Issue #1864.

The OSimFlow pytest workflow failed on every Python version because the
inline coverage gate matched bare ``cloud_campaign_manager.py`` (the form
emitted by ``--cov=scripts``) against canonical ``scripts/cloud_campaign_manager.py``
keys. These tests pin the normalization and aggregation behaviour so the
regression cannot silently return.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from textwrap import dedent

import check_osimflow_coverage as checker
import pytest

# ---------------------------------------------------------------------------
# normalize_filename
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        # coverage.py with --cov=scripts emits bare basenames.
        ("cloud_campaign_manager.py", "scripts/cloud_campaign_manager.py"),
        # coverage.py with --cov=. emits the full repo-relative path.
        ("scripts/cloud_campaign_manager.py", "scripts/cloud_campaign_manager.py"),
        ("./scripts/cloud_campaign_manager.py", "scripts/cloud_campaign_manager.py"),
        # Third-party / installed modules may carry a site-packages prefix.
        (
            "/opt/hostedtoolcache/.../site-packages/scripts/cloud_campaign_manager.py",
            "scripts/cloud_campaign_manager.py",
        ),
        (
            "site-packages/cloud_campaign_manager.py",
            "scripts/cloud_campaign_manager.py",
        ),
        # Windows separators are tolerated.
        ("scripts\\cloud_campaign_manager.py", "scripts/cloud_campaign_manager.py"),
        # Unknown files collapse to scripts/<basename> as well — they simply
        # never match a TARGETS key and are ignored downstream.
        ("foo/bar/state_store.py", "scripts/state_store.py"),
    ],
)
def test_normalize_filename_maps_each_form_to_canonical_key(raw, expected):
    assert checker.normalize_filename(raw) == expected


# ---------------------------------------------------------------------------
# evaluate_coverage
# ---------------------------------------------------------------------------


def _class_elem(
    filename: str,
    lines_valid: int,
    lines_covered: int,
    *,
    with_summary_attrs: bool = True,
) -> ET.Element:
    """Build a ``<class>`` Cobertura element.

    By default emits the ``lines-valid`` / ``lines-covered`` summary
    attributes (older coverage.py shape). Pass ``with_summary_attrs=False``
    to emit only per-line ``<line number hits>`` records, which is what
    coverage.py >= 6 produces and what caused the second half of Issue #1864.
    """

    cls = ET.Element("class")
    cls.set("filename", filename)
    rate = lines_covered / lines_valid if lines_valid else 0.0
    cls.set("line-rate", f"{rate:.4f}")
    if with_summary_attrs:
        cls.set("lines-valid", str(lines_valid))
        cls.set("lines-covered", str(lines_covered))
    else:
        lines_el = ET.SubElement(cls, "lines")
        hit = lines_covered
        for num in range(1, lines_valid + 1):
            ln = ET.SubElement(lines_el, "line")
            ln.set("number", str(num))
            ln.set("hits", "1" if hit > 0 else "0")
            hit = max(0, hit - 1)
    return cls


def _root(*classes: ET.Element, packages: bool = True) -> ET.Element:
    root = ET.Element("coverage")
    if packages:
        pkg = ET.SubElement(root, "package")
        for cls in classes:
            pkg.append(cls)
    else:
        for cls in classes:
            root.append(cls)
    return root


def test_evaluate_coverage_matches_bare_filename_from_cov_scripts():
    # This is the exact regression shape from Issue #1864.
    root = _root(
        _class_elem("cloud_campaign_manager.py", lines_valid=100, lines_covered=68),
        _class_elem("autonomous_parameter_sweep.py", lines_valid=100, lines_covered=84),
        _class_elem("ashrae_benchmark_harness.py", lines_valid=100, lines_covered=84),
    )
    results = checker.evaluate_coverage(root)
    assert {r.path for r in results} == set(checker.TARGETS)
    assert all(r.found for r in results)
    assert all(r.passed for r in results)


def test_evaluate_coverage_matches_prefixed_repo_path():
    root = _root(
        _class_elem(
            "scripts/cloud_campaign_manager.py", lines_valid=100, lines_covered=70
        ),
    )
    results = checker.evaluate_coverage(root)
    ccm = next(r for r in results if r.path == "scripts/cloud_campaign_manager.py")
    assert ccm.found
    assert ccm.percent == pytest.approx(70.0)


def test_evaluate_coverage_strips_site_packages_prefix():
    root = _root(
        _class_elem(
            "/usr/lib/python3.12/site-packages/scripts/cloud_campaign_manager.py",
            lines_valid=50,
            lines_covered=35,
        )
    )
    results = checker.evaluate_coverage(root)
    ccm = next(r for r in results if r.path == "scripts/cloud_campaign_manager.py")
    assert ccm.found
    assert ccm.lines_valid == 50
    assert ccm.lines_covered == 35


def test_evaluate_coverage_aggregates_split_class_entries():
    # Some coverage.py versions emit multiple <class> rows per source file
    # (one per test session / branch report). They must be summed.
    root = _root(
        _class_elem("cloud_campaign_manager.py", lines_valid=60, lines_covered=40),
        _class_elem(
            "scripts/cloud_campaign_manager.py", lines_valid=40, lines_covered=28
        ),
    )
    results = checker.evaluate_coverage(root)
    ccm = next(r for r in results if r.path == "scripts/cloud_campaign_manager.py")
    assert ccm.lines_valid == 100
    assert ccm.lines_covered == 68
    assert ccm.percent == pytest.approx(68.0)


def test_evaluate_coverage_reports_missing_target_as_not_found():
    root = _root(
        _class_elem("cloud_campaign_manager.py", lines_valid=100, lines_covered=68),
    )
    results = checker.evaluate_coverage(root)
    missing = [r for r in results if not r.found]
    paths = {r.path for r in missing}
    assert paths == {
        "scripts/autonomous_parameter_sweep.py",
        "scripts/ashrae_benchmark_harness.py",
    }
    for r in missing:
        assert r.reason == "no coverage data"
        assert not r.passed


def test_evaluate_coverage_marks_below_threshold_as_failed():
    root = _root(
        _class_elem("cloud_campaign_manager.py", lines_valid=100, lines_covered=40),
    )
    results = checker.evaluate_coverage(root)
    ccm = next(r for r in results if r.path == "scripts/cloud_campaign_manager.py")
    assert ccm.found
    assert ccm.percent == pytest.approx(40.0)
    assert not ccm.passed
    assert ccm.reason == "below threshold"


def test_evaluate_coverage_ignores_unrelated_files():
    root = _root(
        _class_elem("state_store.py", lines_valid=100, lines_covered=0),
        _class_elem("cloud_campaign_manager.py", lines_valid=100, lines_covered=68),
        _class_elem("some/random/module.py", lines_valid=10, lines_covered=10),
    )
    results = checker.evaluate_coverage(root)
    ccm = next(r for r in results if r.path == "scripts/cloud_campaign_manager.py")
    assert ccm.lines_valid == 100  # not contaminated by state_store / module


def test_evaluate_coverage_handles_per_line_records_without_summary_attrs():
    # coverage.py >= 6 omits lines-valid/lines-covered on <class> and emits
    # <line number hits> children instead. This is the exact XML shape that
    # masked the second half of the Issue #1864 regression: even with correct
    # path normalization, reading the absent summary attrs yields 0/0.
    root = _root(
        _class_elem(
            "cloud_campaign_manager.py",
            lines_valid=449,
            lines_covered=305,
            with_summary_attrs=False,
        ),
        _class_elem(
            "autonomous_parameter_sweep.py",
            lines_valid=263,
            lines_covered=220,
            with_summary_attrs=False,
        ),
        _class_elem(
            "ashrae_benchmark_harness.py",
            lines_valid=278,
            lines_covered=233,
            with_summary_attrs=False,
        ),
    )
    results = checker.evaluate_coverage(root)
    assert all(r.found for r in results)
    assert all(r.passed for r in results)
    ccm = next(r for r in results if r.path == "scripts/cloud_campaign_manager.py")
    assert ccm.lines_valid == 449
    assert ccm.lines_covered == 305
    assert ccm.percent == pytest.approx(67.93, abs=0.1)


def test_evaluate_coverage_prefers_summary_attrs_when_present():
    # When both shapes are available, the summary attributes win so the
    # checker stays O(attributes) rather than O(lines).
    root = _root(
        _class_elem("cloud_campaign_manager.py", lines_valid=100, lines_covered=68),
    )
    results = checker.evaluate_coverage(root)
    ccm = next(r for r in results if r.path == "scripts/cloud_campaign_manager.py")
    assert ccm.lines_valid == 100
    assert ccm.lines_covered == 68


def test_evaluate_coverage_respects_custom_targets():
    root = _root(
        _class_elem("state_store.py", lines_valid=100, lines_covered=50),
    )
    custom = {"scripts/state_store.py": 60.0}
    results = checker.evaluate_coverage(root, targets=custom)
    assert len(results) == 1
    assert results[0].path == "scripts/state_store.py"
    assert results[0].found
    assert not results[0].passed


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_report_returns_true_and_prints_ok_when_all_pass(capsys):
    results = [
        checker.CoverageResult(
            path="scripts/cloud_campaign_manager.py",
            lines_valid=100,
            lines_covered=68,
            threshold=60.0,
            found=True,
        ),
    ]
    assert checker.report(results) is True
    out = capsys.readouterr().out
    assert "OK  " in out
    assert "scripts/cloud_campaign_manager.py" in out
    assert "68.00%" in out
    assert "::error::" not in out


def test_report_returns_false_and_emits_github_error_annotation(capsys):
    results = [
        checker.CoverageResult(
            path="scripts/cloud_campaign_manager.py",
            lines_valid=100,
            lines_covered=40,
            threshold=60.0,
            found=True,
        ),
        checker.CoverageResult(
            path="scripts/autonomous_parameter_sweep.py",
            lines_valid=0,
            lines_covered=0,
            threshold=60.0,
            found=False,
        ),
    ]
    assert checker.report(results) is False
    out = capsys.readouterr().out
    assert "::error::OSimFlow coverage threshold failures:" in out
    assert "::error::  scripts/cloud_campaign_manager.py: 40.00%" in out
    assert "below threshold" in out
    assert "::error::  scripts/autonomous_parameter_sweep.py:" in out
    assert "no coverage data" in out


# ---------------------------------------------------------------------------
# main / end-to-end
# ---------------------------------------------------------------------------


def _write_xml(tmp_path, body: str):
    xml_path = tmp_path / "coverage.xml"
    xml_path.write_text(dedent(body), encoding="utf-8")
    return xml_path


_PASSING_XML = """
    <coverage>
      <packages>
        <class filename="cloud_campaign_manager.py"
               line-rate="0.68" lines-valid="100" lines-covered="68"/>
        <class filename="autonomous_parameter_sweep.py"
               line-rate="0.84" lines-valid="100" lines-covered="84"/>
        <class filename="ashrae_benchmark_harness.py"
               line-rate="0.84" lines-valid="100" lines-covered="84"/>
      </packages>
    </coverage>
"""


def test_main_returns_zero_on_passing_xml(tmp_path, capsys):
    xml_path = _write_xml(tmp_path, _PASSING_XML)
    rc = checker.main([str(xml_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK  " in out
    assert "::error::" not in out


def test_main_returns_one_when_a_target_is_missing(tmp_path, capsys):
    body = """
        <coverage>
          <packages>
            <class filename="cloud_campaign_manager.py"
                   line-rate="0.68" lines-valid="100" lines-covered="68"/>
          </packages>
        </coverage>
    """
    xml_path = _write_xml(tmp_path, body)
    rc = checker.main([str(xml_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "scripts/autonomous_parameter_sweep.py" in out
    assert "no coverage data" in out


def test_main_returns_one_when_below_threshold(tmp_path, capsys):
    body = """
        <coverage>
          <packages>
            <class filename="cloud_campaign_manager.py"
                   line-rate="0.40" lines-valid="100" lines-covered="40"/>
          </packages>
        </coverage>
    """
    xml_path = _write_xml(tmp_path, body)
    rc = checker.main([str(xml_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "scripts/cloud_campaign_manager.py" in out
    assert "below threshold" in out


def test_main_returns_one_when_xml_missing(tmp_path, capsys):
    rc = checker.main([str(tmp_path / "does_not_exist.xml")])
    assert rc == 1
    out = capsys.readouterr().out
    assert "::error::coverage XML not found" in out


def test_main_returns_one_on_malformed_xml(tmp_path, capsys):
    xml_path = _write_xml(tmp_path, "<coverage><not-closed>")
    rc = checker.main([str(xml_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "::error::failed to parse" in out


def test_main_uses_default_path_when_no_arg_given(monkeypatch, tmp_path, capsys):
    # Redirect DEFAULT_COVERAGE_XML to a passing fixture so the no-arg path
    # can be exercised without touching the real repo layout.
    passing = _write_xml(tmp_path, _PASSING_XML)
    monkeypatch.setattr(checker, "DEFAULT_COVERAGE_XML", str(passing))
    rc = checker.main([])
    assert rc == 0
