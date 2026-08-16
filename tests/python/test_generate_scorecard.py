"""
Unit tests for the release scorecard generator (scripts/generate_scorecard.py).

Covers the module-level API introduced by the #2496 refactor (commit 6170687):
``parse_ashrae``, ``parse_series``, ``parse_gates``, ``parse_readme_throughput``,
``render``, ``load_all``, and ``main``. The script reads from committed sources
under a hard-coded ``REPO`` root and emits a deterministic ``SCORECARD.md``
that CI diffs against the committed copy.

These tests intentionally avoid invoking ``cargo``; they exercise the pure
parsing / rendering logic and the CLI's drift path directly.

Fixes issue #2850 by replacing the previously-skipped legacy ``ScorecardGenerator``
test surface (commit #2496 removed that class) with coverage against the
current module-level API.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_scorecard.py"


def _load_module():
    """Load scripts/generate_scorecard.py as an isolated module by path.

    Registered in ``sys.modules`` so that ``dataclasses`` can resolve
    ``from __future__ import annotations`` string annotations back to the
    module's globals (``Optional[float]`` etc.).
    """
    spec = importlib.util.spec_from_file_location("generate_scorecard", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("generate_scorecard", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen_module():
    return _load_module()


ASHRAE_DOC_TEXT = """\
# ASHRAE Standard 140 Validation Results

*Generated: 2026-04-14 17:28 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 6.2% |
| Passed | 4 |
| Warnings | 2 |
| Failed | 58 |
| Mean Absolute Error | 35.35% |
| Max Deviation | 346.87% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Throughput | 8.51 cases/sec |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Status |
|------|----------------|--------|
| 600 | 4604.57 kWh | ❌ FAIL |
| 610 | 4691.46 kWh | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Status |
|------|----------------|--------|
| 900 | 5052.83 kWh | � FAIL |
| 910 | 5428.96 kWh | ❌ FAIL |

## Systematic Issues

### HVAC Load Calculation

**Count:** 11 metrics
"""

GATES_YAML_TEXT = """\
# Fixture gates.yaml (subset of release_gates.yaml)
validation:
  min_pass_rate: 60.0
  max_mae: 50.0
  known_failures:
    - "900"
    - "600"

benchmark:
  throughput:
    min_configs_per_sec: 150  # Wave 1+1.5 CI runners ~157 configs/sec
  latency:
    max_ms_per_config: 10.0

absolute_min_throughput: 100

# ============================================================================
# CI REQUIRED CHECKS
# ============================================================================
ci:
  required_checks:
    - "ASHRAE 140 Strict Energy Gate (Issue #1333)"
    - "Workspace Check (Issue #2983)"

# ============================================================================
# LOWER SECTION
# ============================================================================
other:
  foo: bar
"""

README_TEXT = """\
# Fluxion

- **Throughput:** ~900 configs/sec throughput in release mode via `BatchOracle`.
"""


@pytest.fixture
def repo_dir(tmp_path, monkeypatch):
    """Provide a tmp_path with stub ASHRAE / gates / README / SCORECARD and
    redirect the module-level path constants to point at it."""
    ashrae = tmp_path / "ASHRAE140_RESULTS.md"
    ashrae.write_text(ASHRAE_DOC_TEXT)
    gates = tmp_path / "release_gates.yaml"
    gates.write_text(GATES_YAML_TEXT)
    readme = tmp_path / "README.md"
    readme.write_text(README_TEXT)
    scorecard = tmp_path / "SCORECARD.md"
    scorecard.write_text("placeholder\n")

    monkeypatch.setattr("builtins.print", lambda *a, **k: None, raising=False)
    return tmp_path, ashrae, gates, readme, scorecard


# ---------------------------------------------------------------------------
# Numeric parser (_num)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cell, expected",
    [
        ("6.2%", 6.2),
        ("64", 64.0),
        ("35.35%", 35.35),
        ("346.87%", 346.87),
        ("0 / 18 cases", 0.0),
        ("1,234.5", 1234.5),
        ("-5", -5.0),
        ("", None),
        ("n/a", None),
        ("abc", None),
    ],
)
def test_num_extracts_leading_numeric(gen_module, cell, expected):
    assert gen_module._num(cell) == expected


# ---------------------------------------------------------------------------
# parse_ashrae
# ---------------------------------------------------------------------------


def test_parse_ashrae_extracts_summary_metrics(gen_module):
    v = gen_module.parse_ashrae(ASHRAE_DOC_TEXT)
    assert v.total == 64
    assert v.passed == 4
    assert v.failed == 58
    assert v.warnings == 2
    assert v.pass_rate == pytest.approx(6.2)
    assert v.mae == pytest.approx(35.35)
    assert v.max_deviation == pytest.approx(346.87)


def test_parse_ashrae_extracts_generated_timestamp(gen_module):
    v = gen_module.parse_ashrae(ASHRAE_DOC_TEXT)
    assert v.generated_utc == "2026-04-14 17:28 UTC"


def test_parse_ashrae_extracts_throughput_only_with_cases_per_sec(gen_module):
    v = gen_module.parse_ashrae(ASHRAE_DOC_TEXT)
    assert v.throughput_cases_per_sec == pytest.approx(8.51)


def test_parse_ashrae_returns_defaults_for_empty(gen_module):
    v = gen_module.parse_ashrae("# Title\n\nbody")
    assert v.total == 0
    assert v.passed == 0
    assert v.failed == 0
    assert v.warnings == 0
    assert v.pass_rate == 0.0
    assert v.mae == 0.0
    assert v.max_deviation == 0.0
    assert v.throughput_cases_per_sec == 0.0
    assert v.generated_utc == ""


def test_parse_ashrae_ignores_sections_outside_tables(gen_module):
    text = "## Summary\n\nno table here\n## Performance Summary\n\nstill no table\n"
    v = gen_module.parse_ashrae(text)
    assert v.total == 0
    assert v.pass_rate == 0.0


# ---------------------------------------------------------------------------
# parse_series
# ---------------------------------------------------------------------------


def test_parse_series_groups_by_subsection(gen_module):
    rows = gen_module.parse_series(ASHRAE_DOC_TEXT)
    by_name = {r.name: r for r in rows}
    assert "Baseline Cases (600 Series)" in by_name
    assert "High-Mass Cases (900 Series)" in by_name
    assert by_name["Baseline Cases (600 Series)"].failed == 2
    assert by_name["High-Mass Cases (900 Series)"].failed == 2


def test_parse_series_counts_pass_warn_fail(gen_module):
    text = (
        "## Detailed Results\n\n"
        "### Mixed\n\n"
        "| Case | Status |\n"
        "|------|--------|\n"
        "| 1 | ✅ PASS |\n"
        "| 2 | ⚠ WARN |\n"
        "| 3 | ❌ FAIL |\n"
        "| 4 | ✅ PASS |\n"
    )
    rows = gen_module.parse_series(text)
    assert len(rows) == 1
    r = rows[0]
    assert r.name == "Mixed"
    assert r.passed == 2
    assert r.warn == 1
    assert r.failed == 1
    assert r.cases == 4
    assert r.pass_rate == pytest.approx(50.0)


def test_parse_series_drops_rows_without_cases(gen_module):
    text = "## Detailed Results\n\n### Empty\n\nno rows\n"
    assert gen_module.parse_series(text) == []


def test_parse_series_skips_sections_outside_detailed_results(gen_module):
    rows = gen_module.parse_series(ASHRAE_DOC_TEXT)
    names = [r.name for r in rows]
    assert "HVAC Load Calculation" not in names


def test_parse_series_stops_after_detailed_results(gen_module):
    text = (
        "## Detailed Results\n\n### Inside\n"
        "| Case | Status |\n|------|--------|\n| 1 | ✅ PASS |\n\n"
        "## Systematic Issues\n\n### Outside\n"
        "| Case | Status |\n|------|--------|\n| 2 | ✅ PASS |\n"
    )
    rows = gen_module.parse_series(text)
    assert [r.name for r in rows] == ["Inside"]


# ---------------------------------------------------------------------------
# parse_gates
# ---------------------------------------------------------------------------


def test_parse_gates_extracts_budget_values(gen_module):
    g = gen_module.parse_gates(GATES_YAML_TEXT)
    assert g.min_pass_rate == 60.0
    assert g.max_mae == 50.0
    assert g.min_throughput == 150.0
    assert g.max_latency_ms == 10.0
    assert g.absolute_min_throughput == 100.0


def test_parse_gates_extracts_known_failures(gen_module):
    g = gen_module.parse_gates(GATES_YAML_TEXT)
    assert g.known_failures == ["900", "600"]


def test_parse_gates_extracts_required_checks(gen_module):
    g = gen_module.parse_gates(GATES_YAML_TEXT)
    assert "ASHRAE 140 Strict Energy Gate (Issue #1333)" in g.required_checks
    assert "Workspace Check (Issue #2983)" in g.required_checks


def test_parse_gates_extracts_ci_throughput_comment(gen_module):
    g = gen_module.parse_gates(GATES_YAML_TEXT)
    assert g.ci_throughput_comment == pytest.approx(157.0)


def test_parse_gates_returns_defaults_for_empty(gen_module):
    g = gen_module.parse_gates("# empty\n")
    assert g.min_pass_rate == 60.0
    assert g.max_mae == 50.0
    assert g.min_throughput == 150.0
    assert g.max_latency_ms == 10.0
    assert g.absolute_min_throughput == 100.0
    assert g.known_failures == []
    assert g.required_checks == []
    assert g.ci_throughput_comment == 0.0


# ---------------------------------------------------------------------------
# parse_readme_throughput
# ---------------------------------------------------------------------------


def test_parse_readme_throughput_extracts_release_figure(gen_module):
    b = gen_module.parse_readme_throughput(README_TEXT)
    assert b.readme_release_throughput == pytest.approx(900.0)


def test_parse_readme_throughput_returns_zero_when_missing(gen_module):
    b = gen_module.parse_readme_throughput("# Fluxion\n\nNo throughput claim here.\n")
    assert b.readme_release_throughput == 0.0


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_embeds_validation_metrics(gen_module):
    v = gen_module.parse_ashrae(ASHRAE_DOC_TEXT)
    series = gen_module.parse_series(ASHRAE_DOC_TEXT)
    g = gen_module.parse_gates(GATES_YAML_TEXT)
    b = gen_module.parse_readme_throughput(README_TEXT)

    out = gen_module.render(v, series, g, b)

    assert "6.2%" in out
    assert "35.35%" in out
    assert "ASHRAE140_RESULTS.md" in out or "ASHRAE 140" in out
    assert "release_gates.yaml" in out
    assert "README.md" in out


def test_render_never_shows_negative_inf_or_zero_pass_in_headline(gen_module):
    v = gen_module.Validation()
    v.pass_rate = 6.2
    v.mae = 35.35
    v.total = 64
    v.passed = 4
    v.warnings = 2
    v.failed = 58
    v.max_deviation = 346.87
    v.throughput_cases_per_sec = 8.51
    g = gen_module.parse_gates(GATES_YAML_TEXT)
    b = gen_module.parse_readme_throughput(README_TEXT)

    out = gen_module.render(v, [], g, b)

    mae_line = next(line for line in out.splitlines() if "Mean Absolute Error" in line)
    assert "35.35%" in mae_line
    assert "-inf%" not in mae_line

    pass_line = next(
        line for line in out.splitlines() if "ASHRAE 140 pass rate" in line
    )
    assert "6.2%" in pass_line
    assert "(0/0)" not in pass_line


def test_render_includes_required_checks_table(gen_module):
    v = gen_module.parse_ashrae(ASHRAE_DOC_TEXT)
    g = gen_module.parse_gates(GATES_YAML_TEXT)
    b = gen_module.parse_readme_throughput(README_TEXT)

    out = gen_module.render(v, [], g, b)

    assert "ASHRAE 140 Strict Energy Gate (Issue #1333)" in out
    assert "Workspace Check (Issue #2983)" in out
    assert "#1333" in out
    assert "#2983" in out


def test_render_includes_known_structural_failures(gen_module):
    v = gen_module.parse_ashrae(ASHRAE_DOC_TEXT)
    g = gen_module.parse_gates(GATES_YAML_TEXT)
    b = gen_module.parse_readme_throughput(README_TEXT)

    out = gen_module.render(v, [], g, b)

    assert "600" in out
    assert "900" in out
    assert "Known Structural Failures" in out


def test_render_is_byte_stable_for_same_input(gen_module):
    v = gen_module.parse_ashrae(ASHRAE_DOC_TEXT)
    series = gen_module.parse_series(ASHRAE_DOC_TEXT)
    g = gen_module.parse_gates(GATES_YAML_TEXT)
    b = gen_module.parse_readme_throughput(README_TEXT)

    out_a = gen_module.render(v, series, g, b)
    out_b = gen_module.render(v, series, g, b)
    assert out_a == out_b


# ---------------------------------------------------------------------------
# load_all
# ---------------------------------------------------------------------------


def test_load_all_returns_parsed_sources(gen_module, repo_dir, monkeypatch):
    root, ashrae, gates, readme, scorecard = repo_dir
    monkeypatch.setattr(gen_module, "ASHRAE_DOC", ashrae)
    monkeypatch.setattr(gen_module, "GATES_YAML", gates)
    monkeypatch.setattr(gen_module, "README_MD", readme)
    monkeypatch.setattr(gen_module, "SCORECARD", scorecard)

    v, series, g, b = gen_module.load_all(verbose=False)
    assert v.total == 64
    assert v.pass_rate == pytest.approx(6.2)
    assert g.min_pass_rate == 60.0
    assert b.readme_release_throughput == pytest.approx(900.0)
    assert any(r.name == "Baseline Cases (600 Series)" for r in series)


def test_load_all_never_calls_cargo(gen_module, repo_dir, monkeypatch):
    root, ashrae, gates, readme, scorecard = repo_dir
    monkeypatch.setattr(gen_module, "ASHRAE_DOC", ashrae)
    monkeypatch.setattr(gen_module, "GATES_YAML", gates)
    monkeypatch.setattr(gen_module, "README_MD", readme)
    monkeypatch.setattr(gen_module, "SCORECARD", scorecard)

    def _fail(*_a, **_k):
        raise AssertionError("load_all must not invoke cargo / shell out")

    monkeypatch.setattr(subprocess, "run", _fail)
    gen_module.load_all(verbose=False)


def test_load_all_exits_when_ashrae_doc_missing(gen_module, tmp_path, monkeypatch):
    gates = tmp_path / "release_gates.yaml"
    gates.write_text(GATES_YAML_TEXT)
    monkeypatch.setattr(gen_module, "ASHRAE_DOC", tmp_path / "missing.md")
    monkeypatch.setattr(gen_module, "GATES_YAML", gates)

    with pytest.raises(SystemExit) as ei:
        gen_module.load_all(verbose=False)
    assert ei.value.code == 2


def test_load_all_exits_when_gates_missing(gen_module, tmp_path, monkeypatch):
    ashrae = tmp_path / "ASHRAE140_RESULTS.md"
    ashrae.write_text(ASHRAE_DOC_TEXT)
    monkeypatch.setattr(gen_module, "ASHRAE_DOC", ashrae)
    monkeypatch.setattr(gen_module, "GATES_YAML", tmp_path / "missing.yaml")

    with pytest.raises(SystemExit) as ei:
        gen_module.load_all(verbose=False)
    assert ei.value.code == 2


# ---------------------------------------------------------------------------
# main (CLI)
# ---------------------------------------------------------------------------


def test_main_writes_scorecard_to_default_path(gen_module, repo_dir, monkeypatch):
    root, ashrae, gates, readme, scorecard = repo_dir
    monkeypatch.setattr(gen_module, "ASHRAE_DOC", ashrae)
    monkeypatch.setattr(gen_module, "GATES_YAML", gates)
    monkeypatch.setattr(gen_module, "README_MD", readme)
    monkeypatch.setattr(gen_module, "SCORECARD", scorecard)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])

    rc = gen_module.main()
    assert rc == 0
    assert scorecard.exists()
    body = scorecard.read_text()
    assert "Fluxion Release Scorecard" in body
    assert "6.2%" in body


def test_main_writes_to_custom_output(gen_module, repo_dir, monkeypatch):
    root, ashrae, gates, readme, scorecard = repo_dir
    monkeypatch.setattr(gen_module, "ASHRAE_DOC", ashrae)
    monkeypatch.setattr(gen_module, "GATES_YAML", gates)
    monkeypatch.setattr(gen_module, "README_MD", readme)
    monkeypatch.setattr(gen_module, "SCORECARD", scorecard)

    custom = root / "custom_scorecard.md"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "-o", str(custom)])

    rc = gen_module.main()
    assert rc == 0
    assert custom.exists()
    assert "Fluxion Release Scorecard" in custom.read_text()


def test_main_check_exits_zero_when_scorecard_matches(
    gen_module, repo_dir, monkeypatch
):
    root, ashrae, gates, readme, scorecard = repo_dir
    monkeypatch.setattr(gen_module, "ASHRAE_DOC", ashrae)
    monkeypatch.setattr(gen_module, "GATES_YAML", gates)
    monkeypatch.setattr(gen_module, "README_MD", readme)

    expected = gen_module.render(
        gen_module.parse_ashrae(ASHRAE_DOC_TEXT),
        gen_module.parse_series(ASHRAE_DOC_TEXT),
        gen_module.parse_gates(GATES_YAML_TEXT),
        gen_module.parse_readme_throughput(README_TEXT),
    )
    scorecard.write_text(expected)
    monkeypatch.setattr(gen_module, "SCORECARD", scorecard)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--check"])

    assert gen_module.main() == 0


def test_main_check_exits_one_on_drift(gen_module, repo_dir, monkeypatch):
    root, ashrae, gates, readme, scorecard = repo_dir
    monkeypatch.setattr(gen_module, "ASHRAE_DOC", ashrae)
    monkeypatch.setattr(gen_module, "GATES_YAML", gates)
    monkeypatch.setattr(gen_module, "README_MD", readme)
    scorecard.write_text("stale content\n")
    monkeypatch.setattr(gen_module, "SCORECARD", scorecard)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--check"])

    assert gen_module.main() == 1


def test_cli_subprocess_writes_scorecard(tmp_path):
    """End-to-end: invoke the script as a subprocess with a custom output path
    and verify it returns 0 and writes a valid scorecard.

    Sources are read from the hard-coded repo root, so we only redirect the
    output path via ``-o``. The drift / missing-source paths are exercised
    in-process by ``main()`` tests above (``test_main_check_exits_zero_when_
    scorecard_matches`` and ``test_main_check_exits_one_on_drift``)."""
    out = tmp_path / "scorecard.md"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "-o", str(out)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert out.exists()
    body = out.read_text()
    assert "Fluxion Release Scorecard" in body
    assert "ASHRAE 140" in body
