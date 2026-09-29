"""Tests for ``scripts/check_scorecard_data_sources_consistent.py`` --
Issues #4208 / #4183.

The gate checks that the committed sources feeding the ``SCORECARD.md``
headline agree: ``validation/performance_history.latest.json`` (preferred)
vs ``docs/ASHRAE140_RESULTS.md`` (fallback), the ``SCORECARD.md`` headline
itself against the canonical source (Issue #4208 -- previously parsed but
never compared), ``README.md`` headline against the canonical source
(Issue #4183 -- README is load-bearing for the scorecard but was
unguarded), and ``docs/KNOWN_ISSUES.md`` for stale SCORECARD
cross-references (Issue #3578).

The tests below drive ``main()`` against hermetic ``tmp_path`` fixture
repos (the ``load_script`` + ``monkeypatch`` pattern): a consistent
fixture passes (exit 0); a SCORECARD/README headline that disagrees with
the performance history / ASHRAE results fails (exit 1); source-vs-source
divergence and stale KNOWN_ISSUES tokens fail (exit 1). The end-to-end
against the real tree is the direct script invocation in
``scorecard-source-consistency.yml`` (path-filtered required check).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCRIPT_NAME = "check_scorecard_data_sources_consistent"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the scorecard consistency gate."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, monkeypatch, tmp_path: Path) -> dict[str, Path]:
    """Point the gate's source constants at a ``tmp_path`` mock repo."""
    paths = {
        "perf": tmp_path / "validation" / "performance_history.latest.json",
        "ashrae": tmp_path / "docs" / "ASHRAE140_RESULTS.md",
        "scorecard": tmp_path / "SCORECARD.md",
        "known_issues": tmp_path / "docs" / "KNOWN_ISSUES.md",
        "readme": tmp_path / "README.md",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(checker, "PERF_SNAPSHOT", paths["perf"])
    monkeypatch.setattr(checker, "ASHRAE_DOC", paths["ashrae"])
    monkeypatch.setattr(checker, "SCORECARD_MD", paths["scorecard"])
    monkeypatch.setattr(checker, "KNOWN_ISSUES", paths["known_issues"])
    monkeypatch.setattr(checker, "README_MD", paths["readme"])
    return paths


def _write_perf(path: Path, pass_rate: float = 9.7826,
                mae: float = 45.27) -> None:
    path.write_text(json.dumps({
        "timestamp": "2026-09-26T12:00:00+00:00",
        "pass_rate": pass_rate,
        "mae": mae,
    }), encoding="utf-8")


def _write_ashrae_doc(path: Path, pass_rate: float = 9.78,
                       mae: float = 45.27) -> None:
    path.write_text(
        "# ASHRAE 140 Results\n"
        "\n"
        "*Generated: 2026-09-26 12:00:00 UTC*\n"
        "\n"
        "| Metric | Value |\n"
        "|---|---|\n"
        f"| Pass Rate | {pass_rate}% |\n"
        f"| Mean Absolute Error | {mae}% |\n",
        encoding="utf-8",
    )


def _write_scorecard(path: Path, pass_rate: str = "9.8",
                     mae: str = "45.27") -> None:
    path.write_text(
        "# Fluxion Scorecard\n"
        "\n"
        "**Last Updated:** 2026-09-26\n"
        "\n"
        f"| ASHRAE 140 pass rate | **{pass_rate}%** (11/84 metrics) |\n"
        f"| Mean Absolute Error (MAE) | **{mae}%** |\n",
        encoding="utf-8",
    )


def _write_known_issues(path: Path, body: str = "") -> None:
    path.write_text(
        "# Known Issues\n"
        "\n"
        "No stale scorecard references in this fixture.\n" + body,
        encoding="utf-8",
    )


def _write_readme(path: Path, pass_rate: str = "9.8",
                  mae: str = "45.27",
                  generated_date: str = "2026-09-26") -> None:
    """Write a minimal README.md with headline metrics matching the regexes
    used by the gate's _load_readme_headline() function."""
    path.write_text(
        "# Fluxion: AI-Accelerated Building Energy Engine\n"
        "\n"
        f"> **Status:** Current ASHRAE 140-2023 validation pass rate is "
        f"**{pass_rate}%** (generated {generated_date}).\n"
        "\n"
        "## Current Validation Status\n"
        "\n"
        "![ASHRAE 140](https://img.shields.io/badge/ASHRAE140-"
        f"{pass_rate}%25%20pass-red)\n"
        "\n"
        "The figures below come from the committed validation suite "
        f"(generated {generated_date}); see SCORECARD.md.\n"
        "\n"
        "| Metric | Current | Target (release gate) | Status |\n"
        "|--------|---------|-----------------------|--------|\n"
        f"| Pass rate (metric-level) | **{pass_rate}%** (11/84) | "
        "≥ 60% | ❌ Fail |\n"
        f"| Mean Absolute Error (MAE) | **{mae}%** | ≤ 50% | ✅ Pass |\n"
        "\n"
        "### Known Limitations\n"
        "\n"
        f"- **Overall accuracy:** {mae}% MAE.\n",
        encoding="utf-8",
    )


def _consistent_repo(checker, monkeypatch, tmp_path: Path) -> dict[str, Path]:
    """A mock repo where all sources agree within tolerance."""
    paths = _redirect(checker, monkeypatch, tmp_path)
    _write_perf(paths["perf"])
    _write_ashrae_doc(paths["ashrae"])
    _write_scorecard(paths["scorecard"])
    _write_known_issues(paths["known_issues"])
    _write_readme(paths["readme"])
    return paths


# ---------------------------------------------------------------------------
# main(): consistent fixture passes
# ---------------------------------------------------------------------------


def test_consistent_fixture_passes(checker, monkeypatch, tmp_path, capsys):
    _consistent_repo(checker, monkeypatch, tmp_path)
    assert checker.main([]) == 0
    out = capsys.readouterr().out
    assert "PASS: the two committed sources agree within tolerance." in out
    assert "no stale SCORECARD cross-references" in out


# ---------------------------------------------------------------------------
# main(): SCORECARD headline disagreement fails (Issue #4208)
# ---------------------------------------------------------------------------


def test_scorecard_headline_pass_rate_disagreement_fails(
        checker, monkeypatch, tmp_path, capsys):
    """A SCORECARD.md headline pass rate that diverges from the
    performance-history canonical source fails -- the Issue #4208 hole
    (headline parsed but never compared)."""
    paths = _consistent_repo(checker, monkeypatch, tmp_path)
    _write_scorecard(paths["scorecard"], pass_rate="25.0")

    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert "SCORECARD.md headline pass_rate diverges" in out


def test_scorecard_headline_mae_disagreement_fails(
        checker, monkeypatch, tmp_path, capsys):
    paths = _consistent_repo(checker, monkeypatch, tmp_path)
    _write_scorecard(paths["scorecard"], mae="10.0")

    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert "SCORECARD.md headline mae diverges" in out


def test_unparseable_scorecard_headline_fails_loud(
        checker, monkeypatch, tmp_path, capsys):
    """A SCORECARD.md whose headline cannot be parsed fails loud rather
    than silent-green (``None`` is outside tolerance by contract)."""
    paths = _consistent_repo(checker, monkeypatch, tmp_path)
    paths["scorecard"].write_text(
        "# Fluxion Scorecard\n\nNo headline table here.\n", encoding="utf-8")

    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert "MISSING ON ONE SIDE" in out


def test_scorecard_headline_falls_back_to_ashrae_doc(
        checker, monkeypatch, tmp_path, capsys):
    """Without a perf-history snapshot the ASHRAE doc is the canonical
    source for the headline comparison."""
    paths = _redirect(checker, monkeypatch, tmp_path)
    _write_ashrae_doc(paths["ashrae"])
    _write_scorecard(paths["scorecard"])
    _write_known_issues(paths["known_issues"])
    # No perf snapshot written: perf-vs-doc fails (missing), but the
    # headline-vs-doc comparison itself must reference the ASHRAE doc.
    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert "canonical (docs/ASHRAE140_RESULTS.md)" in out


# ---------------------------------------------------------------------------
# main(): pre-existing contracts (regression guards)
# ---------------------------------------------------------------------------


def test_perf_ashrae_source_divergence_fails(checker, monkeypatch, tmp_path,
                                             capsys):
    """The original Issue #3535 contract: the two committed sources
    disagreeing beyond tolerance fails."""
    paths = _consistent_repo(checker, monkeypatch, tmp_path)
    _write_ashrae_doc(paths["ashrae"], pass_rate=50.0)

    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert "pass-rate divergence" in out


def test_stale_known_issues_token_fails(checker, monkeypatch, tmp_path,
                                        capsys):
    """The Issue #3578 contract: a stale SCORECARD token quoted in
    KNOWN_ISSUES.md fails."""
    paths = _consistent_repo(checker, monkeypatch, tmp_path)
    _write_known_issues(
        paths["known_issues"],
        body="See SCORECARD.md (Last Updated 2026-08-16): pass rate 14.3%.\n",
    )

    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert "stale SCORECARD cross-reference" in out


# ---------------------------------------------------------------------------
# main(): README.md headline vs canonical source (Issue #4183)
# ---------------------------------------------------------------------------


def test_readme_headline_agrees_with_canonical_passes(
        checker, monkeypatch, tmp_path, capsys):
    """A README.md headline that agrees with the canonical perf-history
    source passes -- the Issue #4183 acceptance criterion (positive case)."""
    paths = _consistent_repo(checker, monkeypatch, tmp_path)
    # README headline matches perf-history values within tolerance.
    _write_readme(paths["readme"], pass_rate="9.8", mae="45.27",
                  generated_date="2026-09-26")

    assert checker.main([]) == 0
    out = capsys.readouterr().out
    # Check that the README headline section passes the check.
    assert "README.md =    9.8" in out
    assert "README.md =  45.27" in out
    # Both pass within tolerance (0.017 pp and 0.000 pp respectively).
    assert "diff = 0.017 pp  ✓" in out
    assert "diff = 0.000 pp  ✓" in out


def test_readme_headline_stale_pass_rate_fails(
        checker, monkeypatch, tmp_path, capsys):
    """A README.md headline with a stale pass rate fails -- the
    Issue #4183 negative regression case (stale figures should be caught)."""
    paths = _consistent_repo(checker, monkeypatch, tmp_path)
    # Plant the stale 14.1% pass rate from the issue #4183 description.
    _write_readme(paths["readme"], pass_rate="14.1", mae="45.27",
                  generated_date="2026-09-07")

    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert "README.md headline pass_rate diverges" in out
    assert "diff" in out and "pp" in out


def test_readme_headline_stale_mae_fails(
        checker, monkeypatch, tmp_path, capsys):
    """A README.md headline with a stale MAE fails -- the
    Issue #4183 negative regression case."""
    paths = _consistent_repo(checker, monkeypatch, tmp_path)
    # Plant the stale 49.82% MAE from the issue #4183 description.
    _write_readme(paths["readme"], pass_rate="9.8", mae="49.82",
                  generated_date="2026-09-07")

    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert "README.md headline mae diverges" in out
    assert "diff" in out and "pp" in out


def test_unparseable_readme_headline_fails_loud(
        checker, monkeypatch, tmp_path, capsys):
    """A README.md whose headline cannot be parsed fails loud rather
    than silent-green (``None`` is outside tolerance by contract)."""
    paths = _consistent_repo(checker, monkeypatch, tmp_path)
    paths["readme"].write_text(
        "# Fluxion\n\nNo validation metrics here.\n", encoding="utf-8")

    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert "MISSING ON ONE SIDE" in out


def test_readme_headline_falls_back_to_ashrae_doc(
        checker, monkeypatch, tmp_path, capsys):
    """Without a perf-history snapshot the ASHRAE doc is the canonical
    source for the README headline comparison (Issue #4183 fallback)."""
    paths = _redirect(checker, monkeypatch, tmp_path)
    _write_ashrae_doc(paths["ashrae"])
    _write_scorecard(paths["scorecard"])
    _write_known_issues(paths["known_issues"])
    _write_readme(paths["readme"])
    # No perf snapshot: README headline should be compared against ASHRAE doc.
    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert "canonical (docs/ASHRAE140_RESULTS.md)" in out
