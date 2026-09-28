"""Tests for ``scripts/check_scorecard_data_sources_consistent.py`` -- Issue #4208.

The gate checks that the committed sources feeding the ``SCORECARD.md``
headline agree: ``validation/performance_history.latest.json`` (preferred)
vs ``docs/ASHRAE140_RESULTS.md`` (fallback), the ``SCORECARD.md`` headline
itself against the canonical source (Issue #4208 -- previously parsed but
never compared), and ``docs/KNOWN_ISSUES.md`` for stale SCORECARD
cross-references (Issue #3578).

The tests below drive ``main()`` against hermetic ``tmp_path`` fixture
repos (the ``load_script`` + ``monkeypatch`` pattern): a consistent
fixture passes (exit 0); a SCORECARD headline that disagrees with the
performance history / ASHRAE results fails (exit 1); source-vs-source
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
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(checker, "PERF_SNAPSHOT", paths["perf"])
    monkeypatch.setattr(checker, "ASHRAE_DOC", paths["ashrae"])
    monkeypatch.setattr(checker, "SCORECARD_MD", paths["scorecard"])
    monkeypatch.setattr(checker, "KNOWN_ISSUES", paths["known_issues"])
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


def _consistent_repo(checker, monkeypatch, tmp_path: Path) -> dict[str, Path]:
    """A mock repo where all sources agree within tolerance."""
    paths = _redirect(checker, monkeypatch, tmp_path)
    _write_perf(paths["perf"])
    _write_ashrae_doc(paths["ashrae"])
    _write_scorecard(paths["scorecard"])
    _write_known_issues(paths["known_issues"])
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
