"""Hermetic tests for the Issue #4160 thermal-selector default-claim anti-drift gate.

The gate scans root markdown docs for the pre-ADR-0017 false claim that
``ThermalSelector::default()`` resolves unconditionally to
``ZoneSolverKind::Gauge`` (in both feature states, with the ``gauge-solver``
feature separately gating a fall-through to legacy 5R1C/9R4C).

ADR-0017 (Issue #3978) replaced that posture: the default is cfg-dependent
and explicit in every build (Gauge with ``--features gauge-solver``,
explicit legacy FiveROneC otherwise), the silent fall-through was removed,
and flip authority sits with the #3986 teacher validation suite.
"""
from __future__ import annotations

import pytest

SCRIPT_NAME = "check_thermal_selector_default_claim"

# The exact false claim that shipped in README.md:29 before Issue #4160.
FALSE_CLAIM_LINE = (
    "the `ZoneSolverKind::Gauge` selector is the unconditional default; "
    "the cargo feature `gauge-solver` separately gates whether the "
    "dispatcher's gauge arm runs unconditionally or falls through to "
    "legacy 5R1C/9R4C (`ThermalSelector::default()` resolves to "
    "`ZoneSolverKind::Gauge` in both feature states)"
)

# The ADR-0017-correct posture (what README.md says after #4160).
CORRECT_POSTURE_LINE = (
    "the default thermal selector is now cfg-dependent and explicit in "
    "every build — `ZoneSolverKind::Gauge` with `--features gauge-solver`, "
    "explicit legacy `FiveROneC` (HighMass specs still auto-promote to "
    "9R4C) in default builds; the silent fall-through was removed, and "
    "an explicit `Gauge` selector in a default build panics loudly at "
    "construction"
)


@pytest.fixture
def checker(load_script):
    return load_script(SCRIPT_NAME)


def _write_mock_repo(tmp_path):
    """Write the four scanned root docs; return tmp_path."""
    for name in ("README.md", "AGENTS.md", "RULES.md", "CONTRIBUTING.md"):
        (tmp_path / name).write_text("# placeholder\n", encoding="utf-8")
    return tmp_path


def test_main_passes_on_correct_posture(checker, monkeypatch, tmp_path):
    """README stating the ADR-0017 posture must not trip the gate."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    _write_mock_repo(tmp_path)
    (tmp_path / "README.md").write_text(
        "# Fluxion\n\n" + CORRECT_POSTURE_LINE + "\n", encoding="utf-8"
    )
    assert checker.main() == 0


def test_main_fails_on_unconditional_gauge_default(checker, monkeypatch, tmp_path):
    """The exact pre-#4160 README claim must be caught (exit 1)."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    _write_mock_repo(tmp_path)
    (tmp_path / "README.md").write_text(
        "# Fluxion\n\n" + FALSE_CLAIM_LINE + "\n", encoding="utf-8"
    )
    assert checker.main() == 1


def test_fenced_code_block_is_excluded(checker, monkeypatch, tmp_path):
    """Quoting the historical false claim inside a fence is not re-drift."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    _write_mock_repo(tmp_path)
    (tmp_path / "README.md").write_text(
        "# Fluxion\n\nHistorical (removed by ADR-0017):\n\n```\n"
        + FALSE_CLAIM_LINE
        + "\n```\n",
        encoding="utf-8",
    )
    assert checker.main() == 0


def test_negated_claim_is_not_flagged(checker, monkeypatch, tmp_path):
    """A sentence explicitly refuting the old claim is the correct posture."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    _write_mock_repo(tmp_path)
    (tmp_path / "README.md").write_text(
        "# Fluxion\n\nThe unconditional Gauge default was removed by ADR-0017; "
        "there is no silent fall-through to legacy 5R1C/9R4C.\n",
        encoding="utf-8",
    )
    assert checker.main() == 0


def test_check_file_reports_line_numbers(checker, tmp_path):
    """_check_file returns the offending line numbers for a planted claim."""
    doc = tmp_path / "README.md"
    doc.write_text("clean line\n" + FALSE_CLAIM_LINE + "\n", encoding="utf-8")
    failures = checker._check_file(doc)
    assert failures, "expected the planted false claim to be flagged"
    assert failures[0][0] == 2
