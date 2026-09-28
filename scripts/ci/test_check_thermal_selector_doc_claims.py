"""
Tests for ``scripts/check_thermal_selector_doc_claims.py`` -- Issue #4160.

The gate exists because ``README.md`` documented a production path that
the repository deliberately deleted: an unconditional ``Gauge`` default
plus a feature-flag-gated "falls through to legacy 5R1C/9R4C" arm
(ADR-0017, Issue #3978, removed both).

A doc-drift gate that is too eager is worse than none -- it forces
contributors to delete correct ADR-0017 wording to get a green build, and
that pressure eventually erodes the very documentation it protects. So
this suite asserts BOTH directions:

  * each of the four false-assertion shapes is reported (the four README /
    ARCHITECTURE.md lines the issue and its audit found);
  * ADR-0017's own canonical wording -- which legitimately uses
    "unconditional" to describe the gauge arm *under the feature* -- is
    reported clean, as are past-tense and explicit-negation phrasings.

Fixtures are synthetic documents, never the real tree, so the suite cannot
rot when unrelated prose changes.
"""
from __future__ import annotations

import sys

import pytest

SCRIPT_NAME = "check_thermal_selector_doc_claims"


@pytest.fixture
def gate(load_script):
    """Freshly-loaded copy of the thermal-selector doc-claim gate."""
    return load_script(SCRIPT_NAME)


def _doc(tmp_path, text: str, name: str = "README.md"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Positive: the false claims must be reported
# ---------------------------------------------------------------------------


def test_reports_both_feature_states_claim(gate, tmp_path):
    """The #4160 headline claim: Gauge is default in *both* feature states."""
    path = _doc(
        tmp_path,
        "The `ZoneSolverKind::Gauge` selector is the unconditional default\n"
        "(`ThermalSelector::default()` resolves to `ZoneSolverKind::Gauge` "
        "in both feature states).\n",
    )
    findings = gate.check_doc(path, rel="README.md")
    assert len(findings) == 1
    assert findings[0].startswith("README.md:2:")
    assert "both feature states" in findings[0]
    assert "ADR-0017" in findings[0]


def test_reports_fall_through_claim(gate, tmp_path):
    """The deleted fall-through arm described as current behavior."""
    path = _doc(
        tmp_path,
        "The dispatcher's gauge arm runs unconditionally or falls through "
        "to legacy 5R1C/9R4C.\n",
    )
    findings = gate.check_doc(path, rel="README.md")
    assert len(findings) == 1
    assert "fall-through" in findings[0]


def test_reports_feature_gates_fall_through_claim(gate, tmp_path):
    """`gauge-solver` "gates whether ... falls through" -- the original sin."""
    path = _doc(
        tmp_path,
        "The cargo feature `gauge-solver` separately gates whether the "
        "dispatcher's gauge arm runs unconditionally or falls through.\n",
    )
    findings = gate.check_doc(path, rel="README.md")
    assert len(findings) == 1
    assert "gauge arm" in findings[0]


def test_reports_bare_unconditional_default_claim(gate, tmp_path):
    """An unqualified "Gauge selector is the unconditional default"."""
    path = _doc(
        tmp_path,
        "In the default build the Gauge selector is now the unconditional "
        "default for every zone.\n",
    )
    findings = gate.check_doc(path, rel="README.md")
    assert len(findings) == 1
    assert "unconditional default" in findings[0]


# ---------------------------------------------------------------------------
# Negative: correct ADR-0017 wording must stay clean
# ---------------------------------------------------------------------------


def test_clean_agents_md_canonical_wording(gate, tmp_path):
    """AGENTS.md's actual ADR-0017 paragraph is the canonical clean form.

    This is a verbatim copy of the committed AGENTS.md wording, which uses
    "unconditional" to describe the gauge arm *under the feature*. If this
    test ever fails, the gate is too eager and would force deletion of
    correct documentation.
    """
    path = _doc(
        tmp_path,
        "- **ADR-0017 interim posture (Issue #3978; supersedes the Phase A8 "
        "two-solver limbo of Issue #3291) — the default thermal selector is "
        "cfg-dependent and explicit in every build; no silent cfg "
        "fall-through anywhere.** With `--features gauge-solver` on, "
        "`ThermalSelector::default()` is `ZoneSolverKind::Gauge` "
        "(`src/sim/thermal_selector.rs`) and the dispatcher's gauge arm is "
        "unconditional — a missing gauge backend is a programming error "
        "that panics loudly; the β-soak gate (Issue #3286) remains the "
        "nightly authority on this arm. In the default build (feature OFF), "
        "`ThermalSelector::default()` is the **explicit legacy `FiveROneC`** "
        "(HighMass specs still auto-promote to 9R4C via "
        "`from_spec_with_selector`, honored by the `FiveROneC` dispatch "
        "arm), and an **explicit `Gauge` selector panics loudly** at "
        "construction naming the feature flag — the silent fall-through to "
        "legacy 5R1C/9R4C was removed.\n",
        name="AGENTS.md",
    )
    assert gate.check_doc(path, rel="AGENTS.md") == []


def test_clean_feature_qualified_unconditional(gate, tmp_path):
    """"Unconditional" scoped to the feature build is correct, not a lie."""
    path = _doc(
        tmp_path,
        "Under `--features gauge-solver` the Gauge selector is the "
        "unconditional default, so neither legacy row is reached without an "
        "explicit `ThermalSelector`.\n",
    )
    assert gate.check_doc(path, rel="ARCHITECTURE.md") == []


def test_clean_explicit_panicking_selector(gate, tmp_path):
    """"rather than falling through" states the correct current behavior."""
    path = _doc(
        tmp_path,
        "An explicit `Gauge` selector in a default build panics at "
        "construction, naming the feature flag, rather than falling through "
        "to a legacy solver.\n",
    )
    assert gate.check_doc(path, rel="ARCHITECTURE.md") == []


def test_clean_historical_superseded_wording(gate, tmp_path):
    """Past-tense history of the removed posture is legitimate."""
    path = _doc(
        tmp_path,
        "Phase A8 made the Gauge selector the unconditional default; this "
        "posture was superseded by ADR-0017 and the silent fall-through "
        "was removed.\n",
    )
    assert gate.check_doc(path, rel="README.md") == []


def test_clean_unrelated_doc(gate, tmp_path):
    """A doc that never mentions the selector is trivially clean."""
    path = _doc(tmp_path, "# Title\n\nSome prose about the build.\n")
    assert gate.check_doc(path, rel="README.md") == []


# ---------------------------------------------------------------------------
# main() surface
# ---------------------------------------------------------------------------


def test_main_returns_one_on_findings(gate, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gate", "--root", str(tmp_path)])
    _doc(
        tmp_path,
        "The `ZoneSolverKind::Gauge` selector is the unconditional default "
        "and falls through to legacy 5R1C/9R4C.\n",
    )
    assert gate.main() == 1
    assert "Unconditional-Gauge" in capsys.readouterr().err


def test_main_returns_zero_when_clean(gate, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gate", "--root", str(tmp_path)])
    _doc(
        tmp_path,
        "Under `--features gauge-solver` the Gauge selector is the "
        "unconditional default.\n",
    )
    assert gate.main() == 0
    assert "OK" in capsys.readouterr().out


def test_main_scans_all_root_docs(gate, tmp_path, monkeypatch, capsys):
    """A finding in a non-README root doc is still reported."""
    monkeypatch.setattr(sys, "argv", ["gate", "--root", str(tmp_path)])
    _doc(tmp_path, "clean prose\n", name="README.md")
    _doc(
        tmp_path,
        "The Gauge selector is the unconditional default for all builds.\n",
        name="ARCHITECTURE.md",
    )
    assert gate.main() == 1
    assert "ARCHITECTURE.md" in capsys.readouterr().err


def test_main_returns_two_when_root_missing(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gate", "--root", str(tmp_path / "nope")])
    assert gate.main() == 2


def test_main_returns_two_when_no_root_docs_present(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gate", "--root", str(tmp_path)])
    assert gate.main() == 2


# ---------------------------------------------------------------------------
# Non-vacuity / scope guards
# ---------------------------------------------------------------------------


def test_script_module_exposes_expected_surface(gate):
    """Guard against an import fixture that silently yields a stub.

    Handoff §7: a control with no real coverage produced vacuously green
    harnesses twice. Assert the loaded module is the real file.
    """
    assert callable(gate.check_doc)
    assert callable(gate.main)
    with open(gate.__file__, encoding="utf-8") as fh:
        src = fh.read()
    assert "ADR-0017" in src
    assert "ROOT_DOCS" in src


def test_root_docs_exclude_adr_records(gate):
    """ADR files are dated records; rewriting one destroys its provenance."""
    assert "AGENTS.md" in gate.ROOT_DOCS
    assert "README.md" in gate.ROOT_DOCS
    assert not any("adr" in name.lower() for name in gate.ROOT_DOCS)


def test_real_tree_is_clean(gate, repo_root):
    """Pin the gate against the live root docs after the #4160 rewrite.

    The synthetic fixtures above prove the matcher works; this proves the
    shipped docs actually pass, so a future regression fails the suite
    rather than waiting for the direct-invocation CI step.
    """
    findings = []
    for name in gate.ROOT_DOCS:
        path = repo_root / name
        if path.is_file():
            findings.extend(gate.check_doc(path, rel=name))
    assert findings == [], findings
