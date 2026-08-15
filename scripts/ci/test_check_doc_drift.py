"""
Tests for ``scripts/check_doc_drift.py`` -- Issue #2895.

Regression guard for doc-comment cycle-state drift in
``fluxion-core/src/lib.rs``. These tests pin the scan primitives, the
tense classifier, the shim detection, and the at-or-below ARCHITECTURE.md
``§"Remaining cycles"`` cross-reference so a silent regex regression
cannot re-introduce a stale "cycle remains" claim that the architecture
drift detector would miss.

Pattern (mirrors ``test_check_physics_sim_cycle.py`` /
``test_check_ashrae_cases_cycle.py``): load the script as a fresh
module, redirect its module-level path constants
(``REPO_ROOT`` / ``LIB_RS`` / ``ARCHITECTURE_MD`` / ``FLUXION_SRC`` /
``FLUXION_CORE_SRC``) at a ``tmp_path`` mock repo, then drive each
scanner + ``main()`` through clean and drift scenarios.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "check_doc_drift"


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the doc-drift script."""
    return load_script(SCRIPT_NAME)


def _write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _redirect(checker, tmp_path, monkeypatch) -> None:
    """Point every module-level path constant at the ``tmp_path`` mock repo."""
    fluxion_core_src = tmp_path / "fluxion-core" / "src"
    fluxion_src = tmp_path / "src"
    lib_rs = fluxion_core_src / "lib.rs"
    arch_md = tmp_path / "ARCHITECTURE.md"
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "LIB_RS", lib_rs)
    monkeypatch.setattr(checker, "ARCHITECTURE_MD", arch_md)
    monkeypatch.setattr(checker, "FLUXION_SRC", fluxion_src)
    monkeypatch.setattr(checker, "FLUXION_CORE_SRC", fluxion_core_src)


# ---------------------------------------------------------------------------
# _normalize_pair
# ---------------------------------------------------------------------------


def test_normalize_pair_strips_crate_prefix(checker):
    a, b = checker._normalize_pair(
        "crate::sim::construction", "fluxion::physics::continuous"
    )
    assert a == "sim::construction"
    assert b == "physics::continuous"


def test_normalize_pair_strips_fluxion_core_prefix(checker):
    a, b = checker._normalize_pair("fluxion_core::assembly", "crate::weather")
    assert a == "assembly"
    assert b == "weather"


def test_normalize_pair_truncates_braces_lists(checker):
    """``fluxion::physics::{wall_spec, method_selector}`` -> ``physics``."""
    a, b = checker._normalize_pair(
        "fluxion::physics::{wall_spec, method_selector}",
        "fluxion::physics::{ctf_coefficients}",
    )
    assert a == "physics"
    assert b == "physics"


# ---------------------------------------------------------------------------
# _is_one_line_reexport_shim
# ---------------------------------------------------------------------------


def test_shim_detects_pub_use_only_file(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    p = _write(tmp_path / "shim.rs", "pub use fluxion_core::tensor::ContinuousField;\n")
    assert checker._is_one_line_reexport_shim(p) is True


def test_shim_detects_two_line_pub_use_only_file(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    p = _write(
        tmp_path / "shim.rs",
        "pub use crate::physics::ctf_coefficients;\npub use crate::physics::fd_discretization;\n",
    )
    assert checker._is_one_line_reexport_shim(p) is True


def test_shim_rejects_file_with_logic(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    p = _write(
        tmp_path / "real.rs",
        "pub fn step(&mut self, dt: f64) -> f64 { 0.0 }\n",
    )
    assert checker._is_one_line_reexport_shim(p) is False


def test_shim_ignores_attribute_lines(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    p = _write(
        tmp_path / "shim.rs",
        "//! Doc comment\n#![allow(nonstandard_style)]\npub use crate::tensor::ContinuousField;\n",
    )
    assert checker._is_one_line_reexport_shim(p) is True


# ---------------------------------------------------------------------------
# _scan_lib_rs_for_cycle_claims
# ---------------------------------------------------------------------------


def test_scan_finds_present_tense_claim(checker, tmp_path, monkeypatch):
    _redirect(checker, tmp_path, monkeypatch)
    _write(
        checker.LIB_RS,
        "//! The `sim::construction ↔ physics::continuous` cycle remains and is the next\n"
        "//! cycle-break target.\n",
    )
    claims = checker._scan_lib_rs_for_cycle_claims()
    assert len(claims) == 1
    line_no, left, right, present = claims[0]
    assert left == "sim::construction"
    assert right == "physics::continuous"
    assert present is True


def test_scan_past_tense_is_not_drift_sensitive(checker, tmp_path, monkeypatch):
    _redirect(checker, tmp_path, monkeypatch)
    _write(
        checker.LIB_RS,
        "//! Issue #2462 broke the last `physics ↔ sim` cycle.\n"
        "//! `sim::construction` Breaks 3 of 5 `physics ↔ sim` cycle edges.\n",
    )
    claims = checker._scan_lib_rs_for_cycle_claims()
    # Both are past-tense / historical — none should be drift-sensitive.
    assert all(c[3] is False for c in claims)
    assert len(claims) == 2


def test_scan_matches_two_side_backticks(checker, tmp_path, monkeypatch):
    _redirect(checker, tmp_path, monkeypatch)
    _write(
        checker.LIB_RS,
        "//! The `sim::construction` ↔ `validation::ashrae_140_cases::Orientation` cycle persists.\n",
    )
    claims = checker._scan_lib_rs_for_cycle_claims()
    assert len(claims) == 1
    _, left, right, present = claims[0]
    assert left == "sim::construction"
    assert right == "validation::ashrae_140_cases::orientation"
    assert present is True


def test_scan_matches_bare_arrow(checker, tmp_path, monkeypatch):
    _redirect(checker, tmp_path, monkeypatch)
    _write(
        checker.LIB_RS,
        "//! the physics<->sim dependency cycle remains.\n",
    )
    claims = checker._scan_lib_rs_for_cycle_claims()
    assert len(claims) == 1
    _, left, right, present = claims[0]
    assert left == "physics"
    assert right == "sim"
    assert present is True


def test_scan_missing_lib_rs_returns_empty(checker, tmp_path, monkeypatch):
    _redirect(checker, tmp_path, monkeypatch)
    # Do NOT create LIB_RS
    assert checker._scan_lib_rs_for_cycle_claims() == []


# ---------------------------------------------------------------------------
# _parse_architecture_remaining_cycles
# ---------------------------------------------------------------------------


ARCH_REMAINING_SECTION = (
    "### Remaining cycles (deferred to follow-up issues)\n\n"
    "- ~~`fluxion::sim::construction` still depends on `fluxion::physics::continuous`.~~\n"
    "  **Resolved by #2462**: moved to `fluxion_core::construction`.\n"
    "- `fluxion::physics::{wall_spec}` reference `fluxion::physics::{ctf_coefficients}`.\n\n"
    "### Next section\n"
)


def test_parse_arch_marks_strikethrough_as_resolved(checker, tmp_path, monkeypatch):
    _redirect(checker, tmp_path, monkeypatch)
    _write(checker.ARCHITECTURE_MD, ARCH_REMAINING_SECTION)
    active, resolved = checker._parse_architecture_remaining_cycles()
    # Strikethrough bullet -> resolved; non-strikethrough -> active.
    assert len(resolved) >= 1
    assert len(active) >= 1
    # The resolved set must contain a key derived from the struck-through pair.
    # The strikethrough regex extracts the first backticked module when no
    # `↔` glyph is present, so we just assert the resolved set is non-empty.
    assert any("construction" in k or "physics" in k for k in resolved)


def test_parse_arch_marks_active_bullet(checker, tmp_path, monkeypatch):
    _redirect(checker, tmp_path, monkeypatch)
    _write(
        checker.ARCHITECTURE_MD,
        "### Remaining cycles (deferred to follow-up issues)\n\n"
        "- `fluxion::sim::construction` still depends on `fluxion::physics::continuous`.\n\n"
        "### Next\n",
    )
    active, resolved = checker._parse_architecture_remaining_cycles()
    assert len(active) >= 1
    assert len(resolved) == 0


def test_parse_arch_missing_section_returns_empty(checker, tmp_path, monkeypatch):
    _redirect(checker, tmp_path, monkeypatch)
    _write(checker.ARCHITECTURE_MD, "# Title\n\nNo cycles here.\n")
    active, resolved = checker._parse_architecture_remaining_cycles()
    assert active == set()
    assert resolved == set()


# ---------------------------------------------------------------------------
# main() — end-to-end scenarios
# ---------------------------------------------------------------------------


def test_main_clean_tree_passes(checker, tmp_path, monkeypatch, capsys):
    """A tree with no drift-triggering claims returns exit 0."""
    _redirect(checker, tmp_path, monkeypatch)
    # Doc-comment with only past-tense cycle mentions.
    _write(
        checker.LIB_RS,
        "//! ## Cycle break (#2462)\n"
        "//! Issue #2462 broke the last `physics ↔ sim` cycle.\n",
    )
    _write(checker.ARCHITECTURE_MD, "# Title\n")
    assert checker.main() == 0
    out = capsys.readouterr().out
    assert "No doc-comment drift" in out


def test_main_flags_two_shim_modules(checker, tmp_path, monkeypatch):
    """Present-tense claim naming two one-line `pub use` shims -> exit 1.

    This is the exact regression that motivated issue #2895: a doc-comment
    claims the `sim::construction ↔ physics::continuous` cycle remains,
    but both modules are now one-line re-export shims.
    """
    _redirect(checker, tmp_path, monkeypatch)
    # Make physics::continuous a one-line shim (real state as of #1718).
    _write(
        tmp_path / "src" / "physics" / "continuous.rs",
        "pub use fluxion_core::tensor::ContinuousField;\n",
    )
    # Make sim::construction a one-line shim too.
    _write(
        tmp_path / "src" / "sim" / "construction.rs",
        "pub use fluxion_core::construction::ConstructionLayer;\n",
    )
    # The offending doc-comment (line 77 is the bug).
    _write(
        checker.LIB_RS,
        "//! The `sim::construction ↔ physics::continuous` cycle remains and is the next\n"
        "//! cycle-break target.\n",
    )
    # ARCHITECTURE.md says it was Resolved by #2462 (real state).
    _write(
        checker.ARCHITECTURE_MD,
        "### Remaining cycles (deferred to follow-up issues)\n\n"
        "- ~~`fluxion::sim::construction` still depends on `fluxion::physics::continuous`.~~\n"
        "  **Resolved by #2462**.\n\n",
    )
    assert checker.main() == 1


def test_main_flags_arch_resolved_pair(checker, tmp_path, monkeypatch):
    """Present-tense claim about a pair ARCHITECTURE.md marks as resolved -> exit 1."""
    _redirect(checker, tmp_path, monkeypatch)
    # No shim files; modules are real code.
    _write(tmp_path / "src" / "sim" / "construction.rs", "pub fn x() {}\n")
    _write(tmp_path / "src" / "physics" / "continuous.rs", "pub fn y() {}\n")
    # Present-tense claim about a pair ARCHITECTURE.md marks resolved.
    _write(
        checker.LIB_RS,
        "//! The `sim::construction ↔ physics::continuous` cycle remains.\n",
    )
    _write(
        checker.ARCHITECTURE_MD,
        "### Remaining cycles (deferred to follow-up issues)\n\n"
        "- ~~`sim::construction` still depends on `physics::continuous`.~~\n"
        "  **Resolved by #2462**.\n\n",
    )
    assert checker.main() == 1


def test_main_passes_when_claim_past_tense(checker, tmp_path, monkeypatch):
    """Historical past-tense claim about a resolved cycle -> exit 0."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / "src" / "sim" / "construction.rs", "pub fn x() {}\n")
    _write(tmp_path / "src" / "physics" / "continuous.rs", "pub fn y() {}\n")
    _write(
        checker.LIB_RS,
        "//! Issue #2462 broke the last `sim::construction` ↔ `physics::continuous` cycle.\n",
    )
    _write(
        checker.ARCHITECTURE_MD,
        "### Remaining cycles (deferred to follow-up issues)\n\n"
        "- ~~`sim::construction` still depends on `physics::continuous`.~~\n"
        "  **Resolved by #2462**.\n\n",
    )
    assert checker.main() == 0
