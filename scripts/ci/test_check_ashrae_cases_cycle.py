"""
Tests for ``scripts/check_ashrae_cases_cycle.py`` -- Issues #1441 + #2495.

Regression guard for the ``sim <-> validation`` cycle. These tests pin the
scan primitives, the leaf-type structural verification, and the at-or-below
baseline rule so a silent regex/allow-list regression cannot re-introduce a
cycle edge or disable enforcement.

Pattern (mirrors ``test_check_physics_sim_cycle.py``): load the script as a
fresh module, redirect its module-level path constants
(``REPO_ROOT`` / ``FLUXION_CORE_SRC`` / ``SIM_DIR`` / ``VALIDATION_DIR`` /
``ASHRAE_CASES_FILE``) at a ``tmp_path`` fixture tree, then drive each
scanner + ``main()`` through clean and offender scenarios.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "check_ashrae_cases_cycle"


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the cycle-check script."""
    return load_script(SCRIPT_NAME)


def _write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _redirect(checker, tmp_path, monkeypatch) -> None:
    """Point every module-level path constant at the ``tmp_path`` mock repo."""
    fluxion_core_src = tmp_path / "fluxion-core" / "src"
    fluxion_src = tmp_path / "src"
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "FLUXION_CORE_SRC", fluxion_core_src)
    monkeypatch.setattr(checker, "FLUXION_SRC", fluxion_src)
    monkeypatch.setattr(checker, "SIM_DIR", fluxion_src / "sim")
    monkeypatch.setattr(checker, "VALIDATION_DIR", fluxion_src / "validation")
    monkeypatch.setattr(
        checker, "ASHRAE_CASES_FILE", fluxion_core_src / "ashrae_cases.rs"
    )


# ---------------------------------------------------------------------------
# _scan_dir_for_prefixes
# ---------------------------------------------------------------------------


def test_scan_dir_clean_returns_empty(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    d = tmp_path / "src" / "sim"
    _write(d / "clean.rs", "fn f() -> i32 { 0 }\n")
    assert checker._scan_dir_for_prefixes(d, ("crate::validation::",)) == []


def test_scan_dir_flags_use_crate_validation(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    d = tmp_path / "src" / "sim"
    _write(
        d / "offender.rs",
        "use crate::validation::ashrae_140_cases::CaseSpec;\n",
    )
    out = checker._scan_dir_for_prefixes(d, ("crate::validation::",))
    assert len(out) == 1
    # offender format is "<rel>:<lineno>: <line>"
    rel, lineno, _ = out[0].split(":", 2)
    assert rel == str((d / "offender.rs").relative_to(tmp_path))
    assert lineno == "1"


def test_scan_dir_ignores_line_comments(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    d = tmp_path / "src" / "sim"
    _write(
        d / "commented.rs",
        "// use crate::validation::ashrae_140_cases::CaseSpec;\nfn f() {}\n",
    )
    assert checker._scan_dir_for_prefixes(d, ("crate::validation::",)) == []


def test_scan_dir_ignores_block_comment_continuations(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    d = tmp_path / "src" / "sim"
    _write(
        d / "block.rs",
        "/*\n * use crate::validation::diagnostics::Foo;\n */\nfn f() {}\n",
    )
    assert checker._scan_dir_for_prefixes(d, ("crate::validation::",)) == []


def test_scan_dir_catches_fully_qualified_usage_not_just_use(
    checker, tmp_path, monkeypatch
):
    """The cycle is driven equally by match arms / expressions, not only `use`."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    d = tmp_path / "src" / "sim"
    _write(
        d / "match.rs",
        "fn m(c: CaseSpec) { match c { crate::validation::X::A => 1 } }\n",
    )
    out = checker._scan_dir_for_prefixes(d, ("crate::validation::",))
    assert len(out) == 1


def test_scan_dir_missing_directory_returns_empty(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    assert (
        checker._scan_dir_for_prefixes(
            tmp_path / "no" / "such" / "dir", ("crate::validation::",)
        )
        == []
    )


# ---------------------------------------------------------------------------
# scan_fluxion_core_for_upward_deps  (fluxion-core must stay acyclic)
# ---------------------------------------------------------------------------


def test_fluxion_core_clean_when_no_upward_refs(checker, tmp_path, monkeypatch):
    _redirect(checker, tmp_path, monkeypatch)
    _write(
        tmp_path / "fluxion-core" / "src" / "ashrae_cases.rs",
        "pub enum Orientation { North, South }\n",
    )
    assert checker.scan_fluxion_core_for_upward_deps() == []


@pytest.mark.parametrize(
    "upward",
    [
        "use crate::sim::construction::ConstructionLayer;",
        "use crate::physics::solver_trait::HeatConductionSolver;",
        "use crate::ai::surrogate::Surrogate;",
        "use crate::validation::ashrae_140_cases::CaseSpec;",
    ],
)
def test_fluxion_core_flags_each_upward_prefix(checker, tmp_path, monkeypatch, upward):
    _redirect(checker, tmp_path, monkeypatch)
    _write(tmp_path / "fluxion-core" / "src" / "bad.rs", upward + "\n")
    out = checker.scan_fluxion_core_for_upward_deps()
    assert len(out) == 1


# ---------------------------------------------------------------------------
# verify_ashrae_cases_module  (all 13 moved leaf types must be present)
# ---------------------------------------------------------------------------


def _all_leaf_types_source() -> str:
    """Synthesize a source file declaring every moved leaf type."""
    types = [
        "Orientation",
        "WindowArea",
        "ConstructionType",
        "ShadingType",
        "ShadingDevice",
        "GlassType",
        "WindowSpec",
        "InternalLoads",
        "HvacSchedule",
        "NightVentilation",
        "BuildingType",
        "GeometrySpec",
        "ConductanceReferences",
    ]
    return "\n".join(f"pub enum {t} {{ Variant }}\n" for t in types)


def test_verify_module_clean_when_all_leaf_types_present(
    checker, tmp_path, monkeypatch
):
    _redirect(checker, tmp_path, monkeypatch)
    _write(checker.ASHRAE_CASES_FILE, _all_leaf_types_source())
    assert checker.verify_ashrae_cases_module() == []


def test_verify_module_reports_missing_file(checker, tmp_path, monkeypatch):
    _redirect(checker, tmp_path, monkeypatch)
    # ASHRAE_CASES_FILE left non-existent
    out = checker.verify_ashrae_cases_module()
    assert len(out) == 1
    assert "missing" in out[0]


def test_verify_module_reports_each_missing_type(checker, tmp_path, monkeypatch):
    _redirect(checker, tmp_path, monkeypatch)
    # Declare only one leaf type -> the other 12 are reported missing.
    _write(checker.ASHRAE_CASES_FILE, "pub enum Orientation { North }\n")
    out = checker.verify_ashrae_cases_module()
    assert len(out) == len(checker.MOVED_LEAF_TYPES) - 1
    assert all("missing pub" in line for line in out)
    assert not any("Orientation" in line for line in out)


def test_verify_module_accepts_struct_or_enum(checker, tmp_path, monkeypatch):
    _redirect(checker, tmp_path, monkeypatch)
    # Mix structs and enums — both satisfy `pub (enum|struct) Name`.
    src = "pub struct Orientation { x: f64 }\n" + _all_leaf_types_source().replace(
        "pub enum Orientation", "pub struct Orientation"
    )
    _write(checker.ASHRAE_CASES_FILE, src)
    assert checker.verify_ashrae_cases_module() == []


# ---------------------------------------------------------------------------
# _check_baseline  (at-or-below baseline rule)
# ---------------------------------------------------------------------------


def test_check_baseline_below_appends_no_failure(checker, capsys):
    failures: list[str] = []
    checker._check_baseline("edge", [], 5, failures)
    assert failures == []
    assert "below baseline 5" in capsys.readouterr().out


def test_check_baseline_at_appends_no_failure(checker, capsys):
    failures: list[str] = []
    checker._check_baseline("edge", ["a", "b"], 2, failures)
    assert failures == []
    assert "at baseline 2" in capsys.readouterr().out


def test_check_baseline_above_appends_failure_with_new_count(checker, capsys):
    failures: list[str] = []
    checker._check_baseline("edge", ["a", "b", "c"], 1, failures)
    # one failure header + the 3 offender lines
    assert len(failures) == 4
    assert "2 NEW edge(s) above baseline" in failures[0]
    assert "above baseline 1" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main()  orchestration
# ---------------------------------------------------------------------------


def test_main_returns_zero_on_clean_fixture(checker, tmp_path, monkeypatch, capsys):
    _redirect(checker, tmp_path, monkeypatch)
    # All leaf types present + no cycle edges anywhere.
    _write(checker.ASHRAE_CASES_FILE, _all_leaf_types_source())
    rc = checker.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "No cycle regression" in out


def test_main_returns_one_when_fluxion_core_has_upward_dep(
    checker, tmp_path, monkeypatch, capsys
):
    _redirect(checker, tmp_path, monkeypatch)
    _write(checker.ASHRAE_CASES_FILE, _all_leaf_types_source())
    _write(
        tmp_path / "fluxion-core" / "src" / "bad.rs",
        "use crate::sim::construction::ConstructionLayer;\n",
    )
    rc = checker.main()
    assert rc == 1
    out = capsys.readouterr().out
    assert "CYCLE REGRESSION DETECTED" in out


def test_main_returns_one_when_sim_to_validation_exceeds_baseline(
    checker, tmp_path, monkeypatch, capsys
):
    _redirect(checker, tmp_path, monkeypatch)
    _write(checker.ASHRAE_CASES_FILE, _all_leaf_types_source())
    # Plant (baseline + 1) crate::validation:: references under src/sim.
    base = checker.BASELINE_SIM_TO_VALIDATION
    lines = [f"use crate::validation::t{i}::X;\n" for i in range(base + 1)]
    _write(checker.SIM_DIR / "burst.rs", "".join(lines))
    rc = checker.main()
    assert rc == 1


def test_main_returns_one_when_leaf_type_missing(
    checker, tmp_path, monkeypatch, capsys
):
    _redirect(checker, tmp_path, monkeypatch)
    _write(checker.ASHRAE_CASES_FILE, "pub enum Orientation { North }\n")
    rc = checker.main()
    assert rc == 1
    assert "CYCLE REGRESSION DETECTED" in capsys.readouterr().out


def test_main_passes_at_documented_baseline_on_real_repo(checker, repo_root, capsys):
    """Smoke test: against the real checkout the gate is a required-check that
    stays green, so ``main()`` must return 0. A scanner regex regression that
    silently inflated an edge count would flip this to 1."""
    rc = checker.main()
    assert rc == 0, capsys.readouterr().out
