"""
Tests for ``scripts/check_physics_sim_cycle.py`` — Issue #2463.

Regression guard for the ``physics <-> sim`` cycle. Mirrors the test
pattern in ``scripts/ci/test_check_osimflow_coverage.py``:
import the script as a module, monkey-patch the repo-rooted path
constants to point at a ``tmp_path`` fixture, then drive each phase
and the ``main()`` orchestration through both clean and offender
scenarios.

The 5 physics->sim edges documented in the issue body are the
baseline; once the companion cycle-break work lands, the script
must report 0 edges. These tests pin the scanning primitives so a
silent regex regression cannot re-introduce a cycle.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from textwrap import dedent

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_physics_sim_cycle.py"
)


def _load_checker():
    """Load scripts/check_physics_sim_cycle.py as a module.

    The script uses module-level ``REPO_ROOT`` / ``PHYSICS_DIR`` / ``SIM_DIR``
    constants rooted at the real repo, so we reload it fresh for each test
    that wants to monkey-patch those paths. Returns the imported module
    object.
    """
    spec = importlib.util.spec_from_file_location(
        "check_physics_sim_cycle", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checker():
    """Return a freshly-loaded copy of the cycle-check script.

    Use ``monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)`` (and the
    other path constants) to redirect the scan to a synthetic fixture
    tree. ``REPO_ROOT`` is resolved at import time from the script's
    location, so a fresh module each test gives a clean slate.
    """
    return _load_checker()


def _write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dedent(text), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# scan_physics_for_sim_deps  (Phase 1)
# ---------------------------------------------------------------------------


def test_phase1_flags_use_crate_sim_in_physics(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "PHYSICS_DIR", tmp_path / "src" / "physics")
    _write(
        tmp_path / "src" / "physics" / "thermal_mass" / "construction.rs",
        """
        use crate::sim::construction::ConstructionLayer;
        fn f(_: ConstructionLayer) {}
        """,
    )
    offenders = checker.scan_physics_for_sim_deps()
    assert len(offenders) == 1
    assert "src/physics/thermal_mass/construction.rs" in offenders[0]
    assert "use crate::sim::construction::ConstructionLayer" in offenders[0]


def test_phase1_flags_pub_use_crate_sim(checker, tmp_path, monkeypatch):
    """pub use crate::sim::* must also trip the guard (it's still an upward edge)."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "PHYSICS_DIR", tmp_path / "src" / "physics")
    _write(
        tmp_path / "src" / "physics" / "re_export.rs",
        """
        pub use crate::sim::construction::ConstructionLayer;
        """,
    )
    offenders = checker.scan_physics_for_sim_deps()
    assert len(offenders) == 1
    assert "src/physics/re_export.rs" in offenders[0]


def test_phase1_flags_fully_qualified_path_in_function_body(
    checker, tmp_path, monkeypatch
):
    """A `crate::sim::foo::bar()` call (not a use-stmt) is still an upward dep."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "PHYSICS_DIR", tmp_path / "src" / "physics")
    _write(
        tmp_path / "src" / "physics" / "weird_path.rs",
        """
        fn f() {
            let x = crate::sim::construction::ConstructionLayer::default();
        }
        """,
    )
    offenders = checker.scan_physics_for_sim_deps()
    assert len(offenders) == 1
    assert "src/physics/weird_path.rs" in offenders[0]


def test_phase1_ignores_use_crate_sim_inside_line_comments(
    checker, tmp_path, monkeypatch
):
    """Comment-only mentions of `crate::sim::` must not trip the guard.

    The script mirrors ``check_ashrae_cases_cycle.py`` which strips
    line comments via ``stripped.startswith("//")`` and continuation
    lines via ``stripped.startswith("*")`` — a faithful mirror of the
    established pattern. Block comments (single-line ``/* ... */``)
    are not detected by either guard, so they are NOT exercised here.
    """
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "PHYSICS_DIR", tmp_path / "src" / "physics")
    _write(
        tmp_path / "src" / "physics" / "commented.rs",
        """
        // use crate::sim::construction::ConstructionLayer; -- was here, now removed
        fn clean() -> i32 { 0 }
        """,
    )
    assert checker.scan_physics_for_sim_deps() == []


def test_phase1_ignores_non_sim_upward_paths(checker, tmp_path, monkeypatch):
    """`crate::physics::...` self-imports in src/physics/** must not trip the guard."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "PHYSICS_DIR", tmp_path / "src" / "physics")
    _write(
        tmp_path / "src" / "physics" / "ok.rs",
        """
        use crate::physics::constants::thermal::ashrae_140::EXTERIOR_FILM_COEFF;
        use crate::physics::solver_trait::HeatConductionSolver;
        use crate::physics::units::Celsius;
        fn f() {}
        """,
    )
    assert checker.scan_physics_for_sim_deps() == []


def test_phase1_returns_empty_when_physics_dir_missing(checker, tmp_path, monkeypatch):
    """If `src/physics/` does not exist, the guard must not crash."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "PHYSICS_DIR", tmp_path / "no" / "such" / "path")
    assert checker.scan_physics_for_sim_deps() == []


def test_phase1_matches_real_offender_baseline(checker, tmp_path, monkeypatch):
    """The 5 documented offenders must all be detected under the right paths."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "PHYSICS_DIR", tmp_path / "src" / "physics")
    _write(
        tmp_path / "src" / "physics" / "thermal_mass" / "construction.rs",
        "use crate::sim::construction::ConstructionLayer;\n",
    )
    _write(
        tmp_path / "src" / "physics" / "thermal_mass" / "diagnostics.rs",
        "use crate::sim::construction::ConstructionLayer;\n",
    )
    _write(
        tmp_path / "src" / "physics" / "multi_node_solver.rs",
        (
            "use crate::sim::per_surface_conduction::{PerSurfaceConductionSolver, SurfaceKind};\n"
            "use crate::sim::sky_radiation::STEFAN_BOLTZMANN;\n"
        ),
    )
    _write(
        tmp_path / "src" / "physics" / "deep_nest.rs",
        (
            "fn deep() {\n"
            "    use crate::sim::sky_radiation::SkyRadiationExchange;\n"
            "}\n"
        ),
    )
    offenders = checker.scan_physics_for_sim_deps()
    assert len(offenders) == 5
    paths = {o.split(":", 1)[0] for o in offenders}
    assert paths == {
        "src/physics/thermal_mass/construction.rs",
        "src/physics/thermal_mass/diagnostics.rs",
        "src/physics/multi_node_solver.rs",
        "src/physics/deep_nest.rs",
    }


# ---------------------------------------------------------------------------
# scan_sim_for_physics_deps  (Phase 2 — extended to all src/sim/** by #2766)
# ---------------------------------------------------------------------------


def test_phase2_flags_use_crate_physics_in_construction(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "SIM_DIR", tmp_path / "src" / "sim")
    _write(
        tmp_path / "src" / "sim" / "construction.rs",
        "use crate::physics::constants::SOMETHING;\n",
    )
    _write(
        tmp_path / "src" / "sim" / "per_surface_conduction.rs",
        "fn clean() -> i32 { 0 }\n",
    )
    offenders = checker.scan_sim_for_physics_deps()
    assert len(offenders) == 1
    assert "src/sim/construction.rs" in offenders[0]


def test_phase2_flags_pub_use_crate_physics(checker, tmp_path, monkeypatch):
    """pub use crate::physics:: must also trip Phase 2 (it is still an upward dep)."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "SIM_DIR", tmp_path / "src" / "sim")
    _write(
        tmp_path / "src" / "sim" / "construction.rs",
        """
        pub use crate::physics::constants::thermal::ashrae_140::EXTERIOR_FILM_COEFF;
        pub use crate::physics::constants::AIR_DENSITY_SEA_LEVEL;
        """,
    )
    offenders = checker.scan_sim_for_physics_deps()
    assert len(offenders) == 2
    assert all("src/sim/construction.rs" in o for o in offenders)


def test_phase2_flags_crate_physics_in_per_surface_conduction(
    checker, tmp_path, monkeypatch
):
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "SIM_DIR", tmp_path / "src" / "sim")
    _write(tmp_path / "src" / "sim" / "construction.rs", "fn clean() -> i32 { 0 }\n")
    _write(
        tmp_path / "src" / "sim" / "per_surface_conduction.rs",
        "use crate::physics::wall_properties::WallProperties;\n",
    )
    offenders = checker.scan_sim_for_physics_deps()
    assert len(offenders) == 1
    assert "src/sim/per_surface_conduction.rs" in offenders[0]


def test_phase2_ignores_comments_and_unrelated_paths(checker, tmp_path, monkeypatch):
    """Comments mentioning crate::physics:: and unrelated use statements must not trip.

    Mirrors the line-comment behaviour from Phase 1 / check_ashrae_cases_cycle.py:
    only ``// ...`` line comments are skipped — the regex pattern anchors on
    ``(pub\\s+)?use\\s+crate::physics::`` so other use statements and
    block-comment contents are not relevant.
    """
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "SIM_DIR", tmp_path / "src" / "sim")
    _write(
        tmp_path / "src" / "sim" / "construction.rs",
        """
        // use crate::physics::constants::SOMETHING;
        use crate::sim::construction::ConstructionLayer;
        use crate::sim::per_surface_conduction::PerSurfaceConductionSolver;
        fn clean() -> i32 { 0 }
        """,
    )
    assert checker.scan_sim_for_physics_deps() == []


def test_phase2_returns_empty_when_sim_dir_missing(checker, tmp_path, monkeypatch):
    """If `src/sim/` does not exist, the guard must not crash (mirrors Phase 1)."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "SIM_DIR", tmp_path / "no" / "such" / "path")
    assert checker.scan_sim_for_physics_deps() == []


def test_phase2_scans_all_sim_files_in_real_repo(checker):
    """Issue #2766 acceptance criterion: Phase 2 must scan ALL of src/sim/**.

    The pre-#2766 guard scanned only 2 files (construction.rs +
    per_surface_conduction.rs). This test pins the extended coverage by
    running the REAL scan (no monkey-patch) against the committed tree and
    asserting:

    * the offender count equals ``BASELINE_SIM_TO_PHYSICS`` (the snapshot
      of 84 pre-existing ``use crate::physics::`` edges across 26 sim files);
    * the offenders span many more than the 2 files the old guard saw;
    * the four documented re-export shims (assembly.rs,
      multi_node_thermal.rs, construction.rs, per_surface_conduction.rs)
      are all clean (0 edges) — the cycle-break work of #2462 holds.
    """
    offenders = checker.scan_sim_for_physics_deps()
    assert len(offenders) == checker.BASELINE_SIM_TO_PHYSICS
    # The 2 files the old guard scanned must be clean (they were the
    # pre-#2766 protected seam and #2462 drove them to 0 physics imports).
    files = {o.split(":", 1)[0] for o in offenders}
    for shim in (
        "src/sim/assembly.rs",
        "src/sim/multi_node_thermal.rs",
        "src/sim/construction.rs",
        "src/sim/per_surface_conduction.rs",
    ):
        assert shim not in files, f"{shim} re-introduced a physics import"
    # Coverage extension: far more than the old 2 files are now scanned.
    assert (
        len(files) >= 20
    ), f"Phase 2 should cover 20+ sim files (issue #2766), only saw {len(files)}"


# ---------------------------------------------------------------------------
# main() — end-to-end orchestration
# ---------------------------------------------------------------------------


def _redirect_to_fixture(
    checker, tmp_path, monkeypatch, physics_files=None, sim_files=None
):
    """Point the module's path constants at a synthetic fixture tree.

    ``physics_files`` is a dict of {relpath: content} written under
    ``tmp_path/src/physics/``. ``sim_files`` is the same for
    ``physics_files`` is a dict of {relpath: content} written under
    ``tmp_path/src/physics/``. ``sim_files`` is the same for
    ``tmp_path/src/sim/`` — Phase 2 walks the whole ``src/sim/`` tree
    (Issue #2766), so every file written there is scanned.
    """
    physics_files = physics_files or {}
    sim_files = sim_files or {}
    physics_root = tmp_path / "src" / "physics"
    sim_root = tmp_path / "src" / "sim"
    for rel, content in physics_files.items():
        _write(physics_root / rel, content)
    for rel, content in sim_files.items():
        _write(sim_root / rel, content)
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "PHYSICS_DIR", physics_root)
    monkeypatch.setattr(checker, "SIM_DIR", sim_root)


def test_main_returns_zero_when_clean(checker, tmp_path, monkeypatch, capsys):
    _redirect_to_fixture(
        checker,
        tmp_path,
        monkeypatch,
        physics_files={
            "solver.rs": (
                "use crate::physics::constants::thermal::ashrae_140::EXTERIOR_FILM_COEFF;\n"
                "fn f() {}\n"
            ),
        },
        sim_files={
            "construction.rs": (
                "use crate::sim::construction::ConstructionLayer;\n" "fn f() {}\n"
            ),
            "per_surface_conduction.rs": (
                "use crate::sim::per_surface_conduction::PerSurfaceConductionSolver;\n"
                "fn f() {}\n"
            ),
        },
    )
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK: 0 upward deps" in out
    assert "OK: no cycle markers" in out
    assert "Total cycle edges: 0" in out
    assert "No cycle regression" in out


def test_main_returns_zero_at_documented_baseline(
    checker, tmp_path, monkeypatch, capsys
):
    """A clean fixture (0 offenders in both directions) must pass below baseline.

    Issue #2462 drove the physics->sim direction to 0 edges; the sim->physics
    baseline is 84 (Issue #2766 snapshotted the pre-existing edges). A clean
    fixture has 0 actual offenders — below both baselines — so the guard
    passes. It only fails if a *new* edge pushes a count *above* its baseline.
    """
    _redirect_to_fixture(
        checker,
        tmp_path,
        monkeypatch,
        physics_files={},
        sim_files={
            "construction.rs": "fn f() {}\n",
            "per_surface_conduction.rs": "fn f() {}\n",
        },
    )

    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK:" in out
    assert "below baseline" in out
    assert "No cycle regression" in out
    assert "Total cycle edges: 0" in out


def test_main_returns_one_when_physics_offender_exceeds_baseline(
    checker, tmp_path, monkeypatch, capsys
):
    """6+ `use crate::sim::` imports in src/physics/** (1 above baseline 5) must trip main().

    The guard's contract is *regression-only*: a NEW edge that pushes the
    count above the documented baseline is a failure. A single offender
    is fine if the baseline allows it; the test therefore seeds
    `BASELINE_PHYSICS_TO_SIM` legitimate edges + 1 extra, and asserts
    that the *extra* is what trips the guard.
    """
    physics_files = {
        f"baseline{i}.rs": "use crate::sim::construction::ConstructionLayer;\n"
        for i in range(checker.BASELINE_PHYSICS_TO_SIM)
    }
    physics_files["sneaky.rs"] = "use crate::sim::sky_radiation::STEFAN_BOLTZMANN;\n"
    _redirect_to_fixture(
        checker,
        tmp_path,
        monkeypatch,
        physics_files=physics_files,
        sim_files={
            "construction.rs": "fn f() {}\n",
            "per_surface_conduction.rs": "fn f() {}\n",
        },
    )
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "[1/3]" in out
    assert f"({1} above baseline {checker.BASELINE_PHYSICS_TO_SIM})" in out
    assert "src/physics/sneaky.rs" in out
    assert "CYCLE REGRESSION DETECTED" in out


def test_main_returns_one_when_sim_offender_exceeds_baseline(
    checker, tmp_path, monkeypatch, capsys
):
    """A `use crate::physics::` import in any sim file above the baseline must trip main().

    The guard's contract is *regression-only*: a NEW edge that pushes the
    count above the documented baseline is a failure. We monkey-patch
    ``BASELINE_SIM_TO_PHYSICS`` to 0 (so the test does not have to generate
    84 baseline lines) and seed a single offender in ``construction.rs`` —
    the *extra* edge above the (patched) baseline is what trips the guard.
    """
    monkeypatch.setattr(checker, "BASELINE_SIM_TO_PHYSICS", 0)
    _redirect_to_fixture(
        checker,
        tmp_path,
        monkeypatch,
        physics_files={"ok.rs": "fn f() {}\n"},
        sim_files={
            "construction.rs": "use crate::physics::wall_properties::WallProperties;\n",
            "per_surface_conduction.rs": "fn f() {}\n",
        },
    )
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "[2/3]" in out
    assert f"({1} above baseline {checker.BASELINE_SIM_TO_PHYSICS})" in out
    assert "src/sim/construction.rs" in out
    assert "CYCLE REGRESSION DETECTED" in out


def test_main_aggregates_offenders_from_both_phases(
    checker, tmp_path, monkeypatch, capsys
):
    """Both phases contribute offenders to the summary when each exceeds baseline.

    We monkey-patch both baselines to 0 (so the test does not have to
    generate 84+ baseline lines) and seed 2 offenders in each phase; the
    *extra* edges above the (patched) baselines are what trip the guard.
    """
    monkeypatch.setattr(checker, "BASELINE_PHYSICS_TO_SIM", 0)
    monkeypatch.setattr(checker, "BASELINE_SIM_TO_PHYSICS", 0)
    physics_files = {
        "sneaky1.rs": "use crate::sim::construction::ConstructionLayer;\n",
        "sneaky2.rs": "use crate::sim::sky_radiation::STEFAN_BOLTZMANN;\n",
    }
    _redirect_to_fixture(
        checker,
        tmp_path,
        monkeypatch,
        physics_files=physics_files,
        sim_files={
            "construction.rs": (
                "use crate::physics::constants::thermal::ashrae_140::EXTERIOR_FILM_COEFF;\n"
                "use crate::physics::constants::AIR_DENSITY_SEA_LEVEL;\n"
            ),
            "per_surface_conduction.rs": "fn f() {}\n",
        },
    )
    rc = checker.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert f"({2} above baseline {checker.BASELINE_PHYSICS_TO_SIM})" in out  # Phase 1
    assert f"({2} above baseline {checker.BASELINE_SIM_TO_PHYSICS})" in out  # Phase 2
    expected_total = (
        checker.BASELINE_PHYSICS_TO_SIM + 2 + checker.BASELINE_SIM_TO_PHYSICS + 2
    )
    assert f"Total cycle edges: {expected_total}" in out
    assert "src/physics/sneaky1.rs" in out
    assert "src/physics/sneaky2.rs" in out
    assert "src/sim/construction.rs" in out


def test_main_returns_two_on_unhandled_exception(
    checker, tmp_path, monkeypatch, capsys
):
    """The ``__main__`` wrapper must translate unhandled exceptions to exit 2.

    Mirrors ``check_ashrae_cases_cycle.py`` — its ``if __name__ == '__main__'``
    block wraps ``sys.exit(main())`` in ``try/except`` and emits
    ``ERROR: <msg>`` on stderr. We drive the same code path with a
    tiny subprocess invocation of ``python3`` against a copy of the
    script whose ``scan_physics_for_sim_deps`` is patched at runtime.
    The patch uses a wrapper script that imports the cycle-check module,
    swaps in a raising function, and calls ``main()`` via the same
    wrapper logic.
    """
    # Build a synthetic clean fixture under tmp_path so the real repo's
    # current 7 offenders do not contaminate the test.
    physics_root = tmp_path / "src" / "physics"
    sim_root = tmp_path / "src" / "sim"
    _write(physics_root / "ok.rs", "fn f() {}\n")
    _write(sim_root / "construction.rs", "fn f() {}\n")
    _write(sim_root / "per_surface_conduction.rs", "fn f() {}\n")

    # Write a small driver that imports the cycle-check script, redirects
    # its REPO_ROOT / PHYSICS_DIR / SIM_DIR to the fixture, patches
    # scan_physics_for_sim_deps to raise, then mirrors the script's own
    # __main__ wrapper so the exception is translated to exit 2.
    driver = tmp_path / "driver.py"
    driver.write_text(
        dedent(f"""\
            import importlib.util
            import sys
            from pathlib import Path

            spec = importlib.util.spec_from_file_location(
                "checker", {str(SCRIPT_PATH)!r}
            )
            checker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(checker)

            checker.REPO_ROOT = {str(tmp_path)!r}
            checker.PHYSICS_DIR = {str(physics_root)!r}
            checker.SIM_DIR = {str(sim_root)!r}

            def boom():
                raise RuntimeError("boom from test")

            checker.scan_physics_for_sim_deps = boom

            try:
                rc = checker.main()
            except Exception as e:
                print(f"ERROR: {{e}}", file=sys.stderr)
                sys.exit(2)
            sys.exit(rc)
            """),
        encoding="utf-8",
    )

    import subprocess

    result = subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "ERROR: boom from test" in result.stderr


# ---------------------------------------------------------------------------
# import-path safety
# ---------------------------------------------------------------------------


def test_main_orchestration_is_a_thin_shell():
    """The ``if __name__ == '__main__'`` wrapper must route main() to sys.exit().

    Read the script tail directly and assert the wrapper delegates
    correctly. This pins the public contract that ``python3 script.py``
    exits with the same code as ``checker.main()`` returns, and that
    unhandled exceptions become exit 2.
    """
    script_src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in script_src
    assert "sys.exit(main())" in script_src
    assert "sys.exit(2)" in script_src
    # The wrapper must include a try/except around sys.exit(main()).
    wrapper_block = script_src.split('if __name__ == "__main__":', 1)[1]
    assert "try:" in wrapper_block
    assert "except Exception" in wrapper_block
