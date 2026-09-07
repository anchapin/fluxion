"""
Tests for ``scripts/check_architecture_drift.py`` -- the Architecture Drift
Detection gate (nightly + on ``src/**/*.rs`` / ``ARCHITECTURE.md`` changes).

These tests pin the *pure* trait-contract primitives that the gate is built
on: Rust-trait parsing, contract (de)serialization, drift detection, the
receiver invariants, and the ARCHITECTURE.md documentation extractors. A
regex regression in any of them would silently disable drift enforcement.

Unlike the cycle-gate tests these functions take strings/dicts directly, so
no ``tmp_path`` redirect is required -- the inputs are synthetic trait
bodies and markdown fragments.
"""

from __future__ import annotations

from textwrap import dedent

import pytest

SCRIPT_NAME = "check_architecture_drift"


@pytest.fixture
def drift(load_script):
    return load_script(SCRIPT_NAME)


# ---------------------------------------------------------------------------
# parse_trait_methods
# ---------------------------------------------------------------------------

TRAIT_SRC = dedent(
    """\
    pub trait HeatConductionSolver: Send + Sync {
        /// Advance the solver one timestep.
        fn step(&mut self, dt: f64) -> f64;
        fn steady_state_flux(&self) -> f64;
        fn energy_storage_rate(&self) -> f64;
    }
    """
)


def test_parse_trait_methods_extracts_signatures(drift):
    methods = drift.parse_trait_methods(TRAIT_SRC, "HeatConductionSolver")
    assert set(methods) == {"step", "steady_state_flux", "energy_storage_rate"}
    assert methods["step"].receiver == "&mut self"
    assert methods["step"].params == ["dt: f64"]
    assert methods["steady_state_flux"].receiver == "&self"
    assert methods["steady_state_flux"].params == []


def test_parse_trait_methods_missing_trait_returns_empty(drift):
    assert drift.parse_trait_methods(TRAIT_SRC, "DoesNotExist") == {}


def test_parse_trait_methods_parses_multiple_params_and_return_types(drift):
    """Realistic declaration-only trait: `fn name(&self, a: T, b: U) -> R;`."""
    src = dedent(
        """\
        pub trait ZoneSolver {
            fn solve(&mut self, dt: f64, steps: usize) -> Result<(), SimError>;
            fn current_temp(&self, zone: usize) -> Celsius;
        }
        """
    )
    methods = drift.parse_trait_methods(src, "ZoneSolver")
    assert set(methods) == {"solve", "current_temp"}
    assert methods["solve"].receiver == "&mut self"
    assert methods["solve"].params == ["dt: f64", "steps: usize"]
    assert "Result" in methods["solve"].return_type
    assert methods["current_temp"].params == ["zone: usize"]


def test_parse_trait_methods_skips_doc_comments(drift):
    src = dedent(
        """\
        pub trait Foo {
            /// fn fake_that_starts_with_fn_in_a_doc_comment() -> ();
            fn real(&self) -> i32;
        }
        """
    )
    methods = drift.parse_trait_methods(src, "Foo")
    assert set(methods) == {"real"}


# ---------------------------------------------------------------------------
# serialize_contract / deserialize_contract round-trip
# ---------------------------------------------------------------------------


def test_contract_roundtrip_preserves_methods(drift):
    methods = drift.parse_trait_methods(TRAIT_SRC, "HeatConductionSolver")
    contract = drift.TraitContract(
        trait_name="HeatConductionSolver",
        source_file="src/physics/solver_trait.rs",
        methods=methods,
    )
    restored = drift.deserialize_contract(drift.serialize_contract(contract))
    assert restored.trait_name == contract.trait_name
    assert restored.source_file == contract.source_file
    assert set(restored.methods) == set(contract.methods)
    assert restored.methods["step"].receiver == "&mut self"
    assert restored.methods["step"].params == contract.methods["step"].params


# ---------------------------------------------------------------------------
# check_contract_drift
# ---------------------------------------------------------------------------


def _contract(drift, name, methods):
    return drift.TraitContract(trait_name=name, source_file="src/x.rs", methods=methods)


def _sig(drift, receiver="&self", params=None, ret="-> f64"):
    return drift.MethodSignature(
        name="m", receiver=receiver, params=params or [], return_type=ret
    )


def test_drift_no_violations_when_identical(drift):
    c = {"T": _contract(drift, "T", {"m": _sig(drift)})}
    assert drift.check_contract_drift(c, c) == []


def test_drift_flags_new_method(drift):
    baseline = {"T": _contract(drift, "T", {"m": _sig(drift)})}
    current = {"T": _contract(drift, "T", {"m": _sig(drift), "new": _sig(drift)})}
    out = drift.check_contract_drift(current, baseline)
    assert len(out) == 1
    assert "New method `new`" in out[0]


def test_drift_flags_removed_method(drift):
    baseline = {"T": _contract(drift, "T", {"m": _sig(drift), "old": _sig(drift)})}
    current = {"T": _contract(drift, "T", {"m": _sig(drift)})}
    out = drift.check_contract_drift(current, baseline)
    assert any("removed from" in v and "`old`" in v for v in out)


def test_drift_flags_receiver_change(drift):
    baseline = {"T": _contract(drift, "T", {"m": _sig(drift, receiver="&self")})}
    current = {"T": _contract(drift, "T", {"m": _sig(drift, receiver="&mut self")})}
    out = drift.check_contract_drift(current, baseline)
    assert any("receiver changed" in v for v in out)


def test_drift_flags_return_type_change(drift):
    baseline = {"T": _contract(drift, "T", {"m": _sig(drift, ret="-> f64")})}
    current = {"T": _contract(drift, "T", {"m": _sig(drift, ret="-> i32")})}
    out = drift.check_contract_drift(current, baseline)
    assert any("return type changed" in v for v in out)


def test_drift_ignores_trait_absent_from_baseline(drift):
    # A brand-new trait (not in baseline) is not a drift by itself.
    baseline = {}
    current = {"T": _contract(drift, "T", {"m": _sig(drift)})}
    assert drift.check_contract_drift(current, baseline) == []


# ---------------------------------------------------------------------------
# check_trait_invariants
# ---------------------------------------------------------------------------


def test_invariants_pass_for_well_formed_heat_solver(drift):
    contracts = {
        "HeatConductionSolver": _contract(
            drift,
            "HeatConductionSolver",
            {
                "step": _sig(drift, receiver="&mut self"),
                "steady_state_flux": _sig(drift, receiver="&self"),
                "energy_storage_rate": _sig(drift, receiver="&self"),
            },
        )
    }
    assert drift.check_trait_invariants(contracts) == []


def test_invariants_flag_step_not_mut_self(drift):
    contracts = {
        "HeatConductionSolver": _contract(
            drift,
            "HeatConductionSolver",
            {"step": _sig(drift, receiver="&self")},
        )
    }
    out = drift.check_trait_invariants(contracts)
    assert len(out) == 1
    assert "step must be `&mut self`" in out[0]


def test_invariants_flag_steady_state_flux_not_ref_self(drift):
    contracts = {
        "HeatConductionSolver": _contract(
            drift,
            "HeatConductionSolver",
            {"steady_state_flux": _sig(drift, receiver="&mut self")},
        )
    }
    out = drift.check_trait_invariants(contracts)
    assert any("steady_state_flux must be `&self`" in v for v in out)


def test_invariants_flag_ventilation_get_ach_not_ref_self(drift):
    contracts = {
        "VentilationSchedule": _contract(
            drift,
            "VentilationSchedule",
            {"get_ach": _sig(drift, receiver="&mut self")},
        )
    }
    out = drift.check_trait_invariants(contracts)
    assert any("get_ach must be `&self`" in v for v in out)


# ---------------------------------------------------------------------------
# extract_documented_traits / extract_documented_files
# ---------------------------------------------------------------------------


def test_extract_documented_traits_catches_backticked_suffixes(drift):
    md = (
        "The swap-point traits `HeatConductionSolver`, `VentilationSchedule`, "
        "and `SolarSource` are documented here.\n"
    )
    traits = drift.extract_documented_traits(md)
    assert "HeatConductionSolver" in traits
    assert "VentilationSchedule" in traits
    assert "SolarSource" in traits


def test_extract_documented_traits_catches_inline_code_trait(drift):
    md = "```rust\npub trait GaugeZoneSolver { fn step(&mut self); }\n```\n"
    assert "GaugeZoneSolver" in drift.extract_documented_traits(md)


def test_extract_documented_traits_catches_supporting_traits_table(drift):
    md = (
        "### Supporting Traits\n\n"
        "| `FooCalculations` | src/x.rs | thing |\n"
        "| `BarLayer` | src/y.rs | thing |\n"
        "\n## Next Section\n"
    )
    traits = drift.extract_documented_traits(md)
    assert "FooCalculations" in traits
    assert "BarLayer" in traits


@pytest.mark.parametrize(
    "fragment",
    [
        "see `src/physics/solver_trait.rs` for the contract",
        "see (src/sim/thermal_model.rs) for details",
    ],
)
def test_extract_documented_files_catches_src_paths(drift, fragment):
    files = drift.extract_documented_files(fragment)
    assert any(f.endswith(".rs") and f.startswith("src/") for f in files)


# ---------------------------------------------------------------------------
# parse_guard_constant / check_cycle_edge_count_drift (Issue #3460)
# ---------------------------------------------------------------------------

# Minimal ARCHITECTURE.md fragment carrying every cycle-count claim the
# #3460 sync guard parses, in the *real* document's shapes (including
# hard line-wraps, to pin the whitespace-normalised matching).
_CYCLE_CLEAN_ARCH = dedent(
    """\
    3. `src/sim/**` → `crate::validation::*` edge count is at or below the
       documented baseline (currently 99). This counts *every* reference.
    4. `src/validation/**` → `crate::sim::*` (baseline 65).
    5. `src/validation/**` → `crate::physics::*` (baseline 65).
    6. `src/validation/**` → `crate::weather::*` (baseline 25).

    drives the engine, weather sources, and physics tensors. As a result ~254
    directional edges remain (99 sim→validation + 65 validation→sim + 65
    validation→physics + 25 validation→weather).

    an 83-edge sim→physics baseline (`BASELINE_SIM_TO_PHYSICS = 83`; more
    history follows here).

    The documented baseline is now **0+83 edges** (0
    physics→sim + 83 sim→physics); the script exits non-zero only on regression
    """
)

_CYCLE_ASHRAE_GUARD = dedent(
    """\
    BASELINE_SIM_TO_VALIDATION = 99  # src/sim -> crate::validation
    BASELINE_VALIDATION_TO_SIM = 65
    BASELINE_VALIDATION_TO_PHYSICS = 65
    BASELINE_VALIDATION_TO_WEATHER = 25
    """
)

_CYCLE_PHYSICS_GUARD = dedent(
    """\
    BASELINE_PHYSICS_TO_SIM = 0
    BASELINE_SIM_TO_PHYSICS = 83  # was 79; +4 for #3324
    """
)


def _redirect_cycle_guards(
    drift, tmp_path, monkeypatch, ashrae=None, physics=None
) -> None:
    """Point the freshly-loaded drift module's cycle-guard paths at
    synthetic ``tmp_path`` guard scripts (mirrors the ``_redirect``
    pattern from ``test_check_required_checks_sync.py``)."""
    ashrae_path = tmp_path / "check_ashrae_cases_cycle.py"
    physics_path = tmp_path / "check_physics_sim_cycle.py"
    ashrae_path.write_text(
        _CYCLE_ASHRAE_GUARD if ashrae is None else ashrae, encoding="utf-8"
    )
    physics_path.write_text(
        _CYCLE_PHYSICS_GUARD if physics is None else physics, encoding="utf-8"
    )
    monkeypatch.setattr(drift, "ASHRAE_CYCLE_GUARD_FILE", ashrae_path)
    monkeypatch.setattr(drift, "PHYSICS_SIM_CYCLE_GUARD_FILE", physics_path)


def test_parse_guard_constant_reads_assignment_and_ignores_comments(
    drift, tmp_path
):
    p = tmp_path / "guard.py"
    p.write_text("BASELINE_X = 42  # trailing comment\n", encoding="utf-8")
    assert drift.parse_guard_constant(p, "BASELINE_X") == 42


def test_parse_guard_constant_returns_none_for_missing_constant_or_file(
    drift, tmp_path
):
    p = tmp_path / "guard.py"
    p.write_text("SOME_OTHER = 1\n", encoding="utf-8")
    assert drift.parse_guard_constant(p, "BASELINE_X") is None
    assert drift.parse_guard_constant(tmp_path / "absent.py", "BASELINE_X") is None


def test_cycle_count_guard_clean_when_docs_match_guards(
    drift, tmp_path, monkeypatch
):
    """Clean case: every documented claim agrees with the mock guard
    constants (including across hard line-wraps)."""
    _redirect_cycle_guards(drift, tmp_path, monkeypatch)
    assert drift.check_cycle_edge_count_drift(_CYCLE_CLEAN_ARCH) == []


def test_cycle_count_guard_passes_on_real_repo(drift, repo_root):
    """Pin the Issue #3460 fix itself: the real ARCHITECTURE.md claims
    must agree with the real cycle-guard constants. A future baseline
    bump that forgets the docs flips this test (and the CI gate)."""
    findings = drift.check_cycle_edge_count_drift(
        (repo_root / "ARCHITECTURE.md").read_text(encoding="utf-8")
    )
    assert findings == []


def test_cycle_count_guard_flags_scalar_baseline_drift(drift, tmp_path, monkeypatch):
    """Planted violation: the guard's BASELINE_SIM_TO_VALIDATION moved to
    100 but the docs still say 99 — the exact #3460 drift class. Both the
    invariant-list claim and the breakdown component compare against the
    same guard constant, so both fire."""
    drifted = _CYCLE_ASHRAE_GUARD.replace(
        "BASELINE_SIM_TO_VALIDATION = 99", "BASELINE_SIM_TO_VALIDATION = 100"
    )
    _redirect_cycle_guards(drift, tmp_path, monkeypatch, ashrae=drifted)
    findings = drift.check_cycle_edge_count_drift(_CYCLE_CLEAN_ARCH)
    assert len(findings) == 2
    joined = "\n".join(findings)
    assert "documents the sim→validation baseline as 99" in joined
    assert "BASELINE_SIM_TO_VALIDATION = 100" in joined
    assert "breakdown documents sim→validation as 99" in joined


def test_cycle_count_guard_flags_backticked_constant_drift(
    drift, tmp_path, monkeypatch
):
    """Planted violation: the backticked ``BASELINE_SIM_TO_PHYSICS = 83``
    mention in the docs drifts from the guard constant."""
    drifted_arch = _CYCLE_CLEAN_ARCH.replace(
        "`BASELINE_SIM_TO_PHYSICS = 83`", "`BASELINE_SIM_TO_PHYSICS = 72`"
    )
    _redirect_cycle_guards(drift, tmp_path, monkeypatch)
    findings = drift.check_cycle_edge_count_drift(drifted_arch)
    assert len(findings) == 1
    assert "documents the sim→physics baseline as 72" in findings[0]
    assert "BASELINE_SIM_TO_PHYSICS = 83" in findings[0]


def test_cycle_count_guard_flags_breakdown_component_drift(
    drift, tmp_path, monkeypatch
):
    """Planted violation: one component of the directional-edge breakdown
    sentence disagrees with its guard baseline (the total literal is
    shifted along with it so only the component comparison fires)."""
    drifted_arch = _CYCLE_CLEAN_ARCH.replace("~254", "~253").replace(
        "+ 65 validation→sim + 65", "+ 64 validation→sim + 65"
    )
    _redirect_cycle_guards(drift, tmp_path, monkeypatch)
    findings = drift.check_cycle_edge_count_drift(drifted_arch)
    assert len(findings) == 1
    assert "breakdown documents validation→sim as 64" in findings[0]
    assert "baseline is 65" in findings[0]


def test_cycle_count_guard_flags_breakdown_total_arithmetic_drift(
    drift, tmp_path, monkeypatch
):
    """Planted violation: the '~N directional edges' total disagrees with
    the sum of its own breakdown components."""
    drifted_arch = _CYCLE_CLEAN_ARCH.replace("~254", "~250")
    _redirect_cycle_guards(drift, tmp_path, monkeypatch)
    findings = drift.check_cycle_edge_count_drift(drifted_arch)
    assert len(findings) == 1
    assert "claims ~250 directional edges" in findings[0]
    assert "sums to 254" in findings[0]


def test_cycle_count_guard_flags_zero_plus_n_drift(drift, tmp_path, monkeypatch):
    """Planted violation: the physics<->sim '**0+N edges**' sentence
    still narrates the pre-#3460 84-edge baseline."""
    drifted_arch = _CYCLE_CLEAN_ARCH.replace("**0+83 edges**", "**0+84 edges**")
    _redirect_cycle_guards(drift, tmp_path, monkeypatch)
    findings = drift.check_cycle_edge_count_drift(drifted_arch)
    assert len(findings) == 1
    assert "0+N baseline headline documents sim→physics as 84" in findings[0]
    assert "BASELINE_SIM_TO_PHYSICS = 83" in findings[0]


def test_cycle_count_guard_flags_removed_claims(drift, tmp_path, monkeypatch):
    """Deleting a documented claim must fail the guard just like drifting
    it — otherwise the sync check can be silenced by removal."""
    _redirect_cycle_guards(drift, tmp_path, monkeypatch)
    findings = drift.check_cycle_edge_count_drift(
        "No cycle narrative here at all.\n"
    )
    # 5 scalar claims + 2 composite sentences all go missing.
    assert len(findings) == 7
    joined = "\n".join(findings)
    assert "no longer documents the sim→validation baseline" in joined
    assert "directional-edge breakdown sentence" in joined
    assert "0+N baseline sentence" in joined


def test_cycle_count_guard_flags_missing_guard_constant(drift, tmp_path, monkeypatch):
    """A guard script that no longer defines the constant must fail
    loudly instead of silently skipping the comparison (the breakdown
    component comparisons skip missing guards, but the scalar claims do
    not — so the missing constants are still named)."""
    _redirect_cycle_guards(
        drift,
        tmp_path,
        monkeypatch,
        ashrae="BASELINE_VALIDATION_TO_SIM = 65\n",
    )
    findings = drift.check_cycle_edge_count_drift(_CYCLE_CLEAN_ARCH)
    assert len(findings) == 3
    joined = "\n".join(findings)
    assert "BASELINE_SIM_TO_VALIDATION not parseable" in joined
    assert "BASELINE_VALIDATION_TO_PHYSICS not parseable" in joined
    assert "BASELINE_VALIDATION_TO_WEATHER not parseable" in joined
