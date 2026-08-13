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
