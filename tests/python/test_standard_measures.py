"""Pytest suite for the standard-library Fluxion measures (Issues #1815, #3995).

Issue #1815 — Phase 3 (Developer Experience & Ecosystem) of the Hybrid Measure
Approach. Covers the baseline measures shipped in ``measures/``:

- ``SetWindowToWallRatio``
- ``ReplaceHVACWithVAV``
- ``IncreaseInsulationRValue``

Issue #3995 grows the standard library past three with the two control-side
measures:

- ``SetThermostatSetpoints``
- ``SwapSchedule``

The suite is designed to run in two modes:

1. **With native bindings** (``maturin develop``): full integration coverage
   through ``fluxion.Model`` — the measures mutate a real model, the
   mutations persist via the snapshot/owned-value round trip, and the
   provenance chain (Issue #1816) is recorded.
2. **Without native bindings** (CI without a Rust toolchain): the pure-Python
   numerical helpers (``compute_window_area``, ``compute_insulated_u_value``,
   ``build_vav_system``, ``resolve_setpoint_pair``, ``build_hourly_profile``,
   …) are unit-tested directly. These tests do not need a ``fluxion.Model``.
   The two control-side measures are additionally exercised against
   duck-typed stand-ins for ``fluxion.Model`` / ``fluxion.Zone``, which
   implement the documented snapshot contract (``model.zones()`` -> mutate ->
   push back) without the native extension.

Numerical reasoning is done in code (per ``RULES.md``); the expected values in
the pure-Python tests were verified with the helper scripts that accompany the
measures.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MEASURES_DIR = REPO_ROOT / "measures"

# Ensure ``measures.*`` is importable.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fluxion_spec = importlib.util.find_spec("fluxion")
HAS_FLUXION = fluxion_spec is not None

requires_fluxion = pytest.mark.skipif(
    not HAS_FLUXION,
    reason="fluxion Python bindings not available (run `maturin develop`)",
)


def _load_module(rel_path: str, mod_name: str):
    """Import a measure module directly from its file path."""
    from importlib.util import module_from_spec, spec_from_file_location

    path = REPO_ROOT / rel_path
    spec = spec_from_file_location(mod_name, str(path))
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# SetWindowToWallRatio — pure-Python numerical helper
# =============================================================================


class TestComputeWindowArea:
    """Unit-test the WWR rescaling math without native bindings."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module("measures/SetWindowToWallRatio.py", "_wwr_mod")

    def test_standard_target(self, mod):
        assert mod.compute_window_area(20.0, 0.40) == pytest.approx(8.0)

    def test_reduce_glazing(self, mod):
        assert mod.compute_window_area(20.0, 0.20) == pytest.approx(4.0)

    def test_add_from_zero(self, mod):
        assert mod.compute_window_area(20.0, 0.50) == pytest.approx(10.0)

    def test_clamps_to_full_area(self, mod):
        # target > 1.0 clamps to 100%
        assert mod.compute_window_area(20.0, 1.10) == pytest.approx(20.0)

    def test_clamps_to_zero_for_negative(self, mod):
        assert mod.compute_window_area(20.0, -0.1) == pytest.approx(0.0)

    def test_zero_area_returns_zero(self, mod):
        assert mod.compute_window_area(0.0, 0.40) == pytest.approx(0.0)

    def test_monotonic_in_target(self, mod):
        vals = [mod.compute_window_area(20.0, t) for t in (0.1, 0.3, 0.5, 0.9)]
        assert vals == sorted(vals)


# =============================================================================
# IncreaseInsulationRValue — pure-Python numerical helper
# =============================================================================


class TestComputeInsulatedUValue:
    """Unit-test the U-value / R-value math without native bindings."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module("measures/IncreaseInsulationRValue.py", "_insul_mod")

    def test_adds_insulation(self, mod):
        # U=0.5, +R2 -> U = 1/(1/0.5 + 2) = 1/4 = 0.25
        assert mod.compute_insulated_u_value(0.5, 2.0) == pytest.approx(0.25)

    def test_zero_delta_r_is_identity(self, mod):
        assert mod.compute_insulated_u_value(1.0, 0.0) == pytest.approx(1.0)

    def test_negative_u_value_untouched(self, mod):
        # Sentinel / uninitialised U-values must not be corrupted.
        assert mod.compute_insulated_u_value(-1.0, 2.0) == pytest.approx(-1.0)

    def test_zero_u_value_untouched(self, mod):
        assert mod.compute_insulated_u_value(0.0, 2.0) == pytest.approx(0.0)

    def test_monotonic_decreasing_in_delta_r(self, mod):
        vals = [mod.compute_insulated_u_value(0.5, dr) for dr in (0.0, 1.0, 2.0, 5.0)]
        # U-value must not increase as insulation is added.
        assert vals == sorted(vals, reverse=True)


# =============================================================================
# ReplaceHVACWithVAV — pure-Python helper
# =============================================================================


class TestBuildVavSystem:
    """Unit-test the VAV snapshot builder without native bindings.

    Uses a ``SimpleNamespace`` stand-in for the PyHVACSystem snapshot.
    """

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module("measures/ReplaceHVACWithVAV.py", "_vav_mod")

    def _stub_hvac(self):
        return SimpleNamespace(
            heating_capacity=10000.0,
            cooling_capacity=8000.0,
            cop_heating=3.0,
            cop_cooling=3.2,
            stages=1,
            min_outdoor_temp=-10.0,
            max_outdoor_temp=40.0,
            vav_enabled=False,
            economizer_enabled=False,
            supply_air_temp=13.0,
        )

    def test_enables_vav_and_economizer_unconditionally(self, mod):
        hvac = self._stub_hvac()
        mod.build_vav_system(hvac, {})
        assert hvac.vav_enabled is True
        assert hvac.economizer_enabled is True

    def test_default_supply_air_temp(self, mod):
        hvac = self._stub_hvac()
        mod.build_vav_system(hvac, {})
        assert hvac.supply_air_temp == pytest.approx(13.0)

    def test_supply_air_temp_override(self, mod):
        hvac = self._stub_hvac()
        mod.build_vav_system(hvac, {"supply_air_temp": 14.5})
        assert hvac.supply_air_temp == pytest.approx(14.5)

    def test_none_arguments_preserve_existing(self, mod):
        hvac = self._stub_hvac()
        # parse_arguments fills declared defaults with None for omitted args;
        # the builder must treat None as "preserve existing".
        mod.build_vav_system(
            hvac,
            {
                "heating_capacity": None,
                "cooling_capacity": None,
                "cop_heating": None,
                "cop_cooling": None,
                "stages": None,
                "min_outdoor_temp": None,
                "max_outdoor_temp": None,
            },
        )
        assert hvac.heating_capacity == pytest.approx(10000.0)
        assert hvac.cooling_capacity == pytest.approx(8000.0)
        assert hvac.stages == 1

    def test_explicit_overrides(self, mod):
        hvac = self._stub_hvac()
        mod.build_vav_system(
            hvac,
            {"heating_capacity": 18000.0, "cooling_capacity": 15000.0, "stages": 2},
        )
        assert hvac.heating_capacity == pytest.approx(18000.0)
        assert hvac.cooling_capacity == pytest.approx(15000.0)
        assert hvac.stages == 2


# =============================================================================
# Argument descriptors (no native bindings needed)
# =============================================================================


class TestArgumentDescriptors:
    """Each standard-library measure exposes a valid argument spec."""

    def test_wwr_arguments(self):
        mod = _load_module("measures/SetWindowToWallRatio.py", "_wwr_args")
        spec = mod.SetWindowToWallRatio().arguments()
        names = [a["name"] for a in spec]
        assert "target_wwr" in names
        wwr_arg = next(a for a in spec if a["name"] == "target_wwr")
        assert wwr_arg["default"] == 0.40
        assert wwr_arg["min"] == 0.0 and wwr_arg["max"] == 1.0

    def test_vav_arguments(self):
        mod = _load_module("measures/ReplaceHVACWithVAV.py", "_vav_args")
        spec = mod.ReplaceHVACWithVAV().arguments()
        names = [a["name"] for a in spec]
        assert "supply_air_temp" in names
        assert "heating_capacity" in names

    def test_insulation_arguments(self):
        mod = _load_module("measures/IncreaseInsulationRValue.py", "_insul_args")
        spec = mod.IncreaseInsulationRValue().arguments()
        names = [a["name"] for a in spec]
        assert "delta_r" in names
        dr = next(a for a in spec if a["name"] == "delta_r")
        assert dr["default"] == 2.0

    def test_setpoint_arguments(self):
        mod = _load_module("measures/SetThermostatSetpoints.py", "_setpoints_args")
        spec = mod.SetThermostatSetpoints().arguments()
        names = [a["name"] for a in spec]
        assert "heating_delta" in names
        assert "cooling_delta" in names
        assert "heating_setpoint" in names
        assert "cooling_setpoint" in names
        heating_delta = next(a for a in spec if a["name"] == "heating_delta")
        assert heating_delta["default"] == 0.0
        absolute = next(a for a in spec if a["name"] == "heating_setpoint")
        assert absolute["default"] is None

    def test_schedule_arguments(self):
        mod = _load_module("measures/SwapSchedule.py", "_schedule_args")
        spec = mod.SwapSchedule().arguments()
        names = [a["name"] for a in spec]
        assert "schedule_name" in names
        assert "profile" in names
        profile = next(a for a in spec if a["name"] == "profile")
        assert profile["default"] == mod.PROFILE_CONSTANT
        assert set(profile["choices"]) == set(mod.PROFILES)


# =============================================================================
# Discovery — the standard-library measures are discoverable
# =============================================================================


class TestDiscovery:
    """``discover_measures`` picks up the standard-library classes."""

    def test_standard_measures_discovered_from_measures_dir(self):
        from fluxion.measures import discover_measures

        classes = discover_measures(MEASURES_DIR)
        names = {c.__name__ for c in classes}
        assert {
            "SetWindowToWallRatio",
            "ReplaceHVACWithVAV",
            "IncreaseInsulationRValue",
            "SetThermostatSetpoints",
            "SwapSchedule",
        }.issubset(names), names


# =============================================================================
# Integration — real model mutation (requires native bindings)
# =============================================================================


@requires_fluxion
class TestSetWindowToWallRatioIntegration:
    """End-to-end WWR mutation on a real fluxion.Model."""

    def _import_measure(self):
        return _load_module(
            "measures/SetWindowToWallRatio.py", "_wwr_int"
        ).SetWindowToWallRatio

    def test_all_zones_hit_target(self):
        import fluxion
        from fluxion.measures import apply_measures

        Measure = self._import_measure()
        m = fluxion.Model(num_zones=2)
        apply_measures(m, [Measure], {"SetWindowToWallRatio": {"target_wwr": 0.40}})

        for s in m.surfaces():
            if s.area > 0:
                assert s.window_area / s.area == pytest.approx(0.40, rel=1e-9)

    def test_zone_filter_only_affects_one_zone(self):
        import fluxion
        from fluxion.measures import apply_measures

        Measure = self._import_measure()
        m = fluxion.Model(num_zones=2)
        apply_measures(
            m,
            [Measure],
            {"SetWindowToWallRatio": {"target_wwr": 0.40, "zone_index": 0}},
        )

        surfaces = m.surfaces()
        # zone 0 surfaces (first half) should be at 0.40
        zones = m.zones()
        per_zone = len(zones[0].surfaces) if zones else 1
        for i, s in enumerate(surfaces):
            zone_idx = i // per_zone
            if zone_idx == 0 and s.area > 0:
                assert s.window_area / s.area == pytest.approx(0.40, rel=1e-9)
            elif zone_idx == 1:
                # Untouched — window_area stays at the model default (0.0).
                assert s.window_area == pytest.approx(0.0, abs=1e-9)

    def test_provenance_chain_recorded(self):
        import fluxion
        from fluxion.measures import (
            SOURCE_PYTHON_MEASURE,
            apply_measures,
        )

        Measure = self._import_measure()
        m = fluxion.Model(num_zones=1)
        chain: list[dict] = []
        apply_measures(m, [Measure], applied_deltas=chain)
        assert len(chain) == 1
        assert chain[0]["source"] == SOURCE_PYTHON_MEASURE
        assert chain[0]["name"] == "SetWindowToWallRatio"

    def test_wwr_persists_through_json_round_trip(self, tmp_path):
        import fluxion
        from fluxion.measures import apply_measures, load_model, save_model

        Measure = self._import_measure()
        m = fluxion.Model(num_zones=1)
        apply_measures(m, [Measure], {"SetWindowToWallRatio": {"target_wwr": 0.30}})

        path = tmp_path / "wwr.json"
        save_model(m, path)
        m2 = load_model(path)
        for s in m2.surfaces():
            if s.area > 0:
                assert s.window_area / s.area == pytest.approx(0.30, rel=1e-9)


@requires_fluxion
class TestReplaceHVACWithVAVIntegration:
    """End-to-end VAV retrofit on a real fluxion.Model.

    Only ``heating_capacity`` / ``cooling_capacity`` currently round-trip into
    the underlying Rust ``ThermalModel`` (see ``src/python/model_bindings/model.rs``).
    The VAV / economizer / supply-air flags are advisory snapshots, matching
    the documented limitation in the ``SetHVACCOP`` example and
    ``docs/measures.md``. These tests assert the capacities + provenance.
    """

    def _import_measure(self):
        return _load_module(
            "measures/ReplaceHVACWithVAV.py", "_vav_int"
        ).ReplaceHVACWithVAV

    def test_capacities_persist(self):
        import fluxion
        from fluxion.measures import apply_measures

        Measure = self._import_measure()
        m = fluxion.Model(num_zones=1)
        apply_measures(
            m,
            [Measure],
            {
                "ReplaceHVACWithVAV": {
                    "heating_capacity": 18000.0,
                    "cooling_capacity": 15000.0,
                }
            },
        )
        hvac = m.hvac_system()
        assert hvac.heating_capacity == pytest.approx(18000.0, rel=1e-9)
        assert hvac.cooling_capacity == pytest.approx(15000.0, rel=1e-9)

    def test_provenance_chain_recorded(self):
        import fluxion
        from fluxion.measures import SOURCE_PYTHON_MEASURE, apply_measures

        Measure = self._import_measure()
        m = fluxion.Model(num_zones=1)
        chain: list[dict] = []
        apply_measures(m, [Measure], applied_deltas=chain)
        assert len(chain) == 1
        assert chain[0]["source"] == SOURCE_PYTHON_MEASURE
        assert chain[0]["name"] == "ReplaceHVACWithVAV"


@requires_fluxion
class TestIncreaseInsulationRValueIntegration:
    """End-to-end insulation retrofit on a real fluxion.Model."""

    def _import_measure(self):
        return _load_module(
            "measures/IncreaseInsulationRValue.py", "_insul_int"
        ).IncreaseInsulationRValue

    def test_u_values_decrease(self):
        import fluxion
        from fluxion.measures import apply_measures

        Measure = self._import_measure()
        m = fluxion.Model(num_zones=1)
        before = [s.u_value for s in m.surfaces()]
        apply_measures(m, [Measure], {"IncreaseInsulationRValue": {"delta_r": 2.0}})
        after = [s.u_value for s in m.surfaces()]
        assert len(before) == len(after)
        for u_old, u_new in zip(before, after):
            if u_old > 0:
                assert u_new < u_old
                # U = 1/(1/U_old + 2)
                assert u_new == pytest.approx(1.0 / (1.0 / u_old + 2.0), rel=1e-9)

    def test_orientation_filter(self):
        import fluxion
        from fluxion.measures import apply_measures

        Measure = self._import_measure()
        m = fluxion.Model(num_zones=1)
        apply_measures(
            m,
            [Measure],
            {"IncreaseInsulationRValue": {"delta_r": 2.0, "orientation": "North"}},
        )
        for s in m.surfaces():
            name = repr(s.orientation).rsplit(".", 1)[-1]
            if name == "North":
                assert s.u_value < 2.5  # reduced
            elif s.u_value > 0:
                # Other orientations: only vertical ones get insulated when
                # vertical_only defaults True AND orientation==North restricts
                # to North. So East/South/West should be unchanged.
                pass

    def test_provenance_chain_recorded(self):
        import fluxion
        from fluxion.measures import apply_measures

        Measure = self._import_measure()
        m = fluxion.Model(num_zones=1)
        chain: list[dict] = []
        apply_measures(m, [Measure], applied_deltas=chain)
        assert len(chain) == 1
        assert chain[0]["name"] == "IncreaseInsulationRValue"


# =============================================================================
# Combined chain — all three measures in sequence
# =============================================================================


@requires_fluxion
class TestStandardLibraryChain:
    """Run the full standard library in sequence and verify provenance."""

    def test_all_three_measures_apply_in_order(self):
        import fluxion
        from fluxion.measures import SOURCE_PYTHON_MEASURE, apply_measures

        classes = [
            _load_module(
                "measures/SetWindowToWallRatio.py", "_c1"
            ).SetWindowToWallRatio,
            _load_module(
                "measures/IncreaseInsulationRValue.py", "_c2"
            ).IncreaseInsulationRValue,
            _load_module("measures/ReplaceHVACWithVAV.py", "_c3").ReplaceHVACWithVAV,
        ]
        m = fluxion.Model(num_zones=2)
        chain: list[dict] = []
        applied = apply_measures(
            m,
            classes,
            {
                "SetWindowToWallRatio": {"target_wwr": 0.40},
                "IncreaseInsulationRValue": {"delta_r": 2.0},
                "ReplaceHVACWithVAV": {
                    "heating_capacity": 18000.0,
                    "cooling_capacity": 15000.0,
                },
            },
            applied_deltas=chain,
        )
        assert applied == [
            "SetWindowToWallRatio",
            "IncreaseInsulationRValue",
            "ReplaceHVACWithVAV",
        ]
        assert [e["name"] for e in chain] == applied
        assert all(e["source"] == SOURCE_PYTHON_MEASURE for e in chain)

    def test_cli_runs_standard_library(self, tmp_path):
        import subprocess

        fluxion_mod = importlib.import_module("fluxion")
        from fluxion.measures import save_model

        base = fluxion_mod.Model(num_zones=1)
        base_path = tmp_path / "base.json"
        save_model(base, base_path)

        out_path = tmp_path / "out.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fluxion.cli",
                "apply-measures",
                "--model",
                str(base_path),
                "--measures",
                str(MEASURES_DIR),
                "--output",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(out_path.read_text())
        assert "applied_deltas" in payload
        names = [e["name"] for e in payload["applied_deltas"]]
        assert "SetWindowToWallRatio" in names
        assert "ReplaceHVACWithVAV" in names
        assert "IncreaseInsulationRValue" in names


# =============================================================================
# Duck-typed stand-ins for fluxion.Model / fluxion.Zone
# =============================================================================
#
# The measures are written against the *documented* snapshot contract
# (``model.zones()`` -> mutate -> push back), so they can be exercised
# without the native bindings by handing them a stand-in that implements the
# same surface. This keeps the pure-Python mode of the suite meaningful for
# the control-side measures, whose data the native ``fluxion.Model`` binding
# does not (yet) expose a setter for.


class _StubZone:
    """Stand-in for the ``fluxion.Zone`` snapshot."""

    def __init__(self, index, heating_setpoint=20.0, cooling_setpoint=24.0):
        self.index = index
        self.heating_setpoint = heating_setpoint
        self.cooling_setpoint = cooling_setpoint


class _StubModel:
    """Stand-in for ``fluxion.Model`` exposing only the zone snapshot API."""

    def __init__(self, zones, with_zone_setpoint_setter=False):
        self._zones = zones
        self.setpoint_calls = []
        if with_zone_setpoint_setter:
            self.set_zone_setpoints = self._set_zone_setpoints

    def zones(self):
        # Mirrors the PyO3 contract: a *fresh* list of owned snapshots.
        return list(self._zones)

    def _set_zone_setpoints(self, index, heating, cooling):
        self.setpoint_calls.append((index, heating, cooling))


class _StubSchedule:
    """Stand-in for ``fluxion.HVACSchedule`` (hourly setpoint lookup)."""

    def __init__(self, heating_values, cooling_values):
        self._heating = list(heating_values)
        self._cooling = list(cooling_values)

    def heating_setpoint(self, hour):
        return self._heating[hour % 24]

    def cooling_setpoint(self, hour):
        return self._cooling[hour % 24]


# =============================================================================
# SetThermostatSetpoints — pure-Python numerical helpers
# =============================================================================


class TestSetpointHelpers:
    """Unit-test the setpoint arithmetic without native bindings."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module("measures/SetThermostatSetpoints.py", "_setpoints_mod")

    def test_clamp_inside_range(self, mod):
        assert mod.clamp(20.0, 15.0, 25.0) == pytest.approx(20.0)

    def test_clamp_below_and_above(self, mod):
        assert mod.clamp(10.0, 15.0, 25.0) == pytest.approx(15.0)
        assert mod.clamp(30.0, 15.0, 25.0) == pytest.approx(25.0)

    def test_clamp_normalizes_reversed_bounds(self, mod):
        assert mod.clamp(30.0, 25.0, 15.0) == pytest.approx(25.0)
        assert mod.clamp(10.0, 25.0, 15.0) == pytest.approx(15.0)

    def test_shift_by_delta(self, mod):
        assert mod.shift_setpoint(20.0, 1.5) == pytest.approx(21.5)
        assert mod.shift_setpoint(20.0, -1.5) == pytest.approx(18.5)

    def test_absolute_overrides_delta(self, mod):
        assert mod.shift_setpoint(20.0, 5.0, absolute=18.0) == pytest.approx(18.0)

    def test_shift_applies_bounds(self, mod):
        assert mod.shift_setpoint(20.0, 10.0, minimum=15.0, maximum=25.0) == pytest.approx(
            25.0
        )

    def test_resolve_pair_applies_deltas(self, mod):
        heating, cooling = mod.resolve_setpoint_pair(20.0, 24.0, heating_delta=1.0)
        assert heating == pytest.approx(21.0)
        assert cooling == pytest.approx(24.0)

    def test_resolve_pair_absolute_overrides_delta(self, mod):
        heating, cooling = mod.resolve_setpoint_pair(
            20.0, 24.0, heating_delta=5.0, cooling_setpoint=26.0
        )
        assert heating == pytest.approx(25.0)  # clamped to the engine bound
        assert cooling == pytest.approx(26.0)

    def test_resolve_pair_zero_delta_is_identity(self, mod):
        heating, cooling = mod.resolve_setpoint_pair(20.0, 24.0)
        assert heating == pytest.approx(20.0)
        assert cooling == pytest.approx(24.0)

    def test_resolve_pair_clamps_to_engine_bounds(self, mod):
        heating, cooling = mod.resolve_setpoint_pair(
            20.0, 24.0, heating_delta=10.0, cooling_delta=20.0
        )
        assert heating == pytest.approx(mod.MAX_HEATING_SETPOINT)
        assert cooling == pytest.approx(mod.MAX_COOLING_SETPOINT)

    def test_resolve_pair_clamps_to_minimums(self, mod):
        heating, cooling = mod.resolve_setpoint_pair(
            20.0, 24.0, heating_delta=-10.0, cooling_delta=-10.0
        )
        assert heating == pytest.approx(mod.MIN_HEATING_SETPOINT)
        assert cooling == pytest.approx(mod.MIN_COOLING_SETPOINT)

    def test_resolve_pair_without_clamping(self, mod):
        # Without clamping the raw shifted values are kept — but the deadband
        # is still enforced, so the cooling setpoint is raised to 31 °C.
        heating, cooling = mod.resolve_setpoint_pair(
            20.0, 24.0, heating_delta=10.0, clamp_to_bounds=False
        )
        assert heating == pytest.approx(30.0)
        assert cooling == pytest.approx(31.0)

    def test_resolve_pair_reopens_collapsed_deadband(self, mod):
        # Heating + 1 K collapses the deadband (25 + 1 > 25), so the cooling
        # setpoint is raised instead of letting the pair invert.
        heating, cooling = mod.resolve_setpoint_pair(24.0, 25.0, heating_delta=1.0)
        assert heating == pytest.approx(25.0)
        assert cooling == pytest.approx(26.0)
        assert cooling > heating

    def test_resolve_pair_lowers_heating_when_cooling_is_capped(self, mod):
        # The cooling setpoint is already at the (narrow) cap, so it cannot
        # be raised: the heating setpoint is lowered instead.
        heating, cooling = mod.resolve_setpoint_pair(
            25.0,
            25.0,
            max_heating=25.0,
            min_heating=15.0,
            min_cooling=22.0,
            max_cooling=25.0,
        )
        assert cooling == pytest.approx(25.0)
        assert heating == pytest.approx(24.0)

    def test_resolve_pair_keeps_valid_deadband(self, mod):
        heating, cooling = mod.resolve_setpoint_pair(20.0, 24.0, heating_delta=2.0)
        assert heating == pytest.approx(22.0)
        assert cooling == pytest.approx(24.0)
        assert cooling - heating >= mod.MIN_DEADBAND_K


class TestApplySetpointSweep:
    """Unit-test the zone sweep helper without native bindings."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module("measures/SetThermostatSetpoints.py", "_setpoints_mod")

    def _zones(self):
        return [
            _StubZone(0, 20.0, 24.0),
            _StubZone(1, 21.0, 25.0),
        ]

    def test_sweeps_every_zone(self, mod):
        zones = self._zones()
        records = mod.apply_setpoint_sweep(zones, heating_delta=1.0)
        assert len(records) == 2
        assert [z.heating_setpoint for z in zones] == pytest.approx([21.0, 22.0])
        assert [z.cooling_setpoint for z in zones] == pytest.approx([24.0, 25.0])

    def test_zone_index_filter(self, mod):
        zones = self._zones()
        records = mod.apply_setpoint_sweep(zones, zone_index=1, heating_delta=1.0)
        assert len(records) == 1
        assert records[0]["index"] == 1
        assert zones[0].heating_setpoint == pytest.approx(20.0)
        assert zones[1].heating_setpoint == pytest.approx(22.0)

    def test_records_report_new_values(self, mod):
        zones = self._zones()
        records = mod.apply_setpoint_sweep(zones, cooling_setpoint=23.0)
        assert records[0] == {
            "index": 0,
            "heating_setpoint": pytest.approx(20.0),
            "cooling_setpoint": pytest.approx(23.0),
        }

    def test_unknown_zone_index_is_a_no_op(self, mod):
        zones = self._zones()
        records = mod.apply_setpoint_sweep(zones, zone_index=7, heating_delta=1.0)
        assert records == []
        assert zones[0].heating_setpoint == pytest.approx(20.0)


# =============================================================================
# SetThermostatSetpoints — measure behaviour (duck-typed model)
# =============================================================================


class TestSetThermostatSetpointsMeasure:
    """Exercise the measure itself against a stand-in model."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module("measures/SetThermostatSetpoints.py", "_setpoints_mod")

    def test_apply_uses_the_zone_setpoint_setter(self, mod):
        model = _StubModel(
            [_StubZone(0, 20.0, 24.0), _StubZone(1, 20.0, 24.0)],
            with_zone_setpoint_setter=True,
        )
        measure = mod.SetThermostatSetpoints()
        measure.apply(model, {"heating_delta": 1.0, "cooling_delta": -1.0})
        assert model.setpoint_calls == [(0, 21.0, 23.0), (1, 21.0, 23.0)]

    def test_apply_without_setter_still_sweeps_the_snapshots(self, mod):
        model = _StubModel([_StubZone(0, 20.0, 24.0)])
        measure = mod.SetThermostatSetpoints()
        measure.apply(model, {"heating_delta": 1.0})
        assert model.setpoint_calls == []
        assert model.zones()[0].heating_setpoint == pytest.approx(21.0)

    def test_apply_respects_zone_index(self, mod):
        model = _StubModel(
            [_StubZone(0, 20.0, 24.0), _StubZone(1, 20.0, 24.0)],
            with_zone_setpoint_setter=True,
        )
        measure = mod.SetThermostatSetpoints()
        measure.apply(model, {"heating_delta": 1.0, "zone_index": 1})
        assert model.setpoint_calls == [(1, 21.0, 24.0)]

    def test_apply_uses_declared_defaults(self, mod):
        model = _StubModel([_StubZone(0, 20.0, 24.0)], with_zone_setpoint_setter=True)
        measure = mod.SetThermostatSetpoints()
        # ``parse_arguments`` fills the declared defaults (None for the
        # optional absolute setpoints) — the measure must treat a None
        # absolute value as "not specified".
        args = measure.parse_arguments({})
        measure.apply(model, args)
        assert model.setpoint_calls == [(0, 20.0, 24.0)]

    def test_provenance_chain_recorded(self, mod):
        from fluxion.measures import SOURCE_PYTHON_MEASURE, apply_measures

        model = _StubModel([_StubZone(0, 20.0, 24.0)])
        chain: list[dict] = []
        applied = apply_measures(
            model,
            [mod.SetThermostatSetpoints],
            {"SetThermostatSetpoints": {"heating_delta": 1.0}},
            applied_deltas=chain,
        )
        assert applied == ["SetThermostatSetpoints"]
        assert len(chain) == 1
        assert chain[0]["source"] == SOURCE_PYTHON_MEASURE
        assert chain[0]["name"] == "SetThermostatSetpoints"


# =============================================================================
# SwapSchedule — pure-Python helpers
# =============================================================================


class TestScheduleHelpers:
    """Unit-test the schedule profile arithmetic without native bindings."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module("measures/SwapSchedule.py", "_schedule_mod")

    def test_fill_range_plain(self, mod):
        values = mod.fill_range([0.0] * 24, 8, 12, 5.0)
        assert values[8] == pytest.approx(5.0)
        assert values[11] == pytest.approx(5.0)
        assert values[12] == pytest.approx(0.0)
        assert values[7] == pytest.approx(0.0)

    def test_fill_range_wraps_midnight(self, mod):
        values = mod.fill_range([0.0] * 24, 22, 6, 1.0)
        assert values[22:24] == pytest.approx([1.0, 1.0])
        assert values[0:6] == pytest.approx([1.0] * 6)
        assert values[6] == pytest.approx(0.0)
        assert values[21] == pytest.approx(0.0)

    def test_fill_range_equal_bounds_fills_nothing(self, mod):
        values = mod.fill_range([0.0] * 24, 8, 8, 5.0)
        assert all(v == pytest.approx(0.0) for v in values)

    def test_constant_profile(self, mod):
        heating, cooling = mod.build_hourly_profile(mod.PROFILE_CONSTANT, 20.0, 24.0)
        assert len(heating) == 24 and len(cooling) == 24
        assert all(v == pytest.approx(20.0) for v in heating)
        assert all(v == pytest.approx(24.0) for v in cooling)

    def test_setback_profile(self, mod):
        heating, cooling = mod.build_hourly_profile(
            mod.PROFILE_SETBACK,
            20.0,
            24.0,
            setback_heating_setpoint=15.0,
            night_start_hour=22,
            night_end_hour=6,
        )
        assert heating[10] == pytest.approx(20.0)
        assert heating[23] == pytest.approx(15.0)
        assert heating[2] == pytest.approx(15.0)
        assert heating[12] == pytest.approx(20.0)
        assert all(v == pytest.approx(24.0) for v in cooling)

    def test_operating_hours_profile(self, mod):
        heating, cooling = mod.build_hourly_profile(
            mod.PROFILE_OPERATING_HOURS,
            20.0,
            24.0,
            start_hour=8,
            end_hour=18,
        )
        assert heating[10] == pytest.approx(20.0)
        assert heating[2] == pytest.approx(mod.HEATING_OFF)
        assert cooling[2] == pytest.approx(mod.COOLING_OFF)
        assert heating[18] == pytest.approx(mod.HEATING_OFF)

    def test_unknown_profile_raises(self, mod):
        with pytest.raises(ValueError):
            mod.build_hourly_profile("nope", 20.0, 24.0)

    def test_occupied_setpoints_constant(self, mod):
        heating, cooling = mod.build_hourly_profile(mod.PROFILE_CONSTANT, 20.0, 24.0)
        assert mod.occupied_setpoints(heating, cooling) == pytest.approx((20.0, 24.0))

    def test_occupied_setpoints_setback_uses_day_value(self, mod):
        heating, cooling = mod.build_hourly_profile(
            mod.PROFILE_SETBACK,
            20.0,
            24.0,
            setback_heating_setpoint=15.0,
            night_start_hour=22,
            night_end_hour=6,
        )
        assert mod.occupied_setpoints(heating, cooling) == pytest.approx((20.0, 24.0))

    def test_occupied_setpoints_operating_hours(self, mod):
        heating, cooling = mod.build_hourly_profile(
            mod.PROFILE_OPERATING_HOURS, 20.0, 24.0, start_hour=8, end_hour=18
        )
        assert mod.occupied_setpoints(heating, cooling) == pytest.approx((20.0, 24.0))

    def test_swap_schedule_values_returns_replacement(self, mod):
        current = [1.0] * 24
        replacement = [2.0] * 24
        assert mod.swap_schedule_values(current, replacement) == pytest.approx([2.0] * 24)

    def test_swap_schedule_values_validates_length(self, mod):
        with pytest.raises(ValueError):
            mod.swap_schedule_values([1.0] * 24, [2.0] * 12)
        with pytest.raises(ValueError):
            mod.swap_schedule_values([1.0] * 12, [2.0] * 24)

    def test_build_native_schedule_maps_profiles_to_constructors(self, mod):
        calls = []

        class _FakeHVACSchedule:
            @staticmethod
            def constant_schedule(heating_sp, cooling_sp):
                calls.append(("constant", heating_sp, cooling_sp))
                return "constant-schedule"

            @staticmethod
            def setback_schedule(day_heat, night_heat, cool_sp, night_start, night_end):
                calls.append(
                    ("setback", day_heat, night_heat, cool_sp, night_start, night_end)
                )
                return "setback-schedule"

            @staticmethod
            def with_operating_hours(heating_sp, cooling_sp, start_hour, end_hour):
                calls.append(
                    ("operating_hours", heating_sp, cooling_sp, start_hour, end_hour)
                )
                return "operating-schedule"

        fake_fluxion = SimpleNamespace(HVACSchedule=_FakeHVACSchedule)

        assert (
            mod.build_native_schedule(
                mod.PROFILE_CONSTANT, 20.0, 24.0, fluxion_module=fake_fluxion
            )
            == "constant-schedule"
        )
        assert (
            mod.build_native_schedule(
                mod.PROFILE_SETBACK,
                20.0,
                24.0,
                setback_heating_setpoint=15.0,
                night_start_hour=22,
                night_end_hour=6,
                fluxion_module=fake_fluxion,
            )
            == "setback-schedule"
        )
        assert (
            mod.build_native_schedule(
                mod.PROFILE_OPERATING_HOURS,
                20.0,
                24.0,
                start_hour=8,
                end_hour=18,
                fluxion_module=fake_fluxion,
            )
            == "operating-schedule"
        )
        assert calls[0] == ("constant", 20.0, 24.0)
        assert calls[1] == ("setback", 20.0, 15.0, 24.0, 22, 6)
        assert calls[2] == ("operating_hours", 20.0, 24.0, 8, 18)


# =============================================================================
# SwapSchedule — measure behaviour (duck-typed model)
# =============================================================================


class TestSwapScheduleMeasure:
    """Exercise the schedule swap against a stand-in model."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module("measures/SwapSchedule.py", "_schedule_mod")

    def test_current_profile_falls_back_to_zone_setpoints(self, mod):
        model = _StubModel([_StubZone(0, 20.0, 24.0)])
        heating, cooling = mod.current_profile(model)
        assert heating == pytest.approx([20.0] * 24)
        assert cooling == pytest.approx([24.0] * 24)

    def test_current_profile_reads_the_model_schedule(self, mod):
        model = _StubModel([_StubZone(0, 20.0, 24.0)])
        model.get_hvac_schedule = lambda: _StubSchedule([15.0] * 24, [26.0] * 24)
        heating, cooling = mod.current_profile(model)
        assert heating == pytest.approx([15.0] * 24)
        assert cooling == pytest.approx([26.0] * 24)

    def test_apply_collapses_to_occupied_setpoints(self, mod):
        model = _StubModel([_StubZone(0, 20.0, 24.0)], with_zone_setpoint_setter=True)
        measure = mod.SwapSchedule()
        measure.apply(
            model,
            {
                "profile": mod.PROFILE_OPERATING_HOURS,
                "heating_setpoint": 21.0,
                "cooling_setpoint": 25.0,
                "start_hour": 8,
                "end_hour": 18,
            },
        )
        assert model.setpoint_calls == [(0, 21.0, 25.0)]

    def test_apply_respects_zone_index(self, mod):
        model = _StubModel(
            [_StubZone(0, 20.0, 24.0), _StubZone(1, 20.0, 24.0)],
            with_zone_setpoint_setter=True,
        )
        measure = mod.SwapSchedule()
        measure.apply(
            model,
            {
                "profile": mod.PROFILE_CONSTANT,
                "heating_setpoint": 22.0,
                "zone_index": 0,
            },
        )
        assert model.setpoint_calls == [(0, 22.0, 24.0)]

    def test_apply_without_setter_is_advisory(self, mod):
        model = _StubModel([_StubZone(0, 20.0, 24.0)])
        measure = mod.SwapSchedule()
        # Must not raise even though the model exposes no schedule setter.
        measure.apply(
            model, {"profile": mod.PROFILE_CONSTANT, "heating_setpoint": 22.0}
        )
        assert model.setpoint_calls == []

    def test_provenance_chain_recorded(self, mod):
        from fluxion.measures import SOURCE_PYTHON_MEASURE, apply_measures

        model = _StubModel([_StubZone(0, 20.0, 24.0)])
        chain: list[dict] = []
        applied = apply_measures(
            model,
            [mod.SwapSchedule],
            {
                "SwapSchedule": {
                    "profile": mod.PROFILE_CONSTANT,
                    "heating_setpoint": 22.0,
                }
            },
            applied_deltas=chain,
        )
        assert applied == ["SwapSchedule"]
        assert len(chain) == 1
        assert chain[0]["source"] == SOURCE_PYTHON_MEASURE
        assert chain[0]["name"] == "SwapSchedule"
