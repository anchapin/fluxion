"""Pytest suite for the standard-library Fluxion measures (Issues #1815, #3995).

Issue #1815 — Phase 3 (Developer Experience & Ecosystem) of the Hybrid Measure
Approach. Covers the baseline measures shipped in ``measures/``:

- ``SetWindowToWallRatio``
- ``ReplaceHVACWithVAV``
- ``IncreaseInsulationRValue``

Issue #3995 grows the standard library past three. First the two
control-side measures:

- ``SetThermostatSetpoints``
- ``SwapSchedule``

and then the load / plant measures:

- ``SetLightingPowerDensity``
- ``SetHVACEfficiency``
- ``SetEquipmentPowerDensity``

and then the shading measure:

- ``AddOverhang``

and then the occupancy-density measure:

- ``SetOccupancyDensity``

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

def _native_bindings_available() -> bool:
    """True when the compiled ``fluxion`` extension is importable.

    The in-tree ``fluxion`` *wrapper* imports cleanly even when the
    compiled extension is missing (Issue #2852), so the mere presence of
    the wrapper module must not be mistaken for usable bindings — the
    integration tests below would fail with ``ModuleNotFoundError: No
    module named 'fluxion.fluxion'`` instead of skipping. ``
    fluxion._NATIVE_IMPORT_ERROR`` is the documented sentinel: ``None``
    when the extension loaded successfully (see ``tests/conftest.py``).
    """
    if importlib.util.find_spec("fluxion") is None:
        return False
    try:
        import fluxion
    except ImportError:
        return False
    return getattr(fluxion, "_NATIVE_IMPORT_ERROR", None) is None


HAS_FLUXION = _native_bindings_available()

requires_fluxion = pytest.mark.skipif(
    not HAS_FLUXION,
    reason="fluxion native bindings not available (run `maturin develop`)",
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

    def test_lighting_arguments(self):
        mod = _load_module("measures/SetLightingPowerDensity.py", "_lpd_args")
        spec = mod.SetLightingPowerDensity().arguments()
        names = [a["name"] for a in spec]
        assert "power_density_w_m2" in names
        density = next(a for a in spec if a["name"] == "power_density_w_m2")
        assert density["default"] == mod.DEFAULT_LIGHTING_POWER_DENSITY_W_M2
        assert density["min"] == mod.MIN_LIGHTING_POWER_DENSITY_W_M2
        assert density["max"] == mod.MAX_LIGHTING_POWER_DENSITY_W_M2

    def test_equipment_arguments(self):
        mod = _load_module("measures/SetEquipmentPowerDensity.py", "_epd_args")
        spec = mod.SetEquipmentPowerDensity().arguments()
        names = [a["name"] for a in spec]
        assert "power_density_w_m2" in names
        density = next(a for a in spec if a["name"] == "power_density_w_m2")
        assert density["default"] == mod.DEFAULT_EQUIPMENT_POWER_DENSITY_W_M2
        assert density["min"] == mod.MIN_EQUIPMENT_POWER_DENSITY_W_M2
        assert density["max"] == mod.MAX_EQUIPMENT_POWER_DENSITY_W_M2

    def test_overhang_arguments(self):
        mod = _load_module("measures/AddOverhang.py", "_overhang_args")
        spec = mod.AddOverhang().arguments()
        names = [a["name"] for a in spec]
        assert "depth" in names
        assert "height" in names
        assert "orientation" in names
        assert "only_with_windows" in names
        depth = next(a for a in spec if a["name"] == "depth")
        assert depth["default"] == mod.DEFAULT_OVERHANG_DEPTH_M
        assert depth["min"] == mod.MIN_OVERHANG_DEPTH_M
        assert depth["max"] == mod.MAX_OVERHANG_DEPTH_M
        height = next(a for a in spec if a["name"] == "height")
        assert height["default"] == mod.DEFAULT_OVERHANG_HEIGHT_M
        assert height["min"] == mod.MIN_OVERHANG_HEIGHT_M
        assert height["max"] == mod.MAX_OVERHANG_HEIGHT_M
        orientation = next(a for a in spec if a["name"] == "orientation")
        assert orientation["default"] == mod.DEFAULT_ORIENTATION
        assert set(orientation["choices"]) == set(mod.ORIENTATION_CHOICES)
        only_windows = next(a for a in spec if a["name"] == "only_with_windows")
        assert only_windows["default"] is False

    def test_hvac_efficiency_arguments(self):
        mod = _load_module("measures/SetHVACEfficiency.py", "_cop_args")
        spec = mod.SetHVACEfficiency().arguments()
        names = [a["name"] for a in spec]
        assert "cop_heating" in names
        assert "cop_cooling" in names
        heating = next(a for a in spec if a["name"] == "cop_heating")
        cooling = next(a for a in spec if a["name"] == "cop_cooling")
        assert heating["default"] is None
        assert cooling["default"] is None
        assert heating["min"] == mod.MIN_COP
        assert cooling["max"] == mod.MAX_COP

    def test_occupancy_arguments(self):
        mod = _load_module("measures/SetOccupancyDensity.py", "_occ_args")
        spec = mod.SetOccupancyDensity().arguments()
        names = [a["name"] for a in spec]
        assert "occupancy_density_persons_per_m2" in names
        density = next(
            a for a in spec if a["name"] == "occupancy_density_persons_per_m2"
        )
        assert density["default"] == mod.DEFAULT_OCCUPANCY_DENSITY_PERSONS_PER_M2
        assert density["min"] == mod.MIN_OCCUPANCY_DENSITY_PERSONS_PER_M2
        assert density["max"] == mod.MAX_OCCUPANCY_DENSITY_PERSONS_PER_M2

    def test_window_properties_arguments(self):
        mod = _load_module("measures/SetWindowProperties.py", "_wp_args")
        spec = mod.SetWindowProperties().arguments()
        names = [a["name"] for a in spec]
        assert "u_value_w_m2k" in names
        assert "shgc" in names
        u_value = next(a for a in spec if a["name"] == "u_value_w_m2k")
        shgc = next(a for a in spec if a["name"] == "shgc")
        assert u_value["default"] is None
        assert shgc["default"] is None
        assert u_value["max"] == mod.MAX_WINDOW_U_VALUE_W_M2K
        assert shgc["max"] == mod.MAX_WINDOW_SHGC


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
            "SetLightingPowerDensity",
            "SetHVACEfficiency",
            "SetEquipmentPowerDensity",
            "AddOverhang",
            "SetWindowProperties",
            "SetOccupancyDensity",
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


# =============================================================================
# SetLightingPowerDensity — duck-typed stand-ins
# =============================================================================


class _StubLighting:
    """Stand-in for a lighting / internal-loads snapshot."""

    def __init__(self, power_density_w_m2=10.0):
        self.lighting_power_density_w_m2 = power_density_w_m2


class _StubLightingModel:
    """Stand-in for ``fluxion.Model`` exposing a lighting snapshot."""

    def __init__(self, power_density_w_m2=10.0, with_setter=True):
        self._lighting = _StubLighting(power_density_w_m2)
        self.lighting_writes = []
        if with_setter:
            self.set_lighting = self._set_lighting

    def lighting(self):
        # Mirrors the PyO3 contract: an owned snapshot handed to the caller.
        return self._lighting

    def _set_lighting(self, lighting):
        self.lighting_writes.append(lighting)


class _StubInternalLoadsModel:
    """Stand-in for a model whose internal loads carry the LPD."""

    def __init__(self, power_density_w_m2=10.0):
        self._loads = _StubLighting(power_density_w_m2)
        self.internal_load_writes = []

    def internal_loads(self):
        return self._loads

    def set_internal_loads(self, loads):
        self.internal_load_writes.append(loads)


class _StubModelWithoutLighting:
    """Stand-in for a model that exposes no lighting object at all."""


# =============================================================================
# SetLightingPowerDensity — argument validation
# =============================================================================


class TestLightingPowerDensityValidation:
    """The lighting density must be a positive, bounded number."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module("measures/SetLightingPowerDensity.py", "_lpd_mod")

    def test_valid_density_is_returned(self, mod):
        assert mod.validate_power_density(8.0) == pytest.approx(8.0)

    def test_zero_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_power_density(0.0)
        assert "> 0" in str(excinfo.value)

    def test_negative_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_power_density(-1.0)

    def test_above_maximum_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_power_density(mod.MAX_LIGHTING_POWER_DENSITY_W_M2 + 1.0)
        assert "<=" in str(excinfo.value)

    def test_below_minimum_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_power_density(mod.MIN_LIGHTING_POWER_DENSITY_W_M2 / 2.0)
        assert ">=" in str(excinfo.value)

    def test_nan_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_power_density(float("nan"))

    def test_infinity_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_power_density(float("inf"))

    def test_non_numeric_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_power_density("bright")

    def test_bounds_are_ordered(self, mod):
        assert (
            mod.MIN_LIGHTING_POWER_DENSITY_W_M2
            < mod.DEFAULT_LIGHTING_POWER_DENSITY_W_M2
            < mod.MAX_LIGHTING_POWER_DENSITY_W_M2
        )


# =============================================================================
# SetLightingPowerDensity — measure behaviour (duck-typed model)
# =============================================================================


class TestSetLightingPowerDensityMeasure:
    """Exercise the lighting measure against stand-in models."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module("measures/SetLightingPowerDensity.py", "_lpd_measure")

    def test_resolve_finds_the_lighting_object(self, mod):
        model = _StubLightingModel()
        target, attribute, path = mod.resolve_lighting_target(model)
        assert target is model._lighting
        assert attribute == "lighting_power_density_w_m2"
        assert path == "lighting"

    def test_resolve_falls_back_to_internal_loads(self, mod):
        model = _StubInternalLoadsModel()
        target, attribute, path = mod.resolve_lighting_target(model)
        assert target is model._loads
        assert attribute == "lighting_power_density_w_m2"
        assert path == "internal_loads"

    def test_resolve_falls_back_to_a_model_attribute(self, mod):
        class _AttributeModel:
            lighting_power_density_w_m2 = 10.0

        target, attribute, path = mod.resolve_lighting_target(_AttributeModel())
        assert attribute == "lighting_power_density_w_m2"
        assert path == "model_attribute"

    def test_resolve_raises_when_the_model_has_no_lighting(self, mod):
        with pytest.raises(mod.LightingNotFoundError) as excinfo:
            mod.resolve_lighting_target(_StubModelWithoutLighting())
        assert "no lighting object" in str(excinfo.value)

    def test_error_is_a_value_error(self, mod):
        assert issubclass(mod.LightingNotFoundError, ValueError)

    def test_apply_mutates_and_persists_the_lighting(self, mod):
        model = _StubLightingModel(power_density_w_m2=10.0)
        mod.SetLightingPowerDensity().apply(model, {"power_density_w_m2": 6.5})
        assert model._lighting.lighting_power_density_w_m2 == pytest.approx(6.5)
        assert len(model.lighting_writes) == 1
        assert (
            model.lighting_writes[0].lighting_power_density_w_m2 == pytest.approx(6.5)
        )

    def test_apply_mutates_the_internal_loads_object(self, mod):
        model = _StubInternalLoadsModel(power_density_w_m2=10.0)
        mod.SetLightingPowerDensity().apply(model, {"power_density_w_m2": 4.0})
        assert model._loads.lighting_power_density_w_m2 == pytest.approx(4.0)
        assert len(model.internal_load_writes) == 1

    def test_apply_sets_a_model_attribute_in_place(self, mod):
        class _AttributeModel:
            lighting_power_density_w_m2 = 10.0

        model = _AttributeModel()
        mod.SetLightingPowerDensity().apply(model, {"power_density_w_m2": 7.0})
        assert model.lighting_power_density_w_m2 == pytest.approx(7.0)

    def test_apply_without_a_setter_is_advisory(self, mod):
        model = _StubLightingModel(power_density_w_m2=10.0, with_setter=False)
        # Must not raise even though the model exposes no lighting setter.
        mod.SetLightingPowerDensity().apply(model, {"power_density_w_m2": 5.0})
        assert model.lighting_writes == []
        assert model._lighting.lighting_power_density_w_m2 == pytest.approx(5.0)

    def test_apply_fails_loudly_without_a_lighting_object(self, mod):
        model = _StubModelWithoutLighting()
        with pytest.raises(mod.LightingNotFoundError):
            mod.SetLightingPowerDensity().apply(model, {"power_density_w_m2": 6.0})

    def test_failed_run_does_not_touch_the_model(self, mod):
        model = _StubLightingModel(power_density_w_m2=10.0)
        with pytest.raises(ValueError):
            # An invalid density is rejected before the model is touched.
            mod.SetLightingPowerDensity().apply(model, {"power_density_w_m2": 0.0})
        assert model._lighting.lighting_power_density_w_m2 == pytest.approx(10.0)
        assert model.lighting_writes == []

    def test_apply_rejects_an_invalid_density(self, mod):
        model = _StubLightingModel(power_density_w_m2=10.0)
        with pytest.raises(ValueError):
            mod.SetLightingPowerDensity().apply(model, {"power_density_w_m2": -3.0})
        assert model.lighting_writes == []

    def test_default_density_is_applied(self, mod):
        model = _StubLightingModel(power_density_w_m2=10.0)
        measure = mod.SetLightingPowerDensity()
        args = measure.parse_arguments({})
        measure.apply(model, args)
        assert (
            model._lighting.lighting_power_density_w_m2
            == pytest.approx(mod.DEFAULT_LIGHTING_POWER_DENSITY_W_M2)
        )

    def test_provenance_chain_recorded(self, mod):
        from fluxion.measures import SOURCE_PYTHON_MEASURE, apply_measures

        model = _StubLightingModel(power_density_w_m2=10.0)
        chain: list[dict] = []
        applied = apply_measures(
            model,
            [mod.SetLightingPowerDensity],
            {"SetLightingPowerDensity": {"power_density_w_m2": 6.0}},
            applied_deltas=chain,
        )
        assert applied == ["SetLightingPowerDensity"]
        assert len(chain) == 1
        assert chain[0]["source"] == SOURCE_PYTHON_MEASURE
        assert chain[0]["name"] == "SetLightingPowerDensity"


# =============================================================================
# SetEquipmentPowerDensity — duck-typed stand-ins
# =============================================================================


class _StubEquipment:
    """Stand-in for an equipment snapshot (canonical field name)."""

    def __init__(self, power_density_w_m2=10.0):
        self.equipment_power_density_w_m2 = power_density_w_m2


class _StubEquipmentAlias:
    """Stand-in carrying the ``equipment_power_density`` alias field."""

    def __init__(self, power_density=10.0):
        self.equipment_power_density = power_density


class _StubEquipmentModel:
    """Stand-in for ``fluxion.Model`` exposing an equipment snapshot."""

    def __init__(self, power_density_w_m2=10.0, with_setter=True):
        self._equipment = _StubEquipment(power_density_w_m2)
        self.equipment_writes = []
        if with_setter:
            self.set_equipment = self._set_equipment

    def equipment(self):
        # Mirrors the PyO3 contract: an owned snapshot handed to the caller.
        return self._equipment

    def _set_equipment(self, equipment):
        self.equipment_writes.append(equipment)


class _StubEquipmentInternalLoadsModel:
    """Stand-in for a model whose internal loads carry the EPD."""

    def __init__(self, power_density_w_m2=10.0):
        self._loads = _StubEquipment(power_density_w_m2)
        self.internal_load_writes = []

    def internal_loads(self):
        return self._loads

    def set_internal_loads(self, loads):
        self.internal_load_writes.append(loads)


class _StubModelWithoutEquipment:
    """Stand-in for a model that exposes no equipment object at all."""


# =============================================================================
# SetEquipmentPowerDensity — argument validation
# =============================================================================


class TestEquipmentPowerDensityValidation:
    """The equipment density must be a positive, bounded number."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module(
            "measures/SetEquipmentPowerDensity.py", "_epd_mod"
        )

    def test_valid_density_is_returned(self, mod):
        assert mod.validate_power_density(8.0) == pytest.approx(8.0)

    def test_zero_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_power_density(0.0)
        assert "> 0" in str(excinfo.value)

    def test_negative_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_power_density(-1.0)

    def test_above_maximum_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_power_density(mod.MAX_EQUIPMENT_POWER_DENSITY_W_M2 + 1.0)
        assert "<=" in str(excinfo.value)

    def test_below_minimum_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_power_density(
                mod.MIN_EQUIPMENT_POWER_DENSITY_W_M2 / 2.0
            )
        assert ">=" in str(excinfo.value)

    def test_nan_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_power_density(float("nan"))

    def test_infinity_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_power_density(float("inf"))

    def test_non_numeric_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_power_density("pluggy")

    def test_bounds_are_ordered(self, mod):
        assert (
            mod.MIN_EQUIPMENT_POWER_DENSITY_W_M2
            < mod.DEFAULT_EQUIPMENT_POWER_DENSITY_W_M2
            < mod.MAX_EQUIPMENT_POWER_DENSITY_W_M2
        )

    def test_upper_bound_exceeds_the_lighting_bound(self, mod):
        # Equipment / plug densities are typically larger than lighting
        # densities, so the equipment bound must not clip lighting-scale
        # studies.
        lighting = _load_module(
            "measures/SetLightingPowerDensity.py", "_lpd_bound"
        )
        assert (
            mod.MAX_EQUIPMENT_POWER_DENSITY_W_M2
            > lighting.MAX_LIGHTING_POWER_DENSITY_W_M2
        )


# =============================================================================
# SetEquipmentPowerDensity — measure behaviour (duck-typed model)
# =============================================================================


class TestSetEquipmentPowerDensityMeasure:
    """Exercise the equipment measure against stand-in models."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module(
            "measures/SetEquipmentPowerDensity.py", "_epd_measure"
        )

    def test_resolve_finds_the_equipment_object(self, mod):
        model = _StubEquipmentModel()
        target, attribute, path = mod.resolve_equipment_target(model)
        assert target is model._equipment
        assert attribute == "equipment_power_density_w_m2"
        assert path == "equipment"

    def test_resolve_accepts_the_alias_field(self, mod):
        class _AliasModel:
            def __init__(self):
                self._equipment = _StubEquipmentAlias(10.0)

            def equipment(self):
                return self._equipment

        target, attribute, path = mod.resolve_equipment_target(_AliasModel())
        assert attribute == "equipment_power_density"
        assert path == "equipment"

    def test_resolve_falls_back_to_internal_loads(self, mod):
        model = _StubEquipmentInternalLoadsModel()
        target, attribute, path = mod.resolve_equipment_target(model)
        assert target is model._loads
        assert attribute == "equipment_power_density_w_m2"
        assert path == "internal_loads"

    def test_resolve_falls_back_to_a_model_attribute(self, mod):
        class _AttributeModel:
            equipment_power_density_w_m2 = 10.0

        target, attribute, path = mod.resolve_equipment_target(_AttributeModel())
        assert attribute == "equipment_power_density_w_m2"
        assert path == "model_attribute"

    def test_resolve_falls_back_to_the_alias_model_attribute(self, mod):
        class _AliasAttributeModel:
            equipment_power_density = 10.0

        target, attribute, path = mod.resolve_equipment_target(
            _AliasAttributeModel()
        )
        assert attribute == "equipment_power_density"
        assert path == "model_attribute"

    def test_resolve_raises_when_the_model_has_no_equipment(self, mod):
        with pytest.raises(mod.EquipmentNotFoundError) as excinfo:
            mod.resolve_equipment_target(_StubModelWithoutEquipment())
        assert "no equipment object" in str(excinfo.value)

    def test_resolve_raises_when_only_lighting_is_present(self, mod):
        class _LightingOnlyModel:
            def __init__(self):
                self._lighting = _StubLighting(10.0)

            def lighting(self):
                return self._lighting

        with pytest.raises(mod.EquipmentNotFoundError):
            mod.resolve_equipment_target(_LightingOnlyModel())

    def test_error_is_a_value_error(self, mod):
        assert issubclass(mod.EquipmentNotFoundError, ValueError)

    def test_apply_mutates_and_persists_the_equipment(self, mod):
        model = _StubEquipmentModel(power_density_w_m2=10.0)
        mod.SetEquipmentPowerDensity().apply(model, {"power_density_w_m2": 6.5})
        assert model._equipment.equipment_power_density_w_m2 == pytest.approx(6.5)
        assert len(model.equipment_writes) == 1
        assert (
            model.equipment_writes[0].equipment_power_density_w_m2
            == pytest.approx(6.5)
        )

    def test_apply_mutates_the_internal_loads_object(self, mod):
        model = _StubEquipmentInternalLoadsModel(power_density_w_m2=10.0)
        mod.SetEquipmentPowerDensity().apply(model, {"power_density_w_m2": 4.0})
        assert model._loads.equipment_power_density_w_m2 == pytest.approx(4.0)
        assert len(model.internal_load_writes) == 1

    def test_apply_writes_the_alias_field(self, mod):
        class _AliasModel:
            def __init__(self):
                self._equipment = _StubEquipmentAlias(10.0)
                self.equipment_writes = []

            def equipment(self):
                return self._equipment

            def set_equipment(self, equipment):
                self.equipment_writes.append(equipment)

        model = _AliasModel()
        mod.SetEquipmentPowerDensity().apply(model, {"power_density_w_m2": 3.0})
        assert model._equipment.equipment_power_density == pytest.approx(3.0)
        assert len(model.equipment_writes) == 1

    def test_apply_writes_a_mapping_shaped_target(self, mod):
        class _MappingModel:
            def __init__(self):
                self._equipment = {"equipment_power_density_w_m2": 10.0}

            def equipment(self):
                return self._equipment

        model = _MappingModel()
        mod.SetEquipmentPowerDensity().apply(model, {"power_density_w_m2": 9.0})
        assert model._equipment["equipment_power_density_w_m2"] == pytest.approx(9.0)

    def test_apply_sets_a_model_attribute_in_place(self, mod):
        class _AttributeModel:
            equipment_power_density_w_m2 = 10.0

        model = _AttributeModel()
        mod.SetEquipmentPowerDensity().apply(model, {"power_density_w_m2": 7.0})
        assert model.equipment_power_density_w_m2 == pytest.approx(7.0)

    def test_apply_without_a_setter_is_advisory(self, mod):
        model = _StubEquipmentModel(power_density_w_m2=10.0, with_setter=False)
        # Must not raise even though the model exposes no equipment setter.
        mod.SetEquipmentPowerDensity().apply(model, {"power_density_w_m2": 5.0})
        assert model.equipment_writes == []
        assert model._equipment.equipment_power_density_w_m2 == pytest.approx(5.0)

    def test_apply_fails_loudly_without_an_equipment_object(self, mod):
        model = _StubModelWithoutEquipment()
        with pytest.raises(mod.EquipmentNotFoundError):
            mod.SetEquipmentPowerDensity().apply(model, {"power_density_w_m2": 6.0})

    def test_failed_run_does_not_touch_the_model(self, mod):
        model = _StubEquipmentModel(power_density_w_m2=10.0)
        with pytest.raises(ValueError):
            # An invalid density is rejected before the model is touched.
            mod.SetEquipmentPowerDensity().apply(model, {"power_density_w_m2": 0.0})
        assert model._equipment.equipment_power_density_w_m2 == pytest.approx(10.0)
        assert model.equipment_writes == []

    def test_apply_rejects_an_invalid_density(self, mod):
        model = _StubEquipmentModel(power_density_w_m2=10.0)
        with pytest.raises(ValueError):
            mod.SetEquipmentPowerDensity().apply(model, {"power_density_w_m2": -3.0})
        assert model.equipment_writes == []

    def test_default_density_is_applied(self, mod):
        model = _StubEquipmentModel(power_density_w_m2=10.0)
        measure = mod.SetEquipmentPowerDensity()
        args = measure.parse_arguments({})
        measure.apply(model, args)
        assert (
            model._equipment.equipment_power_density_w_m2
            == pytest.approx(mod.DEFAULT_EQUIPMENT_POWER_DENSITY_W_M2)
        )

    def test_provenance_chain_recorded(self, mod):
        from fluxion.measures import SOURCE_PYTHON_MEASURE, apply_measures

        model = _StubEquipmentModel(power_density_w_m2=10.0)
        chain: list[dict] = []
        applied = apply_measures(
            model,
            [mod.SetEquipmentPowerDensity],
            {"SetEquipmentPowerDensity": {"power_density_w_m2": 6.0}},
            applied_deltas=chain,
        )
        assert applied == ["SetEquipmentPowerDensity"]
        assert len(chain) == 1
        assert chain[0]["source"] == SOURCE_PYTHON_MEASURE
        assert chain[0]["name"] == "SetEquipmentPowerDensity"


# =============================================================================
# SetOccupancyDensity — duck-typed stand-ins
# =============================================================================


class _StubOccupancy:
    """Stand-in for an occupancy snapshot (canonical field name)."""

    def __init__(self, occupancy_density=0.1):
        self.occupancy_density = occupancy_density


class _StubOccupancyModel:
    """Stand-in for ``fluxion.Model`` exposing an occupancy snapshot."""

    def __init__(self, occupancy_density=0.1, with_setter=True):
        self._occupancy = _StubOccupancy(occupancy_density)
        self.occupancy_writes = []
        if with_setter:
            self.set_occupancy = self._set_occupancy

    def occupancy(self):
        return self._occupancy

    def _set_occupancy(self, occupancy):
        self.occupancy_writes.append(occupancy)


class _StubModelWithoutOccupancy:
    """Stand-in for a model that exposes no occupancy object at all."""


# =============================================================================
# SetOccupancyDensity — argument validation
# =============================================================================


class TestOccupancyDensityValidation:
    """The occupancy density must be a positive, bounded number."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module(
            "measures/SetOccupancyDensity.py", "_occ_validation"
        )

    def test_valid_density_is_returned(self, mod):
        assert mod.validate_occupancy_density(0.1) == pytest.approx(0.1)

    def test_zero_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_occupancy_density(0.0)
        assert "> 0" in str(excinfo.value)

    def test_negative_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_occupancy_density(-0.5)

    def test_below_minimum_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_occupancy_density(
                mod.MIN_OCCUPANCY_DENSITY_PERSONS_PER_M2 / 2.0
            )
        assert ">=" in str(excinfo.value)

    def test_above_maximum_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_occupancy_density(
                mod.MAX_OCCUPANCY_DENSITY_PERSONS_PER_M2 + 1.0
            )
        assert "<=" in str(excinfo.value)

    def test_nan_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_occupancy_density(float("nan"))

    def test_non_numeric_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_occupancy_density("crowded")

    def test_bounds_are_ordered(self, mod):
        assert (
            mod.MIN_OCCUPANCY_DENSITY_PERSONS_PER_M2
            < mod.DEFAULT_OCCUPANCY_DENSITY_PERSONS_PER_M2
            < mod.MAX_OCCUPANCY_DENSITY_PERSONS_PER_M2
        )


# =============================================================================
# SetOccupancyDensity — measure behaviour (duck-typed model)
# =============================================================================


class TestSetOccupancyDensityMeasure:
    """Exercise the occupancy measure against stand-in models."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module("measures/SetOccupancyDensity.py", "_occ_measure")

    def test_resolve_finds_the_occupancy_object(self, mod):
        model = _StubOccupancyModel()
        target, attribute, path = mod.resolve_occupancy_target(model)
        assert target is model._occupancy
        assert attribute == "occupancy_density"
        assert path == "occupancy"

    def test_resolve_raises_when_the_model_has_no_occupancy(self, mod):
        with pytest.raises(mod.OccupancyNotFoundError):
            mod.resolve_occupancy_target(_StubModelWithoutOccupancy())

    def test_apply_mutates_and_persists_the_occupancy(self, mod):
        model = _StubOccupancyModel()
        mod.SetOccupancyDensity().apply(
            model, {"occupancy_density_persons_per_m2": 0.05}
        )
        assert model._occupancy.occupancy_density == pytest.approx(0.05)
        assert model.occupancy_writes == [model._occupancy]

    def test_apply_without_a_setter_is_advisory(self, mod):
        model = _StubOccupancyModel(with_setter=False)
        mod.SetOccupancyDensity().apply(
            model, {"occupancy_density_persons_per_m2": 0.2}
        )
        assert model._occupancy.occupancy_density == pytest.approx(0.2)

    def test_apply_fails_loudly_without_an_occupancy_object(self, mod):
        with pytest.raises(mod.OccupancyNotFoundError):
            mod.SetOccupancyDensity().apply(
                _StubModelWithoutOccupancy(),
                {"occupancy_density_persons_per_m2": 0.1},
            )

    def test_apply_rejects_an_invalid_density(self, mod):
        model = _StubOccupancyModel()
        with pytest.raises(ValueError):
            mod.SetOccupancyDensity().apply(
                model, {"occupancy_density_persons_per_m2": 0.0}
            )
        assert model._occupancy.occupancy_density == pytest.approx(0.1)
        assert model.occupancy_writes == []

    def test_default_density_is_applied(self, mod):
        model = _StubOccupancyModel(occupancy_density=0.5)
        mod.SetOccupancyDensity().apply(model, {})
        assert model._occupancy.occupancy_density == pytest.approx(
            mod.DEFAULT_OCCUPANCY_DENSITY_PERSONS_PER_M2
        )


# =============================================================================
# SetHVACEfficiency — duck-typed stand-ins
# =============================================================================


class _StubHVAC:
    """Stand-in for the ``fluxion.HVACSystem`` snapshot."""

    def __init__(self, cop_heating=3.0, cop_cooling=3.2):
        self.heating_capacity = 10000.0
        self.cooling_capacity = 8000.0
        self.cop_heating = cop_heating
        self.cop_cooling = cop_cooling


class _StubHVACModel:
    """Stand-in for ``fluxion.Model`` exposing the HVAC snapshot."""

    def __init__(self, cop_heating=3.0, cop_cooling=3.2, with_setter=True):
        self._hvac = _StubHVAC(cop_heating, cop_cooling)
        self.hvac_writes = []
        if with_setter:
            self.set_hvac_system = self._set_hvac_system

    def hvac_system(self):
        return self._hvac

    def _set_hvac_system(self, hvac):
        self.hvac_writes.append(hvac)


class _StubModelWithoutHVAC:
    """Stand-in for a model that exposes no HVAC system at all."""


# =============================================================================
# SetHVACEfficiency — argument validation
# =============================================================================


class TestHVACEfficiencyValidation:
    """The COP must be a positive, bounded number."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module("measures/SetHVACEfficiency.py", "_cop_mod")

    def test_valid_cop_is_returned(self, mod):
        assert mod.validate_cop(3.5) == pytest.approx(3.5)

    def test_zero_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_cop(0.0, name="cop_heating")
        assert "cop_heating" in str(excinfo.value)
        assert "> 0" in str(excinfo.value)

    def test_negative_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_cop(-2.0)

    def test_above_maximum_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_cop(mod.MAX_COP + 1.0, name="cop_cooling")
        assert "cop_cooling" in str(excinfo.value)
        assert "<=" in str(excinfo.value)

    def test_below_minimum_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_cop(mod.MIN_COP / 2.0)

    def test_nan_and_infinity_are_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_cop(float("nan"))
        with pytest.raises(ValueError):
            mod.validate_cop(float("inf"))

    def test_non_numeric_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_cop("efficient")

    def test_bounds_are_ordered(self, mod):
        assert 0.0 < mod.MIN_COP < mod.MAX_COP


# =============================================================================
# SetHVACEfficiency — measure behaviour (duck-typed model)
# =============================================================================


class TestSetHVACEfficiencyMeasure:
    """Exercise the plant-efficiency measure against stand-in models."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module("measures/SetHVACEfficiency.py", "_cop_measure")

    def test_resolve_returns_the_hvac_snapshot(self, mod):
        model = _StubHVACModel()
        assert mod.resolve_hvac_snapshot(model) is model._hvac

    def test_resolve_raises_without_an_hvac_system(self, mod):
        with pytest.raises(mod.HVACSystemNotFoundError) as excinfo:
            mod.resolve_hvac_snapshot(_StubModelWithoutHVAC())
        assert "no HVAC system" in str(excinfo.value)

    def test_resolve_raises_when_the_snapshot_has_no_cop_fields(self, mod):
        class _BareSnapshot:
            heating_capacity = 10000.0

        class _BareModel:
            def hvac_system(self):
                return _BareSnapshot()

        with pytest.raises(mod.HVACSystemNotFoundError):
            mod.resolve_hvac_snapshot(_BareModel())

    def test_error_is_a_value_error(self, mod):
        assert issubclass(mod.HVACSystemNotFoundError, ValueError)

    def test_apply_sets_both_cops_and_pushes_back(self, mod):
        model = _StubHVACModel(cop_heating=3.0, cop_cooling=3.2)
        mod.SetHVACEfficiency().apply(
            model, {"cop_heating": 4.0, "cop_cooling": 4.5}
        )
        assert model._hvac.cop_heating == pytest.approx(4.0)
        assert model._hvac.cop_cooling == pytest.approx(4.5)
        assert len(model.hvac_writes) == 1
        assert model.hvac_writes[0].cop_heating == pytest.approx(4.0)

    def test_apply_preserves_the_unspecified_cop(self, mod):
        model = _StubHVACModel(cop_heating=3.0, cop_cooling=3.2)
        mod.SetHVACEfficiency().apply(model, {"cop_heating": 4.0})
        assert model._hvac.cop_heating == pytest.approx(4.0)
        assert model._hvac.cop_cooling == pytest.approx(3.2)

    def test_apply_without_arguments_is_a_no_op(self, mod):
        model = _StubHVACModel(cop_heating=3.0, cop_cooling=3.2)
        measure = mod.SetHVACEfficiency()
        measure.apply(model, measure.parse_arguments({}))
        assert model._hvac.cop_heating == pytest.approx(3.0)
        assert model._hvac.cop_cooling == pytest.approx(3.2)

    def test_apply_without_a_setter_is_advisory(self, mod):
        model = _StubHVACModel(cop_heating=3.0, cop_cooling=3.2, with_setter=False)
        mod.SetHVACEfficiency().apply(model, {"cop_heating": 4.0})
        assert model.hvac_writes == []
        assert model._hvac.cop_heating == pytest.approx(4.0)

    def test_apply_fails_loudly_without_an_hvac_system(self, mod):
        with pytest.raises(mod.HVACSystemNotFoundError):
            mod.SetHVACEfficiency().apply(
                _StubModelWithoutHVAC(), {"cop_heating": 4.0}
            )

    def test_apply_rejects_an_invalid_cop_without_mutating(self, mod):
        model = _StubHVACModel(cop_heating=3.0, cop_cooling=3.2)
        with pytest.raises(ValueError):
            mod.SetHVACEfficiency().apply(model, {"cop_heating": 0.0})
        assert model._hvac.cop_heating == pytest.approx(3.0)
        assert model.hvac_writes == []

    def test_provenance_chain_recorded(self, mod):
        from fluxion.measures import SOURCE_PYTHON_MEASURE, apply_measures

        model = _StubHVACModel()
        chain: list[dict] = []
        applied = apply_measures(
            model,
            [mod.SetHVACEfficiency],
            {"SetHVACEfficiency": {"cop_heating": 4.0, "cop_cooling": 4.5}},
            applied_deltas=chain,
        )
        assert applied == ["SetHVACEfficiency"]
        assert len(chain) == 1
        assert chain[0]["source"] == SOURCE_PYTHON_MEASURE
        assert chain[0]["name"] == "SetHVACEfficiency"


# =============================================================================
# Integration — native bindings
# =============================================================================


@requires_fluxion
class TestSetHVACEfficiencyIntegration:
    """End-to-end plant-efficiency mutation on a real fluxion.Model."""

    def _import_measure(self):
        return _load_module(
            "measures/SetHVACEfficiency.py", "_cop_int"
        ).SetHVACEfficiency

    def test_cops_are_written_on_the_hvac_snapshot(self):
        import fluxion

        model = fluxion.Model(num_zones=1)
        self._import_measure().apply(
            model, {"cop_heating": 4.0, "cop_cooling": 4.5}
        )
        hvac = model.hvac_system()
        assert hvac.cop_heating == pytest.approx(4.0)
        assert hvac.cop_cooling == pytest.approx(4.5)

    def test_invalid_cop_is_rejected(self):
        import fluxion

        model = fluxion.Model(num_zones=1)
        with pytest.raises(ValueError):
            self._import_measure().apply(model, {"cop_heating": -1.0})


@requires_fluxion
class TestSetLightingPowerDensityIntegration:
    """The native model exposes no lighting object — the measure says so."""

    def _import_measure(self):
        return _load_module(
            "measures/SetLightingPowerDensity.py", "_lpd_int"
        ).SetLightingPowerDensity

    def test_native_model_fails_loudly(self):
        import fluxion

        mod = _load_module("measures/SetLightingPowerDensity.py", "_lpd_int2")
        model = fluxion.Model(num_zones=1)
        with pytest.raises(mod.LightingNotFoundError):
            self._import_measure().apply(model, {"power_density_w_m2": 6.0})


@requires_fluxion
class TestSetEquipmentPowerDensityIntegration:
    """The native model exposes no equipment object — the measure says so."""

    def _import_measure(self):
        return _load_module(
            "measures/SetEquipmentPowerDensity.py", "_epd_int"
        ).SetEquipmentPowerDensity

    def test_native_model_fails_loudly(self):
        import fluxion

        mod = _load_module("measures/SetEquipmentPowerDensity.py", "_epd_int2")
        model = fluxion.Model(num_zones=1)
        with pytest.raises(mod.EquipmentNotFoundError):
            self._import_measure().apply(model, {"power_density_w_m2": 6.0})


# =============================================================================
# AddOverhang — duck-typed stand-ins
# =============================================================================


class _StubOrientation:
    """Stand-in for the native ``fluxion.Orientation`` enum member.

    The native enum exposes ``.name`` (``"South"``) and a ``repr`` of
    ``"Orientation.South"``; both spellings are exercised.
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Orientation.{self.name}"


class _StubOrientationReprOnly:
    """Stand-in exposing only the ``repr`` spelling of an orientation."""

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f"Orientation.{self._name}"


class _StubSurface:
    """Stand-in for the native ``fluxion.Surface`` snapshot."""

    def __init__(self, orientation="South", window_area=0.0, area=10.0):
        self.orientation = orientation
        self.window_area = window_area
        self.area = area
        self.u_value = 0.5
        self.overhang_depth = None
        self.overhang_height = None
        self.overhang_calls = []

    def add_overhang(self, depth, height):
        # Mirrors the native helper: sets the shorthand fields and
        # appends a shading device.
        self.overhang_depth = depth
        self.overhang_height = height
        self.overhang_calls.append((depth, height))


class _StubSurfaceModel:
    """Stand-in for ``fluxion.Model`` exposing the surface snapshots."""

    def __init__(self, surfaces, with_setter=True):
        self._surfaces = list(surfaces)
        self.surface_writes = []
        if with_setter:
            self.set_surfaces = self._set_surfaces

    def surfaces(self):
        # Mirrors the PyO3 contract: an owned snapshot handed to the
        # caller.
        return list(self._surfaces)

    def _set_surfaces(self, surfaces):
        self.surface_writes.append(list(surfaces))
        self._surfaces = list(surfaces)


class _StubSurfaceModelWithoutSurfaces:
    """Stand-in for a model that exposes no surfaces accessor at all."""


def _overhang_model():
    """A model with one surface per cardinal orientation."""
    return _StubSurfaceModel(
        [
            _StubSurface("North", window_area=5.0),
            _StubSurface("East", window_area=5.0),
            _StubSurface("South", window_area=5.0),
            _StubSurface("West", window_area=0.0),
            _StubSurface("Up", window_area=0.0),
        ]
    )


# =============================================================================
# AddOverhang — pure-Python helpers (no native bindings needed)
# =============================================================================


class TestAddOverhangHelpers:
    """Unit-test the geometry / filter helpers without native bindings."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module("measures/AddOverhang.py", "_overhang_mod")

    def test_valid_depth_is_returned(self, mod):
        assert mod.validate_depth(1.0) == pytest.approx(1.0)

    def test_depth_below_minimum_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_depth(-0.1)
        assert "depth" in str(excinfo.value)
        assert ">=" in str(excinfo.value)

    def test_depth_above_maximum_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_depth(mod.MAX_OVERHANG_DEPTH_M + 0.1)
        assert "<=" in str(excinfo.value)

    def test_height_bounds_are_enforced(self, mod):
        assert mod.validate_height(mod.MAX_OVERHANG_HEIGHT_M) == pytest.approx(
            mod.MAX_OVERHANG_HEIGHT_M
        )
        with pytest.raises(ValueError):
            mod.validate_height(mod.MAX_OVERHANG_HEIGHT_M + 1.0)
        with pytest.raises(ValueError):
            mod.validate_height(-1.0)

    def test_nan_and_infinity_are_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_depth(float("nan"))
        with pytest.raises(ValueError):
            mod.validate_height(float("inf"))

    def test_non_numeric_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_depth("wide")

    def test_bounds_are_ordered(self, mod):
        assert (
            mod.MIN_OVERHANG_DEPTH_M
            <= mod.DEFAULT_OVERHANG_DEPTH_M
            <= mod.MAX_OVERHANG_DEPTH_M
        )
        assert (
            mod.MIN_OVERHANG_HEIGHT_M
            <= mod.DEFAULT_OVERHANG_HEIGHT_M
            <= mod.MAX_OVERHANG_HEIGHT_M
        )

    def test_orientation_name_reads_the_enum_name(self, mod):
        assert mod.orientation_name(_StubOrientation("South")) == "south"

    def test_orientation_name_reads_the_repr(self, mod):
        assert mod.orientation_name(_StubOrientationReprOnly("West")) == "west"

    def test_orientation_name_accepts_plain_strings(self, mod):
        assert mod.orientation_name("North") == "north"

    def test_normalize_orientation_is_case_insensitive(self, mod):
        assert mod.normalize_orientation("south") == "south"
        assert mod.normalize_orientation("SOUTH") == "south"
        assert mod.normalize_orientation("  West  ") == "west"

    def test_normalize_orientation_accepts_enum_values(self, mod):
        assert mod.normalize_orientation(_StubOrientation("East")) == "east"

    def test_normalize_orientation_accepts_the_all_aliases(self, mod):
        assert mod.normalize_orientation("all") == mod.ORIENTATION_ALL
        assert mod.normalize_orientation("all exterior") == mod.ORIENTATION_ALL
        assert mod.normalize_orientation("all_exterior") == mod.ORIENTATION_ALL

    def test_normalize_orientation_rejects_unknown_spellings(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.normalize_orientation("sideways")
        assert "south" in str(excinfo.value)

    def test_roofs_and_floors_are_not_vertical_orientations(self, mod):
        assert "up" not in mod.VERTICAL_ORIENTATIONS
        assert "down" not in mod.VERTICAL_ORIENTATIONS
        assert "horizontal" not in mod.VERTICAL_ORIENTATIONS

    def test_matches_orientation_matches_the_named_facade(self, mod):
        assert mod.matches_orientation("South", "south") is True
        assert mod.matches_orientation("North", "south") is False

    def test_matches_orientation_all_covers_every_cardinal_facade(self, mod):
        for name in ("North", "East", "South", "West"):
            assert mod.matches_orientation(name, mod.ORIENTATION_ALL) is True
        for name in ("Up", "Down", "Horizontal"):
            assert mod.matches_orientation(name, mod.ORIENTATION_ALL) is False

    def test_is_eligible_respects_the_window_filter(self, mod):
        glazed = _StubSurface("South", window_area=4.0)
        opaque = _StubSurface("South", window_area=0.0)
        assert mod.is_eligible(glazed, "south", False) is True
        assert mod.is_eligible(opaque, "south", False) is True
        assert mod.is_eligible(glazed, "south", True) is True
        assert mod.is_eligible(opaque, "south", True) is False

    def test_is_eligible_respects_the_orientation(self, mod):
        north = _StubSurface("North", window_area=4.0)
        assert mod.is_eligible(north, "south", True) is False
        assert mod.is_eligible(north, "north", True) is True

    def test_select_eligible_returns_the_matching_surfaces(self, mod):
        surfaces = [
            _StubSurface("North", window_area=5.0),
            _StubSurface("South", window_area=5.0),
            _StubSurface("South", window_area=0.0),
        ]
        selected = mod.select_eligible(surfaces, "south", False)
        assert len(selected) == 2
        assert all(s.orientation == "South" for s in selected)

        glazed_only = mod.select_eligible(surfaces, "south", True)
        assert len(glazed_only) == 1
        assert glazed_only[0].window_area == pytest.approx(5.0)


# =============================================================================
# AddOverhang — measure behaviour (duck-typed model)
# =============================================================================


class TestAddOverhangMeasure:
    """Exercise the shading measure against stand-in models."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module("measures/AddOverhang.py", "_overhang_measure")

    def test_error_is_a_value_error(self, mod):
        assert issubclass(mod.NoEligibleSurfaceError, ValueError)

    def test_apply_shadows_the_south_facade_by_default(self, mod):
        model = _overhang_model()
        mod.AddOverhang().apply(model, {})
        shaded = [s for s in model.surfaces() if s.overhang_calls]
        assert len(shaded) == 1
        assert shaded[0].orientation == "South"
        assert shaded[0].overhang_depth == pytest.approx(
            mod.DEFAULT_OVERHANG_DEPTH_M
        )
        assert shaded[0].overhang_height == pytest.approx(
            mod.DEFAULT_OVERHANG_HEIGHT_M
        )

    def test_apply_pushes_the_surfaces_back(self, mod):
        model = _overhang_model()
        mod.AddOverhang().apply(model, {"depth": 0.75, "height": 3.0})
        assert len(model.surface_writes) == 1
        written = model.surface_writes[0]
        assert len(written) == 5
        assert any(s.overhang_depth == pytest.approx(0.75) for s in written)

    def test_apply_honours_the_orientation_argument(self, mod):
        model = _overhang_model()
        mod.AddOverhang().apply(model, {"orientation": "north"})
        shaded = [s for s in model.surfaces() if s.overhang_calls]
        assert len(shaded) == 1
        assert shaded[0].orientation == "North"

    def test_apply_all_covers_every_cardinal_facade(self, mod):
        model = _overhang_model()
        mod.AddOverhang().apply(model, {"orientation": "all"})
        shaded = [s for s in model.surfaces() if s.overhang_calls]
        # North, East, South, West — the roof is never eligible.
        assert len(shaded) == 4
        assert all(s.orientation != "Up" for s in shaded)

    def test_apply_honours_only_with_windows(self, mod):
        model = _overhang_model()
        mod.AddOverhang().apply(
            model, {"orientation": "all", "only_with_windows": True}
        )
        shaded = [s for s in model.surfaces() if s.overhang_calls]
        assert len(shaded) == 3
        assert all(s.window_area > 0.0 for s in shaded)

    def test_apply_raises_when_no_surface_is_eligible(self, mod):
        model = _StubSurfaceModel([_StubSurface("North", window_area=2.0)])
        with pytest.raises(mod.NoEligibleSurfaceError) as excinfo:
            mod.AddOverhang().apply(model, {"orientation": "south"})
        assert "eligible" in str(excinfo.value)
        # The model must be untouched.
        assert model.surface_writes == []
        assert all(not s.overhang_calls for s in model.surfaces())

    def test_apply_raises_when_no_glazed_surface_is_eligible(self, mod):
        model = _StubSurfaceModel([_StubSurface("South", window_area=0.0)])
        with pytest.raises(mod.NoEligibleSurfaceError):
            mod.AddOverhang().apply(model, {"only_with_windows": True})
        assert model.surface_writes == []

    def test_apply_rejects_an_out_of_range_depth(self, mod):
        model = _overhang_model()
        with pytest.raises(ValueError):
            mod.AddOverhang().apply(model, {"depth": -1.0})
        assert model.surface_writes == []
        assert all(not s.overhang_calls for s in model.surfaces())

    def test_apply_rejects_an_out_of_range_height(self, mod):
        model = _overhang_model()
        with pytest.raises(ValueError):
            mod.AddOverhang().apply(
                model, {"height": mod.MAX_OVERHANG_HEIGHT_M + 1.0}
            )
        assert model.surface_writes == []

    def test_apply_rejects_an_unknown_orientation(self, mod):
        model = _overhang_model()
        with pytest.raises(ValueError):
            mod.AddOverhang().apply(model, {"orientation": "diagonal"})
        assert model.surface_writes == []
        assert all(not s.overhang_calls for s in model.surfaces())

    def test_apply_without_a_surfaces_accessor_fails_loudly(self, mod):
        with pytest.raises(mod.NoEligibleSurfaceError):
            mod.AddOverhang().apply(_StubSurfaceModelWithoutSurfaces(), {})

    def test_apply_without_a_setter_is_advisory(self, mod):
        model = _StubSurfaceModel(
            [
                _StubSurface("South", window_area=2.0),
                _StubSurface("North", window_area=2.0),
            ],
            with_setter=False,
        )
        mod.AddOverhang().apply(model, {})
        # No push-back is possible, but the snapshot itself carries the
        # overhang (the advisory pattern of the sibling measures).
        assert model.surface_writes == []
        assert model._surfaces[0].overhang_depth == pytest.approx(
            mod.DEFAULT_OVERHANG_DEPTH_M
        )
        assert model._surfaces[1].overhang_calls == []

    def test_measure_runs_through_the_aot_runner(self, mod):
        from fluxion.measures import SOURCE_PYTHON_MEASURE, apply_measures

        model = _overhang_model()
        chain: list[dict] = []
        applied = apply_measures(
            model,
            [mod.AddOverhang],
            {"AddOverhang": {"depth": 1.5, "orientation": "south"}},
            applied_deltas=chain,
        )
        assert applied == ["AddOverhang"]
        assert len(chain) == 1
        assert chain[0]["source"] == SOURCE_PYTHON_MEASURE
        assert chain[0]["name"] == "AddOverhang"
        shaded = [s for s in model.surfaces() if s.overhang_calls]
        assert len(shaded) == 1
        assert shaded[0].overhang_depth == pytest.approx(1.5)


# =============================================================================
# SetWindowProperties — duck-typed stand-ins
# =============================================================================


class _StubWindowSpec:
    """Stand-in for a Rust ``WindowSpec`` (``src/api/schema.rs``)."""

    def __init__(self, u_value=5.8, shgc=0.86):
        self.window_u_value = u_value
        self.window_shgc = shgc


class _StubWindowSpecModel:
    """Stand-in for a model exposing a window-spec collection."""

    def __init__(self, specs=None, with_setter=True):
        self._specs = list(specs) if specs is not None else [_StubWindowSpec()]
        self.window_writes = []
        if with_setter:
            self.set_window_specs = self._set_window_specs

    def window_specs(self):
        # Mirrors the PyO3 contract: an owned snapshot handed to the
        # caller. The elements are shared, so an in-place mutation of a
        # spec is visible on the model.
        return list(self._specs)

    def _set_window_specs(self, specs):
        self.window_writes.append(list(specs))
        self._specs = list(specs)


class _StubPartialWindowSpec:
    """Stand-in for a window spec that carries only the U-value."""

    def __init__(self, u_value=5.8):
        self.window_u_value = u_value


class _StubWindowsAliasModel:
    """Stand-in exposing the shorter ``windows()`` accessor."""

    def __init__(self, specs=None, with_setter=True):
        self._specs = list(specs) if specs is not None else [_StubWindowSpec()]
        self.window_writes = []
        if with_setter:
            self.set_windows = self._set_windows

    def windows(self):
        return list(self._specs)

    def _set_windows(self, specs):
        self.window_writes.append(list(specs))
        self._specs = list(specs)


class _StubGlazedSurface:
    """Stand-in for a schema-shaped surface nesting its window spec."""

    def __init__(self, u_value=5.8, shgc=0.86):
        self.window = _StubWindowSpec(u_value, shgc)


class _StubSurfaceWithWindowModel:
    """Stand-in whose surfaces carry the glazing fields."""

    def __init__(self, surfaces, with_setter=True):
        self._surfaces = list(surfaces)
        self.surface_writes = []
        if with_setter:
            self.set_surfaces = self._set_surfaces

    def surfaces(self):
        return list(self._surfaces)

    def _set_surfaces(self, surfaces):
        self.surface_writes.append(list(surfaces))
        self._surfaces = list(surfaces)


class _StubModelWithWindowFields:
    """Stand-in whose model object holds the glazing fields itself."""

    def __init__(self, u_value=5.8, shgc=0.86):
        self.window_u_value = u_value
        self.window_shgc = shgc


class _StubModelWithoutWindows:
    """Stand-in for a model that exposes no window spec at all."""


# =============================================================================
# SetWindowProperties — pure validators (no native bindings needed)
# =============================================================================


class TestSetWindowPropertiesValidation:
    """The glazing U-value and SHGC must be valid numbers."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module(
            "measures/SetWindowProperties.py", "_window_props_mod"
        )

    def test_valid_u_value_is_returned(self, mod):
        assert mod.validate_u_value(1.7) == pytest.approx(1.7)

    def test_u_value_cap_is_accepted(self, mod):
        assert mod.validate_u_value(mod.MAX_WINDOW_U_VALUE_W_M2K) == pytest.approx(
            mod.MAX_WINDOW_U_VALUE_W_M2K
        )

    def test_zero_u_value_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_u_value(0.0)
        assert "> 0" in str(excinfo.value)

    def test_negative_u_value_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_u_value(-1.0)

    def test_u_value_above_the_cap_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_u_value(mod.MAX_WINDOW_U_VALUE_W_M2K + 0.1)
        assert "<=" in str(excinfo.value)

    def test_u_value_nan_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_u_value(float("nan"))

    def test_u_value_infinity_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_u_value(float("inf"))

    def test_non_numeric_u_value_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_u_value("single pane")

    def test_valid_shgc_is_returned(self, mod):
        assert mod.validate_shgc(0.35) == pytest.approx(0.35)

    def test_shgc_of_one_is_valid(self, mod):
        assert mod.validate_shgc(1.0) == pytest.approx(1.0)

    def test_zero_shgc_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_shgc(0.0)
        assert "> 0" in str(excinfo.value)

    def test_negative_shgc_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_shgc(-0.2)

    def test_shgc_above_one_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_shgc(1.5)
        assert "<= 1" in str(excinfo.value)

    def test_shgc_nan_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_shgc(float("nan"))

    def test_shgc_infinity_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_shgc(float("inf"))

    def test_non_numeric_shgc_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_shgc("low-e")

    def test_bounds_match_the_rust_validator(self, mod):
        # The Rust validator requires window_u_value > 0 and
        # 0 < window_shgc <= 1 (src/api/schema.rs).
        assert mod.MAX_WINDOW_SHGC == pytest.approx(1.0)
        assert mod.U_VALUE_FIELD == "window_u_value"
        assert mod.SHGC_FIELD == "window_shgc"

    def test_u_value_cap_is_generous(self, mod):
        # Single-pane clear glass is ≈ 5.8 W/m²·K — the cap must sit
        # above it, so a real single-pane upgrade stays modellable.
        assert mod.MAX_WINDOW_U_VALUE_W_M2K > 5.8


# =============================================================================
# SetWindowProperties — measure behaviour (duck-typed model)
# =============================================================================


class TestSetWindowPropertiesMeasure:
    """Exercise the glazing measure against stand-in models."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module(
            "measures/SetWindowProperties.py", "_window_props_measure"
        )

    def test_error_is_a_value_error(self, mod):
        assert issubclass(mod.WindowNotFoundError, ValueError)

    def test_resolve_finds_the_window_spec_collection(self, mod):
        model = _StubWindowSpecModel()
        groups = mod.resolve_window_specs(model)
        assert len(groups) == 1
        assert groups[0].path == "window_specs"
        assert groups[0].specs == [model._specs[0]]

    def test_resolve_finds_the_windows_alias(self, mod):
        model = _StubWindowsAliasModel()
        groups = mod.resolve_window_specs(model)
        assert groups[0].path == "windows"

    def test_resolve_finds_surfaces_carrying_the_window_fields(self, mod):
        model = _StubSurfaceWithWindowModel([_StubGlazedSurface()])
        groups = mod.resolve_window_specs(model)
        assert groups[0].path == "surfaces"
        assert groups[0].specs[0].window_u_value == pytest.approx(5.8)

    def test_resolve_finds_the_model_attributes(self, mod):
        model = _StubModelWithWindowFields()
        groups = mod.resolve_window_specs(model)
        assert groups[0].path == "model_attribute"
        assert groups[0].specs == [model]

    def test_resolve_raises_when_the_model_has_no_window_spec(self, mod):
        with pytest.raises(mod.WindowNotFoundError) as excinfo:
            mod.resolve_window_specs(_StubModelWithoutWindows())
        assert "no window spec" in str(excinfo.value)

    def test_apply_sets_both_glazing_properties(self, mod):
        model = _StubWindowSpecModel()
        mod.SetWindowProperties().apply(
            model, {"u_value_w_m2k": 1.7, "shgc": 0.35}
        )
        spec = model._specs[0]
        assert spec.window_u_value == pytest.approx(1.7)
        assert spec.window_shgc == pytest.approx(0.35)

    def test_apply_with_only_a_u_value_leaves_the_shgc_untouched(self, mod):
        model = _StubWindowSpecModel([_StubWindowSpec(5.8, 0.86)])
        mod.SetWindowProperties().apply(model, {"u_value_w_m2k": 1.7})
        spec = model._specs[0]
        assert spec.window_u_value == pytest.approx(1.7)
        assert spec.window_shgc == pytest.approx(0.86)

    def test_apply_with_only_an_shgc_leaves_the_u_value_untouched(self, mod):
        model = _StubWindowSpecModel([_StubWindowSpec(5.8, 0.86)])
        mod.SetWindowProperties().apply(model, {"shgc": 0.35})
        spec = model._specs[0]
        assert spec.window_shgc == pytest.approx(0.35)
        assert spec.window_u_value == pytest.approx(5.8)

    def test_apply_mutates_every_window_spec(self, mod):
        model = _StubWindowSpecModel(
            [_StubWindowSpec(), _StubWindowSpec(), _StubWindowSpec()]
        )
        mod.SetWindowProperties().apply(
            model, {"u_value_w_m2k": 1.7, "shgc": 0.35}
        )
        assert len(model._specs) == 3
        for spec in model._specs:
            assert spec.window_u_value == pytest.approx(1.7)
            assert spec.window_shgc == pytest.approx(0.35)

    def test_apply_pushes_the_window_specs_back(self, mod):
        model = _StubWindowSpecModel()
        mod.SetWindowProperties().apply(model, {"u_value_w_m2k": 1.7})
        assert len(model.window_writes) == 1
        written = model.window_writes[0]
        assert written[0].window_u_value == pytest.approx(1.7)

    def test_apply_mutates_surfaces_carrying_the_window_fields(self, mod):
        model = _StubSurfaceWithWindowModel(
            [_StubGlazedSurface(), _StubGlazedSurface()]
        )
        mod.SetWindowProperties().apply(
            model, {"u_value_w_m2k": 1.7, "shgc": 0.35}
        )
        for surface in model._surfaces:
            assert surface.window.window_u_value == pytest.approx(1.7)
            assert surface.window.window_shgc == pytest.approx(0.35)
        assert len(model.surface_writes) == 1

    def test_apply_sets_the_model_attributes_in_place(self, mod):
        model = _StubModelWithWindowFields()
        mod.SetWindowProperties().apply(model, {"shgc": 0.35})
        assert model.window_shgc == pytest.approx(0.35)
        assert model.window_u_value == pytest.approx(5.8)

    def test_apply_without_a_setter_is_advisory(self, mod):
        model = _StubWindowSpecModel(with_setter=False)
        # Must not raise even though the model exposes no window setter.
        mod.SetWindowProperties().apply(model, {"u_value_w_m2k": 1.7})
        assert model.window_writes == []
        assert model._specs[0].window_u_value == pytest.approx(1.7)

    def test_apply_rejects_a_spec_that_cannot_take_the_field(self, mod):
        class _PartialModel:
            def __init__(self):
                self._specs = [
                    _StubWindowSpec(5.8, 0.86),
                    _StubPartialWindowSpec(5.8),
                ]
                self.window_writes = []

            def window_specs(self):
                return list(self._specs)

            def set_window_specs(self, specs):
                self.window_writes.append(list(specs))

        model = _PartialModel()
        with pytest.raises(ValueError) as excinfo:
            mod.SetWindowProperties().apply(model, {"shgc": 0.35})
        assert "window_shgc" in str(excinfo.value)
        # The first spec must not have been mutated before the failure.
        assert model._specs[0].window_shgc == pytest.approx(0.86)
        assert model.window_writes == []

    def test_apply_fails_loudly_without_a_window_spec(self, mod):
        model = _StubModelWithoutWindows()
        with pytest.raises(mod.WindowNotFoundError):
            mod.SetWindowProperties().apply(model, {"u_value_w_m2k": 1.7})

    def test_apply_requires_at_least_one_property(self, mod):
        model = _StubWindowSpecModel()
        with pytest.raises(ValueError) as excinfo:
            mod.SetWindowProperties().apply(model, {})
        assert "at least one" in str(excinfo.value)
        assert model._specs[0].window_u_value == pytest.approx(5.8)
        assert model.window_writes == []

    def test_apply_rejects_an_invalid_u_value_before_mutating(self, mod):
        model = _StubWindowSpecModel([_StubWindowSpec(5.8, 0.86)])
        with pytest.raises(ValueError):
            mod.SetWindowProperties().apply(model, {"u_value_w_m2k": 0.0})
        assert model._specs[0].window_u_value == pytest.approx(5.8)
        assert model._specs[0].window_shgc == pytest.approx(0.86)
        assert model.window_writes == []

    def test_apply_rejects_an_invalid_shgc_before_mutating(self, mod):
        model = _StubWindowSpecModel([_StubWindowSpec(5.8, 0.86)])
        with pytest.raises(ValueError):
            mod.SetWindowProperties().apply(model, {"shgc": 1.5})
        assert model._specs[0].window_u_value == pytest.approx(5.8)
        assert model._specs[0].window_shgc == pytest.approx(0.86)
        assert model.window_writes == []

    def test_apply_rejects_a_non_numeric_u_value_before_mutating(self, mod):
        model = _StubWindowSpecModel([_StubWindowSpec(5.8, 0.86)])
        with pytest.raises(ValueError):
            mod.SetWindowProperties().apply(model, {"u_value_w_m2k": "single"})
        assert model.window_writes == []

    def test_arguments_are_optional_and_independent(self, mod):
        spec = mod.SetWindowProperties().arguments()
        names = [a["name"] for a in spec]
        assert names == ["u_value_w_m2k", "shgc"]
        u_value = next(a for a in spec if a["name"] == "u_value_w_m2k")
        shgc = next(a for a in spec if a["name"] == "shgc")
        assert u_value["default"] is None
        assert shgc["default"] is None
        assert u_value["max"] == mod.MAX_WINDOW_U_VALUE_W_M2K
        assert shgc["max"] == mod.MAX_WINDOW_SHGC

    def test_measure_runs_through_the_aot_runner(self, mod):
        from fluxion.measures import SOURCE_PYTHON_MEASURE, apply_measures

        model = _StubWindowSpecModel()
        chain: list[dict] = []
        applied = apply_measures(
            model,
            [mod.SetWindowProperties],
            {"SetWindowProperties": {"u_value_w_m2k": 1.7, "shgc": 0.35}},
            applied_deltas=chain,
        )
        assert applied == ["SetWindowProperties"]
        assert len(chain) == 1
        assert chain[0]["source"] == SOURCE_PYTHON_MEASURE
        assert chain[0]["name"] == "SetWindowProperties"
        assert model._specs[0].window_u_value == pytest.approx(1.7)
        assert model._specs[0].window_shgc == pytest.approx(0.35)


# =============================================================================
# SetInfiltrationRate — duck-typed stand-ins
# =============================================================================


class _StubDailySchedule:
    """Stand-in for a Rust ``DailySchedule`` (``src/sim/schedule.rs``)."""

    def __init__(self, name="Infiltration", values=None):
        self.name = name
        self.values = list(values) if values is not None else [0.5] * 24


class _StubScheduleSet:
    """Stand-in for the Rust ``ScheduleSet`` (``src/api/schema.rs``)."""

    def __init__(self, infiltration=None):
        # ``ScheduleSet.infiltration`` is an ``Option<DailySchedule>``.
        self.infiltration = infiltration


class _StubScheduleSetModel:
    """Stand-in for a model exposing a schedule set."""

    def __init__(self, schedule_set=None, with_setter=True):
        self._schedule_set = (
            schedule_set if schedule_set is not None else _StubScheduleSet()
        )
        self.schedule_set_writes = []
        if with_setter:
            self.set_schedule_set = self._set_schedule_set

    def schedule_set(self):
        return self._schedule_set

    def _set_schedule_set(self, schedule_set):
        self.schedule_set_writes.append(schedule_set)
        self._schedule_set = schedule_set


class _StubSetpoints:
    """Stand-in for the engine's ``SetpointState``."""

    def __init__(self, infiltration_rate=None):
        self.infiltration_rate = infiltration_rate


class _StubSetpointsModel:
    """Stand-in for a model exposing the per-zone infiltration rate."""

    def __init__(self, rates=None):
        self.setpoints = _StubSetpoints(list(rates) if rates is not None else [0.5, 0.5])


class _StubModelWithInfiltrationField:
    """Stand-in whose model object holds the infiltration rate itself."""

    def __init__(self, rate=0.5):
        self.infiltration_ach = rate


class _StubModelWithoutInfiltration:
    """Stand-in for a model that exposes no infiltration representation."""


# =============================================================================
# SetInfiltrationRate — pure validators (no native bindings needed)
# =============================================================================


class TestSetInfiltrationRateValidation:
    """The infiltration rate must be a valid ACH value."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module(
            "measures/SetInfiltrationRate.py", "_set_infiltration_mod"
        )

    def test_valid_rate_is_returned(self, mod):
        assert mod.validate_infiltration_rate(0.2) == pytest.approx(0.2)

    def test_zero_rate_is_accepted(self, mod):
        # A sealed-envelope study is legitimate — zero is not a deletion.
        assert mod.validate_infiltration_rate(0.0) == pytest.approx(0.0)

    def test_negative_rate_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_infiltration_rate(-0.1)
        assert ">= 0" in str(excinfo.value)

    def test_rate_above_the_cap_is_rejected(self, mod):
        with pytest.raises(ValueError) as excinfo:
            mod.validate_infiltration_rate(mod.MAX_INFILTRATION_ACH + 0.1)
        assert "<=" in str(excinfo.value)

    def test_rate_at_the_cap_is_accepted(self, mod):
        assert mod.validate_infiltration_rate(mod.MAX_INFILTRATION_ACH) == pytest.approx(
            mod.MAX_INFILTRATION_ACH
        )

    def test_rate_nan_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_infiltration_rate(float("nan"))

    def test_rate_infinity_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_infiltration_rate(float("inf"))

    def test_non_numeric_rate_is_rejected(self, mod):
        with pytest.raises(ValueError):
            mod.validate_infiltration_rate("tight")

    def test_bounds_are_the_documented_ones(self, mod):
        assert mod.MIN_INFILTRATION_ACH == pytest.approx(0.0)
        assert mod.MAX_INFILTRATION_ACH == pytest.approx(20.0)

    def test_cap_sits_above_the_extreme_in_repo_fixture(self, mod):
        # The most extreme envelope-leakage fixture in the repository is
        # 10 ACH (ventilation_denver_10ach.idf) — the cap must stay above
        # it so a real leakage study remains modellable.
        assert mod.MAX_INFILTRATION_ACH > 10.0

    def test_default_matches_the_engine_default(self, mod):
        # DEFAULT_INFILTRATION_ACH = 0.5 (src/api/server/simulate.rs).
        assert mod.DEFAULT_INFILTRATION_ACH == pytest.approx(0.5)

    def test_flat_daily_schedule_is_24_constant_hours(self, mod):
        profile = mod.flat_daily_schedule(0.2)
        assert len(profile) == 24
        assert all(value == pytest.approx(0.2) for value in profile)

    def test_field_names_match_the_schema_and_the_engine(self, mod):
        # ScheduleSet.infiltration (src/api/schema.rs) and
        # SetpointState.infiltration_rate (setpoint_state.rs).
        assert mod.INFILTRATION_FIELD == "infiltration"
        assert mod.INFILTRATION_RATE_FIELD == "infiltration_rate"


# =============================================================================
# SetInfiltrationRate — measure behaviour (duck-typed model)
# =============================================================================


class TestSetInfiltrationRateMeasure:
    """Exercise the infiltration measure against stand-in models."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_module(
            "measures/SetInfiltrationRate.py", "_set_infiltration_measure"
        )

    def test_error_is_a_value_error(self, mod):
        assert issubclass(mod.InfiltrationNotFoundError, ValueError)

    def test_resolve_finds_the_schedule_set_infiltration(self, mod):
        schedule = _StubDailySchedule()
        model = _StubScheduleSetModel(_StubScheduleSet(schedule))
        targets = mod.resolve_infiltration_targets(model)
        assert len(targets) == 1
        assert targets[0].path == "schedule_set"
        assert targets[0].kind == "schedule"

    def test_resolve_finds_the_per_zone_rate(self, mod):
        model = _StubSetpointsModel([0.5, 0.4])
        targets = mod.resolve_infiltration_targets(model)
        assert len(targets) == 1
        assert targets[0].path == "setpoints"
        assert targets[0].kind == "vector"
        assert targets[0].values == [pytest.approx(0.5), pytest.approx(0.4)]

    def test_resolve_finds_the_model_level_rate(self, mod):
        model = _StubModelWithInfiltrationField(0.5)
        targets = mod.resolve_infiltration_targets(model)
        assert len(targets) == 1
        assert targets[0].path == "model_field"
        assert targets[0].kind == "scalar"

    def test_resolve_raises_when_the_model_has_no_infiltration(self, mod):
        with pytest.raises(mod.InfiltrationNotFoundError) as excinfo:
            mod.resolve_infiltration_targets(_StubModelWithoutInfiltration())
        assert "infiltration" in str(excinfo.value).lower()

    def test_resolve_raises_when_the_schedule_set_has_no_schedule(self, mod):
        # ``ScheduleSet::default()`` carries ``infiltration: None`` — the
        # measure must not invent one.
        model = _StubScheduleSetModel(_StubScheduleSet(None))
        with pytest.raises(mod.InfiltrationNotFoundError):
            mod.resolve_infiltration_targets(model)

    def test_apply_writes_a_flat_daily_schedule(self, mod):
        schedule = _StubDailySchedule()
        model = _StubScheduleSetModel(_StubScheduleSet(schedule))
        mod.SetInfiltrationRate().apply(model, {"infiltration_ach": 0.2})
        assert schedule.values == [pytest.approx(0.2)] * 24

    def test_apply_pushes_the_schedule_set_back(self, mod):
        schedule = _StubDailySchedule()
        model = _StubScheduleSetModel(_StubScheduleSet(schedule))
        mod.SetInfiltrationRate().apply(model, {"infiltration_ach": 0.1})
        assert model.schedule_set_writes == [model._schedule_set]

    def test_apply_writes_every_zone_of_the_rate_vector(self, mod):
        model = _StubSetpointsModel([0.5, 0.5, 0.5])
        mod.SetInfiltrationRate().apply(model, {"infiltration_ach": 0.0})
        assert model.setpoints.infiltration_rate == [pytest.approx(0.0)] * 3

    def test_apply_writes_the_model_level_rate(self, mod):
        model = _StubModelWithInfiltrationField(0.5)
        mod.SetInfiltrationRate().apply(model, {"infiltration_ach": 0.25})
        assert model.infiltration_ach == pytest.approx(0.25)

    def test_apply_validates_before_touching_the_model(self, mod):
        schedule = _StubDailySchedule()
        model = _StubScheduleSetModel(_StubScheduleSet(schedule))
        with pytest.raises(ValueError):
            mod.SetInfiltrationRate().apply(model, {"infiltration_ach": 999.0})
        assert schedule.values == [pytest.approx(0.5)] * 24
        assert model.schedule_set_writes == []

    def test_apply_rejects_a_non_numeric_rate_before_touching_the_model(self, mod):
        schedule = _StubDailySchedule()
        model = _StubScheduleSetModel(_StubScheduleSet(schedule))
        with pytest.raises(ValueError):
            mod.SetInfiltrationRate().apply(model, {"infiltration_ach": "tight"})
        assert schedule.values == [pytest.approx(0.5)] * 24

    def test_apply_fails_loudly_without_an_infiltration_representation(self, mod):
        model = _StubModelWithoutInfiltration()
        with pytest.raises(mod.InfiltrationNotFoundError):
            mod.SetInfiltrationRate().apply(model, {"infiltration_ach": 0.2})

    def test_apply_updates_every_representation_the_model_exposes(self, mod):
        schedule = _StubDailySchedule()
        model = _StubScheduleSetModel(_StubScheduleSet(schedule))
        model.setpoints = _StubSetpoints([0.5, 0.5])
        mod.SetInfiltrationRate().apply(model, {"infiltration_ach": 0.2})
        assert schedule.values == [pytest.approx(0.2)] * 24
        assert model.setpoints.infiltration_rate == [pytest.approx(0.2)] * 2

    def test_arguments_declare_the_ach_argument(self, mod):
        spec = mod.SetInfiltrationRate().arguments()
        assert [a["name"] for a in spec] == ["infiltration_ach"]
        argument = spec[0]
        assert argument["type"] == "double"
        assert argument["default"] == pytest.approx(mod.DEFAULT_INFILTRATION_ACH)
        assert argument["min"] == pytest.approx(mod.MIN_INFILTRATION_ACH)
        assert argument["max"] == pytest.approx(mod.MAX_INFILTRATION_ACH)

    def test_measure_runs_through_the_aot_runner(self, mod):
        from fluxion.measures import SOURCE_PYTHON_MEASURE, apply_measures

        schedule = _StubDailySchedule()
        model = _StubScheduleSetModel(_StubScheduleSet(schedule))
        chain: list[dict] = []
        applied = apply_measures(
            model,
            [mod.SetInfiltrationRate],
            {"SetInfiltrationRate": {"infiltration_ach": 0.2}},
            applied_deltas=chain,
        )
        assert applied == ["SetInfiltrationRate"]
        assert len(chain) == 1
        assert chain[0]["source"] == SOURCE_PYTHON_MEASURE
        assert chain[0]["name"] == "SetInfiltrationRate"
        assert schedule.values == [pytest.approx(0.2)] * 24

    def test_resolve_finds_the_infiltration_of_a_dict_shaped_model(self, mod):
        # The schema model is JSON-shaped: ``schedules.infiltration``.
        model = {"schedules": {"infiltration": {"name": "Infiltration",
                                                "values": [0.5] * 24}}}
        targets = mod.resolve_infiltration_targets(model)
        assert len(targets) == 1
        assert targets[0].path == "schedules"
        assert targets[0].kind == "schedule"

    def test_apply_writes_the_infiltration_of_a_dict_shaped_model(self, mod):
        model = {"schedules": {"infiltration": {"name": "Infiltration",
                                                "values": [0.5] * 24}}}
        mod.SetInfiltrationRate().apply(model, {"infiltration_ach": 0.2})
        assert model["schedules"]["infiltration"]["values"] == [
            pytest.approx(0.2)
        ] * 24
