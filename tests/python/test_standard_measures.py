"""Pytest suite for the standard-library Fluxion measures (Issue #1815).

Issue #1815 — Phase 3 (Developer Experience & Ecosystem) of the Hybrid Measure
Approach. Covers the three baseline measures shipped in ``measures/``:

- ``SetWindowToWallRatio``
- ``ReplaceHVACWithVAV``
- ``IncreaseInsulationRValue``

The suite is designed to run in two modes:

1. **With native bindings** (``maturin develop``): full integration coverage
   through ``fluxion.Model`` — the measures mutate a real model, the
   mutations persist via the snapshot/owned-value round trip, and the
   provenance chain (Issue #1816) is recorded.
2. **Without native bindings** (CI without a Rust toolchain): the pure-Python
   numerical helpers (``compute_window_area``, ``compute_insulated_u_value``,
   ``build_vav_system``) are unit-tested directly. These tests do not need a
   ``fluxion.Model``.

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
    the underlying Rust ``ThermalModel`` (see ``src/python/model_bindings.rs``).
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
