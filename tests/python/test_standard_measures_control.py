"""Pytest suite for the control-side standard-library measures (Issue #3995).

Companion to ``tests/python/test_standard_measures.py``. It covers the parts
of the control-side measures that are not exercised there:

* the **named-schedule resolution** of ``SwapSchedule`` — a schedule that
  does not exist in the model must fail loudly (:class:`ScheduleNotFoundError`),
  never silently do nothing;
* a few extra pure-Python edge cases of the ``SetThermostatSetpoints`` and
  ``SwapSchedule`` helpers.

The suite is designed to run in two modes:

1. **With native bindings** (``maturin develop``): the integration test at
   the bottom exercises the measures against a real ``fluxion.Model``.
2. **Without native bindings** (CI without a Rust toolchain): every
   pure-Python test still runs — the measures are driven through duck-typed
   stand-ins that implement the documented snapshot contract
   (``model.zones()`` -> mutate -> push back) without the native extension.

The native-binding detection follows ``tests/conftest.py`` (Issue #2852): the
``fluxion`` wrapper imports cleanly even when the compiled extension is
missing, so the sentinel ``fluxion._NATIVE_IMPORT_ERROR`` — not the mere
presence of the wrapper — decides whether a binding-dependent test runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MEASURES_DIR = REPO_ROOT / "measures"

# Ensure ``measures.*`` is importable.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _native_extension_available() -> bool:
    """True when the compiled ``fluxion`` extension is importable."""
    if importlib.util.find_spec("fluxion") is None:
        return False
    try:
        import fluxion
    except ImportError:
        return False
    return getattr(fluxion, "_NATIVE_IMPORT_ERROR", None) is None


requires_native = pytest.mark.skipif(
    not _native_extension_available(),
    reason="fluxion native extension not available (run `maturin develop`)",
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


@pytest.fixture(scope="module")
def swap_module():
    return _load_module("measures/SwapSchedule.py", "_swap_control")


@pytest.fixture(scope="module")
def setpoint_module():
    return _load_module("measures/SetThermostatSetpoints.py", "_setpoints_control")


# ---------------------------------------------------------------------------
# Duck-typed stand-ins
# ---------------------------------------------------------------------------


class _StubZone:
    """Stand-in for the ``fluxion.Zone`` snapshot."""

    def __init__(self, index, heating_setpoint=20.0, cooling_setpoint=24.0):
        self.index = index
        self.heating_setpoint = heating_setpoint
        self.cooling_setpoint = cooling_setpoint


class _StubSchedule:
    """Stand-in for a named schedule object (carries a ``name``)."""

    def __init__(self, name):
        self.name = name


class _StubModel:
    """Stand-in for ``fluxion.Model`` that exposes no schedule registry."""

    def __init__(self, zones=None):
        self._zones = list(zones or [])
        self.schedule_writes = []
        self.setpoint_calls = []

    # -- snapshot contract -------------------------------------------------
    def zones(self):
        return list(self._zones)

    def set_zone_setpoints(self, index, heating, cooling):
        self.setpoint_calls.append((index, heating, cooling))


class _StubModelWithRegistry(_StubModel):
    """Stand-in for a model that exposes a schedule registry."""

    def __init__(self, registry, zones=None):
        super().__init__(zones=zones)
        self._registry = list(registry)

    def schedules(self):
        return [_StubSchedule(name) for name in self._registry]


# ---------------------------------------------------------------------------
# Named-schedule resolution — pure Python, no native bindings needed
# ---------------------------------------------------------------------------


class TestScheduleNames:
    """``schedule_names`` probes the model for a schedule registry."""

    def test_no_registry_yields_no_names(self, swap_module):
        model = _StubModel()
        assert swap_module.schedule_names(model) == []

    def test_registry_via_schedules_accessor(self, swap_module):
        model = _StubModelWithRegistry(["Thermostat", "Occupied Heating"])
        assert swap_module.schedule_names(model) == [
            "Thermostat",
            "Occupied Heating",
        ]

    def test_registry_via_schedule_names_accessor(self, swap_module):
        class _NamedModel:
            def schedule_names(self):
                return ["Thermostat"]

        assert swap_module.schedule_names(_NamedModel()) == ["Thermostat"]


class TestResolveScheduleName:
    """``resolve_schedule_name`` fails loudly for an unknown schedule."""

    def test_known_name_in_registry_resolves(self, swap_module):
        model = _StubModelWithRegistry(["Thermostat", "Occupied Heating"])
        assert (
            swap_module.resolve_schedule_name(model, "Occupied Heating")
            == "Occupied Heating"
        )

    def test_unknown_name_in_registry_raises(self, swap_module):
        model = _StubModelWithRegistry(["Thermostat"])
        with pytest.raises(swap_module.ScheduleNotFoundError) as excinfo:
            swap_module.resolve_schedule_name(model, "Missing Schedule")
        message = str(excinfo.value)
        assert "Missing Schedule" in message
        assert "Thermostat" in message

    def test_implicit_schedule_resolves_without_registry(self, swap_module):
        model = _StubModel()
        assert (
            swap_module.resolve_schedule_name(model, swap_module.IMPLICIT_SCHEDULE_NAME)
            == "Thermostat"
        )

    def test_other_name_without_registry_raises(self, swap_module):
        model = _StubModel()
        with pytest.raises(swap_module.ScheduleNotFoundError) as excinfo:
            swap_module.resolve_schedule_name(model, "Occupied Heating")
        assert "Occupied Heating" in str(excinfo.value)
        assert "no schedule registry" in str(excinfo.value)

    def test_error_is_a_value_error(self, swap_module):
        # Callers that only catch ValueError still see the loud failure.
        assert issubclass(swap_module.ScheduleNotFoundError, ValueError)


class TestSwapScheduleLoudFailure:
    """The measure itself refuses to run against a missing schedule."""

    def test_apply_raises_for_unknown_name(self, swap_module):
        model = _StubModelWithRegistry(["Thermostat"])
        measure = swap_module.SwapSchedule()
        with pytest.raises(swap_module.ScheduleNotFoundError):
            measure.apply(model, {"schedule_name": "Missing Schedule"})

    def test_apply_raises_for_unknown_name_without_registry(self, swap_module):
        model = _StubModel()
        measure = swap_module.SwapSchedule()
        with pytest.raises(swap_module.ScheduleNotFoundError):
            measure.apply(model, {"schedule_name": "Occupied Heating"})

    def test_failed_swap_does_not_touch_the_model(self, swap_module):
        model = _StubModelWithRegistry(["Thermostat"])
        measure = swap_module.SwapSchedule()
        with pytest.raises(swap_module.ScheduleNotFoundError):
            measure.apply(model, {"schedule_name": "Missing Schedule"})
        assert model.setpoint_calls == []
        assert model.schedule_writes == []

    def test_known_name_applies(self, swap_module):
        model = _StubModelWithRegistry(["Thermostat"], zones=[_StubZone(0)])
        measure = swap_module.SwapSchedule()
        measure.apply(
            model,
            {
                "schedule_name": "Thermostat",
                "profile": swap_module.PROFILE_CONSTANT,
                "heating_setpoint": 21.0,
                "cooling_setpoint": 25.0,
            },
        )
        assert model.setpoint_calls == [(0, 21.0, 25.0)]

    def test_default_schedule_name_is_the_implicit_one(self, swap_module):
        model = _StubModel(zones=[_StubZone(0)])
        measure = swap_module.SwapSchedule()
        args = measure.parse_arguments({})
        assert args["schedule_name"] == swap_module.IMPLICIT_SCHEDULE_NAME
        measure.apply(model, args)
        assert len(model.setpoint_calls) == 1


# ---------------------------------------------------------------------------
# Extra pure-Python helper coverage
# ---------------------------------------------------------------------------


class TestSetpointHelperEdges:
    """Edge cases of the setpoint helpers (no native bindings needed)."""

    def test_clamp_is_identity_inside_the_range(self, setpoint_module):
        assert setpoint_module.clamp(20.0, 15.0, 25.0) == 20.0

    def test_shift_without_bounds_returns_the_raw_sum(self, setpoint_module):
        assert setpoint_module.shift_setpoint(20.0, 5.0) == 25.0

    def test_absolute_none_keeps_the_delta_behaviour(self, setpoint_module):
        assert setpoint_module.shift_setpoint(20.0, 2.0, None) == 22.0

    def test_sweep_reports_the_swept_values(self, setpoint_module):
        zones = [_StubZone(0, 20.0, 24.0), _StubZone(1, 21.0, 25.0)]
        records = setpoint_module.apply_setpoint_sweep(
            zones, heating_delta=1.0, cooling_delta=-1.0
        )
        assert [r["index"] for r in records] == [0, 1]
        assert records[0]["heating_setpoint"] == 21.0
        assert records[0]["cooling_setpoint"] == 23.0
        # The zone snapshots are mutated in place.
        assert zones[0].heating_setpoint == 21.0

    def test_sweep_without_clamping_can_leave_the_engine_window(
        self, setpoint_module
    ):
        zones = [_StubZone(0, 20.0, 24.0)]
        records = setpoint_module.apply_setpoint_sweep(
            zones,
            heating_delta=10.0,
            clamp_to_bounds=False,
        )
        assert records[0]["heating_setpoint"] == 30.0


class TestScheduleHelperEdges:
    """Edge cases of the schedule helpers (no native bindings needed)."""

    def test_swap_returns_a_copy(self, swap_module):
        current = [20.0] * 24
        replacement = [21.0] * 24
        result = swap_module.swap_schedule_values(current, replacement)
        result[0] = 99.0
        assert replacement[0] == 21.0
        assert current[0] == 20.0

    def test_operating_hours_profile_wrapping_midnight(self, swap_module):
        heating, cooling = swap_module.build_hourly_profile(
            swap_module.PROFILE_OPERATING_HOURS,
            20.0,
            24.0,
            start_hour=22,
            end_hour=6,
        )
        assert heating[22] == 20.0
        assert heating[5] == 20.0
        assert heating[6] == swap_module.HEATING_OFF
        assert cooling[6] == swap_module.COOLING_OFF
        assert cooling[0] == 24.0

    def test_setback_profile_keeps_the_day_value_outside_the_window(
        self, swap_module
    ):
        heating, _ = swap_module.build_hourly_profile(
            swap_module.PROFILE_SETBACK,
            20.0,
            24.0,
            setback_heating_setpoint=15.0,
            night_start_hour=22,
            night_end_hour=6,
        )
        assert heating[23] == 15.0
        assert heating[12] == 20.0

    def test_occupied_setpoints_reject_an_empty_profile(self, swap_module):
        with pytest.raises(ValueError):
            swap_module.occupied_setpoints([], [])


# ---------------------------------------------------------------------------
# Integration — requires the native extension
# ---------------------------------------------------------------------------


@requires_native
class TestSetZoneSetpointsNativeIntegration:
    """Issue #4633 — ``fluxion.Model`` exposes a real zone-setpoint setter."""

    def test_set_and_get_zone_setpoints(self):
        import fluxion

        model = fluxion.Model(num_zones=1)
        model.set_zone_setpoints(0, 18.0, 26.0)
        assert model.get_zone_setpoints() == [(18.0, 26.0)]
        zone = model.zones()[0]
        assert zone.heating_setpoint == 18.0
        assert zone.cooling_setpoint == 26.0

    def test_set_zone_setpoints_validation(self):
        import fluxion

        model = fluxion.Model(num_zones=1)
        with pytest.raises(ValueError):
            model.set_zone_setpoints(1, 18.0, 26.0)
        with pytest.raises(ValueError):
            model.set_zone_setpoints(0, 26.0, 18.0)
        with pytest.raises(ValueError):
            model.set_zone_setpoints(0, 22.0, 22.0)

    def test_setpoints_change_simulated_energy(self):
        import fluxion

        energies = []
        for heating, cooling in [(18.0, 26.0), (23.0, 27.0)]:
            model = fluxion.Model(num_zones=1)
            model.set_zone_setpoints(0, heating, cooling)
            energies.append(model.simulate(1, False))
        assert energies[0] != energies[1]

    def test_measure_persists_through_save_load_round_trip(self, tmp_path):
        import fluxion
        from fluxion.measures import apply_measures, load_model, save_model

        module = _load_module(
            "measures/SetThermostatSetpoints.py", "_setpoints_native_4633"
        )
        model = fluxion.Model(num_zones=1)
        before = model.get_zone_setpoints()
        applied = apply_measures(
            model,
            [module.SetThermostatSetpoints],
            {"SetThermostatSetpoints": {"heating_delta": 1.0, "cooling_delta": 1.0}},
        )
        assert applied == ["SetThermostatSetpoints"]
        after = model.get_zone_setpoints()
        assert after != before
        path = tmp_path / "model.json"
        save_model(model, path)
        loaded = load_model(path)
        assert loaded.get_zone_setpoints() == after


@requires_native
class TestSwapScheduleNativeIntegration:
    """The measure runs against a real ``fluxion.Model``."""

    def test_default_schedule_name_is_accepted(self):
        import fluxion
        from fluxion.measures import apply_measures

        module = _load_module("measures/SwapSchedule.py", "_swap_native")
        model = fluxion.Model(num_zones=1)
        applied = apply_measures(
            model,
            [module.SwapSchedule],
            {"SwapSchedule": {"schedule_name": "Thermostat"}},
        )
        assert applied == ["SwapSchedule"]

    def test_unknown_schedule_name_fails_loudly(self):
        import fluxion
        from fluxion.measures import apply_measures

        module = _load_module("measures/SwapSchedule.py", "_swap_native_missing")
        model = fluxion.Model(num_zones=1)
        with pytest.raises(module.ScheduleNotFoundError):
            apply_measures(
                model,
                [module.SwapSchedule],
                {"SwapSchedule": {"schedule_name": "No Such Schedule"}},
            )
