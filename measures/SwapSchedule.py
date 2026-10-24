"""SwapSchedule — standard-library Fluxion measure (Issue #3995).

Replaces the model's thermostat schedule with an alternative profile — the
canonical *operation-schedule* parametric study in building energy modeling
and the direct analogue of the OpenStudio ``SetThermostatSetpoint`` /
schedule-swap family of measures.

What this measure does
----------------------
1. Resolves the named schedule against the schedules the model exposes — a
   name that does not exist is a **loud failure**
   (:class:`ScheduleNotFoundError`), never a silent no-op.
2. Reads the model's current schedule profile (24 hourly heating / cooling
   setpoint values).
3. Builds the replacement profile from the supplied arguments.
4. Swaps the two — the replacement profile becomes the model's schedule.
5. Pushes the result back into the model through whichever setter the
   binding exposes.

Named schedules
---------------
The native ``fluxion.Model`` binding exposes **no named-schedule
accessor** — the model carries a single, implicit thermostat schedule and
there is no schedule registry to look a name up in. The measure therefore
resolves the name as follows:

* If the model exposes a schedule registry (``model.schedule_names()`` or
  ``model.schedules()``), the requested name must be one of the registered
  names — otherwise the measure raises :class:`ScheduleNotFoundError`.
* Otherwise the only schedule that can be swapped is the implicit
  thermostat schedule, whose canonical name is
  :data:`IMPLICIT_SCHEDULE_NAME` (``"Thermostat"``). Any other name raises
  :class:`ScheduleNotFoundError` with an explicit explanation, because the
  binding cannot verify it.

This is what keeps the measure honest: a parametric run that asks to swap a
schedule the model does not have fails loudly instead of reporting a swap
that never happened.

Profiles
--------
Three profile kinds are supported, mirroring the Rust ``HVACSchedule``
constructors in ``src/sim/schedule.rs``:

``constant``
    Heating and cooling setpoints are held for all 24 hours.

``setback``
    The heating setpoint is held at the day value except over the setback
    window (``[night_start_hour, night_end_hour)``, wrapping midnight) where
    the setback value applies. The cooling setpoint is held constant.

``operating_hours``
    The space is only conditioned over the occupied window
    (``[start_hour, end_hour)``, wrapping midnight). Outside that window the
    heating is switched off (``-100 °C``) and the cooling is switched off
    (``+100 °C``) — the same sentinel values the Rust schedule engine uses.

Provenance (Issue #1816)
------------------------
When run through :func:`fluxion.measures.apply_measures` with an
``applied_deltas`` accumulator, a ``python_measure`` entry is appended
automatically. This measure performs no provenance bookkeeping itself.

Persistence
-----------
The schedule is pushed back through the first setter the binding exposes:

* ``model.set_hvac_schedule(schedule)`` — the model exposes a schedule
  setter; the replacement profile is materialised as a native
  ``fluxion.HVACSchedule``.
* ``model.set_zone_setpoints(zone_index, heating, cooling)`` — the model has
  no schedule setter but does have a zone-setpoint setter; the profile then
  collapses to its occupied setpoints (the values used while the space is
  conditioned).
* neither — the swap is **advisory**: the replacement profile is computed and
  logged but the current ``fluxion.Model`` binding exposes neither setter, so
  it does not round-trip into the serialized model. This mirrors the
  documented advisory fields of ``ReplaceHVACWithVAV``.

Note that the binding exposes a *single* schedule slot per model, so even
when the model carries a schedule registry the replacement profile is
written to that one slot — the named lookup validates the request, it does
not select among several writable schedules.

Run with::

    fluxion apply-measures \\
        --model base.json \\
        --measures measures/ \\
        --measure-args args.json \\
        --output model.rescheduled.json

Where ``args.json`` is::

    {
        "SwapSchedule": {
            "schedule_name": "Thermostat",
            "profile": "setback",
            "heating_setpoint": 20.0,
            "cooling_setpoint": 24.0,
            "setback_heating_setpoint": 15.0,
            "night_start_hour": 22,
            "night_end_hour": 6
        }
    }
"""

from __future__ import annotations

import logging
from typing import Any

from fluxion import FluxionMeasure

_logger = logging.getLogger(__name__)

#: Profile kinds understood by :func:`build_hourly_profile`.
PROFILE_CONSTANT = "constant"
PROFILE_SETBACK = "setback"
PROFILE_OPERATING_HOURS = "operating_hours"
PROFILES: tuple[str, ...] = (PROFILE_CONSTANT, PROFILE_SETBACK, PROFILE_OPERATING_HOURS)

#: Sentinel values the Rust schedule engine uses for "heating off" and
#: "cooling off" (``src/sim/schedule.rs``).
HEATING_OFF = -100.0
COOLING_OFF = 100.0

#: Name of the *implicit* thermostat schedule carried by a model that
#: exposes no schedule registry. The current ``fluxion.Model`` binding has
#: no named-schedule accessor at all — the model carries exactly one
#: thermostat schedule, and this is its canonical name.
IMPLICIT_SCHEDULE_NAME = "Thermostat"

#: Number of hours in a daily schedule profile.
HOURS_PER_DAY = 24


class ScheduleNotFoundError(ValueError):
    """Raised when the schedule to be swapped does not exist in the model.

    A schedule swap that silently does nothing is worse than a loud
    failure: a parametric run would report the swap as applied while the
    model kept its original schedule, and the resulting energy numbers
    would silently belong to the wrong configuration.
    """


def fill_range(
    values: list[float],
    start_hour: int,
    end_hour: int,
    value: float,
) -> list[float]:
    """Return a copy of ``values`` with ``[start_hour, end_hour)`` set to ``value``.

    The range is half-open and wraps around midnight when ``start_hour`` is
    greater than ``end_hour``. When the two bounds are equal nothing is
    filled. This mirrors ``DailySchedule::fill_range`` in
    ``src/sim/schedule.rs``.
    """
    out = list(values)
    start = int(start_hour) % HOURS_PER_DAY
    end = int(end_hour) % HOURS_PER_DAY
    if start == end:
        return out
    if start < end:
        for hour in range(start, end):
            out[hour] = value
    else:
        for hour in range(start, HOURS_PER_DAY):
            out[hour] = value
        for hour in range(0, end):
            out[hour] = value
    return out


def build_hourly_profile(
    profile: str,
    heating_setpoint: float,
    cooling_setpoint: float,
    setback_heating_setpoint: float | None = None,
    night_start_hour: int = 22,
    night_end_hour: int = 6,
    start_hour: int = 8,
    end_hour: int = 18,
) -> tuple[list[float], list[float]]:
    """Build the ``(heating_values, cooling_values)`` of a daily profile.

    Both lists have :data:`HOURS_PER_DAY` entries. See the module docstring
    for the semantics of the three profile kinds.

    Pure function — unit-testable without the native bindings.
    """
    kind = str(profile).strip().lower()
    if kind not in PROFILES:
        raise ValueError(
            f"Unknown schedule profile {profile!r}; expected one of {PROFILES}"
        )

    if kind == PROFILE_CONSTANT:
        heating = [float(heating_setpoint)] * HOURS_PER_DAY
        cooling = [float(cooling_setpoint)] * HOURS_PER_DAY
    elif kind == PROFILE_SETBACK:
        setback = (
            float(heating_setpoint)
            if setback_heating_setpoint is None
            else float(setback_heating_setpoint)
        )
        heating = fill_range(
            [float(heating_setpoint)] * HOURS_PER_DAY,
            night_start_hour,
            night_end_hour,
            setback,
        )
        cooling = [float(cooling_setpoint)] * HOURS_PER_DAY
    else:  # PROFILE_OPERATING_HOURS
        heating = fill_range(
            [HEATING_OFF] * HOURS_PER_DAY,
            start_hour,
            end_hour,
            float(heating_setpoint),
        )
        cooling = fill_range(
            [COOLING_OFF] * HOURS_PER_DAY,
            start_hour,
            end_hour,
            float(cooling_setpoint),
        )

    return heating, cooling


def occupied_setpoints(
    heating_values: list[float],
    cooling_values: list[float],
) -> tuple[float, float]:
    """Return the ``(heating, cooling)`` pair used while the space is occupied.

    The occupied values are the ones of the busiest hour of the profile —
    i.e. the hour with the highest heating setpoint. For a constant profile
    that is the constant value, for a setback profile the day value, and for
    an operating-hours profile the occupied value (the off-hours carry the
    ``-100 °C`` heating sentinel).

    Pure function — unit-testable without the native bindings.
    """
    if not heating_values or not cooling_values:
        raise ValueError("Cannot derive occupied setpoints from an empty profile")
    count = min(len(heating_values), len(cooling_values))
    peak = max(range(count), key=lambda hour: heating_values[hour])
    return heating_values[peak], cooling_values[peak]


def swap_schedule_values(
    current: list[float],
    replacement: list[float],
) -> list[float]:
    """Return ``replacement`` as the new schedule values.

    Both profiles must hold :data:`HOURS_PER_DAY` values — a malformed
    profile is a programming error and raises ``ValueError`` rather than
    silently producing a broken schedule.
    """
    if len(current) != HOURS_PER_DAY:
        raise ValueError(
            f"Current schedule must hold {HOURS_PER_DAY} values, got {len(current)}"
        )
    if len(replacement) != HOURS_PER_DAY:
        raise ValueError(
            f"Replacement schedule must hold {HOURS_PER_DAY} values, "
            f"got {len(replacement)}"
        )
    return list(replacement)


def current_profile(model: Any) -> tuple[list[float], list[float]]:
    """Return the model's current ``(heating_values, cooling_values)`` profile.

    When the model exposes an HVAC schedule getter, the schedule's hourly
    setpoints are read from it. Otherwise the profile is synthesized from the
    zone setpoints (a constant profile), which is what the model uses when it
    carries no explicit schedule.
    """
    getter = getattr(model, "get_hvac_schedule", None)
    if callable(getter):
        schedule = getter()
        heating = [float(schedule.heating_setpoint(hour)) for hour in range(HOURS_PER_DAY)]
        cooling = [float(schedule.cooling_setpoint(hour)) for hour in range(HOURS_PER_DAY)]
        return heating, cooling

    zones = model.zones()
    if zones:
        zone = zones[0]
        return (
            [float(zone.heating_setpoint)] * HOURS_PER_DAY,
            [float(zone.cooling_setpoint)] * HOURS_PER_DAY,
        )
    return [0.0] * HOURS_PER_DAY, [0.0] * HOURS_PER_DAY


def schedule_names(model: Any) -> list[str]:
    """Return the names of the schedules the model exposes.

    Two sources are probed, in order:

    1. ``model.schedule_names()`` — a callable returning an iterable of
       schedule names.
    2. ``model.schedules()`` — a callable returning schedule objects that
       carry a ``name`` attribute.

    An **empty** list means the model exposes no schedule registry: it
    carries a single implicit thermostat schedule (the current
    ``fluxion.Model`` binding has no named-schedule accessor at all).

    Pure function — unit-testable without the native bindings.
    """
    lister = getattr(model, "schedule_names", None)
    if callable(lister):
        try:
            return [str(name) for name in lister()]
        except (AttributeError, TypeError, NotImplementedError):
            # The accessor exists but is not usable on this model — fall
            # through to the next source (and, failing that, to the
            # implicit thermostat schedule).
            pass

    schedules = getattr(model, "schedules", None)
    if callable(schedules):
        try:
            found = list(schedules())
        except (AttributeError, TypeError, NotImplementedError):
            found = []
        names: list[str] = []
        for schedule in found:
            name = getattr(schedule, "name", None)
            if name is not None:
                names.append(str(name))
        return names

    return []


def resolve_schedule_name(model: Any, schedule_name: str) -> str:
    """Resolve ``schedule_name`` against the schedules the model exposes.

    Returns the resolved schedule name. Raises
    :class:`ScheduleNotFoundError` when the named schedule does not
    exist — the measure must never turn a missing schedule into a silent
    no-op.

    Two cases:

    * The model exposes a schedule registry: the requested name must be one
      of the registered names.
    * The model exposes no registry (today's ``fluxion.Model`` binding):
      the only schedule that can be swapped is the implicit thermostat
      schedule :data:`IMPLICIT_SCHEDULE_NAME`.

    Pure function — unit-testable without the native bindings.
    """
    requested = str(schedule_name)
    names = schedule_names(model)

    if names:
        if requested in names:
            return requested
        raise ScheduleNotFoundError(
            f"Schedule {requested!r} not found in the model; "
            f"available schedules: {sorted(names)}"
        )

    if requested == IMPLICIT_SCHEDULE_NAME:
        return requested

    raise ScheduleNotFoundError(
        f"Schedule {requested!r} not found: the model exposes no schedule "
        f"registry (the native binding has no named-schedule accessor), so "
        f"the only schedule that can be swapped is the implicit thermostat "
        f"schedule {IMPLICIT_SCHEDULE_NAME!r}."
    )


def build_native_schedule(
    profile: str,
    heating_setpoint: float,
    cooling_setpoint: float,
    setback_heating_setpoint: float | None = None,
    night_start_hour: int = 22,
    night_end_hour: int = 6,
    start_hour: int = 8,
    end_hour: int = 18,
    fluxion_module: Any = None,
) -> Any:
    """Materialize a profile as a native ``fluxion.HVACSchedule``.

    The native schedule type only exposes the three constructors that mirror
    the Rust engine (``constant_schedule``, ``setback_schedule``,
    ``with_operating_hours``), so the profile kind selects the constructor.

    ``fluxion_module`` may be supplied to inject a stand-in module in tests;
    when omitted the real ``fluxion`` module is imported.
    """
    if fluxion_module is None:
        import fluxion as fluxion_module  # type: ignore[no-redef]

    kind = str(profile).strip().lower()
    if kind == PROFILE_CONSTANT:
        return fluxion_module.HVACSchedule.constant_schedule(
            float(heating_setpoint), float(cooling_setpoint)
        )
    if kind == PROFILE_SETBACK:
        setback = (
            float(heating_setpoint)
            if setback_heating_setpoint is None
            else float(setback_heating_setpoint)
        )
        return fluxion_module.HVACSchedule.setback_schedule(
            float(heating_setpoint),
            setback,
            float(cooling_setpoint),
            int(night_start_hour),
            int(night_end_hour),
        )
    if kind == PROFILE_OPERATING_HOURS:
        return fluxion_module.HVACSchedule.with_operating_hours(
            float(heating_setpoint), float(cooling_setpoint), int(start_hour), int(end_hour)
        )
    raise ValueError(
        f"Unknown schedule profile {profile!r}; expected one of {PROFILES}"
    )


def persist_schedule(
    model: Any,
    profile: str,
    heating_setpoint: float,
    cooling_setpoint: float,
    setback_heating_setpoint: float | None = None,
    night_start_hour: int = 22,
    night_end_hour: int = 6,
    start_hour: int = 8,
    end_hour: int = 18,
    zone_index: int | None = None,
) -> str:
    """Push the swapped schedule back into ``model``.

    Returns the persistence path that was taken:

    * ``"hvac_schedule"`` — the model exposes ``set_hvac_schedule``.
    * ``"zone_setpoints"`` — the model exposes ``set_zone_setpoints``; the
      profile then collapses to its occupied setpoints.
    * ``"advisory"`` — neither setter exists (the current ``fluxion.Model``
      binding is in this situation).
    """
    schedule_setter = getattr(model, "set_hvac_schedule", None)
    if callable(schedule_setter):
        schedule_setter(
            build_native_schedule(
                profile,
                heating_setpoint,
                cooling_setpoint,
                setback_heating_setpoint,
                night_start_hour,
                night_end_hour,
                start_hour,
                end_hour,
            )
        )
        return "hvac_schedule"

    zone_setter = getattr(model, "set_zone_setpoints", None)
    if callable(zone_setter):
        heating_values, cooling_values = build_hourly_profile(
            profile,
            heating_setpoint,
            cooling_setpoint,
            setback_heating_setpoint,
            night_start_hour,
            night_end_hour,
            start_hour,
            end_hour,
        )
        occupied_heating, occupied_cooling = occupied_setpoints(
            heating_values, cooling_values
        )
        for position, zone in enumerate(model.zones()):
            index = int(getattr(zone, "index", position))
            if zone_index is not None and index != int(zone_index):
                continue
            zone_setter(index, occupied_heating, occupied_cooling)
        return "zone_setpoints"

    return "advisory"


class SwapSchedule(FluxionMeasure):
    """Replace the model's thermostat schedule with an alternative profile.

    Parameters
    ----------
    schedule_name : str
        Name of the schedule being replaced (e.g. ``"Thermostat"``). The
        name is resolved against the schedules the model exposes; a name
        that does not exist raises ``ScheduleNotFoundError`` instead of
        silently doing nothing. Default ``"Thermostat"``.
    profile : str
        Profile kind: ``"constant"``, ``"setback"`` or ``"operating_hours"``.
        Default ``"constant"``.
    heating_setpoint : float
        Occupied / day heating setpoint (°C). Default 20.0.
    cooling_setpoint : float
        Occupied / day cooling setpoint (°C). Default 24.0.
    setback_heating_setpoint : float or None
        Heating setpoint during the setback window (°C), used by the
        ``"setback"`` profile. ``None`` (default) reuses
        ``heating_setpoint``.
    night_start_hour : int
        First hour of the setback window (0–23). Default 22.
    night_end_hour : int
        First hour after the setback window (0–23). Default 6.
    start_hour : int
        First occupied hour of the ``"operating_hours"`` profile (0–23).
        Default 8.
    end_hour : int
        First hour after the occupied window (0–23). Default 18.
    zone_index : int or None
        If set, restrict the collapsed setpoint update to that zone
        (0-based). If ``None`` (default), every zone is updated.
    """

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "schedule_name",
                "type": "string",
                "default": "Thermostat",
                "description": "Name of the schedule being replaced.",
            },
            {
                "name": "profile",
                "type": "choice",
                "default": PROFILE_CONSTANT,
                "choices": list(PROFILES),
                "description": (
                    "Replacement profile: constant, setback or operating_hours."
                ),
            },
            {
                "name": "heating_setpoint",
                "type": "double",
                "default": 20.0,
                "min": -100.0,
                "max": 100.0,
                "description": "Occupied / day heating setpoint (°C).",
            },
            {
                "name": "cooling_setpoint",
                "type": "double",
                "default": 24.0,
                "min": -100.0,
                "max": 100.0,
                "description": "Occupied / day cooling setpoint (°C).",
            },
            {
                "name": "setback_heating_setpoint",
                "type": "double",
                "default": None,
                "min": -100.0,
                "max": 100.0,
                "description": (
                    "Heating setpoint during the setback window (°C). "
                    "null reuses heating_setpoint."
                ),
            },
            {
                "name": "night_start_hour",
                "type": "integer",
                "default": 22,
                "min": 0,
                "max": 23,
                "description": "First hour of the setback window (0-23).",
            },
            {
                "name": "night_end_hour",
                "type": "integer",
                "default": 6,
                "min": 0,
                "max": 23,
                "description": "First hour after the setback window (0-23).",
            },
            {
                "name": "start_hour",
                "type": "integer",
                "default": 8,
                "min": 0,
                "max": 23,
                "description": "First occupied hour of the operating_hours profile (0-23).",
            },
            {
                "name": "end_hour",
                "type": "integer",
                "default": 18,
                "min": 0,
                "max": 23,
                "description": "First hour after the occupied window (0-23).",
            },
            {
                "name": "zone_index",
                "type": "integer",
                "default": None,
                "description": (
                    "Zone index to limit the setpoint update to (0-based). "
                    "null applies to every zone."
                ),
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        schedule_name = str(arguments.get("schedule_name", IMPLICIT_SCHEDULE_NAME))
        # Loud failure: refuse to run when the named schedule is not in the
        # model — a silent no-op would poison the parametric sweep.
        resolve_schedule_name(model, schedule_name)
        profile = str(arguments.get("profile", PROFILE_CONSTANT))
        heating_setpoint = float(arguments.get("heating_setpoint", 20.0))
        cooling_setpoint = float(arguments.get("cooling_setpoint", 24.0))
        setback_heating_setpoint = arguments.get("setback_heating_setpoint", None)
        night_start_hour = int(arguments.get("night_start_hour", 22))
        night_end_hour = int(arguments.get("night_end_hour", 6))
        start_hour = int(arguments.get("start_hour", 8))
        end_hour = int(arguments.get("end_hour", 18))
        zone_index = arguments.get("zone_index", None)

        # Read the current profile, then build + swap in the replacement.
        current_heating, current_cooling = current_profile(model)
        replacement_heating, replacement_cooling = build_hourly_profile(
            profile,
            heating_setpoint,
            cooling_setpoint,
            None if setback_heating_setpoint is None else float(setback_heating_setpoint),
            night_start_hour,
            night_end_hour,
            start_hour,
            end_hour,
        )
        new_heating = swap_schedule_values(current_heating, replacement_heating)
        new_cooling = swap_schedule_values(current_cooling, replacement_cooling)
        occupied_heating, occupied_cooling = occupied_setpoints(new_heating, new_cooling)

        path = persist_schedule(
            model,
            profile,
            heating_setpoint,
            cooling_setpoint,
            None if setback_heating_setpoint is None else float(setback_heating_setpoint),
            night_start_hour,
            night_end_hour,
            start_hour,
            end_hour,
            zone_index,
        )

        if path == "advisory":
            _logger.warning(
                "SwapSchedule: the model exposes no schedule setter — the "
                "replacement profile is advisory (computed and logged, not "
                "persisted into the model)."
            )

        _logger.info(
            "SwapSchedule: replaced schedule '%s' with a '%s' profile "
            "(occupied setpoints %.1f/%.1f °C) — persisted via %s",
            schedule_name,
            profile,
            occupied_heating,
            occupied_cooling,
            path,
        )
