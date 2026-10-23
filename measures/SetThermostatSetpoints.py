"""SetThermostatSetpoints — standard-library Fluxion measure (Issue #3995).

Sweeps the heating / cooling thermostat setpoints of every zone in the model
by a configurable delta, or pins them to absolute values. This is the
canonical *control-side* parametric study in building energy modeling and the
direct analogue of the OpenStudio ``SetThermostatSetpoint`` measure.

Definition
----------
For each zone the new setpoints are::

    heating' = heating_setpoint   (absolute)   or   heating + heating_delta
    cooling' = cooling_setpoint   (absolute)   or   cooling + cooling_delta

An absolute value always wins over the corresponding delta, so one measure
invocation can mix "shift the heating setpoint" with "pin the cooling
setpoint". A delta of ``0`` (the default) leaves a setpoint untouched, so the
measure is a no-op unless at least one of the four arguments is set.

Bounds and deadband
-------------------
The Rust parameter validator (``src/api/parameters.rs``) only accepts heating
setpoints in ``[15, 25] °C`` and cooling setpoints in ``[22, 32] °C``, and it
requires the heating setpoint to be strictly below the cooling setpoint. The
helpers below therefore clamp to those bounds by default and re-open the
deadband (``min_deadband``) when a shift would collapse it. Pass
``clamp_to_bounds=False`` to bypass the clamping (e.g. when driving an engine
build with wider bounds).

Provenance (Issue #1816)
------------------------
When run through :func:`fluxion.measures.apply_measures` with an
``applied_deltas`` accumulator, a ``python_measure`` entry is appended
automatically. This measure performs no provenance bookkeeping itself.

Persistence
-----------
The measure follows the snapshot / owned-value contract from
``docs/bindings.md``: the zone list is read **once** via ``model.zones()``,
mutated in place, and pushed back through whichever setter the binding
exposes:

* ``model.set_zone_setpoints(zone_index, heating, cooling)`` — used when the
  model exposes it (the ``fluxion.MultiZoneThermalModel`` binding does).
* ``model.set_zones(zones)`` — used when the model exposes a zone setter.
* neither — the sweep is **advisory**: the new values are visible on the
  mutated zone snapshots but the current ``fluxion.Model`` binding exposes no
  zone-setpoint setter, so they do not round-trip into the serialized model.
  This mirrors the documented advisory fields of ``ReplaceHVACWithVAV``.

Run with::

    fluxion apply-measures \\
        --model base.json \\
        --measures measures/ \\
        --measure-args args.json \\
        --output model.setpoints.json

Where ``args.json`` is::

    {
        "SetThermostatSetpoints": {
            "heating_delta": 1.0,
            "cooling_delta": -1.0
        }
    }

Or from Python::

    from fluxion import Model
    from fluxion.measures import apply_measures
    from measures.SetThermostatSetpoints import SetThermostatSetpoints

    model = Model(num_zones=2)
    apply_measures(model, [SetThermostatSetpoints],
                   {"SetThermostatSetpoints": {"heating_delta": 1.0}})
"""

from __future__ import annotations

import logging
from typing import Any

from fluxion import FluxionMeasure

_logger = logging.getLogger(__name__)

# Engine-side validity bounds for the thermostat setpoints. These mirror the
# constants in ``src/api/parameters.rs`` (``MIN/MAX_HEATING_SETPOINT`` /
# ``MIN/MAX_COOLING_SETPOINT``) — the values the Rust parameter validator
# accepts. They are *not* a physics tuning knob: they are the engine's own
# admissibility window, reproduced here so a measure can never build a model
# the engine rejects.
MIN_HEATING_SETPOINT = 15.0
MAX_HEATING_SETPOINT = 25.0
MIN_COOLING_SETPOINT = 22.0
MAX_COOLING_SETPOINT = 32.0

#: Minimum deadband (K) kept between the heating and the cooling setpoint.
MIN_DEADBAND_K = 1.0


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp ``value`` into ``[minimum, maximum]``.

    The bounds are order-insensitive (a reversed pair is normalized). Split
    out as a pure function so it can be unit-tested without the native
    bindings (RULES.md: numerical reasoning via code).
    """
    low, high = (minimum, maximum) if minimum <= maximum else (maximum, minimum)
    if value < low:
        return low
    if value > high:
        return high
    return value


def shift_setpoint(
    current: float,
    delta: float = 0.0,
    absolute: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return the new setpoint value.

    ``absolute`` wins over ``delta`` when it is not ``None``; otherwise the
    current value is shifted by ``delta``. The optional bounds are applied
    last (only the ones that are given).
    """
    if absolute is not None:
        new_value = float(absolute)
    else:
        new_value = float(current) + float(delta)
    if minimum is not None and maximum is not None:
        return clamp(new_value, float(minimum), float(maximum))
    if minimum is not None:
        return max(new_value, float(minimum))
    if maximum is not None:
        return min(new_value, float(maximum))
    return new_value


def resolve_setpoint_pair(
    heating: float,
    cooling: float,
    heating_delta: float = 0.0,
    cooling_delta: float = 0.0,
    heating_setpoint: float | None = None,
    cooling_setpoint: float | None = None,
    min_heating: float = MIN_HEATING_SETPOINT,
    max_heating: float = MAX_HEATING_SETPOINT,
    min_cooling: float = MIN_COOLING_SETPOINT,
    max_cooling: float = MAX_COOLING_SETPOINT,
    min_deadband: float = MIN_DEADBAND_K,
    clamp_to_bounds: bool = True,
) -> tuple[float, float]:
    """Return the ``(heating, cooling)`` pair after the sweep.

    The resolution order is:

    1. Apply the absolute value (if given) or the delta to the current
       setpoint.
    2. Clamp to the engine's admissible window (unless
       ``clamp_to_bounds`` is False).
    3. Re-open the deadband: if ``heating + min_deadband > cooling``, the
       cooling setpoint is raised to ``heating + min_deadband`` when that
       stays inside the admissible window; otherwise the heating setpoint is
       lowered to ``cooling - min_deadband``. When the bounds make a valid
       deadband impossible the clamped pair is returned unchanged.

    Pure function — unit-testable without the native bindings.
    """
    new_heating = shift_setpoint(heating, heating_delta, heating_setpoint)
    new_cooling = shift_setpoint(cooling, cooling_delta, cooling_setpoint)

    if clamp_to_bounds:
        new_heating = clamp(new_heating, float(min_heating), float(max_heating))
        new_cooling = clamp(new_cooling, float(min_cooling), float(max_cooling))

    if min_deadband > 0.0 and new_heating + float(min_deadband) > new_cooling:
        raised_cooling = new_heating + float(min_deadband)
        if (not clamp_to_bounds) or raised_cooling <= float(max_cooling):
            new_cooling = raised_cooling
        else:
            lowered_heating = new_cooling - float(min_deadband)
            if (not clamp_to_bounds) or lowered_heating >= float(min_heating):
                new_heating = lowered_heating
            # Otherwise the admissible window cannot host a valid deadband;
            # the clamped pair is kept as-is.

    return new_heating, new_cooling


def apply_setpoint_sweep(
    zones: Any,
    zone_index: int | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Sweep the setpoints of ``zones`` in place and return the applied values.

    ``zones`` is any iterable of zone-like objects exposing ``index``,
    ``heating_setpoint`` and ``cooling_setpoint`` (the ``fluxion.Zone``
    snapshot satisfies this). The remaining keyword arguments are forwarded
    to :func:`resolve_setpoint_pair`.

    When ``zone_index`` is given, only the zone with that index is swept.

    Returns one record per swept zone::

        {"index": 0, "heating_setpoint": 21.0, "cooling_setpoint": 24.0}
    """
    records: list[dict[str, Any]] = []
    for position, zone in enumerate(zones):
        index = int(getattr(zone, "index", position))
        if zone_index is not None and index != int(zone_index):
            continue
        heating, cooling = resolve_setpoint_pair(
            float(zone.heating_setpoint),
            float(zone.cooling_setpoint),
            **kwargs,
        )
        zone.heating_setpoint = heating
        zone.cooling_setpoint = cooling
        records.append(
            {
                "index": index,
                "heating_setpoint": heating,
                "cooling_setpoint": cooling,
            }
        )
    return records


def persist_setpoints(model: Any, zones: Any, records: list[dict[str, Any]]) -> str:
    """Push the swept setpoints back into ``model``.

    Returns the name of the persistence path that was taken:

    * ``"zone_setpoints"`` — the model exposes
      ``set_zone_setpoints(zone_index, heating, cooling)`` (the
      ``fluxion.MultiZoneThermalModel`` binding does).
    * ``"zones"`` — the model exposes ``set_zones(zones)``.
    * ``"advisory"`` — neither setter exists, so the change stays on the
      mutated zone snapshots. The current ``fluxion.Model`` binding is in
      this situation: it exposes no zone-setpoint setter, so the sweep is
      advisory until the binding grows one.
    """
    zone_setter = getattr(model, "set_zone_setpoints", None)
    if callable(zone_setter):
        for record in records:
            zone_setter(
                record["index"],
                record["heating_setpoint"],
                record["cooling_setpoint"],
            )
        return "zone_setpoints"

    zones_setter = getattr(model, "set_zones", None)
    if callable(zones_setter):
        zones_setter(zones)
        return "zones"

    return "advisory"


class SetThermostatSetpoints(FluxionMeasure):
    """Shift or pin the heating / cooling thermostat setpoints of every zone.

    Parameters
    ----------
    heating_delta : float
        Shift applied to the heating setpoint (K). Positive warms the
        building. Default 0.0 (no change).
    cooling_delta : float
        Shift applied to the cooling setpoint (K). Positive cools the
        building. Default 0.0 (no change).
    heating_setpoint : float or None
        Absolute heating setpoint (°C). When given it wins over
        ``heating_delta``. ``None`` (default) keeps the delta behaviour.
    cooling_setpoint : float or None
        Absolute cooling setpoint (°C). When given it wins over
        ``cooling_delta``. ``None`` (default) keeps the delta behaviour.
    zone_index : int or None
        If set, sweep only that zone (0-based). If ``None`` (default), sweep
        every zone in the model.
    clamp_to_bounds : bool
        Clamp the swept setpoints to the engine's admissible window
        (heating 15–25 °C, cooling 22–32 °C). Default True.
    min_deadband : float
        Minimum deadband (K) kept between the heating and cooling setpoint.
        Default 1.0.
    """

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "heating_delta",
                "type": "double",
                "default": 0.0,
                "min": -10.0,
                "max": 10.0,
                "description": "Shift applied to the heating setpoint (K).",
            },
            {
                "name": "cooling_delta",
                "type": "double",
                "default": 0.0,
                "min": -10.0,
                "max": 10.0,
                "description": "Shift applied to the cooling setpoint (K).",
            },
            {
                "name": "heating_setpoint",
                "type": "double",
                "default": None,
                "min": MIN_HEATING_SETPOINT,
                "max": MAX_HEATING_SETPOINT,
                "description": (
                    "Absolute heating setpoint (°C). Overrides heating_delta. "
                    "null keeps the delta behaviour."
                ),
            },
            {
                "name": "cooling_setpoint",
                "type": "double",
                "default": None,
                "min": MIN_COOLING_SETPOINT,
                "max": MAX_COOLING_SETPOINT,
                "description": (
                    "Absolute cooling setpoint (°C). Overrides cooling_delta. "
                    "null keeps the delta behaviour."
                ),
            },
            {
                "name": "zone_index",
                "type": "integer",
                "default": None,
                "description": (
                    "Zone index to limit the sweep to (0-based). "
                    "null applies to every zone."
                ),
            },
            {
                "name": "clamp_to_bounds",
                "type": "bool",
                "default": True,
                "description": (
                    "Clamp the swept setpoints to the engine's admissible "
                    "window. Default true."
                ),
            },
            {
                "name": "min_deadband",
                "type": "double",
                "default": MIN_DEADBAND_K,
                "min": 0.0,
                "max": 10.0,
                "description": "Minimum deadband (K) between heating and cooling setpoint.",
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        zone_index = arguments.get("zone_index", None)
        heating_setpoint = arguments.get("heating_setpoint", None)
        cooling_setpoint = arguments.get("cooling_setpoint", None)

        sweep_kwargs: dict[str, Any] = {
            "heating_delta": float(arguments.get("heating_delta", 0.0) or 0.0),
            "cooling_delta": float(arguments.get("cooling_delta", 0.0) or 0.0),
            "heating_setpoint": (
                None if heating_setpoint is None else float(heating_setpoint)
            ),
            "cooling_setpoint": (
                None if cooling_setpoint is None else float(cooling_setpoint)
            ),
            "min_deadband": float(arguments.get("min_deadband", MIN_DEADBAND_K)),
            "clamp_to_bounds": bool(arguments.get("clamp_to_bounds", True)),
        }

        # Snapshot the zones ONCE (PyO3 ``#[pyo3(get)]`` on a Vec returns a
        # fresh list on every access — see docs/bindings.md) and mutate that
        # list in place.
        zones = model.zones()
        records = apply_setpoint_sweep(zones, zone_index=zone_index, **sweep_kwargs)
        path = persist_setpoints(model, zones, records)

        if path == "advisory":
            _logger.warning(
                "SetThermostatSetpoints: the model exposes no zone-setpoint "
                "setter — the swept setpoints are advisory (visible on the "
                "zone snapshots, not persisted into the model)."
            )

        _logger.info(
            "SetThermostatSetpoints: swept %d zone(s) (Δheat=%.2f K, "
            "Δcool=%.2f K) — persisted via %s",
            len(records),
            sweep_kwargs["heating_delta"],
            sweep_kwargs["cooling_delta"],
            path,
        )
