"""ReplaceHVACWithHeatPump — standard-library Fluxion measure (Issue #3995).

Replaces the model's HVAC system with an **air-source heat pump** — the
canonical *electrification* parametric study in building energy modeling:
"what does the bill look like if the gas boiler / chiller loop becomes an
air-source heat pump?" It is the direct analogue of the OpenStudio
``ReplaceHVACWithHeatPump`` measure and the plant-side sibling of
``measures/ReplaceHVACWithVAV.py`` (which electrifies the *air side*) and
``measures/SetHVACEfficiency.py`` (which only sweeps the plant COP).

What this measure does
----------------------
1. **Validates** the requested heating / cooling COP (and the optional
   capacity / operating-envelope overrides) **before** the model is touched,
   so an invalid request never leaves a half-mutated model behind.
2. Reads the model's HVAC snapshot (``model.hvac_system()``) and fails
   loudly with :class:`HVACSystemNotFoundError` when the model carries no
   HVAC system at all.
3. Writes the heat-pump **COPs** onto the snapshot (``cop_heating`` /
   ``cop_cooling``) and, optionally, the plant capacities and the
   outdoor-temperature operating envelope.
4. Records the **heat-pump system type** wherever the model's HVAC
   representation can carry one (see below).
5. Pushes the snapshot back through ``model.set_hvac_system(...)``.

Bounds
------
The COPs are bounded exactly like ``SetHVACEfficiency``'s — the measure
mirrors its ``MIN_COP`` / ``MAX_COP`` constants and its ``validate_cop``
helper rather than inventing new bounds:

* ``MIN_COP`` (0.1) — a COP at or below zero is physically meaningless (the
  plant would consume power without delivering any useful output).
* ``MAX_COP`` (20.0) — a generous upper bound (realistic plant equipment
  sits between 1 and 6); anything above is a modelling error.

The bounds are declared in the measure's ``arguments()`` descriptor (so the
AOT runner rejects an out-of-range value up front) *and* re-checked inside
``apply`` via :func:`validate_cop` (so a direct call cannot slip through).

The heat-pump representation in this repo
-----------------------------------------
The repo **does** know about heat pumps:

* the Rust engine has a dedicated system-type variant,
  ``HVACSystemType::HeatPump`` (``src/sim/hvac/mod.rs``, the enum that also
  carries ``Simple`` / ``VAV`` / ``CAV`` / ``Ideal``), and a full
  ``HeatPump`` struct with ``heating_cop`` / ``cooling_cop`` /
  ``design_temp_heating`` / ``design_temp_cooling`` fields
  (``src/sim/hvac/mod.rs``);
* the Python binding exposes the same enum as ``fluxion.HVACSystemType``
  with a ``HeatPump`` member (``src/python/hvac_bindings.rs``), and
  ``fluxion.hvac`` re-exports it (``src/python/hvac.py``);
* the JSON schema accepts the type through
  ``controls.hvac_system`` (``src/api/schema.rs``).

**However** — and this is the important caveat — the *measure-facing*
HVAC snapshot does not carry a system type. ``PyHVACSystem``
(``src/python/model_bindings/hvac.rs``) exposes only the plant fields
(``heating_capacity``, ``cooling_capacity``, ``cop_heating``,
``cop_cooling``, ``stages``, ``min_outdoor_temp``, ``max_outdoor_temp``,
``vav_enabled``, ``economizer_enabled``, ``supply_air_temp``) and
``apply_hvac_system_to_model`` round-trips **only** the two capacities into
the underlying ``ThermalModel``. The native ``fluxion.Model`` exposes no
``set_hvac_system_type`` either.

So this measure does the only honest thing it can:

* it sets the heat-pump **COPs** (and the optional capacities / operating
  envelope) on the HVAC snapshot — the same advisory snapshot fields
  ``SetHVACEfficiency`` writes, preserved by the Python save/load
  round-trip (``fluxion.measures.save_model`` / ``load_model``);
* it sets the **system type** to ``HVACSystemType.HeatPump`` *whenever the
  model's HVAC representation can carry one* — i.e. when the snapshot has a
  ``system_type`` attribute, or when the model exposes
  ``set_hvac_system_type`` / an ``hvac_system_type`` attribute, or when the
  model exposes a dedicated heat-pump object (``model.heat_pump()``) with
  the Rust ``HeatPump`` field names;
* otherwise the system-type part is **advisory** and the measure says so in
  its log output and in this docstring. It never claims a system-type
  change that did not happen.

A note on field propagation
----------------------------
Only ``heating_capacity`` and ``cooling_capacity`` currently round-trip into
the underlying Rust ``ThermalModel`` (see
``src/python/model_bindings/hvac.rs``). The COP fields are advisory
snapshots — they survive the Python save/load round-trip and record the
modeller's intent, but the Rust simulation does not yet consume them. This
mirrors the documented advisory limitation of ``ReplaceHVACWithVAV`` and
``SetHVACEfficiency``.

Loud failure
------------
A model that exposes no HVAC system at all cannot be retrofitted with a
heat pump; the measure raises :class:`HVACSystemNotFoundError` instead of
reporting an electrification that never happened — the same "fail loudly"
discipline as ``SetHVACEfficiency``'s ``HVACSystemNotFoundError`` and
``SwapSchedule``'s ``ScheduleNotFoundError``.

Provenance (Issue #1816)
------------------------
When run through :func:`fluxion.measures.apply_measures` with an
``applied_deltas`` accumulator, a ``python_measure`` entry is appended
automatically. This measure performs no provenance bookkeeping itself.

Run with::

    fluxion apply-measures \\
        --model base.json \\
        --measures measures/ \\
        --measure-args args.json \\
        --output model.heatpump.json

Where ``args.json`` is::

    {
        "ReplaceHVACWithHeatPump": {
            "cop_heating": 3.0,
            "cop_cooling": 3.5
        }
    }
"""

from __future__ import annotations

import logging
from typing import Any

from fluxion import FluxionMeasure

_logger = logging.getLogger(__name__)

#: Smallest coefficient of performance the measure accepts. A COP at or
#: below zero is physically meaningless. Mirrors ``SetHVACEfficiency``'s
#: ``MIN_COP`` — the same bounds, not new ones.
MIN_COP = 0.1

#: Largest coefficient of performance the measure accepts (realistic plant
#: equipment sits between 1 and 6). Mirrors ``SetHVACEfficiency``'s
#: ``MAX_COP``.
MAX_COP = 20.0

#: Default heating COP (W_th/W_e) of the replacement air-source heat pump.
#: 3.0 is the heating COP the model's HVAC snapshot carries by default
#: (``src/python/model_bindings/hvac.rs``) and a typical air-source design
#: value.
DEFAULT_HEATING_COP = 3.0

#: Default cooling COP (W_th/W_e) of the replacement air-source heat pump.
#: 3.5 is a typical air-source design value (the HVAC snapshot's default
#: cooling COP is 3.2).
DEFAULT_COOLING_COP = 3.5

#: Name of the heat-pump system type in the repo's ``HVACSystemType`` enum
#: (``src/sim/hvac/mod.rs``). Used as the fallback when the native
#: ``fluxion.HVACSystemType`` enum is not importable (e.g. a slim install
#: without the compiled extension).
HEAT_PUMP_SYSTEM_TYPE = "HeatPump"

#: The HVAC snapshot fields this measure always writes.
COP_FIELDS: tuple[str, ...] = ("cop_heating", "cop_cooling")

#: Model accessors that may expose a dedicated heat-pump object carrying the
#: Rust ``HeatPump`` field names (``src/sim/hvac/mod.rs``). The matching
#: setter is used to push the mutated object back.
HEAT_PUMP_ACCESSORS: tuple[tuple[str, str], ...] = (
    ("heat_pump", "set_heat_pump"),
    ("hvac_heat_pump", "set_hvac_heat_pump"),
)

#: The COP field names of the Rust ``HeatPump`` struct.
HEAT_PUMP_COP_FIELDS: tuple[tuple[str, str], ...] = (
    ("heating_cop", "cop_heating"),
    ("cooling_cop", "cop_cooling"),
)


class HVACSystemNotFoundError(ValueError):
    """Raised when the model exposes no HVAC system to replace.

    An electrification study applied to a model without HVAC would report a
    change that never happened: the provenance chain would carry a heat-pump
    delta while the model kept its original plant. The measure fails loudly
    instead.
    """


def validate_cop(
    value: Any,
    name: str = "cop",
    minimum: float = MIN_COP,
    maximum: float = MAX_COP,
) -> float:
    """Return ``value`` as a validated coefficient of performance.

    The COP must be a finite number strictly greater than ``minimum`` and
    no greater than ``maximum``. Anything else raises ``ValueError``.

    Mirrors ``SetHVACEfficiency.validate_cop`` (``measures/SetHVACEfficiency.py``)
    — same bounds, same error semantics — so the two measures accept and
    reject exactly the same plant efficiencies.

    Pure function — unit-testable without the native bindings.
    """
    try:
        cop = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc

    if cop != cop:  # NaN
        raise ValueError(f"{name} must not be NaN")
    if cop in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    if cop <= 0.0:
        raise ValueError(
            f"{name} must be > 0, got {cop}. A non-positive COP means the "
            "plant delivers no useful output for the power it draws."
        )
    if cop < float(minimum):
        raise ValueError(f"{name} must be >= {minimum}, got {cop}")
    if cop > float(maximum):
        raise ValueError(f"{name} must be <= {maximum}, got {cop}")
    return cop


def validate_capacity(value: Any, name: str) -> float:
    """Return ``value`` as a validated plant capacity (W).

    The capacity must be a finite, strictly positive number — a heat pump
    with no capacity is a deletion, not a retrofit. Mirrors the ``min: 1.0``
    floor declared by ``ReplaceHVACWithVAV``.

    Pure function — unit-testable without the native bindings.
    """
    try:
        capacity = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc

    if capacity != capacity:  # NaN
        raise ValueError(f"{name} must not be NaN")
    if capacity in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    if capacity <= 0.0:
        raise ValueError(
            f"{name} must be > 0 W, got {capacity}. A heat pump without "
            "capacity is a deletion, not a retrofit."
        )
    return capacity


def validate_temperature(value: Any, name: str) -> float:
    """Return ``value`` as a validated outdoor temperature (°C).

    Pure function — unit-testable without the native bindings.
    """
    try:
        temperature = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc

    if temperature != temperature:  # NaN
        raise ValueError(f"{name} must not be NaN")
    if temperature in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return temperature


def heat_pump_system_type() -> Any:
    """Return the repo's heat-pump system-type marker.

    Prefers the native ``fluxion.HVACSystemType.HeatPump`` member (the
    Python mirror of the Rust ``HVACSystemType::HeatPump`` variant,
    ``src/sim/hvac/mod.rs`` / ``src/python/hvac_bindings.rs``) and falls
    back to the plain string ``"HeatPump"`` when the native extension is
    not importable (slim install / pure-Python test runs).

    Pure function — unit-testable without the native bindings.
    """
    try:
        from fluxion import HVACSystemType  # type: ignore[attr-defined]
    except (ImportError, AttributeError):  # pragma: no cover — slim install
        return HEAT_PUMP_SYSTEM_TYPE
    return getattr(HVACSystemType, "HeatPump", HEAT_PUMP_SYSTEM_TYPE)


def resolve_hvac_snapshot(model: Any) -> Any:
    """Return the model's HVAC snapshot.

    Raises :class:`HVACSystemNotFoundError` when the model exposes no
    ``hvac_system()`` accessor, or when the snapshot carries none of the
    COP fields this measure writes.

    Mirrors ``SetHVACEfficiency.resolve_hvac_snapshot``.

    Pure function — unit-testable without the native bindings.
    """
    getter = getattr(model, "hvac_system", None)
    snapshot = None
    if callable(getter):
        try:
            snapshot = getter()
        except (AttributeError, TypeError, NotImplementedError) as exc:
            raise HVACSystemNotFoundError(
                "The model's 'hvac_system()' accessor could not be used: "
                f"{exc}"
            ) from exc

    if snapshot is None:
        raise HVACSystemNotFoundError(
            "The model exposes no HVAC system to replace — no "
            "'hvac_system()' accessor was found. Refusing to report a "
            "heat-pump electrification that never happened."
        )

    missing = [field for field in COP_FIELDS if not hasattr(snapshot, field)]
    if len(missing) == len(COP_FIELDS):
        raise HVACSystemNotFoundError(
            "The model's HVAC snapshot carries none of the COP fields "
            f"{COP_FIELDS} — refusing to report a heat-pump "
            "electrification that never happened."
        )

    return snapshot


def build_heat_pump(snapshot: Any, arguments: dict[str, Any]) -> Any:
    """Return the HVAC snapshot configured as an air-source heat pump.

    Writes the heat-pump COP onto the snapshot, plus the optional capacity
    and operating-envelope overrides. Only fields explicitly present in
    ``arguments`` override the existing value, so callers can do a partial
    retrofit (e.g. change the heating COP without touching capacities).

    This pure-ish helper is split out so the field arithmetic can be
    unit-tested without the native bindings.

    The *system type* is deliberately **not** written here: it is the one
    field the current HVAC snapshot cannot carry. It is applied by
    :func:`apply_system_type` wherever the model's representation allows
    it.
    """
    snapshot.cop_heating = float(arguments["cop_heating"])
    snapshot.cop_cooling = float(arguments["cop_cooling"])

    if arguments.get("heating_capacity") is not None:
        snapshot.heating_capacity = float(arguments["heating_capacity"])
    if arguments.get("cooling_capacity") is not None:
        snapshot.cooling_capacity = float(arguments["cooling_capacity"])
    if arguments.get("min_outdoor_temp") is not None:
        snapshot.min_outdoor_temp = float(arguments["min_outdoor_temp"])
    if arguments.get("max_outdoor_temp") is not None:
        snapshot.max_outdoor_temp = float(arguments["max_outdoor_temp"])

    return snapshot


def resolve_system_type_target(
    model: Any, snapshot: Any
) -> tuple[Any, str, str] | None:
    """Locate the place where the HVAC system type can be written.

    Returns ``(target, setter, path)`` where ``target`` is the object
    carrying the system type, ``setter`` is the attribute name to write on
    it, and ``path`` names the route that was found:

    * ``"snapshot_system_type"`` — the HVAC snapshot itself carries a
      ``system_type`` attribute (a binding that mirrors the schema's
      ``controls.hvac_system``).
    * ``"model_setter"`` — the model exposes ``set_hvac_system_type(...)``.
    * ``"model_attribute"`` — the model carries ``hvac_system_type``
      directly.

    Returns ``None`` when the model's HVAC representation cannot carry a
    system type at all — the documented limitation of the current bindings
    (the measure-facing ``PyHVACSystem`` snapshot exposes no system-type
    field, ``src/python/model_bindings/hvac.rs``).

    Pure function — unit-testable without the native bindings.
    """
    if hasattr(snapshot, "system_type"):
        return snapshot, "system_type", "snapshot_system_type"

    setter = getattr(model, "set_hvac_system_type", None)
    if callable(setter):
        return model, "set_hvac_system_type", "model_setter"

    attribute = getattr(model, "hvac_system_type", None)
    if attribute is not None and not callable(attribute):
        return model, "hvac_system_type", "model_attribute"

    return None


def apply_system_type(model: Any, snapshot: Any) -> str:
    """Record the heat-pump system type on the model.

    Returns the route that was taken:

    * ``"snapshot_system_type"`` / ``"model_setter"`` /
      ``"model_attribute"`` — the system type was written.
    * ``"advisory"`` — the model's HVAC representation cannot carry a
      system type; the retrofit is recorded through the COP / capacity
      fields only.

    Never raises: the *absence* of a system-type field is a documented
    limitation of the current bindings, not a failure of the retrofit.
    """
    system_type = heat_pump_system_type()

    resolved = resolve_system_type_target(model, snapshot)
    if resolved is None:
        return "advisory"
    target, setter, path = resolved

    if path == "model_setter":
        try:
            getattr(model, setter)(system_type)
        except (AttributeError, TypeError):  # pragma: no cover — binding mismatch
            return "advisory"
        return path

    try:
        setattr(target, setter, system_type)
    except (AttributeError, TypeError):  # pragma: no cover — read-only binding
        return "advisory"
    return path


def resolve_heat_pump_object(model: Any) -> tuple[Any, str, str] | None:
    """Locate a dedicated heat-pump object on the model, if there is one.

    Returns ``(obj, getter_name, setter_name)`` when the model exposes a
    heat-pump object carrying the Rust ``HeatPump`` COP field names
    (``heating_cop`` / ``cooling_cop``, ``src/sim/hvac/mod.rs``), and
    ``None`` when it does not.

    Pure function — unit-testable without the native bindings.
    """
    for getter_name, setter_name in HEAT_PUMP_ACCESSORS:
        accessor = getattr(model, getter_name, None)
        if not callable(accessor):
            continue
        try:
            obj = accessor()
        except (AttributeError, TypeError, NotImplementedError):
            continue
        if obj is None:
            continue
        if all(hasattr(obj, field) for field, _ in HEAT_PUMP_COP_FIELDS):
            return obj, getter_name, setter_name
    return None


def apply_heat_pump_object(
    model: Any,
    heating_cop: float,
    cooling_cop: float,
) -> str:
    """Write the heat-pump COPs onto a dedicated heat-pump object.

    Returns the route that was taken (``"heat_pump"`` /
    ``"hvac_heat_pump"``) or ``"advisory"`` when the model exposes no such
    object. The mutated object is pushed back through the matching setter
    when one exists.
    """
    resolved = resolve_heat_pump_object(model)
    if resolved is None:
        return "advisory"

    obj, getter_name, setter_name = resolved
    for field, value in (
        ("heating_cop", heating_cop),
        ("cooling_cop", cooling_cop),
    ):
        try:
            setattr(obj, field, float(value))
        except (AttributeError, TypeError):  # pragma: no cover
            return "advisory"

    setter = getattr(model, setter_name, None)
    if callable(setter):
        setter(obj)
    return getter_name


def persist_hvac(model: Any, snapshot: Any) -> str:
    """Push the mutated HVAC snapshot back into ``model``.

    Returns the persistence path that was taken:

    * ``"hvac_system"`` — the model exposes ``set_hvac_system`` and the
      mutated snapshot was written back.
    * ``"advisory"`` — no setter exists, so the new COPs are visible on the
      snapshot only.
    """
    setter = getattr(model, "set_hvac_system", None)
    if callable(setter):
        setter(snapshot)
        return "hvac_system"
    return "advisory"


class ReplaceHVACWithHeatPump(FluxionMeasure):
    """Replace the model's HVAC system with an air-source heat pump.

    Parameters
    ----------
    cop_heating : float
        Heating coefficient of performance of the heat pump (W_th/W_e).
        Must be > 0 and <= ``MAX_COP`` (20.0). Default 3.0.
    cop_cooling : float
        Cooling coefficient of performance of the heat pump (W_th/W_e).
        Must be > 0 and <= ``MAX_COP`` (20.0). Default 3.5.
    heating_capacity : float, optional
        Heating plant capacity (W). When omitted, the existing capacity is
        preserved.
    cooling_capacity : float, optional
        Cooling plant capacity (W). When omitted, the existing capacity is
        preserved.
    min_outdoor_temp : float, optional
        Minimum outdoor temperature for heat-pump operation (°C) — the
        cold-climate cut-off of the air-source unit. When omitted, the
        existing value is preserved.
    max_outdoor_temp : float, optional
        Maximum outdoor temperature for heat-pump operation (°C). When
        omitted, the existing value is preserved.
    """

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "cop_heating",
                "type": "double",
                "default": DEFAULT_HEATING_COP,
                "min": MIN_COP,
                "max": MAX_COP,
                "description": (
                    "Heating coefficient of performance of the heat pump "
                    "(W_th/W_e). Must be > 0 and "
                    f"<= {MAX_COP}. Default {DEFAULT_HEATING_COP}."
                ),
            },
            {
                "name": "cop_cooling",
                "type": "double",
                "default": DEFAULT_COOLING_COP,
                "min": MIN_COP,
                "max": MAX_COP,
                "description": (
                    "Cooling coefficient of performance of the heat pump "
                    "(W_th/W_e). Must be > 0 and "
                    f"<= {MAX_COP}. Default {DEFAULT_COOLING_COP}."
                ),
            },
            {
                "name": "heating_capacity",
                "type": "double",
                "default": None,
                "min": 1.0,
                "description": (
                    "Heating plant capacity (W). Omit to preserve existing."
                ),
            },
            {
                "name": "cooling_capacity",
                "type": "double",
                "default": None,
                "min": 1.0,
                "description": (
                    "Cooling plant capacity (W). Omit to preserve existing."
                ),
            },
            {
                "name": "min_outdoor_temp",
                "type": "double",
                "default": None,
                "description": (
                    "Minimum outdoor temperature for heat-pump operation "
                    "(°C). Omit to preserve existing."
                ),
            },
            {
                "name": "max_outdoor_temp",
                "type": "double",
                "default": None,
                "description": (
                    "Maximum outdoor temperature for heat-pump operation "
                    "(°C). Omit to preserve existing."
                ),
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        # ---- validation FIRST: never touch the model with bad input ----
        heating_cop = validate_cop(
            arguments.get("cop_heating", DEFAULT_HEATING_COP),
            name="cop_heating",
        )
        cooling_cop = validate_cop(
            arguments.get("cop_cooling", DEFAULT_COOLING_COP),
            name="cop_cooling",
        )

        heating_capacity = arguments.get("heating_capacity")
        if heating_capacity is not None:
            heating_capacity = validate_capacity(
                heating_capacity, name="heating_capacity"
            )
        cooling_capacity = arguments.get("cooling_capacity")
        if cooling_capacity is not None:
            cooling_capacity = validate_capacity(
                cooling_capacity, name="cooling_capacity"
            )

        min_outdoor_temp = arguments.get("min_outdoor_temp")
        if min_outdoor_temp is not None:
            min_outdoor_temp = validate_temperature(
                min_outdoor_temp, name="min_outdoor_temp"
            )
        max_outdoor_temp = arguments.get("max_outdoor_temp")
        if max_outdoor_temp is not None:
            max_outdoor_temp = validate_temperature(
                max_outdoor_temp, name="max_outdoor_temp"
            )

        # ---- the model must actually carry an HVAC system ----
        snapshot = resolve_hvac_snapshot(model)

        build_heat_pump(
            snapshot,
            {
                "cop_heating": heating_cop,
                "cop_cooling": cooling_cop,
                "heating_capacity": heating_capacity,
                "cooling_capacity": cooling_capacity,
                "min_outdoor_temp": min_outdoor_temp,
                "max_outdoor_temp": max_outdoor_temp,
            },
        )

        system_type_route = apply_system_type(model, snapshot)
        heat_pump_route = apply_heat_pump_object(
            model, heating_cop, cooling_cop
        )
        persisted = persist_hvac(model, snapshot)

        if system_type_route == "advisory":
            _logger.warning(
                "ReplaceHVACWithHeatPump: the model's HVAC representation "
                "carries no system-type field — the heat-pump system type "
                "is advisory (the retrofit is recorded through the COP / "
                "capacity fields only)."
            )
        if persisted == "advisory":
            _logger.warning(
                "ReplaceHVACWithHeatPump: the model exposes no "
                "'set_hvac_system' setter — the new COPs are advisory "
                "(visible on the HVAC snapshot, not persisted into the "
                "model)."
            )

        _logger.info(
            "ReplaceHVACWithHeatPump: cop_heating=%.3f, cop_cooling=%.3f — "
            "system type via %s, heat-pump object via %s, persisted via %s",
            heating_cop,
            cooling_cop,
            system_type_route,
            heat_pump_route,
            persisted,
        )
