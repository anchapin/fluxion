"""SetInfiltrationRate — standard-library Fluxion measure (Issue #3995).

Sets the **infiltration rate** (air leakage) of the model to a target value
in air changes per hour (ACH). This is the canonical *envelope-tightening*
parametric study in building energy modeling — "how much does sealing the
envelope actually save?" — and the direct analogue of the OpenStudio
``Set infiltration rate`` / air-leakage family of measures.

Nothing else in the standard library can move the air-leakage lever: the
load-side measures (``SetLightingPowerDensity``,
``SetEquipmentPowerDensity``) change internal gains, the plant-side measure
(``SetHVACEfficiency``) changes the HVAC plant, and the envelope measures
(``IncreaseInsulationRValue``, ``SetWindowProperties``,
``SetWindowToWallRatio``) change the *fabric*. Infiltration is the leakage
through the fabric itself and had no standard-library member.

Definition
----------
The infiltration rate is the hourly air-change rate of the zone volume::

    Q_infiltration [W] = ACH [1/h] × V [m³] × ρ_air [kg/m³] × c_p [J/(kg·K)]

The engine applies exactly this relation: the ventilation conductance is
computed as ``h_ve = (infiltration_rate · volume · ρ_air · c_p) / 3600`` with
the comment *"infiltration_rate is in ACH (1/hr)"*
(``src/sim/thermal_model_physics/dispatch_state.rs``), and the per-zone
infiltration rate is handed to the gauge solver under the name
``infiltration_ach`` (``src/sim/thermal_model_physics/step_dispatcher.rs``;
``src/sim/diagnostics_ext.rs`` reads it back as ``infiltration_ach``).

The unit is therefore **ACH (air changes per hour, 1/h)** — proved from the
code, not assumed. The same unit is what the importers produce:
``extract_infiltration_ach`` in ``src/io/idf/convert.rs`` converts an
EnergyPlus ``ZoneInfiltration:DesignFlowRate`` object into ACH
(``AirChanges/Hour`` is already ACH; the flow-per-area methods are
converted with ``· 3600``), and the ASHRAE-140 case spec carries the same
quantity as ``infiltration_ach`` (``fluxion-core/src/ashrae_case_builder.rs``).

The measure writes a **constant rate** — a flat daily infiltration profile.
That is the shape of the standard leakage studies: the envelope leaks at
roughly the same rate in every hour of the day, and the parametric question
is "how much does the *level* of leakage cost?". The engine's own default
is a constant rate too (``DEFAULT_INFILTRATION_ACH = 0.5`` in
``src/api/server/simulate.rs``; ``VectorField::from_scalar(spec.infiltration_ach, …)``
in ``src/sim/thermal_model_core/mod.rs``).

Bounds
------
The rate is validated **before the model is touched**, so an invalid request
never leaves a half-mutated model behind:

* ``MIN_INFILTRATION_ACH`` (0.0 ACH) — a zero rate is *legitimate*: a
  sealed-envelope study ("what if the building had no infiltration at all?")
  is a normal parametric question, and the engine itself models it (the
  ASHRAE-140 case 195 sets the infiltration rate to zero,
  ``src/sim/thermal_model_core/mod.rs``). Unlike the equipment-density
  measure — where zero means "delete the load", not a density study — zero
  infiltration still *is* an infiltration study, so it is accepted.
* ``MAX_INFILTRATION_ACH`` (20.0 ACH) — a generous physical ceiling. The
  most extreme *envelope*-leakage fixture in this repository is 10 ACH
  (``tests/reference_data/energyplus_models/ventilation_denver_10ach.idf``
  and ``tests/reference_data/ventilation/infiltration_denver_10ach.csv``);
  20 ACH is twice that extreme case, so every in-repo leakage study stays
  modellable while a value that can only be a unit error (e.g. 0.5 written
  as 500, or an air-flow rate in m³/h) is still rejected. The 40 ACH figures
  in ``data/ashrae140_holdout.json`` are **night-ventilation** rates (a
  mechanical ventilation system, cases 650 / 950), not envelope
  infiltration, and are deliberately not the yardstick here.

Both bounds live in this measure — the Rust engine has no infiltration-rate
constant to borrow.

Finding the infiltration representation
---------------------------------------
Infiltration reaches the model through **two** documented representations,
and the measure probes for them in this order:

1. **The schema's infiltration schedule** — ``ScheduleSet.infiltration``,
   an ``Option<DailySchedule>`` (``src/api/schema.rs``: ``pub infiltration:
   Option<DailySchedule>``). The measure looks for a schedule set on the
   model (``model.schedule_set()`` / ``model.schedules()``, or the
   ``"schedule_set"`` / ``"schedules"`` key of a dict-shaped model) and
   writes a **flat 24-hour daily schedule** of the target rate into it.
2. **The engine's per-zone infiltration rate** — the
   ``setpoints.infiltration_rate`` field of the thermal model
   (``src/sim/thermal_model_data/setpoint_state.rs``), a per-zone vector in
   ACH. The measure writes the target rate into every zone. This is the
   representation the physics reads (``dispatch_state.rs``).

A third, last-resort probe accepts the rate held **directly on the model**
(``model.infiltration_rate`` / ``model.infiltration_ach``), for a model that
carries the field itself.

Every representation the model exposes is written with the *same* target
rate, so the model stays internally consistent no matter which of them the
engine happens to read.

When the model has **no** infiltration representation at all, the measure
raises :class:`InfiltrationNotFoundError` — a loud failure naming the
missing objects. Applying a leakage study to a model without infiltration
and reporting success would put an infiltration delta on the provenance
chain that belongs to a configuration the model never had. This mirrors the
``LightingNotFoundError`` / ``EquipmentNotFoundError`` /
``HVACSystemNotFoundError`` / ``WindowNotFoundError`` pattern of the other
#3995 measures.

The measure does **not** materialise an infiltration schedule that the model
does not have: ``ScheduleSet::default()`` sets ``infiltration: None``
(``src/api/schema.rs``) and the engine defines no default *schedule* for it
(the engine default of 0.5 ACH lives in the per-zone
``setpoints.infiltration_rate``, not in the schedule set). Inventing a
schedule the model never had would create semantics the engine does not
define, so the measure reports the missing object instead.

Persistence
-----------
The measure follows the snapshot / owned-value contract from
``docs/bindings.md``: every representation is read **once**, mutated in
place, and pushed back through the matching setter when the binding exposes
one:

* ``model.set_schedule_set(...)`` / ``model.set_schedules(...)`` — the
  mutated schedule set is written back.
* neither — the mutation is **advisory**: the new rate is visible on the
  snapshot the binding handed out, but the current ``fluxion.Model`` binding
  exposes no schedule-set accessor, so it does not round-trip into the
  serialized model. This mirrors the documented advisory fields of
  ``ReplaceHVACWithVAV`` and of ``SetWindowProperties``.

On today's native ``fluxion.Model`` binding the measure therefore **fails
loudly**: the binding exposes neither a schedule-set accessor nor a
setpoints accessor (``fluxion.pyi`` — the ``Model`` class exposes only
zones, surfaces, the HVAC system and the temperatures), so a native model
exposes no infiltration representation at all and the measure raises
:class:`InfiltrationNotFoundError`. The measure is written against the
documented schema shape (``ScheduleSet``) and the engine's documented
setpoint shape, and round-trips as soon as the binding grows the accessor;
where a binding exposes the representation but no setter, the change is
**advisory**.

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
        --output model.sealed.json

Where ``args.json`` is::

    {
        "SetInfiltrationRate": {
            "infiltration_ach": 0.2
        }
    }

Worked effect
--------------
Tightening the envelope from 0.5 ACH to 0.2 ACH on the model's schedule set
writes the flat daily profile::

    [0.2, 0.2, ... , 0.2]   (24 values)

and, when the model exposes the per-zone rate, sets every zone of
``setpoints.infiltration_rate`` to ``0.2``. With the engine's relation
``h_ve = ACH · V · ρ_air · c_p / 3600`` the ventilation conductance of a
129.6 m³ zone drops from 21.7 W/K to 8.7 W/K — the ventilation load term
of the zone energy balance shrinks by 60 %.

Or from Python::

    from fluxion import Model
    from fluxion.measures import apply_measures
    from measures.SetInfiltrationRate import SetInfiltrationRate

    model = Model(num_zones=2)
    apply_measures(model, [SetInfiltrationRate],
                   {"SetInfiltrationRate": {"infiltration_ach": 0.2}})
"""

from __future__ import annotations

import logging
from typing import Any

from fluxion import FluxionMeasure

_logger = logging.getLogger(__name__)

#: Smallest infiltration rate the measure accepts (ACH). Zero is a
#: legitimate sealed-envelope study (the engine models it too — ASHRAE-140
#: case 195 sets the infiltration rate to zero).
MIN_INFILTRATION_ACH = 0.0

#: Largest infiltration rate the measure accepts (ACH). The most extreme
#: envelope-leakage fixture in this repository is 10 ACH
#: (``tests/reference_data/energyplus_models/ventilation_denver_10ach.idf``);
#: 20 ACH is twice that extreme case — generous, but still rejecting values
#: that can only be a unit error.
MAX_INFILTRATION_ACH = 20.0

#: Default infiltration rate (ACH) — the engine's own default
#: (``DEFAULT_INFILTRATION_ACH = 0.5`` in ``src/api/server/simulate.rs``).
DEFAULT_INFILTRATION_ACH = 0.5

#: The Rust field name of the infiltration schedule (``ScheduleSet`` in
#: ``src/api/schema.rs``).
INFILTRATION_FIELD = "infiltration"

#: The Rust field name of the per-zone infiltration rate
#: (``SetpointState`` in ``src/sim/thermal_model_data/setpoint_state.rs``).
INFILTRATION_RATE_FIELD = "infiltration_rate"

#: Alias accepted for the per-zone rate / the model-level field — the name
#: the Monte-Carlo parameter sweep and the ASHRAE-140 case spec use for the
#: same physical quantity (``infiltration_ach``).
INFILTRATION_ACH_FIELD = "infiltration_ach"

#: Number of hours in a daily schedule profile.
HOURS_PER_DAY = 24

#: ``(getter, setter)`` pairs probed on the model for the *schedule set*,
#: in order. ``schedule_set`` is the canonical spelling (the Rust
#: ``ScheduleSet``); ``schedules`` is the shorter alias.
SCHEDULE_SET_ACCESSORS: tuple[tuple[str, str], ...] = (
    ("schedule_set", "set_schedule_set"),
    ("schedules", "set_schedules"),
)

#: Field names accepted for the per-zone / model-level infiltration rate.
INFILTRATION_RATE_FIELDS: tuple[str, ...] = (
    INFILTRATION_RATE_FIELD,
    INFILTRATION_ACH_FIELD,
)

#: Sentinel for "field absent" — ``None`` is a legal value (the schema's
#: ``Option<DailySchedule>`` is ``None`` when the model has no schedule).
_MISSING = object()


class InfiltrationNotFoundError(ValueError):
    """Raised when the model exposes no infiltration representation.

    An infiltration study applied to a model without infiltration would
    report a change that never happened: the provenance chain would carry an
    infiltration delta while the model's air leakage stayed untouched. The
    measure fails loudly instead.
    """


def validate_infiltration_rate(
    value: Any,
    minimum: float = MIN_INFILTRATION_ACH,
    maximum: float = MAX_INFILTRATION_ACH,
) -> float:
    """Return ``value`` as a validated infiltration rate (ACH).

    The rate must be a finite number in the closed interval
    ``[minimum, maximum]`` — ``[0, 20]`` ACH by default. Anything else
    raises ``ValueError``.

    Pure function — unit-testable without the native bindings.
    """
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Infiltration rate must be a number, got {value!r}"
        ) from exc

    if rate != rate:  # NaN
        raise ValueError("Infiltration rate must not be NaN")
    if rate in (float("inf"), float("-inf")):
        raise ValueError("Infiltration rate must be finite")
    if rate < float(minimum):
        raise ValueError(
            f"Infiltration rate must be >= {minimum} ACH, got {rate}. A "
            "negative air-change rate is not physical."
        )
    if rate > float(maximum):
        raise ValueError(
            f"Infiltration rate must be <= {maximum} ACH, got {rate}. The "
            "most extreme envelope-leakage fixture in this repository is "
            "10 ACH (ventilation_denver_10ach.idf) — a larger value can "
            "only be a unit error."
        )
    return rate


def flat_daily_schedule(rate: float) -> list[float]:
    """Return the flat daily infiltration profile of ``rate`` (ACH).

    A constant rate is written as a 24-value daily profile — the shape of
    the ``ScheduleSet.infiltration`` ``DailySchedule`` and the shape of the
    engine's own constant-rate default.

    Pure function — unit-testable without the native bindings.
    """
    return [float(rate)] * HOURS_PER_DAY


def _get_field(container: Any, name: str) -> Any:
    """Return the field ``name`` of ``container``, or the ``_MISSING`` sentinel.

    Supports plain attributes (the ``fluxion`` snapshot style) and mapping
    keys (a ``dict``-shaped model).
    """
    if container is None:
        return _MISSING
    if isinstance(container, dict):
        return container.get(name, _MISSING)
    return getattr(container, name, _MISSING)


def _set_field(container: Any, name: str, value: Any) -> None:
    """Write ``value`` into the field ``name`` of ``container``.

    Handles attribute-style and mapping-style containers.
    """
    if isinstance(container, dict) and not hasattr(container, name):
        container[name] = value
        return
    setattr(container, name, value)


def _probe(model: Any, name: str) -> Any:
    """Return the attribute / key ``name`` of ``model``, calling it when callable.

    Returns ``None`` when the attribute is missing or when calling it fails
    — a binding may expose an accessor it cannot serve on this model, in
    which case the probe moves on to the next candidate.
    """
    value = _get_field(model, name)
    if value is _MISSING or value is None:
        return None
    if callable(value):
        try:
            return value()
        except (AttributeError, TypeError, NotImplementedError):
            return None
    return value


def _schedule_hourly_values(schedule: Any) -> list[float] | None:
    """Return the hourly values of a ``DailySchedule``-like object.

    The Rust ``DailySchedule`` carries its profile in ``values``
    (``ScheduleValues::Daily([f64; 24])`` or
    ``ScheduleValues::Weekly([[f64; 24]; 7])``). A dict-shaped schedule
    carries the same list under its ``"values"`` key. ``None`` is returned
    when the object exposes no readable hourly profile.
    """
    values = _get_field(schedule, "values")
    if values is _MISSING or values is None:
        return None
    if isinstance(values, (list, tuple)):
        return list(values)
    # A wrapper around the Rust ``ScheduleValues`` enum.
    daily = _get_field(values, "daily")
    if isinstance(daily, (list, tuple)):
        return list(daily)
    weekly = _get_field(values, "weekly")
    if isinstance(weekly, (list, tuple)):
        flat: list[float] = []
        for day in weekly:
            if not isinstance(day, (list, tuple)):
                return None
            flat.extend(float(v) for v in day)
        return flat
    return None


def _write_schedule_rate(schedule: Any, rate: float) -> None:
    """Write the constant ``rate`` (ACH) into the daily profile of ``schedule``.

    The hourly list is mutated **in place** when possible so that the
    snapshot the binding handed out stays consistent with the model.
    """
    values = _get_field(schedule, "values")
    if isinstance(values, list):
        values[:] = [float(rate)] * len(values)
        return
    daily = _get_field(values, "daily")
    if isinstance(daily, list):
        daily[:] = [float(rate)] * len(daily)
        return
    weekly = _get_field(values, "weekly")
    if isinstance(weekly, list):
        for day in weekly:
            if isinstance(day, list):
                day[:] = [float(rate)] * len(day)
        return
    # Last resort: replace the whole profile.
    _set_field(schedule, "values", flat_daily_schedule(rate))


class InfiltrationTarget:
    """One infiltration representation found on the model.

    Attributes
    ----------
    path : str
        The accessor that produced the representation (``"schedule_set"``,
        ``"schedules"``, ``"setpoints"`` or ``"model_field"``).
    kind : str
        ``"schedule"`` (a flat daily profile), ``"vector"`` (a per-zone
        rate) or ``"scalar"`` (a single rate).
    container : Any
        The object that holds the values (the schedule set, the setpoints
        object or the model itself).
    field : str
        Name of the field / key that holds the values.
    setter : str or None
        Name of the matching setter on the model, or ``None`` when the
        binding exposes none (the change is then advisory).
    owner : Any
        The object pushed back through the setter — the *schedule set* for a
        schedule target (the setter takes the schedule set, not the
        schedule), the container itself otherwise.
    values : list
        The hourly / per-zone values read from the container.
    """

    def __init__(
        self,
        path: str,
        kind: str,
        container: Any,
        field: str,
        setter: str | None,
        values: list[float],
        owner: Any = None,
    ) -> None:
        self.path = path
        self.kind = kind
        self.container = container
        self.field = field
        self.setter = setter
        self.values = list(values)
        self.owner = container if owner is None else owner

    def __repr__(self) -> str:
        return (
            f"InfiltrationTarget(path={self.path!r}, kind={self.kind!r}, "
            f"field={self.field!r}, values={len(self.values)})"
        )

    def write(self, rate: float) -> None:
        """Write the constant ``rate`` (ACH) into this representation."""
        if self.kind == "schedule":
            _write_schedule_rate(self.container, rate)
            return
        if self.kind == "vector":
            current = _get_field(self.container, self.field)
            if isinstance(current, list):
                current[:] = [float(rate)] * len(current)
                return
            _set_field(self.container, self.field, float(rate))
            return
        _set_field(self.container, self.field, float(rate))

    def persist(self, model: Any) -> str:
        """Push the mutated representation back into ``model``.

        Returns the persistence path that was taken:

        * the accessor's own path (``"schedule_set"`` / ``"schedules"``) —
          the matching setter exists and the mutated object was written back.
        * ``"in_place"`` — the values live inside an object of the model
          (``setpoints``) or on the model itself; the in-place mutation
          already reached the model.
        * ``"advisory"`` — no setter exists, so the new rate is visible on
          the snapshot only.
        """
        if self.setter is None:
            # A dict-shaped model holds the very object that was mutated, so
            # the change is already in the model. An attribute-shaped
            # schedule that is only reachable through a snapshot is
            # advisory — the native binding exposes no schedule-set setter.
            if isinstance(self.owner, dict):
                return "in_place"
            return "advisory" if self.kind == "schedule" else "in_place"
        setter = getattr(model, self.setter, None)
        if callable(setter):
            setter(self.owner)
            return self.path
        return "advisory"


def _schedule_set_of(model: Any) -> tuple[Any, str | None] | None:
    """Return the ``(schedule_set, setter)`` of the model, or ``None``."""
    for getter_name, setter_name in SCHEDULE_SET_ACCESSORS:
        schedule_set = _probe(model, getter_name)
        if schedule_set is None:
            continue
        return schedule_set, setter_name
    return None


def resolve_infiltration_targets(model: Any) -> list[InfiltrationTarget]:
    """Locate every infiltration representation the model exposes.

    The probes run in this documented order:

    1. the schema's infiltration schedule — the ``infiltration`` field of
       the model's schedule set (``ScheduleSet.infiltration``, an
       ``Option<DailySchedule>`` in ``src/api/schema.rs``).
    2. the engine's per-zone infiltration rate — the ``infiltration_rate``
       field of the model's setpoints (``SetpointState`` in
       ``src/sim/thermal_model_data/setpoint_state.rs``), in ACH.
    3. the rate held directly on the model (``infiltration_rate`` /
       ``infiltration_ach``).

    Every representation the model exposes is returned — an infiltration
    study applies to the model's whole air-leakage model, not just to the
    first representation that happens to be found.

    Raises :class:`InfiltrationNotFoundError` when the model exposes no
    infiltration representation at all.

    Pure function — unit-testable without the native bindings (it only
    reads the model).
    """
    targets: list[InfiltrationTarget] = []

    # 1 — the schema's infiltration schedule.
    found = _schedule_set_of(model)
    if found is not None:
        schedule_set, setter_name = found
        schedule = _get_field(schedule_set, INFILTRATION_FIELD)
        if schedule is not _MISSING and schedule is not None:
            values = _schedule_hourly_values(schedule)
            if values is None:
                raise ValueError(
                    "The model's infiltration schedule does not expose a "
                    "readable hourly profile ('values') — the infiltration "
                    "rate of this model cannot be written. Refusing to "
                    "report an infiltration change that never happened."
                )
            targets.append(
                InfiltrationTarget(
                    path=_schedule_set_path(model, schedule_set),
                    kind="schedule",
                    container=schedule,
                    field=INFILTRATION_FIELD,
                    setter=setter_name,
                    values=values,
                    owner=schedule_set,
                )
            )

    # 2 — the engine's per-zone infiltration rate.
    setpoints = _probe(model, "setpoints")
    if setpoints is not None:
        for field in INFILTRATION_RATE_FIELDS:
            values = _get_field(setpoints, field)
            if values is _MISSING or values is None:
                continue
            if isinstance(values, (list, tuple)):
                targets.append(
                    InfiltrationTarget(
                        path="setpoints",
                        kind="vector",
                        container=setpoints,
                        field=field,
                        setter=None,
                        values=[float(v) for v in values],
                    )
                )
            elif isinstance(values, (int, float)):
                targets.append(
                    InfiltrationTarget(
                        path="setpoints",
                        kind="scalar",
                        container=setpoints,
                        field=field,
                        setter=None,
                        values=[float(values)],
                    )
                )
            break

    # 3 — the rate held directly on the model.
    for field in INFILTRATION_RATE_FIELDS:
        values = _get_field(model, field)
        if values is _MISSING or values is None:
            continue
        if isinstance(values, (list, tuple)):
            targets.append(
                InfiltrationTarget(
                    path="model_field",
                    kind="vector",
                    container=model,
                    field=field,
                    setter=None,
                    values=[float(v) for v in values],
                )
            )
        elif isinstance(values, (int, float)):
            targets.append(
                InfiltrationTarget(
                    path="model_field",
                    kind="scalar",
                    container=model,
                    field=field,
                    setter=None,
                    values=[float(values)],
                )
            )
        break

    if not targets:
        raise InfiltrationNotFoundError(
            "The model exposes no infiltration representation: no schedule "
            "set with an 'infiltration' schedule (ScheduleSet.infiltration, "
            "src/api/schema.rs), no per-zone 'infiltration_rate' "
            "(SetpointState.infiltration_rate) and no model-level "
            "infiltration rate. Refusing to report an infiltration change "
            "that never happened."
        )

    return targets


def _schedule_set_path(model: Any, schedule_set: Any) -> str:
    """Return the accessor path that produced ``schedule_set``."""
    for getter_name, _setter_name in SCHEDULE_SET_ACCESSORS:
        found = _probe(model, getter_name)
        if found is not None and found is schedule_set:
            return getter_name
    return SCHEDULE_SET_ACCESSORS[0][0]


def apply_infiltration_rate(targets: list[InfiltrationTarget], rate: float) -> int:
    """Write the constant ``rate`` (ACH) into every target.

    Returns the number of representations that were written.
    """
    written = 0
    for target in targets:
        target.write(rate)
        written += 1
    return written


class SetInfiltrationRate(FluxionMeasure):
    """Set the model's infiltration rate (air leakage) to a target ACH.

    Parameters
    ----------
    infiltration_ach : float
        Target infiltration rate in air changes per hour (ACH). Must be in
        ``[0, 20]`` — zero is a legitimate sealed-envelope study.
        Default ``0.5`` (the engine's own default).
    """

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "infiltration_ach",
                "type": "double",
                "default": DEFAULT_INFILTRATION_ACH,
                "min": MIN_INFILTRATION_ACH,
                "max": MAX_INFILTRATION_ACH,
                "description": (
                    "Target infiltration rate (ACH, air changes per hour). "
                    f"Must be in [{MIN_INFILTRATION_ACH}, "
                    f"{MAX_INFILTRATION_ACH}] — 0 is a sealed-envelope "
                    "study."
                ),
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        raw = arguments.get("infiltration_ach", DEFAULT_INFILTRATION_ACH)

        # Validate BEFORE touching the model so an invalid request never
        # leaves a half-mutated model behind.
        rate = validate_infiltration_rate(raw)

        targets = resolve_infiltration_targets(model)

        written = apply_infiltration_rate(targets, rate)

        for target in targets:
            persisted = target.persist(model)
            if persisted == "advisory":
                _logger.warning(
                    "SetInfiltrationRate: the model exposes no '%s' setter "
                    "— the infiltration rate is advisory (visible on the "
                    "snapshot, not persisted into the model).",
                    target.setter or "schedule_set",
                )

        _logger.info(
            "SetInfiltrationRate: %d infiltration representation(s) set to "
            "%.3f ACH",
            written,
            rate,
        )
