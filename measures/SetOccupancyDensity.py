"""SetOccupancyDensity — standard-library Fluxion measure (Issue #3995).

Sets the **occupancy density** (occupants per unit floor area, persons/m²)
of the model's occupancy inputs. This is the people-side sibling of the
load-density measures ``measures/SetLightingPowerDensity.py`` and
``measures/SetEquipmentPowerDensity.py``: "how much does the building
save when it is occupied twice as sparsely?" is the occupancy twin of the
lighting-density study, and it is the canonical OpenStudio
``SetPeopleBuildingType`` / people-density parametric study.

Definition
----------
The occupancy density is the number of occupants per unit of floor area,
in persons per square metre (persons/m²)::

    N_occupants [persons] = OD [persons/m²] × A_floor [m²]

The occupancy *schedule* tells the engine **when** the space is occupied;
the occupancy density tells it **how densely**. Together they give the
occupant count that drives the internal gains (the sensible / latent heat
per person of ``src/sim/occupancy.rs``) and the ventilation demand.

Units and field names
---------------------
The repo's occupancy density is expressed in **persons per square metre**
(people/m²) — not in m²/person — and the measure follows that convention:

* ``occupancy_density`` — the key the BEM checker reads from an
  internal-loads configuration; the checker's own message states the unit
  as *persons/m²* and its typical band as *0.05-0.2*
  (``src/validation/copilot/checker.rs:268``).
* ``occupant_density`` — the accessor of the behaviour-crate occupancy
  generator, documented as *Occupant density [persons/m²]*
  (``fluxion-behavior/src/occupancy.rs:481``).
* ``people_per_area`` — the OpenStudio-interop people field, i.e. *People
  per Zone Floor Area {people/m2}* (``src/interop/osm/types.rs:231``,
  ``src/interop/osm/reader.rs:359``).

The model's own occupancy representation is the ``ScheduleSet.occupancy``
schedule — an hourly *fraction* (0-1) schedule, not a density
(``src/api/schema.rs:492``):

.. code-block:: rust

    pub struct ScheduleSet {
        pub occupancy: DailySchedule,
        ...
    }

The Rust ``ThermalModel`` therefore carries **no occupancy-density field**
of its own, exactly like it carries no equipment-density field (equipment
enters the model through ``ScheduleSet.equipment``, a list of
``EquipmentSpec`` entries — ``src/api/schema.rs:508``). The measure
accordingly *probes* the model for the occupancy / internal-loads object
that carries the density and uses the first one it finds.

Bounds
------
The measure accepts a **strictly positive, bounded** occupancy density:

* ``MIN_OCCUPANCY_DENSITY_PERSONS_PER_M2`` (0.01 persons/m², i.e. one
  occupant per 100 m²) — the smallest density that still describes an
  occupied space. A non-positive density means "nobody is ever here",
  which is a *deletion* of the occupancy, not an occupancy-density study,
  so it is rejected rather than silently applied (the BEM checker makes
  the same call: an occupancy density ``<= 0`` is an **error**, not a
  warning — ``src/validation/copilot/checker.rs:269``). The floor sits
  below the 0.05-0.2 typical band so that a sparsely-used space (a
  warehouse, a rarely-booked meeting room) is still modellable.
* ``MAX_OCCUPANCY_DENSITY_PERSONS_PER_M2`` (10.0 persons/m²) — the upper
  bound of the repository's declared occupancy validity range ("Occupancy
  | 0 to 10 persons/m² | occupant density",
  ``docs/SURROGATE_GOVERNANCE.md:23``). It sits well above the BEM
  checker's 1.0 persons/m² "very high" warning
  (``src/validation/copilot/checker.rs:275``) so that a genuinely dense
  space (an assembly hall, a call centre) stays modellable, while still
  rejecting values that can only be a unit error (e.g. 100 persons/m²,
  or a value that was really m²/person).

Both bounds are checked in :func:`validate_occupancy_density` before the
model is touched, so an invalid request never leaves the model
half-mutated. The bounds live in this measure — the Rust engine has no
occupancy-density constant to borrow.

Finding the occupancy object
----------------------------
The native ``fluxion.Model`` binding does not expose an occupancy
accessor today (the Python surface exposes ``zones()``, ``surfaces()``,
``hvac_system()`` — ``fluxion.pyi:98`` — and no occupancy snapshot). The
measure therefore *probes* the model for the occupancy / internal-loads
object and uses the first one it finds:

1. ``model.occupancy()`` — an occupancy snapshot carrying
   ``occupancy_density`` (the canonical field name, the key the BEM
   checker reads) or one of its aliases.
2. ``model.internal_loads()`` — an internal-loads snapshot carrying the
   same attribute(s) — the BEM checker reads the occupancy density from
   exactly such an internal-loads configuration
   (``src/validation/copilot/checker.rs:268``).
3. ``model.occupancy_density`` / ``model.occupant_density`` /
   ``model.people_per_area`` — the density held directly on the model
   object.

Accepted aliases
----------------
Every probe accepts **all three** spellings of the density field
(``occupancy_density``, ``occupant_density``, ``people_per_area``), all
of them persons/m².

If the model exposes **none** of them, the measure raises
:class:`OccupancyNotFoundError` — a loud failure. Applying an occupancy
measure to a model without occupancy and reporting success would put a
number on the provenance chain that belongs to a configuration the model
never had. This mirrors ``SetLightingPowerDensity``'s
:class:`LightingNotFoundError` and ``SetEquipmentPowerDensity``'s
:class:`EquipmentNotFoundError` pattern.

Persistence
-----------
The measure follows the snapshot / owned-value contract from
``docs/bindings.md``: the occupancy object is read **once**, mutated in
place, and pushed back through the matching setter when the binding
exposes one:

* ``model.set_occupancy(obj)`` / ``model.set_internal_loads(obj)`` — the
  mutated snapshot is written back.
* neither — the mutation is **advisory**: the new density is visible on
  the snapshot the binding handed out, but the current ``fluxion.Model``
  binding has no occupancy setter, so it does not round-trip into the
  serialized model. This mirrors the documented advisory fields of
  ``ReplaceHVACWithVAV``.

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
        --output model.occupancy.json

Where ``args.json`` is::

    {
        "SetOccupancyDensity": {
            "occupancy_density_persons_per_m2": 0.1
        }
    }
"""

from __future__ import annotations

import logging
from typing import Any

from fluxion import FluxionMeasure

_logger = logging.getLogger(__name__)

#: Smallest occupancy density the measure accepts (persons/m²) — one
#: occupant per 100 m². A density of zero means "nobody is ever here",
#: which is not an occupancy-density study.
MIN_OCCUPANCY_DENSITY_PERSONS_PER_M2 = 0.01

#: Largest occupancy density the measure accepts (persons/m²) — the upper
#: bound of the declared occupancy validity range (0 to 10 persons/m²,
#: ``docs/SURROGATE_GOVERNANCE.md:23``). Well above the BEM checker's
#: 1.0 persons/m² "very high" warning
#: (``src/validation/copilot/checker.rs:275``) so a genuinely dense space
#: stays modellable, but far below densities that can only be a unit error.
MAX_OCCUPANCY_DENSITY_PERSONS_PER_M2 = 10.0

#: Default occupancy density (persons/m²). The BEM checker calls 0.05-0.2
#: typical (``src/validation/copilot/checker.rs:280``) and the training
#: data generator defaults to 0.1 persons/m²
#: (``scripts/generate_training_data.py:198``); 0.1 sits in the middle of
#: that band.
DEFAULT_OCCUPANCY_DENSITY_PERSONS_PER_M2 = 0.1

#: Attribute names carrying the occupancy density, in probe order. All
#: three are persons/m²: ``occupancy_density`` is the key the BEM checker
#: reads from an internal-loads configuration
#: (``src/validation/copilot/checker.rs:268``); ``occupant_density`` is
#: the behaviour-crate accessor documented as *Occupant density
#: [persons/m²]* (``fluxion-behavior/src/occupancy.rs:481``);
#: ``people_per_area`` is the OpenStudio-interop *People per Zone Floor
#: Area {people/m2}* field (``src/interop/osm/types.rs:231``).
OCCUPANCY_DENSITY_ATTRIBUTES: tuple[str, ...] = (
    "occupancy_density",
    "occupant_density",
    "people_per_area",
)

#: ``(getter, setter)`` pairs probed on the model, in order. The getter is
#: called to obtain the occupancy / internal-loads object; the setter (when
#: present) writes the mutated snapshot back into the model.
OCCUPANCY_ACCESSORS: tuple[tuple[str, str], ...] = (
    ("occupancy", "set_occupancy"),
    ("internal_loads", "set_internal_loads"),
)


class OccupancyNotFoundError(ValueError):
    """Raised when the model exposes no occupancy object to mutate.

    An occupancy-density study applied to a model without occupancy would
    report a change that never happened: the provenance chain would carry
    an occupancy delta while the model's occupant count stayed untouched.
    The measure fails loudly instead.
    """


def validate_occupancy_density(
    value: Any,
    minimum: float = MIN_OCCUPANCY_DENSITY_PERSONS_PER_M2,
    maximum: float = MAX_OCCUPANCY_DENSITY_PERSONS_PER_M2,
) -> float:
    """Return ``value`` as a validated occupancy density (persons/m²).

    The density must be a finite number strictly greater than ``minimum``
    and no greater than ``maximum``. Anything else raises ``ValueError``.

    Pure function — unit-testable without the native bindings.
    """
    try:
        density = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Occupancy density must be a number, got {value!r}"
        ) from exc

    if density != density:  # NaN
        raise ValueError("Occupancy density must not be NaN")
    if density in (float("inf"), float("-inf")):
        raise ValueError("Occupancy density must be finite")
    if density <= 0.0:
        raise ValueError(
            f"Occupancy density must be > 0 persons/m², got {density}. "
            "A density of zero empties the building — that is a "
            "deletion, not an occupancy-density study."
        )
    if density < float(minimum):
        raise ValueError(
            f"Occupancy density must be >= {minimum} persons/m², "
            f"got {density}"
        )
    if density > float(maximum):
        raise ValueError(
            f"Occupancy density must be <= {maximum} persons/m², "
            f"got {density}"
        )
    return density


def _probe(model: Any, name: str) -> Any:
    """Return ``getattr(model, name)``, calling it when it is callable.

    Returns ``None`` when the attribute is missing or when calling it
    fails — a binding may expose an accessor it cannot serve on this
    model, in which case the probe moves on to the next candidate.
    """
    attribute = getattr(model, name, None)
    if attribute is None:
        return None
    if callable(attribute):
        try:
            return attribute()
        except (AttributeError, TypeError, NotImplementedError):
            return None
    return attribute


def _density_attribute(target: Any) -> str | None:
    """Return the name of the density attribute carried by ``target``.

    Supports plain attributes (the ``fluxion`` snapshot style) and
    mapping keys (a ``dict``-shaped internal-loads object). Returns
    ``None`` when the target carries neither.
    """
    for name in OCCUPANCY_DENSITY_ATTRIBUTES:
        if hasattr(target, name):
            return name
    if isinstance(target, dict):
        for name in OCCUPANCY_DENSITY_ATTRIBUTES:
            if name in target:
                return name
    return None


def resolve_occupancy_target(model: Any) -> tuple[Any, str, str]:
    """Locate the model's occupancy object and the density attribute on it.

    Returns ``(target, attribute, path)`` where ``target`` is the object
    carrying the density, ``attribute`` is the name of the density
    attribute, and ``path`` names the accessor that produced the target
    (``"occupancy"``, ``"internal_loads"`` or ``"model_attribute"``).

    Raises :class:`OccupancyNotFoundError` when the model exposes no
    occupancy object at all.

    Pure function — unit-testable without the native bindings.
    """
    for getter_name, _setter_name in OCCUPANCY_ACCESSORS:
        target = _probe(model, getter_name)
        if target is None:
            continue
        attribute = _density_attribute(target)
        if attribute is None:
            continue
        return target, attribute, getter_name

    # The model may carry the density directly (e.g. a schema-shaped
    # wrapper object).
    for name in OCCUPANCY_DENSITY_ATTRIBUTES:
        attribute = getattr(model, name, None)
        if attribute is None or callable(attribute):
            continue
        return model, name, "model_attribute"

    raise OccupancyNotFoundError(
        "The model exposes no occupancy object to mutate — no "
        "'occupancy'/'internal_loads' accessor carrying "
        f"{OCCUPANCY_DENSITY_ATTRIBUTES[0]} (or "
        f"{OCCUPANCY_DENSITY_ATTRIBUTES[1]}) was found. Refusing to report "
        "an occupancy-density change that never happened."
    )


def set_density(target: Any, attribute: str, value: float) -> None:
    """Write ``value`` into ``target``'s density attribute.

    Handles attribute-style and mapping-style targets.
    """
    if isinstance(target, dict) and not hasattr(target, attribute):
        target[attribute] = float(value)
        return
    setattr(target, attribute, float(value))


def persist_occupancy(
    model: Any,
    target: Any,
    attribute: str,
    value: float,
    path: str,
) -> str:
    """Mutate the occupancy target and push it back into ``model``.

    Returns the persistence path that was taken:

    * ``"occupancy"`` / ``"internal_loads"`` — the matching setter exists
      and the mutated snapshot was written back.
    * ``"model_attribute"`` — the density lives on the model itself; the
      attribute was set in place.
    * ``"advisory"`` — the getter exists but no setter does, so the new
      density is visible on the snapshot only.
    """
    set_density(target, attribute, value)

    if path == "model_attribute":
        return "model_attribute"

    for getter_name, setter_name in OCCUPANCY_ACCESSORS:
        if getter_name != path:
            continue
        setter = getattr(model, setter_name, None)
        if callable(setter):
            setter(target)
            return path
        break

    return "advisory"


class SetOccupancyDensity(FluxionMeasure):
    """Set the occupancy density (persons/m²) of the model's occupancy.

    Parameters
    ----------
    occupancy_density_persons_per_m2 : float
        Target occupancy density in persons per square metre (persons/m²).
        Must be strictly positive and no greater than
        ``MAX_OCCUPANCY_DENSITY_PERSONS_PER_M2`` (10 persons/m²). Default
        0.1 (the middle of the 0.05-0.2 typical band).
    """

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "occupancy_density_persons_per_m2",
                "type": "double",
                "default": DEFAULT_OCCUPANCY_DENSITY_PERSONS_PER_M2,
                "min": MIN_OCCUPANCY_DENSITY_PERSONS_PER_M2,
                "max": MAX_OCCUPANCY_DENSITY_PERSONS_PER_M2,
                "description": (
                    "Target occupancy density (persons/m²). Must be > 0 and "
                    f"<= {MAX_OCCUPANCY_DENSITY_PERSONS_PER_M2}."
                ),
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        density = validate_occupancy_density(
            arguments.get(
                "occupancy_density_persons_per_m2",
                DEFAULT_OCCUPANCY_DENSITY_PERSONS_PER_M2,
            )
        )

        target, attribute, path = resolve_occupancy_target(model)
        persisted = persist_occupancy(model, target, attribute, density, path)

        if persisted == "advisory":
            _logger.warning(
                "SetOccupancyDensity: the model exposes no occupancy "
                "setter — the occupancy density of %.3f persons/m² is "
                "advisory (visible on the occupancy snapshot, not persisted "
                "into the model).",
                density,
            )

        _logger.info(
            "SetOccupancyDensity: occupancy density = %.3f persons/m² "
            "(attribute %r, persisted via %s)",
            density,
            attribute,
            persisted,
        )
