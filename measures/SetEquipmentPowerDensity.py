"""SetEquipmentPowerDensity — standard-library Fluxion measure (Issue #3995).

Sets the **equipment / plug-load power density** (EPD, W/m²) of the model's
equipment (plug & process equipment) inputs. This is the direct analogue of
the OpenStudio ``SetEquipmentPowerDensity`` measure and the sibling of
``measures/SetLightingPowerDensity.py``: "how much does swapping office
plug-loads for efficient ones save?" is the equipment-side twin of the
lighting-density study.

Definition
----------
The equipment power density is the installed plug / process equipment power
per unit of floor area, in watts per square metre (W/m²)::

    P_equipment [W] = EPD [W/m²] × A_floor [m²]

Setting the EPD therefore scales the model's equipment gain (and its
electricity end use) with the floor area of the space it applies to. The
density is the *area-normalized* way of describing a plug load — the same
role the lighting power density plays for the lighting end use.

Bounds
------
The measure accepts a **strictly positive, bounded** EPD:

* ``MIN_EQUIPMENT_POWER_DENSITY_W_M2`` (0.1 W/m²) — the smallest density
  that still describes an equipped space. A non-positive density means "no
  equipment at all", which is a *deletion* of the equipment load, not an
  equipment-density study, so it is rejected rather than silently applied.
  (The floor is the same as the lighting density's: the smallest density
  that is still a density.)
* ``MAX_EQUIPMENT_POWER_DENSITY_W_M2`` (200.0 W/m² ≈ 18.6 W/ft²) — a
  generous upper bound. Typical office plug loads are far below that: the
  BEM checker in ``src/validation/copilot/checker.rs`` flags an equipment
  power density above 30 W/m² as *high* for an office ("typical: 5-15 for
  offices"). The bound is set well above that warning so a dense
  server / telecom room (where per-floor-area densities of 50-100 W/m² are
  normal) can still be modelled, while still rejecting densities that can
  only be a unit error (e.g. 5 000 W/m²). It is deliberately **higher**
  than the lighting bound of 100 W/m²: equipment densities (plug loads,
  process equipment) are typically larger than lighting densities, so the
  lighting bound would clip legitimate equipment studies.

Both bounds are checked in :func:`validate_power_density` before the model
is touched, so an invalid request never leaves the model half-mutated. The
bounds live in this measure — the Rust engine has no equipment-density
constant to borrow.

Finding the equipment object
----------------------------
The native ``fluxion.Model`` binding does not expose an equipment accessor
today (the Rust ``ThermalModel`` carries no equipment-density field —
equipment enters the model through ``ScheduleSet.equipment``, a list of
``EquipmentSpec`` entries, see ``src/api/schema.rs``). The measure
therefore *probes* the model for the equipment / internal-loads object and
uses the first one it finds:

1. ``model.equipment()`` — an equipment snapshot carrying
   ``equipment_power_density_w_m2`` (the canonical field name, mirroring the
   Rust ``ScheduleSet.lighting_power_density_w_m2`` naming for a W/m²
   density) or the short alias ``equipment_power_density`` (the key the BEM
   checker reads from an internal-loads configuration,
   ``src/validation/copilot/checker.rs``).
2. ``model.internal_loads()`` — an internal-loads snapshot carrying the
   same attribute(s).
3. ``model.equipment_power_density_w_m2`` /
   ``model.equipment_power_density`` — the density held directly on the
   model object.

Accepted aliases
----------------
Every probe accepts **both** spellings of the density field:

* ``equipment_power_density_w_m2`` — canonical. Matches the Rust
  ``ScheduleSet`` convention of naming a W/m² density
  ``<quantity>_power_density_w_m2`` (``lighting_power_density_w_m2``).
* ``equipment_power_density`` — the alias used by the BEM checker's
  internal-loads configuration (``src/validation/copilot/checker.rs``).

If the model exposes **none** of them, the measure raises
:class:`EquipmentNotFoundError` — a loud failure. Applying an equipment
measure to a model without equipment and reporting success would put a
number on the provenance chain that belongs to a configuration the model
never had. This mirrors ``SetLightingPowerDensity``'s
:class:`LightingNotFoundError` pattern.

Persistence
-----------
The measure follows the snapshot / owned-value contract from
``docs/bindings.md``: the equipment object is read **once**, mutated in
place, and pushed back through the matching setter when the binding
exposes one:

* ``model.set_equipment(obj)`` / ``model.set_internal_loads(obj)`` — the
  mutated snapshot is written back.
* neither — the mutation is **advisory**: the new density is visible on
  the snapshot the binding handed out, but the current ``fluxion.Model``
  binding has no equipment setter, so it does not round-trip into the
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
        --output model.equipment.json

Where ``args.json`` is::

    {
        "SetEquipmentPowerDensity": {
            "power_density_w_m2": 8.0
        }
    }
"""

from __future__ import annotations

import logging
from typing import Any

from fluxion import FluxionMeasure

_logger = logging.getLogger(__name__)

#: Smallest equipment power density the measure accepts (W/m²). A density of
#: zero means "no equipment", which is not an equipment-density study.
MIN_EQUIPMENT_POWER_DENSITY_W_M2 = 0.1

#: Largest equipment power density the measure accepts (W/m²) — ≈ 18.6 W/ft².
#: Well above the BEM checker's 30 W/m² "high for an office" warning
#: (``src/validation/copilot/checker.rs``) so dense server rooms stay
#: modellable, but far below densities that can only be a unit error.
MAX_EQUIPMENT_POWER_DENSITY_W_M2 = 200.0

#: Default equipment power density (W/m²). The BEM checker calls 5-15 W/m²
#: typical for offices; 10 W/m² sits in the middle of that band.
DEFAULT_EQUIPMENT_POWER_DENSITY_W_M2 = 10.0

#: Attribute names carrying the equipment power density, in probe order.
#: ``equipment_power_density_w_m2`` mirrors the Rust ``ScheduleSet`` naming
#: for a W/m² density (``lighting_power_density_w_m2``,
#: ``src/api/schema.rs``); ``equipment_power_density`` is the key the BEM
#: checker reads from an internal-loads configuration
#: (``src/validation/copilot/checker.rs``).
EQUIPMENT_DENSITY_ATTRIBUTES: tuple[str, ...] = (
    "equipment_power_density_w_m2",
    "equipment_power_density",
)

#: ``(getter, setter)`` pairs probed on the model, in order. The getter is
#: called to obtain the equipment / internal-loads object; the setter (when
#: present) writes the mutated snapshot back into the model.
EQUIPMENT_ACCESSORS: tuple[tuple[str, str], ...] = (
    ("equipment", "set_equipment"),
    ("internal_loads", "set_internal_loads"),
)


class EquipmentNotFoundError(ValueError):
    """Raised when the model exposes no equipment object to mutate.

    An equipment-density study applied to a model without equipment would
    report a change that never happened: the provenance chain would carry an
    equipment delta while the model's equipment load stayed untouched. The
    measure fails loudly instead.
    """


def validate_power_density(
    value: Any,
    minimum: float = MIN_EQUIPMENT_POWER_DENSITY_W_M2,
    maximum: float = MAX_EQUIPMENT_POWER_DENSITY_W_M2,
) -> float:
    """Return ``value`` as a validated equipment power density (W/m²).

    The density must be a finite number strictly greater than ``minimum``
    and no greater than ``maximum``. Anything else raises ``ValueError``.

    Pure function — unit-testable without the native bindings.
    """
    try:
        density = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Equipment power density must be a number, got {value!r}"
        ) from exc

    if density != density:  # NaN
        raise ValueError("Equipment power density must not be NaN")
    if density in (float("inf"), float("-inf")):
        raise ValueError("Equipment power density must be finite")
    if density <= 0.0:
        raise ValueError(
            f"Equipment power density must be > 0 W/m², got {density}. "
            "A density of zero removes the equipment load — that is a "
            "deletion, not an equipment-density study."
        )
    if density < float(minimum):
        raise ValueError(
            f"Equipment power density must be >= {minimum} W/m², got {density}"
        )
    if density > float(maximum):
        raise ValueError(
            f"Equipment power density must be <= {maximum} W/m², got {density}"
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
    for name in EQUIPMENT_DENSITY_ATTRIBUTES:
        if hasattr(target, name):
            return name
    if isinstance(target, dict):
        for name in EQUIPMENT_DENSITY_ATTRIBUTES:
            if name in target:
                return name
    return None


def resolve_equipment_target(model: Any) -> tuple[Any, str, str]:
    """Locate the model's equipment object and the density attribute on it.

    Returns ``(target, attribute, path)`` where ``target`` is the object
    carrying the density, ``attribute`` is the name of the density
    attribute, and ``path`` names the accessor that produced the target
    (``"equipment"``, ``"internal_loads"`` or ``"model_attribute"``).

    Raises :class:`EquipmentNotFoundError` when the model exposes no
    equipment object at all.

    Pure function — unit-testable without the native bindings.
    """
    for getter_name, _setter_name in EQUIPMENT_ACCESSORS:
        target = _probe(model, getter_name)
        if target is None:
            continue
        attribute = _density_attribute(target)
        if attribute is None:
            continue
        return target, attribute, getter_name

    # The model may carry the density directly (e.g. a schema-shaped
    # wrapper object).
    for name in EQUIPMENT_DENSITY_ATTRIBUTES:
        attribute = getattr(model, name, None)
        if attribute is None or callable(attribute):
            continue
        return model, name, "model_attribute"

    raise EquipmentNotFoundError(
        "The model exposes no equipment object to mutate — no "
        "'equipment'/'internal_loads' accessor carrying "
        f"{EQUIPMENT_DENSITY_ATTRIBUTES[0]} (or "
        f"{EQUIPMENT_DENSITY_ATTRIBUTES[1]}) was found. Refusing to report an "
        "equipment-density change that never happened."
    )


def set_density(target: Any, attribute: str, value: float) -> None:
    """Write ``value`` into ``target``'s density attribute.

    Handles attribute-style and mapping-style targets.
    """
    if isinstance(target, dict) and not hasattr(target, attribute):
        target[attribute] = float(value)
        return
    setattr(target, attribute, float(value))


def persist_equipment(
    model: Any,
    target: Any,
    attribute: str,
    value: float,
    path: str,
) -> str:
    """Mutate the equipment target and push it back into ``model``.

    Returns the persistence path that was taken:

    * ``"equipment"`` / ``"internal_loads"`` — the matching setter exists
      and the mutated snapshot was written back.
    * ``"model_attribute"`` — the density lives on the model itself; the
      attribute was set in place.
    * ``"advisory"`` — the getter exists but no setter does, so the new
      density is visible on the snapshot only.
    """
    set_density(target, attribute, value)

    if path == "model_attribute":
        return "model_attribute"

    for getter_name, setter_name in EQUIPMENT_ACCESSORS:
        if getter_name != path:
            continue
        setter = getattr(model, setter_name, None)
        if callable(setter):
            setter(target)
            return path
        break

    return "advisory"


class SetEquipmentPowerDensity(FluxionMeasure):
    """Set the equipment power density (W/m²) of the model's plug loads.

    Parameters
    ----------
    power_density_w_m2 : float
        Target equipment power density in watts per square metre (W/m²).
        Must be strictly positive and no greater than
        ``MAX_EQUIPMENT_POWER_DENSITY_W_M2`` (200 W/m²). Default 10.0
        (mid-range of the 5-15 W/m² typical office plug-load band).
    """

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "power_density_w_m2",
                "type": "double",
                "default": DEFAULT_EQUIPMENT_POWER_DENSITY_W_M2,
                "min": MIN_EQUIPMENT_POWER_DENSITY_W_M2,
                "max": MAX_EQUIPMENT_POWER_DENSITY_W_M2,
                "description": (
                    "Target equipment / plug-load power density (W/m²). "
                    "Must be > 0 and "
                    f"<= {MAX_EQUIPMENT_POWER_DENSITY_W_M2}."
                ),
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        density = validate_power_density(
            arguments.get(
                "power_density_w_m2", DEFAULT_EQUIPMENT_POWER_DENSITY_W_M2
            )
        )

        target, attribute, path = resolve_equipment_target(model)
        persisted = persist_equipment(model, target, attribute, density, path)

        if persisted == "advisory":
            _logger.warning(
                "SetEquipmentPowerDensity: the model exposes no equipment "
                "setter — the equipment power density of %.2f W/m² is "
                "advisory (visible on the equipment snapshot, not persisted "
                "into the model).",
                density,
            )

        _logger.info(
            "SetEquipmentPowerDensity: equipment power density = %.2f W/m² "
            "(attribute %r, persisted via %s)",
            density,
            attribute,
            persisted,
        )
