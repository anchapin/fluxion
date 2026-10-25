"""SetLightingPowerDensity — standard-library Fluxion measure (Issue #3995).

Sets the **lighting power density** (LPD, W/m²) of the model's lighting /
internal-load inputs. This is one of the most common parametric studies in
building energy modeling — the "how much does dimming the lights save?"
sweep — and the direct analogue of the OpenStudio
``SetLightingPowerDensity`` / ``SetLightsPowerDensity`` measures.

Definition
----------
The lighting power density is the installed lighting power per unit of
floor area, in watts per square metre (W/m²)::

    P_lighting [W] = LPD [W/m²] × A_floor [m²]

Setting the LPD therefore scales the model's lighting gain (and its
electricity end use) with the floor area of the space it applies to.

Bounds
------
The measure accepts a **strictly positive, bounded** LPD:

* ``MIN_LIGHTING_POWER_DENSITY_W_M2`` (0.1 W/m²) — the smallest density
  that still describes a lit space. A non-positive density means "no
  lighting at all", which is a *deletion* of the lighting load, not a
  lighting-density study, so it is rejected rather than silently applied.
* ``MAX_LIGHTING_POWER_DENSITY_W_M2`` (100.0 W/m²) — a generous upper
  bound (≈ 9.3 W/ft²); densities above that are a modelling error, not a
  retrofit.

Both bounds are checked in :func:`validate_power_density` before the model
is touched, so an invalid request never leaves the model half-mutated.

Finding the lighting object
---------------------------
The native ``fluxion.Model`` binding does not expose a lighting accessor
today (the Rust ``ThermalModel`` carries no lighting field — lighting is
applied on the API path through ``ScheduleSet.lighting_power_density_w_m2``,
see ``src/api/schema.rs``). The measure therefore *probes* the model for
the lighting / internal-loads object and uses the first one it finds:

1. ``model.lighting()`` — a lighting snapshot carrying
   ``lighting_power_density_w_m2`` (the canonical field name, mirroring the
   Rust ``ScheduleSet``) or the short alias ``power_density_w_m2``.
2. ``model.internal_loads()`` — an internal-loads snapshot carrying the
   same attribute.
3. ``model.lighting_power_density_w_m2`` — the density held directly on
   the model object.

If the model exposes **none** of them, the measure raises
:class:`LightingNotFoundError` — a loud failure. Applying a lighting
measure to a model without lighting and reporting success would put a
number on the provenance chain that belongs to a configuration the model
never had. This mirrors ``SwapSchedule``'s :class:`ScheduleNotFoundError`
pattern.

Persistence
-----------
The measure follows the snapshot / owned-value contract from
``docs/bindings.md``: the lighting object is read **once**, mutated in
place, and pushed back through the matching setter when the binding
exposes one:

* ``model.set_lighting(obj)`` / ``model.set_internal_loads(obj)`` — the
  mutated snapshot is written back.
* neither — the mutation is **advisory**: the new density is visible on
  the snapshot the binding handed out, but the current ``fluxion.Model``
  binding has no lighting setter, so it does not round-trip into the
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
        --output model.lighting.json

Where ``args.json`` is::

    {
        "SetLightingPowerDensity": {
            "power_density_w_m2": 8.0
        }
    }
"""

from __future__ import annotations

import logging
from typing import Any

from fluxion import FluxionMeasure

_logger = logging.getLogger(__name__)

#: Smallest lighting power density the measure accepts (W/m²). A density of
#: zero means "no lighting", which is not a lighting-density study.
MIN_LIGHTING_POWER_DENSITY_W_M2 = 0.1

#: Largest lighting power density the measure accepts (W/m²) — ≈ 9.3 W/ft².
MAX_LIGHTING_POWER_DENSITY_W_M2 = 100.0

#: Default lighting power density (W/m²). Mirrors the office profile's
#: lighting density in ``src/sim/profiles.rs`` (office 10 W/m², retail
#: 12 W/m², school 8 W/m²).
DEFAULT_LIGHTING_POWER_DENSITY_W_M2 = 10.0

#: Attribute names carrying the lighting power density, in probe order.
#: ``lighting_power_density_w_m2`` mirrors the Rust
#: ``ScheduleSet.lighting_power_density_w_m2`` field (``src/api/schema.rs``);
#: ``power_density_w_m2`` is the short alias used by the ``sim::lighting``
#: module.
LIGHTING_DENSITY_ATTRIBUTES: tuple[str, ...] = (
    "lighting_power_density_w_m2",
    "power_density_w_m2",
)

#: ``(getter, setter)`` pairs probed on the model, in order. The getter is
#: called to obtain the lighting / internal-loads object; the setter (when
#: present) writes the mutated snapshot back into the model.
LIGHTING_ACCESSORS: tuple[tuple[str, str], ...] = (
    ("lighting", "set_lighting"),
    ("internal_loads", "set_internal_loads"),
)


class LightingNotFoundError(ValueError):
    """Raised when the model exposes no lighting object to mutate.

    A lighting-density study applied to a model without lighting would
    report a change that never happened: the provenance chain would carry a
    lighting delta while the model's lighting load stayed untouched. The
    measure fails loudly instead.
    """


def validate_power_density(
    value: Any,
    minimum: float = MIN_LIGHTING_POWER_DENSITY_W_M2,
    maximum: float = MAX_LIGHTING_POWER_DENSITY_W_M2,
) -> float:
    """Return ``value`` as a validated lighting power density (W/m²).

    The density must be a finite number strictly greater than ``minimum``
    and no greater than ``maximum``. Anything else raises ``ValueError``.

    Pure function — unit-testable without the native bindings.
    """
    try:
        density = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Lighting power density must be a number, got {value!r}"
        ) from exc

    if density != density:  # NaN
        raise ValueError("Lighting power density must not be NaN")
    if density in (float("inf"), float("-inf")):
        raise ValueError("Lighting power density must be finite")
    if density <= 0.0:
        raise ValueError(
            f"Lighting power density must be > 0 W/m², got {density}. "
            "A density of zero removes the lighting load — that is a "
            "deletion, not a lighting-density study."
        )
    if density < float(minimum):
        raise ValueError(
            f"Lighting power density must be >= {minimum} W/m², got {density}"
        )
    if density > float(maximum):
        raise ValueError(
            f"Lighting power density must be <= {maximum} W/m², got {density}"
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
    for name in LIGHTING_DENSITY_ATTRIBUTES:
        if hasattr(target, name):
            return name
    if isinstance(target, dict):
        for name in LIGHTING_DENSITY_ATTRIBUTES:
            if name in target:
                return name
    return None


def resolve_lighting_target(model: Any) -> tuple[Any, str, str]:
    """Locate the model's lighting object and the density attribute on it.

    Returns ``(target, attribute, path)`` where ``target`` is the object
    carrying the density, ``attribute`` is the name of the density
    attribute, and ``path`` names the accessor that produced the target
    (``"lighting"``, ``"internal_loads"`` or ``"model_attribute"``).

    Raises :class:`LightingNotFoundError` when the model exposes no
    lighting object at all.

    Pure function — unit-testable without the native bindings.
    """
    for getter_name, _setter_name in LIGHTING_ACCESSORS:
        target = _probe(model, getter_name)
        if target is None:
            continue
        attribute = _density_attribute(target)
        if attribute is None:
            continue
        return target, attribute, getter_name

    # The model may carry the density directly (e.g. a schema-shaped
    # wrapper object).
    for name in LIGHTING_DENSITY_ATTRIBUTES:
        attribute = getattr(model, name, None)
        if attribute is None or callable(attribute):
            continue
        return model, name, "model_attribute"

    raise LightingNotFoundError(
        "The model exposes no lighting object to mutate — no "
        "'lighting'/'internal_loads' accessor carrying "
        f"{LIGHTING_DENSITY_ATTRIBUTES[0]} (or "
        f"{LIGHTING_DENSITY_ATTRIBUTES[1]}) was found. Refusing to report a "
        "lighting-density change that never happened."
    )


def set_density(target: Any, attribute: str, value: float) -> None:
    """Write ``value`` into ``target``'s density attribute.

    Handles attribute-style and mapping-style targets.
    """
    if isinstance(target, dict) and not hasattr(target, attribute):
        target[attribute] = float(value)
        return
    setattr(target, attribute, float(value))


def persist_lighting(
    model: Any,
    target: Any,
    attribute: str,
    value: float,
    path: str,
) -> str:
    """Mutate the lighting target and push it back into ``model``.

    Returns the persistence path that was taken:

    * ``"lighting"`` / ``"internal_loads"`` — the matching setter exists
      and the mutated snapshot was written back.
    * ``"model_attribute"`` — the density lives on the model itself; the
      attribute was set in place.
    * ``"advisory"`` — the getter exists but no setter does, so the new
      density is visible on the snapshot only.
    """
    set_density(target, attribute, value)

    if path == "model_attribute":
        return "model_attribute"

    for getter_name, setter_name in LIGHTING_ACCESSORS:
        if getter_name != path:
            continue
        setter = getattr(model, setter_name, None)
        if callable(setter):
            setter(target)
            return path
        break

    return "advisory"


class SetLightingPowerDensity(FluxionMeasure):
    """Set the lighting power density (W/m²) of the model's lighting inputs.

    Parameters
    ----------
    power_density_w_m2 : float
        Target lighting power density in watts per square metre (W/m²).
        Must be strictly positive and no greater than
        ``MAX_LIGHTING_POWER_DENSITY_W_M2`` (100 W/m²). Default 10.0
        (the office profile's lighting density).
    """

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "power_density_w_m2",
                "type": "double",
                "default": DEFAULT_LIGHTING_POWER_DENSITY_W_M2,
                "min": MIN_LIGHTING_POWER_DENSITY_W_M2,
                "max": MAX_LIGHTING_POWER_DENSITY_W_M2,
                "description": (
                    "Target lighting power density (W/m²). Must be > 0 and "
                    f"<= {MAX_LIGHTING_POWER_DENSITY_W_M2}."
                ),
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        density = validate_power_density(
            arguments.get(
                "power_density_w_m2", DEFAULT_LIGHTING_POWER_DENSITY_W_M2
            )
        )

        target, attribute, path = resolve_lighting_target(model)
        persisted = persist_lighting(model, target, attribute, density, path)

        if persisted == "advisory":
            _logger.warning(
                "SetLightingPowerDensity: the model exposes no lighting "
                "setter — the lighting power density of %.2f W/m² is "
                "advisory (visible on the lighting snapshot, not persisted "
                "into the model).",
                density,
            )

        _logger.info(
            "SetLightingPowerDensity: lighting power density = %.2f W/m² "
            "(attribute %r, persisted via %s)",
            density,
            attribute,
            persisted,
        )
