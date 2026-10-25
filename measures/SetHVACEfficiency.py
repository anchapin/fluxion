"""SetHVACEfficiency — standard-library Fluxion measure (Issue #3995).

Sets the **heating and cooling efficiency** (coefficient of performance,
COP) of the model's HVAC plant. This is the canonical *plant-efficiency*
parametric study in building energy modeling — the "what does a heat pump
upgrade save?" sweep — and the direct analogue of the OpenStudio
``SetHVACEfficiency`` / ``SetPlantEfficiency`` measures.

Definition
----------
The coefficient of performance is the useful thermal output per unit of
electrical input (W_th / W_e)::

    COP = Q_useful [W] / P_electric [W]

A higher COP means the same heating / cooling load needs less electricity.
The heating COP (``cop_heating``) and the cooling COP (``cop_cooling``) are
independent — a measure run can set one and keep the other.

Bounds
------
Both COPs must be **strictly positive and bounded**:

* ``MIN_COP`` (0.1) — a COP at or below zero is physically meaningless
  (the plant would consume power without delivering any).
* ``MAX_COP`` (20.0) — a generous upper bound (realistic plant equipment
  sits between 1 and 6); anything above is a modelling error.

The bounds are checked in :func:`validate_cop` **before** the model is
touched, so an invalid request never leaves the model half-mutated.

A note on field propagation
----------------------------
The measure mutates the ``cop_heating`` / ``cop_cooling`` fields of the
model's HVAC snapshot (``model.hvac_system()``) and pushes the snapshot
back through ``model.set_hvac_system(...)``. Only ``heating_capacity`` and
``cooling_capacity`` currently round-trip into the underlying Rust
``ThermalModel`` (``apply_hvac_system_to_model`` in
``src/python/model_bindings/hvac.rs``); the COP fields are **advisory**
snapshots. They survive the Python save/load round-trip
(:func:`fluxion.measures.save_model` / :func:`fluxion.measures.load_model`
carry ``cop_heating`` / ``cop_cooling``) and record the modeller's intent,
but the Rust simulation does not yet consume them. This mirrors the
documented advisory fields of ``ReplaceHVACWithVAV`` and the note in the
``SetHVACCOP`` example.

Loud failure
------------
A model that exposes no HVAC system at all cannot carry a plant
efficiency; the measure raises :class:`HVACSystemNotFoundError` instead of
reporting an efficiency change that never happened — the same
"fail loudly" discipline as ``SwapSchedule``'s
:class:`ScheduleNotFoundError`.

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
        --output model.efficient.json

Where ``args.json`` is::

    {
        "SetHVACEfficiency": {
            "cop_heating": 3.5,
            "cop_cooling": 4.0
        }
    }
"""

from __future__ import annotations

import logging
from typing import Any

from fluxion import FluxionMeasure

_logger = logging.getLogger(__name__)

#: Smallest coefficient of performance the measure accepts. A COP at or
#: below zero is physically meaningless.
MIN_COP = 0.1

#: Largest coefficient of performance the measure accepts (realistic plant
#: equipment sits between 1 and 6).
MAX_COP = 20.0

#: The HVAC snapshot fields this measure writes.
COP_FIELDS: tuple[str, ...] = ("cop_heating", "cop_cooling")


class HVACSystemNotFoundError(ValueError):
    """Raised when the model exposes no HVAC system to mutate.

    A plant-efficiency study applied to a model without an HVAC system
    would report a change that never happened: the provenance chain would
    carry an efficiency delta while the model kept its (missing) plant.
    The measure fails loudly instead.
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


def resolve_hvac_snapshot(model: Any) -> Any:
    """Return the model's HVAC snapshot.

    Raises :class:`HVACSystemNotFoundError` when the model exposes no
    ``hvac_system()`` accessor, or when the snapshot carries none of the
    COP fields this measure writes.

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
            "The model exposes no HVAC system to mutate — no "
            "'hvac_system()' accessor was found. Refusing to report a "
            "plant-efficiency change that never happened."
        )

    missing = [field for field in COP_FIELDS if not hasattr(snapshot, field)]
    if len(missing) == len(COP_FIELDS):
        raise HVACSystemNotFoundError(
            "The model's HVAC snapshot carries none of the COP fields "
            f"{COP_FIELDS} — refusing to report a plant-efficiency change "
            "that never happened."
        )

    return snapshot


def persist_hvac(model: Any, snapshot: Any) -> str:
    """Push the mutated HVAC snapshot back into ``model``.

    Returns the persistence path that was taken:

    * ``"hvac_system"`` — the model exposes ``set_hvac_system`` and the
      mutated snapshot was written back.
    * ``"advisory"`` — no setter exists, so the new COPs are visible on
      the snapshot only.
    """
    setter = getattr(model, "set_hvac_system", None)
    if callable(setter):
        setter(snapshot)
        return "hvac_system"
    return "advisory"


class SetHVACEfficiency(FluxionMeasure):
    """Set the heating and cooling coefficient of performance of the plant.

    Parameters
    ----------
    cop_heating : float or None
        Heating coefficient of performance (W_th/W_e). Must be > 0 and
        <= ``MAX_COP`` (20.0). ``None`` (default) keeps the model's
        current heating COP.
    cop_cooling : float or None
        Cooling coefficient of performance (W_th/W_e). Must be > 0 and
        <= ``MAX_COP`` (20.0). ``None`` (default) keeps the model's
        current cooling COP.
    """

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "cop_heating",
                "type": "double",
                "default": None,
                "min": MIN_COP,
                "max": MAX_COP,
                "description": (
                    "Heating coefficient of performance (W_th/W_e). "
                    "null keeps the model's current heating COP."
                ),
            },
            {
                "name": "cop_cooling",
                "type": "double",
                "default": None,
                "min": MIN_COP,
                "max": MAX_COP,
                "description": (
                    "Cooling coefficient of performance (W_th/W_e). "
                    "null keeps the model's current cooling COP."
                ),
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        cop_heating = arguments.get("cop_heating", None)
        cop_cooling = arguments.get("cop_cooling", None)

        # Validate BEFORE touching the model so an invalid request never
        # leaves a half-mutated HVAC snapshot behind.
        new_heating = (
            None
            if cop_heating is None
            else validate_cop(cop_heating, name="cop_heating")
        )
        new_cooling = (
            None
            if cop_cooling is None
            else validate_cop(cop_cooling, name="cop_cooling")
        )

        snapshot = resolve_hvac_snapshot(model)

        if new_heating is not None:
            snapshot.cop_heating = new_heating
        if new_cooling is not None:
            snapshot.cop_cooling = new_cooling

        persisted = persist_hvac(model, snapshot)

        if persisted == "advisory":
            _logger.warning(
                "SetHVACEfficiency: the model exposes no 'set_hvac_system' "
                "setter — the new COPs are advisory (visible on the HVAC "
                "snapshot, not persisted into the model)."
            )

        _logger.info(
            "SetHVACEfficiency: cop_heating=%s, cop_cooling=%s — persisted "
            "via %s",
            "unchanged" if new_heating is None else f"{new_heating:.3f}",
            "unchanged" if new_cooling is None else f"{new_cooling:.3f}",
            persisted,
        )
