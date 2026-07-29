"""IncreaseInsulationRValue — standard-library Fluxion measure (Issue #1815).

Adds insulation to the opaque envelope by lowering each surface's U-value
(thermal transmittance) as if a layer of additional R-value were inserted.
This is one of the most common envelope retrofit / parametric measures in
building energy modeling and the direct analogue of OpenStudio's
``IncreaseInsulationRValue`` / ``AddInsulation`` measures.

Definition
----------
Thermal transmittance and resistance are reciprocals::

    R_total = 1 / U        U = 1 / R_total

Adding ``ΔR`` (m²·K/W) of insulation to a surface gives::

    R_new = 1 / U_old + ΔR
    U_new = 1 / R_new = 1 / (1 / U_old + ΔR)

Surfaces with ``U <= 0`` (sentinel / uninitialised values) are left untouched
so the measure never corrupts placeholder geometry.

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
        --output model.insulated.json

Where ``args.json`` is::

    {"IncreaseInsulationRValue": {"delta_r": 2.5, "orientation": null}}
"""

from __future__ import annotations

import logging
from typing import Any

from fluxion import FluxionMeasure

_logger = logging.getLogger(__name__)

_VERTICAL_ORIENTATIONS: tuple[str, ...] = ("North", "East", "South", "West")


def compute_insulated_u_value(u_old: float, delta_r: float) -> float:
    """Return the new U-value after adding ``delta_r`` of insulation.

    ``U_new = 1 / (1 / U_old + delta_r)``. Surfaces with ``U <= 0`` (sentinel
    values) are returned unchanged so the measure never divides by zero or
    corrupts placeholder geometry. Split out as a pure function so the
    arithmetic is unit-testable without the native bindings (RULES.md).
    """
    if u_old <= 0.0:
        return u_old
    r_total = 1.0 / u_old + delta_r
    if r_total <= 0.0:
        return u_old
    return 1.0 / r_total


def _orientation_name(value: Any) -> str:
    """Extract the variant name from a PyO3 ``Orientation`` enum value."""
    return repr(value).rsplit(".", 1)[-1]


class IncreaseInsulationRValue(FluxionMeasure):
    """Add insulation to opaque envelope surfaces by lowering their U-value.

    Parameters
    ----------
    delta_r : float
        Additional R-value to add to each qualifying surface (m²·K/W). A
        typical retrofit range is 0.5–5.0. Default 2.0.
    orientation : str or None
        If set (e.g. ``"North"``), only insulate surfaces facing that
        direction. If ``None`` (default), insulate every surface regardless of
        orientation.
    vertical_only : bool
        If True (default), only insulate vertical façade orientations
        (N/E/S/W). Roofs and floors use different insulation conventions;
        set False to include them.
    """

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "delta_r",
                "type": "double",
                "default": 2.0,
                "min": 0.0,
                "max": 20.0,
                "description": "Additional R-value (m²·K/W) to add per surface.",
            },
            {
                "name": "orientation",
                "type": "string",
                "default": None,
                "description": (
                    "Restrict to one orientation (e.g. 'North'). "
                    "null applies to every surface."
                ),
            },
            {
                "name": "vertical_only",
                "type": "bool",
                "default": True,
                "description": (
                    "If true, only insulate vertical façades (N/E/S/W). "
                    "Default true."
                ),
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        delta_r = float(arguments.get("delta_r", 2.0))
        orientation = arguments.get("orientation", None)
        vertical_only = bool(arguments.get("vertical_only", True))

        allowed = set(_VERTICAL_ORIENTATIONS) if vertical_only else None
        if orientation is not None:
            target = {str(orientation)}
            allowed = target if allowed is None else (allowed & target)

        # Snapshot every surface (owned values — see docs/bindings.md).
        surfaces = model.surfaces()
        modified = 0
        for s in surfaces:
            name = _orientation_name(s.orientation)
            if allowed is not None and name not in allowed:
                continue
            s.u_value = compute_insulated_u_value(s.u_value, delta_r)
            modified += 1

        # Push the mutated snapshots back to the model.
        model.set_surfaces(surfaces)

        _logger.info(
            "IncreaseInsulationRValue: added ΔR=%.2f m²·K/W to %d surfaces",
            delta_r,
            modified,
        )
