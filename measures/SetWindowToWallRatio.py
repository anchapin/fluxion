"""SetWindowToWallRatio — standard-library Fluxion measure (Issue #1815).

Dynamically resizes all exterior windows on a zone (or every zone) to a target
window-to-wall ratio (WWR), preserving sill height and overall glazing
topology. This is the most common envelope parametric study in building energy
modeling and the direct analogue of the OpenStudio
``SetWindowToWallRatioByFacade`` measure.

Definition
----------
For each qualifying (exterior, vertical) surface::

    WWR = window_area / gross_wall_area

where ``gross_wall_area`` is the surface's total ``area``. Rescaling to a target
WWR sets ``window_area = target_wwr * area``. The window area is clamped to
``[0, area]`` so it can never exceed the wall or go negative. Only the window
*area* attribute changes — the sill height, window count, and shading topology
are untouched, matching the OpenStudio semantics where the window is grown /
shrunk about its centre.

Why "exterior, vertical" only
-----------------------------
Roofs (``Orientation.Up``) and floors (``Orientation.Down`` / ``Horizontal``)
use skylight-to-roof ratios, not WWR, and behave differently under solar gain.
Interior / partition surfaces carry no glazing. This measure therefore defaults
to the four cardinal vertical orientations; pass ``include_all_orientations`` to
override.

Provenance (Issue #1816)
------------------------
When run through :func:`fluxion.measures.apply_measures` with an
``applied_deltas`` accumulator, a ``python_measure`` entry is appended
automatically. This measure itself performs no provenance bookkeeping — the
runner owns the chain.

Run with::

    fluxion apply-measures \
        --model base.json \
        --measures measures/ \
        --measure-args args.json \
        --output model.wwr40.json

Where ``args.json`` is::

    {"SetWindowToWallRatio": {"target_wwr": 0.40, "zone_index": null}}

Or from Python::

    from fluxion import Model
    from fluxion.measures import apply_measures
    from measures.SetWindowToWallRatio import SetWindowToWallRatio

    model = Model(num_zones=2)
    apply_measures(model, [SetWindowToWallRatio],
                   {"SetWindowToWallRatio": {"target_wwr": 0.40}})
"""

from __future__ import annotations

import logging
from typing import Any

from fluxion import FluxionMeasure

_logger = logging.getLogger(__name__)

# Vertical orientations that carry glazing subject to a WWR. Up/Down/Horizontal
# are roofs/floors and use a separate skylight-to-roof ratio, so they are
# excluded by default.
_VERTICAL_ORIENTATIONS: tuple[str, ...] = ("North", "East", "South", "West")


def compute_window_area(area: float, target_wwr: float) -> float:
    """Return the ``window_area`` that realises ``target_wwr`` on ``area``.

    The result is clamped to ``[0, area]``. This pure function is split out so
    it can be unit-tested without the native bindings (RULES.md: numerical
    reasoning via code).

    Parameters
    ----------
    area : float
        Gross wall area of the surface (m²).
    target_wwr : float
        Desired window-to-wall ratio (fraction in ``[0, 1]``).
    """
    if area <= 0.0:
        return 0.0
    desired = target_wwr * area
    if desired < 0.0:
        return 0.0
    if desired > area:
        return area
    return desired


def _orientation_name(value: Any) -> str:
    """Extract the variant name from a PyO3 ``Orientation`` enum value."""
    return repr(value).rsplit(".", 1)[-1]


class SetWindowToWallRatio(FluxionMeasure):
    """Resize exterior glazing to a target window-to-wall ratio.

    Parameters
    ----------
    target_wwr : float
        Target window-to-wall ratio (fraction 0.0–1.0). Default 0.40, the
        ASHRAE 90.1 prescriptive baseline for many building types.
    zone_index : int or None
        If set, apply only to the surfaces of that zone. If ``None`` (default),
        apply to every zone in the model.
    include_all_orientations : bool
        If True, also resize glazing on ``Up``/``Down``/``Horizontal`` surfaces
        (treated as a skylight ratio). Default False — only vertical façades.
    """

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "target_wwr",
                "type": "double",
                "default": 0.40,
                "min": 0.0,
                "max": 1.0,
                "description": "Target window-to-wall ratio (fraction 0.0–1.0).",
            },
            {
                "name": "zone_index",
                "type": "integer",
                "default": None,
                "description": (
                    "Zone index to limit the measure to (0-based). "
                    "null applies to every zone."
                ),
            },
            {
                "name": "include_all_orientations",
                "type": "bool",
                "default": False,
                "description": (
                    "If true, also resize Up/Down/Horizontal surfaces "
                    "(skylight ratio). Default false (vertical façades only)."
                ),
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        target_wwr = float(arguments.get("target_wwr", 0.40))
        zone_index = arguments.get("zone_index", None)
        include_all = bool(arguments.get("include_all_orientations", False))

        allowed = None if include_all else set(_VERTICAL_ORIENTATIONS)

        # Snapshot the model's surfaces ONCE and mutate that list in place.
        # Accessing ``zone.surfaces`` repeatedly yields fresh PySurface clones
        # (PyO3 ``#[pyo3(get)]`` on a Vec returns a new list each call), so
        # the mutation would be lost on re-read. The flat ``model.surfaces()``
        # list is the canonical snapshot path (see AddSouthOverhang example
        # and docs/bindings.md).
        surfaces = model.surfaces()
        zones = model.zones()
        per_zone = len(zones[0].surfaces) if zones else 0
        if per_zone == 0:
            per_zone = 1

        modified = 0
        total_wall = 0.0
        total_window = 0.0
        for i, surface in enumerate(surfaces):
            name = _orientation_name(surface.orientation)
            if allowed is not None and name not in allowed:
                continue
            # Map the flat index back to a zone index using the per-zone
            # surface count (matches ``reshape_surfaces_for_model`` in
            # src/python/model_bindings.rs).
            surf_zone = i // per_zone
            if zone_index is not None and surf_zone != int(zone_index):
                continue
            new_window = compute_window_area(surface.area, target_wwr)
            surface.window_area = new_window
            modified += 1
            total_wall += surface.area
            total_window += new_window

        # Push the SAME mutated list back to the model.
        model.set_surfaces(surfaces)

        achieved = (total_window / total_wall) if total_wall > 0 else 0.0
        _logger.info(
            "SetWindowToWallRatio: resized %d surfaces to target WWR=%.2f "
            "(achieved aggregate WWR=%.4f)",
            modified,
            target_wwr,
            achieved,
        )
