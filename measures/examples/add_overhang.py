"""AddSouthOverhang — example FluxionMeasure (Issue #1814).

Adds a horizontal overhang to every south-facing opaque surface in the
model. This is the canonical "shading" measure pattern from issue #1812 —
it demonstrates the snapshot/owned-value round trip:

    1. Read ``model.surfaces()`` (snapshots).
    2. Filter for ``Orientation.South``.
    3. Mutate each snapshot via ``surface.add_overhang(...)``.
    4. Push the full flat list back via ``model.set_surfaces(snapshots)``.

Run with::

    fluxion apply-measures \\
        --model base.json \\
        --measures measures/examples/ \\
        --output model.with_overhangs.json

Or from Python::

    from fluxion import Model
    from fluxion.measures import apply_measures
    from measures.examples.add_overhang import AddSouthOverhang

    model = Model(num_zones=2)
    apply_measures(model, [AddSouthOverhang], {"AddSouthOverhang": {"depth": 1.0, "height": 2.5}})
"""

from __future__ import annotations

from typing import Any

from fluxion import FluxionMeasure


class AddSouthOverhang(FluxionMeasure):
    """Add a horizontal overhang to every south-facing opaque surface.

    Parameters
    ----------
    depth : float
        Overhang depth in meters (default 1.0 m).
    height : float
        Mounting height above the window (default 2.5 m).
    only_with_windows : bool
        If True, only attach the overhang to surfaces that already have
        ``window_area > 0`` (default False — apply to all south-facing
        opaque surfaces).
    """

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "depth",
                "type": "double",
                "default": 1.0,
                "min": 0.0,
                "max": 5.0,
                "description": "Overhang depth in meters (m).",
            },
            {
                "name": "height",
                "type": "double",
                "default": 2.5,
                "min": 0.0,
                "max": 10.0,
                "description": "Mounting height above the window (m).",
            },
            {
                "name": "only_with_windows",
                "type": "bool",
                "default": False,
                "description": (
                    "If true, only attach the overhang to surfaces with window_area > 0."
                ),
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        import fluxion

        depth = float(arguments.get("depth", 1.0))
        height = float(arguments.get("height", 2.5))
        only_windows = bool(arguments.get("only_with_windows", False))

        # Snapshot every surface (owned values — see docs/bindings.md).
        surfaces = model.surfaces()
        modified = 0
        for s in surfaces:
            if s.orientation != fluxion.Orientation.South:
                continue
            if only_windows and s.window_area <= 0.0:
                continue
            s.add_overhang(depth=depth, height=height)
            modified += 1

        # Push the mutated snapshots back to the model.
        model.set_surfaces(surfaces)

        # Log via standard logging (no print, so the CLI output stays clean).
        import logging

        logging.getLogger(__name__).info(
            "AddSouthOverhang: attached overhang to %d south-facing surfaces (depth=%.2f m, height=%.2f m)",
            modified,
            depth,
            height,
        )
