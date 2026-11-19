"""SetHVACCOP — example FluxionMeasure (Issue #1814).

Sets the heating and cooling capacity on the model's HVAC system. Demonstrates
the round-trip pattern for HVAC snapshots:

    1. Read ``model.hvac_system()`` (snapshot).
    2. Mutate the snapshot fields.
    3. Push back via ``model.set_hvac_system(snapshot)``.

A note on field propagation
----------------------------
UPDATE 2026-11-19 (loop round 160, ISO 13790 §10.4.5 option R): heating and
cooling capacity **and** the plant COPs (``cop_heating`` / ``cop_cooling``)
are persisted by ``Model.set_hvac_system()`` — the COPs are stored on the
model and consumed by the site/electric metering (electric input =
delivered thermal / COP; the delivered thermal rows are unchanged).
``stages`` etc. remain advisory and read-only snapshots. See
``src/python/model_bindings/model.rs`` for the full ownership story.

Run with::

    fluxion apply-measures \\
        --model base.json \\
        --measures measures/examples/ \\
        --measure-args args.json \\
        --output model.with_high_cop.json

Where ``args.json`` is::

    {
        "SetHVACCOP": {
            "heating_capacity": 15000.0,
            "cooling_capacity": 12000.0
        }
    }
"""

from __future__ import annotations

from typing import Any

from fluxion import FluxionMeasure


class SetHVACCOP(FluxionMeasure):
    """Set heating/cooling capacity on the model's HVAC system.

    Parameters
    ----------
    heating_capacity : float
        Heating capacity in watts (W). Must be > 0. Default 15000.0.
    cooling_capacity : float
        Cooling capacity in watts (W). Must be > 0. Default 12000.0.
    """

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "heating_capacity",
                "type": "double",
                "default": 15000.0,
                "min": 1.0,
                "max": 1.0e7,
                "description": "Heating capacity in watts (W).",
            },
            {
                "name": "cooling_capacity",
                "type": "double",
                "default": 12000.0,
                "min": 1.0,
                "max": 1.0e7,
                "description": "Cooling capacity in watts (W).",
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        hvac = model.hvac_system()
        hvac.heating_capacity = float(arguments.get("heating_capacity", 15000.0))
        hvac.cooling_capacity = float(arguments.get("cooling_capacity", 12000.0))
        # Snapshot COP fields round-trip since loop round 160 (2026-11-19):
        # they are stored on the model and consumed by the site/electric
        # metering (input = delivered / COP).
        hvac.cop_heating = float(arguments.get("cop_heating", hvac.cop_heating))
        hvac.cop_cooling = float(arguments.get("cop_cooling", hvac.cop_cooling))
        model.set_hvac_system(hvac)

        import logging

        logging.getLogger(__name__).info(
            "SetHVACCOP: heating_capacity=%.0f W, cooling_capacity=%.0f W",
            hvac.heating_capacity,
            hvac.cooling_capacity,
        )
