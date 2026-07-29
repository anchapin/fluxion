"""ReplaceHVACWithVAV — standard-library Fluxion measure (Issue #1815).

Removes ideal-loads air systems and attaches an explicit VAV (variable air
volume) system definition to the model. This is the canonical HVAC
retrofit / baseline measure: it converts the default ideal-loads configuration
into a realistic, parameterised VAV system with an integrated air-side
economizer, mirroring the OpenStudio ``AddVAVToAllZones`` /
``ReplaceHVACWithVAV`` measures.

What this measure does
----------------------
1. Reads the model's HVAC snapshot (``model.hvac_system()``).
2. Enables the VAV air-side: ``vav_enabled = True``.
3. Enables the integrated economizer (``economizer_enabled = True``) — standard
   on ASHRAE 90.1 VAV baselines for most climate zones.
4. Sets the cool deck supply-air temperature (``supply_air_temp``).
5. Optionally overrides the heating / cooling plant capacities and COPs.
6. Pushes the snapshot back via ``model.set_hvac_system(...)``.

A note on field propagation
---------------------------
Only ``heating_capacity`` and ``cooling_capacity`` currently round-trip into
the underlying Rust ``ThermalModel`` (see ``src/python/model_bindings.rs``).
The VAV / economizer / supply-air fields are advisory snapshots — they are
preserved through the Python save/load round-trip (``fluxion.measures``) and
represent the modeller's intent, but the Rust simulation does not yet consume
them. This mirrors the documented limitation in the ``SetHVACCOP`` example.

Provenance (Issue #1816)
------------------------
When run through :func:`fluxion.measures.apply_measures` with an
``applied_deltas`` accumulator, a ``python_measure`` entry is appended
automatically. This measure performs no provenance bookkeeping itself.

Run with::

    fluxion apply-measures \
        --model base.json \
        --measures measures/ \
        --measure-args args.json \
        --output model.vav.json

Where ``args.json`` is::

    {
        "ReplaceHVACWithVAV": {
            "heating_capacity": 18000.0,
            "cooling_capacity": 15000.0,
            "supply_air_temp": 13.0
        }
    }
"""

from __future__ import annotations

import logging
from typing import Any

from fluxion import FluxionMeasure

_logger = logging.getLogger(__name__)


def build_vav_system(existing: Any, arguments: dict[str, Any]) -> Any:
    """Return a mutated HVAC snapshot configured as a VAV system.

    ``existing`` is the model's current ``HVACSystem`` snapshot. Only fields
    explicitly present in ``arguments`` override the existing / default value,
    so callers can do a partial retrofit (e.g. flip ``vav_enabled`` without
    touching capacities). This pure-ish helper is split out so the field
    arithmetic can be unit-tested without the native bindings.

    The VAV-specific flags (``vav_enabled``, ``economizer_enabled``,
    ``supply_air_temp``) are always set by the retrofit regardless of whether
    they appear in ``arguments`` — that is the whole point of "replace with
    VAV".
    """
    # VAV retrofit is unconditional — enabling the system type is the measure's
    # raison d'être. Integrated economizer is the ASHRAE 90.1 baseline default
    # for VAV systems in most climate zones.
    existing.vav_enabled = True
    existing.economizer_enabled = True

    # Supply-air temperature: 55 °F ≈ 12.8 °C is the canonical cool-deck setpoint.
    # Honour an explicit override, otherwise use 13.0 °C.
    sat = arguments.get("supply_air_temp", 13.0)
    existing.supply_air_temp = 13.0 if sat is None else float(sat)

    # Optional plant overrides — only applied when the user supplies a value.
    # ``parse_arguments`` fills declared ``default: None`` slots with None, so we
    # test against None rather than key membership.
    if arguments.get("heating_capacity") is not None:
        existing.heating_capacity = float(arguments["heating_capacity"])
    if arguments.get("cooling_capacity") is not None:
        existing.cooling_capacity = float(arguments["cooling_capacity"])
    if arguments.get("cop_heating") is not None:
        existing.cop_heating = float(arguments["cop_heating"])
    if arguments.get("cop_cooling") is not None:
        existing.cop_cooling = float(arguments["cop_cooling"])
    if arguments.get("stages") is not None:
        existing.stages = int(arguments["stages"])
    if arguments.get("min_outdoor_temp") is not None:
        existing.min_outdoor_temp = float(arguments["min_outdoor_temp"])
    if arguments.get("max_outdoor_temp") is not None:
        existing.max_outdoor_temp = float(arguments["max_outdoor_temp"])

    return existing


class ReplaceHVACWithVAV(FluxionMeasure):
    """Replace ideal-loads with an explicit VAV air system.

    Parameters
    ----------
    heating_capacity : float, optional
        Heating plant capacity (W). When omitted, the existing capacity is
        preserved.
    cooling_capacity : float, optional
        Cooling plant capacity (W). When omitted, the existing capacity is
        preserved.
    cop_heating : float, optional
        Heating coefficient of performance (W_th/W_e). When omitted, preserved.
    cop_cooling : float, optional
        Cooling coefficient of performance (W_th/W_e). When omitted, preserved.
    supply_air_temp : float
        VAV cool-deck supply-air temperature (°C). Default 13.0 (≈ 55 °F).
    stages : int, optional
        Number of compressor / burner stages. When omitted, preserved.
    min_outdoor_temp : float, optional
        Minimum outdoor temperature for HVAC operation (°C). When omitted,
        preserved.
    max_outdoor_temp : float, optional
        Maximum outdoor temperature for HVAC operation (°C). When omitted,
        preserved.
    """

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "heating_capacity",
                "type": "double",
                "default": None,
                "min": 1.0,
                "description": "Heating plant capacity (W). Omit to preserve existing.",
            },
            {
                "name": "cooling_capacity",
                "type": "double",
                "default": None,
                "min": 1.0,
                "description": "Cooling plant capacity (W). Omit to preserve existing.",
            },
            {
                "name": "cop_heating",
                "type": "double",
                "default": None,
                "min": 0.1,
                "description": "Heating COP (W_th/W_e). Omit to preserve existing.",
            },
            {
                "name": "cop_cooling",
                "type": "double",
                "default": None,
                "min": 0.1,
                "description": "Cooling COP (W_th/W_e). Omit to preserve existing.",
            },
            {
                "name": "supply_air_temp",
                "type": "double",
                "default": 13.0,
                "min": 5.0,
                "max": 20.0,
                "description": "VAV cool-deck supply-air temperature (°C). Default 13.0 (≈55 °F).",
            },
            {
                "name": "stages",
                "type": "integer",
                "default": None,
                "min": 1,
                "description": "Number of stages. Omit to preserve existing.",
            },
            {
                "name": "min_outdoor_temp",
                "type": "double",
                "default": None,
                "description": "Min outdoor temp for HVAC operation (°C). Omit to preserve.",
            },
            {
                "name": "max_outdoor_temp",
                "type": "double",
                "default": None,
                "description": "Max outdoor temp for HVAC operation (°C). Omit to preserve.",
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        hvac = model.hvac_system()
        build_vav_system(hvac, arguments)
        model.set_hvac_system(hvac)

        _logger.info(
            "ReplaceHVACWithVAV: vav_enabled=True, economizer_enabled=True, "
            "supply_air_temp=%.1f °C, heating=%.0f W, cooling=%.0f W",
            hvac.supply_air_temp,
            hvac.heating_capacity,
            hvac.cooling_capacity,
        )
