#!/usr/bin/env python3
"""
ASHRAE 140 Case Generator for Training Data Generation.

This module implements a generator for all ASHRAE 140 test cases with
parameter variations support, enabling systematic generation of training
datasets covering the full ASHRAE 140 parameter space.

Features:
- Generate all 17 standard ASHRAE 140 cases
- Support parameter variations (U-values ±50%, setpoints ±5°C)
- Output in structured JSON format
- Seed control for reproducibility
- Integration with parameter variation sampler

References:
- ASHRAE Standard 140 test case specifications
- fluxion::validation::ashrae_140_cases (Rust implementation)
"""

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class GlassType(Enum):
    """Window glass types."""

    SINGLE_CLEAR = "SingleClear"
    DOUBLE_CLEAR = "DoubleClear"
    DOUBLE_LOW_E = "DoubleLowE"
    TRIPLE_CLEAR = "TripleClear"
    TRIPLE_LOW_E = "TripleLowE"


class Orientation(Enum):
    """Surface orientation."""

    NORTH = "North"
    EAST = "East"
    SOUTH = "South"
    WEST = "West"
    UP = "Up"  # Roof
    DOWN = "Down"  # Floor
    HORIZONTAL = "Horizontal"


class ConstructionType(Enum):
    """Construction type for ASHRAE 140 cases."""

    LOW_MASS = "LowMass"
    HIGH_MASS = "HighMass"
    SPECIAL = "Special"


class ShadingType(Enum):
    """Shading device types."""

    NONE = "None"
    OVERHANG = "Overhang"
    FINS = "Fins"
    OVERHANG_AND_FINS = "OverhangAndFins"


@dataclass
class WindowSpec:
    """Window specification with thermal and optical properties."""

    u_value: float
    shgc: float
    normal_transmittance: float
    glass_type: GlassType

    @classmethod
    def double_clear_glass(cls) -> "WindowSpec":
        """Creates a double clear glass window (ASHRAE 140 typical)."""
        return cls(
            u_value=3.0,
            shgc=0.789,
            normal_transmittance=0.86156,
            glass_type=GlassType.DOUBLE_CLEAR,
        )


@dataclass
class WindowArea:
    """Window specification with area and orientation."""

    area: float
    orientation: Orientation
    height: float = 2.0
    width: float = 0.0
    sill_height: float = 0.2
    left_offset: float = 0.5

    def __post_init__(self):
        """Calculate width from area if not specified."""
        if self.width == 0.0 and self.area > 0.0:
            self.width = self.area / self.height


@dataclass
class ShadingDevice:
    """Shading device specification."""

    shading_type: ShadingType
    overhang_depth: float = 0.0
    fin_width: float = 0.0
    mounting_height: float = 0.0

    @classmethod
    def none(cls) -> "ShadingDevice":
        """Creates a no-shading specification."""
        return cls(shading_type=ShadingType.NONE)

    @classmethod
    def overhang(cls, depth: float, height: float) -> "ShadingDevice":
        """Creates an overhang shading device."""
        return cls(
            shading_type=ShadingType.OVERHANG,
            overhang_depth=depth,
            mounting_height=height,
        )

    @classmethod
    def fins(cls, width: float) -> "ShadingDevice":
        """Creates shade fins."""
        return cls(shading_type=ShadingType.FINS, fin_width=width)

    @classmethod
    def overhang_and_fins(
        cls, overhang_depth: float, fin_width: float, height: float
    ) -> "ShadingDevice":
        """Creates both overhang and fins."""
        return cls(
            shading_type=ShadingType.OVERHANG_AND_FINS,
            overhang_depth=overhang_depth,
            fin_width=fin_width,
            mounting_height=height,
        )


@dataclass
class InternalLoads:
    """Internal heat gains specification."""

    total_load: float
    radiative_fraction: float
    convective_fraction: float


@dataclass
class HvacSchedule:
    """HVAC control schedule."""

    heating_setpoint: float
    cooling_setpoint: float
    operating_hours: tuple  # (start_hour, end_hour)
    setback_setpoint: Optional[float] = None
    setback_hours: Optional[tuple] = None
    efficiency: float = 1.0

    def is_free_floating(self) -> bool:
        """Returns true if this is a free-floating schedule."""
        return self.efficiency == 0.0 and self.operating_hours == (0, 0)

    @classmethod
    def constant(cls, heating: float, cooling: float) -> "HvacSchedule":
        """Creates a constant HVAC schedule (no setback)."""
        return cls(heating, cooling, (0, 24))

    @classmethod
    def with_setback(
        cls, heating: float, cooling: float, setback: float, start: int, end: int
    ) -> "HvacSchedule":
        """Creates an HVAC schedule with setback."""
        return cls(heating, cooling, (0, 24), setback, (start, end))

    @classmethod
    def with_operating_hours(
        cls, heating: float, cooling: float, start: int, end: int
    ) -> "HvacSchedule":
        """Creates an HVAC schedule with operating hours restriction."""
        return cls(heating, cooling, (start, end))

    @classmethod
    def free_floating(cls) -> "HvacSchedule":
        """Creates a free-floating schedule (no HVAC control)."""
        return cls(0.0, 0.0, (0, 0), efficiency=0.0)


@dataclass
class NightVentilation:
    """Night ventilation specification."""

    fan_capacity: float
    operating_hours: tuple  # (start_hour, end_hour)
    adds_heat: bool = False

    @classmethod
    def case_650(cls) -> "NightVentilation":
        """Creates the ASHRAE 140 Case 650 night ventilation specification."""
        return cls(fan_capacity=1703.16, operating_hours=(18, 7))


@dataclass
class GeometrySpec:
    """Building zone geometry specification."""

    width: float
    depth: float
    height: float


@dataclass
class ConstructionSpec:
    """Construction specification."""

    wall_u_value: float
    roof_u_value: float
    floor_u_value: float


@dataclass
class CaseSpec:
    """Complete ASHRAE 140 test case specification."""

    case_id: str
    description: str
    geometry: List[GeometrySpec]
    construction_type: ConstructionType
    construction: ConstructionSpec
    windows: List[List[WindowArea]]
    window_properties: WindowSpec
    shading: Optional[ShadingDevice]
    internal_loads: List[Optional[InternalLoads]]
    hvac: List[HvacSchedule]
    night_ventilation: Optional[NightVentilation] = None
    infiltration_ach: float = 0.5
    opaque_absorptance: float = 0.6
    num_zones: int = 1

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "case_id": self.case_id,
            "description": self.description,
            "construction_type": self.construction_type.value,
            "geometry": [asdict(g) for g in self.geometry],
            "construction": asdict(self.construction),
            "windows": [
                [
                    {
                        "area": w.area,
                        "orientation": w.orientation.value,
                        "height": w.height,
                        "width": w.width,
                        "sill_height": w.sill_height,
                        "left_offset": w.left_offset,
                    }
                    for w in zone
                ]
                for zone in self.windows
            ],
            "window_properties": {
                "u_value": self.window_properties.u_value,
                "shgc": self.window_properties.shgc,
                "normal_transmittance": self.window_properties.normal_transmittance,
                "glass_type": self.window_properties.glass_type.value,
            },
            "shading": asdict(self.shading) if self.shading else None,
            "internal_loads": [asdict(l) if l else None for l in self.internal_loads],
            "hvac": [
                {
                    "heating_setpoint": h.heating_setpoint,
                    "cooling_setpoint": h.cooling_setpoint,
                    "operating_hours": h.operating_hours,
                    "setback_setpoint": h.setback_setpoint,
                    "setback_hours": h.setback_hours,
                    "efficiency": h.efficiency,
                }
                for h in self.hvac
            ],
            "night_ventilation": asdict(self.night_ventilation)
            if self.night_ventilation
            else None,
            "infiltration_ach": self.infiltration_ach,
            "opaque_absorptance": self.opaque_absorptance,
            "num_zones": self.num_zones,
        }


class ASHRAE140CaseGenerator:
    """
    Generator for ASHRAE 140 test cases with parameter variations.

    This generator can create all 17 standard ASHRAE 140 test cases
    and apply parameter variations for training data generation.

    Example:
        >>> generator = ASHRAE140CaseGenerator(seed=42)
        >>> cases = generator.generate_all_cases()
        >>> variations = generator.generate_variations(
        ...     base_case=Case600,
        ...     u_value_variation=0.5,
        ...     setpoint_variation=5.0,
        ...     num_variations=10
        ... )
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the case generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            logger.info(f"ASHRAE140CaseGenerator initialized with seed {seed}")

    def generate_all_cases(self) -> List[CaseSpec]:
        """
        Generate all 17 standard ASHRAE 140 test cases.

        Returns:
            List of CaseSpec for all standard cases
        """
        cases = []
        cases.append(self.case_600_baseline())
        cases.append(self.case_610_south_shading())
        cases.append(self.case_620_ew_windows())
        cases.append(self.case_630_ew_shading())
        cases.append(self.case_640_setback())
        cases.append(self.case_650_night_vent())
        cases.append(self.case_600ff())
        cases.append(self.case_650ff())
        cases.append(self.case_900_baseline())
        cases.append(self.case_910_south_shading())
        cases.append(self.case_920_ew_windows())
        cases.append(self.case_930_ew_shading())
        cases.append(self.case_940_setback())
        cases.append(self.case_950_night_vent())
        cases.append(self.case_900ff())
        cases.append(self.case_950ff())
        cases.append(self.case_960_sunspace())
        cases.append(self.case_195_solid_conduction())

        logger.info(f"Generated {len(cases)} ASHRAE 140 cases")
        return cases

    def generate_variations(
        self,
        base_case: CaseSpec,
        u_value_variation: float = 0.5,
        setpoint_variation: float = 5.0,
        infiltration_variation: float = 0.2,
        num_variations: int = 10,
        seed: Optional[int] = None,
    ) -> List[CaseSpec]:
        """
        Generate parameter variations of a base case.

        Args:
            base_case: Base case specification to vary
            u_value_variation: U-value variation as fraction of baseline (0.5 = ±50%)
            setpoint_variation: Setpoint variation in °C
            infiltration_variation: Infiltration rate variation in ACH
            num_variations: Number of variations to generate
            seed: Random seed for reproducibility (overrides class seed if provided)

        Returns:
            List of CaseSpec with parameter variations
        """
        if seed is not None:
            random.seed(seed)
        elif self.seed is not None:
            random.seed(self.seed)

        variations = []

        for i in range(num_variations):
            case = self._copy_case(base_case)

            # Vary U-values
            u_scale = 1.0 + random.uniform(-u_value_variation, u_value_variation)
            case.construction.wall_u_value *= u_scale
            case.construction.roof_u_value *= u_scale
            case.construction.floor_u_value *= u_scale

            # Vary window U-value
            case.window_properties.u_value *= u_scale

            # Vary setpoints
            heating_offset = random.uniform(-setpoint_variation, setpoint_variation)
            cooling_offset = random.uniform(-setpoint_variation, setpoint_variation)

            for hvac in case.hvac:
                if not hvac.is_free_floating():
                    hvac.heating_setpoint += heating_offset
                    hvac.cooling_setpoint += cooling_offset
                    if hvac.setback_setpoint:
                        hvac.setback_setpoint += heating_offset

            # Vary infiltration
            inf_offset = random.uniform(-infiltration_variation, infiltration_variation)
            case.infiltration_ach = max(0.0, base_case.infiltration_ach + inf_offset)

            # Update case ID to indicate variation
            case.case_id = f"{base_case.case_id}_var{i + 1:03d}"
            case.description = f"Variation {i + 1} of {base_case.description}"

            variations.append(case)

        logger.info(
            f"Generated {len(variations)} variations of case {base_case.case_id}"
        )
        return variations

    def generate_all_variations(
        self,
        u_value_variation: float = 0.5,
        setpoint_variation: float = 5.0,
        num_variations_per_case: int = 10,
        seed: Optional[int] = None,
    ) -> List[CaseSpec]:
        """
        Generate variations for all standard cases.

        Args:
            u_value_variation: U-value variation as fraction of baseline
            setpoint_variation: Setpoint variation in °C
            num_variations_per_case: Number of variations per case
            seed: Random seed for reproducibility

        Returns:
            List of CaseSpec with variations for all cases
        """
        base_cases = self.generate_all_cases()
        all_variations = []

        for base_case in base_cases:
            variations = self.generate_variations(
                base_case=base_case,
                u_value_variation=u_value_variation,
                setpoint_variation=setpoint_variation,
                num_variations=num_variations_per_case,
                seed=seed,
            )
            all_variations.extend(variations)

        logger.info(
            f"Generated {len(all_variations)} total variations across all cases"
        )
        return all_variations

    def _copy_case(self, case: CaseSpec) -> CaseSpec:
        """Deep copy a case specification."""
        import copy

        return copy.deepcopy(case)

    # ===== Low Mass Cases (600 series) =====

    def case_600_baseline(self) -> CaseSpec:
        """Case 600 - Low mass baseline."""
        return CaseSpec(
            case_id="600",
            description="Low mass baseline - standard construction with south windows",
            geometry=[GeometrySpec(width=8.0, depth=6.0, height=2.7)],
            construction_type=ConstructionType.LOW_MASS,
            construction=ConstructionSpec(
                wall_u_value=0.514, roof_u_value=0.318, floor_u_value=0.197
            ),
            windows=[[WindowArea(area=12.0, orientation=Orientation.SOUTH)]],
            window_properties=WindowSpec.double_clear_glass(),
            shading=None,
            internal_loads=[
                InternalLoads(
                    total_load=200.0, radiative_fraction=0.6, convective_fraction=0.4
                )
            ],
            hvac=[HvacSchedule.constant(20.0, 27.0)],
            infiltration_ach=0.5,
            opaque_absorptance=0.6,
            num_zones=1,
        )

    def case_610_south_shading(self) -> CaseSpec:
        """Case 610 - Low mass with south shading."""
        base = self.case_600_baseline()
        base.case_id = "610"
        base.description = "Low mass with south shading (1m overhang)"
        base.shading = ShadingDevice.overhang(1.0, 2.7)
        return base

    def case_620_ew_windows(self) -> CaseSpec:
        """Case 620 - Low mass with east/west windows."""
        base = self.case_600_baseline()
        base.case_id = "620"
        base.description = "Low mass with east/west windows (6m² each)"
        base.windows = [
            [
                WindowArea(area=6.0, orientation=Orientation.EAST),
                WindowArea(area=6.0, orientation=Orientation.WEST),
            ]
        ]
        return base

    def case_630_ew_shading(self) -> CaseSpec:
        """Case 630 - Low mass with east/west shading."""
        base = self.case_620_ew_windows()
        base.case_id = "630"
        base.description = "Low mass with east/west shading (overhang + fins)"
        base.shading = ShadingDevice.overhang_and_fins(1.0, 1.0, 2.7)
        return base

    def case_640_setback(self) -> CaseSpec:
        """Case 640 - Low mass with thermostat setback."""
        base = self.case_600_baseline()
        base.case_id = "640"
        base.description = "Low mass with thermostat setback (overnight)"
        base.hvac = [HvacSchedule.with_setback(20.0, 27.0, 10.0, 23, 7)]
        return base

    def case_650_night_vent(self) -> CaseSpec:
        """Case 650 - Low mass with night ventilation."""
        base = self.case_600_baseline()
        base.case_id = "650"
        base.description = "Low mass with night ventilation (no heating)"
        base.hvac = [HvacSchedule.with_operating_hours(-100.0, 27.0, 7, 18)]
        base.night_ventilation = NightVentilation.case_650()
        return base

    def case_600ff(self) -> CaseSpec:
        """Case 600FF - Low mass free-floating."""
        base = self.case_600_baseline()
        base.case_id = "600FF"
        base.description = "Low mass free-floating (no HVAC)"
        base.hvac = [HvacSchedule.free_floating()]
        return base

    def case_650ff(self) -> CaseSpec:
        """Case 650FF - Low mass free-floating with night ventilation."""
        base = self.case_600ff()
        base.case_id = "650FF"
        base.description = "Low mass free-floating with night ventilation"
        base.night_ventilation = NightVentilation.case_650()
        return base

    # ===== High Mass Cases (900 series) =====

    def case_900_baseline(self) -> CaseSpec:
        """Case 900 - High mass baseline."""
        return CaseSpec(
            case_id="900",
            description="High mass baseline - concrete construction with south windows",
            geometry=[GeometrySpec(width=8.0, depth=6.0, height=2.7)],
            construction_type=ConstructionType.HIGH_MASS,
            construction=ConstructionSpec(
                wall_u_value=0.688, roof_u_value=0.411, floor_u_value=0.528
            ),
            windows=[[WindowArea(area=12.0, orientation=Orientation.SOUTH)]],
            window_properties=WindowSpec.double_clear_glass(),
            shading=None,
            internal_loads=[
                InternalLoads(
                    total_load=200.0, radiative_fraction=0.6, convective_fraction=0.4
                )
            ],
            hvac=[HvacSchedule.constant(20.0, 27.0)],
            infiltration_ach=0.5,
            opaque_absorptance=0.6,
            num_zones=1,
        )

    def case_910_south_shading(self) -> CaseSpec:
        """Case 910 - High mass with south shading."""
        base = self.case_900_baseline()
        base.case_id = "910"
        base.description = "High mass with south shading (1m overhang)"
        base.shading = ShadingDevice.overhang(1.0, 2.7)
        return base

    def case_920_ew_windows(self) -> CaseSpec:
        """Case 920 - High mass with east/west windows."""
        base = self.case_900_baseline()
        base.case_id = "920"
        base.description = "High mass with east/west windows (6m² each)"
        base.windows = [
            [
                WindowArea(area=6.0, orientation=Orientation.EAST),
                WindowArea(area=6.0, orientation=Orientation.WEST),
            ]
        ]
        return base

    def case_930_ew_shading(self) -> CaseSpec:
        """Case 930 - High mass with east/west shading."""
        base = self.case_920_ew_windows()
        base.case_id = "930"
        base.description = "High mass with east/west shading (overhang + fins)"
        base.shading = ShadingDevice.overhang_and_fins(1.0, 1.0, 2.7)
        return base

    def case_940_setback(self) -> CaseSpec:
        """Case 940 - High mass with thermostat setback."""
        base = self.case_900_baseline()
        base.case_id = "940"
        base.description = "High mass with thermostat setback (overnight)"
        base.hvac = [HvacSchedule.with_setback(20.0, 27.0, 10.0, 23, 7)]
        return base

    def case_950_night_vent(self) -> CaseSpec:
        """Case 950 - High mass with night ventilation."""
        base = self.case_900_baseline()
        base.case_id = "950"
        base.description = "High mass with night ventilation (no heating)"
        base.hvac = [HvacSchedule.with_operating_hours(-100.0, 27.0, 7, 18)]
        base.night_ventilation = NightVentilation.case_650()
        return base

    def case_900ff(self) -> CaseSpec:
        """Case 900FF - High mass free-floating."""
        base = self.case_900_baseline()
        base.case_id = "900FF"
        base.description = "High mass free-floating (no HVAC)"
        base.hvac = [HvacSchedule.free_floating()]
        return base

    def case_950ff(self) -> CaseSpec:
        """Case 950FF - High mass free-floating with night ventilation."""
        base = self.case_900ff()
        base.case_id = "950FF"
        base.description = "High mass free-floating with night ventilation"
        base.night_ventilation = NightVentilation.case_650()
        return base

    # ===== Special Cases =====

    def case_960_sunspace(self) -> CaseSpec:
        """Case 960 - Sunspace (2-zone building)."""
        return CaseSpec(
            case_id="960",
            description="Sunspace - 2-zone building (back-zone + sunspace)",
            geometry=[
                GeometrySpec(width=8.0, depth=6.0, height=2.7),  # Back zone
                GeometrySpec(width=8.0, depth=3.0, height=2.7),  # Sunspace
            ],
            construction_type=ConstructionType.SPECIAL,
            construction=ConstructionSpec(
                wall_u_value=0.514, roof_u_value=0.318, floor_u_value=0.197
            ),
            windows=[
                [
                    WindowArea(area=6.0, orientation=Orientation.SOUTH)
                ],  # Back zone south window
                [
                    WindowArea(area=12.0, orientation=Orientation.SOUTH),
                    WindowArea(area=6.0, orientation=Orientation.EAST),
                    WindowArea(area=6.0, orientation=Orientation.WEST),
                ],  # Sunspace
            ],
            window_properties=WindowSpec.double_clear_glass(),
            shading=None,
            internal_loads=[
                InternalLoads(
                    total_load=200.0, radiative_fraction=0.6, convective_fraction=0.4
                ),
                None,  # Sunspace no loads
            ],
            hvac=[
                HvacSchedule.constant(20.0, 27.0),  # Back zone HVAC
                HvacSchedule.free_floating(),  # Sunspace free-floating
            ],
            infiltration_ach=0.5,
            opaque_absorptance=0.6,
            num_zones=2,
        )

    def case_195_solid_conduction(self) -> CaseSpec:
        """Case 195 - Solid conduction (conduction-only)."""
        return CaseSpec(
            case_id="195",
            description="Solid conduction - no windows, no infiltration, no loads",
            geometry=[GeometrySpec(width=8.0, depth=6.0, height=2.7)],
            construction_type=ConstructionType.SPECIAL,
            construction=ConstructionSpec(
                wall_u_value=0.514, roof_u_value=0.318, floor_u_value=0.197
            ),
            windows=[[]],  # No windows
            window_properties=WindowSpec.double_clear_glass(),
            shading=None,
            internal_loads=[None],  # No loads
            hvac=[HvacSchedule.free_floating()],  # Free-floating
            infiltration_ach=0.0,  # No infiltration
            opaque_absorptance=0.6,
            num_zones=1,
        )


def save_cases_to_json(cases: List[CaseSpec], filepath: str) -> None:
    """
    Save a list of case specifications to a JSON file.

    Args:
        cases: List of CaseSpec objects
        filepath: Output file path
    """
    import json

    def default_serializer(obj):
        """Handle enum serialization."""
        if isinstance(obj, Enum):
            return obj.value
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )

    data = [case.to_dict() for case in cases]
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=default_serializer)
    logger.info(f"Saved {len(cases)} cases to {filepath}")


def load_cases_from_json(filepath: str) -> List[CaseSpec]:
    """
    Load case specifications from a JSON file.

    Args:
        filepath: Input file path

    Returns:
        List of CaseSpec objects
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} cases from {filepath}")
    return data


# CLI Interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASHRAE 140 Case Generator")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output", default="ashrae_140_cases.json", help="Output JSON file"
    )
    parser.add_argument(
        "--u-value-variation",
        type=float,
        default=0.5,
        help="U-value variation fraction (default: 0.5 = ±50%%)",
    )
    parser.add_argument(
        "--setpoint-variation",
        type=float,
        default=5.0,
        help="Setpoint variation in °C (default: 5.0)",
    )
    parser.add_argument(
        "--variations-per-case",
        type=int,
        default=0,
        help="Number of variations per case (0 = no variations, just base cases)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    generator = ASHRAE140CaseGenerator(seed=args.seed)

    if args.variations_per_case > 0:
        cases = generator.generate_all_variations(
            u_value_variation=args.u_value_variation,
            setpoint_variation=args.setpoint_variation,
            num_variations_per_case=args.variations_per_case,
        )
    else:
        cases = generator.generate_all_cases()

    save_cases_to_json(cases, args.output)
    print(f"\nGenerated {len(cases)} ASHRAE 140 cases")
    print(f"Output saved to: {args.output}")
