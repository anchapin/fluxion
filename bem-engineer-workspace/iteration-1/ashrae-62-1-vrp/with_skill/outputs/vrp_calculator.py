"""
ASHRAE 62.1-2022 Ventilation Rate Procedure (VRP) Calculator.

Implements the VRP per Section 6.2 of ASHRAE Standard 62.1-2022.
Rates sourced from Table 6.2.2.1 ("Minimum Ventilation Rates in Breathing Zone").

References:
    ASHRAE Standard 62.1-2022, Table 6.2.2.1
    ASHRAE Standard 62.1-2022, Section 6.2.3 (Zone Calculations)

Units convention:
    area    -> sq ft
    V_bz    -> cfm
    V_oz    -> cfm
    rates   -> cfm/person or cfm/sq ft
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ZoneRate:
    """Minimum ventilation rates for a single occupancy category.

    Attributes:
        people_outdoor_rate: Rp — people outdoor air rate (cfm/person).
        area_outdoor_rate: Ra — area outdoor air rate (cfm/sq ft).
        default_density: Default occupant density from Table 6.2.2.1 (people per 1000 sq ft).
    """

    people_outdoor_rate: float
    area_outdoor_rate: float
    default_density: float


TABLE_6_2_2_1: dict[str, ZoneRate] = {
    "conference_meeting": ZoneRate(
        people_outdoor_rate=5.0,
        area_outdoor_rate=0.06,
        default_density=50,
    ),
    "office_general": ZoneRate(
        people_outdoor_rate=5.0,
        area_outdoor_rate=0.06,
        default_density=5,
    ),
    "classroom_lecture": ZoneRate(
        people_outdoor_rate=7.5,
        area_outdoor_rate=0.12,
        default_density=65,
    ),
    "retail_sales": ZoneRate(
        people_outdoor_rate=7.5,
        area_outdoor_rate=0.12,
        default_density=15,
    ),
    "dining_food_bev": ZoneRate(
        people_outdoor_rate=7.5,
        area_outdoor_rate=0.18,
        default_density=100,
    ),
    "corridor": ZoneRate(
        people_outdoor_rate=0.0,
        area_outdoor_rate=0.06,
        default_density=0,
    ),
    "warehouse_storage": ZoneRate(
        people_outdoor_rate=0.0,
        area_outdoor_rate=0.06,
        default_density=0,
    ),
    "healthcare_exam": ZoneRate(
        people_outdoor_rate=7.5,
        area_outdoor_rate=0.12,
        default_density=20,
    ),
    "lodging_guest_room": ZoneRate(
        people_outdoor_rate=5.0,
        area_outdoor_rate=0.06,
        default_density=10,
    ),
    "gym_fitness": ZoneRate(
        people_outdoor_rate=7.5,
        area_outdoor_rate=0.12,
        default_density=30,
    ),
    "library_reading": ZoneRate(
        people_outdoor_rate=5.0,
        area_outdoor_rate=0.12,
        default_density=25,
    ),
    "auditorium_seating": ZoneRate(
        people_outdoor_rate=5.0,
        area_outdoor_rate=0.06,
        default_density=150,
    ),
    "restroom": ZoneRate(
        people_outdoor_rate=5.0,
        area_outdoor_rate=0.06,
        default_density=10,
    ),
    "kitchen_commercial": ZoneRate(
        people_outdoor_rate=7.5,
        area_outdoor_rate=0.18,
        default_density=20,
    ),
}

DEFAULT_EZ = 1.0
DEFAULT_DS = 1.0
DEFAULT_EP = 1.0


def population_from_density(zone_area_sqft: float, density_per_1000_sqft: float) -> int:
    """Compute zone population using default occupant density from Table 6.2.2.1.

    Pz = (Az / 1000) * default_density, rounded to nearest whole person.
    """
    return max(1, round(zone_area_sqft * density_per_1000_sqft / 1000))


def v_bz(
    zone_area: float,
    zone_population: int,
    rp: float,
    ra: float,
) -> float:
    """Calculate the breathing zone outdoor airflow.

    Per ASHRAE 62.1-2022, Eq. 6.2.3-1:
        V_bz = Rp * Pz + Ra * Az

    Args:
        zone_area: Az — zone floor area (sq ft).
        zone_population: Pz — zone population (people).
        rp: People outdoor air rate from Table 6.2.2.1 (cfm/person).
        ra: Area outdoor air rate from Table 6.2.2.1 (cfm/sq ft).

    Returns:
        V_bz in cfm.
    """
    return rp * zone_population + ra * zone_area


def v_oz(
    vbz: float,
    ez: float = DEFAULT_EZ,
    ds: float = DEFAULT_DS,
    ep: float = DEFAULT_EP,
) -> float:
    """Calculate the zone outdoor airflow.

    Per ASHRAE 62.1-2022, Eq. 6.2.3-2:
        V_oz = V_bz / (Ez * Ds * Ep)

    For a single-zone system or when system-level diversity and plenum
    efficiency are not considered, Ds and Ep default to 1.0.

    Args:
        vbz: Breathing zone outdoor airflow (cfm).
        ez: Zone air distribution effectiveness (Table 6.2.2.2). Default 1.0.
        ds: System diversity factor. Default 1.0 (no diversity).
        ep: Plenum/transfer air effectiveness. Default 1.0.

    Returns:
        V_oz in cfm.
    """
    denominator = ez * ds * ep
    if denominator <= 0:
        raise ValueError(
            f"Denominator factors must be positive: Ez={ez}, Ds={ds}, Ep={ep}"
        )
    return vbz / denominator


def compute_zone_ventilation(
    zone_area: float,
    zone_category: str,
    zone_population: int | None = None,
    ez: float = DEFAULT_EZ,
    ds: float = DEFAULT_DS,
    ep: float = DEFAULT_EP,
) -> dict[str, float | int]:
    """Compute V_bz and V_oz for a zone per ASHRAE 62.1-2022 VRP.

    If zone_population is None, the default occupant density from
    Table 6.2.2.1 is used to derive Pz.

    Args:
        zone_area: Az — zone floor area (sq ft).
        zone_category: Key into TABLE_6_2_2_1 (e.g., "conference_meeting").
        zone_population: Pz — design zone population. None → use default density.
        ez: Zone air distribution effectiveness. Default 1.0.
        ds: System diversity factor. Default 1.0.
        ep: Plenum/transfer air effectiveness. Default 1.0.

    Returns:
        Dictionary with all intermediate and final values.

    Raises:
        KeyError: If zone_category is not in Table 6.2.2.1 lookup.
    """
    if zone_category not in TABLE_6_2_2_1:
        raise KeyError(
            f"Unknown zone category '{zone_category}'. "
            f"Available: {sorted(TABLE_6_2_2_1.keys())}"
        )

    rate = TABLE_6_2_2_1[zone_category]

    if zone_population is None:
        zone_population = population_from_density(zone_area, rate.default_density)

    vbz_val = v_bz(
        zone_area, zone_population, rate.people_outdoor_rate, rate.area_outdoor_rate
    )
    voz_val = v_oz(vbz_val, ez, ds, ep)

    return {
        "zone_category": zone_category,
        "Az_sqft": zone_area,
        "Pz": zone_population,
        "Rp": rate.people_outdoor_rate,
        "Ra": rate.area_outdoor_rate,
        "density_per_1000sqft": rate.default_density,
        "V_bz_cfm": vbz_val,
        "Ez": ez,
        "Ds": ds,
        "Ep": ep,
        "V_oz_cfm": voz_val,
    }


if __name__ == "__main__":
    result = compute_zone_ventilation(
        zone_area=3000,
        zone_category="conference_meeting",
    )

    print("ASHRAE 62.1-2022 VRP Calculation")
    print("=" * 50)
    print(f"Zone Category      : {result['zone_category']}")
    print(f"Area (Az)          : {result['Az_sqft']:.0f} sq ft")
    print(f"Default Density    : {result['density_per_1000sqft']} per 1000 sq ft")
    print(f"Population (Pz)    : {result['Pz']}")
    print(f"Rp (people rate)   : {result['Rp']:.1f} cfm/person")
    print(f"Ra (area rate)     : {result['Ra']:.2f} cfm/sq ft")
    print()
    print(f"V_bz = Rp*Pz + Ra*Az")
    print(
        f"V_bz = {result['Rp']:.1f}*{result['Pz']} + {result['Ra']:.2f}*{result['Az_sqft']:.0f}"
    )
    print(
        f"V_bz = {result['Rp'] * result['Pz']:.1f} + {result['Ra'] * result['Az_sqft']:.1f}"
    )
    print(f"V_bz = {result['V_bz_cfm']:.1f} cfm")
    print()
    print(f"Ez = {result['Ez']:.1f} (ceiling supply, well-mixed)")
    print(f"V_oz = V_bz / Ez = {result['V_bz_cfm']:.1f} / {result['Ez']:.1f}")
    print(f"V_oz = {result['V_oz_cfm']:.1f} cfm")
