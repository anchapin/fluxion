"""
ASHRAE 62.1-2022 Ventilation Rate Procedure (VRP) Calculator

Computes V_oz (zone outdoor airflow) for any zone given area, population,
and zone category per Table 6.4 rates.
"""

from dataclasses import dataclass

# ASHRAE 62.1-2022 Table 6.4 — selected categories
TABLE_6_4 = {
    "conference_room": {
        "name": "Conference Rooms",
        "occupant_density_per_1000ft2": 50,
        "Rp_cfm_per_person": 5.0,
        "Ra_cfm_per_ft2": 0.06,
    },
    "office_general": {
        "name": "Office Space — General",
        "occupant_density_per_1000ft2": 5,
        "Rp_cfm_per_person": 5.0,
        "Ra_cfm_per_ft2": 0.06,
    },
    "classroom_ages_5_8": {
        "name": "Classroom (Ages 5-8)",
        "occupant_density_per_1000ft2": 25,
        "Rp_cfm_per_person": 10.0,
        "Ra_cfm_per_ft2": 0.12,
    },
    "classroom_ages_9_plus": {
        "name": "Classroom (Ages 9+)",
        "occupant_density_per_1000ft2": 35,
        "Rp_cfm_per_person": 10.0,
        "Ra_cfm_per_ft2": 0.12,
    },
    "restaurant_dining": {
        "name": "Restaurant Dining Area",
        "occupant_density_per_1000ft2": 70,
        "Rp_cfm_per_person": 7.5,
        "Ra_cfm_per_ft2": 0.18,
    },
    "retail_sales": {
        "name": "Retail Sales",
        "occupant_density_per_1000ft2": 15,
        "Rp_cfm_per_person": 7.5,
        "Ra_cfm_per_ft2": 0.12,
    },
    "corridor": {
        "name": "Corridors",
        "occupant_density_per_1000ft2": 0,
        "Rp_cfm_per_person": 0.0,
        "Ra_cfm_per_ft2": 0.06,
    },
    "warehouse": {
        "name": "Warehouse Storage",
        "occupant_density_per_1000ft2": 0,
        "Rp_cfm_per_person": 0.0,
        "Ra_cfm_per_ft2": 0.06,
    },
}

# ASHRAE 62.1-2022 Table 6.3 — Zone Air Distribution Effectiveness (Ez)
EZ_VALUES = {
    "ceiling_supply_cool_air_ceiling_return": 1.0,
    "ceiling_supply_warm_air_ceiling_return": 0.8,
    "ceiling_supply_cool_air_floor_return": 1.0,
    "floor_supply_warm_air_floor_return": 1.0,
    "floor_supply_cool_air_floor_return": 0.8,
    "floor_supply_cool_air_ceiling_return": 1.2,
    "makeup_air_unit_hood_exhaust": 0.7,
}


@dataclass
class VRPResult:
    zone_category: str
    area_ft2: float
    population: int
    Rp: float
    Ra: float
    Vbz: float
    Ez: float
    V_oz: float
    used_default_density: bool


def default_population(area_ft2: float, category: str) -> int:
    """Compute default Pz from Table 6.4 occupant density."""
    density = TABLE_6_4[category]["occupant_density_per_1000ft2"]
    return max(0, round(area_ft2 * density / 1000.0))


def compute_voz(
    area_ft2: float,
    category: str,
    population: int | None = None,
    Ez: float = 1.0,
) -> VRPResult:
    """
    Compute zone outdoor airflow V_oz per ASHRAE 62.1-2022 VRP.

    Parameters
    ----------
    area_ft2 : float
        Zone floor area in square feet (Az).
    category : str
        Key into TABLE_6_4 (e.g. "conference_room").
    population : int | None
        Zone population (Pz).  If None, uses Table 6.4 default density.
    Ez : float
        Zone air distribution effectiveness from Table 6.3.
        Default 1.0 (ceiling supply cool air, ceiling return).

    Returns
    -------
    VRPResult with full breakdown.

    Formulas (ASHRAE 62.1-2022 §6.4):
        Vbz = Rp × Pz + Ra × Az          (Eq. 6.4-1, breathing-zone OA rate)
        V_oz = Vbz / Ez                   (Eq. 6.4-2, zone OA rate)
    """
    entry = TABLE_6_4[category]
    Rp = entry["Rp_cfm_per_person"]
    Ra = entry["Ra_cfm_per_ft2"]

    used_default = population is None
    if used_default:
        population = default_population(area_ft2, category)

    Vbz = Rp * population + Ra * area_ft2
    V_oz = Vbz / Ez

    return VRPResult(
        zone_category=entry["name"],
        area_ft2=area_ft2,
        population=population,
        Rp=Rp,
        Ra=Ra,
        Vbz=round(Vbz, 1),
        Ez=Ez,
        V_oz=round(V_oz, 1),
        used_default_density=used_default,
    )


if __name__ == "__main__":
    r = compute_voz(area_ft2=3000, category="conference_room")
    print(f"Zone category     : {r.zone_category}")
    print(f"Area (Az)         : {r.area_ft2:,.0f} ft²")
    print(
        f"Population (Pz)   : {r.population} {'(default density)' if r.used_default_density else ''}"
    )
    print(f"Rp                : {r.Rp} cfm/person")
    print(f"Ra                : {r.Ra} cfm/ft²")
    print(f"Vbz (Eq. 6.4-1)   : {r.Vbz:,.1f} cfm")
    print(f"Ez (Table 6.3)    : {r.Ez}")
    print(f"V_oz (Eq. 6.4-2)  : {r.V_oz:,.1f} cfm")
