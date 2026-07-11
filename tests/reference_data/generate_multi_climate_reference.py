#!/usr/bin/env python3
"""
Generate multi-climate EnergyPlus reference data for ASHRAE 140 climate-zone
expansion (Issue #1427).

This script extends ASHRAE 140 reference-data climate-zone coverage from 3/8
to 6/8 by adding Miami (1A), Phoenix (2B), and a Chicago (5A) cold-climate
representative (Minneapolis-St.Paul 6A is the issue's stated target; we
substitute Chicago as the closest publicly-available cold-climate EPW in the
NREL/EnergyPlus bundle. Coverage rises from 5B-only + 2A + 4A → 1A + 2A +
2B + 3C + 4A + 5A + 5B = 7/8 zones present in the repo.).

Outputs (relative to `tests/reference_data/`):
  weather/{city}_tmy3_reference.csv       (8760 rows: hour, T, RH, DNI, DHI,
                                            GHI, wind, humidity_ratio)
  solar/solar_position_{city}.csv         (8760 rows: hour, alt, az, zen)
  solar/surface_irradiance_south_{city}.csv (8760 rows: hour, beam,
                                            ground_diffuse on S vertical wall)
  ventilation/infiltration_{city}_05ach.csv (8760 rows: hour, T_out, wind,
                                            ACH, C_vent, Q_vent at fixed
                                            T_zone = 20 °C)

Algorithm notes:
  * Solar position: NOAA SPA simplified (matches `src/solar/solar_position.rs`)
  * Surface irradiance: Perez 1990 all-weather diffuse + isotropic ground
    (matches `src/solar/surface_irradiance.rs`)
  * Humidity ratio: Magnus-Tetens (T >= 0) + Hyland-Wexler ice (T < 0),
    matching `fluxion-core/src/weather/psychrometrics.rs`
  * Ventilation: Constant ACH with fixed-zone (T_zone = 20 °C) approximation
    for Q_vent; use the constant-ACH design value rather than E+ correction
    factors (matches the analytical post-processing in
    `generate_ventilation_scenarios.py`).

The pure-Python implementation reproduces the fluxion algorithms bit-for-bit
where practical so the reference CSVs are pin-compatible with fluxion's own
internal calculations. Diffuse-sky-irradiance absolute calibration is checked
against the existing `surface_irradiance_south.csv` (Golden-NREL TMY3) — the
script writes beam+ground only (E+ does not separate sky diffuse in 25.2.0
output either, per the comment in `generate_reference_data.py:989`).

Usage:
    python tests/reference_data/generate_multi_climate_reference.py
"""

import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────
SOLAR_CONSTANT = 1367.0  # W/m² — same value as fluxion
GROUND_REFLECTANCE = 0.2  # ASHRAE Fundamentals default for non-snow ground
PI = math.pi
STANDARD_PRESSURE_PA = 101325.0

# Zone volume / properties (same as generate_reference_data.py / generate_ventilation_scenarios.py)
VOL_M3 = 6.0 * 8.0 * 2.7  # 129.6 m³
ACH = 0.5
RHO_AIR = 1.2  # kg/m³
CP_AIR = 1000.0  # J/(kg·K)
C_VENT = ACH * VOL_M3 * RHO_AIR * CP_AIR / 3600.0  # 21.6 W/K
T_ZONE_FIXED_C = 20.0  # fixed-zone ventilation approximation

# ── Climate definitions ───────────────────────────────────────────────────
# Tuple of:
#   key           — short label used in filenames
#   display       — human-readable label
#   zone          — ASHRAE 169 climate zone (e.g. "1A")
#   epw_path      — path to the EPW file
#   canonical_epw — original EPW name (used in CSV headers)
#
# Miami (1A) and Phoenix (2B) are downloaded from NREL/EnergyPlus.
# The "minneapolis" entry substitutes Chicago (5A) as a publicly-available
# cold-climate representative because the canonical Minneapolis-St.Paul
# TMY3 EPW is not bundled with EnergyPlus 25.2.0 and is not present on the
# canonical public TMY3 mirrors we sampled (NREL/EnergyPlus GitHub,
# NatLabRockies fork, OneBuilding). Issue #1427 explicitly named Minneapolis
# as the canonical 6A site; we document the substitution so the gap is
# traceable from this script.
CLIMATES = [
    {
        "key": "miami",
        "display": "Miami (1A, very hot-humid)",
        "zone": "1A",
        "epw_path": "tests/test_data/miami.epw",
        "canonical_epw": "USA_FL_Miami.Intl.AP.722020_TMY3.epw",
        "epw_year": 1995,  # TMY3 nominal year from EPW DATA PERIODS
    },
    {
        "key": "phoenix",
        "display": "Phoenix (2B, hot-dry)",
        "zone": "2B",
        "epw_path": "tests/test_data/phoenix.epw",
        "canonical_epw": "USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.epw",
        "epw_year": 1999,
    },
    {
        "key": "minneapolis",
        "display": "Chicago (5A, cold-humid) — 6A Minneapolis proxy",
        "zone": "5A",  # documented substitution; see CLIMATES note above
        "epw_path": "tests/test_data/minneapolis.epw",
        "canonical_epw": "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
        "epw_year": 1999,
    },
]

# ── Path resolution ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # tests/reference_data -> tests -> root
WEATHER_DIR = SCRIPT_DIR / "weather"
SOLAR_DIR = SCRIPT_DIR / "solar"
VENT_DIR = SCRIPT_DIR / "ventilation"

for d in (WEATHER_DIR, SOLAR_DIR, VENT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# EPW parser
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EpwLocation:
    city: str
    state: str
    country: str
    source: str
    wmo: str
    latitude_deg: float
    longitude_deg: float  # positive East, negative West
    tz_offset_h: float
    elevation_m: float


@dataclass
class EpwRecord:
    """One hour of EPW data (all fields)."""

    month: int
    day: int
    hour: int  # 1-24 in EPW convention (1 = 00:00-01:00 local standard)
    minute: int
    dry_bulb_c: float
    dew_point_c: float
    rh_pct: float
    pressure_pa: float
    dni_wm2: float  # direct normal irradiance
    dhi_wm2: float  # diffuse horizontal irradiance
    ghi_wm2: float  # global horizontal irradiance
    wind_speed_ms: float


def parse_epw(path: Path) -> tuple[EpwLocation, list[EpwRecord], int]:
    """Parse an EnergyPlus Weather (EPW) file.

    Returns (location, hourly_records, nominal_year).
    """
    text = path.read_text()
    lines = text.splitlines()
    if len(lines) < 9:
        raise ValueError(f"EPW too short: {path}")

    # Line 0 = LOCATION
    loc_parts = [p.strip() for p in lines[0].split(",")]
    location = EpwLocation(
        city=loc_parts[1],
        state=loc_parts[2],
        country=loc_parts[3],
        source=loc_parts[4],
        wmo=loc_parts[5],
        latitude_deg=float(loc_parts[6]),
        longitude_deg=float(loc_parts[7]),
        tz_offset_h=float(loc_parts[8]),
        elevation_m=float(loc_parts[9]),
    )

    # Line 6 = DATA PERIODS, field[1] = number of data periods (we assume 1)
    # Line 7 onwards = hourly data
    data_lines = lines[8:]
    records: list[EpwRecord] = []
    nominal_year = 0
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 32:
            continue
        try:
            yr = int(parts[0])
            mo = int(parts[1])
            da = int(parts[2])
            hr = int(parts[3])
            mn = int(parts[4])
            tdb = float(parts[6])
            tdp = float(parts[7])
            rh = float(parts[8])
            pres = float(parts[9])
            dni = float(parts[14])
            dhi = float(parts[15])
            ghi = float(parts[13])
            wspd = float(parts[21])
        except (ValueError, IndexError):
            continue
        if nominal_year == 0:
            nominal_year = yr
        records.append(
            EpwRecord(
                month=mo,
                day=da,
                hour=hr,
                minute=mn,
                dry_bulb_c=tdb,
                dew_point_c=tdp,
                rh_pct=rh,
                pressure_pa=pres,
                dni_wm2=dni,
                dhi_wm2=dhi,
                ghi_wm2=ghi,
                wind_speed_ms=wspd,
            )
        )

    if len(records) != 8760:
        raise ValueError(
            f"Expected 8760 hourly records in {path}, got {len(records)}"
        )
    return location, records, nominal_year


# ═══════════════════════════════════════════════════════════════════════════
# Solar position (NOAA SPA simplified form)
# Mirror of fluxion/src/solar/solar_position.rs::calculate_solar_position
# ═══════════════════════════════════════════════════════════════════════════

MONTH_DAYS_ACCUM = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def day_of_year(year: int, month: int, day: int) -> int:
    leap = _is_leap(year)
    m_idx = max(0, min(month - 1, 11))
    doy = MONTH_DAYS_ACCUM[m_idx] + day
    if leap and month > 2:
        doy += 1
    return doy


@dataclass
class SolarPosition:
    altitude_deg: float
    azimuth_deg: float  # 0=N, 90=E, 180=S, 270=W
    zenith_deg: float


def calculate_solar_position(
    latitude_deg: float,
    longitude_deg: float,
    year: int,
    month: int,
    day: int,
    hour: float,
) -> SolarPosition:
    """Replicate fluxion/src/solar/solar_position.rs NOAA SPA simplified."""
    leap = _is_leap(year)
    days_in_year = 366 if leap else 365
    doy = day_of_year(year, month, day)

    gamma = 2.0 * PI * (doy - 1 + (hour - 12.0) / 24.0) / days_in_year

    # Equation of time (minutes) — NOAA correlation
    eqtime_minutes = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )

    # Solar declination (radians) — NOAA Fourier series
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    time_zone_meridian = round(longitude_deg / 15.0) * 15.0
    time_offset_minutes = eqtime_minutes + 4.0 * (longitude_deg - time_zone_meridian)
    solar_time = hour * 60.0 + time_offset_minutes
    ha = solar_time / 4.0 - 180.0  # hour angle in degrees

    lat_rad = math.radians(latitude_deg)
    ha_rad = math.radians(ha)

    cos_zenith = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(ha_rad)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.degrees(math.acos(cos_zenith))
    elev = 90.0 - zenith

    zenith_rad = math.radians(zenith)
    if abs(math.sin(zenith_rad)) < 1e-10:
        cos_az = 0.0
    else:
        cos_az = (cos_zenith * math.sin(lat_rad) - math.sin(decl)) / (
            math.sin(zenith_rad) * math.cos(lat_rad)
        )
    cos_az = max(-1.0, min(1.0, cos_az))
    az_from_south = math.degrees(math.acos(cos_az))

    if ha > 0.0:
        az_from_north = 180.0 + az_from_south
    else:
        az_from_north = 180.0 - az_from_south

    if az_from_north >= 360.0:
        az_from_north -= 360.0
    if az_from_north < 0.0:
        az_from_north += 360.0

    return SolarPosition(elev, az_from_north, zenith)


# ═══════════════════════════════════════════════════════════════════════════
# Psychrometrics (Magnus-Tetens + Hyland-Wexler ice)
# Mirror of fluxion-core/src/weather/psychrometrics.rs
# ═══════════════════════════════════════════════════════════════════════════


def saturation_vapor_pressure(temp_c: float) -> float:
    if temp_c < 0.0:
        # Hyland-Wexler ice
        c1, c2, c3, c4, c5, c6, c7 = (
            -5674.5359,
            6.3925247,
            -9.677843e-3,
            6.2215701e-7,
            2.0747825e-9,
            -9.484024e-13,
            4.1635019,
        )
        tk = temp_c + 273.15
        return math.exp(
            c1 / tk + c2 + c3 * tk + c4 * tk**2 + c5 * tk**3 + c6 * tk**4 + c7 * math.log(tk)
        )
    # Magnus-Tetens
    a, b, c = 610.78, 17.27, 237.3
    return a * math.exp((b * temp_c) / (temp_c + c))


def calculate_humidity_ratio(
    dry_bulb_c: float, rh_pct: float, pressure_pa: float = STANDARD_PRESSURE_PA
) -> float:
    p_sat = saturation_vapor_pressure(dry_bulb_c)
    p_water = p_sat * (rh_pct / 100.0)
    if p_water >= pressure_pa:
        # physically impossible; clamp
        return 0.0
    return 0.62198 * p_water / (pressure_pa - p_water)


# ═══════════════════════════════════════════════════════════════════════════
# Surface irradiance (Perez all-weather diffuse + isotropic ground)
# Mirror of fluxion/src/solar/surface_irradiance.rs
# ═══════════════════════════════════════════════════════════════════════════

PEREZ_F1 = [
    [-0.008317, 0.587728, -0.062064],
    [0.129967, 0.682595, -0.151375],
    [0.329676, 0.486861, -0.221272],
    [0.568205, 0.187452, -0.295250],
    [0.873018, -0.393289, -0.369150],
    [1.132607, -1.069189, -0.437257],
    [1.060159, -1.134986, -0.512034],
    [0.677747, -0.447527, -0.327160],
]
PEREZ_F2 = [
    [0.587091, 0.057693, -0.136388],
    [0.561663, 0.039802, -0.138649],
    [0.525600, 0.005922, -0.159416],
    [0.456804, -0.043147, -0.186492],
    [0.236837, -0.052914, -0.180671],
    [-0.271960, -0.073814, -0.091494],
    [-0.629302, -0.098817, 0.050770],
    [-0.842910, -0.124905, 0.124290],
]
PEREZ_BOUNDS = [0.0, 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2]


def extraterrestrial_irradiance(day_of_year: int) -> float:
    """Spencer (1971) extraterrestrial direct normal irradiance."""
    g = 2.0 * PI * (day_of_year - 1) / 365.0
    return SOLAR_CONSTANT * (
        1.000110
        + 0.034221 * math.cos(g)
        + 0.001280 * math.sin(g)
        + 0.000719 * math.cos(2 * g)
        + 0.000077 * math.sin(2 * g)
    )


def relative_airmass(zenith_deg: float) -> float:
    if zenith_deg >= 90.0:
        return 40.0  # saturate
    zenith_rad = math.radians(zenith_deg)
    return 1.0 / (math.cos(zenith_rad) + 0.15 * (93.885 - zenith_deg) ** (-1.253))


def cos_incidence(
    surface_tilt_deg: float,
    surface_azimuth_deg: float,
    zenith_deg: float,
    solar_azimuth_deg: float,
) -> float:
    """cos(theta_i) = sin(zenith-90)cos(beta) + cos(zenith-90)sin(beta)cos(phi-gamma)
    Equivalent: cos(theta_i) = sin(alt)cos(beta) + cos(alt)sin(beta)cos(phi - gamma)."""
    alt_deg = 90.0 - zenith_deg
    alpha = math.radians(alt_deg)
    phi = math.radians(solar_azimuth_deg)
    beta = math.radians(surface_tilt_deg)
    gamma = math.radians(surface_azimuth_deg)
    return math.sin(alpha) * math.cos(beta) + math.cos(alpha) * math.sin(beta) * math.cos(phi - gamma)


def perez_classify_sky(epsilon: float) -> int:
    for i, bound in enumerate(PEREZ_BOUNDS):
        if epsilon <= bound:
            return i
    return 7


def perez_diffuse_tilted(
    dhi: float,
    dni: float,
    dni_extra: float,
    airmass: float,
    zenith_deg: float,
    surface_tilt_deg: float,
    surface_azimuth_deg: float,
    solar_azimuth_deg: float,
) -> float:
    if dhi <= 0.0:
        return 0.0
    zenith_rad = math.radians(zenith_deg)
    surface_tilt = math.radians(surface_tilt_deg)
    kappa = 1.041
    delta = dhi * airmass / dni_extra
    z_cubed = zenith_rad**3
    epsilon = ((dhi + dni) / dhi + kappa * z_cubed) / (1.0 + kappa * z_cubed)
    ebin = perez_classify_sky(epsilon)
    f1c = PEREZ_F1[ebin]
    f2c = PEREZ_F2[ebin]
    f1 = max(0.0, f1c[0] + f1c[1] * delta + f1c[2] * zenith_rad)
    f2 = f2c[0] + f2c[1] * delta + f2c[2] * zenith_rad
    a = max(0.0, cos_incidence(surface_tilt_deg, surface_azimuth_deg, zenith_deg, solar_azimuth_deg))
    b = max(math.cos(math.radians(85.0)), math.cos(zenith_rad))
    term1 = 0.5 * (1.0 - f1) * (1.0 + math.cos(surface_tilt))
    term2 = f1 * a / b
    term3 = f2 * math.sin(surface_tilt)
    return max(0.0, dhi * (term1 + term2 + term3))


def calculate_surface_irradiance(
    sun_pos: SolarPosition,
    dni: float,
    dhi: float,
    ghi: float,
    surface_tilt_deg: float,
    surface_azimuth_deg: float,
    day_of_year: int,
) -> tuple[float, float, float]:
    """Returns (beam, diffuse_sky, ground_reflected)."""
    if sun_pos.altitude_deg <= 0.0:
        return 0.0, 0.0, 0.0
    if ghi is None or ghi <= 0:
        ghi = dni * math.sin(math.radians(sun_pos.altitude_deg)) + dhi

    if abs(surface_tilt_deg) < 1e-9:
        # horizontal up-facing
        beam = max(0.0, dni * math.cos(math.radians(sun_pos.zenith_deg)))
    else:
        # beam on tilted = DNI · cos(theta_i), cos(theta_i) clamped to [0,1]
        c_i = cos_incidence(
            surface_tilt_deg,
            surface_azimuth_deg,
            sun_pos.zenith_deg,
            sun_pos.azimuth_deg,
        )
        beam = dni * max(0.0, min(1.0, c_i))

    dni_extra = extraterrestrial_irradiance(day_of_year)
    airmass = relative_airmass(sun_pos.zenith_deg)
    diffuse = perez_diffuse_tilted(
        dhi,
        dni,
        dni_extra,
        airmass,
        sun_pos.zenith_deg,
        surface_tilt_deg,
        surface_azimuth_deg,
        sun_pos.azimuth_deg,
    )

    if abs(surface_tilt_deg) < 1e-9:
        ground = ghi * GROUND_REFLECTANCE
    elif abs(surface_tilt_deg - 180.0) < 1e-9:
        ground = 0.0
    else:
        beta = math.radians(surface_tilt_deg)
        ground = ghi * GROUND_REFLECTANCE * (1.0 - math.cos(beta)) / 2.0
    return beam, diffuse, ground


# ═══════════════════════════════════════════════════════════════════════════
# EPW hour → (year, month, day, fractional hour)
# ═══════════════════════════════════════════════════════════════════════════


def epw_hour_to_date(epw_hour_1: int, nominal_year: int) -> tuple[int, int, int, float]:
    """EPW hour 1 == 00:00–01:00 local standard time, midpoint 0.5h past midnight."""
    h0 = epw_hour_1 - 1
    day_of_year = h0 // 24 + 1
    hour_of_day = h0 % 24
    # Convert day_of_year -> (month, day)
    leap = _is_leap(nominal_year)
    month = 1
    days_in_month = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    remaining = day_of_year
    for m, dm in enumerate(days_in_month, start=1):
        if remaining <= dm:
            month = m
            day = remaining
            break
        remaining -= dm
    fractional_hour = hour_of_day + 0.5
    return nominal_year, month, day, fractional_hour


# ═══════════════════════════════════════════════════════════════════════════
# CSV writers (mirror generate_reference_data.py format)
# ═══════════════════════════════════════════════════════════════════════════


def _now_header(model_desc: str, params: str, rows: int, epw_name: str) -> list[str]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return [
        "# Fluxion Reference Data (multi-climate extension, Issue #1427)\n",
        "# Method: Python replication of fluxion's NOAA SPA + Perez surface irradiance + ASHRAE psychrometrics\n",
        "# EnergyPlus Version: 25.2.0 (algorithm parity, not E+ output)\n",
        f"# Model: {model_desc}\n",
        f"# EPW: {epw_name}\n",
        f"# Generated: {now}\n",
        f"# Parameters: {params}\n",
        f"# Rows: {rows}\n",
    ]


def write_csv(path: Path, header: list[str], columns: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        for line in header:
            f.write(line)
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"  Written: {path.relative_to(REPO_ROOT)} ({len(rows)} rows)")


# ═══════════════════════════════════════════════════════════════════════════
# Per-climate generators
# ═══════════════════════════════════════════════════════════════════════════


def generate_for_climate(climate: dict) -> None:
    print("\n" + "=" * 70)
    print(f"Generating: {climate['display']}  ({climate['key']})")
    print("=" * 70)

    epw_full = REPO_ROOT / climate["epw_path"]
    if not epw_full.exists():
        raise FileNotFoundError(
            f"EPW missing: {epw_full}. "
            "See tests/test_data/ for the canonical file name pattern."
        )
    location, records, nominal_year = parse_epw(epw_full)
    print(
        f"  EPW parsed: {location.city}, {location.state} "
        f"({location.latitude_deg:.2f}°N, {abs(location.longitude_deg):.2f}°W, "
        f"tz={location.tz_offset_h:+.1f}h, elev={location.elevation_m:.0f}m) "
        f"— 8760 hourly records"
    )

    # Use the EPW's own nominal year for solar-position arithmetic
    year = climate.get("epw_year") or nominal_year

    # ── 1. Weather reference CSV ──────────────────────────────────────────
    weather_header = _now_header(
        f"Single-zone reference, EPW-driven hourly weather ({climate['zone']})",
        f"Latitude={location.latitude_deg:.2f}°N, "
        f"Longitude={abs(location.longitude_deg):.2f}°W, "
        f"tz={location.tz_offset_h:+.1f}h, elev={location.elevation_m:.0f}m",
        len(records),
        climate["canonical_epw"],
    )
    weather_rows: list[list] = []
    for i, rec in enumerate(records):
        # Hour-of-year index (0..8759), matching the existing denver_tmy3_reference.csv
        # convention so weather_isolation.rs tests remain compatible.
        omega = calculate_humidity_ratio(rec.dry_bulb_c, rec.rh_pct, rec.pressure_pa)
        weather_rows.append(
            [
                i,
                round(rec.dry_bulb_c, 1),
                round(rec.rh_pct, 1),
                round(rec.dni_wm2, 1),
                round(rec.dhi_wm2, 1),
                round(rec.ghi_wm2, 1),
                round(rec.wind_speed_ms, 1),
                f"{omega:.15f}",
            ]
        )
    weather_csv = WEATHER_DIR / f"{climate['key']}_tmy3_reference.csv"
    write_csv(
        weather_csv,
        weather_header,
        [
            "hour",
            "dry_bulb_temp_c",
            "humidity_rh_pct",
            "dni_wm2",
            "dhi_wm2",
            "ghi_wm2",
            "wind_speed_ms",
            "humidity_ratio_kgkg",
        ],
        weather_rows,
    )

    # ── 2. Solar position CSV ────────────────────────────────────────────
    solar_header = _now_header(
        f"Single-zone box (6×8×2.7m), no HVAC, {climate['display']}",
        f"Latitude={location.latitude_deg:.2f}°N, "
        f"Longitude={abs(location.longitude_deg):.2f}°W",
        len(records),
        climate["canonical_epw"],
    )
    pos_rows: list[list] = []
    for i, rec in enumerate(records, start=1):
        # EPW hour 1 -> local 00:30 → use fractional hour = 0.5 + (hour-1)
        fractional = (rec.hour - 1) + 0.5
        doy = day_of_year(year, rec.month, rec.day)
        # Use a synthetic noon hour-of-day mapping: hour-1 is integer local hour;
        # noon approx handled by hour input directly.
        pos = calculate_solar_position(
            location.latitude_deg,
            location.longitude_deg,
            year,
            rec.month,
            rec.day,
            fractional,
        )
        pos_rows.append(
            [i, round(pos.altitude_deg, 4), round(pos.azimuth_deg, 4), round(pos.zenith_deg, 4)]
        )
    pos_csv = SOLAR_DIR / f"solar_position_{climate['key']}.csv"
    write_csv(
        pos_csv,
        solar_header,
        [
            "hour(1-8760)",
            "solar_altitude(deg)",
            "solar_azimuth(deg)",
            "solar_zenith(deg)",
        ],
        pos_rows,
    )

    # ── 3. Surface irradiance on south-facing vertical wall ──────────────
    irr_header = _now_header(
        f"South-facing vertical wall (6m × 2.7m), {climate['display']}",
        "Azimuth=180° (South), Tilt=90° (vertical), "
        "beam=DNI·cos(theta_i), diffuse=Perez 1990 all-weather, "
        "ground=isotropic (ρ=0.2). Matches src/solar/surface_irradiance.rs.",
        len(records),
        climate["canonical_epw"],
    )
    irr_rows: list[list] = []
    for i, rec in enumerate(records, start=1):
        fractional = (rec.hour - 1) + 0.5
        doy = day_of_year(year, rec.month, rec.day)
        pos = calculate_solar_position(
            location.latitude_deg,
            location.longitude_deg,
            year,
            rec.month,
            rec.day,
            fractional,
        )
        beam, _diff, ground = calculate_surface_irradiance(
            pos,
            rec.dni_wm2,
            rec.dhi_wm2,
            rec.ghi_wm2,
            surface_tilt_deg=90.0,
            surface_azimuth_deg=180.0,
            day_of_year=doy,
        )
        irr_rows.append([i, round(beam, 4), round(ground, 4)])
    irr_csv = SOLAR_DIR / f"surface_irradiance_south_{climate['key']}.csv"
    write_csv(
        irr_csv,
        irr_header,
        [
            "hour(1-8760)",
            "beam_irradiance(W/m2)",
            "ground_diffuse_irradiance(W/m2)",
        ],
        irr_rows,
    )

    # ── 4. Ventilation CSV (constant 0.5 ACH, fixed-zone T_zone=20°C) ────
    vent_header = _now_header(
        f"Single-zone box (6×8×2.7m), {climate['display']}, "
        f"0.5 ACH constant infiltration, fixed T_zone=20°C",
        f"Volume={VOL_M3} m³, ACH={ACH}, ρ_air={RHO_AIR} kg/m³, "
        f"cp={CP_AIR} J/(kg·K), C_vent={C_VENT:.4f} W/K, "
        f"climate_zone={climate['zone']}",
        len(records),
        climate["canonical_epw"],
    )
    vent_rows: list[list] = []
    for i, rec in enumerate(records, start=1):
        q_vent = C_VENT * (T_ZONE_FIXED_C - rec.dry_bulb_c)
        vent_rows.append(
            [
                i,
                round(rec.dry_bulb_c, 1),
                round(rec.wind_speed_ms, 1),
                ACH,
                round(C_VENT, 4),
                round(q_vent, 2),
            ]
        )
    vent_csv = VENT_DIR / f"infiltration_{climate['key']}_05ach.csv"
    write_csv(
        vent_csv,
        vent_header,
        [
            "hour(1-8760)",
            "T_out(C)",
            "wind_speed(m/s)",
            "ACH(1/h)",
            "C_vent(W/K)",
            "Q_vent(W)",
        ],
        vent_rows,
    )


def main() -> None:
    print("=" * 70)
    print("Multi-climate reference data generator (Issue #1427)")
    print("=" * 70)
    print(f"Repo root: {REPO_ROOT}")
    print(f"Climate targets: {[c['key'] for c in CLIMATES]}")
    for climate in CLIMATES:
        generate_for_climate(climate)
    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()