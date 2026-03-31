#!/usr/bin/env python3
"""
Replicate the exact calculation from phase30_solar_diagnostics.rs to find the bug.
"""

import csv
import math


def calculate_solar_position(latitude_deg, month, day, hour):
    """Replicate Rust solar position calculation."""
    # Day of year (simplified, same as Rust)
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_of_year = sum(days_per_month[: month - 1]) + day

    # Leap year check (2024 is leap year)
    is_leap = True
    if is_leap and month > 2:
        day_of_year += 1

    # NOAA algorithm (from solar.rs)
    days_in_year = 366 if is_leap else 365
    gamma = 2.0 * math.pi * (day_of_year - 1 + (hour - 12.0) / 24.0) / days_in_year

    decl_rad = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    # Simplified hour angle
    ha = (hour - 12.0) * 15.0
    lat_rad = math.radians(latitude_deg)
    ha_rad = math.radians(ha)

    cos_zenith = math.sin(lat_rad) * math.sin(decl_rad) + math.cos(lat_rad) * math.cos(
        decl_rad
    ) * math.cos(ha_rad)
    cos_zenith = max(-1, min(1, cos_zenith))
    zenith = math.degrees(math.acos(cos_zenith))
    altitude = 90.0 - zenith

    # Solar azimuth
    zenith_rad = math.radians(zenith)
    sin_az = -math.cos(decl_rad) * math.sin(lat_rad) * math.sin(ha_rad)
    cos_az = -math.sin(lat_rad) * math.cos(zenith_rad) - math.sin(decl_rad) * math.cos(
        lat_rad
    ) * math.sin(zenith_rad)

    az = math.degrees(math.atan2(sin_az, cos_az))
    if az < 0:
        az += 360.0

    return altitude, zenith, az


def calculate_incidence_cosine(
    surface_tilt_deg, surface_azimuth_deg, altitude_deg, sun_azimuth_deg
):
    """Replicate Rust incidence cosine calculation."""
    alt = math.radians(altitude_deg)
    az = math.radians(sun_azimuth_deg)
    beta = math.radians(surface_tilt_deg)
    gamma = math.radians(surface_azimuth_deg)

    cos_theta = (
        math.sin(beta) * math.sin(gamma) * math.cos(alt) * math.sin(az)
        + math.sin(beta) * math.cos(gamma) * math.cos(alt) * math.cos(az)
        + math.cos(beta) * math.sin(alt)
    )

    return max(0, cos_theta)


def perez_diffuse(dhi, dni, dni_extra, airmass, zenith_deg, tilt_deg, surf_az, sun_az):
    """Replicate Rust Perez calculation."""
    if dhi <= 0:
        return 0.0

    zenith_rad = math.radians(zenith_deg)
    tilt = math.radians(tilt_deg)
    kappa = 1.041
    delta = dhi * airmass / dni_extra

    z_cubed = zenith_rad**3
    epsilon = ((dhi + dni) / dhi + kappa * z_cubed) / (1.0 + kappa * z_cubed)

    # Classify sky clearness
    bounds = [0.0, 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2]
    ebin = 7
    for i, bound in enumerate(bounds):
        if epsilon <= bound:
            ebin = i
            break

    # F1 coefficients (bin 7/8 for clear sky)
    f1c = (
        [0.999852, -1.634380, -0.291495]
        if ebin == 6
        else [0.553776, 0.631414, -0.209172]
    )
    f2c = (
        [-0.165000, 0.060000, 0.000000]
        if ebin == 6
        else [-0.215000, 0.060000, 0.000000]
    )

    f1 = max(0.0, f1c[0] + f1c[1] * delta + f1c[2] * zenith_rad)
    f2 = f2c[0] + f2c[1] * delta + f2c[2] * zenith_rad

    # Cosine of incidence
    cos_inc = calculate_incidence_cosine(tilt_deg, surf_az, 90 - zenith_deg, sun_az)

    a = max(0.0, cos_inc)
    b = max(math.cos(zenith_rad), math.cos(math.radians(85.0)))

    term1 = 0.5 * (1.0 - f1) * (1.0 + math.cos(tilt))
    term2 = f1 * a / b if b > 0 else 0
    term3 = f2 * math.sin(tilt)

    return dhi * max(0.0, term1 + term2 + term3)


# Read sample from diagnostic data
with open("/tmp/solar_diagnostics/case_920_solar.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if (
            5 <= int(row["Month"]) <= 9
            and 13 <= float(row["HourOfDay"]) <= 14
            and row["Orientation"] == "West"
            and float(row["DNI"]) > 800
        ):
            month = int(row["Month"])
            day = int(row["Day"])
            hour = float(row["HourOfDay"])
            dni = float(row["DNI"])
            dhi = float(row["DHI"])
            ghi = float(row["GHI"])
            diffuse_actual = float(row["DiffuseIrradiance_Wm2"])

            print(f"Sample: Month={month}, Day={day}, Hour={hour}")
            print(f"DNI={dni:.1f}, DHI={dhi:.1f}, GHI={ghi:.1f}")
            print(f"Diffuse actual: {diffuse_actual:.1f} W/m²\n")

            # Calculate solar position
            alt, zen, sun_az = calculate_solar_position(39.7, month, day, hour)
            print(f"Solar position: Alt={alt:.1f}°, Zen={zen:.1f}°, Az={sun_az:.1f}°")

            # Calculate incidence for West surface
            cos_inc = calculate_incidence_cosine(90.0, 270.0, alt, sun_az)
            inc_angle = math.degrees(math.acos(max(0, cos_inc)))
            print(f"Incidence: cos={cos_inc:.3f}, angle={inc_angle:.1f}°\n")

            # Calculate Perez diffuse
            dni_extra = 1366.1 * (
                1 + 0.033 * math.cos(2 * math.pi * (month * 30 + day) / 365)
            )
            airmass = (
                1.0 / (math.cos(math.radians(zen)) + 0.50572 * (90 - zen) ** (-1.6364))
                if zen < 90
                else 10
            )

            diffuse_calc = perez_diffuse(
                dhi, dni, dni_extra, airmass, zen, 90.0, 270.0, sun_az
            )

            print(f"Perez calculation:")
            print(f"  DNI_extra: {dni_extra:.1f}")
            print(f"  Airmass: {airmass:.2f}")
            print(f"  Diffuse calculated: {diffuse_calc:.1f} W/m²")
            print(f"  Tilt factor: {diffuse_calc / dhi:.3f}")
            print(f"  Diffuse actual: {diffuse_actual:.1f} W/m²")
            print(
                f"  Ratio (actual/calc): {diffuse_actual / diffuse_calc:.2f}"
                if diffuse_calc > 0
                else "  N/A"
            )

            break
