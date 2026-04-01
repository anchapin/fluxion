#!/usr/bin/env python3
"""
Debug Perez sky model calculation to find the bug.
"""

import csv
import math


def perez_diffuse_tilted(
    dhi,
    dni,
    dni_extra,
    airmass,
    zenith_deg,
    surface_tilt_deg,
    surface_azimuth_deg,
    solar_azimuth_deg,
):
    """Python implementation of Perez model for debugging."""

    if dhi <= 0:
        return 0.0, {}

    zenith_rad = math.radians(zenith_deg)
    surface_tilt = math.radians(surface_tilt_deg)

    kappa = 1.041
    delta = dhi * airmass / dni_extra

    # Sky clearness epsilon
    z_cubed = zenith_rad**3
    numerator = (dhi + dni) / dhi + kappa * z_cubed
    denominator = 1.0 + kappa * z_cubed
    epsilon = numerator / denominator

    # Classify sky clearness
    bounds = [0.0, 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2]
    ebin = 7
    for i, bound in enumerate(bounds):
        if epsilon <= bound:
            ebin = i
            break

    # Perez coefficients (from sky_radiation.rs)
    F1C = [
        [-0.008317, 0.587728, -0.062064],  # Bin 1
        [0.129967, 0.682595, -0.151375],  # Bin 2
        [0.329676, 0.486861, -0.221272],  # Bin 3
        [0.568205, 0.187452, -0.295250],  # Bin 4
        [0.873018, -0.393289, -0.369150],  # Bin 5
        [1.321297, -1.176777, -0.393994],  # Bin 6
        [0.999852, -1.634380, -0.291495],  # Bin 7
        [0.553776, 0.631414, -0.209172],  # Bin 8: clear sky
    ]

    F2C = [
        [0.091000, 0.060000, 0.000000],  # Bin 1
        [0.055000, 0.060000, 0.000000],  # Bin 2
        [0.025000, 0.060000, 0.000000],  # Bin 3
        [-0.015000, 0.060000, 0.000000],  # Bin 4
        [-0.065000, 0.060000, 0.000000],  # Bin 5
        [-0.115000, 0.060000, 0.000000],  # Bin 6
        [-0.165000, 0.060000, 0.000000],  # Bin 7
        [-0.215000, 0.060000, 0.000000],  # Bin 8: clear sky
    ]

    f1c = F1C[ebin]
    f2c = F2C[ebin]

    f1 = max(0.0, f1c[0] + f1c[1] * delta + f1c[2] * zenith_rad)
    f2 = f2c[0] + f2c[1] * delta + f2c[2] * zenith_rad

    # Cosine of incidence angle
    def calc_cos_incidence(tilt, surf_az, zen, sun_az):
        tilt_rad = math.radians(tilt)
        surf_az_rad = math.radians(surf_az)
        zen_rad = math.radians(zen)
        sun_az_rad = math.radians(sun_az)

        return (
            math.sin(tilt_rad)
            * math.sin(surf_az_rad)
            * math.cos(zen_rad)
            * math.sin(sun_az_rad)
            + math.sin(tilt_rad)
            * math.cos(surf_az_rad)
            * math.cos(zen_rad)
            * math.cos(sun_az_rad)
            + math.cos(tilt_rad) * math.sin(zen_rad)
        )

    cos_incidence = calc_cos_incidence(
        surface_tilt_deg, surface_azimuth_deg, zenith_deg, solar_azimuth_deg
    )

    a = max(0.0, cos_incidence)
    b = max(math.cos(zenith_rad), math.cos(math.radians(85.0)))

    term1 = 0.5 * (1.0 - f1) * (1.0 + math.cos(surface_tilt))
    term2 = f1 * a / b if b > 0 else 0
    term3 = f2 * math.sin(surface_tilt)

    total_factor = term1 + term2 + term3
    diffuse_tilted = dhi * max(0.0, total_factor)

    debug_info = {
        "epsilon": epsilon,
        "ebin": ebin,
        "delta": delta,
        "f1": f1,
        "f2": f2,
        "cos_incidence": cos_incidence,
        "a": a,
        "b": b,
        "term1": term1,
        "term2": term2,
        "term3": term3,
        "total_factor": total_factor,
    }

    return diffuse_tilted, debug_info


# Read sample data from simulation
data = []
with open("/tmp/solar_diagnostics/case_920_solar.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(
            {
                "month": int(row["Month"]),
                "day": int(row["Day"]),
                "hour": float(row["HourOfDay"]),
                "orientation": row["Orientation"],
                "dni": float(row["DNI"]),
                "dhi": float(row["DHI"]),
                "diffuse_irr_actual": float(row["DiffuseIrradiance_Wm2"]),
            }
        )

# Test a few sample points
print("=" * 80)
print("PEREZ MODEL DEBUG - Comparing Python vs Rust implementation")
print("=" * 80)

sample_points = [
    r
    for r in data
    if 5 <= r["month"] <= 9
    and 10 <= r["hour"] <= 15
    and r["orientation"] == "West"
    and r["dni"] > 500
][:5]

for i, pt in enumerate(sample_points):
    print(f"\n--- Sample {i + 1} ---")
    print(f"Month: {pt['month']}, Hour: {pt['hour']}, Orientation: {pt['orientation']}")
    print(f"DNI: {pt['dni']:.1f} W/m², DHI: {pt['dhi']:.1f} W/m²")

    # Approximate solar position for West surface at 2pm in summer
    zenith = 25.0  # Altitude = 65°
    solar_azimuth = 240.0  # WSW

    diffuse_calc, debug = perez_diffuse_tilted(
        dhi=pt["dhi"],
        dni=pt["dni"],
        dni_extra=1320.0,
        airmass=1.1,
        zenith_deg=zenith,
        surface_tilt_deg=90.0,
        surface_azimuth_deg=270.0,  # West
        solar_azimuth_deg=solar_azimuth,
    )

    print("\nPerez calculation:")
    print(f"  Sky clearness (epsilon): {debug['epsilon']:.2f} (bin {debug['ebin']})")
    print(f"  Delta: {debug['delta']:.3f}")
    print(f"  F1: {debug['f1']:.3f}")
    print(f"  F2: {debug['f2']:.3f}")
    print(f"  Cos(incidence): {debug['cos_incidence']:.3f}")
    print(f"  Term1 (isotropic): {debug['term1']:.3f}")
    print(f"  Term2 (circumsolar): {debug['term2']:.3f}")
    print(f"  Term3 (horizon): {debug['term3']:.3f}")
    print(f"  Total factor: {debug['total_factor']:.3f}")
    print(f"  Diffuse tilted: {diffuse_calc:.1f} W/m²")
    print(f"  Actual (Rust): {pt['diffuse_irr_actual']:.1f} W/m²")
    print(
        f"  Ratio (calc/actual): {diffuse_calc / pt['diffuse_irr_actual']:.2f}"
        if pt["diffuse_irr_actual"] > 0
        else "  N/A"
    )

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print(
    """
The Perez model calculation shows:
- Term1 (isotropic): ~0.15-0.20 for vertical surface
- Term2 (circumsolar): ~0.1-0.3 depending on incidence angle
- Term3 (horizon): ~-0.2 (negative for clear sky!)

The NEGATIVE term3 is REDUCING the diffuse radiation!

For clear sky (ebin=8), F2C[8] = [-0.215, 0.06, 0.0]
This gives f2 = -0.215 + 0.06 * delta ≈ -0.21

For vertical surface, term3 = f2 * sin(90°) = -0.21

This SUBTRACTS from the total factor!

The issue: F2 coefficients for clear sky should be POSITIVE, not negative.
According to Perez 1990, F2 should enhance horizon brightness for clear skies,
not reduce it.

SUSPECTED BUG: The F2C table may have wrong signs or values.
"""
)
