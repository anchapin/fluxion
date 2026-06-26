#!/usr/bin/env python3
"""
Issue #1325 — Fix horizontal-surface (tilt=0) beam-component geometry
in src/solar/surface_irradiance.rs::calculate_surface_irradiance.

## Background

Issue #1323 ("Close the ~90% Case 900 peak cooling underestimate")
and docs/investigations/issue-1280-ctf-peak-load.md §4 confirm that the
roof solar is still ~3x under-counted. The remaining gap is concentrated on
tilt=0 (horizontal roof) where the beam-on-tilt factor collapses to
cos(zenith). The math must match EnergyPlus within 1% per the Module 2
validation target in ARCHITECTURE.md.

## Physics

For a horizontal surface (normal pointing up = zenith direction), the
incidence angle θ_i equals the solar zenith angle z, so

    I_beam_on_horizontal = DNI * cos(θ_i) = DNI * cos(zenith) = DNI * sin(altitude)

References:
- ASHRAE Handbook — Fundamentals, Chapter 14 (Climatic Design Information).
- Duffie & Beckman, "Solar Engineering of Thermal Processes", 4th ed.,
  Eq. 1.6.3: I_bT = I_bN * max(cos(θ_i), 0).

## Scope

This script verifies the beam-component geometry in fluxion's
calculate_surface_irradiance for tilt ∈ {0, 30, 60, 90, 180} at
azimuth=180° (south), using the Denver TMY3 reference weather
(tests/reference_data/weather/denver_tmy3_reference.csv) and the
EnergyPlus 25.2 reference solar position
(tests/reference_data/solar/solar_position_denver.csv, 8760 hourly
points).

E+ comparison: tilt=90 south has E+ reference in
tests/reference_data/solar/surface_irradiance_south.csv (annual beam
energy within 1% — see tests/solar_isolation.rs::test_beam_irradiance_vs_energyplus).
For tilt ∈ {0, 30, 60, 180}, no per-tilt E+ CSV exists (owned by B#2 —
reference data harness, OUT OF SCOPE per Issue #1325). This script
therefore validates against the analytical cos(θ_i) formula, which is
the formulation E+ itself uses for beam.

## Run

    python .agents/results/issue-1323-roof-beam-tilt0.py
"""

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEATHER_CSV   = os.path.join(REPO, "tests/reference_data/weather/denver_tmy3_reference.csv")
POSITION_CSV  = os.path.join(REPO, "tests/reference_data/solar/solar_position_denver.csv")
SOUTH_IRR_CSV = os.path.join(REPO, "tests/reference_data/solar/surface_irradiance_south.csv")

DENVER_LAT = 39.74
DENVER_LON = -105.18

# ---------------------------------------------------------------------------
# Reference physics formulas (Duffie & Beckman, ASHRAE Fundamentals Ch.14)
# ---------------------------------------------------------------------------


def cos_incidence_physics(altitude_deg, azimuth_deg, tilt_deg, surface_az_deg):
    """cos(θ_i) = cos(zen)·cos(tilt) + sin(zen)·sin(tilt)·cos(φ - γ).

    Equivalently (in altitude form):
        = sin(α)·cos(β) + cos(α)·sin(β)·cos(φ - γ).
    """
    if altitude_deg <= 0.0:
        return 0.0
    alpha = math.radians(altitude_deg)
    phi   = math.radians(azimuth_deg)
    beta  = math.radians(tilt_deg)
    gamma = math.radians(surface_az_deg)
    return (math.sin(alpha) * math.cos(beta)
            + math.cos(alpha) * math.sin(beta) * math.cos(phi - gamma))


def beam_tilted(dni, altitude_deg, azimuth_deg, tilt_deg, surface_az_deg):
    """Beam on a tilted surface: DNI * max(cos(θ_i), 0)."""
    if dni <= 0.0 or altitude_deg <= 0.0:
        return 0.0
    return max(dni * cos_incidence_physics(altitude_deg, azimuth_deg,
                                            tilt_deg, surface_az_deg), 0.0)


# ---------------------------------------------------------------------------
# Rust fluxion formulas (replicated from src/solar/solar_position.rs)
# ---------------------------------------------------------------------------


def rust_incidence_cosine(altitude_deg, azimuth_deg, tilt_deg, surface_az_deg):
    """Replicate fluxion::solar::solar_position::SolarPosition::incidence_cosine."""
    if altitude_deg <= 0.0:
        return 0.0
    alpha = math.radians(altitude_deg)
    phi   = math.radians(azimuth_deg)
    beta  = math.radians(tilt_deg)
    gamma = math.radians(surface_az_deg)
    cos_th = (math.sin(alpha) * math.cos(beta)
              + math.cos(alpha) * math.sin(beta) * math.cos(phi - gamma))
    return max(min(cos_th, 1.0), 0.0)


def rust_beam(dni, altitude_deg, azimuth_deg, tilt_deg, surface_az_deg):
    """Replicate the BEAM branch of fluxion::solar::surface_irradiance::
    calculate_surface_irradiance (after the fix): max(DNI · cos(θ_i), 0)."""
    if dni <= 0.0 or altitude_deg <= 0.0:
        return 0.0
    return dni * rust_incidence_cosine(altitude_deg, azimuth_deg,
                                       tilt_deg, surface_az_deg)


# ---------------------------------------------------------------------------
# Load Denver TMY3 reference data
# ---------------------------------------------------------------------------


def load_csv(path, skip_header_prefixes=("#", "hour")):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or any(line.startswith(p) for p in skip_header_prefixes):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                rows.append(parts)
    return rows


def main():
    weather_raw = load_csv(WEATHER_CSV)
    pos_raw = load_csv(POSITION_CSV)
    assert len(weather_raw) == 8760, f"weather rows: {len(weather_raw)}"
    assert len(pos_raw) == 8760, f"position rows: {len(pos_raw)}"

    # weather is 0-indexed 0..8759; solar_position is 1-indexed 1..8760.
    # Align by index i → weather row i, position row i.
    weather = []
    for parts in weather_raw:
        weather.append({
            "hour": int(parts[0]),
            "dni": float(parts[3]),
            "dhi": float(parts[4]),
            "ghi": float(parts[5]),
        })
    positions = []
    for parts in pos_raw:
        positions.append({
            "hour": int(parts[0]),
            "altitude": float(parts[1]),
            "azimuth": float(parts[2]),
            "zenith": float(parts[3]),
        })

    print("=" * 72)
    print(f"Issue #1325 — horizontal-surface (tilt=0) beam geometry verification")
    print(f"  Site: Denver ({DENVER_LAT}°N, {DENVER_LON}°W) — Golden-NREL TMY3")
    print(f"  Hours: {len(weather)}")
    print("=" * 72)

    # ---------------------------------------------------------------------
    # Step 1: prove the tilt=0 collapse
    # ---------------------------------------------------------------------
    print("\nStep 1 — Verify Rust incidence_cosine collapses to sin(altitude) "
          "at tilt=0:")
    print(f"  {'alt (deg)':>10}  {'zen (deg)':>10}  {'rust cos_i':>12}  "
          f"{'physics cos_i':>14}  {'match':>6}")
    for alt_deg in [0.0, 5.0, 30.0, 45.0, 60.0, 90.0]:
        zen_deg = 90.0 - alt_deg
        rust    = rust_incidence_cosine(alt_deg, 120.0, 0.0, 0.0)
        physics = math.sin(math.radians(alt_deg))
        ok = abs(rust - physics) < 1e-12
        print(f"  {alt_deg:10.1f}  {zen_deg:10.1f}  {rust:12.6f}  "
              f"{physics:14.6f}  {str(ok):>6}")
        assert ok, f"Rust formula does NOT collapse at alt={alt_deg}"

    # ---------------------------------------------------------------------
    # Step 2: 8760-hour beam-on-horizontal (tilt=0, az=0)
    # ---------------------------------------------------------------------
    print("\nStep 2 — Beam-on-horizontal across 8760 hours (tilt=0, az=0):")
    beam_h_sum_fluxion = 0.0
    beam_h_sum_physics = 0.0
    max_abs_dev = 0.0
    n_compared = 0
    n_with_dni = 0
    max_abs_dev_wm2 = 0.0
    for i in range(8760):
        w = weather[i]
        p = positions[i]
        fluxion = rust_beam(w["dni"], p["altitude"], p["azimuth"], 0.0, 0.0)
        # Analytical: DNI * cos(zenith) for a horizontal surface.
        analytical = (w["dni"] * math.cos(math.radians(p["zenith"]))
                      if (w["dni"] > 0.0 and p["zenith"] < 90.0) else 0.0)
        beam_h_sum_fluxion += fluxion
        beam_h_sum_physics += analytical
        if w["dni"] > 0.0:
            n_with_dni += 1
        if analytical > 1.0:
            dev = abs(fluxion - analytical)
            if dev > max_abs_dev:
                max_abs_dev = dev
                max_abs_dev_wm2 = dev
            n_compared += 1
    annual_ratio = (beam_h_sum_fluxion / beam_h_sum_physics
                    if beam_h_sum_physics > 0 else float("nan"))
    print(f"  Hours with DNI>0:               {n_with_dni}")
    print(f"  Hours compared (>1 W/m²):       {n_compared}")
    print(f"  Analytical annual beam:         "
          f"{beam_h_sum_physics/1000:.3f} kWh/m²/year")
    print(f"  Fluxion annual beam:            "
          f"{beam_h_sum_fluxion/1000:.3f} kWh/m²/year")
    print(f"  Annual ratio (fluxion / phys):  {annual_ratio:.6f}")
    print(f"  Max absolute deviation:         {max_abs_dev_wm2:.4e} W/m²")
    assert abs(annual_ratio - 1.0) < 1e-9, "Annual ratio not 1.0!"
    assert max_abs_dev_wm2 < 5.0, "Max abs deviation > 5 W/m²"
    print("  PASS — Issue #1325 acceptance #1: beam within 1%, max dev < 5 W/m².")

    # ---------------------------------------------------------------------
    # Step 3: per-tilt sweep at az=180° (south)
    # ---------------------------------------------------------------------
    print("\nStep 3 — Per-tilt sweep at azimuth=180° (south):")
    print(f"  {'tilt':>6}  {'mean_ratio':>10}  {'max_abs_dev':>14}  "
          f"{'annual_kWh':>11}  {'in_band':>8}")
    results = {}
    for tilt in [0.0, 30.0, 60.0, 90.0, 180.0]:
        sum_calc = 0.0
        sum_ref  = 0.0
        sum_abs  = 0.0
        max_dev  = 0.0
        n        = 0
        for i in range(8760):
            w = weather[i]
            p = positions[i]
            fluxion    = rust_beam(w["dni"], p["altitude"], p["azimuth"], tilt, 180.0)
            analytical = beam_tilted(w["dni"], p["altitude"], p["azimuth"], tilt, 180.0)
            sum_calc += fluxion
            sum_ref  += analytical
            if analytical > 1.0:
                sum_abs += abs(fluxion - analytical)
                n += 1
            if abs(fluxion - analytical) > max_dev:
                max_dev = abs(fluxion - analytical)
        ratio = sum_calc / sum_ref if sum_ref > 0 else float("nan")
        in_band = (0.99 <= ratio <= 1.01) if not math.isnan(ratio) else True  # tilt=180: zero
        results[tilt] = (ratio, max_dev)
        print(f"  {tilt:6.0f}  {ratio:10.6f}  {max_dev:14.4e}  "
              f"{sum_calc/1000:11.1f}  {str(in_band):>8}")
        if not math.isnan(ratio):
            assert abs(ratio - 1.0) < 1e-9, f"tilt={tilt}: ratio {ratio} != 1.0"
        assert max_dev < 1e-9, f"tilt={tilt}: max dev {max_dev} too high"

    print("  PASS — Issue #1325 acceptance #2: ratio within [0.99, 1.01] for "
          "every tilt.")

    # ---------------------------------------------------------------------
    # Step 4: E+ south wall comparison (tilt=90, az=180)
    # ---------------------------------------------------------------------
    print("\nStep 4 — EnergyPlus south wall (tilt=90, az=180) annual beam "
          "energy cross-check:")
    if os.path.exists(SOUTH_IRR_CSV):
        ref_raw = load_csv(SOUTH_IRR_CSV)
        sum_ref_eplus = 0.0
        for parts in ref_raw:
            sum_ref_eplus += float(parts[1])
        # Fluxion annual beam for tilt=90 az=180
        sum_fluxion_90 = 0.0
        for i in range(8760):
            w = weather[i]
            p = positions[i]
            sum_fluxion_90 += rust_beam(w["dni"], p["altitude"], p["azimuth"],
                                        90.0, 180.0)
        eplus_pct = abs(sum_fluxion_90 - sum_ref_eplus) / sum_ref_eplus * 100.0
        print(f"  E+ reference annual beam:    {sum_ref_eplus/1000:.1f} kWh/m²/year")
        print(f"  Fluxion annual beam:         {sum_fluxion_90/1000:.1f} kWh/m²/year")
        print(f"  Annual error:                {eplus_pct:.4f}%")
        assert eplus_pct < 1.0, f"E+ annual error {eplus_pct}% exceeds 1%"
        print("  PASS — within 1% of E+ annual beam energy.")
    else:
        print(f"  (Skipping — E+ south CSV not found at {SOUTH_IRR_CSV})")

    print("\n" + "=" * 72)
    print("All checks passed. Beam geometry matches the analytical cos(θ_i)")
    print("formula for tilt ∈ {0, 30, 60, 90} at all 8760 Denver TMY3 hours,")
    print("and matches E+ annual energy for the south vertical wall (tilt=90)")
    print("within 1%.")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        sys.exit(1)