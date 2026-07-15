#!/usr/bin/env python3
"""
Python verification script for issue #1606 — GaugeSolver diurnal parity vs FiveR1CSolver baseline.

This script simulates both solvers through a 24-hour synthetic Case 900 diurnal
forcing and verifies the acceptance criteria:

1. GaugeSolver flux within ±10% of FiveR1C at every hour
2. Both solvers peak at hour 12, trough at hour 4-5
3. Nighttime negative, daytime positive response (bipolar)
4. Amplitude ≥80 W/m²

Physics notes:
- GaugeSolver: steady-state flux q = (T_eff - T_int) / R_wall where
  T_eff = T_outdoor + solar / h_ext. No thermal mass evolution.
- FiveR1C: lumped-capacitance model with thermal mass that evolves toward
  T_eff over time (τ ≈ 25.6 h for Case 900 200mm concrete).
  On first step, seeds T_mass = (T_int + T_eff) / 2 (steady-state).
  Subsequent steps: transient response with thermal lag.

The ±10% criterion is only achievable if BOTH solvers use the same
steady-state effective-temperature approach. FiveR1C's transient thermal-mass
evolution causes it to lag behind the effective temperature, resulting in
large discrepancies during the diurnal cycle.

Run: python3 .agents/results/issue-1465-diurnal-parity.py
"""

import math
import sys

# Case 900 parameters (matching tests/gauge_validation_case_900.rs)
SOLAR_PEAK_W_M2 = 800.0
T_OUTDOOR_AVG_C = 15.0
T_OUTDOOR_AMP_C = 10.0
T_INDOOR_HVAC_SETPOINT_C = 20.0
H_EXT = 18.3  # W/m²K per ASHRAE 140 v2023
DT_SECONDS = 3600.0

# Wall: 200 mm HW concrete
THICKNESS_M = 0.200
K_W_MK = 0.51
RHO_KG_M3 = 1400.0
CP_J_KGK = 840.0
R_WALL = THICKNESS_M / K_W_MK  # 0.392 m²K/W
C_WALL = RHO_KG_M3 * CP_J_KGK * THICKNESS_M  # 235200 J/m²K


def outdoor_temperature_at(hour: int) -> float:
    h = hour % 24
    return T_OUTDOOR_AVG_C + T_OUTDOOR_AMP_C * math.cos((h - 15.0) / 24.0 * 2.0 * math.pi)


def solar_irradiance_at(hour: int) -> float:
    h = hour % 24
    if 6.0 <= h <= 18.0:
        return SOLAR_PEAK_W_M2 * math.sin((h - 6.0) / 12.0 * math.pi)
    return 0.0


def effective_temperature(hour: int) -> float:
    return outdoor_temperature_at(hour) + solar_irradiance_at(hour) / H_EXT


def simulate_gauge_solver() -> list[float]:
    """Steady-state flux at each hour: q = (T_eff - T_int) / R_wall"""
    fluxes = []
    for hour in range(24):
        t_eff = effective_temperature(hour)
        q = (t_eff - T_INDOOR_HVAC_SETPOINT_C) / R_WALL
        fluxes.append(q)
    return fluxes


def simulate_fiver1c() -> list[float]:
    """FiveR1C lumped-capacitance: evolves T_mass toward T_eff each hour."""
    t_mass = T_INDOOR_HVAC_SETPOINT_C  # initial
    pre_step = True
    fluxes = []

    for hour in range(24):
        t_eff = effective_temperature(hour)
        q_ss = (t_eff - T_INDOOR_HVAC_SETPOINT_C) / R_WALL

        if pre_step:
            # Steady-state seed on first step
            t_mass = (T_INDOOR_HVAC_SETPOINT_C + t_eff) / 2.0
            q = q_ss
            pre_step = False
        else:
            # Transient: Q_ext = (T_eff - T_mass) / R_wall
            q_ext = (t_eff - t_mass) / R_WALL
            dT_mass = q_ext / C_WALL * DT_SECONDS
            t_mass += dT_mass
            q = (t_mass - T_INDOOR_HVAC_SETPOINT_C) / R_WALL

        fluxes.append(q)

    return fluxes


def main() -> int:
    print("=" * 80)
    print("Issue #1606 — GaugeSolver vs FiveR1C Diurnal Parity Verification")
    print("=" * 80)
    print()
    print("Case 900 envelope: 200mm HW concrete, R_wall = {:.4f} m²K/W".format(R_WALL))
    print("Thermal capacity: {:.0f} kJ/m²K, τ = {:.1f} h".format(
        C_WALL / 1000.0, C_WALL * R_WALL / 3600.0))
    print()

    gauge_fluxes = simulate_gauge_solver()
    fiver1c_fluxes = simulate_fiver1c()

    print("Hour | T_out | Solar | T_eff  | Gauge  | 5R1C   | Diff%")
    print("-" * 72)
    all_pass = True
    for hour in range(24):
        t_out = outdoor_temperature_at(hour)
        solar = solar_irradiance_at(hour)
        t_eff = effective_temperature(hour)
        q_gauge = gauge_fluxes[hour]
        q_5r1c = fiver1c_fluxes[hour]
        diff_pct = abs(q_gauge - q_5r1c) / abs(q_5r1c) * 100 if abs(q_5r1c) > 1e-9 else 0.0
        flag = "✗" if diff_pct > 10.0 else " "
        print(f"{hour:4d} | {t_out:5.2f} | {solar:6.1f} | {t_eff:6.2f} | {q_gauge:7.2f} | {q_5r1c:7.2f} | {diff_pct:5.1f}% {flag}")

    print()
    print("=" * 80)
    print("AC1: Per-hour ±10% agreement")
    print("=" * 80)
    failures = []
    for hour in range(24):
        q_gauge = gauge_fluxes[hour]
        q_5r1c = fiver1c_fluxes[hour]
        diff_pct = abs(q_gauge - q_5r1c) / abs(q_5r1c) * 100 if abs(q_5r1c) > 1e-9 else abs(q_gauge) * 100
        if diff_pct > 10.0:
            failures.append((hour, q_gauge, q_5r1c, diff_pct))

    if failures:
        print("FAILED — {} hours exceed ±10%:".format(len(failures)))
        for hour, qg, q5, diff in failures[:5]:
            print("  Hour {:2d}: Gauge={:8.2f} 5R1C={:8.2f} Diff={:.1f}%".format(
                hour, qg, q5, diff))
        if len(failures) > 5:
            print("  ... and {} more".format(len(failures) - 5))
        all_pass = False
    else:
        print("PASS — all 24 hours within ±10%")

    print()
    print("=" * 80)
    print("AC2: Peak at hour 12, trough at hour 4-5")
    print("=" * 80)
    gauge_peak = gauge_fluxes.index(max(gauge_fluxes))
    r1c_peak = fiver1c_fluxes.index(max(fiver1c_fluxes))
    gauge_trough = gauge_fluxes.index(min(gauge_fluxes))
    r1c_trough = fiver1c_fluxes.index(min(fiver1c_fluxes))

    peak_ok = (gauge_peak == 12 and r1c_peak == 12)
    trough_ok = (4 <= gauge_trough <= 5) and (4 <= r1c_trough <= 5)
    print("GaugeSolver peak: hour {:2d} (expected 12) {}".format(gauge_peak, "✓" if gauge_peak == 12 else "✗"))
    print("FiveR1C peak:     hour {:2d} (expected 12) {}".format(r1c_peak, "✓" if r1c_peak == 12 else "✗"))
    print("GaugeSolver trough: hour {:2d} (expected 4-5) {}".format(gauge_trough, "✓" if 4 <= gauge_trough <= 5 else "✗"))
    print("FiveR1C trough:     hour {:2d} (expected 4-5) {}".format(r1c_trough, "✓" if 4 <= r1c_trough <= 5 else "✗"))
    if not peak_ok or not trough_ok:
        all_pass = False

    print()
    print("=" * 80)
    print("AC3: Bipolar response")
    print("=" * 80)
    max_flux = max(gauge_fluxes)
    min_flux = min(gauge_fluxes)
    bipolar = max_flux > 10.0 and min_flux < -10.0
    print("Max flux: {:.2f} W/m² (expected > 10)".format(max_flux))
    print("Min flux: {:.2f} W/m² (expected < -10)")
    print("Bipolar: {}".format("PASS" if bipolar else "FAIL"))
    if not bipolar:
        all_pass = False

    print()
    print("=" * 80)
    print("AC4: Amplitude ≥80 W/m²")
    print("=" * 80)
    amplitude = max_flux - min_flux
    amp_ok = amplitude >= 80.0
    print("Amplitude: {:.2f} W/m² (expected ≥80)".format(amplitude))
    print("Result: {}".format("PASS" if amp_ok else "FAIL"))
    if not amp_ok:
        all_pass = False

    print()
    print("=" * 80)
    print("SUMMARY: {}".format("ALL PASS" if all_pass else "SOME FAILURES"))
    print("=" * 80)
    print()
    print("Physics note: The large discrepancy between GaugeSolver (steady-state)")
    print("and FiveR1C (transient with thermal mass) is expected. The thermal mass")
    print("time constant τ ≈ 25.6 h means FiveR1C's T_mass evolves slowly toward")
    print("T_eff, while GaugeSolver responds instantly. This causes FiveR1C flux")
    print("to lag significantly behind the effective temperature drive.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
