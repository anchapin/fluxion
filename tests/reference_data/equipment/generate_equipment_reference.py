#!/usr/bin/env python3
"""Generate HVAC equipment reference data CSVs (Issue #1933).

These datasets are **analytically derived** from published engineering
correlations — they are NOT EnergyPlus simulation outputs. Each curve's
coefficients and provenance are documented in PROVENANCE.md and in the
leading ``#`` comment header of the emitted CSV (matching the convention
used by the rest of ``tests/reference_data/``).

Outputs (written next to this script):
    fan_affinity_laws.csv
    chiller_capacity_capft.csv
    boiler_part_load_efficiency.csv
    heat_pump_mode_transition.csv

Run from the repository root::

    python tests/reference_data/equipment/generate_equipment_reference.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent


def write_csv(name: str, header_lines: list[str], data_lines: list[str]) -> None:
    path = OUT_DIR / name
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for line in header_lines:
            fh.write(f"# {line}\n" if not line.startswith("#") else f"{line}\n")
        fh.write("\n".join(data_lines))
        fh.write("\n")
    print(f"wrote {path.relative_to(OUT_DIR.parent.parent)} ({len(data_lines)} data rows)")


# ---------------------------------------------------------------------------
# 1. Fan affinity laws (ASHRAE Fundamentals Ch. 21; ASHRAE 90.1 §6.5.3.1)
# ---------------------------------------------------------------------------
def gen_fan_affinity() -> None:
    header = [
        "Analytical Reference: Centrifugal fan affinity laws + constant-speed VIV baseline (Issue #1933)",
        "Source: ASHRAE Handbook — Fundamentals, Ch. 21 (Fans); ASHRAE Standard 90.1-2022 §6.5.3.1",
        "Derivation: For variable-speed drive (VSD), Q/Qf=N/Nf, P/Pf=(N/Nf)^2, W/Wf=(N/Nf)^3.",
        "           Variable-inlet-vane (VIV, constant speed) normalized power is a typical quadratic",
        "           fit (0.13 + 0.35*N + 0.52*N^2) representative of a 60% efficient baseline fan.",
        "Columns: speed_ratio (N/Nf), flow_ratio (Q/Qf), pressure_ratio (P/Pf),",
        "        power_ratio_vsd (W/Wf, cubic), power_ratio_viv (constant-speed VIV)",
        "Status: ANALYTICAL — derived from published fan laws, not EnergyPlus output.",
    ]
    rows = []
    for i in range(0, 11):
        n = i / 10.0
        flow = n
        press = n * n
        power = n * n * n
        viv = 0.13 + 0.35 * n + 0.52 * n * n
        rows.append(
            f"{n:.2f},{flow:.4f},{press:.4f},{power:.4f},{viv:.4f}"
        )
    write_csv("fan_affinity_laws.csv", header,
              ["speed_ratio,flow_ratio,pressure_ratio,power_ratio_vsd,power_ratio_viv"] + rows)


# ---------------------------------------------------------------------------
# 2. Chiller CAPFT biquadratic (AHRI 550/590; EnergyPlus TSD)
# ---------------------------------------------------------------------------
def gen_chiller_capft() -> None:
    # Typical water-cooled centrifugal chiller coefficients (DOE Commercial
    # Reference Buildings / EnergyPlus TSD). T in °C.
    c = [0.958, 0.0179, -0.00037, -0.0010, -0.000007, 0.00021]
    Te_rated, Tc_rated = 6.67, 29.44  # 44°F LCHWT, 85°F ECWT (AHRI 550/590)

    def capft(te: float, tc: float) -> float:
        return (
            c[0] + c[1] * te + c[2] * te * te
            + c[3] * tc + c[4] * tc * tc + c[5] * te * tc
        )

    rated = capft(Te_rated, Tc_rated)
    header = [
        "Analytical Reference: Water-cooled centrifugal chiller CAPFT biquadratic (Issue #1933)",
        "Source: AHRI Standard 550/590 (IP); EnergyPlus Technical Support Document (TSD) commercial reference chiller.",
        "Form: CAPFT(T_evap,T_cond) = c0 + c1*Te + c2*Te^2 + c3*Tc + c4*Tc^2 + c5*Te*Tc",
        f"Coefficients: c0={c[0]}, c1={c[1]}, c2={c[2]}, c3={c[3]}, c4={c[4]}, c5={c[5]}",
        f"Rated point: T_evap={Te_rated} C (44 F LCHWT), T_cond={Tc_rated} C (85 F ECWT); CAPFT_rated={rated:.5f}",
        "Columns: T_evap_C (leaving chilled water), T_cond_C (entering condenser water),",
        "        capft_raw (absolute capacity factor), capft_normalized (ratio to rated)",
        "Status: ANALYTICAL — biquadratic curve fit coefficients, not EnergyPlus output.",
    ]
    rows = []
    for te in [4.0, 5.0, 6.0, 6.67, 7.0, 8.0, 9.0, 10.0]:
        for tc in [20.0, 24.0, 29.44, 32.0, 35.0]:
            v = capft(te, tc)
            rows.append(f"{te:.2f},{tc:.2f},{v:.5f},{v / rated:.5f}")
    write_csv("chiller_capacity_capft.csv", header,
              ["T_evap_C,T_cond_C,capft_raw,capft_normalized"] + rows)


# ---------------------------------------------------------------------------
# 3. Boiler part-load efficiency (ASHRAE 90.1; EnergyPlus TSD)
# ---------------------------------------------------------------------------
def gen_boiler_efficiency() -> None:
    # Normalized Boiler Efficiency Curve (EnergyPlus Boiler:HotWater reference).
    # eta_norm(PLR) = c0 + c1*PLR + c2*PLR^2; eta_actual = eta_rated * eta_norm.
    c = [1.0229, 0.0256, -0.0458]
    rated_eff = 0.80  # ASHRAE 90.1 Table 6.8.1 minimum for non-condensing hot-water
    header = [
        "Analytical Reference: Non-condensing hot-water boiler part-load efficiency (Issue #1933)",
        "Source: ASHRAE Standard 90.1-2022 Table 6.8.1 (rated thermal efficiency); EnergyPlus TSD Normalized Boiler Efficiency Curve.",
        "Form: eta_norm(PLR) = c0 + c1*PLR + c2*PLR^2  (NOT divided by PLR — that is a different curve type)",
        f"Coefficients: c0={c[0]}, c1={c[1]}, c2={c[2]}; rated_thermal_efficiency={rated_eff}",
        "Columns: plr (part-load ratio = load/rated_capacity), eta_norm_ratio (efficiency / rated),",
        "        eta_absolute (combustion efficiency, dimensionless)",
        "Note: PLR=0 row reports standby (idle) efficiency ratio; real operation has a minimum PLR.",
        "Status: ANALYTICAL — published normalized curve, not EnergyPlus output.",
    ]
    rows = [
        f"0.00,{c[0]:.4f},{c[0] * rated_eff:.4f}",
    ]
    for i in range(1, 11):
        plr = i / 10.0
        eta_norm = c[0] + c[1] * plr + c[2] * plr * plr
        rows.append(f"{plr:.2f},{eta_norm:.4f},{eta_norm * rated_eff:.4f}")
    write_csv("boiler_part_load_efficiency.csv", header,
              ["plr,eta_norm_ratio,eta_absolute"] + rows)


# ---------------------------------------------------------------------------
# 4. Air-source heat pump mode transition (AHRI 210/240; ISO 13256)
# ---------------------------------------------------------------------------
def gen_heat_pump_transition() -> None:
    # Linear COP(T_odb) model anchored at AHRI 210/240 heating rating point
    # (8.33°C / 47°F dry-bulb). Balance temperature 18°C above which heating
    # demand is satisfied passively and the unit cycles to cooling.
    a, b = 2.0, 0.07  # COP = a + b*T_odb; COP(8.33)=2.583
    balance_temp = 18.0  # heating/cooling switchover (typical thermostat deadband center)
    header = [
        "Analytical Reference: Air-source heat pump mode transition (Issue #1933)",
        "Source: AHRI Standard 210/240 (unitary air-source heat pumps); ISO 13256-2 water-source.",
        "Model: COP_heating(T_odb) = a + b*T_odb, clamped to >=0; heating mode active for T_odb < balance_temp.",
        f"Coefficients: a={a}, b={b}; balance_temp={balance_temp} C; rated COP at 8.33 C (47 F) = {a + b*8.33:.4f}",
        "Columns: T_odb_C (outdoor dry-bulb), cop_heating, mode (heating | heating_off)",
        "Status: ANALYTICAL — linearized rating curve, not EnergyPlus output.",
    ]
    rows = []
    for t in [-15.0, -10.0, -5.0, 0.0, 2.0, 5.0, 8.33, 10.0, 15.0, 20.0]:
        cop = max(0.0, a + b * t)
        mode = "heating" if t < balance_temp else "heating_off"
        rows.append(f"{t:.2f},{cop:.4f},{mode}")
    write_csv("heat_pump_mode_transition.csv", header,
              ["T_odb_C,cop_heating,mode"] + rows)


def main() -> None:
    gen_fan_affinity()
    gen_chiller_capft()
    gen_boiler_efficiency()
    gen_heat_pump_transition()
    print(f"\nAll equipment reference CSVs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
