# Provenance: ASHRAE 140 Case 900 Gauge Solver Diurnal Reference

**File:** `tests/reference_data/gauge/case_900_diurnal_reference.csv`
**Issue:** #2305 (GaugeSolver Case 900 synthetic CSV vs real E+ data)
**Status:** SYNTHETIC — requires EnergyPlus to generate real hourly data

## Data Status

This file is **SYNTHETIC / ANALYTICAL FIXTURE** — NOT a real EnergyPlus run.
The values are computed from the documented GaugeSolver formula.

## Source of Truth

The values in this CSV are computed from the EXACT GaugeSolver formula
(`src/physics/gauge_solver.rs::GaugeSolver::step_with_boundary_conditions`):

```text
T_outdoor(h) = 15 + 10·cos((h - 15)/24·2π)              # sinusoidal 5-25 °C
Solar(h)     = 800·sin((h - 6)/12·π) for h ∈ [6, 18], else 0  # W/m²
T_sol_air(h) = T_outdoor(h) + Solar(h) / h_ext          # h_ext = 18.3
q_gauge(h)   = (T_sol_air(h) - T_int) / R_wall          # R_wall = 0.392
q_baseline(h)= (T_outdoor(h) - T_int) / R_total          # R_total = 0.572
```

Wall geometry (Case 900 envelope): 200 mm HW concrete, k=0.51 W/mK,
ρ=1400 kg/m³, Cp=840 J/kgK, R_wall=0.392 m²K/W. Interior film h_int=8 W/m²K,
interior setpoint T_int=20 °C. No HVAC, no internal gains, free-floating
solar-only forcing.

## Why Synthetic (Not from EnergyPlus)

The companion annual aggregate reference
(`tests/reference_data/zone_balance/case_900_energy_reference.csv`) is sourced
from NREL/TP-472-6231 Table 3-2 (per `zone_balance/PROVENANCE.md`). That file
contains only annual totals, not hourly data.

An **hourly** EnergyPlus Case 900 CSV would require running EnergyPlus directly.
When such a CSV becomes available (via external E+ simulation), replace this
fixture in a follow-up issue.

## Relationship to Annual Energy Tests

The `#[ignore]` annual ±15% Case 900 energy tolerance tests in
`zone_balance_eplus_isolation.rs` test the **full engine** against the E+
annual aggregate reference. They are independent of this diurnal CSV, which is
used only by the gauge solver shadow-mode validation harness.

The gauge solver (`GaugeSolver`) is a **steady-state** solver with no thermal
capacitance — it computes instantaneous flux, not transient response. Therefore:
- The diurnal CSV cannot validate transient behavior
- Real E+ hourly data would show thermal lag that GaugeSolver cannot reproduce
- The CSV validates that GaugeSolver correctly implements the steady-state formula

## Columns

| Column | Description |
|--------|-------------|
| hour | 0..23 (TMY non-leap day) |
| t_outdoor_c | outdoor air temperature (°C) |
| solar_w_m2 | incident solar irradiance on south wall (W/m²) |
| t_sol_air_c | sol-air temperature (°C) = t_outdoor + solar/h_ext |
| q_baseline_w_m2 | baseline flux without solar, full R_total (W/m²) |
| q_gauge_w_m2 | expected GaugeSolver shadow flux, R_wall only (W/m²) |

## Tolerance Policy

- ±1% for algebraic parity (test asserts GaugeSolver reproduces the q_gauge
  column to within 1% on every hour)
- ±5% for diurnal amplitude diagnostics (the gauge transport's parallel-transport
  stub does not evolve the mass node — flux values are *instantaneous*
  steady-state response to boundary conditions; this is by design for Phase 1b)

## How to Regenerate

If EnergyPlus is available, run the Case 900 simulation and export hourly
surface heat flux data for the south wall. Replace the `q_gauge_w_m2` and
`t_sol_air_c` columns with the E+ values. The outdoor temperature and solar
columns should match the Denver TMY3 spring-week forcing used in the current
synthetic formula.

## Issue Relationship

This fixture supports the Phase 3 GaugeSolver validation harness
(issue #1465). The long-term goal is to replace the synthetic diurnal
profile with real E+ hourly data to validate diurnal swing and phase lag
against production physics. This is tracked in issue #2305.
