# Issue #803 Investigation: Case 610 Annual Heating ~3x Under Expected

**Status:** Root Cause Identified
**Date:** 2026-05-18
**Branch:** `investigation/issue-803-case-610-heating`
**Severity:** CRITICAL — affects ALL 600-series HVAC cases (22/26 tests failing)

## Executive Summary

The reported symptom ("heating ~4x over expected") is **inverted**. The actual simulation
produces **~3x UNDER** the reference values. Case 610 annual heating is 1.55 MWh vs reference
4.36-5.79 MWh. Peak heating is 0.46 kW vs reference 4.30-5.70 kW.

**Root cause:** The `IdealLoadsSystem` HVAC demand formula uses the infiltration airflow
(0.5 ACH) to compute heating/cooling capacity, yielding a maximum ~0.45 kW — far below
the building's actual heat loss rate of ~4.6 kW at design conditions.

## Test Results (All 600-Series)

| Case | Metric | Simulated | Reference | Ratio |
|------|--------|-----------|-----------|-------|
| 610 | Annual Heating | 1.55 MWh | 4.36-5.79 MWh | 0.27x |
| 610 | Peak Heating | 0.46 kW | 4.30-5.70 kW | 0.09x |
| 610 | Annual Cooling | 0.36 MWh | 3.92-6.14 MWh | 0.07x |
| 610 | Peak Cooling | 0.32 kW | 2.20-2.90 kW | 0.13x |
| 620 | Annual Heating | 1.52 MWh | 4.50-6.50 MWh | 0.28x |
| 630 | Annual Heating | 1.61 MWh | 5.05-6.47 MWh | 0.28x |
| 640 | Annual Heating | 1.53 MWh | 2.75-3.80 MWh | 0.47x |
| 650 | Annual Cooling | 0.22 MWh | 4.82-7.06 MWh | 0.03x |
| 600FF | Max Temp | 50.26°C | 64.90-75.10°C | 0.71x |
| 600FF | Min Temp | -8.25°C | -18.80 to -15.60°C | 0.48x |
| 650FF | Max Temp | 47.69°C | 63.20-73.50°C | 0.70x |

**ALL 22 failing tests show values that are too low**, indicating a systematic under-supply
of HVAC energy. The free-float temperature range is also compressed (max too low, min too high),
which is consistent with insufficient thermal coupling.

## Root Cause

### Location
- **File:** `src/sim/hvac/ideal_loads.rs`, lines 139-159
- **Function:** `ZoneIdealLoads::calculate_sensible_heating_load()`
- **Same issue in:** `calculate_sensible_cooling_load()` (lines 103-123)

### Buggy Formula

```rust
pub fn calculate_sensible_heating_load(
    zone_temp: f64,        // = T_i_free (free-floating temperature)
    heating_setpoint: f64,
    supply_air_temp: f64,  // = 40.0°C (hardcoded default)
    zone_volume: f64,      // = 129.6 m³
    air_changes_per_hour: f64, // = 0.5 ACH (infiltration rate)
) -> f64 {
    let airflow_m3s = zone_volume * air_changes_per_hour / 3600.0;
    let mass_flow = airflow_m3s * 1.2;  // 0.0216 kg/s
    let delta_t = (supply_air_temp - zone_temp).max(0.0);
    mass_flow * 1005.0 * delta_t  // WRONG: supply air capacity, not zone demand
}
```

### What it computes vs. what it should compute

**Current (WRONG):** Maximum heating capacity of the infiltration air stream
```
Q = m_dot_infiltration × cp × (T_supply - T_zone)
Q = 0.0216 kg/s × 1005 J/(kg·K) × (40 - T_free)
Q ≈ 21.7 W/K × (40 - T_free)    ← only ~22 W/K effective conductance
```

**Correct:** Power needed to maintain setpoint in the 5R1C model
```
Q = h_tr_is × (T_setpoint - T_free)
Q = 1228 W/K × (20 - T_free)    ← actual thermal coupling to zone
```

### Numerical Proof

For Case 610 at peak conditions (T_free near setpoint):
| Parameter | Value |
|-----------|-------|
| Volume | 129.6 m³ |
| Infiltration ACH | 0.5 |
| Mass flow (0.5 ACH) | 0.0216 kg/s |
| Supply air temp | 40°C |
| h_tr_is (surface-to-air) | 1228 W/K |
| **Current formula capacity** | **0.45 kW** |
| **Reference peak** | **4.30-5.70 kW** |
| **Ratio** | **~10x under** |

The HVAC system physically cannot supply more than ~0.45 kW, while the building
needs ~4.6 kW at design conditions. The zone temperature never reaches setpoint
in cold weather.

### Why annual energy is only 3x (not 10x) under

The system runs more hours trying to maintain setpoint (since it can never quite
reach it during cold periods), partially compensating for the under-capacity. But
it still underestimates by ~3x because the capacity cap limits total energy delivery.

### Downstream Impact

The wrong Q_hvac propagates to:
1. **Temperature update** (line 1164): `T_act = T_free + Q / h_tr_is` — T_act never
   reaches setpoint because Q is too small
2. **Energy accumulation** (line 1176-1183): Heating/cooling energy sums are capped
   at the supply air capacity
3. **Peak power tracking** (lines 1077-1087, 1116-1123): Peaks are limited to
   the supply air capacity (~0.46 kW vs expected ~5 kW)

## Fix Recommendation

### Option A: Direct 5R1C Demand (Recommended)

Replace `calculate_sensible_heating_load` with the physically correct demand formula
that uses the building's actual thermal coupling:

```rust
// In step_physics_5r1c(), after computing t_i_free:
let hvac_demand = h_tr_is * (heating_setpoint - t_i_free).max(0.0);
```

This is already consistent with the temperature update at line 1164:
`T_act = T_free + Q / h_tr_is = T_free + h_tr_is × (sp - T_free) / h_tr_is = sp`

### Option B: Size the ideal loads airflow correctly

If keeping the air-side formula, the airflow must be sized to meet peak demand:
```rust
// Size airflow to deliver enough heat at design conditions
let m_dot_sized = Q_peak / (cp * (T_supply - T_setpoint));
// For Case 610: m_dot_sized = 4600 / (1005 * 20) = 0.229 kg/s
// vs current: m_dot = 0.0216 kg/s (10x too small)
```

### Option A is preferred because:
1. It's the standard approach in ISO 13790 and EnergyPlus ideal loads
2. It directly uses the building's thermal conductance (already computed)
3. No arbitrary supply air temperature parameter needed
4. Works for all building types without sizing calculations

### Files to Modify

1. **`src/sim/hvac/ideal_loads.rs`**: Replace `calculate_sensible_heating_load` and
   `calculate_sensible_cooling_load` with building-conductance-based formulas
2. **`src/sim/thermal_model_physics.rs`**: Pass `h_tr_is` and `den` to the HVAC demand
   calculation instead of relying on the IdealLoadsSystem air-side formula
3. **`src/sim/thermal_model_physics.rs`**: Remove the stale comment at line 740
   about "h_loss × ΔT formula" which was never actually implemented

### Estimated Impact of Fix

- All 22 failing 600-series tests should move much closer to reference values
- Peak heating: 0.46 kW → ~4.5 kW (matches reference)
- Annual heating: 1.55 MWh → ~5 MWh (matches reference)
- Free-float temperatures: Range should expand (max up, min down)
- The `h_tr_is` conductance correctly captures the thermal coupling

## Out-of-Scope Findings

1. **Issue #746 (ground temperature boundary):** The ground coupling is implemented
   via `h_tr_floor` and `t_g`, but may need verification after the HVAC fix.
2. **Night ventilation (Issue #824):** The night-vent air-side path appears correctly
   implemented for Case 650/650FF but should be re-verified after the HVAC fix.
3. **Thermal mass capacitance (Issue #821):** Previous fix appears correct — the
   thermal capacitance is set from actual construction layers, not overwritten.
4. **Stale comment** at line 740 referencing "#872: sensitivity variable removed —
   HVAC demand now uses h_loss × ΔT formula" — this was never actually implemented.

## Evidence Chain

1. Run test → 1.55 MWh simulated vs 4.36-5.79 MWh reference
2. Trace `hvac_demand_from_ideal_loads` → `calculate_sensible_heating_load`
3. Formula uses `m_dot = V × ACH / 3600 × ρ` = 0.0216 kg/s
4. Max capacity = 0.0216 × 1005 × (40 - 20) = 434 W = 0.43 kW
5. Simulated peak = 0.46 kW (matches formula output, not reference)
6. Building heat loss at design = ~4.6 kW (matches reference 4.3-5.7 kW)
7. Root cause confirmed: HVAC capacity formula uses infiltration flow, not building demand
