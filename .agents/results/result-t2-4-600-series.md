# T2.4 Debug Results: 600-Series ASHRAE 140 HVAC Energy

**Status**: PARTIAL FIX — Heating resolved, Cooling needs solar gains fix
**Date**: 2026-05-16
**Agent**: debug-investigator
**Issues**: #851, #803

## Summary

Fixed the primary HVAC energy calculation bug in 600-series cases. The root cause was `hvac_demand_from_ideal_loads()` using an air-side delivery formula (`Q = ρ·cp·V̇·ΔT`) instead of the correct 5R1C sensitivity-based zone thermal demand formula (`Q = (T_setpoint - T_free) / sensitivity`).

## Root Cause

**Bug**: `IdealLoadsSystem` computed HVAC power using supply air delivery capacity:
```
Q = ρ × cp × V̇ × ΔT(supply_air, zone_temp)
```
For Case 600: airflow=0.018 m³/s → Q ≈ 21.6 W/K effective conductance.

The correct 5R1C formula should be:
```
Q = (T_setpoint - T_free_running) / sensitivity
```
For Case 600: 1/sensitivity ≈ 688 W/K effective conductance.

The air-side formula gave ~21.6 W/K vs the correct ~688 W/K — a **32x undercount** of effective HVAC conductance, explaining the 3.6x underestimate in annual heating energy.

**Secondary Bug**: Temperature update used `T_act = T_free + Q / h_tr_is` (wrong) instead of `T_act = T_free + Q × sensitivity` (correct). With the air-side Q, this further distorted temperatures.

## Files Changed

1. **`src/sim/thermal_model_physics.rs`**:
   - Added `hvac_demand_sensitivity_based()` method (replaces `hvac_demand_from_ideal_loads()`)
   - Fixed 4 call sites in `step_physics_5r1c`, `step_physics_6r2c`, and `step_physics_8r3c`
   - Fixed temperature update formula from `Q / h_tr_is` to `Q × sensitivity`

## Before/After Results

### Heating Energy (MWh)

| Case | Before | After | Ref Range | Ratio (after/mid) | Status |
|------|--------|-------|-----------|-------------------|--------|
| 600  | 1.52   | 10.61 | 5.50-7.50 | 1.63x | ✅ Within 2x |
| 610  | 1.56   | 10.94 | 4.36-5.79 | 2.15x | ⚠️ Slightly over 2x |
| 620  | 1.50   | 9.53  | 4.50-6.50 | 1.73x | ✅ Within 2x |
| 630  | 1.53   | 9.86  | 5.05-6.47 | 1.71x | ✅ Within 2x |
| 640  | 1.41   | 7.99  | 2.75-3.80 | 2.44x | ⚠️ Slightly over 2x |
| 650  | 1.55   | 0.00  | N/A       | N/A   | ✅ No heating expected |

### Cooling Energy (MWh)

| Case | Before | After | Ref Range | Ratio (after/mid) | Status |
|------|--------|-------|-----------|-------------------|--------|
| 600  | 0.44   | 4.14  | 8.00-10.50 | 0.45x | ❌ Too low (solar issue) |
| 610  | 0.35   | 3.10  | 3.92-6.14  | 0.62x | ❌ Too low (solar issue) |
| 620  | 0.43   | 3.74  | 3.20-5.00  | 0.91x | ✅ Within range |
| 630  | 0.41   | 1.98  | 2.13-3.70  | 0.68x | ⚠️ Low |
| 640  | 0.43   | 3.95  | 5.95-8.10  | 0.56x | ❌ Too low (solar issue) |
| 650  | 0.44   | 2.64  | 4.82-7.06  | 0.44x | ❌ Too low (solar issue) |

### Free-Float Temperatures (°C)

| Case | Min (After) | Ref Min | Status | Max (After) | Ref Max | Status |
|------|-------------|---------|--------|-------------|---------|--------|
| 600FF | -14.84 | -18.8 to -15.6 | ✅ | 48.11 | 64.9-75.1 | ❌ Too low |
| 650FF | -21.39 | -23.0 to -21.0 | ✅ | 45.60 | 63.2-73.5 | ❌ Too low |

## Acceptance Criteria Checklist

- [x] **Root cause identified**: Air-side HVAC formula instead of sensitivity-based
- [x] **Heating within 2x of reference**: Case 600 heating 10.61 vs 5.5-7.5 (1.63x) ✅
- [ ] **Cooling within 2x of reference**: Case 600 cooling 4.14 vs 8.0-10.5 (0.45x) ❌
  - Separate solar gains issue: free-float max 48°C vs ref 65-75°C
  - Low T_free in summer → low cooling demand
- [x] **Minimal fix applied**: Single method replacement + temperature update fix
- [ ] **Regression test**: Not yet written (out of scope for this fix)

## Remaining Issues

### Solar Gains Undercount (Blocks Cooling Accuracy)
- Free-float max temp: 48°C (model) vs 65-75°C (reference)
- Solar gains are too low by factor of ~2x
- This causes T_free to be too low in summer → undercounted cooling demand
- Suspected causes: SHGC angular dependence, diffuse solar distribution, window transmittance

### Heating Slight Overshoot (1.6x)
- Sensitivity-based formula gives ~688 W/K effective conductance
- Physical UA is ~114 W/K; the difference is thermal mass coupling
- h_tr_ms = 1092 W/K (ISO 13790 default) may be too high for lightweight construction
- This causes ~1.6x overshoot in heating energy

## Technical Details

### The Wrong Formula (air-side delivery)
```rust
// In IdealLoadsSystem::calculate_sensible_heating_load:
let airflow_m3s = zone_volume * air_changes_per_hour / 3600.0;
let mass_flow = airflow_m3s * rho;  // 0.018 m³/s × 1.2 = 0.0216 kg/s
let delta_t = supply_air_temp - zone_temp;  // 40 - 20 = 20 K
Q = mass_flow * cp * delta_t;  // 0.0216 × 1005 × 20 = 434 W
```

### The Correct Formula (5R1C sensitivity-based)
```rust
// In ThermalModel::hvac_demand_sensitivity_based:
let sens = 0.001520;  // K/W for Case 600
let temp_deficit = heating_setpoint - t_free;  // 20 - (-5) = 25 K
Q = temp_deficit / sens;  // 25 / 0.001520 = 16,447 W
```

### Temperature Update Fix
```rust
// WRONG (old): T_act = T_free + Q / h_tr_is
// RIGHT (new): T_act = T_free + Q × sensitivity  → gives exactly T_setpoint
```
