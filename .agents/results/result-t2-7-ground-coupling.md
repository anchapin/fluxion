# T2.7: Fix Ground Coupling / h_tr_floor Calculation

**Issue**: #680
**Status**: DONE
**Date**: 2026-05-16

## Summary

Replaced hardcoded and unjustified `h_tr_floor` values with physics-based calculation from floor construction properties. The floor conductance is now consistently derived as `h_tr_floor = U_floor_construction × A_floor`, where `U_floor_construction` includes the interior film coefficient (5.88 W/m²K for downward heat flow) and ground coupling resistance (0.17 m²K/W per ASHRAE HOAFM).

## What Was Hardcoded

### 1. Default `floor_u_value = 0.039` (RUST constructor)
- **File**: `src/sim/thermal_model_core.rs:2182`
- **Problem**: `0.039` W/m²K implies R_total = 25.6 m²K/W — physically unrealistic for any floor. The ASHRAE 140 target is 0.190 W/m²K.
- **Used by**: `ThermalModel::new()` (bare constructor, unit tests), and `update_derived_parameters()` path.

### 2. `1.2×` multiplier for 900-series
- **File**: `src/sim/thermal_model_core.rs:839` (old line)
- **Problem**: `floor_u * zone_floor_area * 1.2` — no physics justification. Both 600 and 900 series target the same floor U-value of 0.190 W/m²K per ASHRAE 140 Tables B1-1 and B1-3.
- **Effect**: Overestimated ground coupling for high-mass cases by 20%.

### 3. Hardcoded `0.039` for Case 195
- **File**: `src/sim/thermal_model_core.rs:837` (old line), also line 471
- **Problem**: Case 195 was using `0.039 * zone_floor_area` instead of the actual construction U-value.
- **Effect**: Severely underestimated ground coupling for Case 195.

## What Was Replaced With

### Physics-based formula (ISO 13790 §7.2.2.2, ASHRAE 140 Annex B):

```
h_tr_floor = A_floor / R_floor_total

where:
  R_floor_total = R_film_int + R_materials + R_ground_coupling

  R_film_int     = 1 / h_int = 1 / 5.88 = 0.170 m²K/W  (ASHRAE 140, downward heat flow)
  R_materials    = Σ(δ_i / k_i)                           (construction layers)
  R_ground_coupling = 0.17 m²K/W                           (ASHRAE HOAFM, slab-on-grade)
```

This is already implemented in `construction.rs:r_value_total()` when `SurfaceType::Floor` is passed (Issue #588). The fix ensures this path is used consistently for ALL cases.

### Computed values:

| Case Series | Construction | R_materials | R_total | U_floor | A_floor | h_tr_floor |
|---|---|---|---|---|---|---|
| 600 (low-mass) | Timber 25mm + Fiberglass 197mm | 5.104 | 5.444 | 0.184 | 48 m² | 8.82 W/K |
| 900 (high-mass) | Concrete 80mm + Insulation 201mm | 5.182 | 5.522 | 0.181 | 48 m² | 8.69 W/K |
| 195 (solid) | Same as 600 series | 5.104 | 5.444 | 0.184 | 48 m² | 8.82 W/K |
| Default (new()) | — | — | 5.263 | 0.190 | 20 m² | 3.80 W/K |

## Files Changed

1. **`src/sim/thermal_model_core.rs`** — 3 changes:
   - Lines ~819-843: Replaced hardcoded h_tr_floor_vec calculation with single physics-based formula
   - Lines ~468-479: Removed Case 195 special case for `floor_u_value`; all cases now use construction U-value
   - Line 2182: Default `floor_u_value` changed from 0.039 to 0.190

2. **`src/sim/engine.rs`** — 2 changes:
   - Line 544: Test assertion updated from 0.78 to 3.8 (= 0.190 × 20.0)
   - Line 624: Test assertion updated from 0.78 to 3.8

## Acceptance Criteria Checklist

- [x] h_tr_floor calculated from floor construction U-value × floor area (not hardcoded)
- [x] Ground coupling resistance (R=0.17) properly included via `SurfaceType::Floor`
- [x] No unjustified multipliers (removed 1.2× for 900-series)
- [x] All cases (600, 900, 195 series) use same physics-based path
- [x] Default constructor uses realistic ASHRAE 140 floor U-value (0.190)
- [x] All 2449 library tests pass

## Out-of-Scope Dependencies

- **T2.1 (ground temperature BC)**: The ground temperature model (`ConstantGroundTemperature`, `KusudaGroundTemperature`) is used separately from h_tr_floor. The ground coupling resistance (0.17 m²K/W) in the U-value calculation is a steady-state approximation. A more sophisticated ground coupling model could use the Kusuda ground temperature profile, but that is a separate enhancement.
- **Dynamic ground coupling**: The current model uses a fixed R_ground = 0.17 m²K/W. A future enhancement could compute this from soil properties, slab perimeter, and insulation configuration per the ASHRAE Foundation Loading approach.

## Test Results

```
cargo test --lib: 2449 passed, 2 ignored, 0 failures (6.32s)
```
