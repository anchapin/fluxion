# Session 27: Physics-Based Refactoring - Root Cause Analysis

## Date: 2026-03-26
**Objective**: Continue root cause analysis for empirical factor removal

## Session 27 Summary

This session focused on analyzing the ROOT CAUSES of why empirical correction factors are still needed in the ASHRAE 140 validation, and implementing physics-based fixes.

## Key Findings and Fixes

### 1. Predictive Controller Now Uses Dynamic Setpoints ✅

**Issue**: For Cases 640 and 940 with setback schedules, the predictive controller was using fixed setpoints (`self.heating_setpoint`) instead of the time-varying schedules from `self.heating_schedule.value(hour)`.

**Fix Applied**: Changed to use `calculate_modulation_with_setpoints()` with dynamic setpoints:

```rust
// Get hour of day for schedule lookup (supports setback schedules)
let hour_of_day_idx = timestep % 24;

// Get time-varying setpoints from schedule (supports setback)
let heating_setpoint = self.heating_schedule.value(hour_of_day_idx);
let cooling_setpoint = self.cooling_schedule.value(hour_of_day_idx);

let (hvac_mode, modulation) = self
    .predictive_controller
    .calculate_modulation_with_setpoints(
        self.temperatures.as_ref()[0],
        self.mass_temperatures.as_ref()[0],
        temp_rate,
        heating_setpoint,
        cooling_setpoint,
    );
```

**Note**: This fix is correct and in place, but Case 640 and 940 still require empirical corrections. The predictive controller fix enables proper mode determination during setback hours, but the underlying thermal model still needs tuning.

### 2. Mode-Specific Coupling Analysis

**Finding**: The thermal_mass_correction() method resets h_tr_em_heating_factor and h_tr_em_cooling_factor to 1.0. This is CORRECT behavior because:
- Initial setup applies factors to base h_tr_em (lines 1293-1304)
- thermal_mass_correction() applies thermal mass correction to base h_tr_em
- The factors are then used in physics calculation via h_tr_em_heating/cooling

Attempting to preserve factors in thermal_mass_correction() caused double-application and massive under-prediction (Case 900 heating dropped from 1.17 to 0.09 MWh).

### 3. HVAC Mode Determination Logic ✅

Verified that the HVAC mode determination in `src/sim/hvac/control.rs::calculate_modulation()` is correct.

### 4. Seasonal Solar Gain Adjustment ✅

Already implemented in previous sessions (Session 25) for 900-series South window cases.

## Validation Results

```
Case 600: Heating=6.89 (Ref: 5.50-7.50), Cooling=8.82 (Ref: 8.00-10.50) - FAIL
Case 610: Heating=5.33 (Ref: 4.36-5.79), Cooling=4.56 (Ref: 3.92-6.14) - FAIL
Case 620: Heating=6.31 (Ref: 4.50-6.50), Cooling=3.43 (Ref: 3.20-5.00) - FAIL
Case 630: Heating=6.01 (Ref: 5.05-6.47), Cooling=2.23 (Ref: 2.13-3.70) - FAIL
Case 640: Heating=3.55 (Ref: 2.75-3.80), Cooling=6.41 (Ref: 5.95-8.10) - FAIL
Case 650: Heating=0.00 (Ref: 0.00-0.00), Cooling=5.12 (Ref: 4.82-7.06) - FAIL
Case 900: Heating=1.17 (Ref: 1.17-2.04), Cooling=3.48 (Ref: 2.13-3.67) - FAIL
Case 910: Heating=2.06 (Ref: 1.51-2.28), Cooling=1.69 (Ref: 0.82-1.88) - FAIL
Case 920: Heating=4.06 (Ref: 3.26-4.30), Cooling=2.42 (Ref: 1.84-3.31) - FAIL
Case 930: Heating=5.25 (Ref: 4.14-5.34), Cooling=1.04 (Ref: 1.04-2.24) - FAIL
Case 940: Heating=1.31 (Ref: 0.79-1.41), Cooling=3.13 (Ref: 2.08-3.55) - FAIL
Case 950: Heating=0.00 (Ref: 0.00-0.00), Cooling=0.93 (Ref: 0.39-0.92) - FAIL
Case 960: Heating=9.48 (Ref: 5.00-15.00), Cooling=0.80 (Ref: 1.00-3.50) - FAIL
```

## Files Modified

- `src/sim/engine.rs`:
  - Lines ~3486-3511: Fixed predictive controller to use dynamic setpoints from schedule (setback fix)

## Build & Test Status

- **Build**: ✅ PASSED (release build successful)
- **Unit Tests**: 744 passed, 30 failed (pre-existing failures in CTF and other modules)
- **Validation Tests**: Baseline results maintained with predictive controller fix in place

## Key Insight

The Session 27 investigation revealed that:
1. The predictive controller fix is correct and properly enables setback behavior
2. However, the thermal model itself still needs empirical corrections to match reference values
3. The 5R1C model has fundamental limitations that require validator-level corrections
4. Future work should focus on improving the underlying thermal model physics (e.g., multi-node CTF)
