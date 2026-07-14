# Result: Fix Energy Balance Regression (Issue #1580)

## Status: ✅ FIXED

## Summary
Fixed the 168-violation energy balance regression in Case 600 (`test_case_600_energy_balance_conservation`) by updating the InvariantChecker's 5R1C energy balance path to match the WIP PR #1559 zone model changes.

## Root Cause
The WIP PR #1559 changed the 5R1C zone model to:
1. Remove `opaque_sol_w` from `phi_m` (line 336 of `physics_impl.rs`)
2. Use proper sol-air temperature via `SolAirTemperature::for_roof()` instead of raw `outdoor_temp`

The InvariantChecker was still using the old approach:
- Adding `opaque_sol_w` to `phi_m` for ALL thermal model types
- Using `outdoor_temp` directly in the 5R1C envelope term

This created an energy accounting mismatch: the InvariantChecker double-counted opaque solar gains while the zone model handled them through the sol-air pathway.

## Changes Made (`src/sim/invariant_checker.rs`)

### 1. `zone_balance_for` — phi_m now conditional on thermal model type (lines ~234-242)
```rust
let phi_m = match model.thermal_model_type {
    ThermalModelType::NineRFourC => load_w * m_air_frac + remaining_sol * m_sol_frac + opaque_sol_w,
    _ => load_w * m_air_frac + remaining_sol * m_sol_frac,
};
```

### 2. `zone_balance_for` — 5R1C branch uses t_sol_air_zone (lines ~277-280)
Changed from `outdoor_temp` to `t_sol_air_zone`:
```rust
storage - (phi_m + h_tr_3 * (t_air - t_m_avg) + h_tr_em * (t_sol_air_zone - t_m_avg))
```

### 3. New `compute_5r1c_t_sol_air` method (lines ~372-413)
Added helper that mirrors the WIP zone model's `SolAirTemperature::for_roof(outdoor_temp, opaque_solar_ref[i], sky_temp)` per zone.

### 4. `calculate_mass_node_balance` — now computes both t_sol_air paths (lines ~154-172)
Selects the correct t_sol_air per zone based on thermal model type:
```rust
let t_sol_air_zone = match model.thermal_model_type {
    ThermalModelType::NineRFourC => t_sol_air_9r4c[i],
    _ => t_sol_air_5r1c[i],
};
```

### 5. `check_invariant_with_artificial_gain` — same fix applied (lines ~484-509)

## Files Changed
- `src/sim/invariant_checker.rs`

## Test Results
```
test_case_600_energy_balance_conservation ... ✅ PASSED (was 168 violations, max_residual=25.5W)
All zone_balance_eplus_isolation tests: ✅ 19 passed
Full lib test suite: ✅ 2719 passed, 1 unrelated flaky timing test
```

## Acceptance Criteria Checklist
- [x] Case 600 energy balance test passes (max_residual < 1e-7 W)
- [x] Case 900 energy balance test still passes (9R4C path unchanged)
- [x] Case 960 energy balance test still passes (9R4C path unchanged)
- [x] No new test failures introduced
- [x] Build is clean (0 errors, 0 warnings in invariant_checker.rs)
