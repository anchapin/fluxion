---
phase: 02-Thermal-Mass-Dynamics
plan: 05
subsystem: thermal_integration
tags: [test-fix, verification-gap, thermal-mass]
dependency_graph:
  requires:
    - "02-02" (thermal_integration module implementation)
  provides:
    - validated_integration_tests
  affects:
    - "02-VERIFICATION.md" (closes gap)
tech_stack:
  added: []
  patterns: [test-to-implementation-mapping]
key_files:
  created: []
  modified:
    - path: tests/test_thermal_mass_integration.rs
      description: Updated to import actual thermal_integration module functions instead of calling local stubs
decisions:
  - "Import thermal_integration module functions directly rather than maintaining local stubs"
  - "Fix energy balance calculations for implicit methods (backward Euler uses heat flux at new state, Crank-Nicolson uses average)"
metrics:
  duration: 152
  completed_date: "2026-03-09"
---

# Phase 02 Plan 05: Thermal Mass Integration Test Fix Summary

## One-Liner
Fixed thermal mass integration test module to import and test the actual thermal_integration module implementation, closing verification gap with all 8 tests now passing.

## Objective
Fix thermal mass integration test module to import and test the actual thermal_integration module implementation instead of calling local unimplemented!() stubs, closing the verification gap identified in 02-VERIFICATION.md.

## Tasks Completed

### Task 1: Import thermal_integration module functions
**Status:** Completed
**Commit:** `27ec999`
**Files Modified:**
- `tests/test_thermal_mass_integration.rs`

**Changes Made:**
1. Added import statement for thermal_integration module functions:
   ```rust
   use fluxion::sim::thermal_integration::{
       backward_euler_update,
       crank_nicolson_update,
       explicit_euler_update,
   };
   ```

2. Removed local stub functions that returned `unimplemented!()`:
   - Removed `backward_euler_step()` local stub
   - Removed `crank_nicolson_step()` local stub

3. Updated `explicit_euler_step()` to use imported `explicit_euler_update()` function

4. Added wrapper functions to maintain test interface:
   - `backward_euler_step()` calls `backward_euler_update()` with config unpacking
   - `crank_nicolson_step()` calls `crank_nicolson_update()` with config unpacking

5. Fixed energy balance calculation in `test_integration_methods_preserve_energy_balance()`:
   - Backward Euler: Uses heat flux at NEW state (implicit method evaluates at Tm_new)
   - Crank-Nicolson: Uses average of heat flux at OLD and NEW states

**Verification:**
```bash
cargo test --test test_thermal_mass_integration -- --nocapture
```

**Results:**
- All 8 tests now pass (was 3/8 passing)
- Test result: `ok. 8 passed; 0 failed; 0 ignored; 0 measured`
- Tests now validate actual implementation in `src/sim/thermal_integration.rs`

**Test Coverage:**
1. `test_explicit_euler_stable_for_low_thermal_capacitance` - Explicit Euler stable for low Cm
2. `test_explicit_euler_accuracy_limitations_for_high_thermal_capacitance` - Explicit Euler accuracy limits for high Cm
3. `test_backward_euler_numerically_stable_for_high_thermal_capacitance` - Backward Euler numerically stable for high Cm
4. `test_backward_euler_correct_temperature_updates` - Backward Euler produces correct temperature updates
5. `test_crank_nicolson_second_order_accuracy` - Crank-Nicolson is 2nd-order accurate
6. `test_integration_methods_preserve_energy_balance` - Integration methods preserve energy balance over 8760 timesteps
7. `test_integration_methods_handle_heat_flux_sign` - Integration methods correctly handle heating/cooling
8. `test_case_900_thermal_mass_requirements` - Case 900 has extremely high thermal capacitance requiring implicit methods

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed energy balance calculation for implicit integration methods**
- **Found during:** Task 1
- **Issue:** Test `test_integration_methods_preserve_energy_balance` was calculating energy balance incorrectly for implicit methods. It was using heat flux at the old temperature for both backward Euler and Crank-Nicolson, but implicit methods evaluate heat flux at the new state (or average of old/new for Crank-Nicolson).
- **Fix:** Updated energy balance calculations:
  - Backward Euler: `Q_net(Tm_new)` instead of `Q_net(Tm_old)`
  - Crank-Nicolson: `0.5 * (Q_net(Tm_old) + Q_net(Tm_new))` instead of `Q_net(Tm_old)`
- **Files modified:** `tests/test_thermal_mass_integration.rs`
- **Commit:** `27ec999`

## Technical Details

### Function Signature Mapping

**Test Function:**
```rust
fn backward_euler_step(tm_old: f64, config: &ThermalCapacitanceConfig) -> f64
```

**Module Function:**
```rust
pub fn backward_euler_update(
    tm_old: f64,
    dt: f64,
    cm: f64,
    h_tr_em: f64,
    h_tr_ms: f64,
    t_ext: f64,
    t_surface: f64,
    phi_m: f64,
) -> f64
```

**Wrapper Implementation:**
```rust
fn backward_euler_step(tm_old: f64, config: &ThermalCapacitanceConfig) -> f64 {
    backward_euler_update(
        tm_old,
        DT,
        config.cm,
        config.h_tr_em,
        config.h_tr_ms,
        config.t_ext,
        config.t_surface,
        config.phi_m,
    )
}
```

### Energy Balance Physics

**Explicit Euler (forward method):**
```
Cm * (Tm_new - Tm_old) / dt = Q_net(Tm_old)
```
Heat flux evaluated at old state (explicit).

**Backward Euler (implicit method):**
```
Cm * (Tm_new - Tm_old) / dt = Q_net(Tm_new)
```
Heat flux evaluated at new state (implicit), requires solving implicit equation.

**Crank-Nicolson (semi-implicit method):**
```
Cm * (Tm_new - Tm_old) / dt = 0.5 * (Q_net(Tm_old) + Q_net(Tm_new))
```
Heat flux evaluated as average of old and new states, provides 2nd-order accuracy.

## Gap Closure

### Verification Gap (from 02-VERIFICATION.md)
**Original Issue:**
- "test_thermal_mass_integration.rs has 5/8 tests failing (backward_euler_step, crank_nicolson_step functions return unimplemented!() - these are test scaffolds, not testing the actual thermal_integration module)"
- "Test defines local backward_euler_step() and crank_nicolson_step() stubs that return unimplemented!(), not testing the actual src/sim/thermal_integration.rs implementation"

**Resolution:**
- Updated test file to import and call actual thermal_integration module functions
- All 8 tests now pass
- Gap closed ✅

## Status

**Plan Status:** ✅ Complete
**Verification Status:** ✅ All tests passing (8/8)
**Requirements Satisfied:**
- ✅ All 8 tests in test_thermal_mass_integration.rs pass
- ✅ Tests import and call actual thermal_integration module functions
- ✅ No local unimplemented!() stubs remain
- ✅ Tests validate actual implementation in src/sim/thermal_integration.rs
- ✅ Gap from 02-VERIFICATION.md closed

**Phase 2 Status:**
- Phase 2 verification now complete (4/4 must-haves verified)
- Ready for Phase 3 execution (Solar Radiation & External Boundaries)

## Next Steps

Phase 2 is complete. The next phase is:
- **Phase 03:** Solar Radiation & External Boundaries
  - Focus: Fix solar gain calculations and external boundary conditions
  - Address peak cooling load under-prediction and annual cooling energy discrepancies
  - Requirements: SOLAR-01 through SOLAR-04

## Notes

- The thermal_integration module (src/sim/thermal_integration.rs) has 11 unit tests that all pass, confirming correct implementation
- The integration test module (tests/test_thermal_mass_integration.rs) now tests the actual implementation with 8 comprehensive tests
- No architectural changes needed - this was a straightforward test-to-implementation mapping fix
- Energy balance calculations were corrected to match implicit method physics

## Self-Check: PASSED

**Files Created:**
- ✅ `.planning/phases/02-Thermal-Mass-Dynamics/02-05-SUMMARY.md`

**Files Modified:**
- ✅ `tests/test_thermal_mass_integration.rs`

**Commits Created:**
- ✅ `27ec999` - test(02-05): fix thermal mass integration tests to use actual implementation

**Tests Passing:**
- ✅ All 8 tests in test_thermal_mass_integration.rs pass
- ✅ Gap from 02-VERIFICATION.md closed
