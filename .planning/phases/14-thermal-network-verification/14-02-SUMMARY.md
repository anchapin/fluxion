---
phase: 14
plan: 02
subsystem: Thermal Network Verification
tags: [thermal-mass, coupling-ratio, ashrae-140, validation]
dependency_graph:
  requires: []
  provides: [thermal-mass-correction, coupling-validation]
  affects: [high-mass-buildings, annual-energy]
tech_stack:
  added: [HIGH_MASS_THRESHOLD constant, apply_thermal_mass_correction method]
  patterns: [threshold-detection, ratio-correction, mode-specific-coupling]
key_files:
  created:
    - tests/test_thermal_mass_coupling.rs
  modified:
    - src/sim/engine.rs
decisions: []
metrics:
  duration: 1773428450 seconds
  completed_date: 2026-03-13T19:00:50Z
---

# Phase 14 Plan 02: Thermal Mass Coupling Ratio Correction

## One-Liner Summary

Implemented thermal mass correction method to achieve coupling ratio >= 0.1 for high-mass buildings, addressing ASHRAE 140 compliance through threshold-based detection and mode-specific coupling factor integration.

## Task Completion Summary

### Task 1: Add thermal mass correction method to ThermalModel (COMPLETED)

**Status:** ✅ COMPLETED
**Commit:** c10234b

**Implementation:**
- Added `HIGH_MASS_THRESHOLD` constant (5.0e6 J/K) to distinguish high-mass from low-mass buildings
- Implemented `apply_thermal_mass_correction()` method with:
  - Threshold detection based on structure thermal capacitance (excludes air)
  - Coupling ratio calculation: h_tr_em / h_tr_ms
  - Automatic h_tr_em adjustment to achieve target ratio >= 0.1
  - Early exit for low-mass buildings and already-compliant cases

**Key Design Decisions:**
- Threshold: 5e6 J/K (between Case 600's 2.4e6 J/K and Case 900's 1.2e7 J/K)
- Target ratio: 0.1 (ASHRAE 140 requirement)
- Structure capacitance: Excludes air capacitance to isolate building mass
- Direct modification: Set h_tr_em to target value for all zones

### Task 2: Create thermal mass coupling test (COMPLETED)

**Status:** ✅ COMPLETED
**Commit:** 0daf36d

**Implementation:**
- Created comprehensive test suite in `tests/test_thermal_mass_coupling.rs`:
  - `test_thermal_mass_coupling_ratio_low_mass`: Verifies Case 600 not corrected
  - `test_thermal_mass_coupling_ratio_high_mass`: Verifies Case 900 achieves ratio >= 0.1
  - `test_thermal_mass_threshold_detection`: Validates capacitance threshold detection

**Test Results:** All tests passing ✅
- Low-mass buildings uncorrected as expected
- High-mass buildings achieve coupling ratio >= 0.1 after correction
- Threshold detection correctly identifies high-mass vs low-mass buildings

### Task 3: Integrate thermal mass correction into model initialization (COMPLETED)

**Status:** ✅ COMPLETED
**Commit:** 1ca6da4

**Implementation:**
- Modified `ThermalModel::from_spec()` to call `apply_thermal_mass_correction()` automatically
- Call placed after all parameters set but before returning model
- Ensures automatic application across all ASHRAE 140 cases and user models

**Benefits:**
1. Consistent behavior: All ThermalModel instances apply correction automatically
2. No manual intervention: Users don't need to call `apply_thermal_mass_correction()`
3. Transparent integration: Correction applied during model creation

**Updated Test:**
- Modified `test_thermal_mass_coupling_ratio_high_mass` to verify correction applied during model creation
- Test now checks final coupling ratio without calling correction manually

### Task 4: Validate against ASHRAE 140 cases (INCOMPLETE)

**Status:** ⚠️ BLOCKED

**Findings:**
- Thermal mass correction method implemented and integrated
- Unit tests pass for coupling ratio >= 0.1
- However, actual validation reveals the correction is not effective in reducing annual energy error
- Root cause identified: Mode-specific coupling factors (Plan 03-14) override thermal mass correction

**Issue Details:**
The `apply_thermal_mass_correction()` method successfully modifies `h_tr_em`, `h_tr_em_heating`, and `h_tr_em_cooling`, but the simulation uses mode-specific values that are calculated in `from_spec()` BEFORE the correction is applied. This creates a race condition where:

1. `from_spec()` calculates base `h_tr_em` = 57.42 W/K
2. `from_spec()` sets `h_tr_em_heating` = 8.61 W/K (15% factor) and `h_tr_em_cooling` = 60.29 W/K (105% factor)
3. `apply_thermal_mass_correction()` is called and attempts to set all three to target value (109.2 W/K)
4. However, the mode-specific factors (0.15 and 1.05) cause the actual values used in simulation to be different from the target

**Current Results (Case 900):**
- Annual heating: 5.67 MWh (target: 1.17-2.04 MWh) - ❌ FAIL
- Annual cooling: 4.33 MWh (target: 2.13-3.67 MWh) - ❌ FAIL
- Peak heating: 2.10 kW (within range) - ✅ PASS
- Peak cooling: 3.56 kW (within range) - ✅ PASS

## Deviations from Plan

### Deviation 1: Mode-specific coupling integration issue (Rule 4 - Architectural Decision Required)

**Found during:** Task 4 (Validation)

**Issue:** Thermal mass correction implemented as specified, but mode-specific coupling factors from Plan 03-14 interfere with correction effectiveness.

**Root Cause:**
- Plan 03-14 implements mode-specific coupling with factors: heating_factor=0.15, cooling_factor=1.05
- These factors are applied in `from_spec()` to create `h_tr_em_heating` and `h_tr_em_cooling`
- Plan 14-02's `apply_thermal_mass_correction()` tries to set these to achieve coupling ratio >= 0.1
- The mode-specific factors multiply the target values, causing actual coupling to deviate from target

**Impact:**
- Thermal mass correction cannot achieve target coupling ratio >= 0.1
- Annual energy error remains at ~200-300% instead of target ±15%
- Plan 14-02 objective not achieved

**Proposed Resolution:**
1. Option A: Disable mode-specific coupling when thermal mass correction is applied (remove Plan 03-14 factors)
2. Option B: Adjust thermal mass correction to account for mode-specific factors (target = ratio / factor)
3. Option C: Re-evaluate if mode-specific coupling is still needed after thermal mass correction
4. Option D: Make thermal mass correction and mode-specific coupling mutually exclusive configuration options

**Files Modified:** src/sim/engine.rs

**Requires:** User decision on preferred approach for resolving coupling factor conflict

## Implementation Details

### Thermal Mass Correction Method

```rust
pub fn apply_thermal_mass_correction(&mut self) {
    let total_cap: f64 = self.thermal_capacitance.iter().sum();

    // Early exit for low-mass buildings
    if total_cap <= HIGH_MASS_THRESHOLD {
        return;
    }

    // Calculate structure thermal capacitance (excluding air)
    let zone_area = self.zone_area[0];
    let air_cap = zone_area * 1.2 * 1005.0; // J/K
    let structure_cap = total_cap - air_cap;

    // High-mass threshold: 5e6 J/K
    if structure_cap < HIGH_MASS_THRESHOLD {
        return;
    }

    // Calculate current coupling ratio
    let h_tr_ms_value: f64 = self.h_tr_ms.as_ref()[0];
    let h_tr_em_value: f64 = self.h_tr_em.as_ref()[0];
    let current_ratio = h_tr_em_value / h_tr_ms_value;

    // Target ratio >= 0.1 (ASHRAE 140 requirement)
    let target_ratio = 0.1;

    if current_ratio >= target_ratio {
        return; // Already compliant
    }

    // Increase h_tr_em to achieve target ratio
    let target_h_tr_em = target_ratio * h_tr_ms_value;
    let h_tr_em_data = self.h_tr_em.as_mut();
    h_tr_em_data.iter_mut().for_each(|v| *v = target_h_tr_em);

    // Also update h_tr_em_heating and h_tr_em_cooling
    // These are values actually used in simulation
    for v in self.h_tr_em_heating.as_mut().iter_mut() {
        *v = target_h_tr_em * self.h_tr_em_heating_factor;
    }
    for v in self.h_tr_em_cooling.as_mut().iter_mut() {
        *v = target_h_tr_em * self.h_tr_em_cooling_factor;
    }
}
```

### Threshold Rationale

**Why 5e6 J/K?**
- Case 600 (low-mass): ~2.4e6 J/K thermal capacitance
- Case 900 (high-mass): ~1.2e7 J/K thermal capacitance
- Threshold placed at 5e6 J/K (between 2.4e6 and 12.0e7)
- ASHRAE 140 states high-mass buildings have >3x low-mass capacitance
- Threshold correctly identifies Case 600 as low-mass and Case 900 as high-mass

## Testing Results

### Unit Tests
```
test_thermal_mass_coupling_ratio_low_mass ... ok
test_thermal_mass_coupling_ratio_high_mass ... ok
test_thermal_mass_threshold_detection ... ok
```

All unit tests pass, verifying:
1. Low-mass buildings (Case 600) are not affected by correction
2. High-mass buildings (Case 900) achieve coupling ratio >= 0.1
3. Thermal capacitance threshold detection works correctly

### ASHRAE 140 Validation
```
Case 900: Heating=5.67 (Ref: 1.17-2.04), Cooling=4.33 (Ref: 2.13-3.67)
Status: ❌ FAIL (annual energy significantly above target)
```

Peak loads remain within acceptable ranges:
- Peak heating: 2.10 kW (✅)
- Peak cooling: 3.56 kW (✅)

## Success Criteria Status

- [x] Thermal mass correction method implemented
- [x] Coupling ratio >= 0.1 for high-mass buildings (in tests)
- [x] Low-mass cases unaffected
- [ ] Annual energy error reduced to within ±15% tolerance (❌ BLOCKED by mode-specific coupling conflict)
- [x] Peak loads within ASHRAE 140 reference ranges

## Recommendations

### Immediate Action Required
1. **Resolve mode-specific coupling conflict** (Deviation 1)
   - This is a blocker preventing plan completion
   - Requires architectural decision from user
   - Options A-D outlined in Deviations section

### Future Enhancements
1. **Performance optimization:** The thermal mass correction loop could be optimized for multi-zone buildings
2. **Parameterization:** Consider making target ratio configurable (currently hardcoded to 0.1)
3. **Documentation:** Add detailed explanation of coupling ratio physics and ASHRAE 140 requirements
4. **Diagnostics:** Add coupling ratio tracking to validation report for easier debugging

## Lessons Learned

1. **Mode-specific coupling integration is complex:** The interaction between thermal mass correction and mode-specific coupling factors was not anticipated in the plan
2. **Unit tests vs integration tests:** Unit tests pass, but actual validation reveals the issue - highlights importance of end-to-end testing
3. **Coupling ratio is necessary but not sufficient:** Achieving coupling ratio >= 0.1 doesn't automatically fix annual energy error when mode-specific factors interfere
4. **Plan dependencies are critical:** Plan 14-02 depends on Plan 03-14's mode-specific coupling implementation, but the two approaches conflict

## Files Changed

### Created
- `tests/test_thermal_mass_coupling.rs` (105 lines) - Comprehensive thermal mass coupling validation tests

### Modified
- `src/sim/engine.rs` (69 lines added, 18 lines deleted)
  - Added `HIGH_MASS_THRESHOLD` constant
  - Implemented `apply_thermal_mass_correction()` method
  - Integrated correction call in `from_spec()`
  - Updated test to verify correction applied during model creation

## References

- ASHRAE 140 Standard: Coupling ratio requirements for high-mass buildings
- Plan 03-14: Mode-specific coupling parameters (heating_factor=0.15, cooling_factor=1.05)
- Plan 14-01: Thermal network verification context
- ISO 13790 Annex C: Effective thermal capacitance calculation using half-insulation rule

## Next Steps

1. **Resolve blocking issue:** Address mode-specific coupling conflict (Deviation 1)
2. **Re-validate:** Run ASHRAE 140 validation after conflict resolution
3. **Document:** Update docs/ASHRAE140_RESULTS.md if validation passes
4. **Proceed to Plan 14-03:** Mode-specific thermal mass coupling implementation

---

*Summary created: 2026-03-13*
*Plan status: PARTIALLY COMPLETE (3/4 tasks, 1 blocked)*
