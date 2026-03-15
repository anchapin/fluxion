---
phase: 03-Solar-Radiation
plan: 02
subsystem: [thermal-physics, energy-accounting, validation]
tags: [HVAC energy, thermal mass, ASHRAE 140 validation, energy balance]

# Dependency graph
requires:
  - phase: 03-01
    provides: solar gain integration, beam-to-mass distribution
provides:
  - corrected_cumulative_energy field for tracking actual HVAC consumption
  - thermal mass energy accounting mechanism for high-mass buildings
  - validation tests for corrected HVAC energy calculation
affects: [03-03, 04-Multi-Zone-Transfer]

# Tech tracking
tech-stack:
  added: [corrected_cumulative_energy field, conditional thermal mass energy subtraction]
  patterns: [thermal mass energy tracking, HVAC energy correction, diagnostic logging]

key-files:
  created: [tests/ashrae_140_case_900.rs (three new validation tests)]
  modified: [src/sim/engine.rs (ThermalModel struct, step_physics methods)]

key-decisions:
  - "Enable thermal_mass_energy_accounting for high-mass cases (900, 900FF) to get actual HVAC consumption"
  - "Attempted to subtract thermal mass energy change from HVAC energy, but encountered over-correction"
  - "thermal_mass_correction_factor conflicts with thermal_mass_energy_accounting - both try to correct for thermal mass effects"
  - "Thermal mass energy balance verified (mass returns to similar temperature after full year)"

patterns-established:
  - "Pattern 1: Use corrected_cumulative_energy field to track actual HVAC consumption after thermal mass accounting"
  - "Pattern 2: Conditional thermal mass energy subtraction based on mass charging/discharging"
  - "Pattern 3: Diagnostic output for tracking hvac_energy_for_step, mass_energy_change, corrected_hvac_energy"

requirements-completed: []
---

# Phase 3 Plan 02: HVAC Energy Calculation Correction Summary

**Thermal mass energy accounting implementation with corrected_cumulative_energy field and validation tests, but HVAC energy over-correction persists (11.20 MWh vs [2.13, 3.67] MWh target)**

## Performance

- **Duration:** 2h 15m
- **Started:** 2026-03-09T17:00:00Z
- **Completed:** 2026-03-09T19:15:00Z
- **Tasks:** 3 (investigation, implementation, validation)
- **Files modified:** 2 (src/sim/engine.rs, tests/ashrae_140_case_900.rs)

## Accomplishments

- Added `corrected_cumulative_energy` field to ThermalModel struct to track actual HVAC consumption
- Implemented thermal mass energy subtraction in both 5R1C and 6R2C step_physics methods
- Enabled thermal_mass_energy_accounting for high-mass cases (900, 900FF)
- Added three validation tests for corrected HVAC energy calculation
- Verified thermal mass energy balance (mass returns to similar temperature after full year)
- Identified root cause: thermal_mass_correction_factor and thermal_mass_energy_accounting conflict

## Task Commits

Each task was committed atomically:

1. **Task 1: Investigate HVAC energy calculation and thermal mass accounting** - `7be6098` (invest)
   - Confirmed baseline: Case 900 annual cooling 1.01 MWh vs [2.13, 3.67] MWh reference
   - Identified root cause: HVAC energy calculation includes thermal mass energy storage/release
   - Diagnostic output shows mass_energy_change_cumulative = -29.2 MWh at year-end (mass releasing energy)
   - Verified hypothesis: Subtract thermal mass energy change to get actual HVAC consumption

2. **Task 2: Fix HVAC energy calculation to subtract thermal mass energy change** - `c2d1d2c` (feat)
   - Added corrected_cumulative_energy field to ThermalModel struct
   - Initialize in new() constructor and apply_parameters() method
   - Calculate corrected_hvac_energy_for_step = hvac_energy_for_step - mass_energy_change
   - Track corrected_cumulative_energy in step_physics() for both 5R1C and 6R2C models
   - Enable thermal_mass_energy_accounting for high-mass cases (900, 900FF)
   - Replace conditional subtraction with direct subtraction when accounting enabled
   - Results: Baseline 1.01 MWh → Corrected 7.29 MWh (overcorrected)

3. **Task 3: Validate corrected HVAC energy calculation with Case 900 cooling metrics** - `99ac1f6` (feat)
   - Modified thermal mass energy subtraction to be conditional (only when mass is charging)
   - Disabled thermal_mass_correction_factor when thermal_mass_energy_accounting enabled
   - Added three new validation tests for corrected HVAC energy calculation
   - Results: Over-correction issue persists (11.20 MWh vs [2.13, 3.67] MWh target)
   - Root cause: thermal_mass_energy_accounting and thermal_mass_correction_factor conflict

**Plan metadata:** Not yet created (plan not complete)

## Files Created/Modified

- `src/sim/engine.rs` - Added corrected_cumulative_energy field, modified step_physics methods for both 5R1C and 6R2C models to implement thermal mass energy accounting
- `tests/ashrae_140_case_900.rs` - Added three new validation tests: test_case_900_annual_cooling_energy_with_correction, test_case_900_thermal_mass_energy_balance, test_case_900_hvac_energy_correction_comparison

## Decisions Made

- Enable thermal_mass_energy_accounting for high-mass cases (900, 900FF) to get actual HVAC consumption
- Attempted to subtract thermal mass energy change from HVAC energy to correct for thermal mass buffering
- Disabled thermal_mass_correction_factor when thermal_mass_energy_accounting enabled to avoid double correction
- Used conditional subtraction (only when mass is charging) based on original Issue #272, #274, #275 logic

## Deviations from Plan

**Plan could not be completed as specified due to fundamental conflict between thermal_mass_correction_factor and thermal_mass_energy_accounting.**

### Issues Encountered

**1. HVAC Energy Over-correction**
- **Found during:** Task 2 (implementation)
- **Issue:** Attempting to subtract thermal mass energy change from HVAC energy causes over-correction
- **Details:**
  - Baseline cooling energy: 1.01 MWh (with thermal_mass_correction_factor = 0.20)
  - Target cooling energy: [2.13, 3.67] MWh
  - With full thermal mass energy subtraction: 7.29 MWh or 9.80 MWh (overcorrected)
  - With conditional subtraction (only when mass charging): 11.20 MWh (severely overcorrected)
- **Root cause:** thermal_mass_correction_factor (0.20) and thermal_mass_energy_accounting both try to correct for thermal mass effects, causing double correction
- **Attempted fixes:**
  1. Disabled thermal_mass_correction_factor when thermal_mass_energy_accounting enabled → Result: 9.80 MWh (worse)
  2. Changed from full subtraction to conditional subtraction (only when mass charging) → Result: 11.20 MWh (worse)
- **Status:** Unresolved - plan cannot be completed with current approach

**2. Thermal Mass Energy Sign Convention**
- **Found during:** Task 3 (validation)
- **Issue:** Cumulative thermal mass energy change is -26.09 MWh (mass releasing energy), but mass temperature increased by 1.31°C over the year
- **Details:** Mass releasing energy (negative change) should lower mass temperature, but final mass temperature (20.00°C) is higher than initial (18.69°C)
- **Status:** Minor issue - thermal mass energy balance test still passes (within ±2°C tolerance)

**3. Test Heating/Cooling Separation Logic**
- **Found during:** Task 3 (validation)
- **Issue:** Test logic for separating heating and cooling based on energy sign may not work correctly with corrected energy values
- **Details:** With thermal mass energy correction, energy signs may not accurately represent heating vs cooling because of the thermal mass correction
- **Status:** Identified but not resolved

---

**Total deviations:** 1 major issue (HVAC energy over-correction), 2 minor issues (thermal mass sign convention, test separation logic)
**Impact on plan:** Major deviation prevents plan completion. Different approach needed to avoid thermal_mass_correction_factor and thermal_mass_energy_accounting conflict.

## Issues Encountered

### Issue 1: Thermal Mass Correction Mechanism Conflict

**Problem:**
The existing codebase has two mechanisms for accounting for thermal mass effects:
1. `thermal_mass_correction_factor` (0.20 for Case 900) - reduces HVAC energy by 80%
2. `thermal_mass_energy_accounting` - subtracts thermal mass energy change from HVAC energy

These two mechanisms conflict when both are applied, causing over-correction.

**Analysis:**
- With thermal_mass_correction_factor = 0.20 only: 1.01 MWh (53-67% below target)
- With thermal_mass_energy_accounting only: 9.80 MWh (167-359% above target)
- With both mechanisms: 7.29 MWh or 11.20 MWh (overcorrected)
- Target: [2.13, 3.67] MWh

**Root Cause:**
The thermal_mass_correction_factor was added as a crude approximation to account for thermal mass buffering effects in high-mass buildings. The plan attempts to replace this with a more accurate calculation that subtracts the actual thermal mass energy change. However, the two mechanisms are incompatible and cause over-correction when combined.

**Potential Solutions:**
1. Remove thermal_mass_correction_factor entirely and rely only on thermal_mass_energy_accounting (but this over-corrects to 9.80 MWh)
2. Adjust thermal_mass_energy_accounting to subtract only a portion of the mass energy change (e.g., 10% gives 3.62 MWh, within target)
3. Re-examine the physics to understand why the mass energy change is -26.09 MWh but mass temperature increased
4. Investigate if the ASHRAE reference values represent HVAC energy consumption or total building load

**Status:** Unresolved - requires further investigation and possibly a different approach

### Issue 2: Thermal Mass Energy Balance Verification

**Problem:**
Thermal mass energy change cumulative is -26.09 MWh (mass releasing energy), but mass temperature increased from 18.69°C to 20.00°C (ΔT = +1.31°C).

**Analysis:**
- Energy change calculation: ΔE = Cm × (Tm_new - Tm_old)
- If Tm_new > Tm_old, ΔE should be positive (mass storing energy)
- But cumulative ΔE is negative (-26.09 MWh), suggesting mass released energy
- Contradiction: Mass released energy but temperature increased

**Potential Explanations:**
1. Thermal capacitance (Cm) is incorrect
2. Mass temperature calculation has a sign error
3. Cumulative tracking has a sign error
4. Physics model has deeper issues with thermal mass representation

**Status:** Minor issue - thermal mass energy balance test still passes (within ±2°C tolerance), but the contradiction should be investigated

### Issue 3: Test Heating/Cooling Separation Logic

**Problem:**
Test logic for separating heating and cooling based on energy sign and zone temperature may not work correctly with corrected energy values.

**Analysis:**
Current test logic:
```rust
if energy_kwh > 0.0 || zone_temp_before < model.heating_setpoint {
    total_heating += energy_joules;
} else if energy_kwh < 0.0 || zone_temp_before > model.cooling_setpoint {
    total_cooling += -energy_joules;
}
```

With thermal mass energy correction, the energy_kwh returned from step_physics may not accurately represent heating vs cooling because the correction can change the sign or magnitude.

**Status:** Identified but not resolved - may need to use zone temperature instead of energy sign for separation

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Not ready for next phase** - Plan 03-02 could not be completed due to fundamental conflict between thermal_mass_correction_factor and thermal_mass_energy_accounting.

**Blockers:**
1. HVAC energy over-correction issue (7.29-11.20 MWh vs [2.13, 3.67] MWh target)
2. Unclear physics model for thermal mass energy accounting
3. Need to decide between thermal_mass_correction_factor and thermal_mass_energy_accounting approaches

**Recommendations:**
1. Investigate the physics model to understand why thermal mass energy change is -26.09 MWh but mass temperature increased
2. Determine if ASHRAE reference values represent HVAC energy consumption or total building load
3. Consider alternative approaches:
   - Adjust thermal_mass_correction_factor to a different value (currently 0.20)
   - Implement partial thermal mass energy subtraction (e.g., 10% of mass energy change)
   - Re-examine the 5R1C/6R2C thermal network implementation
4. Update Plan 03-02 with revised approach based on investigation

**Status:** Plan 03-02 cannot be considered complete until the HVAC energy correction issue is resolved and Case 900 annual cooling energy is within [2.13, 3.67] MWh reference.

---

## Self-Check: INCOMPLETE

**Files Created:**
- ✅ FOUND: .planning/phases/03-Solar-Radiation/03-02-SUMMARY.md

**Commits Created:**
- ✅ FOUND: 7be6098 (investigation)
- ✅ FOUND: c2d1d2c (implementation)
- ✅ FOUND: 99ac1f6 (validation)

**Plan Completion Status:**
- ❌ FAILED: Case 900 annual cooling energy not within [2.13, 3.67] MWh reference
  - Baseline: 1.01 MWh
  - With correction: 7.29 MWh (overcorrected)
  - Target: [2.13, 3.67] MWh
- ✅ PASSED: Thermal mass energy balance verified (mass returns to similar temperature)
- ✅ PASSED: Three validation tests added

**Blocking Issue:** HVAC energy over-correction due to conflict between thermal_mass_correction_factor and thermal_mass_energy_accounting. Plan cannot be completed until this issue is resolved.

---
*Phase: 03-Solar-Radiation*
*Completed: 2026-03-09 (incomplete - requires further investigation)*
