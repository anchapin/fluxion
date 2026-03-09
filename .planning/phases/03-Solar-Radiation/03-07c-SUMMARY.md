---
phase: 03-Solar-Radiation
plan: 07c
subsystem: thermal-mass-calibration
tags: [thermal-mass, hvac-demand, sensitivity, calibration, investigation]

# Dependency graph
requires:
  - phase: 03-Solar-Radiation
    provides: Solar radiation integration (Plan 03-01), thermal mass dynamics (Plan 03-06), annual energy investigation (Plan 03-07)
provides:
  - Solar beam-to-mass fraction reverted to ASHRAE 140 specification (0.7)
  - Thermal mass conductance analysis and diagnostics
  - Investigation of thermal mass coupling impact on annual energy
  - Understanding of h_tr_em/h_tr_ms ratio effects on HVAC demand
affects:
  - Future thermal mass calibration strategies
  - Potential sensitivity calculation modifications for high-mass buildings

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Solar beam-to-mass fraction = 0.7 for high-mass buildings (ASHRAE 140 spec)
    - Thermal mass coupling enhancement factor tuning approach
    - h_tr_em/h_tr_ms ratio analysis for annual energy diagnosis
    - HVAC sensitivity calculation investigation for high-mass buildings

key-files:
  created:
    - tests/thermal_mass_calibration_diagnostics.rs - Thermal mass conductance analysis diagnostic test
    - .planning/phases/03-Solar-Radiation/03-07c-PLAN.md - Investigation plan
  modified:
    - src/sim/engine.rs - Reverted solar_beam_to_mass_fraction from 0.5 to 0.7

key-decisions:
  - "Solar beam-to-mass fraction reverted to 0.7 (ASHRAE 140 specification) - 0.5 made cooling worse (4.93→5.03 MWh), 0.7 improves cooling (4.93→4.82 MWh)"
  - "Increasing h_tr_em creates trade-off between heating and cooling - 2.0x enhancement: heating +22%, cooling -28%; 1.5x enhancement: heating +11%, cooling -14%"
  - "Thermal mass releases energy primarily to interior (h_tr_ms = 1092 W/K) not exterior (h_tr_em = 57.32 W/K) - This causes HVAC to work against mass energy release, increasing demand"
  - "Single-parameter tuning (h_tr_em enhancement alone) insufficient to resolve annual energy over-prediction - Requires more sophisticated approach (sensitivity modification or time constant correction)"
  - "h_tr_em / h_tr_ms ratio = 0.052 (very low) indicates weak exterior coupling - Sensitivity = 0.002065 K/W (very low, causing high HVAC demand)"

patterns-established:
  - "Solar distribution pattern: 0.7 beam-to-mass for high-mass, 0.0 solar-to-air (ASHRAE 140 spec)"
  - "Thermal mass coupling analysis: Compare h_tr_em/h_tr_ms ratio to identify energy flow paths"
  - "Calibration testing: Test multiple enhancement factors (1.15x, 1.5x, 2.0x) to understand trade-offs"
  - "Diagnostic-first approach: Analyze conductance ratios and sensitivity before implementing fixes"

requirements-completed: []

# Metrics
duration: 45min
completed: 2026-03-09
---

# Phase 3 Plan 07c: Thermal Mass Dynamics Investigation Summary

**Investigation of thermal mass conductances and their impact on annual energy over-prediction for Case 900 high-mass building.**

## Performance

- **Duration:** 45 min
- **Started:** 2026-03-09T20:38:39Z
- **Completed:** 2026-03-09T21:23:39Z
- **Tasks:** 3 (revert solar fraction, analyze conductances, calibrate thermal mass)
- **Files modified:** 2 (1 engine.rs modification, 1 new test file)

## Accomplishments

1. **Solar Beam-to-Mass Fraction Reverted**
   - Reverted solar_beam_to_mass_fraction from 0.5 to 0.7 for Case 900
   - 0.7 is the ASHRAE 140 specification value for high-mass buildings
   - Reverted change from Plan 03-07 that made cooling worse (4.93 → 5.03 MWh)
   - Reverting to 0.7 improves cooling from 4.93 to 4.82 MWh (2.2% improvement)
   - Heating remains at 6.86 MWh (similar to 6.84 MWh with 0.5)
   - Solar distribution validation test now passes (expects 0.7 per ASHRAE 140 spec)

2. **Thermal Mass Conductance Analysis**
   - Created comprehensive diagnostic test in `tests/thermal_mass_calibration_diagnostics.rs`
   - Analyzed h_tr_em and h_tr_ms conductances for Case 900
   - Calculated coupling ratios and sensitivity
   - Identified root cause of annual energy over-prediction

3. **Thermal Mass Coupling Calibration Investigation**
   - Tested multiple coupling_enhancement values: 1.15x, 1.5x, 2.0x
   - Measured impact on annual heating and cooling energy
   - Documented trade-offs between heating and cooling

## Task Commits

Each task was committed atomically:

1. **Task 1: Revert solar_beam_to_mass_fraction to 0.7** - `6566c7b` (fix)
   - Reverted solar_beam_to_mass_fraction from 0.5 to 0.7 for Case 900
   - 0.7 is ASHRAE 140 specification value for high-mass buildings
   - Plan 03-07 reduced to 0.5 but made cooling worse (4.93 → 5.03 MWh)
   - Reverting to 0.7 improves cooling from 4.93 to 4.82 MWh (2.2% improvement)
   - Solar distribution validation test now passes

2. **Task 2: Thermal mass conductance analysis** - `bddee7d` (test)
   - Added comprehensive diagnostic test for thermal mass conductances
   - Tests analyze h_tr_em, h_tr_ms, and their ratio
   - Diagnostics identify thermal mass coupling characteristics
   - Correlates conductance values with annual energy results

3. **Task 3: Thermal mass calibration investigation** - (no commit, findings documented in SUMMARY)
   - Tested coupling_enhancement from 1.15x to 1.5x and 2.0x
   - 2.0x enhancement: heating 6.86→8.40 MWh (+22%), cooling 4.82→3.45 MWh (-28%)
   - 1.5x enhancement: heating 6.86→7.61 MWh (+11%), cooling 4.82→4.15 MWh (-14%)
   - Pattern: Increasing h_tr_em reduces cooling but increases heating
   - Simple parameter tuning creates trade-off between heating and cooling

**Plan metadata:** (no final commit, findings documented in SUMMARY)

## Files Created/Modified

- `tests/thermal_mass_calibration_diagnostics.rs` - Thermal mass conductance analysis diagnostic test
  - test_case_900_thermal_mass_conductance_analysis: Analyzes h_tr_em, h_tr_ms, and sensitivity
  - Calculates coupling ratios and identifies thermal mass coupling characteristics
  - Provides recommendations for thermal mass calibration

- `src/sim/engine.rs` - Reverted solar_beam_to_mass_fraction
  - Line 1008: Changed from 0.5 to 0.7 for Case 900 and high-mass cases
  - Restores ASHRAE 140 specification value

- `.planning/phases/03-Solar-Radiation/03-07c-PLAN.md` - Investigation plan
  - Defines tasks for reverting solar fraction, analyzing conductances, calibrating thermal mass
  - Outlines approaches and expected outcomes

## Decisions Made

**Solar Beam-to-Mass Fraction Reverted to ASHRAE 140 Specification**
- Confirmed 0.7 is the correct value for high-mass buildings per ASHRAE 140
- Plan 03-07 reduced to 0.5 but that made cooling worse
- Reverting to 0.7 provides 2.2% improvement in cooling (4.93 → 4.82 MWh)
- Solar distribution validation test now passes (expects 0.7)
- Maintains ASHRAE 140 specification compliance

**Thermal Mass Conductance Analysis Confirms Hypothesis**
- h_tr_em / h_tr_ms ratio = 0.052 (very low)
- Thermal mass is strongly coupled to interior (h_tr_ms = 1092.00 W/K)
- Thermal mass is weakly coupled to exterior (h_tr_em = 57.32 W/K)
- Sensitivity = 0.002065 K/W (very low, causing high HVAC demand)
- Low sensitivity means HVAC has high demand (Power = ΔT / sensitivity)
- Root cause: Thermal mass releases stored energy primarily to interior, HVAC must work against this

**Thermal Mass Coupling Enhancement Creates Heating-Cooling Trade-off**
- Testing 2.0x enhancement: heating +22% (worse), cooling -28% (better)
- Testing 1.5x enhancement: heating +11% (worse), cooling -14% (better)
- Increasing h_tr_em allows thermal mass to release more energy to exterior
- This reduces cooling demand but increases heating demand (trade-off)
- Single-parameter tuning cannot resolve both heating and cooling over-prediction simultaneously

**Annual Energy Over-Prediction Requires More Sophisticated Approach**
- Simple parameter tuning (h_tr_em enhancement alone) insufficient
- Root cause is complex interaction between thermal mass coupling and HVAC sensitivity
- Thermal mass absorbs solar energy and releases it slowly to interior
- HVAC works against thermal mass energy release for 78.4% of hours
- May need:
  - Sensitivity calculation modification for high-mass buildings
  - Thermal mass time constant correction
  - Adjusting both h_tr_em and h_tr_ms together (complex interaction)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Solar beam-to-mass fraction incorrectly reduced to 0.5**
- **Found during:** Task 1 (revert solar fraction)
- **Issue:** Plan 03-07 reduced solar_beam_to_mass_fraction to 0.5, but this made cooling worse
- **Fix:** Reverted solar_beam_to_mass_fraction from 0.5 to 0.7 for Case 900
- **Files modified:** src/sim/engine.rs (line 1008)
- **Verification:** Cooling improved from 4.93 to 4.82 MWh (2.2% improvement)
- **Committed in:** 6566c7b (Task 1 commit)
- **Impact:** Restores ASHRAE 140 specification value, improves annual cooling

**2. [Rule 4 - Architectural] Annual energy over-prediction requires more sophisticated approach**
- **Found during:** Task 3 (thermal mass calibration)
- **Issue:** Single-parameter tuning (h_tr_em enhancement) creates heating-cooling trade-off
  - 2.0x enhancement: heating +22%, cooling -28%
  - 1.5x enhancement: heating +11%, cooling -14%
- **Proposed change:** Modify sensitivity calculation for high-mass buildings or implement thermal mass time constant correction
- **Why needed:** Current sensitivity (0.002065 K/W) is too low for high-mass, causing high HVAC demand
- **Impact:** Would allow HVAC demand calculation to account for thermal mass time constant effects
- **Alternatives:** Adjust both h_tr_em and h_tr_ms together, or calibrate against ASHRAE 140 reference implementation
- **Decision:** Not implemented in this plan (requires deeper investigation or architectural change)
- **Files modified:** None (findings documented in SUMMARY)
- **Verification:** Annual energies still outside reference ranges (heating 6.86 MWh, cooling 4.82 MWh)
- **Committed in:** None (documented in SUMMARY)
- **Impact:** Identified that simple parameter tuning is insufficient, needs more sophisticated approach

---

**Total deviations:** 1 auto-fixed (1 bug), 1 architectural (not implemented, documented)
**Impact on plan:** Solar fraction reverted successfully, but annual energy objective not achieved. Root cause identified (thermal mass coupling dynamics) but resolution requires more sophisticated approach beyond single-parameter tuning.

## Issues Encountered

**Thermal Mass Coupling Enhancement Creates Heating-Cooling Trade-off**
Multiple thermal mass coupling enhancement values were tested to resolve annual energy over-prediction:

1. **Baseline (1.15x enhancement):**
   - Annual heating: 6.86 MWh (239-491% above reference [1.17, 2.04] MWh)
   - Annual cooling: 4.82 MWh (28-126% above reference [2.13, 3.67] MWh)

2. **1.5x Enhancement:**
   - Annual heating: 7.61 MWh (275-551% above reference) - 11% worse than baseline
   - Annual cooling: 4.15 MWh (13-95% above reference) - 14% better than baseline
   - **Result:** Heating gets worse, cooling improves - trade-off

3. **2.0x Enhancement:**
   - Annual heating: 8.40 MWh (313-619% above reference) - 22% worse than baseline
   - Annual cooling: 3.45 MWh (within [2.13, 3.67] MWh reference!) - 28% better than baseline
   - **Result:** Heating much worse, cooling at reference - trade-off

**Root Cause Analysis:**

The low h_tr_em / h_tr_ms ratio (0.052) means thermal mass releases stored energy primarily to interior (h_tr_ms = 1092 W/K) rather than to exterior (h_tr_em = 57.32 W/K). This causes HVAC to work against thermal mass energy release, increasing annual energy demand.

Increasing h_tr_em allows thermal mass to release more energy to exterior, which reduces cooling demand (less stored energy to cool) but increases heating demand (thermal mass can't retain heat as effectively).

**Diagnostic Findings:**
- Peak loads correct: heating 2.10 kW, cooling 3.54 kW (within tolerance)
- Annual energy over-predicted: heating 6.86 MWh, cooling 4.82 MWh
- HVAC demand often at maximum: average heating 1779 W (close to 2100 W max)
- Thermal mass temperature swing large: 10.16°C to 35.90°C (25.74°C range)
- Mass temp outside setpoints: 76.6% of time (2552 hours below heating, 4164 hours above cooling)
- Sensitivity = 0.002065 K/W (very low, causing high HVAC demand)

**Analysis:**

The pattern of correct peak loads but over-predicted annual energy indicates:
1. HVAC is responding correctly to extreme conditions (peak loads)
2. HVAC is over-responding to moderate conditions (running at high demand when not needed)
3. Thermal mass is absorbing solar energy and releasing it slowly to interior
4. HVAC must work against thermal mass energy release constantly
5. Sensitivity calculation or thermal mass coupling may not account for time constant effects correctly

**Resolution Path Forward:**

Simple parameter tuning (h_tr_em enhancement alone) is insufficient to resolve annual energy over-prediction without creating trade-offs between heating and cooling. Requires more sophisticated approach:

1. **Sensitivity Calculation Modification:**
   - Add thermal mass time constant factor to sensitivity for high-mass buildings
   - This would reduce HVAC demand when thermal mass provides buffering
   - Risk: Affects physics accuracy, may break other cases

2. **Thermal Mass Time Constant Correction:**
   - Implement time constant-based correction for HVAC demand calculation
   - Account for thermal mass response time in demand calculation
   - Risk: Complex to implement and validate

3. **Calibrate Both h_tr_em and h_tr_ms Together:**
   - Adjust conductances together to achieve balanced coupling
   - Need optimization or search to find optimal values
   - Risk: Complex interaction, hard to predict

4. **Calibration Against ASHRAE 140 Reference:**
   - Compare detailed behavior with reference implementation
   - Identify specific calculation discrepancies
   - Risk: Requires access to reference implementation source

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Solar distribution corrected:** Solar beam-to-mass fraction reverted to 0.7 (ASHRAE 140 spec), validation test passing.

**Thermal mass dynamics investigated:** Root cause of annual energy over-prediction identified (thermal mass coupling dynamics, low sensitivity).

**Annual energy objective not achieved:** Current values:
- Heating: 6.86 MWh vs [1.17, 2.04] MWh reference (239-491% above)
- Cooling: 4.82 MWh vs [2.13, 3.67] MWh reference (28-126% above)

**Blockers:**
1. Single-parameter tuning (h_tr_em enhancement) creates heating-cooling trade-off
2. Annual energy over-prediction requires more sophisticated approach
3. May need sensitivity calculation modification or thermal mass time constant correction
4. Trade-off between temperature swing reduction and HVAC energy accuracy not resolved

**Recommendations for Future Work:**

1. **Investigate Sensitivity Calculation for High-Mass Buildings:**
   - Current sensitivity (0.002065 K/W) is too low
   - Modify sensitivity to account for thermal mass time constant
   - Test with multiple ASHRAE 140 cases to ensure no regressions

2. **Implement Thermal Mass Time Constant Correction:**
   - Calculate thermal mass time constant (τ = C / (h_tr_em + h_tr_ms))
   - Apply correction factor based on time constant vs timestep
   - Reduce HVAC demand when thermal mass provides buffering

3. **Multi-Parameter Optimization:**
   - Use optimization or search to find optimal h_tr_em and h_tr_ms values
   - Optimize for both heating and cooling annual energy accuracy
   - Validate against peak loads and temperature swing constraints

4. **Comparison with ASHRAE 140 Reference:**
   - Obtain detailed hour-by-hour data from reference implementation
   - Compare thermal mass temperatures, HVAC demand, sensitivity calculation
   - Identify specific discrepancies in calculation approach

5. **Consider 6R2C Model Re-evaluation:**
   - Plan 03-07 disabled 6R2C for Case 900 with minimal improvement
   - May need to re-evaluate 6R2C parameterization or calculation
   - Two-mass-node model may better capture thermal mass dynamics

---
*Phase: 03-Solar-Radiation*
*Completed: 2026-03-09*
