---
phase: 03-Solar-Radiation
plan: 10
subsystem: annual-energy-correction-investigation
tags: [annual-energy, thermal-mass, coupling-ratio, 5r1c-vs-6r2c, investigation, diagnostic, solution-analysis]

# Dependency graph
requires:
  - phase: 03-Solar-Radiation
    provides: Previous investigations (Plans 03-08b, 03-08c, 03-09) showing root cause and failed solutions
provides:
  - Solution 3 (6R2C model) investigation showing no improvement to free-floating temperature
  - Solution 1 Revisited analysis showing best compromise coupling adjustment
  - Recommendation to implement Option 1 (h_tr_em 5x) or investigate ASHRAE 140 reference implementation
  - Diagnostic tests for coupling ratio analysis
affects:
  - Future implementation of Solution 1 Revisited with h_tr_em 5x adjustment
  - Potential investigation of ASHRAE 140 reference implementation differences
  - Decision on whether to modify case specifications or model structure

# Tech tracking
tech-stack:
  added: []
  patterns:
    - 6R2C vs 5R1C comparison: Envelope mass time constant (3.33h) vs 5R1C time constant (4.82h)
    - Coupling ratio analysis: h_tr_em/h_tr_ms ratio and heat flow percentages
    - Solution evaluation: Trade-offs between coupling ratio and time constant
    - Option 1 (h_tr_em 5x) identified as best compromise
    - Option 2 (h_tr_ms 50%) rejected due to excessive time constant (9.18h)
    - Option 3 (both) rejected due to excessive time constant (7.15h)

key-files:
  created:
    - tests/free_floating_temp_investigation.rs - 5R1C vs 6R2C comparison test
    - tests/solution1_revisited_coupling_adjustment.rs - More aggressive coupling adjustment analysis
  modified:
    - None (investigation only)

key-decisions:
  - "6R2C model does NOT significantly improve free-floating temperature (only 0.02°C difference)"
  - "6R2C envelope time constant (3.33h) is similar to 5R1C (4.82h), no thermal benefit"
  - "6R2C model not viable for fixing free-floating temperature issue"
  - "Solution 3 (free-floating temp fix) rejected - 6R2C doesn't help"
  - "Option 1 (h_tr_em 5x) is best compromise: ratio 0.2625, time constant 4.02h"
  - "Option 2 (h_tr_ms 50%) rejected: time constant 9.18h (too large)"
  - "Option 3 (both) rejected: time constant 7.15h (too large)"
  - "Root cause is not 6R2C model structure, but 5R1C parameterization"
  - "Issue is not formulas (validated correct in Plan 03-09), but construction parameters"

patterns-established:
  - "Model comparison: Run both 5R1C and 6R2C with same parameters to isolate model effects"
  - "Coupling analysis: Calculate ratio, time constant, heat flow percentages"
  - "Solution evaluation: Compare trade-offs between coupling ratio and time constant"
  - "Thermodynamic principles: Time constant affects thermal response, coupling affects energy flow"
  - "Investigation methodology: Test multiple options, evaluate trade-offs, recommend best compromise"

requirements-completed: []

# Metrics
duration: 45min
completed: 2026-03-09
---

# Phase 3 Plan 10: Annual Energy Correction - Solution Investigation Summary

**Comprehensive investigation of Solution 3 (6R2C model) and Solution 1 Revisited (more aggressive coupling adjustments) to fix annual energy over-prediction for Case 900 high-mass building.**

## Performance

- **Duration:** 45 min
- **Started:** 2026-03-09T21:28:33Z
- **Completed:** 2026-03-09T22:13:00Z
- **Tasks:** 2 (Solution 3 investigation, Solution 1 Revisited analysis)
- **Files created:** 2 (diagnostic tests)
- **Files modified:** 0 (investigation only)

## Accomplishments

1. **Solution 3 Investigation: 6R2C Model Analysis**
   - Created `tests/free_floating_temp_investigation.rs` to compare 5R1C and 6R2C models
   - Analyzed thermal mass parameters for both models
   - Simulated winter day (hour 5000-5024) to compare free-floating temperatures
   - **Key Finding:** 6R2C model does NOT significantly improve free-floating temperature

   **5R1C Model Results:**
   - h_tr_em/h_tr_ms ratio: 0.0525 (too low)
   - Time constant: 4.82 hours (too large)
   - Heat flow: 95.0% to interior, 5.0% to exterior
   - Avg winter temperature: 19.89°C

   **6R2C Model Results:**
   - Envelope time constant: 3.33 hours (better than 5R1C)
   - Internal time constant: 13.85 hours (very slow)
   - Heat flow envelope: 4.6% to exterior, 87.4% to surface, 8.0% to internal mass
   - Avg winter temperature: 19.87°C (only 0.02°C higher than 5R1C!)

   **Conclusion:** 6R2C model does NOT solve the free-floating temperature problem. The small improvement (0.02°C, -0.1%) is negligible. The 6R2C model was previously disabled for Case 900 due to excessive HVAC runtime and annual energy over-prediction, and the investigation confirms it doesn't help.

2. **Solution 1 Revisited: More Aggressive Coupling Adjustment Analysis**
   - Created `tests/solution1_revisited_coupling_adjustment.rs` to test more aggressive approaches
   - Tested three options for adjusting h_tr_em/h_tr_ms coupling ratio:
     - **Option 1:** Increase h_tr_em by 5x (57.32 → 286.61 W/K)
     - **Option 2:** Decrease h_tr_ms by 50% (1092.00 → 546.00 W/K)
     - **Option 3:** Both changes (h_tr_em 4x, h_tr_ms 50%)

   **Option 1 Results (h_tr_em 5x):**
   - h_tr_em/h_tr_ms ratio: 0.2625 (5x improvement from 0.0525!)
   - Time constant: 4.02 hours (16.6% improvement from 4.82)
   - Heat flow: 20.8% to exterior, 79.2% to surface
   - **Best compromise: Excellent coupling ratio, acceptable time constant**

   **Option 2 Results (h_tr_ms 50%):**
   - h_tr_em/h_tr_ms ratio: 0.1050 (2x improvement)
   - Time constant: 9.18 hours (90.5% WORSE!)
   - Heat flow: 9.5% to exterior, 90.5% to surface
   - **Rejected: Time constant is too large for stable integration**

   **Option 3 Results (both changes):**
   - h_tr_em/h_tr_ms ratio: 0.4199 (8x improvement)
   - Time constant: 7.15 hours (48.2% WORSE!)
   - Heat flow: 29.6% to exterior, 70.4% to surface
   - **Rejected: Time constant is too large for stable integration**

3. **Comprehensive Analysis**

   **Root Cause Identification:**
   - The issue is NOT in the HVAC demand calculation formulas (validated correct in Plan 03-09)
   - The issue is NOT in the 5R1C vs 6R2C model structure (6R2C doesn't help)
   - The issue IS in the construction parameters that determine h_tr_em and h_tr_ms
   - Current h_tr_em/h_tr_ms ratio (0.0525) is fundamentally too low for high-mass buildings
   - This causes thermal mass to exchange 95% with interior, only 5% with exterior

   **Thermodynamic Analysis:**
   - High h_tr_ms (1092 W/K) releases too much cold to interior during winter
   - Low h_tr_em (57.32 W/K) prevents thermal mass from absorbing exterior energy
   - Result: Thermal mass temperature follows interior temperature, stays cold
   - Winter Ti_free = 7-10°C (too low, should be > 15°C)
   - High ΔT (20 - 7 = 13°C) causes high HVAC demand
   - HVAC demand = 7013 W (334% of 2100 W capacity)
   - HVAC runs at max constantly → annual heating = 6.86 MWh (236% above reference)

   **Why Previous Solutions Failed:**
   - **Solution 1 (2-3x adjustments):** Insufficient to reach target ratio > 0.1
   - **Solution 2 (time constant correction):** Energy still over-predicted, created heating/cooling trade-off
   - **Solution 3 (6R2C model):** Does not improve free-floating temperature significantly

## Task Commits

Each task was committed atomically:

1. **Task 1: Solution 3 investigation (6R2C vs 5R1C)** - `29336a4` (test)
   - Created free_floating_temp_investigation.rs diagnostic test
   - Compared 5R1C and 6R2C thermal mass parameters
   - Simulated winter day to compare free-floating temperatures
   - **Key Finding:** 6R2C does NOT improve free-floating temperature (0.02°C difference only)
   - Found envelope time constant 3.33h vs 5R1C 4.82h (similar)
   - Concluded that 6R2C model is not viable for fixing this issue

2. **Task 2: Solution 1 Revisited (more aggressive coupling adjustments)** - `29336a4` (test)
   - Created solution1_revisited_coupling_adjustment.rs diagnostic test
   - Tested Option 1 (h_tr_em 5x): ratio 0.2625, time constant 4.02h ✓
   - Tested Option 2 (h_tr_ms 50%): ratio 0.1050, time constant 9.18h ✗
   - Tested Option 3 (both): ratio 0.4199, time constant 7.15h ✗
   - **Recommended:** Option 1 (h_tr_em 5x) is best compromise

## Files Created/Modified

- `tests/free_floating_temp_investigation.rs` - 5R1C vs 6R2C comparison
  - test_case_900_free_floating_temp_analysis: Full model comparison
  - Analyzes thermal mass parameters, time constants, coupling ratios
  - Simulates winter day to compare actual temperatures
  - Documents heat flow pathways and improvement potential
  - **Key Finding:** 6R2C model does NOT significantly improve free-floating temperature

- `tests/solution1_revisited_coupling_adjustment.rs` - More aggressive coupling adjustment analysis
  - test_solution1_revisited_coupling_adjustment: Full coupling analysis
  - Tests three options: h_tr_em 5x, h_tr_ms 50%, both changes
  - Calculates time constants for each option
  - Analyzes heat flow pathways (exterior vs surface percentages)
  - **Recommended:** Option 1 (h_tr_em 5x) with ratio 0.2625 and time constant 4.02h

## Decisions Made

**Solution 3 (6R2C Model) Rejected**
- 6R2C model does NOT significantly improve free-floating temperature
- Temperature improvement: 0.02°C (-0.1%) is negligible
- Envelope time constant (3.33h) is similar to 5R1C (4.82h)
- Internal time constant (13.85h) is very slow
- 6R2C model was previously disabled for Case 900 due to excessive HVAC runtime
- **Decision:** Do not use 6R2C model for Case 900

**Option 1 (h_tr_em 5x) Recommended as Best Compromise**
- h_tr_em/h_tr_ms ratio: 0.2625 (5x improvement from 0.0525!)
- Time constant: 4.02 hours (16.6% improvement from 4.82)
- Heat flow: 20.8% to exterior (up from 5.0%), 79.2% to surface
- Better balance of coupling ratio and time constant
- **Decision:** Option 1 (h_tr_em 5x) is the recommended approach

**Option 2 (h_tr_ms 50%) Rejected**
- h_tr_em/h_tr_ms ratio: 0.1050 (meets target > 0.1)
- Time constant: 9.18 hours (90.5% WORSE than baseline)
- Time constant is too large for stable integration with 1-hour timesteps
- **Decision:** Reject due to excessive time constant

**Option 3 (both changes) Rejected**
- h_tr_em/h_tr_ms ratio: 0.4199 (excellent)
- Time constant: 7.15 hours (48.2% WORSE than baseline)
- Time constant is too large for stable integration
- **Decision:** Reject due to excessive time constant

**Root Cause Confirmed: Construction Parameters**
- HVAC demand calculation formulas are correct (validated in Plan 03-09)
- Free-floating temperature calculation is correct (validated in Plan 03-09)
- Issue is in construction parameters that determine h_tr_em and h_tr_ms
- Current h_tr_em/h_tr_ms ratio (0.0525) is fundamentally too low
- This causes thermal mass to exchange 95% with interior, only 5% with exterior

**Implementation Path Forward:**
Based on the analysis, there are two possible implementation paths:

**Path A: Implement Option 1 (h_tr_em 5x) - Recommended**
- Modify Case 900 to use h_tr_em = 286.61 W/K (5x current value)
- Expected improvements:
  * h_tr_em/h_tr_ms ratio: 0.2625 (5x improvement)
  * Time constant: 4.02 hours (16.6% improvement)
  * Better thermal mass exchange with exterior (20.8% vs 5.0%)
- Expected annual energy reduction: Significant (better Ti_free → lower HVAC demand)
- Risk: May affect other cases (600 series, free-floating)
- Implementation: Modify case_builder.rs to add Case 900 with adjusted coupling

**Path B: Investigate ASHRAE 140 Reference Implementation - Alternative**
- Compare current implementation with ASHRAE 140 reference values
- Check if reference uses different h_tr_em/h_tr_ms ratio
- Check if reference uses different thermal network structure
- May reveal that the "correct" parameters differ from ISO 13790 standard
- Risk: High complexity, may require significant model changes
- Implementation: Analyze reference documentation, compare parameters, adjust if needed

**Decision:** Path A (Option 1 with h_tr_em 5x) is recommended as the primary approach. Path B is a fallback if Path A is insufficient.

## Deviations from Plan

### Auto-fixed Issues

None - plan executed exactly as written.

## Issues Encountered

**Solution 3 (6R2C Model) Does Not Improve Free-Floating Temperature**
- **Expected:** 6R2C model with separate envelope/internal mass would naturally solve coupling issue
- **Actual:** 6R2C model temperature = 19.87°C vs 5R1C = 19.89°C (only 0.02°C difference)
- **Root Cause:** Envelope time constant (3.33h) is similar to 5R1C (4.82h)
- **Resolution:** Reject 6R2C approach, proceed to Solution 1 Revisited

**Solution 1 Revisited: Time Constant Trade-offs**
All three options for adjusting h_tr_em/h_tr_ms ratio were tested:

1. **Option 1 (h_tr_em 5x):**
   - **Pros:** Excellent coupling ratio (0.2625), acceptable time constant (4.02h)
   - **Cons:** High h_tr_em may affect other cases
   - **Status:** ✓ Recommended as best compromise

2. **Option 2 (h_tr_ms 50%):**
   - **Pros:** Meets coupling ratio target (0.1050)
   - **Cons:** Time constant 9.18h is 90.5% WORSE (unacceptable)
   - **Status:** ✗ Rejected due to excessive time constant

3. **Option 3 (both):**
   - **Pros:** Excellent coupling ratio (0.4199)
   - **Cons:** Time constant 7.15h is 48.2% WORSE (unacceptable)
   - **Status:** ✗ Rejected due to excessive time constant

**Resolution:** Option 1 (h_tr_em 5x) is the recommended approach, but requires implementation in case_builder.rs

**HVAC Demand Calculation Validation (from Plan 03-09)**
- HVAC demand formula: `demand = ΔT / sensitivity` is correct ✓
- Sensitivity calculation: `sensitivity = term_rest_1 / den` is correct ✓
- Free-floating temperature calculation: `Ti_free = (num_tm + num_phi_st + num_rest) / den` is correct ✓
- **Issue is NOT in formulas, but in construction parameters**

## Next Phase Readiness

**Investigation Complete:** Root cause confirmed, recommended solution identified.

**Current State (from Plan 03-08d):**
- Annual heating: 6.86 MWh (236% above [1.17, 2.04] MWh reference)
- Annual cooling: 4.82 MWh (31% above [2.13, 3.67] MWh reference)
- Annual total: 11.68 MWh (104% above [3.30, 5.71] MWh reference)
- Peak heating: 2.10 kW (perfect, within [1.10, 2.10] kW)
- Peak cooling: 3.57 kW (within [2.10, 3.70] kW)

**Root Cause Confirmed:**
- h_tr_em/h_tr_ms ratio: 0.0525 (target > 0.1)
- Thermal mass exchanges 95% with interior, 5% with exterior
- Winter Ti_free: 7.06°C (should be > 15°C)
- HVAC demand: 7013 W (334% of 2100 W capacity)
- HVAC runs at max constantly → high annual energy

**Recommended Solution:**
- **Option 1 (h_tr_em 5x):** Increase h_tr_em from 57.32 → 286.61 W/K
- Expected h_tr_em/h_tr_ms ratio: 0.2625 (5x improvement)
- Expected time constant: 4.02 hours (16.6% improvement)
- Expected impact: Higher Ti_free, lower HVAC demand, lower annual energy

**Blockers:**
1. Need to implement Option 1 in case_builder.rs (modify Case 900 construction parameters)
2. Need to run full ASHRAE 140 validation after implementation
3. Need to verify that other cases (600 series, free-floating) are not affected
4. May need to investigate ASHRAE 140 reference implementation if Option 1 is insufficient

**Recommendations for Future Work:**

1. **Implement Option 1 (h_tr_em 5x) for Case 900:**
   - Modify case_builder.rs to add new function: `case_900_solution1_revisited()`
   - Set h_tr_em = 286.61 W/K (5x current value: 57.32 W/K)
   - Keep h_tr_ms = 1092.00 W/K (no change)
   - Keep all other Case 900 parameters the same
   - Expected improvements:
     * h_tr_em/h_tr_ms ratio: 0.2625 (5x improvement)
     * Time constant: 4.02 hours (16.6% improvement)
     * Better thermal mass exchange with exterior
     * Higher winter Ti_free (less cold released to interior)
     * Lower HVAC demand
     * Lower annual heating energy

2. **Run Full ASHRAE 140 Validation:**
   - Run all cases (600 series, 900 series, 960, 195)
   - Verify that Case 900 annual heating is within [1.17, 2.04] MWh reference
   - Verify that Case 900 annual cooling is within [2.13, 3.67] MWh reference
   - Verify that Case 900 peak loads remain in range
   - Verify that other cases are not adversely affected

3. **If Option 1 is Insufficient:**
   - Investigate ASHRAE 140 reference implementation
   - Compare reference h_tr_em/h_tr_ms ratio with current values
   - Check if reference uses different thermal network structure
   - May need to adjust thermal mass coupling parameters differently
   - May need to implement more sophisticated thermal model

4. **Alternative: Separate Heating/Cooling Coupling Parameters:**
   - Consider implementing h_tr_em_heating and h_tr_em_cooling (different values)
   - Apply different coupling based on heating/cooling mode
   - May avoid heating/cooling trade-off
   - Higher complexity, but more flexible control

**Implementation Priority:**
1. Implement Option 1 (h_tr_em 5x) in case_builder.rs
2. Run full ASHRAE 140 validation
3. If insufficient, investigate reference implementation
4. As last resort, implement separate heating/cooling coupling parameters

---
*Phase: 03-Solar-Radiation*
*Completed: 2026-03-09*

## Self-Check: PASSED

- [x] Created: tests/free_floating_temp_investigation.rs
- [x] Created: tests/solution1_revisited_coupling_adjustment.rs
- [x] Created: .planning/phases/03-Solar-Radiation/03-10-SUMMARY.md
- [x] Commit: 29336a4 (test: Solution 3 and Solution 1 Revisited investigation tests)
- [x] Solution 3 (6R2C) investigated and rejected (no improvement)
- [x] Solution 1 Revisited analyzed with three options
- [x] Option 1 (h_tr_em 5x) recommended as best compromise
- [x] Option 2 (h_tr_ms 50%) rejected (time constant too large)
- [x] Option 3 (both) rejected (time constant too large)
- [x] Root cause confirmed: h_tr_em/h_tr_ms ratio 0.0525 too low
- [x] Implementation path identified: Option 1 (h_tr_em 5x) in case_builder.rs
