---
phase: 03-Solar-Radiation
plan: 11
subsystem: annual-energy-correction-implementation
tags: [annual-energy, thermal-mass, coupling-ratio, 5r1c, implementation, deviation, investigation]

# Dependency graph
requires:
  - phase: 03-Solar-Radiation
    provides: 03-10-SUMMARY.md with Option 1 (h_tr_em 5x) recommendation
provides:
  - Test results showing Option 1 (h_tr_em 5x) makes annual energy worse
  - Evidence that theoretical analysis was incorrect
  - Recommendation to investigate alternative approaches
affects:
  - Future implementation plans for annual energy correction
  - Need to reconsider coupling enhancement strategy

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Deviation from plan: Option 1 implementation failed
    - Theoretical analysis vs actual test results mismatch
    - h_tr_em enhancement too high causes thermal mass to absorb excessive cold
    - Annual heating energy increases with higher h_tr_em (counter-intuitive)

key-files:
  created:
    - None (investigation only)
  modified:
    - src/sim/engine.rs (temporarily modified, then reverted)

key-decisions:
  - "Option 1 (h_tr_em 5x) implementation FAILED - makes annual heating energy worse"
  - "Theoretical analysis in 03-10-SUMMARY.md was incorrect"
  - "Increasing h_tr_em too much causes thermal mass to absorb too much cold from exterior"
  - "Current state (enhancement=1.15, h_tr_em=57.32 W/K) is optimal for now"
  - "Need alternative approach to fix annual energy over-prediction"

patterns-established:
  - "Implementation verification: Always test recommended changes before committing"
  - "Theoretical analysis vs reality: Validate assumptions with actual test runs"
  - "Thermodynamic behavior: Increasing exterior coupling can increase heating demand in winter"
  - "Counter-intuitive results: Higher h_tr_em can make annual energy worse"

requirements-completed: []

# Metrics
duration: 45min
completed: 2026-03-09
---

# Phase 3 Plan 11: Annual Energy Correction - Option 1 Implementation Summary

**Implementation of Option 1 (h_tr_em 5x) from Plan 03-10 to fix annual energy over-prediction for Case 900 high-mass building. Result: FAILED - Option 1 makes annual energy worse.**

## Performance

- **Duration:** 45 min
- **Started:** 2026-03-09T21:34:44Z
- **Completed:** 2026-03-09T22:19:44Z
- **Tasks:** 2 (implementation attempt, verification)
- **Files created:** 0
- **Files modified:** 1 (temporarily, then reverted)
- **Deviations:** 1 (Option 1 implementation failed)

## Accomplishments

1. **Implementation Attempt: Modified thermal_mass_coupling_enhancement**
   - Changed from 1.15 to 5.75 for Case 900
   - Expected h_tr_em increase: 57.32 W/K → 286.60 W/K (5x)
   - Expected h_tr_em/h_tr_ms ratio improvement: 0.0525 → 0.2625
   - Expected annual heating reduction: Significant (better Ti_free → lower HVAC demand)

2. **Verification: Test Results Show Option 1 Makes Things Worse**
   - Original state (enhancement=1.15): Annual heating = 6.86 MWh
   - With enhancement=5.75: Annual heating = 10.70 MWh (56% WORSE!)
   - Annual cooling also affected: 4.82 MWh → 1.68 MWh (under-predicted)
   - Peak loads remained within range (heating=2.10 kW, cooling=3.25 kW)

## Task Commits

**Attempted commit (later reverted):**
- **Task 1: Implement Option 1 (h_tr_em 5x)** - `071bb05` (feat)
  - Modified src/sim/engine.rs to increase thermal_mass_coupling_enhancement from 1.15 to 5.75
  - Expected: h_tr_em = 286.60 W/K, ratio = 0.2625
  - Actual: h_tr_em = 286.61 W/K, ratio = 0.2625 (correct)
  - **Result: Annual heating = 10.70 MWh (56% worse than 6.86 MWh baseline)**
  - **Decision: Revert commit - Option 1 does not work**

## Deviations from Plan

### Deviation 1: Option 1 Implementation Failed (Critical)

**Found during:** Task 2 (verification)

**Issue:**
- Plan 03-10-SUMMARY.md recommended Option 1 (h_tr_em 5x) as the best approach
- Theoretical analysis predicted: higher h_tr_em → better Ti_free → lower HVAC demand → lower annual energy
- Actual test results: higher h_tr_em → thermal mass absorbs too much cold → worse Ti_free → higher HVAC demand → higher annual energy

**Root Cause:**
- The theoretical analysis missed the thermodynamic reality of winter conditions
- In winter, increasing h_tr_em allows thermal mass to absorb MORE cold from the exterior
- This cold is then released to the interior, making Ti_free lower and increasing heating demand
- The effect dominates over any benefits from better exterior coupling

**Test Results:**
| Enhancement | h_tr_em (W/K) | h_tr_em/h_tr_ms | Annual Heating (MWh) | Annual Cooling (MWh) |
|-------------|---------------|-----------------|----------------------|----------------------|
| 1.15 (original) | 57.32 | 0.0525 | 6.86 (236% above ref) | 4.82 (31% above ref) |
| 2.0 | 99.69 | 0.0913 | 8.40 (worse!) | - |
| 5.75 | 286.61 | 0.2625 | 10.70 (56% worse!) | 1.68 (under-predicted) |

**Resolution:**
- Reverted all changes to src/sim/engine.rs
- Documented that Option 1 is NOT a viable solution
- Current state (enhancement=1.15, h_tr_em=57.32 W/K) is optimal for now
- **Need alternative approach to fix annual energy over-prediction**

## Issues Encountered

**Theoretical Analysis vs Reality Mismatch**
- **Expected:** Increasing h_tr_em from 57.32 W/K to 286.60 W/K would:
  * Improve h_tr_em/h_tr_ms ratio from 0.0525 to 0.2625
  * Improve time constant from 4.82h to 4.02h
  * Increase heat flow to exterior from 5.0% to 20.8%
  * Result in higher Ti_free and lower HVAC demand
- **Actual:** Increasing h_tr_em made annual heating energy 56% worse
- **Root Cause:** Winter thermodynamics - thermal mass absorbs too much cold from exterior
- **Resolution:** Reverted changes, documented findings, need alternative approach

**Current State (at commit afa8a8c):**
- Annual heating: 6.86 MWh (236% above [1.17, 2.04] MWh reference)
- Annual cooling: 4.82 MWh (31% above [2.13, 3.67] MWh reference)
- Annual total: 11.68 MWh (104% above [3.30, 5.71] MWh reference)
- Peak heating: 2.10 kW (perfect, within [1.10, 2.10] kW)
- Peak cooling: 3.57 kW (within [2.10, 3.70] kW)

## Analysis

### Why Option 1 Failed

**Thermodynamic Analysis:**
1. **Winter Conditions:** Outdoor temperature = -10°C, HVAC setpoint = 20°C
2. **Original State (h_tr_em=57.32 W/K):**
   - Thermal mass exchanges 5% with exterior, 95% with interior
   - Thermal mass stays relatively warm (close to interior temp)
   - Ti_free = 7.06°C (cold, but manageable)
   - HVAC demand = 7013 W (high, but not maximum)
3. **With Enhanced h_tr_em (286.61 W/K):**
   - Thermal mass exchanges 21% with exterior, 79% with interior
   - Thermal mass absorbs too much cold from exterior
   - Ti_free drops even further (thermal mass is colder)
   - HVAC demand increases (heating more to maintain setpoint)
   - **Result: Annual heating = 10.70 MWh (56% worse!)**

**The Missing Piece in 03-10 Analysis:**
- 03-10-SUMMARY.md analyzed the coupling ratio and time constant
- It did NOT account for the sign of heat flow in winter
- In summer: High h_tr_em helps thermal mass absorb heat → better Ti_free → lower cooling
- In winter: High h_tr_em helps thermal mass absorb cold → worse Ti_free → higher heating
- The winter effect dominates, making annual heating energy worse

### Counter-Intuitive Result

**Why "Better" Parameters Make Things Worse:**
- h_tr_em/h_tr_ms ratio of 0.2625 is theoretically "better" than 0.0525
- Time constant of 4.02h is theoretically "better" than 4.82h
- BUT: These improvements only help in summer conditions
- In winter, the high h_tr_em causes excessive cold absorption
- **Thermodynamics trumps theoretical coupling ratios**

## Recommendations

### 1. Rejected Approaches

**Option 1 (h_tr_em 5x):** REJECTED
- Makes annual heating energy 56% worse
- Annual cooling becomes under-predicted
- Counter-intuitive but thermodynamically sound

**Option 2 (h_tr_ms 50%):** REJECTED
- Excessive time constant (9.18 hours)
- Unstable integration with 1-hour timesteps

**Option 3 (both changes):** REJECTED
- Excessive time constant (7.15 hours)
- Unstable integration with 1-hour timesteps

### 2. Alternative Approaches to Consider

**A. Investigate ASHRAE 140 Reference Implementation**
- Compare current implementation with ASHRAE 140 reference values
- Check if reference uses different h_tr_em/h_tr_ms ratio
- Check if reference uses different thermal network structure
- May reveal that the "correct" parameters differ from ISO 13790 standard
- Risk: High complexity, may require significant model changes

**B. Separate Heating/Cooling Coupling Parameters**
- Implement h_tr_em_heating and h_tr_em_cooling (different values)
- Apply lower h_tr_em during heating season (reduce cold absorption)
- Apply higher h_tr_em during cooling season (improve heat absorption)
- Complexity: Need season detection or mode-based switching
- May solve heating/cooling trade-off

**C. Time Constant-Based Correction (Revisited)**
- Use different time constant correction for heating vs cooling
- Apply higher sensitivity correction for heating (reduce demand)
- Apply lower sensitivity correction for cooling (maintain demand)
- Issue: Previous attempt (Plan 03-08) created cooling regression

**D. Investigate Construction Parameters**
- Check if ASHRAE 140 reference uses different wall/roof/floor constructions
- May need different U-values or layer ordering
- May need different thermal capacitance values
- Risk: Changes affect all cases, not just Case 900

### 3. Recommended Next Steps

**Priority 1: Investigate ASHRAE 140 Reference Implementation**
- This is the most likely path to success
- ASHRAE 140 reference programs (EnergyPlus, ESP-r, TRNSYS) pass validation
- They must be using correct parameters
- Compare h_tr_em, h_tr_ms, and other 5R1C parameters with reference

**Priority 2: Separate Heating/Cooling Coupling (if Priority 1 insufficient)**
- More complex but addresses the root cause
- Allows different coupling based on heating/cooling mode
- Requires significant code changes

**Priority 3: Accept Current State as Limitation**
- Current state (6.86 MWh heating) may be the best achievable with current approach
- May need to document this as a known limitation
- Focus on other issues (solar gains, peak cooling loads)

## Blockers

1. **Option 1 (h_tr_em 5x) implementation failed** - Makes annual heating energy worse
2. **No obvious alternative fix** - All simple parameter tuning approaches have been tried
3. **Need to investigate ASHRAE 140 reference implementation** - Complex but necessary
4. **May require major model changes** - If reference uses different thermal network structure

## Next Phase Readiness

**Investigation Incomplete:** Option 1 implementation failed, need alternative approach.

**Current State (at commit afa8a8c):**
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

**Failed Solution:**
- **Option 1 (h_tr_em 5x):** Increases annual heating from 6.86 MWh to 10.70 MWh
- **Root cause of failure:** Thermal mass absorbs too much cold from exterior in winter
- **Theoretical analysis was incorrect:** Better coupling ratios don't always mean better energy

**Blockers:**
1. Need to investigate ASHRAE 140 reference implementation
2. May need to implement separate heating/cooling coupling parameters
3. May need to accept current state as a limitation
4. May need major model changes if reference uses different thermal network

**Recommendations for Future Work:**

1. **Investigate ASHRAE 140 Reference Implementation:**
   - Compare reference h_tr_em/h_tr_ms ratio with current values
   - Check if reference uses different thermal network structure
   - May need to adjust thermal mass coupling parameters differently
   - May need to implement more sophisticated thermal model

2. **Consider Separate Heating/Cooling Coupling Parameters:**
   - Implement h_tr_em_heating and h_tr_em_cooling
   - Apply different values based on heating/cooling mode
   - May avoid heating/cooling trade-off
   - Higher complexity, but more flexible control

3. **Alternative: Investigate Construction Parameters:**
   - Check if reference uses different wall/roof/floor U-values
   - Check if reference uses different thermal capacitance values
   - May need different layer ordering or material properties

4. **As Last Resort: Document as Known Limitation:**
   - Current state (6.86 MWh) may be the best achievable
   - Document that ISO 13790 5R1C may not be sufficient for ASHRAE 140 Case 900
   - Recommend using different thermal model for high-mass buildings
   - Focus on other validation issues (solar gains, peak loads)

**Implementation Priority:**
1. Investigate ASHRAE 140 reference implementation
2. If needed, implement separate heating/cooling coupling parameters
3. As last resort, document current state as known limitation

---
*Phase: 03-Solar-Radiation*
*Completed: 2026-03-09*

## Self-Check: FAILED

- [x] Created: .planning/phases/03-Solar-Radiation/03-11-SUMMARY.md
- [x] Commit: 071bb05 (feat: implement Option 1 h_tr_em 5x) - REVERTED
- [x] Option 1 (h_tr_em 5x) implemented and tested
- [x] Test results show Option 1 makes annual heating worse (10.70 vs 6.86 MWh)
- [x] Root cause identified: thermal mass absorbs too much cold from exterior
- [x] All changes reverted to commit afa8a8c
- [x] Alternative approaches recommended (reference investigation, separate coupling)
- [x] Deviation documented in SUMMARY.md
- [ ] **Objective NOT achieved:** Annual energy over-prediction still exists
- [ ] **Success criteria NOT met:** Annual heating still 236% above reference

**Status:** Plan failed - Option 1 implementation made things worse. Need alternative approach.
