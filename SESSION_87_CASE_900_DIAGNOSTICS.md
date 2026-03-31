# Session 87: Case 900 Comprehensive Diagnostics & Root Cause Analysis

**Date:** 2026-03-31
**Previous Session:** Session 86 - Orientation-Dependent Solar Distribution ⚠️ Partial
**Current Pass Rate:** 0% (0/8 Case 900 tests passing)
**Target Pass Rate:** 100% (8/8 Case 900 tests)
**Status:** CRITICAL - Fundamental physics model issues identified

## Session 87 Objectives & Results

### Objective: Fix 8 Failing Case 900 Tests

**Attempted Fix:** Reduced thermal mass coupling enhancement from 1.15 to 0.85
**Result:** No significant improvement - all 8 tests still failing

### Current Test Results

| Test | Reference Range | Current Value | Error | Status |
|------|----------------|---------------|-------|--------|
| Annual Heating | 1.17-2.04 MWh | 8.53 MWh | +318% | ❌ FAIL |
| Annual Cooling | 2.13-3.67 MWh | 0.93 MWh | -56% | ❌ FAIL |
| Peak Heating | 1.10-2.10 kW | 3.20 kW | +52% | ❌ FAIL |
| Peak Cooling | 2.10-3.50 kW | 1.30 kW | -38% | ❌ FAIL |
| FF Min Temp | -6.40 to -1.60°C | -6.06°C | OK | ✅ PASS |
| FF Max Temp | 41.80-46.40°C | 39.20°C | -6% | ❌ FAIL |
| Temp Swing Reduction | 10-25% | 34.3% | +37% | ❌ FAIL |
| Thermal Mass Balance | ±2°C tolerance | >2°C drift | ❌ | ❌ FAIL |

## Critical Root Cause Analysis

### Issue 1: Heating Energy Overprediction (+318%)

**Symptom:** Case 900 heating is 8.53 MWh vs 1.17-2.04 MWh reference

**Root Cause Hypotheses:**
1. **Thermal mass coupling too weak** - Heat not being stored in mass, going directly to HVAC
2. **Solar gain distribution incorrect** - Too little solar going to thermal mass for delayed release
3. **h_tr_em conductance too low** - Exterior-to-mass coupling insufficient
4. **Time constant mismatch** - Model responding too quickly, not capturing thermal lag

**Evidence:**
- Peak heating also overpredicted (+52%) - consistent with heating energy issue
- FF min temp passes (-6.06°C) - winter behavior somewhat correct
- FF max temp fails (39.20°C vs 41.8-46.4°C) - summer behavior wrong

### Issue 2: Cooling Energy Underprediction (-56%)

**Symptom:** Case 900 cooling is 0.93 MWh vs 2.13-3.67 MWh reference

**Root Cause Hypotheses:**
1. **Thermal mass absorbing too much cooling load** - Mass buffering is too effective
2. **Solar gains not reaching zone air** - Too much solar going to mass, not enough immediate cooling load
3. **Sensitivity calculation wrong** - HVAC responding too slowly to cooling needs
4. **Night ventilation effect** - Incorrectly applied to cooling season

**Evidence:**
- Peak cooling also underpredicted (-38%) - consistent with cooling energy issue
- Heating and cooling errors are in OPPOSITE directions - suggests fundamental model structure issue

### Issue 3: Temperature Swing Reduction Too High (34.3% vs 10-25%)

**Symptom:** High-mass building damping is too strong

**Root Cause:**
- Thermal mass coupling enhancement (0.85 or 1.15) not the primary issue
- Thermal capacitance may be too high
- h_tr_ms (mass-to-surface) conductance may be too low, trapping heat in mass

**Evidence:**
- 600FF swing: 68.90°C, 900FF swing: 45.26°C
- Expected: ~19.6% reduction (swing ~55.4°C)
- Actual: 34.3% reduction (swing 45.26°C) - too much damping

### Issue 4: FF Max Temperature Too Low (39.20°C vs 41.8-46.4°C)

**Symptom:** Free-floating building doesn't get hot enough in summer

**Root Cause:**
- Thermal mass absorbing too much solar gain
- Insufficient solar gain reaching zone air
- Ground coupling may be too strong (cooling effect)

**Evidence:**
- FF min temp passes (-6.06°C) - winter cooling is correct
- Summer heating is insufficient - suggests solar distribution issue

## Fundamental Model Structure Issues

### Contradictory Requirements

The model is failing in **opposite directions** for heating vs cooling:
- **Heating:** Overpredicting by 318% (too much energy needed)
- **Cooling:** Underpredicting by 56% (too little energy needed)

This suggests the thermal mass is:
- **In winter:** Not storing heat effectively (HVAC works too hard)
- **In summer:** Storing too much heat (HVAC doesn't work enough)

### Current Solar Distribution (Session 86)

```rust
solar_beam_to_mass_fraction = 0.25  // South cases: 25% to mass, 75% to air/surface
```

**Problem:** This may be BACKWARDS:
- Winter: 75% to air → immediate heating → less HVAC needed → should REDUCE heating
- But heating is OVERpredicted → suggests solar isn't helping enough
- Summer: 75% to air → immediate cooling load → more HVAC needed → should INCREASE cooling
- But cooling is UNDERpredicted → suggests solar isn't creating enough load

### Current Thermal Mass Coupling

```rust
thermal_mass_coupling_enhancement = 0.85  // Reduced from 1.15
```

**Problem:** This factor multiplies h_tr_em:
- Lower value → weaker exterior-to-mass coupling
- Winter: Less heat stored in mass → more immediate heating → should help
- Summer: Less heat stored in mass → more immediate cooling → should hurt
- But both heating AND cooling are wrong → suggests coupling isn't the primary issue

## Recommended Next Steps

### Priority 1: Debug Solar Gain Distribution (CRITICAL)

**Action:** Add detailed solar gain diagnostics to track:
1. Total solar gain per month (heating vs cooling season)
2. Distribution: beam to mass vs beam to air vs diffuse
3. Impact on zone temperature vs mass temperature
4. Correlation with HVAC demand

**Hypothesis:** The solar_beam_to_mass_fraction is causing contradictory effects.

### Priority 2: Investigate Thermal Mass Time Constant

**Action:** Calculate and validate:
1. Actual thermal time constant: τ = C_m / (h_tr_em + h_tr_ms)
2. Compare with expected values (4-6 hours for high-mass)
3. Check if integration method (backward Euler) is appropriate

**Hypothesis:** Time constant may be too short, causing rapid response instead of lag.

### Priority 3: Review h_tr_em and h_tr_ms Calculation

**Action:** Verify physics-based calculations:
1. h_tr_em from ISO 13790 half-insulation rule
2. h_tr_ms from thermal time constant formula
3. Check if values are physically reasonable for 200mm concrete walls

**Current values (from diagnostics):**
- h_tr_em_base = 63.294 W/K
- h_tr_ms_physics = 2014.476 W/K
- Ratio: h_tr_em / h_tr_ms = 0.031 (very low!)

**Hypothesis:** h_tr_em may be too low relative to h_tr_ms, preventing heat from reaching mass.

### Priority 4: Validate Against Simpler Cases

**Action:** Before fixing Case 900, ensure:
1. Case 600 (low-mass) still passes all tests
2. Case 900FF (free-floating) min/max temps are correct
3. Basic heat transfer physics are validated

**Rationale:** Don't break working cases while fixing broken ones.

## Files Requiring Changes

1. **src/sim/engine.rs**
   - Lines ~1500-1600: Solar gain distribution logic
   - Lines ~1800-1900: Thermal mass coupling parameters
   - Lines ~4200-4400: step_physics_5r1c solar distribution

2. **tests/ashrae_140_case_900.rs**
   - Add detailed diagnostic output
   - Track monthly energy breakdown
   - Add solar gain validation tests

3. **New diagnostic binary**
   - `src/bin/session_87_case_900_diagnostics.rs`
   - Track hourly solar gains, temperatures, HVAC demand

## Conclusion

The Case 900 validation issues are **fundamental physics model problems**, not simple tuning issues. The contradictory heating/cooling errors suggest the thermal mass model structure needs re-evaluation.

**Key Insight:** The current approach of adjusting individual factors (coupling enhancement, solar distribution) is insufficient. A comprehensive review of the thermal mass physics is needed, focusing on:
1. Solar gain distribution mechanism
2. Thermal mass time constant
3. h_tr_em and h_tr_ms conductance calculations

**Estimated Effort:** 2-3 sessions (6-9 hours) for comprehensive fix.
