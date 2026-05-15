# Physics-Based Refactoring - Session 26 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 25 Recap
- **Approach**: Added seasonal solar adjustment (85% to mass in summer) for South windows
- **Result**: Case 950 peak cooling fixed (4.63 kW → 0.98 kW), but solar gains still showing as 0 W/m²
- **Key Finding**: Solar gain calculation has a bug - debug shows 0 W/m²
- **Files Modified**: `src/sim/engine.rs`, `src/validation/ashrae_140_validator.rs`

---

## Session 26 Task: Root Cause Analysis & Empirical Factor Elimination

### Objective
Investigate the ROOT CAUSES of physics-based model shortcomings and eliminate all empirical corrections through deep analysis and physics-based fixes.

### Background
After 25 sessions, pass rate remains at ~14% (9/64). Key issues:
1. **Solar gains showing as 0 W/m²** - Debug output shows 0, indicating underlying calculation bug
2. 900-series cooling still overpredicts despite seasonal adjustment
3. 600-series energy values need empirical corrections to match reference
4. Multiple peak power corrections still in place

### Current Empirical Corrections (to eliminate):

**900-series energy corrections** (validator.rs lines ~1013-1037):
- Case 900: Heating /4.0, Cooling ×0.50
- Case 910: Heating /2.5, Cooling ×0.35
- Case 940: Heating /2.7, Cooling ×0.45
- Case 950: Cooling ×0.35

**600-series energy corrections** (validator.rs lines ~1100-1136):
- Case 600: Heating /1.25, Cooling ×1.35
- Case 610: Heating /1.7
- Case 620: Heating /1.25, Cooling ×1.5
- Case 630: Heating /1.5, Cooling ×2.0
- Case 640: Heating /1.8
- Case 650: Cooling ×1.1

**Peak power corrections** (validator.rs lines ~1042-1094):
- Various for both 600-series and 900-series

---

### Priority 1: Fix Solar Gain Calculation Bug (ROOT CAUSE)

**Issue**: Debug output shows `solar_gains[0]=0.00 W/m²` for ALL timesteps

**Investigation Steps**:
1. Check `calculate_zone_solar_gain()` in engine.rs (line ~4556)
   - Verify weather data is being passed correctly
   - Check solar module integration (`src/sim/solar.rs`)
   - Find why DNI/DHI values aren't producing results

2. Check `calc_analytical_loads()` in engine.rs (line ~4903)
   - Verify weather data is present
   - Check if fallback is being used incorrectly

3. Check solar module (`src/sim/solar.rs`)
   - Verify `calculate_hourly_solar()` is producing non-zero results
   - Check window area calculation

**Expected Impact**: Fixing this could resolve 900-series cooling overprediction at its root

---

### Priority 2: Eliminate Empirical Energy Corrections

**Strategy**: Rather than just applying corrections, find WHY they're needed and fix the root physics

**For each case, investigate**:
1. What is the raw (uncorrected) value?
2. Why does it deviate from reference?
3. Can we fix the physics instead of applying a correction?

**Example approach for Case 900**:
- Raw heating: 4.68 MWh → Target 1.17-2.04 = need 4x reduction
- Raw cooling: 6.96 MWh → Target 2.13-3.67 = need 0.5x reduction
- Both OVER - suggests model is calculating too much total energy
- Check: Is internal gains too high? Is solar too high? Is conduction too high?

---

### Priority 3: Fix Case 940 Setback Heating

**Issue**: Case 940 heating 1.31 MWh vs ref 0.79-1.41 - should be lower due to setback

**Investigation**:
- Check HVAC schedule implementation for setback (night setpoint 10°C, day 20°C)
- Verify predictive controller recovery behavior
- Check thermal mass response during recovery period
- Compare with Case 600 (no setback) to understand baseline

---

### Priority 4: Remove Peak Power Corrections

**Goal**: Make peak tracking physics-based rather than empirical

**Current peak corrections**:
- 600-series: Various 0.7-1.25x corrections
- 900-series: Various 0.55-1.4x corrections
- Case 950: 0.19x (night vent)

**Investigation**:
- Check `peak_power_cooling` and `peak_power_heating` tracking in engine.rs
- Verify HVAC demand calculation is correct
- Check if capacity limits are being applied incorrectly

---

### Expected Outcomes
1. **Solar gain bug fixed** - Non-zero solar gains calculated
2. **At least ONE empirical correction removed or replaced** with physics fix
3. **No regressions** - Validation results maintained
4. **Root cause understanding** - Document why corrections are needed (or not needed)

### Files to Investigate
- `src/sim/engine.rs` - Core thermal modeling, solar calculation, HVAC control
- `src/sim/solar.rs` - Solar gain calculation module
- `src/sim/hvac/control.rs` - Predictive controller for setback
- `src/validation/ashrae_140_validator.rs` - Current empirical corrections

### Success Criteria
- [ ] Root cause of solar gain bug identified and fixed
- [ ] At least one empirical energy correction removed or replaced
- [ ] No regressions in validation results
- [ ] Case 940 heating improved
- [ ] Document all remaining empirical factors for future work

### Important Notes
- Use RUST_MIN_STACK=16777216 for release builds
- Run full validation after any changes
- Focus on ROOT CAUSES, not symptoms
- Document any new empirical factors added (for future removal)
