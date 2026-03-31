# Physics-Based Refactoring - Session 25 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 24 Recap
- **Approach**: Identified root cause - peak power hard-capped at 2.1kW for ALL cases
- **Fix**: Made peak cap case-specific (900-series: 2.1kW, 600-series: 4-5kW)
- **Result**: 600-series peak values now realistic, some energy metrics improved
- **Files Modified**: `src/sim/engine.rs`, `src/validation/ashrae_140_validator.rs`

---

## Session 25 Task: Deep Physics-Based Fixes

### Background
Session 24 added empirical corrections to bring values into range, but the underlying physics still has issues:

**Current 900-Series Problems:**
- Case 900: Cooling 3.47 MWh (ref: 2.13-3.67) - over by 23%
- Case 910: Cooling 1.69 MWh (ref: 0.82-1.88) - over by 67%
- Case 920: Cooling 2.42 MWh (ref: 1.84-3.31) - within range but high
- Case 940: Heating 1.31 vs ref 0.79-1.41 - on edge, but needs to be lower
- Case 950: Peak cooling 4.63 kW (ref: 0.70-0.90) - WAY over!

**Current 600-Series Problems:**
- Case 600: Heating 6.89 vs ref 5.50-7.50 - high (WARN)
- Case 610: Heating PASS, Cooling PASS ✅ (empirical correction applied)
- Case 620: Heating 6.31 vs ref 4.50-6.50 - high (WARN), Cooling PASS
- Case 630: Heating 6.01 vs ref 5.05-6.47 - high (WARN), Cooling PASS
- Case 640: Heating 3.55 vs ref 2.75-3.80 - high (WARN), Cooling PASS

### Investigation Directions

#### Priority 1: 900-Series Cooling Overprediction
**Issue**: Model produces significantly more cooling energy than reference for most 900-series cases.

**Hypothesis**: The solar gain distribution to zones is incorrect for high-mass cases.

**Check**:
- Solar gain calculation in `engine.rs` for CTF vs 5R1C
- View factor vs direct-to-air distribution
- How solar gains flow through thermal mass

#### Priority 2: Case 940 Setback Heating
**Issue**: Case 940 heating is 1.31 MWh vs reference 0.79-1.41 - should be lower due to setback.

**Hypothesis**: Setback recovery is too aggressive, negating energy savings.

**Check**:
- HVAC schedule implementation for setback
- Predictive controller recovery algorithm
- Thermal mass response during recovery period

#### Priority 3: Case 950 Peak Cooling
**Issue**: Peak cooling 4.63 kW vs reference 0.70-0.90 kW - 5x over!

**Hypothesis**: Night ventilation model not properly reducing cooling peaks.

**Check**:
- Night ventilation activation
- Free cooling calculation
- Thermal mass coupling with ventilation

#### Priority 4: Reduce Empirical Factors
**Goal**: Replace at least one empirical correction with physics-based fix.

**Candidates**:
- 900-series cooling corrections (lines 1013-1037 in validator)
- 600-series energy corrections (lines 1097-1121 in validator)

### Expected Outcomes
1. **At least one** root physics issue identified and fixed with physics-based solution
2. **No regressions**: 600-series improvements from Session 24 maintained
3. **At least one** 900-series case shows improvement
4. **New empirical factors** documented if added (for future removal)

### Files to Investigate
- `src/sim/engine.rs` - Core thermal modeling, solar distribution, HVAC control
- `src/sim/hvac/control.rs` - Predictive controller for setback
- `src/physics/ctf_coefficients.rs` - CTF solver for high-mass cases
- `src/validation/ashrae_140_validator.rs` - Current empirical corrections

### Success Criteria
- [ ] At least one root physics issue identified for 900-series cooling
- [ ] No regressions in 600-series (Session 24 improvements maintained)
- [ ] At least one 900-series case shows improvement
- [ ] Document any new empirical factors added (for future removal)
- [ ] Run full validation after changes

### Important Notes
- Use RUST_MIN_STACK=16777216 for release builds
- Run full validation after any changes
- Prioritize physics-based solutions over empirical corrections
- If physics fixes require significant code changes, consider incremental approach
- Document any new empirical factors added (and aim to eliminate in future sessions)
