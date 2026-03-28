# Physics-Based Refactoring - Session 27 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 26 Recap
- **Approach**: Investigated solar gain calculation bug (0 W/m² reported)
- **Result**: Solar calculation is CORRECT - 0 W/m² at midnight is expected (timestep 0 = midnight)
- **Key Finding**: The "bug" was incorrect observation timing, not a calculation issue
- **Files Modified**: `src/sim/engine.rs` - Cleaned up debug output

---

## Session 27 Task: Continue Root Cause Analysis for Empirical Factor Removal

### Objective
Continue investigating the ROOT CAUSES of physics-based model shortcomings and eliminate empirical corrections through deep analysis and physics-based fixes.

### Background
After 26 sessions, the model physics is working correctly (solar gains calculated properly). The remaining issues are:
1. **900-series cooling overprediction** - Physics may be correct but needs tuning
2. **Empirical correction factors** - Still in place, need physics-based replacements
3. **600-series heating/cooling** - Need investigation

### Current Empirical Corrections (still in validator.rs):

**900-series energy corrections** (need physics replacement):
- Case 900: Heating /4.0, Cooling ×0.50
- Case 910: Heating /2.5, Cooling ×0.35
- Case 940: Heating /2.7, Cooling ×0.45
- Case 950: Cooling ×0.35

**600-series energy corrections**:
- Case 600: Heating /1.25, Cooling ×1.35
- Case 610: Heating /1.7
- Case 620: Heating /1.25, Cooling ×1.5
- Case 630: Heating /1.5, Cooling ×2.0
- Case 640: Heating /1.8
- Case 650: Cooling ×1.1

---

### Priority 1: Analyze Energy Balance for 900-series Cases

**Issue**: 900-series cooling overpredicts despite correct solar calculation

**Investigation Steps**:
1. Run detailed validation for Case 900 with hourly energy breakdown
2. Check where energy is being added/consumed in the thermal model
3. Compare conduction gains vs solar gains vs internal gains
4. Identify if HVAC is triggering too aggressively

**Questions to Answer**:
- Is the cooling demand calculation correct?
- Is the zone temperature staying above setpoint incorrectly?
- Are thermal coupling factors causing excess heat retention?

---

### Priority 2: Investigate HVAC Mode Determination

**Issue**: HVAC might be triggering cooling when it shouldn't

**Investigation Steps**:
1. Check `calculate_modulation()` in control.rs
2. Verify HVAC mode determination logic (heating/cooling/deadband)
3. Look at how solar gains affect mode determination
4. Check if there's a hysteresis issue causing excess cycling

---

### Priority 3: Analyze Thermal Coupling Factors

**Issue**: h_tr_em coupling factors may need adjustment

**Current factors** (from Session 25):
- Heating coupling: ~0.10-0.15 for 900-series
- Cooling coupling: ~0.10-1.50 for 900-series

**Investigation**:
1. Check if these factors are physically correct
2. Analyze heat transfer paths in 5R1C model
3. Consider if multi-node CTF would help (advanced)

---

### Priority 4: Review Case 940 Setback Implementation

**Issue**: Case 940 heating should be lower due to setback

**Investigation**:
1. Verify night setpoint (10°C) is being applied
2. Check morning recovery heating rate
3. Compare with Case 600 (no setback) baseline

---

### Expected Outcomes
1. **Root cause understanding** for 900-series cooling overprediction
2. **At least ONE empirical correction** analyzed and documented with physics reasoning
3. **No regressions** - Validation results maintained
4. **Documentation** of remaining physics issues

### Files to Investigate
- `src/sim/engine.rs` - Core thermal modeling, HVAC coupling
- `src/sim/hvac/control.rs` - HVAC mode determination
- `src/validation/ashrae_140_validator.rs` - Current empirical corrections
- `src/physics/five_r1c_solver.rs` - 5R1C thermal solver

### Success Criteria
- [ ] Root cause of 900-series cooling overprediction identified
- [ ] HVAC mode determination verified as correct
- [ ] Thermal coupling factors analyzed
- [ ] Case 940 setback behavior verified
- [ ] Document physics reasoning for each empirical factor

### Important Notes
- Use RUST_MIN_STACK=16777216 for release builds
- Run full validation after any changes
- Focus on ROOT CAUSES, not symptoms
- Document any new empirical factors added (for future removal)