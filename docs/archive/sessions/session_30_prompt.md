# Physics-Based Refactoring - Session 30 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 29 Recap
- **Approach**: Reduced empirical correction factors, switched to CTF with FD fallback
- **Result**: Pass rate improved from ~10% to 14.1% (11/64)
- **Key Finding**: 600-series heating now passes, 900-series heating improved
- **Issue**: Cooling still problematic for several cases

---

## Session 30 Task: Fix Cooling Predictions and Improve Pass Rate

### Objective
Build on Session 29's improvements by fixing the remaining cooling prediction issues and further reducing empirical factors through improved physics.

### Background
After 29+ sessions, the thermal model still has issues:
1. **600-series cooling**: Underpredicts (7.51 MWh vs 8.00-10.50 reference for Case 600)
2. **900-series cooling**: Overpredicts for Case 900 (3.47 MWh vs 2.13-3.67 reference)
3. **Case 910, 940 heating**: Need additional tuning
4. **Pass rate**: 14.1% (target is 75%)

### Priority 1: Fix 600-Series Cooling Underprediction

**Current Problem**:
- Case 600 cooling: 7.51 MWh (Ref: 8.00-10.50) - UNDER by ~6%
- Case 620 cooling: 2.74 MWh (Ref: 3.20-5.00) - OK
- Case 630 cooling: 1.67 MWh (Ref: 2.13-3.70) - UNDER by ~27%

**Root Cause Analysis**:
- Need to investigate internal gains and solar distribution
- Check if ventilation heat removal is overestimated
- Verify window solar gains are properly captured

**Solution**:
1. Increase internal gains for 600-series cases
2. Adjust solar distribution to zone air vs walls
3. Verify night ventilation effect on cooling load

### Priority 2: Fix 900-Series Cooling Overprediction

**Current Problem**:
- Case 900 cooling: 3.47 MWh (Ref: 2.13-3.67) - OVER by 22%
- Case 910 cooling: 1.69 MWh (Ref: 0.82-1.88) - OK
- Case 940 cooling: 3.13 MWh (Ref: 2.08-3.55) - OK

**Analysis**:
- Multi-Node CTF helps heating but may cause cooling overprediction
- The solar gain reduction factor may be too aggressive for winter months
- Need to decouple heating vs cooling physics

**Solution**:
1. Apply seasonal solar gain multipliers (summer vs winter)
2. Adjust internal gains for high-mass cases
3. Investigate if CTF is over-predicting heat absorption

### Priority 3: Fix Case 910, 940 Heating

**Current Problem**:
- Case 910 heating: 2.06 MWh (Ref: 1.51-2.28) - Needs ~0.3 MWh reduction
- Case 940 heating: 1.31 MWh (Ref: 0.79-1.41) - Needs ~0.5 MWh reduction

**Analysis**:
- Case 910 has shading - reduced solar gains in summer but should be higher in winter
- Case 940 has setback - recovery heating may be too aggressive

**Solution**:
1. For Case 910: Increase winter solar gains (shading blocks summer sun, not winter)
2. For Case 940: Adjust setback recovery to reduce heating overshoot

### Priority 4: Reduce Remaining Empirical Factors

**Identify remaining corrections** that can be reduced or eliminated:
- Free-floating temperature offsets (lines 776-794)
- Peak corrections for cases now passing
- Any remaining energy corrections

### Expected Outcomes
1. **Improved pass rate** - More cases passing with physics-based approach
2. **Reduced empirical factors** - Replace at least 2 more factors with physics
3. **Better cooling predictions** - Both 600 and 900 series within reference

### Files to Investigate
- `src/sim/engine.rs` - Solar gain calculation, internal gains
- `src/physics/ctf_solver.rs` - CTF solver behavior
- `src/validation/ashrae_140_validator.rs` - Remaining corrections
- `src/sim/solar.rs` - Solar distribution logic

### Success Criteria
- [ ] 600-series cooling within 10% of reference range
- [ ] 900-series cooling within reference range
- [ ] Case 910, 940 heating within reference range
- [ ] At least 2 more empirical factors reduced/removed
- [ ] Pass rate improved to ≥20%
- [ ] No regressions in existing passing cases

### Important Notes
- Use RUST_MIN_STACK=16777216 for release builds
- Run full validation after any changes
- Focus on physics improvements, not empirical tweaks
- Document any new issues found in SESSION_30_SUMMARY.md
