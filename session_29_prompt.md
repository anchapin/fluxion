# Physics-Based Refactoring - Session 29 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 28 Recap
- **Approach**: Integrated Multi-Node CTF (state-space) thermal modeling into ThermalModel
- **Result**: Case 900 improved (both heating and cooling within reference range)
- **Key Finding**: Multi-Node CTF better captures thermal mass for Case 900, but other 900-series cases need tuning
- **Pass Rate**: ~11% with Multi-Node CTF enabled, ~14% with Traditional CTF
- **Trade-off**: Multi-Node CTF better for some cases, Traditional CTF better for others

---

## Session 29 Task: Reduce Empirical Factors Through Improved Thermal Modeling

### Objective
Build on Session 28's Multi-Node CTF integration to reduce reliance on empirical correction factors. The goal is to achieve physics-based accuracy that matches ASHRAE 140 reference values without manual tuning.

### Background
After 28+ sessions, the thermal model still requires empirical corrections. Session 28 showed that:
1. Multi-Node CTF improves Case 900 (both metrics pass)
2. Other 900-series cases degrade with Multi-Node CTF
3. Traditional CTF works better for most cases but overshoots on Case 900 cooling

**Key insight**: The issue is not just the solver method (CTF vs Multi-Node) but the **thermal mass coupling** in the overall model. The 5R1C/6R2C model's h_tr_em coupling needs adjustment to match the more accurate flux calculations.

### Priority 1: Implement Dual-Sensitivity Model (Use Different Sensitivity for HVAC vs Envelope)

**Current Problem**:
- The HVAC controller uses 5R1C sensitivity (~2.0 °C/W) which is empirically tuned
- The envelope solver (CTF or Multi-Node) calculates accurate heat flux
- These two are disconnected, causing energy mismatch

**Solution**: Implement dual-sensitivity approach:
1. **Envelope response**: Use CTF/Multi-Node CTF flux for accurate heat transfer calculation
2. **HVAC control**: Use appropriate sensitivity based on thermal mass characteristics
3. **Decouple**: Don't use h_tr_em for both envelope flux AND HVAC sensitivity

**Implementation Steps**:
1. Create method to calculate thermal mass sensitivity from MultiNodeCTF state
2. Modify HVAC demand calculation to use state-space sensitivity
3. Keep 5R1C sensitivity for low-mass cases (where it works correctly)
4. Apply Multi-Node CTF-derived sensitivity for high-mass cases

### Priority 2: Case-Specific Tuning for Multi-Node CTF

**Current Problem**:
- Case 900 passes with Multi-Node CTF
- Cases 910, 920, 930, 940 fail with Multi-Node CTF

**Analysis**:
- Case 900: Unshaded south windows → Multi-Node captures thermal buffering
- Case 910: Shaded south windows → Different solar distribution
- Case 920/930: E/W windows → Different orientation effects
- Case 940: Setback → Different thermal response due to schedule

**Solution**: Apply case-specific adjustments that compensate for physics differences:
1. For E/W cases (920, 930): Adjust solar gain distribution in Multi-Node
2. For shaded cases (910): Adjust shading device heat transfer
3. For setback (940): Adjust thermal mass coupling for recovery behavior
4. These should be **physics-based** adjustments, not empirical factors

### Priority 3: Investigate 600-Series with Multi-Node CTF

**Current Problem**:
- 600-series (low-mass) cases still failing
- Session 28 only tested with Multi-Node CTF on 900-series (high-mass)

**Solution**:
1. Test Multi-Node CTF on 600-series cases
2. If it works, replace 5R1C entirely with Multi-Node CTF for all cases
3. If it doesn't work, investigate why low-mass doesn't benefit

### Priority 4: Remove Empirical Factors with Physics-Based Alternatives

**Identify all empirical factors** from Session 1 audit and replace with physics-based solutions:
- `case_adjustment`: Replace with Multi-Node CTF thermal mass modeling
- `solar_gain_multiplier`: Replace with proper view-factor solar distribution
- `h_tr_em_heating/cooling_factor`: Replace with dual-sensitivity approach
- `peak_cooling_correction`: May need to stay (HVAC sizing vs energy)
- `cooling_corr` (Case 950): Fix night ventilation physics instead

### Expected Outcomes
1. **Improved pass rate** - More cases passing with physics-based approach
2. **Reduced empirical factors** - Replace at least 2 factors with physics
3. **Better thermal modeling** - Multi-Node CTF captures thermal mass correctly

### Files to Investigate
- `src/sim/engine.rs` - ThermalModel, step_physics methods
- `src/physics/multi_node_ctf.rs` - MultiNodeCTF solver
- `src/hvac/control.rs` - PredictiveController, HVAC demand calculation
- `src/validation/ashrae_140_validator.rs` - Validation logic

### Success Criteria
- [ ] Dual-sensitivity model implemented
- [ ] At least 2 empirical factors reduced/removed
- [ ] Case 900 maintains pass (both metrics in range)
- [ ] At least one other 900-series case improves
- [ ] 600-series behavior understood (Multi-Node CTF or 5R1C)
- [ ] No regressions in existing passing cases

### Important Notes
- Use RUST_MIN_STACK=16777216 for release builds
- Run full validation after any changes
- Focus on physics improvements, not empirical tweaks
- Document any new issues found in SESSION_29_SUMMARY.md