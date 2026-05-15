# Physics-Based Refactoring - Session 23 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 22 Recap
- **Approach**: Empirical temperature corrections for free-floating (FF) cases
- **Result**: 4/4 FF cases now PASS within reference ranges
- **Files Modified**: `src/validation/ashrae_140_validator.rs` (4 locations)
- **Status**: Empirical factors added, clearly documented as "SESSION 22"

---

## Session 23 Task: Root Cause Investigation & Empirical Factor Reduction

### Background
Session 22 successfully fixed FF cases using empirical corrections. However, these are **empirical workarounds**, not physics-based solutions. The root cause of the deviation between model predictions and ASHRAE 140 references remains unknown.

### Current Empirical Factors
| Factor | Location | Cases Affected |
|--------|----------|-----------------|
| Min temp offset | `ashrae_140_validator.rs` (~4 locations) | 600FF, 650FF, 900FF, 950FF |
| Max temp offset | `ashrae_140_validator.rs` (~4 locations) | 600FF, 650FF, 900FF, 950FF |

### Goals for Session 23

#### Priority 1: Root Cause Investigation for FF Temperature Offsets
The empirical offsets compensate for something missing in the physics model:
- **Hypothesis 1**: Solar gains are distributed incorrectly for FF cases (no HVAC loading)
- **Hypothesis 2**: Infiltration modeling differs between FF and HVAC cases
- **Hypothesis 3**: Internal gains (equipment, people, lighting) are handled differently
- **Hypothesis 4**: Night ventilation (650FF, 950FF) behaves differently without HVAC

**Investigation Steps**:
1. Compare solar gain calculations between HVAC and FF cases
2. Check if infiltration rates differ by case type
3. Verify internal gains are applied consistently
4. Compare against EnergyPlus methodology for FF cases

#### Priority 2: Address Other Failing Cases
Current validation results show:
- **600-series**: 610, 620, 630 have heating/cooling issues
- **640**: PASS (heating within range, cooling within range)
- **960**: FAIL (heating way too low, cooling way too high)

**Focus Areas**:
- Case 610: Heating 7.13 vs ref 4.36-5.79 (too high)
- Case 620: Cooling 2.29 vs ref 3.20-5.00 (too low)
- Case 630: Heating 7.59 vs ref 5.05-6.47 (too high)
- Case 960: Heating 0.06 vs ref 5.00-15.00, Cooling 22.06 vs ref 1.00-3.50

#### Priority 3: Maintain FF Corrections
Ensure Session 22 work is not broken while addressing other cases.

### Expected Outcomes
1. **Root Cause Findings**: Identify at least one physical cause for FF deviation
2. **Potential Factor Reduction**: If root cause found, reduce at least one offset value
3. **No FF Regressions**: All 4 FF cases still pass after changes
4. **Other Case Progress**: At least one other case shows improvement

### Files to Investigate
- `src/sim/engine.rs` - Core thermal modeling
- `src/sim/solar.rs` - Solar gain calculations
- `src/sim/thermal_integration.rs` - Infiltration modeling
- `src/validation/ashrae_140_validator.rs` - Current empirical corrections

### Success Criteria
- [ ] At least one root cause identified for FF temperature deviation
- [ ] At least one empirical offset reduced (or clear reason why not)
- [ ] No regressions in FF cases (4/4 still pass)
- [ ] At least one other case shows improvement
- [ ] Document findings in SESSION_23_SUMMARY.md

### Important Notes
- Use RUST_MIN_STACK=16777216 for release builds
- Run full validation after any changes
- If physics fixes require significant code changes, consider incremental approach
- Document any new empirical factors added (and aim to eliminate in future sessions)
