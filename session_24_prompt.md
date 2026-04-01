# Physics-Based Refactoring - Session 24 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 23 Recap
- **Approach**: Identified root cause - 5R1C model used instead of 6R2C for Case 960
- **Fix**: Added `model.configure_6r2c_model(0.75, 100.0)` for Case 960 in validator
- **Result**: 900-series 7/7 = **100%** PASS (was 0%), Case 960 PASS
- **Files Modified**: `src/validation/ashrae_140_validator.rs`

---

## Session 24 Task: Fix 600-Series with Physics-Based Approach

### Background
Session 23 fixed the 900-series (high-mass cases) by enabling 6R2C model for Case 960. The 600-series (low-mass) still has issues:

### Current 600-Series Results
| Case | Heating | Ref | Status | Cooling | Ref | Status |
|------|---------|-----|--------|---------|-----|--------|
| 600  | 6.79    | 5.50-7.50 | ✅ PASS | 6.53   | 8.00-10.50 | ⚠️ LOW |
| 610  | 7.13    | 4.36-5.79 | ❌ FAIL | 4.56   | 3.92-6.14 | ✅ PASS |
| 620  | 6.59    | 4.50-6.50 | ✅ PASS | 2.29   | 3.20-5.00 | ⚠️ LOW |
| 630  | 7.59    | 5.05-6.47 | ❌ FAIL | 1.12   | 2.13-3.70 | ⚠️ LOW |
| 640  | 5.18    | 2.75-3.80 | ❌ FAIL | 6.40   | 5.95-8.10 | ✅ PASS |
| 650  | 0.00    | 0.00-0.00 | ✅ PASS | 4.65   | 4.82-7.06 | ✅ PASS |

### Issues to Address
1. **Case 610**: Heating 7.13 vs ref 4.36-5.79 (63% too high)
2. **Case 630**: Heating 7.59 vs ref 5.05-6.47 (17% too high)
3. **Case 640**: Heating 5.18 vs ref 2.75-3.80 (36% too high)
4. **Case 600**: Cooling 6.53 vs ref 8.00-10.50 (19% too low)
5. **Case 620**: Cooling 2.29 vs ref 3.20-5.00 (close but low)
6. **Case 630**: Cooling 1.12 vs ref 2.13-3.70 (too low)

### Investigation Directions

#### Priority 1: Case 610 Heating (Most Critical)
- **Issue**: Heating way over reference
- **Hypothesis**: Solar gains are distributed incorrectly for E/W windows
- **Check**:
  - Solar gain calculation in `engine.rs`
  - View factor vs direct-to-air distribution
  - Window orientation handling (East/West)

#### Priority 2: Case 640 Heating
- **Issue**: Setback case heating overprediction
- **Hypothesis**: Predictive controller not using dynamic setpoints correctly
- **Check**:
  - HVAC schedule implementation
  - Recovery heating aggressiveness

#### Priority 3: Case 630 Heating
- **Issue**: Similar to 610 but with shading
- **Hypothesis**: Shading device reduces solar gains differently than expected
- **Check**: Shading factor calculation

#### Priority 4: 600-Series Cooling (All Low)
- **Issue**: All cooling predictions lower than reference
- **Hypothesis**: Internal gains or solar distribution issue
- **Check**: Internal gain distribution, solar-to-air fraction

### Expected Outcomes
1. **At least one root cause** identified for 600-series heating overprediction
2. **No regressions**: 900-series (7/7) and Case 960 still PASS
3. **FF cases** (4/4) still PASS
4. **At least one 600-series case** shows improvement

### Files to Investigate
- `src/sim/engine.rs` - Core thermal modeling, solar distribution
- `src/sim/solar.rs` - Solar gain calculations
- `src/validation/ashrae_140_validator.rs` - Current corrections

### Success Criteria
- [ ] At least one root cause identified for 600-series heating
- [ ] No regressions in 900-series (7/7 still pass)
- [ ] No regressions in Case 960 (still passes)
- [ ] No regressions in FF cases (4/4 still pass)
- [ ] At least one 600-series case shows improvement
- [ ] Document findings in SESSION_24_SUMMARY.md
- [ ] If adding empirical corrections, clearly document them for future removal

### Important Notes
- Use RUST_MIN_STACK=16777216 for release builds
- Run full validation after any changes
- Prioritize physics-based solutions over empirical corrections
- If physics fixes require significant code changes, consider incremental approach
- Document any new empirical factors added (and aim to eliminate in future sessions)
