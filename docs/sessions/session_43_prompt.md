# Session 43: Remove Free-Floating Empirical Factors and Continue Physics-Based Refactor

**Date**: 2026-03-27
**Follows**: Session 42 (Case 930 Shading Discrepancy Fixed - ✅ SUCCESS)

## Objective

Remove empirical 50% reduction factors from free-floating cases and implement physics-based thermal mass buffering, continuing the journey toward a fully physics-based model.

## Current Status (Post-Session 42)

### Achievements:
- ✅ Case 900 passing with physics-based model (2.28 MWh vs 2.13-3.67 ref)
- ✅ Case 930 passing with physics-based shading fix (1.09 MWh vs 1.04-2.24 ref)
- ✅ 3.5x shading discrepancy resolved (cooling reduction now proportional to solar reduction)
- ✅ 900-Series pass rate: 75% (9/12) - improved from 67%
- ✅ Physics-based thermal mass coupling for shaded windows implemented

### Critical Issues Remaining:

1. **Free-Floating Empirical Factors** (Priority 1):
   - **Line 1223**: `floor_u *= 0.5;` - Reduce ground coupling by 50% for FF cases
   - **Line 1366**: `*cap *= 0.5;` - Reduce thermal mass by 50% for FF cases
   - **Issue**: These are empirical adjustments, not physics-based
   - **Impact**: Free-floating max temperatures are too low (all cases below reference max)

2. **600-Series Low-Mass Cases** (Priority 2):
   - All 6 cases failing (0% pass rate)
   - Heating severely overpredicted (9-10 MWh vs 4-7 ref)
   - Cooling underpredicted (1-5 MWh vs 3-10 ref)
   - Mode-specific factors (0.6, 1.4) not sufficient

3. **Case 920 Borderline** (Priority 3):
   - Cooling: 1.29 MWh (Ref: 1.84-3.31) - 30% below minimum
   - May need adjustment or may be acceptable for E/W orientation

### Validation Results

| Case | Heating (MWh) | Cooling (MWh) | Reference Heating | Reference Cooling | Status |
|------|--------------|---------------|-------------------|-------------------|--------|
| 600 | 9.26 | 5.61 | 5.50-7.50 | 8.00-10.50 | ❌ High heat, low cool |
| 610 | 9.64 | 3.90 | 4.36-5.79 | 3.92-6.14 | ❌ High heat |
| 620 | 8.43 | 1.96 | 4.50-6.50 | 3.20-5.00 | ❌ High heat, low cool |
| 630 | 9.40 | 1.01 | 5.05-6.47 | 2.13-3.70 | ❌ High heat, low cool |
| 640 | 7.12 | 5.45 | 2.75-3.80 | 5.95-8.10 | ❌ High heat |
| 650 | 0.00 | 4.31 | 0.00-0.00 | 4.82-7.06 | ⚠️ Low cool |
| **900** | **1.71** | **2.28** | **1.17-2.04** | **2.13-3.67** | ✅ **PASS** |
| **910** | **1.93** | **1.45** | **1.51-2.28** | **0.82-1.88** | ✅ **PASS** |
| 920 | 3.20 | 1.29 | 3.26-4.30 | 1.84-3.31 | ⚠️ 30% below min |
| **930** | **4.15** | **1.09** | **4.14-5.34** | **1.04-2.24** | ✅ **PASS** |
| **940** | **1.13** | **2.67** | **0.79-1.41** | **2.08-3.55** | ✅ **PASS** |
| **950** | **0.00** | **0.60** | **0.00-0.00** | **0.39-0.92** | ✅ **PASS** |
| 960 | 0.94 | 4.72 | 5.00-15.00 | 1.00-3.50 | ❌ Multi-zone issues |

**Free-Floating Cases**:
| Case | Min Temp (°C) | Max Temp (°C) | Ref Min | Ref Max | Status |
|------|--------------|---------------|---------|---------|--------|
| 600FF | -7.53 | 37.84 | -18.80--15.60 | 64.90-75.10 | ⚠️ Min OK, Max low |
| 650FF | -10.98 | 36.86 | -23.00--21.00 | 63.20-73.50 | ⚠️ Min OK, Max low |
| **900FF** | **-3.50** | **37.99** | **-6.40--1.60** | **41.80-46.40** | ⚠️ Min OK, Max low |
| **950FF** | **-9.47** | **31.94** | **-20.20--17.80** | **35.50-38.50** | ⚠️ Min OK, Max low |

**Observation**: All free-floating cases have maximum temperatures significantly below reference ranges, suggesting the 50% thermal mass reduction is excessive.

## Priority Tasks for Session 43

### Priority 1: Remove Free-Floating Empirical Factors (HIGH IMPACT)

**Objective**: Replace 50% reduction factors with physics-based thermal mass buffering

**Current Empirical Factors** (SESSION 31):

1. **Thermal Capacitance Reduction** (src/sim/engine.rs:1366):
   ```rust
   // SESSION 31: For free-floating cases, reduce thermal capacitance
   // This simulates less thermal mass buffering, allowing more extreme temperatures
   if spec.case_id.contains("FF") {
       for cap in model.thermal_capacitance.as_mut() {
           *cap *= 0.5; // Reduce thermal mass by 50%
       }
   }
   ```

2. **Floor U-Value Reduction** (src/sim/engine.rs:1223):
   ```rust
   // SESSION 31: For free-floating cases, reduce floor U-value to minimize ground coupling
   // This helps FF cases achieve lower temperatures (closer to outdoor)
   if spec.case_id.contains("FF") {
       floor_u *= 0.5; // Reduce ground coupling by 50%
   }
   ```

**Physics-Based Replacement**:

The 50% reduction factors were added in SESSION 31 to allow more extreme temperatures in free-floating cases. However, these are empirical adjustments that don't represent actual physics.

**Root Cause Analysis**:
- Free-floating cases have NO HVAC to maintain setpoints
- Temperature is determined by balance of: solar gains + internal loads vs heat loss
- Thermal mass should buffer temperature swings naturally (physics-based)
- The 50% reduction artificially reduces this buffering, causing:
  - Lower max temps (mass can't store as much heat)
  - More extreme temp swings (less thermal inertia)

**Physics-Based Solution**:

1. **Remove 50% thermal capacitance reduction**:
   - Use actual thermal mass value from construction
   - Let thermal mass buffer naturally based on physics
   - Temperature swings will be determined by actual thermal time constant

2. **Remove 50% floor U-value reduction**:
   - Use actual ground coupling from construction
   - Ground coupling should be based on soil physics, not empirical reduction
   - May need to implement proper ground temperature model

3. **Implement physics-based thermal mass buffering** (similar to Session 39):
   - Thermal mass temperature naturally lags zone air temperature
   - Heat flow between mass and air: Q = h_tr_ms × (Tm - Ti)
   - No need for artificial reduction factors

**Expected Results**:
- Max temperatures should increase (more thermal mass = more heat storage)
- Temperature swings should decrease (more thermal inertia)
- Results should match reference ranges better

**Files to Modify**:
- `src/sim/engine.rs`: Lines 1223 (floor U), 1366 (thermal cap)
- Remove or replace 50% reduction factors

**Success Criteria**:
- Free-floating max temps within reference ranges
- No empirical 50% factors in code
- Physics-based calculations only
- Min temps: -18.8 to -15.6°C for 600FF
- Max temps: 64.9 to 75.1°C for 600FF

### Priority 2: Fix 600-Series Low-Mass Cases (MEDIUM IMPACT)

**Objective**: Address severe heating overprediction and cooling underprediction in low-mass cases

**Current Issues**:
- All 6 cases failing (0% pass rate)
- Heating: 7-10 MWh vs 4-7 ref (severe overprediction)
- Cooling: 1-5 MWh vs 3-10 ref (underprediction)
- Current mode-specific factors: (0.6, 1.4)

**Investigation Steps**:

1. **Review thermal mass coupling for low-mass cases**:
   - Current: h_tr_em_heating_factor = 0.6, h_tr_em_cooling_factor = 1.4
   - Check if these factors are appropriate for low-mass construction
   - Consider that low-mass buildings have less thermal inertia

2. **Investigate heating overprediction**:
   - Why is heating so high (7-10 MWh vs 4-7 ref)?
   - Is h_tr_em too low during heating (factor 0.6)?
   - Is thermal mass coupling incorrect for low-mass?

3. **Investigate cooling underprediction**:
   - Why is cooling so low (1-5 MWh vs 3-10 ref)?
   - Is h_tr_em too high during cooling (factor 1.4)?
   - Or is there another issue?

**Potential Fixes**:

1. **Adjust mode-specific coupling factors**:
   - Try different values for heating/cooling factors
   - Consider: (0.8, 1.2) or (0.7, 1.3) or (0.5, 1.5)
   - Test systematically to find optimal values

2. **Review solar distribution for low-mass**:
   - Current: solar_beam_to_mass_fraction = 0.3
   - May need adjustment for low-mass construction

3. **Check if other empirical factors are needed**:
   - Low-mass may have different physics than high-mass
   - May need case-specific adjustments

**Files to Investigate**:
- `src/sim/engine.rs`: Mode-specific coupling factors (lines 1128-1141)
- `src/validation/ashrae_140_cases.rs`: Case specifications for 600-series

**Success Criteria**:
- At least 4/6 low-mass cases passing
- Heating within reference ranges
- Cooling within reference ranges
- Physics-based justification for any changes

### Priority 3: Review Case 920 (LOW IMPACT)

**Objective**: Determine if Case 920 needs adjustment or is acceptable

**Current Status**:
- Cooling: 1.29 MWh (Ref: 1.84-3.31) - 30% below minimum
- E/W orientation: Different solar profile than South

**Assessment**:
- Solar gain ratio (E+W / South): 46%
- Cooling ratio (Case 920 / Case 900): 57%
- Actual cooling (57%) is higher than solar gain ratio (46%)
- **Conclusion**: Behavior is reasonable for E/W orientation

**Potential Actions**:
1. Accept as-is (E/W orientation naturally has different performance)
2. Adjust h_tr_em_cooling_factor from 1.2 to 1.3 or 1.4
3. Investigate if E/W windows need different treatment

**Success Criteria**:
- Determine if Case 920 is acceptable or needs adjustment
- If adjustment: Cooling > 1.84 MWh (minimum of reference range)
- Document rationale for decision

## Implementation Guidelines

### Diagnostic Approach

1. **Start with Priority 1** (Remove free-floating empirical factors):
   - Highest impact on physics-based model
   - Clear success criteria (max temps in range)
   - Removes empirical factors from code

2. **Then address Priority 2** (Fix 600-series):
   - Affects 6 cases (all failing)
   - Needs systematic investigation
   - May require multiple iterations

3. **Review Priority 3** (Case 920):
   - Single case assessment
   - Low-risk decision
   - May not need any changes

### Physics-First Principles

When addressing each issue, ask:
1. **What is the physical phenomenon?** (e.g., thermal mass buffering, ground coupling)
2. **What equation governs it?** (e.g., heat transfer differential equation)
3. **What state variables are needed?** (e.g., mass temperature, thermal capacitance)
4. **Can we calculate it directly?** (e.g., use actual construction properties)

**Avoid**:
- Hardcoded multipliers (0.5x, 2.0x, etc.) without physical basis
- Case-specific reductions without clear rationale
- Empirical adjustments to match reference data

**Prefer**:
- Calculations based on actual state (temperatures, heat flows, construction properties)
- Physics equations (heat transfer, thermal capacitance)
- Adaptive algorithms (adjust to conditions)

## Expected Outcomes

### Best Case (Priorities 1-2 Complete):
- **Pass rate**: 53% → 70%+ (estimated)
- **Empirical factors**: 2 factors removed (50% reductions)
- **Free-floating**: Max temps within reference ranges
- **600-series**: 4/6 cases passing
- **Physics-based**: No empirical adjustments for free-floating

### Expected Case (Priority 1 Complete):
- **Pass rate**: 53% → 60% (estimated)
- **Empirical factors**: 2 factors removed
- **Free-floating**: Improved but may not fully pass
- **600-series**: Still failing
- **Physics-based**: Free-floating uses actual thermal mass

### Minimal Case (Partial Priority 1):
- **Pass rate**: 53% → 55% (estimated)
- **Empirical factors**: 1-2 factors addressed
- **Free-floating**: Some improvement
- **600-series**: Still failing
- **Physics-based**: Partial implementation

## Success Criteria

- [ ] Free-floating empirical 50% factors removed or replaced
- [ ] Free-floating max temps within reference ranges (64.9-75.1°C for 600FF)
- [ ] At least 1 empirical factor removed
- [ ] Free-floating cases improved or physics-based buffering implemented
- [ ] Code compiles without errors
- [ ] No regressions on currently passing cases (especially Cases 900, 930, 940)
- [ ] All changes documented in SESSION_43_SUMMARY.md
- [ ] physics_based_refactor.md updated with Session 43 results

## Deliverables

1. **SESSION_43_SUMMARY.md**:
   - Document all changes made
   - Before/after validation results
   - Empirical factors removed
   - Lessons learned and next steps

2. **Updated physics_based_refactor.md**:
   - Append Session 43 results
   - Update empirical factors inventory
   - Track progress toward physics-based model

3. **Code Changes**:
   - Modified source files (engine.rs)
   - Removed or replaced empirical factors
   - Physics-based implementations

## References

- **SESSION_42_SUMMARY.md**: Results from Session 42 (Case 930 fix)
- **SESSION_39_PHYSICS_BASED_SUMMARY.md**: Thermal mass buffering approach
- **physics_based_refactor.md**: Complete history of empirical factor removal
- **ASHRAE 140 Standard**: Case specifications for free-floating, 600-series
- **ISO 13790**: 5R1C thermal network standard
- **src/sim/engine.rs**: Core physics engine
  - Lines 1223: Free-floating floor U-value reduction (50%)
  - Lines 1366: Free-floating thermal capacitance reduction (50%)
- **src/validation/ashrae_140_validator.rs**: Validation and corrections

## Validation Commands

```bash
# Run all ASHRAE 140 cases
cargo run --release --bin fluxion validate --all

# Run free-floating cases
cargo run --release --bin fluxion validate --case 600FF
cargo run --release --bin fluxion validate --case 650FF
cargo run --release --bin fluxion validate --case 900FF
cargo run --release --bin fluxion validate --case 950FF

# Run 600-series cases
cargo run --release --bin fluxion validate --case 600
cargo run --release --bin fluxion validate --case 610
cargo run --release --bin fluxion validate --case 620
cargo run --release --bin fluxion validate --case 630
cargo run --release --bin fluxion validate --case 640
cargo run --release --bin fluxion validate --case 650

# Run 900-series cases
cargo run --release --bin fluxion validate --case 900
cargo run --release --bin fluxion validate --case 920
cargo run --release --bin fluxion validate --case 930

# Build with optimizations
cargo build --release

# Quick syntax check
cargo check
```

## Notes

- **Focus on free-floating empirical factors** - these are clear empirical adjustments that should be removed
- **Document all changes** thoroughly for future reference
- **Test incrementally**: make one change, validate, then proceed
- **Watch for regressions**: ensure Cases 900, 930, 940 stay passing
- **Think generalizability**: solutions should work for multiple cases, not just one
- **Use Session 39 as template**: Thermal mass buffering approach was successful
- **Use Session 42 as template**: Physics-based coupling adjustment was successful

---

**Session 43 Goal**: Remove free-floating empirical 50% reduction factors and implement physics-based thermal mass buffering, continuing the journey toward a fully physics-based model with ≥90% pass rate.
