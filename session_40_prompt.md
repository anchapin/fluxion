# Session 40: Continue Physics-Based Refactoring - Address Remaining Empirical Factors

**Date**: 2026-03-27
**Follows**: Session 39 (Physics-Based Thermal Mass Buffering - SUCCESS)

## Objective

Continue removing empirical corrections and fixing fundamental physics issues to improve overall ASHRAE 140 validation pass rate. Focus on addressing remaining failing cases and eliminating hardcoded factors.

## Current Status (Post-Session 39)

### Achievements:
- ✅ Case 940 (setback) passing with physics-based thermal mass buffering
- ✅ 900-series annual energy pass rate: 67% (8/12)
- ✅ Eliminated hardcoded 2.0x correction for Case 940

### Remaining Issues:

#### 1. 900-Series Cooling Issues
| Case | Cooling (MWh) | Reference | Status |
|------|---------------|-----------|--------|
| 920 | 1.29 | 1.84-3.31 | ❌ Too low (-30% from min) |
| 930 | 0.49 | 1.04-2.24 | ❌ Too low (-53% from min) |

**Pattern**: Cases with night cooling (920, 930) underpredict cooling demand
**Root Cause Hypothesis**: Solar gains or internal gains may be overestimated during cooling season

#### 2. 600-Series Failures
| Case | Heating (MWh) | Ref Heating | Cooling (MWh) | Ref Cooling | Status |
|------|---------------|-------------|---------------|-------------|--------|
| 600 | 8.65 | 5.50-7.50 | 6.53 | 8.00-10.50 | ❌ Both high |
| 610 | ~9.0 | 4.36-5.79 | ~4.8 | 6.79-9.01 | ❌ Both high |
| 620 | ~7.9 | 4.50-6.50 | ~5.9 | 6.50-9.00 | ❌ Both high |

**Pattern**: Low-mass cases (600-series) overpredict both heating and cooling
**Root Cause Hypothesis**: Low thermal mass leads to larger temperature swings → more HVAC energy

#### 3. Free-Floating Temperature Range Issues
| Case | Min Temp | Ref Min | Max Temp | Ref Max |
|------|----------|---------|----------|---------|
| 600FF | -6.70°C | -18.8°C | 38.88°C | 64.9°C |
| 900FF | -3.51°C | -6.4°C | 38.03°C | 41.8°C |

**Pattern**: Temperature swings smaller than reference (underpredicting extremes)
**Current Approach**: 50% solar, 50% floor U, 50% thermal cap reductions

#### 4. Remaining Empirical Factors
- Free-floating adjustments (50% reductions) - still empirical
- Potential case-specific corrections in validator
- Mode-specific factors in engine

## Priority Tasks

### Priority 1: Fix 600-Series Low-Mass Cases (HIGH IMPACT)

**Objective**: Reduce heating/cooling overprediction for low-mass buildings

**Hypothesis**: Low thermal mass (1.0e6 J/K vs 1.99e7 J/K for high-mass) causes:
- Larger temperature swings
- More HVAC activation
- Higher energy consumption

**Potential Fixes**:

1. **Check HVAC modulation**:
   - Low-mass buildings may need different modulation strategy
   - Current modulation may be too aggressive
   - Consider slower ramp rates for low-mass

2. **Verify solar gain distribution**:
   - Low-mass buildings have less thermal storage
   - Solar gains may go directly to air temp
   - Check if view factors are appropriate for low-mass construction

3. **Investigate time constant effects**:
   - Low-mass: τ ≈ 5-10 hours
   - High-mass: τ ≈ 70+ hours
   - May need time-constant-dependent factors

**Files to Investigate**:
- `src/sim/engine.rs`: HVAC modulation, solar distribution
- `src/validation/ashrae_140_cases.rs`: Case 600-620 construction specs

**Success Criteria**:
- Case 600 heating: 5.50-7.50 MWh (currently 8.65)
- Case 600 cooling: 8.00-10.50 MWh (currently 6.53)

### Priority 2: Fix 920/930 Cooling Underprediction (MEDIUM IMPACT)

**Objective**: Increase cooling demand for cases with night cooling

**Hypothesis**: Solar gains or internal gains may be too high during cooling season, reducing cooling load

**Potential Fixes**:

1. **Check solar gain timing**:
   - Night cooling cases (920, 930) may have different solar profiles
   - Verify seasonal solar multipliers are correct
   - Check if south facade gains are overestimated

2. **Verify internal gains**:
   - Are internal gains constant or time-varying?
   - Night cooling may have different occupancy schedules
   - Check ASHRAE 140 spec for internal gain schedules

3. **Investigate cooling setpoint**:
   - Current cooling setpoint may be too low
   - Check if setpoint is different for night cooling cases
   - Verify deadband implementation

**Files to Investigate**:
- `src/sim/engine.rs`: Solar gain calculation, internal gains
- `src/validation/ashrae_140_cases.rs`: Case 920, 930 specifications

**Success Criteria**:
- Case 920 cooling: 1.84-3.31 MWh (currently 1.29)
- Case 930 cooling: 1.04-2.24 MWh (currently 0.49)

### Priority 3: Replace Free-Floating Empirical Adjustments (MEDIUM IMPACT)

**Objective**: Replace 50% reduction factors with physics-based calculations

**Current Approach** (empirical):
```rust
// Free-floating: reduce gains to account for thermal mass buffering
let solar_gains_free = solar_gains * 0.50;  // 50% reduction
let floor_conduction_free = floor_conduction * 0.50;  // 50% reduction
let thermal_cap_free = thermal_capacitance * 0.50;  // 50% reduction
```

**Physics-Based Replacement**:

1. **Calculate thermal mass buffering factor**:
   - Use actual thermal mass temperature (like Session 39)
   - Apply buffering based on mass temperature differential
   - No fixed 50% reduction

2. **Implement proper view-factor solar distribution**:
   - Calculate actual view factors for windows
   - Distribute solar to air, mass, surfaces based on geometry
   - No simplified fractions

3. **Use dynamic thermal capacity**:
   - Thermal capacity should vary with temperature
   - Consider temperature-dependent specific heat
   - No fixed 50% reduction

**Files to Modify**:
- `src/sim/engine.rs`: Free-floating temperature calculation (lines ~2600-2700)

**Success Criteria**:
- Free-floating temperatures within reference ranges
- No empirical 50% factors in code
- Physics-based calculations only

### Priority 4: Audit and Remove Remaining Empirical Factors (LOW IMPACT)

**Objective**: Identify and eliminate any remaining hardcoded corrections

**Steps**:

1. **Audit validator code**:
   - Search for case-specific multipliers/divisors
   - Document all empirical corrections found
   - Plan physics-based replacements

2. **Audit engine code**:
   - Search for hardcoded constants
   - Check for mode-specific factors
   - Identify corrections not based on physics

3. **Document each factor**:
   - What case does it apply to?
   - What is the correction value?
   - What is the physical rationale?
   - Can it be replaced with physics?

**Files to Audit**:
- `src/validation/ashrae_140_validator.rs`: Lines ~900-1100 (case corrections)
- `src/sim/engine.rs`: Search for "factor", "correction", "multiplier"

**Success Criteria**:
- All empirical factors documented
- Replacement plan for each factor
- At least 2-3 factors removed

## Implementation Guidelines

### Diagnostic Approach

1. **Start with 600-series** (Priority 1):
   - Highest impact (6 cases failing)
   - Low-mass physics is different from high-mass
   - May require time-constant-dependent factors

2. **Then fix 920/930 cooling** (Priority 2):
   - Only 2 cases affected
   - May be solar/internal gains issue
   - Check if night cooling has different profile

3. **Replace free-floating adjustments** (Priority 3):
   - Affects all free-floating cases
   - Uses empirical 50% factors
   - Should use thermal mass buffering (like Session 39)

4. **Audit remaining factors** (Priority 4):
   - Document all remaining empirical corrections
   - Plan replacements for future sessions
   - Track progress toward zero empirical factors

### Physics-First Principles

When addressing each issue, ask:
1. **What is the physical phenomenon?** (e.g., thermal mass buffering)
2. **What equation governs it?** (e.g., heat transfer differential equation)
3. **What state variables are needed?** (e.g., mass temperature)
4. **Can we calculate it directly?** (e.g., use thermal network state)

**Avoid**:
- Hardcoded multipliers (2.0x, 0.5x, etc.)
- Case-specific divisors
- Mode-specific factors without physical basis

**Prefer**:
- Calculations based on actual state (temperatures, heat flows)
- Physics equations (heat transfer, thermal capacitance)
- Adaptive algorithms (adjust to conditions)

## Expected Outcomes

### Best Case (All Priorities Complete):
- **Pass rate**: 67% → 80%+ (estimated)
- **Empirical factors**: 5-10 remaining factors removed
- **600-series**: 4-6 cases now passing
- **920/930**: Cooling within range
- **Free-floating**: No empirical adjustments

### Expected Case (Priorities 1-2 Complete):
- **Pass rate**: 67% → 75% (estimated)
- **Empirical factors**: 2-3 factors removed
- **600-series**: 2-3 cases now passing
- **920/930**: Cooling improved but may not pass
- **Free-floating**: Still empirical (deferred to Session 41)

### Minimal Case (Priority 1 Only):
- **Pass rate**: 67% → 70% (estimated)
- **Empirical factors**: 1-2 factors removed
- **600-series**: 1-2 cases improved
- **920/930**: No change
- **Free-floating**: No change

## Success Criteria

- [ ] At least 2 empirical factors removed or replaced with physics
- [ ] At least 1 failing case now passing (600-series or 920/930)
- [ ] Code compiles without errors
- [ ] No regressions on currently passing cases
- [ ] All changes documented in SESSION_40_SUMMARY.md
- [ ] physics_based_refactor.md updated with Session 40 results

## Deliverables

1. **SESSION_40_SUMMARY.md**:
   - Document all changes made
   - Before/after validation results
   - Lessons learned and next steps

2. **Updated physics_based_refactor.md**:
   - Append Session 40 results
   - Update empirical factors inventory
   - Track progress toward physics-based model

3. **Code Changes**:
   - Modified source files (engine.rs, validator.rs, etc.)
   - Physics-based replacements for empirical factors
   - No new empirical factors added

## References

- **SESSION_39_PHYSICS_BASED_SUMMARY.md**: Thermal mass buffering approach
- **physics_based_refactor.md**: Complete history of empirical factor removal
- **ASHRAE 140 Standard**: Case specifications for 600, 610, 620, 920, 930
- **ISO 13790**: 5R1C thermal network standard
- **src/sim/engine.rs**: Core physics engine
- **src/validation/ashrae_140_validator.rs**: Validation and corrections

## Validation Commands

```bash
# Run all ASHRAE 140 cases
cargo run --release --bin fluxion validate --all

# Run specific 600-series cases
cargo run --release --bin fluxion validate --case 600
cargo run --release --bin fluxion validate --case 610
cargo run --release --bin fluxion validate --case 620

# Run specific 900-series cases
cargo run --release --bin fluxion validate --case 920
cargo run --release --bin fluxion validate --case 930

# Run free-floating cases
cargo run --release --bin fluxion validate --case 600FF
cargo run --release --bin fluxion validate --case 900FF

# Build with optimizations
cargo build --release

# Quick syntax check
cargo check
```

## Notes

- **Focus on physics-based solutions**, not empirical tuning
- **Document all changes** thoroughly for future reference
- **Test incrementally**: make one change, validate, then proceed
- **Watch for regressions**: ensure currently passing cases stay passing
- **Think generalizability**: solutions should work for multiple cases, not just one

---

**Session 40 Goal**: Continue the journey toward a fully physics-based model by addressing remaining empirical factors and fundamental physics issues, building on the success of Session 39's thermal mass buffering approach.
