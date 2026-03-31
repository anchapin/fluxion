# Session 44: Investigate 600-Series Low-Mass Cases

**Date**: 2026-03-27
**Follows**: Session 43 (Remove Free-Floating Empirical Factors - ✅ SUCCESS)
**Status**: 📋 PLANNED
**Priority**: HIGH - 600-series cases have 0% pass rate (6/6 failing)

## Objective

Investigate why 600-series low-mass cases are failing validation and determine if the discrepancies are due to:
1. Incorrect physics modeling of low-mass constructions
2. Empirical factors that need adjustment
3. Legitimate differences from reference tools

## Current State

### 600-Series Results (All Failing)

| Case | Heating (MWh) | Ref Range | Status | Cooling (MWh) | Ref Range | Status |
|------|---------------|-----------|--------|---------------|-----------|--------|
| 600 | 9.26 | 5.50-7.50 | ❌ +54% over max | 5.61 | 8.00-10.50 | ❌ -30% below min |
| 610 | 9.64 | 4.36-5.79 | ❌ +67% over max | 3.90 | 3.92-6.14 | ❌ -0.5% below min |
| 620 | 8.43 | 4.50-6.50 | ❌ +30% over max | 1.96 | 3.20-5.00 | ❌ -39% below min |
| 630 | 9.40 | 5.05-6.47 | ❌ +45% over max | 1.01 | 2.13-3.70 | ❌ -53% below min |
| 640 | 7.12 | 2.75-3.80 | ❌ +87% over max | 5.45 | 5.95-8.10 | ❌ -8% below min |
| 650 | 0.00 | 0.00-0.00 | ✅ PASS | 4.31 | 4.82-7.06 | ❌ -11% below min |

**Key Pattern**:
- **Heating**: All cases significantly overpredicting (+30% to +87%)
- **Cooling**: All cases underpredicting (-0.5% to -53%)
- **Case 650**: Only passing metric (heating = 0.00 MWh, as expected)

### 600-Series Free-Floating Results

| Case | Max Temp | Ref Range | Min Temp | Status |
|------|----------|-----------|----------|--------|
| 600FF | 45.66°C | 64.90-75.10°C | -6.09°C | ❌ 20-30°C below max |
| 650FF | 43.71°C | 63.20-73.50°C | -10.49°C | ❌ 20-30°C below max |

Both low-mass free-floating cases have max temps 20-30°C below reference ranges, suggesting:
1. Low-mass buildings may have fundamentally different thermal behavior
2. Current model may not capture low-mass physics correctly
3. Reference tools may use different assumptions for low-mass construction

## Known Issues from Session 40

### Mode-Specific Coupling Factors (Session 40)

**Location**: `src/sim/engine.rs` lines 1128-1133

```rust
// Mode-specific coupling factors for low-mass buildings (600-series)
// Low-mass buildings have faster thermal response than high-mass
let h_tr_em_heating_factor = if case_id.starts_with('6') { 0.6 } else { 1.0 };
let h_tr_em_cooling_factor = if case_id.starts_with('6') { 1.4 } else { 1.0 };
```

**Purpose**: Adjust thermal mass coupling for low-mass vs high-mass buildings
- **Heating factor = 0.6**: Reduces thermal mass coupling during heating
- **Cooling factor = 1.4**: Increases thermal mass coupling during cooling

**Problem**: These factors may be incorrect, causing:
- **Heating overprediction**: Factor 0.6 may be too low (not enough thermal mass buffering)
- **Cooling underprediction**: Factor 1.4 may be too high (too much thermal mass heat sink)

## Investigation Plan

### Priority 1: Diagnose Root Cause

**Step 1: Analyze Thermal Mass Dynamics**
- Compare thermal mass temperatures between 600-series and 900-series
- Check if low-mass thermal mass is heating/cooling too fast
- Verify h_tr_ms conductance values are correct for low-mass construction

**Step 2: Review Mode-Specific Factors**
- Test different values for h_tr_em_heating_factor and h_tr_em_cooling_factor
- Determine if factors should be:
  - Closer to 1.0 (less adjustment)
  - Swapped (heating > cooling)
  - Case-specific (different for each 600-series case)

**Step 3: Compare with Reference Tools**
- Research how EnergyPlus, ESP-r, TRNSYS model low-mass buildings
- Check if reference tools use:
  - Different time constants for low-mass
  - Different thermal mass coupling algorithms
  - Different solar distribution for lightweight construction

### Priority 2: Test Potential Solutions

**Solution A: Adjust Mode-Specific Factors**
- Try h_tr_em_heating_factor = 0.8 or 1.0 (increase from 0.6)
- Try h_tr_em_cooling_factor = 1.0 or 0.8 (decrease from 1.4)
- Test each 600-series case individually

**Solution B: Case-Specific Factors**
- Different factors for each case based on construction details
- Cases 600, 610: Light construction (wood frame)
- Cases 620, 630: Medium construction
- Cases 640, 650: Heavy construction (setback, night ventilation)

**Solution C: Physics-Based Low-Mass Model**
- Implement faster time constants for low-mass buildings
- Reduce thermal mass heat capacity for lightweight construction
- Adjust solar distribution for low thermal mass

### Priority 3: Review Free-Floating Discrepancies

**Investigation**:
- Why are 600FF and 650FF max temps 20-30°C below reference?
- Is this physically correct for low-mass construction?
- Do reference tools use different solar gain assumptions for low-mass?

**Potential Causes**:
1. **Solar gains too low**: Low-mass may need different solar distribution
2. **Heat loss too high**: Low-mass may have higher effective U-values
3. **Thermal mass coupling too high**: Mass absorbing too much heat
4. **Reference tool differences**: Different assumptions for low-mass physics

## Expected Outcomes

### Best Case: Physics-Based Fix
- Identify correct physics model for low-mass buildings
- Adjust mode-specific factors based on thermal mass principles
- Achieve ≥50% pass rate for 600-series (3/6 cases passing)

### Medium Case: Partial Improvement
- Improve some cases but not all
- May need case-specific adjustments
- Achieve ≥25% pass rate for 600-series (1-2/6 cases passing)

### Worst Case: Fundamental Differences
- Current model correctly represents low-mass physics
- Reference tools use fundamentally different assumptions
- May need to accept discrepancies as legitimate differences

## Success Criteria

- [ ] Root cause of 600-series failures identified
- [ ] At least 1-2 600-series cases passing (≥25% pass rate)
- [ ] Better understanding of low-mass vs high-mass thermal physics
- [ ] Decision on whether to adjust factors or accept differences
- [ ] Investigation documented in SESSION_44_SUMMARY.md
- [ ] physics_based_refactor.md updated with findings

## Files to Examine

1. **`src/sim/engine.rs`**:
   - Lines 1128-1133: Mode-specific coupling factors
   - Lines 1390-1395: 600-series factor application
   - Lines 2338-2345: Time constant sensitivity correction

2. **`src/validation/ashrae_140_validator.rs`**:
   - 600-series case specifications
   - Construction details for each case
   - Reference ranges and tolerances

3. **Session Documents**:
   - `SESSION_40_SUMMARY.md`: Original implementation of mode-specific factors
   - `SESSION_43_SUMMARY.md`: Free-floating results
   - `physics_based_refactor.md`: Overall progress tracking

## Diagnostic Commands

```bash
# Run 600-series cases
cargo run --release --bin fluxion validate --case 600
cargo run --release --bin fluxion validate --case 610
cargo run --release --bin fluxion validate --case 620
cargo run --release --bin fluxion validate --case 630
cargo run --release --bin fluxion validate --case 640
cargo run --release --bin fluxion validate --case 650

# Run free-floating 600-series
cargo run --release --bin fluxion validate --case 600FF
cargo run --release --bin fluxion validate --case 650FF

# Run all cases for comparison
cargo run --release --bin fluxion validate --all

# Build with optimizations
cargo build --release

# Quick syntax check
cargo check
```

## Additional Context

### Low-Mass vs High-Mass Construction

**600-Series (Low-Mass)**:
- Lightweight walls (wood frame, etc.)
- Less thermal mass per unit area
- Faster thermal response
- Lower heat capacity

**900-Series (High-Mass)**:
- Heavyweight walls (concrete, etc.)
- More thermal mass per unit area
- Slower thermal response
- Higher heat capacity

### Key Physics Question

**Do low-mass buildings have fundamentally different thermal behavior than high-mass buildings?**

- **Hypothesis 1**: Yes - low-mass buildings heat/cool faster, have different time constants
- **Hypothesis 2**: No - same physics, just different parameters (thermal mass values)

**Session 44 will test these hypotheses** and determine the correct modeling approach.

## References

- **SESSION_40_SUMMARY.md**: Original implementation of mode-specific factors
- **SESSION_43_SUMMARY.md**: Free-floating results and remaining discrepancies
- **physics_based_refactor.md**: Overall progress and remaining work
- **ASHRAE 140 Standard**: 600-series case specifications and construction details
- **ISO 13790**: 5R1C thermal network standard for low-mass buildings

---

**Session 44 Goal**: Investigate and diagnose 600-series low-mass case failures, with target of achieving ≥25% pass rate (1-2/6 cases passing) and better understanding of low-mass thermal physics.
