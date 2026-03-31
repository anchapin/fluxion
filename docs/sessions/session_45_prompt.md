# Session 45: Fix 5R1C Conductance Calculation for Low-Mass Buildings

**Date**: 2026-03-27
**Follows**: Session 44 (Investigate 600-Series Low-Mass Cases - Root Cause Identified)
**Status**: 📋 PLANNED
**Priority**: HIGH - Addresses root cause of 600-series failures (0% pass rate)

## Objective

Investigate and fix the 5R1C conductance calculation for low-mass buildings to resolve the time constant contradiction:
- **Current**: Model reports "LOW-MASS" but τ = 5 hours (should be 1-2 hours)
- **Expected**: τ ≈ 1-2 hours for low-mass construction
- **Goal**: Correct h_tr_ms and h_tr_is conductances to produce proper time constants

## Problem Statement from Session 44

**Contradiction Identified**:
```
Case 600 Thermal Properties:
  Total Thermal Capacitance: 2.40e6 J/K
  Mass Class: LOW-MASS
  Coupling Ratio (h_tr_em / h_tr_ms): 0.080

Time Constant Analysis:
  Time Constant (τ): 5.00 hours
  ✓  Slow response (τ ≥ 4 hours) - HIGH MASS
```

**Issue**: Thermal capacitance is correct for low-mass (2.4e6 J/K), but time constant is wrong (5 hours vs expected 1-2 hours).

**Root Cause**: 5R1C conductances are calculated incorrectly for low-mass construction:
- `h_tr_ms` (mass→surface) = 1092 W/K - TOO HIGH for low-mass
- `h_tr_is` (surface→interior) = 550 W/K - May be incorrect
- Result: τ = C / U_total is too high → wrong energy balance

## Investigation Plan

### Priority 1: Understand Current Conductance Calculation

**Step 1: Examine Conductance Code**
- File: `src/sim/engine.rs` lines 1100-1300
- Understand how `h_tr_ms` and `h_tr_is` are calculated
- Check if there are differences between low-mass and high-mass construction
- Identify formulas used for conductance calculation

**Step 2: Review ASHRAE 140 Specifications**
- Check ASHRAE 140 standard for low-mass construction requirements
- Verify expected conductance values for Case 600
- Compare with current implementation

**Step 3: Compare with Reference Tools**
- Research how EnergyPlus calculates conductances for low-mass
- Check ISO 13790 standard for 5R1C network parameters
- Identify if there should be construction-specific adjustments

### Priority 2: Test Alternative Conductances

**Hypothesis**: Low-mass buildings need different h_tr_ms/h_tr_is ratios than high-mass

**Test Matrix**:
| Test | h_tr_ms | h_tr_is | Expected τ | Goal |
|------|---------|---------|------------|------|
| Current | 1092 W/K | 550 W/K | 5.0 hours | Baseline |
| Test 1 | 500 W/K | 550 W/K | ~2.5 hours | Reduce h_tr_ms |
| Test 2 | 300 W/K | 550 W/K | ~1.5 hours | Match expected τ |
| Test 3 | 1092 W/K | 800 W/K | ~4.0 hours | Increase h_tr_is |
| Test 4 | 500 W/K | 800 W/K | ~2.0 hours | **TARGET** |

**Success Criteria**:
- Time constant τ ≈ 1-2 hours for low-mass
- Heating loads reduced toward reference range
- Cooling loads increased toward reference range
- At least 1-2 cases passing (≥25% pass rate)

### Priority 3: Implement Physics-Based Solution

**If tests successful**:
1. Identify correct formula for low-mass conductances
2. Implement construction-specific conductance calculation
3. Apply to all 600-series cases
4. Validate against reference ranges
5. Document changes

**If tests unsuccessful**:
1. Accept that current approach may be fundamentally different from reference tools
2. Document as "5R1C Model Limitation - Low-Mass Construction"
3. Focus on improving 900-series results

## Expected Outcomes

### Best Case: Conductance Fix
- Identify correct h_tr_ms/h_tr_is for low-mass
- Implement construction-specific calculation
- Achieve ≥25% pass rate for 600-series (1-2/6 cases passing)
- Time constant τ ≈ 1-2 hours (correct for low-mass)

### Medium Case: Partial Improvement
- Improve some cases but not all
- May need case-specific conductance adjustments
- Achieve ≥10% pass rate (improvement from 0%)

### Worst Case: Fundamental Differences
- Current conductance calculation is correct for our model
- Reference tools use fundamentally different assumptions
- Accept discrepancies as legitimate differences

## Technical Background

### 5R1C Thermal Network

The ISO 13790 5R1C network consists of:
- **R1**: h_tr_w - Windows (exterior → interior)
- **R2**: h_ve - Ventilation (exterior → interior)
- **R3**: h_tr_is - Interior surface (surface → interior)
- **R4**: h_tr_ms - Mass surface (mass → surface)
- **R5**: h_tr_em - Exterior mass (exterior → mass)

**Time Constant**: τ = C / U_total
- C = Total thermal capacitance (J/K)
- U_total = Total conductance (W/K)
- τ should be 1-2 hours for low-mass, 4-5 hours for high-mass

### Conductance Calculation

Current implementation (from Session 44 diagnostics):
```
h_tr_em: 87.36 W/K (exterior->mass)
h_tr_ms: 1092.00 W/K (mass->surface)  ← TOO HIGH?
h_tr_is: 550.62 W/K (surface->interior)
h_tr_w:  36.00 W/K (windows)
h_ve:    21.71 W/K (ventilation)
```

**Total Conductance**:
```
U_total = h_tr_w + h_ve + (h_tr_is * h_tr_em) / (h_tr_is + h_tr_em)
U_total = 36 + 21.71 + (550.62 * 87.36) / (550.62 + 87.36)
U_total = 57.71 + 75.4 = 133.11 W/K
```

**Time Constant**:
```
τ = C / U_total = 2.40e6 J/K / 133.11 W/K = 18033 seconds = 5.0 hours
```

**To achieve τ = 1.5 hours**:
```
U_total = C / τ = 2.40e6 J/K / (1.5 * 3600 s) = 444 W/K
```

This requires reducing `h_tr_ms` from 1092 W/K to ~300 W/K (factor of 3.6 reduction).

## Files to Examine

1. **`src/sim/engine.rs`**:
   - Lines 1100-1300: Conductance calculation from spec
   - Lines 1200-1250: h_tr_ms and h_tr_is calculation
   - Lines 1300-1350: Mode-specific coupling factor application

2. **`src/validation/ashrae_140_cases.rs`**:
   - Lines 1900+: Case 600 specifications
   - Construction type definitions
   - Material properties

3. **`src/sim/construction.rs`**:
   - Material thermal properties
   - Assembly calculations
   - Conductance formulas

## Diagnostic Commands

```bash
# Examine conductance calculation
grep -n "h_tr_ms\|h_tr_is" src/sim/engine.rs | head -20

# Check construction type handling
grep -n "low_mass\|high_mass" src/sim/engine.rs | head -20

# Run 600-series diagnostic
cargo run --release --bin diagnose_600_series

# Test Case 600
cargo run --release --bin fluxion validate --case 600

# Build for testing
cargo build --release
```

## Success Criteria

- [ ] Root cause of incorrect time constant identified
- [ ] Correct conductance calculation implemented for low-mass
- [ ] Time constant τ ≈ 1-2 hours for low-mass (verified)
- [ ] At least 1-2 600-series cases passing (≥25% pass rate)
- [ ] Heating loads reduced toward reference range
- [ ] Cooling loads increased toward reference range
- [ ] Changes documented in SESSION_45_SUMMARY.md
- [ ] physics_based_refactor.md updated with results

## References

- **SESSION_44_SUMMARY.md**: Root cause identification and test results
- **ISO 13790**: 5R1C thermal network standard
- **ASHRAE 140**: Case 600 specifications and construction details
- **Session 40**: Original implementation of mode-specific factors
- **Session 33**: Physics-based baseline establishment

---

**Session 45 Goal**: Fix the 5R1C conductance calculation for low-mass buildings to produce correct time constants (τ ≈ 1-2 hours) and improve 600-series validation pass rate from 0% to ≥25%.
