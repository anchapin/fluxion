# Session 49: CTF Integration for All 900-Series Cases

**Date**: 2026-03-27
**Follows**: Session 48 (CTF Solver Audit and Initial Enablement)
**Status**: 📋 PLANNED
**Priority**: 🔴 CRITICAL - Complete CTF enablement for high-mass buildings
**Estimated Duration**: 1 week
**Prerequisite**: Session 48 successful (CTF working for Case 900)

## Objective

Enable CTF solver for all 900-series high-mass cases (900, 910, 920, 930, 940, 950) and validate results. This completes the CTF implementation for high-mass buildings and establishes whether CTF can achieve the target ≥67% pass rate for 900-series.

## Context

### Session 48 Results

**Assuming Session 48 was successful**:
- ✅ CTF solver audited and understood
- ✅ CTF enabled for Case 900
- ✅ Case 900 running with CTF
- ✅ Results showing improvement or comparable to 5R1C

### Current 900-Series Status (5R1C Model)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 1.71 MWh (Ref: 1.17-2.04) | 2.28 MWh (Ref: 2.13-3.67) | 1.26 kW (Ref: 1.80-2.40) | 2.35 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 1.93 MWh (Ref: 1.51-2.28) | 1.45 MWh (Ref: 0.82-1.88) | 1.28 kW (Ref: 1.90-2.50) | 1.74 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 3.20 MWh (Ref: 3.26-4.30) | 2.59 MWh (Ref: 1.84-3.31) | 1.93 kW (Ref: 2.10-2.80) | 1.56 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 4.15 MWh (Ref: 4.14-5.34) | 1.09 MWh (Ref: 1.04-2.24) | 2.03 kW (Ref: 2.30-3.00) | 0.95 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 1.13 MWh (Ref: 0.79-1.41) | 2.67 MWh (Ref: 2.08-3.55) | 1.64 kW (Ref: 1.90-2.50) | 3.00 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 0.00 MWh (Ref: 0.00-0.00) | 0.60 MWh (Ref: 0.39-0.92) | 0.00 kW (Ref: 0.00-0.00) | 3.14 kW (Ref: 0.70-0.90) | ❌ FAIL |

**Key Patterns**:
- Annual heating: 5/6 close to or within range
- Annual cooling: 5/6 close to or within range
- Peak heating: All below minimum (8-33%)
- Peak cooling: Mixed (some above, some below)

### Why CTF Should Help

1. **Distributed Thermal Mass**: CTF captures heat distribution through wall layers
2. **Transient Response**: Better handling of thermal lag effects
3. **Peak Loads**: Should improve peak load prediction (less averaging)
4. **EnergyPlus Methodology**: Same approach as reference tool

## Implementation Plan

### Phase 1: Verify Session 48 Changes (Day 1)

**Step 1: Confirm CTF Working for Case 900**

```bash
# Run Case 900 with CTF
cargo run --release --bin fluxion validate --case 900

# Verify results are improved
# Compare with 5R1C baseline from Session 48
```

**Step 2: Review Session 48 Findings**

- [ ] Read `SESSION_48_SUMMARY.md`
- [ ] Understand any issues encountered
- [ ] Review fixes applied
- [ ] Check performance metrics

**Step 3: Validate Code Quality**

```bash
# Check for regressions
cargo test --release

# Check code style
cargo fmt --check
cargo clippy --release
```

### Phase 2: Enable CTF for All 900-Series Cases (Days 2-3)

**Step 1: Verify CTF Enablement Logic**

From Session 48, CTF should already be enabled for all high-mass cases:
```rust
// In ThermalModel::from_spec()
if matches!(spec.construction_type, ConstructionType::HighMass) {
    model.ctf_enabled = true;
    // ... CTF initialization
}
```

**Verification**:
- [ ] All 900-series cases have `ConstructionType::HighMass`
- [ ] CTF is enabled for all of them
- [ ] CTF coefficients calculated correctly for each construction

**Step 2: Test Each Case Individually**

```bash
# Test each 900-series case
for case in 900 910 920 930 940 950; do
    echo "Testing Case $case with CTF..."
    cargo run --release --bin fluxion validate --case $case
    echo "---"
done
```

**Expected**:
- All cases run without errors
- Energy conservation maintained
- Results differ from 5R1C

### Phase 3: Validation and Comparison (Days 4-5)

**Step 1: Create Comparison Table**

For each case, compare 5R1C vs CTF vs Reference:

**Case 900**:
| Metric | 5R1C | CTF | Reference | 5R1C Status | CTF Status |
|--------|------|-----|-----------|-------------|------------|
| Annual Heating | 1.71 | ? | 1.17-2.04 | ✅ | ? |
| Annual Cooling | 2.28 | ? | 2.13-3.67 | ✅ | ? |
| Peak Heating | 1.26 | ? | 1.80-2.40 | ❌ 30% below | ? |
| Peak Cooling | 2.35 | ? | 1.60-2.10 | ❌ 12% above | ? |

**Repeat for Cases 910-950**

**Step 2: Analyze Improvements**

For each metric category:
- [ ] Annual heating: How many cases pass with CTF vs 5R1C?
- [ ] Annual cooling: How many cases pass with CTF vs 5R1C?
- [ ] Peak heating: How much improvement?
- [ ] Peak cooling: How much improvement?

**Step 3: Identify Patterns**

- [ ] Which cases benefit most from CTF?
- [ ] Which metrics improve most?
- [ ] Are there any regressions?
- [ ] Are results consistent with expectations?

### Phase 4: Debug and Tuning (Days 5-6)

**If Results Are Good** (≥67% pass rate):

- [ ] Document success
- [ ] Proceed to Session 50
- [ ] Consider CTF complete for 900-series

**If Results Are Mixed** (40-67% pass rate):

**Investigation Areas**:

1. **CTF Coefficients**:
   - [ ] Are coefficients correct for each construction?
   - [ ] Check Cases 910, 920, 930 (different window orientations)
   - [ ] Check Cases 940, 950 (different configurations)

2. **Timestep Issues**:
   - [ ] Is 1-hour timestep sufficient?
   - [ ] Try 30-minute timestep (1800s)
   - [ ] Compare results

3. **Integration Issues**:
   - [ ] Is CTF properly integrated with HVAC control?
   - [ ] Check for coupling with thermal mass
   - [ ] Verify boundary conditions

4. **Construction-Specific Issues**:
   - [ ] Case 910: North windows (different solar profile)
   - [ ] Case 920: E/W windows (no shading)
   - [ ] Case 930: E/W windows (with shading)
   - [ ] Case 940: Large south windows
   - [ ] Case 950: Night ventilation

**If Results Are Poor** (<40% pass rate):

- [ ] Review CTF implementation for bugs
- [ ] Compare with EnergyPlus CTF methodology
- [ ] Consider alternative approaches
- [ ] May need to revisit Session 48

### Phase 5: Performance Optimization (Day 6)

**Step 1: Profile CTF Solver**

```bash
# Profile Case 900
time cargo run --release --bin fluxion validate --case 900

# Profile all 900-series cases
time for case in 900 910 920 930 940 950; do
    cargo run --release --bin fluxion validate --case $case
done
```

**Targets**:
- Single case: <5 seconds
- All 900-series: <30 seconds total

**Step 2: Optimize If Needed**

If performance is poor:
- [ ] Profile hot spots
- [ ] Optimize coefficient calculation
- [ ] Cache repeated calculations
- [ ] Consider selective CTF enablement

### Phase 6: Documentation (Day 7)

**Deliverables**:

1. **CTF Integration Report** (`SESSION_49_CTF_INTEGRATION.md`):
   - All 900-series cases results
   - 5R1C vs CTF comparison
   - Pass rate analysis
   - Performance metrics

2. **Session Summary** (`SESSION_49_SUMMARY.md`):
   - What was accomplished
   - Pass rate achieved
   - Lessons learned
   - Next steps for Session 50

3. **Update Documentation**:
   - Update `ASHRAE140_ROADMAP.md` with progress
   - Update `physics_based_refactor.md` with CTF findings

## Success Criteria

- [ ] CTF enabled for all 900-series cases
- [ ] All cases run successfully without errors
- [ ] Energy conservation maintained (<1% imbalance)
- [ ] **Pass rate ≥67%** (4/6 cases or 16/24 metrics)
- [ ] Performance <5s per case
- [ ] Results documented and analyzed

## Go/No-Go Decision for Session 50

**Go (Proceed to Session 50) if**:
- ✅ CTF working for all 900-series cases
- ✅ Pass rate ≥50% (at least 3/6 cases)
- ✅ No fundamental issues with CTF
- ✅ Performance acceptable

**Conditional Go (Proceed with Caution) if**:
- ⚠️ Pass rate 33-50% (2-3/6 cases)
- ⚠️ Some cases showing improvement
- ⚠️ Needs tuning but fundamentally sound
- → Proceed to Session 50 but continue CTF development in parallel

**No-Go (Debug CTF Further) if**:
- ❌ Pass rate <33% (0-1/6 cases)
- ❌ Fundamental issues with CTF
- ❌ Energy conservation violated
- ❌ Cannot proceed without fixes

## Expected Outcomes

### Best Case: CTF Highly Successful (≥90% pass rate)

- CTF solver works excellently for all 900-series
- Most metrics passing validation
- Peak loads significantly improved
- Clear demonstration that CTF is superior to 5R1C
- **Recommendation**: Use CTF for all high-mass cases, deprecate 5R1C

### Medium Case: CTF Moderately Successful (50-67% pass rate)

- CTF shows improvement over 5R1C
- Some cases passing, some still failing
- Peak loads improved but not perfect
- **Recommendation**: Use CTF for 900-series, continue tuning
- May need sub-hourly timesteps (Session 52) for full improvement

### Worst Case: CTF Not Successful (<33% pass rate)

- CTF results similar to or worse than 5R1C
- Fundamental issues with implementation
- **Recommendation**: Debug CTF or consider FD solver instead
- May need 2-3 weeks to fix CTF before proceeding

## Detailed Case Analysis

### Case 900 (Baseline)
- **Construction**: High-mass concrete, south windows
- **Expected**: Good candidate for CTF
- **5R1C Issues**: Peak loads below reference
- **CTF Expectation**: Peak loads should improve

### Case 910 (North Windows)
- **Construction**: High-mass, north windows
- **Expected**: Lower solar gains
- **5R1C Issues**: Peak cooling above reference
- **CTF Expectation**: Better thermal mass handling

### Case 920 (E/W Windows, No Shading)
- **Construction**: High-mass, E/W windows
- **Expected**: High morning/afternoon solar
- **5R1C Issues**: Annual heating below minimum
- **CTF Expectation**: Better thermal lag handling

### Case 930 (E/W Windows, With Shading)
- **Construction**: High-mass, E/W windows, shading
- **Expected**: Reduced solar gains
- **5R1C Issues**: Peak loads below reference
- **CTF Expectation**: Similar to Case 920 but with shading effects

### Case 940 (Large South Windows)
- **Construction**: High-mass, large south windows
- **Expected**: High solar gains
- **5R1C Issues**: Peak cooling well above reference
- **CTF Expectation**: Better solar gain distribution

### Case 950 (Night Ventilation)
- **Construction**: High-mass, night ventilation
- **Expected**: Thermal mass discharge at night
- **5R1C Issues**: Peak cooling very high above reference
- **CTF Expectation**: Better night ventilation effects

## Files to Modify

1. **`src/sim/engine.rs`**:
   - Lines 1700-1710: CTF enablement (verify all high-mass cases)
   - Lines 3200-3250: CTF integration in solve loop (if not complete)

2. **`src/physics/ctf_coefficients.rs`**:
   - Verify coefficient calculation for all constructions
   - Add debug output if needed

3. **`src/physics/ctf_solver.rs`**:
   - Verify solver works for all cases
   - Add performance optimizations if needed

## Commands to Run

```bash
# Verify CTF enabled for all 900-series
grep -A 5 "HighMass" src/sim/engine.rs | grep ctf_enabled

# Test each case
for case in 900 910 920 930 940 950; do
    echo "=== Case $case ==="
    cargo run --release --bin fluxion validate --case $case
    echo ""
done

# Compare with 5R1C (if 5R1C results saved)
# diff 5r1c_results.txt ctf_results.txt

# Check energy conservation
# (Add debug output to solver if needed)

# Profile performance
time cargo run --release --bin fluxion validate --case 900

# Generate comparison report
# (Create script to compare all cases)
```

## Research Questions

1. **Pass Rate**: What pass rate does CTF achieve for 900-series?
2. **Peak Loads**: Does CTF improve peak load prediction?
3. **Annual Energies**: Are annual energies improved or maintained?
4. **Case Variability**: Which cases benefit most from CTF?
5. **Performance**: Is CTF performance acceptable?

## Dependencies

- **Session 48**: CTF audit and initial enablement complete
- **CTF Infrastructure**: Working for Case 900
- **900-Series Cases**: All high-mass constructions

## Next Session

**Session 50**: FD Solver Audit and Initial Enablement
- Prerequisite: Session 49 successful (CTF working for 900-series)
- Scope: Audit FD solver and enable for Case 600
- Goal: Test FD on low-mass building

## Alternatives if CTF Fails

**Option A**: Debug and Fix CTF (2-3 weeks)
- Identify root cause of failures
- Fix implementation issues
- Re-test all 900-series cases

**Option B**: Use FD Solver for All Cases
- FD may work better for both low and high mass
- Test FD on 900-series (instead of CTF)
- May need more development

**Option C**: Accept 5R1C with Improvements
- Continue using 5R1C with empirical corrections
- Focus on sub-hourly timesteps
- May not achieve 90% target

## References

- **`SESSION_48_SUMMARY.md`**: CTF audit and initial enablement
- **`docs/ASHRAE140_ROADMAP.md`**: Phase 1 (CTF for High-Mass)
- **`docs/ASHRAE140_QUICKSTART.md`**: Quick start guide
- **ASHRAE 140 Standard**: 900-series case specifications
- **ISO 13790**: CTF model specification
- **EnergyPlus Documentation**: CTF implementation reference

---

**Session 49 Goal**: Enable CTF solver for all 900-series high-mass cases and validate results, achieving ≥67% pass rate and demonstrating that CTF is superior to 5R1C for high-mass buildings.
