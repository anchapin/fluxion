# Session 51: FD Integration for All 600-Series Cases

**Date**: 2026-03-27
**Follows**: Session 50 (FD Solver Audit and Initial Enablement)
**Status**: 📋 PLANNED
**Priority**: 🔴 CRITICAL - Complete FD enablement for low-mass buildings
**Estimated Duration**: 1 week
**Prerequisite**: Session 50 successful (FD working for Case 600)

## Objective

Enable FD solver for all 600-series low-mass cases (600, 610, 620, 630, 640, 650) and validate results. This completes the FD implementation for low-mass buildings and establishes whether FD can achieve the target ≥50% pass rate for 600-series.

## Context

### Session 50 Results

**Assuming Session 50 was successful**:
- ✅ FD solver audited and understood
- ✅ FD enabled for Case 600
- ✅ Case 600 running with FD
- ✅ Heating overprediction reduced
- ✅ Cooling underprediction reduced

### Current 600-Series Status (5R1C Model)

| Case | Annual Heating | Ref Range | Error | Annual Cooling | Ref Range | Error |
|------|----------------|-----------|-------|----------------|-----------|-------|
| 600 | 8.65 MWh | 5.50-7.50 | **+54% over** | 6.53 MWh | 8.00-10.50 | **-30% below** |
| 610 | 9.08 MWh | 4.36-5.79 | **+67% over** | 4.56 MWh | 3.92-6.14 | **-0.5% below** |
| 620 | 7.90 MWh | 4.50-6.50 | **+30% over** | 2.29 MWh | 3.20-5.00 | **-39% below** |
| 630 | 9.04 MWh | 5.05-6.47 | **+45% over** | 1.12 MWh | 2.13-3.70 | **-53% below** |
| 640 | 7.12 MWh | 2.75-3.80 | **+87% over** | 5.45 MWh | 5.95-8.10 | **-8% below** |
| 650 | 0.00 MWh | 0.00-0.00 | ✅ | 4.65 MWh | 4.82-7.06 | **-11% below** |

**Key Patterns**:
- Heating: All cases significantly overprediction (+30% to +87%)
- Cooling: Most cases underprediction (-53% to -0.5%)
- Root cause: 5R1C lumped thermal mass not suitable for low-mass

### Why FD Should Help

1. **Nodal Resolution**: Multiple nodes capture thermal gradients
2. **Rapid Transients**: Handles fast temperature changes in low-mass
3. **Small Thermal Mass**: Correctly models low thermal capacitance
4. **Spatial Distribution**: Better heat distribution through wall

## Implementation Plan

### Phase 1: Verify Session 50 Changes (Day 1)

**Step 1: Confirm FD Working for Case 600**

```bash
# Run Case 600 with FD
cargo run --release --bin fluxion validate --case 600

# Verify results are improved
# Compare with 5R1C baseline from Session 50
```

**Step 2: Review Session 50 Findings**

- [ ] Read `SESSION_50_SUMMARY.md`
- [ ] Understand any issues encountered
- [ ] Review fixes applied
- [ ] Check performance metrics
- [ ] Note optimal timestep and nodal resolution

**Step 3: Validate Code Quality**

```bash
# Check for regressions
cargo test --release

# Check code style
cargo fmt --check
cargo clippy --release
```

### Phase 2: Enable FD for All 600-Series Cases (Days 2-3)

**Step 1: Verify FD Enablement Logic**

From Session 50, FD should already be enabled for all low-mass cases:
```rust
// In ThermalModel::from_spec()
if matches!(spec.construction_type, ConstructionType::LowMass) {
    model.fd_enabled = true;
    model.fd_timestep = 300.0; // 5-minute timestep
    // ... FD initialization
}
```

**Verification**:
- [ ] All 600-series cases have `ConstructionType::LowMass`
- [ ] FD is enabled for all of them
- [ ] FD timestep appropriate for each construction
- [ ] Nodal resolution appropriate

**Step 2: Test Each Case Individually**

```bash
# Test each 600-series case
for case in 600 610 620 630 640 650; do
    echo "Testing Case $case with FD..."
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

For each case, compare 5R1C vs FD vs Reference:

**Case 600**:
| Metric | 5R1C | FD | Reference | 5R1C Error | FD Error | Improvement |
|--------|------|-----|-----------|------------|----------|-------------|
| Annual Heating | 8.65 | ? | 5.50-7.50 | +54% over | ? | ? |
| Annual Cooling | 6.53 | ? | 8.00-10.50 | -30% below | ? | ? |
| Peak Heating | 4.43 | ? | 2.80-3.80 | +62% over | ? | ? |
| Peak Cooling | 5.04 | ? | 4.80-6.20 | +5% over | ? | ? |

**Repeat for Cases 610-650**

**Step 2: Analyze Improvements**

For each metric category:
- [ ] Annual heating: How much reduction in overprediction?
- [ ] Annual cooling: How much reduction in underprediction?
- [ ] Peak heating: How much improvement?
- [ ] Peak cooling: How much improvement?
- [ ] Pass rate: How many cases passing now?

**Step 3: Identify Patterns**

- [ ] Which cases benefit most from FD?
- [ ] Which metrics improve most?
- [ ] Are there any regressions?
- [ ] Are results consistent with expectations?

**Step 4: Free-Floating Analysis**

```bash
# Run free-floating cases
cargo run --release --bin fluxion validate --case 600FF
cargo run --release --bin fluxion validate --case 650FF
```

**Expected**:
- Max temperatures should increase (closer to reference)
- FD should capture temperature swings better
- Should be within 10°C of reference (vs 20-30°C with 5R1C)

### Phase 4: Debug and Tuning (Days 5-6)

**If Results Are Good** (≥50% pass rate):

- [ ] Document success
- [ ] Proceed to Session 52
- [ ] Consider FD complete for 600-series

**If Results Are Mixed** (33-50% pass rate):

**Investigation Areas**:

1. **Timestep Issues**:
   - [ ] Is 5-minute timestep appropriate for all cases?
   - [ ] Try 10-minute timestep (600s) for faster execution
   - [ ] Try 15-minute timestep (900s) if stable
   - [ ] Compare results

2. **Nodal Resolution**:
   - [ ] Are 5 nodes appropriate for all constructions?
   - [ ] Try 3 nodes for faster execution
   - [ ] Try 7 nodes for better accuracy
   - [ ] Compare results

3. **Construction-Specific Issues**:
   - [ ] Case 610: North windows (different solar profile)
   - [ ] Case 620: Exterior insulation (different thermal dynamics)
   - [ ] Case 630: E/W windows
   - [ ] Case 640: High window-to-wall ratio
   - [ ] Case 650: Night setback

4. **Boundary Conditions**:
   - [ ] Indoor film coefficient
   - [ ] Outdoor film coefficient
   - [ ] Solar gain distribution
   - [ ] Internal gain distribution

**If Results Are Poor** (<33% pass rate):

- [ ] Review FD implementation for bugs
- [ ] Consider alternative approaches:
   - Use CTF for low-mass instead
   - Hybrid FD/CTF approach
   - Accept 5R1C limitations

### Phase 5: Performance Optimization (Day 6)

**Step 1: Profile FD Solver**

```bash
# Profile Case 600
time cargo run --release --bin fluxion validate --case 600

# Profile all 600-series cases
time for case in 600 610 620 630 640 650; do
    cargo run --release --bin fluxion validate --case $case
done
```

**Targets**:
- Single case: <30 seconds
- All 600-series: <3 minutes total

**Step 2: Optimize If Needed**

If performance is poor:
- [ ] Reduce nodal resolution (3 nodes instead of 5)
- [ ] Increase timestep (10-15 minutes if stable)
- [ ] Optimize solver code
- [ ] Use adaptive timestepping
- [ ] Cache repeated calculations

### Phase 6: Documentation (Day 7)

**Deliverables**:

1. **FD Integration Report** (`SESSION_51_FD_INTEGRATION.md`):
   - All 600-series cases results
   - 5R1C vs FD comparison
   - Pass rate analysis
   - Free-floating temperature analysis
   - Performance metrics

2. **Session Summary** (`SESSION_51_SUMMARY.md`):
   - What was accomplished
   - Pass rate achieved
   - Lessons learned
   - Next steps for Session 52

3. **Update Documentation**:
   - Update `ASHRAE140_ROADMAP.md` with progress
   - Update `physics_based_refactor.md` with FD findings

## Success Criteria

- [ ] FD enabled for all 600-series cases
- [ ] All cases run successfully without errors
- [ ] Energy conservation maintained (<1% imbalance)
- [ ] **Pass rate ≥50%** (3/6 cases or 12/24 metrics)
- [ ] Heating overprediction reduced by ≥50%
- [ ] Cooling underprediction reduced by ≥50%
- [ ] Free-floating temps within 10°C of reference
- [ ] Performance <30s per case

## Go/No-Go Decision for Session 52

**Go (Proceed to Session 52) if**:
- ✅ FD working for all 600-series cases
- ✅ Pass rate ≥33% (at least 2/6 cases)
- ✅ Significant improvement over 5R1C
- ✅ Performance acceptable (<60s per case)

**Conditional Go (Proceed with Optimization) if**:
- ⚠️ Pass rate 17-33% (1-2/6 cases)
- ⚠️ Some cases showing improvement
- ⚠️ Needs tuning but fundamentally sound
- ⚠️ Performance slower than expected
- → Proceed to Session 52 but continue FD optimization

**No-Go (Reconsider Approach) if**:
- ❌ Pass rate <17% (0-1/6 cases)
- ❌ No significant improvement over 5R1C
- ❌ Fundamental issues with FD
- ❌ Performance unacceptable (>2 min per case)

## Expected Outcomes

### Best Case: FD Highly Successful (≥67% pass rate)

- FD solver works excellently for all 600-series
- Most metrics passing validation
- Heating overprediction reduced to <20%
- Cooling underprediction reduced to <20%
- Free-floating temps within 5-10°C of reference
- Performance acceptable (<30s per case)
- **Recommendation**: Use FD for all low-mass cases, deprecate 5R1C

### Medium Case: FD Moderately Successful (33-50% pass rate)

- FD shows significant improvement over 5R1C
- Some cases passing, some still failing
- Heating overprediction reduced 30-50%
- Cooling underprediction reduced 30-50%
- Free-floating temps improved but still low
- Performance acceptable
- **Recommendation**: Use FD for 600-series, continue tuning
- May need sub-hourly timesteps (Session 52) for full improvement

### Worst Case: FD Not Successful (<33% pass rate)

- FD results similar to or worse than 5R1C
- Fundamental issues with implementation
- Performance unacceptable
- **Recommendation**: Consider CTF for low-mass or accept 5R1C limitations

## Detailed Case Analysis

### Case 600 (Baseline)
- **Construction**: Low-mass lightweight, south windows
- **Expected**: Good candidate for FD
- **5R1C Issues**: Heating +54% over, Cooling -30% below
- **FD Expectation**: Both metrics should improve significantly

### Case 610 (North Windows)
- **Construction**: Low-mass, north windows
- **Expected**: Lower solar gains
- **5R1C Issues**: Heating +67% over (worst), Cooling -0.5% below
- **FD Expectation**: Better handling of low solar gains

### Case 620 (Exterior Insulation)
- **Construction**: Low-mass with exterior insulation
- **Expected**: Different thermal dynamics
- **5R1C Issues**: Heating +30% over, Cooling -39% below
- **FD Expectation**: Better insulation modeling

### Case 630 (E/W Windows)
- **Construction**: Low-mass, E/W windows
- **Expected**: Morning/afternoon solar peaks
- **5R1C Issues**: Heating +45% over, Cooling -53% below (worst)
- **FD Expectation**: Better transient handling

### Case 640 (High WWR)
- **Construction**: Low-mass, high window-to-wall ratio
- **Expected**: High solar gains
- **5R1C Issues**: Heating +87% over (worst), Cooling -8% below
- **FD Expectation**: Better solar gain distribution

### Case 650 (Night Setback)
- **Construction**: Low-mass, night setback
- **Expected**: Night temperature reduction
- **5R1C Issues**: Heating 0% (ok), Cooling -11% below
- **FD Expectation**: Better setback modeling

## Files to Modify

1. **`src/sim/engine.rs`**:
   - Lines 1700-1710: FD enablement (verify all low-mass cases)
   - Lines 3200-3250: FD integration in solve loop (if not complete)

2. **`src/physics/fd_solver.rs`**:
   - Verify solver works for all cases
   - Add performance optimizations if needed
   - Adjust nodal resolution if needed

## Commands to Run

```bash
# Verify FD enabled for all 600-series
grep -A 5 "LowMass" src/sim/engine.rs | grep fd_enabled

# Test each case
for case in 600 610 620 630 640 650; do
    echo "=== Case $case ==="
    cargo run --release --bin fluxion validate --case $case
    echo ""
done

# Test free-floating
cargo run --release --bin fluxion validate --case 600FF
cargo run --release --bin fluxion validate --case 650FF

# Compare with 5R1C (if 5R1C results saved)

# Check performance
time cargo run --release --bin fluxion validate --case 600

# Generate comparison report
# (Create script to compare all cases)
```

## Research Questions

1. **Pass Rate**: What pass rate does FD achieve for 600-series?
2. **Heating Overprediction**: How much is heating overprediction reduced?
3. **Cooling Underprediction**: How much is cooling underprediction reduced?
4. **Free-Floating**: Are free-floating temps closer to reference?
5. **Performance**: Is FD performance acceptable?

## Dependencies

- **Session 50**: FD audit and initial enablement complete
- **FD Infrastructure**: Working for Case 600
- **600-Series Cases**: All low-mass constructions

## Next Session

**Session 52**: Sub-Hourly Timesteps for Peak Loads
- Prerequisite: Session 51 successful (FD working for 600-series)
- Scope: Implement 15-minute timesteps for all cases
- Goal: Improve peak load prediction for both 600 and 900 series

## Alternatives if FD Fails

**Option A**: Use CTF for Low-Mass (2-3 weeks)
- Test CTF on 600-series
- May work better than FD
- Simpler (no CFL constraint)

**Option B**: Hybrid FD/CTF Approach
- Use FD for cases with rapid transients
- Use CTF for cases with slow transients
- Auto-select based on thermal characteristics

**Option C**: Accept 5R1C with Empirical Corrections
- Continue using 5R1C
- Add empirical corrections for low-mass
- May never achieve good results

## References

- **`SESSION_50_SUMMARY.md`**: FD audit and initial enablement
- **`docs/ASHRAE140_ROADMAP.md`**: Phase 2 (FD for Low-Mass)
- **`docs/ASHRAE140_QUICKSTART.md`**: Quick start guide
- **`SESSION_45_SUMMARY.md`**: 600-series investigation findings
- **ASHRAE 140 Standard**: 600-series case specifications
- **Finite Difference Methods**: Numerical heat transfer reference

---

**Session 51 Goal**: Enable FD solver for all 600-series low-mass cases and validate results, achieving ≥50% pass rate and demonstrating that FD can address the systematic heating overprediction and cooling underprediction in low-mass buildings.
