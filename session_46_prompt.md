# Session 46: Address Case 920 Borderline Results

**Date**: 2026-03-27
**Follows**: Session 45 (Accept 600-series as legitimate model differences - ✅ COMPLETE)
**Status**: 📋 PLANNED
**Priority**: MEDIUM - Case 920 is 30% below minimum, one of two remaining 900-series failures

## Objective

Investigate and fix Case 920 which is currently 30% below the minimum reference range for annual cooling energy.

## Current State

### Case 920 Results

**Current Validation**:
- **Annual Heating**: 3.20 MWh (Ref: 3.26-4.30 MWh) - ✅ Just within range (2% below min)
- **Annual Cooling**: 1.29 MWh (Ref: 1.84-3.31 MWh) - ❌ 30% below minimum
- **Peak Heating**: 1.93 kW (Ref: 2.10-2.80 kW) - ❌ 8% below minimum
- **Peak Cooling**: 1.22 kW (Ref: 1.40-1.90 kW) - ❌ 13% below minimum

**Status**: Cooling energy significantly below reference - needs investigation.

### Case 920 Specifications

From `src/validation/ashrae_140_cases.rs`:
- **Construction**: Low-mass with east/west windows (6m² each)
- **Windows**: Double clear glass, east/west orientation
- **Shading**: None (unlike Case 930 which has overhang + fins)
- **Floor Area**: 48 m² (8m × 6m × 2.7m high)
- **Internal Loads**: 200W (60% radiative, 40% convective)
- **HVAC Setpoints**: 20°C heating, 27°C cooling
- **Infiltration**: 0.5 ACH

### Comparison with Similar Cases

| Case | Windows | Shading | Cooling (MWh) | Ref Range | Status |
|------|---------|---------|---------------|-----------|--------|
| 920 | E/W 6m² each | None | 1.29 | 1.84-3.31 | ❌ -30% below min |
| 930 | E/W 6m² each | Overhang + fins | 1.09 | 1.04-2.24 | ✅ PASS |

**Key Insight**: Case 930 (with shading) passes validation, but Case 920 (without shading) fails by 30%. This is unexpected - shading should REDUCE cooling, not increase it.

## Investigation Plan

### Priority 1: Compare Case 920 vs Case 930

**Step 1: Analyze Energy Balance**
- Compare hourly cooling loads between Case 920 and 930
- Check if shading is having the opposite effect (increasing cooling instead of decreasing)
- Verify solar gain calculations for E/W windows

**Step 2: Check Session 42 Fix**
- Session 42 fixed Case 930 with reduced cooling coupling (h_tr_em_cooling_factor = 0.5)
- Verify this fix isn't negatively affecting Case 920
- Test if Case 920 needs different coupling factors

### Priority 2: Test Potential Solutions

**Solution A: Adjust Cooling Coupling for Case 920**
- Current: h_tr_em_cooling_factor = 1.2 (from Session 40, line 1132)
- Test: Increase to 1.4 or 1.6 to increase cooling loads
- Rationale: More coupling to exterior mass = more heat rejection = higher cooling

**Solution B: Check Solar Gains**
- Verify E/W window solar gain calculations
- Check if beam/diffuse split is correct for E/W orientation
- Test if solar gains are too low (causing low cooling demand)

**Solution C: Review Internal Load Distribution**
- Check if 200W internal loads are being applied correctly
- Verify radiative/convective split (60%/40%)
- Test different distributions

### Priority 3: Compare with Reference Tools

If the above doesn't reveal the issue:
- Research how EnergyPlus models Case 920
- Check if there are case-specific corrections in reference tools
- Consider if this is a legitimate difference

## Expected Outcomes

### Best Case: Simple Parameter Adjustment
- Identify incorrect parameter value
- Adjust and achieve ≥90% pass rate for Case 920
- No regressions on other cases

### Medium Case: Partial Improvement
- Improve Case 920 but not to passing level
- May need to accept as borderline case
- Document as known limitation

### Worst Case: Fundamental Difference
- Current implementation is correct per ASHRAE 140
- Reference tools use different assumptions
- Accept as legitimate model difference

## Success Criteria

- [ ] Root cause of Case 920 low cooling identified
- [ ] Case 920 cooling within reference range (≥1.84 MWh)
- [ ] No regressions on other 900-series cases
- [ ] Changes documented in SESSION_46_SUMMARY.md
- [ ] physics_based_refactor.md updated with results

## Files to Examine

1. **`src/sim/engine.rs`**:
   - Lines 1128-1145: Mode-specific coupling factors (Case 920 = 0.8, 1.2)
   - Lines 1300-1400: Solar gain distribution
   - Lines 3600-3800: Cooling load calculation

2. **`src/validation/ashrae_140_cases.rs`**:
   - Lines 1940-1954: Case 920 specifications
   - E/W window construction details

3. **Session Documents**:
   - `SESSION_42_SUMMARY.md`: Case 930 fix (reduced cooling coupling)
   - `SESSION_45_SUMMARY.md`: 600-series investigation context

## Diagnostic Commands

```bash
# Run Case 920 validation
cargo run --release --bin fluxion validate --case 920

# Compare with Case 930
cargo run --release --bin fluxion validate --case 930

# Run both cases
cargo run --release --bin fluxion validate --case 920 --case 930

# Build for testing
cargo build --release
```

## Additional Context

### Session 42 Background

Case 930 had a 3.5x cooling discrepancy that was fixed by reducing the cooling coupling factor:
```rust
// Session 42 fix for Case 930 (with shading)
"930" => (0.8, 0.5),  // Reduced cooling coupling to prevent excessive heat loss to mass
```

**Question**: Should Case 920 (without shading) also use reduced cooling coupling, or is the current value (1.2) correct?

### Key Question

Why does Case 930 (with shading) have HIGHER cooling coupling efficiency (0.5 factor) than Case 920 (without shading, 1.2 factor)?

**Expected**: Shading reduces solar gains → less cooling needed
**Actual**: Case 930 passes, Case 920 fails (30% below minimum)

This suggests the coupling factors may need adjustment for Case 920.

## References

- **SESSION_42_SUMMARY.md**: Case 930 fix and shading analysis
- **SESSION_45_SUMMARY.md**: 600-series investigation (completed)
- **ASHRAE 140 Standard**: Case 920 and 930 specifications
- **ISO 13790**: 5R1C thermal network standard

---

**Session 46 Goal**: Investigate and fix Case 920 cooling underprediction (30% below minimum), achieving ≥90% pass rate for 900-series cases.
