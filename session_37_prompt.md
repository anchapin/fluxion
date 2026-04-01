# Session 37 Prompt: Enable CTF for 900-Series to Fix Physics Limitations

**Date**: 2026-03-27
**Objective**: Enable CTF (Conduction Transfer Function) model for 900-series cases to fix the fundamental 5R1C physics limitations exposed by Session 36's empirical factor removal.

---

## Session 36 Results

**What Was Done**:
1. ✅ Removed empirical 45% summer solar reduction
2. ✅ Reverted solar distribution to physics-based (50% to mass)
3. ✅ Updated coupling factors to physics-based values

**Current Pass Rate**: ~1.6% (very low)

**Key Issues Exposed**:
| Issue | Cases Affected | Root Cause |
|-------|---------------|------------|
| 900-series heating underpredicts | 900, 910, 920, 930 | 5R1C doesn't handle winter solar well |
| 900-series cooling overpredicts | 900, 910, 940 | 5R1C doesn't handle summer solar well |
| 600-series heating overpredicts | 600, 610, 630 | Low-mass thermal dynamics differ |

---

## Root Cause Analysis

The 5R1C (5 Resistance, 1 Capacitance) model has fundamental limitations:
1. **Single thermal mass node** - Can't model multi-layer wall dynamics
2. **Simple surface conductance** - Doesn't capture CTF-based heat transfer
3. **Lumped solar distribution** - Can't differentiate beam vs diffuse, orientation effects

The removed empirical factors were compensating for these model limitations.

---

## Solution: Enable CTF for 900-Series

The CTF (Conduction Transfer Function) model provides:
- **Multi-layer wall dynamics** - Proper thermal mass modeling
- **Accurate surface temperatures** - Time-varying heat transfer
- **Physics-based solar handling** - Better than lumped approximations

### Task 1: Enable CTF for All 900-Series Cases

**Location**: `src/validation/ashrae_140_validator.rs`, function `enable_advanced_solver()`

**Current State**: CTF only enabled for Case 900

**Change Required**: Enable for ALL cases starting with '9'

```rust
// Current (only Case 900):
if spec.case_id == "900" {
    model.enable_ctf(&layers, 3600.0, 50);
}

// New (all 900-series):
if spec.case_id.starts_with('9') {
    model.enable_ctf(&layers, 3600.0, 50);
}
```

### Task 2: Verify CTF Configuration for High-Mass Cases

**Location**: `src/sim/engine.rs`, case setup

**Ensure**:
- Wall layers properly configured for high-mass construction
- Thermal mass parameters correct for concrete/masonry
- CTF timestep matches simulation (3600s)

### Task 3: Test CTF-Only (No 5R1C fallback)

After enabling CTF, verify:
- CTF solver is actually used for 900-series
- No fallback to 5R1C (which has the empirical issues)
- Results improve for both heating and cooling

---

## Alternative: Keep Hybrid Approach

If CTF alone doesn't work, consider hybrid:

1. **900-series**: Use CTF for walls, 5R1C for internal mass
2. **600-series**: Keep 5R1C (works better for low-mass)
3. **Tune each model separately**

---

## Expected Outcome

With CTF enabled for 900-series:
- Solar gains handled more accurately (CTF-based)
- Thermal mass dynamics captured correctly
- Empirical factors no longer needed
- Pass rate should improve from ~2% to ~20-30%

---

## Success Criteria

- [ ] CTF enabled for ALL 900-series cases (900, 910, 920, 930, 940, 950, 960)
- [ ] 900-series heating improved (toward reference range)
- [ ] 900-series cooling improved (toward reference range)
- [ ] Code compiles without errors
- [ ] Target: ≥20% pass rate

---

## Files to Modify

1. `src/validation/ashrae_140_validator.rs`:
   - Lines ~1280-1290: Enable CTF for all 900-series

2. Test and validate results

---

## Important Notes

1. **Don't just enable CTF** - verify it's actually being used
2. **Check for fallback** - ensure 5R1C isn't used as fallback
3. **Monitor performance** - CTF is computationally heavier
4. **Test each case** - don't assume all 900-series behave the same
