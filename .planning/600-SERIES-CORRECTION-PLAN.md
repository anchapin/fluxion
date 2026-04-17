# Plan: Standardize 600-Series Cases to Use from_spec() with Physics-Based Corrections

## Context

Session 77 identified that Case 600 had duplicate code paths:
1. **Case600Model** (dedicated file) - used per-surface film coefficients
2. **from_spec()** path - used uniform 8.29 W/m²K (wrong for 48m² zone)

**Actions taken:**
- Fixed `from_spec()` h_tr_is to use per-surface film coefficients (commit da996a5)
- Removed Case600Model and its tests (commit bcf9a59)
- Changed correction factor from 2.42 to 1.0 for Case 600 (commit d320402)

Now applying same standardization to remaining 600-series cases.

## Current State Analysis

### Correction Factors (engine.rs:1066-1070)

| Case | heating_corr | cooling_corr | Notes |
|------|--------------|--------------|-------|
| 600 | 1.0 | 1.0 | ✅ Already fixed |
| 600FF | 1.0 | 1.0 | ✅ Already fixed |
| 610 | 1.7 | 1.0 | May be over-correcting |
| 620 | 2.69 | 0.64 | May be over-correcting |
| 630 | 1.7 | 1.0 | May be over-correcting |
| 640 | 1.7 | 1.0 | May be over-correcting |
| 650 | 1.7 | 1.0 | May be over-correcting |
| 650FF | 1.7 | 1.0 | May be over-correcting |

### Reference Ranges (benchmark.rs)

| Case | Heat Ref (MWh) | Cool Ref (MWh) | Type |
|------|----------------|----------------|------|
| 600 | 5.50-7.50 | 8.00-10.50 | Low-mass |
| 610 | 4.36-5.79 | 3.92-6.14 | Low-mass |
| 620 | 4.50-6.50 | 3.20-5.00 | Low-mass |
| 630 | 5.05-6.47 | 2.13-3.70 | Low-mass |
| 640 | 2.75-3.80 | 5.95-8.10 | Low-mass |
| 650 | 0.00 | 4.82-7.06 | Low-mass (heat=0) |
| 650FF | N/A | N/A | Free-float |

## Root Cause

All 600-series cases are **low-mass** buildings (τ ~ 2 hours). The correction factors were calibrated when:
1. Physics was broken (wrong h_tr_is)
2. Other bugs caused energy prediction errors

Now that physics is fixed, correction factors > 1.0 are likely **over-correcting**.

## Proposed Actions

### Phase 1: Set All Low-Mass 600-Series Corrections to 1.0

**Rationale:** Low-mass buildings (τ ~ 2h) don't have the thermal damping issues that high-mass buildings (τ > 100h) have. If raw energy is under-predicting, dividing by >1 makes it worse.

```rust
// Change from:
"610" => (1.7, 1.0),
"620" => (2.69, 0.64),
"630" | "640" | "650" | "650FF" => (1.7, 1.0),

// Change to:
"610" | "620" | "630" | "640" | "650" | "650FF" => (1.0, 1.0),
```

### Phase 2: Verify No Dedicated Model Files Exist

Check for any remaining dedicated model implementations:
- Case610Model, Case620Model, etc.
- Similar to the removed case_600.rs

```bash
# Search for any dedicated 6xx model files
rg "Case6[0-9]{2}Model" --type rust
```

### Phase 3: Run Validation to Verify Results

Run the ASHRAE 140 validator and check if results are within reference ranges.

```bash
cargo test ashrae_140_validation --release
```

Expected outcome:
- Raw energy should be closer to reference ranges
- May need further tuning if still off

### Phase 4: Investigate High-Mass 900-Series (Future)

The 900-series cases have τ > 100 hours and genuinely need correction factors due to thermal mass absorption effects. Don't change those yet - they require separate analysis.

## Success Criteria

1. All 600-series cases (610, 620, 630, 640, 650, 650FF) use (1.0, 1.0) corrections
2. No dedicated model files exist for 600-series (only from_spec)
3. Validation results show raw energy closer to reference ranges
4. 900-series corrections remain unchanged (separate issue)

## Implementation Order

1. Change correction factors in engine.rs
2. Commit with message: "fix(engine): remove correction factors for low-mass 600-series cases"
3. Run validation tests
4. If results improved, keep changes
5. If results degraded, investigate and adjust

## Notes

- Case 650 has heating=0 (night vent cooling only) - verify this is handled correctly
- Case 650FF is free-floating (no HVAC) - should have no heating/cooling energy
- High-mass 900-series (900, 910, 920, 930, 940, 950) have different physics and need separate analysis
