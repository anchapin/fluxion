# Result: Final Case 900 Validation and MultiNodeHvacRunner Deprecation

## Status: COMPLETE

## Summary
Added `#[deprecated]` attribute to `MultiNodeHvacRunner` struct and documented final Case 900 validation status.

## Changes

### 1. `src/sim/multi_node_hvac_runner.rs` (modified)
- Added `#[deprecated(since = "0.9.0", note = "...")]` attribute to `MultiNodeHvacRunner` struct
- Deprecation reason: "Use multi-node thermal model with inline HVAC control instead"

### 2. `.planning/debug/875-final-validation.md` (created)
- Documented Case 900 test pass/fail status
- Identified blockers: Issue #876 (Crank-Nicolson) for annual heating and min temp

## Case 900 Test Results

| Metric | Status | Current | Reference | Blocker |
|--------|--------|---------|-----------|---------|
| Annual Heating | FAIL | 4.25 MWh | [1.17, 2.04] MWh | #876 Crank-Nicolson |
| 900FF Min Temp | FAIL | -0.50°C | [-6.40, -1.60]°C | #876 Crank-Nicolson |
| Temp Swing Reduction | FAIL | 12.7% | [30, 55]% | #876 Crank-Nicolson |
| Solar Beam Fraction | FAIL | 0.6 → 46.49°C | ≤46.4°C max | #700 calibration |

**Passing tests: 12** (including annual cooling, peak heating/cooling, max temperatures)

## Acceptance Criteria Checklist

- [x] MultiNodeHvacRunner marked as deprecated
- [x] All possible Case 900 metrics validated
- [x] Documented summary of pass/fail status
- [x] `cargo build --lib` succeeds (with expected deprecation warnings)
- [x] `cargo test -- case_900` runs (12 passed, 4 failed - expected)

## Next Steps

1. **Issue #876**: Implement Crank-Nicolson to fix annual heating over-prediction
2. **Issue #700**: Calibrate `solar_beam_to_mass_fraction` parameter (0.8 brings max temp in range)
3. **Migration**: Update `tests/case_900ff_multinode_validation.rs` to use inline HVAC

## Commit

```
feat!: deprecate MultiNodeHvacRunner in favor of inline HVAC (Issue #875)
```
