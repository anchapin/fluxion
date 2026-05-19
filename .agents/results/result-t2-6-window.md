# T2.6: Verify Window Properties for High-Mass (900-series)

**Status**: FIXED
**Issue**: #699

## Summary

Verified window properties (U-value, SHGC, glass type) for all 900-series ASHRAE 140 cases. Found and fixed one bug: **Case 900FF was using `single_clear_glass()` (U=5.8, SHGC=0.86) instead of `double_clear_glass()` (U=2.10, SHGC=0.77)**.

## Findings

### Window Properties Found

All 600/900-series cases use `WindowSpec::double_clear_glass()`:
- **U-value**: 2.10 W/(m^2 K)
- **SHGC**: 0.77
- **Normal transmittance**: 0.703
- **Glass type**: DoubleClear

### Bug: Case 900FF Used Wrong Glass Type

| Case | Before Fix | After Fix |
|------|-----------|-----------|
| 900FF | `single_clear_glass()` (U=5.8, SHGC=0.86) | `double_clear_glass()` (U=2.10, SHGC=0.77) |
| All other 900-series | `double_clear_glass()` | No change |
| All 600-series | `double_clear_glass()` | No change |

**Root cause**: Case 900FF was the only case in the 600/900 series that used `single_clear_glass()`. Per ASHRAE 140, all cases within a series (600 or 900) must use the same window type. The free-floating variants differ only in HVAC control (no thermostat), not window construction.

### Secondary Finding: U-value Discrepancy

The codebase has a discrepancy between authoritative constants and the active `WindowSpec`:

| Source | U-value | SHGC |
|--------|---------|------|
| `materials.rs` constants (WINDOW_U_VALUE) | 3.0 W/(m^2 K) | 0.787 |
| `WindowSpec::double_clear_glass()` (active) | 2.10 W/(m^2 K) | 0.77 |
| `WindowProperties::double_clear()` (solar calc) | N/A | 0.787 |

The `materials.rs` constants are defined but **never used** — the actual values come from `WindowSpec::double_clear_glass()`. The `WindowSpec` values (U=2.10, SHGC=0.77) are documented as "from official ASHRAE 140 Table 6.3.1 / BESTEST window dataset". This discrepancy should be investigated in a separate task but was NOT changed in this fix to avoid regressions.

### Data Flow Verified

1. `CaseBuilder::case_900_baseline()` creates `CaseSpec` with `WindowSpec::double_clear_glass()`
2. `thermal_model_core.rs` line 1594-1598: `WindowSpec.shgc` and `WindowSpec.normal_transmittance` are used to create `WindowProperties` for solar calculations
3. `thermal_model_core.rs` line 444: `WindowSpec.u_value` is used for `model.window_u_value`
4. `thermal_model_core.rs` line 801: `spec.window_properties.u_value * zone_window_area` for conductance

## Files Changed

1. **`src/validation/ashrae_140_cases.rs`**:
   - Line ~2458: Changed Case 900FF from `single_clear_glass()` to `double_clear_glass()`
   - Line ~1697: Fixed stale comment (U=3.0 -> U=2.10, h_tr_w=36.0 -> 25.2)
   - Added 3 regression tests at end of file

## Regression Tests Added

1. `test_900_series_window_properties_consistency` — Verifies all 900-series cases use identical double_clear_glass()
2. `test_600_series_window_properties_consistency` — Verifies all 600-series cases use identical double_clear_glass()
3. `test_window_spec_matches_ashrae_140_constants` — Verifies double_clear_glass() properties are in valid ranges

## Test Results

- All 3 new tests: PASS
- All 23 ashrae_140_cases tests: PASS
- Compilation check: PASS

## Acceptance Criteria Checklist

- [x] Window U-value verified for 900-series cases (all use 2.10 W/(m^2 K))
- [x] Window SHGC verified for 900-series cases (all use 0.77)
- [x] Case 900FF bug fixed (was using wrong glass type)
- [x] Stale comment corrected in ConductanceReferences
- [x] Regression tests added to prevent future regressions
- [x] All existing tests still pass
