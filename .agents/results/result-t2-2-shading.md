# T2.2: Fix Shading Overlap for Cases 630/930

**Status**: FIXED
**Issue**: #747
**Date**: 2026-05-16

## Summary

Fixed double-counting of shadow overlap when both overhang and fin shading devices cast shadows on the same window (ASHRAE 140 Cases 630/930). The previous code simply summed overhang shadow area and fin shadow area without subtracting their intersection.

## Root Cause

In `src/sim/shading.rs`, `calculate_shaded_fraction()` computed:

```
shaded_area = overhang_shadow + fin_shadow
```

The overhang shadow is a horizontal strip across the **full width** of the window (`shaded_height × window.width`), while the fin shadow is a vertical strip across the **full height** (`window.height × shaded_width`). When both are present, the corner region where they overlap (`shaded_height × shaded_width`) was counted twice.

This is a textbook inclusion-exclusion error.

## Fix Applied

**File**: `src/sim/shading.rs`

Refactored the calculation to track individual shadow dimensions (height/width) instead of raw areas, then apply inclusion-exclusion:

```
shaded_area = (overhang_shaded_height × window.width)
            + Σ(fin_shaded_width × window.height)
            - Σ(overhang_shaded_height × fin_shaded_width)
```

The last term subtracts the overlap for each fin that casts a shadow simultaneously with the overhang.

### Changes:
1. Replaced `calculate_overhang_shadow_area()` → `calculate_overhang_shaded_height()` (returns dimension, not area)
2. Replaced `calculate_fin_shadow_area()` → `calculate_fin_shaded_width()` (returns dimension, not area)
3. Updated `calculate_shaded_fraction()` to apply overlap correction
4. Added 3 regression tests:
   - `test_overhang_fin_overlap_no_double_counting` — verifies overlap subtraction with 45°/45° sun
   - `test_overhang_fin_no_overlap_different_azimuth` — verifies no false correction when fin doesn't shade
   - `test_case_630_shading_overlap_correction` — Case 630 geometry with overhang at 2.7m height

## Test Results

```
cargo test --lib sim::shading
6 passed, 2433 filtered out (1 suite, 0.00s)
```

All existing tests continue to pass (they use overhang-only or fin-only configurations where no overlap exists).

## Impact Analysis

- **Cases 610/910** (overhang only): No change — no fins present, overlap term is zero
- **Cases 630/930** (overhang + fins): Reduced shading → less solar blocking → slightly higher cooling loads, lower heating loads
- **Cases 600/900** (no shading): No change
- No other shading accumulation patterns found in the codebase

## Files Changed

| File | Change |
|------|--------|
| `src/sim/shading.rs` | Refactored `calculate_shaded_fraction()` with inclusion-exclusion; renamed internal helpers; added 3 regression tests |

## Acceptance Criteria Checklist

- [x] Overhang + fin shadow overlap corrected for Cases 630/930
- [x] No double-counting of shaded areas (inclusion-exclusion applied)
- [x] Regression tests added
- [x] Existing tests pass
- [x] No impact on non-shading cases (600/900)
- [x] No impact on overhang-only cases (610/910)
