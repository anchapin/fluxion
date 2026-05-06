## Problem Description

ASHRAE 140 Table 3 specifies different window types for low-mass vs high-mass cases:
- **600 series (Low-mass)**: Single pane clear glass, 6mm, SHGC ≈ 0.86
- **900 series (High-mass)**: Double pane clear glass, SHGC ≈ 0.76

Currently, **ALL cases (600, 600FF, 900, 900FF) use `WindowSpec::double_clear_glass()` with SHGC=0.789**.

## Investigation Findings

### Code Locations
- `WindowSpec::double_clear_glass()` defined at `src/validation/ashrae_140_cases.rs:78-82`
- Returns SHGC=0.789, U=3.0, transmittance=0.86156
- **No `single_clear_glass()` method exists** - Case 600 incorrectly uses double glass

### Affected Cases
- `case_600()` at line 2148: Uses `double_clear_glass()`
- `case_600ff()` at line 2262: Uses `double_clear_glass()`
- `case_900ff()` at line 2443: Uses `double_clear_glass()` (correct for high-mass)

### Impact on Issue #666

| Case | Current Window | Should Be | Impact |
|------|---------------|-----------|--------|
| 600FF | Double glass (SHGC 0.789) | Single glass (SHGC ~0.86) | Solar gain TOO LOW |
| 900FF | Double glass (SHGC 0.789) | Double glass (SHGC ~0.76) | Solar gain slightly HIGH |

**However**, the free-floating cases show temperatures that are TOO HIGH (137°C vs 46°C reference), which contradicts the finding that 900FF has slightly low SHGC.

## Investigation Still Needed

The temperature discrepancy suggests additional issues beyond just SHGC:
1. Check if `solar_gain` calculation is correct in `src/sim/solar.rs:385`
2. Verify HVAC is truly disabled for FF cases (setpoint override check)
3. The 900FF temperature being too high cannot be explained by SHGC alone

## Proposed Fix

1. Add `WindowSpec::single_clear_glass()` with SHGC ≈ 0.86 for 600 series
2. Adjust 900 series SHGC to match ASHRAE 140-2022 Table 3 value (~0.76)

## Files to Modify
- `src/validation/ashrae_140_cases.rs` - Add single_clear_glass(), update case_600/600FF

## Verification
- Compare temperature results before/after the SHGC correction
- Ensure 600FF max temp doesn't decrease below reference range
- Ensure 900FF behavior is consistent