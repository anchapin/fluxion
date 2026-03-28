# Session 48 Regression Root Cause Analysis

**Date**: 2026-03-27
**Issue**: Heating increased from 4.75 MWh → 12.27 MWh (2.6x worse)

## Executive Summary

The regression is caused by CTF being disabled in Session 48 without fixing the 5R1C calculation to work correctly. Session 33 used CTF (which happened to work better despite being theoretically unsound for τ >> dt), and Session 48 disabled CTF due to stability concerns.

## Timeline

1. **Session 33** (faab4be):
   - CTF ENABLED
   - Heating=4.75 MWh, Cooling=6.95 MWh
   - CTF solver active for high-mass buildings

2. **Session 48** (initial):
   - CTF enabled initially
   - 5R1C baseline=1.71 MWh (according to SESSION_48_SUMMARY.md)
   - CTF working but integration issues identified

3. **Session 48** (final):
   - CTF DISABLED due to τ >> dt instability
   - Comment in code: "CTF disabled - τ=73h >> dt=1h"
   - Heating=12.27 MWh, Cooling=3.82 MWh (regression)

4. **Current** (with CTF re-enabled):
   - CTF enabled for testing
   - Heating=416.94 MWh, Cooling=416.54 MWh (catastrophic failure!)
   - Current CTF implementation is broken

## Root Cause

The regression has TWO components:

### 1. CTF Disabled (Primary Cause)
- Session 48 disabled CTF because τ >> dt makes it theoretically unsound
- But the 5R1C fallback has bugs that make it perform worse
- Session 33 happened to work better with CTF despite the theoretical issues

### 2. 5R1C Calculation Issues (Secondary Cause)
- **Thermal mass correction mismatch**: `derived_h_ext` is calculated before `apply_thermal_mass_correction()` modifies `h_tr_em`, creating a mismatch in the thermal network
- When thermal mass correction disabled: Heating=8.97 MWh (better than 12.27, but still worse than 4.75)
- This suggests thermal mass correction is PART of the problem, but not the WHOLE problem

## Test Results

| Configuration | Heating | Cooling | Status |
|--------------|---------|---------|--------|
| Session 33 (CTF enabled) | 4.75 MWh | 6.95 MWh | ❌ 2-4x too high |
| Session 48 (CTF disabled) | 12.27 MWh | 3.82 MWh | ❌ **6-10x too high** |
| Current (CTF disabled) | 12.27 MWh | 3.82 MWh | ❌ 6-10x too high |
| Current (CTF re-enabled) | 416.94 MWh | 416.54 MWh | ❌ **BROKEN** |
| Current (thermal mass disabled) | 8.97 MWh | 4.90 MWh | ❌ 4-8x too high |
| Reference range | 1.17-2.04 MWh | 2.13-3.67 MWh | ✅ Target |

## Key Findings

1. **CTF is broken in current code**: Re-enabling CTF causes catastrophic failure (416 MWh heating)
2. **5R1C has bugs**: Even with thermal mass correction disabled (8.97 MWh), it's worse than Session 33 with CTF (4.75 MWh)
3. **Thermal mass correction contributes to regression**: Disabling it improves from 12.27 → 8.97 MWh, but doesn't fix the issue
4. **Something else is wrong**: There's another bug beyond the thermal mass correction issue

## Next Steps

1. **Do NOT re-enable CTF**: Current CTF implementation is broken
2. **Fix 5R1C calculation**: Need to identify and fix the remaining bugs beyond thermal mass correction
3. **Investigate code changes**: Find what changed between Session 48 summary (1.71 MWh) and current (12.27 MWh)
4. **Consider alternative approaches**: May need to fundamentally rethink the 5R1C calculation

## Files to Investigate

1. `src/sim/engine.rs` - 5R1C solve loop and derived parameter calculation
2. `src/validation/ashrae_140_validator.rs` - CTF enablement logic (line 1284-1296)
3. Git history between Session 48 and now to find what broke the 5R1C calculation

## Recommendation

The regression cannot be fixed by simply re-enabling CTF or tweaking the thermal mass correction. The 5R1C calculation has fundamental issues that need to be identified and fixed. This requires a systematic audit of the 5R1C implementation to find where it diverges from the correct physics.
