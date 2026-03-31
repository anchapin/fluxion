# Session 48 Regression - RESOLVED

**Date**: 2026-03-27
**Issue**: Investigation of "regression" from 4.75 MWh → 12.27 MWh heating

## Resolution

**The regression was NEVER in the main codebase.** Current main (commit cbf8bfa) already produces correct results:

| Test | Heating | Cooling | Status |
|------|---------|---------|--------|
| Session 33 (faab4be) | 4.75 MWh | 6.95 MWh | ✅ Working baseline |
| Current main (cbf8bfa) | 4.75 MWh | 6.95 MWh | ✅ Same as Session 33 |
| Reference range | 1.17-2.04 MWh | 2.13-3.67 MWh | ❌ 2-4x too high |

## Key Findings

1. **No regression in main code**: Session 33 → Current main produces identical results
2. **Session 48 reports are inconsistent**: Session 48_SUMMARY.md claims 1.71 MWh heating, but:
   - This doesn't match Session 33 baseline (4.75 MWh)
   - This doesn't match current main (4.75 MWh)
   - Likely from local uncommitted changes or different test configuration

3. **Real issue**: The actual problem is that BOTH Session 33 and current main produce 4.75 MWh heating, which is:
   - 2.3x the midpoint of the reference range (1.60 MWh)
   - 4.75 MWh vs 1.17-2.04 MWh reference
   - Still failing validation despite CTF being enabled

## Investigation Summary

### What Was Tested

1. ✅ Checked out Session 33 commit (faab4be) → 4.75 MWh with CTF
2. ✅ Checked out current main (cbf8bfa) → 4.75 MWh with CTF
3. ✅ Compared code between Session 33 and current main → No significant differences
4. ✅ Verified CTF is enabled in both cases

### What Was Discovered

The "Session 48 regression" analysis was based on incorrect assumptions:
- Session 48 documents claimed 1.71 MWh baseline (unverified)
- Current main actually matches Session 33 exactly (4.75 MWh)
- No code changes caused a regression from 4.75 → 12.27 MWh

### The Real Problem

The actual issue is NOT a regression, but that the baseline itself (4.75 MWh) is still wrong:

**Case 900 Validation Status**:
- Annual Heating: 4.75 MWh (Reference: 1.17-2.04 MWh) → **2.3x too high**
- Annual Cooling: 6.95 MWh (Reference: 2.13-3.67 MWh) → **2x too high**

Both metrics are **2-4x above the reference range**, even with CTF enabled.

## Next Steps

The regression investigation is complete. The real work needed is:

1. **Investigate why heating is 2.3x too high** (4.75 vs 1.60 MWh reference midpoint)
2. **Investigate why cooling is 2x too high** (6.95 vs 2.90 MWh reference midpoint)
3. **Consider alternative approaches**:
   - Check if there are other empirical factors that need removal
   - Verify the 5R1C network topology is correct
   - Compare with EnergyPlus hourly results to find the discrepancy
   - Consider whether the reference tools themselves are using different assumptions

## Conclusion

**STATUS**: ✅ Regression investigation complete - no regression found

**CURRENT STATE**:
- Main code matches Session 33 baseline exactly
- CTF is working as designed
- Results are consistent but still 2-4x above reference range

**RECOMMENDATION**: Focus on understanding why the baseline itself is 2-4x too high, rather than chasing regressions that don't exist.
