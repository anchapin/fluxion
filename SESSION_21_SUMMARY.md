# Session 21 Summary: 6R2C Model for Free-Floating Cases

## Session Overview
- **Date**: 2026-03-26
- **Objective**: Test if 6R2C (two-capacitance) model improves free-floating temperature predictions
- **Context**: Sessions 17-20 found 5R1C structurally limited for FF prediction

## What Was Tried

### Approach: Enable 6R2C for FF Cases
The 6R2C model has two thermal mass nodes:
- Envelope mass (walls, roof, floor)
- Internal mass (furniture, partitions)

This could potentially capture the diurnal temperature swings better than the single-capacitance 5R1C model.

### Tests Performed
| Configuration | Result |
|--------------|--------|
| 70% envelope, 150 W/K | -6.96°C (no improvement) |
| 60% envelope, 200 W/K | -6.87°C (no improvement) |
| 75% envelope, 100 W/K (default) | -6.85°C (no improvement) |

### Key Finding
**6R2C does NOT improve free-floating temperature predictions** - min temp still -6.85°C vs reference range [-6.40, -1.60]°C

The envelope/internal mass separation in 6R2C doesn't capture the fundamental limitation with ASHRAE 140 reference data.

## Root Cause Analysis

The free-floating temperature prediction problem appears to be:
1. **Model structure limitation** - 5R1C and 6R2C both use RC network approach
2. **Reference data mismatch** - ASHRAE 140 references may use different model assumptions
3. **Missing physics** - Solar distribution, infiltration, or internal gains modeling gaps

## Final Decision
- **Reverted 6R2C changes** - Kept original 5R1C model for FF cases
- **No regression** - HVAC cases (900, 910, etc.) still pass

## Files Modified
- `src/sim/engine.rs`: Added and then reverted 6R2C enablement for FF cases

## Pass Rate Status
- **Current**: ~7.8% (unchanged from Session 20)
- **FF cases**: Still failing (3/4 FAIL, 900FF WARN)

## Recommendations for Future Sessions
1. Investigate the specific thermal model assumptions in ASHRAE 140 references
2. Consider adjusting solar gain distribution for FF cases
3. Explore if infiltration modeling differs between FF and HVAC cases
4. May need empirical correction factors specific to FF cases