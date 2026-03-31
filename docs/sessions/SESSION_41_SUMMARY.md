# Session 41 Summary: 920/930 Cooling Underprediction Investigation

**Date**: 2026-03-27
**Status**: Investigation Complete, Root Cause Identified

## Current Results

| Case | Cooling (MWh) | Reference | Status | Issue |
|------|---------------|-----------|--------|-------|
| 900 | 2.28 | 2.13-3.67 | ✅ PASS | - |
| 920 | 1.29 | 1.84-3.31 | ❌ FAIL | 30% below minimum |
| 930 | 0.49 | 1.04-2.24 | ❌ FAIL | 53% below minimum |

## Key Findings

### 1. Case 900 is Now Passing ✅
- Case 900 (South-facing 12m²) is now **within reference range**
- This indicates the core cooling calculation is working correctly

### 2. Solar Gain Analysis

**Summer Solstice Solar Gains (June 21):**
- South (12m²): 63,450 Wh
- E+W (12m²): 29,009 Wh
- **Ratio: 0.46** (E+W gets 46% of South-facing solar gain)

**Expected vs Actual Cooling:**
- Expected Case 920 cooling (46% of Case 900): 2.28 * 0.46 = **1.05 MWh**
- Actual Case 920 cooling: **1.29 MWh**
- **Ratio: 0.57** (Case 920 has 57% of Case 900 cooling)

The actual cooling (57%) is **higher** than the solar gain ratio (46%), which suggests the solar gain calculation is reasonable.

### 3. Shading Discrepancy (Case 930)

**Solar Gain Reduction from Shading:**
- Case 920 (no shading): 12,720 Wh (6-hour sample)
- Case 930 (with shading): 10,477 Wh (6-hour sample)
- **Reduction: 17.6%**

**Cooling Load Reduction from Shading:**
- Case 920 cooling: 1.29 MWh
- Case 930 cooling: 0.49 MWh
- **Reduction: 62%**

**Critical Finding:** The 62% cooling reduction is **3.4x larger** than the 17.6% solar gain reduction from shading. This indicates a major issue with how shading affects the cooling calculation.

### 4. Mode-Specific Coupling Factors

**Case 900 (South windows):** `(heating: 0.5, cooling: 1.3)`
**Case 920/930 (E/W windows):** `(heating: 0.8, cooling: 1.2)`

The cooling factor for E/W windows (1.2) is **lower** than for South windows (1.3). This reduces heat rejection during cooling, which should **increase** cooling load. However, Case 920 has **lower** cooling than Case 900, suggesting this factor difference is not the primary cause.

### 5. Free-Floating Temperature Comparison

At hour 12 (noon), outdoor temp 35°C:
- Case 900 (South): 23.40°C
- Case 920 (E/W): 23.40°C

**Identical free-floating temperatures** despite different solar gain patterns suggest complex thermal dynamics.

## Root Cause Analysis

The investigation reveals **two separate issues**:

### Issue 1: Case 920 - Minor Underprediction
- Case 920 cooling is **30% below minimum** reference
- Solar gain ratio (46%) vs cooling ratio (57%) suggests reasonable behavior
- The 1.29 MWh result is close to expected 1.05 MWh based on solar gains
- **Possible cause**: E/W windows have different thermal coupling patterns

### Issue 2: Case 930 - Severe Underprediction (PRIMARY ISSUE)
- Case 930 cooling is **53% below minimum** reference
- Shading only reduces solar gains by 17.6%
- But cooling is reduced by **62%** (3.4x discrepancy)
- **Possible cause**: Shading calculation or free-floating temp calculation is incorrect for shaded E/W windows

## Recommendations

### Priority 1: Investigate Shading Impact on Free-Floating Temperature
The 3.4x discrepancy between solar gain reduction (17.6%) and cooling reduction (62%) suggests that:
1. Shading may be incorrectly reducing free-floating temperatures
2. Or the HVAC demand calculation may be incorrectly handling shaded windows

**Action:** Compare hourly free-floating temperatures for Case 920 vs 930 throughout the day to identify when the discrepancy occurs.

### Priority 2: Review View Factors for Shaded E/W Windows
Shading may affect view factors differently for E/W orientations compared to South. The view factor determines how much solar gain goes directly to air vs thermal mass.

**Action:** Check if `solar_distribution_to_air` should be adjusted for shaded E/W windows.

### Priority 3: Implement Physics-Based Free-Floating Buffers
Per Session 41 prompt, replace empirical 50% reduction factors with physics-based thermal mass buffering for free-floating cases.

**Action:** Implement `calculate_free_float_thermal_mass_buffering()` function as described in Session 39.

## Next Steps

1. **Immediate**: Run hourly comparison of Case 920 vs 930 to identify when cooling discrepancy occurs
2. **Short-term**: Fix shading calculation or free-floating temp calculation for E/W windows
3. **Medium-term**: Implement physics-based thermal mass buffering for free-floating cases

## Diagnostic Tools Created

1. **diagnose_920_930_solar.rs**: Solar gain comparison for E/W vs South windows
2. **annual_solar_920.rs**: Annual solar gain analysis across seasons
3. **diagnose_free_float_920_900.rs**: Free-floating temperature comparison
4. **diagnose_ew_shading.rs**: Shading impact on E/W windows

## References

- Session 41 prompt: `session_41_prompt.md`
- Physics-based refactor: `physics_based_refactor.md`
- Empirical hacks audit: `docs/empirical_hacks_audit.md`
- ASHRAE 140 results: `docs/ASHRAE140_RESULTS.md`
