# Session 36 Prompt: Deep Physics Fix - Remove All Empirical Factors

**Date**: 2026-03-27
**Objective**: Fix fundamental physics issues so ALL empirical correction factors can be removed while passing ASHRAE 140 validation.

---

## Current State

**Pass Rate**: 1.6% (strict 5% tolerance)

### Session 35 Achievement
- 900-series cooling now WITHIN REFERENCE RANGE for 5/6 cases
- BUT achieved through EMPIRICAL 45% summer solar reduction

### The Problem
Session 35's success came from an EMPIRICAL BAND-AID:
```rust
// Line ~3144 in engine.rs - THIS IS EMPIRICAL, NOT PHYSICS!
let summer_solar_reduction = if is_summer_month && self.case_id.starts_with('9') {
    0.55 // 45% reduction - NOT PHYSICS-BASED!
};
```

This is NOT acceptable - we need physics-based solutions!

---

## Root Causes to Fix

### 1. 900-Series Heating Underprediction

| Case | Current | Reference | Issue |
|------|---------|-----------|-------|
| 920 | 1.98 MWh | 3.26-4.30 | 40% under |
| 930 | 2.79 MWh | 4.14-5.34 | 37% under |
| 910 | 1.42 MWh | 1.51-2.28 | Near |

**Root Cause Analysis**:
- E/W windows (920, 930) have different solar geometry than South windows
- Morning (East) vs afternoon (West) sun angles are different
- Current model treats all orientations the same

### 2. 600-Series Heating Overprediction

| Case | Current | Reference | Issue |
|------|---------|-----------|-------|
| 600 | 8.65 MWh | 5.50-7.50 | 30% over |
| 610 | 9.08 MWh | 4.36-5.79 | 57% over |
| 630 | 9.02 MWh | 5.05-6.47 | 39% over |

**Root Cause Analysis**:
- Low-mass buildings have different thermal dynamics
- Internal gains (200 W/m²) may not be applied correctly
- HVAC sensitivity too high for low-mass

### 3. Free-Floating Temperature Issues

| Case | Current Max | Reference Max | Issue |
|------|-------------|----------------|-------|
| 900FF | 32.17°C | 41.8-46.4 | Too low |
| 950FF | 29.46°C | 35.5-38.5 | Too low |

**Root Cause**: Summer solar reduction affects FF cases too

### 4. Peak Load Issues

Many peak heating/cooling values don't match references - needs equipment sizing review

---

## Key Files to Investigate

1. **`src/sim/engine.rs`**:
   - Lines 3137-3169: **REMOVE** summer solar reduction empirical factor
   - Lines 1119-1130: Coupling factors
   - Lines 1414-1437: Solar distribution
   - HVAC sensitivity calculations

2. **Internal gains** - Check if 600-series properly apply 200 W/m²

---

## Session 36 Tasks

### Task 1: Remove Empirical Summer Solar Reduction (CRITICAL)

**Location**: Lines ~3137-3169 in `engine.rs`

**Current Code** (REMOVE THIS):
```rust
let summer_solar_reduction = if is_summer_month && self.case_id.starts_with('9') {
    if self.case_id.as_str() == "920" || self.case_id.as_str() == "930" {
        1.0 // E/W windows: no reduction
    } else {
        0.55 // South windows: reduce by 45% - EMPIRICAL!
    }
} else {
    1.0
};
let sol_w = solar_ref[i] * area_ref[i] * summer_solar_reduction;
```

**Why Remove**: This is empirical, not physics-based!

**What to Replace With**:
- Fix the ROOT CAUSE, not symptoms
- Consider: solar radiation direction, surface orientation, time of day
- Consider: HVAC sensitivity adjustments instead of solar reduction

### Task 2: Fix E/W Window Cases (920, 930) Heating

E/W windows have different physics than South windows:
- East windows: morning sun (low angle, high reflection)
- West windows: afternoon sun (high angle)
- Need WINTER boost, not summer reduction

**Potential Fix**:
- Add winter solar gain boost for E/W orientations
- Different coupling factors for E/W vs South

### Task 3: Fix 600-Series

**Issues**:
1. Internal gains may not be applied
2. Thermal mass coupling wrong for low-mass
3. HVAC sensitivity too high

**Potential Fix**:
- Verify internal loads = 200 W/m² applied
- Set h_tr_em coupling = 1.0 for 600-series
- Adjust HVAC sensitivity calculation

### Task 4: Fix Free-Floating Cases

**Current Problem**: Summer solar reduction affects FF cases

**Potential Fix**:
- DON'T apply summer reduction to FF cases
- Use different thermal mass treatment

---

## Expected Outcome

After removing empirical factors:
- Pass rate may DROP temporarily
- Then fix ROOT CAUSES to bring it back up
- Final goal: ≥10% pass rate with ZERO empirical factors

---

## Success Criteria

- [ ] REMOVED: Summer solar reduction empirical factor
- [ ] 900-series cooling still within reference (without empirical band-aid)
- [ ] 900-series heating fixed
- [ ] 600-series fixed
- [ ] Free-floating temperatures fixed
- [ ] Code compiles without errors
- [ ] Target: ≥10% pass rate with physics-only solutions

---

## Files to Modify

1. `src/sim/engine.rs`:
   - REMOVE lines ~3144-3164 (summer solar reduction)
   - ADD physics-based fixes for:
     - E/W window heating
     - 600-series issues
     - Free-floating cases

2. Document all changes in `SESSION_36_SUMMARY.md`

---

## Important Notes

1. **Don't just tweak numbers** - understand the physics
2. **If something works, understand WHY** before changing
3. **Test each change** - run validation to see impact
4. **Document empirical factors** - track what we're removing
