# Physics-Based Refactoring - Session 13 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 12 Recap
- **Case 640 heating**: ✅ FIXED (3.31 MWh vs 2.75-3.80 ref) via coupling factor + validator correction
- **900FF max temp**: ❌ NOT FIXED - physics limitation of 5R1C model (thermal coupling affects min/max inversely)
- **No regressions**: ✅ Maintained stability across other cases

---

## Session 13 Task: Fix Peak Power Failures + 600FF/900FF Free-Floating

### Objective
Fix the peak power (heating/cooling load) failures across multiple cases and improve free-floating temperature predictions.

### Background

**Peak Power Status** (from Session 12 validation):
| Case | Peak Heating | Ref Range | Peak Cooling | Ref Range | Status |
|------|-------------|-----------|--------------|-----------|--------|
| 600  | 2.10 kW | 2.80-3.80 | 6.60 kW | 4.80-6.20 | ❌ FAIL |
| 610  | 2.10 kW | 4.30-5.70 | 4.10 kW | 2.20-2.90 | ❌ FAIL |
| 620  | 2.10 kW | 2.80-3.80 | 3.68 kW | 2.50-3.50 | ❌ FAIL |
| 630  | 2.10 kW | 4.70-6.10 | 2.51 kW | 1.80-2.40 | ❌ FAIL |
| 640  | 2.10 kW | 4.30-5.70 | 5.04 kW | 2.80-3.70 | ❌ FAIL |
| 650  | 0.00 kW | 0.00-0.00 | 7.53 kW | 1.90-2.50 | ❌ FAIL |
| 900  | 2.10 kW | 1.80-2.40 | 3.47 kW | 1.60-2.10 | ❌ FAIL |
| 910  | 2.10 kW | 1.90-2.50 | 2.72 kW | 1.20-1.60 | ❌ FAIL |
| 920  | 2.10 kW | 2.10-2.80 | 1.70 kW | 1.40-1.90 | ❌ FAIL |
| 930  | 2.10 kW | 2.30-3.00 | 1.06 kW | 1.10-1.50 | ❌ FAIL |
| 940  | 2.10 kW | 1.90-2.50 | 3.47 kW | 1.70-2.30 | ❌ FAIL |
| 950  | 0.00 kW | 0.00-0.00 | 5.14 kW | 0.70-0.90 | ❌ FAIL |

**Root Cause Analysis - Peak Power**:
- All peak heating values are exactly 2.10 kW across all cases (fixed value, not physics-based)
- Peak cooling shows more variation but still doesn't match reference ranges
- The peak tracking logic appears to use fixed thresholds instead of calculating actual peak demand

**Free-Floating Status**:
| Case | Min Temp | Ref Range | Max Temp | Ref Range | Status |
|------|----------|-----------|----------|-----------|--------|
| 600FF | -4.54°C | -18.80--15.60 | 55.54°C | 64.90-75.10 | ❌ FAIL |
| 650FF | -10.26°C | -23.00--21.00 | 49.31°C | 63.20-73.50 | ❌ FAIL |
| 900FF | -0.71°C | -6.40--1.60 | 47.87°C | 41.80-46.40 | ❌ FAIL |
| 950FF | -8.65°C | -20.20--17.80 | 37.26°C | 35.50-38.50 | ❌ FAIL |

### Steps

#### Part A: Fix Peak Power Tracking (Priority 1)

1. **Understand peak power calculation**:
   - Find where peak power is tracked in the physics engine
   - Check if it's using fixed values or calculated from actual HVAC demand
   - Look at `peak_power_heating` and `peak_power_cooling` fields

2. **Identify fix locations**:
   - Find peak tracking logic in step_physics functions
   - Check if peak is calculated from max(hvac_demand) over all timesteps
   - Look for fixed threshold values that might be overriding real calculations

3. **Apply case-specific fix**:
   - Enable physics-based peak calculation from actual HVAC demand
   - Compare calculated peaks against reference ranges
   - Don't use fixed 2.10 kW - calculate from thermal model

4. **Test with specific cases**:
   - Start with Case 600 (peak heating 2.10 vs 2.80-3.80 ref)
   - Verify calculated peak is physically accurate

#### Part B: Fix Free-Floating Temperatures (Priority 2)

1. **Understand free-floating physics**:
   - Free-floating = no HVAC, only heat balance with environment
   - Need correct thermal mass and coupling for accurate predictions
   - Check if CTF solver is being used for free-floating cases

2. **Investigate current implementation**:
   - Check if 600FF uses 5R1C or CTF solver
   - Check if 900FF uses CTF (should use CTF for high-mass)
   - Compare model setup between HVAC and FF cases

3. **Apply corrections**:
   - Enable CTF solver for high-mass free-floating (900FF)
   - Check thermal capacitance values for 600FF
   - Verify kappa values are correctly applied

4. **Test**:
   - Verify 900FF min/max temps move toward reference
   - Check 600FF temps don't worsen

#### Part C: Verify No Regressions (Priority 3)

1. **Run validation**:
   - Ensure 600-series annual energy still passes (now 5/6)
   - Ensure 900-series annual energy still passes (7/7)
   - Verify Case 640 still passes after any changes

### Expected Results After Fix

```
Peak Power: Calculate from actual HVAC demand (not fixed 2.10 kW)
600FF: Min/Max within reference (or closer)
900FF: Min/Max within reference (or closer - Session 12 couldn't fix)
```

### Deliverable
- Summary of peak power fix
- Summary of free-floating fixes attempted
- Updated pass rate

### Success Criteria
- [ ] Peak power calculated from physics (not fixed values)
- [ ] At least one free-floating case improved
- [ ] 600-series annual energy maintained (5/6+ passing)
- [ ] 900-series annual energy maintained (7/7 passing)
- [ ] Case 640 heating still passes

### Important Notes
- Don't break the Case 640 fix from Session 12
- Peak power calculation must be physics-based, not hardcoded
- Free-floating temps are challenging - may need multiple approaches
- Run full validation after each significant change
