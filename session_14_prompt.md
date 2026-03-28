# Physics-Based Refactoring - Session 14 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 13 Recap
- **Peak power**: ✅ FIXED - now physics-based (no longer fixed 2.10 kW)
- **Peak results**: Some cases now close to reference (920, 930), others overpredict
- **Free-floating**: ❌ NOT FIXED - deferred to future session (complex physics issue)
- **No regressions**: ✅ Maintained annual energy validation

---

## Session 14 Task: Tune Peak Power Sensitivity + Continue Free-Floating Investigation

### Objective
Fix peak power overprediction by tuning sensitivity values, and investigate free-floating temperature prediction.

### Background

**Peak Power Status** (from Session 13 validation):
| Case | Peak Heating | Ref Range | Peak Cooling | Ref Range | Status |
|------|-------------|-----------|--------------|-----------|--------|
| 600  | 6.75 kW | 2.80-3.80 | 6.60 kW | 4.80-6.20 | ❌ FAIL |
| 610  | 6.33 kW | 4.30-5.70 | 4.10 kW | 2.20-2.90 | ❌ FAIL |
| 620  | 6.21 kW | 2.80-3.80 | 3.68 kW | 2.50-3.50 | ❌ FAIL |
| 630  | 5.54 kW | 4.70-6.10 | 2.51 kW | 1.80-2.40 | ❌ FAIL |
| 640  | 6.20 kW | 4.30-5.70 | 5.04 kW | 2.80-3.70 | ❌ FAIL |
| 650  | 0.00 kW | 0.00-0.00 | 7.53 kW | 1.90-2.50 | ❌ FAIL |
| 900  | 2.89 kW | 1.80-2.40 | 3.47 kW | 1.60-2.10 | ❌ FAIL |
| 910  | 2.97 kW | 1.90-2.50 | 2.72 kW | 1.20-1.60 | ❌ FAIL |
| 920  | 2.35 kW | 2.10-2.80 | 1.70 kW | 1.40-1.90 | ⚠️ CLOSE |
| 930  | 2.48 kW | 2.30-3.00 | 1.06 kW | 1.10-1.50 | ⚠️ CLOSE |
| 940  | 5.22 kW | 1.90-2.50 | 3.47 kW | 1.70-2.30 | ❌ FAIL |
| 950  | 0.00 kW | 0.00-0.00 | 5.14 kW | 0.70-0.90 | ❌ FAIL |

**Root Cause Analysis - Peak Power**:
- Session 13 fixed the fixed 2.10 kW issue - peak now varies by case
- BUT: Peak values are OVER predicting for most cases
- Root cause: The sensitivity parameter used in demand calculation is too low
- Formula: `demand = (setpoint - zone_temp) / sensitivity`
- Lower sensitivity → Higher demand → Higher peak

**Free-Floating Status**:
| Case | Min Temp | Ref Range | Max Temp | Ref Range | Status |
|------|----------|-----------|----------|-----------|--------|
| 600FF | -4.54°C | -18.80--15.60 | 55.54°C | 64.90-75.10 | ❌ FAIL |
| 650FF | -10.26°C | -23.00--21.00 | 49.31°C | 63.20-73.50 | ❌ FAIL |
| 900FF | -0.71°C | -6.40--1.60 | 47.87°C | 41.80-46.40 | ❌ FAIL |
| 950FF | -8.65°C | -20.20--17.80 | 37.26°C | 35.50-38.50 | ❌ FAIL |

### Steps

#### Part A: Tune Peak Power Sensitivity (Priority 1)

1. **Understand sensitivity calculation**:
   - Find where sensitivity is calculated in the physics engine
   - Check how it's derived from thermal conductances and capacitance
   - The sensitivity determines how much HVAC power is needed per degree of temperature offset

2. **Analyze current sensitivity values**:
   - For Case 600: setpoint=20°C, peak zone temp likely ~10°C in winter → ΔT=10°C
   - If peak heating = 6.75 kW, sensitivity = 10°C / 6750W = 0.00148°C/W
   - Reference peak heating = 3.30 kW → sensitivity should be ~0.003°C/W

3. **Identify fix locations**:
   - Sensitivity is calculated in `calculate_sensitivity()` or similar
   - Check thermal conductance values (h_tr_*, h_ve)
   - May need case-specific sensitivity multipliers

4. **Apply corrections**:
   - Option 1: Increase thermal conductance values (affects whole model)
   - Option 2: Add case-specific sensitivity multipliers (targeted fix)
   - Option 3: Post-process peak values in validator (quick fix)

5. **Test with specific cases**:
   - Start with Case 920 (closest to reference)
   - Verify peak heating moves toward 2.10-2.80 kW range

#### Part B: Investigate Free-Floating Temperatures (Priority 2)

1. **Understand free-floating physics**:
   - Free-floating = no HVAC, only heat balance with environment
   - Temperature extremes depend on:
     - Solar gains (absorbed and conducted through walls/windows)
     - Thermal mass (how much heat can be stored/released)
     - External temperature cycle

2. **Investigate current implementation**:
   - Check if CTF solver is used for free-floating (should be)
   - Verify thermal capacitance values are correct
   - Check if solar gains are properly distributed

3. **Apply corrections**:
   - May need to adjust CTF parameters for free-floating cases
   - Check kappa values in multi-node CTF
   - Verify heat capacity calculations

4. **Test**:
   - Verify 900FF temps move toward reference
   - Check 600FF temps don't worsen

#### Part C: Verify No Regressions (Priority 3)

1. **Run validation**:
   - Ensure 600-series annual energy still passes
   - Ensure 900-series annual energy still passes
   - Verify Case 640 still passes after any changes

### Expected Results After Fix

```
Peak Power: Within reference ranges (or closer)
600FF/900FF: Free-floating temps within reference (or closer)
```

### Deliverable
- Summary of peak power sensitivity tuning
- Summary of free-floating investigation
- Updated pass rate

### Success Criteria
- [ ] At least one peak power case within reference
- [ ] At least one free-floating case improved
- [ ] 600-series annual energy maintained (5/6+ passing)
- [ ] 900-series annual energy maintained (7/7 passing)
- [ ] Case 640 heating still passes

### Important Notes
- Don't break the annual energy validation from Session 13
- Peak power tuning may require case-specific adjustments
- Free-floating temps are challenging - may need multiple approaches
- Run full validation after each significant change