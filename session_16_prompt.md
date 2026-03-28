# Physics-Based Refactoring - Session 16 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 15 Recap
- **Free-floating temps**: ✅ HVAC bug fixed (setpoints/schedules updated)
- **Free-floating results**: Still failing - thermal MODEL PARAMETERS need tuning
- **No regressions**: ✅ Annual energy and peak power unchanged

---

## Session 16 Task: Fix Thermal Model Parameters for Free-Floating Temperatures

### Objective
Fix free-floating temperature prediction by tuning thermal model parameters (conductances, solar gains) rather than empirical corrections.

### Background

**Session 15 Root Cause Fixed**: 
- HVAC schedules were initialized with 0.0 setpoints, causing cooling to always trigger
- Fixed by setting extreme setpoints (-999/999) AND updating schedules

**Remaining Issue**: Thermal model parameters need tuning:
- Min temps TOO WARM: Not losing enough heat to exterior in winter
  - 600FF: -4.54°C vs -18.80°C target (14°C too warm)
  - 900FF: -0.71°C vs -6.40°C target (5.7°C too warm)
  
- Max temps inconsistent: 
  - 600FF max 55.54°C vs 64.90°C target (9°C too cold)
  - 900FF max 47.87°C vs 46.40°C target (1.5°C too warm - opposite problem!)

**Hypothesis**: 
The 5R1C thermal network parameters need adjustment for free-floating (no HVAC) mode:
- h_tr_em (exterior-mass conductance) may be too low
- h_ve (ventilation conductance) may need adjustment
- Solar gain distribution may need different factors for FF mode
- CTF parameters for high-mass cases (900FF, 950FF) may not be applied correctly

### Steps

#### Part A: Investigate Thermal Parameters for Free-Floating (Priority 1)

1. **Check h_tr_em values**:
   - For 600FF (low-mass): Should be higher to allow more heat loss
   - For 900FF (high-mass): Should be lower due to thermal mass buffering
   
2. **Check h_ve (ventilation)**:
   - Infiltration rate affects heat loss
   - Current 0.5 ACH may need adjustment for FF cases

3. **Check solar gain distribution**:
   - In FF mode, all solar gains go to zone (no HVAC offset)
   - May need to reduce direct-to-air fraction for FF cases

4. **Verify CTF is working for 900FF/950FF**:
   - Session 15 output showed CTF enabled: "[Solver] Case 900FF: Enabled CTF solver"
   - Verify CTF parameters are being applied correctly

#### Part B: Tune Parameters for Free-Floating (Priority 2)

1. **Increase exterior heat loss**:
   - Try increasing h_tr_em by 20-50% for FF cases
   - Or increase window U-value for FF cases

2. **Adjust solar gains**:
   - May need to reduce solar gains in summer (prevent overheating)
   - Or increase in winter (help warming)

3. **Case-specific tuning**:
   - 600FF: Needs more heat loss (min temp too warm)
   - 900FF: Needs different balance (min too warm, max slightly too high)

#### Part C: Verify No Regressions (Priority 3)

1. **Run validation**:
   - Ensure 600-series annual energy still passes
   - Ensure 900-series annual energy still passes
   - Verify peak power improvements maintained

### Expected Results After Fix

```
Free-Floating: At least partially improved
600FF: Min ~-15°C (was -4.54°C), Max ~65°C (was 55.54°C)  
900FF: Min ~-4°C (was -0.71°C), Max ~44°C (was 47.87°C)
```

### Deliverable
- Summary of thermal parameter investigation
- Implementation of parameter tuning (if found)
- Updated pass rate

### Success Criteria
- [ ] At least one free-floating case shows improvement
- [ ] No regressions in annual energy (600-series, 900-series)
- [ ] Peak power improvements maintained
- [ ] Document findings for future sessions

### Important Notes
- Don't break the annual energy validation from Session 14/15
- Free-floating temps are challenging - may need case-specific approach
- Focus on physics-based parameters, NOT empirical corrections
- Run full validation after each significant change