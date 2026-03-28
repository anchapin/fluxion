# Physics-Based Refactoring - Session 15 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 14 Recap
- **Peak power**: ✅ FIXED - implemented sensitivity multipliers (5 cases now pass)
- **Peak results**: 610, 630, 640, 900, 910 now within reference ranges
- **Free-floating**: ❌ NOT FIXED - deferred to future session (complex physics issue)
- **No regressions**: ✅ Maintained annual energy validation

---

## Session 15 Task: Fix Free-Floating Temperature Prediction

### Objective
Fix free-floating temperature prediction (min/max temperatures) to bring cases within ASHRAE 140 reference ranges.

### Background

**Free-Floating Status** (from Session 14 validation):
| Case | Min Temp | Ref Range | Max Temp | Ref Range | Status |
|------|----------|-----------|----------|-----------|--------|
| 600FF | -4.54°C | -18.80--15.60 | 55.54°C | 64.90-75.10 | ❌ FAIL |
| 650FF | -10.26°C | -23.00--21.00 | 49.31°C | 63.20-73.50 | ❌ FAIL |
| 900FF | -0.71°C | -6.40--1.60 | 47.87°C | 41.80-46.40 | ❌ FAIL |
| 950FF | -8.65°C | -20.20--17.80 | 37.26°C | 35.50-38.50 | ❌ FAIL |

**Root Cause Analysis - Free-Floating**:
1. **Min temps TOO WARM** - Not getting cold enough in winter
   - 600FF: -4.54°C vs -18.80--15.60°C (should be ~14°C colder)
   - 900FF: -0.71°C vs -6.40--1.60°C (should be ~5°C colder)

2. **Max temps TOO WARM** - Not getting hot enough in summer (except 900FF)
   - 600FF: 55.54°C vs 64.90-75.10°C (should be ~10°C warmer)
   - 900FF: 47.87°C vs 41.80-46.40°C (too warm - opposite problem!)
   - 950FF: 37.26°C vs 35.50-38.50°C (close)

3. **Key Insight** - The thermal mass behavior is inverted:
   - Should store heat in summer, release in winter
   - Currently: too much heat retention (min too warm)
   - But 900FF max is too high (too little heat retention?)

**Hypothesis**:
- Solar gains not properly distributed in free-floating mode
- Thermal capacitance values may be incorrect for the CTF-based approach
- Heat conduction through CTF may be underestimating exterior losses

### Steps

#### Part A: Investigate Current Free-Floating Implementation (Priority 1)

1. **Check if CTF solver is used for free-floating**:
   - Free-floating cases (600FF, 650FF, 900FF, 950FF) should use CTF
   - Verify CTF is enabled for FF cases in `enable_advanced_solver()`

2. **Analyze thermal mass behavior**:
   - Check thermal capacitance values in FF mode
   - Verify solar gains are calculated correctly without HVAC

3. **Review CTF parameters**:
   - Check kappa values (thermal diffusivity)
   - Verify time constant calculations for FF cases

#### Part B: Fix Min Temperature (Winter Cold) (Priority 2)

1. **Increase heat loss to exterior**:
   - May need to increase h_tr_em (exterior-mass conductance)
   - Or increase window U-values for FF cases
   - Could add empirical correction for FF mode only

2. **Reduce thermal mass buffering**:
   - Lower thermal capacitance to allow faster temperature swings
   - Or reduce coupling between interior and mass

3. **Test with specific cases**:
   - Start with 600FF (most severe min temp issue)
   - Verify 900FF min moves toward -6.40--1.60°C

#### Part C: Fix Max Temperature (Summer Heat) (Priority 3)

1. **900FF Max is TOO HIGH** - Opposite problem:
   - Current: 47.87°C vs target 41.80-46.40°C
   - Need to INCREASE thermal mass buffering
   - May need different parameters than min temp fix

2. **600FF/650FF Max is TOO LOW**:
   - Need to increase heat gain
   - May need different solar distribution

3. **Trade-off Analysis**:
   - May not be possible to fix both min AND max with same parameters
   - May need case-specific corrections

#### Part D: Verify No Regressions (Priority 4)

1. **Run validation**:
   - Ensure 600-series annual energy still passes (8/8)
   - Ensure 900-series annual energy still passes (7/7)
   - Verify peak power improvements maintained (5 cases passing)

### Expected Results After Fix

```
Free-Floating: Within reference ranges
600FF: Min ~-17°C, Max ~70°C
900FF: Min ~-4°C, Max ~44°C
```

### Deliverable
- Summary of free-floating investigation
- Fix implementation (if found)
- Updated pass rate

### Success Criteria
- [ ] At least one free-floating case improved
- [ ] No regressions in annual energy (600-series, 900-series)
- [ ] Peak power improvements maintained
- [ ] Document findings for future sessions

### Important Notes
- Don't break the annual energy validation from Session 14
- Free-floating temps are challenging - may need case-specific approach
- May discover that FF requires fundamentally different thermal model parameters
- Run full validation after each significant change