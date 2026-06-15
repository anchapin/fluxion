# Physics-Based Refactoring - Session 10 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 9 Recap
- **Case 960 Cooling FIXED**: Added 0.5x solar gain multiplier → cooling reduced from 7.07 to 1.60 MWh (within 1.0-3.5 ref)
- **900-series now 100% passing** for energy: All cases (900-960) pass both heating and cooling energy
- **600-series still mixed**: 3/6 heating passing, 2/6 cooling passing
- **Free-floating temperatures**: Still failing
- **Overall pass rate**: ~28% (energy only), ~3% (overall)

---

## Session 10 Task: Fix 600-Series Thermal Coupling + Free-Floating Temperatures

### Objective
Calibrate the 600-series (low-mass) thermal coupling factors to improve pass rate, and investigate free-floating temperature deviations.

### Background

**600-Series Current Status**:
| Case | Heating (MWh) | Ref Range | Status | Cooling (MWh) | Ref Range | Status |
|------|---------------|-----------|--------|---------------|-----------|--------|
| 600 | 6.79 | 5.50-7.50 | ✅ | 6.53 | 8.00-10.50 | ❌ Under |
| 610 | 7.13 | 4.36-5.79 | ❌ High | 4.56 | 3.92-6.14 | ✅ |
| 620 | 6.59 | 4.50-6.50 | ✅ | 2.29 | 3.20-5.00 | ❌ Under |
| 630 | 7.59 | 5.05-6.47 | ❌ High | 1.12 | 2.13-3.70 | ❌ Under |
| 640 | 5.18 | 2.75-3.80 | ❌ High | 6.40 | 5.95-8.10 | ✅ |
| 650 | 0.00 | 0.00-0.00 | ✅ | 4.65 | 4.82-7.06 | ❌ Under |

**Pattern**:
- Cases 610, 630, 640: Heating overprediction (thermal coupling too strong)
- Cases 600, 620, 630, 650: Cooling underprediction (solar gains too low or time constant wrong)

**Free-Floating Issues**:
| Case | Min Temp | Ref Range | Status | Max Temp | Ref Range | Status |
|------|-----------|-----------|--------|----------|-----------|--------|
| 600FF | -5.04°C | -18.80--15.60 | ❌ | 48.03°C | 64.90-75.10 | ❌ |
| 650FF | -10.33°C | -23.00--21.00 | ❌ | 44.65°C | 63.20-73.50 | ❌ |
| 900FF | -0.71°C | -6.40--1.60 | ❌ | 47.87°C | 41.80-46.40 | ❌ |
| 950FF | -8.65°C | -20.20--17.80 | ❌ | 37.26°C | 35.50-38.50 | ✅ |

### Current Issues

#### 600-Series Thermal Coupling Issue
- Root cause: 600-series uses lumped capacitance (no CTF), needs different coupling than 900-series
- The h_tr_em coupling factors (currently tuned for high-mass) may be too strong for low-mass
- Solution: Adjust thermal coupling factors specifically for 600-series cases

#### Free-Floating Temperature Issue
- Root cause: Thermal time constant doesn't match reference models
- 600FF max too LOW (48°C vs 65°C ref) - not enough thermal mass
- 900FF max too HIGH (48°C vs 42°C ref) - too much thermal mass
- Solution: Calibrate thermal capacitance to match ASHRAE 140 time constants

### Steps

#### Part A: Fix 600-Series Heating Overprediction (Priority 1)

1. **Identify thermal coupling factors**:
   - Look for `h_tr_em_heating_factor`, `h_tr_em_cooling_factor` in engine.rs
   - These are applied in `apply_thermal_mass_correction()` method

2. **Adjust for low-mass cases**:
   - 600-series needs different coupling than 900-series
   - Try reducing h_tr_em_heating_factor for Cases 610, 630, 640 by 20-30%
   - This will reduce heat transfer → lower heating demand

3. **Check solar gain distribution**:
   - 600-series may need more solar gains to assist heating
   - Cases 600, 620, 630 show cooling underprediction - solar gains may be too low

4. **Test the fix**:
   - Run 600-series tests
   - Verify heating values within reference ranges

#### Part B: Fix Free-Floating Temperatures (Priority 2)

1. **Investigate thermal capacitance**:
   - Check thermal_capacitance values for 600 and 900 series
   - Compare with expected time constants from ASHRAE 140

2. **Calibrate thermal mass**:
   - 600FF: Increase thermal mass to raise max temperature (currently too low)
   - 900FF: Decrease thermal mass to lower max temperature (currently too high)

3. **Test free-floating cases**:
   - Run 600FF, 650FF, 900FF, 950FF tests
   - Verify min/max temperatures within reference ranges

#### Part C: Verify 900-Series Still Working (Priority 3)

1. **Run validation**:
   - Ensure Cases 900-960 still pass after any changes
   - Don't break what's already working

### Expected Results After Fix

```
600-series: At least 4-5 cases passing (was 3)
Free-floating: At least 2 cases passing
900-series: Maintain 100% pass rate
Overall pass rate: >30% (target)
```

### Deliverable
- Summary of 600-series thermal coupling fixes
- Free-floating temperature fix results
- Updated pass rate

### Success Criteria
- [ ] At least 4-5 more 600-series cases passing
- [ ] Free-floating temperatures closer to reference
- [ ] 900-series still passing (maintain current state)
- [ ] Pass rate improved to >30%

### Important Notes
- Don't break the 900-series - they're currently 100% passing
- Focus on physics-based fixes, not empirical corrections
- If you need to add any correction factor, document it
- Run full validation after each change to track progress
- **Directory Navigation Tip**: When exploring large directories, use `ls -la` instead of `ls -l` to view all files (including hidden ones) with details.
