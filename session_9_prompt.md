# Physics-Based Refactoring - Session 9 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 8 Recap
- Verified solar gains are working correctly (7422 W at noon, 0 at night)
- Case 960 cooling still overpredicts: 7.07 MWh vs 1.0-3.5 MWh ref (+102%)
- 600-series: 3 passing heating, 3 passing cooling
- Free-floating temps: 600FF too low, 900FF too high
- Overall pass rate: 3.1% (2/64)

---

## Session 9 Task: Fix Case 960 Cooling + 600-Series Thermal Calibration

### Objective
Complete the Case 960 fix and calibrate 600-series thermal coupling factors to improve pass rate.

### Background
Session 8 results:
- **Case 960**: Heating=6.02 MWh ✅ (ref: 5-15), Cooling=7.07 MWh ❌ (ref: 1-3.5)
  - Sunspace thermal buffering insufficient
- **600-series**: Mixed results - 3 pass heating, 3 pass cooling
- **Free-floating**: Max temps incorrect - thermal time constant mismatch

### Current Issues

#### Case 960 Remaining Issue
- Cooling: 7.07 MWh vs 1.0-3.5 MWh ref (+102% over max)
- Root cause: Sunspace heat transfer to back-zone too aggressive
- Solution options:
  1. Apply solar gain multiplier (0.5-0.7) to sunspace
  2. Increase sunspace thermal mass for more buffering
  3. Reduce inter-zone coupling further

#### 600-Series Issues
| Case | Heating (MWh) | Ref Range | Status | Cooling (MWh) | Ref Range | Status |
|------|---------------|-----------|--------|---------------|-----------|--------|
| 600 | 6.79 | 5.50-7.50 | ✅ | 6.53 | 8.00-10.50 | ❌ Under |
| 610 | 7.13 | 4.36-5.79 | ❌ High | 4.56 | 3.92-6.14 | ✅ |
| 620 | 6.59 | 4.50-6.50 | ✅ | 2.29 | 3.20-5.00 | ❌ Under |
| 630 | 7.59 | 5.05-6.47 | ❌ High | 1.12 | 2.13-3.70 | ❌ Under |
| 640 | 5.18 | 2.75-3.80 | ❌ High | 6.40 | 5.95-8.10 | ✅ |
| 650 | 0.00 | 0.00-0.00 | ✅ | 4.65 | 4.82-7.06 | ❌ Under |

**Pattern**:
- Cases 610/630/640: Heating overprediction (thermal coupling too strong)
- Cases 600/620/630/650: Cooling underprediction (solar gains too low or time constant wrong)

### Steps

#### Part A: Fix Case 960 Cooling (Priority 1)

1. **Apply solar gain multiplier for Case 960**:
   - In `calc_analytical_loads()` or `calculate_zone_solar_gain()`, add case-specific multiplier
   - Target: Reduce sunspace solar gains by 50-70%
   - This will reduce heat transfer to back-zone

2. **Alternative: Increase sunspace thermal mass**:
   - Find where sunspace capacitance is defined
   - Increase thermal mass for more buffering effect
   - This stores more heat in sunspace, reducing transfer to back-zone

3. **Verify the fix**:
   - Run Case 960 test
   - Target: Cooling < 3.5 MWh

#### Part B: Calibrate 600-Series Thermal Coupling (Priority 2)

1. **Identify thermal coupling factors**:
   - Look for `h_tr_em`, `h_tr_ms`, `h_tr_is` in 5R1C model
   - These control heat transfer between air, surface, and mass nodes

2. **Adjust for low-mass cases**:
   - 600-series uses lumped capacitance (no CTF)
   - Need different coupling than 900-series (high-mass with CTF)
   - Try: Reduce h_tr_em by 20-30% for heating, increase for cooling

3. **Check solar gain distribution**:
   - 600-series may need different solar distribution than 900-series
   - Verify winter solar gains are sufficient for heating assistance

4. **Apply targeted fixes**:
   - Don't use empirical corrections - fix the physics
   - Document any remaining factors that can't be resolved

#### Part C: Fix Free-Floating Temperatures (Priority 3)

1. **Investigate thermal time constant**:
   - Check thermal capacitance (C) values for 600 and 900 series
   - Compare with expected time constants from ASHRAE 140

2. **Calibrate thermal mass behavior**:
   - For 600FF: Increase thermal mass to raise max temperature
   - For 900FF: Decrease thermal mass to lower max temperature

3. **Test free-floating cases**:
   - Run 600FF, 650FF, 900FF, 950FF tests
   - Verify min/max temperatures within reference ranges

### Expected Results After Fix

```
Case 960: Heating ~6 MWh ✅, Cooling ~2.5 MWh ✅ (target)
600-series: At least 4-5 cases passing
Free-floating: Temperatures within ±5% of reference
Overall pass rate: >10% (target: 32/64)
```

### Deliverable
- Summary of Case 960 cooling fix
- 600-series thermal coupling calibration results
- Free-floating temperature fix
- Updated pass rate

### Success Criteria
- [ ] Case 960 cooling reduced from 7 MWh to within reference (<3.5 MWh)
- [ ] At least 4-5 more 600-series cases passing
- [ ] Free-floating temperatures closer to reference
- [ ] Pass rate improved to >10%

### Important Notes
- Focus on Case 960 first - it's the biggest issue
- For 600-series, find the root cause, don't just add band-aids
- If you need to add any correction factor, document it as a known issue to be resolved later
- Run full validation after each change to track progress
- **Directory Navigation Tip**: When exploring large directories, use `ls -la` instead of `ls -l` to view all files (including hidden ones) with details. For very large directories, use `ls -la | head -30` to view incrementally and avoid loading too much output at once.