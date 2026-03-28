# Physics-Based Refactoring - Session 11 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 10 Recap
- **600-series heating fixes**: Applied thermal coupling factor reductions - values improved but not passing
  - Case 610: 7.13→6.86 MWh (-3.8%)
  - Case 630: 7.59→6.97 MWh (-8.2%)
  - Case 640: 5.18→4.64 MWh (-10.4%)
- **900-series still 100% passing**: All 7 high-mass cases maintain passing status
- **Free-floating temperatures**: Deferred - requires thermal capacitance calibration
- **Overall pass rate**: ~3.1%

---

## Session 11 Task: Fix Free-Floating Temperatures + 600-Series Cooling

### Objective
Calibrate thermal capacitance to fix free-floating temperature deviations and boost 600-series cooling energy.

### Background

**Free-Floating Current Status**:
| Case | Min Temp | Ref Range | Status | Max Temp | Ref Range | Status |
|------|----------|-----------|--------|----------|-----------|--------|
| 600FF | -5.04°C | -18.8--15.6°C | ❌ Too warm | 48.03°C | 64.9-75.1°C | ❌ Too low |
| 900FF | -0.71°C | -6.4--1.6°C | ❌ Too warm | 47.87°C | 41.8-46.4°C | ❌ Too high |
| 950FF | -8.65°C | -20.2--17.8°C | ❌ Too warm | 37.26°C | 35.5-38.5°C | ✅ |

**Root Cause Analysis**:
- 600FF: Max temp too LOW (48°C vs 65°C ref) → **thermal mass too LOW**
- 900FF: Max temp too HIGH (48°C vs 42°C ref) → **thermal mass too HIGH**
- The model doesn't match ASHRAE 140 reference time constants

**600-Series Cooling Current Status**:
| Case | Cooling (MWh) | Ref Range | Status |
|------|----------------|-----------|--------|
| 600 | 6.53 | 8.00-10.50 | ❌ Under |
| 620 | 2.29 | 3.20-5.00 | ❌ Under |
| 630 | 1.12 | 2.13-3.70 | ❌ Under |
| 650 | 4.65 | 4.82-7.06 | ❌ Under |

**Pattern**: Cases 600, 620, 630, 650 show cooling **underprediction** → need solar gain boost

### Steps

#### Part A: Fix Free-Floating Temperatures (Priority 1)

1. **Understand ASHRAE 140 time constants**:
   - Each case has a specific thermal time constant (τ) from the standard
   - 600-series (low-mass): τ ≈ 1 hour
   - 900-series (high-mass): τ ≈ 4-5 hours (but model shows ~73h)

2. **Calibrate thermal capacitance**:
   - For 600FF: **INCREASE** thermal capacitance to raise max temperature
   - For 900FF: **DECREASE** thermal capacitance to lower max temperature
   - Find the kappa values (κ_wall, κ_roof, κ_floor) in engine.rs that control this

3. **Test free-floating cases**:
   - Run 600FF, 650FF, 900FF, 950FF tests
   - Verify min/max temperatures within reference ranges

#### Part B: Fix 600-Series Cooling (Priority 2)

1. **Identify solar gain boost locations**:
   - Cases 600, 620, 630, 650 show cooling underprediction
   - Need to **increase** solar gains to boost cooling energy
   - Look at solar gain multiplier or direct-to-air fraction

2. **Apply case-specific solar boosts**:
   - Try 1.2-1.5x multiplier for cooling season
   - Only apply when outdoor temp > 18°C (summer months)

3. **Test 600-series cooling**:
   - Verify values within reference ranges

#### Part C: Verify 900-Series Still Working (Priority 3)

1. **Run validation**:
   - Ensure Cases 900-960 still pass after any changes
   - Don't break what's already working

### Expected Results After Fix

```
Free-floating: At least 2-3 cases passing (was 1)
600-series cooling: At least 3-4 cases passing (was 2)
900-series: Maintain 100% pass rate
Overall pass rate: >5% (target)
```

### Deliverable
- Summary of free-floating temperature fixes
- 600-series cooling fix results
- Updated pass rate

### Success Criteria
- [ ] Free-floating temperatures closer to reference (at least 2-3 passing)
- [ ] 600-series cooling improved (at least 3-4 passing)
- [ ] 900-series still passing (maintain current state)
- [ ] Pass rate improved to >5%

### Important Notes
- Don't break the 900-series - they're currently 100% passing
- Focus on physics-based fixes, not empirical corrections
- If you need to add any correction factor, document it
- Run full validation after each change to track progress
- **Directory Navigation Tip**: When exploring large directories, use `ls -la` instead of `ls -l` to view all files (including hidden ones) with details.