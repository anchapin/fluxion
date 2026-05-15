# Physics-Based Refactoring - Session 8 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 7 Recap
- Fixed Case 960 inter-zone heat transfer sign convention (lines 3343-3348 in engine.rs)
- Increased inter-zone coupling from 1.5 to 9 W/K for Case 960
- Case 960 cooling reduced from 66 MWh to 7.07 MWh (89% improvement!)
- 600-series investigation deferred to Session 8
- Overall pass rate: 3.1% (no change - need to test more cases)

---

## Session 8 Task: Fix Case 960 Cooling + 600-Series Investigation

### Objective
Complete the Case 960 fix (reduce remaining cooling overprediction) and investigate 600-series thermal model issues.

### Background
Session 7 results:
- **Case 960**: Heating=6.02 MWh ✅ (ref: 5-15), Cooling=7.07 MWh ❌ (ref: 1-3.5)
  - Still 2x over max reference
- **600-series**: Mixed results with cooling underprediction and heating overprediction

### Current Issues by Case Type

#### Case 960 Remaining Issue
- Cooling: 7.07 MWh vs 1.0-3.5 MWh ref (+102% over max)
- Root cause: Sunspace solar gains are too high / not properly distributed
- The sunspace zone (Zone 1) should act as a buffer but current model allows too much heat transfer to back-zone

#### 600-Series Issues (All using 5R1C model)
| Case | Heating (MWh) | Ref Range | Status | Cooling (MWh) | Ref Range | Status |
|------|---------------|-----------|--------|---------------|-----------|--------|
| 600 | 6.79 | 5.50-7.50 | ✅ | 6.53 | 8.00-10.50 | ❌ Under |
| 610 | 7.13 | 4.36-5.79 | ❌ High | 4.56 | 3.92-6.14 | ✅ |
| 620 | 6.59 | 4.50-6.50 | ✅ | 2.29 | 3.20-5.00 | ❌ Under |
| 630 | 7.59 | 5.05-6.47 | ❌ High | 1.12 | 2.13-3.70 | ❌ Under |
| 640 | 5.18 | 2.75-3.80 | ❌ High | 6.40 | 5.95-8.10 | ✅ |
| 650 | 0.00 | 0.00-0.00 | ✅ | 4.65 | 4.82-7.06 | ❌ Under |

### Steps

#### Part A: Fix Case 960 Remaining Cooling (Priority 1)

1. **Investigate solar gains to sunspace**:
   - Check how solar gains are calculated for Zone 1 (sunspace)
   - The sunspace has 6 m² of south-facing windows vs 12 m² for back-zone
   - During summer, sunspace heats up but should NOT transfer all heat to back-zone

2. **Check HVAC demand calculation for multi-zone**:
   - The sunspace (Zone 1) is free-floating - no HVAC
   - Only back-zone (Zone 0) has HVAC
   - Verify HVAC demand is calculated only for Zone 0

3. **Apply targeted fix**:
   - Option 1: Reduce solar gains to sunspace (multiplier)
   - Option 2: Increase inter-zone coupling further (allow more heat to "stay" in sunspace)
   - Option 3: Adjust thermal mass of sunspace (larger capacitance = more buffering)

#### Part B: Investigate 600-Series (Priority 2)

1. **Run individual 600-series diagnostics**:
```bash
cargo test --test ashrae_140_case_600_series 2>&1 | tail -30
```

2. **Analyze thermal mass behavior**:
   - 600-series uses 5R1C (lumped capacitance)
   - Compare thermal time constants between 600 and 900 series
   - Check if thermal coupling factors need adjustment

3. **Identify root causes**:
   - Cases 610/630/640: Heating overprediction
   - Cases 600/620/630/650: Cooling underprediction
   - Check HVAC demand calculations
   - Verify solar gain distribution

4. **Apply targeted fixes**:
   - Don't use empirical corrections - fix the physics
   - Document any remaining empirical factors

#### Part C: Free-Floating Temperature Fix (Priority 3)

1. **Check temperature ranges**:
   - 600FF: Max 48°C vs 64.9-75.1°C ref (too low)
   - 900FF: Max 47.87°C vs 41.8-46.4°C ref (too high)
   - Need correct thermal mass behavior

2. **Investigate thermal coupling**:
   - Check `h_tr_em` and `h_tr_ms` coupling factors
   - Verify thermal mass temperature calculation
   - Compare free-floating vs HVAC-controlled behavior

### Expected Architecture After Fix

```
Thermal Model
├── 5R1C Model (for 600-series)
│   ├── Zone air node (Ti)
│   ├── Thermal mass node (Tm)
│   └── Lumped capacitance
├── CTF Model (for 900-series) - Working ✅
│   └── Transient conduction
└── Multi-zone Model (for Case 960) - Needs final fix
    ├── Back-zone (Zone 0) - HVAC controlled
    └── Sunspace (Zone 1) - Free-floating buffer
```

### Deliverable
- Summary of Case 960 cooling fix
- 600-series investigation results
- Free-floating temperature investigation

### Success Criteria
- [ ] Case 960 cooling reduced from 7 MWh to within reference (1.0-3.5 MWh)
- [ ] 600-series pass rate improved (at least 2-3 cases passing)
- [ ] Free-floating temperatures closer to reference ranges
- [ ] No new empirical factors added (fix physics, not apply band-aids)

### Important Notes
- Focus on Case 960 first - the remaining issue is just 2x over
- For 600-series, identify root cause of heating/cooling mispredictions
- Don't add empirical corrections - fix the underlying model
- If you need to add any correction factor, document it as a known issue to be resolved later
