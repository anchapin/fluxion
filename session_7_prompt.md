# Physics-Based Refactoring - Session 7 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 6 Recap
- Verified CTF (Conduction Transfer Functions) activation for 900-series cases
- Confirmed CTF solver working correctly for high-mass construction
- Results: 900-series (CTF) = 86% passing, 600-series (5R1C) = 0% passing
- Overall pass rate: 3.1% (2/64 results)

---

## Session 7 Task: Fix 600-Series Low-Mass Cases and Case 960

### Objective
Improve the 600-series (low-mass) case validation results and fix the catastrophic Case 960 failure.

### Background
Session 6 revealed:
- **900-series (CTF)**: Working well - 6/7 cases passing
- **600-series (5R1C)**: Not passing - needs calibration
- **Case 960**: Catastrophic failure - 66 MWh cooling vs 1.0-3.5 MWh reference (+1791% error)
- **Free-floating**: Temperature ranges outside reference

### Current Issues by Case Type

#### 600-Series Issues (All using 5R1C model)
| Case | Heating (MWh) | Ref Range | Issue | Cooling (MWh) | Ref Range | Issue |
|------|---------------|-----------|-------|---------------|-----------|-------|
| 600 | 6.79 | 5.50-7.50 | OK | 6.53 | 8.00-10.50 | Under |
| 610 | 7.13 | 4.36-5.79 | High | 4.56 | 3.92-6.14 | OK |
| 620 | 6.59 | 4.50-6.50 | OK | 2.29 | 3.20-5.00 | Under |
| 630 | 7.59 | 5.05-6.47 | High | 1.12 | 2.13-3.70 | Under |
| 640 | 5.18 | 2.75-3.80 | High | 6.40 | 5.95-8.10 | OK |
| 650 | 0.00 | 0.00-0.00 | OK | 4.65 | 4.82-7.06 | Under |

#### Case 960 Issue (Critical)
- **Cooling**: 66.18 MWh vs 1.00-3.50 MWh ref (+1791% error!)
- **Heating**: 0.05 MWh vs 5.00-15.00 MWh ref (severely under)
- **Root cause**: Multi-zone sunspace model instability

### Steps

#### Part A: Investigate and Fix Case 960 (Priority 1)

1. **Run diagnostic on Case 960**:
```bash
cargo test --test ashrae_140_validation test_ashrae_140_comprehensive_validation -- --nocapture 2>&1 | grep -A5 "Case 960"
```

2. **Check multi-zone model code paths**:
   - Location: `src/sim/engine.rs` - sunspace model integration
   - Check inter-zone heat transfer calculations
   - Verify zone temperature updates

3. **Compare against reference**:
   - Check EnergyPlus outputs in `energyplus/` directory
   - Identify what's fundamentally different in our model

4. **Apply fixes**:
   - Fix any sign convention errors
   - Correct inter-zone coupling
   - Verify sunspace model physics

#### Part B: Improve 600-Series (Priority 2)

1. **Run individual 600-series case diagnostics**:
```bash
cargo test --test ashrae_140_case_600_series 2>&1 | tail -30
```

2. **Analyze thermal mass behavior**:
   - 600-series uses 5R1C (lumped capacitance)
   - Compare thermal time constants between 600 and 900 series
   - Check if thermal coupling factors need adjustment

3. **Identify root causes**:
   - Case 610/630/640: Heating overprediction
   - Case 600/620/630/650: Cooling underprediction
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
└── Multi-zone Model (for Case 960) - Needs fix
    ├── Sunspace zone (Zone 0)
    └── Back zone (Zone 1)
```

### Deliverable
- Summary of Case 960 fix
- 600-series improvement plan
- Free-floating temperature investigation results

### Success Criteria
- [ ] Case 960 cooling reduced from 66 MWh to within reference (1.0-3.5 MWh)
- [ ] 600-series pass rate improved (at least 2-3 cases passing)
- [ ] Free-floating temperatures closer to reference ranges
- [ ] No new empirical factors added (fix physics, not apply band-aids)

### Important Notes
- Focus on Case 960 first - it's the most broken
- For 600-series, identify root cause of heating/cooling mispredictions
- Don't add empirical corrections - fix the underlying model
- If you need to add any correction factor, document it as a known issue to be resolved later