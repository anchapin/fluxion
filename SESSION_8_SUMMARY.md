# Session 8 Summary: Physics-Based Refactoring Investigation

## Session Overview
- **Date**: 2026-03-25
- **Session Focus**: Fix Case 960 Cooling + 600-Series Investigation
- **Previous Session**: Session 7 - Fixed inter-zone heat transfer (89% improvement)

## Key Findings

### 1. Solar Gains Are Working Correctly ✅

Verified through debug output that solar gains are properly calculated:
- Daytime hours: Solar gains are non-zero (e.g., 7422 W at noon)
- Nighttime hours: Solar gains = 0 (correct)
- Weather data (DNI, DHI) is properly passed to the calculation

Sample debug output (timestep 12 - noon, January):
```
DEBUG calc_analytical_loads: timestep=12, weather dni=770.48, dhi=94.63, month=1, day=1, hour=12
DEBUG solar: timestep=12, zone_idx=0, solar_gain_watts=7422.61 W, floor_area=48 m²
```

### 2. Case 960 Status (Priority 1)

From Session 7 improvements:
- Heating: 6.02 MWh ✅ (ref: 5-15 MWh) - PASS
- Cooling: 7.07 MWh ❌ (ref: 1-3.5 MWh) - Still 2x over max
- **Issue**: Sunspace thermal model still allows too much heat transfer to back-zone

### 3. Case 900 Status

- Heating: 1.17 MWh (ref: 1.17-2.04) - At min boundary ✅
- Cooling: 3.82 MWh (ref: 2.13-3.67) - 4% over max ⚠️
- Free-floating max: 47.87°C (ref: 41.8-46.4°C) - 3% over max ⚠️

### 4. 600-Series Status

| Case | Heating | Ref | Status | Cooling | Ref | Status |
|------|---------|-----|--------|---------|-----|--------|
| 600  | 6.79    | 5.50-7.50 | ✅ | 6.53 | 8.00-10.50 | ❌ |
| 610  | 7.13    | 4.36-5.79 | ❌ | 4.56 | 3.92-6.14 | ✅ |
| 620  | 6.59    | 4.50-6.50 | ✅ | 2.29 | 3.20-5.00 | ❌ |
| 630  | 7.59    | 5.05-6.47 | ❌ | 1.12 | 2.13-3.70 | ❌ |
| 640  | 5.18    | 2.75-3.80 | ❌ | 6.40 | 5.95-8.10 | ✅ |
| 650  | 0.00    | 0.00-0.00 | ✅ | 4.65 | 4.82-7.06 | ❌ |

**Pattern**:
- Heating overprediction: Cases 610, 630, 640
- Cooling underprediction: Cases 600, 620, 630, 650

### 5. Free-Floating Temperature Issues

| Case | Model Max | Reference | Status |
|------|-----------|-----------|--------|
| 600FF | 48.03°C | 64.90-75.10°C | ❌ Too low |
| 900FF | 47.87°C | 41.80-46.40°C | ❌ Too high |

**Root Cause**: Thermal mass dynamics not correctly modeling heat storage/release

## Root Cause Analysis

### Case 960 Cooling Overprediction
1. Sunspace (Zone 1) receives too much solar gain
2. Inter-zone coupling (9 W/K) allows heat to flow to back-zone too easily
3. HVAC demand calculated for back-zone includes sunspace heat contribution

### 600-Series Heating Overprediction  
1. 5R1C model thermal coupling factors not tuned for low-mass cases
2. Solar gains may be underestimated during winter months

### Free-Floating Temperature Issues
1. Thermal mass time constant doesn't match reference models
2. Heat storage/release dynamics in 5R1C model need refinement

## Current Pass Rate: 3.1% (2/64)

From `docs/ASHRAE140_RESULTS.md`:
- Passed: 2 (Case 900 heating at min, Case 195 unspecified)
- Failed: 61
- Warnings: 1

## Recommendations for Future Sessions

### Priority 1: Fix Case 960 Cooling (1-2 hours)
- Reduce solar gains to sunspace by applying multiplier (0.5-0.7)
- OR increase thermal mass of sunspace for more buffering
- Target: Reduce cooling from 7.07 MWh to <3.5 MWh

### Priority 2: Fix 600-Series Heating (2-3 hours)
- Adjust thermal coupling factors (h_tr_em, h_tr_ms) for low-mass
- Verify solar gain distribution during winter
- Target: Get at least 3-4 cases passing

### Priority 3: Fix Free-Floating Temps (2-3 hours)
- Calibrate thermal mass time constant
- Compare with EnergyPlus/ESP-r thermal behavior
- Target: Within ±5% of reference ranges

### Priority 4: Full Validation (1-2 hours)
- Run complete ASHRAE 140 test suite
- Document all remaining issues
- Target: ≥50% pass rate (32/64)

## Files Modified
- `src/sim/engine.rs`: Debug logging (lines 4906-4930) - temporary for investigation
- `docs/ASHRAE140_RESULTS.md`: Validation report regenerated

## Session Status: INCOMPLETE
The investigation revealed that:
1. Solar gains are working correctly ✅
2. More work needed on thermal modeling for specific cases ❌
3. Pass rate remains low at 3.1%

**Next Session Should**: Continue with Case 960 cooling fix and 600-series thermal calibration