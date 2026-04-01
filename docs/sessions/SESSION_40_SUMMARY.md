# Session 40: Heating Overprediction Root Cause Identified

**Date**: 2026-03-28
**Status**: 🔍 Root cause identified - Ti_free calculation issue in CTF mode
**Goal**: Investigate why heating is 2.2x overprediction

---

## Executive Summary

Successfully identified the root cause of heating overprediction in Case 900. The issue is that `Ti_free` (free-floating temperature) is **9.75°C lower than `Ti_actual` during heating hours** when CTF mode is enabled. This causes the HVAC system to overpredict heating demand by 2.8x. Cooling is within range (3.04 MWh vs 2.13-3.67 MWh expected).

---

## Key Findings

### Primary Issue: Ti_free Calculation in CTF Mode

**Problem**: The steady-state CTF approximation for free-floating temperature produces Ti_free that is 9.75°C too low during heating hours.

**Evidence** (diagnose_heating_overprediction.rs):
```
Average Ti_free (heating): 10.25°C
Average Ti_actual (heating): 20.00°C
Average Ti diff (heating): -9.75°C
```

**Root Cause**: HVAC demand calculation uses Ti_free:
```
Q_heating = (T_setpoint - Ti_free) / sensitivity
```

When Ti_free is 9.75°C too low, temp difference is too large, causing 2.8x heating overprediction.

### Secondary Issue: CTF Sensitivity

**Problem**: CTF sensitivity is 26% higher than 5R1C.

**Evidence**:
```
Sensitivity (5R1C): 0.013777 °C/W
Sensitivity (CTF): 0.017329 °C/W
Ratio: 1.258 (26% higher)
```

---

## Case 900 Validation Results

| Session | Heating | Cooling | Status |
|---------|----------|----------|--------|
| Session 35 (baseline) | 1.74 MWh | 9.25 MWh | Heating OK, Cooling 2.5x over |
| Session 38 (CTF free-floating) | 4.76 MWh | 1.96 MWh | ❌ Heating 2.3x over, Cooling OK |
| Session 39 (steady-state) | 4.49 MWh | 3.04 MWh | ❌ Heating 2.2x over, Cooling OK |
| **Session 40 (root cause)** | **4.49 MWh** | **3.04 MWh** | **🔍 Root cause identified** |

**Reference** (EnergyPlus):
- Heating: 1.17-2.04 MWh
- Cooling: 2.13-3.67 MWh

---

## Required Next Steps

### Priority 1: Fix Ti_free Calculation for CTF Mode

The current steady-state approximation ignores thermal inertia. Need to account for CTF thermal mass effect in Ti_free calculation.

### Priority 2: Improve CTF Sensitivity Calculation

Current CTF sensitivity is 26% higher than 5R1C. Need to include thermal mass effect in sensitivity calculation.

---

## Files Created

1. `src/bin/diagnose_heating_overprediction.rs` - Heating overprediction diagnostic
2. `src/bin/test_5r1c_vs_ctf.rs` - 5R1C vs CTF comparison
3. `SESSION_40_SUMMARY.md` - This summary document

---

## Success Criteria

| Criterion | Status |
|------------|--------|
| Root cause identified | ✅ COMPLETE (Ti_free too low by 9.75°C) |
| Secondary issues identified | ✅ COMPLETE (CTF sensitivity +26%) |
| Fix for Ti_free calculation | ❌ TODO |
| Heating < 2.5 MWh | ❌ FAIL (4.49 MWh) |

---

**Status**: 🔍 Root cause identified - Ti_free calculation issue in CTF mode
**Next**: Fix Ti_free calculation for CTF mode
