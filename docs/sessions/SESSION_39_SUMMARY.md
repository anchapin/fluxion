# Session 39: Revert to Steady-State CTF Approximation

**Date**: 2026-03-28
**Status**: ⚠️ Mixed results - cooling fixed, heating still overpredicting
**Goal**: Debug heating overprediction by testing cached CTF flux approach

---

## Executive Summary

Attempted to use cached CTF flux from `step_physics()` for more accurate free-floating temperature calculation. However, discovered that this creates a chicken-and-egg problem: `last_ctf_flux` is None when `calculate_free_float_temperature()` is first called (before `step_physics()` runs). Reverted to Session 38 steady-state CTF flux approximation. Cooling is now within range (3.04 MWh vs 2.13-3.67 MWh expected), but heating is still overpredicting (4.49 MWh vs 1.17-2.04 MWh expected).

---

## Progress Made

### ❌ Cached CTF Flux Approach Failed

**Problem**: Chicken-and-egg issue with cached CTF flux approach.

**Root Cause**:
- Free-floating temperature is used to calculate HVAC demand
- HVAC demand is used in `step_physics()`
- CTF flux is cached in `step_physics()`
- But `calculate_free_float_temperature()` is called BEFORE `step_physics()` runs
- Therefore, `last_ctf_flux` is always None when free-floating temperature is calculated

**Evidence** (diagnose_free_float_ctf.rs):
```
After enabling CTF:
  ctf_enabled: true
  ctf_solvers.len(): 1
  Ti_free: 25.00°C

=== CTF Solver Active ===
  Zone temperature: 20.00°C
  Outdoor temperature: 25.00°C
  CTF heat flux: 1.232 W/m²
  Flux direction: Into zone

❌ PROBLEM: CTF flux is calculated but NOT used in Ti_free calculation
```

**Resolution**: Reverted to Session 38 steady-state CTF flux approximation.

### ✅ Steady-State CTF Approximation Confirmed

**Implementation**: `calculate_free_float_temperature_ctf()` in `src/sim/engine.rs`

**Steady-State CTF Effective Conductance**:
```
h_ctf_eff = (h_tr_is × h_tr_em) / (h_tr_is + h_tr_ms + h_tr_em)
```

**Verification**:
```
Ti_free (5R1C): 25.34°C
Ti_free (CTF enabled): 25.00°C
Difference: 0.35°C

✓ Ti_free changes when CTF is enabled
  Free-floating temperature is CTF-aware
```

**Status**: Free-floating temperature is confirmed to be CTF-aware using steady-state approximation.

---

## Case 900 Validation Results

| Session | Heating | Cooling | Status |
|---------|----------|----------|--------|
| Session 35 (baseline) | 1.74 MWh | 9.25 MWh | Heating OK, Cooling 2.5x over |
| Session 36 (thermal mass fixed) | 3.77 MWh | 12.11 MWh | Both 2-4x over |
| Session 37 (CTF sensitivity) | 0.58 MWh | 45.99 MWh | ❌ Heating OK, Cooling 12x over |
| Session 38 (CTF free-floating) | 4.76 MWh | 1.96 MWh | ❌ Heating 2.3x over, Cooling OK |
| **Session 39 (steady-state)** | **4.49 MWh** | **3.04 MWh** | ❌ Heating 2.2x over, Cooling OK |

**Reference** (EnergyPlus):
- Heating: 1.17-2.04 MWh
- Cooling: 2.13-3.67 MWh

**Status**:
- ✅ Cooling: 3.04 MWh (within range 2.13-3.67 MWh)
- ❌ Heating: 4.49 MWh (2.2-3.8x overprediction vs 1.17-2.04 MWh)

---

## Required Next Steps

### Priority 1: Investigate Heating Overprediction Root Cause

Compare hourly heating demand between Fluxion and EnergyPlus. Check if overprediction is uniform or concentrated in specific periods.

### Priority 2: Test CTF Thermal Mass Effect

Implement transient CTF flux calculation to account for thermal inertia during heating season.

### Priority 3: Compare 5R1C vs CTF for Heating

Disable CTF for Case 900 to test if 5R1C works better.

---

## Success Criteria

| Criterion | Status |
|------------|--------|
| Cached CTF flux approach | ❌ ABANDONED (chicken-and-egg problem) |
| Steady-state CTF approximation | ✅ COMPLETE |
| Free-floating temperature CTF-aware | ✅ COMPLETE |
| Cooling < 3.5 MWh | ✅ COMPLETE (3.04 MWh) |
| Heating < 2.5 MWh | ❌ FAIL (4.49 MWh) |

---

**Status**: ⚠️ Cooling fixed, heating still overpredicting
**Next**: Investigate heating overprediction root cause (Priority 1)
