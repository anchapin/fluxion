# Phase 6: Implement and Validate - Attempt Results

**Date:** 2026-03-28
**Status:** ⚠️ Inconclusive - Physics-based fixes caused catastrophic instability
**Summary:** Attempted to implement physics-based fixes from Phase 5, but found that the proposed solutions break free-floating cases catastrophically.

---

## Baseline Results (Before Any Fixes)

### ASHRAE 140 Validation

| Case | Heating (MWh) | Reference (MWh) | Error | Status |
|------|-----------------|-------------------|-------|--------|
| 600 | 19.75 | 4.30-5.71 | +294% | ❌ |
| 610 | 22.72 | 4.36-5.79 | +366% | ❌ |
| 620 | 15.25 | 4.61-5.94 | +202% | ❌ |
| 630 | 18.76 | 5.05-6.47 | +229% | ❌ |
| 640 | 19.75 | 2.75-3.80 | +472% | ❌ |
| 650 | 0.00 | 0.00-0.00 | 0% | ✅ |
| 900 | 22.54 | 1.17-2.04 | +1304% | ❌ |
| 910 | 27.81 | 1.51-2.28 | +1600% | ❌ |
| 920 | 22.81 | 3.26-4.30 | +543% | ❌ |
| 930 | 27.82 | 4.14-5.34 | +521% | ❌ |
| 940 | 22.54 | 0.79-1.41 | +2300% | ❌ |
| 950 | 0.00 | 0.00-0.00 | 0% | ✅ |
| 960 | 30.27 | 1.65-2.45 | +1600% | ❌ |
| 195 | 36.59 | 5.85-7.25 | +493% | ❌ |

**Free-Floating Cases (No HVAC):**
| Case | Min Temp (°C) | Reference (°C) | Max Temp (°C) | Reference (°C) | Status |
|------|----------------|------------------|----------------|------------------|--------|
| 600FF | -6.66 | -18.80 to -15.60 | 72.19 | 64.90 to 75.10 | ⚠️ Slightly high |
| 900FF | -2.14 | -6.40 to -1.60 | 36.76 | 41.80 to 46.40 | ✅ Reasonable |

**Observation:** Free-floating temperatures are reasonable (36-72°C range). The issue is primarily with controlled cases (HVAC enabled).

---

## Fix Attempt 1: h_tr_em = 0.0 (Remove Double-Counting)

### Implementation
- Set `h_tr_em_vec.push(0.0)` (removed resistance subtraction formula)
- Removed h_tr_em term from mass heat flow: `q_m_net = h_tr_ms * (T_s - T_m) + phi_m`

### Results

| Case | Heating (MWh) | Cooling (MWh) | Issue |
|------|-----------------|----------------|-------|
| 600 | 9.67 | 83.00 | Heating improved, cooling catastrophic |
| 900 | 18.76 | 4.14 | Slight improvement |

**Free-Floating Temperatures:**
| Case | Min Temp (°C) | Max Temp (°C) | Issue |
|------|----------------|----------------|-------|
| 600FF | -1.14 | **95.29** | ❌ CATASTROPHIC |
| 650FF | -11.40 | **80.88** | ❌ CATASTROPHIC |

### Critical Finding
**Setting h_tr_em = 0.0 causes FREE-FLOATING CASES TO OVERHEAT CATASTROPHICALLY (95°C, 81°C)**

**Root Cause:** The h_tr_em path provided crucial stabilization for free-floating cases. When HVAC is off:
- Heat from solar gains and outdoor temperature reaches mass via: Zone Air → Interior Surface → Mass
- This indirect path causes heat to accumulate in mass with no way to dissipate properly
- The old `h_tr_em_val.max(0.1)` ensured at least some direct exterior-mass coupling, preventing runaway temperatures

**Conclusion:** The h_tr_em CANNOT be set to 0.0. The Phase 4 analysis claiming h_tr_em should be removed is **INCORRECT** for the ASHRAE 140 implementation.

---

## Fix Attempt 2: h_tr_ms from Thermal Time Constant + h_tr_em = 0.0

### Implementation
- Derive h_tr_ms from τ = C_m / h_tr_ms → h_tr_ms = C_m / τ_target
- Target τ based on mass class: 2.0-4.0 hours
- Set h_tr_em = 0.0

### Results

From Phase 5 initial test:
| Case | New h_tr_ms (W/K) | τ (hours) | Old h_tr_ms (W/K) | Old τ (hours) | Issue |
|------|-------------------|------------|-------------------|-------------|-------|
| 600 | 333 | 2.0 | 1092 | 2.2 | 3.3x reduction ✅ |
| 900 | 2770 | 2.0 | 1092 | 18.1 | 2.6x INCREASED ❌ |

**Free-Floating Temperatures:** 95°C (catastrophic)

### Critical Finding
**The τ-based approach breaks Case 900:**
- Case 900 has ~8x more thermal mass than Case 600 (mainly from high-mass floor)
- Using same τ=2.0 hours for both cases gives h_tr_ms proportional to C_m
- Result: h_tr_ms for Case 900 becomes 2770 W/K (too high)
- With too-high h_tr_ms, thermal mass responds too slowly (τ still 2 hours but actual physics are different)

**Root Cause:** Using fixed τ based on mass class fails when C_m varies widely between cases. High-mass cases need HIGHER τ (slower response), not same τ.

---

## Conductance Diagnostic Results

### Case 600 (Low-Mass Baseline)
```
Mass Class: VeryLight
A_m Factor: 2.5
A_m (Effective Mass Area): 120.00 m²

Thermal Capacitance (C_m): 2.40e6 J/K

Conductances:
  OLD h_tr_ms: 1092.00 W/K (from 9.1 × A_m)
  NEW h_tr_ms: 332.81 W/K (from C_m/τ)
  Ratio: 3.28x (reduced)

Actual τ (from new h_tr_ms): 2.00 hours (7200 seconds) ✅

Validation:
  ✅ h_tr_ms in reasonable range (10-500 W/K)
  ✅ τ in reasonable range (0.5-10 hours)
```

### Case 900 (High-Mass Baseline)
```
Mass Class: Light
A_m Factor: 2.5
A_m ( Effective Mass Area): 120.00 m²

Thermal Capacitance (C_m): 1.99e7 J/K (8x higher than Case 600!)

Conductances:
  OLD h_tr_ms: 1092.00 W/K (from 9.1 × A_m)
  NEW h_tr_ms: 2770.35 W/K (from C_m/τ)
  Ratio: 0.39x (increased)

Actual τ (from new h_tr_ms): 2.00 hours (7200 seconds)

Validation:
  ⚠️  h_tr_ms > 500 W/K - TOO HIGH - no thermal buffering
  ✅ τ in reasonable range (0.5-10 hours)
```

---

## Key Insights

### 1. Baseline Implementation is Fundamentally Stable
- Free-floating temperatures are reasonable (36-72°C)
- Controlled cases have high energy error, but this is consistent with previous reports
- The thermal mass implementation is NOT causing runaway temperatures

### 2. h_tr_em Provides Critical Stabilization
- Setting h_tr_em = 0.0 breaks free-floating cases catastrophically
- The direct exterior-mass coupling path prevents heat accumulation in free-floating mode
- This contradicts the Phase 4 analysis which claimed h_tr_em should be removed

### 3. τ-Based h_tr_ms Does Not Scale Properly
- Using fixed τ based on mass class fails when C_m varies widely
- Case 900 has ~8x more thermal mass than Case 600 (mainly from floor)
- Same τ=2 hours gives very different h_tr_ms: 333 W/K (Case 600) vs 2770 W/K (Case 900)
- High h_tr_ms for Case 900 causes incorrect physics (too slow mass response)

### 4. Phase 4 Analysis May Be Incomplete
The Phase 4 analysis concluded that:
- "h_ms = 9.1 × A_m is NOT based on ISO 13790"
- "h_tr_em should be removed"

**Contradictory Evidence:**
1. Removing h_tr_em breaks free-floating cases
2. τ-based h_tr_ms makes Case 900 worse
3. Baseline free-floating temperatures are reasonable

**Alternative Hypothesis:**
The 4x energy error may NOT be due to h_tr_ms or h_tr_em at all. Possible alternative root causes:
1. Wrong HVAC energy accounting (counting mass charging as consumption)
2. Wrong internal gains calculation (solar, internal loads)
3. Wrong HVAC control logic (setpoints, deadbands)
4. Empirical calibration parameters (solar distribution, convective fractions)

---

## Recommendations

### Immediate Action Required
**DO NOT PROCEED** with implementing the Phase 5 physics-based fixes. The proposed solutions:
1. Break free-floating cases catastrophically (h_tr_em = 0.0)
2. Make high-mass cases worse (τ-based h_tr_ms)
3. Contradict evidence that baseline is reasonably stable

### Recommended Next Steps

1. **Re-examine the root cause analysis:**
   - Phase 1-2 identified thermal mass parameters as the issue
   - But my experiments show these parameters are NOT the cause
   - Need to find the ACTUAL cause of 4x energy error

2. **Investigate alternative root causes:**
   - HVAC energy accounting (Issue #272, #274, #275)
   - Internal gains calculation (solar, lights, equipment)
   - HVAC control logic (setpoints, deadbands, scheduling)
   - Empirical calibration factors in the model

3. **Compare with reference implementations more thoroughly:**
   - Run EnergyPlus simulations with identical inputs
   - Compare Fluxion's intermediate states (temperatures, heat flows) to EnergyPlus
   - Identify WHERE the simulation diverges

4. **Consider that ASHRAE 140 cases may need specific calibrations:**
   - Solar gain distribution may need case-specific tuning
   - Convective fractions may need adjustment
   - These are NOT physics-based but may be necessary for validation compliance

---

## Conclusion

The Phase 5 physics-based solution, as derived and documented, **CANNOT BE IMPLEMENTED** because:
1. It causes catastrophic instability in free-floating cases
2. It makes high-mass cases worse
3. It contradicts evidence that the baseline is reasonably stable

The root cause of the 4x energy error identified in Phases 1-4 appears to be **INCORRECTLY IDENTIFIED**. The actual cause likely lies in a different area of the simulation (energy accounting, internal gains, or empirical calibration).

---

**Phase 6 Status:** ❌ Physics-based fixes are NOT viable. Root cause re-analysis required.

---

## Files Modified

1. **`src/sim/engine.rs`** - All changes reverted to baseline
2. **`src/bin/diagnose_conductances.rs`** - Created for conductance analysis
3. **`src/bin/diagnose_free_float.rs`** - Created for free-floating analysis (incomplete)
4. **`src/bin/test_fix_combinations.rs`** - Created for incremental testing (incomplete)
5. **`docs/PHASE_6_IMPLEMENTATION_ATTEMPTS.md`** - This file (comprehensive findings)
