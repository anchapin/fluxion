# Session 83: TDD Physics Refactoring - Phase 1 - Emergency Assessment

**Date:** 2026-03-31
**Status:** DIAGNOSTIC - Critical discrepancy found
**Goal:** Identify and fix core physics issues causing massive empirical correction needs

## CRITICAL FINDING

SESSION_82 documented that Case 900 heating was **8.32 MWh raw**, requiring **÷26.8** correction to match 1.66 MWh target.

**ACTUAL CURRENT DATA** from validation run shows Case 900 heating is **0.53 MWh**, which is **UNDERPREDICTING** against 1.17-2.04 reference range.

**This represents a 16x swing** from overprediction to underprediction.

## Current Raw Physics Values (NO CORRECTIONS)

### 900-Series Heating (Current vs Reference):
| Case | Current (Raw) | Reference Min | Reference Max | Status |
|------|---------------|---------------|---------------|--------|
| 900 | 0.53 | 1.17 | 2.04 | -55% UNDERPREDICTION |
| 910 | 0.62 | 1.51 | 2.28 | -59% UNDERPREDICTION |
| 920 | 7.26 | 3.26 | 4.30 | +69% OVERPREDICTION |
| 930 | 8.90 | 4.14 | 5.34 | +67% OVERPREDICTION |
| 940 | 0.36 | 0.79 | 1.41 | -55% UNDERPREDICTION |
| 950 | 0.00 | 0.00 | 0.00 | ✅ PASS |

### 900-Series Cooling (Current vs Reference):
| Case | Current (Raw) | Reference Min | Reference Max | Status |
|------|---------------|---------------|---------------|--------|
| 900 | 2.05 | 2.13 | 3.67 | -4% UNDERPREDICTION |
| 910 | 1.29 | 0.82 | 1.88 | -29% UNDERPREDICTION |
| 920 | 1.94 | 1.84 | 3.31 | -5% UNDERPREDICTION |
| 930 | 1.52 | 1.04 | 2.24 | -32% UNDERPREDICTION |
| 950 | 0.23 | 0.39 | 0.92 | -41% UNDERPREDICTION |

## Root Cause Analysis

### Possible Explanation 1: h_tr_em Scaling is TOO AGGRESSIVE

The SESSION 82 change from 0.3 to 0.8 exponent on h_tr_em scaling may have OVER-reduced the heating coupling, causing:
- Strong thermal mass coupling reduces zone air temperature → less heating demand
- But the physics is wrong - high-mass buildings should have MORE zone heating demand due to delayed response

**Hypothesis:** Cm_ratio.powf(0.8) is making h_tr_em TOO LARGE for high-mass buildings, which means too much heat flows to mass too quickly, leaving zone air too cold.

### Possible Explanation 2: h_tr_ms (Mass-to-Surface) Coupling Issue

The h_tr_ms calculation based on thermal time constant (Session 82 line 1399) may be:
- Too high: Drains too much heat from mass back to zone surfaces
- Too low: Prevents heat from mass from reaching zone air

### Possible Explanation 3: Solar Gain Distribution Problem

Solar gains may be going:
- Mostly to thermal mass (stored, slowly released) → Not enough immediate cooling/heating effect
- Not enough to zone air (immediate effect)

## Refactoring Plan - Phase 1

### Step 1: Revert h_tr_em Scaling Exponent (Diagnostic) ✅ TESTED

**Result:** EXPONENT IS NOT THE CULPRIT

Testing showed exponent reversal from 0.8 to 0.3 made things WORSE:
- Case 900: 0.53 → 0.37 (MORE underprediction)
- Case 920: 7.26 → 5.03 (Less overprediction but still wrong direction)

**Conclusion:** The h_tr_em exponent (0.8 vs 0.3) is not the root issue. Something else is causing the contradictory behavior:
- South window cases (900, 910): Heating underpredicted
- E/W window cases (920, 930): Heating overpredicted

This pattern suggests the issue is in **case-specific solar gain distribution or window orientation handling**, not in base h_tr_em scaling.

### Step 2: Verify h_tr_ms Calculation

**File:** `src/sim/engine.rs` lines 1399-1402

Check if h_tr_ms calculation is physically reasonable:
```rust
// Formula: h_tr_ms = C_m / τ
// Where τ is thermal time constant in seconds
let h_ms_physics = if tau_seconds > 0.0 {
    total_thermal_cap / tau_seconds  // C_m / τ
} else {
    0.1  // Fallback
};
```

**Action:** Add debug output to validate values against ASHRAE 140 expected ranges

### Step 3: Check Solar Gain Distribution

**File:** `src/sim/engine.rs` lines 4200-4300 (step_physics_5r1c)

Verify that solar gains are being properly distributed:
- Beam solar: 70% to mass, 30% to air (high-mass cases)
- Check if the 70/30 split is still being applied

### Step 4: Run Validation and Document

After each step, run full validation and document actual vs expected values:

```bash
RUST_MIN_STACK=16777216 cargo test --release --lib ashrae_140_validator::tests::validate_analytical_engine 2>&1 | grep "Case 9[0-9][0-9]:"
```

## Success Criteria

- Case 900 heating: Between 1.17-2.04 MWh (without empirical correction)
- Case 920 heating: Between 3.26-4.30 MWh (without empirical correction)
- Case 930 heating: Between 4.14-5.34 MWh (without empirical correction)

## Timeline

- Step 1 (Exponent revert): 30 minutes
- Step 2 (h_tr_ms verification): 1 hour
- Step 3 (Solar distribution check): 1 hour
- Step 4 (Validation & docs): 30 minutes

**Total: 3 hours**

## Diagnostic Findings (Updated)

### Step 1 Testing Confirmed - h_tr_em Exponent NOT the Issue
- Reverting 0.8→0.3 made South cases WORSE (0.53→0.37 for Case 900)
- Keep exponent at 0.8

### Window Setup is Correct
- Case 900: with_south_window(12.0) - 12m² south ✅
- Case 920: with_ew_windows(6.0) - 6m² east + 6m² west ✅
- Window areas correctly stored and retrieved in `window_area_by_zone_and_orientation()`
- Solar gains correctly calculated per orientation in `calculate_zone_solar_gain()`

### Pattern Analysis
The contradictory behavior (South underpredicted, E/W overpredicted) suggests:
1. Maybe it's NOT a solar gain problem - maybe it's h_tr_ms or zone air coupling
2. Strong h_tr_em coupling for South cases might be draining heat from zone air too fast
3. Weak h_tr_em coupling for E/W cases might not be draining enough

**Hypothesis:** The h_tr_em.powf(0.8) scaling is TOO STRONG for high-mass cases, causing the zone air to underheat due to excessive heat transfer to mass.

## Next Session (84)

### Investigation Path (Priority)
1. Check h_tr_ms calculation - maybe TOO HIGH (drains mass too fast to surface)
2. Review thermal network balance - maybe z mass-surface coupling is inverted
3. Verify solar gain distribution fractions (70/30 split is this case-specific or case-invariant?)

### If Root Cause Found
- Phase 2: Fix core physics (h_tr_em, h_tr_ms, or solar distribution)
- Phase 3: Reduce empirical corrections to 1.0
- Phase 4: Achieve >85% pass rate without corrections
