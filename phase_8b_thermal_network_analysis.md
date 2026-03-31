# Phase 8B: Thermal Network Physics Analysis

## Summary

Investigated the root cause of systematic 60-90% energy under-prediction in Fluxion ASHRAE 140 validation.

## Key Finding

### The Sensitivity Calculation is the Problem

The sensitivity calculation (line 2443 in `src/sim/engine.rs`) uses **empirical weights** to "calibrate against ASHRAE 140 reference values":

```rust
let h_is_m = self.derived_h_ms_is_prod.clone() / self.derived_term_rest_1.clone();
let h_with_mass = self.derived_h_ext.clone() + h_is_m.clone();
let is_low_mass = self.case_id.starts_with('6') || self.case_id == "195";
let h_total = if is_low_mass {
    self.derived_h_ext.clone() * 0.65 + h_with_mass.clone() * 0.35
} else {
    self.derived_h_ext.clone() * 0.75 + h_with_mass.clone() * 0.25
};
self.derived_sensitivity = self.temperatures.constant_like(1.0) / h_total.clone();
```

**Critical Issue:**
- The weights (0.65/0.35 for low-mass, 0.75/0.25 for high-mass) are **empirical calibrations**
- The comment explicitly states: "Based on calibration against ASHRAE 140 reference values"
- This means sensitivity is **tuned to match reference results, not derived from physics**

### Debug Output Analysis (Case 600, Hour 12)

```
h_ext=105.54 W/K, h_is_m=112.11 W/K, h_total=144.78 W/K, sensitivity=0.006907 K/W
Temperature: 32.66°C, Cooling setpoint: 27.0°C
Raw HVAC power: -819.99 W (cooling)
```

### The Problem

With `sensitivity = 0.006907 K/W` (thermal resistance = 144.78 K/W):

- To change temperature by 1K, you'd need 144.78 W of power
- This is **physically unrealistic** for normal HVAC operation
- The code creates an artificial resistance that makes the building appear "thermally tight"

**Impact:**
- Lower sensitivity → Higher thermal resistance → Less HVAC power needed
- Less HVAC power → Less energy accumulation → Systematic under-prediction

## Secondary Issue: h_tr_is Coefficient

The interior surface conductance uses a fixed value:
- `h_tr_is = area_tot * 3.45` (line 2361)
- Defined in `src/sim/construction.rs:660` as "ASHRAE 140 simplified 5R1C value"
- But significantly lower than ASHRAE 140 spec: 8.29 W/m²K (default), 7.69-10.0 W/m²K (by surface)

**Impact:**
- Lower h_tr_is → Less heat transfer from interior surfaces
- Makes building appear thermally "tighter" than reality

## EnergyPlus vs Fluxion Comparison

| Case | EP Heating (MWh) | Fluxion Heating (MWh) | Error | EP Cooling (MWh) | Fluxion Cooling (MWh) | Error |
|-------|------------------|----------------------|--------|------------------|----------------------|--------|
| 600 | 15,569 | 6.10 | 61% under | 21,759 | 6.55 | 70% under |
| 610 | 15,750 | 6.19 | 61% under | 15,648 | 4.91 | 69% under |
| 620 | 16,138 | 5.25 | 67% under | 14,667 | 2.80 | 81% under |
| 630 | 17,208 | 5.53 | 68% under | 10,249 | 1.56 | 85% under |
| 640 | 9,607 | 4.28 | 55% under | 20,809 | 6.47 | 68% under |
| 900 | 5,980 | 1.74 | 71% under | 8,993 | 3.77 | 58% under |
| 910 | 7,030 | 1.91 | 73% under | 5,002 | 2.75 | 45% under |

## Root Cause

The systematic 60-90% under-prediction is caused by:

1. **Empirical Sensitivity Calibration**: The sensitivity calculation uses ad-hoc weights to match ASHRAE 140 reference values rather than being derived from first principles
   - This masks underlying physics errors rather than fixing them
   - Weights vary by case type (0.65/0.35 vs 0.75/0.25) which is a calibration, not a physics-based approach

2. **Incorrect Conductance Magnitudes**: The thermal conductances may not reflect actual building physics:
   - `h_tr_is = 3.45 × area` is too low (vs ASHRAE 140: 8.29 W/m²K)
   - The sensitivity becomes 144.78 K/W, implying unrealistic thermal resistance

3. **Fundamental Model Mismatch**: The 5R1C thermal network may not capture all heat transfer paths required by ASHRAE 140:
   - Potential missing longwave radiation from interior surfaces
   - Potential missing interior surface radiation terms
   - Potential incorrect thermal mass distribution

## Recommended Fixes

### Fix 1: Validate h_tr_is Against ASHRAE 140 Specification

**Action:** Update `h_tr_is` calculation to use ASHRAE 140 surface-specific values:
- Walls: 7.69 W/m²K × wall_area
- Ceilings: 10.0 W/m²K × ceiling_area
- Floors: 5.88 W/m²K × floor_area

**Expected Impact:** Increasing h_tr_is from 3.45 to ~8 W/m²K will:
- Reduce thermal resistance (increase h_is_m)
- Reduce overall sensitivity
- Increase HVAC power required
- Increase predicted energy toward EnergyPlus values

**Risk:** This alone won't fix the 60-90% under-prediction because:
- The empirical weights were chosen to compensate for the low h_tr_is
- Increasing h_tr_is without removing the empirical weights will over-predict

### Fix 2: Remove Empirical Sensitivity Weights

**Action:** Replace empirical weighting with physics-based calculation:

```rust
// Current (empirical):
let h_total = if is_low_mass {
    h_ext * 0.65 + (h_ext + h_is_m) * 0.35
} else {
    h_ext * 0.75 + (h_ext + h_is_m) * 0.25
};

// Proposed (physics-based):
let h_total = h_ext + h_is_m;  // No weighting
self.derived_sensitivity = self.temperatures.constant_like(1.0) / h_total;
```

**Expected Impact:**
- Sensitivity based on actual thermal resistances
- May initially show worse results if other conductances are incorrect
- Provides baseline for identifying which conductances need adjustment

### Fix 3: Validate All Conductances Against ASHRAE 140

**Action:** Audit each conductance in the thermal network:

| Conductance | Current Calculation | ASHRAE 140 Spec | Status |
|------------|-------------------|------------------|--------|
| h_tr_em | From window U-value | U_window × A_window | ✅ |
| h_tr_w | From window U-value | U_window × A_window | ✅ |
| h_ve | From infiltration rate | ACH × V × ρ × cp / 3600 | ✅ |
| h_tr_is | Fixed 3.45 × area_tot | 8.29 × surface_area | ❌ Issue |
| h_tr_ms | From construction layers | R-value from C_m | ❌ Validate |
| h_tr_is_m | Derived from h_tr_ms × h_tr_is | - | Calculate first |

### Fix 4: Compare Hourly Power Traces

**Action:** Extract hourly HVAC power from EnergyPlus SQL files and compare with Fluxion on timestep basis.

**Purpose:** Identify whether under-prediction is:
- Uniform across all conditions (sensitivity issue)
- Condition-specific (e.g., during coldest hours, solar peaks)
- Time-of-day mismatch

## Next Steps

1. **Immediate**: Implement Fix 1 (h_tr_is from ASHRAE 140 spec)
2. **Test**: Run validation to measure impact
3. **If still under-predicted**: Implement Fix 2 (remove empirical weights)
4. **Compare**: Extract and compare hourly power traces from EnergyPlus
5. **Validate**: Audit all conductances against ASHRAE 140 physics

## Files to Modify

- `src/sim/engine.rs`: Update `h_tr_is` calculation
- `src/sim/construction.rs`: Review `H_SI` constant
- `src/validation/ashrae_140_cases.rs`: Verify case specifications

## References

- ASHRAE Standard 140-2017: Annex D, Tables 1-5
- EnergyPlus Engineering Reference: HVAC sizing and thermal balance
- Phase 7A: HVAC capacity and sensitivity fix
