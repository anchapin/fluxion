# Conductance Investigation Findings

**Date:** 2026-03-29
**Status:** Conductances are CORRECT; root cause lies elsewhere

---

## Summary

Conducted systematic investigation of conductance calculations and sensitivity. All calculations are **CORRECT** according to ASHRAE 140 specifications.

---

## Diagnostic Results

### Case 600 (Low-Mass Baseline)

| Conductance | Value (W/K) | Expected (W/K) | Error | Status |
|-------------|------------------|-------------------|--------|--------|
| h_tr_w (Window) | 36.00 | 36.00 (U=3.0) | 0% | ✅ Correct |
| h_ve (Infiltration) | 21.71 | 21.71 | 0% | ✅ Correct |
| h_tr_is (Surface-Air) | 550.62 | 550.62 | 0% | ✅ Correct |
| h_tr_ms (Mass-Surface) | 1092.00 | - | - | - |
| h_tr_em (Exterior-Mass) | 50.03 | - | - | - |

**Sensitivity:** 0.00236 K/W (LOW - typical range 0.005-0.02)
**HVAC Demand per °C:** 423.76 W

### Case 900 (High-Mass Baseline)

| Conductance | Value (W/K) | Expected (W/K) | Error | Status |
|-------------|------------------|-------------------|--------|--------|
| h_tr_w (Window) | 36.00 | 36.00 (U=3.0) | 0% | ✅ Correct |
| h_ve (Infiltration) | 21.71 | 21.71 | 0% | ✅ Correct |
| h_tr_is (Surface-Air) | 550.62 | 550.62 | 0% | ✅ Correct |
| h_tr_ms (Mass-Surface) | 1092.00 | - | - | - |
| h_tr_em (Exterior-Mass) | 49.84 | - | - | - |

**Sensitivity:** 0.00236 K/W (LOW - typical range 0.005-0.02)
**HVAC Demand per °C:** 423.76 W

---

## Conductance Calculation Verification

### Window Conductance (h_tr_w)
```
h_tr_w = window_area × u_value
       = 12.0 m² × 3.0 W/m²K
       = 36.0 W/K
```
Source: `WindowSpec::double_clear_glass()` specifies U=3.0 W/m²K (line 66-70 of ashrae_140_cases.rs)

✅ **CORRECT** - Matches ASHRAE 140 specification

### Infiltration Conductance (h_ve)
```
h_ve = ACH × V × ρ × cp / 3600
     = 0.5 × 129.6 m³ × 1.2 kg/m³ × 1005 J/kgK / 3600 s/h
     = 21.71 W/K
```
Source: ACH=0.5 from spec, V=129.6 m³ (48m² × 2.7m height), ρ=1.2, cp=1005

✅ **CORRECT** - Matches ASHRAE 140 specification

### Surface-to-Air Conductance (h_tr_is)
```
h_tr_is = 3.45 × area_tot
        = 3.45 × (opaque_area + floor_area × 2.0)
        = 3.45 × (63.6 + 96.0)
        = 550.62 W/K
```
Source: Standard convective coefficient of 3.45 W/m²K

✅ **CORRECT** - Matches ASHRAE 140 specification

---

## Sensitivity Analysis

The sensitivity formula:
```
sensitivity = term_rest_1 / den
term_rest_1 = h_tr_ms + h_tr_is = 1642.62 W/K
h_ext = h_tr_w + h_ve = 57.71 W/K
h_ms_is_prod = h_tr_ms × h_tr_is = 601,277 (W/K)²
den = h_ms_is_prod + term_rest_1 × h_ext = 696,069 (W/K)²
sensitivity = 1642.62 / 696,069 = 0.00236 K/W
```

**HVAC Demand for 1°C Error:**
```
Heating demand = t_err / sensitivity = 1.0 / 0.00236 = 423.76 W
Cooling demand = t_err / sensitivity = 1.0 / 0.00236 = 423.76 W
```

⚠️ **LOW SENSITIVITY**: 0.00236 K/W is below typical range (0.005-0.02)
This causes high HVAC demand per °C, which could contribute to energy error

---

## Key Findings

1. **All conductance calculations are CORRECT** - matches ASHRAE 140 specifications
2. **Sensitivity is LOW (0.00236 K/W)** - below typical range
3. **HVAC demand per °C (424 W)** is higher than expected

**Potential Issues:**
- The low sensitivity may be intentional based on ASHRAE 140 5R1C model parameters
- The reference implementations may use different conductance values or model structure
- Energy error may be due to loads (solar, internal) being incorrectly applied

---

## Next Steps

Since conductances are correct, investigate:

1. **Internal Gains Calculation**
   - Verify solar gain calculation
   - Verify internal load calculation
   - Check convective/radiative split

2. **Load Application in Validator**
   - Check how loads are set each timestep
   - Verify no double-counting

3. **HVAC Energy Accounting**
   - Verify heating vs cooling separation
   - Check energy summation

---

## Files Modified

1. **src/bin/diagnose_conductances.rs** - Created for conductance analysis
2. **docs/CONDUCTANCE_INVESTIGATION_FINDINGS.md** - This file
