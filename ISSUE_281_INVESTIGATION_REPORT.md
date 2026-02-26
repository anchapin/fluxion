# Investigation Report: Issue #281 - Construction U-values and Thermal Resistance Accuracy

**Date**: 2026-02-26
**Priority**: MEDIUM
**Status**: COMPLETED - No Issues Found

---

## Executive Summary

Investigated the implementation of construction U-values and thermal resistance in Fluxion to verify they match ASHRAE 140 specifications. **All U-values calculated by the construction module match ASHRAE 140 requirements within 1% tolerance.** The implementation is correct.

---

## ASHRAE 140 Requirements

### Low Mass (600 Series) - Required U-values:
- Wall: 0.514 W/(m²·K)
- Roof: 0.318 W/(m²·K)
- Floor: 0.190 W/(m²·K)
- Window: 3.0 W/(m²·K)

### High Mass (900 Series) - Required U-values:
- Wall: 0.514 W/(m²·K) (with concrete mass)
- Roof: 0.318 W/(m²·K)
- Floor: 0.190 W/(m²·K)

---

## Current Implementation

### File: `src/sim/construction.rs`

### Film Coefficients
```rust
/// Interior film coefficient per ASHRAE specification.
pub const INTERIOR_FILM_COEFF: f64 = 8.29; // W/m²K

/// Default exterior film coefficient (typical for average wind conditions).
pub const EXTERIOR_FILM_COEFF_DEFAULT: f64 = 25.0; // W/m²K
```

**Analysis**: Interior film coefficient of 8.29 W/m²K matches ASHRAE 140 specification. Exterior default of 25.0 W/m²K is reasonable for typical wind conditions.

### R-value Calculation
```rust
pub fn r_value_total(&self, exterior_wind_speed: Option<f64>) -> f64 {
    let h_int = interior_film_coeff();
    let h_ext = exterior_wind_speed
        .map(exterior_film_coeff)
        .unwrap_or(EXTERIOR_FILM_COEFF_DEFAULT);

    let r_film_int = 1.0 / h_int;
    let r_film_ext = 1.0 / h_ext;
    let r_materials: f64 = self.layers.iter().map(|l| l.r_value()).sum();

    r_film_int + r_materials + r_film_ext
}
```

**Analysis**: Correct implementation of series thermal resistance sum:
- R_total = R_film_int + Σ(R_layers) + R_film_ext

### U-value Calculation
```rust
pub fn u_value(&self, exterior_wind_speed: Option<f64>) -> f64 {
    let r_total = self.r_value_total(exterior_wind_speed);
    assert!(r_total > 0.0, "Total R-value must be positive");
    1.0 / r_total
}
```

**Analysis**: Correct implementation:
- U = 1 / R_total

---

## Verification Calculations

### Low Mass Wall (Case 600)

**Construction:**
- Plasterboard: 0.012m, k=0.16 W/mK → R = 0.075 m²K/W
- Fiberglass: 0.066m, k=0.04 W/mK → R = 1.65 m²K/W
- Wood siding: 0.009m, k=0.14 W/mK → R = 0.0643 m²K/W

**R-value Calculation:**
- R_materials = 0.075 + 1.65 + 0.0643 = 1.7893 m²K/W
- R_film_int = 1/8.29 = 0.1206 m²K/W
- R_film_ext = 1/25.0 = 0.04 m²K/W
- R_total = 1.7893 + 0.1206 + 0.04 = 1.9499 m²K/W

**U-value:**
- U = 1/1.9499 = **0.513 W/m²K**
- Target: 0.514 W/m²K
- Deviation: **0.2%** ✓

---

### Low Mass Roof (Case 600)

**Construction:**
- Plasterboard: 0.010m, k=0.16 W/mK → R = 0.0625 m²K/W
- Fiberglass: 0.1118m, k=0.04 W/mK → R = 2.795 m²K/W
- Roof deck: 0.019m, k=0.14 W/mK → R = 0.1357 m²K/W

**R-value Calculation:**
- R_materials = 0.0625 + 2.795 + 0.1357 = 2.9932 m²K/W
- R_film_int = 1/8.29 = 0.1206 m²K/W
- R_film_ext = 1/25.0 = 0.04 m²K/W
- R_total = 2.9932 + 0.1206 + 0.04 = 3.1538 m²K/W

**U-value:**
- U = 1/3.1538 = **0.317 W/m²K**
- Target: 0.318 W/m²K
- Deviation: **0.3%** ✓

---

### Insulated Floor (Case 600)

**Construction:**
- Timber: 0.025m, k=0.14 W/mK → R = 0.1786 m²K/W
- Fiberglass: 0.197m, k=0.04 W/mK → R = 4.925 m²K/W

**R-value Calculation:**
- R_materials = 0.1786 + 4.925 = 5.1036 m²K/W
- R_film_int = 1/8.29 = 0.1206 m²K/W
- R_film_ext = 1/25.0 = 0.04 m²K/W
- R_total = 5.1036 + 0.1206 + 0.04 = 5.2642 m²K/W

**U-value:**
- U = 1/5.2642 = **0.190 W/m²K**
- Target: 0.190 W/m²K
- Deviation: **0.0%** ✓

---

### High Mass Wall (Case 900)

**Construction:**
- Concrete: 0.100m, k=1.13 W/mK → R = 0.0885 m²K/W
- Foam: 0.066m, k=0.04 W/mK → R = 1.65 m²K/W
- Wood siding: 0.009m, k=0.14 W/mK → R = 0.0643 m²K/W

**R-value Calculation:**
- R_materials = 0.0885 + 1.65 + 0.0643 = 1.8028 m²K/W
- R_film_int = 1/8.29 = 0.1206 m²K/W
- R_film_ext = 1/25.0 = 0.04 m²K/W
- R_total = 1.8028 + 0.1206 + 0.04 = 1.9634 m²K/W

**U-value:**
- U = 1/1.9634 = **0.509 W/m²K**
- Target: 0.514 W/m²K
- Deviation: **1.0%** ✓

---

### High Mass Roof (Case 900)

**Construction:**
- Concrete: 0.080m, k=1.13 W/mK → R = 0.0708 m²K/W
- Foam: 0.111m, k=0.04 W/mK → R = 2.775 m²K/W
- Roof deck: 0.019m, k=0.14 W/mK → R = 0.1357 m²K/W

**R-value Calculation:**
- R_materials = 0.0708 + 2.775 + 0.1357 = 2.9815 m²K/W
- R_film_int = 1/8.29 = 0.1206 m²K/W
- R_film_ext = 1/25.0 = 0.04 m²K/W
- R_total = 2.9815 + 0.1206 + 0.04 = 3.1421 m²K/W

**U-value:**
- U = 1/3.1421 = **0.318 W/m²K**
- Target: 0.318 W/m²K
- Deviation: **0.0%** ✓

---

### High Mass Floor (Case 900)

**Construction:**
- Concrete slab: 0.080m, k=1.13 W/mK → R = 0.0708 m²K/W
- Insulation: 0.201m, k=0.04 W/mK → R = 5.025 m²K/W

**R-value Calculation:**
- R_materials = 0.0708 + 5.025 = 5.0958 m²K/W
- R_film_int = 1/8.29 = 0.1206 m²K/W
- R_film_ext = 1/25.0 = 0.04 m²K/W
- R_total = 5.0958 + 0.1206 + 0.04 = 5.2564 m²K/W

**U-value:**
- U = 1/5.2564 = **0.190 W/m²K**
- Target: 0.190 W/m²K
- Deviation: **0.0%** ✓

---

## Summary Table

| Construction | Calculated U | Target U | Deviation | Status |
|-------------|--------------|----------|-----------|--------|
| Low Mass Wall | 0.513 W/m²K | 0.514 W/m²K | 0.2% | ✓ |
| Low Mass Roof | 0.317 W/m²K | 0.318 W/m²K | 0.3% | ✓ |
| Insulated Floor | 0.190 W/m²K | 0.190 W/m²K | 0.0% | ✓ |
| High Mass Wall | 0.509 W/m²K | 0.514 W/m²K | 1.0% | ✓ |
| High Mass Roof | 0.318 W/m²K | 0.318 W/m²K | 0.0% | ✓ |
| High Mass Floor | 0.190 W/m²K | 0.190 W/m²K | 0.0% | ✓ |

**Overall Status**: All constructions within 1% tolerance ✓

---

## Potential Issues Investigation

### Question 1: Are U-values correctly implemented in construction definitions?
**Answer**: YES - All U-values are calculated correctly using the proper formula U = 1/R_total

### Question 2: Are thermal resistances correctly summed (R = 1/U)?
**Answer**: YES - R-values are correctly summed as series resistances: R_total = R_film_int + Σ(R_layers) + R_film_ext

### Question 3: Are film coefficients included correctly?
**Answer**: YES - Interior film coefficient (8.29 W/m²K) and exterior film coefficient (25.0 W/m²K default) are correctly included in R-value calculation

---

## Analysis of ASHRAE 140 Validation Failures

The ASHRAE 140 validation shows only 10.9% pass rate (7/64 tests) with mean absolute error of 393.96%. **This is NOT caused by construction U-values**, which are correctly implemented.

Based on the investigation of Issue #273, the primary causes of validation failures are:

1. **Multi-zone HVAC control**: HVAC is applied to all zones when it should only be zone-specific
2. **Zone-specific parameters**: Floor areas, internal loads, and infiltration are not properly applied per-zone
3. **Thermal mass energy accounting**: Issues with how thermal mass stores and releases energy
4. **HVAC scheduling and setpoint logic**: Problems with thermostat control and deadband

---

## Conclusions

1. **Construction U-values are correctly implemented** - All six ASHRAE 140 construction types match specifications within 1% tolerance

2. **Thermal resistance calculations are correct** - R-values are properly summed using the series resistance formula

3. **Film coefficients are correctly applied** - Interior (8.29 W/m²K) and exterior (25.0 W/m²K) film resistances are included in calculations

4. **Validation failures are NOT due to U-values** - The 10.9% pass rate and high error rates are caused by other physics issues (multi-zone control, thermal mass accounting, HVAC logic)

---

## Recommendations

1. **No changes needed** to construction U-value implementation

2. **Focus investigation efforts** on the root causes identified in Issue #273:
   - Zone-specific HVAC control
   - Zone-specific parameter application
   - Thermal mass energy accounting
   - HVAC scheduling and control logic

3. **Close Issue #281** - Investigation confirms U-value implementation is correct

---

## References

- ASHRAE Standard 140-2023
- Construction Implementation: `src/sim/construction.rs`
- ASHRAE 140 Cases: `src/validation/ashrae_140_cases.rs`
- Issue #273 Investigation Report
- ASHRAE 140 Progress Tracker: `ASHRAE_140_PROGRESS.md`

---

**Investigation completed**: 2026-02-26
**Investigator**: Fluxion AI Agent
**Status**: Ready for review
