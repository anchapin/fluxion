# T3.7: Remove Placeholder Conductances from ashrae_140_cases.rs

**Status**: COMPLETE
**Issue**: #731
**Date**: 2026-05-16

## Summary

Replaced all placeholder thermal conductance values in `ConductanceReferences::case600_reference_conductances()` with physics-derived values calculated from the actual construction materials, film coefficients, and building geometry specified in ASHRAE 140.

## Files Changed

1. `src/validation/ashrae_140_cases.rs` — Replaced 4 placeholder values + 1 slightly inaccurate value with physics-derived calculations
2. `tests/test_conductance_calculations.rs` — Updated `h_tr_ms` range check to accommodate correct physics value

## Placeholders Found & Replaced

All values in `case600_reference_conductances()` (lines 1684-1712):

| Conductance | Placeholder | Physics Value | Derivation |
|-------------|-------------|---------------|------------|
| **h_tr_em** | 123.45 | **56.72** W/K | Σ(U_opaque × A): wall(0.5119×63.6=32.56) + roof(0.3198×48.0=15.35) + floor(0.1837×48.0=8.82) |
| **h_tr_w** | 25.20 | **25.20** W/K | Already correct: U=2.10 W/m²K × A=12.0 m² |
| **h_tr_ms** | 89.01 | **1092.00** W/K | ISO 13790 Annex C eq C.3: h_ms_coeff=9.1 W/m²K × A_m=2.5×48.0=120 m² |
| **h_tr_is** | 234.56 | **1343.60** W/K | Σ(h_int×A): walls(7.69×75.6=581.36) + ceiling(10.0×48.0=480.0) + floor(5.88×48.0=282.24) |
| **h_ve** | 21.72 | **22.16** W/K | ρ×cp×(ACH/3600)×V = 1.225×1005×(0.5/3600)×129.6 |

### Physics Derivation Details

**Building geometry** (Case 600): 8.0m × 6.0m × 2.7m
- Floor area = 48.0 m², Volume = 129.6 m³
- Gross wall = 75.60 m², Net opaque wall = 63.60 m² (12 m² south window)

**Film coefficients** (ASHRAE 140-2023 v2023):
- Interior wall: 7.69 W/m²K, Ceiling: 10.0 W/m²K, Floor: 5.88 W/m²K
- Exterior: 29.3 W/m²K, Ground coupling: R=0.17 m²K/W

**Construction U-values** (from material properties):
- Wall (plasterboard 12mm + fiberglass 66mm + wood siding 9mm): U = 0.5119 W/m²K
- Roof (plasterboard 10mm + fiberglass 111.8mm + roof deck 19mm): U = 0.3198 W/m²K
- Floor (timber 25mm + fiberglass 197mm): U = 0.1837 W/m²K (with ground coupling)

## Test Results

| Test | Status |
|------|--------|
| `test_ashrae_140_case_600_reference_values` | PASS |
| `test_overall_conductance_correctness` | PASS (updated range) |
| `test_conductance_units` | PASS |
| `test_h_tr_em_calculation` | PASS |
| `test_h_tr_is_calculation` | PASS |
| `test_h_tr_ms_calculation` | PASS |
| `test_h_tr_w_calculation` | PASS |
| `test_thermal_bridge_effects` | PASS |
| `test_internal_gain_modeling` | PASS |
| `test_layer_by_layer_r_value_calculation` | PASS |

**Pre-existing failures** (NOT caused by this change):
- `test_ashrae_film_coefficient_application` — expects EXTERIOR_FILM_COEFF_DEFAULT=25.0 but code uses 29.3 (v2023)
- `test_window_property_validation` — expects u_value=3.0 but Case 600 has 2.1 (double clear glass)

## Acceptance Criteria Checklist

- [x] All thermal conductances in ashrae_140_cases.rs are physics-calculated, not placeholder values
- [x] Each value has documented derivation (material properties, areas, film coefficients)
- [x] h_tr_w (window) was already correct — confirmed and preserved
- [x] h_tr_em, h_tr_ms, h_tr_is replaced with physics-derived values
- [x] h_ve corrected from 21.72 to 22.16 W/K
- [x] No hardcoded "123.45", "89.01", "234.56" magic numbers remain
- [x] All related tests pass
- [x] Crate compiles clean (`cargo check` passes)

## Notes for Other Agents

- The `ConductanceReferences` struct is only used in `test_conductance_calculations.rs` for reference validation — it does NOT feed into the actual thermal model computation
- The thermal model (`thermal_model_core.rs`) computes its own h_tr_em, h_tr_ms, h_tr_is from first principles per-zone during simulation
- These reference values serve as ground-truth for validating that the simulation's computed conductances match expectations
