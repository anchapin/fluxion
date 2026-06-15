# T3.6: Remove Empirical Corrections from Validation Harness

**Status**: COMPLETE
**Issues**: #724, #739
**Date**: 2026-05-16

## Summary

Removed all empirical correction factors from the validation harness. The harness now tests pure physics model predictions against ASHRAE 140 reference values with no post-simulation adjustments.

## Files Changed

| File | Change |
|------|--------|
| `src/validation/ashrae_140_validator.rs` | Removed 5 correction blocks, fixed 4 misleading comments |
| `src/validation/thermal_mass.rs` | Removed `calculate_thermal_mass_correction()` function, correction factor fields, and related tests |

## Corrections Found and Removed

### 1. SESSION 81 TDD Empirical Correction Factors (ashrae_140_validator.rs:1216-1265)
- **What**: `heating_correction`, `cooling_correction`, `peak_cooling_correction`, `peak_heating_correction` variables (all set to 1.0 but with infrastructure to apply non-trivial multipliers)
- **Why added**: To compensate for model formulation gaps during development
- **Removed**: Entire block including the Case 960 stub (`cooling_cop = 1.0`, `heating_efficiency = 1.0`)

### 2. Session 91/93 Post-Simulation Multipliers (ashrae_140_validator.rs:1267-1305)
- **What**: Case-specific multipliers applied in `Informed` mode only:
  - Case 900: `heating /= 4.0`, `cooling *= 0.50`
  - Case 910: `heating /= 2.5`, `cooling *= 0.35`
  - Case 940: `heating /= 2.7`, `cooling *= 0.45`
  - Case 950: `cooling *= 0.35`
- **Why added**: "The simplified 5R1C thermal network needs empirical corrections to match ASHRAE 140 reference values"
- **Removed**: Entire `if self.validation_mode == ValidationMode::Informed` block with all case multipliers

### 3. Thermal Mass Correction Factor (thermal_mass.rs)
- **What**: `calculate_thermal_mass_correction()` function producing a `sqrt(1/cap_ratio)` factor clamped to [0.2, 1.0], stored in `low_mass_correction_factor` and `high_mass_correction_factor` fields
- **Why added**: Empirical formula to predict thermal buffering effects
- **Removed**: Function, struct fields, all calculation/validation code, and 8 tests that tested the correction factor itself

### 4. Debug Print Referencing Correction (ashrae_140_validator.rs:1780-1783)
- **What**: `println!("DEBUG Case 600: correction_factor={}", model.time_constant_sensitivity_correction)`
- **Removed**: Debug line referencing `time_constant_sensitivity_correction`

### 5. Misleading Comments (ashrae_140_validator.rs)
- Line 945: "h_corr correction applied" -> "physics-only, no empirical corrections"
- Line 2680: "applies proper calibration and correction factors" -> "physics-only, no empirical corrections"
- ValidationMode enum docs: Removed references to "corrections and calibrated values"

## Corrections Kept (Legitimate Physics)

| Item | Location | Reason |
|------|----------|--------|
| Case 960 COP/efficiency conversion (COP=3.0, η=0.9) | ashrae_140_validator.rs:2694-2698 | ASHRAE 140 specified equipment parameters — physics-based unit conversion from thermal to electrical energy |
| kWh→Joules conversion (3.6e6) | Multiple files | Physical constant |
| Air properties (ρ=1.2, cp=1005.0) | thermal_mass.rs:93 | Physical constants |
| Stefan-Boltzmann constant | report.rs | Physical constant |
| BH correction in statistical.rs | statistical.rs | Benjamini-Hochberg procedure — legitimate statistical method, not physics correction |
| `time_constant_sensitivity_correction` field in ThermalModel | sim/thermal_model_data.rs | Already initialized to 1.0 (no-op); field retained for struct compatibility but never applied |
| `ab_testing.rs` mock multipliers | ab_testing.rs:390-396 | Test framework stubs returning mock data, not applied to simulation results |
| `adaptive_calibration.rs` corrections | adaptive_calibration.rs | Training/calibration module, not part of validation harness |

## Test Results

```
lib tests (validation): 659 passed, 0 failed
lib tests (thermal_mass): 63 passed, 0 failed
cargo check: compiles with 0 warnings
```

Pre-existing failures in `thermal_mass_coupling_tests` (6 failures related to tau/capacitance calculation assertions) are unchanged — not caused by this task.

## Acceptance Criteria Checklist

- [x] No correction factors in validation harness
- [x] No "fudge factors" or empirical adjustments applied to simulation results
- [x] Clean physics-only model used for all validation comparisons
- [x] Post-simulation multipliers for cases 900/910/940/950 removed
- [x] `calculate_thermal_mass_correction()` function removed
- [x] Correction factor fields removed from `ThermalMassValidationResult`
- [x] Legitimate physics constants and ASHRAE-specified values preserved
- [x] Case 960 COP conversion preserved (ASHRAE 140 specified, not empirical)
- [x] All validation tests pass
- [x] No compiler warnings
