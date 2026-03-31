# Session 80: Remove Empirical Corrections from Validator

**Date:** 2026-03-31
**Objective:** Remove all empirical correction factors from `src/validation/ashrae_140_validator.rs` to enable fully physics-based validation

## Changes Made

### File Modified: `src/validation/ashrae_140_validator.rs`

Removed the following empirical correction factors from the `validate_analytical_engine()` function:

#### 1. SESSION 78 Corrections (Removed)
These were large correction factors for systemic heating overprediction:
- **Case 900**: Heating divided by 26.8×, Cooling multiplied by 1.717×
- **Case 910**: Heating divided by 23.6×
- **Case 940**: Heating divided by 31.5×
- **Case 950**: Cooling multiplied by 0.35× (night ventilation correction)

#### 2. SESSION 70 Corrections (Removed)
Sunspace COP correction for Case 960:
- **Case 960**: `cooling_cop = 2.2`, `heating_efficiency = 0.95`

#### 3. SESSION 69 Peak Corrections (Removed)
Peak load tuning factors:
- **Case 920/920FF**: Peak cooling × 0.65
- **Case 930/930FF**: Peak cooling × 0.65, Peak heating × 1.10
- **Case 940/940FF**: Peak cooling × 0.70
- **Case 950/950FF**: Peak cooling × 0.40

### Code Change Summary

**Before:**
```rust
for partial in partials {
    if let (Some(data), Some(mut results)) = (partial.data, partial.results) {
        // === SESSION 78: Restore Empirical Correction Factors ===
        // ... 60+ lines of empirical corrections ...

        if partial.case_id == "960" {
            let cooling_cop = 2.2;
            let heating_efficiency = 0.95;
            results.annual_heating_mwh /= heating_efficiency;
            results.annual_cooling_mwh /= cooling_cop;
        }
        // ... more corrections ...
    }
}
```

**After:**
```rust
for partial in partials {
    if let (Some(data), Some(results)) = (partial.data, partial.results) {
        // Physics-based validation - no empirical corrections applied
        // All results are raw model output for transparent validation
    }
}
```

## Impact

### Validation Transparency
- All validation results now reflect the **raw physics model output**
- No hidden correction factors masking model inaccuracies
- Transparent comparison against ASHRAE 140 reference values

### Expected Results
With empirical corrections removed, validation results will show the true state of the physics model:
- Cases that were passing due to corrections may now fail
- The magnitude of errors will reveal which physics components need improvement
- This enables targeted physics-based fixes rather than empirical tuning

### Next Steps
1. Run full validation suite to see raw model performance
2. Identify largest discrepancies between model and reference
3. Implement physics-based fixes for root causes:
   - CTF solver heat flux calculation
   - Thermal mass coupling conductances
   - Solar gain distribution to thermal mass
   - Night ventilation modeling
   - Multi-node CTF zone coupling

## Verification
- `cargo check` passes successfully (exit code 0)
- No new warnings introduced by this change
