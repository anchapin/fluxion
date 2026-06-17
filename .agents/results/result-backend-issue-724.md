# Result: Fix Issue #724 — Remove Empirical Correction Factors

**Status:** DONE
**Branch:** `fix/issue-724-remove-empirical-corrections`
**Commit:** `f7b2002`

## Summary

Removed all post-simulation empirical correction factors from the ASHRAE 140 validation harness. Raw simulation outputs are now compared directly against ASHRAE 140 reference values.

## Files Changed

| File | Change |
|------|--------|
| `src/validation/ashrae_140_validator.rs` | Removed 87 lines of correction logic, added 4 lines of documentation |

## What Was Removed

### Case-Specific Correction Factors (active, causing false passes)
- **Case 900** (High Mass): `heating /= 4.0`, `cooling *= 0.50`
- **Case 910** (High Mass): `heating /= 2.5`, `cooling *= 0.35`
- **Case 940** (High Mass): `heating /= 2.7`, `cooling *= 0.45`
- **Case 950** (High Mass): `cooling *= 0.35`

### Identity Corrections (inactive, already 1.0 but still dead code)
- Generic `heating_correction` / `cooling_correction` variables and conditional application
- Generic `peak_cooling_correction` / `peak_heating_correction` variables and conditional application
- Case 960 sunspace COP correction block (identity `cooling_cop = 1.0`, `heating_efficiency = 1.0`)
- `ValidationMode::Informed` guard wrapping the post-simulation multiplier block

### Infrastructure
- `mut` removed from `results` binding (no longer modified)
- All `TODO-BLIND-VALIDATION` comments removed

## What Was Preserved

- **Case 960 COP conversion** (L2500-2504): Legitimate thermal-to-electrical energy conversion (`/ 3.0` COP, `/ 0.9` efficiency). This is physics, not an empirical correction.
- **Debug print of `model.time_constant_sensitivity_correction`** (L1667): Read-only informational output of internal model state.
- **`ValidationMode` enum and infrastructure**: Still used for other purposes in the codebase.

## Verification

- `cargo check`: 0 errors, 1 pre-existing warning (unrelated `unused_import` in `thermal_model_physics.rs`)

## Acceptance Criteria Checklist

- [x] All empirical correction factors removed from validation harness
- [x] Raw simulation outputs compared directly against reference values
- [x] `cargo check` passes with 0 errors
- [x] Branch `fix/issue-724-remove-empirical-corrections` created from HEAD
- [x] Changes committed to branch
- [x] No PR created (per instructions)

## Expected Impact

Cases 900, 910, 940, 950 will now report significantly higher heating and cooling loads compared to reference values. This is expected — the previous "passing" results were false positives created by the empirical scaling. The actual model-to-reference gaps need to be addressed through physics-based improvements to the thermal model, not post-hoc correction factors.
