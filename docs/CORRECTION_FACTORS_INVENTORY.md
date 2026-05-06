# ASHRAE 140 Correction Factors Inventory

**Purpose**: Catalog all correction and calibration infrastructure for blind validation preparation.
**Context**: ASHRAE 140 validation is currently "informed" — case IDs are known, corrections applied post-simulation, and benchmark ranges are calibrated for 5R1C. This document enables true "blind" validation where corrections are removed.

**Plan**: [A-01](https://github.com/anchapin/fluxion/issues/662) — Phase A.1 of ASHRAE 140 Blind Validation Plan v1.3

---

## 1. Post-Simulation Multipliers

### 1.1 Case 900 — High Mass Heavy Adjustment

| Property | Value |
|----------|-------|
| **Location** | `src/validation/ashrae_140_validator.rs:1129-1132` |
| **Type** | Post-simulation multiplier |
| **Current Values** | `annual_heating_mwh /= 4.0`, `annual_cooling_mwh *= 0.50` |
| **Affected Metric** | Annual heating and cooling energy |
| **Effect if Removed** | Heating would be ~4x higher, cooling ~2x higher than reference |
| **Removal Method** | Delete lines 1129-1132 (the `if partial.case_id == "900"` block) |
| **Verification** | Compare uncalibrated simulation to ASHRAE 140 reference ranges |

### 1.2 Case 910 — High Mass Medium Adjustment

| Property | Value |
|----------|-------|
| **Location** | `src/validation/ashrae_140_validator.rs:1134-1137` |
| **Type** | Post-simulation multiplier |
| **Current Values** | `annual_heating_mwh /= 2.5`, `annual_cooling_mwh *= 0.35` |
| **Affected Metric** | Annual heating and cooling energy |
| **Effect if Removed** | Heating would be ~2.5x higher, cooling ~2.86x higher than reference |
| **Removal Method** | Delete lines 1134-1137 (the `if partial.case_id == "910"` block) |
| **Verification** | Compare uncalibrated simulation to ASHRAE 140 reference ranges |

### 1.3 Case 940 — High Mass Medium Adjustment

| Property | Value |
|----------|-------|
| **Location** | `src/validation/ashrae_140_validator.rs:1139-1142` |
| **Type** | Post-simulation multiplier |
| **Current Values** | `annual_heating_mwh /= 2.7`, `annual_cooling_mwh *= 0.45` |
| **Affected Metric** | Annual heating and cooling energy |
| **Effect if Removed** | Heating would be ~2.7x higher, cooling ~2.22x higher than reference |
| **Removal Method** | Delete lines 1139-1142 (the `if partial.case_id == "940"` block) |
| **Verification** | Compare uncalibrated simulation to ASHRAE 140 reference ranges |

### 1.4 Case 950 — High Mass Cooling Only

| Property | Value |
|----------|-------|
| **Location** | `src/validation/ashrae_140_validator.rs:1144-1146` |
| **Type** | Post-simulation multiplier |
| **Current Values** | `annual_cooling_mwh *= 0.35` |
| **Affected Metric** | Annual cooling energy only |
| **Effect if Removed** | Cooling would be ~2.86x higher than reference |
| **Removal Method** | Delete lines 1144-1146 (the `if partial.case_id == "950"` block) |
| **Verification** | Compare uncalibrated simulation to ASHRAE 140 reference ranges |

---

## 2. 6R2C Model Correction Constants

### 2.1 Time Constant Sensitivity Correction

| Property | Value |
|----------|-------|
| **Location** | `src/sim/thermal_model_core.rs:326-331` |
| **Type** | Calibration constant |
| **Current Value** | `time_constant_sensitivity_correction_6r2c = 5.2` |
| **Affected Metric** | Thermal time constant for 6R2C model |
| **Effect if Removed** | Model would use τ = Cm / (h_tr_ms + h_tr_em) directly, potentially under-representing thermal mass |
| **Removal Method** | Set `time_constant_sensitivity_correction_6r2c = 1.0` for uncorrected physics |
| **Verification** | Compare simulation τ to measured ASHRAE 140 response curves |

### 2.2 Cooling Sensitivity Correction

| Property | Value |
|----------|-------|
| **Location** | `src/sim/thermal_model_core.rs:330-331` |
| **Type** | Calibration constant |
| **Current Value** | `cooling_sensitivity_correction_6r2c = 1.74` |
| **Affected Metric** | Cooling response sensitivity |
| **Effect if Removed** | Model cooling response would be dampened by factor of 1.74 |
| **Removal Method** | Set `cooling_sensitivity_correction_6r2c = 1.0` for uncorrected physics |
| **Verification** | Compare free-floating temperature profiles to ASHRAE 140 reference |

---

## 3. Case-Specific 6R2C Configuration

### 3.1 High Mass Cases (900 Series) Envelope Fraction

| Property | Value |
|----------|-------|
| **Location** | `src/sim/thermal_model_core.rs:1211-1222` |
| **Type** | Model configuration |
| **Current Values** | `configure_6r2c_model(0.75, 100.0, None)` — 75% envelope, 25% internal mass |
| **Affected Metric** | All 900 series case results |
| **Effect if Removed** | 6R2C model would use default mass distribution instead of case-specific calibration |
| **Removal Method** | Conditional block at lines 1217-1222 applies 75% envelope fraction only for cases starting with "9" (except "960"). Remove or make conditional on `ValidationMode::Informed` |
| **Verification** | Run 900 series with default configuration and compare to reference |

**Code Context**:
```rust
// SESSION 23 FIX: Enable 6R2C model for proper envelope/internal mass separation
// SESSION 76 FIX: Solar gain distribution was REVERSED - fixed to proper 60%/40% split
if spec.case_id.starts_with("9") && spec.case_id != "960" {
    // For high-mass buildings: 75% envelope mass, 25% internal mass
    // Conductance between masses: 100 W/K (typical for concrete construction)
    model.configure_6r2c_model(0.75, 100.0, None);
}
```

---

## 4. Thermal Mass Correction Function

### 4.1 Thermal Mass Correction Factor Calculation

| Property | Value |
|----------|-------|
| **Location** | `src/validation/thermal_mass.rs:50-66` |
| **Type** | Correction formula |
| **Current Values** | `calculate_thermal_mass_correction()` with reference capacitance 2.4e6 J/K |
| **Affected Metric** | All high-mass case results via correction factor |
| **Effect if Removed** | High-mass correction factor would default to 1.0 (no correction) |
| **Removal Method** | Function exists but is not actively used in validation pipeline — verify by checking `validate_thermal_mass()` calls |
| **Verification** | Compare output of `validate_thermal_mass()` with and without correction applied |

**Note**: This function appears to be used for validation testing rather than in the main validation pipeline. The correction factor calculation uses `1.0 / cap_ratio.sqrt()` clamped to [0.2, 1.0].

---

## 5. Adaptive Calibration System

### 5.1 Hourly Recalibration Loop

| Property | Value |
|----------|-------|
| **Location** | `src/validation/adaptive_calibration.rs` (entire file) |
| **Type** | Continuous calibration |
| **Current Behavior** | Multi-stage hourly recalibration with trigger-based updates |
| **Affected Metrics** | All metrics if calibration is active during simulation |
| **Effect if Removed** | System would use default `CalibrationState` values without adaptation |
| **Removal Method** | Do not call `AdaptiveHourlyCalibrator::process_observation()` during validation runs |
| **Verification** | Run validation with and without calibration triggers active |

**Key Components**:
- `SmartMeterPatternAnalyzer` (lines 105-210) — classifies bias patterns
- `TriggerDetector` (lines 213-301) — detects recalibration triggers
- `AdaptiveHourlyCalibrator` (lines 315-509) — performs 4-step calibration loop
- `BiasPattern` enum (lines 26-35) — UniversalBias, SeasonalBias, MixedBias, NoBias

**Default CalibrationState** (`adaptive_calibration.rs:65-76`):
```rust
thermal_conductivity: 0.16,
specific_heat: 840.0,
density: 2400.0,
infiltration_rate: 0.5,
internal_gain_multiplier: 1.0,
solar_gain_multiplier: 1.0,
```

---

## 6. Benchmark Calibrated Ranges

### 6.1 5R1C Calibrated Benchmark Ranges

| Property | Value |
|----------|-------|
| **Location** | `src/validation/benchmark.rs:108-110` |
| **Type** | Adjusted reference ranges |
| **Current Behavior** | Comment states "These ranges are calibrated for the 5R1C thermal network model" |
| **Affected Metric** | All ASHRAE 140 validation pass/fail determinations |
| **Effect if Removed** | Validation would use actual ASHRAE 140 reference values instead of calibrated ones |
| **Removal Method** | Update benchmark data to use ASHRAE 140-2023 values directly from standard |
| **Verification** | Compare benchmark.rs values to ASHRAE 140-2023 Table values |

**Example Comment** (lines 107-111):
```rust
// Case 600 - Baseline (Low Mass)
// Note: These ranges are calibrated for the 5R1C thermal network model
// The ASHRAE 140 reference values are based on detailed hourly simulation
// Our model uses simplified 5R1C thermal network with different solar distribution
```

---

## 7. CTF Primary Surface Temperature Coupling

### 7.1 High-Mass Free-Floating Cases (900FF, 950FF)

| Property | Value |
|----------|-------|
| **Location** | `src/sim/thermal_model_core.rs:1228-1239` |
| **Type** | Model enhancement for specific cases |
| **Current Behavior** | Enables CTF solver with iterative zone coupling for high-mass free-floating cases |
| **Affected Cases** | 900FF, 950FF only |
| **Effect if Removed** | Free-floating temperature predictions would be less accurate for these cases |
| **Removal Method** | Remove conditional block or guard with `ValidationMode::Informed` check |
| **Verification** | Compare 900FF and 950FF free-floating results with and without CTF |

**Code Context**:
```rust
// SESSION 89: CTF-primary surface temperature coupling for high-mass free-floating cases.
// The 6R2C lumped model's τ ≈ 26h under-represents thermal mass (concrete h₁₂ = 771 W/K
// is too high). The CTF solver captures multi-layer conduction dynamics correctly (τ ≈ 120-200h).
if spec.case_id == "900FF" || spec.case_id == "950FF" {
    use crate::physics::ctf_coefficients::CTFMaterial;
    // Wall layers match Materials::high_mass_wall() from construction.rs:
    let wall_layers = vec![
        CTFMaterial::new("Concrete Block", 0.100, 0.51, 1400.0, 1000.0),
        CTFMaterial::new("Foam Insulation", 0.0615, 0.04, 10.0, 1400.0),
        CTFMaterial::new("Wood Siding", 0.009, 0.14, 500.0, 1300.0),
    ];
    model.enable_ctf(&wall_layers, 3600.0, 50);
    model.ctf_primary = true;
}
```

---

## Summary Table

| Correction Type | Location | Cases Affected | Removal Complexity |
|-----------------|----------|----------------|-------------------|
| Post-simulation multipliers | `ashrae_140_validator.rs:1129-1146` | 900, 910, 940, 950 | Low (delete blocks) |
| 6R2C time constant | `thermal_model_core.rs:330` | All 900 series | Medium (set to 1.0) |
| 6R2C cooling sensitivity | `thermal_model_core.rs:331` | All 900 series | Medium (set to 1.0) |
| 6R2C envelope config | `thermal_model_core.rs:1211-1222` | 900 series (not 960) | Medium (guard with mode) |
| Thermal mass correction | `thermal_mass.rs:50-66` | Validation only | N/A (not in pipeline) |
| Adaptive calibration | `adaptive_calibration.rs` | All if active | High (requires mode check) |
| Benchmark ranges | `benchmark.rs:108-110` | All validation | High (requires new ref data) |
| CTF coupling | `thermal_model_core.rs:1228-1239` | 900FF, 950FF | Medium (guard with mode) |

---

## Implementation Notes

### TODO-BLIND-VALIDATION Markers

Add markers at each location for easy grep-based inventory:
- `ashrae_140_validator.rs:1129-1146` — Post-simulation multipliers
- `thermal_model_core.rs:326-331` — 6R2C correction constants
- `thermal_model_core.rs:1211-1222` — 6R2C case-specific config
- `thermal_model_core.rs:1228-1239` — CTF coupling for free-floating
- `thermal_mass.rs:50-66` — Thermal mass correction function
- `benchmark.rs:108-110` — Calibrated benchmark ranges

### Verification Command

```bash
grep -n "TODO-BLIND-VALIDATION" src/validation/ashrae_140_validator.rs src/sim/thermal_model_core.rs src/validation/thermal_mass.rs src/validation/benchmark.rs | wc -l
# Expected: ≥ 15 occurrences
```

### ValidationMode Design

Implement `ValidationMode::Blind` in `Ashrae140Validator`:
- Takes only `CaseSpec` (no case ID string exposed to validation logic)
- Does not apply post-simulation multipliers
- Uses default thermal model configuration (no 6R2C special-casing)
- Uses raw benchmark data (not calibrated ranges)

---

## References

- ASHRAE Standard 140-2023
- EnergyPlus BESTEST reports
- Issue #662: ASHRAE 140 Blind Validation Plan v1.3