---
phase: A-baseline-stripping
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - docs/CORRECTION_FACTORS_INVENTORY.md
  - src/validation/ashrae_140_validator.rs
  - src/sim/thermal_model_core.rs
  - src/validation/thermal_mass.rs
  - src/validation/adaptive_calibration.rs
  - src/validation/benchmark.rs
autonomous: true
requirements:
  - BASELINE-01
  - BASELINE-02

must_haves:
  truths:
    - "Every correction factor location is documented with its current value and intended removal method"
    - "The validator can run in 'correction-free mode' for baseline measurement"
  artifacts:
    - path: "docs/CORRECTION_FACTORS_INVENTORY.md"
      provides: "Complete catalog of all correction infrastructure"
      min_lines: 100
    - path: "src/validation/ashrae_140_validator.rs"
      provides: "Commented corrections with removal TODO markers"
      contains: "CorrectionFactor"
  key_links:
    - from: "src/validation/ashrae_140_validator.rs"
      to: "docs/CORRECTION_FACTORS_INVENTORY.md"
      via: "All corrections documented inline"
---

<objective>
Catalog ALL correction and calibration infrastructure in the codebase. Document every location where correction factors, empirical adjustments, or case-specific tuning exists. This establishes the complete inventory needed to strip all corrections for true physics-only validation.
</objective>

<execution_context>
@/home/alex/.agents/get-shit-done/workflows/execute-plan.md
@/home/alex/.agents/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/validation/ashrae_140_validator.rs (lines 1074-1146 for correction factors)
@src/sim/thermal_model_core.rs (lines 326-331 for correction constants)
@src/validation/thermal_mass.rs (thermal mass corrections)
@src/validation/adaptive_calibration.rs (hourly recalibration)
@src/validation/benchmark.rs (calibrated ranges)
@docs/ASHRAE140_RESULTS.md (current pass/fail state)

From deep dive:
- Post-simulation case-specific corrections at ashrae_140_validator.rs:1129-1146
- 6R2C correction constants (5.2, 1.74) stored in ThermalModelData but not actively applied
- Case-specific 6R2C configuration for 900 series at thermal_model_core.rs:1211-1222
- Benchmark ranges "calibrated for 5R1C" at benchmark.rs:108-110
- Thermal mass correction factors in thermal_mass.rs
- Adaptive calibration system in adaptive_calibration.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Document all correction factor locations</name>
  <files>docs/CORRECTION_FACTORS_INVENTORY.md</files>
  <action>
Create docs/CORRECTION_FACTORS_INVENTORY.md with complete catalog. For each correction, document:
1. **Location** — exact file and line number
2. **Type** — post-simulation multiplier, calibration constant, configuration hint, benchmark adjustment
3. **Current value** — the actual numerical value
4. **What it affects** — which cases, which metrics
5. **Effect if removed** — what happens to pass rate (known or hypothesized)
6. **Removal method** — specific code change needed
7. **Verification** — how to confirm removal didn't break something else

Organize by correction type:

### A. Post-Simulation Case-Specific Corrections
Location: ashrae_140_validator.rs lines 1129-1146
Current values:
- Case 900: heating ÷ 4.0, cooling × 0.50
- Case 910: heating ÷ 2.5, cooling × 0.35
- Case 940: heating ÷ 2.7, cooling × 0.45
- Case 950: cooling × 0.35

### B. 6R2C Calibration Constants
Location: thermal_model_core.rs lines 326-331
- time_constant_sensitivity_correction_6r2c = 5.2
- cooling_sensitivity_correction_6r2c = 1.74

### C. Case-Specific Model Configuration
Location: thermal_model_core.rs lines 1211-1222
- 900-series (except 960): configure_6r2c_model(0.75, 100.0, None)
- h_tr_ms defaults to 40% of ISO 13790 value for 6R2C

### D. Thermal Mass Correction Factors
Location: thermal_mass.rs
- low_mass_correction_factor
- high_mass_correction_factor
- per-case computation based on mass class

### E. Adaptive Calibration System
Location: adaptive_calibration.rs
- Multi-stage hourly recalibration
- Bias pattern detection
- Parameter adjustment mechanisms

### F. Benchmark Range Adjustment
Location: benchmark.rs lines 108-110
- Reference ranges "calibrated for 5R1C model"
- Different from true ASHRAE 140 reference values
  </action>
  <verify>
    File exists with 6 sections (A-F), each with complete Location/Type/Value/Affects/Removal/Verification fields
  </verify>
  <done>Complete correction inventory documented with removal methods verified</done>
</task>

<task type="auto">
  <name>Task 2: Mark corrections in source code with removal markers</name>
  <files>src/validation/ashrae_140_validator.rs, src/sim/thermal_model_core.rs, src/validation/thermal_mass.rs</files>
  <action>
For each correction location in the inventory, add inline comments in the source code:

```rust
// TODO-BLIND-VALIDATION: Remove this correction factor
// See docs/CORRECTION_FACTORS_INVENTORY.md Section A.1
// Effect if removed: Case 900 heating will increase ~4x
// Ticket: Track in issue tracker
```

Add struct field comments in ThermalModelData for the 6R2C corrections at thermal_model_core.rs:326-331:

```rust
/// Correction factor for 6R2C time constant
/// TODO-BLIND-VALIDATION: Remove, replace with physics-based derivation
/// Current value (5.2) is empirically calibrated, not physics-derived
time_constant_sensitivity_correction_6r2c: f64,

/// Correction factor for 6R2C cooling sensitivity
/// TODO-BLIND-VALIDATION: Remove, replace with physics-based derivation
/// Current value (1.74) is empirically calibrated, not physics-derived
cooling_sensitivity_correction_6r2c: f64,
```

In thermal_mass.rs, add markers around the correction factor computations.

DO NOT remove any code yet — just document and mark.
  </action>
  <verify>
    grep -n "TODO-BLIND-VALIDATION" src/validation/ashrae_140_validator.rs src/sim/thermal_model_core.rs src/validation/thermal_mass.rs | wc -l
    Expected: ≥ 15 occurrences (multiple per correction type)
  </verify>
  <done>All corrections marked with removal TODO markers in source code</done>
</task>

<task type="auto">
  <name>Task 3: Create correction-free configuration mode</name>
  <files>src/validation/ashrae_140_validator.rs, src/sim/thermal_model_core.rs</files>
  <action>
Add a ValidationMode enum and configuration to ashrae_140_validator.rs:

```rust
pub enum ValidationMode {
    Informed  // Current mode: uses case ID, applies corrections
    Blind     // Physics-only: no case ID, no corrections
}

pub struct ValidatorConfig {
    pub mode: ValidationMode,
    // ... other config
}

impl Ashrae140Validator {
    pub fn new_blind() -> Self {
        Self::new_with_config(ValidatorConfig {
            mode: ValidationMode::Blind,
            ..Default::default()
        })
    }
}
```

In blind mode, the validator must:
1. NOT look up case ID before simulation
2. NOT apply any post-simulation multipliers
3. Use default thermal model configuration (no case-type hints)
4. Compare against true reference values (not calibrated ranges)

Add method:
```rust
pub fn validate_blind(&self, case_spec: &CaseSpec) -> ValidationResult
```

This method takes only the case specification — no case ID string.
  </action>
  <verify>
    cargo check --lib
    grep -n "ValidationMode" src/validation/ashrae_140_validator.rs
    grep -n "validate_blind" src/validation/ashrae_140_validator.rs
  </verify>
  <done>Blind validation mode implemented, takes only CaseSpec (no case ID)</done>
</task>

</tasks>

<verification>
Run: cargo test --test ashrae_140_validation 2>&1 | tail -20

Verify:
- All existing tests still pass
- New validate_blind method is callable
- TODO-BLIND-VALIDATION markers are present in source
</verification>

<success_criteria>
- docs/CORRECTION_FACTORS_INVENTORY.md exists with complete catalog
- All correction locations marked with TODO-BLIND-VALIDATION in source
- ValidationMode::Blind implemented and callable
- cargo check passes
</success_criteria>

<output>
After completion, create `.planning/phases/A-baseline-stripping/A-01-SUMMARY.md`
</output>