---
phase: B-physics-fixes
plan: B-02
type: execute
wave: 2
depends_on: ["B-01"]
files_modified:
  - tests/thermal_mass_time_constant_validation.rs
  - src/sim/thermal_model_core.rs
  - src/sim/construction.rs
autonomous: true
requirements:
  - PHYSICS-02

must_haves:
  truths:
    - "Thermal time constant τ matches ISO 13790 calculated values within ±5%"
    - "h_tr_ms (conductance from mass to interior surface) is correct per construction layers"
  artifacts:
    - path: "tests/thermal_mass_time_constant_validation.rs"
      provides: "Time constant comparison test"
    - path: ".planning/baseline/THERMAL_MASS_ANALYSIS.md"
      provides: "τ discrepancy analysis"
---

<objective>
Fix thermal mass time constant (τ = Cm/(h_tr_ms + h_tr_em)) calculation to match ISO 13790 values. The 900 series cases show annual energy failures that suggest τ is wrong — either Cm is wrong, or h_tr_ms is wrong.
</objective>

<context>
From deep dive analysis:
- τ calculation at thermal_model_core.rs lines 963-966
- h_tr_ms represents conductance from thermal mass node to interior surface
- Current h_tr_ms uses actual layer properties but may have errors
- 6R2C model uses 40% of ISO 13790 h_tr_ms value (empirically calibrated)
- Cm (thermal capacitance) calculated from ISO 13790 Annex C effective capacitance per area

The 6R2C correction factors (5.2, 1.74) are empirically derived. If τ is wrong, these factors paper over the error.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create time constant diagnostic test</name>
  <files>tests/thermal_mass_time_constant_validation.rs</files>
  <action>
Create tests/thermal_mass_time_constant_validation.rs that:

1. For each construction type (LowMass, HighMass):
   - Compute τ using ISO 13790 formula (from construction.rs iso_13790_effective_capacitance_per_area)
   - Compute τ using current Fluxion implementation
   - Compare the two

2. For each case (600, 900):
   - Log computed h_tr_ms, h_tr_em, Cm, τ
   - Compare τ against reference values if available

3. Output:
   - Per-construction-type τ comparison table
   - h_tr_ms breakdown (wall contribution, roof contribution, floor contribution)
   - Cm breakdown by layer

This test should reveal WHERE the time constant calculation diverges from ISO 13790.
  </action>
  <verify>
    cargo test --test thermal_mass_time_constant_validation -- --nocapture 2>&1 | head -80
  </verify>
  <done>Time constant diagnostic test exists and produces τ breakdown</done>
</task>

<task type="auto">
  <name>Task 2: Identify time constant discrepancy</name>
  <files>tests/thermal_mass_time_constant_validation.rs, src/sim/thermal_model_core.rs, src/sim/construction.rs</files>
  <action>
Run the diagnostic and identify root cause of τ discrepancy.

Check:
1. Cm calculation: Does iso_13790_effective_capacitance_per_area() use correct κ values from Table C.2?
2. h_tr_ms: Is the conductance calculation using correct R-values for each layer?
3. Half-insulation rule: Does the model correctly identify which layers contribute to thermal mass?
4. Surface area weighting: Are A_wall, A_roof, A_floor correctly computed and weighted?

For each discrepancy:
- Document current τ value
- Document expected τ value per ISO 13790
- Identify which calculation step causes the divergence

Output findings to .planning/baseline/THERMAL_MASS_ANALYSIS.md
  </action>
  <verify>
    File exists: .planning/baseline/THERMAL_MASS_ANALYSIS.md
  </verify>
  <done>Time constant discrepancies identified with root cause analysis</done>
</task>

<task type="auto">
  <name>Task 3: Implement time constant fix</name>
  <files>src/sim/construction.rs, src/sim/thermal_model_core.rs</files>
  <action>
Based on analysis, implement the fix:

### If Cm calculation is wrong:
Verify/fix iso_13790_effective_capacitance_per_area() to use correct Table C.2 κ values:
```rust
// κ values from ISO 13790 Table C.2:
// Heavy mass: κ ≈ 160+ kJ/m²K
// Low mass: κ ≈ 40-80 kJ/m²K
```

### If h_tr_ms is wrong:
Review and fix the h_tr_ms calculation at thermal_model_core.rs lines 757-834:
```rust
// h_tr_ms = A_wall / R_wall_layers where R comes from actual layer properties
// Ensure exterior-to-mass R-value uses correct layer identification
```

### If 6R2C configuration is wrong:
The 6R2C model uses 40% of ISO 13790 h_tr_ms — this is empirically calibrated.
Fix: Derive h_tr_ms for 6R2C from first principles, not calibration.

After fix:
1. Re-run thermal_mass_time_constant_validation
2. Confirm τ matches ISO 13790 within ±5%
3. Run blind validation: cargo test --test ashrae_140_blind_validation
4. 900 series annual energy should improve
  </action>
  <verify>
    cargo test --test thermal_mass_time_constant_validation 2>&1 | grep -E "(PASS|FAIL|τ|tau)"
    cargo test --test ashrae_140_blind_validation 2>&1 | grep -E "(900.*heating|900.*cooling)" | head -5
  </verify>
  <done>Time constant fix implemented, τ matches ISO 13790 within ±5%, 900 series annual energy improved</done>
</task>

</tasks>

<verification>
Run: cargo test --test thermal_mass_time_constant_validation -- --nocapture

Verify:
- τ breakdown produced for LowMass and HighMass constructions
- Error between ISO 13790 formula and implementation is identified
- Physics-based fix implemented (no correction factors)
</verification>

<success_criteria>
- Time constant diagnostic test exists and runs
- τ discrepancy root causes identified
- τ matches ISO 13790 within ±5% after fix
- 900 series annual energy within ±20% (improvement from baseline)
- Blind validation shows measurable improvement
</success_criteria>

<output>
After completion, create `.planning/phases/B-physics-fixes/B-02-SUMMARY.md`
</output>