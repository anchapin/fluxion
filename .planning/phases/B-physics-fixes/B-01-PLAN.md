---
phase: B-physics-fixes
plan: B-01
type: execute
wave: 1
depends_on: ["A-02"]
files_modified:
  - tests/solar_distribution_validation.rs
  - src/sim/solar.rs
  - src/sim/thermal_model_core.rs
autonomous: true
requirements:
  - PHYSICS-01

must_haves:
  truths:
    - "Solar gain profiles for Case 900 match EnergyPlus hourly data within ±10%"
    - "Beam/diffuse/ground-reflected split is correct per ISO 13790 Section 10"
  artifacts:
    - path: "tests/solar_distribution_validation.rs"
      provides: "Hourly solar gain comparison test"
    - path: "data/solar_reference_profiles/case_900_hourly.csv"
      provides: "EnergyPlus hourly solar gain reference data"
---

<objective>
Fix solar distribution to match reference programs (EnergyPlus/ESP-r) for heavy-mass cases. The 900 series cases show systematic cooling over-prediction (cooling × 0.35-0.50 corrections applied). This suggests solar gains are being distributed incorrectly between air and thermal mass nodes.
</objective>

<execution_context>
@/home/alex/.agents/get-shit-done/workflows/execute-plan.md
</execution_context>

<context>
From deep dive analysis:
- Solar gains calculated in src/sim/solar.rs lines 234-390
- ISO 13790 distribution at thermal_model_core.rs lines 1056-1069
- f_ms (mass surface fraction) determines solar-to-air vs solar-to-mass split
- Perez sky model used for diffuse calculation on tilted surfaces
- Ground reflectance configurable but may not match ASHRAE 140 assumptions

Key equations:
```rust
// Solar to air fraction per ISO 13790:
let solar_to_air_frac = 0.1 * (1.0 - f_ms) + f_ms;
let solar_to_mass_frac = 1.0 - solar_to_air_frac;

// Surface irradiance:
beam = dni * incidence_cosine(tilt, azimuth)
diffuse = PerezSkyModel::calculate_diffuse_tilted(...)
ground_reflected = ghi * ground_reflectance * (1 - surface_tilt.cos()) / 2.0
```

Hypothesis: For high-mass buildings (900 series), the f_ms value is causing too much solar to go to air (causing over-prediction of cooling load). OR the beam vs diffuse split is wrong (causing incorrect angle-of-incidence effects).
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create solar distribution diagnostic test</name>
  <files>tests/solar_distribution_validation.rs, data/solar_reference_profiles/</files>
  <action>
Create tests/solar_distribution_validation.rs that:

1. Runs Case 600 and Case 900 (representative low-mass and high-mass)
2. Captures hourly solar gain breakdown for each zone:
   - Total incident solar (W)
   - Beam component (W)
   - Diffuse component (W)
   - Ground-reflected component (W)
   - Solar gain to air (W)
   - Solar gain to mass (W)
3. Compares against reference hourly profiles (EnergyPlus output)

4. Outputs per-case report:
   - Hourly time series for key summer day (July 15)
   - Daily totals for comparison
   - Beam/diffuse ratio

Use the existing test infrastructure. Reference data should come from EnergyPlus validation output for Case 600 and Case 900.

If EnergyPlus reference data doesn't exist, create data/solar_reference_profiles/case_600_hourly.csv and case_900_hourly.csv with current Fluxion output (labeled as "Fluxion current" — will compare after fix).

DO NOT use the existing ashrae_140_validator for this test — create standalone test that captures intermediate solar values.
  </action>
  <verify>
    cargo test --test solar_distribution_validation -- --nocapture 2>&1 | head -100
  </verify>
  <done>Solar distribution diagnostic test exists and produces hourly breakdown</done>
</task>

<task type="auto">
  <name>Task 2: Identify solar distribution discrepancy</name>
  <files>tests/solar_distribution_validation.rs, src/sim/solar.rs, src/sim/thermal_model_core.rs</files>
  <action>
Run the diagnostic test and analyze the discrepancy between Fluxion and reference.

Compare:
1. Total daily solar radiation (kWh/day) — should match TMY data
2. Beam vs diffuse split — Perez model may differ from EnergyPlus sky model
3. Solar-to-air vs solar-to-mass distribution — f_ms calculation

For each discrepancy, identify the ROOT CAUSE:

### Possible issues to check:
1. Perez model parameters — does Fluxion use same Perez coefficients as EnergyPlus?
2. Ground reflectance — ASHRAE 140 may specify 0.2, Fluxion may use different default
3. Surface tilt effect on diffuse — ISO 13790 vs EnergyPlus implementation
4. Angular SHGC correction — ashrae_140_window_shgc_ratio() function behavior
5. f_ms (mass surface fraction) calculation — does it match ISO 13790 Table C.2?

For each issue:
- Document current behavior
- Document expected behavior per ISO 13790 / EnergyPlus
- Estimate impact on cooling load prediction

Output findings to .planning/baseline/SOLAR_DISTRIBUTION_ANALYSIS.md with:
- Per-case hourly comparison tables
- Root cause identification for each discrepancy
- Recommended fix approach
  </action>
  <verify>
    File exists: .planning/baseline/SOLAR_DISTRIBUTION_ANALYSIS.md
    Contains: Hourly comparison, root cause analysis, fix recommendations
  </verify>
  <done>Solar distribution discrepancies identified with root cause analysis</done>
</task>

<task type="auto">
  <name>Task 3: Implement solar distribution fix</name>
  <files>src/sim/solar.rs, src/sim/thermal_model_core.rs</files>
  <action>
Based on the analysis, implement the fix. Common fixes include:

### If Perez model parameters differ:
Update Perez coefficients to match EnergyPlus implementation.
Check: PerezSkyModel struct in src/sim/solar.rs

### If ground reflectance is wrong:
```rust
// In calculate_surface_irradiance():
// ASHRAE 140 specifies ground reflectance of 0.2 for standard cases
// Confirm this is being used, not a different default
let ground_reflectance = 0.2;  // Standard ASHRAE 140 value
```

### If f_ms calculation is wrong:
Check iso_13790_effective_capacitance_per_area() in construction.rs
Verify f_ms = A_m / (6 × A_f) per ISO 13790 Section C.2

### If beam/diffuse split is wrong:
Review incidence_cosine() function — angle calculation may have error
Review Perez diffuse tilt factor calculation

After implementing fix:
1. Re-run solar_distribution_validation.rs
2. Confirm improvement: 900 series cooling should be closer to reference
3. Run blind validation: cargo test --test ashrae_140_blind_validation
4. Document before/after comparison

DO NOT add any correction factors — fix must be physics-based.
  </action>
  <verify>
    cargo test --test solar_distribution_validation 2>&1 | grep -E "(PASS|FAIL|error)"
    cargo test --test ashrae_140_blind_validation 2>&1 | grep -E "(900.*cooling|Case 900)" | head -10
  </verify>
  <done>Solar distribution fix implemented, 900 series cooling improved without corrections</done>
</task>

</tasks>

<verification>
Run full solar validation:
cargo test --test solar_distribution_validation -- --nocapture

Verify:
- Test produces hourly breakdown for 600 and 900 cases
- Before/after comparison shows improvement
- No correction factors added
</verification>

<success_criteria>
- Solar distribution diagnostic test exists and runs
- Discrepancy root causes identified and documented
- Physics-based fix implemented (no correction factors)
- 900 series cooling shows measurable improvement
- Blind validation pass rate increases (expected: small gain at this stage)
</success_criteria>

<output>
After completion, create `.planning/phases/B-physics-fixes/B-01-SUMMARY.md`
</output>