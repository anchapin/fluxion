---
phase: B-physics-fixes
plan: B-03
type: execute
wave: 3
depends_on: ["B-02"]
files_modified:
  - tests/free_floating_temperature_validation.rs
  - src/sim/thermal_model_core.rs
  - src/sim/hvac.rs
autonomous: true
requirements:
  - PHYSICS-03

must_haves:
  truths:
    - "Free-floating temperature profiles match reference within ±2°C"
    - "HVAC is truly inactive in free-float mode (no heating/cooling load)"
  artifacts:
    - path: "tests/free_floating_temperature_validation.rs"
      provides: "Free-float temperature comparison test"
    - path: ".planning/baseline/FREE_FLOAT_ANALYSIS.md"
      provides: "Free-float failure root cause analysis"
---

<objective>
Fix free-floating temperature failures. The baseline shows extreme failures: 900FF max temp is 125°C when reference is 41-46°C. This suggests either HVAC is incorrectly active, internal gains are double-counted, or thermal damping is wrong.
</objective>

<context>
From ASHRAE140_RESULTS.md:
- 600FF: max 105.85°C (ref: 64.90-75.10°C)
- 650FF: max 103.59°C (ref: 63.20-73.50°C)
- 900FF: max 125.49°C (ref: 41.80-46.40°C)
- 950FF: max 123.99°C (ref: 35.50-38.50°C)

These are physically impossible temperatures for a building. Root causes to investigate:
1. HVAC incorrectly active in free-float mode
2. Internal gains (occupants, equipment, lighting) double-counted
3. Thermal mass not damping correctly (solver instability)
4. Weather data handling error for unconditioned case
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create free-floating diagnostic test</name>
  <files>tests/free_floating_temperature_validation.rs</files>
  <action>
Create tests/free_floating_temperature_validation.rs that:

1. Runs free-floating cases: 600FF, 650FF, 900FF, 950FF
2. Captures hourly:
   - Indoor air temperature (°C)
   - HVAC state (active/inactive, heating/cooling demand)
   - Internal gain totals (W) — occupants, equipment, lighting
   - Solar gain totals (W)
   - Thermal mass node temperature (°C)
3. Compares against reference diurnal temperature profiles
4. Reports:
   - Min/max temperatures per case
   - HVAC energy consumption (should be 0 for free-float)
   - Diurnal temperature swing vs reference

The test should reveal WHY temperatures are physically impossible.
  </action>
  <verify>
    cargo test --test free_floating_temperature_validation -- --nocapture 2>&1 | head -100
  </verify>
  <done>Free-floating diagnostic test exists and produces temperature/hvac/gain breakdown</done>
</task>

<task type="auto">
  <name>Task 2: Identify free-float failure root cause</name>
  <files>tests/free_floating_temperature_validation.rs, src/sim/thermal_model_core.rs, src/sim/hvac.rs</files>
  <action>
Run the diagnostic and identify root cause.

### Check 1: HVAC State
- Is HVAC truly off in free-float mode?
- If any heating/cooling load is computed, that's a bug
- Expected: HVAC energy = 0 for all free-float cases

### Check 2: Internal Gains
- Are internal gains (occupants, equipment, lighting) being summed correctly?
- Check if gains are being double-counted (added to both thermal network AND HVAC load)
- Expected: Sum of (occupants + equipment + lighting) = total internal gain

### Check 3: Thermal Damping
- Does the thermal mass provide correct damping of diurnal temperature swing?
- For high-mass (900FF), temperature swing should be SMALLER than low-mass (600FF)
- If swing is too large, thermal mass isn't providing enough damping

### Check 4: Solver Stability
- Check for numerical instability in implicit solver
- Very high temperatures can indicate solver divergence
- Look for timestep reduction or instability indicators

For each potential root cause, document:
- Evidence from diagnostic test
- Expected behavior
- Recommended fix approach

Output to .planning/baseline/FREE_FLOAT_ANALYSIS.md
  </action>
  <verify>
    File exists: .planning/baseline/FREE_FLOAT_ANALYSIS.md
    Contains: Root cause identification, evidence, fix approach
  </verify>
  <done>Free-floating failure root causes identified</done>
</task>

<task type="auto">
  <name>Task 3: Implement free-float fix</name>
  <files>src/sim/thermal_model_core.rs, src/sim/hvac.rs</files>
  <action>
Based on root cause analysis, implement fix:

### If HVAC is incorrectly active:
Fix HVAC control logic for free-float mode:
```rust
// In free-float mode, HVAC system should be completely inactive
// No heating demand, no cooling demand, no energy consumption
if spec.is_free_floating() {
    hvac_output = HvacOutput::zero();  // No heating, no cooling
}
```

### If internal gains are double-counted:
Fix gain accounting — ensure gains are counted once in thermal network only.
Check: Where are gains added to the thermal network? Are they also added to HVAC load?

### If thermal damping is insufficient:
Fix thermal mass configuration — ensure Cm is large enough to damp diurnal swing.
Check: For 900FF (high mass), Cm should produce τ ≈ 26+ hours.

### If solver is unstable:
Fix implicit solver parameters for free-float cases.
Check: Are there timestep issues in long-simulation runs without HVAC load?

After fix:
1. Re-run free_floating_temperature_validation
2. Confirm min/max temperatures are physically reasonable (10-50°C range, not 125°C)
3. Confirm HVAC energy = 0 for all free-float cases
4. Run blind validation: cargo test --test ashrae_140_blind_validation
  </action>
  <verify>
    cargo test --test free_floating_temperature_validation 2>&1 | grep -E "(max|min|°C|PASS|FAIL)"
    cargo test --test ashrae_140_blind_validation 2>&1 | grep -E "(FF|PASS|FAIL)" | head -20
  </verify>
  <done>Free-floating temperature fix implemented, temperatures in 10-50°C range, HVAC inactive</done>
</task>

</tasks>

<verification>
Run: cargo test --test free_floating_temperature_validation -- --nocapture

Verify:
- Free-float max temps are physically reasonable (<50°C, not 125°C)
- HVAC energy = 0 for free-float cases
- Diurnal swing matches reference pattern
</verification>

<success_criteria>
- Free-floating diagnostic test exists and runs
- Root causes identified and documented
- Fix implemented (physics-based, no correction factors)
- Free-float max temps within ±2°C of reference
- All free-float cases pass OR root cause clearly documented as architectural issue
</success_criteria>

<output>
After completion, create `.planning/phases/B-physics-fixes/B-03-SUMMARY.md`
</output>