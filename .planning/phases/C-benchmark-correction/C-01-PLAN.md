---
phase: C-benchmark-correction
plan: C-01
type: execute
wave: 1
depends_on: ["B-03"]
files_modified:
  - data/ashrae_140_true_reference/
  - src/validation/benchmark.rs
autonomous: true
requirements:
  - BENCH-01

must_haves:
  truths:
    - "Benchmark ranges reflect true EnergyPlus/ESP-r/TRNSYS reference values, not 'calibrated for 5R1C'"
  artifacts:
    - path: "data/ashrae_140_true_reference/"
      provides: "True ASHRAE reference data for all cases"
---

<objective>
Replace "calibrated for 5R1C" benchmark ranges with true ASHRAE 140 reference values. This phase may reveal that current "passing" cases were only passing because the benchmark was adjusted to match the broken model.
</objective>

<context>
From deep dive: benchmark.rs lines 108-110 comment explicitly states ranges are "calibrated for 5R1C model" — not true reference values. This means the current validation is circular: model is validated against itself.

This phase must source actual EnergyPlus/ESP-r/TRNSYS outputs from ASHRAE 140 standard test suite.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Source true ASHRAE 140 reference data</name>
  <files>data/ashrae_140_true_reference/</files>
  <action>
Create data/ashrae_140_true_reference/ directory with:

1. **energyplus_references/** — EnergyPlus output files for each case
   - Annual energy (heating, cooling) in MWh
   - Peak loads (heating, cooling) in kW
   - Monthly energy breakdown
   - Source: ASHRAE 140 standard validation files (version to be documented)

2. **esp_r_references/** — ESP-r reference outputs (if available)
3. **trnsys_references/** — TRNSYS reference outputs (if available)

4. **metadata.yaml** — Documentation of source programs and versions:
   ```yaml
   energyplus_version: "23.x"
   esp_r_version: "latest"
   trnsys_version: "18"
   source: "ASHRAE 140 standard test suite"
   date_obtained: "2026-05-XX"
   ```

If actual reference data cannot be sourced from official ASHRAE documents, create data/ with current Fluxion outputs labeled as "PROVISIONAL - needs verification against official ASHRAE reference data."

Document in BENCHMARK_SOURCING_ISSUES.md what was available vs what was not.
  </action>
  <verify>
    ls -la data/ashrae_140_true_reference/
    cat data/ashrae_140_true_reference/metadata.yaml
  </verify>
  <done>True ASHRAE reference data sourced and documented</done>
</task>

<task type="auto">
  <name>Task 2: Update benchmark.rs to use true references</name>
  <files>src/validation/benchmark.rs</files>
  <action>
Update benchmark.rs to load true reference values instead of calibrated ranges:

1. Replace hardcoded calibrated ranges with loading from data/ashrae_140_true_reference/
2. Remove the comment "calibrated for 5R1C model" — this was a known accommodation
3. Ensure benchmark loading is case-agnostic (no case-type-specific adjustments)
4. Document the change: "Reference values now represent actual EnergyPlus/ESP-r/TRNSYS outputs per ASHRAE 140 standard"

After update:
```rust
// Before (calibrated):
"900" => EnergyRange { min: 1.17, max: 2.04, calibrated_for_5r1c: true }

// After (true reference):
"900" => EnergyRange { min: 1.17, max: 2.04, source: "EnergyPlus 23.x" }
```

Run: cargo test --test ashrae_140_blind_validation to see effect of true references
  </action>
  <verify>
    cargo check --lib
    cargo test --test ashrae_140_blind_validation 2>&1 | grep -E "(PASS|FAIL|baseline)" | head -20
  </verify>
  <done>benchmark.rs uses true reference data, calibrated ranges removed</done>
</task>

</tasks>

<verification>
Run: cargo test --test ashrae_140_blind_validation

Verify:
- benchmark.rs loads from data/ashrae_140_true_reference/
- No "calibrated for 5R1C" comments remain
- Pass rate reflects true model accuracy vs reference programs
</verification>

<success_criteria>
- True reference data sourced and documented (or provisional with issues noted)
- benchmark.rs no longer contains "calibrated for 5R1C" language
- Blind validation uses true reference values for comparison
</success_criteria>

<output>
After completion, create `.planning/phases/C-benchmark-correction/C-01-SUMMARY.md`
</output>