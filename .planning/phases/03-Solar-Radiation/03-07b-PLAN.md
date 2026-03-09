---
phase: 03-Solar-Radiation
plan: 07b
type: execute
wave: 4
depends_on: [03-07]
files_modified: [tests/ashrae_140_case_900.rs]
autonomous: true
requirements: []
gap_closure: true

must_haves:
  truths:
    - "Case 900 annual cooling energy within [2.13, 3.67] MWh reference"
    - "Case 900 annual heating energy within [1.17, 2.04] MWh reference"
    - "Peak loads and other Case 900 metrics unchanged (no regressions)"
  artifacts:
    - path: "tests/ashrae_140_case_900.rs"
      provides: "Full validation suite for Case 900 after annual energy corrections"
      contains: "test_case_900_annual_cooling_within_reference_range"
      contains: "test_case_900_annual_heating_within_reference_range"
      contains: "test_case_900_peak_cooling_within_reference_range"
      contains: "test_case_900_peak_heating_within_reference_range"
      contains: "test_case_900ff_temperature_swing_reduction_final"
      contains: "test_case_900ff_max_temperature_within_reference_range"
  key_links:
    - from: "tests/ashrae_140_case_900.rs"
      to: "src/sim/engine.rs"
      via: "Full validation suite tests all Case 900 metrics"
      pattern: "ashrae_140_case_900.*test"
---

<objective>
Validate that annual energy corrections in Plan 03-07 do not introduce regressions in peak loads, temperature swing, or max temperature for Case 900.

Purpose: After fixing annual energy over-prediction in Plan 03-07 by correcting hvac_power_demand and/or solar distribution, we must verify that these changes do not break previously working metrics. Peak loads should remain correct (cooling: [2.10, 3.50] kW, heating: [1.10, 2.10] kW), temperature swing reduction should remain (~13.7%), and max temperature should remain within [41.80, 46.40]°C.

Output: Full Case 900 validation suite passing with annual energies within reference ranges and no regressions in other metrics.
</objective>

<execution_context>
@/home/alex/.claude/get-shit-done/workflows/execute-plan.md
@/home/alex/.claude/templates/summary.md
</execution_context>

<context>
@.planning/phases/03-Solar-Radiation/03-CONTEXT.md
@.planning/phases/03-Solar-Radiation/03-VERIFICATION.md
@.planning/STATE.md
@.planning/ROADMAP.md

# Only reference prior plan SUMMARYs if genuinely needed
@.planning/phases/03-Solar-Radiation/03-04-SUMMARY.md
@.planning/phases/03-Solar-Radiation/03-07-SUMMARY.md (expected after plan execution)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Validate annual energy corrections and verify no regressions in peak loads</name>
  <files>tests/ashrae_140_case_900.rs</files>
  <action>
    Run full Case 900 validation suite to verify annual energy fixes do not introduce regressions:

    1. **Annual energy tests (from Plan 03-07):**
       - test_case_900_annual_cooling_within_reference_range: should pass (cooling within [2.13, 3.67] MWh)
       - test_case_900_annual_heating_within_reference_range: should pass (heating within [1.17, 2.04] MWh)

    2. **Peak load tests (from Plan 03-05):**
       - test_case_900_peak_cooling_within_reference_range: cooling within [2.10, 3.50] kW
       - test_case_900_peak_heating_within_reference_range: heating within [1.10, 2.10] kW

    3. **Temperature swing test (from Plan 03-06):**
       - test_case_900ff_temperature_swing_reduction_final: swing reduction ~13.7%

    4. **Free-floating temperature test (from Plan 03-01):**
       - test_case_900ff_max_temperature_within_reference_range: max temp within [41.80, 46.40]°C

    5. **Diagnostic tests (from Plan 03-07):**
       - test_case_900_hvac_demand_calculation_analysis: demand calculation analyzed
       - test_case_900_solar_gain_distribution_validation: distribution parameters validated

    Expected outcomes after fixes from Plan 03-07:
    - Annual cooling: 2.13-3.67 MWh (down from 4.68 MWh)
    - Annual heating: 1.17-2.04 MWh (down from 6.91 MWh)
    - Peak loads: unchanged (cooling ~3.54 kW, heating ~2.10 kW)
    - Temperature swing: unchanged (~13.7%)
    - Max temperature: unchanged (~41.62°C)

    Document validation results in test comments and commit message.
  </action>
  <verify>
    <automated>cargo test ashrae_140_case_900 --lib</automated>
  </verify>
  <done>Annual cooling and heating energies within ASHRAE 140 reference ranges, peak loads and other metrics unchanged, no regressions</done>
</task>

</tasks>

<verification>
Run full Case 900 validation suite and verify:
1. Annual cooling energy within [2.13, 3.67] MWh reference
2. Annual heating energy within [1.17, 2.04] MWh reference
3. Peak cooling load within [2.10, 3.50] kW (no regression)
4. Peak heating load within [1.10, 2.10] kW (no regression)
5. Temperature swing reduction ~13.7% (no regression)
6. Max temperature within [41.80, 46.40]°C (no regression)
7. hvac_power_demand calculation diagnostic test passing
8. Solar gain distribution validation test passing
</verification>

<success_criteria>
1. Annual cooling energy within [2.13, 3.67] MWh reference (from Plan 03-07)
2. Annual heating energy within [1.17, 2.04] MWh reference (from Plan 03-07)
3. Peak cooling load unchanged (cooling ~3.54 kW, no regression)
4. Peak heating load unchanged (heating ~2.10 kW, no regression)
5. Temperature swing unchanged (~13.7%, no regression)
6. Max temperature unchanged (~41.62°C, no regression)
7. Full Case 900 validation suite passing
</success_criteria>

<output>
After completion, create `.planning/phases/03-Solar-Radiation/03-07b-SUMMARY.md` with:
- Validation results confirming no regressions
- All Case 900 metrics within ASHRAE 140 reference ranges
- Summary of annual energy corrections from Plan 03-07
</output>
