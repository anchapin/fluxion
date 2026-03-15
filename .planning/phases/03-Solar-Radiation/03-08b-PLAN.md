---
phase: 03-Solar-Radiation
plan: 08b
type: execute
wave: 5
depends_on: [03-08, 03-07b]
files_modified: [tests/ashrae_140_case_900.rs, tests/ashrae_140_free_floating.rs]
autonomous: true
requirements: []
gap_closure: true

must_haves:
  truths:
    - "Case 900FF temperature swing reduction ~19.6% (from 13.7%)"
    - "Case 900FF max temperature remains within [41.80, 46.40]°C reference"
    - "Thermal mass coupling parameters (h_tr_em, h_tr_ms) correctly tuned for high-mass buildings"
    - "No regressions in peak loads or annual energies from Plans 03-07, 03-08"
  artifacts:
    - path: "tests/ashrae_140_case_900.rs"
      provides: "Full validation suite for Case 900 after thermal mass tuning"
      contains: "test_case_900ff_temperature_swing_reduction_final"
      contains: "test_case_900ff_max_temperature_within_reference_range"
      contains: "test_case_900_peak_cooling_within_reference_range"
      contains: "test_case_900_peak_heating_within_reference_range"
      contains: "test_case_900_annual_cooling_within_reference_range"
      contains: "test_case_900_annual_heating_within_reference_range"
    - path: "tests/ashrae_140_free_floating.rs"
      provides: "Thermal mass validation tests"
      contains: "test_thermal_mass_lag_and_damping"
  key_links:
    - from: "tests/ashrae_140_case_900.rs"
      to: "src/sim/engine.rs"
      via: "Full validation suite tests all Case 900 metrics after thermal mass tuning"
      pattern: "ashrae_140_case_900.*test"
    - from: "tests/ashrae_140_free_floating.rs"
      to: "src/sim/engine.rs"
      via: "Thermal mass dynamics validation"
      pattern: "thermal_mass_lag_and_damping"
---

<objective>
Validate that thermal mass coupling enhancements in Plan 03-08 achieve temperature swing reduction target without introducing regressions in peak loads, annual energies, or other metrics.

Purpose: After applying thermal mass coupling enhancements in Plan 03-08 (tuning h_tr_em and h_tr_ms together) to achieve ~19.6% temperature swing reduction, we must verify that these changes do not break previously working metrics from Plans 03-07 (annual energies) and 03-05 (peak loads). Max temperature should remain within [41.80, 46.40]°C.

Output: Full Case 900 and free-floating validation suite passing with temperature swing reduction ~19.6% and no regressions in other metrics.
</objective>

<execution_context>
@/home/alex/.claude/get-shit-done/workflows/execute-plan.md
@/home/alex/.claude/templates/summary.md
</execution_context>

<context>
@.planning/phases/03-Solar-Radiation/03-CONTEXT.md
@.planning/phases/03-Solar-Radiation/03-RESEARCH.md
@.planning/phases/03-Solar-Radiation/03-VERIFICATION.md
@.planning/STATE.md
@.planning/ROADMAP.md

# Only reference prior plan SUMMARYs if genuinely needed
@.planning/phases/03-Solar-Radiation/03-06-SUMMARY.md
@.planning/phases/03-Solar-Radiation/03-07b-SUMMARY.md (expected after plan execution)
@.planning/phases/03-Solar-Radiation/03-08-SUMMARY.md (expected after plan execution)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Validate thermal mass tuning and verify no regressions in all Case 900 metrics</name>
  <files>tests/ashrae_140_case_900.rs, tests/ashrae_140_free_floating.rs</files>
  <action>
    Run full Case 900 and free-floating validation suite to verify thermal mass tuning achieves targets without regressions:

    1. **Temperature swing reduction (from Plan 03-08):**
       - test_case_900ff_temperature_swing_reduction_final: should pass (~19.6% reduction)

    2. **Max temperature (from Plan 03-01):**
       - test_case_900ff_max_temperature_within_reference_range: should pass (within [41.80, 46.40]°C)

    3. **Peak loads (from Plan 03-05):**
       - test_case_900_peak_cooling_within_reference_range: cooling within [2.10, 3.50] kW
       - test_case_900_peak_heating_within_reference_range: heating within [1.10, 2.10] kW

    4. **Annual energies (from Plan 03-07):**
       - test_case_900_annual_cooling_within_reference_range: cooling within [2.13, 3.67] MWh
       - test_case_900_annual_heating_within_reference_range: heating within [1.17, 2.04] MWh

    5. **Thermal mass validation tests:**
       - test_case_900ff_thermal_mass_coupling_parameters: should pass (coupling parameters validated)
       - test_case_900_thermal_capacitance_validation: should pass (thermal capacitance verified)
       - test_case_900ff_thermal_mass_coupling_tuning_options: should pass (enhancement strategies tested)

    6. **Free-floating tests:**
       - test_thermal_mass_lag_and_damping: should pass (thermal mass dynamics validated)
       - All 10 free-floating tests should pass

    Expected outcomes after optimal tuning from Plan 03-08:
    - Temperature swing reduction: ~19.6% (achieved target, up from 13.7%)
    - Max temperature: within [41.80, 46.40]°C (maintained)
    - Peak loads: unchanged (cooling ~3.54 kW, heating ~2.10 kW)
    - Annual energies: unchanged (cooling ~2.9 MWh, heating ~1.7 MWh from Plan 03-07)
    - All free-floating tests: passing (10/10)

    Document validation results in test comments and commit message.
  </action>
  <verify>
    <automated>cargo test ashrae_140_case_900 ashrae_140_free_floating --lib</automated>
  </verify>
  <done>Temperature swing reduction ~19.6% achieved, max temperature within reference range, all Case 900 and free-floating tests passing, no regressions</done>
</task>

</tasks>

<verification>
Run full Case 900 and free-floating validation suite and verify:
1. Temperature swing reduction ~19.6% for Case 900FF
2. Max temperature within [41.80, 46.40]°C for Case 900FF
3. Peak cooling load within [2.10, 3.50] kW (no regression)
4. Peak heating load within [1.10, 2.10] kW (no regression)
5. Annual cooling energy within [2.13, 3.67] MWh (no regression from Plan 03-07)
6. Annual heating energy within [1.17, 2.04] MWh (no regression from Plan 03-07)
7. Thermal capacitance values match ASHRAE 140 specifications
8. All free-floating tests passing (10/10)
9. All Case 900 tests passing
</verification>

<success_criteria>
1. Temperature swing reduction ~19.6% achieved for Case 900FF (from Plan 03-08)
2. Max temperature maintained within [41.80, 46.40]°C reference range
3. Peak loads unchanged (no regressions from Plan 03-05)
4. Annual energies unchanged (no regressions from Plan 03-07)
5. Thermal capacitance values verified against ASHRAE 140 specifications
6. Thermal mass coupling enhancement strategy validated
7. All free-floating tests passing (10/10)
8. Full Case 900 validation suite passing
</success_criteria>

<output>
After completion, create `.planning/phases/03-Solar-Radiation/03-08b-SUMMARY.md` with:
- Validation results confirming temperature swing reduction target achieved
- No regressions in peak loads, annual energies, or other metrics
- Summary of thermal mass coupling enhancements from Plan 03-08
- All Phase 3 gap closure plans complete
</output>
