---
phase: 03-Solar-Radiation
plan: 07c
type: execute
wave: 4
depends_on: [03-07]
files_modified: [src/sim/engine.rs, tests/ashrae_140_case_900.rs]
autonomous: true
requirements: []
gap_closure: true

must_haves:
  truths:
    - "Case 900 annual cooling energy within [2.13, 3.67] MWh reference"
    - "Case 900 annual heating energy within [1.17, 2.04] MWh reference"
    - "HVAC demand calculation for high-mass buildings accounts for thermal mass effects correctly"
    - "Thermal mass conductances (h_tr_em, h_tr_ms) calibrated for annual energy accuracy"
  artifacts:
    - path: "src/sim/engine.rs"
      provides: "Reverted solar_beam_to_mass_fraction to 0.7 and calibrated thermal mass conductances"
      contains: "solar_beam_to_mass_fraction"
      contains: "h_tr_em"
      contains: "h_tr_ms"
    - path: "tests/ashrae_140_case_900.rs"
      provides: "Annual energy validation tests"
      contains: "test_case_900_annual_cooling_within_reference_range"
      contains: "test_case_900_annual_heating_within_reference_range"
  key_links:
    - from: "src/sim/engine.rs::solar_beam_to_mass_fraction"
      to: "Solar gain distribution in step_physics"
      via: "70% of beam solar goes to thermal mass per ASHRAE 140"
    - from: "src/sim/engine.rs::h_tr_em, h_tr_ms"
      to: "Thermal mass coupling in 5R1C thermal network"
      via: "Exterior-Mass and Mass-Surface conductances control thermal mass response"
    - from: "hvac_power_demand()"
      to: "Annual energy accumulation"
      via: "Sensitivity calculation determines HVAC demand magnitude"
---

<objective>
Continue investigation of thermal mass dynamics to fix annual energy over-prediction for Case 900.

Context from Plan 03-07:
- Plan 03-07 completed its tasks but the objective was NOT achieved
- Annual cooling: 4.69 MWh vs [2.13, 3.67] MWh reference (28-120% above)
- Annual heating: 6.90 MWh vs [1.17, 2.04] MWh reference (239-491% above)
- HVAC runs 78.4% of time vs expected ~50%
- Sensitivity = 0.0021 K/W (very low, causing high HVAC demand)
- Free-floating temperature often 7-10°C below heating setpoint
- Solar_beam_to_mass_fraction was changed to 0.5 but should be reverted to 0.7 (made cooling worse)

Investigation focus:
1. Revert solar_beam_to_mass_fraction to 0.7 (ASHRAE 140 specification)
2. Investigate thermal mass conductances: h_tr_em (Exterior -> Mass), h_tr_ms (Mass -> Surface)
3. Analyze sensitivity calculation in HVAC demand for high-mass buildings
4. Consider thermal mass coupling parameters (α_em, α_ms) and their impact on annual energy

Expected outcome:
- Annual cooling energy within [2.13, 3.67] MWh
- Annual heating energy within [1.17, 2.04] MWh
- HVAC runtime closer to ~50% of hours
- Free-floating temperatures closer to 20°C during winter
</objective>

<execution_context>
@/home/alex/.claude/get-shit-done/workflows/execute-plan.md
@/home/alex/.claude/get-shit-done/templates/summary.md
@/home/alex/.claude/get-shit-done/references/checkpoints.md
@/home/alex/.claude/get-shit-done/references/tdd.md
</execution_context>

<context>
@.planning/phases/03-Solar-Radiation/03-CONTEXT.md
@.planning/phases/03-Solar-Radiation/03-RESEARCH.md
@.planning/phases/03-Solar-Radiation/03-VERIFICATION.md
@.planning/STATE.md
@.planning/ROADMAP.md
@.planning/phases/03-Solar-Radiation/03-07-SUMMARY.md

# Only reference prior plan SUMMARYs if genuinely needed
@.planning/phases/03-Solar-Radiation/03-04-SUMMARY.md
@.planning/phases/03-Solar-Radiation/03-06-SUMMARY.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Revert solar_beam_to_mass_fraction to 0.7 and validate</name>
  <files>src/sim/engine.rs, tests/ashrae_140_case_900.rs</files>
  <behavior>
    - Test 1: solar_beam_to_mass_fraction should be 0.7 for Case 900 (ASHRAE 140 specification)
    - Test 2: Solar distribution validation test should pass after revert
    - Test 3: Annual cooling energy should not be worse than current 4.93 MWh
    - Test 4: Annual heating energy should not be worse than current 6.84 MWh
  </behavior>
  <action>
    Revert solar_beam_to_mass_fraction from 0.5 to 0.7 for Case 900:
    
    1. In src/sim/engine.rs, line 1008, change:
       FROM: "900" | "910" | "920" | "930" | "940" | "950" => 0.5
       TO:   "900" | "910" | "920" | "930" | "940" | "950" => 0.7
    
    2. This reverts the change made in Plan 03-07 that made cooling worse (4.93 → 5.03 MWh)
    
    3. Run tests to validate:
       - test_case_900_solar_gain_distribution_validation should pass (expects 0.7)
       - test_case_900_annual_cooling_within_reference_range
       - test_case_900_annual_heating_within_reference_range
    
    4. Document results and commit the revert if tests pass
  </action>
  <verify>
    <automated>cargo test --test ashrae_140_case_900 test_case_900_solar_gain_distribution_validation --lib</automated>
  </verify>
  <done>Solar beam-to-mass fraction reverted to 0.7, solar distribution test passing</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Analyze thermal mass conductances (h_tr_em, h_tr_ms) and their impact on annual energy</name>
  <files>src/sim/engine.rs, tests/ashrae_140_case_900.rs</files>
  <behavior>
    - Test 1: h_tr_em should be in appropriate range for Case 900 high-mass building
    - Test 2: h_tr_ms should be in appropriate range for Case 900 high-mass building
    - Test 3: Ratio h_tr_em / h_tr_ms should reflect thermal mass coupling characteristics
    - Test 4: Sensitivity calculation should account for thermal mass effects appropriately
  </behavior>
  <action>
    Analyze thermal mass conductances for Case 900:
    
    1. Read current h_tr_em and h_tr_ms values from Case 900 model initialization
    2. Compare with ASHRAE 140 reference values (if available in documentation)
    3. Analyze the ratio h_tr_em / h_tr_ms (currently 0.05 per diagnostic output)
    4. Investigate if h_tr_em is too low (weak exterior coupling) or h_tr_ms is too high
    
    Key observations from Plan 03-07 diagnostics:
    - h_tr_em = 57.32 W/K (after 1.15x enhancement)
    - h_tr_ms = 1092.00 W/K
    - Ratio h_tr_em / h_tr_ms = 0.05 (very low, thermal mass weakly coupled to exterior)
    - Sensitivity = 0.0021 K/W (very low, causing high HVAC demand)
    
    Hypothesis: The low h_tr_em / h_tr_ms ratio means thermal mass is:
    - Strongly coupled to interior (h_tr_ms = 1092 W/K)
    - Weakly coupled to exterior (h_tr_em = 57.32 W/K)
    - This causes thermal mass to release stored energy primarily to interior
    - HVAC must work against this energy release, increasing demand
    
    Create diagnostic test in tests/ashrae_140_case_900.rs:
    - test_case_900_thermal_mass_conductance_analysis() - analyzes h_tr_em, h_tr_ms, and their ratio
    - Correlates conductance values with annual energy results
    - Tests sensitivity calculation impact of different conductance ratios
    
    Reference Plan 03-06: Thermal mass coupling enhancement (1.15x) was applied to h_tr_em
    but may need additional adjustment or different approach for annual energy accuracy.
  </action>
  <verify>
    <automated>cargo test --test ashrae_140_case_900 test_case_900_thermal_mass_conductance_analysis --lib -- --nocapture</automated>
  </verify>
  <done>Thermal mass conductances analyzed, relationship to annual energy documented</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Calibrate thermal mass conductances to reduce annual energy over-prediction</name>
  <files>src/sim/engine.rs</files>
  <behavior>
    - Test 1: Calibrated h_tr_em should reduce annual heating energy toward [1.17, 2.04] MWh
    - Test 2: Calibrated h_tr_em should reduce annual cooling energy toward [2.13, 3.67] MWh
    - Test 3: Peak loads should remain within reference ranges (no regression)
    - Test 4: Temperature swing reduction should remain >12% (maintain damping effect)
  </behavior>
  <action>
    Based on findings from Task 2, calibrate thermal mass conductances:
    
    **Approach 1: Increase h_tr_em (Exterior -> Mass coupling)**
    - Hypothesis: Higher h_tr_em allows thermal mass to release stored energy to exterior
    - Current: h_tr_em = 57.32 W/K (after 1.15x enhancement from base)
    - Try: Increase coupling_enhancement factor from 1.15 to 1.5 or 2.0
    - Expected: Thermal mass releases more energy to exterior, less to interior
    - Risk: May reduce temperature swing reduction (trade-off)
    
    **Approach 2: Adjust both h_tr_em and h_tr_ms together**
    - Hypothesis: Need balanced coupling between exterior and interior
    - Try: Increase h_tr_em while decreasing h_tr_ms
    - Expected: Thermal mass releases energy more evenly to exterior and interior
    - Risk: Complex interaction, hard to predict
    
    **Approach 3: Modify sensitivity calculation for high-mass buildings**
    - Hypothesis: Current sensitivity (0.0021 K/W) is too small for high-mass
    - Try: Add thermal mass time constant factor to sensitivity calculation
    - Expected: HVAC demand reduced for high-mass buildings
    - Risk: Affects physics accuracy, may break other cases
    
    Implementation steps:
    1. Test Approach 1 first (simplest, least risk)
    2. If Approach 1 insufficient, try Approach 2
    3. Only use Approach 3 if conductance calibration fails
    
    Apply changes to src/sim/engine.rs:
    - Line 780: Change coupling_enhancement for Case 900 from 1.15 to test value
    - Or modify h_tr_em/h_tr_ms calculation directly if needed
    
    Test after each change:
    - Annual cooling energy
    - Annual heating energy
    - Peak loads
    - Temperature swing reduction
    - Max temperature
    
    Commit successful calibration with clear justification.
  </action>
  <verify>
    <automated>cargo test --test ashrae_140_case_900 test_case_900_annual_cooling_within_reference_range test_case_900_annual_heating_within_reference_range --lib</automated>
  </verify>
  <done>Thermal mass conductances calibrated, annual energies within reference ranges</done>
</task>

</tasks>

<verification>
After completing thermal mass calibration, verify:
1. Annual cooling energy within [2.13, 3.67] MWh reference
2. Annual heating energy within [1.17, 2.04] MWh reference
3. Peak cooling load within [2.10, 3.50] kW (no regression)
4. Peak heating load within [1.10, 2.10] kW (no regression)
5. Temperature swing reduction maintained (>12%)
6. Max temperature within [41.80, 46.40]°C
7. solar_beam_to_mass_fraction = 0.7 (ASHRAE 140 spec)
</verification>

<success_criteria>
1. Solar beam-to-mass fraction reverted to 0.7 (ASHRAE 140 specification)
2. Thermal mass conductances analyzed and relationship to annual energy documented
3. Thermal mass conductances calibrated to reduce annual energy over-prediction
4. Case 900 annual cooling energy within [2.13, 3.67] MWh reference
5. Case 900 annual heating energy within [1.17, 2.04] MWh reference
6. Peak loads and temperature swing maintained (no regressions)
</success_criteria>

<output>
After completion, create `.planning/phases/03-Solar-Radiation/03-07c-SUMMARY.md` with:
- Solar beam-to-mass fraction revert details
- Thermal mass conductance analysis results
- Calibration approach and results
- Annual energy improvements achieved
