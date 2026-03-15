---
phase: 08-Critical-Issue-Resolution
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/sim/engine.rs (likely)
  - src/validation/ashrae_140_cases.rs (maybe)
  - tests/ashrae_140_case_960_sunspace.rs (diagnostics)
autonomous: false
requirements:
  - CASE960-01
  - CASE960-02
  - CASE960-03
must_haves:
  truths:
    - "Case 960 annual cooling energy falls within 1.0-3.5 MWh reference range"
    - "No regression in other validated cases (600-950 series) after fix"
    - "Root cause is documented with before/after metrics and code changes"
    - "Validation suite passes for all cases including 960"
  artifacts:
    - path: "tests/debug_960_summer.rs"
      provides: "Diagnostic test showing summer hourly behavior"
      contains:
        - "Summer week simulation with hourly logging"
        - "Temperature profiles for both zones"
        - "Solar gains and inter-zone heat transfer rates"
    - path: "docs/CASE_960_ROOT_CAUSE.md"
      provides: "Root cause analysis and fix documentation"
      contains:
        - "Symptom: cooling 4.53 MWh vs reference 1.0-3.5 MWh"
        - "Root cause identified: [to be filled]"
        - "Fix implemented: [to be filled]"
        - "Verification data showing post-fix results"
  key_links:
    - from: "Case 960 validation test"
      to: "Annual cooling within range"
      via: "cargo test test_ashrae_140_case_960"
      pattern: "Annual Cooling"
    - from: "Inter-zone temperature analysis"
      to: "Sunspace should be warmer than back-zone in summer"
      via: "zone temperature traces"
      pattern: "Zone 1 temp > Zone 0 temp (summer)"
    - from: "Solar gains calculation"
      to: "calculate_zone_solar_gain"
      via: "Proc thorough solar debugging"
      pattern: "solar_gain_watts > 0 for sunspace"
---
<objective>
Fix Case 960 annual cooling failure (currently 4.53 MWh, reference max 3.5 MWh) by identifying and correcting the root cause in multi-zone heat transfer and/or solar gain distribution.
</objective>

<execution_context>
@/home/alex/.claude/get-shit-done/workflows/execute-plan.md
@/home/alex/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/08-Critical-Issue-Resolution/08-CONTEXT.md (will be created)
@.planning/phases/08-Critical-Issue-Resolution/08-RESEARCH.md (will be created)
@.planning/phases/08-Critical-Issue-Resolution/08-VERIFICATION.md (will be created)

# Current State (from investigation on 2026-03-11):
- Annual cooling: 4.53 MWh (reference: 1.0-3.5 MWh) - FAIL
- Annual heating: 5.78 MWh (reference: 5.0-15.0 MWh) - PASS
- Peak cooling: 3.79 kW (reference: 0.0-4.0 kW) - borderline
- Inter-zone temperature difference: sunspace 4.79°C COLDER than back-zone (should be warmer)
- Solar gains appear to be zero in simulation logs despite positive DNI/DHI

# Root Cause Analysis (Pre-hypothesis):
The excessive cooling load likely stems from incorrect inter-zone heat transfer direction. The sunspace should act as a thermal buffer, absorbing solar gains and transferring heat to the back-zone, reducing cooling demand. Instead, the sunspace is colder, suggesting:
1. Solar gains not being applied to sunspace (or being subtracted)
2. Inter-zone conductance sign error (heat flowing backwards)
3. Missing or incorrect solar distribution to sunspace windows
4. Free-floating sunspace losing excessive heat to exterior

# Plan Approach:
1. First, ensure all Phase 7 work is complete and code compiles cleanly
2. Create diagnostic tests with detailed hourly logging for summer peak days
3. Investigate solar gain calculation path for Zone 1 (sunspace)
4. Verify inter-zone heat transfer implementation (sign, magnitude, components)
5. Check HVAC efficiency assumptions (COP for cooling)
6. Compare with reference data if available via web search
7. Implement fix based on findings
8. Validate with full ASHRAE 140 suite (no regressions)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Ensure Phase 7 completion and fix compilation errors</name>
  <files>
    - all src/ files with compilation errors
    - Phase 7 summary documents
  </files>
  <action>
    1. Verify Phase 7 plans 07-10 and 07-11 are truly complete (code changes present)
    2. Fix remaining compilation errors in:
       - src/analysis/sensitivity.rs (unused import already fixed)
       - src/sim/thermal_integration.rs (PI import)
       - src/validation/report.rs (PathBuf import)
       - Any other errors blocking test compilation
    3. Run `cargo check --all-targets` to ensure clean build
    4. Document Phase 7 completion status in .planning/STATE.md
  </action>
  <verify>
    cargo check --all-targets 2>&1 | grep -E "error|warning" | head -20
  </verify>
  <done>Codebase compiles cleanly; Phase 7 complete.</done>
</task>

<task type="auto">
  <name>Task 2: Create enhanced diagnostic test for summer behavior</name>
  <files>
    - tests/debug_960_summer.rs (new)
  </files>
  <action>
    1. Write a focused test that simulates a peak summer week (e.g., July 15-21)
    2. Log hourly for both zones:
       - Outdoor temperature
       - Zone air temperatures
       - Solar gains (W/m² and total W)
       - Inter-zone heat transfer components (conduction, radiation, ventilation)
       - HVAC power demand
       - Free-floating temperature (T_i_free)
    3. Print summary statistics: mean temps, total solar gain, total inter-zone transfer
    4. Run the test and capture output to file
    5. Analyze whether sunspace receives solar gains and whether it's warmer than back-zone
  </action>
  <verify>
    cargo test test_960_summer_debug --release -- --nocapture 2>&1 | tee /tmp/summer_debug.log
  </verify>
  <done>Diagnostic test created and run; summer behavior data collected.</done>
</task>

<task type="auto">
  <name>Task 3: Investigate solar gain calculation for Zone 1 (sunspace)</name>
  <files>
    - src/sim/engine.rs (calculate_zone_solar_gain, calc_analytical_loads)
    - src/validation/ashrae_140_cases.rs (Case 960 spec)
  </files>
  <action>
    1. From diagnostic output, check if `calculate_zone_solar_gain` returns positive values for Zone 1
    2. Verify that Zone 1 surfaces include south-facing windows with correct area (6 m²)
    3. Check that `self.surfaces[1]` contains the window and that orientation is South
    4. Verify weather data DNI/DHI are valid during summer hours
    5. Check if any shading or window property issues cause zero gain
    6. If solar gains are zero, trace through `calculate_hourly_solar` to see return value
    7. If solar gains >0 but not showing in model state, check assignment to `self.solar_gains`
  </action>
  <verify>
    Analyze diagnostic log lines:
    - "solar: timestep=..., zone_idx=1, solar_gain_watts=..."
    - Should be >0 for summer hours; if 0, bug in calculation
  </verify>
  <done>Solar gain issue identified (or cleared as not the problem).</done>
</task>

<task type="auto">
  <name>Task 4: Investigate inter-zone heat transfer direction and magnitude</name>
  <files>
    - src/sim/engine.rs (step_physics_5r1c, lines 2104-2166)
  </files>
  <action>
    1. Review the inter-zone calculation:
       - `delta_t_cond = temps[1] - temps[0]` (sunspace - back)
       - `q_cond = h_iz * delta_t_cond`
       - `q_rad` uses (T_sunspace^4 - T_back^4)
       - `q_vent` from sunspace to back
       - Total: `q_iz_total = q_cond + q_rad + q_vent`
       - Applied as `[-q_iz_total, q_iz_total]` to [back, sunspace]
    2. Log these components hourly during summer to verify:
       - Is q_cond positive (should be positive if sunspace > back)
       - Is q_rad positive
       - Is q_vent positive
    3. If q_iz_total is negative, heat flows from back to sunspace (wrong direction)
    4. Check if h_iz (conductance) value is reasonable (~111 W/K for Case 960?)
    5. Verify common wall area and U-value used to compute h_iz
  </action>
  <verify>
    In summer, expect: T_sunspace > T_back => q_iz_total > 0 (heat to back)
    If T_sunspace < T_back, either zone temps are wrong or solar gains insufficient
  </verify>
  <done>Inter-zone heat transfer verified as correct (or bug identified).</done>
</task>

<task type="auto">
  <name>Task 5: Check HVAC efficiency assumptions and cooling energy accounting</name>
  <files>
    - src/sim/engine.rs (hvac_power_demand, step_physics_5r1c)
    - src/validation/benchmark.rs (reference ranges)
  </files>
  <action>
    1. Determine whether `hvac_power_demand` returns thermal load or electrical power
    2. Check if cooling COP is applied anywhere in the energy calculation
    3. Compare reference ranges: Are they for electrical or thermal energy? (EnergyPlus typically reports electrical)
    4. If we report thermal but reference is electrical, divide by COP (~3.0) to get electrical
    5. Check if heating also needs efficiency correction
    6. If efficiency correction needed, add hvac_heating_cop/hvac_cooling_cop fields and apply
  </action>
  <verify>
    Reread benchmark.rs comments: "Reference data from EnergyPlus, ESP-r, TRNSYS, and DOE2"
    EnergyPlus reports HVAC electricity, not thermal load. So we likely need COP division.
  </verify>
  <done>HVAC efficiency issue identified and corrected if needed.</done>
</task>

<task type="auto">
  <name>Task 6: Web search for ASHRAE 140 Case 960 reference values and typical behavior</name>
  <files>
    - External research only
  </files>
  <action>
    1. Search: "ASHRAE 140 Case 960 sunspace annual cooling energy" to find typical values
    2. Search: "EnergyPlus Case 960 validation results" for raw numbers
    3. Look for ASHRAE 140-2023 standard tables (may need access)
    4. Goal: Confirm that sunspace should REDUCE cooling load compared to single-zone baseline
    5. Expected: Case 960 cooling should be less than Case 600 (8-10.5 MWh), typically 1-3 MWh
    6. Document findings in research file
  </action>
  <verify>
    Found credible source stating case 960 cooling range or showing sunspace reduces cooling
  </verify>
  <done>Reference data collected and documented.</done>
</task>

<task type="auto">
  <name>Task 7: Implement fix based on root cause</name>
  <files>
    - Depends on findings from Tasks 2-6
  </files>
  <action>
    1. After identifying root cause, implement targeted fix:
       - If solar gains zero: fix `calculate_zone_solar_gain` or surfaces construction
       - If inter-zone sign wrong: flip sign in q_iz_total application
       - If HVAC efficiency missing: add COP division in energy accounting
       - If conductance wrong: adjust h_iz calculation from common wall specs
    2. Update ThermalModel or CaseSpec builder as needed
    3. Ensure changes are localized and well-commented
    4. Run `cargo fmt && cargo clippy`
  </action>
  <verify>
    Compilation successful; no clippy warnings
  </verify>
  <done>Fix implemented and code quality checked.</done>
</task>

<task type="auto">
  <name>Task 8: Validate fix with Case 960 test</name>
  <files>
    - tests/ashrae_140_case_960_sunspace.rs
    - src/validation/ashrae_140_validator.rs (full suite)
  </files>
  <action>
    1. Run `cargo test test_ashrae_140_case_960 --release`
    2. Verify Case 960 annual cooling is within 1.0-3.5 MWh
    3. Verify heating still within 5.0-15.0 MWh
    4. Check peak metrics are acceptable
    5. If still failing, iterate: return to Task 2/3/4 with new diagnostics
  </action>
  <verify>
    Test output shows: "Annual Cooling: X MWh (reference: 1.0-3.5 MWh)" with X in range
  </verify>
  <done>Case 960 validation passes.</done>
</task>

<task type="auto">
  <name>Task 9: Run full ASHRAE 140 validation suite to check for regressions</name>
  <files>
    - all validation tests
  </files>
  <action>
    1. Run `cargo test validate_all_cases --release` or `fluxion validate --all`
    2. Check that all previously passing cases (600-950) still pass
    3. Ensure no new failures introduced
    4. If regression, identify cause and adjust fix (CASE960-02)
    5. Document any intentional changes to other cases
  </action>
  <verify>
    No regression in annual heating/cooling for single-zone cases ±5% tolerance
  </verify>
  <done>Full suite passes; no regressions; ready for commit.</done>
</task>

<task type="auto">
  <name>Task 10: Document root cause, fix, and verification results</name>
  <files>
    - docs/CASE_960_ROOT_CAUSE.md (new)
    - KNOWN_ISSUES.md (update)
    - ASHRAE140_RESULTS.md (will be regenerated)
    - .planning/phases/08-Critical-Issue-Resolution/08-SUMMARY.md (auto)
  </files>
  <action>
    1. Create `docs/CASE_960_ROOT_CAUSE.md` with:
       - Symptom: cooling 4.53 MWh vs reference 1.0-3.5 MWh
       - Investigation process: diagnostics, solar gains check, inter-zone analysis, HVAC efficiency
       - Root cause: [specific code issue]
       - Fix: code changes made, files modified
       - Before/after numbers
       - Implications for model accuracy
    2. Update `KNOWN_ISSUES.md` to mark MULTI-01 as resolved or update description
    3. Regenerate ASHRAE140_RESULTS.md by running validation
    4. Create 08-SUMMARY.md per GSD template
    5. Commit with message: `fix(validation): resolve Case 960 cooling over-prediction (CASE960-02)`
  </action>
  <verify>
    git diff docs/CASE_960_ROOT_CAUSE.md shows comprehensive documentation
    git log -1 shows proper commit message
  </verify>
  <done>Documentation complete; changes committed.</done>
</task>

</tasks>
