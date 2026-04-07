---
phase: M2-zone-hvac-controls
gap_closure: true
plan: 07
type: execute
wave: 1
depends_on: []
files_modified:
  - src/hvac/zone_control.rs
  - tests/hvac/zone_control_tests.rs
  - src/hvac/mod.rs
  - src/cli/hvac_commands.rs
autonomous: true
requirements: [MZ-03, MZ-04, MZ-10]

must_haves:
  truths:
    - "HVAC control tests compile without errors"
    - "Zone-level HVAC control logic works with correct imports and API"
    - "CLI HVAC commands are properly integrated"
  artifacts:
    - path: "src/hvac/zone_control.rs"
      provides: "Fixed ThermalModel imports and proper module structure"
      min_lines: 280
    - path: "tests/hvac/zone_control_tests.rs"
      provides: "Working HVAC control validation tests with correct VectorField API"
      min_lines: 200
    - path: "src/cli/hvac_commands.rs"
      provides: "Fully implemented HVAC CLI commands with proper integration"
      min_lines: 350
  key_links:
    - from: "src/hvac/zone_control.rs"
      to: "src/thermal/thermal_model.rs"
      via: "correct import path"
      pattern: "crate::thermal::thermal_model::ThermalModel"
    - from: "tests/hvac/zone_control_tests.rs"
      to: "src/hvac/zone_control.rs"
      via: "test validation"
      pattern: "test_zone_control"
    - from: "src/cli/hvac_commands.rs"
      to: "src/hvac/zone_control.rs"
      via: "CLI integration"
      pattern: "ZoneControl::new"
---

<objective>
Fix critical compilation errors and complete CLI integration for HVAC controls

Purpose: Resolve remaining compilation issues and implement full CLI functionality
Output: Working HVAC control system with functional tests and complete CLI integration
</objective>

<execution_context>
@/home/alex/.config/opencode/get-shit-done/workflows/execute-plan.md
@/home/alex/.config/opencode/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/M2-zone-hvac-controls/M2-01-SUMMARY.md
@.planning/phases/M2-zone-hvac-controls/M2-03-SUMMARY.md
@.planning/phases/M2-zone-hvac-controls/M2-04-SUMMARY.md
@.planning/phases/M2-zone-hvac-controls/M2-VERIFICATION.md
@src/physics/cta/mod.rs
@src/thermal/thermal_model.rs
@src/hvac/zone_setpoints.rs
@src/hvac/zone_control.rs
@tests/hvac/zone_control_tests.rs
@src/cli/hvac_commands.rs
@src/cli/multi_zone.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Fix ThermalModel import path in zone_control.rs</name>
  <files>src/hvac/zone_control.rs</files>
  <action>
    Correct the ThermalModel import path:
    - Change `use crate::thermal::ThermalModel;` to `use crate::thermal::thermal_model::ThermalModel;`
    - Verify all ThermalModel usage compiles correctly
    - Ensure Arc<ThermalModel> usage is preserved
    - Maintain existing control logic (1000W per °C difference)
  </action>
  <verify>
    <automated>cargo check --lib 2>&1 | grep -i "zone_control" || echo "No zone_control errors"</automated>
  </verify>
  <done>
    zone_control.rs compiles without import errors
    ThermalModel integration working correctly
    Control logic unchanged
  </done>
</task>

<task type="auto">
  <name>Task 2: Fix VectorField API usage in HVAC control tests</name>
  <files>tests/hvac/zone_control_tests.rs</files>
  <action>
    Replace all `.get()` method calls with proper VectorField API:
    - Change `vector_field.get(index)` to `vector_field.as_slice()[index]`
    - Update all test assertions that use the incorrect API
    - Ensure proper bounds checking for zone indices
    - Maintain existing test logic and validation
    
    Specific fixes needed:
    - Line 194: current_temps.get(0) → current_temps.as_slice()[0]
    - Line 210: current_temps.get(1) → current_temps.as_slice()[1]
    - Line 227: current_temps.get(2) → current_temps.as_slice()[2]
    - Line 246: result.get(0) → result.as_slice()[0]
    - Line 247: result.get(1) → result.as_slice()[1]
    - Line 262: result.get(2) → result.as_slice()[2]
  </action>
  <verify>
    <automated>cargo check --lib 2>&1 | grep -i "zone_control_tests" || echo "No zone_control_tests errors"</automated>
  </verify>
  <done>
    HVAC control tests compile without VectorField API errors
    All test assertions use correct as_slice() indexing
    Test logic preserved (same validation, edge cases)
  </done>
</task>

<task type="auto">
  <name>Task 3: Fix zone_setpoints module imports</name>
  <files>src/hvac/mod.rs, src/hvac/zone_control.rs</files>
  <action>
    Ensure proper module structure and imports:
    - In src/hvac/mod.rs: Add `pub mod zone_setpoints;` declaration
    - In src/hvac/zone_control.rs: Change `super::zone_setpoints::ZoneSetpoints` to `crate::hvac::zone_setpoints::ZoneSetpoints`
    - Verify all zone_setpoints usage compiles correctly
    - Maintain existing functionality
  </action>
  <verify>
    <automated>cargo check --lib 2>&1 | grep -i "zone_setpoints" || echo "No zone_setpoints errors"</automated>
  </verify>
  <done>
    zone_setpoints module properly declared and imported
    All zone_setpoints usage compiles without errors
    Functionality preserved
  </done>
</task>

</tasks>

<verification>
- cargo check --lib completes without errors
- cargo test zone_control_tests compiles successfully
- fluxion multi-zone hvac --help shows all commands
- fluxion multi-zone hvac status runs without errors
</verification>

<success_criteria>
- cargo check --lib completes without errors
- cargo test zone_control_tests compiles successfully
- All HVAC control tests pass
- CLI HVAC commands work properly
- Python bindings can be built and tested
</success_criteria>

<output>
After completion, create `.planning/phases/M2-zone-hvac-controls/M2-07-SUMMARY.md`
</output>
