---
phase: M2-zone-hvac-controls
plan: 05
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/hvac/zone_control_tests.rs
  - src/hvac/zone_control.rs
  - src/cli/hvac_commands.rs
autonomous: true
requirements: [MZ-03, MZ-04, MZ-10]
gap_closure: true

must_haves:
  truths:
    - "HVAC control tests compile and run successfully"
    - "Zone-level HVAC control logic works with correct VectorField API"
    - "CLI HVAC commands are fully implemented and functional"
  artifacts:
    - path: "tests/hvac/zone_control_tests.rs"
      provides: "Working HVAC control validation tests"
      min_lines: 300
    - path: "src/hvac/zone_control.rs"
      provides: "Fixed ThermalModel imports and VectorField usage"
      min_lines: 280
    - path: "src/cli/hvac_commands.rs"
      provides: "Fully implemented HVAC CLI commands"
      min_lines: 250
  key_links:
    - from: "src/hvac/zone_control.rs"
      to: "src/thermal/thermal_model.rs"
      via: "correct import path"
      pattern: "crate::thermal::ThermalModel"
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
Fix critical gaps in M2 phase: VectorField API compatibility, ThermalModel imports, and CLI integration

Purpose: Resolve compilation errors and implement missing functionality to achieve phase goals
Output: Working HVAC control system with functional tests and CLI integration
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
  <name>Task 1: Fix VectorField API usage in HVAC control tests</name>
  <files>tests/hvac/zone_control_tests.rs</files>
  <action>
    Replace all `.get()` method calls with proper VectorField API:
    - Change `vector_field.get(index)` to `vector_field.as_slice()[index]`
    - Update all 6 test assertions that use the incorrect API
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
  <name>Task 2: Fix ThermalModel import path in zone_control.rs</name>
  <files>src/hvac/zone_control.rs</files>
  <action>
    Correct the ThermalModel import path:
    - Change `use crate::thermal::thermal_model::ThermalModel;` to `use crate::thermal::ThermalModel;`
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
  <name>Task 3: Implement actual HVAC integration in CLI handlers</name>
  <files>src/cli/hvac_commands.rs</files>
  <action>
    Replace TODO comments with actual HVAC integration:
    
    In handle_setpoints():
    - Implement SetHeating: call zone_setpoints.set_heating_setpoint(zone_id, temp)
    - Implement SetCooling: call zone_setpoints.set_cooling_setpoint(zone_id, temp)
    - Implement SetDeadband: call zone_setpoints.set_deadband(zone_id, deadband)
    - Implement Show: display current setpoints for specified zone or all zones
    
    In handle_simulate():
    - Create ZoneControl instance with thermal model
    - Run simulation loop for specified number of steps
    - Collect energy inputs and status for each step
    - Output CSV if requested (zone_id,step,temperature,energy,status)
    
    In handle_status():
    - Get current zone status from ZoneControl
    - Display heating/cooling/off status for each zone
    - Show current energy consumption
    
    Add proper error handling and validation throughout.
  </action>
  <verify>
    <automated>cargo build --release 2>&1 | grep -i "hvac_commands" || echo "No hvac_commands errors"</automated>
  </verify>
  <done>
    All CLI HVAC commands fully implemented
    Integration with ZoneControl and ZoneSetpoints working
    Error handling and validation complete
  </done>
</task>

</tasks>

<verification>
- cargo test zone_control_tests passes (all HVAC control tests)
- cargo build --release succeeds (CLI integration working)
- fluxion multi-zone hvac --help shows all commands
- fluxion multi-zone hvac status runs without errors
</verification>

<success_criteria>
- cargo check --lib completes without errors
- cargo test zone_control_tests passes all tests
- cargo build --release succeeds
- fluxion multi-zone hvac --help shows comprehensive command list
- fluxion multi-zone hvac status displays current HVAC status
</success_criteria>

<output>
After completion, create `.planning/phases/M2-zone-hvac-controls/M2-05-SUMMARY.md`
</output>
