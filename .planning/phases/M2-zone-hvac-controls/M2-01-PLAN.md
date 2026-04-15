---
phase: M2-zone-hvac-controls
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/hvac/zone_setpoints.rs
  - src/hvac/zone_control.rs
  - tests/hvac/zone_control_tests.rs
autonomous: true
requirements: [MZ-03, MZ-04]

must_haves:
  truths:
    - "Zone-specific heating/cooling setpoints can be configured"
    - "Independent HVAC control logic maintains zone temperatures"
    - "HVAC control respects deadband settings"
  artifacts:
    - path: "src/hvac/zone_setpoints.rs"
      provides: "Zone-specific setpoint management"
      min_lines: 150
    - path: "src/hvac/zone_control.rs"
      provides: "Independent HVAC control logic"
      min_lines: 200
    - path: "tests/hvac/zone_control_tests.rs"
      provides: "HVAC control validation tests"
      min_lines: 120
  key_links:
    - from: "src/hvac/zone_control.rs"
      to: "src/thermal/thermal_model.rs"
      via: "temperature feedback loop"
      pattern: "get_zone_temperature"
    - from: "tests/hvac/zone_control_tests.rs"
      to: "src/hvac/zone_control.rs"
      via: "test validation"
      pattern: "test_zone_control"
---

<objective>
Implement zone-level HVAC setpoints and control logic

Purpose: Enable independent temperature control for each thermal zone in multi-zone buildings
Output: Zone setpoint management system and HVAC control logic with comprehensive tests
</objective>

<execution_context>
@/home/alex/.config/opencode/get-shit-done/workflows/execute-plan.md
@/home/alex/.config/opencode/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/M1-multi-zone-foundation/M1-01-SUMMARY.md
@src/thermal/thermal_model.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement zone-specific setpoints</name>
  <files>src/hvac/zone_setpoints.rs</files>
  <action>
    Create ZoneSetpoints struct with per-zone heating/cooling setpoints and deadband configuration.
    Implement methods for:
    - new() constructor with default values
    - set_heating_setpoint(zone_id, temperature)
    - set_cooling_setpoint(zone_id, temperature)
    - set_deadband(zone_id, deadband)
    - get_heating_setpoint(zone_id) -> f64
    - get_cooling_setpoint(zone_id) -> f64
    - get_deadband(zone_id) -> f64
    - validate_setpoints() -> Result<(), String>

    Use VectorField for zone storage to match thermal model architecture.
    Ensure setpoints are in valid temperature range (10.0°C to 40.0°C).
    Deadband should be positive and <= 5.0°C.
  </action>
  <verify>
    <automated>cargo test zone_setpoints -- --nocapture</automated>
  </verify>
  <done>
    ZoneSetpoints struct implemented with validation
    All setpoint methods working correctly
    Unit tests pass
  </done>
</task>

<task type="auto">
  <name>Task 2: Implement zone-level HVAC control logic</name>
  <files>src/hvac/zone_control.rs</files>
  <action>
    Create ZoneControl struct that implements independent HVAC control for each zone.
    Key components:
    - ZoneControl::new(thermal_model: Arc<ThermalModel>, setpoints: ZoneSetpoints)
    - update_zone_controls(&mut self, current_temperatures: &VectorField) -> VectorField
    - get_zone_hvac_status(zone_id: usize) -> HVACStatus
    - calculate_energy_input(zone_id: usize, current_temp: f64) -> f64

    Implement control logic:
    - If current_temp < (heating_setpoint - deadband/2): enable heating
    - If current_temp > (cooling_setpoint + deadband/2): enable cooling
    - Otherwise: no HVAC action (deadband)
    - Return energy input in Watts for each zone

    Use HVACStatus enum: Heating, Cooling, Off
    Ensure thread-safe access to thermal model using Arc.
  </action>
  <verify>
    <automated>cargo test zone_control -- --nocapture</automated>
  </verify>
  <done>
    ZoneControl struct with independent per-zone logic
    Deadband control working correctly
    Energy calculation returns valid Watt values
  </done>
</task>

<task type="auto">
  <name>Task 3: Create HVAC control integration tests</name>
  <files>tests/hvac/zone_control_tests.rs</files>
  <action>
    Create comprehensive test suite for zone-level HVAC control:
    - test_setpoint_validation(): validates temperature ranges and deadband constraints
    - test_heating_control(): verifies heating activates when temp below setpoint
    - test_cooling_control(): verifies cooling activates when temp above setpoint
    - test_deadband_control(): ensures no action within deadband range
    - test_independent_zone_control(): validates zones don't interfere with each other
    - test_energy_calculation(): checks Watt calculations are reasonable
    - test_hvac_status_transitions(): tests state changes between heating/cooling/off

    Use test thermal model with 3 zones for testing.
    Mock temperatures to test edge cases.
  </action>
  <verify>
    <automated>cargo test zone_control_tests -- --nocapture</automated>
  </verify>
  <done>
    All HVAC control tests passing
    Edge cases covered (boundary temperatures, invalid inputs)
    Independent zone behavior verified
  </done>
</task>

</tasks>

<verification>
- Zone setpoints can be configured and validated
- HVAC control logic maintains temperatures within deadband
- Independent zone control verified (no cross-zone interference)
- Energy calculations return reasonable values
</verification>

<success_criteria>
- cargo test zone_setpoints passes (all setpoint tests)
- cargo test zone_control passes (all control logic tests)
- cargo test zone_control_tests passes (all integration tests)
- No panics or errors in HVAC control operations
</success_criteria>

<output>
After completion, create `.planning/phases/M2-zone-hvac-controls/M2-01-SUMMARY.md`
</output>
