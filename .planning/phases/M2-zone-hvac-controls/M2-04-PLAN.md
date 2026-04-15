---
phase: M2-zone-hvac-controls
plan: 04
type: execute
wave: 1
depends_on: [M2-01]
files_modified:
  - src/hvac/zone_setpoints.rs
  - src/hvac/zone_control.rs
  - src/lib.rs
  - src/python/bindings.rs
  - src/python/mod.rs
  - src/python/hvac_bindings.rs
autonomous: true
requirements: [MZ-09]
gap_closure: true

must_haves:
  truths:
    - "Python bindings compile successfully with python-bindings feature"
    - "HVAC modules use correct VectorField API"
    - "ThermalModel imports resolve correctly"
    - "Python module registration works properly"
  artifacts:
    - path: "src/hvac/zone_setpoints.rs"
      provides: "VectorField API compatibility"
      min_lines: 180
    - path: "src/hvac/zone_control.rs"
      provides: "Correct ThermalModel imports and VectorField usage"
      min_lines: 280
    - path: "src/lib.rs"
      provides: "Python module declarations"
      contains: "mod python"
    - path: "src/python/hvac_bindings.rs"
      provides: "Python HVAC bindings"
      min_lines: 150
  key_links:
    - from: "src/hvac/zone_setpoints.rs"
      to: "src/physics/cta/VectorField"
      via: "as_slice indexing"
      pattern: "as_slice\(\)"
    - from: "src/hvac/zone_control.rs"
      to: "src/thermal/thermal_model::ThermalModel"
      via: "correct import path"
      pattern: "crate::thermal::thermal_model"
    - from: "src/lib.rs"
      to: "src/python"
      via: "module declaration"
      pattern: "pub mod python"
    - from: "src/python/mod.rs"
      to: "src/python/hvac_bindings"
      via: "submodule registration"
      pattern: "hvac_bindings"
---

<objective>
Fix technical blockers in M2 phase to enable successful Python bindings build

Purpose: Resolve VectorField API incompatibility, ThermalModel import issues, and Python module registration problems
Output: Working Python bindings that compile and can be tested
</objective>

<execution_context>
@/home/alex/.agents/get-shit-done/workflows/execute-plan.md
@/home/alex/.agents/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/M2-zone-hvac-controls/M2-01-SUMMARY.md
@.planning/phases/M2-zone-hvac-controls/M2-02-SUMMARY.md
@src/physics/cta/mod.rs
@src/thermal/thermal_model.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Fix VectorField API usage in zone_setpoints.rs</name>
  <files>src/hvac/zone_setpoints.rs</files>
  <action>
    Update VectorField API usage to use correct methods:
    - Replace any .get()/.set() calls with as_slice() indexing
    - Use vector_field.as_slice()[zone_index] pattern for access
    - Ensure proper bounds checking for zone indices
    - Maintain existing validation logic (temperature ranges, deadband validation)
    - Keep VectorField storage pattern intact
  </action>
  <verify>
    <automated>cargo check --lib 2>&1 | grep -i "zone_setpoints" || echo "No zone_setpoints errors"</automated>
  </verify>
  <done>
    zone_setpoints.rs compiles without VectorField API errors
    All validation logic preserved
    Temperature and deadband ranges still enforced
  </done>
</task>

<task type="auto">
  <name>Task 2: Fix ThermalModel imports and VectorField usage in zone_control.rs</name>
  <files>src/hvac/zone_control.rs</files>
  <action>
    Fix import paths and VectorField usage:
    - Change import from `use crate::thermal::thermal_model::ThermalModel;` to correct path
    - Update VectorField access patterns to use as_slice() indexing
    - Fix zone_status initialization to use proper VectorField constructor
    - Ensure Arc<ThermalModel> usage is correct
    - Maintain existing control logic (proportional control, deadband logic)
  </action>
  <verify>
    <automated>cargo check --lib 2>&1 | grep -i "zone_control" || echo "No zone_control errors"</automated>
  </verify>
  <done>
    zone_control.rs compiles without import or API errors
    Control logic unchanged (1000W per °C difference)
    Thread-safe Arc usage preserved
  </done>
</task>

<task type="auto">
  <name>Task 3: Add Python module declarations and fix PyO3 usage</name>
  <files>src/lib.rs, src/python/mod.rs, src/python/bindings.rs</files>
  <action>
    Fix Python module registration and PyO3 API usage:

    In src/lib.rs:
    - Add `pub mod python;` declaration to register Python module
    - Ensure proper feature flag wrapping: #[cfg(feature = "python-bindings")]

    In src/python/mod.rs:
    - Replace `pyo3::import(_py, "fluxion")` with correct PyO3 import pattern
    - Use `PyModule::import(_py, "fluxion")` instead
    - Fix module registration to use proper PyO3 macros

    In src/python/bindings.rs:
    - Replace incorrect `_py` usage with proper Python parameter
    - Use `py: Python` parameter in functions that need it
    - Fix PyO3 method calls to use current API
  </action>
  <verify>
    <automated>cargo check --features python-bindings 2>&1 | grep -E "(error|python)" | head -10 || echo "Python module checks pass"</automated>
  </verify>
  <done>
    Python module declarations added
    PyO3 API usage corrected
    Feature flags properly applied
  </done>
</task>

</tasks>

<verification>
- cargo check --features python-bindings succeeds
- No VectorField API errors in HVAC modules
- ThermalModel imports resolve correctly
- Python module registration works
</verification>

<success_criteria>
- cargo check --features python-bindings completes without errors
- cargo build --features python-bindings compiles successfully
- Python bindings can be imported (maturin develop succeeds)
- All existing HVAC functionality preserved
</success_criteria>

<output>
After completion, create `.planning/phases/M2-zone-hvac-controls/M2-04-SUMMARY.md`
</output>

