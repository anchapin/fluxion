---
phase: M2-zone-hvac-controls
plan: 06
type: execute
wave: 2
depends_on: [M2-05]
files_modified:
  - src/python/hvac_bindings.rs
  - src/python/mod.rs
  - src/lib.rs
autonomous: true
requirements: [MZ-09]
gap_closure: true

must_haves:
  truths:
    - "Python bindings compile successfully with python-bindings feature"
    - "HVAC modules use correct VectorField API"
    - "ThermalModel imports resolve correctly"
    - "Python module registration works properly"
    - "Python API can be imported and used"
  artifacts:
    - path: "src/python/hvac_bindings.rs"
      provides: "Python HVAC bindings with fixed API usage"
      min_lines: 150
    - path: "src/python/mod.rs"
      provides: "Proper HVAC module registration"
      min_lines: 50
    - path: "src/lib.rs"
      provides: "Python module declarations with feature flags"
      contains: "mod python"
  key_links:
    - from: "src/python/hvac_bindings.rs"
      to: "src/hvac/zone_setpoints.rs"
      via: "PyO3 FFI"
      pattern: "pyo3::wrap_pyfunction"
    - from: "src/python/hvac_bindings.rs"
      to: "src/hvac/zone_control.rs"
      via: "PyO3 FFI"
      pattern: "pyo3::wrap_pyfunction"
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
Enable and verify Python bindings for HVAC functionality

Purpose: Resolve remaining Python bindings issues and verify full functionality
Output: Working Python API for zone-level HVAC control with comprehensive tests
</objective>

<execution_context>
@/home/alex/.config/opencode/get-shit-done/workflows/execute-plan.md
@/home/alex/.config/opencode/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/M2-zone-hvac-controls/M2-02-SUMMARY.md
@.planning/phases/M2-zone-hvac-controls/M2-04-SUMMARY.md
@.planning/phases/M2-zone-hvac-controls/M2-05-SUMMARY.md
@src/physics/cta/mod.rs
@src/thermal/thermal_model.rs
@src/hvac/zone_setpoints.rs
@src/hvac/zone_control.rs
@src/python/hvac_bindings.rs
@src/python/mod.rs
@src/lib.rs
@tests/python/test_hvac_bindings.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Enable HVAC bindings module registration</name>
  <files>src/python/mod.rs, src/python/hvac_bindings.rs</files>
  <action>
    Uncomment and fix HVAC bindings registration:
    
    In src/python/mod.rs:
    - Uncomment the hvac_bindings module declaration
    - Add hvac submodule to PyInit_fluxion() registration
    - Use proper PyO3 registration pattern
    
    In src/python/hvac_bindings.rs:
    - Ensure all PyO3 API usage is correct
    - Fix any remaining VectorField API issues
    - Verify ThermalModel references are correct
    - Ensure proper error handling with PyErr
    
    Add comprehensive logging for debugging if needed.
  </action>
  <verify>
    <automated>cargo check --features python-bindings 2>&1 | grep -E "(error|hvac)" | head -10 || echo "Python bindings check pass"</automated>
  </verify>
  <done>
    HVAC bindings module properly registered
    All PyO3 API usage correct
    No compilation errors in Python bindings
  </done>
</task>

<task type="auto">
  <name>Task 2: Build and test Python bindings</name>
  <files>src/python/hvac_bindings.rs, tests/python/test_hvac_bindings.py</files>
  <action>
    Build Python bindings and run comprehensive tests:
    
    - Run `maturin develop --features python-bindings`
    - Test Python import: `python -c "import fluxion.hvac; print('HVAC module imported successfully')"`
    - Run Python tests: `pytest tests/python/test_hvac_bindings.py -v`
    - Test all HVAC functionality through Python API:
      * ZoneSetpoints creation and configuration
      * ZoneControl creation and operations
      * Energy calculation accuracy
      * Error handling and validation
    
    Fix any issues found during testing.
  </action>
  <verify>
    <automated>maturin develop --features python-bindings && python -c "import fluxion.hvac; print('HVAC module imported successfully')" && pytest tests/python/test_hvac_bindings.py -v</automated>
  </verify>
  <done>
    Python bindings build successfully
    HVAC module imports without errors
    All Python tests pass
    Python API behavior matches Rust implementation
  </done>
</task>

<task type="auto">
  <name>Task 3: Verify end-to-end Python HVAC functionality</name>
  <files>tests/python/test_hvac_bindings.py</files>
  <action>
    Create comprehensive end-to-end test:
    
    - Test multi-zone HVAC control through Python API
    - Verify setpoint configuration works for all zones
    - Test HVAC control logic with various temperature scenarios
    - Validate energy calculations match expected values
    - Test error conditions (invalid zones, temperatures, etc.)
    - Ensure thread safety in multi-zone operations
    
    Add test cases for:
    - 3-zone system with different setpoints
    - Temperature transitions across deadband
    - Energy calculation accuracy (±5% tolerance)
    - Error handling for edge cases
  </action>
  <verify>
    <automated>pytest tests/python/test_hvac_bindings.py::test_end_to_end_hvac_control -v</automated>
  </verify>
  <done>
    End-to-end HVAC control test passing
    Multi-zone functionality verified
    Energy calculations accurate
    Error handling comprehensive
  </done>
</task>

</tasks>

<verification>
- cargo check --features python-bindings succeeds
- maturin develop --features python-bindings completes without errors
- python -c "import fluxion.hvac" succeeds
- pytest tests/python/test_hvac_bindings.py passes all tests
- Python API behavior matches Rust implementation
</verification>

<success_criteria>
- cargo check --features python-bindings completes without errors
- maturin develop --features python-bindings succeeds
- python -c "import fluxion.hvac" executes successfully
- pytest tests/python/test_hvac_bindings.py passes all tests
- All HVAC functionality accessible through Python API
- Energy calculations match between Rust and Python implementations
</success_criteria>

<output>
After completion, create `.planning/phases/M2-zone-hvac-controls/M2-06-SUMMARY.md`
</output>
