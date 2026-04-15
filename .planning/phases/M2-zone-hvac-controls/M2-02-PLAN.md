---
phase: M2-zone-hvac-controls
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - src/python/bindings.rs
  - src/python/hvac_bindings.rs
  - tests/python/test_hvac_bindings.py
autonomous: true
requirements: [MZ-09]

must_haves:
  truths:
    - "Python API exposes zone setpoint configuration"
    - "Python API exposes zone control functionality"
    - "Python bindings match Rust implementation behavior"
  artifacts:
    - path: "src/python/hvac_bindings.rs"
      provides: "Python bindings for HVAC functionality"
      min_lines: 180
    - path: "tests/python/test_hvac_bindings.py"
      provides: "Python binding validation tests"
      min_lines: 100
  key_links:
    - from: "src/python/hvac_bindings.rs"
      to: "src/hvac/zone_setpoints.rs"
      via: "PyO3 FFI"
      pattern: "pyo3::wrap_pyfunction"
    - from: "src/python/hvac_bindings.rs"
      to: "src/hvac/zone_control.rs"
      via: "PyO3 FFI"
      pattern: "pyo3::wrap_pyfunction"
---

<objective>
Extend Python API with multi-zone HVAC bindings

Purpose: Enable Python users to configure and control zone-level HVAC through PyO3 bindings
Output: Complete Python API for multi-zone HVAC operations with validation tests
</objective>

<execution_context>
@/home/alex/.config/opencode/get-shit-done/workflows/execute-plan.md
@/home/alex/.config/opencode/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/M1-multi-zone-foundation/M1-03-SUMMARY.md
@src/python/bindings.rs
@src/hvac/zone_setpoints.rs
@src/hvac/zone_control.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create HVAC Python bindings module</name>
  <files>src/python/hvac_bindings.rs</files>
  <action>
    Create new PyO3 module for HVAC bindings:

    #[pymodule]
    fn hvac(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PyZoneSetpoints>()?;
        m.add_class::<PyZoneControl>()?;
        m.add_function(wrap_pyfunction!(create_zone_setpoints, m)?)?;
        m.add_function(wrap_pyfunction!(create_zone_control, m)?)?;
        Ok(())
    }

    Implement Python wrappers:
    - PyZoneSetpoints: wraps ZoneSetpoints with PyO3 getters/setters
    - PyZoneControl: wraps ZoneControl with control methods
    - create_zone_setpoints(num_zones: usize) -> PyZoneSetpoints
    - create_zone_control(thermal_model: PyThermalModel, setpoints: PyZoneSetpoints) -> PyZoneControl

    Add to main bindings.rs:
    - Register hvac module in init_fluxion()
    - Export hvac module in __init__.py
  </action>
  <verify>
    <automated>maturin develop && python -c "import fluxion.hvac; print('HVAC module imported successfully')"</automated>
  </verify>
  <done>
    HVAC Python module compiles and imports
    All wrapper classes implemented
    Module registered in main bindings
  </done>
</task>

<task type="auto">
  <name>Task 2: Implement PyZoneSetpoints wrapper</name>
  <files>src/python/hvac_bindings.rs</files>
  <action>
    Implement PyZoneSetpoints class:

    #[pyclass]
    struct PyZoneSetpoints {
        inner: ZoneSetpoints,
    }

    #[pymethods]
    impl PyZoneSetpoints {
        #[new]
        fn new(num_zones: usize) -> Self {
            Self { inner: ZoneSetpoints::new(num_zones) }
        }

        fn set_heating_setpoint(&mut self, zone_id: usize, temp: f64) -> PyResult<()>
        fn set_cooling_setpoint(&mut self, zone_id: usize, temp: f64) -> PyResult<()>
        fn set_deadband(&mut self, zone_id: usize, deadband: f64) -> PyResult<()>
        fn get_heating_setpoint(&self, zone_id: usize) -> f64
        fn get_cooling_setpoint(&self, zone_id: usize) -> f64
        fn get_deadband(&self, zone_id: usize) -> f64
        fn validate(&self) -> PyResult<()>
    }

    Add error handling for invalid zone IDs and temperature ranges.
    Return PyErr with descriptive messages for validation failures.
  </action>
  <verify>
    <automated>python -c "import fluxion.hvac; s = fluxion.hvac.ZoneSetpoints(3); s.set_heating_setpoint(0, 21.0); print('Setpoints working')"</automated>
  </verify>
  <done>
    PyZoneSetpoints fully functional
    Error handling implemented
    Validation working correctly
  </done>
</task>

<task type="auto">
  <name>Task 3: Implement PyZoneControl wrapper and create Python tests</name>
  <files>src/python/hvac_bindings.rs, tests/python/test_hvac_bindings.py</files>
  <action>
    Implement PyZoneControl class:

    #[pyclass]
    struct PyZoneControl {
        inner: Arc<Mutex<ZoneControl>>,
    }

    #[pymethods]
    impl PyZoneControl {
        fn update_controls(&self, temperatures: Vec<f64>) -> PyResult<Vec<f64>>
        fn get_zone_status(&self, zone_id: usize) -> String
        fn get_energy_input(&self, zone_id: usize) -> f64
    }

    Create Python test file test_hvac_bindings.py:
    - test_setpoint_creation(): validates ZoneSetpoints creation
    - test_setpoint_getters_setters(): tests all getter/setter methods
    - test_setpoint_validation(): checks error handling
    - test_zone_control_creation(): validates ZoneControl creation
    - test_hvac_control_update(): tests control logic with mock temperatures
    - test_energy_calculation(): verifies energy output values

    Use pytest framework with approximate equality checks.
  </action>
  <verify>
    <automated>pytest tests/python/test_hvac_bindings.py -v</automated>
  </verify>
  <done>
    PyZoneControl wrapper functional
    Python tests passing
    Energy calculations match Rust implementation
  </done>
</task>

</tasks>

<verification>
- Python HVAC module imports successfully
- ZoneSetpoints creation and configuration working
- ZoneControl update and status methods functional
- Python tests validate all functionality
</verification>

<success_criteria>
- maturin develop completes without errors
- python -c "import fluxion.hvac" succeeds
- pytest tests/python/test_hvac_bindings.py passes all tests
- Python API behavior matches Rust implementation
</success_criteria>

<output>
After completion, create `.planning/phases/M2-zone-hvac-controls/M2-02-SUMMARY.md`
</output>
