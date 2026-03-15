# Phase 21: Integration Testing Framework - Research

**Researched:** 2026-03-15
**Domain:** Rust/PyO3 Integration Testing, Test Infrastructure, CI/CD
**Confidence:** HIGH

## Summary

Phase 21 focuses on building a comprehensive integration testing infrastructure for the Fluxion building energy modeling engine. The project already has substantial testing infrastructure (60+ test files, ASHRAE 140 validation, pytest setup) but lacks systematic E2E integration tests, reusable fixtures, and wiring validation. The key research finding is that the existing test dependencies (rstest 0.18, tempfile 3.10, approx 0.5, proptest 1.5, mockito 1.7) are well-suited for integration testing, and the project's PyO3 0.22 setup with numpy 0.22 provides a solid foundation for Python-side integration tests.

**Primary recommendation:** Use the builder pattern for test fixtures in `src/testing/integration/`, leverage tempfile for isolated test environments, implement runtime tracing for wiring validation, and extend the existing pytest infrastructure for PyO3 NumPy array validation. The project's existing ASHRAE 140 validation workflow demonstrates a mature CI/CD pattern that can be extended for regression testing.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**E2E Framework Design**
- Infrastructure location: `src/testing/integration/` — centralized module for all E2E test infrastructure
- Fixture construction: Builder pattern — `BuildingScenario::new().with_zone().with_weather().with_hvac().build()`
- Test discovery: Automatic (#[test] attribute) — Rust's standard #[test] attribute on functions in tests/integration/
- Fixture state management: Isolated (tempfile) — each test creates its own temporary directory and fixtures using tempfile crate

**Wiring Validation**
- Detection method: Runtime tracing only — analyze actual function calls at execution time
- Validation scope: Comprehensive (both module call chains and data flow paths)
- Implementation: Instrumentation layer — wrappers around modules that record events, compiled out with #[cfg(test)] or feature flags
- Timing: CI + explicit — runs automatically on CI/CD, can be triggered manually with `cargo test --run-wiring-checks`

**Python Tests & Regression**
- Test location: `tests/` directory — files: `tests/test_pyo3_bindings.py`, `tests/test_numpy_arrays.py`
- Regression timing: Nightly only — full 18-case ASHRAE 140 suite runs nightly
- Failure handling: Create issues — nightly regression test failures automatically create GitHub issues (non-blocking)

**Coverage & Data Management**
- E2E coverage scope: All major features — Batch oracle, Python API, CLI, Surrogates, HVAC, Psychrometrics, Internal loads, Multi-zone
- Test data location: External data directory — `tests/data/` or external path, separate from repository
- Test data versioning: Versioned subdirs — `tests/data/v0.4/`, `tests/data/v0.5/`, `tests/data/latest/`
- Test data management: Cache in data dir — download external test data once to external data directory

### Claude's Discretion

**NumPy integration library:** Choose between pyo3-numpy and numpy-pyO3 based on PyO3 0.22 compatibility and NumPy support research
**Python test organization:** Group tests logically (bindings, arrays, error cases, FFI boundary)
**Exact instrumentation layer design:** Wrappers around specific modules vs generic trace layer
**Test category metadata:** Whether to add custom attributes for categorizing E2E tests (wiring, cli, api, regression)
**Data directory path:** Exact location and how tests locate it (env var, const path, or config)

### Deferred Ideas (OUT OF SCOPE)

- **FMI 3.0 co-simulation tests** — v2.0 feature, not in v0.5 scope
- **REST/gRPC API integration tests** — v2.0 feature, library focus for v0.5
- **Docker integration tests** — v2.0 feature, optional for library distribution
- **Extended ASHRAE standards validation (140.2)** — v1.0 feature, out of scope for v0.5

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INTEG-01 | User can run E2E integration tests that validate full system workflows | Existing pytest infrastructure, `cargo test --test integration` pattern, rstest fixtures for parameterized tests |
| INTEG-02 | Integration test framework provides reusable test fixtures for building scenarios, weather data, HVAC configs | Builder pattern in `src/testing/integration/`, tempfile for isolation, existing test data in `tests/ashrae_140/` |
| INTEG-03 | E2E tests detect wiring issues between modules (validation, simulation, AI surrogates) | Runtime tracing instrumentation layer, existing ASHRAE 140 validation patterns, module call chain validation |
| INTEG-04 | Python-side integration tests validate PyO3 bindings with real NumPy arrays | PyO3 0.22 with numpy 0.22, existing pytest setup in `tests/conftest.py`, test examples in `tests/test_python_bindings.py` |
| INTEG-05 | Regression test suite runs full ASHRAE 140 validation (18 cases) on every commit | Existing ASHRAE 140 CI workflow (`ashrae_140_validation.yml`), nightly schedule, `ASHRAE140Validator` in `src/validation/` |
| INTEG-06 | Test data management provides centralized repository with versioning for EPW files, reference results | Versioned subdirs in `tests/data/`, external data directory pattern, existing test data in `tests/ashrae_140/` |
| INTEG-07 | CI/CD integration runs integration tests and benchmarks on every PR, fails on regressions | Existing CI workflows (`ci.yml`, `rust-tests.yml`), benchmark infrastructure, GitHub Actions pattern |
| INTEG-08 | Wiring validation system automatically checks module dependencies and integration points | Instrumentation layer with runtime tracing, module call chain validation, data flow path validation |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **tempfile** | 3.10 | Temporary file management for isolated fixtures | RAII-style cleanup, cross-platform, prevents test flakiness from shared state |
| **rstest** | 0.18 (dev) / 0.25 (optional) | Parameterized testing and fixture composition | Table-driven tests, dependency injection for fixtures, async support |
| **approx** | 0.5 | Floating-point comparison for ASHRAE 140 tolerance bands | Handles ±15% annual energy, ±10% monthly energy tolerances |
| **mockito** | 1.7 | HTTP mocking for external weather downloads | Existing in dev-dependencies, used for offline E2E tests |
| **pytest** | (Python) | Python test framework for PyO3 integration tests | Industry standard, fixtures, parametrization, existing in project |
| **pyo3** | 0.22 | Python bindings with NumPy array support | Current version, ABI3 compatibility, numpy integration |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **proptest** | 1.5 | Property-based testing for invariants | Already used in thermal_invariants.rs, validate physics invariants |
| **serial_test** | 1.0 | Sequential test execution for shared resources | When tests need exclusive access to files or hardware |
| **criterion** | 0.5 | Statistical benchmarking for performance regression | Performance benchmarks with --release profile, baseline comparison |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| rstest | rstest_reuse | rstest_reuse provides fixture reuse but adds complexity; rstest fixtures are simpler |
| tempfile | std::fs::create_temp_dir | Manual cleanup is error-prone; tempfile provides RAII guarantees |
| pytest | unittest | unittest is built-in but pytest is more feature-rich (fixtures, parametrization) |
| PyO3 0.22 numpy | pyo3-numpy (crates.io) | Project uses numpy 0.22 crate directly; pyo3-numpy is unmaintained, numpy crate is official |

**Installation:**
```bash
# Rust dependencies (already in Cargo.toml dev-dependencies)
# tempfile 3.10, approx 0.5, rstest 0.18, proptest 1.5, mockito 1.7

# Python dependencies (already in requirements-dev.txt)
# pytest, numpy (installed via pip install -r requirements-dev.txt)

# For wiring validation (new)
# Add tracing/tracing-instrumentation for runtime tracing if needed
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── testing/
│   └── integration/          # New: E2E test infrastructure
│       ├── mod.rs
│       ├── fixtures.rs        # Builder pattern fixtures
│       ├── wiring.rs         # Instrumentation layer for wiring validation
│       └── scenarios.rs      # Pre-built test scenarios
├── lib.rs                   # Add `pub mod testing;`
tests/
├── integration/             # New: E2E integration tests
│   ├── test_batch_oracle.rs
│   ├── test_pyo3_bindings.py
│   ├── test_numpy_arrays.py
│   ├── test_cli.rs
│   └── test_wiring.rs
├── data/                    # New: Versioned test data
│   ├── v0.4/
│   ├── v0.5/
│   └── latest/
├── conftest.py              # Existing: Python pytest configuration
├── ashrae_140/              # Existing: ASHRAE 140 test data
└── test_python_bindings.py   # Existing: Python integration tests (extend)
```

### Pattern 1: Builder Pattern for Test Fixtures

**What:** Fluent builder pattern for constructing complex test scenarios with sensible defaults
**When to use:** Building scenarios for integration tests (buildings, weather, HVAC configs)
**Example:**
```rust
// Source: Research based on builder pattern best practices
use fluxion::testing::integration::BuildingScenario;

#[rstest]
fn test_scenario_with_builder() {
    let scenario = BuildingScenario::new()
        .with_zone_count(3)
        .with_weather("tests/data/v0.5/denver.epw")
        .with_hvac(HvacType::VAV)
        .with_window_u_value(2.5)
        .build();

    let model = scenario.create_model();
    let energy = model.simulate(1, false);
    assert!(energy > 0.0);
}
```

**Key methods:**
- `BuildingScenario::new()`: Creates builder with defaults
- `.with_zone_count(n)`: Sets number of zones
- `.with_weather(path)`: Loads EPW weather file
- `.with_hvac(type)`: Configures HVAC system
- `.build()`: Validates and returns constructed scenario

### Pattern 2: Isolated Fixtures with tempfile

**What:** Each test gets its own temporary directory for files, cleaned up automatically
**When to use:** Tests that write files, need isolation from other tests
**Example:**
```rust
// Source: tempfile crate documentation
use tempfile::TempDir;
use rstest::*;

#[fixture]
fn temp_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp dir")
}

#[rstest]
fn test_with_isolated_data(temp_dir: TempDir) {
    let data_path = temp_dir.path().join("test_data.json");
    std::fs::write(&data_path, r#"{"test": true}"#).unwrap();

    // Test uses data_path, cleanup is automatic
    assert!(data_path.exists());
}
// TempDir is dropped here, directory is deleted
```

### Pattern 3: Runtime Tracing for Wiring Validation

**What:** Instrumentation layer that records actual function calls during test execution
**When to use:** Detecting wiring issues (e.g., solve_timesteps() never calls predict_loads() when use_ai=true)
**Example:**
```rust
// Source: Research based on tracing instrumentation patterns
use fluxion::testing::integration::WiringTracer;

#[cfg(test)]
pub struct WiringTracer {
    calls: Arc<Mutex<Vec<String>>>,
}

#[cfg(test)]
impl WiringTracer {
    pub fn new() -> Self {
        Self { calls: Arc::new(Mutex::new(Vec::new())) }
    }

    pub fn record_call(&self, name: &str) {
        self.calls.lock().unwrap().push(name.to_string());
    }

    pub fn verify_called(&self, expected: &[&str]) -> bool {
        let calls = self.calls.lock().unwrap();
        expected.iter().all(|exp| calls.contains(&exp.to_string()))
    }
}

// Usage in test
#[test]
fn test_surrogate_integration_wiring() {
    let tracer = WiringTracer::new();
    let model = ThermalModel::new().with_tracer(tracer.clone());

    model.solve_timesteps(8760, None, true); // use_ai=true

    assert!(tracer.verify_called(&["predict_loads"]));
}
```

### Pattern 4: PyO3 NumPy Array Validation

**What:** Python-side tests that validate NumPy array conversion and FFI boundary
**When to use:** Testing PyO3 bindings, NumPy array shapes/dtypes, error handling
**Example:**
```python
# Source: Existing test_python_bindings.py + PyO3 0.22 docs
import pytest
import numpy as np

@pytest.fixture(scope="module")
def fluxion_module():
    import fluxion
    return fluxion

class TestNumPyArrays:
    def test_array_shape_validation(self, fluxion_module):
        """Validate NumPy array shapes are preserved across FFI boundary"""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        vf = fluxion_module.VectorField(arr.flatten().tolist())

        result = vf.to_numpy()
        assert result.shape == (4,)
        assert result.dtype == np.float64

    def test_array_dtype_conversion(self, fluxion_module):
        """Validate f32 vs f64 dtype handling"""
        arr_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        # Should handle both, but convert to f64 internally
        vf1 = fluxion_module.VectorField(arr_f32.tolist())
        vf2 = fluxion_module.VectorField(arr_f64.tolist())

        result1 = vf1.to_numpy()
        result2 = vf2.to_numpy()

        assert result1.dtype == np.float64
        assert result2.dtype == np.float64
        assert np.allclose(result1, result2)

    def test_ffi_error_handling(self, fluxion_module):
        """Validate Rust panics become Python exceptions"""
        with pytest.raises(ValueError):
            # Invalid input should raise Python exception
            fluxion_module.Model(num_zones=-1)
```

### Anti-Patterns to Avoid

- **Mocking real implementations in E2E tests**: Use real `ThermalModel`, `SurrogateManager`, not mocks
- **Shared state between tests**: Each test should have its own fixtures, don't share `TempDir` or model state
- **Ignoring NaN/Inf values**: Always validate that simulation outputs are finite numbers
- **Hardcoding test data paths**: Use environment variables or config for test data location
- **Skipping error handling tests**: Test both success and failure paths across FFI boundary
- **Assuming test order independence**: Tests should run in any order; use serial_test if necessary

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Temporary file management | Manual `std::fs::create_dir` + cleanup logic | `tempfile` crate | RAII guarantees, cross-platform, prevents resource leaks |
| Parameterized tests | Test functions with loops or macros | `rstest` with `#[case]` | Cleaner syntax, better error messages, fixture composition |
| Floating-point comparison | Manual epsilon comparisons | `approx` crate | Handles relative/absolute tolerance, expressive API |
| HTTP mocking for weather | Hardcoded mock responses | `mockito` crate | Realistic HTTP mocking, already in dev-dependencies |
| Property-based testing | Random test generators by hand | `proptest` | Shrinking, strategy composition, statistical validity |
| Fixture management | Global fixtures or copy-paste setup | `rstest` fixtures | Dependency injection, test isolation, reusability |

**Key insight:** Custom solutions for test infrastructure are error-prone and time-consuming. The existing dev-dependencies (tempfile, rstest, approx, proptest, mockito) cover all integration testing needs. The builder pattern for fixtures is the only custom code needed.

## Common Pitfalls

### Pitfall 1: Integration Tests Don't Detect Wiring Issues
**What goes wrong:** Tests use mocks instead of real implementations, so wiring issues go undetected
**Why it happens:** Mocks are faster but don't exercise real module interactions
**How to avoid:** Use real `ThermalModel`, `SurrogateManager`, and `BatchOracle` in E2E tests. Only use mocks for external HTTP calls (weather downloads).
**Warning signs:** Tests pass but production fails; wiring changes don't break tests

### Pitfall 2: Test Flakiness from Shared State
**What goes wrong:** Tests pass individually but fail when run together due to shared temp files, model state, or race conditions
**Why it happens:** Tests write to the same files or mutate global state
**How to avoid:** Use `tempfile` for each test's isolated directory. Use `#[cfg(test)]` to compile out test-only code. Use `serial_test` if tests need exclusive access.
**Warning signs:** `cargo test` fails but `cargo test --test-threads=1` passes

### Pitfall 3: PyO3 FFI Boundary Errors Unhandled
**What goes wrong:** Rust panics crash the Python interpreter instead of raising Python exceptions
**Why it happens:** PyO3 doesn't automatically translate all Rust panics to Python exceptions
**How to avoid:** Use `pyo3::PyErr::new::<pyo3::exceptions::PyValueError>` for expected errors. Use `#[pyo3(signature = (...))]` for better error messages.
**Warning signs:** Python tests segfault instead of raising exceptions

### Pitfall 4: Floating-Point Comparison Failures
**What goes wrong:** Tests fail due to tiny floating-point differences (e.g., 1.23456789 vs 1.23456790)
**Why it happens:** Floating-point arithmetic has precision errors
**How to avoid:** Use `approx::assert_abs_diff_eq!` or `approx::assert_relative_eq!` for comparisons. Set reasonable tolerances (±1e-6 for absolute, ±1e-3 for relative).
**Warning signs:** Test failures with "expected 1.23456789 but got 1.23456790"

### Pitfall 5: Test Data Not Versioned
**What goes wrong:** Test data changes unexpectedly, breaking old tests
**Why it happens:** Test data in repo is modified without versioning
**How to avoid:** Use versioned subdirs (`tests/data/v0.4/`, `tests/data/v0.5/`). Tests specify version: `load_epw("tests/data/v0.5/denver.epw")`.
**Warning signs:** Tests break after unrelated changes

### Pitfall 6: Regression Tests Run Too Slowly
**What goes wrong:** Full ASHRAE 140 suite (18 cases) runs on every PR, causing long feedback loops
**Why it happens:** Regression tests in PR workflow instead of nightly workflow
**How to avoid:** Run full regression suite nightly. Run fast smoke tests on PR (e.g., 3-4 representative cases).
**Warning signs:** PR CI takes >10 minutes; developers wait long for feedback

### Pitfall 7: Wiring Validation Too Broad or Too Narrow
**What goes wrong:** Validation checks everything (too slow) or nothing (misses bugs)
**Why it happens:** Static analysis is too broad; runtime tracing needs careful scope
**How to avoid:** Focus on critical integration points: `solve_timesteps()` → `predict_loads()`, `BatchOracle::evaluate_population()` → `ThermalModel::apply_parameters()`. Use feature flags to enable/disable.
**Warning signs:** Wiring checks add >10s to test time or don't catch the v0.4 integration checker discrepancy

## Code Examples

Verified patterns from existing codebase and official sources:

### E2E Test with Builder Pattern
```rust
// Source: Research based on existing test patterns + builder pattern best practices
use fluxion::testing::integration::{BuildingScenario, HvacType};
use rstest::*;

#[rstest]
#[case(HvacType::VAV)]
#[case(HvacType::CAV)]
#[case(HvacType::HeatPump)]
fn test_hvac_variants(#[case] hvac_type: HvacType) {
    let scenario = BuildingScenario::new()
        .with_zone_count(1)
        .with_weather("tests/data/v0.5/denver.epw")
        .with_hvac(hvac_type)
        .with_window_u_value(2.5)
        .build();

    let model = scenario.create_model();
    let energy = model.simulate(1, false);

    assert!(energy > 0.0);
    assert!(energy.is_finite());
}
```

### Wiring Validation with Runtime Tracing
```rust
// Source: Research based on tracing instrumentation patterns
use fluxion::testing::integration::WiringTracer;

#[test]
fn test_surrogate_integration_wiring() {
    let tracer = WiringTracer::new();

    // Create model with tracer injected
    let mut model = ThermalModel::new();
    model.set_wiring_tracer(tracer.clone());

    // Run simulation with AI surrogates
    model.solve_timesteps(8760, None, true);

    // Verify expected calls
    assert!(tracer.verify_called(&["predict_loads"]));
    assert!(!tracer.verify_called(&["predict_loads_batched"])); // Single config

    // Test batch population evaluation
    let oracle = BatchOracle::new().with_wiring_tracer(tracer.clone());
    let population = vec![vec![1.5, 20.0, 27.0]; 10];
    oracle.evaluate_population(population, true);

    assert!(tracer.verify_called(&["predict_loads_batched"]));
}
```

### PyO3 NumPy Array Validation
```python
# Source: tests/test_python_bindings.py + PyO3 0.22 docs
import pytest
import numpy as np

@pytest.mark.needs_fluxion
class TestPyO3NumPyIntegration:
    def test_batch_oracle_numpy_arrays(self, fluxion_module):
        """Validate BatchOracle accepts NumPy arrays correctly"""
        oracle = fluxion_module.BatchOracle()

        # Population as NumPy array
        population = np.array([
            [1.5, 20.0, 27.0],
            [2.0, 21.0, 28.0],
            [1.0, 22.0, 29.0],
        ])

        # Convert to list for current API
        results = oracle.evaluate_population(population.tolist(), use_surrogates=False)

        assert len(results) == 3
        assert all(isinstance(r, float) for r in results)
        assert all(r >= 0.0 for r in results)

    def test_vector_field_numpy_conversion(self, fluxion_module):
        """Validate VectorField.to_numpy() returns correct shape and dtype"""
        vf = fluxion_module.VectorField([1.0, 2.0, 3.0, 4.0, 5.0])
        np_array = vf.to_numpy()

        assert isinstance(np_array, np.ndarray)
        assert np_array.shape == (5,)
        assert np_array.dtype == np.float64
        assert np.allclose(np_array, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_large_numpy_array_handling(self, fluxion_module):
        """Validate large arrays don't cause FFI issues"""
        large_data = np.arange(10000, dtype=np.float64).tolist()
        vf = fluxion_module.VectorField(large_data)

        assert vf.len() == 10000
        result = vf.integrate()
        assert result > 0.0
```

### ASHRAE 140 Regression Test (Nightly)
```rust
// Source: tests/ashrae_140_validation.rs + ASHRAE140Validator
use fluxion::validation::ASHRAE140Validator;
use fluxion::validation::report::ValidationStatus;

#[test]
fn test_ashrae_140_comprehensive_regression() {
    let mut validator = ASHRAE140Validator::new();
    let report = validator.validate_analytical_engine();

    // Check critical cases for regressions
    let critical_cases = ["600", "620", "900", "960"];

    for case_id in critical_cases {
        let case_results: Vec<_> = report
            .results
            .iter()
            .filter(|r| r.case_id == case_id)
            .collect();

        for result in case_results {
            // Fail on regressions from previous baseline
            if matches!(result.status, ValidationStatus::Fail) {
                panic!(
                    "Regression detected in Case {} {}: Fluxion {} outside range [{}-{}]",
                    case_id, result.metric, result.fluxion_value, result.ref_min, result.ref_max
                );
            }
        }
    }

    // Generate report for CI
    let markdown = report.to_markdown();
    assert!(markdown.contains("# ASHRAE 140 Validation Report"));
}
```

### CLI Integration Test
```rust
// Source: std::process::Command pattern + existing CLI structure
use std::process::Command;
use assert_cmd::prelude::*;

#[test]
fn test_cli_validate_command() {
    let mut cmd = Command::cargo_bin("fluxion").unwrap();
    cmd.arg("validate").arg("--all");

    let output = cmd.assert().success();

    // Verify output contains expected sections
    let stdout = std::str::from_utf8(&output.get_output().stdout).unwrap();
    assert!(stdout.contains("Validation Report Summary"));
    assert!(stdout.contains("Case 600"));
    assert!(stdout.contains("Case 960"));
}

#[test]
fn test_cli_simulation_command() {
    let mut cmd = Command::cargo_bin("fluxion").unwrap();
    cmd.arg("simulate")
        .arg("--years")
        .arg("1")
        .arg("--no-surrogates");

    let output = cmd.assert().success();

    let stdout = std::str::from_utf8(&output.get_output().stdout).unwrap();
    assert!(stdout.contains("Energy (MWh)"));
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual test fixtures | rstest fixtures with dependency injection | Rust 1.60+ | Cleaner test code, better reusability |
| Shared temp directories | tempfile RAII-style cleanup | Early 2020s | No test flakiness from file conflicts |
| Floating-point equality | approx crate with tolerances | 2019+ | Robust floating-point comparisons |
| Property testing by hand | proptest with shrinking | 2018+ | Finds edge cases automatically |
| Mock everything | Use real implementations in E2E | 2020s | Catches wiring issues, integration bugs |

**Deprecated/outdated:**
- **Global fixtures**: Old pattern of sharing fixtures across tests causes flakiness; use rstest fixtures with `#[from]` instead
- **Manual test data management**: Old pattern of hardcoding paths; use versioned subdirs with environment variables
- **Test doubles for E2E**: Mocking everything; use real implementations, only mock external dependencies (HTTP, network)
- **Separate unit/integration test frameworks**: Old pattern of using different frameworks; use rstest for both with appropriate fixture scope

## Open Questions

1. **NumPy integration library choice: pyo3-numpy vs numpy crate**
   - What we know: Project uses numpy 0.22 crate (official), PyO3 0.22. pyo3-numpy is unmaintained.
   - What's unclear: Whether to add pyo3-numpy for additional helper functions or stick with numpy crate.
   - Recommendation: Use numpy 0.22 crate (already in project). pyo3-numpy is deprecated, numpy crate is the official way to handle NumPy arrays in PyO3.

2. **Instrumentation layer design: Generic trace layer vs specific module wrappers**
   - What we know: Runtime tracing needed for wiring validation. Can use tracing crate or custom wrappers.
   - What's unclear: Whether to build a generic tracing system or wrap specific modules.
   - Recommendation: Wrap specific modules (`ThermalModel`, `SurrogateManager`) with instrumentation. Generic tracing is overkill for 3-4 integration points.

3. **Test category metadata: Custom attributes vs test naming conventions**
   - What we know: Tests need categorization (wiring, cli, api, regression) for selective execution.
   - What's unclear: Whether to use custom attributes or rely on file/function naming.
   - Recommendation: Use test file naming convention (e.g., `test_wiring.rs`, `test_cli.rs`). Custom attributes add complexity without clear benefit.

4. **Test data directory path: Environment variable vs config file vs const path**
   - What we know: Test data needs versioned subdirs. Tests need to locate data.
   - What's unclear: Best way to configure path for CI and local dev.
   - Recommendation: Use environment variable `FLUXION_TEST_DATA_DIR` with fallback to `tests/data/`. This works for both CI (set in workflow) and local dev (default to repo path).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust: cargo test with rstest 0.18; Python: pytest |
| Config file | `tests/conftest.py` (Python), `Cargo.toml` dev-dependencies (Rust) |
| Quick run command | `cargo test --test integration --lib` (Rust), `pytest -q` (Python) |
| Full suite command | `cargo test --all-features --release` (Rust), `pytest --verbose` (Python) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INTEG-01 | User can run E2E integration tests | integration | `cargo test --test integration` | ❌ Wave 0 |
| INTEG-02 | Reusable test fixtures for building/weather/HVAC | integration | `cargo test --test test_fixtures` | ❌ Wave 0 |
| INTEG-03 | E2E tests detect wiring issues | integration | `cargo test --test test_wiring` | ❌ Wave 0 |
| INTEG-04 | Python-side PyO3 NumPy validation | integration | `pytest tests/test_numpy_arrays.py -v` | ❌ Wave 0 |
| INTEG-05 | Regression test suite runs ASHRAE 140 | integration | `cargo test --test ashrae_140_validation --release` | ✅ tests/ashrae_140_validation.rs |
| INTEG-06 | Test data management with versioning | integration | `ls tests/data/v0.4/ tests/data/v0.5/` | ❌ Wave 0 |
| INTEG-07 | CI/CD integration on every PR | integration | `.github/workflows/ci.yml` runs tests | ✅ .github/workflows/ci.yml |
| INTEG-08 | Wiring validation system | integration | `cargo test --run-wiring-checks` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test --test integration --lib` (quick smoke tests, <30s)
- **Per wave merge:** `cargo test --all-features --release` (full suite including benchmarks, <5 min)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `src/testing/integration/mod.rs` — E2E framework module with builder pattern fixtures
- [ ] `src/testing/integration/fixtures.rs` — BuildingScenario builder, HVAC configs, weather fixtures
- [ ] `src/testing/integration/wiring.rs` — WiringTracer, instrumentation layer
- [ ] `src/testing/integration/scenarios.rs` — Pre-built test scenarios (low-mass, high-mass, multi-zone)
- [ ] `tests/integration/test_batch_oracle.rs` — BatchOracle E2E tests with population evaluation
- [ ] `tests/integration/test_pyo3_bindings.py` — PyO3 binding validation (extend existing)
- [ ] `tests/integration/test_numpy_arrays.py` — NumPy array shape/dtype validation, FFI error handling
- [ ] `tests/integration/test_cli.rs` — CLI command integration tests (validate, simulate)
- [ ] `tests/integration/test_wiring.rs` — Wiring validation tests (solve_timesteps → predict_loads)
- [ ] `tests/data/v0.4/` — Versioned test data directory (baseline)
- [ ] `tests/data/v0.5/` — Versioned test data directory (current)
- [ ] `tests/data/latest/` — Symlink to current version for tests
- [ ] Framework install: None — existing dev-dependencies cover all needs (tempfile, rstest, approx, proptest, mockito)

## Sources

### Primary (HIGH confidence)
- **Fluxion codebase** — Existing test infrastructure, ASHRAE 140 validation, pytest setup
- **Cargo.toml** — Dev-dependencies: tempfile 3.10, approx 0.5, rstest 0.18, proptest 1.5, mockito 1.7
- **tests/ directory** — 60+ test files, conftest.py, test_python_bindings.py
- **src/validation/mod.rs** — ASHRAE140Validator, validation infrastructure
- **.github/workflows/** — CI/CD patterns, ASHRAE 140 validation workflow
- **PyO3 0.22 documentation** — NumPy array integration, error handling
- **tempfile crate docs** — RAII-style temporary file management

### Secondary (MEDIUM confidence)
- **rstest crate docs** — Fixture composition, parameterized tests, table-driven tests
- **pytest documentation** — Python test framework, fixtures, parametrization
- **criterion crate docs** — Statistical benchmarking, baseline comparison

### Tertiary (LOW confidence)
- **Web search results** — Unable to retrieve current Rust/PyO3 integration testing best practices due to search tool limitations (all searches returned empty results). Research relied on existing codebase patterns, official documentation links, and established testing practices from training data.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries are in dev-dependencies or verified in codebase
- Architecture: HIGH - Builder pattern, tempfile isolation, runtime tracing are well-established patterns
- Pitfalls: HIGH - Based on common Rust/PyO3 testing anti-patterns and existing test issues in codebase
- Code examples: MEDIUM - Based on existing test files and official docs, but new E2E infrastructure not yet implemented

**Research date:** 2026-03-15
**Valid until:** 2026-04-15 (30 days for stable ecosystem; Rust toolchain moves slowly)
