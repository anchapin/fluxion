# Testing Patterns

**Analysis Date:** 2026-03-08

## Test Framework

**Runner:**
- `cargo test` - Rust's built-in test framework
- `pytest` - Python test framework (for Python bindings)
- Config: `Cargo.toml` (dev-dependencies), `pyproject.toml` ([tool.pytest.ini_options])

**Assertion Library:**
- Rust: Built-in `assert!`, `assert_eq!`, `assert_ne!` macros
- Python: `pytest` assertions

**Benchmarking:**
- `criterion = "0.5"` - Rust benchmarking framework
- Config files in `benches/` directory

**Run Commands:**
```bash
cargo test                          # Run all tests
cargo test -- --nocapture           # Run with output
cargo test -- --test-threads=1      # Single-threaded (for debugging)
cargo test test_thermal_model       # Run specific test
cargo bench                         # Run benchmarks
pytest tests/                       # Run Python tests
```

## Test File Organization

**Location:**
- **Rust unit tests**: Co-located in source files using `#[cfg(test)]` modules
- **Rust integration tests**: Separate files in `tests/` directory
- **Python tests**: Separate files in `tests/*.py`
- **Benchmarks**: Separate files in `benches/` directory

**Naming:**
- Rust unit tests: `mod tests { ... }` within source files
- Rust integration tests: `test_*.rs` or `*_test.rs` (e.g., `test_case_195_heat_balance.rs`)
- Python tests: `test_*.py` (e.g., `test_python_bindings.py`)
- ASHRAE 140 tests: `ashrae_140_*.rs` (e.g., `ashrae_140_integration.rs`)

**Structure:**
```
tests/
├── ashrae_140_integration.rs         # Comprehensive ASHRAE 140 suite
├── ashrae_140_free_floating.rs      # Free-floating tests
├── test_case_195_heat_balance.rs    # Case 195 physics validation
├── test_issue_273_multi_zone_parameters.rs  # Issue-specific tests
├── test_python_bindings.py          # Python binding tests
└── conftest.py                      # Pytest fixtures

src/
├── sim/engine.rs                    # Contains mod tests { ... }
├── physics/cta.rs                   # Contains mod tests { ... }
└── ...                              # 42 files with inline tests

benches/
├── cta_bench.rs                     # CTA performance benchmarks
├── engine_bench.rs                  # Engine performance benchmarks
└── cta_perf_comparison.rs           # Implementation comparisons
```

## Test Structure

**Suite Organization:**

**Rust Unit Tests (co-located):**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_model_creation() {
        let model = ThermalModel::<VectorField>::new(10);
        assert_eq!(model.num_zones, 10);
        assert_eq!(model.temperatures.len(), 10);
    }

    #[test]
    fn test_apply_parameters_updates_model() {
        let mut model = ThermalModel::<VectorField>::new(10);
        let params = vec![1.5, 20.0, 27.0];
        model.apply_parameters(&params);
        assert_eq!(model.window_u_value, 1.5);
    }
}
```

**Rust Integration Tests (separate files):**
```rust
use fluxion::validation::ASHRAE140Validator;

#[test]
fn test_ashrae_140_comprehensive() {
    let mut validator = ASHRAE140Validator::new();
    let report = validator.validate_analytical_engine();
    report.print_summary();

    let pass_rate = report.pass_rate();
    println!("Overall ASHRAE 140 Pass Rate: {:.1}%", pass_rate * 100.0);
}
```

**Python Tests:**
```python
import pytest
import numpy as np
from fluxion import Model, BatchOracle

@pytest.fixture(scope="module")
def model():
    return Model(num_zones=1)

class TestImportModule:
    def test_module_imports(self, fluxion_module):
        assert fluxion_module is not None
        assert hasattr(fluxion_module, "Model")
```

**Patterns:**
- **Setup**: Create model/spec in test body or via fixtures (Python)
- **Teardown**: Not needed (Rust tests clean up automatically)
- **Assertion**: Use `assert_eq!` for exact matches, `assert!` with epsilon for floating-point
- **Epsilon comparison**: `const EPSILON: f64 = 1e-9; assert!(abs_diff < EPSILON)`

## Mocking

**Framework:** No dedicated mocking framework. Uses direct instantiation and conditional compilation.

**Patterns:**

**Surrogate Mocking:**
```rust
// Create mock SurrogateManager (no model loaded)
let surrogates = SurrogateManager::new();

// Load real model for integration tests
if Path::new("models/surrogate.onnx").exists() {
    surrogates.load_onnx("models/surrogate.onnx")?;
}
```

**Conditional Testing:**
```rust
#[test]
#[ignore = "Issue #62 pending merge"]
fn test_case_610_shading() {
    // Test code here
}
```

**Python Fixture:**
```python
@pytest.fixture(scope="module")
def fluxion_module():
    try:
        import fluxion
        return fluxion
    except ImportError:
        pytest.skip("fluxion Python bindings not available")
```

**What to Mock:**
- Surrogate models (use `SurrogateManager::new()` for mock)
- External services (test with local data files)
- File I/O (check file existence before loading)

**What NOT to Mock:**
- Physics calculations (test real implementation)
- Thermal network solver (use real `ThermalModel`)
- Vector field operations (test real CTA implementation)

## Fixtures and Factories

**Test Data:**

**ASHRAE 140 Cases:**
```rust
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

let spec = ASHRAE140Case::Case195.spec();
let model = ThermalModel::<VectorField>::from_spec(&spec);
```

**Parameter Vectors:**
```rust
let params = vec![1.5, 20.0, 27.0]; // [window_u_value, heating_setpoint, cooling_setpoint]
model.apply_parameters(&params);
```

**Weather Data:**
```rust
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

let weather_data = DenverTmyWeather::get_data();
```

**Location:**
- ASHRAE 140 case specs: `src/validation/ashrae_140_cases.rs`
- Weather data: `src/weather/denver.rs`, `src/weather/epw.rs`
- Test data files: `models/` directory (gitignored)

**Python Conftest:**
```python
# tests/conftest.py
@pytest.fixture(scope="module")
def model(fluxion_module):
    return fluxion_module.Model(num_zones=1)

@pytest.fixture(scope="module")
def batch_oracle(fluxion_module):
    return fluxion_module.BatchOracle()
```

## Coverage

**Requirements:** No enforced coverage target, but high coverage expected for critical physics code.

**View Coverage:**
```bash
# Rust coverage (requires tarpaulin or similar)
cargo tarpaulin --out Html

# Python coverage (configured in pyproject.toml)
pytest --cov=fluxion tests/
```

**Coverage Configuration (Python):**
```toml
[tool.coverage.run]
source = ["fluxion"]
omit = ["*test*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

## Test Types

**Unit Tests:**
- **Scope**: Individual functions and methods
- **Approach**: Co-located in `#[cfg(test)]` modules
- **Examples**:
  - `ThermalModel::new()` creates valid model
  - `apply_parameters()` updates model state
  - `VectorField` arithmetic operations

**Integration Tests:**
- **Scope**: Multiple components working together
- **Approach**: Separate files in `tests/` directory
- **Examples**:
  - ASHRAE 140 case validation (full year simulation)
  - Multi-zone parameter application
  - Heat balance calculation verification
  - Python-Rust FFI integration

**E2E Tests:**
- **Framework**: Not used (Rust testing pattern)
- Alternative: ASHRAE 140 integration tests serve as end-to-end validation

**Performance Tests:**
- **Framework**: `criterion` benchmarking
- **Location**: `benches/` directory
- **Examples**:
  - `cta_bench.rs` - Vector field operation performance
  - `engine_bench.rs` - Thermal model solve performance
  - `cta_perf_comparison.rs` - Implementation comparisons

## Common Patterns

**Async Testing:** Not applicable (no async tests in codebase)

**Error Testing:**
```rust
#[test]
fn test_invalid_weights() {
    let result = NeuralScalarField::new(vec![]);
    assert!(result.is_err());
}

#[test]
fn test_validation_error() {
    let controller = IdealHVACController::new(30.0, 20.0); // Invalid: heating > cooling
    let result = controller.validate();
    assert!(result.is_err());
}
```

**Floating-Point Comparison:**
```rust
const EPSILON: f64 = 1e-9;

assert!(
    (model.zone_area[0] - 20.0).abs() < EPSILON,
    "Zone area should be 20.0"
);

assert!(
    model.h_tr_w[0] > 19.0 && model.h_tr_w[0] < 21.0,
    "h_tr_w should be in range [19.0, 21.0]"
);
```

**Issue-Specific Tests:**
```rust
// test_issue_274_thermal_mass.rs
#[test]
fn test_issue_274_thermal_mass_correction() {
    // Test thermal mass energy correction
    // ...
}
```

**Diagnostic Output:**
```rust
#[test]
fn test_heat_balance_calculation() {
    // Run physics calculation
    // ...
    println!("=== Model Conductances ===");
    println!("h_tr_em (envelope-mass): {:.2} W/K", h_tr_em);
    // ...
}
```

## Pre-Commit Test Enforcement

**Hooks:**
- `cargo fmt` - Format check
- `cargo check` - Compilation check
- `cargo audit` - Security vulnerabilities
- `cargo clippy` - Additional lints
- Python: `ruff-check`, `ruff-format`, `black`, `isort`, `mypy`
- Custom: `batch-oracle-pattern`, `rust-doc-check`

**Run manually:**
```bash
pre-commit run --all-files
```

**CI/CD:**
- ASHRAE 140 validation runs on every PR
- All tests must pass before merge

## Test Data Management

**ASHRAE 140 Test Cases:**
- Defined in `src/validation/ashrae_140_cases.rs`
- Enum-based case selection: `ASHRAE140Case::Case195`, `Case600`, etc.
- Reference ranges embedded in test specs

**Weather Data:**
- Denver TMY weather: `src/weather/denver.rs`
- EPW file support: `src/weather/epw.rs`
- Test cases use embedded weather data

**ONNX Models:**
- Test model: `tests_tmp_dummy.onnx` (for CI testing)
- Production models: `models/` directory (gitignored)
- Graceful skip when model not present:
  ```rust
  if Path::new("models/surrogate.onnx").exists() {
      surrogates.load_onnx("models/surrogate.onnx")?;
  } else {
      // Use mock predictions
  }
  ```

## Debugging Tests

**Single-threaded mode:**
```bash
cargo test -- --test-threads=1
```

**Verbose output:**
```bash
cargo test -- --nocapture --show-output
```

**Specific test:**
```bash
cargo test test_thermal_model_creation
```

**Debug mode:**
```bash
cargo test -- --nocapture
# Add debug prints with println!
```

---

*Testing analysis: 2026-03-08*
