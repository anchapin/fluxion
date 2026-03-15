# Phase 11: API & Robustness - Research

**Researched:** 2026-03-13
**Domain:** PyO3 Python bindings API design, error handling, input validation, and robustness patterns
**Confidence:** MEDIUM

## Summary

Phase 11 focuses on simplifying the Python API and strengthening error handling for Fluxion's Rust-based Building Energy Modeling engine. The current API has several usability issues:

1. **Error Handling**: Uses generic `PyRuntimeError` and `PyValueError` without domain-specific exception types
2. **Parameter Validation**: Bounds checking exists (`MIN_U_VALUE`, `MAX_U_VALUE`, etc.) but is not exposed to Python users
3. **Type Safety**: Parameter vectors use `Vec<f64>` which is error-prone and lacks semantic meaning
4. **Validation**: `validate_parameters()` exists in Rust but is not accessible from Python
5. **Error Recovery**: ONNX Runtime failures are not handled with fallback to analytical mode
6. **Logging**: Limited control over logging verbosity from Python
7. **Extreme Data**: No explicit handling for out-of-range weather data

**Primary recommendation:** Implement a comprehensive API polishing strategy that includes custom PyO3 exception types, parameter bounds exposure, dedicated parameter validation methods, and robust error recovery for ONNX failures—all while maintaining backward compatibility for existing `BatchOracle` and `Model` APIs.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `pyo3` | 0.22 | Python bindings, error handling, type conversion | De facto standard for Rust-Python FFI; supports custom exceptions, `FromPyObject` trait, seamless numpy integration |
| `thiserror` | 1.0 (already in deps) | Rust error type definitions | Provides derive macros for custom error types with Display/Error trait implementations |
| `anyhow` | 1.0 (already in deps) | Error chain composition | Simplifies error propagation across module boundaries |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pyo3-log` | Latest (if needed) | Python logging bridge | When bridging Rust `log` crate to Python's `logging` module |
| `tracing` | Latest (if needed) | Structured logging | When replacing `env_logger` for more granular control |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `pyo3` | `cpython` | PyO3 has better Rust ergonomics, safer FFI, better documentation |
| `thiserror` | Manual `impl Display/Error` | `thiserror` reduces boilerplate and ensures consistent error formatting |

**Installation:**
```bash
# All dependencies already in Cargo.toml
# No additional installation needed
```

## Architecture Patterns

### Recommended API Structure

```
src/lib.rs (Python bindings)
├── Custom Exception Definitions
│   ├── FluxionError (base exception)
│   ├── ValidationError (parameter bounds, NaN/Inf)
│   ├── SurrogateError (ONNX Runtime failures)
│   └── SimulationError (physics failures, NaN propagation)
├── BatchOracle API
│   ├── new()
│   ├── evaluate_population()
│   ├── evaluate_population_numpy()
│   ├── get_parameter_bounds() (NEW)
│   ├── validate_parameters() (NEW)
│   └── ParameterBounds struct (NEW)
└── Model API
    ├── new()
    ├── simulate()
    ├── load_surrogate()
    ├── get_parameter_bounds() (NEW)
    ├── validate_parameters() (NEW)
    └── ParameterBounds struct (NEW)
```

### Pattern 1: Custom PyO3 Exception Types

**What:** Define domain-specific exceptions that map to Python's exception hierarchy

**When to use:** When error conditions have semantic meaning beyond generic runtime/value errors

**Example:**
```rust
// Source: PyO3 documentation pattern for custom exceptions
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::PyResult;
use thiserror::Error;

// Define Rust-side error types
#[derive(Error, Debug)]
pub enum FluxionError {
    #[error("Parameter validation failed: {0}")]
    Validation(String),

    #[error("ONNX Runtime error: {0}")]
    Surrogate(String),

    #[error("Simulation error: {0}")]
    Simulation(String),
}

// Create Python exceptions
create_exception!(fluxion, FluxionError, PyException);
create_exception!(fluxion, ValidationError, FluxionError);
create_exception!(fluxion, SurrogateError, FluxionError);
create_exception!(fluxion, SimulationError, FluxionError);

// Convert Rust errors to Python exceptions
impl From<FluxionError> for PyErr {
    fn from(err: FluxionError) -> PyErr {
        match err {
            FluxionError::Validation(msg) => ValidationError::new_err(msg.to_string()),
            FluxionError::Surrogate(msg) => SurrogateError::new_err(msg.to_string()),
            FluxionError::Simulation(msg) => SimulationError::new_err(msg.to_string()),
        }
    }
}

#[pymodule]
fn fluxion(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register exceptions
    m.add("FluxionError", _py.get_type_bound::<FluxionError>())?;
    m.add("ValidationError", _py.get_type_bound::<ValidationError>())?;
    m.add("SurrogateError", _py.get_type_bound::<SurrogateError>())?;
    m.add("SimulationError", _py.get_type_bound::<SimulationError>())?;
    Ok(())
}
```

### Pattern 2: Parameter Bounds Exposure

**What:** Expose design variable ranges as a structured Python object

**When to use:** When users need to know valid parameter ranges for optimization

**Example:**
```rust
#[cfg(feature = "python-bindings")]
#[pyclass]
#[derive(Clone)]
pub struct ParameterBounds {
    #[pyo3(get)]
    pub min_u_value: f64,

    #[pyo3(get)]
    pub max_u_value: f64,

    #[pyo3(get)]
    pub min_heating_setpoint: f64,

    #[pyo3(get)]
    pub max_heating_setpoint: f64,

    #[pyo3(get)]
    pub min_cooling_setpoint: f64,

    #[pyo3(get)]
    pub max_cooling_setpoint: f64,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl ParameterBounds {
    #[staticmethod]
    fn get_bounds() -> Self {
        ParameterBounds {
            min_u_value: BatchOracle::MIN_U_VALUE,
            max_u_value: BatchOracle::MAX_U_VALUE,
            min_heating_setpoint: BatchOracle::MIN_HEATING_SETPOINT,
            max_heating_setpoint: BatchOracle::MAX_HEATING_SETPOINT,
            min_cooling_setpoint: BatchOracle::MIN_COOLING_SETPOINT,
            max_cooling_setpoint: BatchOracle::MAX_COOLING_SETPOINT,
        }
    }
}

// In BatchOracle Python API
#[pymethods]
impl BatchOracle {
    fn get_parameter_bounds(&self) -> ParameterBounds {
        ParameterBounds::get_bounds()
    }

    fn validate_parameters(&self, params: Vec<f64>) -> PyResult<()> {
        Self::validate_parameters(&params)
            .map_err(|e| FluxionError::Validation(e))
            .map_err(|e| e.into())
    }
}
```

### Pattern 3: Dedicated Parameter Type

**What:** Replace `Vec<f64>` with a typed struct that enforces semantic meaning

**When to use:** When parameter vectors have fixed structure and semantic meaning

**Example:**
```rust
#[cfg(feature = "python-bindings")]
#[pyclass]
#[derive(Clone)]
pub struct BuildingParameters {
    #[pyo3(get, set)]
    pub window_u_value: f64,

    #[pyo3(get, set)]
    pub heating_setpoint: f64,

    #[pyo3(get, set)]
    pub cooling_setpoint: f64,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl BuildingParameters {
    #[new]
    fn new(
        window_u_value: f64,
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> PyResult<Self> {
        // Validate on construction
        if window_u_value < BatchOracle::MIN_U_VALUE || window_u_value > BatchOracle::MAX_U_VALUE {
            return Err(ValidationError::new_err(format!(
                "Window U-value must be between {} and {} W/m²K",
                BatchOracle::MIN_U_VALUE,
                BatchOracle::MAX_U_VALUE
            )));
        }
        // ... other validations

        Ok(BuildingParameters {
            window_u_value,
            heating_setpoint,
            cooling_setpoint,
        })
    }

    fn to_vec(&self) -> Vec<f64> {
        vec![
            self.window_u_value,
            self.heating_setpoint,
            self.cooling_setpoint,
        ]
    }
}

// Maintain backward compatibility: accept both Vec<f64> and BuildingParameters
impl TryFrom<BuildingParameters> for Vec<f64> {
    type Error = String;

    fn try_from(params: BuildingParameters) -> Result<Self, Self::Error> {
        Ok(params.to_vec())
    }
}
```

### Pattern 4: Error Recovery with Fallback

**What:** Wrap ONNX inference with fallback to analytical mode on failure

**When to use:** When surrogate failures should not crash the simulation

**Example:**
```rust
impl SurrogateManager {
    pub fn predict_loads_with_fallback(&self, temps: &[f64]) -> Result<Vec<f64>, FluxionError> {
        match self.predict_loads_batched(&[temps.to_vec()]) {
            Ok(mut loads) => Ok(loads.pop().unwrap()),
            Err(e) => {
                // Log the error but don't fail
                log::warn!("ONNX inference failed, falling back to analytical: {}", e);

                // Fallback to analytical calculation
                self.analytical_loads(temps)
                    .map_err(|e| FluxionError::Surrogate(format!(
                        "ONNX and analytical fallback both failed: {}",
                        e
                    )))
            }
        }
    }

    fn analytical_loads(&self, temps: &[f64]) -> Result<Vec<f64>, String> {
        // Implement analytical load calculation
        // ... (existing analytical logic from Phase 8/9)
        Ok(vec![1.2; temps.len()]) // Mock for example
    }
}
```

### Anti-Patterns to Avoid

- **Generic PyRuntimeError for all errors**: Users cannot catch specific error types for different handling strategies
- **Hardcoded parameter bounds in docs only**: Bounds should be accessible programmatically
- **No Python-side validation**: Let invalid parameters cross FFI boundary only to fail in Rust
- **Silent ONNX failures**: Errors should be logged, not silently ignored
- **Breaking BatchOracle API**: Maintain `Vec<Vec<f64>>` signature for backward compatibility; add typed APIs alongside

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Custom error formatting | Manual string concatenation | `thiserror` derive macro | Reduces boilerplate, ensures consistent `Display/Error` implementations |
| Parameter validation logic | Custom validator functions | `thiserror` + PyO3 `FromPyObject` | Leverages PyO3's type conversion system |
| Exception hierarchy | Manual exception registration | `pyo3::create_exception!` macro | Handles Python exception base class relationships automatically |
| Logging bridge | Manual `print!` statements | `pyo3-log` or `tracing` | Integrates with Python's `logging` module, supports log levels |
| Fallback logic | Manual try-catch patterns | `Result` combinator methods | Rust's `Result` type with `?` operator provides clean error propagation |

**Key insight:** PyO3 provides robust infrastructure for Python-Rust interop. Hand-rolling solutions creates maintenance burden and misses edge cases that the framework already handles.

## Common Pitfalls

### Pitfall 1: Inconsistent Error Types
**What goes wrong:** Mixing `PyRuntimeError`, `PyValueError`, and custom exceptions makes it hard for users to catch specific errors
**Why it happens:** Legacy code using generic PyO3 exceptions without domain-specific structure
**How to avoid:** Define a clear exception hierarchy from the start and enforce its use via code review
**Warning signs:** Multiple `PyRuntimeError::new_err` calls with different semantic meanings

### Pitfall 2: Bounds Not Exposed to Python
**What goes wrong:** Python users hardcode bounds from docs instead of using programmatic APIs
**Why it happens:** Bounds are Rust constants not exported to Python module
**How to avoid:** Create `get_parameter_bounds()` method and document it in Python API
**Warning signs:** Python test files with hardcoded `MIN_U_VALUE = 0.1` constants

### Pitfall 3: No NaN/Inf Detection in Validation
**What goes wrong:** `NaN` or `Inf` values pass range checks but cause physics failures later
**Why it happens:** Range checks like `value >= min && value <= max` don't catch `NaN`
**How to avoid:** Use `value.is_finite()` before range validation
**Warning signs:** Simulation returning `NaN` energy values without clear error messages

### Pitfall 4: Breaking Backward Compatibility
**What goes wrong:** Existing Python code breaks after API changes
**Why it happens:** Removing or renaming methods without maintaining old signatures
**How to avoid:** Add new methods alongside old ones; mark old ones `#[deprecated]` if needed
**Warning signs:** User complaints after release; CI tests failing on old API usage

### Pitfall 5: Not Testing Error Paths
**What goes wrong:** Error handling code has bugs that only manifest in production
**Why it happens:** Focus on happy path testing; error paths untested
**How to avoid:** Write tests for all error conditions, including ONNX failures
**Warning signs:** Low test coverage on `#[pymethods]` that return `PyResult`

## Code Examples

Verified patterns from PyO3 and best practices:

### Custom Exception Registration
```rust
// Source: PyO3 documentation (https://pyo3.rs/)
use pyo3::create_exception;

// Create exception hierarchy
create_exception!(fluxion, FluxionError, pyo3::exceptions::PyException);
create_exception!(fluxion, ValidationError, FluxionError);
create_exception!(fluxion, SurrogateError, FluxionError);

#[pymodule]
fn fluxion(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("FluxionError", _py.get_type_bound::<FluxionError>())?;
    m.add("ValidationError", _py.get_type_bound::<ValidationError>())?;
    m.add("SurrogateError", _py.get_type_bound::<SurrogateError>())?;
    Ok(())
}
```

### Parameter Validation with Friendly Errors
```rust
#[pymethods]
impl BatchOracle {
    fn validate_parameters(&self, params: Vec<f64>) -> PyResult<()> {
        // Check for NaN/Inf first
        for (i, &val) in params.iter().enumerate() {
            if !val.is_finite() {
                return Err(ValidationError::new_err(format!(
                    "Parameter at index {} is not finite (got: {})",
                    i, val
                )));
            }
        }

        // Check bounds
        if let Some(&u_value) = params.get(Self::U_VALUE_INDEX) {
            if !(Self::MIN_U_VALUE..=Self::MAX_U_VALUE).contains(&u_value) {
                return Err(ValidationError::new_err(format!(
                    "Window U-value ({:.2} W/m²K) out of range [{:.1}, {:.1}]",
                    u_value, Self::MIN_U_VALUE, Self::MAX_U_VALUE
                )));
            }
        }

        // ... other validations

        Ok(())
    }
}
```

### ONNX Error Recovery
```rust
impl SurrogateManager {
    pub fn predict_loads_with_recovery(
        &self,
        temps: &[f64],
    ) -> Result<Vec<f64>, FluxionError> {
        // Try ONNX first
        if let Some(pool) = &self.session_pool {
            match pool.predict_loads_batched(&[temps.to_vec()]) {
                Ok(mut loads) => {
                    log::info!("ONNX inference successful");
                    return Ok(loads.pop().unwrap());
                }
                Err(e) => {
                    log::warn!("ONNX inference failed: {}", e);
                    // Fall through to analytical
                }
            }
        }

        // Fallback to analytical
        log::info!("Using analytical load calculation");
        self.analytical_loads(temps)
            .map_err(|e| FluxionError::Surrogate(e))
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Generic `PyRuntimeError` for all errors | Domain-specific exception hierarchy | PyO3 0.13+ (2019) | Users can catch specific errors for different handling |
| Hardcoded parameter bounds in docs | `get_parameter_bounds()` method | Industry best practice | Programmatic discovery of valid parameter ranges |
| `Vec<f64>` for parameters | Optional typed `BuildingParameters` struct | PyO3 type system evolution | Better IDE support, compile-time checks |
| Silent ONNX failures | Fallback to analytical + logging | Production reliability pattern | Graceful degradation, better debugging |

**Deprecated/outdated:**
- Manual exception registration without `create_exception!` macro (PyO3 < 0.7)
- String-based error codes instead of exception types (Python 2 era)
- Silent failures in FFI boundary (pre-PyO3 error handling)

## Open Questions

1. **Should `BuildingParameters` replace `Vec<f64>` or coexist?**
   - What we know: Replacing would be breaking change; coexistence maintains backward compatibility
   - What's unclear: User adoption patterns and migration strategy
   - Recommendation: Coexist for v0.3; consider deprecating `Vec<f64>` in v1.0

2. **Should `validate_parameters()` return a boolean or raise exception?**
   - What we know: Pythonic APIs should raise exceptions on invalid input
   - What's unclear: Performance impact of exception-based validation
   - Recommendation: Raise `ValidationError` exception for consistency

3. **How detailed should error messages be?**
   - What we know: Need balance between helpfulness and verbosity
   - What's unclear: User preference for short vs detailed messages
   - Recommendation: Include parameter index, value, and valid range in validation errors

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo (native Rust testing) |
| Config file | None (uses default Cargo.toml config) |
| Quick run command | `cargo test -p fluxion api::` |
| Full suite command | `cargo test -p fluxion` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| API-01 | Custom exception types for BatchOracle | unit | `cargo test -p fluxion test_custom_exceptions -x` | ❌ NEW |
| API-02 | get_parameter_bounds() method | integration | `cargo test -p fluxion test_parameter_bounds -x` | ❌ NEW |
| API-03 | BuildingParameters struct type safety | unit | `cargo test -p fluxion test_building_parameters -x` | ❌ NEW |
| API-04 | validate_parameters() in Python API | integration | `cargo test -p fluxion test_python_validate -x` | ❌ NEW |
| API-05 | Standardize return types across APIs | unit | `cargo test -p fluxion test_return_types -x` | ❌ NEW |
| ROBUST-01 | NaN/Inf detection in apply_parameters | unit | `cargo test -p fluxion test_nan_detection -x` | ❌ NEW |
| ROBUST-02 | ONNX fallback to analytical mode | integration | `cargo test -p fluxion test_onnx_fallback -x` | ❌ NEW |
| ROBUST-03 | Logging verbosity control | unit | `cargo test -p fluxion test_logging_control -x` | ❌ NEW |
| ROBUST-04 | Extreme weather data handling | integration | `cargo test -p fluxion test_extreme_weather -x` | ❌ NEW |
| ROBUST-05 | Thread-safe SessionPool initialization | unit | `cargo test -p fluxion test_session_pool_thread_safe -x` | ❌ NEW |
| BUG-03 | Correct misleading error messages | unit | `cargo test -p fluxion test_error_messages -x` | ❌ NEW |

### Sampling Rate
- **Per task commit:** `cargo test -p fluxion api:: robust::` (quick subset)
- **Per wave merge:** `cargo test -p fluxion` (full suite)
- **Phase gate:** All 12 test files passing + manual Python API testing

### Wave 0 Gaps
- [ ] `tests/test_api_exceptions.rs` — covers API-01
- [ ] `tests/test_parameter_bounds.rs` — covers API-02
- [ ] `tests/test_building_parameters.rs` — covers API-03
- [ ] `tests/test_python_validation.rs` — covers API-04
- [ ] `tests/test_return_types.rs` — covers API-05
- [ ] `tests/test_robustness.rs` — covers ROBUST-01, ROBUST-02, ROBUST-03, ROBUST-04, ROBUST-05
- [ ] `tests/test_bug_03_error_messages.rs` — covers BUG-03

## Sources

### Primary (HIGH confidence)
- PyO3 0.22 Documentation - Error handling, custom exceptions, `create_exception!` macro
- PyO3 0.22 Documentation - `FromPyObject` trait for type conversion and validation
- PyO3 0.22 Documentation - `#[pymodule]` and exception registration
- Fluxion source code (`src/lib.rs`) - Current error handling patterns (lines 63, 80, 118, 499, 801, 838, 874, 1005, 1736, 1794-1818)
- Fluxion source code (`src/lib.rs`) - Current parameter validation (lines 526-527, 539-576)

### Secondary (MEDIUM confidence)
- PyO3 GitHub Repository - Examples of custom exception hierarchies
- thiserror crate documentation - Derive macros for error types
- Python logging module documentation - Integration patterns with Rust logging

### Tertiary (LOW confidence)
- (Web search returned no results; relying on official docs and source code analysis)

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - PyO3 0.22 well-established; thiserror standard in Rust ecosystem
- Architecture: MEDIUM - Patterns based on PyO3 best practices; untested in this codebase
- Pitfalls: HIGH - Identified from current codebase inspection and common FFI issues

**Research date:** 2026-03-13
**Valid until:** 30 days (stable domain, PyO3 0.22 LTS)
