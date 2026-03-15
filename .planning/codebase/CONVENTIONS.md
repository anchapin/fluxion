# Coding Conventions

**Analysis Date:** 2026-03-08

## Naming Patterns

**Files:**
- `snake_case.rs` for all Rust source files
- Modules match directory structure: `src/sim/engine.rs` → `sim::engine`
- Test files: `test_*.py` for Python, `*_test.rs` or `test_*.rs` for Rust integration tests

**Functions:**
- `snake_case` for all function names
- Public functions: `pub fn calculate_power(...)`
- Private functions: `fn get_daily_cycle(...)`
- Methods follow same pattern: `pub fn new(...)`, `pub fn with_stages(...)`

**Variables:**
- `snake_case` for all variables and fields
- Constants: `UPPER_SNAKE_CASE` (e.g., `MIN_U_VALUE`, `MAX_SETPOINT`)
- Temporal variables: Use descriptive names like `heating_threshold`, `cooling_threshold`
- Physics parameters: Prefix with physical quantity (e.g., `heating_setpoint`, `cooling_setpoint`, `window_u_value`)

**Types:**
- `PascalCase` for structs, enums, and traits
- `ThermalModel<T>`, `VectorField`, `IdealHVACController`, `HVACMode`
- Type parameters use single capital letters: `<T: ContinuousTensor<f64>>`
- Enum variants: `PascalCase` (e.g., `HVACMode::Heating`, `InferenceBackend::CPU`)

## Code Style

**Formatting:**
- `cargo fmt` enforced via pre-commit hook
- Standard Rust formatting (4-space indentation)
- Line length: No strict limit, but typically under 100 characters
- Blank lines: One blank line between functions/methods

**Linting:**
- `cargo check` enforced via pre-commit hook
- `cargo clippy` for additional lints
- `cargo audit` for security vulnerabilities (pre-commit)
- Custom pre-commit hooks:
  - `batch-oracle-pattern`: Enforces single-level parallelism in BatchOracle
  - `rust-doc-check`: Validates doc comments on public API
  - `perf-baseline` (optional): Performance regression detection

**Python:**
- `black` for formatting (line-length: 88)
- `ruff` for linting (extends E, F, I rules)
- `isort` for import sorting
- `mypy` for type checking

## Import Organization

**Order:**
1. `use crate::...` (internal modules)
2. `use std::...` (standard library)
3. `use third_party::...` (external crates)
4. `use super::*` in test modules only

**Path Aliases:**
- No path aliases configured (all imports are explicit)
- Module re-exports in `src/lib.rs` for public API convenience

**Common Patterns:**
```rust
use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;
use std::ops::{Add, Mul, Sub};
use anyhow::Result;
```

**Conditional Imports:**
```rust
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(test)]
use super::*;
```

## Error Handling

**Patterns:**
- **Public API**: Returns `Result<T, String>` for recoverable errors
  - Example: `pub fn new(weights: Vec<T>) -> Result<Self, String>`
- **Internal code**: Uses `expect()` for truly unrecoverable conditions
  - Example: `tx.send(temps).expect("Failed to send state to coordinator")`
- **ONNX integration**: Returns `Result<T, Box<dyn std::error::Error>>`
- **Validation**: Returns `Result<(), String>` from validation methods

**Dependencies:**
- `anyhow = "1.0"` - Error composition (available but not heavily used)
- `thiserror = "1.0"` - Custom error types (available but not heavily used)

**Common Error Patterns:**
```rust
// Validation errors
pub fn validate(&self) -> Result<(), String> {
    if self.heating_setpoint > self.cooling_setpoint {
        return Err("Heating setpoint must be below cooling setpoint".to_string());
    }
    Ok(())
}

// Constructor errors
pub fn new(weights: Vec<T>) -> Result<Self, String> {
    if weights.len() == 0 {
        return Err("Weights cannot be empty".to_string());
    }
    Ok(Self { weights })
}
```

## Logging

**Framework:** `tracing` or `env_logger` (not detected in code - likely uses `println!` for debug output)

**Patterns:**
- Debug output: `println!("=== Model Conductances ===")` in tests
- Debug assertions: `const EPSILON: f64 = 1e-9; assert!(abs_diff < EPSILON)`
- Diagnostic output: Uses `DiagnosticCollector` and `DiagnosticConfig` for detailed validation output

**When to Log:**
- Test diagnostics: Print model parameters and intermediate values
- Validation output: Use `report.print_summary()` for structured output
- Performance: Benchmarks use `criterion` library

## Comments

**When to Comment:**
- Document physics assumptions and calibrations (e.g., Issue #274 thermal mass correction)
- Explain ASHRAE 140 case-specific logic
- Note temporary workarounds with `// TODO:` comments
- Document derived constants and their calculations

**JSDoc/TSDoc:**
- **Required**: All public structs, enums, traits, and functions must have `///` doc comments
- Enforced by `rust-doc-check` pre-commit hook
- Example format:
```rust
/// Ideal HVAC controller with deadband and staging support.
///
/// This controller implements ASHRAE 140 compliant HVAC control with:
/// - Dual setpoint control (heating and cooling)
/// - Deadband between heating and cooling setpoints
/// - Optional staging for multi-stage systems
///
/// # Arguments
/// * `heating_setpoint` - Heating setpoint in °C
/// * `cooling_setpoint` - Cooling setpoint in °C
///
/// # Returns
/// HVAC power in Watts (positive = heating, negative = cooling)
pub fn calculate_power(&self, zone_temp: f64, free_float_temp: f64, sensitivity: f64) -> f64 {
    // ...
}
```

**Comment Patterns:**
- Use `///` for public API documentation
- Use `//` for inline comments
- Use `//!` for module-level documentation
- Comment complex calculations with physical units
- Reference issue numbers for workarounds (e.g., `// Issue #274`)

## Function Design

**Size:** No strict limit, but typically under 50 lines for clarity. Physics calculation functions can be longer (e.g., `ThermalModel::solve_timesteps`).

**Parameters:**
- Use individual parameters for small function signatures (≤3-4 parameters)
- Use builder pattern or config structs for complex initialization
- Example: `IdealHVACController::with_stages(heating, cooling, heating_stages, cooling_stages, ...)`
- Pass `&self` for read-only access, `&mut self` for mutations

**Return Values:**
- Use explicit return types (never infer)
- Return `Result<T, E>` for fallible operations
- Return `Option<T>` for optional values
- For physics calculations, return `f64` for scalar results, `VectorField` for vector results

**Examples:**
```rust
// Small function
pub fn determine_mode(&self, zone_temp: f64) -> HVACMode {
    if zone_temp < self.heating_threshold {
        HVACMode::Heating
    } else if zone_temp > self.cooling_threshold {
        HVACMode::Cooling
    } else {
        HVACMode::Off
    }
}

// Complex builder
pub fn with_stages(
    heating_setpoint: f64,
    cooling_setpoint: f64,
    heating_stages: u8,
    cooling_stages: u8,
    heating_capacity_per_stage: f64,
    cooling_capacity_per_stage: f64,
) -> Self { ... }
```

## Module Design

**Exports:**
- Public API re-exported in `src/lib.rs`
- Internal modules use `pub mod` for inter-module access
- Test modules use `mod tests { ... }` pattern with `#[cfg(test)]`

**Barrel Files:**
- Each directory has a `mod.rs` file that re-exports public items
- Example: `src/sim/mod.rs` exports `boundary`, `components`, `engine`, etc.
- Re-exports in `src/lib.rs` for convenience:
  ```rust
  pub use sim::thermal_model::{PhysicsThermalModel, SurrogateThermalModel, ...};
  pub use sim::construction::{Construction, ConstructionLayer, MassClass};
  ```

**Module Structure:**
- `src/physics/` - CTA, continuous fields, geometry tensors
- `src/sim/` - Thermal model, HVAC, solar, shading, etc.
- `src/ai/` - Surrogates, neural fields, batch inference
- `src/validation/` - ASHRAE 140, diagnostics, benchmarks
- `src/weather/` - Weather data sources (EPW, etc.)

**Conditional Compilation:**
```rust
#[cfg(feature = "python-bindings")]
mod python_api;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_something() { ... }
}
```

## Concurrency

**Threading:**
- Use `rayon` for data parallelism (population-level evaluation)
- Use `tokio` for async operations
- Critical: **Single-level parallelism only** (enforced by `batch-oracle-pattern` hook)
- Do NOT nest `par_iter()` inside other parallel loops

**Shared State:**
- Use `Arc<Mutex<T>>` for shared mutable state
- Use `OnceLock` for lazy-initialized static data (e.g., `DAILY_CYCLE`)
- Use `crossbeam::channel` for thread communication

## Constants

**Physical Constants:**
- Define at module or struct level with descriptive names
- Include units in doc comments
- Example: `hvac_heating_capacity: f64 // 100 kW (very high, won't be a limit)`

**Calibration Constants:**
- Document ASHRAE 140 calibration values with issue references
- Example: `solar_distribution_to_air = 0.75 // Low-mass: 75% to air, 25% to thermal mass`

---

*Convention analysis: 2026-03-08*
