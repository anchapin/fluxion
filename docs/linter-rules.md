# Fluxion Linter Rules

> **TL;DR**: Fluxion-specific linting rules for physics correctness, code quality, and test reliability.
> **Owned by**: Fluxion team
> **Reviewed**: 2026-07-13

This document defines Fluxion-specific linting rules beyond standard Rust clippy. These rules enforce physics correctness, prevent common simulation bugs, and ensure test reliability.

---

## Physics Rules (FLX-PHYSICS-*)

### FLX-PHYSICS-001: No Unchecked Division in Physics Equations

**Severity:** Error

**Description:** Physics equations involving division must not have denominator values that could be zero or near-zero. This includes thermal resistances, areas, time steps, and mass flow rates.

**Anti-pattern:**
```rust
let r_value = 1.0 / (ua_sum - other.ua); // Could be zero
let htc = q_dot / (t_surface - t_air);   // Could divide by zero if equal temps
```

**Correct:**
```rust
let r_value = 1.0 / (ua_sum - other.ua).max(MIN_UA_DIFF);
let delta_t = t_surface - t_air;
if delta_t.abs() > MIN_TEMP_DIFF {
    htc = q_dot / delta_t;
}
```

**Reference:** ARCHITECTURE.md §Module Boundaries

---

### FLX-PHYSICS-002: Energy Balance Must Close Within Tolerance

**Severity:** Error

**Description:** All energy balance checks must use relative tolerance, not absolute tolerance. For building energy simulations, use 1e-6 relative tolerance for single precision, 1e-10 for double precision.

**Anti-pattern:**
```rust
assert!(energy_imbalance < 1e-3); // Absolute tolerance inappropriate
```

**Correct:**
```rust
let relative_imbalance = (inputs - outputs) / inputs.max(outputs).abs();
assert!(relative_imbalance.abs() < 1e-6);
```

---

### FLX-PHYSICS-003: State Variables Must Be Physical

**Severity:** Error

**Description:** Zone temperatures, surface temperatures, and humidity ratios must be checked for physical bounds before use in calculations.

**Anti-pattern:**
```rust
let new_temp = old_temp + dt * (q_net / (rho_air * cp_air * volume));
// No bounds check
```

**Correct:**
```rust
let new_temp = old_temp + dt * (q_net / (rho_air * cp_air * volume));
let new_temp = new_temp.clamp(MIN_ZONE_TEMP, MAX_ZONE_TEMP);
```

**Bounds:**
| Variable | Min | Max | Unit |
|----------|-----|-----|------|
| Zone temperature | -50 | 80 | °C |
| Surface temperature | -50 | 200 | °C |
| Humidity ratio | 0 | 0.05 | kg/kg |
| Pressure | 50000 | 150000 | Pa |

---

### FLX-PHYSICS-004: Time Step Validation for Conduction Solvers

**Severity:** Error

**Description:** Conduction solvers (5R1C, CTF, FD) must validate that the time step meets stability criteria before computing.

**Anti-pattern:**
```rust
fn solve(&mut self, dt: f64) {
    let alpha = self.diffusivity;
    let dx = self.thickness / (NODES - 1) as f64;
    let fo = alpha * dt / dx.powi(2);
    // No check on Fourier number
}
```

**Correct:**
```rust
fn solve(&mut self, dt: f64) -> Result<(), SolverError> {
    let alpha = self.diffusivity;
    let dx = self.thickness / (NODES - 1) as f64;
    let fo = alpha * dt / dx.powi(2);
    if fo > 0.5 {
        return Err(SolverError::StabilityViolation {
            fourier: fo,
            max_allowed: 0.5,
            dt_required: 0.5 * dx.powi(2) / alpha,
        });
    }
    // proceed with solution
}
```

---

### FLX-PHYSICS-005: Solar Position Validation

**Severity:** Warning

**Description:** Solar calculations must validate latitude, longitude, and hour angles. Declination and hour angle must be within physical bounds.

**Anti-pattern:**
```rust
let declination = 23.45 * f64::sin(2.0 * PI * (284 + day_of_year) / 365.0);
// No validation that result is in expected range
```

**Correct:**
```rust
let declination = 23.45_f64.to_radians() * f64::sin(2.0 * PI * (284 + day_of_year) / 365.0);
// Declination should be between -23.45 and 23.45 degrees
assert!(declination.abs() <= 23.5, "Declination {} out of physical bounds", declination);
```

---

## Code Quality Rules (FLX-CODE-*)

### FLX-CODE-001: No Floating Point Equality

**Severity:** Error

**Description:** Never use `==` or `!=` for floating point comparison. Use `f64::abs(a - b) < tolerance` or `relative_eq(a, b)` from `approx`.

**Anti-pattern:**
```rust
if result == expected { ... }
```

**Correct:**
```rust
use approx::relative_eq;
if relative_eq!(result, expected, max_relative = 1e-6) { ... }
```

---

### FLX-CODE-002: Explicit Type Conversions

**Severity:** Warning

**Description:** All numeric type conversions must be explicit with `.into()`, `.as_()`, or `f64::from()`.

**Anti-pattern:**
```rust
let x = 5; // implicit i32 to f64
let area = length * width; // mixed units possible
```

**Correct:**
```rust
let x = 5.0_f64;
let area = (length_m * width_m).into_inner(); // explicit units
```

---

### FLX-CODE-003: Error Handling for External Data

**Severity:** Error

**Description:** Weather data, reference data, and configuration files must be validated on load. Return `Result<T, Error>` for all data loading functions.

**Anti-pattern:**
```rust
fn load_epw(path: &str) -> WeatherData {
    // Could panic on invalid data
}
```

**Correct:**
```rust
fn load_epw(path: &Path) -> Result<WeatherData, WeatherError> {
    let content = std::fs::read_to_string(path)?;
    parse_epw(&content).ok_or(WeatherError::ParseFailure)?
}
```

---

## Test Rules (FLX-TEST-*)

### FLX-TEST-001: No Opacity in Test Names

**Severity:** Warning

**Description:** Test function names must clearly indicate what is being tested, what condition is being exercised, and what is expected.

**Anti-pattern:**
```rust
#[test]
fn test_calc() { ... }

#[test]
fn test_fail() { ... }
```

**Correct:**
```rust
#[test]
fn conduction_solver_returns_nan_when_dt_exceeds_stability_limit() { ... }

#[test]
fn zone_energy_balance_closes_within_1e6_relative_tolerance() { ... }
```

**Name template:** `{unit}_{method}_{scenario}_{expected}`

---

### FLX-TEST-002: Golden Reference Tests Must Be Deterministic

**Severity:** Error

**Description:** Tests comparing against reference data must set seeds for any random number generators and must be marked as `#[test]` not `#[test_each_seed]`.

**Anti-pattern:**
```rust
#[test]
fn test_surrogate_output() {
    let output = model.predict(&input);
    assert_eq!(output, REFERENCE_OUTPUT); // Flaky if RNG inside model
}
```

**Correct:**
```rust
#[test]
fn test_surrogate_output_deterministic() {
    let mut rng = SmallRng::seed_from_u64(42);
    let output = model.predict(&input, &mut rng);
    assert_eq!(output, REFERENCE_OUTPUT);
}
```

---

### FLX-TEST-003: Mock Data Must Be Physically Plausible

**Severity:** Warning

**Description:** Mock data used in tests must be within physical bounds (see FLX-PHYSICS-003).

**Anti-pattern:**
```rust
let mock_weather = WeatherData {
    temp: 1000.0, // Unrealistic
    humidity: 5.0, // Unrealistic
    ...
};
```

**Correct:**
```rust
let mock_weather = WeatherData {
    temp: 25.0, // 25°C indoor
    humidity: 0.005, // 50% RH at 25°C
    ...
};
```

---

## Running the Linters

### Pre-commit Lint
```bash
./scripts/precommit-lint.sh
```

### Agent-assisted Lint
```bash
./scripts/agent-lint.sh
```

### Individual Rule Checks
```bash
cargo clippy -- -A clippy::all -W flutter::FLX-PHYSICS-001
```

---

## Adding New Rules

1. Add rule to this document with FLX-{CATEGORY}-{NNN} naming
2. Update scripts/precommit-lint.sh if applicable
3. Update scripts/agent-lint.sh if applicable
4. Add tests for the rule
5. Update doc-inventory.md if rule affects documentation

