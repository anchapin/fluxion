# Agent Coding Conventions

**Fluxion v0.4+** — AI coding standards for fluxion-specific conventions not covered by
generic rules in `RULES.md`, `.claude/skills/clean-code/SKILL.md`, or
`.claude/skills/refactoring-patterns/SKILL.md`.

Cross-agent reviewers use this document as a checklist. When conventions conflict,
this document takes precedence for fluxion-specific physics, testing, and API conventions.

---

## Table of Contents

1. [Physics Code Conventions](#1-physics-code-conventions)
2. [Rust Conventions](#2-rust-conventions)
3. [Test Conventions](#3-test-conventions)
4. [API Conventions](#4-api-conventions)

---

## 1. Physics Code Conventions

### 1.1 Floating-Point Math Verification

**All floating-point math must be verified in Python before committing.**

Physics implementations must include a Python verification script in `.agents/results/`
that independently computes the expected output. This catches LLM arithmetic errors
before they reach CI.

```python
# .agents/results/solar_position_verification.py
import math

def verify_solar_position(lat_deg, day_of_year, hour):
    # Independent implementation — not copied from fluxion
    declination = 23.45 * math.sin(math.radians(360/365 * (day_of_year - 81)))
    # ... compute altitude, azimuth
    return altitude_deg, azimuth_deg
```

**Rule**: When adding or modifying any thermal, solar, or ventilation calculation,
include a verification script. Do not commit physics changes without it.

### 1.2 Named Constants Over Magic Numbers

**No bare magic numbers — use named constants from `src/physics/constants/`.**

```rust
// WRONG
let flux = (T_ext - T_int) / 29.3;

// CORRECT
use crate::physics::constants::EXTERIOR_FILM_COEFF;
let flux = (T_ext - T_int) / EXTERIOR_FILM_COEFF;
```

### 1.3 Exterior Film Coefficient

**Always use `EXTERIOR_FILM_COEFF` (18.3), never `1.0 / 29.3` or any equivalent.**

```rust
// WRONG — computed from inverse resistance, obscures intent
let h_ext = 1.0 / 29.3;

// CORRECT — ASHRAE 140-2023 Table X, explicit intent
use crate::physics::constants::EXTERIOR_FILM_COEFF;
let h_ext = EXTERIOR_FILM_COEFF;
```

The named constant documents the source standard and carries uncertainty metadata.
See `docs/PHYSICAL_CONSTANTS.md` for full constant definitions.

### 1.4 Stateful vs. Query Methods

**`HeatConductionSolver::step()` is state-advancing; `surface_heat_flux()` must not call it.**

The `step()` method mutates solver state (advances the thermal mass node).
`steady_state_flux()` is a pure query — it must not call `step()` or depend on
prior `step()` invocations. This distinction is critical because ML surrogate
swap-points rely on `steady_state_flux()` for parity checks with no state dependence.

```rust
// WRONG — surface_heat_flux() mutates state
fn surface_heat_flux(&mut self, ...) -> HeatFlux {
    self.step(...)?  // Mutates internal state!
    self.saved_flux
}

// CORRECT — pure query, no side effects
fn surface_heat_flux(&self, ...) -> HeatFlux {
    self.steady_state_flux(...)?  // Deterministic, stateless
}
```

See `src/physics/solver_trait.rs` for the full trait contract.

### 1.5 NaN/Inf Rejection at Module Boundaries

**Use `debug_assert!` to reject NaN/Inf at module entry/exit points.**

Numerical instability must be caught at the physics module boundary, not silently
propagated through the simulation:

```rust
pub fn step(...) -> Result<HeatFlux, SolverError> {
    debug_assert!(
        flux.is_finite(),
        "NaN/Inf flux after step(): solver={}", self.name()
    );
    debug_assert!(
        T_interior.is_finite() && T_exterior.is_finite(),
        "NaN/Inf temperature inputs"
    );
    // ... solver logic
}
```

`debug_assert!` (not `assert!`) is used so these checks are compiled out in release builds.

### 1.6 Energy Conservation (CI Gate)

**The energy balance `flux_in + flux_out + generation = 0` must hold at every timestep.**

Energy conservation is enforced as a CI gate. For zone-level simulations:

```rust
let total_flux = solar_gain + conduction_gain + ventilation_gain + internal_gain
    + hvac_cooling + hvac_heating;
debug_assert!(
    total_flux.abs() < 1e-3,  // 1 mW tolerance for a typical zone
    "Energy balance violation: {:.6} W", total_flux
);
```

When adding new flux terms, update the energy balance check and add a test
that verifies the balance holds for a known scenario.

---

## 2. Rust Conventions

### 2.1 Error Handling

**Use `Result<T, SolverError>` over `.unwrap()` in physics paths.**

```rust
// WRONG
let flux = solver.step(...).unwrap();

// CORRECT
let flux = solver.step(...)?;
```

For the zone-level thermal model, use `PhysicsError` and `PhysicsResult<T>` from
`src/physics/solver_trait.rs`:

```rust
pub type PhysicsResult<T> = Result<T, PhysicsError>;

fn simulate(&mut self, steps: u32) -> PhysicsResult<f64> {
    let flux = self.solver.step(...)?;
    // ...
    Ok(total_energy)
}
```

### 2.2 Trait Objects

**Use `Box<dyn HeatConductionSolver + Send + Sync>` for runtime solver dispatch.**

```rust
// WRONG — concrete type leaks abstraction
let solver = FiveR1CSolver::new();

// CORRECT — trait object, enables solver swapping
let solver: Box<dyn HeatConductionSolver + Send + Sync> = Box::new(FiveR1CSolver::new());
```

### 2.3 No Unsafe in Physics-Critical Paths

**Do not use `unsafe` in any code that participates in the thermal solve loop.**

Memory safety in physics code is achieved through Rust's ownership system.
Unsafe blocks prevent static analysis and may hide bugs in a domain where
numerical stability is already a concern.

### 2.4 Document Trait Contract Methods

**All trait methods must document input ranges and output units.**

```rust
/// Advance solver by one timestep.
///
/// # Arguments
/// * `timestep` - Timestep duration [s], must be positive
/// * `T_interior` - Interior air temperature [°C], valid range: -50 to 80
/// * `T_exterior` - Exterior air temperature [°C], valid range: -60 to 60
/// * `h_interior` - Interior convective HTC [W/m²·K], must be positive
/// * `h_exterior` - Exterior convective HTC [W/m²·K], must be positive
///
/// # Returns
/// Heat flux [W/m²]: positive = heat flowing into zone
fn step(
    &mut self,
    timestep: Time,
    T_interior: Temperature,
    T_exterior: Temperature,
    h_interior: HeatTransferCoefficient,
    h_exterior: HeatTransferCoefficient,
) -> Result<HeatFlux, SolverError>;
```

### 2.5 Module-Level `pub(crate)`

**Internal helpers use `pub(crate)` visibility, not `pub`.**

```rust
// Internal helper — not part of public API
pub(crate) fn compute_thermal_resistance(...) -> f64 { ... }

// Public API — documented, versioned
pub fn simulate(...) -> PhysicsResult<f64> { ... }
```

---

## 3. Test Conventions

### 3.1 Test Naming

**Test names follow `fn module_description_scenario_result()`.**

```rust
// WRONG — vague
#[test]
fn test_flux() { ... }

// CORRECT — describes module, physical scenario, expected outcome
#[test]
fn solar_altitude_denver_summer_solstice_within_0_5deg() { ... }
```

### 3.2 Reference Data — No Hardcoded Values

**Use `tests/reference_data/` CSVs, not hardcoded values.**

Reference CSVs are generated from EnergyPlus 25.2.0 and represent ground truth.
Hardcoded values drift from E+ and are not acceptable.

```rust
// WRONG
assert!((flux - 142.7).abs() < 1.0);

// CORRECT
let reference = load_reference_csv("conduction/step_response_200mm_concrete.csv")?;
assert!((flux - reference[row].flux).abs() / reference[row].flux < 0.01);  // 1%
```

See `tests/reference_data/README.md` for available datasets.

### 3.3 Tolerances

| Quantity | Tolerance | Rationale |
|----------|-----------|-----------|
| Energy (Wh, kWh) | 1% | Energy is cumulative; small % errors accumulate |
| Temperature (°C) | 0.5°C | ASHRAE 140-2023 zone air temperature tolerance |
| Solar angles (°deg) | 0.5°deg | NOAA algorithm vs. E+ empirical fit |

```rust
const TOLERANCE_ENERGY_REL: f64 = 0.01;   // 1%
const TOLERANCE_TEMP_ABS: f64 = 0.5;      // 0.5 °C
const TOLERANCE_ANGLE_ABS: f64 = 0.5;     // 0.5 °deg
```

### 3.4 Python Verification Scripts

**Physics tests require a Python verification script in `.agents/results/`.**

The script must independently implement the algorithm (not port the Rust code)
and compare its output to the Rust result:

```python
# .agents/results/conduction_step_response_verification.py
import csv, sys
import numpy as np

def main():
    rust_output = load_csv(sys.argv[1])
    python_output = compute_step_response()

    max_rel_err = max(abs(r - p) / abs(p) for r, p in zip(rust_output, python_output))
    print(f"Max relative error: {max_rel_err:.4%}")
    assert max_rel_err < 0.01, f"Exceeds 1% tolerance: {max_rel_err:.4%}"

if __name__ == "__main__":
    main()
```

Run via `python .agents/results/<test>_verification.py` before committing.

---

## 4. API Conventions

### 4.1 Document Input Ranges and Error Conditions

**Every public API must document valid input ranges and error conditions.**

```rust
/// Simulate building thermal performance for multiple years.
///
/// # Arguments
/// * `years` — Number of simulation years, valid range: 1 to 100
/// * `use_surrogates` — Use ML surrogate models (faster, approximate)
///                      vs. full physics (slower, reference-quality)
///
/// # Errors
/// Returns `PhysicsError::Initialization` if weather data is not loaded.
/// Returns `PhysicsError::InvalidState` if `years == 0`.
///
/// # Performance
/// Surrogate mode: ~1 ms/building/year
/// Full physics mode: ~50 ms/building/year
pub fn simulate(&mut self, years: u32, use_surrogates: bool) -> PhysicsResult<f64> {
    // ...
}
```

### 4.2 Breaking Changes Require ADR

**Any change that breaks the public API or changes physics behavior requires
an Architecture Decision Record in `docs/adr/`.**

See existing ADRs in `docs/adr/` (e.g., `0002-promote-9r4c-high-mass-default.md`)
for the format. A new ADR must include:

- **Status**: Proposed / Accepted / Deprecated
- **Context**: What changed and why
- **Decision**: What the team agreed to
- **Consequences**: Breaking aspects, migration path

Do not make breaking changes without an accepted ADR.

### 4.3 Deprecation Requires Migration Path

**No deprecation without a migration path documented in `CHANGELOG.md`.**

```rust
// DEPRECATED: Use `simulate_with_weather(weather: &WeatherData)` instead.
/// This method uses embedded Denver TMY3 weather and will be removed in v0.6.
/// Migration: call `simulate_with_weather(&denver_tmy3_weather())` instead.
#[deprecated(since = "0.4.2", note = "Use `simulate_with_weather` with explicit weather data")]
pub fn simulate_years(&mut self, years: u32) -> PhysicsResult<f64> {
    self.simulate_with_weather(years, &DENVER_TMY3)
}
```

The deprecation notice must be added to `CHANGELOG.md` under the current version
with a migration path description.

---

## References

- `src/physics/solver_trait.rs` — HeatConductionSolver trait, SolverError, PhysicsError
- `src/physics/constants/mod.rs` — Named physical constants
- `docs/PHYSICAL_CONSTANTS.md` — Constant source standards and uncertainty
- `docs/adr/` — Architecture decision records
- `tests/reference_data/README.md` — Reference data catalog
- `AI_CODING_STRATEGY_ADOPTION_PLAN.md` §11 — Original convention requirements
