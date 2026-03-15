# Phase 1: Foundation - Core Validation Fixes - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

## Phase Boundary

Correct fundamental 5R1C thermal network parameterization and HVAC load calculations to reduce 61% failure rate and 78.79% MAE. This is a bug-fix phase focused on achieving ASHRAE 140 validation compliance through targeted physics corrections.

**Scope Constraints:**
- High-impact fixes only: window-related conductances (h_tr_em, h_tr_w) and HVAC load calculation
- Validate lightweight cases only: 600, 610, 620, 630, 640, 650
- Defer thermal mass dynamics (Case 900) to Phase 2
- No new capabilities - only correcting existing physics

## Implementation Decisions

### Fix Strategy

**Test-driven development approach:**
- Write failing unit tests for each conductance first
- Fix implementation to make tests pass
- Validate against ASHRAE 140 reference values
- Write all tests before touching implementation (complete test suite upfront)

**Testing sequence:**
- Conductance unit tests first — isolate root cause of heating over-prediction
- Validate against ASHRAE 140 Case 600 reference values (well-documented)
- Use full case simulation as first test (Case 600 full year)
- Research-guided fix: apply known issues from research (conductance units, window U-value application), then write tests to validate

**Implementation approach:**
- Extract helper methods for conductance calculations (calc_h_tr_em(), calc_h_tr_w(), etc.)
- Fix high-impact areas first: window-related conductances before other conductances
- Verify fixes using energy balance check: sum of all loads = energy change in thermal mass
- Indirect validation: run Case 600 simulation and check results against reference (not direct field access)

### Testing Methodology

**Test coverage:**
- Both unit tests and integration tests together (comprehensive approach)
- Unit tests: test each 5R1C conductance independently
- Integration tests: full ASHRAE 140 case simulations validating annual/monthly energy and peak loads

**Test structure:**
- Unit tests in separate files: `src/sim/tests/test_conductance_calculations.rs`
- Integration tests in separate files: `tests/ashrae_140_validation.rs`
- Parameterized tests using rstest framework for tests across multiple envelope properties

**Execution:**
- Run all tests in development loop: `cargo test --test-threads=1` after each fix
- CI/CD enforces all tests: gate commits on test failures

**Diagnostic integration:**
- Use existing diagnostic framework (DiagnosticCollector) in integration tests
- Collect hourly data and energy breakdowns for debugging failed cases

### Scope Boundary

**In scope for Phase 1:**
- Fix window-related conductances: h_tr_em (exterior-to-mass) and h_tr_w (exterior-to-interior)
- Fix HVAC load calculation (Ti_free temperature source, sign convention)
- Validate all lightweight cases: 600, 610, 620, 630, 640, 650

**Out of scope (deferred):**
- Thermal mass dynamics fixes (Case 900) — Phase 2
- Other conductances (h_tr_ms, h_tr_is, h_ve) — defer to later phases if not critical
- Solar radiation and external boundary fixes — Phase 3
- Inter-zone heat transfer fixes — Phase 4

**Deliverables:**
- Code fixes for conductances and HVAC load calculation
- Updated ASHRAE140_RESULTS.md with new validation numbers
- Updated STATE.md and ROADMAP.md with phase progress

### Claude's Discretion

- Exact implementation of helper method signatures and internal structure
- Specific test case organization within files
- Validation of conductance values before tests (diagnostic analysis optional)
- Order of testing individual conductances within the all-at-once approach

## Existing Code Insights

### Reusable Assets

**Diagnostic infrastructure:**
- `DiagnosticCollector` — event-driven data collection with configurable output
- `DiagnosticConfig` — enable/disable diagnostics, hourly output, energy breakdowns
- `HourlyData` — hourly temperature, solar, HVAC, internal load tracking
- `EnergyBreakdown` — component-level energy analysis (envelope, infiltration, solar, internal gains)
- `DiagnosticReport` — comprehensive Markdown/CSV/JSON report generation

**Validation framework:**
- `ASHRAE140Validator` — core validation logic coordinating test execution
- `CaseSpec` — building geometry, materials, HVAC, weather specifications with builder pattern
- `BenchmarkData` — reference values from EnergyPlus, ESP-r, TRNSYS with min/max ranges
- Comparison engine — tolerance-based validation with Pass/Warning/Fail status (±5% tolerance band)

**Thermal model traits:**
- `ThermalModelTrait` — modular interface for physics/surrogate models
- `PhysicsThermalModel` — physics-based 5R1C implementation
- `HVACSchedule` — heating/cooling setpoints with free-floating support
- `HvacSpec` — HVAC setpoints (heating <20°C, cooling >27°C)

### Established Patterns

**Thermal network structure:**
- ISO 13790 5R1C model with conductances stored as `VectorField` for CTA operations
- `apply_parameters()` method maps gene vector to model state and broadcasts conductances
- `solve_timesteps()` uses CTA operations (element-wise +, *, /) for 5R1C solving

**Builder pattern:**
- `CaseBuilder` for flexible test case construction with optional parameters
- Allows custom case specifications beyond predefined ASHRAE 140 cases

**Schedule management:**
- `Schedule` type with value(hour) method for time-dependent setpoints
- `free_floating()` variant for cases without HVAC control

### Integration Points

**Where new code connects:**
- `src/sim/engine.rs` — `ThermalModel::apply_parameters()` applies conductance changes
- `src/sim/construction.rs` — `Assemblies` and `Construction` provide envelope R-values
- `src/validation/ashrae_140_cases.rs` — `CaseSpec` provides baseline case parameters
- `tests/ashrae_140_validation.rs` — existing integration tests for validation

## Specific Ideas

No specific reference requirements provided. Open to standard ASHRAE 140-compliant approaches.

## Deferred Ideas

None — discussion stayed within phase scope (high-impact fixes only, no new capabilities suggested).

---

*Phase: 01-CONTEXT.md*
*Context gathered: 2026-03-09*
