# Phase 21: Integration Testing Framework - Context

**Gathered:** 2026-03-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Build comprehensive integration testing infrastructure that catches wiring issues, provides reusable fixtures, validates Python-Rust FFI boundary, and prevents regressions before they reach production.

**What this delivers:**
- E2E integration test framework with reusable fixtures for building/weather/HVAC scenarios — INTEG-01, INTEG-02
- Wiring validation system that detects integration issues between modules — INTEG-03, INTEG-08
- Python-side integration tests for PyO3 bindings with NumPy array validation — INTEG-04
- Regression test suite that runs full ASHRAE 140 validation (18 cases) — INTEG-05
- Test data management with centralized repository and versioning — INTEG-06
- CI/CD integration that runs tests and benchmarks on every PR — INTEG-07

This phase establishes the testing foundation that all subsequent phases (Validation Gap Resolution, Production Readiness) depend on for detecting regressions and catching wiring issues.

</domain>

<decisions>
## Implementation Decisions

### E2E Framework Design

**Infrastructure location:** `src/testing/integration/`
- Centralized module for all E2E test infrastructure
- Fixtures are public API, can be used across multiple test files
- Clear separation from unit tests and integration test consumers

**Fixture construction:** Builder pattern
- Example: `BuildingScenario::new().with_zone().with_weather().with_hvac().build()`
- Flexible and explicit construction
- Easy to extend with additional configuration options

**Test discovery:** Automatic (#[test] attribute)
- Rust's standard #[test] attribute on functions in tests/integration/
- Simple, explicit, no manual registration required
- Can optionally add metadata via attributes if needed

**Fixture state management:** Isolated (tempfile)
- Each test creates its own temporary directory and fixtures
- No shared state between tests
- Use tempfile crate for RAII-style cleanup
- Research-recommended approach to avoid test flakiness

### Wiring Validation

**Detection method:** Runtime tracing only
- Analyze actual function calls at execution time
- Catches wiring issues that static analysis would miss
- Aligned with research recommendation to avoid Pitfall 1

**Validation scope:** Comprehensive (both module call chains and data flow paths)
- Module call chains: Verify `solve_timesteps()` calls `predict_loads()` when `use_ai=true`
- Data flow paths: Validate weather → simulation → output, loads → surrogates → physics
- Most comprehensive wiring validation approach

**Implementation:** Instrumentation layer
- Wrappers around modules that record events
- Can be compiled out with `#[cfg(test)]` or feature flags
- Clean, explicit, no runtime overhead when not testing

**Timing:** CI + explicit
- Runs automatically on CI/CD pipeline
- Can be triggered manually with flag: `cargo test --run-wiring-checks`
- Balance between safety and development speed

### Python Tests & Regression

**NumPy integration library:** Claude's discretion
- Planner/researcher to evaluate pyo3-numpy vs numpy-pyO3 based on PyO3 0.22 compatibility
- Focus on NumPy array type safety and PyO3 integration patterns

**Test location:** `tests/` directory
- Files: `tests/test_pyo3_bindings.py`, `tests/test_numpy_arrays.py`
- Aligns with existing structure where `conftest.py` already lives in tests/
- Can share fixtures with Rust integration tests via common test data

**Regression timing:** Nightly only
- Full 18-case ASHRAE 140 suite runs nightly
- Faster PR feedback loop while ensuring main branch validity
- ~2-5 min execution time acceptable for nightly workflow

**Failure handling:** Create issues
- Nightly regression test failures automatically create GitHub issues
- Non-blocking approach: doesn't prevent PR merges
- Tracks problems for manual triage
- Separate from PR-blocking unit/integration tests

### Coverage & Data Management

**E2E coverage scope:** All major features
- Batch oracle throughput (population evaluation)
- Python API (BatchOracle, Model classes)
- CLI commands (validate, simulation)
- Surrogate integration (AI surrogate calls)
- HVAC equipment (VAV, CAV, HeatPump, Chiller, Boiler)
- Psychrometrics (dew point, humidity ratio, enthalpy, wet-bulb)
- Internal loads (lighting, equipment, occupancy with schedules)
- Multi-zone physics (inter-zone conductance, zonal coupling)
- Comprehensive but maintainable

**Test data location:** External data directory
- Separate from repository: `tests/data/` or external path
- Good for large datasets (EPW files, reference results)
- Can be gitignored if large, referenced in documentation

**Test data versioning:** Versioned subdirs
- Structure: `tests/data/v0.4/`, `tests/data/v0.5/`, `tests/data/latest/`
- Clear version boundaries, git history preserves all versions
- Tests specify version: `load_epw("tests/data/v0.4/usa_ca_san_francisco_2019.epw")`

**Test data management:** Cache in data dir
- Download external test data once to external data directory
- Tests use cached data for deterministic execution
- Real HTTP download code tested (unlike mocked HTTP)
- Combine with mockito for E2E tests that require offline behavior

### Claude's Discretion

**NumPy integration library:** Choose between pyo3-numpy and numpy-pyO3 based on PyO3 0.22 compatibility and NumPy support research
**Python test organization:** Group tests logically (bindings, arrays, error cases, FFI boundary)
**Exact instrumentation layer design:** Wrappers around specific modules vs generic trace layer
**Test category metadata:** Whether to add custom attributes for categorizing E2E tests (wiring, cli, api, regression)
**Data directory path:** Exact location and how tests locate it (env var, const path, or config)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets

**Test dependencies (already in Cargo.toml):**
- `tempfile 3.10` — Temporary file management for isolated fixtures
- `approx 0.5` — Floating-point comparison for ASHRAE 140 tolerance bands
- `rstest 0.25` — Parameterized testing for table-driven tests and fixture composition
- `proptest 1.5` — Property-based testing (already used in thermal_invariants.rs)
- `mockito 1.7` — HTTP mocking for external weather downloads

**Existing test infrastructure:**
- `tests/` directory with 60+ test files
- `tests/ashrae_140/` subdirectory for validation test data
- `tests/conftest.py` — Python pytest configuration (exists but no Python tests yet)
- `src/validation/mod.rs` — ASHRAE140Validator for regression tests
- `src/validation/report.rs` — ValidationStatus and report generation

### Established Patterns

**Test organization:**
- Unit tests in `src/` files with `#[cfg(test)]` modules
- Integration tests in `tests/` directory (standard Rust pattern)
- ASHRAE 140 tests organized by case series (600-series, 800-series, 900-series)

**Validation pattern:**
- `ASHRAE140Validator::validate_analytical_engine()` runs all cases
- Report generation with `to_markdown()` for CI output
- Status enum: `Pass | Warning | Fail`

**Python integration:**
- PyO3 bindings exposed in `src/lib.rs`
- `BatchOracle` and `Model` classes with NumPy array support
- CI runs `pytest -q` (currently no tests pass)

### Integration Points

**Where new E2E code connects:**
- `src/lib.rs` — Add `pub mod testing` for E2E framework
- `tests/integration/` — Create E2E test files consuming framework
- `src/testing/integration/` — New module: fixtures, wiring checker, test runner
- `.github/workflows/ci.yml` — Add E2E test step
- `tests/conftest.py` — Add Python pytest fixtures for integration tests
- `tests/data/` — External data directory for EPW files and reference results

**Module wiring validation points:**
- `src/sim/engine.rs` — `ThermalModel::solve_timesteps()` should call `SurrogateManager::predict_loads()`
- `src/ai/surrogate.rs` — `SurrogateManager::predict_loads_batched()` integration
- `src/lib.rs` — `BatchOracle::evaluate_population()` parallelism and surrogate integration

</code_context>

<specifics>
## Specific Ideas

**Wiring validation should catch:** When `use_ai=true` but `solve_timesteps()` never calls `predict_loads()` — this was a real issue in v0.4 (integration checker discrepancies).

**Nightly regression workflow:** Separate GitHub Action `.github/workflows/nightly_regression.yml` that runs full ASHRAE 140 suite and creates issues on failure.

**Test fixtures for Python:** Need to validate NumPy array shapes, dtypes (f32 vs f64), and error handling across PyO3 boundary (panic → Python exception).

**CLI integration tests:** Use `std::process::Command` to run `fluxion validate --all` and verify exit codes and output format.

**Performance regression detection:** Use Criterion's baseline comparison to detect >10% slowdown from baseline (INTEG-07 requirement).

</specifics>

<deferred>
## Deferred Ideas

**FMI 3.0 co-simulation tests** — v2.0 feature, not in v0.5 scope
**REST/gRPC API integration tests** — v2.0 feature, library focus for v0.5
**Docker integration tests** — v2.0 feature, optional for library distribution
**Extended ASHRAE standards validation (140.2)** — v1.0 feature, out of scope for v0.5

</deferred>

---

*Phase: 21-integration-testing-framework*
*Context gathered: 2026-03-15*
