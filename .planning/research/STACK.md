# Stack Research

**Domain:** Building Energy Modeling (BEM) - Rust-based Physics Engine with Integration Testing
**Researched:** 2026-03-15
**Confidence:** MEDIUM

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Rust | 2021 edition | Core language for physics engine | Memory safety, zero-cost abstractions, excellent performance for numerical computing |
| criterion | 0.5 | Statistical benchmarking framework | Industry standard for Rust benchmarking, provides statistical confidence in detecting performance regressions, supports baseline comparison and variance testing |
| proptest | 1.5 | Property-based testing | Standard Rust property-based testing library, excels at finding edge cases through random input generation, already in use for thermal invariants |
| tempfile | 3.10 | Temporary file management | Rust standard for testing with temporary files/directories, essential for E2E tests that need file I/O |
| approx | 0.5 | Floating-point comparison | Standard for approximate floating-point comparisons in scientific computing, handles NaN/Inf properly |
| rand | 0.8 | Random number generation | Essential for generating test populations and synthetic data, used in existing benchmarks |

### Testing Framework Additions

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| rstest | 0.25 | Parameterized testing | Table-driven tests, fixtures with parameters, test cases with multiple inputs (already in dev-dependencies, optional in Cargo.toml) |
| serial_test | 1.0 | Sequential test execution | Prevents race conditions in tests with shared state, useful for parallel test debugging |
| mockito | 1.7 | HTTP mocking | For integration tests that mock external HTTP services (weather downloads, reference data fetches) |
| anyhow | 1.0 | Error handling | Already in dependencies, use for test error aggregation and reporting |

### Validation Gap Resolution Tools

| Technology | Version | Purpose | Why Needed |
|------------|---------|---------|-------------|
| statrs | 0.18.0 | Statistical computing | Already in dependencies, used for NMBE, CV(RMSE), FDR calculations in validation framework |
| faer | 0.23.2 | Linear algebra | Already in dependencies, used for thermal network matrix operations and 8R3C evaluations |
| ndarray | 0.16 | Numerical arrays | Already in dependencies, compatible with CTA VectorField for thermal mass accuracy improvements |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| cargo test | Built-in test runner | Run unit tests, integration tests, benchmarks |
| cargo bench | Built-in benchmark runner | Criterion-based benchmarking, uses release profile automatically |
| cargo flamegraph | Performance profiling | Generate flamegraphs for hot path identification |
| dhat | Heap profiling | Track memory allocations in hot loops (already in dev-dependencies) |

## Installation

```bash
# Core testing (already in Cargo.toml)
cargo test

# Property-based testing (already in dev-dependencies)
cargo test thermal_invariants

# Benchmarks (already in dev-dependencies)
cargo bench --bench performance_regression -- --baseline phase10

# Integration testing additions
cargo add --dev rstest serial_test
cargo add --dev mockito # if mocking HTTP services

# For E2E test framework
# No additional crates needed - use existing:
# - tempfile for temp file management
# - approx for floating-point comparison
# - proptest for property-based testing
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| criterion | divan | divan provides faster benchmarking but less mature ecosystem, criterion has better CI integration |
| proptest | quickcheck | quickcheck is older with fewer strategies, proptest has better documentation and more active maintenance |
| approx | float-cmp | float-cmp is more complex, approx is simpler and sufficient for ASHRAE 140 tolerance bands (±15%) |
| rstest | test-case | test-case is less maintained, rstest has better async support and fixture composition |
| tempfile | std::fs::remove_dir_all | Manual cleanup is error-prone, tempfile ensures RAII-style cleanup |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| custom test frameworks | Reinvents wheel, maintenance burden | Use built-in cargo test + established libraries (criterion, proptest) |
| assert! for floating-point | Fails due to precision | Use approx::assert_relative_eq with tolerance parameter |
| manual variance testing | Inconsistent results | Use Criterion's built-in statistical analysis (mean, median, std dev) |
| hardcoded test data | Doesn't catch edge cases | Use proptest for random generation or rstest for table-driven tests |
| mockall for HTTP mocking | Overkill for simple cases | Use mockito for HTTP mocks, mockall only if trait mocking needed |

## Stack Patterns by Variant

**If testing thermal mass accuracy improvements:**
- Use proptest with thermal capacitance strategies (LOW_MASS_CONFIG, MEDIUM_MASS_CONFIG, HIGH_MASS_CONFIG)
- Benchmark with criterion to compare explicit vs implicit integration methods
- Use statrs for statistical validation metrics (NMBE, CV(RMSE))

**If testing Case 960 sunspace validation:**
- Use tempfile for temporary weather file generation
- Use approx::assert_relative_eq with ±15% annual energy tolerance
- Use rstest for table-driven tests across different shading configurations

**If testing 8R3C thermal network evaluation:**
- Use faer for matrix operations (eigenvalue decomposition, linear solving)
- Use ndarray for multi-dimensional thermal state arrays
- Benchmark against 5R1C baseline using criterion baseline comparison

**If testing E2E integration workflow:**
- Use mockito to mock weather downloads and reference data fetches
- Use tempfile for temporary simulation output files
- Use serial_test for tests that share global state or require sequential execution

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| criterion@0.5 | Rust 2021 edition | Requires stable Rust, no nightly features needed |
| proptest@1.5 | std library only | No external dependencies, works with all Rust editions |
| statrs@0.18.0 | faer@0.23.2 | Used together for validation metrics and linear algebra |
| ndarray@0.16 | faer@0.23.2 | Compatible array types for thermal mass calculations |
| tempfile@3.10 | Rust 1.70+ | Requires recent std library features |

## Integration Points

### Existing Integration (No Changes Needed)
- **Validation Framework**: `src/validation/` module with ASHRAE140Validator, already uses statrs
- **Property-Based Testing**: `tests/thermal_invariants.rs` uses proptest 1.5, no changes needed
- **Benchmarking**: `benches/performance_regression.rs` uses criterion 0.5, baseline system established
- **Statistical Validation**: `tests/test_statistical_validation.rs` uses statrs 0.18.0 for NMBE/CV(RMSE)

### New Integration Points for v0.5
- **E2E Test Framework**: Create `tests/e2e/` directory with tempfile and approx
- **Integration Test Orchestration**: Use rstest for fixture composition across multiple test modules
- **Mock Infrastructure**: Add mockito for weather API and reference data service mocking
- **Test Parallelization**: Use serial_test for sequential execution where needed, rayon for population-level parallelism (already in dependencies)

### Python Integration (No Changes Needed)
- **PyO3 Bindings**: Already integrated, Python tests in `api/tests/` use pytest
- **ONNX Runtime**: Already using ort@2.0.0-rc.10, no changes needed for surrogate testing

## Sources

- [criterion - Rust Documentation](https://docs.rs/criterion/latest/criterion/) - HIGH confidence (official docs)
- [proptest - Rust Documentation](https://docs.rs/proptest/latest/proptest/) - HIGH confidence (official docs)
- [The Rust Programming Language - Test Organization](https://doc.rust-lang.org/book/ch11-03-test-organization.html) - HIGH confidence (official docs)
- [Fluxion Existing Stack](/home/alex/Projects/fluxion/Cargo.toml) - HIGH confidence (actual project configuration)
- [Fluxion Testing Patterns](/home/alex/Projects/fluxion/.planning/TESTING.md) - HIGH confidence (internal documentation)
- [Fluxion Contributing Guide](/home/alex/Projects/fluxion/docs/CONTRIBUTING.md) - HIGH confidence (internal documentation)
- [Fluxion Performance Benchmarks](/home/alex/Projects/fluxion/docs/PERFORMANCE_BENCHMARKS.md) - HIGH confidence (internal documentation)
- [Fluxion Architecture](/home/alex/Projects/fluxion/docs/ARCHITECTURE.md) - HIGH confidence (internal documentation)
- [Phase 10 Baseline System](/home/alex/Projects/fluxion/benches/baseline/phase10/README.md) - HIGH confidence (internal documentation)

---
*Stack research for: Building Energy Modeling - Integration Testing & Validation*
*Researched: 2026-03-15*
