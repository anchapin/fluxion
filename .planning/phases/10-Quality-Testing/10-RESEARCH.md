# Phase 10: Quality & Testing - Research

**Researched:** 2026-03-12
**Domain:** Rust testing infrastructure, coverage measurement, test determinism, and benchmarking
**Confidence:** MEDIUM

## Summary

Phase 10 focuses on achieving >80% test coverage and eliminating flaky tests in Fluxion's Rust-based Building Energy Modeling engine. The project currently has:
- 56 integration test files in `tests/` directory
- 234 source files with embedded unit tests
- Existing Tarpaulin configuration for code coverage
- Criterion benchmarking infrastructure in place
- No property-based testing framework (proptest) currently installed

**Primary recommendation:** Install and integrate `proptest` for property-based testing of thermal invariants, enhance Tarpaulin coverage configuration, establish deterministic testing patterns for Rayon parallel code, and implement performance regression tests using Criterion.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cargo-tarpaulin` | Latest (0.35.1+) | Code coverage measurement | Industry-standard Rust coverage tool, integrates with CI, supports XML/HTML/LCOV formats |
| `proptest` | 1.0+ | Property-based testing | Rust's canonical property testing framework, inspired by QuickCheck, automatic test generation and shrinking |
| `criterion` | 0.5+ (already in dev-deps) | Performance benchmarking | Rust's de facto benchmarking library, provides statistical analysis, stable measurements, CI integration |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `rstest` | 0.18+ (already in dev-deps) | Test fixtures and parameterized tests | For setup/teardown patterns and running same test with multiple inputs |
| `tempfile` | 3.10+ (already in dev-deps) | Temporary file management | For tests that need filesystem isolation |
| `approx` | 0.5+ (already in dev-deps) | Floating-point comparison | For approximate equality in physics calculations |
| `mockito` | 1.7+ (already in dev-deps) | HTTP mocking | For testing external API integrations |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `proptest` | `quickcheck` | `proptest` has better shrinking, more active development, Rust-specific |
| `tarpaulin` | `cargo-llvm-cov` | `tarpaulin` has easier setup, better CI integration, ptrace support for Linux |
| `criterion` | Custom `std::time::Instant` | Criterion provides statistics, auto-tuning, comparison with baselines |

**Installation:**
```bash
# Install cargo-tarpaulin for coverage
cargo install cargo-tarpaulin

# Add proptest to dev-dependencies
# (Will add to Cargo.toml [dev-dependencies] section)
cargo add --dev proptest
```

## Architecture Patterns

### Recommended Test Structure
```
src/
├── physics/
│   ├── cta.rs              # Unit tests in #[cfg(test)] module
│   ├── nd_array.rs         # Unit tests embedded
│   └── ...
├── sim/
│   ├── engine.rs           # Unit tests for ThermalModel
│   └── ...
├── ai/
│   ├── surrogate.rs        # Unit tests for SurrogateManager
│   └── ...
└── tests/                # Integration tests
    ├── ashrae_140_case_600.rs       # ASHRAE validation
    ├── test_batch_oracle_throughput.rs   # Performance tests
    ├── test_conductance_calculations.rs  # Physics validation
    └── properties/                       # Property-based tests
        ├── thermal_invariants.rs         # Energy conservation, etc.
        └── numeric_stability.rs         # Floating-point properties
```

### Pattern 1: Property-Based Testing with Proptest
**What:** Automatically generate test cases to verify invariants hold across random inputs
**When to use:** Mathematical properties, invariants that should always hold, edge case discovery
**Example:**
```rust
// Source: https://docs.rs/proptest/latest/proptest/
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_energy_conservation_heating(load in -1000.0..1000.0_f64) {
        // Energy conservation: sum of all loads should be balanced
        let model = setup_test_model();
        let result = model.apply_loads(vec![load; model.num_zones()]);

        // Total energy change should equal applied load
        prop_assert!(abs(result.energy_change - load) < 1e-6);
    }

    #[test]
    fn test_temperature_bounds(temp in -50.0..100.0_f64) {
        // Temperatures should remain within physical bounds
        let mut model = ThermalModel::new(1);
        model.set_temperature(0, temp);

        let result = model.step_physics();
        prop_assert!(result.temperatures.iter().all(|&t| t >= -273.15 && t <= 5000.0));
    }
}
```

### Pattern 2: Deterministic Parallel Testing
**What:** Ensure Rayon-based parallel code produces consistent results across runs
**When to use:** Testing BatchOracle, parallel validation, any rayon::par_iter usage
**Example:**
```rust
use rayon::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn test_parallel_batch_deterministic() {
    // Force deterministic random generation
    let mut rng = StdRng::seed_from_u64(42);
    let population: Vec<Vec<f64>> = (0..100)
        .map(|_| {
            vec![
                rng.gen_range(0.1..5.0),  // U-value
                rng.gen_range(15.0..30.0),  // Setpoint
            ]
        })
        .collect();

    // Run multiple times with seeded thread pool
    let results1 = run_batch_oracle(&population);
    let results2 = run_batch_oracle(&population);

    // Results should be identical
    assert_eq!(results1, results2, "Parallel execution should be deterministic");
}

fn run_batch_oracle(population: &[Vec<f64>]) -> Vec<f64> {
    population
        .par_iter()
        .map(|params| evaluate_config(params))
        .collect()
}
```

### Pattern 3: Performance Regression Tests
**What:** Benchmarks with variance thresholds to detect performance regressions
**When to use:** Critical paths, batch oracle throughput, thermal model solve time
**Example:**
```rust
// benches/performance_regression.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_thermal_model_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("thermal_model");

    // Test different zone counts
    for zones in [1, 5, 10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(zones),
            zones,
            |b, &zones| {
                let model = ThermalModel::new(*zones);
                b.iter(|| {
                    let mut model_clone = model.clone();
                    model_clone.solve_timesteps(8760, false, false);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(perf_benches, bench_thermal_model_solve);
criterion_main!(perf_benches);
```

### Anti-Patterns to Avoid
- **Shared static state in tests:** Tests should not depend on mutable static variables, causes race conditions
- **Assuming test execution order:** Tests must be independent and runnable in any order
- **Hardcoded test data instead of strategies:** Use proptest strategies for broader coverage
- **Ignoring floating-point precision:** Use `approx` crate for appropriate tolerance in physics calculations
- **Testing implementation details:** Test public contracts and invariants, not internal structure

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Code coverage measurement | Custom instrumentation | `cargo-tarpaulin` | Handles ptrace/llvm backends, CI integration, report generation |
| Property test generation | Manual test case enumeration | `proptest` | Automatic shrinking, comprehensive coverage, finds edge cases |
| Benchmark statistics | Manual timing loops | `criterion` | Statistical analysis, auto-tuning, baseline comparison |
| Test fixtures | Copy-paste setup code | `rstest` | Parameterized tests, lifecycle hooks, cleaner code |
| Deterministic randomness | System time seeding | `StdRng::seed_from_u64` | Reproducible tests, debugging support |
| Flaky test detection | Manual re-runs | Tarpaulin + CI gating | Automated detection, trend tracking |

**Key insight:** Custom test infrastructure is brittle and hard to maintain. Established Rust testing tools provide battle-tested solutions for common problems like test isolation, coverage measurement, and performance regression detection.

## Common Pitfalls

### Pitfall 1: Flaky Tests with Rayon Parallelism
**What goes wrong:** Tests using `par_iter()` sometimes pass, sometimes fail due to thread scheduling differences
**Why it happens:** Rayon's work-stealing scheduler doesn't guarantee execution order, thread pools aren't reset between tests
**How to avoid:**
1. Use seeded random number generation: `StdRng::seed_from_u64(42)`
2. Force single-threaded for nondeterministic tests: `RAYON_NUM_THREADS=1 cargo test`
3. Test both single-threaded and multi-threaded modes
4. Avoid tests that depend on execution order
**Warning signs:** Tests that fail intermittently with "assertion failed" without code changes

### Pitfall 2: Shared State Between Tests
**What goes wrong:** Tests interfere with each other's state, causing cascading failures
**Why it happens:** Static variables, global mutable state, file system artifacts not cleaned up
**How to avoid:**
1. Never use `static mut` or `lazy_static!` with mutable state
2. Use test fixtures with setup/teardown (`rstest` lifecycle hooks)
3. Use `tempfile` crate for isolated file system resources
4. Ensure each test creates its own instances, doesn't reuse
**Warning signs:** Tests pass when run alone but fail in `cargo test` (all together)

### Pitfall 3: Inadequate Floating-Point Comparison
**What goes wrong:** Tests fail due to tiny numerical differences in physics calculations
**Why it happens:** Floating-point arithmetic isn't exact, cumulative errors accumulate
**How to avoid:**
1. Use `approx` crate: `assert_relative_eq!(actual, expected, max_relative=1e-6)`
2. Define appropriate tolerance per calculation type
3. Test for physical invariants (energy conservation) not exact values
4. Use absolute error for small values, relative for large
**Warning signs:** Tests fail with "assertion failed: left != right" on tiny differences

### Pitfall 4: Missing Coverage of Error Paths
**What goes wrong:** Coverage >80% achieved but error handling untested
**Why it happens:** Developers focus on happy path, `Result::Err` branches ignored
**How to avoid:**
1. Force error conditions in tests: invalid parameters, missing files, network failures
2. Use `?` operator chains and test each error case
3. Add tests for panic conditions (boundary values, division by zero)
4. Use Tarpaulin's `--fail-under 80` to enforce threshold
**Warning signs:** Coverage shows red on error-handling lines

### Pitfall 5: Performance Regressions Go Undetected
**What goes wrong:** Changes slow down critical code, only discovered in production
**Why it happens:** Benchmarks not run in CI, no regression thresholds configured
**How to avoid:**
1. Add Criterion benchmarks to CI pipeline
2. Use `cargo bench -- --save-baseline` to establish baselines
3. Check against baseline with `cargo bench -- --baseline main`
4. Set variance thresholds (<5% for critical metrics)
5. Run benchmarks in release mode only (debug builds are misleading)
**Warning signs:** Benchmark times increase across commits without CI gating

### Pitfall 6: Property Tests Not Shrinking to Minimal Examples
**What goes wrong:** Property tests fail but don't identify the minimal failing case
**Why it happens:** Poor strategy definitions, custom types without `Arbitrary` impl
**How to avoid:**
1. Use built-in strategies: `any::<f64>()`, `0..100.0`
2. Implement `Arbitrary` trait for custom domain types
3. Use `prop_compose!` to build complex strategies from simple ones
4. Add `prop_assume!` for preconditions that must hold
**Warning signs:** Proptest reports large failing inputs, hard to debug

## Code Examples

Verified patterns from official sources:

### Running Tarpaulin Coverage
```bash
# Run coverage on library code
cargo tarpaulin --lib --out xml --output-dir coverage/

# Run with HTML report
cargo tarpaulin --lib --out html --output-dir coverage/

# Fail if coverage below 80%
cargo tarpaulin --lib --fail-under 80

# Run specific test
cargo tarpaulin --lib test_thermal_model_energy_conservation

# Source: https://github.com/xd009642/tarpaulin
```

### Property-Based Test with Proptest
```rust
use proptest::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #[test]
        fn test_vector_field_associativity(a in any::<f64>(), b in any::<f64>(), c in any::<f64>()) {
            // Vector addition is associative
            let v1 = VectorField::new(vec![a, b, c]);
            let v2 = VectorField::new(vec![a, b, c]);
            let v3 = VectorField::new(vec![a, b, c]);

            let sum1 = v1.clone() + v2.clone();
            let sum2 = sum1 + v3;

            let sum3 = v1.clone() + v2.clone();
            let sum4 = sum3 + v3;

            assert_eq!(sum2, sum4);
        }

        #[test]
        fn test_thermal_energy_conservation(initial_temp in 20.0..25.0_f64, load in -500.0..500.0_f64) {
            let mut model = setup_thermal_model();
            model.set_temperature(0, initial_temp);

            let energy_before = model.total_energy();
            model.apply_loads(vec![load]);
            model.step_physics();
            let energy_after = model.total_energy();

            // Energy should be conserved (within numerical precision)
            let delta = abs(energy_after - energy_before - load);
            prop_assert!(delta < 1e-6, "Energy not conserved: {}", delta);
        }
    }
}

// Source: https://docs.rs/proptest/latest/proptest/
```

### Deterministic Test with Seeded RNG
```rust
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn test_batch_oracle_deterministic_results() {
    let seed = 42;
    let population = generate_population_with_seed(seed, 100);

    // Run 10 times, all should produce identical results
    let mut results = Vec::new();
    for _ in 0..10 {
        results.push(evaluate_population(&population));
    }

    // All results should be equal
    for result in &results[1..] {
        assert_eq!(result, &results[0], "Results not deterministic");
    }
}

fn generate_population_with_seed(seed: u64, size: usize) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size)
        .map(|_| {
            vec![
                rng.gen_range(0.1..5.0),   // Window U-value
                rng.gen_range(15.0..30.0),  // HVAC setpoint
            ]
        })
        .collect()
}

// Source: Rust rand crate documentation
```

### Performance Regression Test with Criterion
```rust
use criterion::{black_box, Criterion, Throughput};

fn bench_batch_oracle_throughput(c: &mut Criterion) {
    let oracle = setup_batch_oracle();
    let population = generate_test_population(1000);

    let mut group = c.benchmark_group("batch_oracle");
    group.throughput(Throughput::Elements(1000));
    group.sample_size(100); // Increase samples for stability

    group.bench_function("analytical", |b| {
        b.iter(|| {
            oracle.evaluate_population(black_box(population.clone()), false);
        });
    });

    group.bench_function("with_surrogates", |b| {
        b.iter(|| {
            oracle.evaluate_population(black_box(population.clone()), true);
        });
    });

    group.finish();
}

// Run with baseline comparison
// cargo bench -- --save-baseline main
// cargo bench -- --baseline main

// Source: Criterion.rs book: https://bheisler.github.io/criterion.rs/book/
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual coverage scripts | `cargo-tarpaulin` | 2019+ | Automated, CI-integrated, multiple report formats |
| Example-based testing | Property-based testing | 2016+ | Broader coverage, edge case discovery, automatic shrinking |
| Manual timing loops | `criterion` benchmarking | 2017+ | Statistical analysis, regression detection, auto-tuning |
| Hardcoded test data | Strategy-based generation | 2016+ | More comprehensive, finds unexpected inputs |
| Non-deterministic tests | Seeded RNG + isolation | 2020+ | Reproducible failures, easier debugging |

**Deprecated/outdated:**
- `cargo test --nocapture` for debugging: Use `env_logger` with `RUST_LOG=debug` for structured logging
- `std::time::Instant` for benchmarks: Use `criterion` for statistical significance
- Test-specific mocking frameworks: Use trait injection and `mockall` for clean mocking
- Manual flaky test detection: Use Tarpaulin's CI integration with `--fail-under`

## Open Questions

1. **Proptest Strategy Definition for ThermalModel**
   - What we know: Need to generate valid parameter vectors (U-value 0.1-5.0, setpoint 15-30°C)
   - What's unclear: Optimal strategy for multi-zone configurations, how to handle invalid combinations
   - Recommendation: Start with simple strategies, iterate based on failing cases

2. **Determinism in Parallel ASHRAE Validation**
   - What we know: ASHRAE validation tests use `par_iter()` for multi-case parallelism
   - What's unclear: Whether validation results should be strictly deterministic or within tolerance
   - Recommendation: Define tolerance bounds for parallel validation, use seeded RNG where possible

3. **Performance Baseline Establishment**
   - What we know: Need to establish baselines for critical benchmarks (batch oracle, thermal model solve)
   - What's unclear: Acceptable variance threshold (<5% stated in requirements), baseline storage strategy
   - Recommendation: Run 10 iterations to establish baseline, use geometric mean for throughput

4. **Coverage Exclusions**
   - What we know: Some code should be excluded from coverage (e.g., debug-only functions)
   - What's unclear: Which files/modules to exclude, how to document exclusions
   - Recommendation: Use `#[cfg(not(tarpaulin_include))` attribute for debug code, document in Tarpaulin config

5. **Test Execution Time Budget**
   - What we know: Full test suite must complete in reasonable time for CI
   - What's unclear: Maximum acceptable duration, tradeoff between coverage and speed
   - Recommendation: Target <5 minutes for full suite, use parallel test execution with `--test-threads`

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Built-in `cargo test` + proptest 1.0 + Criterion 0.5 |
| Config file | `.tarpaulin.toml` (coverage) + `Criterion` inline config |
| Quick run command | `cargo test --lib` |
| Full suite command | `cargo test` (unit + integration) |
| Coverage command | `cargo tarpaulin --lib --fail-under 80 --out xml` |
| Benchmark command | `cargo bench` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TEST-01 | Unit test coverage >80% | coverage | `cargo tarpaulin --lib --fail-under 80` | ❌ Need baseline |
| TEST-02 | Property-based tests for thermal invariants | unit | `cargo test thermal_invariants` | ❌ Wave 0 |
| TEST-03 | Integration tests for edge cases | integration | `cargo test edge_cases` | ⚠️ Partial |
| TEST-04 | Deterministic results with seeded thread pools | unit | `cargo test deterministic` | ❌ Wave 0 |
| TEST-05 | Performance regression tests with <5% variance | benchmark | `cargo bench -- --baseline main` | ✅ Exists, needs gating |
| TEST-06 | No shared state between tests | integration | `cargo test --test-threads=4` | ⚠️ Needs verification |
| BUG-04 | Flaky tests eliminated across 10 runs | regression | `for i in {1..10}; do cargo test; done` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test --lib` (quick unit tests, <30s)
- **Per wave merge:** `cargo tarpaulin --lib --fail-under 75` (coverage check, <2min)
- **Phase gate:** Full suite green: `cargo test` (all tests) + `cargo tarpaulin --lib --fail-under 80` + `cargo bench -- --baseline main`

### Wave 0 Gaps
- [ ] `tests/properties/thermal_invariants.rs` — property-based tests for energy conservation, temperature bounds
- [ ] `tests/properties/numeric_stability.rs` — property-based tests for floating-point operations
- [ ] `tests/test_deterministic_parallel.rs` — deterministic tests for Rayon-based parallel code
- [ ] `tests/test_edge_cases.rs` — integration tests for extreme parameters, zero loads, boundary conditions
- [ ] `tests/test_flaky_detection.rs` — test harness to run tests 10 times and report failures
- [ ] `proptest` installation: `cargo add --dev proptest` — add to dev-dependencies
- [ ] Coverage baseline: Run `cargo tarpaulin --lib --out xml` to establish initial coverage
- [ ] Performance baselines: Run `cargo bench -- --save-baseline main` for critical benchmarks
- [ ] Test isolation audit: Review all 234 test files for shared static state

## Sources

### Primary (HIGH confidence)
- [Tarpaulin GitHub Repository](https://github.com/xd009642/tarpaulin) - Coverage tool configuration, usage patterns, CI integration
- [Proptest Documentation](https://docs.rs/proptest/latest/proptest/) - Property-based testing API, strategies, shrinking
- [Criterion.rs Book](https://bheisler.github.io/criterion.rs/book/) - Benchmarking patterns, statistical analysis, regression detection
- [Rust Testing Book](https://doc.rust-lang.org/book/ch11-00-testing.html) - Unit and integration test patterns, test organization

### Secondary (MEDIUM confidence)
- [Fluxion CONTRIBUTING.md](/home/alex/Projects/fluxion/docs/CONTRIBUTING.md) - Existing testing guidelines, test structure
- [Fluxion CLAUDE.md](/home/alex/Projects/fluxion/CLAUDE.md) - Project-specific testing patterns, batch oracle testing
- [Fluxion .tarpaulin.toml](/home/alex/Projects/fluxion/.tarpaulin.toml) - Current coverage configuration

### Tertiary (LOW confidence)
- Web search results for Rust testing best practices - Unable to verify due to search tool limitations, relying on training data
- Property-based testing patterns for physics simulations - General knowledge, needs validation against domain-specific requirements

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - Tarpaulin and proptest are well-established, but proptest integration with thermal modeling needs validation
- Architecture: MEDIUM - Test patterns are standard, but deterministic parallel testing with Rayon requires investigation
- Pitfalls: HIGH - Based on documented Rust testing issues and Fluxion's existing codebase

**Research date:** 2026-03-12
**Valid until:** 2026-04-11 (30 days for stable testing infrastructure, 7 days for rapidly evolving tools like proptest)
