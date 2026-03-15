# Pitfalls Research: v0.5 Production Foundation

**Domain:** Building Energy Modeling (BEM) — Adding Integration Testing, Validation Gap Resolution, and Production Readiness
**Researched:** 2026-03-15
**Confidence:** MEDIUM (Based on project context and software engineering best practices; limited external sources due to search tool issues)

---

## Executive Summary

Adding integration testing, validation gap resolution, and production readiness to an existing Building Energy Modeling system presents systematic challenges that differ from initial validation work. The most critical pitfalls fall into three categories: (1) **Integration tests that don't detect wiring issues** (the very problem they're meant to prevent, as seen in v0.4's integration checker discrepancies), (2) **Validation gap fixes that introduce regressions** in previously passing ASHRAE 140 cases, and (3) **Production artifacts that become stale** as the codebase evolves. Fluxion's current status (37/37 v0.4 requirements satisfied, known Case 960 annual cooling failure, 8R3C evaluation pending, high-mass accuracy 229-322% above reference) demonstrates that these pitfalls are real and costly. Prevention requires: (a) designing integration tests to exercise real component wiring through E2E workflows, (b) running full ASHRAE 140 validation suite after each gap fix, (c) implementing automated documentation freshness checks, and (d) adding monitoring alongside stability guarantees.

---

## Critical Pitfalls

### Pitfall 1: Integration Tests That Don't Detect Wiring Issues

**What goes wrong:**
Integration tests pass all components individually but fail to detect when components are incorrectly wired together. For example, a test might verify `ThermalModel::solve_timesteps()` produces valid temperatures, and another test verifies `SurrogateManager::predict_loads()` returns valid loads, but neither test catches if `solve_timesteps()` never calls `predict_loads()` when `use_ai=true`.

**Why it happens:**
- Tests use mocks or stubs that bypass real wiring (testing components in isolation)
- Tests focus on happy-path scenarios that don't exercise error propagation paths
- Tests don't verify data flow across module boundaries
- Over-reliance on unit tests that don't catch integration issues
- Integration checker runs before gap closure, verification reports after (timing mismatch)

**Consequences:**
- **Integration checker reports partial integrations** while tests pass (happened in v0.4 with WEATHER-03, WEATHER-04, WEATHER-05)
- **Wiring issues ship to production** despite green test suite
- **Debugging requires manual investigation** instead of failing tests
- **Integration checker discrepancies** vs verification reports (seen in v0.4)

**Prevention:**
1. **Use real implementations in integration tests** (not mocks) whenever possible
2. **Create E2E tests** that trace data from Python API through Rust to final output
3. **Verify component interaction** by asserting side effects (e.g., `predict_loads()` was called N times)
4. **Add integration tests for known failure modes** (weather data pipeline, ASHRAE 140 case setup)
5. **Include negative tests** that verify error handling across boundaries
6. **Test through Python API** not just Rust constructors

**Warning signs:**
- All tests pass but integration checker reports partial integrations
- Verification reports disagree with integration checker analysis
- Tests use `#[cfg(test)]` mocks for all cross-module calls
- No test file imports from multiple modules simultaneously
- Tests only use Rust constructors, not Python API

**Phase to address:**
Phase 1 (Integration Testing Framework) — Design tests to detect wiring issues before they accumulate.

---

### Pitfall 2: Validation Gap Fixes That Introduce Regressions

**What goes wrong:**
When fixing known validation gaps (Case 960 annual cooling failure, 8R3C thermal network evaluation, high-mass accuracy), the fix breaks previously passing validation cases. For example, adding thermal mass correction for high-mass buildings might improve those cases but cause 600-series free-floating tests to fail.

**Why it happens:**
- Validation cases tested individually, not as a suite
- Fixes applied to core physics without running full ASHRAE 140 validation suite
- Thermal network parameter changes affect multiple cases with different mass levels
- Tests run only specific case numbers, not the full suite
- Root cause not fully understood (e.g., thermal mass energy accounting vs parameter tuning)

**Consequences:**
- **New regressions introduced** while fixing known issues
- **Validation score degrades** despite fixes (e.g., 18/18 passing → 15/18 passing)
- **Wasted effort** reverting fixes and re-implementing
- **Loss of confidence** in physics changes
- **Case-specific fixes** break other cases in same series (e.g., 960 fix breaks 920, 930, 940)

**Prevention:**
1. **Always run complete ASHRAE 140 validation suite** after physics changes (all 18 cases)
2. **Add automated regression tests** that run before every commit
3. **Use git bisect** to identify which change caused a regression
4. **Document thermal parameter sensitivity** for each case (which parameters affect which cases)
5. **Maintain baseline reference values** for all cases, not just the one being fixed
6. **Run 900-series together** as regression test when fixing Case 960
7. **Validate root cause** before applying fixes (energy accounting vs parameter tuning)

**Warning signs:**
- Manual validation reports only show case being fixed, not full suite
- CI runs ASHRAE 140 validation manually rather than automatically
- Test suite runs only subset of cases (e.g., only 600-series or only 900-series)
- No baseline reference values stored for regression detection
- Test files named `test_case_960.rs` but no `test_900_series.rs`

**Phase to address:**
Phase 2 (Validation Gap Resolution) — Run full validation suite after each fix.

---

### Pitfall 3: Performance Benchmarks That Don't Reflect Real Workloads

**What goes wrong:**
Benchmarks measure single-configuration latency (<100ms target) but don't test population-level throughput (10,000 configs/second target). Alternatively, benchmarks test with debug builds or small test data that don't reflect production loads (8760 timesteps, multi-zone buildings).

**Why it happens:**
- Benchmarks use unrealistic inputs (e.g., 10 timesteps instead of 8760)
- Benchmarks run in debug mode, missing release optimizations
- Benchmarks test `solve_timesteps()` in isolation without `evaluate_population()` overhead
- No benchmark for the actual use case (BatchOracle with 10,000 configs)
- Focus on "fast benchmarks" rather than "realistic benchmarks"

**Consequences:**
- **Misleading performance claims** (<100ms per config but 10k configs take hours)
- **Production performance issues** despite green benchmarks
- **Performance regressions undetected** (benchmarks don't catch slowdowns)
- **Optimization effort wasted** on wrong hot paths
- **Throughput requirements unmet** (10,000 configs/second target missed)

**Prevention:**
1. **Create benchmarks that mirror production workloads** (8760 timesteps, multi-zone)
2. **Always benchmark with `--release` profile** (LTO, codegen-units=1)
3. **Add benchmarks for both single-config latency** and population throughput
4. **Use realistic population sizes** (100, 1000, 10000) to scale correctly
5. **Benchmark both analytical and surrogate paths** (use_ai=false vs true)
6. **Include Python-Rust FFI boundary crossing** in benchmarks
7. **Document benchmark methodology** in output (release profile, input size, etc.)

**Warning signs:**
- Benchmarks complete in microseconds (real simulation takes milliseconds)
- Benchmark output doesn't mention release profile
- No benchmark for `BatchOracle::evaluate_population()`
- Benchmark uses single-zone instead of multi-zone buildings
- Benchmarks test 10 timesteps instead of 8760

**Phase to address:**
Phase 3 (Production Readiness) — Benchmarks must reflect real workloads.

---

### Pitfall 4: Documentation That Becomes Stale Before Release

**What goes wrong:**
Production documentation is written early in the milestone but becomes outdated as code changes. For example, API documentation describes `ThermalModel` parameters that were refactored, or deployment guide references build commands that no longer work.

**Why it happens:**
- Documentation written as a "documentation task" after code is complete
- No automated checks that documentation examples compile/run
- Docs stored separately from code, making updates easier to miss
- PR reviews don't include doc string updates as a requirement
- Examples hardcoded with old API signatures

**Consequences:**
- **Users can't follow outdated examples** (code doesn't compile)
- **Deployment fails** for users following stale documentation
- **Support burden increases** (users report issues from outdated docs)
- **Confusion about API changes** (docs don't reflect current state)
- **Loss of trust** in documentation quality

**Prevention:**
1. **Write documentation alongside code** (docs-as-code approach)
2. **Add doctest examples** to public APIs and run them in CI
3. **Use `cargo doc --no-deps`** to check doc links and examples
4. **Make doc updates a required checkbox** in PR template
5. **Generate API docs automatically** from Rust doc comments
6. **Include "run example" commands** in user-facing docs and verify they work
7. **Add CI job that builds examples** to catch stale code snippets

**Warning signs:**
- Documentation section in PR template is optional or missing
- Doc examples have `// TODO: verify this works` comments
- No CI job checks `cargo doc`
- Manual README updates without corresponding code review
- Examples don't compile or use outdated API

**Phase to address:**
Phase 3 (Production Readiness) — Automate documentation freshness checks.

---

### Pitfall 5: 5R1C to 8R3C Migration Without Comprehensive Validation

**What goes wrong:**
Migrating from 5R1C (5 resistor, 1 capacitor) to 8R3C (8 resistor, 3 capacitor) thermal network to fix high-mass accuracy, but not validating that the new network matches reference data for low-mass buildings. This could cause regressions in currently passing cases (600-series, 800-series) while improving high-mass cases (900-series).

**Why it happens:**
- Focus on fixing high-mass cases without checking low-mass impact
- 8R3C adds complexity (more conductances, more capacitances) making parameter mapping error-prone
- No reference values available for 8R3C comparison
- Assumption that "more complex = more accurate" without validation
- 8R3C implemented as replacement, not optional enhancement

**Consequences:**
- **Regressions in low-mass cases** (600-series, 800-series fail)
- **Validation score drops** from 18/18 to 10/18 passing
- **Performance degrades** (more complex network slower than 5R1C)
- **Wasted effort** if 8R3C doesn't improve high-mass accuracy significantly
- **Loss of trust** in thermal network changes

**Prevention:**
1. **Maintain both 5R1C and 8R3C implementations** during transition (feature flag)
2. **Run full ASHRAE 140 suite on both networks** before migration
3. **Document which cases require 5R1C vs 8R3C** (not one-size-fits-all)
4. **Validate 8R3C against high-mass reference data** before enabling
5. **Add performance benchmark** to ensure 8R3C doesn't violate <100ms per config
6. **Consider 8R3C as optional enhancement**, not mandatory replacement
7. **Validate 8R3C for low-mass cases** before migration (not just high-mass)

**Warning signs:**
- No plan to run full validation suite on 8R3C implementation
- Implementation removes 5R1C code entirely (no fallback)
- No performance impact analysis for additional complexity
- ASSUMPTION: "8R3C is always better than 5R1C"
- No documentation of which cases should use which network

**Phase to address:**
Phase 2 (Validation Gap Resolution) — Comprehensive validation before 8R3C enablement.

---

### Pitfall 6: Integration Tests That Are Too Brittle

**What goes wrong:**
Integration tests are so tightly coupled to implementation details that they fail when harmless refactoring occurs. For example, a test that asserts `ThermalModel` has exactly 5 conductances breaks when adding internal mass conductance (making it 6), even though the physics is correct.

**Why it happens:**
- Tests assert internal state structure instead of observable behavior
- Tests mock out dependencies in ways that couple to implementation
- Tests use exact floating-point comparisons instead of tolerance-based checks
- Tests don't use stable public APIs for assertions
- Tests access internal fields directly instead of public methods

**Consequences:**
- **Refactoring blocked by brittle tests** (correct changes fail tests)
- **Developers ignore or disable brittle tests** (loss of test coverage)
- **Test suite becomes unmaintainable** (constant updating needed)
- **False negatives** (tests fail when code is correct)
- **Loss of confidence** in test suite

**Prevention:**
1. **Test through public APIs** (BatchOracle, Model) not internal structs
2. **Use tolerance-based assertions** for floating-point values (e.g., `assert_abs_diff_eq!` with tolerance)
3. **Test behavior (output energy, temperature profile)** not structure (number of fields)
4. **Make tests independent of implementation details** (e.g., don't assert conductance count)
5. **Use golden file testing** for complex outputs (compare to known-good outputs)
6. **Avoid accessing internal fields** in tests (use public methods only)
7. **Design tests for refactor-friendliness** (focus on invariants, not structure)

**Warning signs:**
- Tests access `pub(crate)` or `pub` fields directly instead of methods
- Tests use `assert_eq!` on floating-point values without tolerance
- Tests fail when harmless refactoring (e.g., rename field, extract method)
- Test imports from internal modules (e.g., `use fluxion::sim::engine::*`)
- Tests assert struct field counts or field names

**Phase to address:**
Phase 1 (Integration Testing Framework) — Design robust tests from the start.

---

### Pitfall 7: Production Stability Guarantees Without Monitoring

**What goes wrong:**
Production release promises "stability guarantees" but has no monitoring to detect when those guarantees are violated. For example, claiming "<100ms per configuration" but no production metrics to alert when latency exceeds 200ms.

**Why it happens:**
- Stability defined by lab benchmarks, not production reality
- No logging/metrics infrastructure in place
- No alerting thresholds defined or configured
- Assumption that "if it works in CI, it works in production"
- Focus on shipping features, not operational readiness

**Consequences:**
- **Stability violations undetected** (latency spikes, error rates)
- **SLAs breached without knowledge** (users experience degraded performance)
- **Post-mortem debugging difficult** (no metrics to analyze failures)
- **Reactive instead of proactive** (users report issues before monitoring catches them)
- **Loss of trust** in stability claims

**Prevention:**
1. **Define stability metrics upfront** (latency percentiles, error rates, throughput)
2. **Add logging/metrics collection** before production deployment
3. **Set up alerting for metric violations** (e.g., p95 latency >150ms)
4. **Create dashboards for real-time monitoring**
5. **Include load testing** with production-like traffic patterns
6. **Document escalation procedures** when stability thresholds are breached
7. **Ship monitoring alongside features** (not as afterthought)

**Warning signs:**
- Stability section in release notes has no metrics or monitoring plan
- No logging library configured (e.g., `tracing`, `env_logger`)
- No CI job simulates production load
- "Stability" is defined qualitatively ("it's stable") not quantitatively
- No alerting thresholds documented

**Phase to address:**
Phase 3 (Production Readiness) — Monitoring must ship with stability guarantees.

---

### Pitfall 8: Case 960 Fix That Breaks Other 900-Series Cases

**What goes wrong:**
Fixing Case 960 annual cooling failure (currently 4.53 MWh vs reference) by adjusting HVAC equipment or thermal parameters causes other 900-series cases (920, 930, 940) to exceed ASHRAE 140 tolerance bands (±15% annual energy, ±10% monthly energy).

**Why it happens:**
- Case 960 tested in isolation, not as part of 900-series suite
- HVAC equipment changes affect all cases with similar equipment
- Thermal mass adjustments affect all high-mass cases similarly
- No regression test that runs all 900-series cases together
- Root cause not fully understood (COP correction vs thermal physics)

**Consequences:**
- **New 900-series failures** introduced while fixing Case 960
- **Validation score degrades** (18/18 → 15/18 passing)
- **Case-specific corrections** break similar cases
- **Wasted effort** iterating on fixes that don't generalize
- **Incomplete 900-series compliance** (only Case 960 passes)

**Prevention:**
1. **Always test complete ASHRAE 140 suite**, not just case being fixed
2. **Add specific regression test for 900-series cases** (run all 900-series together)
3. **Document HVAC equipment differences** between 920, 930, 940, 960
4. **Analyze why 960 fails before applying fixes** (root cause analysis)
5. **Consider case-specific parameter tuning vs global physics changes**
6. **Validate COP corrections** don't break other 900-series cases
7. **Run 900-series suite as gate** before committing Case 960 fix

**Warning signs:**
- Test file named `test_case_960.rs` but no `test_900_series.rs`
- Fix applied to global HVAC equipment without checking other cases
- No documentation of 900-series equipment differences
- Manual validation report shows only case 960 results
- Fix tested in isolation, not with full suite

**Phase to address:**
Phase 2 (Validation Gap Resolution) — Run 900-series regression test.

---

### Pitfall 9: High-Mass Accuracy Fix Without Thermal Mass Energy Accounting

**What goes wrong:**
Attempting to fix high-mass annual energy accuracy (currently exceeds reference by 229-322%) by adjusting thermal network parameters, but the root cause is that thermal mass energy changes aren't being accounted for in HVAC demand calculations. The code has `ThermalModel::thermal_mass_energy_accounting: bool` but it's not being used correctly.

**Why it happens:**
- Misdiagnosis of the problem (parameter tuning vs energy accounting)
- Existing code infrastructure for thermal mass accounting (`mass_energy_change_cumulative`) but not integrated
- Focus on "tuning conductances" instead of "fixing energy balance"
- No test that validates thermal mass energy conservation
- Pressure to "just make it pass" without understanding root cause

**Consequences:**
- **Parameter tuning doesn't fix fundamental issue** (229-322% error persists)
- **Energy accounting problems hidden** by parameter adjustments
- **Wasted effort** on incorrect fixes
- **Physics becomes non-conservative** (energy not conserved)
- **Future regressions likely** (fix doesn't address root cause)

**Prevention:**
1. **Verify thermal mass energy accounting is working** before tuning parameters
2. **Add test that tracks mass energy change** over simulation year
3. **Validate energy balance**: energy_in = energy_out + mass_energy_change
4. **Don't rely on parameter tuning** to fix fundamental energy accounting issues
5. **Use thermal mass correction factor as last resort**, not first
6. **Validate energy conservation** with thermal mass enabled vs disabled
7. **Check mass_energy_change_cumulative** is non-zero for high-mass cases

**Warning signs:**
- Fix focuses on adjusting `h_tr_em`, `h_tr_ms`, etc. without checking `mass_energy_change_cumulative`
- Thermal mass energy accounting flag exists but tests don't verify it
- High-mass errors are >200% (suggests fundamental issue, not tuning)
- No test explicitly asserts energy conservation with thermal mass
- Parameter tuning applied before validating energy accounting

**Phase to address:**
Phase 2 (Validation Gap Resolution) — Fix energy accounting before tuning.

---

### Pitfall 10: Integration Tests That Don't Catch Python-Rust Boundary Issues

**What goes wrong:**
Integration tests in Rust pass, but Python API has issues that aren't detected. For example, `BatchOracle::evaluate_population()` works correctly in Rust tests, but Python bindings panic when passed incorrectly shaped NumPy arrays or fail to handle exceptions properly.

**Why it happens:**
- No Python-side tests (only Rust unit tests)
- Integration tests use Rust constructors instead of Python API
- Tests don't exercise error handling across FFI boundary
- PyO3 error propagation not validated
- Assumption that "if it works in Rust, it works in Python"

**Consequences:**
- **Python API panics** in production (unhandled exceptions)
- **FFI overhead undetected** (performance issues in Python)
- **Error messages confusing** (Rust panics vs Python exceptions)
- **Users can't use Python API** (bugs only surface in Python)
- **Integration issues ship** despite green Rust tests

**Prevention:**
1. **Add Python integration tests** alongside Rust tests
2. **Test with real NumPy arrays** (not just `Vec<Vec<f64>>`)
3. **Test error handling** (invalid parameters, NaN values, wrong array shapes)
4. **Verify exception propagation** from Rust to Python
5. **Test with realistic population sizes** (1000+) to catch FFI overhead
6. **Include FFI boundary in benchmarks** (measure `evaluate_population` from Python)
7. **Add smoke test** that imports fluxion and runs basic operations

**Warning signs:**
- All tests in `tests/` directory are Rust files (`.rs`)
- No `tests/` directory with Python files (`.py`)
- Tests call Rust constructors directly (`ThermalModel::new()`) not Python API
- No tests import `fluxion` Python module
- Benchmarks don't measure Python-Rust FFI overhead

**Phase to address:**
Phase 1 (Integration Testing Framework) — Include Python-side tests.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skipping full ASHRAE 140 suite for quick iteration | Faster iteration during development | High regression risk, validation gaps | Never - always run full suite before commits |
| Using mocks in integration tests | Faster tests, easier setup | Misses wiring issues, brittle tests | Only when external service unavailable (weather API, etc.) |
| Manual validation reports instead of automated | No CI infrastructure needed | Inconsistent, doesn't catch regressions | Only for one-off validation, not for regression testing |
| Documentation written after code | Unblocked development | Stale docs, outdated examples | Never - docs-as-code approach |
| Benchmarks on debug builds | Faster benchmark iteration | Misleading performance data | Never - always benchmark release builds |
| Case-specific fixes without regression testing | Quick fix for known issue | Breaks other cases, technical debt accumulation | Never - always run full validation suite |
| Hardcoding tolerance values in tests | Quick assertion | Tests fail when expected error bounds change | Acceptable for reference cases (ASHRAE 140), configurable otherwise |
| Delaying monitoring until post-production | Faster initial release | Stability violations undetected, reactive debugging | Never - ship monitoring with features |

---

## Integration Gotchas

Common mistakes when connecting to external services or components.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Weather data parsing (EPW) | Only testing V2 format, missing V3/AMY/IWEC edge cases | Test all EPW versions in integration suite (V2, V3, AMY, IWEC) |
| Python-Rust FFI (PyO3) | Testing Rust functions directly, not Python API | Write Python tests that import `fluxion` and test through Python interface |
| ONNX surrogate integration | Using mock surrogates in all tests | Include real ONNX model tests (with dummy models in tests/) |
| ASHRAE 140 validation | Running cases individually, not as suite | Create integration test that runs all 18 cases sequentially |
| HVAC equipment modules | Testing equipment in isolation | Test equipment integrated with thermal network (e.g., chiller with 5R1C) |
| Psychrometrics calculations | Testing formulas in isolation | Test psychrometrics integrated with weather data pipeline |
| Internal loads with schedules | Testing schedule logic separately | Test schedules integrated with occupancy and HVAC demand |
| Integration checker discrepancies | Ignoring integration checker when tests pass | Investigate discrepancy, verify wiring manually if needed |
| Thermal mass energy accounting | Testing thermal mass without energy tracking | Test mass_energy_change_cumulative tracking and usage |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Debug builds for benchmarks | All benchmarks <10ms, unrealistic performance | Always use `--release` profile, document in benchmark output | Always - never trust debug build performance |
| Single-config benchmarks | Benchmarks pass, but 10k-config population takes hours | Add population throughput benchmarks (100, 1000, 10000 configs) | At scale (100+ configs) |
| Small timestep counts | Benchmark uses 10 timesteps, production uses 8760 | Always benchmark with 8760 timesteps (full year) | At realistic workloads |
| No Python-Rust FFI overhead measurement | Rust benchmarks fast, Python API slow | Include FFI boundary in benchmarks (via `evaluate_population`) | At population level (100+ configs) |
| Ignoring rayon thread pool contention | Single thread fast, parallel tests slower | Profile with realistic core counts, tune thread pool size | At high parallelism (8+ cores) |
| No memory profiling | Fast at start, memory leak slows down over time | Add memory tracking in benchmarks (valgrind, heaptrack) | After many iterations (10k+ evaluations) |
| Optimizing wrong hot loop | Optimized minor function, main loop still slow | Profile with `cargo flamegraph` to find actual bottlenecks | Always - profile before optimizing |
| 8R3C performance not benchmarked | 8R3C more accurate but 5x slower | Add performance benchmark, ensure <100ms per config | At production scale (1000+ configs) |

---

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Panics on invalid user input | Denial of service, crashes optimizer loop | Use `Result<T, E>` for error handling, validate inputs early |
| Unvalidated parameter vectors | Division by zero, NaN propagation | Validate all parameters (U-value range, setpoint range) before physics |
| Out-of-bounds array access | Memory safety violation, potential exploit | Use Rust's bounds checking, validate array lengths |
| Integer overflow in timestep calculations | Incorrect simulation results, crashes | Use checked arithmetic or saturating math for timestep calculations |
| Deserialization untrusted data | RCE via malicious ONNX models or EPW files | Validate file structure, use safe deserialization libraries |
| Thread safety violations | Data races, undefined behavior | Leverage Rust's ownership model, avoid `unsafe` without justification |
| Exception handling across FFI boundary | Panic in Rust propagates as crash in Python | Catch all Rust exceptions, convert to Python `RuntimeError` |
| Resource exhaustion (memory, file handles) | System crashes, denial of service | Add resource limits, cleanup in error paths |

---

## UX Pitfalls

Common user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Cryptic error messages ("internal error") | Users can't debug, abandon tool | Provide actionable errors ("U-value 0.01 below minimum 0.1") |
| No progress indication for long simulations | Users think process hung, kill it | Add progress callback or logging (every 1000 timesteps) |
| Missing examples in API docs | Users can't figure out how to use | Include code examples for all public APIs |
| Inconsistent parameter ordering | Users pass wrong parameters, get wrong results | Use named parameters in Python API, document parameter order |
| No validation feedback | Users submit invalid configs, wait for nothing | Validate inputs early, return errors before expensive simulation |
| Hardcoded paths (weather files) | Users can't run on their machines | Make paths configurable, document required files |
| Unclear stability guarantees | Users don't know what to expect | Document quantitative metrics (latency, throughput, error rates) |
| Missing migration guides | Users can't upgrade from v0.4 to v0.5 | Provide upgrade instructions, breaking changes documented |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Integration testing framework:** Often missing Python-side tests — verify tests in `tests/*.py` alongside `tests/*.rs`
- [ ] **Case 960 fix:** Often missing 900-series regression test — verify all 900-series cases still pass after fix
- [ ] **8R3C thermal network:** Often missing low-mass validation — verify 600-series and 800-series cases pass with 8R3C
- [ ] **High-mass accuracy:** Often missing thermal mass energy accounting validation — verify `mass_energy_change_cumulative` tracked and used
- [ ] **Performance benchmarks:** Often missing population throughput benchmarks — verify benchmarks for 100, 1000, 10000 configs
- [ ] **Documentation:** Often missing doctest examples — verify `cargo test --doc` passes and examples compile
- [ ] **Stability guarantees:** Often missing monitoring/alerting — verify metrics collection and alert thresholds defined
- [ ] **Integration tests:** Often missing real implementation usage — verify tests use real implementations, not mocks
- [ ] **Production readiness:** Often missing deployment guide — verify README includes build/install instructions
- [ ] **FFI overhead:** Often missing from benchmarks — verify benchmarks measure Python-Rust boundary crossing
- [ ] **Error handling:** Often missing across FFI boundary — verify Rust exceptions converted to Python exceptions
- [ ] **Energy accounting:** Often missing thermal mass validation — verify energy conservation with thermal mass enabled

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Integration tests don't catch wiring issues | HIGH | 1. Add comprehensive E2E tests through Python API 2. Run full ASHRAE 140 validation 3. Fix wiring issues discovered 4. Add tests for newly discovered interactions |
| Validation fix causes regressions | MEDIUM | 1. Run full ASHRAE 140 suite to identify all failing cases 2. Git bisect to find commit causing regression 3. Revert or refactor fix to be more targeted 4. Add regression test to prevent recurrence |
| Benchmarks don't reflect real workloads | MEDIUM | 1. Add realistic workload benchmarks (8760 timesteps, multi-zone) 2. Profile release build to find actual bottlenecks 3. Re-baseline performance metrics with realistic data 4. Update release criteria to use new benchmarks |
| Documentation becomes stale | LOW | 1. Audit all docs against current code 2. Add doctest examples and verify they compile 3. Add `cargo doc --no-deps` to CI to catch stale docs 4. Update docs in PRs that change code |
| 5R1C to 8R3C migration breaks cases | HIGH | 1. Revert to 5R1C implementation 2. Add 8R3C as feature-gated alternative 3. Run full validation on both implementations 4. Only enable 8R3C for high-mass cases after validation |
| Integration tests too brittle | LOW | 1. Refactor tests to use public APIs instead of internal state 2. Replace exact comparisons with tolerance-based assertions 3. Add golden file testing for complex outputs 4. Update PR review checklist to check test robustness |
| No monitoring for stability | MEDIUM | 1. Add logging/metrics collection infrastructure 2. Define stability metrics and alert thresholds 3. Set up dashboards and alerting 4. Add load testing to verify monitoring catches violations |
| Case 960 fix breaks other 900-series | MEDIUM | 1. Run 900-series regression test to identify failing cases 2. Analyze root cause of 960 failure more carefully 3. Apply more targeted fix specific to 960 4. Add regression test for all 900-series cases |
| High-mass fix ignores energy accounting | HIGH | 1. Revert parameter tuning changes 2. Implement proper thermal mass energy accounting 3. Validate energy conservation with mass tracking 4. Re-tune parameters only after energy accounting works |
| Python-Rust boundary issues undetected | MEDIUM | 1. Add Python integration tests alongside Rust tests 2. Test with real NumPy arrays and error cases 3. Verify exception propagation across FFI 4. Add FFI overhead to benchmarks |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Integration tests don't catch wiring issues | Phase 1 (Integration Testing) | E2E tests detect wiring issues, full ASHRAE 140 suite passes |
| Validation fix causes regressions | Phase 2 (Validation Gaps) | Full ASHRAE 140 suite runs after each fix, regression tests added |
| Benchmarks don't reflect real workloads | Phase 3 (Production Readiness) | Benchmarks use release profile, realistic workloads, documented in CI |
| Documentation becomes stale | Phase 3 (Production Readiness) | Doctest examples compile, `cargo doc` in CI, docs updated in PRs |
| 5R1C to 8R3C migration breaks cases | Phase 2 (Validation Gaps) | Both implementations tested, full ASHRAE 140 suite validates 8R3C |
| Integration tests too brittle | Phase 1 (Integration Testing) | Tests use public APIs, tolerance assertions, robust to refactoring |
| No monitoring for stability | Phase 3 (Production Readiness) | Metrics collection deployed, alerting configured, dashboards created |
| Case 960 fix breaks other 900-series | Phase 2 (Validation Gaps) | 900-series regression test passes, all 900-series cases validated |
| High-mass fix ignores energy accounting | Phase 2 (Validation Gaps) | Thermal mass energy accounting validated, energy balance test passes |
| Python-Rust boundary issues undetected | Phase 1 (Integration Testing) | Python tests pass, NumPy array handling tested, FFI overhead measured |

---

## Sources

### Project Context (HIGH confidence - actual project data)
- **Fluxion PROJECT.md** - v0.5 requirements, known limitations (Case 960, 8R3C, high-mass accuracy)
- **Fluxion ARCHITECTURE.md** - BatchOracle pattern, thermal network structure
- **Fluxion CLAUDE.md** - Development workflows, testing strategies, common pitfalls
- **Fluxion test files** - Existing test patterns, integration approaches
- **Fluxion v0.4 validation status** - 37/37 requirements satisfied, integration checker discrepancies

### Knowledge Sources (MEDIUM confidence - training data + domain knowledge)
- **Software engineering best practices** - Integration testing, documentation-as-code, monitoring
- **Scientific computing V&V** - Verification & Validation principles, regression testing
- **Production deployment practices** - Monitoring, stability guarantees, performance benchmarks
- **Rust-Python integration patterns** - PyO3 best practices, FFI error handling

### Gap in External Verification (LOW confidence - search tool issues)
Due to web search tool issues, external sources (integration testing methodologies, scientific simulation testing best practices, production deployment monitoring standards) could not be verified against current sources. Research relies on training data and project context.

**Consider external validation for:**
- Integration testing best practices for scientific computing
- ASHRAE 140 community guidance on regression testing
- Production monitoring standards for scientific software
- Python-Rust FFI testing patterns
- Thermal network validation beyond ASHRAE 140

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| Integration testing pitfalls | MEDIUM | Based on software engineering best practices and project context; limited external verification |
| Validation gap resolution pitfalls | HIGH | Informed by Fluxion's known Case 960 failure, 8R3C evaluation pending, high-mass accuracy issues |
| Performance benchmark pitfalls | MEDIUM | Based on software engineering best practices; limited external verification |
| Documentation pitfalls | MEDIUM | Based on software engineering best practices and project context |
| Production readiness pitfalls | MEDIUM | Based on software engineering best practices; limited external verification |
| Python-Rust FFI pitfalls | MEDIUM | Based on PyO3 documentation and common FFI patterns |
| Thermal network migration pitfalls | HIGH | Informed by Fluxion's 5R1C architecture and 8R3C evaluation requirement |
| High-mass accuracy pitfalls | HIGH | Informed by Fluxion's 229-322% high-mass error and thermal mass accounting infrastructure |

---

## Research Limitations

**Web search tool issues:** The web search tool returned no results for queries about integration testing, scientific simulation validation, and production deployment best practices. I relied on training data and project context, which provides high confidence for Fluxion-specific issues but medium confidence for general best practices.

**Fluxion-specific validation:** Many pitfalls are inferred from Fluxion's current state (v0.4 success, known limitations) rather than documented external sources. The integration checker discrepancies in v0.4 provide strong evidence for Pitfall 1, but external verification would strengthen confidence.

**External source access:** I did not have access to current external sources for integration testing methodologies, scientific computing testing best practices, or production deployment monitoring standards. References to these topics are based on training data.

---

## Actionable Recommendations for v0.5

Based on identified pitfalls, here are prioritized actions:

### Phase 1: Integration Testing Framework
1. Design E2E tests through Python API (not just Rust constructors)
2. Add tests that verify component interaction (side effects, call counts)
3. Include Python-side tests alongside Rust tests
4. Test with real NumPy arrays and error cases
5. Add integration tests for known failure modes (weather, ASHRAE 140 setup)
6. Design robust tests (public APIs, tolerance assertions, not brittle)

### Phase 2: Validation Gap Resolution
1. Run full ASHRAE 140 suite after each physics change
2. Add 900-series regression test when fixing Case 960
3. Validate thermal mass energy accounting before parameter tuning
4. Validate 8R3C against both high-mass and low-mass cases
5. Maintain both 5R1C and 8R3C during 8R3C migration
6. Document which parameters affect which cases

### Phase 3: Production Readiness
1. Create realistic workload benchmarks (8760 timesteps, multi-zone, population throughput)
2. Always benchmark with `--release` profile
3. Add documentation freshness checks (doctest examples, `cargo doc` in CI)
4. Define stability metrics and implement monitoring
5. Add FFI overhead to benchmarks
6. Write docs-as-code (alongside implementation)
7. Include deployment guide and upgrade instructions

---

*Pitfalls research for: v0.5 Production Foundation*
*Researched: 2026-03-15*
