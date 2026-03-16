# Codebase Concerns

**Analysis Date:** 2026-03-08

## Tech Debt

**Thermal Mass Energy Accounting:**
- Issue: Test validation skipped due to incomplete thermal mass energy accounting
- Files: `src/sim/engine.rs` (lines 3376-3385, 3422, 3451, 3502)
- Impact: Physics validation incomplete; steady-state heat transfer tests bypassed
- Fix approach: Implement proper thermal mass energy change tracking across timesteps; integrate energy balance validation

**Large Monolithic Files:**
- Issue: `src/sim/engine.rs` at 4,059 lines is difficult to navigate and maintain
- Files: `src/sim/engine.rs`, `src/validation/ashrae_140_cases.rs` (2,021 lines), `src/lib.rs` (1,746 lines)
- Impact: Reduced code comprehension, harder to test individual components, increased merge conflict risk
- Fix approach: Split `ThermalModel` into focused modules (solver, controller, initialization, validation); extract ASHRAE case definitions to separate files

**Excessive Cloning in Hot Path:**
- Issue: 194+ clone operations in `src/sim/engine.rs` creating performance overhead
- Files: `src/sim/engine.rs`
- Impact: BatchOracle throughput degradation; unnecessary memory allocations in parallel execution
- Fix approach: Implement borrow-based patterns for immutable data; use `Arc` for shared state; audit each clone for necessity

**Mutable State Overuse:**
- Issue: 67+ `let mut` declarations in `src/sim/engine.rs` indicating imperative style
- Files: `src/sim/engine.rs`
- Impact: Reduced thread safety guarantees; harder to reason about state changes in parallel code
- Fix approach: Refactor to functional style where possible; encapsulate mutable state in clear ownership boundaries

**Unsafe Code in PyO3 Bindings:**
- Issue: Multiple `unsafe` blocks for NumPy array slicing
- Files: `src/lib.rs` (lines 170, 773, 1608, 1613), `src/physics/cta.rs` (lines 282, 285)
- Impact: Memory safety responsibility on maintainers; potential for UB if assumptions violated
- Fix approach: Use safe alternatives like `to_vec()` where performance permits; document invariants for each unsafe block; consider PyO3's safer slice APIs

## Known Bugs

**ASHRAE 140 Validation Failures:**
- Symptoms: 39/64 validation metrics failing (61% failure rate); systematic over-prediction of heating loads
- Files: `docs/ASHRAE140_RESULTS.md`, `src/sim/engine.rs`
- Trigger: Running `fluxion validate --all` or ASHRAE 140 test cases
- Workaround: None; validation failures indicate physics model discrepancies
- Root causes under investigation: High-mass building cases (900-series) show significant deviations; peak heating values inconsistent with reference

**Night Ventilation Temperature Clamping:**
- Symptoms: Inconsistent thermal mass energy balance when night ventilation active
- Files: `src/sim/engine.rs` (lines 2031-2045, 2047-2049)
- Trigger: Night ventilation scenarios with variable infiltration
- Workaround: None; affects energy consumption accuracy
- Impact: Incorrect EUI calculations for buildings with night cooling strategies

**Negative EUI Clamping:**
- Symptoms: Negative energy use intensity values clamped to 0.0 in BatchOracle
- Files: `src/lib.rs` (lines 710-712)
- Trigger: Deadband operation with thermal mass charging > HVAC input
- Workaround: None; masks underlying physics issue
- Impact: Energy conservation violations; invalid physics results

## Security Considerations

**Environment Variable Exposure:**
- Risk: `.env` files exist but may contain sensitive API keys or credentials
- Files: `.gitignore` (references `.env`, `.env.local`)
- Current mitigation: Git ignores .env files, but no enforcement of secret management
- Recommendations: Implement `.env.example` template with placeholder values; document required vs. optional environment variables; add pre-commit hook to detect committed secrets

**Unsafe Pointer Dereferencing:**
- Risk: Memory corruption in PyO3 bindings if NumPy arrays not contiguous
- Files: `src/lib.rs`, `src/physics/cta.rs`
- Current mitigation: `as_slice()?` returns Result, but unsafe block assumes success
- Recommendations: Validate array contiguity before unsafe conversion; add explicit bounds checking; consider replacing with safer PyO3 APIs

**ONNX Model Loading:**
- Risk: Loading untrusted ONNX models could execute arbitrary code via custom operators
- Files: `src/ai/surrogate.rs` (SessionPool, MultiDeviceSessionPool)
- Current mitigation: None; trusts model files from disk
- Recommendations: Implement model signature verification; restrict operator allowlist; validate model metadata before loading

## Performance Bottlenecks

**BatchOracle Coordinator-Worker Overhead:**
- Problem: Channel-based coordination creates synchronization overhead for every timestep
- Files: `src/lib.rs` (lines 648-700, 812-872)
- Cause: Cross-beam channels allocated per worker; 8,760 round trips per simulation year
- Improvement path: Batch multiple timesteps per channel round trip; consider lock-free ring buffer; evaluate if time-first loop overhead outweighs benefits

**Sensitivity Tensor Recalculation:**
- Problem: Sensitivity tensor recalculated at every timestep for variable ventilation
- Files: `src/sim/engine.rs` (lines 2047-2049)
- Cause: Issue #301, #366 - h_ext changes with night ventilation require recalculation
- Improvement path: Cache sensitivity tensors for discrete ventilation states; precompute for common scenarios; invalidate cache only when ventilation state changes

**Session Pool Contention:**
- Problem: Multiple threads competing for ONNX session pool access
- Files: `src/ai/surrogate.rs` (SessionPool with Mutex<Vec<Session>>)
- Cause: Single mutex guards entire session pool; no per-thread session caching
- Improvement path: Implement thread-local session caching; use sharded mutexes; evaluate lock-free session distribution

**VectorField Clone Overhead:**
- Problem: Excessive cloning of VectorField in physics calculations
- Files: `src/sim/engine.rs` (lines 1999-2023 show repeated `.clone()`)
- Cause: CTA API designed with copy-on-write semantics not leveraged
- Improvement path: Implement borrowing for read-only operations; use `Arc<VectorField>` for shared immutable state; refactor CTA to support references

## Fragile Areas

**ThermalModel State Mutation:**
- Files: `src/sim/engine.rs` (4,059-line monolithic struct)
- Why fragile: 40+ mutable fields; complex initialization order; state scattered across public methods
- Safe modification: Add state transition invariants; implement builder pattern for initialization; extract state management to separate `ModelState` struct
- Test coverage: High for ASHRAE cases, but low for edge cases and state transitions

**Distributed Inference Coordination:**
- Files: `src/sim/distributed_inference.rs`, `src/lib.rs` (BatchOracle coordinator loop)
- Why fragile: Cross-beam channels + rayon + tokio creates complex concurrency; worker disconnect causes panic
- Safe modification: Add timeout handling for channel operations; implement graceful worker failure recovery; add comprehensive integration tests
- Test coverage: Limited; relies on smoke tests; no failure injection testing

**ASHRAE 140 Case Specifications:**
- Files: `src/validation/ashrae_140_cases.rs` (2,021 lines)
- Why fragile: Hardcoded values; no automated verification against ASHRAE spec documents; risk of drift
- Safe modification: Add checksums or hashes for reference data; implement schema validation for case specs; add CI check against ASHRAE 140 reference
- Test coverage: High (validated against reference ranges), but no regression detection for spec changes

**Python-Rust Boundary (PyO3):**
- Files: `src/lib.rs` (PyVectorField, BatchOracle, Model classes)
- Why fragile: Unsafe array slicing; manual parameter vector parsing (hardcoded column counts); no type safety across boundary
- Safe modification: Use PyO3's `#[pyo3(from_py_with)]` for custom type conversion; add validation in Rust before processing; implement Python-side type hints
- Test coverage: Minimal; only smoke tests present; no negative test cases for malformed input

## Scaling Limits

**Memory per Simulation:**
- Current capacity: ~100MB per ThermalModel (VectorField allocations, 8,760 timesteps)
- Limit: Memory bound for >10,000 concurrent simulations (~1GB+), crashes with OOM on 8GB systems
- Scaling path: Implement streaming timesteps; use memory-mapped files for weather data; reduce per-simulation state via shared caches

**ONNX Session Pool Size:**
- Current capacity: Default pool size not specified; assumed 1-2 sessions
- Limit: CPU bound for >100 concurrent inferences without GPU acceleration; session creation overhead
- Scaling path: Auto-scaling session pool based on throughput; implement GPU session pooling with multi-device support (already started in MultiDeviceSessionPool)

**BatchOracle Population Size:**
- Current capacity: Tested up to ~1,000 configurations
- Limit: Channel coordination overhead becomes dominant at >10,000 configs; Python-Rust serialization cost
- Scaling path: Implement chunked evaluation; use shared memory for population data; consider zero-copy serialization (e.g., `arrow` format)

**ASHRAE 140 Validation Time:**
- Current capacity: 9 cases × 8,760 timesteps = 78,840 simulation steps
- Limit: Validation takes >10 minutes in CI; blocks rapid iteration
- Scaling path: Parallelize case execution; cache intermediate results; implement differential validation (only changed cases)

## Dependencies at Risk

**ONNX Runtime (ort = "2.0.0-rc.10"):**
- Risk: Release candidate version; API instability between RC versions
- Impact: SurrogateManager breaking changes; SessionPool API incompatibility; GPU backend provider changes
- Migration plan: Pin to stable 2.0.0 when released; implement abstraction layer over ONNX APIs; add version detection in SessionPool

**PyO3 (0.22):**
- Risk: Breaking changes in PyO3 0.23+ expected for GIL lifetime changes
- Impact: Python bindings may require significant refactoring; unsafe code may become invalid
- Migration plan: Monitor PyO3 changelog; isolate PyO3-specific code to thin wrapper layer; add compatibility tests

**ndarray (0.16):**
- Risk: 0.17+ in development with potential breaking changes to array iteration APIs
- Impact: CTA operations may break; NumPy interop requires updates
- Migration plan: Implement adapter pattern for ndarray operations; restrict ndarray usage to well-defined interfaces

**rayon (1.10):**
- Risk: Thread pool configuration changes in future versions; work-stealing algorithm differences
- Impact: BatchOracle performance regression; potential deadlocks in nested parallelism
- Migration plan: Pin rayon version strictly; add benchmark regression tests; document thread pool assumptions

## Missing Critical Features

**Energy Conservation Validation:**
- Problem: No automated validation that energy input = energy output across simulation
- Blocks: Confidence in physics accuracy; detection of numerical drift
- Impact: Silent physics errors; thermal mass energy accounting issues go undetected

**Error Propagation from Surrogates:**
- Problem: ONNX inference errors panicking or returning dummy values without detection
- Blocks: Reliable surrogate fallback to analytical mode
- Impact: Invalid results when ONNX model fails; no graceful degradation

**Test Coverage for Edge Cases:**
- Problem: Missing tests for: NaN propagation, infinity handling, extreme parameter values
- Blocks: Confidence in production robustness
- Impact: Production crashes from unhandled edge cases

**Benchmark Regression Detection:**
- Problem: CI runs benchmarks but no automated regression checking
- Blocks: Performance regressions caught too late
- Impact: Silent performance degradation across releases

## Test Coverage Gaps

**Negative Input Handling:**
- What's not tested: Negative temperatures, negative conductances, negative zone areas
- Files: `src/sim/engine.rs`, `src/lib.rs`
- Risk: Physics calculations may produce NaN or incorrect results
- Priority: High (validation could produce invalid configurations)

**Concurrent Access Patterns:**
- What's not tested: Multiple Python threads calling BatchOracle simultaneously; race conditions in SessionPool
- Files: `src/lib.rs`, `src/ai/surrogate.rs`
- Risk: Data races in production; undetected deadlocks
- Priority: High (multi-threaded Python use case)

**Error Recovery in Coordinator Loop:**
- What's not tested: Worker disconnection mid-simulation; channel timeout; ONNX session failure
- Files: `src/lib.rs` (lines 648-700), `src/ai/surrogate.rs`
- Risk: Panic cascades in batch evaluation; lost population data
- Priority: Medium (production reliability)

**Parameter Validation Edge Cases:**
- What's not tested: Boundary values (MIN_U_VALUE, MAX_SETPOINT), empty populations, single-element populations
- Files: `src/lib.rs` (validate_parameters)
- Risk: Incorrect validation logic producing false positives/negatives
- Priority: Medium (API correctness)

**Memory Leak Detection:**
- What's not tested: Long-running simulations with repeated BatchOracle calls; SessionPool lifecycle
- Files: All Rust modules with allocation
- Risk: Memory growth in long-running services; OOM in production
- Priority: Medium (deployment stability)

**ASHRAE 140 Regression:**
- What's not tested: Automated comparison to previous validation results; drift detection
- Files: `src/validation/ashrae_140_validator.rs`
- Risk: Physics regressions go undetected between releases
- Priority: High (validation integrity)

---

*Concerns audit: 2026-03-08*
