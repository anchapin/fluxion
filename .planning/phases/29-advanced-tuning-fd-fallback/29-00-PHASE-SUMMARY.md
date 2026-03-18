# Phase 29 Plan: Advanced Tuning & FD Fallback Integration

**Phase:** 29 - Advanced Physics Tuning & FD Fallback
**Status:** READY FOR EXECUTION
**Created:** 2026-03-18
**Effort:** 3-4 weeks (3 waves)

---

## Phase Summary

**Goal:** Implement advanced tuning strategies for CTF coefficients and FD solver with robust fallback integration.

**Context:** Phase 28 integrated CTF/FD solvers into production thermal model. Phase 29 optimizes accuracy, performance, and robustness through advanced tuning strategies.

**Success Criteria:**
1. ✅ CTF pole extraction 10× faster (Newton-Raphson vs bisection)
2. ✅ CTF coefficients stable for τ up to 10 hours
3. ✅ FD solver accurate to ±3% for very high-mass walls
4. ✅ Hybrid CTF/FD fallback with >99.9% success rate
5. ✅ CTF throughput ≥800 configs/sec
6. ✅ FD throughput ≥500 configs/sec
7. ✅ Method selection overhead <1ms per wall
8. ✅ ASHRAE 140 high-mass cases within ±15% tolerance

---

## Execution Strategy

### Wave 1: CTF Tuning (Week 1-2)
**Focus:** Optimize CTF coefficient calculation for speed and accuracy

**Plans:**
- 29-01: Newton-Raphson pole extraction
- 29-02: Coefficient convergence acceleration
- 29-03: Stability enhancement & validation

**Deliverables:**
- 10× faster pole extraction
- Stable coefficients for τ < 10h
- Condition number monitoring

---

### Wave 2: FD Tuning (Week 2-3)
**Focus:** Optimize FD solver accuracy and performance

**Plans:**
- 29-04: Node distribution optimization
- 29-05: Implicit scheme enhancement
- 29-06: Surface balance coupling

**Deliverables:**
- Geometric node grading
- Adaptive timestep control
- Robin boundary condition linearization

---

### Wave 3: Fallback Integration (Week 3-4)
**Focus:** Integrate hybrid CTF/FD fallback with performance optimization

**Plans:**
- 29-07: Fallback decision tree implementation
- 29-08: Coefficient caching & parallel solving
- 29-09: Validation & benchmarking

**Deliverables:**
- Robust fallback logic
- Performance optimization
- Comprehensive validation

---

## Detailed Plans

### Plan 29-01: Newton-Raphson Pole Extraction

**Effort:** 2-3 days
**Priority:** HIGH
**Dependencies:** None

**Tasks:**

1. **Implement analytical derivative computation**
   - Add `compute_derivative_analytical()` method to `CTFCalculator`
   - Derive dA/ds formula for multi-layer transmission matrix
   - Unit test: verify derivative accuracy vs finite difference

2. **Replace bisection with Newton-Raphson**
   - Modify `find_pole_bisection()` → `find_pole_newton()`
   - Add convergence monitoring (max 50 iterations, tol 1e-10)
   - Add divergence detection (|s| > 1e10)
   - Fallback to bisection if Newton-Raphson fails

3. **Complex conjugate pairing**
   - Add `enforce_conjugate_pairs()` method
   - Handle oscillatory response in multi-layer walls
   - Ensure real-valued coefficients

4. **Performance benchmarking**
   - Measure speedup vs bisection (target: 10×)
   - Test accuracy vs analytical solutions
   - Profile pole extraction for various wall types

**Files to Modify:**
- `src/physics/ctf_coefficients.rs` - Add Newton-Raphson implementation
- `tests/test_ctf_pole_extraction.rs` - Add unit tests

**Verification:**
- ✅ Newton-Raphson converges in <20 iterations (typical)
- ✅ Speedup ≥10× vs bisection
- ✅ Accuracy ±1% vs analytical solutions
- ✅ All existing CTF tests pass

---

### Plan 29-02: Coefficient Convergence Acceleration

**Effort:** 2-3 days
**Priority:** HIGH
**Dependencies:** 29-01 complete

**Tasks:**

1. **Exponential decay window**
   - Add `apply_decay_window()` method
   - Implement adaptive decay factor (σ = 0.80-0.95 based on τ)
   - Renormalize coefficients after applying window

2. **Optimal coefficient count determination**
   - Add `determine_optimal_coefficient_count()` method
   - Implement ASHRAE convergence criterion (|X_j|/|X_0| < 10⁻⁶)
   - Set minimum coefficients per wall type (6-50)

3. **Residue normalization**
   - Ensure sum(Y) = U-value (steady-state consistency)
   - Normalize X, Z coefficients similarly
   - Add unit tests for steady-state verification

4. **Dominant pole prioritization**
   - Add `select_dominant_poles()` method
   - Sort poles by contribution (|residue|/|Re(pole)|)
   - Retain top N poles (configurable, default 50)

**Files to Modify:**
- `src/physics/ctf_coefficients.rs` - Add convergence acceleration
- `src/physics/ctf_solver.rs` - Integrate optimal coefficient count

**Verification:**
- ✅ Coefficients decay smoothly (ratio < 0.1 after 20 terms)
- ✅ Sum(Y) = U-value within ±0.1%
- ✅ Minimum 6 coefficients, maximum 50
- ✅ Dominant poles capture >99% of response

---

### Plan 29-03: Stability Enhancement & Validation

**Effort:** 1-2 days
**Priority:** MEDIUM
**Dependencies:** 29-02 complete

**Tasks:**

1. **Condition number monitoring**
   - Add `check_condition_number()` method
   - Detect ill-conditioned transmission matrices
   - Threshold: κ < 10⁶ for stability

2. **Timestep adaptation**
   - Implement Δt ≤ τ/10 rule for high-mass walls
   - Add `adapt_timestep()` method
   - Minimum 15 minutes, maximum 1 hour

3. **CTF feasibility validation**
   - Add `validate_ctf_feasibility()` method
   - Check wall thickness (<0.5m)
   - Check homogeneous concrete ratio (<0.8 for thick walls)
   - Check layer count (<20)

4. **Comprehensive validation suite**
   - Test stability for various wall constructions
   - Verify timestep adaptation logic
   - Profile condition numbers for ASHRAE 140 cases

**Files to Modify:**
- `src/physics/ctf_coefficients.rs` - Add stability checks
- `src/physics/method_selector.rs` - Add feasibility validation

**Verification:**
- ✅ Condition number detected for unstable walls
- ✅ Timestep adapted automatically for high-mass
- ✅ CTF feasibility validated before computation
- ✅ No NaN/Inf coefficients produced

---

### Plan 29-04: FD Node Distribution Optimization

**Effort:** 2-3 days
**Priority:** HIGH
**Priority:** HIGH
**Dependencies:** None (parallel with CTF tuning)

**Tasks:**

1. **Geometric grading implementation**
   - Add `geometric_distribution()` function
   - Grade nodes near surfaces (grading ratio 0.1-0.2)
   - Uniform spacing in center region

2. **Minimum node requirements**
   - Implement nodes-per-layer rule (minimum 5)
   - Add Fourier number-based node count
   - Ensure Fo ≤ 0.5 for accuracy

3. **Interface node placement**
   - Add `place_interface_nodes()` function
   - Place nodes exactly at material boundaries
   - Handle multi-layer walls correctly

4. **Node count optimization**
   - Benchmark accuracy vs node count
   - Determine optimal node count (20-60 typical)
   - Add configuration option for node density

**Files to Modify:**
- `src/physics/fd_discretization.rs` - Add node distribution strategies
- `src/physics/fd_solver.rs` - Integrate optimized discretization

**Verification:**
- ✅ Geometric grading reduces error by 50% (vs uniform)
- ✅ Minimum 5 nodes per layer enforced
- ✅ Interface nodes placed correctly
- ✅ Optimal node count determined (20-60 typical)

---

### Plan 29-05: Implicit Scheme Enhancement

**Effort:** 2-3 days
**Priority:** MEDIUM
**Dependencies:** 29-04 complete

**Tasks:**

1. **Thomas algorithm with partial pivoting**
   - Modify `thomas_algorithm()` to add pivoting
   - Handle near-zero pivot elements
   - Add numerical stability checks

2. **Fourier number limit enforcement**
   - Add `check_fourier_number()` method
   - Enforce Fo ≤ 0.5 for accuracy
   - Auto-adapt timestep if Fo exceeds limit

3. **Adaptive timestep strategy**
   - Implement `adaptive_fd_timestep()` function
   - Scan all nodes for minimum stable timestep
   - Minimum 15 minutes, maximum 1 hour

4. **Energy conservation verification**
   - Add energy balance check to FD solver
   - Verify ±1% tolerance for annual simulations
   - Log energy imbalance if exceeds threshold

**Files to Modify:**
- `src/physics/fd_solver.rs` - Add stability enhancements
- `src/physics/fd_surface_balance.rs` - Add energy balance check

**Verification:**
- ✅ Partial pivoting prevents division by zero
- ✅ Fourier number ≤ 0.5 enforced
- ✅ Timestep adapted automatically
- ✅ Energy balance closes within ±1%

---

### Plan 29-06: Surface Balance Coupling

**Effort:** 1-2 days
**Priority:** MEDIUM
**Dependencies:** 29-05 complete

**Tasks:**

1. **Robin boundary condition linearization**
   - Add `robin_boundary_coeff()` function
   - Linearize q = h·(T_s - T_f) + q_solar for implicit scheme
   - Handle both interior and exterior boundaries

2. **Sol-air temperature calculation**
   - Implement `sol_air_temperature()` function
   - Include solar absorptance term
   - Include longwave radiation loss term

3. **Longwave radiation exchange**
   - Add `longwave_radiation_loss()` function
   - Linearized radiation heat transfer
   - Sky temperature model (clear vs overcast)

4. **Integration with thermal model**
   - Wire surface balance into `step_physics()`
   - Pass boundary conditions to FD solver
   - Extract surface heat flux for zone energy balance

**Files to Modify:**
- `src/physics/fd_surface_balance.rs` - Add surface balance models
- `src/sim/engine.rs` - Integrate FD surface coupling

**Verification:**
- ✅ Robin BC implemented correctly
- ✅ Sol-air temperature matches ASHRAE formula
- ✅ Longwave radiation modeled accurately
- ✅ Surface flux matches analytical solutions

---

### Plan 29-07: Fallback Decision Tree Implementation

**Effort:** 2-3 days
**Priority:** HIGH
**Dependencies:** 29-03, 29-06 complete

**Tasks:**

1. **Decision tree implementation**
   - Add `select_method()` to `SolverSelector`
   - Implement τ < 2h → 5R1C, τ ≥ 2h → CTF logic
   - Add CTF feasibility check before selection

2. **Fallback triggers**
   - Implement coefficient divergence detection
   - Implement pole extraction failure detection
   - Implement condition number exceeded detection

3. **FD fallback integration**
   - Auto-switch to FD when CTF fails
   - Log fallback reason for debugging
   - Ensure seamless handoff between solvers

4. **Error handling & logging**
   - Add detailed logging for fallback events
   - Create fallback statistics (count, reasons)
   - Add warning for frequent fallback cases

**Files to Modify:**
- `src/physics/method_selector.rs` - Add decision tree
- `src/physics/solver_manager.rs` - Integrate fallback logic

**Verification:**
- ✅ Decision tree selects correct method
- ✅ CTF failures caught and redirected to FD
- ✅ Fallback logged with reason
- ✅ >99.9% success rate (fallback or primary)

---

### Plan 29-08: Coefficient Caching & Parallel Solving

**Effort:** 2-3 days
**Priority:** MEDIUM
**Dependencies:** 29-07 complete

**Tasks:**

1. **CTF coefficient caching**
   - Add `CTFCache` struct with hit/miss tracking
   - Cache coefficients by wall construction hash
   - Integrate caching into solver initialization

2. **Parallel FD solver**
   - Use rayon for parallel wall solving
   - Implement `solve_all_walls_parallel()` function
   - Ensure thread-safe solver instances

3. **Performance profiling**
   - Measure cache hit rate (target: >90%)
   - Measure parallel speedup (target: 4-8× on 8-core)
   - Profile memory usage for caching

4. **Configuration options**
   - Add cache size limit (configurable)
   - Add parallelism control (thread count)
   - Add performance metrics export

**Files to Modify:**
- `src/physics/ctf_solver.rs` - Add caching
- `src/physics/fd_solver.rs` - Add parallel solving
- `src/sim/engine.rs` - Integrate performance optimizations

**Verification:**
- ✅ Cache hit rate >90% for population simulations
- ✅ Parallel speedup 4-8× on 8-core CPU
- ✅ Memory usage reasonable (<100MB for cache)
- ✅ Configuration options working

---

### Plan 29-09: Validation & Benchmarking

**Effort:** 2-3 days
**Priority:** HIGH
**Dependencies:** All previous plans complete

**Tasks:**

1. **Unit test suite**
   - Pole extraction accuracy tests
   - Coefficient convergence tests
   - FD steady-state tests
   - Surface balance tests
   - Fallback logic tests

2. **Integration tests**
   - ASHRAE 140 Case 900 (high-mass)
   - ASHRAE 140 Case 920 (very high-mass)
   - ASHRAE 140 Case 940 (extreme high-mass)
   - CTF vs FD comparison (±5% agreement)

3. **Performance benchmarks**
   - CTF throughput (target: ≥800 configs/sec)
   - FD throughput (target: ≥500 configs/sec)
   - Method selection overhead (<1ms per wall)
   - Cache performance (hit rate, memory)

4. **ASHRAE 140 validation**
   - Run full 140 validation suite
   - Target: ±15% tolerance for all cases
   - Generate validation report
   - Compare vs Phase 28 results

**Files to Create:**
- `tests/test_ctf_tuning.rs` - CTF tuning tests
- `tests/test_fd_tuning.rs` - FD tuning tests
- `tests/test_fallback_integration.rs` - Fallback tests
- `docs/PHASE_29_VALIDATION_REPORT.md` - Validation results

**Verification:**
- ✅ All unit tests pass (20+ tests)
- ✅ Integration tests pass (ASHRAE 140 cases)
- ✅ Performance targets met
- ✅ ASHRAE 140 cases within ±15% tolerance

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Newton-Raphson doesn't converge | Low | Medium | Fallback to bisection |
| FD timestep too small | Medium | Low | Adaptive timestep with minimum |
| Caching uses too much memory | Low | Low | Configurable cache size limit |
| Fallback logic too aggressive | Medium | Medium | Tunable thresholds, logging |

### Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CTF tuning takes longer | Medium | Low | Parallel FD tuning |
| Validation reveals issues | High | Medium | Buffer time (1 week) |
| Performance below target | Low | Medium | Optimization iteration |

---

## Resource Requirements

**Personnel:**
- 1 Rust developer (full-time, 3-4 weeks)
- 1 Validation engineer (part-time, 1 week)

**Compute:**
- Development: Standard workstation (8-core CPU)
- Validation: Cloud instance for parallel testing

**Dependencies:**
- Phase 28 complete (CTF/FD solvers integrated)
- ASHRAE 140 test cases available
- Performance profiling tools (criterion, perf)

---

## Files to Create/Modify

### New Files
- `src/physics/ctf_cache.rs` - Coefficient caching
- `tests/test_ctf_tuning.rs` - CTF tuning tests
- `tests/test_fd_tuning.rs` - FD tuning tests
- `tests/test_fallback_integration.rs` - Fallback tests
- `docs/PHASE_29_VALIDATION_REPORT.md` - Validation results
- `docs/CTF_TUNING_GUIDE.md` - CTF tuning documentation
- `docs/FD_SOLVER_GUIDE.md` - FD solver documentation

### Modified Files
- `src/physics/ctf_coefficients.rs` - Newton-Raphson, stability, convergence
- `src/physics/ctf_solver.rs` - Caching integration
- `src/physics/fd_discretization.rs` - Node distribution optimization
- `src/physics/fd_solver.rs` - Stability, parallel solving
- `src/physics/fd_surface_balance.rs` - Surface balance coupling
- `src/physics/method_selector.rs` - Fallback decision tree
- `src/physics/solver_manager.rs` - Fallback integration
- `src/sim/engine.rs` - Performance optimizations

---

## Success Measures

**Phase 29 is complete when:**

1. ✅ Newton-Raphson pole extraction implemented (10× speedup)
2. ✅ CTF coefficients stable for τ < 10 hours
3. ✅ FD node distribution optimized (geometric grading)
4. ✅ FD timestep adaptation working (Fo ≤ 0.5)
5. ✅ Hybrid CTF/FD fallback integrated (>99.9% success)
6. ✅ Coefficient caching working (>90% hit rate)
7. ✅ Parallel FD solving working (4-8× speedup)
8. ✅ All unit tests passing (20+ tests)
9. ✅ ASHRAE 140 cases within ±15% tolerance
10. ✅ Performance targets met (CTF ≥800, FD ≥500 configs/sec)

---

## Next Actions

1. **Execute Phase 29** using `/gsd:execute-phase 29`
   - Wave 1: CTF tuning (Week 1-2)
   - Wave 2: FD tuning (Week 2-3)
   - Wave 3: Fallback integration (Week 3-4)

2. **Plan Phase 30** using `/gsd:plan-phase 30`
   - Documentation completion
   - v0.7.0 release preparation
   - User guide updates

3. **Update project state**
   - Update `.planning/STATE.md` with Phase 29 progress
   - Update `.planning/ROADMAP.md` with timeline
   - Update `docs/PERFORMANCE_TUNING.md` with solver performance data

---

## Appendix: Implementation Code Snippets

### Newton-Raphson Pole Extraction

```rust
fn find_pole_newton(&self, s_guess: Complex64, max_iter: usize) -> Option<Complex64> {
    let mut s = s_guess;
    let tol = 1e-10;

    for _ in 0..max_iter {
        let A = self.compute_A(s);
        let dA_ds = self.compute_derivative_analytical(s);

        if dA_ds.norm() < 1e-15 {
            return None; // Derivative too small
        }

        let delta = A / dA_ds;
        s = s - delta;

        if delta.norm() < tol {
            return Some(s); // Converged
        }

        if s.norm() > 1e10 {
            return None; // Diverging
        }
    }

    None // Didn't converge
}
```

### Geometric Node Distribution

```rust
fn geometric_distribution(
    total_thickness: f64,
    num_nodes: usize,
    grading_ratio: f64
) -> Vec<f64> {
    let mut positions = Vec::with_capacity(num_nodes);
    let center_dx = total_thickness / num_nodes as f64;
    let surface_dx = center_dx * grading_ratio;

    for i in 0..num_nodes {
        let progress = i as f64 / (num_nodes - 1) as f64;
        let dx = surface_dx + (center_dx - surface_dx)
                 * (2.0 * (progress - 0.5).abs()).powf(2.0);

        if i == 0 {
            positions.push(dx / 2.0);
        } else {
            positions.push(positions[i-1] + dx);
        }
    }

    positions
}
```

### Fallback Decision Tree

```rust
fn select_method(&self, wall: &BuildingAssembly) -> SolverMethod {
    let tau = self.calculate_time_constant(wall);

    if tau < self.threshold_hours {
        return SolverMethod::FiveR1C;
    }

    if self.enable_fallback {
        if !self.validate_ctf_feasibility(wall) {
            return SolverMethod::FiniteDifference;
        }
    }

    SolverMethod::CTF
}
```

---

*Phase 29 Plan created: 2026-03-18*
*Ready for execution*
