# Physics Test Coverage Baseline Analysis

**Date:** 2026-03-29
**Coverage Tool:** cargo-llvm-cov
**Total Physics Lines of Code:** ~12,000

## Summary

| Metric | Value | Target | Gap |
|--------|-------|--------|-----|
| Overall Physics Coverage | ~67% | 90% | -23% |
| Files Meeting 90% Target | 1/20 | 20/20 | -19 |
| Files Below 50% | 3/20 | 0/20 | -3 |

## Detailed Module Coverage

### Priority P0 (Critical Path) - Target: 90%

| Module | Line Coverage | Status | Gap | Test Count | Notes |
|--------|--------------|--------|-----|------------|-------|
| `cta.rs` | 89.92% | ✅ | 0% | 10 | Good - minor gaps in edge cases |
| `ctf_coefficients.rs` | 74.65% | ⚠️ | -15% | 12 | Missing coefficient validation tests |
| `ctf_solver.rs` | 67.36% | ⚠️ | -23% | 20 | Missing time-varying boundary conditions |
| `fd_discretization.rs` | 80.47% | ⚠️ | -10% | 10 | Missing non-uniform grid tests |
| `fd_solver.rs` | 69.61% | ⚠️ | -20% | 13 | Missing stability limit tests |
| `five_r1c_solver.rs` | 83.70% | ⚠️ | -6% | 2 | Missing HVAC control paths |
| `newton_solver.rs` | 85.22% | ⚠️ | -5% | 0 | Missing edge case tests |
| `ctf_solver_wrapper.rs` | 84.81% | ⚠️ | -5% | 6 | Missing method switching tests |
| `fd_solver_wrapper.rs` | 80.24% | ⚠️ | -10% | 3 | Missing wrapper error handling |

**P0 Average:** 79.5% (Target: 90%, Gap: -10.5%)

### Priority P1 (Core Physics) - Target: 90%

| Module | Line Coverage | Status | Gap | Test Count | Notes |
|--------|--------------|--------|-----|------------|-------|
| `convection.rs` | 44.93% | 🔴 | -45% | 0 | **CRITICAL** - No tests for convection |
| `radiation.rs` | 80.38% | ⚠️ | -10% | 0 | Missing solar angle tests |
| `combined_heat_balance.rs` | 78.76% | ⚠️ | -11% | 0 | Missing multi-surface coupling |
| `per_surface_model.rs` | N/A | 🔴 | -90% | 0 | **NOT IN REPORT** - Likely 0% |
| `per_surface_ctf.rs` | N/A | 🔴 | -90% | 0 | **NOT IN REPORT** - Likely 0% |
| `fd_surface_balance.rs` | 34.70% | 🔴 | -55% | 15 | Missing boundary condition tests |
| `geometry_tensor.rs` | 56.03% | 🔴 | -34% | 12 | Missing tensor operations |
| `nd_array.rs` | 30.73% | 🔴 | -59% | 5 | Missing array operations |

**P1 Average:** ~44% (Target: 90%, Gap: -46%)

### Priority P2 (Supporting Code) - Target: 85%

| Module | Line Coverage | Status | Gap | Test Count | Notes |
|--------|--------------|--------|-----|------------|-------|
| `method_selector.rs` | 95.48% | ✅ | 0% | 11 | Excellent |
| `solver_manager.rs` | 76.14% | ⚠️ | -9% | 12 | Missing state preservation |
| `continuous.rs` | 92.68% | ✅ | 0% | 3 | Good |
| `solver_trait.rs` | 0.00% | ⚠️ | -100% | 0 | Trait-only (low priority) |
| `ctf_flux_direction_test.rs` | N/A | 🔴 | -100% | 0 | **NOT IN REPORT** - Test file |

**P2 Average:** ~53% (Target: 85%, Gap: -32%)

### Constants (Priority P3)

| Module | Line Coverage | Status | Gap | Notes |
|--------|--------------|--------|-----|-------|
| `annex_c.rs` | 0.00% | ⚠️ | -100% | Constant values (hard to test) |

## Critical Gaps by Category

### 1. Convection (CRITICAL - 44.93%)
**Missing Test Areas:**
- Natural convection coefficient calculation
- Forced convection (internal/external)
- Reynolds number calculations
- Nusselt number correlations
- Mixed convection regimes

**Impact:** Heat transfer calculations use unvalidated convection coefficients.

### 2. Radiation (80.38%)
**Missing Test Areas:**
- Solar incidence angle calculations
- Beam/diffuse radiation split
- Longwave sky radiation
- Surface-to-surface view factors

**Impact:** Solar gain calculations may be inaccurate.

### 3. Per-Surface Models (0% estimated)
**Missing Test Areas:**
- All of `per_surface_model.rs` (705 lines)
- All of `per_surface_ctf.rs` (1,236 lines)

**Impact:** Multi-zone simulations use unvalidated per-surface physics.

### 4. CTF Solver (67.36%)
**Missing Test Areas:**
- Time-varying boundary conditions
- CTF coefficient validation
- Heat flux direction tracking
- Multi-layer wall verification

**Impact:** Thermal mass dynamics may be inaccurate.

### 5. FD Solver (69.61%)
**Missing Test Areas:**
- Stability limit verification
- Non-uniform grids
- Implicit/explicit method switching
- Temperature profile evolution

**Impact:** Finite-difference solver may diverge for certain conditions.

## Files Not In Coverage Report

| File | Lines | Reason |
|------|-------|--------|
| `per_surface_model.rs` | 705 | Likely has no testable code paths |
| `per_surface_ctf.rs` | 1,236 | Likely has no testable code paths |
| `mod.rs` | 25 | Module exports only |
| `ctf_flux_direction_test.rs` | 146 | Test-only file |

## Test Count Analysis

| Module | Public Functions | Current Tests | Tests/Function | Target Tests |
|--------|----------------|---------------|----------------|--------------|
| `cta.rs` | ~40 | 10 | 0.25 | 25 |
| `ctf_coefficients.rs` | ~56 | 12 | 0.21 | 30 |
| `ctf_solver.rs` | ~31 | 20 | 0.65 | 45 |
| `convection.rs` | ~25 | 0 | 0.00 | 12 |
| `radiation.rs` | ~17 | 0 | 0.00 | 18 |
| `fd_solver.rs` | ~27 | 13 | 0.48 | 35 |
| `five_r1c_solver.rs` | ~11 | 2 | 0.18 | 15 |
| `newton_solver.rs` | ~37 | 0 | 0.00 | 10 |
| `fd_surface_balance.rs` | ~26 | 15 | 0.58 | 30 |

**Average tests per function:** 0.35 (Target: 1.0)

## Recommendations

### Immediate Actions (Week 1)

1. **Add Convection Tests** (Highest Priority)
   - Implement `tests/physics/convection_validation.rs`
   - Target: 12 new tests
   - Expected coverage increase: +45% (44% → 89%)

2. **Add Radiation Tests**
   - Implement `tests/physics/radiation_validation.rs`
   - Target: 18 new tests
   - Expected coverage increase: +10% (80% → 90%)

3. **Investigate Missing Coverage Reports**
   - Determine why `per_surface_model.rs` and `per_surface_ctf.rs` don't appear
   - Add test scaffolding if needed

### Short-term Actions (Weeks 2-4)

4. **Enhance CTF Coefficient Tests**
   - Add EP reference validation
   - Target: +18 new tests
   - Expected coverage: +15% (75% → 90%)

5. **Enhance FD Solver Tests**
   - Add stability and convergence tests
   - Target: +22 new tests
   - Expected coverage: +20% (70% → 90%)

6. **Add Newton Solver Tests**
   - Convergence, divergence, edge cases
   - Target: 10 new tests
   - Expected coverage: +5% (85% → 90%)

### Medium-term Actions (Weeks 5-6)

7. **Per-Surface Model Tests**
   - Implement complete test suite
   - Target: 25 new tests
   - Expected coverage: 0% → 80%

8. **Integration Test Enhancement**
   - Add energy balance verification
   - Add multi-surface coupling tests
   - Target: 20 new tests

## Coverage Trend Tracking

| Date | Overall Physics | P0 | P1 | P2 |
|-------|----------------|-----|-----|-----|
| 2026-03-29 (Baseline) | 67% | 79.5% | 44% | 53% |
| Target | 90% | 90% | 90% | 85% |
| Gap | -23% | -10.5% | -46% | -32% |

## Test Execution Summary

```
test result: FAILED. 708 passed; 25 failed; 6 ignored
```

**Failing Physics Tests:**
- `physics::combined_heat_balance::tests::test_combined_solver_multiple_surfaces`
- `physics::combined_heat_balance::tests::test_combined_solver_simple_case`
- `physics::combined_heat_balance::tests::test_combined_solver_with_hvac`
- `physics::convection::tests::test_forced_convection_exterior`
- `physics::convection::tests::test_interior_convection_ashrae140_defaults`
- `physics::convection::tests::test_natural_convection_floor_ceiling`
- `physics::ctf_coefficients::tests::test_ctf_coefficients_asymmetric_wall`
- `physics::newton_solver::tests::test_max_iterations`
- `physics::newton_solver::tests::test_newton_with_relaxation`
- `physics::newton_solver::tests::test_numerical_jacobian`

**Note:** Fixing failing tests will naturally improve coverage.

## Conclusion

The physics codebase has significant coverage gaps, particularly in:
1. **Convection** (44.93%) - Critical for heat transfer accuracy
2. **Per-surface models** (0% estimated) - Critical for multi-zone
3. **Radiation** (80.38%) - Needs +10% for 90% target
4. **CTF solver** (67.36%) - Core thermal mass dynamics
5. **FD solver** (69.61%) - Alternative solver path

The plan to achieve 90%+ coverage requires approximately **200 new tests** over 6 weeks, with emphasis on using EnergyPlus reference data for validation.

## Next Steps

1. ✅ Complete Phase 1 (Baseline Assessment)
2. ⏭️ Proceed to Phase 2 (EP Oracle Setup)
3. ⏭️ Implement convection tests (highest impact)
4. ⏭️ Implement radiation tests (high impact)
5. ⏭️ Fix existing failing tests
6. ⏭️ Add coverage enforcement to CI/CD
