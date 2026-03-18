# Phase 28 Planning Complete

**Date:** 2026-03-17
**Phase:** 28 - CTF/FD Solver Integration
**Status:** ✅ PLANNED (6 plans ready to execute)

---

## Summary

Successfully created comprehensive plan for Phase 28: CTF/FD Solver Integration.

**Phase 28 Goal:** Integrate advanced heat conduction solvers (CTF and Finite Difference) into production thermal model with automatic method selection.

**v0.7 Context:** Complete high-mass accuracy work started in v0.6 (75% improvement) by integrating CTF/FD solvers to achieve ±15% tolerance for all ASHRAE 140 cases.

---

## Plans Created

| Plan | Name | Effort | Key Deliverables |
|------|------|--------|------------------|
| 28-01 | CTF Solver Core Integration | 2-3 days | CTF wired into thermal model |
| 28-02 | Finite Difference Solver Integration | 2-3 days | FD wired as fallback option |
| 28-03 | Method Selector Implementation | 1-2 days | Automatic solver selection (τ > 2h) |
| 28-04 | Thermal Model Refactoring | 2-3 days | Trait-based solver abstraction |
| 28-05 | Python API and CLI Integration | 1-2 days | User-facing configuration |
| 28-06 | Unit Tests for CTF/FD Solvers | 1-2 days | 20+ comprehensive tests |

**Total Effort:** 9-15 days (2-3 weeks) + 1-3 weeks buffer = **4-6 weeks total**

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `.planning/phases/28-ctf-fd-solver-integration/28-00-PHASE-SUMMARY.md` | ~350 | Phase overview |
| `.planning/phases/28-ctf-fd-solver-integration/28-01-PLAN.md` | ~300 | CTF integration plan |
| `.planning/phases/28-ctf-fd-solver-integration/28-02-PLAN.md` | ~300 | FD integration plan |
| `.planning/phases/28-ctf-fd-solver-integration/28-03-PLAN.md` | ~350 | Method selector plan |
| `.planning/phases/28-ctf-fd-solver-integration/28-04-PLAN.md` | ~350 | Solver abstraction plan |
| `.planning/phases/28-ctf-fd-solver-integration/28-05-PLAN.md` | ~350 | Python API & CLI plan |
| `.planning/phases/28-ctf-fd-solver-integration/28-06-PLAN.md` | ~400 | Unit tests plan |

**Total:** ~2,400 lines of detailed planning documentation

---

## Execution Strategy

### Wave 1: Core Integration (Week 1-2)
**Parallel:**
- Plan 28-01: CTF Solver Core Integration
- Plan 28-02: Finite Difference Solver Integration

**Deliverable:** Both solvers wired into thermal model

---

### Wave 2: Method Selection & Abstraction (Week 2-3)
**Sequential:**
- Plan 28-03: Method Selector Implementation
- Plan 28-04: Thermal Model Refactoring

**Deliverable:** Automatic solver selection with trait abstraction

---

### Wave 3: API & Testing (Week 3-4)
**Parallel:**
- Plan 28-05: Python API and CLI Integration
- Plan 28-06: Unit Tests for CTF/FD Solvers

**Deliverable:** User-facing configuration + comprehensive tests

---

## Requirements Coverage

### SOLVER Requirements (v0.7)

| Requirement | Plan | Status |
|-------------|------|--------|
| SOLVER-01: CTF integrated | 28-01 | ✅ Planned |
| SOLVER-02: FD integrated | 28-02 | ✅ Planned |
| SOLVER-03: Auto method selection | 28-03 | ✅ Planned |
| SOLVER-04: Threshold configuration | 28-03, 28-05 | ✅ Planned |
| SOLVER-05: Zero-copy data sharing | 28-04 | ✅ Planned |
| SOLVER-06: Python API & CLI | 28-05 | ✅ Planned |

**Coverage:** 6/6 SOLVER requirements (100%)

---

## Technical Approach

### CTF Integration (Plan 28-01)

**Existing Code:**
- `src/physics/ctf_solver.rs` (427 lines)
- `src/physics/ctf_coefficients.rs` (483 lines)

**Integration:**
1. Precompute CTF coefficients during initialization
2. Maintain temperature/flux history buffers
3. Calculate heat flux using CTF formula
4. Wire into thermal model timestep loop

**Formula:**
```
q''_int,t = -Z₀·T_int,t + Σ(X_j·T_ext,t-j) - Σ(Y_j·T_int,t-j) - Σ(Φ_j·q''_t-j)
```

---

### FD Integration (Plan 28-02)

**Existing Code:**
- `src/physics/fd_solver.rs` (655 lines)
- `src/physics/fd_discretization.rs`

**Integration:**
1. Create wall discretization from layers
2. Apply Robin boundary conditions
3. Solve tridiagonal system (Thomas algorithm)
4. Extract surface heat flux

**Algorithm:**
```
Implicit BTCS: -Fo·T_{i-1}^{n+1} + (1+2Fo)·T_i^{n+1} - Fo·T_{i+1}^{n+1} = T_i^n
```

---

### Method Selection (Plan 28-03)

**Strategy:**
- Calculate thermal mass time constant (τ)
- τ < 2 hours → 5R1C (low mass, fast)
- τ ≥ 2 hours → CTF (high mass, accurate)
- CTF fails → FD (fallback)

**Formula:**
```
τ = (Σ ρ_i · c_p,i · L_i) / (h_interior + h_exterior)
```

**Typical Values:**
- Wood frame: τ ≈ 0.5-1.5h → 5R1C
- Concrete (200mm): τ ≈ 3-5h → CTF
- Adobe (300mm): τ ≈ 8-12h → CTF/FD

---

### Solver Abstraction (Plan 28-04)

**Trait Interface:**
```rust
pub trait HeatConductionSolver: Send + Sync {
    fn name(&self) -> &str;
    fn initialize(&mut self, wall: &WallAssembly) -> Result<(), SolverError>;
    fn step(&mut self, timestep: f64, T_int: f64, T_ext: f64,
            h_int: f64, h_ext: f64) -> Result<f64, SolverError>;
    fn energy_storage_rate(&self) -> f64;
    fn is_valid(&self) -> bool;
}
```

**Implementations:**
- `FiveR1CSolver` - Baseline 5R1C thermal network
- `CTFSolverWrapper` - CTF solver wrapper
- `FDSolverWrapper` - FD solver wrapper

---

### Python API & CLI (Plan 28-05)

**Python API:**
```python
class ModelConfig:
    solver_method: str = "auto"  # "auto", "5r1c", "ctf", "fd"
    solver_threshold_hours: float = 2.0
    ctf_history_size: int = 50
    fd_nodes_per_layer: int = 10
```

**CLI:**
```bash
fluxion simulate --solver-method auto \
                 --solver-threshold-hours 2.0 \
                 --ctf-history-size 50 \
                 --fd-nodes-per-layer 10
```

---

### Unit Tests (Plan 28-06)

**Test Categories:**
1. CTF coefficient calculation (2 tests)
2. CTF heat flux calculation (2 tests)
3. FD discretization (2 tests)
4. FD steady-state (2 tests)
5. FD transient response (1 test)
6. Energy conservation (1 test)
7. Solver comparison (2 tests)

**Total:** 12+ tests

**Success Criteria:**
- All tests pass
- Energy balance closes within ±1%
- CTF vs FD agreement within ±10%

---

## Verification Criteria

### Code Quality
- ✅ `cargo check --release` passes
- ✅ `cargo clippy` passes
- ✅ `cargo fmt --check` passes

### Functional Testing
- ✅ `cargo test ctf` passes (8+ tests)
- ✅ `cargo test fd` passes (8+ tests)
- ✅ `cargo test solver` passes (4+ tests)

### Performance
- ✅ CTF throughput ≥800 configs/sec
- ✅ FD throughput ≥500 configs/sec
- ✅ Method selection overhead <1ms
- ✅ Abstraction overhead <5%

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CTF integration complex | Medium | High | FD fallback available |
| Performance below target | Low | Medium | Optimize coefficient caching |
| Regression on low-mass | Low | High | Extensive testing |
| Edge cases | Medium | Low | FD fallback for CTF failures |

### Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Integration takes longer | Medium | Medium | Parallel development |
| Debugging solver issues | Medium | Medium | Buffer time (1-3 weeks) |
| API changes needed | Low | Low | Minimal API first |

---

## Next Actions

1. **Execute Phase 28** using `/gsd:execute-phase 28`
   - Wave 1: Core integration (Week 1-2)
   - Wave 2: Method selection & abstraction (Week 2-3)
   - Wave 3: API & testing (Week 3-4)

2. **Plan Phase 29** using `/gsd:plan-phase 29`
   - Full ASHRAE 140 validation (18 cases)
   - Performance benchmarking
   - Validation report

3. **Plan Phase 30** using `/gsd:plan-phase 30`
   - Documentation completion
   - v0.7.0 release preparation

---

## Success Measures

**Phase 28 is complete when:**

1. ✅ CTF solver integrated and working
2. ✅ FD solver integrated and working
3. ✅ Method selector chooses appropriate solver automatically
4. ✅ All solvers produce physically realistic results
5. ✅ Python API and CLI configuration working
6. ✅ All unit tests passing (20+ tests)
7. ✅ Energy balance closes within ±1%
8. ✅ Performance targets met (≥800 configs/sec CTF)

---

## Files Updated

| File | Change |
|------|--------|
| `.planning/STATE.md` | Updated for Phase 28 planned |
| `.planning/ROADMAP.md` | Updated progress table |
| `.planning/phases/28-ctf-fd-solver-integration/` | Created with 7 files |

---

## Context from v0.6

**v0.6 Achievement:** 75% improvement in high-mass accuracy
- Case 920 heating: 2.29 MWh (31% below reference, down from 229-322%)
- CTF/FD solvers implemented but experimental

**v0.7 Goal:** Complete high-mass accuracy
- Target: ±15% annual energy tolerance for all ASHRAE 140 cases
- Method: Integrate CTF/FD solvers into production thermal model
- Success: Case 900 heating 1.17-2.04 MWh, cooling 2.13-3.67 MWh

---

*Planning summary created: 2026-03-17*
