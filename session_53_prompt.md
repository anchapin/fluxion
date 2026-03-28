# Session 53: Multi-Method Solver Manager

**Date**: 2026-03-27
**Follows**: Session 52 (Sub-Hourly Timesteps)
**Status**: 📋 PLANNED
**Priority**: 🟡 HIGH - Auto-select optimal solver for each case
**Estimated Duration**: 1 week
**Prerequisite**: Sessions 49-52 successful (CTF, FD, sub-hourly all working)

## Objective

Implement SolverManager to automatically select the optimal thermal solver (5R1C, CTF, or FD) based on building characteristics. This completes the multi-method solver infrastructure and ensures each case uses the most appropriate solver.

## Context

### Current Solver Status
- **5R1C**: Working but limited accuracy
- **CTF**: Enabled for 900-series (high-mass) - Session 49
- **FD**: Enabled for 600-series (low-mass) - Session 51
- **Sub-hourly**: 15-minute timesteps - Session 52

### Problem
Currently each case uses a manually-selected solver. Need automatic selection based on:
- Thermal capacitance (high vs low mass)
- Construction complexity
- Performance requirements

### SolverManager Infrastructure
Already implemented in `src/physics/solver_manager.rs` but not integrated.

## Implementation Plan

### Phase 1: Audit SolverManager (Days 1-2)
- Review existing SolverManager implementation
- Understand auto-selection logic
- Test each solver independently
- Verify switching works without energy imbalances

### Phase 2: Implement Auto-Selection Logic (Days 2-3)

**Decision Tree**:
```rust
fn select_solver(spec: &CaseSpec) -> SolverType {
    let thermal_cap = spec.construction.calculate_thermal_capacitance();

    if thermal_cap < 5.0e6 {
        // Low-mass: Use FD solver
        SolverType::FiniteDifference
    } else if thermal_cap > 1.0e7 {
        // High-mass: Use CTF solver
        SolverType::ConductionTransferFunction
    } else {
        // Medium-mass or simple: Use 5R1C
        SolverType::FiveResistanceOneCapacitance
    }
}
```

### Phase 3: Integration Testing (Days 3-4)
- Test all 18 cases with auto-selected solvers
- Verify correct solver chosen for each case
- Check for energy conservation when switching
- Validate results match manual solver selection

### Phase 4: Performance Optimization (Days 4-5)
- Profile each solver
- Add caching where beneficial
- Optimize hot paths
- Target: <5s per case

### Phase 5: Documentation (Days 6-7)
- Document solver selection criteria
- Create user guide
- Update validation report

## Success Criteria
- [ ] Auto-selection working for all cases
- [ ] Each case uses optimal solver
- [ ] No energy conservation issues
- [ ] Overall pass rate ≥85%
- [ ] Performance <5s per case

## Expected Outcomes
- **Best Case**: All cases auto-select correctly, 90%+ pass rate
- **Medium Case**: Most cases correct, 85% pass rate, need tuning
- **Worst Case**: Selection issues, need debugging

## Files to Modify
- `src/physics/solver_manager.rs`
- `src/sim/engine.rs` (integration)
- `src/validation/ashrae_140_validator.rs`

## Commands
```bash
# Test auto-selection
cargo run --release --bin fluxion validate --all --auto-solver

# Verify solver selection
grep -r "SolverType" validation_results.txt

# Profile performance
time cargo run --release --bin fluxion validate --all
```

## Next Session
**Session 54**: Free-Floating Temperature Validation

## References
- `docs/ASHRAE140_ROADMAP.md` Phase 4
- `src/physics/solver_manager.rs`

---

**Session 53 Goal**: Implement automatic solver selection based on building characteristics, achieving optimal accuracy/speed tradeoff and ≥85% overall pass rate.
