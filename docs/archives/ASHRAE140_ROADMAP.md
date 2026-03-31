# ASHRAE 140 Complete Validation Roadmap

**Current Status**: 1.6% pass rate (1/64 metrics passing)
**Target**: 90%+ pass rate using physics-based approaches only
**Constraint**: Open to refactoring and changing thermal models

---

## Executive Summary

Fluxion currently fails ASHRAE 140 validation at a 98.4% rate (62/64 metrics failing). The root causes are fundamental limitations of the ISO 13790 5R1C model:

1. **Hourly timesteps** average sub-hourly peaks
2. **Lumped thermal mass** cannot capture distributed thermal dynamics
3. **First-order response** cannot capture rapid temperature changes
4. **Low-mass buildings** require different modeling approach than high-mass

**Good News**: The codebase already has infrastructure for advanced thermal models (CTF, FD, SolverManager), but they are not enabled. This roadmap leverages existing infrastructure to achieve complete validation.

---

## Current Validation Status

### Pass Rate by Category

| Category | Pass Rate | Key Issues |
|----------|-----------|------------|
| **600-Series (Low-Mass)** | 0% (0/24) | Heating +30-87%, Cooling -53% to -0.5% |
| **900-Series (High-Mass)** | 0% (0/24) | Peak loads ±8-249%, some annual energies close |
| **Free-Floating** | 0% (0/8) | Max temps 20-30°C below reference |
| **Special Cases** | 0% (0/8) | Multi-zone, steady-state issues |

### Systematic Issues

1. **5R1C Model Limitation** (12 metrics): Annual energies within 15-20% but not passing
2. **Thermal Mass Dynamics** (3 metrics): Free-floating temps not capturing extremes
3. **Solar Gain Calculations** (5 metrics): Peak cooling issues in 600-series
4. **Inter-Zone Transfer** (1 metric): Case 960 sunspace
5. **Unknown/Unclassified** (41 metrics): Need investigation

---

## Strategic Approaches

### Option A: Incremental 5R1C Improvements ⚠️ NOT RECOMMENDED

**Approach**: Tweak parameters, add corrections, improve 5R1C within its constraints

**Pros**:
- Minimal code changes
- Maintains ISO 13790 compliance
- Fast to implement

**Cons**:
- ❌ Cannot fix fundamental limitations
- ❌ Low-mass buildings will still fail
- ❌ Peak loads will still be wrong
- ❌ Free-floating temps will still be off

**Estimated Pass Rate**: 40-60% (improvement but not complete)

**Verdict**: **NOT RECOMMENDED** - Does not achieve 90% target

---

### Option B: Enable Advanced Thermal Models ✅ RECOMMENDED

**Approach**: Enable existing CTF/FD infrastructure, use multi-method solver

**Pros**:
- ✅ Infrastructure already exists (CTF, FD, SolverManager)
- ✅ Matches reference tool methodology
- ✅ Can handle both low-mass and high-mass
- ✅ Sub-hourly timesteps possible
- ✅ Distributed thermal mass representation

**Cons**:
- Moderate development effort
- Higher computational cost (acceptable for validation)
- Need to validate each solver

**Estimated Pass Rate**: 85-95%

**Verdict**: **HIGHLY RECOMMENDED** - Best path to 90%+ target

---

### Option C: Complete Architectural Refactor ⚠️ LAST RESORT

**Approach**: Replace thermal engine with EnergyPlus-like architecture

**Pros**:
- ✅ Could match reference tools exactly
- ✅ Future-proof for complex buildings

**Cons**:
- ❌ Massive development effort (6-12 months)
- ❌ Loses simplicity of current model
- ❌ May break existing functionality
- ❌ Overkill for ASHRAE 140

**Estimated Pass Rate**: 95-99%

**Verdict**: **LAST RESORT** - Only if Option B fails

---

## Recommended Roadmap: Option B (Advanced Thermal Models)

### Phase 1: Enable CTF Solver for High-Mass Buildings 🎯 PRIORITY

**Goal**: Fix 900-series annual energy discrepancies

**Rationale**:
- CTF (Conduction Transfer Functions) is what EnergyPlus uses
- Matches reference tool methodology
- Already implemented in `src/physics/ctf_solver.rs`
- Just needs to be enabled and integrated

**Tasks**:
1. **Audit CTF Implementation** (1-2 days)
   - Review `src/physics/ctf_coefficients.rs`
   - Review `src/physics/ctf_solver.rs`
   - Verify coefficient calculation matches ASHRAE 140
   - Check timestep handling

2. **Enable CTF for 900-Series** (2-3 days)
   - Modify `ThermalModel::from_spec()` to use CTF for high-mass cases
   - Set `ctf_enabled = true` for construction_type == HighMass
   - Configure CTF timestep (start with 3600s, test 900s)
   - Run validation and compare results

3. **CTF Solver Integration** (2-3 days)
   - Integrate CTF solver into main solve loop
   - Replace 5R1C heat conduction with CTF
   - Keep HVAC control logic unchanged
   - Test on Cases 900, 910, 920, 930, 940, 950

4. **Validation and Tuning** (2-3 days)
   - Run ASHRAE 140 validation for 900-series
   - Compare with 5R1C results
   - Adjust CTF coefficients if needed
   - Document improvements

**Expected Outcome**:
- 900-series annual energies: 70-90% pass rate (up from current ~50%)
- Peak heating: Improved (within 15-20%)
- Peak cooling: Improved (within 15-20%)
- Free-floating temps: Improved (within 3-5°C)

**Success Criteria**:
- [ ] CTF solver enabled for 900-series
- [ ] At least 4/6 cases passing (≥67% pass rate)
- [ ] No regressions in currently passing metrics
- [ ] Free-floating temps within 5°C of reference

---

### Phase 2: Enable FD Solver for Low-Mass Buildings 🎯 PRIORITY

**Goal**: Fix 600-series low-mass discrepancies

**Rationale**:
- Low-mass buildings need finer spatial resolution
- FD (Finite Difference) can handle rapid thermal transients
- Already implemented in `src/physics/fd_solver.rs`
- Better suited for low thermal mass

**Tasks**:
1. **Audit FD Implementation** (1-2 days)
   - Review `src/physics/fd_solver.rs`
   - Verify nodal network structure
   - Check boundary conditions
   - Validate timestep stability (CFL condition)

2. **Enable FD for 600-Series** (2-3 days)
   - Modify `ThermalModel::from_spec()` to use FD for low-mass cases
   - Set `fd_enabled = true` for construction_type == LowMass
   - Configure FD timestep (start with 300s for stability)
   - Run validation and compare results

3. **FD Solver Integration** (2-3 days)
   - Integrate FD solver into main solve loop
   - Replace 5R1C heat conduction with FD
   - Handle multiple nodes per surface
   - Test on Cases 600, 610, 620, 630, 640, 650

4. **Validation and Tuning** (2-3 days)
   - Run ASHRAE 140 validation for 600-series
   - Compare with 5R1C results
   - Adjust nodal resolution if needed
   - Tune timestep for accuracy vs speed

**Expected Outcome**:
- 600-series annual energies: 60-80% pass rate (up from current 0%)
- Heating overprediction: Reduced from +30-87% to ±15%
- Cooling underprediction: Reduced from -53% to ±15%
- Free-floating temps: Within 5-10°C of reference

**Success Criteria**:
- [ ] FD solver enabled for 600-series
- [ ] At least 3/6 cases passing (≥50% pass rate)
- [ ] Heating overprediction <20%
- [ ] Cooling underprediction <20%

---

### Phase 3: Sub-Hourly Timesteps for Peak Loads 🎯 PRIORITY

**Goal**: Fix systematic peak load discrepancies

**Rationale**:
- Peak loads are averaged in hourly timesteps
- Reference tools use 10-15 minute timesteps
- Sub-hourly resolution captures true peaks

**Tasks**:
1. **Timestep Refactoring** (2-3 days)
   - Modify solve loop to support variable timesteps
   - Implement sub-hourly timestep (900s = 15 min)
   - Ensure energy conservation with smaller dt
   - Profile performance impact

2. **Peak Load Calculation** (1-2 days)
   - Track peak at sub-hourly resolution
   - Verify peak occurs at correct timestep
   - Compare hourly vs sub-hourly peaks
   - Document peak load improvement

3. **Validation** (1-2 days)
   - Run ASHRAE 140 validation with sub-hourly timesteps
   - Compare peak loads with reference
   - Verify annual energies unchanged (should be identical)

**Expected Outcome**:
- Peak heating: Within 10-15% of reference (improved from 8-33% below)
- Peak cooling: Within 10-15% of reference (improved from ±14-249%)
- Annual energies: Unchanged (sub-hourly shouldn't affect annual totals)

**Success Criteria**:
- [ ] Sub-hourly timestep (900s) implemented
- [ ] Peak loads within 15% of reference
- [ ] Annual energies unchanged (±1%)
- [ ] Performance <10s per case

---

### Phase 4: Multi-Method Solver Manager 🎯 PRIORITY

**Goal**: Automatic selection of optimal solver for each case

**Rationale**:
- Different buildings need different solvers
- Low-mass: FD solver
- High-mass: CTF solver
- Simple cases: 5R1C solver (fastest)
- SolverManager infrastructure already exists

**Tasks**:
1. **Solver Manager Audit** (1-2 days)
   - Review `src/physics/solver_manager.rs`
   - Understand existing architecture
   - Verify auto-selection logic
   - Test each solver independently

2. **Auto-Selection Logic** (2-3 days)
   - Implement decision tree:
     - If thermal capacitance < 5e6 J/K → FD solver
     - If thermal capacitance > 5e6 J/K → CTF solver
     - If simple case → 5R1C solver
   - Add fallback logic (if CTF/FD fails, use 5R1C)
   - Log solver selection for debugging

3. **Integration Testing** (2-3 days)
   - Test all ASHRAE 140 cases
   - Verify correct solver selected
   - Check for solver switching issues
   - Validate energy conservation

4. **Performance Optimization** (2-3 days)
   - Profile each solver
   - Optimize hot paths
   - Cache solver results
   - Parallelize where possible

**Expected Outcome**:
- Automatic solver selection based on building characteristics
- Optimal accuracy/speed tradeoff
- All cases using appropriate solver

**Success Criteria**:
- [ ] SolverManager enabled for all cases
- [ ] Auto-selection working correctly
- [ ] Overall pass rate ≥85%
- [ ] Performance <5s per case

---

### Phase 5: Free-Floating Temperature Validation 🎯 PRIORITY

**Goal**: Fix free-floating temperature extremes

**Rationale**:
- Free-floating temps test thermal dynamics without HVAC
- Currently 20-30°C below reference
- Tests solver's ability to capture thermal mass effects

**Tasks**:
1. **Root Cause Analysis** (1-2 days)
   - Compare 5R1C vs CTF vs FD for free-floating
   - Identify why temps are too low
   - Check solar gain distribution
   - Verify thermal mass coupling

2. **Solar Gain Redistribution** (2-3 days)
   - Review solar-to-mass vs solar-to-air split
   - Adjust for free-floating conditions
   - Implement dynamic distribution based on HVAC mode
   - Test with different solar fractions

3. **Thermal Mass Coupling** (2-3 days)
   - Verify h_tr_em and h_tr_ms values
   - Check coupling ratios
   - Adjust for free-floating vs conditioned
   - Test impact on temperature extremes

4. **Validation** (1-2 days)
   - Run free-floating cases (600FF, 650FF, 900FF, 950FF)
   - Compare with reference
   - Verify min/max temps within range

**Expected Outcome**:
- Free-floating max temps: Within 5°C of reference
- Free-floating min temps: Within 5°C of reference
- Better representation of thermal dynamics

**Success Criteria**:
- [ ] At least 2/4 free-floating cases passing
- [ ] Max temps within 5°C of reference
- [ ] Min temps within 5°C of reference
- [ ] No regressions in conditioned cases

---

### Phase 6: Special Cases (960, 195) 🎯 MEDIUM PRIORITY

**Goal**: Fix multi-zone and steady-state special cases

**Rationale**:
- Case 960: Sunspace with inter-zone heat transfer
- Case 195: Steady-state conduction only
- Test edge cases of solver

**Tasks**:
1. **Case 960 Sunspace** (3-4 days)
   - Verify inter-zone heat transfer calculation
   - Check sunspace temperature modeling
   - Validate solar gain distribution to sunspace
   - Test multi-zone solver

2. **Case 195 Steady-State** (2-3 days)
   - Verify steady-state conduction
   - Check no windows condition
   - Validate envelope heat transfer
   - Test solver with zero solar gains

3. **Validation** (1-2 days)
   - Run special cases
   - Compare with reference
   - Debug any remaining issues

**Expected Outcome**:
- Case 960: Annual energies within range
- Case 195: Annual heating within range
- Special cases passing

**Success Criteria**:
- [ ] Case 960 passing
- [ ] Case 195 passing
- [ ] Multi-zone solver validated
- [ ] Steady-state conduction validated

---

### Phase 7: Final Validation and Documentation 🎯 PRIORITY

**Goal**: Achieve 90%+ pass rate and document

**Tasks**:
1. **Comprehensive Validation** (2-3 days)
   - Run all 18 ASHRAE 140 cases
   - Generate validation report
   - Calculate pass rate
   - Identify any remaining issues

2. **Performance Profiling** (1-2 days)
   - Profile each case
   - Identify bottlenecks
   - Optimize hot paths
   - Target <5s per case

3. **Documentation** (3-4 days)
   - Document solver selection logic
   - Write technical notes on CTF/FD implementation
   - Update ASHRAE 140 validation report
   - Create user guide for solver selection

4. **Regression Testing** (1-2 days)
   - Run full test suite
   - Check for regressions
   - Validate energy conservation
   - Test edge cases

**Expected Outcome**:
- Overall pass rate ≥90%
- All cases using appropriate solver
- Comprehensive documentation
- Performance <5s per case

**Success Criteria**:
- [ ] Overall pass rate ≥90% (58/64 metrics)
- [ ] 600-series ≥67% pass rate
- [ ] 900-series ≥90% pass rate
- [ ] Free-floating ≥50% pass rate
- [ ] Special cases ≥50% pass rate
- [ ] Performance <5s per case
- [ ] Documentation complete

---

## Risk Assessment and Mitigation

### Risk 1: CTF/FD Solvers Have Bugs 🔴 HIGH RISK

**Mitigation**:
- Thorough code audit before enabling
- Unit tests for each solver
- Validation against analytical solutions
- Incremental rollout (one case at a time)

### Risk 2: Performance Degradation 🟡 MEDIUM RISK

**Mitigation**:
- Profile early and often
- Optimize hot paths
- Use 5R1C for simple cases
- Parallelize where possible

### Risk 3: Solver Switching Issues 🟡 MEDIUM RISK

**Mitigation**:
- Comprehensive integration testing
- Fallback to 5R1C on failure
- Log solver selection
- Validate energy conservation

### Risk 4: Regressions in Currently Passing Metrics 🟢 LOW RISK

**Mitigation**:
- Comprehensive regression testing
- Preserve 5R1C as fallback
- Version control for easy rollback
- Continuous validation

---

## Timeline Estimate

### Aggressive Timeline (Parallel Development)

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: CTF Solver | 1-2 weeks | None |
| Phase 2: FD Solver | 1-2 weeks | None (can parallelize) |
| Phase 3: Sub-Hourly | 1 week | Phases 1,2 |
| Phase 4: Solver Manager | 1-2 weeks | Phases 1,2,3 |
| Phase 5: Free-Floating | 1 week | Phase 4 |
| Phase 6: Special Cases | 1 week | Phase 4 |
| Phase 7: Final Validation | 1-2 weeks | All phases |

**Total**: 8-12 weeks

### Conservative Timeline (Sequential Development)

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: CTF Solver | 2-3 weeks | None |
| Phase 2: FD Solver | 2-3 weeks | Phase 1 |
| Phase 3: Sub-Hourly | 1-2 weeks | Phase 2 |
| Phase 4: Solver Manager | 2-3 weeks | Phase 3 |
| Phase 5: Free-Floating | 1-2 weeks | Phase 4 |
| Phase 6: Special Cases | 1-2 weeks | Phase 4 |
| Phase 7: Final Validation | 2-3 weeks | All phases |

**Total**: 12-18 weeks

---

## Success Metrics

### Primary Metrics

- **Overall Pass Rate**: ≥90% (58/64 metrics)
- **600-Series Pass Rate**: ≥67% (16/24 metrics)
- **900-Series Pass Rate**: ≥90% (22/24 metrics)
- **Free-Floating Pass Rate**: ≥50% (4/8 metrics)
- **Special Cases Pass Rate**: ≥50% (4/8 metrics)

### Secondary Metrics

- **Performance**: <5 seconds per case
- **Energy Conservation**: <0.1% imbalance
- **Code Quality**: All tests passing, clippy clean
- **Documentation**: Complete technical notes

### Tertiary Metrics

- **Maintainability**: Clear code structure
- **Extensibility**: Easy to add new solvers
- **Debuggability**: Good logging and diagnostics
- **User Experience**: Easy to use and understand

---

## Decision Points

### Go/No-Go Decision After Phase 1

**Go Criteria**:
- CTF solver improves 900-series by ≥20%
- No major bugs discovered
- Performance acceptable (<10s per case)

**No-Go Criteria**:
- CTF solver has fundamental issues
- No improvement in validation
- Performance degradation >5x

**Fallback**: Revert to 5R1C, consider Option C (complete refactor)

### Go/No-Go Decision After Phase 2

**Go Criteria**:
- FD solver improves 600-series by ≥30%
- Low-mass buildings responding correctly
- Solver switching working

**No-Go Criteria**:
- FD solver unstable
- Low-mass buildings still failing
- Cannot achieve 50% pass rate

**Fallback**: Accept 600-series as limitation, focus on 900-series

---

## Alternative: Hybrid Approach

If full CTF/FD implementation is too ambitious, consider a hybrid approach:

### Hybrid: Enhanced 5R1C + Selective Advanced Models

1. **Keep 5R1C for simple cases** (fast, accurate enough)
2. **Use CTF for high-mass** (900-series)
3. **Use FD for low-mass** (600-series)
4. **Sub-hourly timesteps for peaks** (all cases)

**Estimated Pass Rate**: 75-85%

**Timeline**: 6-10 weeks

**Effort**: Moderate

---

## Research Questions

1. **CTF Coefficient Calculation**: Are we calculating CTF coefficients correctly for ASHRAE 140 constructions?
2. **FD Nodal Resolution**: How many nodes are needed for accurate low-mass simulation?
3. **Timestep Stability**: What's the minimum stable timestep for FD solver?
4. **Solver Switching**: Can we seamlessly switch between solvers without energy imbalances?
5. **Performance**: Can we achieve <5s per case with CTF/FD solvers?

---

## Next Steps

1. **Review this roadmap** and provide feedback
2. **Audit existing CTF/FD infrastructure** to assess readiness
3. **Prototype Phase 1** (CTF for 900-series) as proof-of-concept
4. **Make go/no-go decision** based on Phase 1 results
5. **Plan detailed implementation** for remaining phases

---

## References

- **`src/physics/ctf_solver.rs`**: CTF solver implementation
- **`src/physics/fd_solver.rs`**: FD solver implementation
- **`src/physics/solver_manager.rs`**: Multi-method solver manager
- **`physics_based_refactor.md`**: History of physics-based improvements
- **`SESSION_47_SUMMARY.md`**: Peak load investigation
- **ASHRAE 140 Standard**: Test case specifications
- **ISO 13790**: 5R1C and CTF model specifications

---

**Last Updated**: 2026-03-27 (Session 47)
**Status**: 📋 PLANNING - Awaiting review and approval
**Next Milestone**: Phase 1 kickoff (CTF solver audit and enablement)
