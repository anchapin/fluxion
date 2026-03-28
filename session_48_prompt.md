# Session 48: CTF Solver Audit and Initial Enablement

**Date**: 2026-03-27
**Follows**: Session 47 (Peak Load Investigation - Complete)
**Status**: 📋 PLANNED
**Priority**: 🔴 CRITICAL - First phase of advanced thermal model enablement
**Estimated Duration**: 1 week

## Objective

Audit the Conduction Transfer Function (CTF) solver implementation and enable it for Case 900 as a proof-of-concept. This is the first step in moving from the 5R1C model to advanced thermal models that match EnergyPlus methodology.

## Context

### Current State

- **5R1C Model**: Used for all ASHRAE 140 cases
- **900-Series Pass Rate**: ~50% for annual energies (some close, some failing)
- **Peak Loads**: Systematically 8-33% below reference
- **Root Cause**: 5R1C model limitations (hourly timesteps, lumped thermal mass)

### Why CTF?

**CTF (Conduction Transfer Functions)** is the same methodology used by EnergyPlus:
- Handles distributed thermal mass
- Captures transient heat conduction through walls
- Matches reference tool approach
- Already implemented in `src/physics/ctf_solver.rs`

### What's Already Implemented

The codebase has CTF infrastructure but it's not enabled:
- `src/physics/ctf_coefficients.rs` - CTF coefficient calculation
- `src/physics/ctf_solver.rs` - CTF solver implementation
- `ThermalModel.ctf_enabled` - Flag to enable CTF (currently false)
- `ThermalModel.ctf_solvers` - Vector of CTF solver instances
- `ThermalModel.ctf_timestep` - Timestep duration (currently 3600s)

## Investigation Plan

### Phase 1: CTF Implementation Audit (Days 1-2)

**Step 1: Review CTF Coefficient Calculation**

Read and analyze `src/physics/ctf_coefficients.rs`:
- [ ] How are CTF coefficients calculated?
- [ ] Do they follow ASHRAE 140 specifications?
- [ ] Are wall constructions correctly represented?
- [ ] Verify coefficient values for Case 900 concrete wall

**Step 2: Review CTF Solver Implementation**

Read and analyze `src/physics/ctf_solver.rs`:
- [ ] How does the solver calculate surface temperatures?
- [ ] What is the time-stepping method?
- [ ] Are boundary conditions correct?
- [ ] Check for stability issues

**Step 3: Check Integration Points**

Search for CTF usage in codebase:
```bash
grep -r "ctf_enabled\|CTFSolver\|ctf_coefficients" src/sim/engine.rs
grep -r "CTFCalculator" src/
```

Questions to answer:
- [ ] Where would CTF be integrated into solve loop?
- [ ] Are there any existing hooks for CTF?
- [ ] What changes are needed to enable CTF?

### Phase 2: Initial Enablement for Case 900 (Days 3-4)

**Step 1: Enable CTF Flag**

Modify `src/sim/engine.rs` in `ThermalModel::from_spec()`:
```rust
// Around line 1700, after thermal mass correction
// Enable CTF for high-mass buildings
if matches!(spec.construction_type, ConstructionType::HighMass) {
    model.ctf_enabled = true;
    model.ctf_timestep = 3600.0; // Start with 1-hour timestep

    // Pre-compute CTF coefficients
    let ctf_calc = CTFCalculator::new();
    model.ctf_coefficients = Some(ctf_calc.compute_coefficients(&spec.construction));

    // Initialize CTF solvers (one per zone)
    model.ctf_solvers = spec.geometry.iter()
        .map(|g| CTFSolver::new(model.ctf_coefficients.as_ref().unwrap()))
        .collect();
}
```

**Step 2: Integrate CTF into Solve Loop**

Locate where heat conduction is calculated (around line 3200 in solve loop):
```rust
// In solve_timesteps method
if self.ctf_enabled {
    // Use CTF solver for heat conduction
    for (zone_idx, ctf_solver) in self.ctf_solvers.iter_mut().enumerate() {
        let surface_temps = ctf_solver.solve_step(
            &self.outdoor_temp,
            &self.temperatures,
            self.ctf_timestep,
        );
        // Update surface temperatures in model
        // TODO: Complete integration
    }
}
```

**Step 3: Test on Case 900**

```bash
# Build with CTF enabled
cargo build --release

# Run validation on Case 900
cargo run --release --bin fluxion validate --case 900
```

**Expected Results**:
- CTF solver runs without errors
- Energy conservation maintained
- Results differ from 5R1C (hopefully improved)

### Phase 3: Validation and Comparison (Days 5-7)

**Step 1: Compare CTF vs 5R1C**

Create comparison table for Case 900:

| Metric | 5R1C Result | CTF Result | Reference | 5R1C Error | CTF Error |
|--------|-------------|------------|-----------|------------|-----------|
| Annual Heating | 1.71 MWh | ? | 1.17-2.04 MWh | Within range | ? |
| Annual Cooling | 2.28 MWh | ? | 2.13-3.67 MWh | Within range | ? |
| Peak Heating | 1.26 kW | ? | 1.80-2.40 kW | 30% below | ? |
| Peak Cooling | 2.35 kW | ? | 1.60-2.10 kW | 12% above | ? |

**Step 2: Analyze Improvements**

- [ ] Did CTF improve annual heating?
- [ ] Did CTF improve annual cooling?
- [ ] Did CTF improve peak heating?
- [ ] Did CTF improve peak cooling?
- [ ] Are results closer to reference?

**Step 3: Debug Issues**

If CTF results are worse or have errors:
- [ ] Check coefficient calculation
- [ ] Verify solver integration
- [ ] Validate energy conservation
- [ ] Check boundary conditions

### Phase 4: Documentation (Day 7)

**Deliverables**:

1. **CTF Audit Report** (`SESSION_48_CTF_AUDIT.md`):
   - CTF coefficient calculation review
   - CTF solver implementation review
   - Integration points identified
   - Issues found and fixes applied

2. **Case 900 Results** (`SESSION_48_RESULTS.md`):
   - 5R1C vs CTF comparison
   - Validation against reference
   - Performance comparison
   - Recommendations for proceeding

3. **Session Summary** (`SESSION_48_SUMMARY.md`):
   - What was accomplished
   - What was learned
   - Next steps for Session 49

## Success Criteria

- [ ] CTF implementation audited and understood
- [ ] CTF enabled for Case 900
- [ ] CTF solver integrated into solve loop
- [ ] Case 900 runs successfully with CTF
- [ ] Results compared with 5R1C
- [ ] Decision made: proceed to Session 49 or debug further

## Go/No-Go Decision for Session 49

**Go (Proceed to Session 49) if**:
- ✅ CTF solver runs without errors
- ✅ Energy conservation maintained (<1% imbalance)
- ✅ Results show improvement or are comparable to 5R1C
- ✅ No fundamental issues with CTF implementation

**No-Go (Debug or Alternative) if**:
- ❌ CTF solver has critical bugs
- ❌ Energy conservation violated (>5% imbalance)
- ❌ Results significantly worse than 5R1C
- ❌ CTF implementation incomplete or incorrect

## Expected Outcomes

### Best Case: CTF Successful

- CTF solver works correctly
- Case 900 results improved (especially peak loads)
- Clear path to enabling CTF for all 900-series cases
- Proceed to Session 49 with confidence

### Medium Case: CTF Partially Successful

- CTF solver works but needs tuning
- Case 900 results mixed (some metrics better, some worse)
- Need to debug and adjust CTF implementation
- May need extra time before Session 49

### Worst Case: CTF Not Ready

- CTF implementation has fundamental issues
- Cannot enable without major refactoring
- Need to consider alternative approaches:
  - Fix CTF implementation (2-3 weeks)
  - Use FD solver instead
  - Accept 5R1C limitations

## Risks and Mitigations

### Risk 1: CTF Implementation Incomplete

**Mitigation**:
- Thorough audit before enabling
- Incremental testing (one component at a time)
- Fallback to 5R1C if issues arise

### Risk 2: Energy Conservation Violated

**Mitigation**:
- Add energy conservation checks
- Validate each timestep
- Compare total energy in/out

### Risk 3: Performance Degradation

**Mitigation**:
- Profile CTF solver
- Optimize hot spots
- Use 5R1C for simple cases

### Risk 4: Solver Integration Issues

**Mitigation**:
- Understand existing architecture
- Add clear integration points
- Test incrementally

## Files to Examine

1. **`src/physics/ctf_coefficients.rs`**:
   - CTF coefficient calculation
   - Wall construction handling
   - Coefficient validation

2. **`src/physics/ctf_solver.rs`**:
   - CTF solver implementation
   - Time-stepping method
   - Boundary conditions

3. **`src/sim/engine.rs`**:
   - Lines 893-1700: `ThermalModel::from_spec()` - where CTF will be enabled
   - Lines 2974-3200: `solve_timesteps()` - where CTF will be integrated
   - Lines 490-520: CTF-related struct fields

4. **`src/validation/ashrae_140_cases.rs`**:
   - Case 900 specifications
   - High-mass construction details
   - Wall assembly definitions

## Commands to Run

```bash
# Audit CTF implementation
code src/physics/ctf_coefficients.rs
code src/physics/ctf_solver.rs

# Check current CTF usage
grep -r "ctf_enabled" src/
grep -r "CTFSolver" src/
grep -r "CTFCalculator" src/

# Build with changes
cargo build --release

# Run Case 900 validation
cargo run --release --bin fluxion validate --case 900

# Compare with 5R1C baseline
# (Save 5R1C results first, then compare with CTF)

# Check energy conservation
# (Add debug output if needed)
```

## Research Questions

1. **CTF Coefficients**: Are CTF coefficients calculated correctly for ASHRAE 140 constructions?
2. **Timestep Stability**: Is 1-hour timestep stable for CTF solver?
3. **Boundary Conditions**: Are indoor/outdoor boundary conditions correct?
4. **Energy Conservation**: Does CTF maintain energy conservation?
5. **Integration**: How does CTF integrate with existing HVAC control logic?

## Dependencies

- **Session 47**: Peak load investigation complete
- **CTF Infrastructure**: Already implemented (not enabled)
- **Case 900**: High-mass baseline case (relatively simple)

## Next Session

**Session 49**: Enable CTF for all 900-series cases
- Prerequisite: Session 48 successful
- Scope: Enable CTF for Cases 900-950
- Goal: Achieve ≥67% pass rate for 900-series

## References

- **`docs/ASHRAE140_ROADMAP.md`**: Complete validation roadmap
- **`docs/ASHRAE140_QUICKSTART.md`**: Quick start guide
- **`SESSION_47_SUMMARY.md`**: Peak load investigation findings
- **`physics_based_refactor.md`**: History of physics-based improvements
- **ASHRAE 140 Standard**: Case 900 specifications
- **ISO 13790**: CTF model specification

---

**Session 48 Goal**: Audit CTF solver implementation and enable it for Case 900 as proof-of-concept, demonstrating that advanced thermal models can improve validation results.
