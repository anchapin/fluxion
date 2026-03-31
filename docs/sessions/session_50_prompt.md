# Session 50: FD Solver Audit and Initial Enablement

**Date**: 2026-03-27
**Follows**: Session 49 (CTF Integration for 900-Series)
**Status**: 📋 PLANNED
**Priority**: 🔴 CRITICAL - First phase of low-mass building support
**Estimated Duration**: 1 week
**Prerequisite**: Session 49 successful (CTF working for 900-series)

## Objective

Audit the Finite Difference (FD) solver implementation and enable it for Case 600 as a proof-of-concept. This addresses the 600-series low-mass cases which currently have 0% pass rate with the 5R1C model.

## Context

### Current 600-Series Status (5R1C Model)

| Case | Annual Heating | Ref Range | Error | Annual Cooling | Ref Range | Error |
|------|----------------|-----------|-------|----------------|-----------|-------|
| 600 | 8.65 MWh | 5.50-7.50 | **+54% over** | 6.53 MWh | 8.00-10.50 | **-30% below** |
| 610 | 9.08 MWh | 4.36-5.79 | **+67% over** | 4.56 MWh | 3.92-6.14 | **-0.5% below** |
| 620 | 7.90 MWh | 4.50-6.50 | **+30% over** | 2.29 MWh | 3.20-5.00 | **-39% below** |
| 630 | 9.04 MWh | 5.05-6.47 | **+45% over** | 1.12 MWh | 2.13-3.70 | **-53% below** |
| 640 | 7.12 MWh | 2.75-3.80 | **+87% over** | 5.45 MWh | 5.95-8.10 | **-8% below** |
| 650 | 0.00 MWh | 0.00-0.00 | ✅ | 4.65 MWh | 4.82-7.06 | **-11% below** |

**Pattern**:
- **Heating**: Systematic overprediction (+30% to +87%)
- **Cooling**: Systematic underprediction (-53% to -0.5%)
- **Root Cause**: 5R1C lumped thermal mass not suitable for low-mass buildings

### Free-Floating Temperature Evidence

**600-Series Free-Floating Results**:
| Case | Max Temp | Reference | Difference |
|------|----------|-----------|------------|
| 600FF | 45.66°C | 64.90-75.10°C | **20-30°C below** |
| 650FF | 43.71°C | 63.20-73.50°C | **20-30°C below** |

This confirms that low-mass buildings behave fundamentally differently in our model.

### Why Finite Difference?

**FD (Finite Difference)** is better suited for low-mass buildings:
- **Nodal Resolution**: Multiple nodes through wall thickness
- **Rapid Transients**: Captures fast temperature changes
- **Low Thermal Mass**: Handles small thermal capacitance correctly
- **Flexibility**: Can adjust nodal resolution based on needs

### What's Already Implemented

The codebase has FD infrastructure but it's not enabled:
- `src/physics/fd_solver.rs` - FD solver implementation
- `ThermalModel.fd_enabled` - Flag to enable FD (currently false)
- `ThermalModel.fd_solvers` - Vector of FD solver instances
- `ThermalModel.fd_timestep` - Timestep duration (currently 3600s)

## Investigation Plan

### Phase 1: FD Implementation Audit (Days 1-2)

**Step 1: Review FD Solver Implementation**

Read and analyze `src/physics/fd_solver.rs`:
- [ ] How is the nodal network structured?
- [ ] What is the spatial discretization?
- [ ] How many nodes through wall thickness?
- [ ] What is the time-stepping method?
- [ ] Is it explicit or implicit?

**Step 2: Check Stability Criteria**

For FD solvers, timestep must satisfy CFL condition:
- [ ] What is the stability criterion?
- [ ] What timestep is required for stability?
- [ ] Is 1-hour timestep stable for low-mass?
- [ ] Do we need 5-minute or smaller timestep?

**Step 3: Verify Boundary Conditions**

- [ ] Indoor boundary condition (surface temperature)
- [ ] Outdoor boundary condition (ambient temperature)
- [ ] Solar gain boundary condition
- [ ] Inter-nodal conductances

**Step 4: Check Integration Points**

Search for FD usage in codebase:
```bash
grep -r "fd_enabled\|FDSolver\|ImplicitFDSolver" src/sim/engine.rs
grep -r "fd_solver" src/
```

Questions to answer:
- [ ] Where would FD be integrated into solve loop?
- [ ] Are there any existing hooks for FD?
- [ ] What changes are needed to enable FD?

### Phase 2: Initial Enablement for Case 600 (Days 3-4)

**Step 1: Determine Appropriate Timestep**

For low-mass buildings, FD typically requires smaller timesteps:
- **CFL Condition**: `dt ≤ dx² / (2α)` where α is thermal diffusivity
- **For typical wall**: 5-15 minutes may be needed
- **Start with**: 5 minutes (300 seconds) for stability

**Step 2: Enable FD Flag**

Modify `src/sim/engine.rs` in `ThermalModel::from_spec()`:
```rust
// Around line 1700, after thermal mass correction
// Enable FD for low-mass buildings
if matches!(spec.construction_type, ConstructionType::LowMass) {
    model.fd_enabled = true;
    model.fd_timestep = 300.0; // 5-minute timestep for stability

    // Initialize FD solvers with nodal network
    model.fd_solvers = spec.geometry.iter()
        .map(|g| {
            // Create nodal network through wall thickness
            let num_nodes = 5; // 5 nodes through wall
            let wall_thickness = 0.3; // 300 mm typical
            let dx = wall_thickness / num_nodes as f64;

            ImplicitFDSolver::new(
                &spec.construction.wall,
                g.wall_area,
                num_nodes,
                dx,
                model.fd_timestep,
            )
        })
        .collect();
}
```

**Step 3: Integrate FD into Solve Loop**

Locate where heat conduction is calculated (around line 3200 in solve loop):
```rust
// In solve_timesteps method
if self.fd_enabled {
    // Use FD solver for heat conduction
    for (zone_idx, fd_solver) in self.fd_solvers.iter_mut().enumerate() {
        let surface_temps = fd_solver.solve_step(
            &self.outdoor_temp,
            &self.temperatures,
            self.fd_timestep,
        );
        // Update surface temperatures in model
        // TODO: Complete integration
    }
}
```

**Step 4: Handle Sub-Hourly Timesteps**

With 5-minute timestep, we need 12x more timesteps for annual simulation:
```rust
// Instead of 8760 steps, use 8760 * 12 = 105120 steps
let num_steps_per_hour = (3600.0 / model.fd_timestep) as usize;
let total_steps = 8760 * num_steps_per_hour;

for step in 0..total_steps {
    let hour_of_day = (step / num_steps_per_hour) % 24;
    // ... solve step
}
```

**Step 5: Test on Case 600**

```bash
# Build with FD enabled
cargo build --release

# Run validation on Case 600
cargo run --release --bin fluxion validate --case 600
```

**Expected Results**:
- FD solver runs without errors
- Energy conservation maintained
- Heating overprediction reduced
- Cooling underprediction reduced

### Phase 3: Validation and Comparison (Days 5-6)

**Step 1: Compare FD vs 5R1C**

Create comparison table for Case 600:

| Metric | 5R1C Result | FD Result | Reference | 5R1C Error | FD Error | Improvement |
|--------|-------------|-----------|-----------|------------|----------|-------------|
| Annual Heating | 8.65 MWh | ? | 5.50-7.50 MWh | +54% over | ? | ? |
| Annual Cooling | 6.53 MWh | ? | 8.00-10.50 MWh | -30% below | ? | ? |
| Peak Heating | 4.43 kW | ? | 2.80-3.80 kW | +62% over | ? | ? |
| Peak Cooling | 5.04 kW | ? | 4.80-6.20 kW | +5% over | ? | ? |

**Step 2: Analyze Improvements**

- [ ] Did FD reduce heating overprediction?
- [ ] Did FD reduce cooling underprediction?
- [ ] Did FD improve peak loads?
- [ ] Are results closer to reference?
- [ ] Is energy balance improved?

**Step 3: Check Free-Floating Temperatures**

```bash
# Run free-floating case
cargo run --release --bin fluxion validate --case 600FF
```

**Expected**:
- Max temperature should increase (closer to 64.90-75.10°C)
- FD should capture temperature swings better than 5R1C

**Step 4: Debug Issues**

If FD results are worse or have errors:
- [ ] Check timestep stability (CFL condition)
- [ ] Verify nodal resolution (try more nodes)
- [ ] Check boundary conditions
- [ ] Validate energy conservation

### Phase 4: Performance Analysis (Day 6)

**Step 1: Profile FD Solver**

```bash
# Profile Case 600 with FD
time cargo run --release --bin fluxion validate --case 600
```

**Expected Performance**:
- FD with 5-minute timestep: 12x slower than hourly
- Target: <30 seconds per case
- If slower: Need optimization

**Step 2: Optimization Strategies**

If performance is poor:
- [ ] Reduce nodal resolution (try 3 nodes instead of 5)
- [ ] Increase timestep (try 10 minutes if stable)
- [ ] Optimize solver code
- [ ] Use adaptive timestepping

### Phase 5: Documentation (Day 7)

**Deliverables**:

1. **FD Audit Report** (`SESSION_50_FD_AUDIT.md`):
   - FD solver implementation review
   - Nodal network structure
   - Stability analysis (CFL condition)
   - Timestep selection rationale

2. **Case 600 Results** (`SESSION_50_RESULTS.md`):
   - 5R1C vs FD comparison
   - Validation against reference
   - Free-floating temperature analysis
   - Performance metrics

3. **Session Summary** (`SESSION_50_SUMMARY.md`):
   - What was accomplished
   - What was learned
   - Next steps for Session 51

## Success Criteria

- [ ] FD implementation audited and understood
- [ ] FD enabled for Case 600
- [ ] FD solver integrated into solve loop
- [ ] Case 600 runs successfully with FD
- [ ] Heating overprediction reduced by ≥50%
- [ ] Cooling underprediction reduced by ≥50%
- [ ] Energy conservation maintained (<1% imbalance)
- [ ] Performance <30s per case

## Go/No-Go Decision for Session 51

**Go (Proceed to Session 51) if**:
- ✅ FD solver runs without errors
- ✅ Heating overprediction reduced (at least 25%)
- ✅ Cooling underprediction reduced (at least 25%)
- ✅ Energy conservation maintained
- ✅ No fundamental issues with FD

**Conditional Go (Proceed with Tuning) if**:
- ⚠️ FD runs but results mixed
- ⚠️ Some improvement but not enough
- ⚠️ Performance slower than expected
- ⚠️ Needs timestep/nodal tuning
- → Proceed to Session 51 but continue optimization

**No-Go (Reconsider Approach) if**:
- ❌ FD solver unstable or crashing
- ❌ No improvement over 5R1C
- ❌ Energy conservation violated
- ❌ Performance unacceptable (>60s per case)

## Expected Outcomes

### Best Case: FD Highly Successful

- FD solver works excellently for Case 600
- Heating overprediction reduced from +54% to <15%
- Cooling underprediction reduced from -30% to <15%
- Free-floating temps much closer to reference
- Performance acceptable (<30s)
- Clear path to enabling FD for all 600-series

### Medium Case: FD Moderately Successful

- FD shows improvement over 5R1C
- Heating overprediction reduced 25-50%
- Cooling underprediction reduced 25-50%
- Free-floating temps improved but still low
- Performance slow but acceptable
- Needs tuning but fundamentally sound

### Worst Case: FD Not Ready

- FD implementation has stability issues
- Timestep too small for practical use
- Results similar to or worse than 5R1C
- Performance unacceptable
- Need to consider:
  - Fix FD implementation (2-3 weeks)
  - Use CTF for low-mass (may work better)
  - Accept 5R1C limitations

## Technical Deep Dive: FD Solver Considerations

### Nodal Resolution

**More Nodes**:
- ✅ Better accuracy
- ✅ Better capture thermal gradients
- ❌ More computation
- ❌ Smaller timestep needed (CFL)

**Fewer Nodes**:
- ✅ Faster computation
- ✅ Larger timestep possible
- ❌ Less accurate
- ❌ May miss thermal gradients

**Recommendation**: Start with 5 nodes, adjust based on results

### Timestep Selection

**CFL Condition**:
```
dt ≤ dx² / (2α)
```
Where:
- `dx` = nodal spacing
- `α` = thermal diffusivity

**For Typical Wall**:
- `dx` = 0.06 m (5 nodes through 0.3 m wall)
- `α` ≈ 1e-6 m²/s (concrete)
- `dt` ≤ (0.06)² / (2 × 1e-6) = 1800 seconds (30 minutes)

**For Low-Mass Wall**:
- `α` may be higher (less thermal mass)
- `dt` may need to be 5-15 minutes
- Start with 5 minutes for safety

### Implicit vs Explicit FD

**Implicit** (Unconditionally stable):
- ✅ Larger timestep possible
- ✅ No CFL constraint
- ❌ Requires matrix solve each timestep
- ❌ More complex implementation

**Explicit** (Conditionally stable):
- ✅ Simple implementation
- ✅ Fast per timestep
- ❌ CFL constraint limits timestep
- ❌ May need very small dt

**Recommendation**: Use implicit FD (already implemented as `ImplicitFDSolver`)

## Files to Examine

1. **`src/physics/fd_solver.rs`**:
   - FD solver implementation
   - Nodal network structure
   - Time-stepping method
   - Stability criteria

2. **`src/sim/engine.rs`**:
   - Lines 893-1700: `ThermalModel::from_spec()` - where FD will be enabled
   - Lines 2974-3200: `solve_timesteps()` - where FD will be integrated
   - Lines 507-513: FD-related struct fields

3. **`src/validation/ashrae_140_cases.rs`**:
   - Case 600 specifications
   - Low-mass construction details
   - Wall assembly definitions

## Commands to Run

```bash
# Audit FD implementation
code src/physics/fd_solver.rs

# Check current FD usage
grep -r "fd_enabled" src/
grep -r "FDSolver" src/
grep -r "ImplicitFDSolver" src/

# Build with changes
cargo build --release

# Run Case 600 validation
cargo run --release --bin fluxion validate --case 600

# Run free-floating case
cargo run --release --bin fluxion validate --case 600FF

# Check performance
time cargo run --release --bin fluxion validate --case 600

# Compare with 5R1C baseline
# (Save 5R1C results first, then compare with FD)
```

## Research Questions

1. **Nodal Resolution**: How many nodes are needed for accurate low-mass simulation?
2. **Timestep Stability**: What is the minimum stable timestep for FD?
3. **Energy Conservation**: Does FD maintain energy conservation with small timesteps?
4. **Integration**: How does FD integrate with existing HVAC control logic?
5. **Performance**: Can FD achieve acceptable performance (<30s per case)?

## Dependencies

- **Session 49**: CTF working for 900-series (proves advanced models work)
- **FD Infrastructure**: Already implemented (not enabled)
- **Case 600**: Low-mass baseline case (relatively simple)

## Next Session

**Session 51**: FD Integration for All 600-Series Cases
- Prerequisite: Session 50 successful (FD working for Case 600)
- Scope: Enable FD for Cases 600-650
- Goal: Achieve ≥50% pass rate for 600-series

## Alternatives if FD Fails

**Option A**: Use CTF for Low-Mass
- CTF may work better than FD for low-mass
- Test CTF on 600-series (instead of FD)
- Simpler than FD (no CFL constraint)

**Option B**: Hybrid Approach
- Use FD for cases with rapid transients
- Use CTF for cases with slow transients
- Auto-select based on thermal characteristics

**Option C**: Accept 5R1C with Corrections
- Continue using 5R1C with empirical corrections
- May never achieve good results for low-mass
- Not recommended for 90% target

## References

- **`SESSION_49_SUMMARY.md`**: CTF integration results
- **`docs/ASHRAE140_ROADMAP.md`**: Phase 2 (FD for Low-Mass)
- **`docs/ASHRAE140_QUICKSTART.md`**: Quick start guide
- **`SESSION_45_SUMMARY.md`**: 600-series investigation findings
- **ASHRAE 140 Standard**: Case 600 specifications
- **ISO 13790**: FD model guidelines
- **Finite Difference Methods**: Numerical heat transfer reference

---

**Session 50 Goal**: Audit FD solver implementation and enable it for Case 600 as proof-of-concept, demonstrating that FD can address the systematic heating overprediction and cooling underprediction in low-mass buildings.
