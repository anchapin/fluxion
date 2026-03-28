# ASHRAE 140 Complete Validation - Quick Start Task List

**Current Status**: 1.6% pass rate (1/64 metrics)
**Target**: 90%+ pass rate
**Approach**: Enable existing advanced thermal models (CTF, FD)

---

## IMMEDIATE ACTIONS (This Week)

### Task 1: Audit CTF Solver Infrastructure ⚡ URGENT

**Goal**: Verify CTF solver is ready for production use

**Steps**:
1. Read `src/physics/ctf_coefficients.rs` - understand coefficient calculation
2. Read `src/physics/ctf_solver.rs` - understand solver logic
3. Check if CTF coefficients match ASHRAE 140 specifications
4. Verify timestep handling (3600s vs sub-hourly)
5. Test CTF solver on Case 900 (simple high-mass case)

**Commands**:
```bash
# Read CTF implementation
code src/physics/ctf_coefficients.rs
code src/physics/ctf_solver.rs

# Create test script
cat > test_ctf.rs << 'EOF'
// Test CTF solver on Case 900
use fluxion::physics::ctf_solver::CTFSolver;
use fluxion::physics::ctf_coefficients::CTFCalculator;

fn main() {
    println!("Testing CTF solver infrastructure...");
    // TODO: Implement CTF test
}
EOF
```

**Time**: 2-4 hours
**Output**: CTF audit report, ready/not-ready decision

---

### Task 2: Audit FD Solver Infrastructure ⚡ URGENT

**Goal**: Verify FD solver is ready for production use

**Steps**:
1. Read `src/physics/fd_solver.rs` - understand FD implementation
2. Check nodal network structure
3. Verify boundary conditions
4. Test CFL condition for timestep stability
5. Test FD solver on Case 600 (simple low-mass case)

**Commands**:
```bash
# Read FD implementation
code src/physics/fd_solver.rs

# Check for stability issues
grep -r "CFL\|stability\|timestep" src/physics/fd_solver.rs
```

**Time**: 2-4 hours
**Output**: FD audit report, ready/not-ready decision

---

### Task 3: Test Solver Manager ⚡ URGENT

**Goal**: Verify multi-method solver manager works

**Steps**:
1. Read `src/physics/solver_manager.rs`
2. Understand auto-selection logic
3. Test solver switching
4. Verify energy conservation when switching

**Commands**:
```bash
# Read solver manager
code src/physics/solver_manager.rs

# Check current usage
grep -r "solver_manager\|ctf_enabled\|fd_enabled" src/sim/engine.rs
```

**Time**: 2-3 hours
**Output**: Solver manager audit report

---

## PHASE 1 TASKS (Week 1-2: Enable CTF for High-Mass)

### Task 1.1: Enable CTF for Case 900

**File**: `src/sim/engine.rs`
**Location**: Lines 893-1700 (ThermalModel::from_spec)

**Change**:
```rust
// Around line 1700, after thermal mass correction
// Enable CTF for high-mass buildings
if matches!(spec.construction_type, ConstructionType::HighMass) {
    model.ctf_enabled = true;
    model.ctf_timestep = 3600.0; // Start with 1-hour timestep

    // Pre-compute CTF coefficients
    let ctf_calc = CTFCalculator::new();
    model.ctf_coefficients = Some(ctf_calc.compute_coefficients(&spec.construction));

    // Initialize CTF solvers
    model.ctf_solvers = spec.geometry.iter()
        .map(|g| CTFSolver::new(model.ctf_coefficients.as_ref().unwrap()))
        .collect();
}
```

**Test**:
```bash
cargo run --release --bin fluxion validate --case 900
```

**Expected**: Improved results for Case 900

---

### Task 1.2: Integrate CTF into Solve Loop

**File**: `src/sim/engine.rs`
**Location**: Lines 2974-3200 (solve_timesteps method)

**Change**: Add CTF path in solve loop

```rust
// Around line 3200, in solve loop
if self.ctf_enabled {
    // Use CTF solver for heat conduction
    for (zone_idx, ctf_solver) in self.ctf_solvers.iter_mut().enumerate() {
        let surface_temps = ctf_solver.solve_step(
            &self.outdoor_temp,
            &self.zone_temperatures,
            self.ctf_timestep,
        );
        // Update surface temperatures
        // ...
    }
} else {
    // Use existing 5R1C logic
    // ... existing code ...
}
```

**Test**: Run validation on all 900-series cases

---

### Task 1.3: Validate CTF Results

**Command**:
```bash
# Run all 900-series cases
for case in 900 910 920 930 940 950; do
    echo "Testing Case $case with CTF..."
    cargo run --release --bin fluxion validate --case $case
done
```

**Expected**:
- Annual heating: 70-90% pass rate (up from ~50%)
- Annual cooling: 70-90% pass rate
- Peak loads: Improved but not perfect

**Success**: ≥4/6 cases passing (67% pass rate)

---

## PHASE 2 TASKS (Week 3-4: Enable FD for Low-Mass)

### Task 2.1: Enable FD for Case 600

**File**: `src/sim/engine.rs`
**Location**: Same as Task 1.1

**Change**:
```rust
// Around line 1700, after thermal mass correction
// Enable FD for low-mass buildings
if matches!(spec.construction_type, ConstructionType::LowMass) {
    model.fd_enabled = true;
    model.fd_timestep = 300.0; // 5-minute timestep for stability

    // Initialize FD solvers with nodal network
    model.fd_solvers = spec.geometry.iter()
        .map(|g| {
            let num_nodes = 5; // 5 nodes through wall thickness
            ImplicitFDSolver::new(
                &spec.construction.wall,
                g.wall_area,
                num_nodes,
            )
        })
        .collect();
}
```

**Test**:
```bash
cargo run --release --bin fluxion validate --case 600
```

---

### Task 2.2: Integrate FD into Solve Loop

**File**: `src/sim/engine.rs`
**Location**: Same as Task 1.2

**Change**: Add FD path in solve loop

```rust
// Around line 3200, in solve loop
if self.fd_enabled {
    // Use FD solver for heat conduction
    for (zone_idx, fd_solver) in self.fd_solvers.iter_mut().enumerate() {
        let surface_temps = fd_solver.solve_step(
            &self.outdoor_temp,
            &self.zone_temperatures,
            self.fd_timestep,
        );
        // Update surface temperatures
        // ...
    }
}
```

**Test**: Run validation on all 600-series cases

---

### Task 2.3: Validate FD Results

**Command**:
```bash
# Run all 600-series cases
for case in 600 610 620 630 640 650; do
    echo "Testing Case $case with FD..."
    cargo run --release --bin fluxion validate --case $case
done
```

**Expected**:
- Heating overprediction: Reduced from +30-87% to ±20%
- Cooling underprediction: Reduced from -53% to ±20%
- At least 3/6 cases passing (50% pass rate)

---

## PHASE 3 TASKS (Week 5: Sub-Hourly Timesteps)

### Task 3.1: Implement Sub-Hourly Timestep

**File**: `src/sim/engine.rs`
**Location**: Lines 2974-2987 (solve_timesteps method)

**Change**: Add support for 900s (15-minute) timestep

```rust
// Add parameter to solve_timesteps
pub fn solve_timesteps_with_dt(
    &mut self,
    steps: usize,
    surrogates: &SurrogateManager,
    use_ai: bool,
    lighting: Option<&LightingSchedule>,
    equipment: Option<&[Box<dyn Equipment>]>,
    occupancy: Option<&OccupancyProfile>,
    dt_seconds: f64, // Timestep duration in seconds
) -> f64 {
    // ... implementation ...
}

// For validation, use sub-hourly timestep
let dt = 900.0; // 15 minutes
model.solve_timesteps_with_dt(8760 * 4, &surrogates, false, None, None, None, dt);
// Note: 4x timesteps for 15-minute resolution
```

---

### Task 3.2: Update Peak Tracking

**File**: `src/sim/engine.rs`
**Location**: Lines 3709-3715 (peak tracking)

**Change**: Ensure peak tracking works with sub-hourly timesteps

```rust
// Peak tracking should already work with sub-hourly timesteps
// Just verify it's capturing sub-hourly peaks
```

**Test**:
```bash
# Compare hourly vs sub-hourly peaks
./target/release/diagnose_peak_loads 900
```

---

### Task 3.3: Validate Peak Loads

**Command**:
```bash
# Run validation with sub-hourly timesteps
RUST_LOG=info cargo run --release --bin fluxion validate --all
```

**Expected**:
- Peak heating: Within 10-15% of reference
- Peak cooling: Within 10-15% of reference
- Annual energies: Unchanged (±1%)

---

## PHASE 4 TASKS (Week 6-7: Multi-Method Solver)

### Task 4.1: Implement Auto-Selection Logic

**File**: `src/sim/engine.rs` or `src/physics/solver_manager.rs`

**Change**:
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
        // Medium-mass: Use 5R1C
        SolverType::FiveResistanceOneCapacitance
    }
}
```

---

### Task 4.2: Test Solver Switching

**Command**:
```bash
# Test all cases with auto-selected solvers
cargo run --release --bin fluxion validate --all
```

**Expected**:
- Each case uses appropriate solver
- No energy conservation issues
- Overall pass rate ≥85%

---

## PHASE 5 TASKS (Week 8: Free-Floating)

### Task 5.1: Fix Free-Floating Temps

**Investigation needed**:
- Why are free-floating temps 20-30°C below reference?
- Is it solar gain distribution?
- Is it thermal mass coupling?
- Is it solver limitation?

**Test**:
```bash
# Compare free-floating results with different solvers
cargo run --release --bin fluxion validate --case 600FF
cargo run --release --bin fluxion validate --case 900FF
```

---

## PHASE 6 TASKS (Week 9: Special Cases)

### Task 6.1: Fix Case 960 (Sunspace)

**Investigation needed**:
- Inter-zone heat transfer
- Sunspace temperature modeling
- Solar gain distribution

**Test**:
```bash
cargo run --release --bin fluxion validate --case 960
```

---

### Task 6.2: Fix Case 195 (Steady-State)

**Investigation needed**:
- Steady-state conduction
- Zero windows condition
- Envelope heat transfer

**Test**:
```bash
cargo run --release --bin fluxion validate --case 195
```

---

## PHASE 7 TASKS (Week 10-11: Final Validation)

### Task 7.1: Comprehensive Validation

**Command**:
```bash
# Run full validation
cargo run --release --bin fluxion validate --all > validation_results.txt

# Generate report
cat validation_results.txt | grep -E "PASS|FAIL"
```

**Expected**:
- Overall pass rate ≥90%
- All metrics documented

---

### Task 7.2: Performance Profiling

**Command**:
```bash
# Profile each case
time cargo run --release --bin fluxion validate --case 900
time cargo run --release --bin fluxion validate --case 600
```

**Expected**:
- Performance <5s per case

---

### Task 7.3: Documentation

**Deliverables**:
1. Technical notes on CTF implementation
2. Technical notes on FD implementation
3. Solver selection guide
4. Updated ASHRAE 140 validation report

---

## SUCCESS CHECKLIST

- [ ] Overall pass rate ≥90% (58/64 metrics)
- [ ] 600-series ≥67% pass rate (16/24)
- [ ] 900-series ≥90% pass rate (22/24)
- [ ] Free-floating ≥50% pass rate (4/8)
- [ ] Special cases ≥50% pass rate (4/8)
- [ ] Performance <5s per case
- [ ] Documentation complete
- [ ] All tests passing
- [ ] No regressions

---

## QUICK START COMMANDS

```bash
# 1. Audit existing infrastructure
code src/physics/ctf_solver.rs
code src/physics/fd_solver.rs
code src/physics/solver_manager.rs

# 2. Build and test
cargo build --release
cargo test --release

# 3. Run validation baseline
cargo run --release --bin fluxion validate --all

# 4. Enable CTF for one case (prototype)
# Edit src/sim/engine.rs to enable CTF for Case 900
cargo run --release --bin fluxion validate --case 900

# 5. Compare results
# Did CTF improve Case 900 results?
# If yes: Proceed with full implementation
# If no: Debug or consider alternative approach
```

---

**Estimated Total Time**: 8-12 weeks (aggressive) to 12-18 weeks (conservative)
**Key Success Factor**: CTF/FD solvers are already implemented - just need to enable and integrate

**First Milestone**: End of Week 2 - CTF solver enabled for 900-series, ≥67% pass rate
