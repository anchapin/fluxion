# Physics Test Suite

## Organization

The physics test suite is organized to ensure comprehensive coverage of Fluxion's thermal modeling physics.

### Test Files

| File | Purpose | Status |
|------|---------|--------|
| `ashrae_140_coverage.rs` | ASHRAE 140 validation with coverage tracking | Planned |
| `convection_validation.rs` | Natural and forced convection tests | Embedded in module |
| `radiation_validation.rs` | Longwave and shortwave radiation tests | Embedded in module |
| `ctf_coefficient_validation.rs` | CTF coefficients vs EnergyPlus | Embedded in module |
| `newton_solver_validation.rs` | Convergence and stability tests | Embedded in module |
| `per_surface_validation.rs` | Per-surface heat balance tests | Embedded in module |
| `combined_heat_balance.rs` | Combined convection+radiation tests | Embedded in module |

### Embedded Module Tests

Most physics tests are embedded directly in their respective modules under `src/physics/`:

- `cta.rs` - Continuous Tensor Abstraction operations and numerical properties (10 tests)
- `ctf_coefficients.rs` - CTF coefficient calculation (37 tests)
- `ctf_solver.rs` - CTF solver implementation (20 tests)
- `fd_solver.rs` - Finite difference solver (13 tests)
- `five_r1c_solver.rs` - 5R1C thermal network solver (2 tests)
- `newton_solver.rs` - Newton-Raphson solver (0 tests)
- `convection.rs` - Convection coefficient calculations (0 tests)
- `radiation.rs` - Radiation heat transfer (0 tests)
- `combined_heat_balance.rs` - Combined heat balance (0 tests)
- `method_selector.rs` - Solver method selection (11 tests)
- `solver_manager.rs` - Solver state management (12 tests)
- `per_surface_model.rs` - Per-surface thermal model (0 tests)
- `per_surface_ctf.rs` - Per-surface CTF formulation (0 tests)
- `fd_surface_balance.rs` - Finite difference surface balance (37 tests)
- `fd_discretization.rs` - Finite difference discretization (10 tests)
- `ctf_solver_wrapper.rs` - CTF solver wrapper (6 tests)
- `fd_solver_wrapper.rs` - FD solver wrapper (3 tests)
- `geometry_tensor.rs` - Geometric tensor operations (12 tests)
- `nd_array.rs` - N-dimensional array operations (5 tests)
- `continuous.rs` - Continuous field trait (3 tests)

**Total embedded physics tests: ~245**

## Running Tests

### All Physics Tests
```bash
# Run all physics module tests
cargo test --lib physics::

# Run with output for debugging
cargo test --lib physics:: -- --nocapture

# Run tests with coverage report
cargo llvm-cov test --lib --features default -- --test-threads=1
```

### Specific Module Tests
```bash
# CTA module
cargo test --lib physics::cta::

# CTF coefficients
cargo test --lib physics::ctf_coefficients::

# Convection
cargo test --lib physics::convection::

# Radiation
cargo test --lib physics::radiation::
```

### Integration Tests
```bash
# ASHRAE 140 validation suite
cargo test ashrae_140

# Critical path tests
cargo test test_critical_paths

# Coverage enhancement tests
cargo test test_coverage_enhancement
```

### With Performance Metrics
```bash
# Run tests with timing information
cargo test --lib physics:: -- --nocapture --test-threads=1

# Run benchmarks
cargo bench --bench cta_bench
```

## EnergyPlus Reference Data

### Reference Data Structure

EnergyPlus reference data is managed through the EP oracle framework:

```
refdata/
├── ep/                       # EP reference results (generated)
│   ├── Case_600_results.json
│   ├── Case_900_results.json
│   └── ...
├── epw/                      # Weather files (TMY format)
│   ├── Denver.epw
│   └── ...
└── ep_test_cases.toml         # Test case catalog
```

### Generating Reference Data

```bash
# Set EnergyPlus installation directory
export ENERGYPLUS_INSTALL_DIR=/path/to/EnergyPlus

# Generate reference for a specific case
python tools/ep_oracle.py generate --case 600

# Generate all test cases
python tools/ep_oracle.py generate --all-cases
```

### EP Oracle Tool Commands

```bash
# Generate EP reference for a specific case
python tools/ep_oracle.py generate --case 600

# Generate all test cases
python tools/ep_oracle.py generate --all-cases

# Compare Fluxion and EP results
python tools/ep_oracle.py compare \
  --fluxion fluxion_output.json \
  --ep refdata/ep/Case_600_results.json

# Validate Fluxion against EP
python tools/ep_oracle.py validate \
  --test-case 600 \
  --fluxion-output fluxion_output.json
```

## Validation Framework

### EP Oracle Validation

The EP oracle framework provides validation against EnergyPlus reference data:

```rust
use fluxion::validation::ep_oracle::{EPOracle, FluxionResults};

let oracle = EPOracle::new()?;
let fluxion_results = FluxionResults {
    temperatures: vec![/* ... */],
    fluxes: vec![/* ... */],
    energy: 12345.0,
};
let report = oracle.validate(&fluxion_results);

if report.passed {
    println!("Validation passed!");
} else {
    println!("Validation failed:");
    if let Some(temp) = report.temperature {
        println!("  Temperature RMSE: {:.2}", temp.rmse);
    }
}
```

### Validation Criteria

Default validation thresholds:
- **Maximum Absolute Error**: 1.0K for temperatures, 1W for fluxes
- **Maximum Relative Error**: 5% for all metrics
- **Minimum Correlation**: R² ≥ 0.95
- **Maximum RMSE**: 0.5K for temperatures

Custom thresholds can be configured via `ValidationCriteria` struct.

## ASHRAE 140 Validation

### Running ASHRAE 140 Tests

```bash
# Run full ASHRAE 140 validation suite
cargo test --lib ashrae_140

# Run with detailed diagnostic output
RUST_LOG=debug cargo test --lib ashrae_140 -- --nocapture

# Generate validation report
cargo test generate_validation_report -- --nocapture
```

### Coverage Tracking

Phase 4 adds coverage tracking to ASHRAE 140 tests:

```rust
#[test]
fn test_case_600_full_coverage() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Track which code paths are exercised
    let mut coverage = CoverageTracker::new();

    // Run full year simulation
    for hour in 0..8760 {
        coverage.mark_path("solve_timestep");
        model.solve_timesteps(hour + 1, &surrogates, false, None, None, None);
    }

    // Verify critical paths were hit
    assert!(coverage.path_hit("conduction"), "Conduction path not hit");
    assert!(coverage.path_hit("convection"), "Convection path not hit");
    assert!(coverage.path_hit("radiation"), "Radiation path not hit");
    assert!(coverage.path_hit("hvac_control"), "HVAC control path not hit");
}
```

### Tracked Paths

- `conduction` - Heat conduction through building envelope
- `convection` - Convective heat transfer (interior/exterior)
- `radiation` - Longwave and shortwave radiation exchange
- `hvac_control` - HVAC heating/cooling demand calculation
- `interzone_transfer` - Heat transfer between zones
- `solar_gain` - Solar radiation gain calculations
- `thermal_mass` - Thermal mass storage effects
- `surface_balance` - Surface energy balance solver

## Test Case Catalog Format

### EP Test Cases (`refdata/ep_test_cases.toml`)

```toml
[[case]]
id = "600"
name = "Heavyweight - Summer"
category = "conduction"
floor_area = 48.0
walls_u = 0.358
roof_u = 0.226
floor_u = 0.398
window_u = 2.943
window_area = 6.0
setpoint_heating = 20.0
setpoint_cooling = 27.0
epw = "refdata/epw/Denver.epw"
ep_reference = "refdata/ep/Case600.sql"
```

### Categories

- `conduction` - Heat conduction tests (600, 900 series)
- `convection` - Natural and forced convection
- `radiation` - Longwave and shortwave radiation
- `ctf` - Conduction Transfer Function tests
- `newton` - Newton solver convergence tests
- `per_surface` - Per-surface model tests
- `combined` - Combined heat balance tests
- `fd` - Finite difference solver tests
- `energy_balance` - Annual and diurnal energy balance

## Adding New Tests

### Adding Module Tests

1. Add test module to the source file under `src/physics/`:
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn test_new_feature() {
           // Test implementation
       }
   }
   ```

2. Run tests to verify:
   ```bash
   cargo test --lib physics::module_name::
   ```

3. Add documentation for the test

### Adding EnergyPlus-Validated Tests

1. Add test case to `refdata/ep_test_cases.toml`

2. Generate EnergyPlus reference:
   ```bash
   python tools/ep_oracle.py generate --case YOUR_CASE_ID
   ```

3. Create test file under `tests/physics/`:
   ```rust
   #[test]
   fn test_your_case_energyplus() {
       use fluxion::validation::ep_oracle::{EPOracle, FluxionResults};

       let fluxion_results = run_fluxion_simulation("YOUR_CASE_ID");
       let oracle = EPOracle::new()?;
       let report = oracle.validate(&fluxion_results);

       assert!(report.passed, "Validation failed: {:?}", report);
   }
   ```

4. Run and verify:
   ```bash
   cargo test test_your_case_energyplus
   ```

### Adding Regression Tests

Use the regression test generator:

```bash
# After running Fluxion and EnergyPlus simulations
python tools/generate_regression_tests.py \
  --case-id 600 \
  --fluxion-output fluxion_600.json \
  --ep-output ep_600.json \
  --test-name case_600_regression
```

This generates Rust test code with proper assertions against EP reference data.

## Troubleshooting

### Common Issues

#### "EnergyPlus not found"
```
Error: ENERGYPLUS_INSTALL_DIR not set or EnergyPlus not found
```

**Solution:**
```bash
export ENERGYPLUS_INSTALL_DIR=/path/to/EnergyPlus
python tools/ep_oracle.py generate --case 600
```

#### "Test timeout in solver"
```bash
Error: Test timed out after 30 seconds
```

**Solution:**
- Check for convergence issues in Newton solver
- Verify time constant vs timestep ratio
- Use implicit methods for stiff systems

#### "Coverage below threshold"
```bash
Error: Physics coverage below 90%
```

**Solution:**
- Run `cargo llvm-cov report --html --output-dir coverage/`
- Open `coverage/index.html` to identify uncovered code
- Add tests for uncovered paths

#### "EPW file not found"
```bash
Error: Weather file not found: refdata/epw/Denver.epw
```

**Solution:**
- Download EPW files from [EnergyPlus Weather Data](https://energyplus.net/weather)
- Place in `refdata/epw/`
- Update `ep_test_cases.toml` with correct paths

### Debugging Test Failures

#### Enable Debug Output
```bash
RUST_LOG=debug cargo test test_name -- --nocapture
```

#### Run Single Test
```bash
cargo test test_name -- --exact --nocapture
```

#### Check Backtrace
```bash
RUST_BACKTRACE=1 cargo test test_name -- --nocapture
```

### Performance Issues

#### Slow Test Execution
```bash
# Run tests in parallel (default)
cargo test --lib physics::

# Run single-threaded for debugging
cargo test --lib physics:: -- --test-threads=1
```

#### Memory Usage
```bash
# Run with memory profiler
cargo test --lib physics:: -- --release
```

## Coverage Reports

### Generate Coverage Report

```bash
# Generate HTML coverage report
cargo llvm-cov report --html --output-dir coverage/html

# Generate LCOV format for Codecov
cargo llvm-cov report --lcov --output-path coverage/lcov.info

# Generate JSON for custom analysis
cargo llvm-cov report --json --output-path coverage/coverage.json
```

### Coverage Targets

| Module | Target | Current | Gap |
|--------|--------|---------|-----|
| CTA | 90% | 89.92% | -0.08% |
| CTF Coefficients | 90% | 74.65% | -15.35% |
| CTF Solver | 90% | 67.36% | -22.64% |
| FD Solver | 90% | 69.61% | -20.39% |
| 5R1C Solver | 90% | 83.70% | -6.30% |
| Newton Solver | 90% | 85.22% | -4.78% |
| Convection | 90% | 44.93% | -45.07% |
| Radiation | 90% | 80.38% | -9.62% |
| Per-Surface Model | 90% | 0% | -90% |
| FD Surface Balance | 90% | 34.70% | -55.30% |

**Overall Goal: 90%+ coverage for all physics modules**

## Related Documentation

- [PHYSICS_TEST_COVERAGE_PLAN.md](../../docs/PHYSICS_TEST_COVERAGE_PLAN.md) - Detailed coverage plan
- [PHASE2_COMPLETION.md](../../docs/PHASE2_COMPLETION.md) - EP oracle setup
- [ASHRAE140_RESULTS.md](../../docs/ASHRAE140_RESULTS.md) - ASHRAE 140 validation results
- [ARCHITECTURE.md](../../docs/ARCHITECTURE.md) - Fluxion architecture
- [CONTRIBUTING.md](../../docs/CONTRIBUTING.md) - Development guidelines
