# Session 73: Test-Driven Development with EnergyPlus Integration

## Executive Summary

This session establishes a comprehensive test-driven development framework that leverages EnergyPlus and OpenStudio-MCP resources to improve physics engine accuracy. We address critical temperature instability issues identified in Session 72 and create a robust validation pipeline.

## Session 72 Critical Issues (Inherited)

| Test | Case | Issue | Severity |
|------|------|-------|----------|
| `test_temperature_stability_case_900` | 900 | Temperature starts at 164.82°C | CRITICAL |
| `test_free_floating_stability_case_900ff` | 900FF | Temperature starts at 148.93°C | CRITICAL |
| `test_free_floating_stability_case_600ff` | 600FF | Setpoints not configured for free-floating | HIGH |

**Pass Rate:** 9/12 (75%) - 3 tests failing

## Root Cause Analysis

### Issue 1: Case 900 Temperature Instability (164.82°C at step 0)

**Symptom:** High-mass case (900) shows extreme temperature immediately at the first timestep.

**Investigation Findings:**
1. Initial temperatures are correctly set to 20°C in `ThermalModel::new()`
2. The issue occurs during the first `step_physics()` call
3. Case 900 uses 6R2C model with CTF solver enabled
4. Solar gain calculation may be producing extreme values

**Hypothesis:** The CTF solver or solar gain distribution is producing invalid results on the first timestep, possibly due to:
- Invalid initial CTF history values
- Extreme solar gain calculation for high-mass cases
- Incorrect sensitivity calculation leading to extreme HVAC response

### Issue 2: Free-Floating Case 600FF Setpoint Configuration

**Symptom:** Free-floating case doesn't have extreme setpoints (-999°C heating, 999°C cooling).

**Investigation Findings:**
- The `from_spec()` function may not be correctly identifying free-floating cases
- HVAC schedules may not be set correctly for FF cases

## Test-Driven Development Plan

### Phase 1: Fix Critical Temperature Instability (RED → GREEN)

#### Task 1.1: Add Diagnostic Tests for First Timestep

Create focused tests that isolate the first timestep behavior:

```rust
// New test file: tests/first_timestep_diagnostics.rs

#[test]
fn test_case_900_first_timestep_temperatures() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Record initial state
    let initial_zone_temp = model.temperatures.as_ref()[0];
    let initial_mass_temp = model.mass_temperatures.as_ref()[0];

    assert_eq!(initial_zone_temp, 20.0, "Initial zone temp should be 20°C");
    assert_eq!(initial_mass_temp, 20.0, "Initial mass temp should be 20°C");

    // Run first timestep
    let hvac_kwh = model.step_physics(0, 10.0, 3600.0);

    // Check temperatures after first step
    let zone_temp = model.temperatures.as_ref()[0];
    let mass_temp = model.mass_temperatures.as_ref()[0];

    // Temperatures should remain reasonable (< 50°C change in one hour)
    assert!(
        zone_temp > -30.0 && zone_temp < 70.0,
        "Zone temp after step 0: {:.2}°C (unreasonable)", zone_temp
    );
    assert!(
        mass_temp > -30.0 && mass_temp < 70.0,
        "Mass temp after step 0: {:.2}°C (unreasonable)", mass_temp
    );
}

#[test]
fn test_case_900_solar_gains_first_timestep() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Check solar gains before step
    assert_eq!(model.solar_gains.as_ref()[0], 0.0, "Solar gains should be 0 before calc");

    // Trigger load calculation
    model.calc_analytical_loads(0, true);

    // Check solar gains after calculation
    let solar_gain = model.solar_gains.as_ref()[0];
    assert!(
        solar_gain.is_finite() && solar_gain >= 0.0 && solar_gain < 1000.0,
        "Solar gain at timestep 0: {:.2} W/m² (unreasonable)", solar_gain
    );
}
```

#### Task 1.2: Fix CTF Solver Initial Conditions

The CTF solver needs proper initial conditions. Add validation:

```rust
// In src/physics/ctf_solver.rs

impl CTFSolver {
    pub fn new(coefficients: CTFCoefficients, config: CTFSolverConfig) -> Self {
        // Validate coefficients
        assert!(
            coefficients.validate(),
            "CTF coefficients are invalid"
        );

        // Initialize history with zeros (stable initial condition)
        Self {
            coefficients,
            config,
            history: vec![0.0; config.history_size],
            initialized: true,
        }
    }
}
```

#### Task 1.3: Fix Solar Gain Calculation for High-Mass Cases

The solar gain calculation may be producing extreme values. Add bounds checking:

```rust
// In src/sim/engine.rs, step_physics_5r1c/step_physics_6r2c

// Clamp solar gains to reasonable range
let solar_gain_clamped = solar_gain_watts.clamp(0.0, 5000.0); // Max 5000 W for small building
```

### Phase 2: EnergyPlus Integration for Validation

#### Task 2.1: Create EnergyPlus Reference Data Generator

Use OpenStudio-MCP to generate reference data for all ASHRAE 140 cases:

```python
# tools/generate_energyplus_reference.py

#!/usr/bin/env python3
"""
Generate EnergyPlus reference data for ASHRAE 140 validation cases.
Uses OpenStudio-MCP to create and simulate building models.
"""

import asyncio
import json
from pathlib import Path
from mcp_client import create_mcp_session, call_tool

ASHRAE_140_CASES = [
    "600", "610", "620", "630", "640", "650",
    "900", "910", "920", "930", "940", "950", "960"
]

async def generate_reference_for_case(case_id: str):
    """Generate EnergyPlus reference data for a specific case."""
    async with create_mcp_session() as session:
        # Create building model using MCP tools
        # Run simulation
        # Extract hourly results
        pass

async def main():
    """Generate reference data for all cases."""
    for case_id in ASHRAE_140_CASES:
        print(f"Generating reference data for Case {case_id}...")
        await generate_reference_for_case(case_id)
```

#### Task 2.2: Create Comparison Test Suite

```rust
// tests/energyplus_comparison_tests.rs

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

/// Compare Fluxion results with EnergyPlus reference data
#[test]
fn test_case_900_vs_energyplus_annual_energy() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Load EnergyPlus reference data
    let ep_reference = load_energyplus_reference("900");

    // Run Fluxion simulation
    let fluxion_results = run_annual_simulation(&mut model);

    // Compare annual energy
    let heating_error = (fluxion_results.heating - ep_reference.heating).abs() / ep_reference.heating;
    let cooling_error = (fluxion_results.cooling - ep_reference.cooling).abs() / ep_reference.cooling;

    // Allow 10% error for annual energy
    assert!(
        heating_error < 0.10,
        "Heating error: {:.1}% (Fluxion: {:.2} MWh, EP: {:.2} MWh)",
        heating_error * 100.0, fluxion_results.heating, ep_reference.heating
    );
    assert!(
        cooling_error < 0.10,
        "Cooling error: {:.1}% (Fluxion: {:.2} MWh, EP: {:.2} MWh)",
        cooling_error * 100.0, fluxion_results.cooling, ep_reference.cooling
    );
}

/// Compare hourly zone temperatures
#[test]
fn test_case_900_vs_energyplus_hourly_temperatures() {
    // Compare hourly zone air temperatures
    // Allow ±2°C error for 95% of hours
}

/// Compare peak loads
#[test]
fn test_case_900_vs_energyplus_peak_loads() {
    // Compare peak heating and cooling loads
    // Allow 10% error
}
```

### Phase 3: Continuous Validation Pipeline

#### Task 3.1: Create Automated Validation Script

```python
# tools/continuous_validation.py

#!/usr/bin/env python3
"""
Continuous validation pipeline for Fluxion physics engine.
Runs tests, compares with EnergyPlus, and generates reports.
"""

import subprocess
import json
from pathlib import Path

def run_rust_tests():
    """Run Rust test suite and capture results."""
    result = subprocess.run(
        ["cargo", "test", "--test", "step_physics_unit_tests", "--", "--nocapture"],
        capture_output=True,
        text=True
    )
    return parse_test_results(result.stdout)

def generate_validation_report():
    """Generate comprehensive validation report."""
    report = {
        "test_results": run_rust_tests(),
        "energyplus_comparison": run_energyplus_comparison(),
        "regression_check": check_for_regressions(),
    }
    return report

if __name__ == "__main__":
    report = generate_validation_report()
    print(json.dumps(report, indent=2))
```

## Implementation Status

### Completed
- [x] Session 72 test infrastructure created
- [x] Critical issues identified
- [x] TDD plan established

### In Progress
- [ ] Fix Case 900 temperature instability
- [ ] Fix Case 900FF temperature instability
- [ ] Fix Case 600FF setpoint configuration

### Pending
- [ ] EnergyPlus reference data generation
- [ ] Comparison test suite implementation
- [ ] Continuous validation pipeline

## Success Criteria

1. **All 12 unit tests passing** (currently 9/12 = 75%)
2. **Case 900 temperature remains within -40°C to 60°C range**
3. **Free-floating temperatures remain within realistic bounds**
4. **EnergyPlus comparison within 10% for annual energy**
5. **Overall ASHRAE 140 validation pass rate ≥90%**

## Files to Modify

1. `src/sim/engine.rs` - Fix temperature instability in step_physics
2. `src/physics/ctf_solver.rs` - Add coefficient validation
3. `tests/step_physics_unit_tests.rs` - Update failing tests
4. `tests/first_timestep_diagnostics.rs` (NEW) - Add diagnostic tests
5. `tools/generate_energyplus_reference.py` (NEW) - EnergyPlus integration
6. `tests/energyplus_comparison_tests.rs` (NEW) - Comparison tests

## Next Steps

1. **Immediate:** Run diagnostic tests to identify exact cause of temperature instability
2. **Short-term:** Fix critical issues to achieve 100% unit test pass rate
3. **Medium-term:** Integrate EnergyPlus for reference validation
4. **Long-term:** Establish continuous validation pipeline
