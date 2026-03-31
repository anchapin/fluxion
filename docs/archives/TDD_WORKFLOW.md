# Test-Driven Development Workflow for Physics Accuracy

This document describes the TDD workflow for improving fundamental physics accuracy in Fluxion using EnergyPlus and OpenStudio-MCP as reference resources.

## Overview

The TDD framework provides a systematic approach to validating and improving physics calculations:

1. **Test First**: Write tests that define expected physics behavior before implementation
2. **Reference Validation**: Compare against EnergyPlus/DOE reference data or analytical solutions
3. **Incremental Improvement**: Fix one physics component at a time
4. **Regression Prevention**: All tests must pass before merging changes

## Physics Domains

The framework covers 10 fundamental physics domains:

| Domain | Description | Tolerance |
|--------|-------------|-----------|
| `HeatConduction` | Heat transfer through walls, roofs, floors | 2% |
| `SolarRadiation` | Solar radiation absorption and transmission | 5% |
| `ThermalMass` | Thermal mass storage and release effects | 5% |
| `HVACLoads` | Heating and cooling load calculations | 5% |
| `AirExchange` | Infiltration and ventilation heat transfer | 5% |
| `InterZoneTransfer` | Heat transfer between thermal zones | 5% |
| `GroundCoupling` | Ground heat transfer and slab losses | 10% |
| `InternalGains` | Internal heat from occupants and equipment | 5% |
| `WindowHeatTransfer` | Window conduction and solar gain | 5% |
| `LongwaveRadiation` | Longwave radiation exchange between surfaces | 5% |

## Quick Start

### Running All Tests

```bash
# Run all physics domain tests
cargo run --bin tdd_validator

# Generate report to specific file
cargo run --bin tdd_validator -- --output reports/tdd_report.md
```

### Running Specific Domain Tests

```bash
# Test heat conduction only
cargo run --bin tdd_validator -- --domain heat-conduction

# Test with custom tolerance
cargo run --bin tdd_validator -- --tolerance 0.03
```

### Using Reference Data

```bash
# Load EnergyPlus reference data
cargo run --bin tdd_validator -- --reference data/energyplus_references.json
```

## Workflow

### Step 1: Create a Test Case

When implementing a new physics feature or fixing a bug, first create a test case:

```rust
use fluxion::testing::tdd_framework::{
    TestCaseResult, PhysicsDomain, TestStatus
};

// Create a test for steady-state heat conduction
let result = TestCaseResult::pass(
    "HC-001",                              // Test ID
    "Steady-state wall conduction",        // Test name
    PhysicsDomain::HeatConduction,         // Domain
    computed_value,                        // Fluxion's result
    reference_value,                       // Expected value
    "W",                                   // Units
);
```

### Step 2: Define Reference Values

Reference values can come from:

1. **Analytical Solutions**: Exact mathematical solutions (e.g., Q = U×A×ΔT)
2. **EnergyPlus Simulations**: Run EnergyPlus for the same building configuration
3. **ASHRAE Reference Data**: Published reference values (e.g., ASHRAE 140)
4. **ISO Standards**: Values from ISO 13790 or related standards

### Step 3: Run Tests and Verify

```bash
# Run tests and see results
cargo run --bin tdd_validator -- --domain heat-conduction --verbose
```

### Step 4: Fix Failing Tests

If a test fails:

1. **Investigate Root Cause**: Why is Fluxion's result different from reference?
2. **Fix the Physics**: Update the implementation to match physics
3. **Re-run Tests**: Verify the fix doesn't break other tests
4. **Document the Change**: Add comments explaining the fix

### Step 5: Add to Regression Suite

Once a test passes, it becomes part of the regression suite. All future changes must keep these tests passing.

## Creating Reference Data Files

Reference data can be stored in JSON format:

```json
{
  "cases": {
    "HC-001": {
      "source": "Analytical: Q = U × A × ΔT",
      "value": 100.0,
      "uncertainty": 0.0,
      "units": "W",
      "metadata": {
        "u_value": "0.5 W/m²K",
        "area": "10 m²",
        "delta_t": "20 K"
      }
    },
    "SR-001": {
      "source": "EnergyPlus 22.2",
      "value": 1361.0,
      "uncertainty": 0.01,
      "units": "W/m²",
      "metadata": {
        "description": "Solar constant at top of atmosphere"
      }
    }
  }
}
```

## Integration with CI/CD

Add the TDD validator to your CI pipeline:

```yaml
# .github/workflows/physics-validation.yml
name: Physics Validation

on: [push, pull_request]

jobs:
  tdd-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run TDD Physics Tests
        run: |
          cargo run --bin tdd_validator -- --output tdd_report.md
      - name: Upload Report
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: tdd-report
          path: tdd_report.md
```

## Best Practices

### 1. Test Isolation

Each test should be independent and not rely on the state of other tests.

### 2. Meaningful Test IDs

Use a consistent naming convention:
- `HC-001`: Heat Conduction test 1
- `SR-002`: Solar Radiation test 2
- `TM-003`: Thermal Mass test 3

### 3. Document Assumptions

Always document the assumptions behind reference values:
- Weather conditions
- Material properties
- Boundary conditions

### 4. Appropriate Tolerances

Set tolerances based on:
- Uncertainty in reference data
- Numerical precision limits
- Physical variability

### 5. Fail Fast in Development

Use `--fail-fast` during development to quickly identify issues:

```bash
cargo run --bin tdd_validator -- --fail-fast --domain heat-conduction
```

## EnergyPlus Integration

### Using OpenStudio-MCP

The framework can integrate with OpenStudio-MCP for reference simulations:

```python
# tools/energyplus_reference.py
import subprocess
import json

def run_energyplus_simulation(building_config):
    """Run EnergyPlus simulation and extract results."""
    # Create IDF file from config
    # Run EnergyPlus
    # Parse output and return results
    pass

def generate_reference_data():
    """Generate reference data for TDD framework."""
    reference_db = {"cases": {}}

    # Heat conduction reference
    config = {
        "wall_u_value": 0.5,
        "area": 10.0,
        "indoor_temp": 20.0,
        "outdoor_temp": 0.0,
    }
    result = run_energyplus_simulation(config)
    reference_db["cases"]["HC-001"] = {
        "source": "EnergyPlus 22.2",
        "value": result["heat_loss"],
        "uncertainty": 0.02,
        "units": "W",
    }

    return reference_db
```

### Extracting EnergyPlus Results

Use the existing tools in `tools/extract_energyplus_components.py`:

```bash
# Extract reference data from EnergyPlus simulations
python tools/extract_energyplus_components.py --output data/energyplus_references.json
```

## Troubleshooting

### Test Fails but Physics is Correct

1. Check if the reference value is correct
2. Verify units are consistent
3. Check for sign conventions (heat gain vs loss)
4. Consider if tolerance is too strict

### Test Passes but Physics is Wrong

1. Check if the test is actually validating the right thing
2. Verify the reference value is accurate
3. Consider if tolerance is too lenient
4. Add more specific test cases

### Performance Issues

If tests are slow:
1. Use analytical solutions instead of simulations where possible
2. Cache reference data
3. Run only affected domains during development

## Contributing

When contributing physics improvements:

1. **Create a branch** for your changes
2. **Write tests first** that define expected behavior
3. **Implement the fix** to make tests pass
4. **Run full test suite** to check for regressions
5. **Update documentation** with your changes
6. **Submit a PR** with test results

## Resources

- [ISO 13790](https://www.iso.org/standard/41974.html): Calculation of energy use for heating and cooling
- [ASHRAE Standard 140](https://www.ashrae.org/standards-research-technology/standards-amp-guidelines/standards-addenda/140-2017-addenda): Building Energy Simulation Test Procedures
- [EnergyPlus Documentation](https://energyplus.net/documentation)
- [OpenStudio SDK](https://openstudio.net/)

## Appendix: Test Case Templates

### Heat Conduction Test Template

```rust
#[test]
fn test_steady_state_conduction() {
    // Given: Wall with known properties
    let u_value = 0.5; // W/m²K
    let area = 10.0; // m²
    let t_in = 20.0; // °C
    let t_out = 0.0; // °C

    // When: Calculate heat transfer
    let q = u_value * area * (t_in - t_out);

    // Then: Match analytical solution
    let expected = 100.0; // W
    assert!((q - expected).abs() < 0.01);
}
```

### Solar Radiation Test Template

```rust
#[test]
fn test_solar_altitude_angle() {
    // Given: Location and time
    let latitude = 40.0_f64.to_radians();
    let declination = 23.45_f64.to_radians();
    let hour_angle = 0.0; // Solar noon

    // When: Calculate altitude
    let sin_alpha = latitude.sin() * declination.sin()
                  + latitude.cos() * declination.cos() * hour_angle.cos();
    let alpha = sin_alpha.asin().to_degrees();

    // Then: Match expected value
    let expected = 73.45; // degrees at summer solstice
    assert!((alpha - expected).abs() < 0.1);
}
