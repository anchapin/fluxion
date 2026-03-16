# Adding New ASHRAE 140 Reference Cases

This guide provides step-by-step instructions for adding new ASHRAE 140 reference cases to Fluxion.

## Overview

ASHRAE 140 is a standard test suite for building energy simulation programs. Adding new reference cases allows Fluxion to validate its accuracy against a wider range of building configurations and compare results against established simulation programs.

## Prerequisites

- Fluxion installed and built (`cargo build && maturin develop`)
- ASHRAE 140 specification (available from ASHRAE)
- Reference data from at least two established simulation programs (EnergyPlus, ESP-r, TRNSYS, DOE2)
- Familiarity with Rust and thermal network concepts

## Step 1: Create Case Specification File

Create a new case file in `src/validation/ashrae_140/case_XXXX.rs` (where XXXX is the case ID).

### Example Case Specification

```rust
// src/validation/ashrae_140/case_1000.rs
use crate::sim::engine::ThermalModel;
use crate::physics::cta::VectorField;
use crate::validation::ashrae_140_cases::CaseSpec;

pub fn create_case_1000() -> CaseSpec {
    // Zone configuration
    let mut spec = CaseSpec::default();
    spec.num_zones = 1;

    // Building geometry
    let mut geometry = crate::validation::ashrae_140_cases::Geometry::default();
    geometry.floor_area = 20.0; // m²
    geometry.height = 3.0; // m
    geometry.wall_area = geometry.floor_area * geometry.height;
    spec.geometry = vec![geometry];

    // Window properties
    let mut window = crate::validation::ashrae_140_cases::Window::default();
    window.area = geometry.floor_area * 0.15; // 15% window-to-floor ratio
    window.u_value = Some(1.5); // W/m²K
    window.shgc = Some(0.6); // Solar heat gain coefficient
    spec.windows = vec![vec![window]];

    // Construction properties
    let mut construction = crate::validation::ashrae_140_cases::Construction::default();
    construction.wall_u_value = Some(0.5); // W/m²K
    construction.roof_u_value = Some(0.3); // W/m²K
    spec.construction = construction;

    // HVAC configuration
    let mut hvac = crate::validation::ashrae_140_cases::HVACSchedule::default();
    hvac.heating_setpoint = 20.0; // °C
    hvac.cooling_setpoint = 27.0; // °C
    hvac.heating_enabled = true;
    hvac.cooling_enabled = true;
    spec.hvac = vec![hvac];

    // Internal loads
    let mut internal_loads = crate::validation::ashrae_140_cases::InternalLoads::default();
    internal_loads.lighting = 5.0; // W/m²
    internal_loads.equipment = 10.0; // W/m²
    internal_loads.occupancy = 3.0; // W/m²
    spec.internal_loads = vec![Some(internal_loads)];

    // Infiltration
    spec.infiltration_ach = 0.5; // Air changes per hour

    // Thermal mass (5R1C parameter)
    spec.thermal_mass_c = 1_000_000.0; // J/K

    // 5R1C conductances
    spec.h_tr_em = Some(57.42); // Exterior to mass (W/K)
    spec.h_tr_ms = Some(1087.5); // Mass to surface (W/K)
    spec.h_tr_is = Some(0.0); // Surface to interior (W/K)
    spec.h_tr_w = Some(0.0); // Exterior to interior via windows (W/K)
    spec.h_ve = Some(0.0); // Ventilation (W/K)

    spec
}
```

### Key Parameters to Define

1. **Zone Configuration**
   - `num_zones`: Number of thermal zones

2. **Geometry**
   - `floor_area`: Floor area per zone (m²)
   - `height`: Zone height (m)
   - `wall_area`: Exterior wall area (m²)

3. **Window Properties**
   - `area`: Total window area (m²)
   - `u_value`: Window U-value (W/m²K)
   - `shgc`: Solar heat gain coefficient

4. **Construction Properties**
   - `wall_u_value`: Wall U-value (W/m²K)
   - `roof_u_value`: Roof U-value (W/m²K)

5. **HVAC Configuration**
   - `heating_setpoint`: Heating setpoint (°C)
   - `cooling_setpoint`: Cooling setpoint (°C)
   - `heating_enabled`: Whether heating is active
   - `cooling_enabled`: Whether cooling is active

6. **Internal Loads**
   - `lighting`: Lighting load (W/m²)
   - `equipment`: Equipment load (W/m²)
   - `occupancy`: Occupancy load (W/m²)

7. **Thermal Mass and 5R1C Conductances**
   - `thermal_mass_c`: Thermal capacitance (J/K)
   - `h_tr_em`: Exterior to mass conductance (W/K)
   - `h_tr_ms`: Mass to surface conductance (W/K)
   - `h_tr_is`: Surface to interior conductance (W/K)
   - `h_tr_w`: Exterior to interior via windows (W/K)
   - `h_ve`: Ventilation conductance (W/K)

## Step 2: Add Case to ASHRAE140Case Enum

Update `src/validation/ashrae_140_cases.rs` to include the new case:

```rust
#[derive(Debug, Clone, Copy)]
pub enum ASHRAE140Case {
    // ... existing cases ...
    Case1000, // New case
}
```

Implement the `spec()` method:

```rust
impl ASHRAE140Case {
    pub fn spec(&self) -> CaseSpec {
        match self {
            // ... existing cases ...
            ASHRAE140Case::Case1000 => crate::validation::ashrae_140::case_1000::create_case_1000(),
        }
    }

    pub fn number(&self) -> &'static str {
        match self {
            // ... existing cases ...
            ASHRAE140Case::Case1000 => "1000",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            // ... existing cases ...
            ASHRAE140Case::Case1000 => "Custom test case 1000",
        }
    }
}
```

## Step 3: Define Reference Data

Add reference data to `src/validation/benchmark.rs`:

```rust
pub fn get_all_benchmark_data() -> HashMap<String, BenchmarkData> {
    let mut data = HashMap::new();

    // ... existing cases ...

    // Add new case
    data.insert(
        "1000".to_string(),
        BenchmarkData {
            annual_heating_min: 1.5,  // MWh (minimum across reference programs)
            annual_heating_max: 2.0,  // MWh (maximum across reference programs)
            annual_cooling_min: 3.0,  // MWh
            annual_cooling_max: 4.0,  // MWh
            peak_heating_min: 2.5,     // kW
            peak_heating_max: 3.5,     // kW
            peak_cooling_min: 4.0,     // kW
            peak_cooling_max: 5.0,     // kW
            min_free_float_min: -10.0,  // °C
            min_free_float_max: -8.0,   // °C
            max_free_float_min: 40.0,    // °C
            max_free_float_max: 45.0,    // °C
        },
    );

    data
}
```

### Reference Data Sources

Reference data should come from at least two established simulation programs:

- **EnergyPlus**: DOE's flagship building energy simulation program
- **ESP-r**: Research-grade simulation from University of Strathclyde
- **TRNSYS**: Transient System Simulation Tool
- **DOE2**: Legacy DOE simulation program

For each metric, collect both the minimum and maximum values across these programs to create a reference range.

## Step 4: Update Validator

Update `src/validation/ashrae_140_validator.rs` to include the new case in validation runs:

```rust
impl ASHRAE140Validator {
    pub fn validate_analytical_engine(&mut self) -> BenchmarkReport {
        let mut report = BenchmarkReport::new();
        let benchmark_data = benchmark::get_all_benchmark_data();
        let weather = DenverTmyWeather::new();

        // Add new case to validation list
        let cases = vec![
            // ... existing cases ...
            ASHRAE140Case::Case1000, // New case
        ];

        for case in cases {
            let case_id = case.number();
            if let Some(data) = benchmark_data.get(&case_id) {
                // ... validation logic ...
            }
        }

        report
    }
}
```

## Step 5: Run Validation and Check Results

### Run Specific Case

```bash
# Test specific case
cargo test test_case_1000 --lib
```

### Run All Cases

```bash
# Run all ASHRAE 140 validation cases
cargo test test_ashrae_140_validation --lib

# Or use CLI (if available)
fluxion validate --case 1000
fluxion validate --all
```

### Expected Output

The validation output should show:

```
Case 1000: Heating=1.75 MWh (Ref: 1.50-2.00), Cooling=3.5 MWh (Ref: 3.00-4.00)
Status: PASS
```

## Step 6: Debug and Iterate

### Common Issues and Solutions

#### Issue 1: Conductance Values Incorrect

**Symptoms:** Annual energy values significantly outside reference range

**Debug Steps:**
1. Compare conductance calculations with ASHRAE 140 specification formulas
2. Verify units (W/K vs. W/m²K)
3. Check thermal mass coupling ratio (h_tr_em / h_tr_ms)

**Solution:** Adjust conductance values based on ASHRAE 140 formulas:
- `h_tr_em`: Transmission from exterior to thermal mass
- `h_tr_ms`: Transmission from thermal mass to interior surface
- `h_tr_is`: Transmission from interior surface to interior air
- `h_tr_w`: Transmission from exterior to interior (windows)
- `h_ve`: Ventilation heat transfer

#### Issue 2: Solar Gains Overestimated

**Symptoms:** Cooling energy too high compared to reference

**Debug Steps:**
1. Check window SHGC values
2. Verify solar incidence angle calculations
3. Compare hourly solar gain traces with reference

**Solution:**
- Reduce SHGC if needed
- Verify solar algorithm matches ASHRAE 140 specification
- Use diagnostic CSV export to compare hourly profiles

#### Issue 3: Free-Floating Temperatures Incorrect

**Symptoms:** Min/max temperatures outside reference range for free-floating cases

**Debug Steps:**
1. Enable diagnostic output: `ASHRAE140Validator::with_full_diagnostics()`
2. Export hourly data to CSV
3. Compare temperature profiles with reference traces

**Solution:**
- Adjust thermal mass parameters
- Verify internal load schedule
- Check infiltration rates

### Using Diagnostic Tools

Fluxion provides diagnostic tools to help debug cases:

```rust
let validator = ASHRAE140Validator::with_full_diagnostics();
let (report, diagnostic_report) = validator.validate_with_diagnostics();
diagnostic_report.print_summary();
diagnostic_report.export_hourly_csv("case_1000_hourly.csv")?;
```

This provides:
- Hourly temperature profiles
- Energy breakdown (conduction, infiltration, solar, internal gains)
- Peak load timing
- Comparison with reference programs

## Step 7: Add Multi-Reference Data (Optional)

For per-program validation, update `docs/ashrae_140_references.json`:

```json
{
  "cases": {
    "1000": {
      "annual_heating": {
        "EnergyPlus": {"min": 1.5, "max": 1.8},
        "ESP-r": {"min": 1.4, "max": 1.7},
        "TRNSYS": {"min": 1.6, "max": 2.0}
      },
      "annual_cooling": {
        "EnergyPlus": {"min": 3.0, "max": 3.5},
        "ESP-r": {"min": 3.2, "max": 3.8},
        "TRNSYS": {"min": 3.1, "max": 3.7}
      }
    }
  }
}
```

This enables the validator to provide per-program pass/fail status and identify which reference programs Fluxion agrees with.

## Example: Adding Case 1000

Here's a complete example of adding Case 1000:

### 1. Create Case File

```rust
// src/validation/ashrae_140/case_1000.rs
use crate::validation::ashrae_140_cases::CaseSpec;

pub fn create_case_1000() -> CaseSpec {
    let mut spec = CaseSpec::default();
    spec.num_zones = 1;

    // Geometry: 20 m² floor, 3m height
    let mut geometry = crate::validation::ashrae_140_cases::Geometry::default();
    geometry.floor_area = 20.0;
    geometry.height = 3.0;
    geometry.wall_area = geometry.floor_area * geometry.height;
    spec.geometry = vec![geometry];

    // Windows: 15% window-to-floor ratio, U=1.5 W/m²K, SHGC=0.6
    let mut window = crate::validation::ashrae_140_cases::Window::default();
    window.area = geometry.floor_area * 0.15;
    window.u_value = Some(1.5);
    window.shgc = Some(0.6);
    spec.windows = vec![vec![window]];

    // Construction: Wall U=0.5 W/m²K, Roof U=0.3 W/m²K
    let mut construction = crate::validation::ashrae_140_cases::Construction::default();
    construction.wall_u_value = Some(0.5);
    construction.roof_u_value = Some(0.3);
    spec.construction = construction;

    // HVAC: Heat 20°C, Cool 27°C
    let mut hvac = crate::validation::ashrae_140_cases::HVACSchedule::default();
    hvac.heating_setpoint = 20.0;
    hvac.cooling_setpoint = 27.0;
    hvac.heating_enabled = true;
    hvac.cooling_enabled = true;
    spec.hvac = vec![hvac];

    // Internal loads: Light 5 W/m², Equipment 10 W/m², Occupancy 3 W/m²
    let mut internal_loads = crate::validation::ashrae_140_cases::InternalLoads::default();
    internal_loads.lighting = 5.0;
    internal_loads.equipment = 10.0;
    internal_loads.occupancy = 3.0;
    spec.internal_loads = vec![Some(internal_loads)];

    // Infiltration: 0.5 ACH
    spec.infiltration_ach = 0.5;

    // Thermal mass: 1,000,000 J/K
    spec.thermal_mass_c = 1_000_000.0;

    // 5R1C conductances
    spec.h_tr_em = Some(57.42);
    spec.h_tr_ms = Some(1087.5);
    spec.h_tr_is = Some(0.0);
    spec.h_tr_w = Some(0.0);
    spec.h_ve = Some(0.0);

    spec
}
```

### 2. Update Enum

```rust
// src/validation/ashrae_140_cases.rs
#[derive(Debug, Clone, Copy)]
pub enum ASHRAE140Case {
    // ... existing cases ...
    Case1000,
}

impl ASHRAE140Case {
    pub fn spec(&self) -> CaseSpec {
        match self {
            // ... existing cases ...
            ASHRAE140Case::Case1000 => crate::validation::ashrae_140::case_1000::create_case_1000(),
        }
    }

    pub fn number(&self) -> &'static str {
        match self {
            // ... existing cases ...
            ASHRAE140Case::Case1000 => "1000",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            // ... existing cases ...
            ASHRAE140Case::Case1000 => "Custom test case 1000",
        }
    }
}
```

### 3. Add Reference Data

```rust
// src/validation/benchmark.rs
pub fn get_all_benchmark_data() -> HashMap<String, BenchmarkData> {
    let mut data = HashMap::new();

    // ... existing cases ...

    data.insert(
        "1000".to_string(),
        BenchmarkData {
            annual_heating_min: 1.5,
            annual_heating_max: 2.0,
            annual_cooling_min: 3.0,
            annual_cooling_max: 4.0,
            // ... other metrics ...
        },
    );

    data
}
```

### 4. Test and Validate

```bash
# Run tests
cargo test test_case_1000 --lib

# Or run full validation
cargo test test_ashrae_140_validation --lib
```

## Troubleshooting

### Compilation Errors

**Error:** `cannot find function 'create_case_1000'`

**Solution:** Add module declaration in `src/validation/ashrae_140_cases.rs`:
```rust
pub mod case_1000;
pub use case_1000::create_case_1000;
```

### Validation Failures

**Symptom:** Case fails validation consistently

**Debug Steps:**
1. Verify reference data is correct (check units and ranges)
2. Enable full diagnostics: `ASHRAE140Validator::with_full_diagnostics()`
3. Export hourly CSV: `diagnostic_report.export_hourly_csv("debug.csv")?`
4. Compare hourly profiles with reference programs
5. Check conductance calculations against ASHRAE 140 formulas

### Parameter Sensitivity

To understand which parameters most affect validation results, use sensitivity analysis:

```rust
use crate::validation::report::{BenchmarkReport, SensitivityResult, get_parameter_ranges, normalize_sensitivity};

// Collect sensitivity data
let sensitivity_data = vec![
    ("window_u_value".to_string(), sensitivity_coefficient),
    ("hvac_setpoint".to_string(), sensitivity_coefficient),
    // ... more parameters
];

// Normalize and rank
let ranges = get_parameter_ranges();
let normalized = normalize_sensitivity(&sensitivity_data, &ranges);

// Print rankings
for result in normalized {
    println!("Rank {}: {} (normalized: {:.4})",
             result.ranking,
             result.parameter_name,
             result.normalized_coefficient);
}
```

## Best Practices

1. **Use ASHRAE 140 specification formulas** for conductance calculations
2. **Document parameter sources** (which ASHRAE 140 section they come from)
3. **Compare with multiple reference programs** to understand acceptable ranges
4. **Test with both analytical and surrogate modes** if applicable
5. **Use diagnostic tools** for detailed debugging
6. **Add tests** for the new case in the validation test suite
7. **Update documentation** with case description and expected behavior

## References

- **ASHRAE 140 Standard:** ASHRAE Standard 140 - Standard Method of Test for the Evaluation of Building Energy Analysis Computer Programs
- **Fluxion Architecture:** `docs/ARCHITECTURE.md` - Overall system architecture
- **Validation Framework:** `src/validation/ashrae_140_validator.rs` - Validator implementation
- **Diagnostic Tools:** `src/validation/diagnostic.rs` - Diagnostic collection and reporting
- **Known Limitations:** `docs/KNOWN_LIMITATIONS.md` - 5R1C model limitations

## Contributing

When adding new ASHRAE 140 cases:

1. Follow this guide step-by-step
2. Add comprehensive tests for the new case
3. Document any assumptions or deviations from ASHRAE 140 spec
4. Update validation reports with new case results
5. Consider contributing reference data to the multi-reference database

For questions or issues, please open an issue on the Fluxion GitHub repository.
