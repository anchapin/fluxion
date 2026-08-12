# Extending Fluxion with Custom Thermal Models

This tutorial provides a complete guide for developers who want to extend Fluxion with custom thermal models. You'll learn about the BatchOracle and Model classes, the 5R1C thermal network structure, and how to implement custom thermal models for specific building scenarios.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Thermal Model Structure](#thermal-model-structure)
3. [Implementing `heat_transfer()` Method](#implementing-heat_transfer-method)
4. [Integration with SimulationDiagnostics](#integration-with-simulationdiagnostics)
5. [Complete Working Example](#complete-working-example)
6. [Testing and Validation](#testing-and-validation)

---

## Project Overview

### BatchOracle vs Model

Fluxion provides two main classes for building simulation, each designed for different use cases:

#### BatchOracle (High-Throughput Evaluation)

**Purpose:** Evaluate thousands of building configurations in parallel for optimization workflows.

**Use Cases:**
- Population evaluation for genetic algorithms
- Quantum optimization with D-Wave annealers
- Parametric studies with large design spaces
- Real-time optimization scenarios

**Key Characteristics:**
- Uses `rayon` for data parallelism across configurations
- Minimizes Python-Rust boundary crossings
- Returns aggregate fitness scores (EUI - Energy Use Intensity)
- Optimized for throughput (>1,000 configs/sec on 8-core CPU)

**Example:**
```python
import fluxion

# Create oracle
oracle = fluxion.BatchOracle()

# Define population (parameter vectors)
population = [
    [1.5, 21.0],  # [window_u_value, hvac_setpoint]
    [2.0, 22.0],
    [2.5, 20.0],
]

# Evaluate entire population at once
results = oracle.evaluate_population(population, use_surrogates=False)
# Returns: [energy_use_1, energy_use_2, energy_use_3]
```

#### Model (Single-Building Analysis)

**Purpose:** Detailed single-configuration simulation for validation and debugging.

**Use Cases:**
- ASHRAE 140 validation
- Hourly temperature trace analysis
- Detailed energy breakdown studies
- Debugging thermal behavior

**Key Characteristics:**
- Single-configuration focus
- Detailed diagnostic output (hourly temperatures, loads, HVAC demand)
- Peak load tracking
- CSV export capabilities

**Example:**
```python
import fluxion

# Create model
model = fluxion.Model()

# Simulate for 1 year
total_energy = model.simulate(years=1, use_surrogates=False)
print(f"Annual energy: {total_energy} MWh")

# Get diagnostic data
diagnostics = model.get_diagnostics()
peak_heating = diagnostics.get_peak_heating()
print(f"Peak heating load: {peak_heating} kW")
```

**When to Use Each:**

| Scenario | Use BatchOracle | Use Model |
|----------|-----------------|-----------|
| Evaluating 1000+ configurations | ✅ | ❌ (too slow) |
| ASHRAE 140 validation | ❌ | ✅ |
| Optimization loops | ✅ | ❌ |
| Debugging thermal behavior | ❌ | ✅ |
| Hourly trace analysis | ❌ | ✅ |
| Real-time optimization | ✅ | ❌ |

---

## Thermal Model Structure

Fluxion uses an ISO 13790-compliant **5R1C Thermal Network** with **Continuous Tensor Abstraction (CTA)**. This section explains the core components you'll work with when extending thermal models.

### ThermalModel Struct

The `ThermalModel` struct contains all state variables and parameters needed for building simulation:

```rust
pub struct ThermalModel {
    pub num_zones: usize,

    // State variables (CTA VectorFields)
    pub temperatures: VectorField,       // Zone air temperatures (°C)
    pub mass_temperatures: VectorField,  // Thermal mass temperatures (°C)
    pub loads: VectorField,              // Total thermal loads (Watts)

    // Design variables
    pub window_u_value: f64,             // Window thermal transmittance (W/m²K)
    pub hvac_setpoint: f64,              // HVAC target temperature (°C)

    // 5R1C Parameters (CTA VectorFields)
    pub h_tr_em: VectorField,  // Transmission: Exterior -> Mass (W/K)
    pub h_tr_ms: VectorField,  // Transmission: Mass -> Surface (W/K)
    pub h_tr_is: VectorField,  // Transmission: Surface -> Interior (W/K)
    pub h_tr_w: VectorField,   // Transmission: Exterior -> Interior (W/K)
    pub h_ve: VectorField,     // Ventilation: Exterior -> Interior (W/K)

    // Thermal mass parameters (Issues #274, #317)
    pub thermal_mass_correction_factor: f64,
    pub thermal_mass_energy_accounting: bool,
    pub previous_mass_temperatures: VectorField,
    pub mass_energy_change_cumulative: f64,
}
```

### 5R1C Thermal Network Components

The 5R1C (5 Resistance, 1 Capacitance) network represents heat transfer through a building:

**Temperature Nodes:**
- `temperatures`: Zone air temperature for each zone (interior)
- `mass_temperatures`: Thermal mass temperature for each zone

**Heat Transfer Paths (Conductances):**
- `h_tr_em`: Exterior → Mass (transmission through exterior surface)
- `h_tr_ms`: Mass → Surface (thermal coupling)
- `h_tr_is`: Surface → Interior (surface to air coupling)
- `h_tr_w`: Exterior → Interior (window transmission)
- `h_ve`: Exterior → Interior (ventilation)

**Design Variables:**
- `window_u_value`: Window thermal transmittance (affects `h_tr_w`)
- `hvac_setpoint`: Target indoor temperature

### Continuous Tensor Abstraction (CTA)

Fluxion uses CTA to abstract vector operations, enabling future GPU acceleration:

```rust
use fluxion::physics::cta::VectorField;

// Create a VectorField (8760 hourly values)
let loads = VectorField::new(vec![100.0; 8760]);

// Element-wise operations
let total_load = loads.integrate();  // Sum all hourly values
let peak_load = loads.max();         // Find maximum value

// CTA operations are optimized for vectorization
```

**Key Point:** The physics engine uses CTA operations (`+`, `*`, `/`) on `VectorField` types, not raw `Vec<f64>`. This enables GPU acceleration while maintaining simple code.

### Parameter Vector Semantics

When using BatchOracle or Model with custom parameters, the parameter vector format is critical:

**Element 0: Window U-value**
- Range: 0.1 – 5.0 W/m²K
- Affects: `h_tr_w` (window conductance), `h_tr_em` (exterior transmission)
- Lower values = better insulation, less heat transfer

**Element 1: HVAC Setpoint**
- Range: 15 – 30°C
- Affects: HVAC activation threshold
- Higher setpoints = less heating, more cooling

**Future Elements (Planned):**
- Element 2: Thermal mass capacitance
- Element 3: Infiltration rate
- Element 4: Solar heat gain coefficient (SHGC)

**Constants:**
```rust
const MIN_U_VALUE: f64 = 0.1;
const MAX_U_VALUE: f64 = 5.0;
const MIN_SETPOINT: f64 = 15.0;
const MAX_SETPOINT: f64 = 30.0;
```

### Multi-Zone Models

For buildings with multiple zones, `num_zones` specifies the number of thermal zones:

```rust
// Single-zone model
let single_zone = ThermalModel::new(1);

// Multi-zone model (3 zones)
let multi_zone = ThermalModel::new(3);
```

Each zone has its own temperature, mass temperature, and load state. The conductances (`h_tr_*`) and design variables (`window_u_value`, `hvac_setpoint`) are broadcast to all zones unless specifically configured.

### Cloning for Parallelism

`ThermalModel` implements `Clone`, which enables the batch parallel pattern:

```rust
use rayon::prelude::*;

// Base model (configured once)
let base_model = ThermalModel::new(1);
base_model.apply_parameters(&[2.0, 21.0]);

// Evaluate multiple configurations in parallel
let results: Vec<f64> = population
    .par_iter()
    .map(|params| {
        // Clone model for each thread
        let mut model = base_model.clone();
        model.apply_parameters(params);
        model.solve_timesteps(8760, &surrogates, false)
    })
    .collect();
```

**Critical:** Clone at the population level, not inside timesteps. This avoids nested parallelism and maximizes performance.

---

## Implementing `heat_transfer()` Method

The `heat_transfer()` method is the primary extension point for custom thermal models. This section shows how to implement it.

### Method Signature

```rust
fn heat_transfer(&self) -> VectorField
```

**Returns:** Total heat transfer per zone (Watts) as a `VectorField` of hourly values.

### Solar Gain Integration

Solar radiation is a major heat source in buildings. Fluxion calculates solar gains based on window properties and sun position:

```rust
impl ThermalModel<VectorField> {
    fn heat_transfer(&self) -> VectorField {
        let mut total_heat_transfer = VectorField::zeros(8760);

        // Get solar radiation from weather data
        let solar_radiation = self.weather.solar_horizontal;

        // Solar gain through windows
        let window_area = self.geometry.window_area;  // m²
        let shgc = 0.7;  // Solar Heat Gain Coefficient

        // Calculate solar gain for each hour
        for hour in 0..8760 {
            let solar_gain = solar_radiation[hour] * window_area * shgc;
            total_heat_transfer.data[hour] += solar_gain;
        }

        total_heat_transfer
    }
}
```

### Inter-Zone Heat Transfer

For multi-zone models, heat can transfer between zones (e.g., through interior walls):

```rust
fn heat_transfer(&self) -> VectorField {
    let mut total_heat_transfer = VectorField::zeros(8760);

    // Solar gains (as above)
    total_heat_transfer += self.calculate_solar_gains();

    // Inter-zone heat transfer
    if self.num_zones > 1 {
        for zone in 0..self.num_zones {
            for neighbor in 0..self.num_zones {
                if zone != neighbor {
                    // Heat transfer between zones
                    let temp_diff = self.temperatures - self.temperatures_of(neighbor);
                    let coupling = self.inter_zone_coupling(zone, neighbor);  // W/K
                    let transfer = coupling * temp_diff;
                    total_heat_transfer += transfer;
                }
            }
        }
    }

    total_heat_transfer
}
```

### Ventilation Heat Transfer

Ventilation exchanges indoor and outdoor air, bringing in outdoor temperature:

```rust
fn heat_transfer(&self) -> VectorField {
    let mut total_heat_transfer = VectorField::zeros(8760);

    // Solar gains
    total_heat_transfer += self.calculate_solar_gains();

    // Ventilation heat transfer
    let air_density = 1.2;  // kg/m³
    let specific_heat = 1000.0;  // J/kgK

    for hour in 0..8760 {
        let outdoor_temp = self.weather.dry_bulb_temp[hour];
        let indoor_temp = self.temperatures.data[hour];
        let temp_diff = outdoor_temp - indoor_temp;

        // Heat transfer due to ventilation
        // Q = m_dot * cp * ΔT
        // m_dot = ρ * V_dot (mass flow rate)
        let volumetric_flow_rate = 0.5;  // m³/s (infiltration rate)
        let mass_flow_rate = air_density * volumetric_flow_rate;  // kg/s
        let ventilation_heat = mass_flow_rate * specific_heat * temp_diff;  // W

        total_heat_transfer.data[hour] += ventilation_heat;
    }

    total_heat_transfer
}
```

### Complete `heat_transfer()` Example

Here's a complete implementation combining all components:

```rust
impl ThermalModel<VectorField> {
    fn heat_transfer(&self) -> VectorField {
        let mut total_heat_transfer = VectorField::zeros(8760);
        let air_density = 1.2;  // kg/m³
        let specific_heat = 1000.0;  // J/kgK

        for hour in 0..8760 {
            // 1. Solar gains through windows
            let solar_radiation = self.weather.solar_horizontal[hour];
            let window_area = self.geometry.window_area;
            let shgc = 0.7;  // Solar Heat Gain Coefficient
            let solar_gain = solar_radiation * window_area * shgc;
            total_heat_transfer.data[hour] += solar_gain;

            // 2. Ventilation heat transfer
            let outdoor_temp = self.weather.dry_bulb_temp[hour];
            let indoor_temp = self.temperatures.data[hour];
            let temp_diff = outdoor_temp - indoor_temp;
            let volumetric_flow_rate = 0.5;  // m³/s (infiltration rate)
            let mass_flow_rate = air_density * volumetric_flow_rate;
            let ventilation_heat = mass_flow_rate * specific_heat * temp_diff;
            total_heat_transfer.data[hour] += ventilation_heat;

            // 3. Inter-zone heat transfer (for multi-zone models)
            if self.num_zones > 1 {
                for zone in 0..self.num_zones {
                    for neighbor in 0..self.num_zones {
                        if zone != neighbor {
                            let coupling = self.inter_zone_coupling(zone, neighbor);
                            let zone_temp = self.temperatures_of(zone).data[hour];
                            let neighbor_temp = self.temperatures_of(neighbor).data[hour];
                            let transfer = coupling * (neighbor_temp - zone_temp);
                            total_heat_transfer.data[hour] += transfer;
                        }
                    }
                }
            }
        }

        total_heat_transfer
    }
}
```

---

## Integration with SimulationDiagnostics

`SimulationDiagnostics` collects detailed diagnostic data during simulation, including hourly temperatures, loads, and HVAC demand.

### Diagnostic Data Collection

```rust
use fluxion::sim::diagnostics::SimulationDiagnostics;

// Create model with diagnostics
let mut model = ThermalModel::new(1);
model.enable_diagnostics();

// Run simulation
model.solve_timesteps(8760, &surrogates, false);

// Get diagnostic data
let diagnostics = model.get_diagnostics();
```

### Hourly Temperature Tracking

`SimulationDiagnostics` tracks zone temperatures at each timestep:

```rust
let diagnostics = model.get_diagnostics();

// Get hourly temperatures for all zones
let hourly_temps = diagnostics.get_hourly_temperatures();

// Get temperature for a specific zone (0-indexed)
let zone_0_temps = hourly_temps.get_zone(0);

// Access specific hour
let hour_100_temp = zone_0_temps.data[100];
println!("Temperature at hour 100: {:.2}°C", hour_100_temp);
```

### Load Tracking

`SimulationDiagnostics` tracks thermal loads (heating/cooling demand):

```rust
let diagnostics = model.get_diagnostics();

// Get hourly loads (positive = heating, negative = cooling)
let hourly_loads = diagnostics.get_hourly_loads();

// Calculate total annual energy
let annual_heating = hourly_loads.heating_energy_mwh();
let annual_cooling = hourly_loads.cooling_energy_mwh();
println!("Annual heating: {:.2} MWh", annual_heating);
println!("Annual cooling: {:.2} MWh", annual_cooling);
```

### Peak Load Tracking

Peak loads are important for HVAC system sizing:

```rust
let diagnostics = model.get_diagnostics();

// Get peak loads
let peak_heating = diagnostics.get_peak_heating();  // kW
let peak_cooling = diagnostics.get_peak_cooling();  // kW

println!("Peak heating load: {:.2} kW", peak_heating);
println!("Peak cooling load: {:.2} kW", peak_cooling);
```

### Free-Floating Temperature Calculation

Free-floating temperature is the indoor temperature without HVAC operation:

```rust
let diagnostics = model.get_diagnostics();

// Get free-floating temperatures
let free_float_temps = diagnostics.get_free_floating_temperatures();

// Analyze thermal mass effect
let avg_free_float = free_float_temps.mean();
println!("Average free-floating temp: {:.2}°C", avg_free_float);
```

### CSV Export

Export diagnostic data to CSV for analysis in Excel, Python, or other tools:

```rust
use std::fs::File;
use std::io::Write;

let diagnostics = model.get_diagnostics();

// Export hourly temperatures
let mut csv_file = File::create("hourly_temperatures.csv")?;
writeln!(csv_file, "Hour,Temperature_C")?;
for (hour, &temp) in diagnostics.get_hourly_temperatures().data.iter().enumerate() {
    writeln!(csv_file, "{},{:.2}", hour, temp)?;
}

// Export hourly loads
let mut loads_file = File::create("hourly_loads.csv")?;
writeln!(loads_file, "Hour,Load_W")?;
for (hour, &load) in diagnostics.get_hourly_loads().data.iter().enumerate() {
    writeln!(loads_file, "{},{:.2}", hour, load)?;
}
```

### Diagnostic Summary

Generate a summary report of simulation results:

```rust
let diagnostics = model.get_diagnostics();

println!("=== Simulation Summary ===");
println!("Total simulation time: {} hours", diagnostics.get_simulation_hours());
println!("Peak heating: {:.2} kW at hour {}", diagnostics.get_peak_heating(), diagnostics.get_peak_heating_hour());
println!("Peak cooling: {:.2} kW at hour {}", diagnostics.get_peak_cooling(), diagnostics.get_peak_cooling_hour());
println!("Annual heating: {:.2} MWh", diagnostics.get_annual_heating_mwh());
println!("Annual cooling: {:.2} MWh", diagnostics.get_annual_cooling_mwh());
println!("Total energy: {:.2} MWh", diagnostics.get_annual_energy_mwh());
```

---

## Complete Working Example

This section provides a complete, runnable example of extending Fluxion with a custom thermal model.

### Example: Custom Office Building

Let's create a custom thermal model for an office building and compare it against the ASHRAE 140 Case 600 baseline.

```rust
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec};
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Custom office building thermal model
struct OfficeBuilding {
    base_model: ThermalModel<VectorField>,
    occupancy_schedule: Vec<f64>,  // Occupancy fraction (0-1) per hour
    equipment_load: Vec<f64>,      // Equipment internal heat gain (W) per hour
    lighting_load: Vec<f64>,       // Lighting internal heat gain (W) per hour
}

impl OfficeBuilding {
    /// Create a new office building model
    fn new(spec: &CaseSpec) -> Self {
        let base_model = ThermalModel::<VectorField>::from_spec(spec);

        // Generate occupancy schedule (9 AM - 5 PM, Mon-Fri)
        let occupancy_schedule = (0..8760)
            .map(|hour| {
                let hour_of_day = hour % 24;
                let day_of_week = (hour / 24) % 7;
                if day_of_week < 5 && hour_of_day >= 9 && hour_of_day < 17 {
                    1.0  // Full occupancy
                } else {
                    0.0  // Unoccupied
                }
            })
            .collect();

        // Equipment load (higher during occupied hours)
        let equipment_load = occupancy_schedule
            .iter()
            .map(|&occ| occ * 2000.0)  // 2 kW peak equipment load
            .collect();

        // Lighting load (higher during occupied hours)
        let lighting_load = occupancy_schedule
            .iter()
            .map(|&occ| occ * 1500.0)  // 1.5 kW peak lighting load
            .collect();

        OfficeBuilding {
            base_model,
            occupancy_schedule,
            equipment_load,
            lighting_load,
        }
    }

    /// Apply custom parameters
    fn apply_parameters(&mut self, params: &[f64]) {
        self.base_model.apply_parameters(params);
    }

    /// Calculate internal heat gains (occupants + equipment + lighting)
    fn internal_heat_gains(&self) -> VectorField {
        let mut gains = VectorField::zeros(8760);

        for hour in 0..8760 {
            let occupancy_gains = self.occupancy_schedule[hour] * 100.0;  // 100 W per occupant
            gains.data[hour] = self.equipment_load[hour] + self.lighting_load[hour] + occupancy_gains;
        }

        gains
    }

    /// Simulate for a year
    fn simulate_year(&mut self) -> (f64, f64, f64, f64) {
        let weather = DenverTmyWeather::new();
        const STEPS: usize = 8760;

        let mut annual_heating_joules = 0.0;
        let mut annual_cooling_joules = 0.0;
        let mut peak_heating_watts: f64 = 0.0;
        let mut peak_cooling_watts: f64 = 0.0;

        // Enable diagnostics
        self.base_model.enable_diagnostics();

        for step in 0..STEPS {
            // Get weather data
            let weather_data = weather.get_hourly_data(step).unwrap();

            // Update weather data on model
            self.base_model.set_weather(weather_data.clone());

            // Calculate heat transfer (base + internal gains)
            let base_heat_transfer = self.base_model.heat_transfer();
            let internal_gains = self.internal_heat_gains();
            let total_heat_transfer = &base_heat_transfer + &internal_gains;

            // Apply heat transfer to model
            self.base_model.set_heat_transfer(total_heat_transfer);

            // Step physics
            let hvac_energy_for_step = self.base_model.step_physics(step, weather_data.dry_bulb_temp);

            // Track peaks
            if hvac_energy_for_step > 0.0 {
                // Heating
                peak_heating_watts = peak_heating_watts.max(hvac_energy_for_step);
            } else {
                // Cooling
                let cooling_demand = -hvac_energy_for_step;
                peak_cooling_watts = peak_cooling_watts.max(cooling_demand);
            }

            // Accumulate energy (J = W × s)
            annual_heating_joules += hvac_energy_for_step.max(0.0) * 3600.0;
            annual_cooling_joules += hvac_energy_for_step.min(0.0).abs() * 3600.0;
        }

        // Convert to MWh
        let annual_heating_mwh = annual_heating_joules / 3.6e9;
        let annual_cooling_mwh = annual_cooling_joules / 3.6e9;
        let peak_heating_kw = peak_heating_watts / 1000.0;
        let peak_cooling_kw = peak_cooling_watts / 1000.0;

        (annual_heating_mwh, annual_cooling_mwh, peak_heating_kw, peak_cooling_kw)
    }
}

fn main() {
    println!("Custom Office Building Simulation");
    println!("==================================\n");

    // Create custom office building
    let case_600_spec = ASHRAE140Case::Case600.spec();
    let mut office = OfficeBuilding::new(&case_600_spec);

    // Apply parameters
    office.apply_parameters(&[2.0, 21.0]);  // [window_u_value, hvac_setpoint]

    // Simulate for a year
    let (heating, cooling, peak_heat, peak_cool) = office.simulate_year();

    println!("Office Building Results:");
    println!("  Annual heating: {:.2} MWh", heating);
    println!("  Annual cooling: {:.2} MWh", cooling);
    println!("  Peak heating: {:.2} kW", peak_heat);
    println!("  Peak cooling: {:.2} kW", peak_cool);

    // Compare with Case 600 baseline
    println!("\nComparison with Case 600 Baseline:");
    println!("  Case 600 heating: ~2.13 MWh");
    println!("  Case 600 cooling: ~0.93 MWh");
    println!("  Difference (heating): {:+.2} MWh ({:+.1}%)",
        heating - 2.13, ((heating - 2.13) / 2.13) * 100.0);
    println!("  Difference (cooling): {:+.2} MWh ({:+.1}%)",
        cooling - 0.93, ((cooling - 0.93) / 0.93) * 100.0);

    // Get diagnostic data
    let diagnostics = office.base_model.get_diagnostics();
    println!("\nDiagnostic Summary:");
    println!("  Peak heating hour: {}", diagnostics.get_peak_heating_hour());
    println!("  Peak cooling hour: {}", diagnostics.get_peak_cooling_hour());
    println!("  Free-floating average: {:.2}°C", diagnostics.get_free_floating_temperatures().mean());
}
```

### BatchOracle Usage Example

Evaluate multiple office building configurations in parallel:

```rust
use fluxion::BatchOracle;

fn main() {
    let oracle = BatchOracle::new().unwrap();

    // Define population (different window U-values and HVAC setpoints)
    let population = vec![
        vec![1.5, 20.0],  // Good insulation, lower setpoint
        vec![2.0, 21.0],  // Medium insulation, medium setpoint
        vec![2.5, 22.0],  // Poor insulation, higher setpoint
    ];

    // Evaluate population
    let results = oracle.evaluate_population(population, false).unwrap();

    println!("Batch Evaluation Results:");
    for (i, energy) in results.iter().enumerate() {
        println!("  Config {}: {:.2} MWh", i, energy);
    }
}
```

### Model Usage Example

Detailed single-configuration analysis:

```rust
use fluxion::Model;

fn main() {
    let model = Model::new().unwrap();

    // Apply parameters
    model.apply_parameters(&[2.0, 21.0]);

    // Simulate for 1 year
    let total_energy = model.simulate(1, false);
    println!("Total annual energy: {:.2} MWh", total_energy);

    // Get diagnostic data
    let diagnostics = model.get_diagnostics();

    // Export to CSV
    diagnostics.export_to_csv("simulation_results.csv").unwrap();

    println!("Hourly results exported to simulation_results.csv");
}
```

---

## Testing and Validation

### Error Handling

Always handle errors properly when working with custom thermal models:

```rust
use fluxion::sim::engine::ValidationError;

fn safe_simulation() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = ThermalModel::new(1);

    // Validate parameters before applying
    let params = vec![2.0, 21.0];
    if let Err(e) = model.validate_parameters(&params) {
        return Err(e.into());
    }

    // Apply parameters
    model.apply_parameters(&params);

    // Simulate and handle errors
    let surrogates = SurrogateManager::new()?;
    let energy = model.solve_timesteps(8760, &surrogates, false);

    // Check for NaN/Inf
    if energy.is_nan() || energy.is_infinite() {
        return Err(ValidationError::InvalidSimulationResult(energy).into());
    }

    println!("Simulation successful: {:.2} MWh", energy);
    Ok(())
}
```

### Testing Against Baseline

Validate your custom model against known baselines:

```rust
#[test]
fn test_custom_model_against_case_600() {
    let case_600_spec = ASHRAE140Case::Case600.spec();
    let mut custom_model = OfficeBuilding::new(&case_600_spec);
    custom_model.apply_parameters(&[2.0, 21.0]);

    let (heating, cooling, _, _) = custom_model.simulate_year();

    // Compare with Case 600 baseline (within ±15% tolerance)
    assert!((heating - 2.13).abs() / 2.13 < 0.15,
        "Heating {:.2} MWh outside 15% tolerance of baseline 2.13 MWh", heating);
    assert!((cooling - 0.93).abs() / 0.93 < 0.15,
        "Cooling {:.2} MWh outside 15% tolerance of baseline 0.93 MWh", cooling);
}
```

### Performance Testing

Ensure your custom model meets performance targets:

```rust
use std::time::Instant;

#[test]
fn test_custom_model_performance() {
    let start = Instant::now();

    let case_600_spec = ASHRAE140Case::Case600.spec();
    let mut custom_model = OfficeBuilding::new(&case_600_spec);
    custom_model.apply_parameters(&[2.0, 21.0]);
    custom_model.simulate_year();

    let duration = start.elapsed();

    // Should complete in <100ms for single configuration
    assert!(duration.as_millis() < 100,
        "Simulation took {}ms, expected <100ms", duration.as_millis());
}
```

---

## Next Steps

- **Explore API Reference:** See `docs/API_REFERENCE.md` for comprehensive BatchOracle and Model documentation
- **Read Architecture:** See `docs/ARCHITECTURE.md` for deep dive into Fluxion's design
- **Study Examples:** Check `examples/` directory for more working examples
- **Contribute:** See `docs/CONTRIBUTING.md` for contribution guidelines

## See Also

- [API Reference](../API_REFERENCE.md) - Comprehensive BatchOracle and Model documentation
- [Architecture Overview](../ARCHITECTURE.md) - Deep dive into Fluxion's design
- [ASHRAE 140 Validation](../ASHRAE_140_RESULTS.md) - Validation methodology and results
- [Known Limitations](../KNOWN_ISSUES.md) - Current model limitations and planned improvements
