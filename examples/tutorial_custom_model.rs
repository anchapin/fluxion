use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec};
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Custom office building thermal model with internal heat gains
///
/// This example demonstrates how to extend Fluxion with a custom thermal model
/// that includes occupancy, equipment, and lighting schedules.
#[derive(Clone)]
struct OfficeBuilding {
    base_model: ThermalModel<VectorField>,
    occupancy_schedule: Vec<f64>, // Occupancy fraction (0-1) per hour
    equipment_load: Vec<f64>,     // Equipment internal heat gain (W) per hour
    lighting_load: Vec<f64>,      // Lighting internal heat gain (W) per hour
}

impl OfficeBuilding {
    /// Create a new office building model from an ASHRAE 140 case specification
    ///
    /// # Arguments
    /// * `spec` - Case specification defining building geometry and properties
    ///
    /// # Returns
    /// Initialized office building model with default schedules
    fn new(spec: &CaseSpec) -> Self {
        let base_model = ThermalModel::<VectorField>::from_spec(spec);

        // Generate occupancy schedule (9 AM - 5 PM, Mon-Fri)
        let occupancy_schedule = (0..8760)
            .map(|hour| {
                let hour_of_day = hour % 24;
                let day_of_week = (hour / 24) % 7;
                if day_of_week < 5 && hour_of_day >= 9 && hour_of_day < 17 {
                    1.0 // Full occupancy during business hours
                } else {
                    0.0 // Unoccupied outside business hours
                }
            })
            .collect();

        // Equipment load (higher during occupied hours)
        let equipment_load = occupancy_schedule
            .iter()
            .map(|&occ| occ * 2000.0) // 2 kW peak equipment load
            .collect();

        // Lighting load (higher during occupied hours)
        let lighting_load = occupancy_schedule
            .iter()
            .map(|&occ| occ * 1500.0) // 1.5 kW peak lighting load
            .collect();

        OfficeBuilding {
            base_model,
            occupancy_schedule,
            equipment_load,
            lighting_load,
        }
    }

    /// Apply custom parameters to the model
    ///
    /// # Arguments
    /// * `params` - Parameter vector: [window_u_value, hvac_setpoint]
    ///
    /// # Parameter Bounds
    /// - Window U-value: 0.1 - 5.0 W/m²K
    /// - HVAC Setpoint: 15 - 30°C
    ///
    /// # Returns
    /// Result indicating success or validation error
    fn apply_parameters(&mut self, params: &[f64]) -> Result<(), String> {
        // Validate parameter count
        if params.len() != 2 {
            return Err(format!("Expected 2 parameters, got {}", params.len()));
        }

        // Validate parameter bounds
        const MIN_U_VALUE: f64 = 0.1;
        const MAX_U_VALUE: f64 = 5.0;
        const MIN_SETPOINT: f64 = 15.0;
        const MAX_SETPOINT: f64 = 30.0;

        if params[0] < MIN_U_VALUE || params[0] > MAX_U_VALUE {
            return Err(format!(
                "Window U-value {:.2} outside range [{:.1}, {:.1}] W/m²K",
                params[0], MIN_U_VALUE, MAX_U_VALUE
            ));
        }

        if params[1] < MIN_SETPOINT || params[1] > MAX_SETPOINT {
            return Err(format!(
                "HVAC setpoint {:.2}°C outside range [{:.1}, {:.1}]°C",
                params[1], MIN_SETPOINT, MAX_SETPOINT
            ));
        }

        // Apply parameters to base model
        self.base_model.apply_parameters(params);
        Ok(())
    }

    /// Calculate internal heat gains from occupants, equipment, and lighting
    ///
    /// # Returns
    /// VectorField of hourly internal heat gains (Watts)
    fn internal_heat_gains(&self) -> VectorField {
        let mut gains = VectorField::zeros(8760);

        for hour in 0..8760 {
            // Occupant heat gain (100 W per occupant during occupied hours)
            let occupancy_gains = self.occupancy_schedule[hour] * 100.0;

            // Total internal gains = equipment + lighting + occupants
            gains.data[hour] =
                self.equipment_load[hour] + self.lighting_load[hour] + occupancy_gains;
        }

        gains
    }

    /// Simulate building for one year
    ///
    /// # Returns
    /// Tuple of (annual_heating_mwh, annual_cooling_mwh, peak_heating_kw, peak_cooling_kw)
    fn simulate_year(&mut self) -> (f64, f64, f64, f64) {
        let weather = DenverTmyWeather::new();
        const STEPS: usize = 8760;

        let mut annual_heating_joules = 0.0;
        let mut annual_cooling_joules = 0.0;
        let mut peak_heating_watts: f64 = 0.0;
        let mut peak_cooling_watts: f64 = 0.0;

        // Enable diagnostics for detailed output
        self.base_model.enable_diagnostics();

        for step in 0..STEPS {
            // Get weather data
            let weather_data = weather.get_hourly_data(step).unwrap();

            // Update weather data on model for solar gain calculation
            self.base_model.set_weather(weather_data.clone());

            // Calculate heat transfer (base + internal gains)
            let base_heat_transfer = self.base_model.heat_transfer();
            let internal_gains = self.internal_heat_gains();
            let total_heat_transfer = &base_heat_transfer + &internal_gains;

            // Apply total heat transfer to model
            self.base_model.set_heat_transfer(total_heat_transfer);

            // Step physics (analytical path, no surrogates)
            let hvac_energy_for_step = self
                .base_model
                .step_physics(step, weather_data.dry_bulb_temp);

            // Track peaks
            if hvac_energy_for_step > 0.0 {
                // Heating
                peak_heating_watts = peak_heating_watts.max(hvac_energy_for_step);
            } else {
                // Cooling (store as positive value)
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

        (
            annual_heating_mwh,
            annual_cooling_mwh,
            peak_heating_kw,
            peak_cooling_kw,
        )
    }

    /// Get diagnostic data from simulation
    fn get_diagnostics(&self) -> &fluxion::sim::diagnostics::SimulationDiagnostics {
        self.base_model.get_diagnostics()
    }
}

/// BatchOracle usage example: Evaluate multiple configurations in parallel
fn batch_oracle_example() {
    println!("\n=== BatchOracle Example ===");

    // Note: This would normally use fluxion::BatchOracle from Python bindings
    // Here we demonstrate the pattern manually for educational purposes

    let case_600_spec = ASHRAE140Case::Case600.spec();

    // Define population (different window U-values and HVAC setpoints)
    let configurations = vec![
        vec![1.5, 20.0], // Good insulation, lower setpoint
        vec![2.0, 21.0], // Medium insulation, medium setpoint (baseline)
        vec![2.5, 22.0], // Poor insulation, higher setpoint
    ];

    println!(
        "Evaluating {} configurations in parallel...",
        configurations.len()
    );

    // Evaluate each configuration
    let mut results = Vec::new();
    for (i, params) in configurations.iter().enumerate() {
        let mut building = OfficeBuilding::new(&case_600_spec);
        building.apply_parameters(params).unwrap();
        let (heating, cooling, _, _) = building.simulate_year();
        let total_energy = heating + cooling;
        results.push((i, total_energy));
        println!(
            "  Config {}: U={:.1} W/m²K, Setpoint={:.1}°C → {:.2} MWh",
            i, params[0], params[1], total_energy
        );
    }

    // Find optimal configuration
    let best = results
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    println!(
        "\nBest configuration: Config {} ({:.2} MWh)",
        best.0, best.1
    );
}

/// Model usage example: Detailed single-configuration analysis
fn model_example() {
    println!("\n=== Model Example ===");

    let case_600_spec = ASHRAE140Case::Case600.spec();
    let mut building = OfficeBuilding::new(&case_600_spec);

    // Apply parameters
    building.apply_parameters(&[2.0, 21.0]).unwrap();

    // Simulate for 1 year
    let (heating, cooling, peak_heat, peak_cool) = building.simulate_year();
    let total_energy = heating + cooling;

    println!("Single-configuration simulation:");
    println!("  Annual heating: {:.2} MWh", heating);
    println!("  Annual cooling: {:.2} MWh", cooling);
    println!("  Total energy: {:.2} MWh", total_energy);
    println!("  Peak heating: {:.2} kW", peak_heat);
    println!("  Peak cooling: {:.2} kW", peak_cool);

    // Get diagnostic data
    let diagnostics = building.get_diagnostics();

    println!("\nDiagnostic Summary:");
    println!(
        "  Peak heating hour: {}",
        diagnostics.get_peak_heating_hour()
    );
    println!(
        "  Peak cooling hour: {}",
        diagnostics.get_peak_cooling_hour()
    );
    println!(
        "  Free-floating average: {:.2}°C",
        diagnostics.get_free_floating_temperatures().mean()
    );
    println!(
        "  Total simulation hours: {}",
        diagnostics.get_simulation_hours()
    );
}

/// Comparison with Case 600 baseline
fn baseline_comparison() {
    println!("\n=== Baseline Comparison ===");

    let case_600_spec = ASHRAE140Case::Case600.spec();
    let mut building = OfficeBuilding::new(&case_600_spec);

    // Apply parameters
    building.apply_parameters(&[2.0, 21.0]).unwrap();

    // Simulate for 1 year
    let (heating, cooling, peak_heat, peak_cool) = building.simulate_year();

    // Compare with Case 600 baseline
    let baseline_heating = 2.13; // MWh
    let baseline_cooling = 0.93; // MWh

    println!("Custom Office Building Results:");
    println!("  Annual heating: {:.2} MWh", heating);
    println!("  Annual cooling: {:.2} MWh", cooling);
    println!("  Peak heating: {:.2} kW", peak_heat);
    println!("  Peak cooling: {:.2} kW", peak_cool);

    println!("\nCase 600 Baseline:");
    println!("  Heating: ~2.13 MWh");
    println!("  Cooling: ~0.93 MWh");

    let heating_diff = heating - baseline_heating;
    let cooling_diff = cooling - baseline_cooling;
    let heating_pct = (heating_diff / baseline_heating) * 100.0;
    let cooling_pct = (cooling_diff / baseline_cooling) * 100.0;

    println!("\nDifference from Baseline:");
    println!("  Heating: {:+.2} MWh ({:+.1}%)", heating_diff, heating_pct);
    println!("  Cooling: {:+.2} MWh ({:+.1}%)", cooling_diff, cooling_pct);

    // Check ASHRAE 140 tolerance (±15%)
    let heating_pass = heating_pct.abs() < 15.0;
    let cooling_pass = cooling_pct.abs() < 15.0;

    println!("\nASHRAE 140 Tolerance Check (±15%):");
    println!(
        "  Heating: {} ({:.1}% vs ±15%)",
        if heating_pass { "PASS" } else { "FAIL" },
        heating_pct.abs()
    );
    println!(
        "  Cooling: {} ({:.1}% vs ±15%)",
        if cooling_pass { "PASS" } else { "FAIL" },
        cooling_pct.abs()
    );
}

/// Error handling demonstration
fn error_handling_example() {
    println!("\n=== Error Handling Example ===");

    let case_600_spec = ASHRAE140Case::Case600.spec();

    // Test 1: Invalid parameter count
    println!("Test 1: Invalid parameter count");
    let mut building = OfficeBuilding::new(&case_600_spec);
    match building.apply_parameters(&[2.0]) {
        Ok(_) => println!("  ERROR: Should have failed"),
        Err(e) => println!("  ✓ Expected error: {}", e),
    }

    // Test 2: Window U-value out of bounds (too low)
    println!("\nTest 2: Window U-value out of bounds (too low)");
    let mut building = OfficeBuilding::new(&case_600_spec);
    match building.apply_parameters(&[0.05, 21.0]) {
        Ok(_) => println!("  ERROR: Should have failed"),
        Err(e) => println!("  ✓ Expected error: {}", e),
    }

    // Test 3: Window U-value out of bounds (too high)
    println!("\nTest 3: Window U-value out of bounds (too high)");
    let mut building = OfficeBuilding::new(&case_600_spec);
    match building.apply_parameters(&[6.0, 21.0]) {
        Ok(_) => println!("  ERROR: Should have failed"),
        Err(e) => println!("  ✓ Expected error: {}", e),
    }

    // Test 4: HVAC setpoint out of bounds (too low)
    println!("\nTest 4: HVAC setpoint out of bounds (too low)");
    let mut building = OfficeBuilding::new(&case_600_spec);
    match building.apply_parameters(&[2.0, 10.0]) {
        Ok(_) => println!("  ERROR: Should have failed"),
        Err(e) => println!("  ✓ Expected error: {}", e),
    }

    // Test 5: HVAC setpoint out of bounds (too high)
    println!("\nTest 5: HVAC setpoint out of bounds (too high)");
    let mut building = OfficeBuilding::new(&case_600_spec);
    match building.apply_parameters(&[2.0, 35.0]) {
        Ok(_) => println!("  ERROR: Should have failed"),
        Err(e) => println!("  ✓ Expected error: {}", e),
    }

    // Test 6: Valid parameters
    println!("\nTest 6: Valid parameters");
    let mut building = OfficeBuilding::new(&case_600_spec);
    match building.apply_parameters(&[2.0, 21.0]) {
        Ok(_) => println!("  ✓ Success: Parameters applied"),
        Err(e) => println!("  ERROR: Should have succeeded - {}", e),
    }
}

fn main() {
    println!("Fluxion Tutorial: Custom Thermal Model Extension");
    println!("================================================\n");

    println!("This example demonstrates how to extend Fluxion with custom thermal models.");
    println!(
        "We create an OfficeBuilding model with occupancy, equipment, and lighting schedules.\n"
    );

    // Run examples
    baseline_comparison();
    model_example();
    batch_oracle_example();
    error_handling_example();

    println!("\n=== Summary ===");
    println!("This tutorial showed:");
    println!("  1. Creating custom thermal models with internal heat gains");
    println!("  2. Implementing parameter validation");
    println!("  3. Simulating single and multiple configurations");
    println!("  4. Comparing against ASHRAE 140 baselines");
    println!("  5. Handling errors properly");
    println!("\nSee docs/tutorials/extending_fluxion.md for the complete tutorial.");
}
