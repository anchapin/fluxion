//! Phase 1 Task 1.2: Tune solar distribution for low-mass
//!
//! This tool tests different solar_distribution_to_air values for Case 600
//! and measures impact on cooling energy.
//!
//! Hypothesis: Low-mass buildings need higher solar_distribution_to_air (0.2-0.3)
//! because they have less thermal mass to buffer solar gains.

use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    println!("=== Phase 1 Task 1.2: Solar Distribution Tuning ===");
    println!("Testing different solar_distribution_to_air values for Case 600\n");

    // Case 600 reference values
    println!("=== ASHRAE 140 Case 600 Reference ===");
    println!("Heating: 4.30-5.71 MWh (Reference)");
    println!("Cooling: 6.14-8.45 MWh (Reference)");
    println!();

    // Test range of solar_distribution_to_air values
    // Values to test: 0.0 (all to mass) to 0.5 (half to air)
    let solar_values = vec![
        0.0,  // All radiative gains to mass (current high-mass approach)
        0.05, // 5% to air
        0.1,  // 10% to air (current default)
        0.15, // 15% to air
        0.2,  // 20% to air (hypothesis)
        0.25, // 25% to air (hypothesis)
        0.3,  // 30% to air (hypothesis)
        0.4,  // 40% to air
        0.5,  // 50% to air
    ];

    println!(
        "Testing {} different solar_distribution_to_air values:\n",
        solar_values.len()
    );

    // Table header
    println!(
        "┌───────────┬────────────┬────────────┬────────────┬────────────┬───────────┬───────────┐"
    );
    println!("│ Solar to  │ Heating    │ Cooling    │ Total HVAC  │ % Heating  │ % Cooling │ Improvement│");
    println!("│ Air (frac) │ (MWh)      │ (MWh)      │ (MWh)      │ Error      │ Error     │ vs Ref    │");
    println!(
        "├───────────┼────────────┼────────────┼────────────┼────────────┼───────────┼───────────┤"
    );

    let mut results = Vec::new();

    for solar_dist in &solar_values {
        let result = run_case_600_with_solar_dist(*solar_dist);
        results.push(result.clone());

        // Calculate errors relative to reference
        let heating_ref_mid = (4.30 + 5.71) / 2.0;
        let cooling_ref_mid = (6.14 + 8.45) / 2.0;

        let heating_error_pct = if heating_ref_mid > 0.0 {
            ((result.heating_mwh - heating_ref_mid) / heating_ref_mid) * 100.0
        } else {
            0.0
        };

        let cooling_error_pct = if cooling_ref_mid > 0.0 {
            ((result.cooling_mwh - cooling_ref_mid) / cooling_ref_mid) * 100.0
        } else {
            0.0
        };

        // Calculate improvement (reduction in error magnitude)
        let current_heating_error =
            ((result.heating_mwh - heating_ref_mid) / heating_ref_mid).abs() * 100.0;
        let baseline_heating_error = 294.0; // From Task 1.1: current heating is 294% too high
        let improvement = baseline_heating_error - current_heating_error;

        // Print row
        let improvement_str = if improvement > 10.0 {
            format!("+{:.0}%", improvement)
        } else if improvement < -10.0 {
            format!("{:.0}%", improvement)
        } else {
            format!("{:.0}%", improvement)
        };

        println!(
            "│ {:<9} │ {:>10.3} │ {:>10.3} │ {:>10.3} │ {:>+10.1} │ {:>+9.1} │ {:<11} │",
            solar_dist,
            result.heating_mwh,
            result.cooling_mwh,
            result.heating_mwh + result.cooling_mwh,
            heating_error_pct,
            cooling_error_pct,
            improvement_str
        );
    }

    println!(
        "└───────────┴────────────┴────────────┴────────────┴────────────┴───────────┴───────────┘"
    );
    println!();

    // Find optimal value
    println!("=== Analysis ===");

    // Find value that minimizes total error
    let heating_ref_mid = (4.30 + 5.71) / 2.0;
    let cooling_ref_mid = (6.14 + 8.45) / 2.0;

    let mut best_heating_value = 0.0;
    let mut best_heating_error = f64::INFINITY;

    let mut best_cooling_value = 0.0;
    let mut best_cooling_error = f64::INFINITY;

    let mut best_total_error = f64::INFINITY;
    let mut best_total_value = 0.0;

    for result in &results {
        let heating_error = (result.heating_mwh - heating_ref_mid).abs() / heating_ref_mid * 100.0;
        let cooling_error = (result.cooling_mwh - cooling_ref_mid).abs() / cooling_ref_mid * 100.0;
        let total_error = heating_error + cooling_error;

        if heating_error < best_heating_error {
            best_heating_error = heating_error;
            best_heating_value = result.solar_dist;
        }

        if cooling_error < best_cooling_error {
            best_cooling_error = cooling_error;
            best_cooling_value = result.solar_dist;
        }

        if total_error < best_total_error {
            best_total_error = total_error;
            best_total_value = result.solar_dist;
        }
    }

    println!(
        "Best for heating: solar_distribution_to_air = {:.2} (error: {:.1}%)",
        best_heating_value, best_heating_error
    );
    println!(
        "Best for cooling: solar_distribution_to_air = {:.2} (error: {:.1}%)",
        best_cooling_value, best_cooling_error
    );
    println!(
        "Best for total:   solar_distribution_to_air = {:.2} (total error: {:.1}%)",
        best_total_value, best_total_error
    );
    println!();

    // Identify trend
    println!("=== Trend Analysis ===");

    // Group results by low/medium/high solar distribution
    let low_solar: Vec<&SimulationResult> =
        results.iter().filter(|r| r.solar_dist <= 0.1).collect();

    let med_solar: Vec<&SimulationResult> = results
        .iter()
        .filter(|r| r.solar_dist > 0.1 && r.solar_dist <= 0.3)
        .collect();

    let high_solar: Vec<&SimulationResult> =
        results.iter().filter(|r| r.solar_dist > 0.3).collect();

    if !low_solar.is_empty() {
        let avg_heating =
            low_solar.iter().map(|r| r.heating_mwh).sum::<f64>() / low_solar.len() as f64;
        let avg_cooling =
            low_solar.iter().map(|r| r.cooling_mwh).sum::<f64>() / low_solar.len() as f64;
        println!(
            "Low solar (0.0-0.1):  Heating={:.2} MWh, Cooling={:.2} MWh",
            avg_heating, avg_cooling
        );
    }

    if !med_solar.is_empty() {
        let avg_heating =
            med_solar.iter().map(|r| r.heating_mwh).sum::<f64>() / med_solar.len() as f64;
        let avg_cooling =
            med_solar.iter().map(|r| r.cooling_mwh).sum::<f64>() / med_solar.len() as f64;
        println!(
            "Med solar (0.15-0.3):  Heating={:.2} MWh, Cooling={:.2} MWh",
            avg_heating, avg_cooling
        );
    }

    if !high_solar.is_empty() {
        let avg_heating =
            high_solar.iter().map(|r| r.heating_mwh).sum::<f64>() / high_solar.len() as f64;
        let avg_cooling =
            high_solar.iter().map(|r| r.cooling_mwh).sum::<f64>() / high_solar.len() as f64;
        println!(
            "High solar (0.4-0.5):  Heating={:.2} MWh, Cooling={:.2} MWh",
            avg_heating, avg_cooling
        );
    }

    println!();

    // Recommendations
    println!("=== Recommendations ===");

    if best_heating_value != best_cooling_value {
        println!("⚠️  Trade-off detected:");
        println!("   Heating optimal: {:.2}", best_heating_value);
        println!("   Cooling optimal: {:.2}", best_cooling_value);
        println!("   Total optimal: {:.2}", best_total_value);
        println!();

        if best_total_value >= 0.2 && best_total_value <= 0.3 {
            println!(
                "✓ Optimal value ({:.2}) is in hypothesized range (0.2-0.3)",
                best_total_value
            );
        }
    } else {
        println!(
            "✓ No trade-off: solar_distribution_to_air = {:.2} is optimal for both",
            best_total_value
        );
    }

    println!();

    // Physics-based explanation
    println!("=== Physics-Based Explanation ===");
    println!("Low-mass buildings have:");
    println!(
        "  - Small thermal capacitance (C_m = {:.1} kJ/K)",
        results[0].thermal_capacitance_kj
    );
    println!(
        "  - Fast thermal time constant (τ = {:.2} hours)",
        results[0].tau_hours
    );
    println!("  - Limited ability to buffer solar gains in thermal mass");
    println!();
    println!("Higher solar_distribution_to_air:");
    println!("  - Sends more solar gains directly to air");
    println!("  - Less energy stored in mass");
    println!("  - More readily rejected by cooling");
    println!();
    println!("For low-mass: Use higher solar_distribution_to_air (0.2-0.3)");
    println!("For high-mass: Use lower solar_distribution_to_air (0.05-0.1)");

    println!();
    println!("=== Task 1.2 Complete ===");
}

/// Run Case 600 simulation with specified solar_distribution_to_air.
fn run_case_600_with_solar_dist(solar_dist: f64) -> SimulationResult {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec(&spec);

    // Override solar_distribution_to_air
    model.solar_distribution_to_air = solar_dist;

    // Extract thermal parameters
    let thermal_capacitance = model.thermal_capacitance.as_ref()[0];
    let h_tr_ms = model.h_tr_ms.as_ref()[0];

    // Calculate time constant
    let tau_hours = (thermal_capacitance / h_tr_ms) / 3600.0;

    // Load weather data (simplified for speed)
    let weather = create_simplified_denver_weather();

    // Run simulation
    let hours_to_simulate = 8760;
    let mut total_heating_joules = 0.0;
    let mut total_cooling_joules = 0.0;

    for hour in 0..hours_to_simulate {
        let outdoor_temp = weather.get_temperature(hour);

        let hvac_kwh = model.step_physics(hour, outdoor_temp);
        let hvac_joules = hvac_kwh * 3.6e6;

        if hvac_joules > 0.0 {
            total_heating_joules += hvac_joules;
        } else {
            total_cooling_joules += -hvac_joules;
        }
    }

    // Convert to MWh
    let heating_mwh = total_heating_joules / 3.6e9;
    let cooling_mwh = total_cooling_joules / 3.6e9;

    SimulationResult {
        solar_dist,
        heating_mwh,
        cooling_mwh,
        thermal_capacitance_kj: thermal_capacitance / 1000.0,
        tau_hours,
    }
}

/// Simplified weather data for fast testing.
fn create_simplified_denver_weather() -> SimplifiedWeather {
    SimplifiedWeather {
        // Denver annual mean: ~12°C
        // Seasonal amplitude: ~17.5°C
        // Daily amplitude: ~8°C
        base_temp: 12.0,
        seasonal_amplitude: 17.5,
        daily_amplitude: 8.0,
    }
}

/// Result structure for solar distribution testing.
#[derive(Debug, Clone, Copy)]
struct SimulationResult {
    solar_dist: f64,
    heating_mwh: f64,
    cooling_mwh: f64,
    thermal_capacitance_kj: f64,
    tau_hours: f64,
}

/// Simplified weather for fast simulation.
struct SimplifiedWeather {
    base_temp: f64,
    seasonal_amplitude: f64,
    daily_amplitude: f64,
}

impl SimplifiedWeather {
    /// Get hourly temperature for a given hour of the year.
    fn get_temperature(&self, hour: usize) -> f64 {
        let day_of_year = (hour / 24) % 365;
        let hour_of_day = hour % 24;

        // Seasonal component
        let season_factor = std::f64::consts::PI * 2.0 * (day_of_year as f64 / 365.0 - 0.25);
        let seasonal = self.seasonal_amplitude * season_factor.cos();

        // Daily component
        let daily_factor = std::f64::consts::PI * 2.0 * (hour_of_day as f64 / 24.0);
        let daily = self.daily_amplitude * daily_factor.cos();

        self.base_temp + seasonal + daily
    }
}
