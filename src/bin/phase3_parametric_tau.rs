//! Phase 3 Recommendation 1: Parametric τ study for VeryLight mass class
//!
//! Test different target τ values to find optimal balance between
//! heating and cooling errors for 600-series cases.

use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

fn main() {
    println!("=== Phase 3: Parametric τ Study for VeryLight Mass ===");
    println!("Testing Case 600 with different target τ values\n");

    // Case 600 reference
    println!("=== ASHRAE 140 Case 600 Reference ===");
    println!("Heating: 4.30-5.71 MWh (midpoint: 5.01 MWh)");
    println!("Cooling: 6.14-8.45 MWh (midpoint: 7.30 MWh)");
    println!();

    // Test τ values around and beyond current 6.0 hours
    let tau_values = vec![
        3.0,  // Below current (faster response)
        4.0,  // Below current
        5.0,  // Below current
        6.0,  // Current optimal from Phase 1
        7.0,  // Above current (slower response)
        8.0,  // Above current
        10.0, // Significantly above (slower)
    ];

    println!("┌───────┬─────────────┬────────────────┬────────────┬──────────┬──────────┐");
    println!("│ τ (h)  │ h_tr_ms (W/K) │ Heating (MWh) │ H % Error │ Cool MWh │ C % Err │");
    println!("├───────┼─────────────┼────────────────┼────────────┼──────────┼──────────┤");

    let mut results = Vec::new();

    for tau_hours in &tau_values {
        let result = test_case_600_with_tau(*tau_hours);
        print_tau_result(*tau_hours, &result);
        results.push((*tau_hours, result));
    }

    println!("└───────┴─────────────┴────────────────┴────────────┴──────────┴──────────┘");
    println!();

    // Analysis
    println!("=== Analysis ===");

    let heating_ref_mid = (4.30 + 5.71) / 2.0;
    let cooling_ref_mid = (6.14 + 8.45) / 2.0;

    // Find optimal for heating
    let mut best_heating_tau = 0.0;
    let mut best_heating_error = f64::INFINITY;

    // Find optimal for cooling
    let mut best_cooling_tau = 0.0;
    let mut best_cooling_error = f64::INFINITY;

    // Find optimal for balanced
    let mut best_balanced_tau = 0.0;
    let mut best_balanced_error = f64::INFINITY;

    for (tau_hours, result) in &results {
        let heating_error =
            ((result.heating_mwh - heating_ref_mid) / heating_ref_mid).abs() * 100.0;
        let cooling_error =
            ((result.cooling_mwh - cooling_ref_mid) / cooling_ref_mid).abs() * 100.0;
        let balanced_error = heating_error.abs() + cooling_error.abs();

        if heating_error < best_heating_error {
            best_heating_error = heating_error;
            best_heating_tau = *tau_hours;
        }

        if cooling_error < best_cooling_error {
            best_cooling_error = cooling_error;
            best_cooling_tau = *tau_hours;
        }

        if balanced_error < best_balanced_error {
            best_balanced_error = balanced_error;
            best_balanced_tau = *tau_hours;
        }
    }

    println!(
        "Best τ for heating: {:.1} hours ({:.1}% error)",
        best_heating_tau, best_heating_error
    );
    println!(
        "Best τ for cooling: {:.1} hours ({:.1}% error)",
        best_cooling_tau, best_cooling_error
    );
    println!(
        "Best τ for balanced: {:.1} hours ({:.1}% total error)",
        best_balanced_tau, best_balanced_error
    );
    println!();

    // Trade-off analysis
    println!("=== Trade-off Analysis ===");
    println!("Current configuration: τ = 6.0 hours (from Phase 1)");
    println!();

    // Find results
    let current_opt = results.iter().find(|(t, _)| *t == 6.0).unwrap();
    let heating_err_6h = ((current_opt.1.heating_mwh - heating_ref_mid) / heating_ref_mid) * 100.0;
    let cooling_err_6h = ((current_opt.1.cooling_mwh - cooling_ref_mid) / cooling_ref_mid) * 100.0;

    println!("Current (τ=6.0h):");
    println!(
        "  Heating: {:.3} MWh ({:.1}% error)",
        current_opt.1.heating_mwh, heating_err_6h
    );
    println!(
        "  Cooling: {:.3} MWh ({:.1}% error)",
        current_opt.1.cooling_mwh, cooling_err_6h
    );
    println!();

    // Compare with best balanced
    let balanced_opt = results
        .iter()
        .find(|(t, _)| *t == best_balanced_tau)
        .unwrap();
    let heating_err_best =
        ((balanced_opt.1.heating_mwh - heating_ref_mid) / heating_ref_mid) * 100.0;
    let cooling_err_best =
        ((balanced_opt.1.cooling_mwh - cooling_ref_mid) / cooling_ref_mid) * 100.0;

    println!("Best balanced (τ={:.1}h):", best_balanced_tau);
    println!(
        "  Heating: {:.3} MWh ({:.1}% error)",
        balanced_opt.1.heating_mwh, heating_err_best
    );
    println!(
        "  Cooling: {:.3} MWh ({:.1}% error)",
        balanced_opt.1.cooling_mwh, cooling_err_best
    );
    println!();

    let heating_improvement = heating_err_6h - heating_err_best;
    let cooling_improvement = cooling_err_6h - cooling_err_best;

    println!("Improvement:");
    if heating_improvement > 0.0 {
        println!("  Heating: +{:.1}% improvement", heating_improvement);
    } else if heating_improvement < 0.0 {
        println!("  Heating: {:.1}% worse", heating_improvement);
    } else {
        println!("  Heating: No change");
    }

    if cooling_improvement > 0.0 {
        println!("  Cooling: +{:.1}% improvement", cooling_improvement);
    } else if cooling_improvement < 0.0 {
        println!("  Cooling: {:.1}% worse", cooling_improvement);
    } else {
        println!("  Cooling: No change");
    }

    println!();
    println!("=== Physics Interpretation ===");
    println!("τ (thermal time constant) determines how fast thermal mass responds:");
    println!("  - Lower τ = faster response = less heat storage = more HVAC energy");
    println!("  - Higher τ = slower response = more heat storage = less HVAC energy");
    println!();
    println!("For Case 600 (VeryLight mass):");
    println!("  - Current τ=6.0h: Good cooling (-8%), but heating +80%");
    println!("  - Suggests: τ may need adjustment for better heating/cooling balance");
    println!();
    println!("Trade-off:");
    println!("  - Too low τ → Cooling passes, heating fails (current)");
    println!("  - Too high τ → Heating passes, cooling fails");
    println!("  - Optimal τ → Both pass (goal)");

    println!();
    println!("=== Phase 3 Recommendation 1 Complete ===");
}

/// Test Case 600 with specific τ value.
fn test_case_600_with_tau(tau_hours: f64) -> SimulationResult {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec(&spec);

    // Override h_tr_ms to use target τ
    let thermal_capacitance = model.thermal_capacitance.as_ref()[0];
    let target_tau_seconds = tau_hours * 3600.0;
    let h_tr_ms_override = thermal_capacitance / target_tau_seconds;

    // Override h_tr_ms using from_scalar
    model.h_tr_ms = fluxion::physics::cta::VectorField::from_scalar(h_tr_ms_override, 1);

    // Weather
    let weather = DenverTmyWeather::new();

    // Run simulation
    let hours_to_simulate = 8760;
    let mut total_heating_joules = 0.0;
    let mut total_cooling_joules = 0.0;

    for hour in 0..hours_to_simulate {
        let weather_data = weather.get_hourly_data(hour).unwrap();

        // Simple solar model using actual weather
        let floor_area = spec.geometry[0].floor_area();
        let solar_gain = weather_data.dni * floor_area * 0.8 * 0.5; // Simplified

        // Set loads
        let internal_gains = spec
            .internal_loads
            .get(0)
            .or(spec.internal_loads.first())
            .and_then(|l| l.as_ref())
            .map_or(200.0, |l| l.total_load);
        let internal_loads = vec![internal_gains];
        let solar_loads = vec![solar_gain];

        model.set_loads(&internal_loads);
        model.set_solar_loads(&solar_loads);

        // Step physics
        let hvac_kwh = model.step_physics(hour, weather_data.dry_bulb_temp);

        // Track energy
        if hvac_kwh > 0.0 {
            total_heating_joules += hvac_kwh * 3.6e6;
        } else {
            total_cooling_joules += (-hvac_kwh) * 3.6e6;
        }
    }

    SimulationResult {
        heating_mwh: total_heating_joules / 3.6e9,
        cooling_mwh: total_cooling_joules / 3.6e9,
        h_tr_ms: h_tr_ms_override,
    }
}

/// Print τ test result in table format.
fn print_tau_result(tau_hours: f64, result: &SimulationResult) {
    let heating_ref_mid = (4.30 + 5.71) / 2.0;
    let cooling_ref_mid = (6.14 + 8.45) / 2.0;

    let heating_error = ((result.heating_mwh - heating_ref_mid) / heating_ref_mid) * 100.0;
    let cooling_error = ((result.cooling_mwh - cooling_ref_mid) / cooling_ref_mid) * 100.0;

    println!(
        "│ {:>5.1} │ {:>13.2} │ {:>14.3} │ {:>+10.1} │ {:>8.3} │ {:>+8.1} │",
        tau_hours,
        result.h_tr_ms,
        result.heating_mwh,
        heating_error,
        result.cooling_mwh,
        cooling_error
    );
}

/// Simulation result.
#[derive(Debug, Clone, Copy)]
struct SimulationResult {
    heating_mwh: f64,
    cooling_mwh: f64,
    h_tr_ms: f64,
}
