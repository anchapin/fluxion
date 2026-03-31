//! Diagnostic test for excessive heating in ASHRAE 140 cases
//!
//! This test helps diagnose why annual heating is 3-13x overpredicted.

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::{denver::DenverTmyWeather, WeatherSource};

/// Test: Diagnose heating calculation for Case 600
#[test]
fn test_heating_diagnostic_case_600() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    // Track heating/cooling for first 1000 timesteps
    let mut heating_hours = 0;
    let mut cooling_hours = 0;
    let mut total_heating_watts = 0.0;
    let mut total_cooling_watts = 0.0;
    let mut max_heating_watts: f64 = 0.0;

    // Get reference data
    let heating_setpoint = 20.0;
    let cooling_setpoint = 27.0;

    println!("\n=== Case 600 Heating Diagnostic (First 1000 timesteps) ===");
    println!(
        "Heating setpoint: {}°C, Cooling setpoint: {}°C",
        heating_setpoint, cooling_setpoint
    );
    println!("Window U-value: {:.2} W/m²K", model.window_u_value);
    println!("h_tr_em: {:.2} W/K", model.h_tr_em.as_ref()[0]);
    println!("h_tr_w: {:.2} W/K", model.h_tr_w.as_ref()[0]);
    println!("h_ve: {:.2} W/K", model.h_ve.as_ref()[0]);
    println!(
        "Solar distribution to air: {:.2}",
        model.solar_distribution_to_air
    );
    println!("Convective fraction: {:.2}", model.convective_fraction);
    println!();

    // Run first 1000 timesteps and track patterns
    for timestep in 0..1000 {
        let surrogates = SurrogateManager::new().expect("Failed to create surrogate manager");

        // Get outdoor temp for this hour
        let outdoor_temp = weather.get_hourly_data(timestep).unwrap().dry_bulb_temp;

        // Get current indoor temp before step
        let ti_before = model.temperatures.as_ref()[0];
        let ti_free_before = {
            // Calculate Ti_free manually for diagnostic
            let h_ext_base = &model.derived_h_ext;
            let term_rest_1 = &model.derived_term_rest_1;
            let den = &model.derived_den;

            // Get current values
            let tm = model.envelope_mass_temperatures.as_ref()[0];
            let num_tm = model.derived_h_ms_is_prod.as_ref()[0] * tm;

            // Simplified: no internal loads for this diagnostic
            let num_phi_st = 0.0;
            let num_rest = term_rest_1.as_ref()[0] * h_ext_base.as_ref()[0] * outdoor_temp;

            let ti_free = (num_tm + num_phi_st + num_rest) / den.as_ref()[0];
            ti_free
        };

        // Run one timestep
        let _eui = model.solve_timesteps(1, &surrogates, false, None, None, None);

        // Get indoor temp after step
        let ti_after = model.temperatures.as_ref()[0];

        // Calculate HVAC demand (simplified)
        let hvac_demand = if ti_free_before < heating_setpoint {
            heating_setpoint - ti_free_before
        } else if ti_free_before > cooling_setpoint {
            -(ti_free_before - cooling_setpoint) // Negative for cooling
        } else {
            0.0
        };

        // Track heating/cooling
        if hvac_demand > 0.0 {
            heating_hours += 1;
            total_heating_watts += hvac_demand;
            max_heating_watts = max_heating_watts.max(hvac_demand);
        } else if hvac_demand < 0.0 {
            cooling_hours += 1;
            total_cooling_watts += -hvac_demand;
        }

        // Print sample data every 100 timesteps
        if timestep % 100 == 0 {
            println!("Timestep {:4}: Outdoor={:6.1}°C, Ti_before={:6.1}°C, Ti_after={:6.1}°C, Ti_free={:6.1}°C, HVAC={:7.1}W",
                timestep, outdoor_temp, ti_before, ti_after, ti_free_before, hvac_demand);
        }
    }

    println!("\n=== Summary (First 1000 timesteps) ===");
    println!(
        "Heating hours: {} ({:.1}%)",
        heating_hours,
        heating_hours as f64 / 10.0
    );
    println!(
        "Cooling hours: {} ({:.1}%)",
        cooling_hours,
        cooling_hours as f64 / 10.0
    );
    println!(
        "Avg heating power: {:.2} W",
        total_heating_watts / heating_hours as f64
    );
    println!(
        "Avg cooling power: {:.2} W",
        total_cooling_watts / cooling_hours as f64
    );
    println!("Max heating power: {:.2} W", max_heating_watts);
    println!();

    // Run full year to compare with reference
    let mut model_full = ThermalModel::from_spec(&spec);
    let surrogates = SurrogateManager::new().expect("Failed to create surrogate manager");
    let _eui = model_full.solve_timesteps(8760, &surrogates, false, None, None, None);

    println!("=== Full Year Results ===");
    println!(
        "Annual heating: {:.2} MWh (Ref: 5.50-7.50 MWh)",
        model_full.annual_heating_energy
    );
    println!(
        "Annual cooling: {:.2} MWh (Ref: 8.00-10.50 MWh)",
        model_full.annual_cooling_energy
    );
    println!(
        "Peak heating: {:.2} kW",
        model_full.get_peak_heating_power_kw()
    );
    println!(
        "Peak cooling: {:.2} kW",
        model_full.get_peak_cooling_power_kw()
    );
}
