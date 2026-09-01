//! High-resolution diagnostic test for Case 900 peak loads.
//!
//! Objective: Identify root cause of peak load overestimation in high-mass buildings.
//! Exports hourly internal state variables to CSV for direct comparison with EnergyPlus.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::validation::diagnostics::SimulationDiagnostics;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;
use std::path::Path;

#[test]
fn test_case_900_peak_diagnostic() {
    // 1. Load Case 900 specification
    let case_900 = ASHRAE140Case::Case900;
    let spec = case_900.spec();

    // 2. Load weather (ASHRAE 140 reference weather - Denver TMY)
    let weather = DenverTmyWeather::new();

    // 3. Build thermal model
    let mut model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("default selector must initialize");

    // 4. Attach diagnostics
    let diag = SimulationDiagnostics::new(model.hvac.num_zones, 8760);
    model.set_diagnostics(Some(diag));

    // 5. Run full year simulation
    println!("Running Case 900 simulation (8760 hours)...");

    let num_zones = model.hvac.num_zones;

    for step in 0..8760 {
        let weather_data = weather.get_hourly_data(step).unwrap();

        // Update weather data on model for solar gain calculation
        model.solar.weather = Some(weather_data.clone());

        // Match ashrae_140_validator logic:
        // Set dynamic setpoints from spec (handles setback cases)
        if let Some(hvac_schedule) = spec.hvac.first() {
            let hour = (step % 24) as u8;
            let heating_sp = hvac_schedule
                .heating_setpoint_at_hour(hour)
                .unwrap_or(hvac_schedule.heating_setpoint);
            let cooling_sp = model.setpoints.cooling_schedule.value(hour as usize);
            model.setpoints.heating_setpoint = heating_sp;
            model.setpoints.cooling_setpoint = cooling_sp;

            if spec.hvac.len() > 1 {
                let mut heating_sps = vec![heating_sp; num_zones];
                let mut cooling_sps = vec![cooling_sp; num_zones];
                for (zone_idx, hvac) in spec.hvac.iter().enumerate() {
                    if zone_idx < num_zones {
                        let h_sp = hvac
                            .heating_setpoint_at_hour(hour)
                            .unwrap_or(hvac.heating_setpoint);
                        let c_sp = model.setpoints.cooling_schedule.value(hour as usize);
                        heating_sps[zone_idx] = h_sp;
                        cooling_sps[zone_idx] = c_sp;
                    }
                }
                model.setpoints.heating_setpoints = VectorField::new(heating_sps);
                model.setpoints.cooling_setpoints = VectorField::new(cooling_sps);
            }
        }

        // Set internal loads
        let mut internal_loads: Vec<f64> = Vec::with_capacity(num_zones);
        for zone_idx in 0..num_zones {
            let internal_gains = spec
                .internal_loads
                .get(zone_idx)
                .or(spec.internal_loads.first())
                .and_then(|l| l.as_ref())
                .map_or(0.0, |l| l.total_load);

            let floor_area = spec
                .geometry
                .get(zone_idx)
                .or(spec.geometry.first())
                .map_or(20.0, |g| g.floor_area());

            internal_loads.push(internal_gains / floor_area);
        }
        model.set_loads(&internal_loads);

        // Step physics
        let _hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    // 6. Export diagnostics to CSV
    let csv_path = "case_900_peak_hourly.csv";
    let diag = model
        .get_diagnostics()
        .expect("Diagnostics should be attached");
    diag.export_csv(csv_path)
        .expect("Should export CSV successfully");

    println!("Exported diagnostics to {}", csv_path);

    // 7. Report peak values
    let mut max_heating = 0.0;
    let mut max_cooling = 0.0;
    let mut peak_heating_hour = 0;
    let mut peak_cooling_hour = 0;

    for i in 0..diag.hours.len() {
        let hvac = diag.loads.hvac[i][0]; // Zone 0
        if hvac > max_heating {
            max_heating = hvac;
            peak_heating_hour = diag.hours[i];
        }
        if hvac < max_cooling {
            max_cooling = hvac;
            peak_cooling_hour = diag.hours[i];
        }
    }

    println!(
        "Peak Heating: {:.2} W at hour {}",
        max_heating, peak_heating_hour
    );
    println!(
        "Peak Cooling: {:.2} W at hour {}",
        -max_cooling, peak_cooling_hour
    );

    // Success criteria: CSV exists
    assert!(Path::new(csv_path).exists());
}
