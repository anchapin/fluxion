//! Quick check of Case 900 cooling energy after thermal mass correction fix

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

#[test]
fn test_case_900_cooling_with_thermal_mass_correction() {
    println!("\n=== Case 900 Cooling Energy Check (Phase 30 Wave 2 Fix) ===\n");

    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    println!(
        "Correction factor applied: {}",
        model.time_constant_sensitivity_correction
    );
    println!("Expected: 4.0 (symmetric thermal mass correction)");

    // Run 365 days (8760 hours)
    // Removed unused variables

    for step in 0..8760 {
        if let Ok(weather_data) = weather.get_hourly_data(step) {
            model.weather = Some(weather_data.clone());
            let _energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

            // Log progress every 24 hours
            if step % 24 == 0 {
                println!(
                    "Day {}: H={:.2} MWh, C={:.2} MWh",
                    step / 24,
                    model.annual_heating_energy / 1000.0,
                    model.annual_cooling_energy / 1000.0
                );
            }
        }
    }

    let heating_total_mwh = model.annual_heating_energy / 1000.0;
    let cooling_total_mwh = model.annual_cooling_energy / 1000.0;

    println!("\n=== Results ===");
    println!("Total heating: {:.2} MWh", heating_total_mwh);
    println!("Total cooling: {:.2} MWh", cooling_total_mwh);
    println!("\n=== Phase 29 Results (before fix) ===");
    println!("Heating: 5.90 MWh (with 4.0x correction applied)");
    println!("Cooling: 6.13 MWh (NO correction - this is the bug!)");
    println!("\n=== Expected (within ASHRAE 140 ranges) ===");
    println!("Heating: ~1.5 MWh (within [1.17, 2.04] MWh target)");
    println!("Cooling: ~9.0 MWh (within [8.00, 10.50] MWh target)");
    println!("\n=== Target Ranges (ASHRAE 140) ===");
    println!("Heating target: [1.17, 2.04] MWh");
    println!("Cooling target: [8.00, 10.50] MWh");

    println!("\n=== Analysis ===");
    if heating_total_mwh >= 1.17 - 0.5 && heating_total_mwh <= 2.04 + 0.5 {
        println!("✅ Heating within tolerance");
    } else {
        println!("❌ Heating out of tolerance!");
    }

    if cooling_total_mwh >= 8.00 - 0.5 && cooling_total_mwh <= 10.50 + 0.5 {
        println!("✅ Cooling within tolerance");
    } else {
        println!("❌ Cooling out of tolerance!");
    }
}
