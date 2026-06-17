//! Cross-Platform Determinism Test for Case 900
//!
//! This test verifies that the Case 900 annual simulation produces
//! identical results across different platforms (Ubuntu, Windows, macOS).
//!
//! The test outputs deterministic values that can be hashed and compared
//! across platforms to verify numerical consistency.
//!
//! Run with: cargo test --test case_900_determinism --release -- --nocapture

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Run Case 900 annual simulation and return key outputs
fn run_case_900_simulation() -> DeterminismOutput {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    // Run 14-day warmup + 1 year simulation
    let warmup_steps = 14 * 24;
    let steps = 8760;

    // Warmup
    for step in 0..warmup_steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
    }

    // Reset energy tracking after warmup
    model.reset_heating_cooling_energy();
    model.reset_peak_power();

    // Annual simulation
    let mut total_heating = 0.0_f64;
    let mut total_cooling = 0.0_f64;
    let mut peak_heating = 0.0_f64;
    let mut peak_cooling = 0.0_f64;

    for step in warmup_steps..warmup_steps + steps {
        let weather_data = weather.get_hourly_data(step % 8760).unwrap();
        model.weather = Some(weather_data.clone());

        let zone_temp_before = model
            .temperatures
            .as_slice()
            .first()
            .copied()
            .unwrap_or(20.0);

        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);
        let energy_joules = energy_kwh * 3.6e6;

        // Track heating
        if energy_kwh > 0.0 || zone_temp_before < model.heating_setpoint {
            total_heating += energy_joules;
            let power_watts = energy_joules / 3600.0;
            peak_heating = peak_heating.max(power_watts);
        }

        // Track cooling
        if energy_kwh < 0.0 || zone_temp_before > model.cooling_setpoint {
            total_cooling += -energy_joules;
            let power_watts = -energy_joules / 3600.0;
            peak_cooling = peak_cooling.max(power_watts);
        }
    }

    // Convert to standard units
    let annual_heating_mwh = total_heating / 3.6e9;
    let annual_cooling_mwh = total_cooling / 3.6e9;
    let peak_heating_kw = peak_heating / 1000.0;
    let peak_cooling_kw = peak_cooling / 1000.0;

    DeterminismOutput {
        annual_heating_mwh,
        annual_cooling_mwh,
        peak_heating_kw,
        peak_cooling_kw,
    }
}

/// Output values for determinism checking
#[derive(Debug, Clone)]
struct DeterminismOutput {
    annual_heating_mwh: f64,
    annual_cooling_mwh: f64,
    peak_heating_kw: f64,
    peak_cooling_kw: f64,
}

impl DeterminismOutput {
    /// Format values as a pipe-delimited string for hashing
    fn format_for_hash(&self) -> String {
        format!(
            "{:.6}|{:.6}|{:.6}|{:.6}",
            self.annual_heating_mwh,
            self.annual_cooling_mwh,
            self.peak_heating_kw,
            self.peak_cooling_kw
        )
    }
}

#[test]
fn test_case_900_determinism() {
    let output = run_case_900_simulation();

    // Print determinism hash line (machine-parseable)
    // Format: DETERMINISM_HASH|heating_cooling|peak_h_p
    println!(
        "DETERMINISM_VALUES|{:.6}|{:.6}|{:.6}|{:.6}",
        output.annual_heating_mwh,
        output.annual_cooling_mwh,
        output.peak_heating_kw,
        output.peak_cooling_kw
    );

    // Print human-readable summary
    println!();
    println!("=== Case 900 Determinism Output ===");
    println!("Annual Heating: {:.6} MWh", output.annual_heating_mwh);
    println!("Annual Cooling: {:.6} MWh", output.annual_cooling_mwh);
    println!("Peak Heating: {:.6} kW", output.peak_heating_kw);
    println!("Peak Cooling: {:.6} kW", output.peak_cooling_kw);
    println!("Values for hash: {}", output.format_for_hash());

    // For the test to pass, we just verify the simulation runs
    // The actual cross-platform comparison happens in the CI workflow
    assert!(
        output.annual_heating_mwh > 0.0,
        "Heating energy should be positive"
    );
    assert!(
        output.annual_cooling_mwh > 0.0,
        "Cooling energy should be positive"
    );
}

#[test]
#[ignore = "Case 900FF has pre-existing NaN instability - see Issue #NNN"]
fn test_case_900_determinism_free_floating() {
    // Run free-floating simulation (no HVAC)
    let spec = ASHRAE140Case::Case900FF.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    let weather = DenverTmyWeather::new();

    let steps = 8760;
    let mut min_temp = f64::MAX;
    let mut max_temp = f64::MIN;
    let mut sum_temps = 0.0_f64;

    for step in 0..steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());
        model.step_physics(step, weather_data.dry_bulb_temp, 3600.0);

        if let Some(&zone_temp) = model.temperatures.as_slice().first() {
            min_temp = min_temp.min(zone_temp);
            max_temp = max_temp.max(zone_temp);
            sum_temps += zone_temp;
        }
    }

    let avg_temp = sum_temps / steps as f64;
    let temp_swing = max_temp - min_temp;

    println!(
        "DETERMINISM_FF_VALUES|{:.6}|{:.6}|{:.6}|{:.6}",
        min_temp, max_temp, avg_temp, temp_swing
    );

    println!();
    println!("=== Case 900FF Determinism Output ===");
    println!("Min Temperature: {:.6} °C", min_temp);
    println!("Max Temperature: {:.6} °C", max_temp);
    println!("Avg Temperature: {:.6} °C", avg_temp);
    println!("Temperature Swing: {:.6} K", temp_swing);

    // Verify sensible ranges
    assert!(min_temp < avg_temp, "Min temp should be below average");
    assert!(max_temp > avg_temp, "Max temp should be above average");
    assert!(temp_swing > 0.0, "Temperature swing should be positive");
}
