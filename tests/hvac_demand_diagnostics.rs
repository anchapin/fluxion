//! HVAC demand calculation diagnostic tests for Plan 03-07
//!
//! This test analyzes hvac_power_demand calculation and solar gain distribution
//! to identify causes of annual energy over-prediction for Case 900.
//!
//! Current issues:
//! - Annual heating: 6.91 MWh vs [1.17, 2.04] MWh (239-491% over)
//! - Annual cooling: 4.68 MWh vs [2.13, 3.67] MWh (27-120% over)
//! - Peak loads are correct (heating: 2.10 kW, cooling: 3.54 kW)
//! - Mass energy change: -159.4 MWh (mass releasing energy)
//! - Total solar gain: 15.50 MWh
//!
//! Root cause hypothesis: Too much solar energy is being absorbed by thermal mass
//! (solar_beam_to_mass_fraction = 0.7), which then releases energy slowly,
//! causing HVAC to work against mass heating/cooling.

use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::weather::WeatherSource;

/// Task 1: Analyze HVAC demand calculation behavior
///
/// This test simulates Case 900 for a full year and analyzes:
/// 1. How often HVAC runs when within deadband
/// 2. Distribution of heating vs cooling demand
/// 3. Relationship between demand magnitude and temperature difference
#[test]
fn test_case_900_hvac_demand_analysis() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model =
        fluxion::sim::engine::ThermalModel::<fluxion::physics::cta::VectorField>::from_spec(&spec);
    let weather = fluxion::weather::denver::DenverTmyWeather::new();

    let steps = 8760; // 1 year
    let heating_setpoint = 20.0;
    let cooling_setpoint = 27.0;

    let mut demand_in_deadband_count = 0;
    let mut heating_hours = 0;
    let mut cooling_hours = 0;
    let mut off_hours = 0;

    let mut heating_demand_sum: f64 = 0.0;
    let mut cooling_demand_sum: f64 = 0.0;
    let mut max_heating_demand: f64 = 0.0;
    let mut max_cooling_demand: f64 = 0.0;

    let mut mass_temp_below_setpoint_count = 0;
    let mut mass_temp_above_setpoint_count = 0;

    for step in 0..steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());

        // Run physics step
        let energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp);

        // Track mass temperature behavior
        if let Some(&mass_temp) = model.mass_temperatures.as_slice().first() {
            if mass_temp < heating_setpoint {
                mass_temp_below_setpoint_count += 1;
            } else if mass_temp > cooling_setpoint {
                mass_temp_above_setpoint_count += 1;
            }
        }

        // Analyze HVAC demand based on energy sign
        if energy_kwh > 0.01 {
            // Heating
            heating_hours += 1;
            let demand_w = energy_kwh * 1000.0; // kWh to W
            heating_demand_sum += demand_w;
            max_heating_demand = max_heating_demand.max(demand_w);
        } else if energy_kwh < -0.01 {
            // Cooling
            cooling_hours += 1;
            let demand_w = energy_kwh.abs() * 1000.0; // kWh to W
            cooling_demand_sum += demand_w;
            max_cooling_demand = max_cooling_demand.max(demand_w);
        } else {
            // Off
            off_hours += 1;

            // Check if demand is near zero but mass temperature is outside setpoints
            if let Some(&mass_temp) = model.mass_temperatures.as_slice().first() {
                if mass_temp < heating_setpoint || mass_temp > cooling_setpoint {
                    demand_in_deadband_count += 1;
                }
            }
        }
    }

    println!("\n=== Task 1: HVAC Demand Analysis ===");
    println!("Total hours: {}", steps);
    println!(
        "Heating hours: {} ({:.1}%)",
        heating_hours,
        heating_hours as f64 / steps as f64 * 100.0
    );
    println!(
        "Cooling hours: {} ({:.1}%)",
        cooling_hours,
        cooling_hours as f64 / steps as f64 * 100.0
    );
    println!(
        "Off hours: {} ({:.1}%)",
        off_hours,
        off_hours as f64 / steps as f64 * 100.0
    );
    println!();
    println!(
        "Average heating demand: {:.2} W",
        heating_demand_sum / heating_hours as f64
    );
    println!(
        "Average cooling demand: {:.2} W",
        cooling_demand_sum / cooling_hours as f64
    );
    println!("Max heating demand: {:.2} W", max_heating_demand);
    println!("Max cooling demand: {:.2} W", max_cooling_demand);
    println!();
    println!(
        "Mass temperature below heating setpoint: {} hours",
        mass_temp_below_setpoint_count
    );
    println!(
        "Mass temperature above cooling setpoint: {} hours",
        mass_temp_above_setpoint_count
    );
    println!(
        "HVAC demand when mass temp outside setpoints: {}",
        demand_in_deadband_count
    );

    // Diagnostics:
    // 1. If heating hours are high (>50%), heating demand is over-estimated
    // 2. If cooling hours are high (>50%), cooling demand is over-estimated
    // 3. If demand_in_deadband_count is high (>1000), HVAC is not responding to thermal mass effects
    // 4. If mass_temp_below/above_setpoint_count are both high, mass is storing too much energy

    assert!(
        heating_hours < 4000,
        "Heating hours should be <50% of year, got {}",
        heating_hours
    );
    assert!(
        cooling_hours < 4000,
        "Cooling hours should be <50% of year, got {}",
        cooling_hours
    );
}

/// Task 2: Validate solar gain distribution parameters
///
/// This test checks that solar gain distribution parameters match ASHRAE 140 specifications:
/// - solar_beam_to_mass_fraction should be 0.7 (70% to mass)
/// - solar_distribution_to_air should be 0.3 (30% to air)
#[test]
fn test_case_900_solar_gain_distribution_validation() {
    let spec = ASHRAE140Case::Case900.spec();
    let model =
        fluxion::sim::engine::ThermalModel::<fluxion::physics::cta::VectorField>::from_spec(&spec);

    println!("\n=== Task 2: Solar Gain Distribution Validation ===");
    println!(
        "solar_distribution_to_air: {:.2}",
        model.solar_distribution_to_air
    );
    println!(
        "solar_beam_to_mass_fraction: {:.2}",
        model.solar_beam_to_mass_fraction
    );

    // ASHRAE 140 specification for Case 900 (high-mass):
    // - 30% of solar gains go to air (solar_distribution_to_air = 0.3)
    // - 70% of solar gains go to thermal mass (solar_beam_to_mass_fraction = 0.7)

    let expected_air_fraction = 0.3;
    let expected_mass_fraction = 0.7;

    let air_fraction_diff = (model.solar_distribution_to_air - expected_air_fraction).abs();
    let mass_fraction_diff = (model.solar_beam_to_mass_fraction - expected_mass_fraction).abs();

    println!("Expected air fraction: {:.2}", expected_air_fraction);
    println!("Expected mass fraction: {:.2}", expected_mass_fraction);
    println!("Air fraction diff: {:.4}", air_fraction_diff);
    println!("Mass fraction diff: {:.4}", mass_fraction_diff);

    // Tolerance: ±0.01 (1%)
    assert!(
        air_fraction_diff < 0.01,
        "solar_distribution_to_air should be {:.2} ±0.01, got {:.2}",
        expected_air_fraction,
        model.solar_distribution_to_air
    );

    assert!(
        mass_fraction_diff < 0.01,
        "solar_beam_to_mass_fraction should be {:.2} ±0.01, got {:.2}",
        expected_mass_fraction,
        model.solar_beam_to_mass_fraction
    );

    // Verify that solar_beam_to_mass_fraction = 1.0 - solar_distribution_to_air
    let sum = model.solar_distribution_to_air + model.solar_beam_to_mass_fraction;
    assert!(
        (sum - 1.0).abs() < 0.01,
        "solar_distribution_to_air + solar_beam_to_mass_fraction should equal 1.0, got {:.4}",
        sum
    );
}

/// Task 3: Analyze solar gain impact on thermal mass
///
/// This test simulates Case 900 for a full year and analyzes:
/// 1. How much solar energy is absorbed by thermal mass
/// 2. How thermal mass temperature correlates with solar gains
/// 3. Whether thermal mass is acting as a heat source or sink
#[test]
fn test_case_900_solar_mass_interaction_analysis() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model =
        fluxion::sim::engine::ThermalModel::<fluxion::physics::cta::VectorField>::from_spec(&spec);
    let weather = fluxion::weather::denver::DenverTmyWeather::new();

    let steps = 8760; // 1 year

    let mut total_solar_gain = 0.0;
    let mut mass_energy_change = 0.0;

    let mut mass_min_temp = f64::MAX;
    let mut mass_max_temp = f64::MIN;

    let mut summer_mass_min_temp = f64::MAX;
    let mut summer_mass_max_temp = f64::MIN;

    let mut winter_mass_min_temp = f64::MAX;
    let mut winter_mass_max_temp = f64::MIN;

    let _prev_mass_temp = model
        .mass_temperatures
        .as_slice()
        .first()
        .copied()
        .unwrap_or(20.0);

    for step in 0..steps {
        let weather_data = weather.get_hourly_data(step).unwrap();
        model.weather = Some(weather_data.clone());

        // Track solar gains
        let solar_gain_w = model.solar_gains.as_slice().first().copied().unwrap_or(0.0);
        total_solar_gain += solar_gain_w;

        // Get previous mass temp before step
        let prev_mass_temp = model
            .mass_temperatures
            .as_slice()
            .first()
            .copied()
            .unwrap_or(20.0);

        // Run physics step
        model.step_physics(step, weather_data.dry_bulb_temp);

        // Get new mass temp
        let curr_mass_temp = model
            .mass_temperatures
            .as_slice()
            .first()
            .copied()
            .unwrap_or(20.0);

        // Track mass energy change
        let mass_temp_change = curr_mass_temp - prev_mass_temp;
        mass_energy_change += mass_temp_change.abs();

        // Track mass temperature range
        mass_min_temp = mass_min_temp.min(curr_mass_temp);
        mass_max_temp = mass_max_temp.max(curr_mass_temp);

        // Track seasonal mass temperatures
        let month = ((step / 730) % 12) + 1; // Approximate month

        if month >= 6 && month <= 8 {
            // Summer (June-August)
            summer_mass_min_temp = summer_mass_min_temp.min(curr_mass_temp);
            summer_mass_max_temp = summer_mass_max_temp.max(curr_mass_temp);
        } else if month >= 12 || month <= 2 {
            // Winter (Dec-Feb)
            winter_mass_min_temp = winter_mass_min_temp.min(curr_mass_temp);
            winter_mass_max_temp = winter_mass_max_temp.max(curr_mass_temp);
        }
    }

    // Calculate solar energy going to mass (70% of total)
    let solar_to_mass = total_solar_gain * model.solar_beam_to_mass_fraction;

    println!("\n=== Task 3: Solar-Mass Interaction Analysis ===");
    println!("Total annual solar gain: {:.2} MWh", total_solar_gain / 1e6);
    println!("Solar to mass (70%): {:.2} MWh", solar_to_mass / 1e6);
    println!(
        "Solar to air (30%): {:.2} MWh",
        (total_solar_gain * model.solar_distribution_to_air) / 1e6
    );
    println!();
    println!(
        "Mass temperature range: {:.2}°C to {:.2}°C",
        mass_min_temp, mass_max_temp
    );
    println!(
        "Summer mass temp range: {:.2}°C to {:.2}°C",
        summer_mass_min_temp, summer_mass_max_temp
    );
    println!(
        "Winter mass temp range: {:.2}°C to {:.2}°C",
        winter_mass_min_temp, winter_mass_max_temp
    );
    println!();
    println!(
        "Cumulative mass energy change: {:.2} MWh",
        mass_energy_change / 1e6
    );

    // Diagnostics:
    // 1. If solar_to_mass is high (>10 MWh), mass absorbs too much solar energy
    // 2. If mass_max_temp - mass_min_temp > 20°C, mass has large thermal swings
    // 3. If summer_mass_max > 30°C, mass is overheating in summer
    // 4. If winter_mass_min < 10°C, mass is cooling too much in winter

    assert!(
        solar_to_mass < 12e6, // <12 MWh
        "Solar to mass should be <12 MWh, got {:.2} MWh",
        solar_to_mass / 1e6
    );

    assert!(
        mass_max_temp - mass_min_temp < 25.0, // <25°C swing
        "Mass temperature swing should be <25°C, got {:.2}°C",
        mass_max_temp - mass_min_temp
    );
}

/// Comprehensive analysis of hvac_power_demand calculation issues
///
/// This test identifies specific problems in the HVAC demand calculation:
/// 1. Is demand calculated when Ti_free is within deadband?
/// 2. Is demand magnitude proportional to temperature difference?
/// 3. Are sensitivity values appropriate for high-mass buildings?
#[test]
fn test_hvac_power_demand_calculation_issues() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model =
        fluxion::sim::engine::ThermalModel::<fluxion::physics::cta::VectorField>::from_spec(&spec);

    // Get conductance values to understand sensitivity
    let h_tr_ms = model.h_tr_ms.as_slice().first().copied().unwrap_or(0.0);
    let h_tr_is = model.h_tr_is.as_slice().first().copied().unwrap_or(0.0);

    println!("\n=== Task 1 Detailed: HVAC Power Demand Calculation ===");
    println!("h_tr_ms (mass-to-surface): {:.2} W/K", h_tr_ms);
    println!("h_tr_is (surface-to-interior): {:.2} W/K", h_tr_is);
    println!("Thermal model type: {:?}", model.thermal_model_type);

    // Calculate expected sensitivity for Case 900
    // Sensitivity = h_tr_ms * h_tr_is / (h_tr_ms * h_tr_is + ...)
    let term_rest_1 = h_tr_ms * h_tr_is;
    let expected_sensitivity_base = term_rest_1 / (term_rest_1 + 100.0); // Simplified

    println!(
        "Expected sensitivity base: {:.6} °C/W",
        expected_sensitivity_base
    );

    // For high-mass buildings, sensitivity should be:
    // - High (large temperature change per unit power) because thermal mass buffers temperature
    // - This means HVAC needs more power to change temperature
    // - But the current calculation may be over-estimating demand

    println!();
    println!("HVAC demand calculation analysis:");
    println!("- Uses Ti_free (free-floating temp) for mode determination");
    println!("- Uses sensitivity to convert temp difference to power");
    println!("- Heating capacity clamped to 2100 W");
    println!("- Cooling capacity clamped to 100000 W");
    println!();
    println!("Issue: Annual energy over-prediction suggests:");
    println!("  1. Sensitivity may be too low (HVAC thinks it needs more power)");
    println!("  2. Or thermal mass is absorbing too much solar energy");
    println!("  3. Or HVAC is running when it shouldn't be (deadband issue)");

    // The test doesn't assert specific values because it's diagnostic
    // The key insight is that we need to investigate sensitivity calculation
    // and solar gain distribution to fix annual energy over-prediction
}
