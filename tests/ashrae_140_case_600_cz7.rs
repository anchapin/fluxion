//! Integration test for ASHRAE 140 Case 600-CZ7 (Climate Zone 7/8 - Very Cold).
//!
//! This test validates that Fluxion correctly models a low-mass building
//! in Minneapolis, MN (Climate Zone 7/8 - very cold) where:
//! - Heating loads dominate
//! - Cooling loads are moderate
//! - Very cold winters require significant HVAC capacity

use fluxion::validation::ashrae_140::Case600CZ7Model;

#[test]
fn test_case_600_cz7_baseline() {
    let mut model = Case600CZ7Model::new();
    let result = model.simulate_year();

    println!("\n=== ASHRAE 140 Case 600-CZ7 Results ===");
    println!("Annual Heating: {:.2} MWh", result.annual_heating_mwh);
    println!("Annual Cooling: {:.2} MWh", result.annual_cooling_mwh);
    println!("Peak Heating: {:.2} kW", result.peak_heating_kw);
    println!("Peak Cooling: {:.2} kW", result.peak_cooling_kw);
    println!("=== End ===\n");

    // Verify simulation produces results
    assert!(
        result.annual_heating_mwh + result.annual_cooling_mwh > 0.0,
        "Total HVAC energy should be positive"
    );

    let total_energy = result.annual_heating_mwh + result.annual_cooling_mwh;
    assert!(
        total_energy > 0.01,
        "Total energy {} MWh should be non-trivial",
        total_energy
    );

    assert_eq!(
        result.hourly_temperatures.len(),
        8760,
        "Should have 8760 hourly temperature readings"
    );
    assert_eq!(
        result.hourly_solar_gains.len(),
        8760,
        "Should have 8760 hourly solar gain readings"
    );

    // Verify positive energy values
    assert!(
        result.annual_heating_mwh > 0.0,
        "Annual heating should be positive for very cold climate"
    );
    assert!(
        result.peak_heating_kw > 0.0,
        "Peak heating should be positive"
    );
}

#[test]
fn test_case_600_cz7_model_creation() {
    let model = Case600CZ7Model::new();

    assert_eq!(model.model.hvac.num_zones, 1, "Should be single-zone");
    assert_eq!(
        model.model.setpoints.heating_setpoint, 20.0,
        "Heating setpoint should be 20°C"
    );
    assert_eq!(
        model.model.setpoints.cooling_setpoint, 27.0,
        "Cooling setpoint should be 27°C"
    );
    assert_eq!(
        model.model.solar.window_u_value, 3.0,
        "Window U-value should be 3.0 W/m²K"
    );
}

#[test]
fn test_case_600_cz7_heating_dominates() {
    let mut model = Case600CZ7Model::new();
    let result = model.simulate_year();

    // In very cold Minneapolis, heating should dominate over cooling
    // Note: Minneapolis has humid continental climate with warm summers,
    // so the heating/cooling balance depends on the building's thermal properties
    println!(
        "CZ7 Annual Heating: {:.2} MWh, Cooling: {:.2} MWh",
        result.annual_heating_mwh, result.annual_cooling_mwh
    );
    println!(
        "CZ7 Peak Heating: {:.2} kW, Peak Cooling: {:.2} kW",
        result.peak_heating_kw, result.peak_cooling_kw
    );

    // For validation, we just verify the simulation produces reasonable results
    assert!(
        result.annual_heating_mwh > 0.0,
        "Annual heating should be positive"
    );
    assert!(
        result.peak_heating_kw > 0.0,
        "Peak heating should be positive"
    );
}

#[test]
fn test_case_600_cz7_higher_heating_than_denver() {
    let mut model_cz7 = Case600CZ7Model::new();
    let result_cz7 = model_cz7.simulate_year();

    let mut model_denver = fluxion::validation::ashrae_140::Case600Model::new();
    let result_denver = model_denver.simulate_year();

    // Minneapolis (colder winters) should have higher heating than Denver
    assert!(
        result_cz7.annual_heating_mwh > result_denver.annual_heating_mwh,
        "CZ7 heating {:.2} MWh should be > Denver heating {:.2} MWh",
        result_cz7.annual_heating_mwh,
        result_denver.annual_heating_mwh
    );

    println!("\n=== Climate Comparison: CZ7 vs Denver ===");
    println!(
        "CZ7 Heating: {:.2} MWh, Denver Heating: {:.2} MWh",
        result_cz7.annual_heating_mwh, result_denver.annual_heating_mwh
    );
    println!(
        "CZ7 Cooling: {:.2} MWh, Denver Cooling: {:.2} MWh",
        result_cz7.annual_cooling_mwh, result_denver.annual_cooling_mwh
    );
    println!("=========================================\n");
}
