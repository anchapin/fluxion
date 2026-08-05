//! Integration test for ASHRAE 140 Case 600-CZ3 (Climate Zone 3 - Hot-Humid).
//!
//! This test validates that Fluxion correctly models a low-mass building
//! in Miami, FL (Climate Zone 3 - hot-humid) where:
//! - Cooling loads dominate
//! - Heating loads are minimal
//! - High humidity affects sensible and latent cooling loads

use fluxion::validation::ashrae_140::Case600CZ3Model;

#[test]
fn test_case_600_cz3_baseline() {
    let mut model = Case600CZ3Model::new();
    let result = model.simulate_year();

    println!("\n=== ASHRAE 140 Case 600-CZ3 Results ===");
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
        result.annual_cooling_mwh > 0.0,
        "Annual cooling should be positive for hot-humid climate"
    );
    assert!(
        result.peak_cooling_kw > 0.0,
        "Peak cooling should be positive"
    );
}

#[test]
fn test_case_600_cz3_model_creation() {
    let model = Case600CZ3Model::new();

    assert_eq!(model.model.num_zones, 1, "Should be single-zone");
    assert_eq!(
        model.model.heating_setpoint, 20.0,
        "Heating setpoint should be 20°C"
    );
    assert_eq!(
        model.model.cooling_setpoint, 27.0,
        "Cooling setpoint should be 27°C"
    );
    assert_eq!(
        model.model.window_u_value, 3.0,
        "Window U-value should be 3.0 W/m²K"
    );
}

#[test]
fn test_case_600_cz3_cooling_dominates() {
    let mut model = Case600CZ3Model::new();
    let result = model.simulate_year();

    // In hot-humid Miami, cooling should dominate over heating
    assert!(
        result.annual_cooling_mwh > result.annual_heating_mwh,
        "CZ3 cooling {:.2} MWh should dominate heating {:.2} MWh",
        result.annual_cooling_mwh,
        result.annual_heating_mwh
    );

    // Peak cooling should be higher than peak heating
    assert!(
        result.peak_cooling_kw > result.peak_heating_kw,
        "CZ3 peak cooling {:.2} kW should exceed peak heating {:.2} kW",
        result.peak_cooling_kw,
        result.peak_heating_kw
    );
}

#[test]
fn test_case_600_cz3_higher_cooling_than_denver() {
    let mut model_cz3 = Case600CZ3Model::new();
    let result_cz3 = model_cz3.simulate_year();

    let mut model_denver = fluxion::validation::ashrae_140::Case600Model::new();
    let result_denver = model_denver.simulate_year();

    // Miami should have higher cooling than Denver (Zone 5)
    assert!(
        result_cz3.annual_cooling_mwh > result_denver.annual_cooling_mwh,
        "CZ3 cooling {:.2} MWh should be > Denver cooling {:.2} MWh",
        result_cz3.annual_cooling_mwh,
        result_denver.annual_cooling_mwh
    );

    // Denver should have higher heating than Miami
    assert!(
        result_denver.annual_heating_mwh > result_cz3.annual_heating_mwh,
        "Denver heating {:.2} MWh should be > CZ3 heating {:.2} MWh",
        result_denver.annual_heating_mwh,
        result_cz3.annual_heating_mwh
    );

    println!("\n=== Climate Comparison: CZ3 vs Denver ===");
    println!(
        "CZ3 Cooling: {:.2} MWh, Denver Cooling: {:.2} MWh",
        result_cz3.annual_cooling_mwh, result_denver.annual_cooling_mwh
    );
    println!(
        "CZ3 Heating: {:.2} MWh, Denver Heating: {:.2} MWh",
        result_cz3.annual_heating_mwh, result_denver.annual_heating_mwh
    );
    println!("=========================================\n");
}
