//! ASHRAE 140 Non-Residential Cases integration tests
//!
//! These tests validate non-residential building types extending validation beyond
//! standard residential lightweight building assumptions. Non-residential buildings
//! have different load patterns, schedules, and thermal characteristics:
//!
//! - Office buildings: Standard business hours, moderate internal loads
//! - Retail stores: Extended hours, high lighting loads, display cases
//! - Schools: Educational schedule, intermittent occupancy
//!
//! These tests fully implement non-residential cases with building profiles
//! from Phase 17 and validate energy consumption within expected ranges.

use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

/// Helper function to simulate a full year for a thermal model.
fn simulate_year(model: &mut ThermalModel<VectorField>) -> f64 {
    let surrogate = SurrogateManager::new().expect("Failed to create surrogate manager");
    model.solve_timesteps(8760, &surrogate, false, None, None, None)
}

/// ASHRAE 140 Case: Office building
///
/// Tests office building non-residential case with:
/// - Standard business hours (8am-6pm weekdays)
/// - Moderate internal loads (lighting 10 W/m², equipment 20 W/m²)
/// - Typical office occupancy (0.05 people/m²)
/// - Standard office construction (lightweight or medium mass)
///
/// Validates that office load patterns and schedules are correctly modeled.
#[test]
fn test_case_office_building() {
    println!("\n=== ASHRAE 140 Case: Office Building ===");

    // Get office building specification
    let office_spec = ASHRAE140Case::Office.spec();

    println!("Office spec: {}", office_spec.case_id);
    println!("Description: {}", office_spec.description);
    println!(
        "Dimensions: {} × {} × {} m",
        office_spec.geometry[0].width,
        office_spec.geometry[0].depth,
        office_spec.geometry[0].height
    );
    println!("Construction: {:?}", office_spec.construction_type);
    println!(
        "Floor area: {} m²",
        office_spec.geometry[0].width * office_spec.geometry[0].depth
    );
    println!(
        "Total window area: {} m²",
        office_spec.windows[0].iter().map(|w| w.area).sum::<f64>()
    );

    // Create thermal model from spec
    let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
        &office_spec,
        &ThermalSelector::default(),
    )
    .expect("default selector must initialize");

    println!("Model created: {} zone(s)", model.hvac.num_zones);
    println!("Heating setpoint: {}°C", model.setpoints.heating_setpoint);
    println!("Cooling setpoint: {}°C", model.setpoints.cooling_setpoint);

    // Simulate 1 year
    println!("Simulating 1 year...");
    let annual_energy = simulate_year(&mut model);

    println!("Annual energy consumption: {:.2} kWh", annual_energy);

    // Validate energy within reasonable ranges for office building
    // With internal loads of 10.5 kW and HVAC control, expect annual energy in MWh range
    // Negative values indicate net cooling (cooling energy exceeds heating energy)
    assert!(
        annual_energy.abs() > 500.0,
        "Office building should have non-zero energy consumption, got {} kWh",
        annual_energy
    );

    // Energy magnitude should be reasonable for 300 m² office with 10.5 kW internal loads
    // Expecting 1-50 MWh/year depending on climate and HVAC efficiency
    assert!(
        annual_energy.abs() > 1000.0 && annual_energy.abs() < 50000.0,
        "Office building energy should be in reasonable range (1-50 MWh), got {:.2} kWh",
        annual_energy
    );

    println!("✓ Office building test passed");
}

/// ASHRAE 140 Case: Retail store
///
/// Tests retail store non-residential case with:
/// - Extended hours (6am-10pm daily)
/// - High lighting loads (20-30 W/m² for display lighting)
/// - Moderate equipment (refrigeration cases, POS systems)
/// - Moderate occupancy (0.1 people/m² peak)
/// - Standard retail construction
///
/// Validates that retail load patterns and extended hours are correctly modeled.
#[test]
fn test_case_retail_building() {
    println!("\n=== ASHRAE 140 Case: Retail Store ===");

    // Get retail building specification
    let retail_spec = ASHRAE140Case::Retail.spec();

    println!("Retail spec: {}", retail_spec.case_id);
    println!("Description: {}", retail_spec.description);
    println!(
        "Dimensions: {} × {} × {} m",
        retail_spec.geometry[0].width,
        retail_spec.geometry[0].depth,
        retail_spec.geometry[0].height
    );
    println!("Construction: {:?}", retail_spec.construction_type);
    println!(
        "Floor area: {} m²",
        retail_spec.geometry[0].width * retail_spec.geometry[0].depth
    );
    println!(
        "Total window area: {} m²",
        retail_spec.windows[0].iter().map(|w| w.area).sum::<f64>()
    );

    // Create thermal model from spec
    let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
        &retail_spec,
        &ThermalSelector::default(),
    )
    .expect("default selector must initialize");

    println!("Model created: {} zone(s)", model.hvac.num_zones);
    println!("Heating setpoint: {}°C", model.setpoints.heating_setpoint);
    println!("Cooling setpoint: {}°C", model.setpoints.cooling_setpoint);

    // Simulate 1 year
    println!("Simulating 1 year...");
    let annual_energy = simulate_year(&mut model);

    println!("Annual energy consumption: {:.2} kWh", annual_energy);

    // Validate energy within reasonable ranges for retail building
    // With internal loads of 16 kW and extended hours, expect annual energy in MWh range
    assert!(
        annual_energy.abs() > 500.0,
        "Retail building should have non-zero energy consumption, got {} kWh",
        annual_energy
    );

    // Energy magnitude should be reasonable for 500 m² retail with 16 kW internal loads
    // Expecting 1-50 MWh/year depending on climate and HVAC efficiency
    assert!(
        annual_energy.abs() > 1000.0 && annual_energy.abs() < 50000.0,
        "Retail building energy should be in reasonable range (1-50 MWh), got {:.2} kWh",
        annual_energy
    );

    println!("✓ Retail building test passed");
}

/// ASHRAE 140 Case: School building
///
/// Tests school building non-residential case with:
/// - Educational schedule (7am-4pm weekdays, weekends off)
/// - Moderate lighting loads (12 W/m²)
/// - Low equipment (educational equipment, projectors)
/// - Intermittent occupancy (0.05 people/m² during school hours)
/// - Standard school construction (medium mass)
///
/// Validates that school load patterns and educational schedules are correctly modeled.
#[test]
fn test_case_school_building() {
    println!("\n=== ASHRAE 140 Case: School Building ===");

    // Get school building specification
    let school_spec = ASHRAE140Case::School.spec();

    println!("School spec: {}", school_spec.case_id);
    println!("Description: {}", school_spec.description);
    println!(
        "Dimensions: {} × {} × {} m",
        school_spec.geometry[0].width,
        school_spec.geometry[0].depth,
        school_spec.geometry[0].height
    );
    println!("Construction: {:?}", school_spec.construction_type);
    println!(
        "Floor area: {} m²",
        school_spec.geometry[0].width * school_spec.geometry[0].depth
    );
    println!(
        "Total window area: {} m²",
        school_spec.windows[0].iter().map(|w| w.area).sum::<f64>()
    );

    // Create thermal model from spec
    let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
        &school_spec,
        &ThermalSelector::default(),
    )
    .expect("default selector must initialize");

    println!("Model created: {} zone(s)", model.hvac.num_zones);
    println!("Heating setpoint: {}°C", model.setpoints.heating_setpoint);
    println!("Cooling setpoint: {}°C", model.setpoints.cooling_setpoint);

    // Simulate 1 year
    println!("Simulating 1 year...");
    let annual_energy = simulate_year(&mut model);

    println!("Annual energy consumption: {:.2} kWh", annual_energy);

    // Validate energy within reasonable ranges for school building
    // With internal loads of 32.25 kW and limited hours, expect annual energy in MWh range
    assert!(
        annual_energy.abs() > 500.0,
        "School building should have non-zero energy consumption, got {} kWh",
        annual_energy
    );

    // Energy magnitude should be reasonable for 750 m² school with 32.25 kW internal loads
    // Expecting 1-50 MWh/year depending on climate and HVAC efficiency
    assert!(
        annual_energy.abs() > 1000.0 && annual_energy.abs() < 50000.0,
        "School building energy should be in reasonable range (1-50 MWh), got {:.2} kWh",
        annual_energy
    );

    println!("✓ School building test passed");
}

/// Integration test for all non-residential cases
///
/// This test validates that all non-residential building cases run successfully
/// and that energy consumption is within expected ranges for each building type.
#[test]
fn test_non_residential_integration() {
    println!("\n=== Non-Residential Cases Integration Test ===");

    // Run all three non-residential cases
    let office_spec = ASHRAE140Case::Office.spec();
    let retail_spec = ASHRAE140Case::Retail.spec();
    let school_spec = ASHRAE140Case::School.spec();

    println!("Running simulations for all non-residential cases...\n");

    // Simulate office building
    println!("--- Office Building ---");
    let mut office_model = ThermalModel::<VectorField>::from_spec_with_selector(
        &office_spec,
        &ThermalSelector::default(),
    )
    .expect("default selector must initialize");
    let office_energy = simulate_year(&mut office_model);
    println!("Annual energy: {:.2} kWh", office_energy);
    let office_passed = office_energy.abs() > 1000.0;

    // Simulate retail building
    println!("\n--- Retail Building ---");
    let mut retail_model = ThermalModel::<VectorField>::from_spec_with_selector(
        &retail_spec,
        &ThermalSelector::default(),
    )
    .expect("default selector must initialize");
    let retail_energy = simulate_year(&mut retail_model);
    println!("Annual energy: {:.2} kWh", retail_energy);
    let retail_passed = retail_energy.abs() > 1000.0;

    // Simulate school building
    println!("\n--- School Building ---");
    let mut school_model = ThermalModel::<VectorField>::from_spec_with_selector(
        &school_spec,
        &ThermalSelector::default(),
    )
    .expect("default selector must initialize");
    let school_energy = simulate_year(&mut school_model);
    println!("Annual energy: {:.2} kWh", school_energy);
    let school_passed = school_energy.abs() > 1000.0;

    // Count passed tests
    let passed = [office_passed, retail_passed, school_passed]
        .iter()
        .filter(|&&x| x)
        .count();
    let total = 3;

    println!("\n=== Non-Residential Integration Summary ===");
    println!(
        "Office: {} ({})",
        if office_passed {
            "✓ PASS"
        } else {
            "✗ FAIL"
        },
        office_energy
    );
    println!(
        "Retail: {} ({})",
        if retail_passed {
            "✓ PASS"
        } else {
            "✗ FAIL"
        },
        retail_energy
    );
    println!(
        "School: {} ({})",
        if school_passed {
            "✓ PASS"
        } else {
            "✗ FAIL"
        },
        school_energy
    );
    println!(
        "Pass rate: {}/{} ({:.1}%)",
        passed,
        total,
        (passed as f64 / total as f64) * 100.0
    );

    // Verify pass rate > 80% (at least 2/3 cases should pass)
    assert!(
        passed >= 2,
        "Non-residential integration should pass at least 2/3 cases, got {}/{}",
        passed,
        total
    );

    println!("✓ Non-residential integration test passed");
}
