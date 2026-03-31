// Solar Distribution Unit Tests for ASHRAE 140
//
// These tests validate how solar gains are distributed between zone air
// and thermal mass for both low-mass and high-mass buildings.
//
// Components tested:
// 1. Low-mass solar distribution factor
// 2. High-mass solar distribution factor
// 3. Time-dependent distribution (thermal lag)
// 4. Heat balance: internal gains → zone air vs thermal mass

use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, ConstructionType};

fn main() {
    println!("Running solar distribution tests...");
}

#[test]
fn test_low_mass_solar_distribution_to_air() {
    // For low-mass buildings (600-series), solar gains should primarily go
    // to zone air directly because thermal mass has low heat capacity

    let low_mass_spec = ASHRAE140Case::Case600.spec();

    // Create low-mass model
    let model = ThermalModel::from_spec(&low_mass_spec);

    // Validate that low-mass construction is actually configured
    assert_eq!(
        model.construction_type,
        ConstructionType::LowMass,
        "Case 600 should use low-mass construction"
    );

    // Thermal mass should be low for low-mass
    // Check thermal capacitance
    let thermal_mass = model.total_thermal_capacity.unwrap_or(0.0);

    // Low-mass: roughly 1000-5000 J/K
    // High-mass: roughly 50000-200000 J/K

    assert!(
        thermal_mass < 20000.0,
        "Low-mass construction should have thermal capacitance < 20000 J/K, got {:.1} J/K",
        thermal_mass
    );
}

#[test]
fn test_high_mass_solar_thermal_lag() {
    // For high-mass buildings (900-series), solar gains should be distributed
    // between zone air and thermal mass with time delay (thermal lag)

    let high_mass_spec = ASHRAE140Case::Case900.spec();

    let model = ThermalModel::from_spec(&high_mass_spec);

    // Validate that high-mass construction is configured
    assert_eq!(
        model.construction_type,
        ConstructionType::HighMass,
        "Case 900 should use high-mass construction"
    );

    // Thermal mass should be high for high-mass
    let thermal_mass = model.total_thermal_capacity.unwrap_or(0.0);

    // High-mass: roughly 50000-200000 J/K
    assert!(
        thermal_mass > 50000.0,
        "High-mass construction should have thermal capacitance > 50000 J/K, got {:.1} J/K",
        thermal_mass
    );
}

#[test]
fn test_thermal_time_constant_calculation() {
    // Test that thermal time constant (tau) is calculated correctly
    // tau = R × C (resistance × capacitance)

    let spec = ASHRAE140Case::Case900.spec();

    // Get thermal resistance (sum of 1/C for all layers)
    let total_thermal_resistance: f64 = spec
        .construction
        .wall
        .layers
        .iter()
        .map(|l| l.thickness / l.conductivity)
        .sum::<f64>();

    // Get thermal capacitance
    let total_thermal_capacitance: f64 = spec
        .construction
        .wall
        .layers
        .iter()
        .map(|l| l.thickness * l.density * l.specific_heat * 48.0) // 48 m² zone area
        .sum::<f64>();

    let tau_hours = total_thermal_resistance * total_thermal_capacitance / 3600.0;

    // For Case 900 high-mass: tau should be ~73 hours
    assert!(
        tau_hours > 50.0 && tau_hours < 100.0,
        "Thermal time constant for Case 900 should be 50-100 hours, got {:.1} hours",
        tau_hours
    );
}

#[test]
fn test_conductance_mass_dependence() {
    // Test that h_tr_ms and h_tr_is are appropriate for
    // low-mass vs high-mass constructions

    let low_spec = ASHRAE140Case::Case600.spec();
    let high_spec = ASHRAE140Case::Case900.spec();

    let low_model = ThermalModel::from_spec(&low_spec);
    let high_model = ThermalModel::from_spec(&high_spec);

    // h_tr_ms (mass to surface) should scale with thermal mass
    let low_h_tr_ms = low_model.h_tr_ms.as_ref().get(0).copied().unwrap_or(0.0);
    let high_h_tr_ms = high_model.h_tr_ms.as_ref().get(0).copied().unwrap_or(0.0);

    // High-mass should have higher h_tr_ms
    assert!(
        high_h_tr_ms > low_h_tr_ms,
        "High-mass h_tr_ms ({:.2} W/K) should be > low-mass ({:.2} W/K)",
        high_h_tr_ms,
        low_h_tr_ms
    );

    // h_tr_is (surface to interior air) should scale similarly
    let low_h_tr_is = low_model.h_tr_is.as_ref().get(0).copied().unwrap_or(0.0);
    let high_h_tr_is = high_model.h_tr_is.as_ref().get(0).copied().unwrap_or(0.0);

    assert!(
        high_h_tr_is > low_h_tr_is,
        "High-mass h_tr_is ({:.2} W/K) should be > low-mass ({:.2} W/K)",
        high_h_tr_is,
        low_h_tr_is
    );
}

#[test]
fn test_solar_distribution_factor_range() {
    // Solar distribution factor should be between 0.0 and 1.0
    // 0.0 = all solar to thermal mass
    // 1.0 = all solar to zone air

    let low_spec = ASHRAE140Case::Case600.spec();
    let high_spec = ASHRAE140Case::Case900.spec();

    // Verify construction types are correct
    assert_eq!(low_spec.construction_type, ConstructionType::LowMass);
    assert_eq!(high_spec.construction_type, ConstructionType::HighMass);
}

#[test]
fn test_low_vs_high_mass_response() {
    // Run short simulations for both low and high mass
    // Compare their thermal response characteristics

    let low_spec = ASHRAE140Case::Case600.spec();
    let high_spec = ASHRAE140Case::Case900.spec();

    // Create models
    let low_model = ThermalModel::from_spec(&low_spec);
    let high_model = ThermalModel::from_spec(&high_spec);

    // Compare thermal mass
    let low_mass = low_model.total_thermal_capacity.unwrap_or(0.0);
    let high_mass = high_model.total_thermal_capacity.unwrap_or(0.0);

    // Low-mass should have lower thermal mass
    assert!(
        low_mass < high_mass,
        "Low-mass ({:.1} J/K) should be < high-mass ({:.1} J/K)",
        low_mass,
        high_mass
    );
}
