//! Test to validate HVAC sensitivity correction factor for high-mass buildings (Issue #470)

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_case_900_thermal_mass_correction_factor() {
    // Test that thermal_mass_correction_factor is set correctly for Case 900
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("=== Thermal Mass Correction Factor Validation (Issue #470) ===");
    println!();
    println!("Case 900 (high-mass):");
    println!(
        "  thermal_mass_correction_factor: {:.2}",
        model.thermal_mass_correction_factor
    );
    println!("  Expected: 4.0 (balanced compromise for heating and cooling)");
    println!();

    // Verify correction factor is set to 4.0 for Case 900
    assert!(
        (model.thermal_mass_correction_factor - 4.0).abs() < 0.01,
        "thermal_mass_correction_factor should be 4.0 for Case 900, got {:.2}",
        model.thermal_mass_correction_factor
    );

    println!("✅ Thermal mass correction factor validated");
}

#[test]
fn test_case_600_thermal_mass_correction_factor() {
    // Test that thermal_mass_correction_factor is 1.0 for low-mass buildings
    let spec = ASHRAE140Case::Case600.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("=== Low-Mass Case Validation ===");
    println!();
    println!("Case 600 (low-mass):");
    println!(
        "  thermal_mass_correction_factor: {:.2}",
        model.thermal_mass_correction_factor
    );
    println!("  Expected: 1.0 (no correction, τ ≈ 1 hour)");
    println!();

    // Verify correction factor is 1.0 for Case 600
    assert!(
        (model.thermal_mass_correction_factor - 1.0).abs() < 0.01,
        "thermal_mass_correction_factor should be 1.0 for Case 600, got {:.2}",
        model.thermal_mass_correction_factor
    );

    println!("✅ Low-mass correction factor validated");
}

#[test]
fn test_case_900ff_thermal_mass_correction_factor() {
    // Test that thermal_mass_correction_factor is 1.0 for free-floating cases
    let spec = ASHRAE140Case::Case900FF.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("=== Free-Floating Case Validation ===");
    println!();
    println!("Case 900FF (free-floating high-mass):");
    println!(
        "  thermal_mass_correction_factor: {:.2}",
        model.thermal_mass_correction_factor
    );
    println!("  Expected: 1.0 (no correction for free-floating)");
    println!();

    // Verify correction factor is 1.0 for free-floating case
    assert!(
        (model.thermal_mass_correction_factor - 1.0).abs() < 0.01,
        "thermal_mass_correction_factor should be 1.0 for Case 900FF, got {:.2}",
        model.thermal_mass_correction_factor
    );

    println!("✅ Free-floating correction factor validated");
}
