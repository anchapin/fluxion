//! Unit tests for constants module.

use fluxion::physics::constants::thermal::ashrae_140::{
    EXTERIOR_FILM_COEFF, INTERIOR_FILM_COEFF, SOLAR_ABSORPTANCE_DEFAULT,
};
use fluxion::physics::constants::thermal::iso_13790::{
    calculate_effective_thermal_mass, THERMAL_MASS_HEAVY, THERMAL_MASS_HEAVY_UPPER,
    THERMAL_MASS_LIGHT, THERMAL_MASS_LIGHT_UPPER, THERMAL_MASS_MEDIUM, THERMAL_MASS_MEDIUM_UPPER,
    THERMAL_MASS_VERY_HEAVY, THERMAL_MASS_VERY_LIGHT,
};

#[test]
fn test_ashrae_140_interior_film_coeff() {
    assert_eq!(INTERIOR_FILM_COEFF, 8.29);
}

#[test]
fn test_ashrae_140_exterior_film_coeff() {
    assert_eq!(EXTERIOR_FILM_COEFF, 18.3);
}

#[test]
fn test_ashrae_140_solar_absorptance() {
    assert_eq!(SOLAR_ABSORPTANCE_DEFAULT, 0.7);
}

#[test]
fn test_ashrae_140_constants_are_positive() {
    assert!(INTERIOR_FILM_COEFF > 0.0);
    assert!(EXTERIOR_FILM_COEFF > 0.0);
    assert!(SOLAR_ABSORPTANCE_DEFAULT > 0.0);
    assert!(SOLAR_ABSORPTANCE_DEFAULT <= 1.0);
}

// ISO 13790 Annex C tests

#[test]
fn test_iso_13790_thermal_mass_thresholds() {
    assert_eq!(THERMAL_MASS_VERY_LIGHT, 50.0);
    assert_eq!(THERMAL_MASS_LIGHT, 50.0);
    assert_eq!(THERMAL_MASS_LIGHT_UPPER, 150.0);
    assert_eq!(THERMAL_MASS_MEDIUM, 150.0);
    assert_eq!(THERMAL_MASS_MEDIUM_UPPER, 260.0);
    assert_eq!(THERMAL_MASS_HEAVY, 260.0);
    assert_eq!(THERMAL_MASS_HEAVY_UPPER, 370.0);
    assert_eq!(THERMAL_MASS_VERY_HEAVY, 370.0);
}

#[test]
fn test_calculate_effective_thermal_mass() {
    // Test with concrete layer: 0.1m * 2300 kg/m³ * 840 J/kgK = 193200 J/m²K = 193.2 kJ/m²K
    let layers = vec![(0.1, 2300.0, 840.0)];
    let thermal_mass = calculate_effective_thermal_mass(&layers);
    assert!((thermal_mass - 193.2).abs() < 0.1);
}

#[test]
fn test_calculate_effective_thermal_mass_multiple_layers() {
    // Test with two layers
    let layers = vec![
        (0.1, 2300.0, 840.0), // 193.2 kJ/m²K
        (0.05, 50.0, 840.0),  // 2.1 kJ/m²K
    ];
    let thermal_mass = calculate_effective_thermal_mass(&layers);
    assert!((thermal_mass - 195.3).abs() < 0.1);
}
