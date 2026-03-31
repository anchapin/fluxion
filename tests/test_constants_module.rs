//! Unit tests for constants module.

use fluxion::physics::constants::atmospheric as atm;
use fluxion::physics::constants::solar::ashrae_140 as solar;
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
    let layers = vec![(0.1, 2300.0, 840.0)];
    let thermal_mass = calculate_effective_thermal_mass(&layers);
    assert!((thermal_mass - 193.2).abs() < 0.1);
}

#[test]
fn test_calculate_effective_thermal_mass_multiple_layers() {
    let layers = vec![(0.1, 2300.0, 840.0), (0.05, 50.0, 840.0)];
    let thermal_mass = calculate_effective_thermal_mass(&layers);
    assert!((thermal_mass - 195.3).abs() < 0.1);
}

#[test]
fn test_atmospheric_pressure() {
    assert!((atm::STANDARD_ATMOSPHERIC_PRESSURE - 101325.0).abs() < 1.0);
    assert!(atm::STANDARD_ATMOSPHERIC_PRESSURE > 0.0);
}

#[test]
fn test_air_density() {
    assert!((atm::AIR_DENSITY_SEA_LEVEL - 1.225).abs() < 0.01);
    assert!(atm::AIR_DENSITY_SEA_LEVEL > 0.0);
}

#[test]
fn test_air_specific_heat() {
    assert!((atm::AIR_SPECIFIC_HEAT - 1005.0).abs() < 5.0);
    assert!(atm::AIR_SPECIFIC_HEAT > 0.0);
}

#[test]
fn test_gas_constants() {
    assert!((atm::SPECIFIC_GAS_CONSTANT_DRY_AIR - 287.05).abs() < 1.0);
    assert!(atm::SPECIFIC_GAS_CONSTANT_DRY_AIR > 0.0);
    assert!((atm::SPECIFIC_GAS_CONSTANT_WATER_VAPOR - 461.52).abs() < 1.0);
    assert!(atm::SPECIFIC_GAS_CONSTANT_WATER_VAPOR > 0.0);
    assert!(atm::SPECIFIC_GAS_CONSTANT_WATER_VAPOR > atm::SPECIFIC_GAS_CONSTANT_DRY_AIR);
}

#[test]
fn test_atmospheric_lapse_rate() {
    assert!((atm::ATMOSPHERIC_LAPSE_RATE - 0.0065).abs() < 0.0001);
    assert!(atm::ATMOSPHERIC_LAPSE_RATE > 0.0);
}

#[test]
fn test_gravity_acceleration() {
    assert!((atm::GRAVITY_ACCELERATION - 9.80665).abs() < 0.001);
    assert!(atm::GRAVITY_ACCELERATION > 0.0);
}

#[test]
fn test_standard_temperature_sea_level() {
    assert!((atm::STANDARD_TEMPERATURE_SEA_LEVEL - 288.15).abs() < 0.5);
    assert!(atm::STANDARD_TEMPERATURE_SEA_LEVEL > 273.0);
}

#[test]
fn test_solar_constant() {
    assert!((solar::SOLAR_CONSTANT - 1361.0).abs() < 10.0);
    assert!(solar::SOLAR_CONSTANT > 1000.0);
    assert!(solar::SOLAR_CONSTANT < 1500.0);
}

#[test]
fn test_solar_declination_coefficient() {
    assert!((solar::SOLAR_DECLINATION_COEFFICIENT - 23.45).abs() < 0.1);
    assert!(solar::SOLAR_DECLINATION_COEFFICIENT > 23.0);
    assert!(solar::SOLAR_DECLINATION_COEFFICIENT < 24.0);
}

#[test]
fn test_hour_angle_coefficient() {
    assert!((solar::HOUR_ANGLE_COEFFICIENT - 15.0).abs() < 0.1);
    assert!(solar::HOUR_ANGLE_COEFFICIENT > 0.0);
}

#[test]
fn test_zenith_angle_noon() {
    assert!((solar::ZENITH_ANGLE_NOON - 0.0).abs() < 0.01);
}

#[test]
fn test_atmospheric_extinction_coefficient() {
    assert!((solar::ATMOSPHERIC_EXTINCTION_COEFFICIENT - 0.2).abs() < 0.1);
    assert!(solar::ATMOSPHERIC_EXTINCTION_COEFFICIENT > 0.0);
    assert!(solar::ATMOSPHERIC_EXTINCTION_COEFFICIENT < 1.0);
}

#[test]
fn test_diffuse_fraction_coefficient() {
    assert!((solar::DIFFUSE_FRACTION_COEFFICIENT - 0.1).abs() < 0.05);
    assert!(solar::DIFFUSE_FRACTION_COEFFICIENT > 0.0);
    assert!(solar::DIFFUSE_FRACTION_COEFFICIENT < 0.5);
}

#[test]
fn test_all_constants_are_finite() {
    assert!(atm::STANDARD_ATMOSPHERIC_PRESSURE.is_finite());
    assert!(atm::AIR_DENSITY_SEA_LEVEL.is_finite());
    assert!(atm::AIR_SPECIFIC_HEAT.is_finite());
    assert!(atm::SPECIFIC_GAS_CONSTANT_DRY_AIR.is_finite());
    assert!(atm::SPECIFIC_GAS_CONSTANT_WATER_VAPOR.is_finite());
    assert!(atm::ATMOSPHERIC_LAPSE_RATE.is_finite());
    assert!(atm::GRAVITY_ACCELERATION.is_finite());
    assert!(atm::STANDARD_TEMPERATURE_SEA_LEVEL.is_finite());
    assert!(solar::SOLAR_CONSTANT.is_finite());
    assert!(solar::SOLAR_DECLINATION_COEFFICIENT.is_finite());
    assert!(solar::HOUR_ANGLE_COEFFICIENT.is_finite());
    assert!(solar::ZENITH_ANGLE_NOON.is_finite());
    assert!(solar::ATMOSPHERIC_EXTINCTION_COEFFICIENT.is_finite());
    assert!(solar::DIFFUSE_FRACTION_COEFFICIENT.is_finite());
    assert!(INTERIOR_FILM_COEFF.is_finite());
    assert!(EXTERIOR_FILM_COEFF.is_finite());
    assert!(SOLAR_ABSORPTANCE_DEFAULT.is_finite());
}

#[test]
fn test_all_thermal_constants_positive() {
    assert!(atm::STANDARD_ATMOSPHERIC_PRESSURE > 0.0);
    assert!(atm::AIR_DENSITY_SEA_LEVEL > 0.0);
    assert!(atm::AIR_SPECIFIC_HEAT > 0.0);
    assert!(atm::SPECIFIC_GAS_CONSTANT_DRY_AIR > 0.0);
    assert!(atm::SPECIFIC_GAS_CONSTANT_WATER_VAPOR > 0.0);
    assert!(atm::ATMOSPHERIC_LAPSE_RATE > 0.0);
    assert!(atm::GRAVITY_ACCELERATION > 0.0);
    assert!(atm::STANDARD_TEMPERATURE_SEA_LEVEL > 0.0);
    assert!(solar::SOLAR_CONSTANT > 0.0);
    assert!(solar::SOLAR_DECLINATION_COEFFICIENT > 0.0);
    assert!(solar::HOUR_ANGLE_COEFFICIENT > 0.0);
    assert!(INTERIOR_FILM_COEFF > 0.0);
    assert!(EXTERIOR_FILM_COEFF > 0.0);
}

#[test]
fn test_ideal_gas_law_consistency() {
    let calculated_pressure = atm::AIR_DENSITY_SEA_LEVEL
        * atm::SPECIFIC_GAS_CONSTANT_DRY_AIR
        * atm::STANDARD_TEMPERATURE_SEA_LEVEL;
    let error_pct = ((calculated_pressure - atm::STANDARD_ATMOSPHERIC_PRESSURE)
        / atm::STANDARD_ATMOSPHERIC_PRESSURE
        * 100.0)
        .abs();
    assert!(error_pct < 1.0, "Ideal gas law error: {:.2}%", error_pct);
}

#[test]
fn test_solar_absorptance_range() {
    assert!(SOLAR_ABSORPTANCE_DEFAULT >= 0.0);
    assert!(SOLAR_ABSORPTANCE_DEFAULT <= 1.0);
    assert!(SOLAR_ABSORPTANCE_DEFAULT >= 0.5);
}
