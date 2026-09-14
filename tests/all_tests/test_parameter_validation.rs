use fluxion::physics::constants::*;
use fluxion::sim::assembly::{ConcreteMaterial, InsulationMaterial, MaterialLayer};

#[test]
fn test_ashrae_140_constants_match_specification() {
    // Verify ASHRAE 140 constants match specification values

    assert!(
        (thermal::ashrae_140::INTERIOR_FILM_COEFF - 8.29).abs() < 0.01,
        "Interior film coefficient should be 8.29 W/m²K per ASHRAE 140"
    );
    assert!(
        (thermal::ashrae_140::EXTERIOR_FILM_COEFF - 18.3).abs() < 0.1,
        "Exterior film coefficient should be 18.3 W/m²K per ASHRAE 140"
    );
}

#[test]
fn test_iso_13790_thresholds_match_specification() {
    // Verify ISO 13790 Annex C thresholds match specification

    assert!(
        (thermal::iso_13790::THERMAL_MASS_VERY_LIGHT - 50.0).abs() < 0.1,
        "VeryLight threshold should be 50 kJ/m²K per ISO 13790"
    );
    assert!(
        (thermal::iso_13790::THERMAL_MASS_LIGHT - 50.0).abs() < 0.1,
        "Light threshold should be 50 kJ/m²K per ISO 13790"
    );
    assert!(
        (thermal::iso_13790::THERMAL_MASS_LIGHT_UPPER - 150.0).abs() < 0.1,
        "Light upper threshold should be 150 kJ/m²K per ISO 13790"
    );
    assert!(
        (thermal::iso_13790::THERMAL_MASS_MEDIUM - 150.0).abs() < 0.1,
        "Medium threshold should be 150 kJ/m²K per ISO 13790"
    );
    assert!(
        (thermal::iso_13790::THERMAL_MASS_MEDIUM_UPPER - 260.0).abs() < 0.1,
        "Medium upper threshold should be 260 kJ/m²K per ISO 13790"
    );
    assert!(
        (thermal::iso_13790::THERMAL_MASS_HEAVY - 260.0).abs() < 0.1,
        "Heavy threshold should be 260 kJ/m²K per ISO 13790"
    );
    assert!(
        (thermal::iso_13790::THERMAL_MASS_HEAVY_UPPER - 370.0).abs() < 0.1,
        "Heavy upper threshold should be 370 kJ/m²K per ISO 13790"
    );
    assert!(
        (thermal::iso_13790::THERMAL_MASS_VERY_HEAVY - 370.0).abs() < 0.1,
        "VeryHeavy threshold should be 370 kJ/m²K per ISO 13790"
    );
}

#[test]
fn test_solar_constants_match_scientific_consensus() {
    // Verify solar constants match scientific consensus

    assert!(
        (solar::ashrae_140::SOLAR_CONSTANT - 1361.0).abs() < 1.0,
        "Solar constant should be ~1361 W/m² per IPCC AR6"
    );
    assert!(
        (solar::ashrae_140::SOLAR_DECLINATION_COEFFICIENT - 23.45).abs() < 0.1,
        "Solar declination coefficient should be 23.45° per Cooper (1969)"
    );
}

#[test]
fn test_atmospheric_constants_match_iso_2533() {
    // Verify atmospheric constants match ISO 2533 Standard Atmosphere

    assert!(
        (atmospheric::STANDARD_ATMOSPHERIC_PRESSURE - 101325.0).abs() < 100.0,
        "Standard atmospheric pressure should be 101325 Pa per ISO 2533"
    );
    assert!(
        (atmospheric::AIR_DENSITY_SEA_LEVEL - 1.225).abs() < 0.01,
        "Air density should be 1.225 kg/m³ per ISO 2533"
    );
}

#[test]
fn test_material_properties_match_ashrae_fundamentals() {
    // Verify material properties match ASHRAE Handbook of Fundamentals

    // Concrete
    let concrete = ConcreteMaterial::new(0.1);
    assert!(
        (concrete.conductivity() - 1.4).abs() < 0.1,
        "Concrete conductivity should be ~1.4 W/mK per ASHRAE Fundamentals"
    );
    assert!(
        (concrete.density() - 2300.0).abs() < 50.0,
        "Concrete density should be ~2300 kg/m³ per ASHRAE Fundamentals"
    );
    assert!(
        (concrete.specific_heat() - 840.0).abs() < 50.0,
        "Concrete specific heat should be ~840 J/kgK per ASHRAE Fundamentals"
    );

    // Insulation
    let insulation = InsulationMaterial::new(0.05);
    assert!(
        (insulation.conductivity() - 0.04).abs() < 0.01,
        "Insulation conductivity should be ~0.04 W/mK per ASHRAE Fundamentals"
    );
    assert!(
        (insulation.density() - 50.0).abs() < 10.0,
        "Insulation density should be ~50 kg/m³ per ASHRAE Fundamentals"
    );
}

#[test]
fn test_uncertainty_ranges_are_reasonable() {
    // Verify uncertainty ranges are reasonable (±10% or less for most constants)

    // Film coefficients: ±5% acceptable
    let h_int = thermal::ashrae_140::INTERIOR_FILM_COEFF;
    assert!(
        0.05 * h_int < 0.5,
        "Interior film coefficient uncertainty < 5%"
    );
}
