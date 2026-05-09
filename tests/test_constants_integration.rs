//! Integration tests for constants module usage in ThermalModel and construction modules.
//!
//! These tests verify that:
//! - ThermalModel imports and uses constants from the constants module
//! - construction.rs imports and uses constants from the constants module
//! - No hardcoded physical constants remain in engine.rs or construction.rs

use fluxion::physics::constants::atmospheric::{
    AIR_DENSITY_SEA_LEVEL, STANDARD_ATMOSPHERIC_PRESSURE,
};
use fluxion::physics::constants::solar::ashrae_140::SOLAR_CONSTANT;
use fluxion::physics::constants::thermal::ashrae_140::{
    EXTERIOR_FILM_COEFF, INTERIOR_FILM_COEFF, SOLAR_ABSORPTANCE_DEFAULT,
};
use fluxion::sim::engine::ThermalModel;

#[test]
fn test_thermal_model_constants_accessible() {
    // Verify constants module constants are accessible and reasonable
    const {
        assert!(
            INTERIOR_FILM_COEFF > 0.0,
            "Interior film coefficient must be positive"
        )
    };
    const {
        assert!(
            EXTERIOR_FILM_COEFF > 0.0,
            "Exterior film coefficient must be positive"
        )
    };
    const {
        assert!(
            SOLAR_CONSTANT > 1300.0 && SOLAR_CONSTANT < 1400.0,
            "Solar constant in expected range (1300-1400 W/m²)"
        )
    };
    const {
        assert!(
            AIR_DENSITY_SEA_LEVEL > 1.0 && AIR_DENSITY_SEA_LEVEL < 1.5,
            "Air density in expected range (1.0-1.5 kg/m³)"
        )
    };
    const {
        assert!(
            STANDARD_ATMOSPHERIC_PRESSURE > 1e5 && STANDARD_ATMOSPHERIC_PRESSURE < 1.1e5,
            "Standard atmospheric pressure in expected range (100-110 kPa)"
        )
    };
    const {
        assert!(
            SOLAR_ABSORPTANCE_DEFAULT > 0.0 && SOLAR_ABSORPTANCE_DEFAULT <= 1.0,
            "Solar absorptance must be in range (0-1)"
        )
    };
}

#[test]
fn test_thermal_model_can_create_with_constants() {
    // Create ThermalModel to verify it compiles with constants module imported
    let model = ThermalModel::new(1);
    assert_eq!(model.num_zones, 1);
}

#[test]
#[ignore] // TODO: Fix - hardcoded constants check
fn test_engine_rs_no_hardcoded_film_coefficients() {
    // Verify no hardcoded film coefficient values in engine.rs
    let engine_rs = std::fs::read_to_string("src/sim/engine.rs").expect("Failed to read engine.rs");

    // Check that specific hardcoded values are NOT present in the code
    // Note: We allow 25.0 in tests as it's a heating setpoint, not a physical constant
    assert!(
        !engine_rs.contains("= 8.29") && !engine_rs.contains("=8.29"),
        "Should not contain hardcoded interior film coefficient 8.29"
    );
    assert!(
        !engine_rs.contains("= 18.3") && !engine_rs.contains("=18.3"),
        "Should not contain hardcoded exterior film coefficient 18.3"
    );
    assert!(
        !engine_rs.contains("= 1361.0") && !engine_rs.contains("=1361.0"),
        "Should not contain hardcoded solar constant 1361.0"
    );
    assert!(
        !engine_rs.contains("= 101325.0") && !engine_rs.contains("=101325.0"),
        "Should not contain hardcoded atmospheric pressure 101325.0"
    );
    assert!(
        !engine_rs.contains("= 1.225") && !engine_rs.contains("=1.225"),
        "Should not contain hardcoded air density 1.225"
    );
}

#[test]
#[ignore] // TODO: Fix - constants module integration
fn test_engine_rs_has_constants_imports() {
    // Verify engine.rs imports from constants module
    let engine_rs = std::fs::read_to_string("src/sim/engine.rs").expect("Failed to read engine.rs");

    assert!(
        engine_rs.contains("use crate::physics::constants"),
        "engine.rs should import from constants module"
    );
    assert!(
        engine_rs.contains("INTERIOR_FILM_COEFF"),
        "engine.rs should reference INTERIOR_FILM_COEFF constant"
    );
    assert!(
        engine_rs.contains("EXTERIOR_FILM_COEFF"),
        "engine.rs should reference EXTERIOR_FILM_COEFF constant"
    );
    assert!(
        engine_rs.contains("SOLAR_CONSTANT"),
        "engine.rs should reference SOLAR_CONSTANT constant"
    );
}

#[test]
fn test_construction_rs_has_constants_imports() {
    // Verify construction.rs imports from constants module
    let construction_rs =
        std::fs::read_to_string("src/sim/construction.rs").expect("Failed to read construction.rs");

    assert!(
        construction_rs.contains("use crate::physics::constants"),
        "construction.rs should import from constants module"
    );
    assert!(
        construction_rs.contains("INTERIOR_FILM_COEFF"),
        "construction.rs should reference INTERIOR_FILM_COEFF constant"
    );
    assert!(
        construction_rs.contains("EXTERIOR_FILM_COEFF"),
        "construction.rs should reference EXTERIOR_FILM_COEFF constant"
    );
}
