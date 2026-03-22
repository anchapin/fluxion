//! Thermal Mass Coupling Validation Test Cases
//!
//! This module implements validation test cases for thermal mass coupling behavior,
//! addressing the ASHRAE 140 requirement for coupling ratio > 0.1 for high-mass buildings.
//!
//! ## Validations Performed:
//! - Low-mass buildings (Case 600) should NOT be corrected
//! - High-mass buildings (Case 900) should be corrected to coupling ratio > 0.1
//! - Thermal capacitance threshold detection (5e6 J/K boundary)

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_thermal_mass_coupling_ratio_low_mass() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Calculate initial coupling ratio
    let h_tr_ms_initial: f64 = model.h_tr_ms.as_ref()[0];
    let h_tr_em_initial: f64 = model.h_tr_em.as_ref()[0];
    let _initial_ratio = h_tr_em_initial / h_tr_ms_initial;

    // Apply thermal mass correction
    model.apply_thermal_mass_correction();

    // Verify low-mass building was NOT corrected
    let h_tr_ms_final: f64 = model.h_tr_ms.as_ref()[0];
    let h_tr_em_final: f64 = model.h_tr_em.as_ref()[0];
    let final_ratio = h_tr_em_final / h_tr_ms_final;

    assert_eq!(
        h_tr_em_initial, h_tr_em_final,
        "Low-mass building h_tr_em should not change"
    );
    let initial_ratio = h_tr_em_initial / h_tr_ms_initial;
    assert_eq!(
        initial_ratio, final_ratio,
        "Low-mass building coupling ratio should not change"
    );
}

#[test]
fn test_thermal_mass_coupling_ratio_high_mass() {
    let spec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Verify high-mass building was corrected during model creation
    let h_tr_ms: f64 = model.h_tr_ms.as_ref()[0];
    let h_tr_em: f64 = model.h_tr_em.as_ref()[0];
    let final_ratio = h_tr_em / h_tr_ms;

    assert!(
        final_ratio >= 0.1,
        "High-mass building coupling ratio {} should be >= 0.1 after correction",
        final_ratio
    );
    assert!(h_tr_ms > 0.0, "h_tr_ms should be positive");
    assert!(h_tr_em > 0.0, "h_tr_em should be positive");
}

#[test]
fn test_thermal_mass_threshold_detection() {
    let spec = ASHRAE140Case::Case600.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Case 600 should be low-mass
    let total_cap: f64 = model.thermal_capacitance.iter().sum();
    let zone_area = model.zone_area[0];
    let air_cap = zone_area * 1.2 * 1005.0;
    let structure_cap = total_cap - air_cap;

    assert!(
        structure_cap < 5.0e6,
        "Case 600 structure capacitance {} J/K should be below high-mass threshold 5.0e6 J/K",
        structure_cap
    );

    // Case 900 should be high-mass
    let spec_900 = ASHRAE140Case::Case900.spec();
    let model_900 = ThermalModel::<VectorField>::from_spec(&spec_900);

    let total_cap_900: f64 = model_900.thermal_capacitance.iter().sum();
    let zone_area_900 = model_900.zone_area[0];
    let air_cap_900 = zone_area_900 * 1.2 * 1005.0;
    let structure_cap_900 = total_cap_900 - air_cap_900;

    assert!(
        structure_cap_900 > 5.0e6,
        "Case 900 structure capacitance {} J/K should exceed high-mass threshold 5.0e6 J/K",
        structure_cap_900
    );
}

#[test]
fn test_thermal_mass_coupling_mode_specific_disabled() {
    let spec = ASHRAE140Case::Case900.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Verify coupling ratios achieve target >= 0.1 in both modes
    let h_tr_ms_value: f64 = model.h_tr_ms.as_ref()[0];
    let h_tr_em_heating_value: f64 = model.h_tr_em_heating.as_ref()[0];
    let h_tr_em_cooling_value: f64 = model.h_tr_em_cooling.as_ref()[0];

    // Heating coupling ratio (no factor applied since mode-specific coupling is disabled)
    let heating_ratio = h_tr_em_heating_value / h_tr_ms_value;

    // Cooling coupling ratio (no factor applied since mode-specific coupling is disabled)
    let cooling_ratio = h_tr_em_cooling_value / h_tr_ms_value;

    // Both ratios should be >= 0.1 (target)
    assert!(
        heating_ratio >= 0.1,
        "Heating coupling ratio {} should be >= 0.1",
        heating_ratio
    );
    assert!(
        cooling_ratio >= 0.1,
        "Cooling coupling ratio {} should be >= 0.1",
        cooling_ratio
    );

    // Verify mode-specific factors are disabled (set to 1.0)
    let heating_factor = model.h_tr_em_heating_factor;
    let cooling_factor = model.h_tr_em_cooling_factor;

    assert_eq!(
        heating_factor, 1.0,
        "Heating factor should be 1.0 (disabled), got {}",
        heating_factor
    );
    assert_eq!(
        cooling_factor, 1.0,
        "Cooling factor should be 1.0 (disabled), got {}",
        cooling_factor
    );

    // Verify heating and cooling values are equal (no mode-specific difference)
    assert_eq!(
        h_tr_em_heating_value, h_tr_em_cooling_value,
        "Heating and cooling coupling should be equal when mode-specific coupling is disabled"
    );

    println!("DEBUG: Thermal mass correction with mode-specific coupling disabled:");
    println!("  Heating factor: {:.2} (disabled)", heating_factor);
    println!("  Cooling factor: {:.2} (disabled)", cooling_factor);
    println!("  Heating coupling ratio: {:.3}", heating_ratio);
    println!("  Cooling coupling ratio: {:.3}", cooling_ratio);
}

#[test]
fn test_thermal_mass_correction_low_mass_unchanged() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);

    // Calculate initial h_tr_em values
    let h_tr_em_initial: f64 = model.h_tr_em.as_ref()[0];
    let h_tr_em_heating_initial: f64 = model.h_tr_em_heating.as_ref()[0];
    let h_tr_em_cooling_initial: f64 = model.h_tr_em_cooling.as_ref()[0];

    // Apply thermal mass correction (should exit early for low-mass)
    model.apply_thermal_mass_correction();

    // Verify values unchanged (low-mass should not be corrected)
    let h_tr_em_final: f64 = model.h_tr_em.as_ref()[0];
    let h_tr_em_heating_final: f64 = model.h_tr_em_heating.as_ref()[0];
    let h_tr_em_cooling_final: f64 = model.h_tr_em_cooling.as_ref()[0];

    assert_eq!(
        h_tr_em_initial, h_tr_em_final,
        "Low-mass base h_tr_em should not change"
    );
    assert_eq!(
        h_tr_em_heating_initial, h_tr_em_heating_final,
        "Low-mass heating h_tr_em should not change"
    );
    assert_eq!(
        h_tr_em_cooling_initial, h_tr_em_cooling_final,
        "Low-mass cooling h_tr_em should not change"
    );
}
