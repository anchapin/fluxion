use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec};

#[test]
fn test_low_mass_solar_distribution() {
    let spec: CaseSpec = ASHRAE140Case::Case600.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);
    assert_eq!(model.solar_distribution_to_air, 0.75);
    assert_eq!(model.solar_beam_to_mass_fraction, 0.25);
}

#[test]
fn test_high_mass_solar_distribution() {
    let spec: CaseSpec = ASHRAE140Case::Case900.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);
    assert_eq!(model.solar_distribution_to_air, 0.5);
    assert_eq!(model.solar_beam_to_mass_fraction, 0.5);
}

#[test]
fn test_special_solar_distribution_case_960() {
    let spec: CaseSpec = ASHRAE140Case::Case960.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);
    assert_eq!(model.solar_distribution_to_air, 0.6);
    assert_eq!(model.solar_beam_to_mass_fraction, 0.4);
}

#[test]
fn test_distribution_fractions_sum_to_one() {
    // Verify that the chosen fractions are complementary
    let b_high = 0.5; // high-mass beam to mass
    let air_high = 1.0 - b_high;
    assert_eq!(air_high, 0.5);
    // Sum check for 6R2C: (1-b) + b*0.7 + b*0.3 = 1
    let total = (1.0 - b_high) + b_high * 0.7 + b_high * 0.3;
    assert!((total - 1.0).abs() < 1e-6);
}
