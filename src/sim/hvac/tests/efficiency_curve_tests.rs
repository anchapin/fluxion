//! Unit tests for efficiency curve models

use fluxion::sim::hvac::efficiency_curves::{default_ahri_coefficients, EfficiencyCurve};

#[test]
fn test_polynomial_efficiency_curves() {
    // Test cubic polynomial evaluation
    let coeffs = [3.5, 0.0, 0.0, 0.0];
    let curve = EfficiencyCurve::new(coeffs, 0.02, -5.0);

    // Test at PLR = 1.0 (full load)
    let cop_full_load = curve.cop_at(1.0, -5.0);
    assert!((cop_full_load - 3.5).abs() < 0.1); // Constant COP

    // Test at PLR = 0.5 (part load)
    let cop_part_load = curve.cop_at(0.5, -5.0);
    assert!((cop_part_load - 3.5).abs() < 0.1); // Same COP at part load

    // Test at PLR = 0.0 (no load)
    let cop_no_load = curve.cop_at(0.0, -5.0);
    assert!((cop_no_load - 3.5).abs() < 0.1); // Same COP at no load

    // Test temperature degradation
    let cop_design_temp = curve.cop_at(1.0, -5.0);
    let cop_cold_temp = curve.cop_at(1.0, -15.0);
    assert!(cop_cold_temp < cop_design_temp); // Degraded at cold temp

    // Test minimum COP (30% of rated)
    let cop_extreme_temp = curve.cop_at(1.0, -50.0);
    assert!(cop_extreme_temp >= 3.5 * 0.3); // Minimum 30% of rated
}

#[test]
fn test_horner_method_evaluation() {
    // Verify Horner's method matches direct evaluation
    let coeffs = [1.0, 2.0, 3.0, 4.0];
    let curve = EfficiencyCurve::new(coeffs, 0.0, 0.0);

    // Direct evaluation: 1 + 2*x + 3*x² + 4*x³
    let x = 2.0;
    let direct = 1.0 + 2.0 * x + 3.0 * x.powi(2) + 4.0 * x.powi(3);

    // Horner's method: ((4*x + 3)*x + 2)*x + 1
    let horner = curve.evaluate_polynomial(x);

    assert!((direct - horner).abs() < 1e-10); // Should match exactly
}

#[test]
fn test_ahri_coefficient_loading() {
    // Test default AHRI coefficients
    let config = default_ahri_coefficients();

    // Verify heat pump coefficients
    assert_eq!(config.heatpump_heating.plr.len(), 4); // 4 coefficients
    assert_eq!(config.heatpump_heating.design_temp, -5.0);
    assert_eq!(config.heatpump_cooling.design_temp, 35.0);

    // Verify chiller coefficients
    assert_eq!(config.chiller.plr.len(), 4);
    assert_eq!(config.chiller.design_temp, 35.0);

    // Verify boiler coefficients
    assert_eq!(config.boiler.plr.len(), 4);
    assert_eq!(config.boiler.design_temp, -5.0);

    // Create efficiency curves from AHRI coefficients
    let hp_heating_curve: EfficiencyCurve = (&config.heatpump_heating).into();
    let cop_at_design = hp_heating_curve.cop_at(1.0, -5.0);
    assert!((cop_at_design - 3.5).abs() < 0.1); // Rated COP 3.5
}

#[test]
fn test_efficiency_curve_constant() {
    // Test that constant polynomial produces same COP at all PLR values
    let coeffs = [3.5, 0.0, 0.0, 0.0];
    let curve = EfficiencyCurve::new(coeffs, 0.0, 0.0);

    let cop_100 = curve.cop_at(1.0, 0.0);
    let cop_75 = curve.cop_at(0.75, 0.0);
    let cop_50 = curve.cop_at(0.5, 0.0);
    let cop_25 = curve.cop_at(0.25, 0.0);
    let cop_0 = curve.cop_at(0.0, 0.0);

    // Constant: same COP at all PLR values
    assert_eq!(cop_0, 3.5);
    assert_eq!(cop_0, cop_100);
    assert_eq!(cop_25, cop_50);
    assert_eq!(cop_50, cop_75);
}

#[test]
fn test_temperature_coefficient() {
    // Test that temperature coefficient degrades COP linearly
    let coeffs = [3.5, 0.0, 0.0, 0.0]; // No PLR degradation
    let temp_coeff = 0.02; // 2% per degree
    let curve = EfficiencyCurve::new(coeffs, temp_coeff, -5.0);

    let cop_design = curve.cop_at(1.0, -5.0);
    let cop_5_deg_colder = curve.cop_at(1.0, -10.0);
    let cop_10_deg_colder = curve.cop_at(1.0, -15.0);

    // Linear degradation: 2% per degree
    let degradation_5 = (cop_design - cop_5_deg_colder) / cop_design;
    let degradation_10 = (cop_design - cop_10_deg_colder) / cop_design;

    assert!((degradation_5 - 0.10).abs() < 0.01); // ~10% at 5°C colder
    assert!((degradation_10 - 0.20).abs() < 0.01); // ~20% at 10°C colder
}
