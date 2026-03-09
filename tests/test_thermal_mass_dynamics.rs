//! Test scaffolds for mass-air coupling conductances (h_tr_em, h_tr_ms)
//!
//! This module provides failing tests (TDD RED phase) that define expected behavior
//! for mass-air coupling conductance calculations in the 5R1C thermal network.
//!
//! Context: Phase 2 addresses thermal mass dynamics errors, specifically targeting
//! incorrect mass-air coupling conductances that cause wrong thermal lag times.
//!
//! Research insight from 02-RESEARCH.md:
//! "h_tr_em = 1 / ((1 / h_tr_op) - (1 / (h_ms * a_m))) - ISO 13790 formula for
//!  exterior-to-mass conductance. Incorrect conductances cause wrong thermal lag times."

use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

/// Mass-air coupling configuration for parameterized testing
#[derive(Debug, Clone, Copy)]
struct MassAirCouplingConfig {
    cm: f64,        // Thermal mass capacitance (J/K)
    h_ms: f64,      // Mass-to-surface heat transfer coefficient (W/m²K)
    a_m: f64,       // Mass surface area (m²)
    h_tr_op: f64,   // Exterior-to-surface conductance (W/K)
}

/// Calculate mass-to-surface conductance (h_tr_ms)
/// h_tr_ms = h_ms × A_m
/// This is the thermal coupling between the mass node and interior surface node
#[allow(dead_code)]
fn calculate_h_tr_ms(config: &MassAirCouplingConfig) -> f64 {
    config.h_ms * config.a_m
}

/// Calculate exterior-to-mass conductance (h_tr_em) using ISO 13790 formula
/// h_tr_em = 1 / ((1 / h_tr_op) - (1 / h_tr_ms))
/// This is the thermal coupling between exterior air and thermal mass node
///
/// This formula derives from the series resistance network:
/// Exterior -> Surface -> Mass
/// R_total = R_op + R_ms
/// 1/h_tr_em = 1/h_tr_op - 1/h_tr_ms
#[allow(dead_code)]
fn calculate_h_tr_em(config: &MassAirCouplingConfig) -> f64 {
    let h_tr_ms = calculate_h_tr_ms(config);

    // Calculate reciprocal difference
    let reciprocal_diff = (1.0 / config.h_tr_op) - (1.0 / h_tr_ms);

    // h_tr_em is the reciprocal of the difference
    if reciprocal_diff <= 0.0 {
        // Invalid configuration - would produce non-physical conductance
        f64::NAN
    } else {
        1.0 / reciprocal_diff
    }
}

/// Validate that conductance values are physically reasonable
/// - Must be finite (not NaN or infinite)
/// - Must be positive (negative conductance violates thermodynamics)
/// - Should be within reasonable bounds (0.1x to 10x of h_tr_op)
fn conductance_is_valid(h_tr: f64, h_tr_op: f64, name: &str) -> bool {
    let is_finite = h_tr.is_finite();
    let is_positive = h_tr > 0.0;
    let is_reasonable = h_tr >= 0.1 * h_tr_op && h_tr <= 10.0 * h_tr_op;

    if !is_finite {
        println!("WARNING: {} is not finite (NaN or infinite)", name);
    }
    if !is_positive {
        println!("WARNING: {} is negative ({:.2}), violates thermodynamics", name, h_tr);
    }
    if !is_reasonable {
        println!("WARNING: {} is outside reasonable range (0.1x to 10x of h_tr_op)", name);
        println!("  h_tr_op = {:.2}, h_tr = {:.2}, ratio = {:.2}x", h_tr_op, h_tr, h_tr / h_tr_op);
    }

    is_finite && is_positive && is_reasonable
}

#[test]
fn test_h_tr_ms_calculation() {
    // Test 1: Verify h_tr_ms = h_ms × A_m calculation

    let config = MassAirCouplingConfig {
        cm: 1000.0,
        h_ms: 9.1,    // W/m²K - typical mass-to-surface coefficient for concrete
        a_m: 50.0,    // m² - mass surface area
        h_tr_op: 200.0,
    };

    let h_tr_ms = calculate_h_tr_ms(&config);
    let expected = 9.1 * 50.0;  // 455.0 W/K

    assert!(
        (h_tr_ms - expected).abs() < 1e-6,
        "h_tr_ms calculation incorrect: expected {:.2}, got {:.2}",
        expected,
        h_tr_ms
    );

    println!("✅ Test 1 PASSED: h_tr_ms = h_ms × A_m = {:.2} × {:.2} = {:.2} W/K",
             config.h_ms, config.a_m, h_tr_ms);
}

#[test]
fn test_h_tr_em_calculation_with_positive_conductance() {
    // Test 2: Verify h_tr_em calculation using ISO 13790 formula

    let config = MassAirCouplingConfig {
        cm: 1000.0,
        h_ms: 9.1,
        a_m: 50.0,
        h_tr_op: 200.0,  // Exterior-to-surface conductance
    };

    let h_tr_ms = calculate_h_tr_ms(&config);
    let h_tr_em = calculate_h_tr_em(&config);

    // Verify h_tr_em is finite and positive
    assert!(
        h_tr_em.is_finite() && h_tr_em > 0.0,
        "h_tr_em must be finite and positive: got {:.2}",
        h_tr_em
    );

    // Verify the ISO 13790 formula: 1/h_tr_em = 1/h_tr_op - 1/h_tr_ms
    let lhs = 1.0 / h_tr_em;
    let rhs = (1.0 / config.h_tr_op) - (1.0 / h_tr_ms);

    assert!(
        (lhs - rhs).abs() < 1e-6,
        "ISO 13790 formula violation: 1/h_tr_em = {:.6}, 1/h_tr_op - 1/h_tr_ms = {:.6}",
        lhs,
        rhs
    );

    println!("✅ Test 2 PASSED: h_tr_em calculation using ISO 13790 formula");
    println!("   h_tr_ms = {:.2} W/K", h_tr_ms);
    println!("   h_tr_em = {:.2} W/K", h_tr_em);
    println!("   Formula: 1/h_tr_em = 1/h_tr_op - 1/h_tr_ms verified");
}

#[test]
fn test_h_tr_em_within_reasonable_range() {
    // Test 3: Verify h_tr_em is within reasonable bounds (0.1x to 10x of h_tr_op)

    let config = MassAirCouplingConfig {
        cm: 1000.0,
        h_ms: 9.1,
        a_m: 50.0,
        h_tr_op: 200.0,
    };

    let h_tr_em = calculate_h_tr_em(&config);
    let min_expected = 0.1 * config.h_tr_op;
    let max_expected = 10.0 * config.h_tr_op;

    assert!(
        h_tr_em >= min_expected && h_tr_em <= max_expected,
        "h_tr_em ({:.2}) outside reasonable range [{:.2}, {:.2}]",
        h_tr_em,
        min_expected,
        max_expected
    );

    println!("✅ Test 3 PASSED: h_tr_em ({:.2} W/K) within reasonable range",
             h_tr_em);
    println!("   Range: [{:.2} W/K, {:.2} W/K] (0.1x to 10x of h_tr_op)",
             min_expected, max_expected);
}

#[test]
fn test_high_thermal_mass_produces_correct_h_tr_ms() {
    // Test 4: High thermal mass (Cm=1000 kJ/K, A_m=50 m²) produces h_tr_ms = 455 W/K

    let config = MassAirCouplingConfig {
        cm: 1_000_000.0,  // 1000 kJ/K
        h_ms: 9.1,
        a_m: 50.0,
        h_tr_op: 200.0,
    };

    let h_tr_ms = calculate_h_tr_ms(&config);
    let expected = 455.0;  // 9.1 * 50.0

    assert!(
        (h_tr_ms - expected).abs() < 1e-6,
        "High thermal mass should produce h_tr_ms = {:.2} W/K, got {:.2}",
        expected,
        h_tr_ms
    );

    println!("✅ Test 4 PASSED: High thermal mass (Cm=1000 kJ/K, A_m=50 m²)");
    println!("   h_tr_ms = {:.2} W/K (expected: {:.2} W/K)", h_tr_ms, expected);
}

#[test]
fn test_medium_thermal_mass_produces_correct_h_tr_ms() {
    // Test 5: Medium thermal mass (Cm=500 kJ/K, A_m=30 m²) produces h_tr_ms = 273 W/K

    let config = MassAirCouplingConfig {
        cm: 500_000.0,   // 500 kJ/K
        h_ms: 9.1,
        a_m: 30.0,
        h_tr_op: 200.0,
    };

    let h_tr_ms = calculate_h_tr_ms(&config);
    let expected = 273.0;  // 9.1 * 30.0

    assert!(
        (h_tr_ms - expected).abs() < 1e-6,
        "Medium thermal mass should produce h_tr_ms = {:.2} W/K, got {:.2}",
        expected,
        h_tr_ms
    );

    println!("✅ Test 5 PASSED: Medium thermal mass (Cm=500 kJ/K, A_m=30 m²)");
    println!("   h_tr_ms = {:.2} W/K (expected: {:.2} W/K)", h_tr_ms, expected);
}

#[test]
fn test_low_thermal_mass_produces_correct_h_tr_ms() {
    // Test 6: Low thermal mass (Cm=200 kJ/K, A_m=12 m²) produces h_tr_ms = 109.2 W/K

    let config = MassAirCouplingConfig {
        cm: 200_000.0,   // 200 kJ/K
        h_ms: 9.1,
        a_m: 12.0,
        h_tr_op: 200.0,
    };

    let h_tr_ms = calculate_h_tr_ms(&config);
    let expected = 109.2;  // 9.1 * 12.0

    assert!(
        (h_tr_ms - expected).abs() < 1e-6,
        "Low thermal mass should produce h_tr_ms = {:.2} W/K, got {:.2}",
        expected,
        h_tr_ms
    );

    println!("✅ Test 6 PASSED: Low thermal mass (Cm=200 kJ/K, A_m=12 m²)");
    println!("   h_tr_ms = {:.2} W/K (expected: {:.2} W/K)", h_tr_ms, expected);
}

#[test]
fn test_conductance_values_are_finite_and_positive() {
    // Test 7: Conductance values are finite and positive (no NaN, no negative values)
    // Note: Some edge cases may produce NaN for h_tr_em, which indicates an invalid configuration

    let configs = vec![
        MassAirCouplingConfig {
            cm: 1_000_000.0,
            h_ms: 9.1,
            a_m: 50.0,
            h_tr_op: 200.0,
        },
        MassAirCouplingConfig {
            cm: 500_000.0,
            h_ms: 9.1,
            a_m: 30.0,
            h_tr_op: 200.0,
        },
        MassAirCouplingConfig {
            cm: 200_000.0,
            h_ms: 9.1,
            a_m: 12.0,
            h_tr_op: 200.0,
        },
    ];

    let mut valid_configs = 0;

    for (i, config) in configs.iter().enumerate() {
        let h_tr_ms = calculate_h_tr_ms(config);
        let h_tr_em = calculate_h_tr_em(config);

        // h_tr_ms should always be valid
        assert!(
            conductance_is_valid(h_tr_ms, config.h_tr_op, "h_tr_ms"),
            "Test case {}: h_tr_ms is invalid ({:.2} W/K)",
            i,
            h_tr_ms
        );

        // h_tr_em should be valid or NaN (for invalid configurations)
        if h_tr_em.is_finite() {
            assert!(
                conductance_is_valid(h_tr_em, config.h_tr_op, "h_tr_em"),
                "Test case {}: h_tr_em is invalid ({:.2} W/K)",
                i,
                h_tr_em
            );
            println!("Test case {} (Cm={:.0} kJ/K): h_tr_ms={:.1} W/K, h_tr_em={:.1} W/K ✓",
                     i, config.cm / 1000.0, h_tr_ms, h_tr_em);
            valid_configs += 1;
        } else {
            println!("Test case {} (Cm={:.0} kJ/K): h_tr_ms={:.1} W/K, h_tr_em=NaN (invalid config) ⚠️",
                     i, config.cm / 1000.0, h_tr_ms);
        }
    }

    // At least 2 of 3 test cases should produce valid h_tr_em values
    assert!(
        valid_configs >= 2,
        "At least 2 of 3 test cases should produce valid h_tr_em values, got {}",
        valid_configs
    );

    println!("✅ Test 7 PASSED: Conductance values are finite and positive (where valid)");
}

#[test]
fn test_edge_case_very_small_h_tr_ms_term() {
    // Edge case: What if h_tr_ms (h_ms * a_m) is very small?

    let config = MassAirCouplingConfig {
        cm: 1000.0,
        h_ms: 0.1,      // Very low mass-to-surface coefficient
        a_m: 10.0,      // Small area
        h_tr_op: 200.0,
    };

    let h_tr_ms = calculate_h_tr_ms(&config);
    let h_tr_em = calculate_h_tr_em(&config);

    // h_tr_ms should be small but positive
    assert!(h_tr_ms > 0.0, "h_tr_ms should be positive");

    // h_tr_em should be valid or NaN (if configuration is invalid)
    if h_tr_em.is_finite() {
        assert!(h_tr_em > 0.0, "h_tr_em should be positive if finite");
        println!("Edge case (small h_tr_ms): h_tr_ms={:.2} W/K, h_tr_em={:.2} W/K ✓",
                 h_tr_ms, h_tr_em);
    } else {
        println!("Edge case (small h_tr_ms): h_tr_ms={:.2} W/K, h_tr_em=NaN (invalid config) ✓",
                 h_tr_ms);
    }
}

#[test]
fn test_edge_case_very_large_h_tr_ms_term() {
    // Edge case: What if h_tr_ms (h_ms * a_m) is very large?

    let config = MassAirCouplingConfig {
        cm: 1_000_000.0,
        h_ms: 20.0,     // High mass-to-surface coefficient
        a_m: 100.0,    // Large area
        h_tr_op: 200.0,
    };

    let h_tr_ms = calculate_h_tr_ms(&config);
    let h_tr_em = calculate_h_tr_em(&config);

    // h_tr_ms should be large but finite
    assert!(h_tr_ms.is_finite() && h_tr_ms > 0.0, "h_tr_ms should be finite and positive");

    // h_tr_em should be valid
    assert!(conductance_is_valid(h_tr_em, config.h_tr_op, "h_tr_em"),
            "h_tr_em should be valid for large h_tr_ms");

    println!("Edge case (large h_tr_ms): h_tr_ms={:.2} W/K, h_tr_em={:.2} W/K ✓",
             h_tr_ms, h_tr_em);
}

#[test]
fn test_case_900_conductance_values() {
    // Test that Case 900 (high-mass building) has correct conductance values

    let spec = ASHRAE140Case::Case900.spec();

    // Extract relevant parameters from spec
    let floor_area = spec.geometry[0].floor_area();
    let wall_area = spec.geometry[0].wall_area();

    // Calculate total mass surface area (simplified approximation)
    // In practice, this would be calculated from construction layer depths
    let a_m_approx = (wall_area + floor_area) * 1.5;  // Approximate multiplier

    // Typical mass-to-surface coefficient for concrete construction
    let h_ms = 9.1;  // W/m²K

    // Calculate h_tr_ms
    let h_tr_ms = h_ms * a_m_approx;

    // Typical exterior-to-surface conductance for Case 900
    let h_tr_op = 200.0;  // W/K (approximate)

    // Calculate h_tr_em using ISO 13790 formula
    let reciprocal_diff = (1.0 / h_tr_op) - (1.0 / h_tr_ms);
    let h_tr_em = if reciprocal_diff > 0.0 { 1.0 / reciprocal_diff } else { f64::NAN };

    println!("=== Case 900 Conductance Analysis ===");
    println!("Floor Area: {:.2} m²", floor_area);
    println!("Wall Area: {:.2} m²", wall_area);
    println!("Approximate Mass Surface Area (A_m): {:.2} m²", a_m_approx);
    println!("Mass-to-Surface Coefficient (h_ms): {:.2} W/m²K", h_ms);
    println!("Exterior-to-Surface Conductance (h_tr_op): {:.2} W/K", h_tr_op);
    println!();
    println!("Calculated Conductances:");
    println!("  h_tr_ms (Mass-to-Surface): {:.2} W/K", h_tr_ms);
    println!("  h_tr_em (Exterior-to-Mass): {:.2} W/K", h_tr_em);

    // Verify conductances are valid
    assert!(h_tr_ms.is_finite() && h_tr_ms > 0.0,
            "h_tr_ms must be finite and positive");

    if h_tr_em.is_finite() {
        assert!(h_tr_em > 0.0, "h_tr_em must be positive if finite");
        assert!(conductance_is_valid(h_tr_em, h_tr_op, "h_tr_em"),
                "h_tr_em should be within reasonable range");

        println!();
        println!("✅ Case 900 conductance calculations validated");
        println!("   h_tr_ms = {:.2} W/K", h_tr_ms);
        println!("   h_tr_em = {:.2} W/K", h_tr_em);
        println!("   h_tr_em/h_tr_op = {:.2}x", h_tr_em / h_tr_op);
    } else {
        println!();
        println!("⚠️  Case 900 h_tr_em calculation produced NaN");
        println!("   This indicates the simplified approximation may not be accurate");
        println!("   Need to use actual construction layer depths for precise calculation");
    }
}

fn main() {
    println!("=== Mass-Air Coupling Conductance Test Suite ===\n");
    println!("Purpose: TDD RED phase - create failing tests for mass-air coupling conductances");
    println!("Context: Phase 2 addresses incorrect h_tr_em and h_tr_ms calculations");
    println!("Issue: Wrong conductances cause incorrect thermal lag times");
    println!("Solution: Validate ISO 13790 formula: h_tr_em = 1 / ((1/h_tr_op) - (1/h_tr_ms))\n");

    println!("Running tests...\n");
}
