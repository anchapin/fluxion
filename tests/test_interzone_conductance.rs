//! Unit tests for inter-zone conductance calculation.
//!
//! Tests validate the formula h_tr_iz = A_common / R_common_wall
//! for thermal conductance between adjacent building zones.

use fluxion::sim::construction::Assemblies;
use fluxion::sim::interzone::{
    calculate_directional_interzone_conductance, calculate_interzone_conductance,
};

/// Common wall area from Case 960 specification (8m x 2.7m)
const COMMON_WALL_AREA: f64 = 21.6; // m²

/// Concrete wall thickness for Case 960 (200mm)
const CONCRETE_THICKNESS: f64 = 0.200; // m

/// Expected R-value for 200mm concrete wall (materials only)
/// R = thickness / k = 0.200 / 1.13 = 0.177 m²K/W for concrete
const CONCRETE_R_VALUE: f64 = 0.177; // m²K/W

/// Expected conductance from first principles: h = A / R
const EXPECTED_CONDUCTANCE: f64 = COMMON_WALL_AREA / CONCRETE_R_VALUE; // W/K

/// Tolerance for numerical comparison (1%)
const TOLERANCE_PCT: f64 = 1.0;

#[test]
fn test_interzone_conductance_from_first_principles() {
    // Test conductance calculation from first principles
    // h_tr_iz = A_common / R_common_wall

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);
    let h_tr_iz = calculate_interzone_conductance(COMMON_WALL_AREA, &wall);

    // Expected: h = 21.6 / 0.14 = 154.3 W/K
    let expected = EXPECTED_CONDUCTANCE;

    println!("Inter-zone conductance: {:.2} W/K", h_tr_iz);
    println!("Expected: {:.2} W/K", expected);
    println!(
        "Difference: {:.2}%",
        ((h_tr_iz - expected).abs() / expected) * 100.0
    );

    assert!(
        (h_tr_iz - expected).abs() / expected * 100.0 < TOLERANCE_PCT,
        "Conductance {:.2} W/K differs from expected {:.2} W/K by more than {}%",
        h_tr_iz,
        expected,
        TOLERANCE_PCT
    );
}

#[test]
fn test_interzone_conductance_case_960_spec() {
    // Test with exact Case 960 specifications
    // Verify that the implementation matches the documented spec

    let wall = Assemblies::concrete_wall(0.200);
    let area = 21.6; // Common wall area for Case 960

    let h_tr_iz = calculate_interzone_conductance(area, &wall);

    // Case 960 spec: h = A / R = 21.6 / 0.177 = 122.0 W/K
    // (R = thickness / k = 0.200 / 1.13 = 0.177 m²K/W for concrete)
    let expected = 122.0;

    assert!(
        (h_tr_iz - expected).abs() < 1.0,
        "Expected ~122.0 W/K for Case 960, got {:.2} W/K",
        h_tr_iz
    );
}

#[test]
fn test_directional_conductance_asymmetric() {
    // Test directional conductance for asymmetric insulation
    // This is critical for sunspace/back-zone configurations

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);

    // Asymmetric insulation:
    // - Zone 0 (back-zone): R = 2.0 m²K/W (insulation facing back-zone)
    // - Zone 1 (sunspace): R = 0.0 m²K/W (no insulation facing sunspace)
    let r_insulation_zone_0 = 2.0;
    let r_insulation_zone_1 = 0.0;

    let (h_iz_0_to_1, h_iz_1_to_0) = calculate_directional_interzone_conductance(
        COMMON_WALL_AREA,
        &wall,
        r_insulation_zone_0,
        r_insulation_zone_1,
    );

    // Expected calculations:
    // h_iz_0_to_1 = 21.6 / (0.14 + 2.0) = 10.0 W/K
    // h_iz_1_to_0 = 21.6 / (0.14 + 0.0) = 154.3 W/K
    let expected_h_0_to_1 = COMMON_WALL_AREA / (CONCRETE_R_VALUE + r_insulation_zone_0);
    let expected_h_1_to_0 = COMMON_WALL_AREA / (CONCRETE_R_VALUE + r_insulation_zone_1);

    println!(
        "h_iz_0_to_1: {:.2} W/K (expected {:.2} W/K)",
        h_iz_0_to_1, expected_h_0_to_1
    );
    println!(
        "h_iz_1_to_0: {:.2} W/K (expected {:.2} W/K)",
        h_iz_1_to_0, expected_h_1_to_0
    );
    println!("Ratio: {:.2}×", h_iz_1_to_0 / h_iz_0_to_1);

    assert!(
        (h_iz_0_to_1 - expected_h_0_to_1).abs() < 0.5,
        "Expected h_iz_0_to_1 ~{:.2} W/K, got {:.2} W/K",
        expected_h_0_to_1,
        h_iz_0_to_1
    );

    assert!(
        (h_iz_1_to_0 - expected_h_1_to_0).abs() < 1.0,
        "Expected h_iz_1_to_0 ~{:.2} W/K, got {:.2} W/K",
        expected_h_1_to_0,
        h_iz_1_to_0
    );

    // Verify physical meaning: insulation on zone 0 side reduces heat flow from zone 0 -> 1
    assert!(
        h_iz_0_to_1 < h_iz_1_to_0,
        "Insulation should reduce h_iz_0_to_1 relative to h_iz_1_to_0"
    );

    // Verify the ~12.3× difference due to insulation
    let ratio = h_iz_1_to_0 / h_iz_0_to_1;
    assert!(
        (ratio - 12.3).abs() < 1.0,
        "Expected ~12.3× ratio, got {:.1}×",
        ratio
    );
}

#[test]
fn test_directional_conductance_symmetric() {
    // Test that symmetric insulation produces equal directional values

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);

    // Symmetric insulation: R = 2.0 m²K/W on both sides
    let r_insulation = 2.0;

    let (h_iz_0_to_1, h_iz_1_to_0) = calculate_directional_interzone_conductance(
        COMMON_WALL_AREA,
        &wall,
        r_insulation,
        r_insulation,
    );

    // Both should equal: h = 21.6 / (0.14 + 2.0) = 10.0 W/K
    let expected = COMMON_WALL_AREA / (CONCRETE_R_VALUE + r_insulation);

    assert!(
        (h_iz_0_to_1 - expected).abs() < 0.5,
        "Expected h_iz_0_to_1 ~{:.2} W/K, got {:.2} W/K",
        expected,
        h_iz_0_to_1
    );

    assert!(
        (h_iz_1_to_0 - expected).abs() < 0.5,
        "Expected h_iz_1_to_0 ~{:.2} W/K, got {:.2} W/K",
        expected,
        h_iz_1_to_0
    );

    // Verify equality for symmetric construction
    assert!(
        (h_iz_0_to_1 - h_iz_1_to_0).abs() < 0.1,
        "Symmetric insulation should produce equal conductances"
    );
}

#[test]
fn test_directional_conductance_no_insulation() {
    // Test that no additional insulation reduces to single conductance

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);

    // No additional insulation on either side
    let (h_iz_0_to_1, h_iz_1_to_0) =
        calculate_directional_interzone_conductance(COMMON_WALL_AREA, &wall, 0.0, 0.0);

    // Both should equal single-directional calculation
    let expected = calculate_interzone_conductance(COMMON_WALL_AREA, &wall);

    assert!(
        (h_iz_0_to_1 - expected).abs() < 1.0,
        "Expected h_iz_0_to_1 ~{:.2} W/K, got {:.2} W/K",
        expected,
        h_iz_0_to_1
    );

    assert!(
        (h_iz_1_to_0 - expected).abs() < 1.0,
        "Expected h_iz_1_to_0 ~{:.2} W/K, got {:.2} W/K",
        expected,
        h_iz_1_to_0
    );

    assert!(
        (h_iz_0_to_1 - h_iz_1_to_0).abs() < 0.1,
        "No insulation should produce equal conductances"
    );
}

#[test]
fn test_interzone_conductance_zero_area() {
    // Test edge case: zero area should give zero conductance

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);
    let h_tr_iz = calculate_interzone_conductance(0.0, &wall);

    assert_eq!(h_tr_iz, 0.0, "Zero area should give zero conductance");
}

#[test]
fn test_interzone_conductance_negative_area() {
    // Test edge case: negative area should give negative conductance (physically invalid but mathematically correct)

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);
    let h_tr_iz = calculate_interzone_conductance(-10.0, &wall);

    assert!(
        h_tr_iz < 0.0,
        "Negative area should give negative conductance"
    );
}
