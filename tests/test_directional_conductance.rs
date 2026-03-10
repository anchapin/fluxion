//! Unit tests for bidirectional inter-zone conductance (directional h_tr_iz).
//!
//! Tests validate asymmetric insulation handling and directionality
//! in inter-zone heat transfer calculations.

use fluxion::sim::construction::Assemblies;
use fluxion::sim::interzone::calculate_directional_interzone_conductance;

/// Common wall area from Case 960 specification (8m x 2.7m)
const COMMON_WALL_AREA: f64 = 21.6; // m²

/// Concrete wall thickness for Case 960 (200mm)
const CONCRETE_THICKNESS: f64 = 0.200; // m

/// Expected R-value for 200mm concrete wall (materials only)
const CONCRETE_R_VALUE: f64 = 0.177; // m²K/W

/// Tolerance for numerical comparison (1%)
const TOLERANCE_PCT: f64 = 1.0;

#[test]
fn test_h_iz_0_to_1() {
    // Test bidirectional conductance with asymmetric insulation
    // Zone 0 (back-zone): Heavy insulation
    // Zone 1 (sunspace): No insulation

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
    // h_iz_0_to_1 = A / (R_base + R_insulation_0) = 21.6 / (0.177 + 2.0) = 9.92 W/K
    // h_iz_1_to_0 = A / (R_base + R_insulation_1) = 21.6 / (0.177 + 0.0) = 122.0 W/K
    let expected_h_0_to_1 = COMMON_WALL_AREA / (CONCRETE_R_VALUE + r_insulation_zone_0);
    let expected_h_1_to_0 = COMMON_WALL_AREA / (CONCRETE_R_VALUE + r_insulation_zone_1);

    println!("h_iz_0_to_1: {:.2} W/K (expected {:.2} W/K)", h_iz_0_to_1, expected_h_0_to_1);
    println!("h_iz_1_to_0: {:.2} W/K (expected {:.2} W/K)", h_iz_1_to_0, expected_h_1_to_0);
    println!("Ratio: {:.2}×", h_iz_1_to_0 / h_iz_0_to_1);

    assert!(
        (h_iz_0_to_1 - expected_h_0_to_1).abs() / expected_h_0_to_1 * 100.0 < TOLERANCE_PCT,
        "Expected h_iz_0_to_1 ~{:.2} W/K, got {:.2} W/K",
        expected_h_0_to_1, h_iz_0_to_1
    );

    assert!(
        (h_iz_1_to_0 - expected_h_1_to_0).abs() / expected_h_1_to_0 * 100.0 < TOLERANCE_PCT,
        "Expected h_iz_1_to_0 ~{:.2} W/K, got {:.2} W/K",
        expected_h_1_to_0, h_iz_1_to_0
    );

    // Verify physical meaning: insulation on zone 0 side reduces heat flow from zone 0 -> 1
    assert!(
        h_iz_0_to_1 < h_iz_1_to_0,
        "Insulation should reduce h_iz_0_to_1 relative to h_iz_1_to_0"
    );

    // Verify the ~12.3× difference due to insulation
    let ratio = h_iz_1_to_0 / h_iz_0_to_1;
    let expected_ratio = (CONCRETE_R_VALUE + r_insulation_zone_0) / (CONCRETE_R_VALUE + r_insulation_zone_1);

    assert!(
        (ratio - expected_ratio).abs() < 0.5,
        "Expected ~{:.1}× ratio, got {:.1}×",
        expected_ratio, ratio
    );
}

#[test]
fn test_h_iz_1_to_0() {
    // Test bidirectional conductance with asymmetric insulation (reverse direction)
    // Zone 0 (back-zone): Heavy insulation
    // Zone 1 (sunspace): No insulation
    // This test verifies h_iz_1_to_0 specifically

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);

    let r_insulation_zone_0 = 2.0;
    let r_insulation_zone_1 = 0.0;

    let (h_iz_0_to_1, h_iz_1_to_0) = calculate_directional_interzone_conductance(
        COMMON_WALL_AREA,
        &wall,
        r_insulation_zone_0,
        r_insulation_zone_1,
    );

    // h_iz_1_to_0 should be high (no insulation on sunspace side)
    let expected_h_1_to_0 = COMMON_WALL_AREA / CONCRETE_R_VALUE;

    println!("h_iz_1_to_0: {:.2} W/K (expected {:.2} W/K)", h_iz_1_to_0, expected_h_1_to_0);

    assert!(
        (h_iz_1_to_0 - expected_h_1_to_0).abs() / expected_h_1_to_0 * 100.0 < TOLERANCE_PCT,
        "Expected h_iz_1_to_0 ~{:.2} W/K, got {:.2} W/K",
        expected_h_1_to_0, h_iz_1_to_0
    );

    // Verify h_iz_1_to_0 >> h_iz_0_to_1 (sunspace to back-zone has high conductance)
    assert!(
        h_iz_1_to_0 > h_iz_0_to_1 * 10.0,
        "h_iz_1_to_0 should be > 10× h_iz_0_to_1 due to asymmetric insulation"
    );
}

#[test]
fn test_bidirectional_conductance_asymmetric() {
    // Test bidirectional conductance with asymmetric insulation
    // Base wall: 200mm concrete, R_base = 0.14 m²K/W, Area = 21.6 m²
    // Zone 0 (back-zone): Insulation R = 2.0 m²K/W on interior side
    // Zone 1 (sunspace): No additional insulation

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);

    // Asymmetric insulation
    let r_insulation_zone_0 = 2.0;
    let r_insulation_zone_1 = 0.0;

    let (h_iz_0_to_1, h_iz_1_to_0) = calculate_directional_interzone_conductance(
        COMMON_WALL_AREA,
        &wall,
        r_insulation_zone_0,
        r_insulation_zone_1,
    );

    // Expected calculations:
    // h_iz_0_to_1 = 21.6 / (0.177 + 2.0) = 9.92 W/K
    // h_iz_1_to_0 = 21.6 / (0.177 + 0.0) = 122.0 W/K
    let expected_h_0_to_1 = COMMON_WALL_AREA / (CONCRETE_R_VALUE + r_insulation_zone_0);
    let expected_h_1_to_0 = COMMON_WALL_AREA / (CONCRETE_R_VALUE + r_insulation_zone_1);

    println!("h_iz_0_to_1: {:.2} W/K (expected {:.2} W/K)", h_iz_0_to_1, expected_h_0_to_1);
    println!("h_iz_1_to_0: {:.2} W/K (expected {:.2} W/K)", h_iz_1_to_0, expected_h_1_to_0);
    println!("Ratio: {:.2}×", h_iz_1_to_0 / h_iz_0_to_1);

    assert!(
        (h_iz_0_to_1 - expected_h_0_to_1).abs() < 0.5,
        "Expected h_iz_0_to_1 ~{:.2} W/K, got {:.2} W/K",
        expected_h_0_to_1, h_iz_0_to_1
    );

    assert!(
        (h_iz_1_to_0 - expected_h_1_to_0).abs() < 1.0,
        "Expected h_iz_1_to_0 ~{:.2} W/K, got {:.2} W/K",
        expected_h_1_to_0, h_iz_1_to_0
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
fn test_symmetric_insulation() {
    // Test symmetric insulation (no directionality)
    // Both zones have same additional insulation R = 2.0 m²K/W

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);

    let r_insulation = 2.0;

    let (h_iz_0_to_1, h_iz_1_to_0) = calculate_directional_interzone_conductance(
        COMMON_WALL_AREA,
        &wall,
        r_insulation,
        r_insulation,
    );

    // Both should equal: h = 21.6 / (0.177 + 2.0) = 9.92 W/K
    let expected = COMMON_WALL_AREA / (CONCRETE_R_VALUE + r_insulation);

    assert!(
        (h_iz_0_to_1 - expected).abs() < 0.5,
        "Expected h_iz_0_to_1 ~{:.2} W/K, got {:.2} W/K",
        expected, h_iz_0_to_1
    );

    assert!(
        (h_iz_1_to_0 - expected).abs() < 0.5,
        "Expected h_iz_1_to_0 ~{:.2} W/K, got {:.2} W/K",
        expected, h_iz_1_to_0
    );

    // Verify equality for symmetric construction
    assert!(
        (h_iz_0_to_1 - h_iz_1_to_0).abs() < 0.1,
        "Symmetric insulation should produce equal conductances"
    );
}

#[test]
fn test_no_additional_insulation() {
    // Test no additional insulation (single conductance)
    // Both zones have R = 0.0 m²K/W (no additional insulation)

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);

    let (h_iz_0_to_1, h_iz_1_to_0) = calculate_directional_interzone_conductance(
        COMMON_WALL_AREA,
        &wall,
        0.0,
        0.0,
    );

    // Both should equal single-directional calculation
    let expected = COMMON_WALL_AREA / CONCRETE_R_VALUE;

    assert!(
        (h_iz_0_to_1 - expected).abs() < 1.0,
        "Expected h_iz_0_to_1 ~{:.2} W/K, got {:.2} W/K",
        expected, h_iz_0_to_1
    );

    assert!(
        (h_iz_1_to_0 - expected).abs() < 1.0,
        "Expected h_iz_1_to_0 ~{:.2} W/K, got {:.2} W/K",
        expected, h_iz_1_to_0
    );

    assert!(
        (h_iz_0_to_1 - h_iz_1_to_0).abs() < 0.1,
        "No insulation should produce equal conductances"
    );
}

#[test]
fn test_extreme_asymmetry() {
    // Test extreme asymmetry
    // Zone 0: Heavy insulation R = 5.0 m²K/W
    // Zone 1: No insulation R = 0.0 m²K/W

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);

    let r_insulation_zone_0 = 5.0;
    let r_insulation_zone_1 = 0.0;

    let (h_iz_0_to_1, h_iz_1_to_0) = calculate_directional_interzone_conductance(
        COMMON_WALL_AREA,
        &wall,
        r_insulation_zone_0,
        r_insulation_zone_1,
    );

    // h_iz_0_to_1 = 21.6 / (0.177 + 5.0) = 4.2 W/K (very low)
    // h_iz_1_to_0 = 21.6 / (0.177 + 0.0) = 122.0 W/K (high)
    let expected_h_0_to_1 = COMMON_WALL_AREA / (CONCRETE_R_VALUE + r_insulation_zone_0);
    let expected_h_1_to_0 = COMMON_WALL_AREA / (CONCRETE_R_VALUE + r_insulation_zone_1);

    println!("h_iz_0_to_1: {:.2} W/K (expected {:.2} W/K)", h_iz_0_to_1, expected_h_0_to_1);
    println!("h_iz_1_to_0: {:.2} W/K (expected {:.2} W/K)", h_iz_1_to_0, expected_h_1_to_0);
    println!("Ratio: {:.2}×", h_iz_1_to_0 / h_iz_0_to_1);

    assert!(
        (h_iz_0_to_1 - expected_h_0_to_1).abs() < 0.5,
        "Expected h_iz_0_to_1 ~{:.2} W/K, got {:.2} W/K",
        expected_h_0_to_1, h_iz_0_to_1
    );

    assert!(
        (h_iz_1_to_0 - expected_h_1_to_0).abs() < 1.0,
        "Expected h_iz_1_to_0 ~{:.2} W/K, got {:.2} W/K",
        expected_h_1_to_0, h_iz_1_to_0
    );

    // Verify extreme asymmetry (> 25× difference)
    let ratio = h_iz_1_to_0 / h_iz_0_to_1;
    assert!(
        ratio > 25.0,
        "Expected > 25× ratio for extreme asymmetry, got {:.1}×",
        ratio
    );
}

#[test]
fn test_moderate_asymmetry() {
    // Test moderate asymmetry
    // Zone 0: Moderate insulation R = 1.0 m²K/W
    // Zone 1: Light insulation R = 0.2 m²K/W

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);

    let r_insulation_zone_0 = 1.0;
    let r_insulation_zone_1 = 0.2;

    let (h_iz_0_to_1, h_iz_1_to_0) = calculate_directional_interzone_conductance(
        COMMON_WALL_AREA,
        &wall,
        r_insulation_zone_0,
        r_insulation_zone_1,
    );

    // h_iz_0_to_1 = 21.6 / (0.177 + 1.0) = 18.4 W/K
    // h_iz_1_to_0 = 21.6 / (0.177 + 0.2) = 56.0 W/K
    let expected_h_0_to_1 = COMMON_WALL_AREA / (CONCRETE_R_VALUE + r_insulation_zone_0);
    let expected_h_1_to_0 = COMMON_WALL_AREA / (CONCRETE_R_VALUE + r_insulation_zone_1);

    println!("h_iz_0_to_1: {:.2} W/K (expected {:.2} W/K)", h_iz_0_to_1, expected_h_0_to_1);
    println!("h_iz_1_to_0: {:.2} W/K (expected {:.2} W/K)", h_iz_1_to_0, expected_h_1_to_0);
    println!("Ratio: {:.2}×", h_iz_1_to_0 / h_iz_0_to_1);

    assert!(
        (h_iz_0_to_1 - expected_h_0_to_1).abs() < 0.5,
        "Expected h_iz_0_to_1 ~{:.2} W/K, got {:.2} W/K",
        expected_h_0_to_1, h_iz_0_to_1
    );

    assert!(
        (h_iz_1_to_0 - expected_h_1_to_0).abs() < 1.0,
        "Expected h_iz_1_to_0 ~{:.2} W/K, got {:.2} W/K",
        expected_h_1_to_0, h_iz_1_to_0
    );

    // Verify moderate asymmetry (~3× difference)
    let ratio = h_iz_1_to_0 / h_iz_0_to_1;
    let expected_ratio = (CONCRETE_R_VALUE + r_insulation_zone_0) / (CONCRETE_R_VALUE + r_insulation_zone_1);

    assert!(
        (ratio - expected_ratio).abs() < 0.5,
        "Expected ~{:.1}× ratio, got {:.1}×",
        expected_ratio, ratio
    );
}

#[test]
fn test_zero_area() {
    // Test edge case: zero area should give zero conductance

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);
    let (h_iz_0_to_1, h_iz_1_to_0) = calculate_directional_interzone_conductance(
        0.0, &wall, 2.0, 0.0,
    );

    assert_eq!(h_iz_0_to_1, 0.0, "Zero area should give zero conductance");
    assert_eq!(h_iz_1_to_0, 0.0, "Zero area should give zero conductance");
}

#[test]
fn test_negative_area() {
    // Test edge case: negative area should give negative conductance (physically invalid but mathematically correct)

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);
    let (h_iz_0_to_1, h_iz_1_to_0) = calculate_directional_interzone_conductance(
        -10.0, &wall, 2.0, 0.0,
    );

    assert!(h_iz_0_to_1 < 0.0, "Negative area should give negative conductance");
    assert!(h_iz_1_to_0 < 0.0, "Negative area should give negative conductance");
}

#[test]
fn test_high_insulation_both_sides() {
    // Test high insulation on both sides
    // Both zones have R = 5.0 m²K/W

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);

    let r_insulation = 5.0;

    let (h_iz_0_to_1, h_iz_1_to_0) = calculate_directional_interzone_conductance(
        COMMON_WALL_AREA,
        &wall,
        r_insulation,
        r_insulation,
    );

    // Both should equal: h = 21.6 / (0.177 + 5.0) = 4.2 W/K
    let expected = COMMON_WALL_AREA / (CONCRETE_R_VALUE + r_insulation);

    assert!(
        (h_iz_0_to_1 - expected).abs() < 0.5,
        "Expected h_iz_0_to_1 ~{:.2} W/K, got {:.2} W/K",
        expected, h_iz_0_to_1
    );

    assert!(
        (h_iz_1_to_0 - expected).abs() < 0.5,
        "Expected h_iz_1_to_0 ~{:.2} W/K, got {:.2} W/K",
        expected, h_iz_1_to_0
    );

    // Verify equality and low conductance due to high insulation
    assert!(
        (h_iz_0_to_1 - h_iz_1_to_0).abs() < 0.1,
        "Symmetric high insulation should produce equal conductances"
    );

    assert!(
        h_iz_0_to_1 < 10.0,
        "High insulation should produce low conductance (< 10 W/K)"
    );
}

#[test]
fn test_conductance_scaling_with_area() {
    // Test that conductance scales linearly with area

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);

    let area_1 = 10.0; // m²
    let area_2 = 20.0; // m²

    let (h_iz_0_to_1_1, _) = calculate_directional_interzone_conductance(
        area_1, &wall, 2.0, 0.0,
    );

    let (h_iz_0_to_1_2, _) = calculate_directional_interzone_conductance(
        area_2, &wall, 2.0, 0.0,
    );

    let ratio = h_iz_0_to_1_2 / h_iz_0_to_1_1;
    let expected_ratio = 2.0;

    println!("Doubling area: {:.2}× (expected {:.2}×)", ratio, expected_ratio);

    assert!(
        (ratio - expected_ratio).abs() < 0.01,
        "Conductance should scale linearly with area, expected ratio {:.2}, got {:.2}",
        expected_ratio, ratio
    );
}

#[test]
fn test_conductance_inverse_with_insulation() {
    // Test that conductance is inversely proportional to total R-value

    let wall = Assemblies::concrete_wall(CONCRETE_THICKNESS);

    let r_insulation_1 = 1.0;
    let r_insulation_2 = 2.0;

    let (h_iz_0_to_1_1, _) = calculate_directional_interzone_conductance(
        COMMON_WALL_AREA, &wall, r_insulation_1, 0.0,
    );

    let (h_iz_0_to_1_2, _) = calculate_directional_interzone_conductance(
        COMMON_WALL_AREA, &wall, r_insulation_2, 0.0,
    );

    // Doubling R should approximately halve conductance
    let ratio = h_iz_0_to_1_1 / h_iz_0_to_1_2;
    let expected_ratio = (CONCRETE_R_VALUE + r_insulation_2) / (CONCRETE_R_VALUE + r_insulation_1);

    println!("Doubling R-value: {:.2}× (expected {:.2}×)", ratio, expected_ratio);

    assert!(
        (ratio - expected_ratio).abs() < 0.1,
        "Conductance should be inversely proportional to R, expected ratio {:.2}, got {:.2}",
        expected_ratio, ratio
    );
}
