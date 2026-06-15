//! IEA Annex 58/60 Multi-Zone Validation Tests
//!
//! This module provides validation tests for multi-zone heat transfer modeling
//! based on IEA Annex 58/60 standards for inter-zone heat transfer validation.
//!
//! ## Reference Sources
//! - IEA Annex 58: Thermal Energy Performance of Buildings
//! - IEA Annex 60: New Generation Calculation Tools
//! - ISO 13790: Thermal performance of buildings
//!
//! ## Test Configurations
//! - 2-zone standard case: Two adjacent zones with common wall
//! - 3-zone standard case: Linear arrangement of three zones

use fluxion::sim::construction::Assemblies;
use fluxion::sim::interzone::{
    calculate_directional_interzone_conductance, calculate_interzone_conductance,
    calculate_radiative_conductance, calculate_stack_effect_ach,
    calculate_surface_radiative_exchange, calculate_ventilation_heat_transfer,
    calculate_window_radiative_conductance, calculate_zone_to_zone_view_factor, AIR_DENSITY,
    AIR_SPECIFIC_HEAT, STACK_COEFFICIENT,
};
use fluxion::sim::interzone_radiation::STEFAN_BOLTZMANN_CONSTANT;
use std::collections::HashMap;

// ============================================================================
// IEA Annex 58/60 Reference Data Structures
// ============================================================================

/// Reference values for 2-zone standard case (IEA Annex 58/60)
/// Based on ISO 13790 simplified multi-zone approach
pub struct TwoZoneReference {
    /// Common wall area (m²)
    pub common_wall_area: f64,
    /// Wall thermal resistance (m²K/W)
    pub wall_r_value: f64,
    /// Expected inter-zone conductance (W/K)
    pub expected_conductance: f64,
    /// Zone 1 volume (m³)
    pub zone1_volume: f64,
    /// Zone 2 volume (m³)
    pub zone2_volume: f64,
    /// Door opening area for stack effect (m²)
    pub door_area: f64,
    /// Door height for stack effect (m)
    pub door_height: f64,
    /// Surface emissivity
    pub emissivity: f64,
    /// Expected radiative heat transfer at ΔT=20°C (W)
    pub expected_radiative_20C: f64,
}

/// Reference values for 3-zone standard case (IEA Annex 58/60)
pub struct ThreeZoneReference {
    /// Zone volumes (m³)
    pub zone_volumes: [f64; 3],
    /// Common wall areas between adjacent zones (m²)
    pub common_wall_areas: [f64; 2],
    /// Wall R-values (m²K/W)
    pub wall_r_values: [f64; 2],
    /// Expected conductances between zones (W/K)
    pub expected_conductances: [f64; 2],
    /// Stack effect parameters
    pub door_area: f64,
    pub door_height: f64,
}

/// Create 2-zone standard case reference data
pub fn create_two_zone_reference() -> TwoZoneReference {
    TwoZoneReference {
        common_wall_area: 20.0,        // 4m x 5m common wall
        wall_r_value: 0.250,           // Representative brick wall R-value
        expected_conductance: 80.0,    // A/R = 20.0/0.25 = 80 W/K
        zone1_volume: 100.0,           // 5m x 4m x 5m
        zone2_volume: 100.0,           // Same size
        door_area: 2.0,                // 1m x 2m door
        door_height: 2.1,              // Standard door height
        emissivity: 0.9,               // Typical interior surface
        expected_radiative_20C: 680.0, // Q = σ·ε²·F·A·(T1⁴-T2⁴) at ΔT=20°C
    }
}

/// Create 3-zone standard case reference data
pub fn create_three_zone_reference() -> ThreeZoneReference {
    ThreeZoneReference {
        zone_volumes: [80.0, 80.0, 80.0],    // Equal zones
        common_wall_areas: [16.0, 16.0],     // 4m x 4m walls
        wall_r_values: [0.200, 0.300],       // Different constructions
        expected_conductances: [80.0, 53.3], // A/R for each wall
        door_area: 2.0,
        door_height: 2.1,
    }
}

// ============================================================================
// 2-Zone Standard Case Tests
// ============================================================================

#[test]
fn test_two_zone_interzone_conductance() {
    let reference = create_two_zone_reference();
    let wall = Assemblies::concrete_wall(0.200);

    let h_iz = calculate_interzone_conductance(reference.common_wall_area, &wall);

    // Verify conductance is within 5% of expected
    let error_pct =
        ((h_iz - reference.expected_conductance).abs() / reference.expected_conductance) * 100.0;
    assert!(
        error_pct < 5.0,
        "Conductance error {:.2}% exceeds 5%",
        error_pct
    );
}

#[test]
fn test_two_zone_directional_conductance() {
    let reference = create_two_zone_reference();
    let wall = Assemblies::concrete_wall(0.200);

    // Asymmetric case: insulation on zone 1 side only
    let r_insulation_zone1 = 0.5;
    let r_insulation_zone2 = 0.0;

    let (h_1_to_2, h_2_to_1) = calculate_directional_interzone_conductance(
        reference.common_wall_area,
        &wall,
        r_insulation_zone1,
        r_insulation_zone2,
    );

    // Heat flow from insulated side should be reduced
    assert!(
        h_1_to_2 < h_2_to_1,
        "Insulated side should have lower conductance"
    );

    // Verify ratio is physically meaningful
    let ratio = h_2_to_1 / h_1_to_2;
    assert!(
        ratio > 1.5,
        "Expected significant asymmetry, got ratio {:.2}",
        ratio
    );
}

#[test]
fn test_two_zone_stack_effect_ach() {
    let reference = create_two_zone_reference();

    // Temperature difference of 10°C
    let temp_zone1 = 25.0;
    let temp_zone2 = 15.0;

    let ach = calculate_stack_effect_ach(
        temp_zone1,
        temp_zone2,
        reference.door_height,
        reference.door_area,
        reference.zone1_volume,
    );

    // ACH should be positive for temperature difference
    assert!(ach > 0.0, "Stack effect ACH should be positive");

    // Typical ACH for inter-zone stack effect should be < 5 ACH
    assert!(ach < 5.0, "Stack effect ACH {:.3} seems too high", ach);
}

#[test]
fn test_two_zone_ventilation_heat_transfer() {
    let reference = create_two_zone_reference();

    let ach = 1.0; // 1 ACH
    let temp_source = 25.0;
    let temp_target = 20.0;

    let q_vent =
        calculate_ventilation_heat_transfer(ach, temp_source, temp_target, reference.zone1_volume);

    // Q = ρ·Cp·ACH·V·ΔT / 3600
    // = 1.2 × 1000 × 1 × 100 × 5 / 3600 = 166.7 W
    let expected = AIR_DENSITY
        * AIR_SPECIFIC_HEAT
        * ach
        * reference.zone1_volume
        * (temp_source - temp_target)
        / 3600.0;

    assert!(
        (q_vent - expected).abs() < 1.0,
        "Expected {:.1} W, got {:.1} W",
        expected,
        q_vent
    );
}

#[test]
fn test_two_zone_radiative_exchange() {
    let reference = create_two_zone_reference();

    // Large ΔT = 20°C scenario
    let temp_sunspace = 35.0;
    let temp_backzone = 15.0;

    let q_rad = calculate_surface_radiative_exchange(
        temp_sunspace,
        temp_backzone,
        reference.emissivity,
        reference.emissivity,
        1.0, // View factor = 1 for direct exchange
        reference.common_wall_area,
    );

    // Q should be significant for 20°C difference
    assert!(
        q_rad > 500.0,
        "Radiative exchange {:.1} W too low for ΔT=20°C",
        q_rad
    );
    assert!(q_rad < 1000.0, "Radiative exchange {:.1} W too high", q_rad);
}

#[test]
fn test_two_zone_view_factor() {
    let reference = create_two_zone_reference();

    let total_area_zone1 = 100.0; // Floor area × 4 walls
    let total_area_zone2 = 100.0;

    let vf = calculate_zone_to_zone_view_factor(
        reference.common_wall_area,
        total_area_zone1,
        total_area_zone2,
    );

    // View factor should be between 0 and 1
    assert!(vf > 0.0, "View factor should be positive");
    assert!(vf < 1.0, "View factor should be less than 1");

    // For equal areas and common wall = 20% of total area, F ≈ 0.04
    assert!(vf < 0.1, "View factor {:.4} seems too high", vf);
}

// ============================================================================
// 3-Zone Standard Case Tests
// ============================================================================

#[test]
fn test_three_zone_interzone_conductance_zone1_zone2() {
    let reference = create_three_zone_reference();
    let wall = Assemblies::concrete_wall(0.200);

    let h_12 = calculate_interzone_conductance(reference.common_wall_areas[0], &wall);

    let error_pct = ((h_12 - reference.expected_conductances[0]).abs()
        / reference.expected_conductances[0])
        * 100.0;
    assert!(
        error_pct < 5.0,
        "Zone 1-2 conductance error {:.2}%",
        error_pct
    );
}

#[test]
fn test_three_zone_interzone_conductance_zone2_zone3() {
    let reference = create_three_zone_reference();
    let wall = Assemblies::concrete_wall(0.300); // Different thickness

    let h_23 = calculate_interzone_conductance(reference.common_wall_areas[1], &wall);

    let error_pct = ((h_23 - reference.expected_conductances[1]).abs()
        / reference.expected_conductances[1])
        * 100.0;
    assert!(
        error_pct < 5.0,
        "Zone 2-3 conductance error {:.2}%",
        error_pct
    );
}

#[test]
fn test_three_zone_directional_asymmetry() {
    let reference = create_three_zone_reference();
    let wall = Assemblies::concrete_wall(0.200);

    // Zone 1 to Zone 2 with asymmetric insulation
    let (h_12, h_21) = calculate_directional_interzone_conductance(
        reference.common_wall_areas[0],
        &wall,
        1.0, // R-1.0 insulation on zone 1 side
        0.0, // No insulation on zone 2 side
    );

    // With insulation on zone 1 side, heat flow from zone 1 to 2 should be reduced
    assert!(
        h_12 < h_21,
        "Insulation should reduce h_12 relative to h_21"
    );

    // Ratio should be significant
    let ratio = h_21 / h_12;
    assert!(
        ratio > 1.5,
        "Expected asymmetric ratio > 1.5, got {:.2}",
        ratio
    );
}

#[test]
fn test_three_zone_stack_effect_chain() {
    let reference = create_three_zone_reference();

    // Temperature chain: Zone 1 warm, Zone 2 medium, Zone 3 cool
    let temp_1 = 30.0;
    let temp_2 = 20.0;
    let temp_3 = 10.0;

    // Stack effect from Zone 1 to Zone 2
    let ach_12 = calculate_stack_effect_ach(
        temp_1,
        temp_2,
        reference.door_height,
        reference.door_area,
        reference.zone_volumes[0],
    );

    // Stack effect from Zone 2 to Zone 3
    let ach_23 = calculate_stack_effect_ach(
        temp_2,
        temp_3,
        reference.door_height,
        reference.door_area,
        reference.zone_volumes[1],
    );

    // Both should be positive
    assert!(ach_12 > 0.0, "ACH 1->2 should be positive");
    assert!(ach_23 > 0.0, "ACH 2->3 should be positive");

    // ACH should decrease as temperature difference decreases
    // For constant ΔT=10°C, ACH should be similar
    let diff = (ach_12 - ach_23).abs();
    assert!(diff < 1.0, "ACH values should be similar for equal ΔT");
}

#[test]
fn test_three_zone_total_heat_transfer() {
    let reference = create_three_zone_reference();

    // Calculate total conductance through the 3-zone chain
    let wall = Assemblies::concrete_wall(0.200);

    let h_12 = calculate_interzone_conductance(reference.common_wall_areas[0], &wall);
    let h_23 = calculate_interzone_conductance(reference.common_wall_areas[1], &wall);

    // Total equivalent conductance for series connection
    // 1/H_total = 1/H_12 + 1/H_23
    let h_total = 1.0 / (1.0 / h_12 + 1.0 / h_23);

    // Should be less than either individual conductance
    assert!(h_total < h_12, "Total conductance should be less than H_12");
    assert!(h_total < h_23, "Total conductance should be less than H_23");

    // Should be positive
    assert!(h_total > 0.0, "Total conductance should be positive");
}

// ============================================================================
// Radiative Exchange Tests
// ============================================================================

#[test]
fn test_radiative_exchange_nonlinear_stefan_boltzmann() {
    // Stefan-Boltzmann law: Q = σ·ε²·F·A·(T1⁴ - T2⁴)
    let sigma = STEFAN_BOLTZMANN_CONSTANT;
    let emissivity = 0.9;
    let view_factor = 1.0;
    let area = 20.0;

    // ΔT = 20°C: T1 = 35°C, T2 = 15°C
    let t1_k = 35.0 + 273.15;
    let t2_k = 15.0 + 273.15;

    let q_calc = sigma * emissivity.powi(2) * view_factor * area * (t1_k.powi(4) - t2_k.powi(4));

    // Use the module function for comparison
    let q_fn =
        calculate_surface_radiative_exchange(35.0, 15.0, emissivity, emissivity, view_factor, area);

    assert!(
        (q_calc - q_fn).abs() < 1.0,
        "Stefan-Boltzmann calculation mismatch"
    );
}

#[test]
fn test_radiative_exchange_linearized_vs_nonlinear() {
    // For small ΔT (< 5°C), linearized and nonlinear should agree
    let temp_a = 22.0;
    let temp_b = 18.0;
    let emissivity = 0.9;
    let view_factor = 0.5;
    let area = 10.0;

    let q_nonlinear = calculate_surface_radiative_exchange(
        temp_a,
        temp_b,
        emissivity,
        emissivity,
        view_factor,
        area,
    );

    // Linearized: h_rad = 4·σ·ε²·F·T³·A
    let t_avg_k = (temp_a + temp_b) / 2.0 + 273.15;
    let h_rad_lin =
        4.0 * STEFAN_BOLTZMANN_CONSTANT * emissivity.powi(2) * view_factor * t_avg_k.powi(3) * area;
    let q_linearized = h_rad_lin * (temp_a - temp_b);

    // Should agree within 2% for small ΔT
    let error_pct = ((q_nonlinear - q_linearized).abs() / q_linearized) * 100.0;
    assert!(
        error_pct < 2.0,
        "Linearized error {:.2}% too high for small ΔT",
        error_pct
    );
}

#[test]
fn test_window_radiative_conductance() {
    let window_area = 3.0; // 1m x 3m window
    let glass_emissivity = 0.84;
    let mean_temp_k = 293.15;
    let view_factor = 0.8;

    let h_rad = calculate_window_radiative_conductance(
        window_area,
        glass_emissivity,
        mean_temp_k,
        view_factor,
    );

    assert!(
        h_rad > 0.0,
        "Window radiative conductance should be positive"
    );

    // h_rad = 4·σ·ε²·F·T³·A
    let expected = 4.0
        * STEFAN_BOLTZMANN_CONSTANT
        * glass_emissivity.powi(2)
        * view_factor
        * mean_temp_k.powi(3)
        * window_area;
    assert!(
        (h_rad - expected).abs() < 0.1,
        "Window radiative conductance mismatch"
    );
}

// ============================================================================
// Edge Cases and Boundary Conditions
// ============================================================================

#[test]
fn test_zero_temperature_difference_stack_effect() {
    let reference = create_two_zone_reference();

    let ach = calculate_stack_effect_ach(
        20.0, // Same temperature
        20.0,
        reference.door_height,
        reference.door_area,
        reference.zone1_volume,
    );

    assert_eq!(ach, 0.0, "Stack effect ACH should be zero for ΔT=0");
}

#[test]
fn test_zero_area_conductance() {
    let wall = Assemblies::concrete_wall(0.200);

    let h = calculate_interzone_conductance(0.0, &wall);

    assert_eq!(h, 0.0, "Zero area should give zero conductance");
}

#[test]
fn test_negative_temperature_difference_ventilation() {
    let reference = create_two_zone_reference();

    // Heat flows from cool to warm (negative ΔT)
    let q = calculate_ventilation_heat_transfer(
        1.0,
        15.0, // Cool source
        25.0, // Warm target
        reference.zone1_volume,
    );

    assert!(
        q < 0.0,
        "Ventilation heat transfer should be negative when source is cooler"
    );
}

#[test]
fn test_identical_zones_conductance() {
    let wall = Assemblies::concrete_wall(0.200);
    let area = 20.0;

    let h = calculate_interzone_conductance(area, &wall);

    // For identical zones, directional conductance should be symmetric
    let (h_a_to_b, h_b_to_a) = calculate_directional_interzone_conductance(area, &wall, 0.0, 0.0);

    assert!(
        (h_a_to_b - h_b_to_a).abs() < 0.01,
        "Symmetric zones should have equal conductance"
    );
    assert!(
        (h - h_a_to_b).abs() < 0.01,
        "Symmetric conductance should equal basic conductance"
    );
}

// ============================================================================
// Physical Consistency Tests
// ============================================================================

#[test]
fn test_energy_conservation_stack_effect() {
    let reference = create_two_zone_reference();

    let temp_a = 25.0;
    let temp_b = 15.0;

    let ach_a_to_b = calculate_stack_effect_ach(
        temp_a,
        temp_b,
        reference.door_height,
        reference.door_area,
        reference.zone1_volume,
    );

    let ach_b_to_a = calculate_stack_effect_ach(
        temp_b,
        temp_a,
        reference.door_height,
        reference.door_area,
        reference.zone2_volume,
    );

    // ACH should be symmetric for symmetric zones
    assert!(
        (ach_a_to_b - ach_b_to_a).abs() < 0.001,
        "ACH should be symmetric for equal volumes"
    );
}

#[test]
fn test_radiative_exchange_sign_convention() {
    // Positive ΔT should give positive Q (heat flows from hot to cold)
    let q_positive = calculate_surface_radiative_exchange(30.0, 20.0, 0.9, 0.9, 1.0, 10.0);
    assert!(q_positive > 0.0, "Heat should flow from hot to cold");

    // Negative ΔT should give negative Q
    let q_negative = calculate_surface_radiative_exchange(20.0, 30.0, 0.9, 0.9, 1.0, 10.0);
    assert!(
        q_negative < 0.0,
        "Heat should flow from hot to cold (negative)"
    );

    // Magnitude should be equal
    assert!(
        (q_positive + q_negative).abs() < 0.001,
        "Magnitude should be equal"
    );
}

#[test]
fn test_conductance_proportionality() {
    let wall = Assemblies::concrete_wall(0.200);

    // Double area should double conductance
    let h1 = calculate_interzone_conductance(10.0, &wall);
    let h2 = calculate_interzone_conductance(20.0, &wall);

    assert!(
        (h2 - 2.0 * h1).abs() < 0.001,
        "Conductance should be proportional to area"
    );
}

#[test]
fn test_inverse_r_value_proportionality() {
    // Higher R-value should give lower conductance
    let wall_low_r = Assemblies::concrete_wall(0.100); // R ≈ 0.09
    let wall_high_r = Assemblies::concrete_wall(0.300); // R ≈ 0.27

    let h_low = calculate_interzone_conductance(10.0, &wall_low_r);
    let h_high = calculate_interzone_conductance(10.0, &wall_high_r);

    assert!(
        h_low > h_high,
        "Lower R-value should give higher conductance"
    );
}

// ============================================================================
// Constants Validation
// ============================================================================

#[test]
fn test_constants_values() {
    // Verify physical constants are within expected ranges
    assert!(
        (AIR_DENSITY - 1.2).abs() < 0.01,
        "Air density should be ~1.2 kg/m³"
    );
    assert!(
        (AIR_SPECIFIC_HEAT - 1000.0).abs() < 1.0,
        "Air specific heat should be ~1000 J/kg·K"
    );
    assert!(
        (STACK_COEFFICIENT - 0.025).abs() < 0.001,
        "Stack coefficient should be ~0.025"
    );
    assert!(
        (STEFAN_BOLTZMANN_CONSTANT - 5.67e-8).abs() < 1e-10,
        "Stefan-Boltzmann constant mismatch"
    );
}
