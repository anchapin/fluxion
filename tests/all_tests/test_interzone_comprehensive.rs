//! Comprehensive tests for inter-zone heat transfer modeling.
//!
//! This test suite covers:
//! - Directional conductance calculations
//! - Stack effect ventilation
//! - Radiative conductance between zones
//! - Ventilation heat transfer
//! - Edge cases and boundary conditions

use fluxion::sim::construction::Assemblies;
use fluxion::sim::interzone::{
    calculate_directional_interzone_conductance, calculate_interzone_conductance,
    calculate_radiative_conductance, calculate_stack_effect_ach,
    calculate_ventilation_heat_transfer, calculate_window_radiative_conductance,
    calculate_zone_to_zone_view_factor, AIR_DENSITY, AIR_SPECIFIC_HEAT, STACK_COEFFICIENT,
};
use fluxion::sim::view_factors::{
    build_zone_view_factors, hottels_rectangular_view_factor, hottels_rectangular_view_factor_pair,
    reciprocal_view_factor, CommonWallGeometry,
};

// ============================================================================
// Interzone Conductance Tests
// ============================================================================

#[test]
fn test_interzone_conductance_basic() {
    // Simple concrete wall
    let wall = Assemblies::concrete_wall(0.200);
    let area = 10.0; // m²

    let h = calculate_interzone_conductance(area, &wall);

    // Should be positive and reasonable
    assert!(h > 0.0);
    assert!(h < 1000.0);
}

#[test]
fn test_interzone_conductance_proportional_to_area() {
    let wall = Assemblies::concrete_wall(0.200);

    let h1 = calculate_interzone_conductance(10.0, &wall);
    let h2 = calculate_interzone_conductance(20.0, &wall);

    // Doubling area should double conductance
    assert!((h2 - 2.0 * h1).abs() < 1e-10);
}

#[test]
fn test_interzone_conductance_inversely_proportional_to_r_value() {
    // Thicker wall = higher R-value = lower conductance
    let wall_thin = Assemblies::concrete_wall(0.100);
    let wall_thick = Assemblies::concrete_wall(0.200);
    let area = 10.0;

    let h_thin = calculate_interzone_conductance(area, &wall_thin);
    let h_thick = calculate_interzone_conductance(area, &wall_thick);

    // Thicker wall should have lower conductance
    assert!(h_thick < h_thin);

    // Ratio should be approximately 2:1 (double thickness = half conductance)
    let ratio = h_thin / h_thick;
    assert!((ratio - 2.0).abs() < 0.1);
}

#[test]
fn test_directional_conductance_symmetric_insulation() {
    let wall = Assemblies::concrete_wall(0.200);
    let area = 21.6;

    // Symmetric insulation
    let (h_a_to_b, h_b_to_a) = calculate_directional_interzone_conductance(area, &wall, 1.0, 1.0);

    // Should be equal
    assert!((h_a_to_b - h_b_to_a).abs() < 1e-10);
}

#[test]
fn test_directional_conductance_asymmetric_insulation() {
    let wall = Assemblies::concrete_wall(0.200);
    let area = 21.6;

    // More insulation on side A → lower conductance from A to B
    let (h_a_to_b, h_b_to_a) = calculate_directional_interzone_conductance(area, &wall, 2.0, 0.0);

    // h_a_to_b should be lower (more resistance on side A)
    assert!(h_a_to_b < h_b_to_a);

    // Verify the ratio is reasonable
    let ratio = h_b_to_a / h_a_to_b;
    assert!(ratio > 1.0);
}

#[test]
fn test_directional_conductance_no_insulation() {
    let wall = Assemblies::concrete_wall(0.200);
    let area = 21.6;

    // No additional insulation
    let (h_a_to_b, h_b_to_a) = calculate_directional_interzone_conductance(area, &wall, 0.0, 0.0);

    // Should be equal and match basic conductance
    assert!((h_a_to_b - h_b_to_a).abs() < 1e-10);

    let h_basic = calculate_interzone_conductance(area, &wall);
    assert!((h_a_to_b - h_basic).abs() < 1e-10);
}

#[test]
fn test_directional_conductance_extreme_insulation() {
    let wall = Assemblies::concrete_wall(0.200);
    let area = 10.0;

    // Very high insulation on side A
    let (h_a_to_b, h_b_to_a) = calculate_directional_interzone_conductance(area, &wall, 100.0, 0.0);

    // h_a_to_b should be very small
    assert!(h_a_to_b < 1.0);
    assert!(h_b_to_a > h_a_to_b);
}

// ============================================================================
// Stack Effect Ventilation Tests
// ============================================================================

#[test]
fn test_stack_effect_basic() {
    // Basic stack effect with temperature difference
    let temp_a = 25.0; // Warmer zone
    let temp_b = 15.0; // Cooler zone
    let door_height = 2.0;
    let door_area = 2.0;
    let zone_volume = 64.8; // m³ (Case 960 back-zone volume)

    let ach = calculate_stack_effect_ach(temp_a, temp_b, door_height, door_area, zone_volume);

    // Should be positive
    assert!(ach > 0.0);

    // Should be reasonable (typically 0.1-10 ACH for buildings)
    assert!(ach < 50.0);
}

#[test]
fn test_stack_effect_symmetric() {
    // Stack effect should be symmetric (depends on |ΔT|)
    let door_height = 2.0;
    let door_area = 2.0;
    let zone_volume = 64.8;

    let ach1 = calculate_stack_effect_ach(25.0, 15.0, door_height, door_area, zone_volume);
    let ach2 = calculate_stack_effect_ach(15.0, 25.0, door_height, door_area, zone_volume);

    assert!((ach1 - ach2).abs() < 1e-10);
}

#[test]
fn test_stack_effect_zero_temperature_difference() {
    // No temperature difference → no stack effect
    let ach = calculate_stack_effect_ach(20.0, 20.0, 2.0, 2.0, 64.8);

    assert!(ach.abs() < 1e-10);
}

#[test]
fn test_stack_effect_proportional_to_sqrt_delta_t() {
    let door_height = 2.0;
    let door_area = 2.0;
    let zone_volume = 64.8;

    // Stack effect follows √(ΔT) relationship
    let ach1 = calculate_stack_effect_ach(21.0, 20.0, door_height, door_area, zone_volume); // ΔT = 1
    let ach4 = calculate_stack_effect_ach(24.0, 20.0, door_height, door_area, zone_volume); // ΔT = 4

    // ach4 should be approximately 2× ach1 (√4 = 2, √1 = 1)
    let ratio = ach4 / ach1;
    assert!((ratio - 2.0).abs() < 0.1);
}

#[test]
fn test_stack_effect_door_height_effect() {
    let temp_a = 25.0;
    let temp_b = 15.0;
    let door_area = 2.0;
    let zone_volume = 64.8;

    // Taller door → lower ACH (flow per unit volume)
    let ach_short = calculate_stack_effect_ach(temp_a, temp_b, 1.0, door_area, zone_volume);
    let ach_tall = calculate_stack_effect_ach(temp_a, temp_b, 4.0, door_area, zone_volume);

    // Shorter door should have higher ACH
    assert!(ach_short > ach_tall);
}

#[test]
fn test_stack_effect_large_temperature_difference() {
    // Sunspace can get very hot (40°C+) while back-zone is cool (15°C)
    let temp_sunspace = 45.0;
    let temp_backzone = 15.0;
    let door_height = 2.1;
    let door_area = 2.0;
    let zone_volume = 64.8;

    let ach = calculate_stack_effect_ach(
        temp_sunspace,
        temp_backzone,
        door_height,
        door_area,
        zone_volume,
    );

    assert!(ach > 0.0);
    assert!(ach < 100.0); // Should be reasonable
}

// ============================================================================
// Ventilation Heat Transfer Tests
// ============================================================================

#[test]
fn test_ventilation_heat_transfer_basic() {
    let ach = 1.0;
    let temp_source = 25.0;
    let temp_target = 20.0;
    let volume = 100.0;

    let q = calculate_ventilation_heat_transfer(ach, temp_source, temp_target, volume);

    // Heat should flow from source to target (positive)
    assert!(q > 0.0);

    // Should be reasonable magnitude
    assert!(q < 10000.0);
}

#[test]
fn test_ventilation_heat_transfer_direction() {
    let ach = 1.0;
    let volume = 100.0;

    // Heat flows from warm to cool
    let q1 = calculate_ventilation_heat_transfer(ach, 25.0, 20.0, volume);
    let q2 = calculate_ventilation_heat_transfer(ach, 20.0, 25.0, volume);

    assert!(q1 > 0.0); // From 25°C to 20°C
    assert!(q2 < 0.0); // From 20°C to 25°C (negative = heat loss from target)

    // Magnitudes should be equal
    assert!((q1 + q2).abs() < 1e-10);
}

#[test]
fn test_ventilation_heat_transfer_proportional_to_ach() {
    let temp_source = 25.0;
    let temp_target = 20.0;
    let volume = 100.0;

    let q1 = calculate_ventilation_heat_transfer(0.5, temp_source, temp_target, volume);
    let q2 = calculate_ventilation_heat_transfer(1.0, temp_source, temp_target, volume);
    let q3 = calculate_ventilation_heat_transfer(2.0, temp_source, temp_target, volume);

    // Should be linear with ACH
    assert!((q2 - 2.0 * q1).abs() < 1e-10);
    assert!((q3 - 2.0 * q2).abs() < 1e-10);
}

#[test]
fn test_ventilation_heat_transfer_proportional_to_volume() {
    let ach = 1.0;
    let temp_source = 25.0;
    let temp_target = 20.0;

    let q1 = calculate_ventilation_heat_transfer(ach, temp_source, temp_target, 50.0);
    let q2 = calculate_ventilation_heat_transfer(ach, temp_source, temp_target, 100.0);
    let q3 = calculate_ventilation_heat_transfer(ach, temp_source, temp_target, 200.0);

    // Should be linear with volume
    assert!((q2 - 2.0 * q1).abs() < 1e-10);
    assert!((q3 - 2.0 * q2).abs() < 1e-10);
}

#[test]
fn test_ventilation_heat_transfer_proportional_to_delta_t() {
    let ach = 1.0;
    let volume = 100.0;

    let q5 = calculate_ventilation_heat_transfer(ach, 25.0, 20.0, volume); // ΔT = 5
    let q10 = calculate_ventilation_heat_transfer(ach, 30.0, 20.0, volume); // ΔT = 10
    let q20 = calculate_ventilation_heat_transfer(ach, 40.0, 20.0, volume); // ΔT = 20

    // Should be linear with ΔT
    assert!((q10 - 2.0 * q5).abs() < 1e-10);
    assert!((q20 - 2.0 * q10).abs() < 1e-10);
}

#[test]
fn test_ventilation_heat_transfer_zero_delta_t() {
    let ach = 1.0;
    let volume = 100.0;

    let q = calculate_ventilation_heat_transfer(ach, 20.0, 20.0, volume);

    assert!(q.abs() < 1e-10);
}

#[test]
fn test_ventilation_heat_transfer_zero_ach() {
    let temp_source = 25.0;
    let temp_target = 20.0;
    let volume = 100.0;

    let q = calculate_ventilation_heat_transfer(0.0, temp_source, temp_target, volume);

    assert!(q.abs() < 1e-10);
}

#[test]
fn test_ventilation_heat_transfer_unit_consistency() {
    // Verify: ρ·Cp·ACH·V·ΔT / 3600 = W
    // (kg/m³)·(J/kg·K)·(1/hr)·(m³)·K / (s/hr) = J/s = W

    let ach = 1.0;
    let temp_source = 21.0;
    let temp_target = 20.0;
    let volume = 1.0;

    let q = calculate_ventilation_heat_transfer(ach, temp_source, temp_target, volume);

    // Expected: 1.2 * 1000 * 1 * 1 * 1 / 3600 = 0.333... W
    let expected = AIR_DENSITY * AIR_SPECIFIC_HEAT * ach * volume * 1.0 / 3600.0;
    assert!((q - expected).abs() < 1e-10);
}

// ============================================================================
// Radiative Conductance Tests
// ============================================================================

#[test]
fn test_radiative_conductance_basic() {
    let area = 10.0;
    let emissivity = 0.9;
    let mean_temp_k = 293.15; // 20°C
    let view_factor = 0.5;

    let h_r = calculate_radiative_conductance(area, emissivity, mean_temp_k, view_factor);

    assert!(h_r > 0.0);
    assert!(h_r < 100.0); // Should be reasonable
}

#[test]
fn test_radiative_conductance_proportional_to_area() {
    let emissivity = 0.9;
    let mean_temp_k = 293.15;
    let view_factor = 0.5;

    let h1 = calculate_radiative_conductance(10.0, emissivity, mean_temp_k, view_factor);
    let h2 = calculate_radiative_conductance(20.0, emissivity, mean_temp_k, view_factor);

    assert!((h2 - 2.0 * h1).abs() < 1e-10);
}

#[test]
fn test_radiative_conductance_proportional_to_emissivity_squared() {
    let area = 10.0;
    let mean_temp_k = 293.15;
    let view_factor = 0.5;

    let h1 = calculate_radiative_conductance(area, 0.5, mean_temp_k, view_factor);
    let h2 = calculate_radiative_conductance(area, 0.9, mean_temp_k, view_factor);

    // Should scale with ε²
    let ratio = h2 / h1;
    let expected_ratio = (0.9 * 0.9) / (0.5 * 0.5);
    assert!((ratio - expected_ratio).abs() < 0.1);
}

#[test]
fn test_radiative_conductance_proportional_to_t_cubed() {
    let area = 10.0;
    let emissivity = 0.9;
    let view_factor = 0.5;

    let t1_k = 283.15; // 10°C
    let t2_k = 293.15; // 20°C

    let h1 = calculate_radiative_conductance(area, emissivity, t1_k, view_factor);
    let h2 = calculate_radiative_conductance(area, emissivity, t2_k, view_factor);

    // Should increase with temperature (T³ dependence)
    assert!(h2 > h1);

    // Ratio should be approximately (T2/T1)³
    let ratio = h2 / h1;
    let expected_ratio = (t2_k / t1_k).powi(3);
    assert!((ratio - expected_ratio).abs() < 0.1);
}

#[test]
fn test_radiative_conductance_proportional_to_view_factor() {
    let area = 10.0;
    let emissivity = 0.9;
    let mean_temp_k = 293.15;

    let h1 = calculate_radiative_conductance(area, emissivity, mean_temp_k, 0.25);
    let h2 = calculate_radiative_conductance(area, emissivity, mean_temp_k, 0.5);

    assert!((h2 - 2.0 * h1).abs() < 1e-10);
}

#[test]
fn test_window_radiative_conductance() {
    let window_area = 6.0;
    let glass_emissivity = 0.84; // Typical for clear glass
    let mean_temp_k = 293.15;
    let view_factor = 1.0; // Direct view between windows

    let h_r = calculate_window_radiative_conductance(
        window_area,
        glass_emissivity,
        mean_temp_k,
        view_factor,
    );

    assert!(h_r > 0.0);
    assert!(h_r < 50.0);
}

// ============================================================================
// View Factor Tests
// ============================================================================

#[test]
fn test_zone_to_zone_view_factor_basic() {
    let common_wall = 10.0;
    let total_area_1 = 50.0;
    let total_area_2 = 50.0;

    let f = calculate_zone_to_zone_view_factor(common_wall, total_area_1, total_area_2);

    assert!(f >= 0.0);
    assert!(f <= 1.0);
}

#[test]
fn test_zone_to_zone_view_factor_proportional_to_common_area() {
    let total_area_1 = 50.0;
    let total_area_2 = 50.0;

    let f1 = calculate_zone_to_zone_view_factor(5.0, total_area_1, total_area_2);
    let f2 = calculate_zone_to_zone_view_factor(10.0, total_area_1, total_area_2);

    // Should increase with common area
    assert!(f2 > f1);
}

#[test]
fn test_zone_to_zone_view_factor_inversely_proportional_to_total_area() {
    let common_wall = 10.0;

    let f1 = calculate_zone_to_zone_view_factor(common_wall, 50.0, 50.0);
    let f2 = calculate_zone_to_zone_view_factor(common_wall, 100.0, 100.0);

    // Larger total area → smaller view factor
    assert!(f1 > f2);
}

#[test]
fn test_zone_to_zone_view_factor_zero_common_area() {
    let f = calculate_zone_to_zone_view_factor(0.0, 50.0, 50.0);
    assert!(f.abs() < 1e-10);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_interzone_heat_transfer_combined() {
    // Simulate heat transfer between two zones with both conduction and ventilation

    // Zone A: 25°C, Zone B: 20°C
    let temp_a = 25.0;
    let temp_b = 20.0;

    // Common wall: 200mm concrete, 21.6 m²
    let wall = Assemblies::concrete_wall(0.200);
    let area = 21.6;

    // Conductive heat transfer
    let h_cond = calculate_interzone_conductance(area, &wall);
    let q_cond = h_cond * (temp_a - temp_b);

    // Stack effect ventilation
    let door_height = 2.1;
    let door_area = 2.0;
    let zone_volume = 64.8;
    let ach = calculate_stack_effect_ach(temp_a, temp_b, door_height, door_area, zone_volume);

    // Ventilation heat transfer
    let volume_b = 64.8; // m³
    let q_vent = calculate_ventilation_heat_transfer(ach, temp_a, temp_b, volume_b);

    // Total heat transfer from A to B
    let q_total = q_cond + q_vent;

    assert!(q_total > 0.0); // Heat flows from warm to cool
    assert!(q_total < 10000.0); // Should be reasonable
}

#[test]
fn test_case_960_sunspace_scenario() {
    // Simulate Case 960: Sunspace at 35°C, Back-zone at 20°C

    let temp_sunspace = 35.0;
    let temp_backzone = 20.0;

    // Common wall: 200mm concrete, 21.6 m²
    let wall = Assemblies::concrete_wall(0.200);
    let common_area = 21.6;

    // Conductive coupling
    let h_cond = calculate_interzone_conductance(common_area, &wall);
    let q_cond = h_cond * (temp_sunspace - temp_backzone);

    // Window-to-window radiation (6 m² glazing)
    let window_area = 6.0;
    let glass_emissivity = 0.84;
    let mean_temp_k = (temp_sunspace + temp_backzone) / 2.0 + 273.15;
    let view_factor = 1.0; // Aligned windows

    let h_rad = calculate_window_radiative_conductance(
        window_area,
        glass_emissivity,
        mean_temp_k,
        view_factor,
    );
    let q_rad = h_rad * (temp_sunspace - temp_backzone);

    // Stack effect ventilation through door
    let door_height = 2.1;
    let door_area = 2.0;
    let zone_volume = 64.8;
    let ach = calculate_stack_effect_ach(
        temp_sunspace,
        temp_backzone,
        door_height,
        door_area,
        zone_volume,
    );
    let volume_backzone = 64.8;
    let q_vent =
        calculate_ventilation_heat_transfer(ach, temp_sunspace, temp_backzone, volume_backzone);

    // Total heat transfer from sunspace to back-zone
    let q_total = q_cond + q_rad + q_vent;

    // All components should contribute positively
    assert!(q_cond > 0.0);
    assert!(q_rad > 0.0);
    assert!(q_vent > 0.0);
    assert!(q_total > 0.0);

    // Conductive should be dominant (large concrete wall)
    assert!(q_cond > q_rad);
}

// ============================================================================
// Edge Cases and Constants
// ============================================================================

#[test]
fn test_constants_are_reasonable() {
    // Verify physical constants are within expected ranges
    assert!((STACK_COEFFICIENT - 0.025).abs() < 1e-10);
    assert!((AIR_DENSITY - 1.2).abs() < 0.1);
    assert!((AIR_SPECIFIC_HEAT - 1000.0).abs() < 100.0);
}

#[test]
fn test_interzone_conductance_very_thin_wall() {
    // Very thin wall (high conductance)
    let wall = Assemblies::concrete_wall(0.010); // 10mm
    let area = 10.0;

    let h = calculate_interzone_conductance(area, &wall);

    assert!(h > 0.0);
    assert!(h > 100.0); // Should be very high
}

#[test]
fn test_interzone_conductance_very_thick_wall() {
    // Very thick wall (low conductance)
    let wall = Assemblies::concrete_wall(1.0); // 1m
    let area = 10.0;

    let h = calculate_interzone_conductance(area, &wall);

    assert!(h > 0.0);
    // For 1m concrete (k≈1.13), R = 1.0/1.13 ≈ 0.885 m²K/W
    // h = A/R = 10/0.885 ≈ 11.3 W/K
    assert!(h < 20.0); // Should be low but not extremely low
}

#[test]
fn test_stack_effect_very_small_door() {
    let temp_a = 25.0;
    let temp_b = 15.0;
    let door_height = 2.0;
    let door_area = 0.01; // Very small
    let zone_volume = 64.8;

    let ach = calculate_stack_effect_ach(temp_a, temp_b, door_height, door_area, zone_volume);

    assert!(ach > 0.0);
}

#[test]
fn test_ventilation_heat_transfer_extreme_temperatures() {
    // Test with extreme temperature difference (sunspace scenario)
    let ach = 2.0;
    let temp_source = 50.0; // Very hot sunspace
    let temp_target = 10.0; // Cool back-zone
    let volume = 65.0;

    let q = calculate_ventilation_heat_transfer(ach, temp_source, temp_target, volume);

    assert!(q > 0.0);
    assert!(q < 10000.0); // Should be reasonable
}

#[test]
fn test_radiative_conductance_zero_emissivity() {
    let area = 10.0;
    let emissivity = 0.0;
    let mean_temp_k = 293.15;
    let view_factor = 0.5;

    let h_r = calculate_radiative_conductance(area, emissivity, mean_temp_k, view_factor);

    assert!(h_r.abs() < 1e-10);
}

#[test]
fn test_radiative_conductance_zero_view_factor() {
    let area = 10.0;
    let emissivity = 0.9;
    let mean_temp_k = 293.15;
    let view_factor = 0.0;

    let h_r = calculate_radiative_conductance(area, emissivity, mean_temp_k, view_factor);

    assert!(h_r.abs() < 1e-10);
}

// ============================================================================
// Issue #1444 — Hottel view factor reciprocity under asymmetric geometry
// ============================================================================
//
// The previous implementation of `hottels_rectangular_view_factor` returned
// `(common / A_a) * min(common / A_b, 1)`, which is symmetric in A and B and
// violates the reciprocity identity `F_AB * A_A = F_BA * A_B` whenever the two
// surfaces have different areas.  For the case `8m × 3m` vs `8m × 2m` at
// separation 0.1 m the residual was 5.33 m² — radiative energy was not
// conserved across the common wall.  These tests guard against regressions.

const RECIPROCITY_TOL_1444: f64 = 1e-9;

/// Issue #1444 example: 8 × 3 vs 8 × 2, common-wall geometry.
/// Old code returned 0.667 for both directions ⇒ `F_AB*A_A = 16`,
/// `F_BA*A_B = 10.67`, residual 5.33.  New code returns F_AB = 16/24 and
/// F_BA = 1.0 ⇒ both products equal 16.
#[test]
fn test_issue_1444_hottel_reciprocity_8x3_vs_8x2() {
    let a_a = 8.0 * 3.0;
    let a_b = 8.0 * 2.0;

    let f_ab = hottels_rectangular_view_factor(8.0, 3.0, 8.0, 2.0, 0.1);
    let f_ba_direct = hottels_rectangular_view_factor(8.0, 2.0, 8.0, 3.0, 0.1);
    let f_ba_reciprocal = reciprocal_view_factor(f_ab, a_a, a_b);

    // Directional — F_AB ≠ F_BA in general.
    assert!((f_ab - 16.0 / 24.0).abs() < 1e-9, "F_AB = {f_ab:.6}");
    assert!((f_ba_direct - 1.0).abs() < 1e-6, "F_BA = {f_ba_direct:.6}");

    // Reciprocity: both A-weighted products must equal 16 m².
    let residual = (f_ab * a_a - f_ba_reciprocal * a_b).abs();
    assert!(
        residual < RECIPROCITY_TOL_1444,
        "reciprocity violated: F_AB*A_A={:.6e} F_BA*A_B={:.6e} residual={:.3e}",
        f_ab * a_a,
        f_ba_reciprocal * a_b,
        residual
    );
}

/// Reciprocity across 15 random rectangular configurations spanning
/// the common cases (aligned equal, aligned asymmetric, slight offset, large
/// separation, partial overlap).
#[test]
fn test_issue_1444_reciprocity_random_rectangles() {
    let configs: &[(f64, f64, f64, f64, f64)] = &[
        (8.0, 3.0, 8.0, 2.0, 0.1), // issue #1444 example
        (8.0, 3.0, 8.0, 2.9, 0.1),
        (8.0, 3.0, 8.0, 3.0, 0.0),
        (10.0, 4.0, 6.0, 2.0, 0.2),
        (5.0, 5.0, 5.0, 5.0, 0.1),
        (2.0, 1.5, 4.0, 1.0, 0.5),
        (1.0, 1.0, 3.0, 3.0, 0.0),
        (12.0, 2.0, 4.0, 2.0, 0.05),
        (8.0, 3.0, 8.0, 2.0, 2.0),
        (1.5, 1.0, 4.0, 2.5, 0.3),
        (6.0, 4.0, 2.0, 1.0, 0.0),
        (20.0, 5.0, 4.0, 1.0, 0.1),
        (3.0, 2.0, 6.0, 4.0, 0.1),
        (8.0, 3.0, 4.0, 1.0, 0.0),
        (8.0, 3.0, 8.0, 1.0, 0.05),
    ];
    for &(a_l, a_w, b_l, b_w, sep) in configs {
        let (f_ab, f_ba) = hottels_rectangular_view_factor_pair(a_l, a_w, b_l, b_w, sep);
        let a_a = a_l * a_w;
        let a_b = b_l * b_w;
        let residual = (f_ab * a_a - f_ba * a_b).abs();
        assert!(
            residual < RECIPROCITY_TOL_1444,
            "reciprocity violated for ({a_l}x{a_w}, {b_l}x{b_w}, sep={sep}): \
             F_AB={f_ab:.9e} F_BA={f_ba:.9e} residual={residual:.3e}"
        );
        // F_AB must be ≤ 1 (it's a directional view factor).
        assert!((0.0..=1.0).contains(&f_ab), "F_AB out of [0, 1]: {f_ab}");
        // F_BA is allowed to exceed 1 only when the receiving zone is smaller
        // (every ray from the small zone hits the larger zone).
        assert!(f_ba >= 0.0, "F_BA negative: {f_ba}");
    }
}

/// Perpendicular orientations do not exchange radiation across a common wall
/// (orthogonal surfaces).  For two zero-area or zero-overlap configurations
/// the reciprocity residual is trivially zero.
#[test]
fn test_issue_1444_zero_overlap_reciprocity() {
    // B entirely disjoint from A's footprint ⇒ F_AB = F_BA = 0.
    let (f_ab, f_ba) = hottels_rectangular_view_factor_pair(8.0, 3.0, 0.0, 2.0, 0.1);
    assert_eq!(f_ab, 0.0);
    assert_eq!(f_ba, 0.0);

    // Zero area on either side: reciprocity holds trivially.
    let (f_ab, f_ba) = hottels_rectangular_view_factor_pair(0.0, 3.0, 8.0, 2.0, 0.1);
    assert_eq!(f_ab, 0.0);
    assert_eq!(f_ba, 0.0);
}

/// `build_zone_view_factors` must satisfy per-wall reciprocity for any
/// rectangular configuration.
#[test]
fn test_issue_1444_matrix_builder_reciprocity() {
    let walls = vec![
        CommonWallGeometry {
            zone_a: 0,
            zone_b: 1,
            a_length: 8.0,
            a_width: 3.0,
            b_length: 8.0,
            b_width: 2.0,
            separation: 0.1,
        },
        CommonWallGeometry {
            zone_a: 0,
            zone_b: 2,
            a_length: 8.0,
            a_width: 3.0,
            b_length: 4.0,
            b_width: 1.0,
            separation: 0.2,
        },
        CommonWallGeometry {
            zone_a: 1,
            zone_b: 2,
            a_length: 8.0,
            a_width: 2.0,
            b_length: 8.0,
            b_width: 2.5,
            separation: 0.05,
        },
    ];
    // Building the matrix triggers a `debug_assert!` per wall that checks
    // `F_AB * A_A == F_BA * A_B`.  Re-run here as a release-mode test too.
    let m = build_zone_view_factors(3, &walls);

    for w in &walls {
        let (i, j) = (w.zone_a, w.zone_b);
        let f_ab = m[(j, i)]; // F[j, i] = view factor from i to j
        let f_ba = m[(i, j)]; // F[i, j] = view factor from j to i
        let a_a = w.a_length * w.a_width;
        let a_b = w.b_length * w.b_width;
        let residual = (f_ab * a_a - f_ba * a_b).abs();
        assert!(
            residual < RECIPROCITY_TOL_1444,
            "matrix reciprocity violated for wall {w:?}: \
             F_AB*A_A={:.6e} F_BA*A_B={:.6e} residual={:.3e}",
            f_ab * a_a,
            f_ba * a_b,
            residual
        );
    }
}
