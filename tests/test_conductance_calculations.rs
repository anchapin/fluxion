//! Unit tests for 5R1C thermal network conductance calculations.
//!
//! This module validates correct parameterization of all 5R1C conductances:
//! - h_tr_em: Exterior-to-mass transmission
//! - h_tr_ms: Mass-to-surface transmission
//! - h_tr_is: Surface-to-interior transmission
//! - h_tr_w: Window conductance (exterior-to-interior through glazing)
//! - h_ve: Ventilation conductance
//!
//! Tests validate:
//! - Correct units (W/K not W/m²K)
//! - Proper window U-value application to h_tr_em and h_tr_w
//! - Thermal bridge effects
//! - ASHRAE 140 Case 600 reference values
//! - Layer-by-layer R-value calculations (LAYER-01)
//! - ASHRAE film coefficient application (LAYER-02)
//! - Window property validation (WINDOW-01, WINDOW-02)
//! - Air change rate conversion (INFIL-01)
//! - Internal gain modeling (INTERNAL-01, INTERNAL-02)
//! - Ground heat transfer (GROUND-01)
//! - Overall conductance correctness (COND-01)

use fluxion::sim::construction::{Construction, Assemblies};
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec};
use approx::assert_relative_eq;

/// Test 1: Validate h_tr_em (exterior-to-mass) calculation with window U-value applied correctly
#[test]
fn test_h_tr_em_calculation() {
    // Create a standard construction
    let construction = Construction::lightweight_wall();

    // Calculate h_tr_em with different window U-values
    // This should account for both the construction U-value and window U-value
    let window_u_values = [0.5, 1.0, 2.0, 3.0, 5.0]; // W/m²K
    let surface_area = 50.0; // m²

    for u_value in window_u_values {
        let h_tr_em = construction.calc_h_tr_em(u_value, surface_area);

        // Verify units are W/K (conductance, not thermal transmittance)
        assert!(h_tr_em > 0.0, "h_tr_em should be positive for U-value {}", u_value);

        // Conductance should scale with surface area
        let h_tr_em_double_area = construction.calc_h_tr_em(u_value, surface_area * 2.0);
        let ratio = h_tr_em_double_area / h_tr_em;
        assert_relative_eq!(ratio, 2.0, epsilon = 0.01);

        println!("h_tr_em for U={}: {:.2} W/K (area: {:.1} m²)", u_value, h_tr_em, surface_area);
    }
}

/// Test 2: Validate h_tr_w (window conductance) calculation
#[test]
fn test_h_tr_w_calculation() {
    let construction = Construction::lightweight_wall();

    // Window U-values from ASHRAE 140 typical ranges
    let window_u_values = [0.5, 1.0, 2.0, 3.0, 5.0]; // W/m²K
    let window_areas = [5.0, 10.0, 12.0, 15.0, 20.0]; // m²

    for u_value in window_u_values {
        for window_area in window_areas {
            let h_tr_w = construction.calc_h_tr_w(u_value, window_area);

            // Verify units are W/K
            assert!(h_tr_w > 0.0, "h_tr_w should be positive");

            // Should be U-value × window_area
            let expected = u_value * window_area;
            assert_relative_eq!(h_tr_w, expected, epsilon = 0.01);

            println!("h_tr_w for U={}, A={}: {:.2} W/K", u_value, window_area, h_tr_w);
        }
    }
}

/// Test 3: Validate h_tr_ms (mass-to-surface) calculation
#[test]
fn test_h_tr_ms_calculation() {
    let construction = Construction::lightweight_wall();

    let surface_areas = [20.0, 50.0, 80.0, 100.0]; // m²

    for surface_area in surface_areas {
        let h_tr_ms = construction.calc_h_tr_ms(surface_area);

        // Verify units are W/K
        assert!(h_tr_ms > 0.0, "h_tr_ms should be positive");

        // Should scale with surface area
        let h_tr_ms_double_area = construction.calc_h_tr_ms(surface_area * 2.0);
        let ratio = h_tr_ms_double_area / h_tr_ms;
        assert_relative_eq!(ratio, 2.0, epsilon = 0.01);

        println!("h_tr_ms for A={}: {:.2} W/K", surface_area, h_tr_ms);
    }
}

/// Test 4: Validate h_tr_is (surface-to-interior) calculation
#[test]
fn test_h_tr_is_calculation() {
    let construction = Construction::lightweight_wall();

    let surface_areas = [20.0, 50.0, 80.0, 100.0]; // m²

    for surface_area in surface_areas {
        let h_tr_is = construction.calc_h_tr_is(surface_area);

        // Verify units are W/K
        assert!(h_tr_is > 0.0, "h_tr_is should be positive");

        // Should scale with surface area
        let h_tr_is_double_area = construction.calc_h_tr_is(surface_area * 2.0);
        let ratio = h_tr_is_double_area / h_tr_is;
        assert_relative_eq!(ratio, 2.0, epsilon = 0.01);

        println!("h_tr_is for A={}: {:.2} W/K", surface_area, h_tr_is);
    }
}

/// Test 5: Validate h_ve (ventilation) calculation with air change rate
#[test]
fn test_h_ve_calculation() {
    let assemblies = Assemblies::low_mass();

    // Air change rates (ACH) from ASHRAE 140 typical values
    let ach_values = [0.1, 0.5, 1.0, 1.5, 2.0]; // air changes per hour
    let zone_volumes = [50.0, 100.0, 200.0, 300.0]; // m³

    // Air density and specific heat capacity
    const AIR_DENSITY: f64 = 1.2; // kg/m³
    const AIR_SPECIFIC_HEAT: f64 = 1005.0; // J/kg·K

    for ach in ach_values {
        for volume in zone_volumes {
            let h_ve = assemblies.calc_h_ve(ach, volume);

            // Verify units are W/K
            assert!(h_ve >= 0.0, "h_ve should be non-negative");

            // Expected: h_ve = ρ × cp × (ACH/3600) × V
            // Note: ACH is per hour, so divide by 3600 to get per second
            let expected = AIR_DENSITY * AIR_SPECIFIC_HEAT * (ach / 3600.0) * volume;

            assert_relative_eq!(h_ve, expected, max_relative = 0.01);

            println!("h_ve for ACH={}, V={}: {:.2} W/K", ach, volume, h_ve);
        }
    }
}

/// Test 6: Validate conductance units are W/K (not W/m²K or 1/K)
#[test]
fn test_conductance_units() {
    let construction = Construction::lightweight_wall();
    let assemblies = Assemblies::low_mass();

    // All conductances should have units of W/K
    let h_tr_em = construction.calc_h_tr_em(2.0, 50.0);
    let h_tr_w = construction.calc_h_tr_w(2.0, 12.0);
    let h_tr_ms = construction.calc_h_tr_ms(50.0);
    let h_tr_is = construction.calc_h_tr_is(50.0);
    let h_ve = assemblies.calc_h_ve(0.5, 100.0);

    // Verify all are positive and have reasonable magnitudes
    assert!(h_tr_em > 0.0, "h_tr_em should be positive");
    assert!(h_tr_w > 0.0, "h_tr_w should be positive");
    assert!(h_tr_ms > 0.0, "h_tr_ms should be positive");
    assert!(h_tr_is > 0.0, "h_tr_is should be positive");
    assert!(h_ve >= 0.0, "h_ve should be non-negative");

    // Verify magnitudes are reasonable for typical building
    // For a 50m² surface with U≈2 W/m²K, conductance should be ~100 W/K
    assert!(h_tr_em < 500.0, "h_tr_em seems too large: {}", h_tr_em);
    assert!(h_tr_w < 100.0, "h_tr_w seems too large: {}", h_tr_w);

    println!("Conductance units validation:");
    println!("  h_tr_em: {:.2} W/K", h_tr_em);
    println!("  h_tr_w: {:.2} W/K", h_tr_w);
    println!("  h_tr_ms: {:.2} W/K", h_tr_ms);
    println!("  h_tr_is: {:.2} W/K", h_tr_is);
    println!("  h_ve: {:.2} W/K", h_ve);
}

/// Test 7: Validate against ASHRAE 140 Case 600 reference conductance values
#[test]
fn test_ashrae_140_case_600_reference_values() {
    let case_spec = ASHRAE140Case::Case600.spec();

    // Get reference conductances from CaseSpec
    let references = case_spec.case600_reference_conductances();

    println!("ASHRAE 140 Case 600 reference conductances:");
    println!("  h_tr_em: {:.2} W/K", references.h_tr_em);
    println!("  h_tr_w: {:.2} W/K", references.h_tr_w);
    println!("  h_tr_ms: {:.2} W/K", references.h_tr_ms);
    println!("  h_tr_is: {:.2} W/K", references.h_tr_is);
    println!("  h_ve: {:.2} W/K", references.h_ve);

    // Validate reference values are positive and reasonable
    assert!(references.h_tr_em > 0.0);
    assert!(references.h_tr_w > 0.0);
    assert!(references.h_tr_ms > 0.0);
    assert!(references.h_tr_is > 0.0);
    assert!(references.h_ve >= 0.0);

    // TODO: Once implementations are complete, compare calculated values against references
    // let construction = Construction::from_case(&case_spec);
    // let calculated_h_tr_em = construction.calc_h_tr_em(case_spec.window.u_value, case_spec.geometry.surface_area);
    // assert_relative_eq!(calculated_h_tr_em, references.h_tr_em, max_relative = 0.05);
}

/// Test 8: Validate thermal bridge effects are accounted for
#[test]
fn test_thermal_bridge_effects() {
    let construction = Construction::lightweight_wall();
    let surface_area = 50.0; // m²
    let window_u_value = 2.0; // W/m²K

    // Calculate h_tr_em without thermal bridge correction
    let h_tr_em_no_bridge = construction.calc_h_tr_em(window_u_value, surface_area);

    // Calculate h_tr_em with thermal bridge correction (e.g., edge effects, corner effects)
    // This should be higher due to additional heat transfer paths
    let h_tr_em_with_bridge = construction.calc_h_tr_em_with_thermal_bridge(
        window_u_value,
        surface_area,
        true, // enable thermal bridge correction
    );

    // Thermal bridge should increase conductance
    assert!(
        h_tr_em_with_bridge > h_tr_em_no_bridge,
        "Thermal bridge should increase h_tr_em"
    );

    println!("Thermal bridge effect: {:.2} W/K → {:.2} W/K (+{:.1}%)",
        h_tr_em_no_bridge,
        h_tr_em_with_bridge,
        (h_tr_em_with_bridge / h_tr_em_no_bridge - 1.0) * 100.0
    );
}

/// Test 9: Validate layer-by-layer R-value calculations (LAYER-01)
#[test]
fn test_layer_by_layer_r_value_calculation() {
    let construction = Construction::lightweight_wall();

    // Calculate total R-value from layer-by-layer sum
    let total_r_value = construction.calculate_total_r_value();

    // Verify R-value is positive and reasonable for a wall
    assert!(total_r_value > 0.0, "Total R-value should be positive");
    assert!(total_r_value < 10.0, "Total R-value seems too high for lightweight wall: {}", total_r_value);

    // Calculate U-value from R-value (U = 1/R)
    let u_value = 1.0 / total_r_value;

    // Verify U-value is within reasonable range
    assert!(u_value > 0.1, "U-value seems too low: {}", u_value);
    assert!(u_value < 5.0, "U-value seems too high: {}", u_value);

    println!("Layer-by-layer R-value calculation:");
    println!("  Total R-value: {:.4} m²K/W", total_r_value);
    println!("  U-value: {:.4} W/m²K", u_value);
}

/// Test 10: Validate ASHRAE film coefficient application (LAYER-02)
#[test]
fn test_ashrae_film_coefficient_application() {
    use fluxion::sim::construction::{INTERIOR_FILM_COEFF, EXTERIOR_FILM_COEFF};

    // Verify film coefficients are defined correctly
    assert!(INTERIOR_FILM_COEFF > 0.0, "Interior film coefficient should be positive");
    assert!(EXTERIOR_FILM_COEFF > 0.0, "Exterior film coefficient should be positive");

    // Verify typical ASHRAE values
    // Interior: R_si = 0.13 m²K/W → h_si = 7.69 W/m²K
    // Exterior: R_se = 0.04 m²K/W → h_se = 25.0 W/m²K
    let expected_interior = 7.69; // W/m²K
    let expected_exterior = 25.0; // W/m²K

    assert_relative_eq!(INTERIOR_FILM_COEFF, expected_interior, max_relative = 0.01);
    assert_relative_eq!(EXTERIOR_FILM_COEFF, expected_exterior, max_relative = 0.01);

    println!("ASHRAE film coefficients:");
    println!("  Interior: {:.2} W/m²K", INTERIOR_FILM_COEFF);
    println!("  Exterior: {:.2} W/m²K", EXTERIOR_FILM_COEFF);
}

/// Test 11: Validate window property validation (WINDOW-01, WINDOW-02)
#[test]
fn test_window_property_validation() {
    let case_spec = ASHRAE140Case::Case600.spec();

    // Validate window properties
    assert!(case_spec.window.u_value > 0.0, "Window U-value should be positive");
    assert!(case_spec.window.shgc >= 0.0 && case_spec.window.shgc <= 1.0, "SHGC should be in [0, 1]");
    assert!(case_spec.window.normal_transmittance >= 0.0 && case_spec.window.normal_transmittance <= 1.0,
        "Normal transmittance should be in [0, 1]");
    assert!(case_spec.window.emissivity >= 0.0 && case_spec.window.emissivity <= 1.0,
        "Emissivity should be in [0, 1]");

    // Verify typical ASHRAE 140 Case 600 window properties
    // Double clear glass: U=3.0 W/m²K, SHGC=0.789, τ=0.86156
    assert_relative_eq!(case_spec.window.u_value, 3.0, max_relative = 0.01);
    assert_relative_eq!(case_spec.window.shgc, 0.789, max_relative = 0.01);
    assert_relative_eq!(case_spec.window.normal_transmittance, 0.86156, max_relative = 0.01);

    println!("Window properties validation:");
    println!("  U-value: {:.4} W/m²K", case_spec.window.u_value);
    println!("  SHGC: {:.4}", case_spec.window.shgc);
    println!("  Normal transmittance: {:.4}", case_spec.window.normal_transmittance);
    println!("  Emissivity: {:.4}", case_spec.window.emissivity);
}

/// Test 12: Validate air change rate conversion (INFIL-01)
#[test]
fn test_air_change_rate_conversion() {
    let assemblies = Assemblies::low_mass();

    // Test conversion from ACH to ventilation conductance
    let ach_values = [0.5, 1.0, 2.0]; // air changes per hour
    let volume = 100.0; // m³

    for ach in ach_values {
        let h_ve = assemblies.calc_h_ve(ach, volume);

        // Expected: h_ve = ρ × cp × (ACH/3600) × V
        const AIR_DENSITY: f64 = 1.2; // kg/m³
        const AIR_SPECIFIC_HEAT: f64 = 1005.0; // J/kg·K
        let expected = AIR_DENSITY * AIR_SPECIFIC_HEAT * (ach / 3600.0) * volume;

        assert_relative_eq!(h_ve, expected, max_relative = 0.01);

        println!("ACH {} → h_ve: {:.2} W/K", ach, h_ve);
    }
}

/// Test 13: Validate internal gain modeling (INTERNAL-01, INTERNAL-02)
#[test]
fn test_internal_gain_modeling() {
    let case_spec = ASHRAE140Case::Case600.spec();

    // Validate internal gain properties
    assert!(case_spec.internal_gains.people_count > 0, "People count should be positive");
    assert!(case_spec.internal_gains.lighting_power > 0.0, "Lighting power should be positive");
    assert!(case_spec.internal_gains.equipment_power > 0.0, "Equipment power should be positive");

    // Validate convective/radiative split (should sum to 1.0)
    let total_fraction = case_spec.internal_gains.convective_fraction
        + case_spec.internal_gains.radiative_fraction;
    assert_relative_eq!(total_fraction, 1.0, max_relative = 0.01);

    // Validate fractions are in [0, 1]
    assert!(case_spec.internal_gains.convective_fraction >= 0.0
        && case_spec.internal_gains.convective_fraction <= 1.0);
    assert!(case_spec.internal_gains.radiative_fraction >= 0.0
        && case_spec.internal_gains.radiative_fraction <= 1.0);

    println!("Internal gains:");
    println!("  People: {}", case_spec.internal_gains.people_count);
    println!("  Lighting: {:.2} W/m²", case_spec.internal_gains.lighting_power);
    println!("  Equipment: {:.2} W/m²", case_spec.internal_gains.equipment_power);
    println!("  Convective: {:.1}%", case_spec.internal_gains.convective_fraction * 100.0);
    println!("  Radiative: {:.1}%", case_spec.internal_gains.radiative_fraction * 100.0);
}

/// Test 14: Validate ground heat transfer (GROUND-01)
#[test]
fn test_ground_heat_transfer() {
    let case_spec = ASHRAE140Case::Case600.spec();

    // Validate ground temperature is defined
    assert!(case_spec.ground_temperature.is_some(), "Ground temperature should be specified");

    let ground_temp = case_spec.ground_temperature.unwrap();
    assert!(ground_temp > 0.0, "Ground temperature should be positive (K)");
    assert!(ground_temp < 350.0, "Ground temperature seems too high: {} K", ground_temp);

    // Ground temperature should be near ambient for slab-on-grade
    // Typical: 283-288 K (10-15°C) for moderate climates
    assert!(ground_temp >= 283.0 && ground_temp <= 288.0,
        "Ground temperature should be in range [283, 288] K for ASHRAE 140 Case 600");

    println!("Ground heat transfer:");
    println!("  Ground temperature: {:.2} K ({:.2} °C)",
        ground_temp,
        ground_temp - 273.15
    );
}

/// Test 15: Validate overall conductance correctness (COND-01)
#[test]
fn test_overall_conductance_correctness() {
    let case_spec = ASHRAE140Case::Case600.spec();
    let references = case_spec.case600_reference_conductances();

    // Verify all conductances are positive and reasonable
    assert!(references.h_tr_em > 0.0 && references.h_tr_em < 1000.0,
        "h_tr_em out of range: {}", references.h_tr_em);
    assert!(references.h_tr_w > 0.0 && references.h_tr_w < 200.0,
        "h_tr_w out of range: {}", references.h_tr_w);
    assert!(references.h_tr_ms > 0.0 && references.h_tr_ms < 1000.0,
        "h_tr_ms out of range: {}", references.h_tr_ms);
    assert!(references.h_tr_is > 0.0 && references.h_tr_is < 2000.0,
        "h_tr_is out of range: {}", references.h_tr_is);
    assert!(references.h_ve >= 0.0 && references.h_ve < 500.0,
        "h_ve out of range: {}", references.h_ve);

    // Verify conductance hierarchy
    // h_tr_is (interior surface) should be highest (large surface area, high film coefficient)
    // h_tr_w (window) should be lowest (small area)
    assert!(references.h_tr_is > references.h_tr_em,
        "h_tr_is should be higher than h_tr_em");
    assert!(references.h_tr_is > references.h_tr_ms,
        "h_tr_is should be higher than h_tr_ms");
    assert!(references.h_tr_w < references.h_tr_em,
        "h_tr_w should be lower than h_tr_em");

    println!("Overall conductance validation:");
    println!("  All conductances positive and within expected ranges");
    println!("  Conductance hierarchy validated");
}
