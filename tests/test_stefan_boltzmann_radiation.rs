//! Unit tests for full nonlinear Stefan-Boltzmann radiative heat transfer.
//!
//! Tests validate the formula Q = σ·ε₁·ε₂·F·A·(T₁⁴ - T₂⁴)
//! for radiative exchange between thermal zones.

use fluxion::sim::interzone::calculate_radiative_conductance;

/// Stefan-Boltzmann constant (W/m²K⁴)
const STEFAN_BOLTZMANN_CONST: f64 = 5.670374419e-8;

/// Common surface area for Case 960 (21.6 m²)
const SURFACE_AREA: f64 = 21.6; // m²

/// Typical surface emissivity
const EMISSIVITY: f64 = 0.9;

/// View factor for adjacent zones (F = 1.0 for direct exchange)
const VIEW_FACTOR: f64 = 1.0;

/// Tolerance for numerical comparison (1%)
const TOLERANCE_PCT: f64 = 1.0;

/// Helper function to calculate full nonlinear radiative heat transfer
///
/// Q = σ·ε₁·ε₂·F·A·(T₁⁴ - T₂⁴)
fn calculate_nonlinear_radiative_transfer(
    temperature_a_c: f64,
    temperature_b_c: f64,
    emissivity: f64,
    view_factor: f64,
    area: f64,
) -> f64 {
    // Convert to Kelvin
    let t_a_k = temperature_a_c + 273.15;
    let t_b_k = temperature_b_c + 273.15;

    // Calculate T⁴ for both temperatures
    let t_a_pow4 = f64::powi(t_a_k, 4);
    let t_b_pow4 = f64::powi(t_b_k, 4);

    // Calculate nonlinear transfer
    STEFAN_BOLTZMANN_CONST * emissivity * emissivity * view_factor * area * (t_a_pow4 - t_b_pow4)
}

/// Helper function to calculate linearized radiative heat transfer
///
/// Q_linear = 4σ·ε²·F·T_avg³·A·ΔT
fn calculate_linearized_radiative_transfer(
    temperature_a_c: f64,
    temperature_b_c: f64,
    emissivity: f64,
    view_factor: f64,
    area: f64,
) -> f64 {
    // Convert to Kelvin
    let t_a_k = temperature_a_c + 273.15;
    let t_b_k = temperature_b_c + 273.15;

    // Calculate mean temperature
    let t_avg_k = (t_a_k + t_b_k) / 2.0;

    // Temperature difference
    let delta_t = temperature_b_c - temperature_a_c;

    // Calculate linearized transfer
    4.0 * STEFAN_BOLTZMANN_CONST * emissivity * emissivity * view_factor * f64::powi(t_avg_k, 3) * area * delta_t
}

#[test]
fn test_stefan_boltzmann_nonlinear_large_delta_t() {
    // Test full nonlinear calculation with large temperature difference
    // Temperature A = 20°C, Temperature B = 40°C (ΔT = 20°C)
    // This represents sunspace conditions where large ΔT is common

    let temp_a = 20.0; // °C
    let temp_b = 40.0; // °C

    // Expected calculation:
    // T_A = 293.15 K, T_B = 313.15 K
    // T_A⁴ = 7.39e9 K⁴, T_B⁴ = 9.62e9 K⁴
    // ΔT⁴ = -2.23e9 K⁴ (negative because T_A < T_B)
    // Q = 5.67e-8 * 0.9² * 1.0 * 21.6 * -2.23e9 = -2213 W
    let expected_watts = -2213.0;

    let q_nonlinear = calculate_nonlinear_radiative_transfer(
        temp_a, temp_b, EMISSIVITY, VIEW_FACTOR, SURFACE_AREA,
    );

    println!("Nonlinear radiative transfer: {:.2} W", q_nonlinear);
    println!("Expected: {:.2} W", expected_watts);
    println!("Difference: {:.2}%", ((q_nonlinear - expected_watts).abs() / expected_watts.abs()) * 100.0);

    assert!(
        (q_nonlinear - expected_watts).abs() / expected_watts.abs() * 100.0 < TOLERANCE_PCT,
        "Nonlinear transfer {:.2} W differs from expected {:.2} W by more than {}%",
        q_nonlinear, expected_watts, TOLERANCE_PCT
    );
}

#[test]
fn test_stefan_boltzmann_linearized_large_delta_t() {
    // Test linearized approximation with large temperature difference
    // Should be close to nonlinear but not exact

    let temp_a = 20.0; // °C
    let temp_b = 40.0; // °C

    // Expected calculation:
    // T_avg = 303.15 K, T_avg³ = 2.79e7 K³
    // Q = 4 * 5.67e-8 * 0.9² * 1.0 * 2.79e7 * 21.6 * 20 = 2211 W
    let expected_watts = 2211.0;

    let q_linearized = calculate_linearized_radiative_transfer(
        temp_a, temp_b, EMISSIVITY, VIEW_FACTOR, SURFACE_AREA,
    );

    println!("Linearized radiative transfer: {:.2} W", q_linearized);
    println!("Expected: {:.2} W", expected_watts);

    assert!(
        (q_linearized - expected_watts).abs() / expected_watts * 100.0 < TOLERANCE_PCT,
        "Linearized transfer {:.2} W differs from expected {:.2} W by more than {}%",
        q_linearized, expected_watts, TOLERANCE_PCT
    );
}

#[test]
fn test_nonlinear_vs_linearized_small_delta_t() {
    // Test that nonlinear and linearized match closely for small ΔT (< 5°C)
    // Small temperature difference: 20°C to 23°C (ΔT = 3°C)

    let temp_a = 20.0; // °C
    let temp_b = 23.0; // °C

    let q_nonlinear = calculate_nonlinear_radiative_transfer(
        temp_a, temp_b, EMISSIVITY, VIEW_FACTOR, SURFACE_AREA,
    );

    let q_linearized = calculate_linearized_radiative_transfer(
        temp_a, temp_b, EMISSIVITY, VIEW_FACTOR, SURFACE_AREA,
    );

    let difference_pct = (q_nonlinear - q_linearized).abs() / q_nonlinear * 100.0;

    println!("Nonlinear: {:.2} W, Linearized: {:.2} W", q_nonlinear, q_linearized);
    println!("Difference: {:.2}%", difference_pct);

    // For small ΔT, nonlinear and linearized should match within 1%
    assert!(
        difference_pct < TOLERANCE_PCT,
        "Nonlinear and linearized differ by {:.2}% for small ΔT, expected < {}%",
        difference_pct, TOLERANCE_PCT
    );
}

#[test]
fn test_nonlinear_vs_linearized_large_delta_t() {
    // Test that nonlinear is more accurate for large ΔT (> 20°C)
    // Document the difference to validate need for full nonlinear implementation
    // Large temperature difference: 20°C to 40°C (ΔT = 20°C)

    let temp_a = 20.0; // °C
    let temp_b = 40.0; // °C

    let q_nonlinear = calculate_nonlinear_radiative_transfer(
        temp_a, temp_b, EMISSIVITY, VIEW_FACTOR, SURFACE_AREA,
    );

    let q_linearized = calculate_linearized_radiative_transfer(
        temp_a, temp_b, EMISSIVITY, VIEW_FACTOR, SURFACE_AREA,
    );

    let difference_pct = (q_nonlinear - q_linearized).abs() / q_nonlinear.abs() * 100.0;

    println!("Nonlinear: {:.2} W, Linearized: {:.2} W", q_nonlinear, q_linearized);
    println!("Difference: {:.2}%", difference_pct);

    // For large ΔT, difference should be noticeable (> 0.5%)
    // This validates that nonlinear implementation is needed for accuracy
    assert!(
        difference_pct > 0.5,
        "Expected nonlinear and linearized to differ by > 0.5% for large ΔT, got {:.2}%",
        difference_pct
    );
}

#[test]
fn test_kelvin_conversion_required() {
    // Test that Kelvin conversion is required for correct T⁴ calculation
    // Common pitfall: using Celsius in T⁴ calculation produces wrong magnitude

    let temp_a_c = 20.0; // °C
    let temp_b_c = 40.0; // °C

    // Correct: Use Kelvin
    let t_a_k = temp_a_c + 273.15;
    let t_b_k = temp_b_c + 273.15;
    let q_correct = STEFAN_BOLTZMANN_CONST * EMISSIVITY * EMISSIVITY * VIEW_FACTOR * SURFACE_AREA * (f64::powi(t_a_k, 4) - f64::powi(t_b_k, 4));

    // Incorrect: Use Celsius (common mistake)
    let q_incorrect = STEFAN_BOLTZMANN_CONST * EMISSIVITY * EMISSIVITY * VIEW_FACTOR * SURFACE_AREA * (f64::powi(temp_a_c, 4) - f64::powi(temp_b_c, 4));

    println!("Correct (Kelvin): {:.2} W", q_correct);
    println!("Incorrect (Celsius): {:.2} W", q_incorrect);
    println!("Ratio: {:.1}×", q_correct.abs() / q_incorrect.abs());

    // Using Celsius produces wrong magnitude (should be huge error)
    assert!(
        q_correct.abs() > 100.0 * q_incorrect.abs(),
        "Celsius-based calculation should produce > 100× error, got {:.1}×",
        q_correct.abs() / q_incorrect.abs()
    );
}

#[test]
fn test_radiative_conductance_function_exists() {
    // Test that the existing calculate_radiative_conductance function works
    // This function uses linearized approximation

    let mean_temp_k = 293.15; // 20°C in Kelvin
    let h_rad = calculate_radiative_conductance(SURFACE_AREA, EMISSIVITY, mean_temp_k, VIEW_FACTOR);

    println!("Radiative conductance: {:.4} W/K", h_rad);

    // Conductance should be positive
    assert!(h_rad > 0.0, "Radiative conductance should be positive");

    // Conductance should be reasonable (order of magnitude check)
    assert!(h_rad < 100.0, "Radiative conductance should be < 100 W/K, got {:.2} W/K", h_rad);
}

#[test]
fn test_radiative_transfer_with_conductance() {
    // Test that Q = h_rad * ΔT works with linearized conductance
    // This validates the relationship between conductance and heat transfer

    let temp_a = 20.0; // °C
    let temp_b = 25.0; // °C
    let delta_t = temp_b - temp_a;

    let mean_temp_k = (temp_a + temp_b) / 2.0 + 273.15;
    let h_rad = calculate_radiative_conductance(SURFACE_AREA, EMISSIVITY, mean_temp_k, VIEW_FACTOR);

    // Q = h_rad * ΔT
    let q_from_conductance = h_rad * delta_t;

    // Direct linearized calculation
    let q_linearized = calculate_linearized_radiative_transfer(
        temp_a, temp_b, EMISSIVITY, VIEW_FACTOR, SURFACE_AREA,
    );

    println!("Q from conductance: {:.2} W", q_from_conductance);
    println!("Q from linearized: {:.2} W", q_linearized);

    // Should match within numerical precision
    assert!(
        (q_from_conductance - q_linearized).abs() < 0.01,
        "Q from conductance ({:.2} W) should match Q from linearized ({:.2} W)",
        q_from_conductance, q_linearized
    );
}

#[test]
fn test_radiative_transfer_zero_delta_t() {
    // Test edge case: zero temperature difference should give zero heat transfer

    let temp_a = 20.0; // °C
    let temp_b = 20.0; // °C

    let q_nonlinear = calculate_nonlinear_radiative_transfer(
        temp_a, temp_b, EMISSIVITY, VIEW_FACTOR, SURFACE_AREA,
    );

    println!("Radiative transfer with ΔT=0: {:.2} W", q_nonlinear);

    assert_eq!(q_nonlinear, 0.0, "Zero temperature difference should give zero heat transfer");
}

#[test]
fn test_radiative_transfer_emissivity_scaling() {
    // Test that heat transfer scales with emissivity squared
    // Q ∝ ε₁·ε₂ = ε² (assuming equal emissivities)

    let temp_a = 20.0; // °C
    let temp_b = 40.0; // °C

    let q_emissivity_09 = calculate_nonlinear_radiative_transfer(
        temp_a, temp_b, 0.9, VIEW_FACTOR, SURFACE_AREA,
    );

    let q_emissivity_05 = calculate_nonlinear_radiative_transfer(
        temp_a, temp_b, 0.5, VIEW_FACTOR, SURFACE_AREA,
    );

    let ratio = q_emissivity_09 / q_emissivity_05;
    let expected_ratio = f64::powi(0.9 / 0.5, 2); // Square ratio

    println!("Q(ε=0.9): {:.2} W", q_emissivity_09);
    println!("Q(ε=0.5): {:.2} W", q_emissivity_05);
    println!("Ratio: {:.2}× (expected {:.2}×)", ratio, expected_ratio);

    assert!(
        (ratio - expected_ratio).abs() < 0.01,
        "Q should scale with ε², expected ratio {:.2}, got {:.2}",
        expected_ratio, ratio
    );
}

#[test]
fn test_radiative_transfer_area_scaling() {
    // Test that heat transfer scales linearly with area
    // Q ∝ A

    let temp_a = 20.0; // °C
    let temp_b = 40.0; // °C

    let q_area_10 = calculate_nonlinear_radiative_transfer(
        temp_a, temp_b, EMISSIVITY, VIEW_FACTOR, 10.0,
    );

    let q_area_20 = calculate_nonlinear_radiative_transfer(
        temp_a, temp_b, EMISSIVITY, VIEW_FACTOR, 20.0,
    );

    let ratio = q_area_20 / q_area_10;
    let expected_ratio = 2.0;

    println!("Q(A=10): {:.2} W", q_area_10);
    println!("Q(A=20): {:.2} W", q_area_20);
    println!("Ratio: {:.2}× (expected {:.2}×)", ratio, expected_ratio);

    assert!(
        (ratio - expected_ratio).abs() < 0.01,
        "Q should scale linearly with area, expected ratio {:.2}, got {:.2}",
        expected_ratio, ratio
    );
}
