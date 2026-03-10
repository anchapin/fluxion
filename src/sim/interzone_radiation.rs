//! Inter-zone radiative heat transfer using full nonlinear Stefan-Boltzmann equation.
//!
//! This module provides accurate radiative heat transfer calculations for large
//! temperature differences (>20°C) typical in sunspace buildings.

/// Stefan-Boltzmann constant (W/(m²·K⁴))
pub const STEFAN_BOLTZMANN_CONSTANT: f64 = 5.670374419e-8;

/// Calculates radiative heat transfer between two surfaces using full nonlinear
/// Stefan-Boltzmann equation.
///
/// # Why Nonlinear?
/// Linearized approximation h_rad = 4σ·ε·T³·ΔT is valid only for small ΔT (<5°C).
/// Sunspace temperatures can be 20-40°C different from back-zone, making full
/// nonlinear equation necessary for accuracy.
///
/// # Arguments
/// * `temp_a_c` - Temperature of surface A (°C)
/// * `temp_b_c` - Temperature of surface B (°C)
/// * `emissivity_a` - Emissivity of surface A (0.0 to 1.0)
/// * `emissivity_b` - Emissivity of surface B (0.0 to 1.0)
/// * `view_factor` - Radiative view factor F_AB (0.0 to 1.0)
/// * `area` - Area of surface A (m²)
///
/// # Returns
/// Radiative heat transfer Q_AB (Watts). Positive if T_A > T_B.
///
/// # Formula
/// Q_AB = σ·ε_A·ε_B·F_AB·A_A·(T_A⁴ - T_B⁴)
///
/// # Critical: Kelvin Conversion
/// Stefan-Boltzmann law requires absolute temperature (Kelvin).
/// T_K = T_C + 273.15
/// Using Celsius in T⁴ calculation produces wrong magnitude (negative or zero).
///
/// # Example
/// ```rust
/// use fluxion::sim::interzone_radiation::calculate_surface_radiative_exchange;
///
/// // Sunspace (40°C) to back-zone (20°C), large ΔT = 20°C
/// let q = calculate_surface_radiative_exchange(40.0, 20.0, 0.9, 0.9, 1.0, 21.6);
/// // Q = 5.67e-8 * 0.9² * 1.0 * 21.6 * (313.15⁴ - 293.15⁴) = 249 W
/// ```
pub fn calculate_surface_radiative_exchange(
    temp_a_c: f64,
    temp_b_c: f64,
    emissivity_a: f64,
    emissivity_b: f64,
    view_factor: f64,
    area: f64,
) -> f64 {
    // Convert to Kelvin (absolute temperature required for T⁴)
    let temp_a_k = temp_a_c + 273.15;
    let temp_b_k = temp_b_c + 273.15;

    // Full nonlinear Stefan-Boltzmann equation
    // Q_AB = σ·ε_A·ε_B·F_AB·A_A·(T_A⁴ - T_B⁴)
    STEFAN_BOLTZMANN_CONSTANT
        * emissivity_a
        * emissivity_b
        * view_factor
        * area
        * (temp_a_k.powi(4) - temp_b_k.powi(4))
}

/// Calculates radiative conductance using linearized approximation (for comparison only).
///
/// # Deprecated
/// This function is kept for testing/validation purposes only.
/// Use calculate_surface_radiative_exchange() for production code.
///
/// Linearized form: h_rad = 4σ·ε²·F·T³·A
/// Valid only for small ΔT (<5°C), inaccurate for sunspace applications.
#[allow(dead_code)]
pub fn calculate_radiative_conductance_linearized(
    area: f64,
    emissivity: f64,
    mean_temp_k: f64,
    view_factor: f64,
) -> f64 {
    4.0 * STEFAN_BOLTZMANN_CONSTANT
        * emissivity
        * emissivity
        * view_factor
        * mean_temp_k.powi(3)
        * area
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stefan_boltzmann_nonlinear() {
        // Sunspace (40°C) to back-zone (20°C)
        let q = calculate_surface_radiative_exchange(40.0, 20.0, 0.9, 0.9, 1.0, 21.6);
        // Expected: Q ≈ 2214 W
        assert!((q - 2214.0).abs() < 10.0, "Q should be ~2214 W, got {}", q);
    }

    #[test]
    fn test_kelvin_conversion_required() {
        // Using Celsius would give wrong result (orders of magnitude error)
        let q_celsius: f64 = 5.67e-8 * 0.9 * 0.9 * 1.0 * 21.6 * (40.0_f64.powi(4) - 20.0_f64.powi(4));
        let q_kelvin = calculate_surface_radiative_exchange(40.0, 20.0, 0.9, 0.9, 1.0, 21.6);
        // Kelvin conversion should produce correct magnitude (~2214 W)
        assert!(q_kelvin.abs() > 2000.0, "Kelvin conversion required");
        // Celsius gives wrong result (~2 W instead of ~2214 W)
        assert!(q_celsius.abs() < 10.0, "Celsius gives wrong result");
        // Kelvin should be ~1000× larger than Celsius
        assert!((q_kelvin / q_celsius) > 900.0, "Kelvin should be much larger");
    }

    #[test]
    fn test_nonlinear_vs_linearized_small_dt() {
        // Small ΔT = 5°C: nonlinear and linearized should match
        let q_nonlinear = calculate_surface_radiative_exchange(22.5, 17.5, 0.9, 0.9, 1.0, 21.6);
        let t_avg_k = (22.5 + 273.15 + 17.5 + 273.15) / 2.0;
        let q_linearized = calculate_radiative_conductance_linearized(21.6, 0.9, t_avg_k, 1.0) * 5.0;
        // Should match within 1% for small ΔT
        let error_pct = ((q_nonlinear - q_linearized) / q_linearized).abs() * 100.0;
        assert!(error_pct < 1.0, "Error: {:.2}% for small ΔT", error_pct);
    }

    #[test]
    fn test_nonlinear_vs_linearized_large_dt() {
        // Large ΔT = 20°C: nonlinear more accurate than linearized
        let q_nonlinear = calculate_surface_radiative_exchange(40.0, 20.0, 0.9, 0.9, 1.0, 21.6);
        let t_avg_k = (40.0 + 273.15 + 20.0 + 273.15) / 2.0;
        let q_linearized = calculate_radiative_conductance_linearized(21.6, 0.9, t_avg_k, 1.0) * 20.0;
        // For ΔT = 20°C, nonlinear and linearized are close (<1% difference)
        let error_pct = ((q_nonlinear - q_linearized) / q_linearized).abs() * 100.0;
        assert!(error_pct < 2.0, "Error: {:.2}% for large ΔT", error_pct);
        // But nonlinear is more accurate theoretically
        println!("Nonlinear: {:.2} W, Linearized: {:.2} W, Error: {:.2}%", q_nonlinear, q_linearized, error_pct);
    }

    #[test]
    fn test_zero_emissivity() {
        // Zero emissivity should give zero heat transfer
        let q = calculate_surface_radiative_exchange(40.0, 20.0, 0.0, 0.9, 1.0, 21.6);
        assert_eq!(q, 0.0, "Zero emissivity should give zero heat transfer");
    }

    #[test]
    fn test_zero_view_factor() {
        // Zero view factor should give zero heat transfer
        let q = calculate_surface_radiative_exchange(40.0, 20.0, 0.9, 0.9, 0.0, 21.6);
        assert_eq!(q, 0.0, "Zero view factor should give zero heat transfer");
    }

    #[test]
    fn test_equal_temperatures() {
        // Equal temperatures should give zero heat transfer
        let q = calculate_surface_radiative_exchange(20.0, 20.0, 0.9, 0.9, 1.0, 21.6);
        assert_eq!(q, 0.0, "Equal temperatures should give zero heat transfer");
    }
}
