//! Inter-zone radiative heat transfer using full nonlinear Stefan-Boltzmann equation.
//!
//! This module provides accurate radiative heat transfer calculations for large
//! temperature differences (>20°C) typical in sunspace buildings.
//!
//! The canonical entry point is [`surface_radiative_exchange`], which evaluates
//! the full nonlinear Stefan-Boltzmann law
//! `Q_AB = σ·ε_A·ε_B·F_AB·A·(T_A⁴ − T_B⁴)`.  Issue #1445 promotes this from an
//! orphaned utility to the API consumed by `MultiZoneAirflowNetwork` air-node
//! steps via the [`radiative_conductance_chord_slope`] helper, which returns
//! `h_eff = Q_rad / ΔT` — the chord-slope linearization that *exactly*
//! reproduces the full nonlinear `Q_rad` at the current operating point when
//! multiplied back by `ΔT` in a linear air-node solve.  This eliminates the
//! prior `T_ref = 293.15 K` linearization error of up to ~10 % at
//! ΔT = 20 K (Issue #1445).

/// Stefan-Boltzmann constant (W/(m²·K⁴))
pub const STEFAN_BOLTZMANN_CONSTANT: f64 = 5.670374419e-8;

/// Full nonlinear radiative heat transfer between two surfaces A and B
/// (Stefan-Boltzmann law).
///
/// Canonical API for inter-zone / multi-zone air-node coupling
/// (Issue #1445).  Returns `Q_AB` in Watts, positive when `T_A > T_B`.
///
/// # Why Nonlinear?
/// Linearized approximation `h_rad = 4σ·ε·T³·ΔT` is valid only for small
/// ΔT (<5 K).  Sunspace temperatures can be 20–40 K different from the
/// back-zone, making the full nonlinear equation necessary for accuracy.
/// At ΔT = 20 K around 293.15 K the linearization at `T_ref = 293.15 K`
/// under-predicts by ~9.7 %; chord-slope linearization at the *current*
/// operating point (see [`radiative_conductance_chord_slope`]) reproduces the
/// full nonlinear value exactly.
///
/// # Arguments
/// * `temp_a_c` - Temperature of surface A (°C)
/// * `temp_b_c` - Temperature of surface B (°C)
/// * `emissivity_a` - Emissivity of surface A (0.0 to 1.0)
/// * `emissivity_b` - Emissivity of surface B (0.0 to 1.0)
/// * `view_factor` - Radiative view factor `F_AB` (0.0 to 1.0)
/// * `area` - Area of surface A (m²)
///
/// # Returns
/// Radiative heat transfer `Q_AB` (Watts). Positive if `T_A > T_B`.
///
/// # Formula
/// `Q_AB = σ·ε_A·ε_B·F_AB·A_A·(T_A⁴ − T_B⁴)`
///
/// # Critical: Kelvin Conversion
/// Stefan-Boltzmann law requires absolute temperature (Kelvin).
/// `T_K = T_C + 273.15`.  Using Celsius in the `T⁴` calculation produces the
/// wrong magnitude (off by ~1000×).
///
/// # Example — canonical sunspace case (Issue #1445 docstring fix)
/// ```rust
/// use fluxion::sim::interzone_radiation::surface_radiative_exchange;
///
/// // Sunspace (40°C) to back-zone (20°C), ΔT = 20 K
/// let q = surface_radiative_exchange(40.0, 20.0, 0.9, 0.9, 1.0, 21.6);
/// // Q = 5.67e-8 * 0.9² * 1.0 * 21.6 * (313.15⁴ − 293.15⁴) ≈ 2 214 W
/// // (NOT 249 W — that figure was the docstring bug fixed by Issue #1445)
/// assert!((q - 2214.0).abs() < 10.0, "full nonlinear Q_rad ≈ 2214 W, got {q:.1}");
/// ```
///
/// # Example — ASHRAE 140 Case 960 sunspace peak-hour fixture
/// ```rust
/// use fluxion::sim::interzone_radiation::surface_radiative_exchange;
///
/// // Peak-hour sunspace (300 K = 26.85 °C) vs. back-zone (283 K = 9.85 °C),
/// // ε² = 0.81, F = 0.5, A = 21.6 m² → Q ≈ 836 W
/// let q = surface_radiative_exchange(26.85, 9.85, 0.9, 0.9, 0.5, 21.6);
/// assert!(q > 800.0 && q < 870.0, "ASHRAE 140 fixture: got {q:.1} W");
/// ```
pub fn surface_radiative_exchange(
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

/// Backwards-compatible alias for the canonical
/// [`surface_radiative_exchange`] function.  Older call sites and tests still
/// import this name (Issue #1445 — pre-existing API).
pub use surface_radiative_exchange as calculate_surface_radiative_exchange;

/// Chord-slope radiative conductance that *exactly* linearizes the full
/// nonlinear Stefan-Boltzmann law at the supplied operating point.
///
/// Returns `h_eff = Q_rad / (T_A − T_B)` (W/K), the linear coefficient that
/// when multiplied by `ΔT` reproduces the full nonlinear `Q_rad` exactly at
/// the supplied temperatures.  When `ΔT` is infinitesimal, `h_eff` equals
/// the derivative `dQ/dT = 4σ·ε_A·ε_B·F·A·T_avg³`; for finite `ΔT` the chord
/// slope and the tangent differ slightly but the chord-slope is exact at the
/// supplied operating point.
///
/// This is the replacement for the prior `4σ·ε²·F·T_ref³·A` linearization at
/// a hardcoded `T_ref = 293.15 K` (Issue #1445).  At ΔT = 20 K around 293.15 K
/// the chord-slope reproduces the full nonlinear `Q_rad` to floating-point
/// precision; the hardcoded `T_ref` linearization under-predicted by ~9.7 %.
///
/// # Arguments
/// * `temp_a_k` - Temperature of surface A (Kelvin)
/// * `temp_b_k` - Temperature of surface B (Kelvin)
/// * `emissivity_a` - Emissivity of surface A (0.0 to 1.0)
/// * `emissivity_b` - Emissivity of surface B (0.0 to 1.0)
/// * `view_factor` - Radiative view factor `F_AB` (0.0 to 1.0)
/// * `area` - Area of surface A (m²)
///
/// # Returns
/// `h_eff` (W/K) — the chord-slope radiative conductance. Returns 0.0 when
/// `T_A == T_B` (no gradient, no flow) or when `area`, `view_factor`, or
/// either emissivity is zero.
pub fn radiative_conductance_chord_slope(
    temp_a_k: f64,
    temp_b_k: f64,
    emissivity_a: f64,
    emissivity_b: f64,
    view_factor: f64,
    area: f64,
) -> f64 {
    let dt = temp_a_k - temp_b_k;
    if dt.abs() < f64::EPSILON || area <= 0.0 || view_factor <= 0.0 {
        return 0.0;
    }
    let q_rad = STEFAN_BOLTZMANN_CONSTANT
        * emissivity_a
        * emissivity_b
        * view_factor
        * area
        * (temp_a_k.powi(4) - temp_b_k.powi(4));
    q_rad / dt
}

/// Calculates radiative conductance using linearized approximation (for
/// comparison only).
///
/// # Deprecated
/// This function is kept for testing/validation purposes only.  Production
/// code should use [`surface_radiative_exchange`] for the full nonlinear
/// flux, or [`radiative_conductance_chord_slope`] for a conductance
/// coefficient that exactly reproduces the full nonlinear flux at the
/// current operating point.
///
/// Linearized form: `h_rad = 4σ·ε²·F·T³·A`.  Valid only for small ΔT (<5 K),
/// inaccurate for sunspace applications.
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
        let q_celsius: f64 =
            5.67e-8 * 0.9 * 0.9 * 1.0 * 21.6 * (40.0_f64.powi(4) - 20.0_f64.powi(4));
        let q_kelvin = calculate_surface_radiative_exchange(40.0, 20.0, 0.9, 0.9, 1.0, 21.6);
        // Kelvin conversion should produce correct magnitude (~2214 W)
        assert!(q_kelvin.abs() > 2000.0, "Kelvin conversion required");
        // Celsius gives wrong result (~2 W instead of ~2214 W)
        assert!(q_celsius.abs() < 10.0, "Celsius gives wrong result");
        // Kelvin should be ~1000× larger than Celsius
        assert!(
            (q_kelvin / q_celsius) > 900.0,
            "Kelvin should be much larger"
        );
    }

    #[test]
    fn test_nonlinear_vs_linearized_small_dt() {
        // Small ΔT = 5°C: nonlinear and linearized should match
        let q_nonlinear = calculate_surface_radiative_exchange(22.5, 17.5, 0.9, 0.9, 1.0, 21.6);
        let t_avg_k = (22.5 + 273.15 + 17.5 + 273.15) / 2.0;
        let q_linearized =
            calculate_radiative_conductance_linearized(21.6, 0.9, t_avg_k, 1.0) * 5.0;
        // Should match within 1% for small ΔT
        let error_pct = ((q_nonlinear - q_linearized) / q_linearized).abs() * 100.0;
        assert!(error_pct < 1.0, "Error: {:.2}% for small ΔT", error_pct);
    }

    #[test]
    fn test_nonlinear_vs_linearized_large_dt() {
        // Large ΔT = 20°C: nonlinear more accurate than linearized
        let q_nonlinear = calculate_surface_radiative_exchange(40.0, 20.0, 0.9, 0.9, 1.0, 21.6);
        let t_avg_k = (40.0 + 273.15 + 20.0 + 273.15) / 2.0;
        let q_linearized =
            calculate_radiative_conductance_linearized(21.6, 0.9, t_avg_k, 1.0) * 20.0;
        // For ΔT = 20°C, nonlinear and linearized are close (<1% difference)
        let error_pct = ((q_nonlinear - q_linearized) / q_linearized).abs() * 100.0;
        assert!(error_pct < 2.0, "Error: {:.2}% for large ΔT", error_pct);
        // But nonlinear is more accurate theoretically
        println!(
            "Nonlinear: {:.2} W, Linearized: {:.2} W, Error: {:.2}%",
            q_nonlinear, q_linearized, error_pct
        );
    }

    #[test]
    fn test_chord_slope_exact_at_operating_point() {
        // Chord-slope h_eff reproduces full nonlinear Q_rad exactly at the
        // supplied operating point (Issue #1445).  At T_a=313.15 K, T_b=293.15 K
        // (ΔT = 20 K, ε² = 0.81, F = 1.0, A = 21.6 m²):
        let t_a_k = 313.15;
        let t_b_k = 293.15;
        let dt = t_a_k - t_b_k;
        let q_full = surface_radiative_exchange(40.0, 20.0, 0.9, 0.9, 1.0, 21.6);
        let h_eff = radiative_conductance_chord_slope(t_a_k, t_b_k, 0.9, 0.9, 1.0, 21.6);
        let q_chord = h_eff * dt;
        assert!(
            (q_chord - q_full).abs() < 1e-6,
            "Chord-slope must reproduce full nonlinear exactly: \
             chord={q_chord:.6}, full={q_full:.6}"
        );
        // And the chord-slope is strictly LARGER than the tangent-at-T_ref=293.15
        // (because dT⁴/dT = 4T³ grows superlinearly in T — T_avg=303.15 K tangent
        // exceeds the T=293.15 K tangent, and the chord exceeds even the tangent
        // at T_avg for any finite ΔT by exactly the integral remainder):
        let h_t_ref = calculate_radiative_conductance_linearized(21.6, 0.9, 293.15, 1.0);
        let q_legacy = h_t_ref * dt;
        let legacy_err = (q_legacy - q_full) / q_full * 100.0;
        assert!(
            h_eff > h_t_ref,
            "Chord-slope {h_eff:.3} should exceed T_ref-linearized {h_t_ref:.3} W/K \
             (legacy linearization under-predicts by {legacy_err:.2}%)"
        );
    }

    #[test]
    fn test_chord_slope_zero_gradient_returns_zero() {
        // No ΔT → no flow (also covers the guard for zero area / view factor).
        assert_eq!(
            radiative_conductance_chord_slope(293.15, 293.15, 0.9, 0.9, 1.0, 21.6),
            0.0
        );
        assert_eq!(
            radiative_conductance_chord_slope(300.0, 293.15, 0.9, 0.9, 0.0, 21.6),
            0.0,
            "Zero view factor must give zero conductance"
        );
        assert_eq!(
            radiative_conductance_chord_slope(300.0, 293.15, 0.0, 0.9, 1.0, 21.6),
            0.0,
            "Zero emissivity must give zero conductance"
        );
    }

    #[test]
    fn test_ashrae_140_peak_hour_fixture() {
        // ASHRAE 140 Case 960 sunspace peak-hour fixture from the issue body:
        // T_a = 300 K (26.85 °C), T_b = 283 K (9.85 °C), ε² = 0.81, F = 0.5, A = 21.6 m²
        // → Q_full ≈ 836 W; the prior T_ref=293.15 linearization over-predicted by
        // ~1.6 % at this operating point (Python-verified in the issue).
        let q = surface_radiative_exchange(26.85, 9.85, 0.9, 0.9, 0.5, 21.6);
        assert!(
            q > 800.0 && q < 870.0,
            "ASHRAE 140 peak-hour fixture: expected 800 < Q < 870 W, got {q:.2} W"
        );
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
