//! Thermal mass integration methods for stable numerical simulation.
//!
//! This module provides implicit integration methods for thermal mass updates
//! to address instability issues with explicit Euler integration when
//! thermal capacitance is high (> 500 J/K).
//!
//! # Background
//!
//! Explicit Euler integration (Tm_new = Tm_old + dt * Q/Cm) becomes unstable
//! for high thermal capacitance systems when dt > Cm / (h_tr_em + h_tr_ms).
//! With dt = 3600s (1 hour) and Cm > 500 J/K, this condition is often violated,
//! leading to oscillatory or divergent solutions.
//!
//! Implicit methods (backward Euler, Crank-Nicolson) are unconditionally stable
//! and handle stiff thermal systems robustly.

use crate::api::error::FluxionError;

#[cfg(test)]
use std::f64::consts::PI;

/// Validates the timestep / thermal-capacitance pair shared by every
/// integrator in this module (Issue #3638).
///
/// A degenerate `dt` or `cm` (zero, negative, or non-finite — e.g. a faulty
/// HVAC override driving `cm = 0` through an adaptive timestep) previously
/// panicked mid-simulation and aborted the whole process. Every integrator
/// now surfaces a typed [`FluxionError::Validation`] instead so callers on
/// the production physics paths can degrade gracefully.
fn validate_dt_cm(dt: f64, cm: f64) -> Result<(), FluxionError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(FluxionError::Validation(format!(
            "Time step dt must be positive and finite, got {dt}"
        )));
    }
    if !cm.is_finite() || cm <= 0.0 {
        return Err(FluxionError::Validation(format!(
            "Thermal capacitance cm must be positive and finite, got {cm}"
        )));
    }
    Ok(())
}

/// Thermal integration method for mass temperature updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalIntegrationMethod {
    /// Explicit Euler integration (forward method)
    /// Stable only when dt < Cm / (h_tr_em + h_tr_ms)
    ExplicitEuler,

    /// Backward Euler integration (implicit method)
    /// Unconditionally stable, 1st-order accurate
    BackwardEuler,

    /// Crank-Nicolson integration (semi-implicit method)
    /// Unconditionally stable, 2nd-order accurate
    CrankNicolson,
}

/// Selects the appropriate integration method based on thermal capacitance.
///
/// For high thermal capacitance (> 500 J/K), uses Crank-Nicolson for
/// 2nd-order accuracy and unconditional stability per ISO 13790 §C.4.
/// For low thermal capacitance, explicit Euler is sufficient and faster.
///
/// # Arguments
/// * `cm` - Thermal capacitance (J/K)
///
/// # Returns
/// * `CrankNicolson` if cm > 500 J/K (ISO 13790 recommended)
/// * `ExplicitEuler` otherwise
///
/// # Example
/// ```
/// use fluxion::sim::thermal_integration::{
///     select_integration_method, ThermalIntegrationMethod,
/// };
///
/// let method = select_integration_method(1000.0);
/// assert_eq!(method, ThermalIntegrationMethod::CrankNicolson);
/// ```
pub fn select_integration_method(cm: f64) -> ThermalIntegrationMethod {
    // ISO 13790 §C.4 recommends Crank-Nicolson for high thermal mass
    // Crank-Nicolson is unconditionally stable (A-stable) and 2nd-order accurate
    const HIGH_MASS_THRESHOLD: f64 = 500.0;

    if cm > HIGH_MASS_THRESHOLD {
        ThermalIntegrationMethod::CrankNicolson
    } else {
        ThermalIntegrationMethod::ExplicitEuler
    }
}

/// Backward Euler solver for implicit thermal mass update.
///
/// Solves the implicit equation:
/// Cm * (Tm_new - Tm_old) / dt = h_tr_em * (t_ext - Tm_new) + h_tr_ms * (t_surface - Tm_new) + phi_m
///
/// Rearranged to solve for Tm_new:
/// (Cm/dt + h_tr_em + h_tr_ms) * Tm_new = Cm/dt * Tm_old + h_tr_em * t_ext + h_tr_ms * t_surface + phi_m
///
/// # Arguments
/// * `tm_old` - Previous mass temperature (°C)
/// * `dt` - Time step (seconds)
/// * `cm` - Thermal capacitance (J/K)
/// * `h_tr_em` - Exterior-to-mass conductance (W/K)
/// * `h_tr_ms` - Mass-to-surface conductance (W/K)
/// * `t_ext` - Exterior temperature (°C)
/// * `t_surface` - Surface temperature (°C)
/// * `phi_m` - Direct gains to thermal mass (W)
///
/// # Returns
/// * New mass temperature (°C)
///
/// # Errors
/// * [`FluxionError::Validation`] when `dt <= 0.0`, `cm <= 0.0`, or either
///   value is non-finite (Issue #3638 — was a `panic!`).
///
/// # Stability
/// Unconditionally stable for any time step size.
///
/// # Accuracy
/// 1st-order accurate: error = O(dt^2)
///
/// # Example
/// ```
/// use fluxion::sim::thermal_integration::backward_euler_update;
///
/// let tm_new = backward_euler_update(
///     20.0,  // tm_old
///     3600.0, // dt (1 hour)
///     1000.0, // cm
///     10.0,   // h_tr_em
///     100.0,  // h_tr_ms
///     -5.0,   // t_ext
///     22.0,   // t_surface
///     500.0,  // phi_m
/// )
/// .unwrap();
/// ```
#[allow(clippy::too_many_arguments)]
pub fn backward_euler_update(
    tm_old: f64,
    dt: f64,
    cm: f64,
    h_tr_em: f64,
    h_tr_ms: f64,
    t_ext: f64,
    t_surface: f64,
    phi_m: f64,
) -> Result<f64, FluxionError> {
    validate_dt_cm(dt, cm)?;

    // Calculate denominator: (Cm/dt + h_tr_em + h_tr_ms)
    let denom = cm / dt + h_tr_em + h_tr_ms;

    // Calculate numerator: Cm/dt * Tm_old + h_tr_em * t_ext + h_tr_ms * t_surface + phi_m
    let numer = cm / dt * tm_old + h_tr_em * t_ext + h_tr_ms * t_surface + phi_m;

    // Return new temperature
    Ok(numer / denom)
}

/// ISO 13790 Crank-Nicolson mass temperature update (§C.4).
///
/// The Crank-Nicolson scheme averages the conductance terms between old and new
/// time steps for 2nd-order accuracy. The mass energy balance is:
///
///   Cm/dt × (Tm_new − Tm_old) = H_tr_em × (t_ext − Tm_avg) + H_tr_3 × (t_sup − Tm_avg) + phi_m
///
/// where Tm_avg = 0.5 × (Tm_old + Tm_new). Rearranged:
///
///   Tm_new = [Tm_old × (Cm/dt − ½(H_tr_em + H_tr_3)) + H_tr_em × t_ext + H_tr_3 × t_sup + phi_m]
///            / [Cm/dt + ½(H_tr_em + H_tr_3)]
///
/// Issue #917 fix: the previous version omitted the H_tr_em × t_ext + H_tr_3 × t_sup
/// driving terms, leaving the mass node coupled only to phi_m (solar gains). This
/// suppressed all free-floating temperatures by ~30 °C.
///
/// # Arguments
/// * `tm_prev` - Previous mass temperature (°C)
/// * `dt` - Time step in seconds
/// * `cm` - Thermal capacitance of mass (J/K)
/// * `h_tr_3` - ISO 13790 combined conductance H_tr_3 (W/K)
/// * `h_tr_em` - Mass-to-exterior conductance (W/K)
/// * `t_ext` - Exterior (sol-air) temperature driving the h_tr_em path (°C)
/// * `t_sup` - Supply / surface temperature driving the h_tr_3 path (°C)
/// * `phi_m_tot` - Total heat flow to mass node (W), includes HVAC via network
///
/// # Returns
/// * New mass temperature (°C)
///
/// # Errors
/// * [`FluxionError::Validation`] when `dt <= 0.0`, `cm <= 0.0`, or either
///   value is non-finite (Issue #3638 — was a `panic!`).
#[allow(clippy::too_many_arguments)]
pub fn crank_nicolson_iso13790(
    tm_prev: f64,
    dt: f64,
    cm: f64,
    h_tr_3: f64,
    h_tr_em: f64,
    t_ext: f64,
    t_sup: f64,
    phi_m_tot: f64,
) -> Result<f64, FluxionError> {
    validate_dt_cm(dt, cm)?;

    let cm_dt = cm / dt;
    let half_cond = 0.5 * (h_tr_3 + h_tr_em);

    let denom = cm_dt + half_cond;
    let numer = tm_prev * (cm_dt - half_cond) + h_tr_em * t_ext + h_tr_3 * t_sup + phi_m_tot;

    // Check for negative denominator (can happen if conductances > Cm/dt)
    if denom <= 0.0 {
        // Fall back to forward Euler to avoid instability
        return Ok(tm_prev
            + dt / cm * (h_tr_em * (t_ext - tm_prev) + h_tr_3 * (t_sup - tm_prev) + phi_m_tot));
    }

    Ok(numer / denom)
}

/// Backward Euler solver for thermal mass with 2 conductances (no exterior path).
///
/// For 6R2C envelope mass: h_tr_em is NOT included in the heat balance.
/// The envelope mass receives heat from:
///   - T_s via h_tr_ms (surface-to-mass conductance)
///   - Tm_int via h_tr_me (internal-to-envelope-mass conductance)
///
/// Heat balance:
/// Cm * (Tm_new - Tm_old) / dt = h_tr_ms * (T_s - Tm_new) + h_tr_me * (Tm_int - Tm_new) + phi_m
///
/// Rearranged:
/// (Cm/dt + h_tr_ms + h_tr_me) * Tm_new = Cm/dt * Tm_old + h_tr_ms * T_s + h_tr_me * Tm_int + phi_m
///
/// # Arguments
/// * `tm_old` - Previous mass temperature (°C)
/// * `dt` - Time step (seconds)
/// * `cm` - Thermal capacitance (J/K)
/// * `h_tr_ms` - Surface-to-mass conductance (W/K)
/// * `h_tr_me` - Internal-mass-to-envelope-mass conductance (W/K)
/// * `t_surface` - Surface temperature (°C)
/// * `t_int` - Internal mass temperature (°C)
/// * `phi_m` - Direct gains to thermal mass (W)
///
/// # Returns
/// * New mass temperature (°C)
///
/// # Errors
/// * [`FluxionError::Validation`] when `dt <= 0.0`, `cm <= 0.0`, or either
///   value is non-finite (Issue #3638 — was a `panic!`).
#[allow(clippy::too_many_arguments)]
pub fn backward_euler_update_2cond(
    tm_old: f64,
    dt: f64,
    cm: f64,
    h_tr_ms: f64,
    h_tr_me: f64,
    t_surface: f64,
    t_int: f64,
    phi_m: f64,
) -> Result<f64, FluxionError> {
    validate_dt_cm(dt, cm)?;

    // Calculate denominator: (Cm/dt + h_tr_ms + h_tr_me)
    let denom = cm / dt + h_tr_ms + h_tr_me;

    // Calculate numerator: Cm/dt * Tm_old + h_tr_ms * t_surface + h_tr_me * t_int + phi_m
    let numer = cm / dt * tm_old + h_tr_ms * t_surface + h_tr_me * t_int + phi_m;

    // Return new temperature
    Ok(numer / denom)
}

/// Backward Euler solver for thermal mass with 2 conductances using H_tr_3.
///
/// For high-mass envelopes (Case 900+), the correct thermal coupling is through
/// H_tr_3 (the combined air-side conductance ≈ 40 W/K), NOT h_tr_ms + h_tr_me
/// (which gives ≈ 1450 W/K and a time constant of ~1.9 hours instead of ~69 hours).
///
/// This function replaces `backward_euler_update_2cond` for high-mass cases where
/// the envelope mass should be thermally coupled to the zone air through the slow
/// H_tr_3 path, not the fast surface-to-mass path.
///
/// Heat balance:
/// Cm * (Tm_new - Tm_old) / dt = h_tr_3 * (t_zone - Tm_new) + phi_m
///
/// Rearranged:
/// (Cm/dt + h_tr_3) * Tm_new = Cm/dt * Tm_old + h_tr_3 * t_zone + phi_m
///
/// # Arguments
/// * `tm_old` - Previous mass temperature (°C)
/// * `dt` - Time step (seconds)
/// * `cm` - Thermal capacitance (J/K)
/// * `h_tr_3` - ISO 13790 combined conductance H_tr_3 (W/K) ≈ 40 W/K for Case 900
/// * `t_zone` - Zone air temperature (°C)
/// * `phi_m` - Direct gains to thermal mass (W)
///
/// # Returns
/// * New mass temperature (°C)
///
/// # Errors
/// * [`FluxionError::Validation`] when `dt <= 0.0`, `cm <= 0.0`, or either
///   value is non-finite (Issue #3638 — was a `panic!`).
#[allow(clippy::too_many_arguments)]
pub fn backward_euler_update_2cond_h_tr3(
    tm_old: f64,
    dt: f64,
    cm: f64,
    h_tr_3: f64,
    t_zone: f64,
    phi_m: f64,
) -> Result<f64, FluxionError> {
    validate_dt_cm(dt, cm)?;

    // Calculate denominator: (Cm/dt + h_tr_3)
    let denom = cm / dt + h_tr_3;

    // Calculate numerator: Cm/dt * Tm_old + h_tr_3 * t_zone + phi_m
    let numer = cm / dt * tm_old + h_tr_3 * t_zone + phi_m;

    // Return new temperature
    Ok(numer / denom)
}

/// Crank-Nicolson solver for semi-implicit thermal mass update.
///
/// Uses average of old and new heat fluxes for 2nd-order accuracy:
/// Cm * (Tm_new - Tm_old) / dt = 0.5 * (Q_old + Q_new)
///
/// Where:
/// Q_old = h_tr_em * (t_ext - Tm_old) + h_tr_ms * (t_surface - Tm_old) + phi_m
/// Q_new = h_tr_em * (t_ext - Tm_new) + h_tr_ms * (t_surface - Tm_new) + phi_m
///
/// Rearranged to solve for Tm_new:
/// (Cm/dt + 0.5 * h_tr_em + 0.5 * h_tr_ms) * Tm_new = Cm/dt * Tm_old + 0.5 * Q_old + 0.5 * (h_tr_em * t_ext + h_tr_ms * t_surface + phi_m)
///
/// # Arguments
/// * `tm_old` - Previous mass temperature (°C)
/// * `dt` - Time step (seconds)
/// * `cm` - Thermal capacitance (J/K)
/// * `h_tr_em` - Exterior-to-mass conductance (W/K)
/// * `h_tr_ms` - Mass-to-surface conductance (W/K)
/// * `t_ext` - Exterior temperature (°C)
/// * `t_surface` - Surface temperature (°C)
/// * `phi_m` - Direct gains to thermal mass (W)
///
/// # Returns
/// * New mass temperature (°C)
///
/// # Errors
/// * [`FluxionError::Validation`] when `dt <= 0.0`, `cm <= 0.0`, or either
///   value is non-finite (Issue #3638 — was a `panic!`).
///
/// # Stability
/// Unconditionally stable (A-stable).
///
/// # Accuracy
/// 2nd-order accurate: error = O(dt^3), better than backward Euler for oscillatory systems.
///
/// # Example
/// ```
/// use fluxion::sim::thermal_integration::crank_nicolson_update;
///
/// let tm_new = crank_nicolson_update(
///     20.0,  // tm_old
///     3600.0, // dt (1 hour)
///     1000.0, // cm
///     10.0,   // h_tr_em
///     100.0,  // h_tr_ms
///     -5.0,   // t_ext
///     22.0,   // t_surface
///     500.0,  // phi_m
/// )
/// .unwrap();
/// ```
#[allow(clippy::too_many_arguments)]
pub fn crank_nicolson_update(
    tm_old: f64,
    dt: f64,
    cm: f64,
    h_tr_em: f64,
    h_tr_ms: f64,
    t_ext: f64,
    t_surface: f64,
    phi_m: f64,
) -> Result<f64, FluxionError> {
    validate_dt_cm(dt, cm)?;

    // Calculate old heat flux
    let q_old = h_tr_em * (t_ext - tm_old) + h_tr_ms * (t_surface - tm_old) + phi_m;

    // Calculate total conductance
    let a = h_tr_em + h_tr_ms;

    // Calculate constant term (independent of Tm_new)
    let b = h_tr_em * t_ext + h_tr_ms * t_surface + phi_m;

    // Calculate denominator: (Cm/dt + 0.5 * a)
    let denom = cm / dt + 0.5 * a;

    // Calculate numerator: Cm/dt * Tm_old + 0.5 * q_old + 0.5 * b
    let numer = cm / dt * tm_old + 0.5 * q_old + 0.5 * b;

    // Return new temperature
    Ok(numer / denom)
}

/// Crank-Nicolson solver for semi-implicit thermal mass update with THREE conductances.
///
/// For 6R2C model, the envelope mass receives heat from:
/// - Exterior (h_tr_em, t_ext = sol-air temperature)
/// - Surface (h_tr_ms, t_surface = surface temperature)
/// - Internal mass (h_tr_me, t_int = internal mass temperature)
///
/// Uses average of old and new heat fluxes for 2nd-order accuracy:
/// Cm * (Tm_new - Tm_old) / dt = 0.5 * (Q_old + Q_new)
///
/// Where Q_old = h_tr_em*(t_ext-Tm_old) + h_tr_ms*(t_surface-Tm_old) + h_tr_me*(t_int-Tm_old) + phi_m
///
/// # Arguments
/// * `tm_old` - Previous mass temperature (°C)
/// * `dt` - Time step (seconds)
/// * `cm` - Thermal capacitance (J/K)
/// * `h_tr_em` - Exterior-to-mass conductance (W/K)
/// * `h_tr_ms` - Mass-to-surface conductance (W/K)
/// * `h_tr_me` - Mass-to-internal-mass conductance (W/K)
/// * `t_ext` - Exterior temperature/sol-air (°C)
/// * `t_surface` - Surface temperature (°C)
/// * `t_int` - Internal mass temperature (°C)
/// * `phi_m` - Direct gains to thermal mass (W)
///
/// # Returns
/// * New mass temperature (°C)
///
/// # Errors
/// * [`FluxionError::Validation`] when `dt <= 0.0`, `cm <= 0.0`, or either
///   value is non-finite (Issue #3638 — was a `panic!`).
#[allow(clippy::too_many_arguments)]
pub fn crank_nicolson_update_3cond(
    tm_old: f64,
    dt: f64,
    cm: f64,
    h_tr_em: f64,
    h_tr_ms: f64,
    h_tr_me: f64,
    t_ext: f64,
    t_surface: f64,
    t_int: f64,
    phi_m: f64,
) -> Result<f64, FluxionError> {
    validate_dt_cm(dt, cm)?;

    // Calculate old heat flux from all three paths
    let q_old = h_tr_em * (t_ext - tm_old)
        + h_tr_ms * (t_surface - tm_old)
        + h_tr_me * (t_int - tm_old)
        + phi_m;

    // Calculate total conductance
    let a = h_tr_em + h_tr_ms + h_tr_me;

    // Calculate constant term (independent of Tm_new)
    let b = h_tr_em * t_ext + h_tr_ms * t_surface + h_tr_me * t_int + phi_m;

    // Calculate denominator: (Cm/dt + 0.5 * a)
    let denom = cm / dt + 0.5 * a;

    // Calculate numerator: Cm/dt * Tm_old + 0.5 * q_old + 0.5 * b
    let numer = cm / dt * tm_old + 0.5 * q_old + 0.5 * b;

    // Return new temperature
    Ok(numer / denom)
}

/// Explicit Euler solver for thermal mass update (forward method).
///
/// Simple forward integration:
/// Tm_new = Tm_old + dt * (Q_net / Cm)
///
/// Where Q_net = h_tr_em * (t_ext - Tm_old) + h_tr_ms * (t_surface - Tm_old) + phi_m
///
/// # Arguments
/// * `tm_old` - Previous mass temperature (°C)
/// * `dt` - Time step (seconds)
/// * `cm` - Thermal capacitance (J/K)
/// * `h_tr_em` - Exterior-to-mass conductance (W/K)
/// * `h_tr_ms` - Mass-to-surface conductance (W/K)
/// * `t_ext` - Exterior temperature (°C)
/// * `t_surface` - Surface temperature (°C)
/// * `phi_m` - Direct gains to thermal mass (W)
///
/// # Returns
/// * New mass temperature (°C)
///
/// # Errors
/// * [`FluxionError::Validation`] when `dt <= 0.0`, `cm <= 0.0`, or either
///   value is non-finite (Issue #3638 — was a `panic!`).
///
/// # Stability
/// Conditionally stable when dt < Cm / (h_tr_em + h_tr_ms).
/// For typical building parameters with Cm > 500 J/K and dt = 3600s,
/// this condition is often violated, leading to instability.
///
/// # Accuracy
/// 1st-order accurate: error = O(dt^2)
///
/// # Example
/// ```
/// use fluxion::sim::thermal_integration::explicit_euler_update;
///
/// let tm_new = explicit_euler_update(
///     20.0,  // tm_old
///     3600.0, // dt (1 hour)
///     200.0,  // cm (low thermal mass)
///     10.0,   // h_tr_em
///     100.0,  // h_tr_ms
///     -5.0,   // t_ext
///     22.0,   // t_surface
///     500.0,  // phi_m
/// )
/// .unwrap();
/// ```
#[allow(clippy::too_many_arguments)]
pub fn explicit_euler_update(
    tm_old: f64,
    dt: f64,
    cm: f64,
    h_tr_em: f64,
    h_tr_ms: f64,
    t_ext: f64,
    t_surface: f64,
    phi_m: f64,
) -> Result<f64, FluxionError> {
    validate_dt_cm(dt, cm)?;

    // Calculate net heat flux
    let q_net = h_tr_em * (t_ext - tm_old) + h_tr_ms * (t_surface - tm_old) + phi_m;

    // Update temperature
    Ok(tm_old + (q_net / cm) * dt)
}

/// Checks if explicit Euler is stable for given parameters.
///
/// Stability condition: dt < Cm / (h_tr_em + h_tr_ms)
///
/// # Arguments
/// * `dt` - Time step (seconds)
/// * `cm` - Thermal capacitance (J/K)
/// * `h_tr_em` - Exterior-to-mass conductance (W/K)
/// * `h_tr_ms` - Mass-to-surface conductance (W/K)
///
/// # Returns
/// * `true` if stable, `false` otherwise
///
/// # Example
/// ```
/// use fluxion::sim::thermal_integration::is_explicit_euler_stable;
///
/// let stable = is_explicit_euler_stable(3600.0, 1000.0, 10.0, 100.0);
/// assert!(!stable); // High mass, likely unstable
/// ```
pub fn is_explicit_euler_stable(dt: f64, cm: f64, h_tr_em: f64, h_tr_ms: f64) -> bool {
    let total_conductance = h_tr_em + h_tr_ms;
    if total_conductance <= 0.0 {
        return true; // No heat transfer, trivially stable
    }
    dt < cm / total_conductance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_integration_method_low_mass() {
        // Low thermal mass: explicit Euler is fine
        assert_eq!(
            select_integration_method(200.0),
            ThermalIntegrationMethod::ExplicitEuler
        );
    }

    #[test]
    fn test_select_integration_method_high_mass() {
        // High thermal mass: use CrankNicolson (ISO 13790 compliant)
        assert_eq!(
            select_integration_method(1000.0),
            ThermalIntegrationMethod::CrankNicolson
        );
    }

    #[test]
    fn test_select_integration_method_threshold() {
        // At threshold: explicit Euler for low mass, CrankNicolson for high mass (ISO 13790)
        assert_eq!(
            select_integration_method(500.0),
            ThermalIntegrationMethod::ExplicitEuler
        );
        assert_eq!(
            select_integration_method(501.0),
            ThermalIntegrationMethod::CrankNicolson
        );
    }

    #[test]
    fn test_backward_euler_basic() {
        // Simple heating scenario
        let tm_old = 20.0;
        let dt = 3600.0;
        let cm = 1000.0;
        let h_tr_em = 10.0;
        let h_tr_ms = 100.0;
        let t_ext = 25.0;
        let t_surface = 22.0;
        let phi_m = 500.0;

        let tm_new =
            backward_euler_update(tm_old, dt, cm, h_tr_em, h_tr_ms, t_ext, t_surface, phi_m)
                .expect("valid thermal integration inputs");

        // Temperature should increase due to heating
        assert!(tm_new > tm_old);
        // The temperature should be reasonable (bounded by physics)
        assert!(tm_new < 50.0); // Upper bound for reasonable temperature
    }

    #[test]
    fn test_backward_euler_cooling() {
        // Simple cooling scenario
        let tm_old = 25.0;
        let dt = 3600.0;
        let cm = 1000.0;
        let h_tr_em = 10.0;
        let h_tr_ms = 100.0;
        let t_ext = -5.0;
        let t_surface = 20.0;
        let phi_m = 0.0;

        let tm_new =
            backward_euler_update(tm_old, dt, cm, h_tr_em, h_tr_ms, t_ext, t_surface, phi_m)
                .expect("valid thermal integration inputs");

        // Temperature should decrease due to cooling
        assert!(tm_new < tm_old);
        // But not below exterior (thermal mass provides thermal lag)
        assert!(tm_new > t_ext);
    }

    #[test]
    fn test_crank_nicolson_accuracy() {
        // Compare with backward Euler on a simple heating scenario
        // Note: Crank-Nicolson and backward Euler can differ significantly for large time steps
        // This test just verifies both methods produce reasonable results
        let tm_old = 20.0;
        let dt = 3600.0;
        let cm = 1000.0;
        let h_tr_em = 10.0;
        let h_tr_ms = 100.0;
        let t_ext = 25.0;
        let t_surface = 22.0;
        let phi_m = 500.0;

        let tm_be =
            backward_euler_update(tm_old, dt, cm, h_tr_em, h_tr_ms, t_ext, t_surface, phi_m)
                .expect("valid thermal integration inputs");
        let tm_cn =
            crank_nicolson_update(tm_old, dt, cm, h_tr_em, h_tr_ms, t_ext, t_surface, phi_m)
                .expect("valid thermal integration inputs");

        // Both should give increasing temperatures
        assert!(tm_be > tm_old);
        assert!(tm_cn > tm_old);

        // Both should be in reasonable range
        assert!(tm_be < 50.0);
        assert!(tm_cn < 50.0);
    }

    #[test]
    fn test_explicit_euler_basic() {
        // Simple heating scenario with low thermal mass
        let tm_old = 20.0;
        let dt = 3600.0;
        let cm = 200.0; // Low thermal mass
        let h_tr_em = 5.0;
        let h_tr_ms = 50.0;
        let t_ext = 25.0;
        let t_surface = 22.0;
        let phi_m = 100.0;

        let tm_new =
            explicit_euler_update(tm_old, dt, cm, h_tr_em, h_tr_ms, t_ext, t_surface, phi_m)
                .expect("valid thermal integration inputs");

        // Temperature should increase
        assert!(tm_new > tm_old);
    }

    #[test]
    fn test_is_explicit_euler_stable() {
        let dt = 3600.0;
        let cm = 200.0;
        let h_tr_em = 5.0;
        let h_tr_ms = 50.0;

        // Check stability criterion: dt < cm / (h_tr_em + h_tr_ms)
        // For these values: 3600 < 200 / 55 = 3.63, which is FALSE
        // So explicit Euler should be UNSTABLE
        assert!(!is_explicit_euler_stable(dt, cm, h_tr_em, h_tr_ms));

        // High mass: even more unstable
        assert!(!is_explicit_euler_stable(dt, 1000.0, h_tr_em, h_tr_ms));

        // Very small time step: stable
        assert!(is_explicit_euler_stable(0.1, cm, h_tr_em, h_tr_ms));
    }

    #[test]
    fn test_backward_euler_invalid_dt_returns_typed_error() {
        // Issue #3638: dt <= 0 must return a typed validation error, not panic.
        for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = backward_euler_update(20.0, dt, 1000.0, 10.0, 100.0, -5.0, 22.0, 500.0)
                .expect_err("invalid dt must yield Err");
            assert!(
                matches!(err, FluxionError::Validation(ref msg) if msg.contains("Time step dt")),
                "dt={dt} produced wrong error: {err:?}"
            );
        }
    }

    #[test]
    fn test_backward_euler_invalid_cm_returns_typed_error() {
        // Issue #3638: cm <= 0 must return a typed validation error, not panic.
        for cm in [0.0, -1000.0, f64::NAN, f64::INFINITY] {
            let err = backward_euler_update(20.0, 3600.0, cm, 10.0, 100.0, -5.0, 22.0, 500.0)
                .expect_err("invalid cm must yield Err");
            assert!(
                matches!(err, FluxionError::Validation(ref msg) if msg.contains("Thermal capacitance cm")),
                "cm={cm} produced wrong error: {err:?}"
            );
        }
    }

    /// Issue #3638 regression: every integrator returns a typed
    /// `FluxionError::Validation` for degenerate `dt` values instead of
    /// panicking. `dt = 0.0` is the exact scenario named in the issue.
    #[test]
    fn test_issue_3638_all_integrators_reject_degenerate_dt() {
        let degenerate_dts = [0.0, -3600.0, f64::NAN, f64::INFINITY];

        for &dt in &degenerate_dts {
            let err = backward_euler_update(20.0, dt, 1000.0, 10.0, 100.0, -5.0, 22.0, 500.0)
                .expect_err("backward_euler_update must reject degenerate dt");
            assert!(matches!(err, FluxionError::Validation(_)));

            let err = crank_nicolson_iso13790(20.0, dt, 1000.0, 40.0, 10.0, -5.0, 22.0, 500.0)
                .expect_err("crank_nicolson_iso13790 must reject degenerate dt");
            assert!(matches!(err, FluxionError::Validation(_)));

            let err = backward_euler_update_2cond(20.0, dt, 1000.0, 100.0, 30.0, 22.0, 19.0, 500.0)
                .expect_err("backward_euler_update_2cond must reject degenerate dt");
            assert!(matches!(err, FluxionError::Validation(_)));

            let err = backward_euler_update_2cond_h_tr3(20.0, dt, 1000.0, 40.0, 21.0, 500.0)
                .expect_err("backward_euler_update_2cond_h_tr3 must reject degenerate dt");
            assert!(matches!(err, FluxionError::Validation(_)));

            let err = crank_nicolson_update(20.0, dt, 1000.0, 10.0, 100.0, -5.0, 22.0, 500.0)
                .expect_err("crank_nicolson_update must reject degenerate dt");
            assert!(matches!(err, FluxionError::Validation(_)));

            let err = crank_nicolson_update_3cond(
                20.0, dt, 1000.0, 10.0, 100.0, 30.0, -5.0, 22.0, 19.0, 500.0,
            )
            .expect_err("crank_nicolson_update_3cond must reject degenerate dt");
            assert!(matches!(err, FluxionError::Validation(_)));

            let err = explicit_euler_update(20.0, dt, 200.0, 5.0, 50.0, 25.0, 22.0, 100.0)
                .expect_err("explicit_euler_update must reject degenerate dt");
            assert!(matches!(err, FluxionError::Validation(_)));
        }
    }

    /// Issue #3638 regression: every integrator returns a typed
    /// `FluxionError::Validation` for degenerate `cm` values instead of
    /// panicking. `cm = 0.0` via a faulty HVAC override is the exact
    /// scenario named in the issue.
    #[test]
    fn test_issue_3638_all_integrators_reject_degenerate_cm() {
        let degenerate_cms = [0.0, -1000.0, f64::NAN, f64::INFINITY];

        for &cm in &degenerate_cms {
            let err = backward_euler_update(20.0, 3600.0, cm, 10.0, 100.0, -5.0, 22.0, 500.0)
                .expect_err("backward_euler_update must reject degenerate cm");
            assert!(matches!(err, FluxionError::Validation(_)));

            let err = crank_nicolson_iso13790(20.0, 3600.0, cm, 40.0, 10.0, -5.0, 22.0, 500.0)
                .expect_err("crank_nicolson_iso13790 must reject degenerate cm");
            assert!(matches!(err, FluxionError::Validation(_)));

            let err = backward_euler_update_2cond(20.0, 3600.0, cm, 100.0, 30.0, 22.0, 19.0, 500.0)
                .expect_err("backward_euler_update_2cond must reject degenerate cm");
            assert!(matches!(err, FluxionError::Validation(_)));

            let err = backward_euler_update_2cond_h_tr3(20.0, 3600.0, cm, 40.0, 21.0, 500.0)
                .expect_err("backward_euler_update_2cond_h_tr3 must reject degenerate cm");
            assert!(matches!(err, FluxionError::Validation(_)));

            let err = crank_nicolson_update(20.0, 3600.0, cm, 10.0, 100.0, -5.0, 22.0, 500.0)
                .expect_err("crank_nicolson_update must reject degenerate cm");
            assert!(matches!(err, FluxionError::Validation(_)));

            let err = crank_nicolson_update_3cond(
                20.0, 3600.0, cm, 10.0, 100.0, 30.0, -5.0, 22.0, 19.0, 500.0,
            )
            .expect_err("crank_nicolson_update_3cond must reject degenerate cm");
            assert!(matches!(err, FluxionError::Validation(_)));

            let err = explicit_euler_update(20.0, 3600.0, cm, 5.0, 50.0, 25.0, 22.0, 100.0)
                .expect_err("explicit_euler_update must reject degenerate cm");
            assert!(matches!(err, FluxionError::Validation(_)));
        }
    }

    /// Issue #3638 golden values: prove the `Result` conversion did not
    /// alter valid-input numerics.
    ///
    /// Provenance: expected values computed with python3 (RULES.md Rule 0)
    /// by mirroring the exact IEEE-754 operation order of each integrator
    /// on the pre-#3638 implementation, then hardcoded here. Assertions
    /// are exact (`assert_eq!`) — bit-identical — so any drift in the
    /// arithmetic fails the test.
    #[test]
    fn test_issue_3638_golden_values_valid_inputs_bit_identical() {
        // backward_euler_update: denom = 1000/3600 + 10 + 100,
        // numer = (1000/3600)*20 + 10*(-5) + 100*22 + 500
        let got =
            backward_euler_update(20.0, 3600.0, 1000.0, 10.0, 100.0, -5.0, 22.0, 500.0).unwrap();
        assert_eq!(got, 24.08060453400504);

        // crank_nicolson_iso13790: cm_dt = 1000/3600, half_cond = 0.5*(40+10),
        // numer = 20*(cm_dt - half_cond) + 10*(-5) + 40*22 + 500
        let got =
            crank_nicolson_iso13790(20.0, 3600.0, 1000.0, 40.0, 10.0, -5.0, 22.0, 500.0).unwrap();
        assert_eq!(got, 33.05494505494505);

        // backward_euler_update_2cond: denom = 1000/3600 + 100 + 30,
        // numer = (1000/3600)*20 + 100*22 + 30*19 + 500
        let got = backward_euler_update_2cond(20.0, 3600.0, 1000.0, 100.0, 30.0, 22.0, 19.0, 500.0)
            .unwrap();
        assert_eq!(got, 25.142857142857146);

        // backward_euler_update_2cond_h_tr3: denom = 1000/3600 + 40,
        // numer = (1000/3600)*20 + 40*21 + 500
        let got =
            backward_euler_update_2cond_h_tr3(20.0, 3600.0, 1000.0, 40.0, 21.0, 500.0).unwrap();
        assert_eq!(got, 33.40689655172414);

        // crank_nicolson_update: q_old = 10*(-25) + 100*2 + 500, a = 110,
        // b = -50 + 2200 + 500, numer = (1000/3600)*20 + 0.5*q_old + 0.5*b
        let got =
            crank_nicolson_update(20.0, 3600.0, 1000.0, 10.0, 100.0, -5.0, 22.0, 500.0).unwrap();
        assert_eq!(got, 28.14070351758794);

        // crank_nicolson_update_3cond: q_old = 10*(-25) + 100*2 + 30*(-1) + 500,
        // a = 140, b = -50 + 2200 + 570 + 500
        let got = crank_nicolson_update_3cond(
            20.0, 3600.0, 1000.0, 10.0, 100.0, 30.0, -5.0, 22.0, 19.0, 500.0,
        )
        .unwrap();
        assert_eq!(got, 25.97628458498024);

        // explicit_euler_update: q_net = 5*5 + 50*2 + 100, tm = 20 + (225/200)*3600
        let got = explicit_euler_update(20.0, 3600.0, 200.0, 5.0, 50.0, 25.0, 22.0, 100.0).unwrap();
        assert_eq!(got, 4070.0);
    }

    #[test]
    fn test_energy_balance_conservation() {
        // Test that integration methods preserve energy balance
        // over a full day with sinusoidal forcing

        let dt = 3600.0; // 1 hour
        let cm = 1000.0;
        let h_tr_em = 10.0;
        let h_tr_ms = 100.0;
        let t_surface = 20.0;
        let phi_m = 0.0;

        // Simulate 24 hours with sinusoidal exterior temperature
        let mut tm = 20.0;

        for hour in 0..24 {
            let t_ext = 20.0 + 10.0 * ((hour as f64 / 24.0) * 2.0 * PI).sin();

            let tm_old = tm;
            tm = backward_euler_update(tm_old, dt, cm, h_tr_em, h_tr_ms, t_ext, t_surface, phi_m)
                .expect("valid thermal integration inputs");
        }

        // Over a full sinusoidal cycle, the net energy should be close to zero
        // (thermal mass returns to near initial temperature)
        // Allow for numerical error
        let final_tm = tm;
        let initial_tm = 20.0;
        let tm_change = (final_tm - initial_tm).abs();

        // Temperature change should be small (< 1°C over full cycle)
        assert!(
            tm_change < 1.0,
            "Final temp: {}, Initial: {}, Change: {}",
            final_tm,
            initial_tm,
            tm_change
        );
    }
}
