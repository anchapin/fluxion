//! Free helpers for the multi-node thermal solver (extracted from
//! `mod.rs` to keep the parent file under the Issue #3457 module-size
//! ratchet ceiling — Issue #3790, decomposition stage 2).
//!
//! Coherent module boundary: pure functions and one data-carrying
//! struct (`SurfaceExteriorTemperatures`) that do not depend on
//! `MultiNodeSolver` internals. They are reused by the solver's
//! backward-Euler step and the air-node balance.
//!
//! ## Contents
//! - `h_series` / `h_series_strict` — series conductance combination
//!   (Issue #1281)
//! - `air_sky_conductance` — linearized sky-radiative conductance
//!   (Issue #1858)
//! - `internal_node_envelope_temperature` — conductance-weighted envelope
//!   average for the internal-node coupling (Issue #1859)
//! - `per_surface_t_s` — per-surface surface temperature for the
//!   parallel-resistance 9R4C coupling (Issue #1281)
//! - `SurfaceExteriorTemperatures` — per-surface exterior boundary
//!   temperatures (Issue #863)

use fluxion_core::physics_constants::STEFAN_BOLTZMANN;

/// Series combination of two conductances (Issue #1281, parallel-resistance
/// coupling network for 9R4C).
///
/// `h_series(a, b) = (a × b) / (a + b)` is the conductance of `a` and `b` placed
/// in series. It is symmetric, strictly positive when both inputs are positive,
/// and bounded above by `min(a, b)`.
///
/// In the parallel-resistance formulation, each per-surface mass-to-air path is
/// the series pair `(h_tr_ms_k, h_tr_is)`, so `h_path_k = h_series(h_tr_ms_k, h_tr_is)`.
///
/// Returns 0.0 for degenerate inputs (a≤0 or b≤0); caller is expected to
/// validate inputs upstream. A `debug_assert!` fires in debug builds to
/// catch configuration errors early.
#[inline]
pub fn h_series(a: f64, b: f64) -> f64 {
    debug_assert!(
        a > 0.0 && b > 0.0,
        "h_series called with degenerate inputs: a={}, b={}",
        a,
        b
    );
    if a <= 0.0 || b <= 0.0 {
        return 0.0;
    }
    (a * b) / (a + b)
}

/// Strict version of `h_series` that returns `Err` instead of emitting a
/// `debug_assert!` for degenerate inputs. This enables release-mode testing
/// of the error path.
#[inline]
pub fn h_series_strict(a: f64, b: f64) -> Result<f64, &'static str> {
    if a <= 0.0 || b <= 0.0 {
        return Err("h_series called with degenerate inputs");
    }
    Ok((a * b) / (a + b))
}

/// Compute the linearized sky-radiative conductance [W/K] for the 9R4C air node
/// (Issue #1858 — closes the ~0.6 °C high-mass free-float night-min residual).
///
/// Issue #2872 — applies a per-surface sky view factor `f_sky` to the mass
/// node boundary so the longwave sky-radiation exchange is distributed
/// across the envelope instead of being concentrated on the roof. The
/// canonical values are `f_sky_wall = 0.5` (vertical wall, half sky dome),
/// `f_sky_roof = 1.0` (horizontal roof, full sky dome), and `f_sky_floor
/// = 0.0` (slab-on-grade, no sky view).
///
/// The 9R4C air-node energy balance previously had only four terms
/// (`h_tr_is · T_s`, `(h_ve + h_ve_night) · T_out`, `φ_ia`), which algebraically
/// bounds the free-floating air temperature below by `min(T_surface, T_out)`.
/// Under clear-sky radiative cooling the air temperature should be able to drop
/// *below* the outdoor dry-bulb; this conductance adds that path.
///
/// The exchange is modeled as a linearized longwave conductance between the air
/// node and the effective sky temperature:
///
/// ```text
/// h_rad_sky = ε · F_sky · 4 · σ · T_mean³ · A_aperture     [W/K]
/// ```
///
/// where `T_mean = (T_air + T_sky) / 2` [K] and `σ` is the Stefan–Boltzmann
/// constant. This is the same linearization used by
/// `SkyRadiationExchange::radiative_coefficient`, scaled by the radiative
/// aperture area to yield a total conductance compatible with the per-zone
/// `h_tr_is` / `h_ve` terms.
///
/// All inputs are physics-derived — emissivity, sky-view factor (from surface
/// tilt via `SkyRadiationExchange::tilted_surface`), aperture area (building
/// geometry), temperatures (EPW `sky_temperature()` + current air estimate) —
/// so no case-specific tuning constant is introduced (RULES.md).
///
/// Returns 0.0 for degenerate inputs (non-positive emissivity / view factor /
/// aperture), which makes the sky path a no-op for callers that do not supply
/// sky data — preserving backward compatibility.
#[inline]
pub fn air_sky_conductance(
    emissivity: f64,
    sky_view_factor: f64,
    aperture_area: f64,
    t_air_c: f64,
    t_sky_c: f64,
) -> f64 {
    if aperture_area <= 0.0 || emissivity <= 0.0 || sky_view_factor <= 0.0 {
        return 0.0;
    }
    let t_air_k = t_air_c + 273.15;
    let t_sky_k = t_sky_c + 273.15;
    let t_mean = (t_air_k + t_sky_k) / 2.0;
    if !t_mean.is_finite() || t_mean <= 0.0 {
        return 0.0;
    }
    4.0 * emissivity * sky_view_factor * STEFAN_BOLTZMANN * t_mean.powi(3) * aperture_area
}

/// Compute the conductance-weighted envelope temperature for internal node coupling
/// (Issue #1859).
///
/// ISO 13790 §C.3 specifies that the internal mass couples to the envelope
/// surfaces through the series combination of (surface-to-mass, mass-to-internal):
/// `h_me_k = h_series(h_tr_ms_k, h_tr_me)` per surface k.
///
/// The effective envelope temperature driving heat flow into the internal node is
/// the h_me-weighted average of the three envelope mass temperatures:
/// `t_env_avg = Σ(h_me_k × T_m_k) / Σ(h_me_k)` for k ∈ {wall, roof, floor}.
///
/// This replaces the unweighted arithmetic mean `(T_wall + T_roof + T_floor) / 3.0`
/// which over-weights whichever envelope happens to be hotter, suppressing the
/// internal node's damping effect on diurnal air temperature swing.
///
/// Degenerate cases (h_tr_me <= 0 or all h_series ~ 0) fall back to the simple
/// arithmetic mean.
#[inline]
pub(super) fn internal_node_envelope_temperature(
    t_wall: f64,
    t_roof: f64,
    t_floor: f64,
    h_ms_wall: f64,
    h_ms_roof: f64,
    h_ms_floor: f64,
    h_tr_me: f64,
) -> f64 {
    if h_tr_me <= 0.0 {
        return (t_wall + t_roof + t_floor) / 3.0;
    }
    let h_me_w = h_series(h_ms_wall, h_tr_me);
    let h_me_r = h_series(h_ms_roof, h_tr_me);
    let h_me_f = h_series(h_ms_floor, h_tr_me);
    let h_me_sum = h_me_w + h_me_r + h_me_f;
    if h_me_sum > 1e-6 {
        (h_me_w * t_wall + h_me_r * t_roof + h_me_f * t_floor) / h_me_sum
    } else {
        (t_wall + t_roof + t_floor) / 3.0
    }
}

/// Per-surface surface temperature for the parallel-resistance 9R4C coupling
/// (Issue #1281).
///
/// Steady-state solution of the (mass → T_s → air) series pair, given the
/// current mass temperature `t_m`, the surface-to-air conductance `h_is`,
/// and the air temperature `t_air`:
/// ...
/// T_s = (h_tr_ms × t_m + h_tr_is × t_air) / (h_tr_ms + h_tr_is)
/// ```
///
/// Equivalent to `t_air + (h_tr_ms / (h_tr_ms + h_tr_is)) × (t_m − t_air)`.
/// Degenerate cases (`h_tr_ms + h_tr_is` near zero, or non-finite inputs)
/// fall back to the air temperature.
#[inline]
pub(super) fn per_surface_t_s(t_m: f64, h_tr_ms: f64, h_tr_is: f64, t_air: f64) -> f64 {
    let denom = h_tr_ms + h_tr_is;
    if !denom.is_finite() || denom < 1e-10 {
        return t_air;
    }
    if !t_m.is_finite() || !t_air.is_finite() {
        return t_air;
    }
    (h_tr_ms * t_m + h_tr_is * t_air) / denom
}

/// Per-surface exterior boundary temperatures for the multi-node solver (Issue #863).
///
/// Each envelope node (wall, roof, floor) can have its own exterior boundary
/// temperature, computed from sol-air temperature calculations.
///
/// - Wall/Roof: sol-air temperature (accounts for solar irradiance, longwave radiation)
/// - Floor: ground temperature (ground-coupled)
#[derive(Debug, Clone)]
pub struct SurfaceExteriorTemperatures {
    /// Sol-air temperature for the wall exterior boundary (°C)
    pub t_ext_wall: f64,
    /// Sol-air temperature for the roof exterior boundary (°C)
    pub t_ext_roof: f64,
    /// Ground temperature for the floor exterior boundary (°C)
    pub t_ext_floor: f64,
}

impl SurfaceExteriorTemperatures {
    /// Create with a uniform exterior temperature (legacy fallback).
    pub fn uniform(t: f64) -> Self {
        Self {
            t_ext_wall: t,
            t_ext_roof: t,
            t_ext_floor: t,
        }
    }
}
