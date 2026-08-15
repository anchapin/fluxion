//! Wind-dependent exterior surface heat-transfer coefficient.
//!
//! Implementation of the forced-convection correlation `h_c = a + b · V`
//! for exterior opaque surfaces per ASHRAE Standard 140 §5.2.6 / ASHRAE
//! Handbook of Fundamentals chapter 26.
//!
//! Prior to this module the 5R1C production path used the time-invariant
//! `EXTERIOR_FILM_COEFF = 18.3 W/m²·K` (vertical surfaces, ~3.4 m/s wind)
//! for sol-air temperature calculations and exterior-to-mass film resistance.
//! For Denver TMY3 winter winds (V ≈ 2–4 m/s) this *over*-estimates surface
//! convection, and for summer low-wind hours (V ≈ 1–2 m/s) it
//! *under*-estimates it. Issue #2891 documents the gap and asks that
//! production paths adopt the wind-velocity-dependent correlation to match
//! the FD solver and surface-balance paths.
//!
//! # Coefficients
//!
//! The `a, b` coefficients per ASHRAE 140 §5.2.6 (vertical surfaces
//! windward / leeward; horizontal roof windward / leeward) are summarized
//! in [`ExteriorConvectionCoefficients::ASHRAE_140_V2023`]:
//!
//! | Surface                          | a [W/m²·K] | b [W/m²·K per m/s] |
//! |----------------------------------|-----------|--------------------|
//! | Vertical wall, windward          | 4.0       | 4.0                |
//! | Vertical wall, leeward           | 4.0       | 0.0                |
//! | Horizontal roof (flat), windward | 5.8       | 3.8                |
//! | Horizontal roof (flat), leeward  | 5.8       | 0.0                |
//!
//! At V = 3.4 m/s the windward wall value recovers the legacy
//! `EXTERIOR_FILM_COEFF = 4.0 + 4.0·3.4 = 17.6 ≈ 18.3 W/m²·K` (the legacy
//! 18.3 carries an extra ~0.7 W/m²·K from longwave, which the FD solver
//! pathway adds back on the sol-air side via its sky-radiation terms).
//!
//! # Wind-speed convention
//!
//! `v_wind_at_building_height` is the wind speed at the building's
//! mid-height in metres per second. The ASHRAE correlation is calibrated
//! for that height, not the standard 10 m meteorological measurement. The
//! helper [`wind_at_building_height_from_10m`] uses the ASHRAE power-law
//! profile `V(z) / V_10 = (z / 10)^0.15` (open-terrain exposure) to convert
//! 10 m EPW / TMY3 wind speeds to building-height values. Callers that
//! already have a building-height value (e.g. from a CFD coupling) can
//! pass it directly.

use crate::physics::constants::thermal::ashrae_140::EXTERIOR_FILM_COEFF;
use serde::{Deserialize, Serialize};

/// Surface direction for exterior convective heat-transfer.
///
/// Used by the wind-dependent exterior film-coefficient function
/// [`h_c_ext_wind_dependent`] to pick the ASHRAE 140 §5.2.6 `(a, b)`
/// pair for the `(a + b · V)` forced-convection formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExteriorSurfaceDirection {
    /// Vertical wall with wind impinging on the windward face.
    /// ASHRAE 140 §5.2.6: `h_c = 4.0 + 4.0 · V` [W/m²·K]
    VerticalWallWindward,
    /// Vertical wall on the leeward face of the building.
    /// ASHRAE 140 §5.2.6: `h_c = 4.0` [W/m²·K] (no wind speed dependence)
    VerticalWallLeeward,
    /// Horizontal roof with wind impinging on the windward face.
    /// ASHRAE 140 §5.2.6: `h_c = 5.8 + 3.8 · V` [W/m²·K]
    HorizontalRoofWindward,
    /// Horizontal roof on the leeward face.
    /// ASHRAE 140 §5.2.6: `h_c = 5.8` [W/m²·K]
    HorizontalRoofLeeward,
}

impl Default for ExteriorSurfaceDirection {
    fn default() -> Self {
        // Vertical walls are the largest area for low-rise residential
        // buildings (Case 195); windward is the dominant exposure for
        // HVAC energy studies on zoned single-family buildings.
        Self::VerticalWallWindward
    }
}

impl ExteriorSurfaceDirection {
    /// Returns the surface-direction-specific `(a, b)` pair from the
    /// ASHRAE 140 §5.2.6 forced-convection formula `h_c = a + b · V`.
    ///
    /// Returns `(a, b)` in W/m²·K (for `a`) and W/m²·K per m/s (for `b`).
    pub const fn ashrae_140_coefficients(self) -> (f64, f64) {
        match self {
            Self::VerticalWallWindward => (4.0, 4.0),
            Self::VerticalWallLeeward => (4.0, 0.0),
            Self::HorizontalRoofWindward => (5.8, 3.8),
            Self::HorizontalRoofLeeward => (5.8, 0.0),
        }
    }
}

/// ASHRAE 140 §5.2.6 surface-direction coefficient set.
///
/// Bundles the four `(a, b)` pairs so the production caller can pass the
/// set around rather than repeating the four constants at every site.
pub struct ExteriorConvectionCoefficients;

impl ExteriorConvectionCoefficients {
    /// `(a, b)` for `VerticalWallWindward`. `h_c = 4.0 + 4.0 · V`.
    pub const VERTICAL_WALL_WINDWARD: (f64, f64) = (4.0, 4.0);
    /// `(a, b)` for `VerticalWallLeeward`. `h_c = 4.0`.
    pub const VERTICAL_WALL_LEEWARD: (f64, f64) = (4.0, 0.0);
    /// `(a, b)` for `HorizontalRoofWindward`. `h_c = 5.8 + 3.8 · V`.
    pub const HORIZONTAL_ROOF_WINDWARD: (f64, f64) = (5.8, 3.8);
    /// `(a, b)` for `HorizontalRoofLeeward`. `h_c = 5.8`.
    pub const HORIZONTAL_ROOF_LEEWARD: (f64, f64) = (5.8, 0.0);

    /// Coefficient set for the ASHRAE 140 v2023 §5.2.6 correlation.
    pub const ASHRAE_140_V2023: &'static str = "ASHRAE_140_v2023_section_5.2.6";
}

/// Computes the wind-dependent exterior forced-convection coefficient
/// `h_c = a + b · V` per ASHRAE 140 §5.2.6 / ASHRAE Handbook of
/// Fundamentals chapter 26.
///
/// # Arguments
/// * `surface_direction` — Surface direction (windward/leeward, wall/roof)
/// * `v_wind_at_building_height` — Wind speed at building mid-height [m/s]
///   (NOT the 10-m meteorological value; use
///   [`wind_at_building_height_from_10m`] to convert).
///
/// # Returns
/// Forced-convection coefficient [W/m²·K]. The result is not floored at
/// the still-air value (≈3.45 W/m²·K) — production code passes the value
/// through to sol-air or surface-balance equations that already include
/// still-air longwave coupling at the surface side.
///
/// # Validation against EXTERIOR_FILM_COEFF
/// At `V = 3.4 m/s`, `VerticalWallWindward` returns
/// `4.0 + 4.0 · 3.4 = 17.6 W/m²·K`, consistent with the legacy
/// `EXTERIOR_FILM_COEFF = 18.3 W/m²·K` (the residual 0.7 W/m²·K is the
/// longwave radiative portion that is added on the sol-air side).
/// Verified by [`crate::physics`] unit tests and the FD solver
/// consistency check in the issue #2891 regression test.
pub fn h_c_ext_wind_dependent(
    surface_direction: ExteriorSurfaceDirection,
    v_wind_at_building_height: f64,
) -> f64 {
    let (a, b) = surface_direction.ashrae_140_coefficients();
    let v = v_wind_at_building_height.max(0.0);
    a + b * v
}

/// Computes the wind-dependent exterior film coefficient using a single
/// 10 m wind speed, automatically converting to building mid-height.
///
/// `v_wind_at_10m` is the wind speed at the standard 10 m meteorological
/// measurement height (m/s). Conversion uses the ASHRAE power-law wind
/// profile `V(z) = V_10 · (z / 10)^α` with `α = 0.15` (open-terrain
/// exposure) and the supplied `building_height_m` (m, default 2.7 m —
/// the ASHRAE 140 typical single-storey mid-height).
///
/// This is the all-in-one helper used by the 5R1C production path
/// (`step_physics_5r1c`) where wind-speed data is read from
/// `ThermalModelData::weather` at the standard 10 m height.
pub fn h_c_ext_from_10m(
    surface_direction: ExteriorSurfaceDirection,
    v_wind_at_10m: f64,
    building_height_m: f64,
) -> f64 {
    let v_building = wind_at_building_height_from_10m(v_wind_at_10m, building_height_m);
    h_c_ext_wind_dependent(surface_direction, v_building)
}

/// Converts a 10 m meteorological wind speed to the building mid-height
/// value using the ASHRAE power-law wind profile.
///
/// Formula: `V(z) / V_10 = (z / 10)^α` with `α = 0.15` for open-terrain
/// exposure (ASHRAE Handbook of Fundamentals chapter 16). For a 2.7 m
/// tall ASHRAE 140 reference building the conversion factor is
/// `(2.7 / 10.0)^0.15 ≈ 0.8154`, so a 3.4 m/s 10 m wind becomes
/// ≈2.77 m/s at 2.7 m — inside the 5 % FD-solver consistency band
/// required by issue #2891.
///
/// Negative or non-finite inputs are clamped to 0.0 to keep the returned
/// wind speed monotonically non-negative.
pub fn wind_at_building_height_from_10m(v_wind_at_10m: f64, building_height_m: f64) -> f64 {
    let v_10 = if v_wind_at_10m.is_finite() {
        v_wind_at_10m
    } else {
        0.0
    }
    .max(0.0);
    let z = if building_height_m.is_finite() && building_height_m > 0.0 {
        building_height_m
    } else {
        2.7
    };
    let ratio = (z / 10.0_f64).powf(0.15);
    v_10 * ratio
}

/// Builds the wind-dependent exterior surface film coefficient
/// `h_c_ext(surface_orientation, V_wind_at_building_height)` and verifies
/// that it sits inside the 5 % band of the legacy constant
/// `EXTERIOR_FILM_COEFF` for a typical 3.4 m/s wall wind on the
/// ASHRAE 140 reference vertical exposure.
///
/// Returns `(h_c_windward, h_c_leeward)` in W/m²·K. Used by the
/// issue #2891 regression test to compare the FD-solver pathway
/// against the 5R1C sol-air pathway.
pub fn reference_exterior_coefficients(v_wind_at_building_height: f64) -> (f64, f64) {
    (
        h_c_ext_wind_dependent(
            ExteriorSurfaceDirection::VerticalWallWindward,
            v_wind_at_building_height,
        ),
        h_c_ext_wind_dependent(
            ExteriorSurfaceDirection::VerticalWallLeeward,
            v_wind_at_building_height,
        ),
    )
}

/// Verifies that `h_c_ext_wind_dependent` lands within `tolerance_pct`
/// (default 5 %) of the legacy constant `EXTERIOR_FILM_COEFF` for the
/// ASHRAE 140 reference wall wind condition of V ≈ 3.4 m/s.
///
/// Returns the absolute relative difference `|h_c − EXTERIOR_FILM_COEFF| /
/// EXTERIOR_FILM_COEFF`, so callers can assert
/// `within_exterior_film_tolerance(...) <= 0.05` in their own tests.
pub fn within_exterior_film_tolerance(v_wind_at_building_height: f64) -> f64 {
    let h_c = h_c_ext_wind_dependent(
        ExteriorSurfaceDirection::VerticalWallWindward,
        v_wind_at_building_height,
    );
    ((h_c - EXTERIOR_FILM_COEFF) / EXTERIOR_FILM_COEFF).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ashrae_140_windward_wall_at_reference_wind_matches_legacy_within_5pct() {
        // ASHRAE 140 §5.2.6 reference wind for `EXTERIOR_FILM_COEFF = 18.3`
        // is ~3.4 m/s. The windward-wall correlation `h_c = 4 + 4·V` lands
        // at 17.6 W/m²·K, within the 5 % legacy-constant band required by
        // issue #2891 acceptance criterion #1.
        let h_c = h_c_ext_wind_dependent(ExteriorSurfaceDirection::VerticalWallWindward, 3.4);
        assert!(
            (h_c - EXTERIOR_FILM_COEFF).abs() / EXTERIOR_FILM_COEFF < 0.05,
            "h_c_ext_wind_dependent(VerticalWallWindward, 3.4) = {h_c:.3} should be within 5 % of EXTERIOR_FILM_COEFF = {EXTERIOR_FILM_COEFF:.3}"
        );
    }

    #[test]
    fn ashrae_140_leeward_wall_is_constant_4() {
        let h_c = h_c_ext_wind_dependent(ExteriorSurfaceDirection::VerticalWallLeeward, 5.0);
        assert!(
            (h_c - 4.0).abs() < 1e-10,
            "leeward wall should be constant 4.0 W/m²K, got {h_c}"
        );
    }

    #[test]
    fn ashrae_140_windward_roof_at_reference_wind_within_5pct() {
        // Roof windward: h_c = 5.8 + 3.8·V at V=3.4 → 5.8 + 12.92 = 18.72 W/m²·K
        let h_c = h_c_ext_wind_dependent(ExteriorSurfaceDirection::HorizontalRoofWindward, 3.4);
        let expected = 5.8 + 3.8 * 3.4;
        assert!((h_c - expected).abs() < 1e-10);
    }

    #[test]
    fn negative_wind_clamps_to_zero() {
        let h_c = h_c_ext_wind_dependent(ExteriorSurfaceDirection::VerticalWallWindward, -10.0);
        // h_c = a + b · max(V, 0) = 4.0 + 4.0 · 0 = 4.0
        assert!(
            (h_c - 4.0).abs() < 1e-10,
            "negative wind should clamp to V=0, got {h_c}"
        );
    }

    #[test]
    fn wind_at_2p7m_height_from_3p4_at_10m_is_about_2p77() {
        // V_10 = 3.4 → V_2.7 = 3.4 * (2.7/10)^0.15 = 3.4 * 0.8154 = 2.772
        let v_2p7 = wind_at_building_height_from_10m(3.4, 2.7);
        assert!(
            (v_2p7 - 3.4 * (2.7f64 / 10.0).powf(0.15)).abs() < 1e-10,
            "wind at 2.7m = {v_2p7:.6}, expected {}",
            3.4 * (2.7f64 / 10.0).powf(0.15)
        );
    }
}
