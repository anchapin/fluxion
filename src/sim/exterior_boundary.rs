//! Per-surface exterior boundary for the 5R1C production path (Issue #4166).
//!
//! The 5R1C production step used to build **one** sol-air temperature per zone
//! and apply it to the whole envelope via `h_tr_em * (t_sol_air - T_mass)`.
//! That lumped value baked in three physical errors:
//!
//! 1. **Sky view factor.** `SolAirTemperature::for_roof` applies the horizontal
//!    (`F_sky = 1.0`) sky view factor. Walls see roughly half the sky dome
//!    (`F_sky = (1 + cos(tilt)) / 2 ≈ 0.5` for `tilt = 90°`), so the lumped
//!    value charged walls roughly twice their correct longwave radiative loss.
//! 2. **Wind coefficients.** The lumped path hard-selected
//!    `ExteriorSurfaceDirection::HorizontalRoofWindward` (`a = 5.8, b = 3.8`)
//!    for every surface. Walls were evaluated with the roof's coefficients and
//!    no surface was ever evaluated as leeward.
//! 3. **Collapsed incidence.** The area-averaged `opaque_solar` collapsed
//!    south/north/east/west incidence into one number.
//!
//! This module provides the per-surface exterior boundary vector: per surface,
//! its tilt, azimuth, sky view factor, wind-dependent exterior film coefficient
//! (from the per-surface `(a, b)` pair selected by actual wind direction), and
//! its own absorbed irradiance. The per-surface values aggregate to the
//! zone-level `(h_tr_em, t_sol_air)` pair with correct area weighting, so the
//! downstream `h_tr_em * (t_sol_air - T_mass)` form is preserved exactly:
//!
//! ```text
//! Q = Σ_s h_tr_em,s * (T_sol_air,s − T_mass)
//!   = h_tr_em * (T_sol_air,eff − T_mass)
//! where h_tr_em = Σ_s h_tr_em,s
//!   and T_sol_air,eff = Σ_s (h_tr_em,s * T_sol_air,s) / h_tr_em
//! ```
//!
//! The sol-air formula mirrors `SolAirTemperature::for_wall_with_f_sky` /
//! `for_roof` (`crate::sim::sky_radiation`) exactly; it is re-implemented here
//! (rather than imported) so this module adds no `use crate::physics::` edge
//! — the `physics ↔ sim` cycle guard (`scripts/check_physics_sim_cycle.py`)
//! counts those, and the caller (`step_5r1c.rs`) already owns the
//! `exterior_convection` import.
//!
//! # References
//! - ASHRAE 140 §5.2.6 (wind-dependent exterior film coefficients)
//! - ASHRAE 140-2023 Annex B1 (solar absorptance: 0.7 roof, 0.6 walls)
//! - ISO 13790 §12.3.2 (`F_sky = (1 + cos(tilt)) / 2`)

use crate::sim::sky_radiation::STEFAN_BOLTZMANN;

/// Solar absorptance for horizontal roof surfaces (ASHRAE 140-2023 Annex B1-3).
/// Matches the `alpha_roof` convention in `calculate_zone_solar_gain`.
pub const ABSORPTANCE_ROOF: f64 = 0.7;

/// Solar absorptance for vertical wall surfaces (ASHRAE 140-2023 Annex B1-2).
/// Matches the `alpha_wall` convention in `calculate_zone_solar_gain`.
pub const ABSORPTANCE_WALL: f64 = 0.6;

/// Per-surface exterior boundary condition (Issue #4166).
///
/// All fields are plain values — the caller maps its surface representation
/// (orientation → tilt/azimuth) and computes `h_c_ext` from the per-surface
/// `(a, b)` wind-coefficient pair before constructing this struct.
#[derive(Debug, Clone)]
pub struct ExteriorBoundarySurface {
    /// Opaque area of the surface (m², net of window area).
    pub area_opaque_m2: f64,
    /// Surface azimuth (degrees, meteorological: 0°=North, 90°=East).
    /// -1.0 for horizontal surfaces (roof/floor).
    pub azimuth_deg: f64,
    /// Surface tilt from horizontal (degrees): 0° = roof, 90° = wall.
    pub tilt_deg: f64,
    /// Sky view factor: `F_sky = (1 + cos(tilt)) / 2` (1.0 roof, 0.5 wall).
    pub f_sky: f64,
    /// Exterior convective film coefficient (W/m²·K) from the surface's own
    /// `(a, b)` pair: `h_c = a + b · V_building`.
    pub h_c_ext: f64,
    /// Windward/leeward convection exposure selected using the actual wind
    /// direction (Issue #4166). `None` for horizontal surfaces (azimuth < 0)
    /// or when the weather source provides no wind direction (windward
    /// fallback).
    pub windward: Option<bool>,
    /// The `(a, b)` convection coefficient pair used for `h_c_ext`
    /// (e.g., vertical wall windward `(4.0, 4.0)`, leeward `(4.0, 0.0)`).
    pub convection_ab: (f64, f64),
    /// Incident solar irradiance, beam + diffuse (W/m², ground excluded).
    pub irradiance_beam_diffuse_wm2: f64,
    /// Ground-reflected solar irradiance (W/m²).
    pub irradiance_ground_wm2: f64,
    /// Solar absorptance (0.7 roof / 0.6 walls per ASHRAE 140 Annex B1).
    pub absorptance: f64,
    /// Exterior longwave emissivity (per-zone, from the construction spec).
    pub emissivity: f64,
    /// Sum of opaque-layer R-values for this surface class (m²·K/W, no films).
    pub r_materials: f64,
}

impl ExteriorBoundarySurface {
    /// Sky view factor from surface tilt (degrees from horizontal).
    ///
    /// `F_sky = (1 + cos(tilt)) / 2` — the same convention as
    /// `SkyRadiationExchange::tilted_surface` (`sky_radiation.rs`) and
    /// ISO 13790 §12.3.2. Tilt 0° (roof) → 1.0; tilt 90° (wall) → 0.5.
    pub fn sky_view_factor_from_tilt(tilt_deg: f64) -> f64 {
        let tilt_rad = tilt_deg.to_radians();
        ((1.0 + tilt_rad.cos()) / 2.0).clamp(0.0, 1.0)
    }

    /// Sol-air temperature for this surface (°C).
    ///
    /// Mirrors `SolAirTemperature::for_wall_with_f_sky` with `f_sky = 1.0`
    /// reducing to the `for_roof` formula:
    /// ```text
    /// T_sol = T_out + α·I/h_c − F_sky·ε·ΔR/h_c
    /// ΔR = σ·(T_sky,K⁴ − T_out,K⁴)
    /// ```
    /// where `I` is the total incident irradiance (beam + diffuse + ground).
    /// The `(1 − F_sky)` fraction radiates to the ground, approximated as
    /// ambient outdoor air (no net exchange) — the same convention as
    /// `for_wall_with_f_sky`.
    pub fn sol_air_temperature(&self, outdoor_temp_c: f64, sky_temp_c: f64) -> f64 {
        let h_c = self.h_c_ext.max(1.0);
        let alpha = self.absorptance.clamp(0.0, 1.0);
        let eps = self.emissivity.clamp(0.0, 1.0);
        let total_irradiance = self.irradiance_beam_diffuse_wm2 + self.irradiance_ground_wm2;
        let solar_term = alpha * total_irradiance / h_c;

        let t_out_k = outdoor_temp_c + 273.15;
        let t_sky_k = sky_temp_c + 273.15;
        let delta_r = STEFAN_BOLTZMANN * (t_sky_k.powi(4) - t_out_k.powi(4));
        let longwave_term = eps * delta_r / h_c;

        let f_sky = self.f_sky.clamp(0.0, 1.0);
        outdoor_temp_c + solar_term - f_sky * longwave_term
    }

    /// Per-surface `h_tr_em` contribution (W/K): `A / (R_materials + 1/h_c)`.
    ///
    /// Mirrors `h_tr_em_wind_dependent` (`step_5r1c.rs`, Issue #3063) with the
    /// surface's own `h_c_ext` instead of the lumped roof-windward value.
    pub fn h_tr_em(&self) -> f64 {
        let h_c = self.h_c_ext.max(1.0);
        self.area_opaque_m2 / (self.r_materials + 1.0 / h_c)
    }
}

/// Returns `true` when a vertical surface is windward of the wind.
///
/// * `wind_from_deg` — meteorological wind direction: the direction FROM which
///   the wind blows, degrees clockwise from north (EPW convention).
/// * `surface_azimuth_deg` — outward-normal azimuth of the surface, degrees
///   clockwise from north (0 = North, 90 = East).
///
/// The wind velocity vector points toward `wind_from + 180°`. The surface is
/// windward when that vector has a negative component along the outward
/// normal (the wind strikes the face), i.e. `|wrap(wind_to − azimuth)| > 90°`.
pub fn is_windward(wind_from_deg: f64, surface_azimuth_deg: f64) -> bool {
    let wind_to = (wind_from_deg + 180.0) % 360.0;
    let mut diff = (wind_to - surface_azimuth_deg) % 360.0;
    if diff < 0.0 {
        diff += 360.0;
    }
    if diff > 180.0 {
        diff = 360.0 - diff;
    }
    diff > 90.0
}

/// Zone-level exterior boundary aggregated from the per-surface vector.
///
/// Returns `(h_tr_em, t_sol_air_eff)` where `h_tr_em = Σ_s h_tr_em,s` and
/// `t_sol_air_eff` is the `h_tr_em`-weighted mean sol-air temperature, so
/// `h_tr_em * (t_sol_air_eff − T_mass) = Σ_s h_tr_em,s * (T_sol_air,s − T_mass)`
/// exactly. Falls back to `outdoor_temp` when the zone has no exterior
/// opaque area.
pub fn aggregate_zone_boundary(
    surfaces: &[ExteriorBoundarySurface],
    outdoor_temp_c: f64,
    sky_temp_c: f64,
) -> (f64, f64) {
    let mut h_tr_em = 0.0;
    let mut h_weighted_sol_air = 0.0;
    for s in surfaces {
        let h = s.h_tr_em();
        h_tr_em += h;
        h_weighted_sol_air += h * s.sol_air_temperature(outdoor_temp_c, sky_temp_c);
    }
    let t_sol_air_eff = if h_tr_em > 1e-12 {
        h_weighted_sol_air / h_tr_em
    } else {
        outdoor_temp_c
    };
    (h_tr_em, t_sol_air_eff)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #4166 acceptance criterion 2: a wall-only and a roof-only zone
    /// with identical area must produce different sol-air values, consistent
    /// with the 0.5 vs 1.0 sky view factor.
    ///
    /// With zero solar irradiance the sol-air formula reduces to
    /// `T_sol = T_out − F_sky·ε·ΔR/h_c`, so the wall/roof difference is
    /// exactly `(1.0 − 0.5)·ε·ΔR/h_c` when both use the same `h_c`.
    #[test]
    fn wall_only_and_roof_only_zones_differ_by_view_factor() {
        let outdoor = 30.0;
        let sky = 10.0;
        let h_c = 15.0;
        let eps = 0.9;
        let area = 50.0;

        let wall = ExteriorBoundarySurface {
            area_opaque_m2: area,
            azimuth_deg: 180.0,
            tilt_deg: 90.0,
            f_sky: ExteriorBoundarySurface::sky_view_factor_from_tilt(90.0),
            h_c_ext: h_c,
            windward: Some(true),
            convection_ab: (4.0, 4.0),
            irradiance_beam_diffuse_wm2: 0.0,
            irradiance_ground_wm2: 0.0,
            absorptance: ABSORPTANCE_WALL,
            emissivity: eps,
            r_materials: 2.0,
        };
        let roof = ExteriorBoundarySurface {
            area_opaque_m2: area,
            azimuth_deg: -1.0,
            tilt_deg: 0.0,
            f_sky: ExteriorBoundarySurface::sky_view_factor_from_tilt(0.0),
            h_c_ext: h_c,
            windward: None,
            convection_ab: (5.8, 3.8),
            irradiance_beam_diffuse_wm2: 0.0,
            irradiance_ground_wm2: 0.0,
            absorptance: ABSORPTANCE_ROOF,
            emissivity: eps,
            r_materials: 2.0,
        };

        assert!(
            (wall.f_sky - 0.5).abs() < 1e-12,
            "wall F_sky = {}",
            wall.f_sky
        );
        assert!(
            (roof.f_sky - 1.0).abs() < 1e-12,
            "roof F_sky = {}",
            roof.f_sky
        );

        let t_wall = wall.sol_air_temperature(outdoor, sky);
        let t_roof = roof.sol_air_temperature(outdoor, sky);

        // Analytic expectation: ΔT = (F_sky,roof − F_sky,wall)·ε·ΔR/h_c.
        let t_out_k = outdoor + 273.15;
        let t_sky_k = sky + 273.15;
        let delta_r = STEFAN_BOLTZMANN * (t_sky_k.powi(4) - t_out_k.powi(4));
        let expected_diff = 0.5 * eps * delta_r / h_c;

        // ΔR is negative (sky colder than air), so −F_sky·ε·ΔR/h_c is positive:
        // the roof (F_sky = 1.0) gets the full positive LW offset and sits
        // *above* the wall. The wall−roof difference is 0.5·ε·ΔR/h_c.
        assert!(
            t_roof > t_wall,
            "roof sol-air ({t_roof:.4}) should be above wall sol-air ({t_wall:.4}) for cold sky"
        );
        let actual_diff = t_wall - t_roof;
        assert!(
            (actual_diff - expected_diff).abs() < 1e-9,
            "wall−roof sol-air diff {actual_diff:.6} != analytic {expected_diff:.6}"
        );
    }

    #[test]
    fn sky_view_factor_tilt_convention() {
        assert!((ExteriorBoundarySurface::sky_view_factor_from_tilt(0.0) - 1.0).abs() < 1e-12);
        assert!((ExteriorBoundarySurface::sky_view_factor_from_tilt(90.0) - 0.5).abs() < 1e-12);
        assert!((ExteriorBoundarySurface::sky_view_factor_from_tilt(180.0) - 0.0).abs() < 1e-12);
        // 45° tilt → (1 + cos45°)/2 ≈ 0.8536
        let f45 = ExteriorBoundarySurface::sky_view_factor_from_tilt(45.0);
        assert!(
            (f45 - 0.8535533905932737).abs() < 1e-12,
            "F_sky(45°) = {f45}"
        );
    }

    #[test]
    fn windward_selection_matches_meteorological_convention() {
        // Wind FROM the north (0°) blows toward the south: north wall windward.
        assert!(is_windward(0.0, 0.0));
        assert!(!is_windward(0.0, 180.0));
        // Wind FROM the east (90°): east wall windward, west wall leeward.
        assert!(is_windward(90.0, 90.0));
        assert!(!is_windward(90.0, 270.0));
        // Wind FROM the south-west (225°): south and west walls windward.
        assert!(is_windward(225.0, 180.0));
        assert!(is_windward(225.0, 270.0));
        assert!(!is_windward(225.0, 0.0));
        assert!(!is_windward(225.0, 90.0));
    }

    #[test]
    fn aggregate_preserves_energy_form() {
        // Two surfaces with different sol-air values: the aggregate must
        // satisfy h_tr_em·(t_eff − T_m) = Σ h_s·(T_s − T_m) exactly.
        let s1 = ExteriorBoundarySurface {
            area_opaque_m2: 30.0,
            azimuth_deg: 180.0,
            tilt_deg: 90.0,
            f_sky: 0.5,
            h_c_ext: 17.6,
            windward: Some(true),
            convection_ab: (4.0, 4.0),
            irradiance_beam_diffuse_wm2: 400.0,
            irradiance_ground_wm2: 50.0,
            absorptance: ABSORPTANCE_WALL,
            emissivity: 0.9,
            r_materials: 2.0,
        };
        let s2 = ExteriorBoundarySurface {
            area_opaque_m2: 20.0,
            azimuth_deg: -1.0,
            tilt_deg: 0.0,
            f_sky: 1.0,
            h_c_ext: 18.7,
            windward: None,
            convection_ab: (5.8, 3.8),
            irradiance_beam_diffuse_wm2: 800.0,
            irradiance_ground_wm2: 0.0,
            absorptance: ABSORPTANCE_ROOF,
            emissivity: 0.9,
            r_materials: 3.0,
        };
        let surfaces = [s1.clone(), s2.clone()];
        let (h_tr_em, t_eff) = aggregate_zone_boundary(&surfaces, 25.0, 5.0);
        let t_mass = 22.0;
        let lumped = h_tr_em * (t_eff - t_mass);
        let exact: f64 = surfaces
            .iter()
            .map(|s| s.h_tr_em() * (s.sol_air_temperature(25.0, 5.0) - t_mass))
            .sum();
        assert!(
            (lumped - exact).abs() < 1e-9,
            "lumped {lumped:.9} != per-surface sum {exact:.9}"
        );
        // h_tr_em is the plain sum of per-surface contributions.
        let expected_h: f64 = surfaces.iter().map(|s| s.h_tr_em()).sum();
        assert!((h_tr_em - expected_h).abs() < 1e-12);
    }

    #[test]
    fn aggregate_empty_zone_falls_back_to_outdoor() {
        let (h, t) = aggregate_zone_boundary(&[], 21.0, 6.0);
        assert_eq!(h, 0.0);
        assert_eq!(t, 21.0);
    }
}
