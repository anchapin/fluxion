//! ASHRAE 140 free-float sol-air attribution diagnostic (Issue #4166).
//!
//! This diagnostic runs BEFORE the per-surface exterior-boundary production
//! change. For each ASHRAE 140 free-float case (600FF, 650FF, 900FF, 950FF)
//! it builds the production `ThermalModel`, extracts the zone's actual opaque
//! surfaces, and computes the zone sol-air temperature five ways at a
//! representative Denver summer peak hour:
//!
//! 1. `lumped` — the current production method: one `SolAirTemperature`
//!    built with `HorizontalRoofWindward` coefficients, `for_roof`, fed the
//!    area-averaged opaque solar gain.
//! 2. `f_sky only` — per-surface sky view factor (`F_sky = (1+cos tilt)/2`),
//!    everything else lumped.
//! 3. `wind only` — per-surface `(a, b)` wind coefficients with
//!    windward/leeward selection from actual wind direction, everything else
//!    lumped.
//! 4. `irradiance only` — per-orientation incident irradiance with the
//!    per-class absorptance (0.7 roof / 0.6 walls), everything else lumped.
//! 5. `full per-surface` — all three fixes (the #4166 target).
//!
//! The printed table attributes the lumped→per-surface delta to each
//! component. Run with `--nocapture` to see the report:
//! `cargo test --test all_tests exterior_boundary_diagnostic -- --nocapture`.

use fluxion::physics::cta::VectorField;
use fluxion::physics::exterior_convection::{
    h_c_ext_wind_dependent, wind_at_building_height_from_10m, ExteriorSurfaceDirection,
};
use fluxion::sim::exterior_boundary::{
    aggregate_zone_boundary, is_windward, ExteriorBoundarySurface, ABSORPTANCE_ROOF,
    ABSORPTANCE_WALL,
};
use fluxion::sim::sky_radiation::SolAirTemperature;
use fluxion::sim::thermal_model_core::ThermalModel;
use fluxion::solar::surface_irradiance::{
    calculate_surface_irradiance, Orientation as SolarOrientation,
};
use fluxion::solar::{calculate_solar_position, SolarPosition};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion_core::ashrae_cases::Orientation;

/// Representative Denver summer peak hour (TMY3, July 21 14:00 solar time).
/// Documented synthetic values — the diagnostic attributes *differences*
/// between sol-air variants, which are robust to the exact hour chosen.
const DIAG_OUTDOOR_TEMP_C: f64 = 33.0;
const DIAG_SKY_TEMP_C: f64 = 12.0;
const DIAG_DNI_WM2: f64 = 880.0;
const DIAG_DHI_WM2: f64 = 140.0;
const DIAG_WIND_SPEED_10M_MS: f64 = 3.8;
/// Meteorological wind direction (FROM), degrees clockwise from north.
const DIAG_WIND_FROM_DEG: f64 = 200.0;
const DIAG_EMISSIVITY: f64 = 0.9;
const DIAG_GROUND_REFLECTANCE: f64 = 0.2;

/// Per-surface data extracted from the production model for one zone.
struct DiagSurface {
    area_opaque_m2: f64,
    tilt_deg: f64,
    azimuth_deg: f64, // -1 for horizontal
    is_roof: bool,
    r_materials: f64,
    u_value: f64,
    irradiance_beam_diffuse: f64,
    irradiance_ground: f64,
}

fn orientation_to_tilt(o: &Orientation) -> Option<f64> {
    match o {
        Orientation::North | Orientation::East | Orientation::South | Orientation::West => {
            Some(90.0)
        }
        Orientation::Up | Orientation::Horizontal => Some(0.0),
        Orientation::Down => None, // floor: not an exterior boundary
    }
}

fn to_solar_orientation(o: &Orientation) -> Option<SolarOrientation> {
    match o {
        Orientation::North => Some(SolarOrientation::North),
        Orientation::East => Some(SolarOrientation::East),
        Orientation::South => Some(SolarOrientation::South),
        Orientation::West => Some(SolarOrientation::West),
        Orientation::Up | Orientation::Horizontal => Some(SolarOrientation::Horizontal),
        Orientation::Down => None,
    }
}

/// Builds the per-surface boundary vector for a case's zone 0.
/// Returns (surfaces, floor_area_m2).
fn build_diag_surfaces(case_id: &str) -> (Vec<DiagSurface>, f64) {
    let case = ASHRAE140Case::from_case_id(case_id).expect("known FF case");
    let spec = case.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);
    let zone_surfaces = &model.solar.surfaces[0];

    // Denver, July 21, 14:00 solar time.
    let sun_pos = calculate_solar_position(39.7, -105.0, 2024, 7, 21, 14.0, None);
    let day_of_year = 202; // July 21

    let mut out = Vec::new();
    for s in zone_surfaces {
        let tilt = match orientation_to_tilt(&s.orientation) {
            Some(t) => t,
            None => continue,
        };
        let solar_orientation = match to_solar_orientation(&s.orientation) {
            Some(o) => o,
            None => continue,
        };
        let irr = calculate_surface_irradiance(
            &sun_pos,
            DIAG_DNI_WM2,
            DIAG_DHI_WM2,
            None,
            solar_orientation,
            DIAG_GROUND_REFLECTANCE,
            day_of_year,
        );
        let area_opaque = (s.area - s.window_area).max(0.0);
        if area_opaque <= 0.0 {
            continue;
        }
        let is_roof = tilt < 45.0;
        // R_materials from the U-value: R_total = 1/U, subtract film
        // resistances to isolate the material stack (mirrors the
        // h_tr_em_wind_dependent convention in step_5r1c.rs).
        let r_materials = (1.0 / s.u_value - 0.13 - 0.04).max(0.5);
        out.push(DiagSurface {
            area_opaque_m2: area_opaque,
            tilt_deg: tilt,
            azimuth_deg: s.orientation.azimuth_deg(),
            is_roof,
            r_materials,
            u_value: s.u_value,
            irradiance_beam_diffuse: irr.beam_wm2 + irr.diffuse_wm2,
            irradiance_ground: irr.ground_reflected_wm2,
        });
    }
    // Floor area for the production opaque_solar normalization.
    let floor_area = model
        .0
        .setpoints
        .zone_area
        .as_ref()
        .first()
        .copied()
        .unwrap_or(48.0);
    (out, floor_area)
}

fn windward_direction(is_roof: bool, azimuth_deg: f64) -> ExteriorSurfaceDirection {
    if is_roof {
        ExteriorSurfaceDirection::HorizontalRoofWindward
    } else if is_windward(DIAG_WIND_FROM_DEG, azimuth_deg) {
        ExteriorSurfaceDirection::VerticalWallWindward
    } else {
        ExteriorSurfaceDirection::VerticalWallLeeward
    }
}

/// Current production (lumped) sol-air, replicating step_5r1c.rs exactly:
/// `total_opaque_gain = Σ(opaque_area·U·I·α·R_e)`, normalized by floor area,
/// then `for_roof` with α=0.7 and HorizontalRoofWindward coefficients.
fn lumped_sol_air(
    surfaces: &[DiagSurface],
    u_values: &[f64],
    floor_area_m2: f64,
    wind_speed_h: f64,
) -> f64 {
    const R_E: f64 = 1.0 / 18.3; // EXTERIOR_FILM_COEFF_DEFAULT
    let total_opaque_gain: f64 = surfaces
        .iter()
        .zip(u_values.iter())
        .map(|(s, &u)| {
            let alpha = if s.is_roof {
                ABSORPTANCE_ROOF
            } else {
                ABSORPTANCE_WALL
            };
            s.area_opaque_m2 * u * (s.irradiance_beam_diffuse + s.irradiance_ground) * alpha * R_E
        })
        .sum();
    let opaque_solar = total_opaque_gain / floor_area_m2;
    let h_c = h_c_ext_wind_dependent(
        ExteriorSurfaceDirection::HorizontalRoofWindward,
        wind_speed_h,
    );
    // Production uses alpha_sol_default = 0.7 (ashrae_140_default).
    let calc = SolAirTemperature::new(0.7, DIAG_EMISSIVITY, h_c);
    calc.for_roof(DIAG_OUTDOOR_TEMP_C, opaque_solar, DIAG_SKY_TEMP_C)
}

fn per_surface_boundary(s: &DiagSurface, wind_speed_h: f64) -> ExteriorBoundarySurface {
    let dir = windward_direction(s.is_roof, s.azimuth_deg);
    let windward = if s.azimuth_deg < 0.0 {
        None
    } else {
        Some(is_windward(DIAG_WIND_FROM_DEG, s.azimuth_deg))
    };
    ExteriorBoundarySurface {
        area_opaque_m2: s.area_opaque_m2,
        azimuth_deg: s.azimuth_deg,
        tilt_deg: s.tilt_deg,
        f_sky: ExteriorBoundarySurface::sky_view_factor_from_tilt(s.tilt_deg),
        h_c_ext: h_c_ext_wind_dependent(dir, wind_speed_h),
        windward,
        convection_ab: dir.ashrae_140_coefficients(),
        irradiance_beam_diffuse_wm2: s.irradiance_beam_diffuse,
        irradiance_ground_wm2: s.irradiance_ground,
        absorptance: if s.is_roof {
            ABSORPTANCE_ROOF
        } else {
            ABSORPTANCE_WALL
        },
        emissivity: DIAG_EMISSIVITY,
        r_materials: s.r_materials,
    }
}

#[test]
fn exterior_boundary_ff_sol_air_attribution() {
    let wind_speed_h = wind_at_building_height_from_10m(DIAG_WIND_SPEED_10M_MS, 2.7);

    println!();
    println!("=== Issue #4166 pre-change diagnostic: FF-case sol-air attribution ===");
    println!("Denver peak hour: T_out={DIAG_OUTDOOR_TEMP_C}°C, T_sky={DIAG_SKY_TEMP_C}°C,");
    println!("  DNI={DIAG_DNI_WM2}, DHI={DIAG_DHI_WM2}, wind {DIAG_WIND_SPEED_10M_MS} m/s from {DIAG_WIND_FROM_DEG}°");
    println!();

    for case_id in ["600FF", "650FF", "900FF", "950FF"] {
        let (surfaces, floor_area_m2) = build_diag_surfaces(case_id);
        assert!(!surfaces.is_empty(), "{case_id}: no exterior surfaces");
        let u_values: Vec<f64> = surfaces.iter().map(|s| s.u_value).collect();

        let t_lumped = lumped_sol_air(&surfaces, &u_values, floor_area_m2, wind_speed_h);

        // Sequential attribution: each step adds one physical fix.
        // Step 1: per-orientation irradiance + per-class absorptance.
        let v_irr: Vec<ExteriorBoundarySurface> = surfaces
            .iter()
            .map(|s| {
                let mut b = per_surface_boundary(s, wind_speed_h);
                // Keep F_sky + wind at lumped values.
                b.f_sky = 1.0;
                b.h_c_ext = h_c_ext_wind_dependent(
                    ExteriorSurfaceDirection::HorizontalRoofWindward,
                    wind_speed_h,
                );
                b
            })
            .collect();
        let (_, t_irr) = aggregate_zone_boundary(&v_irr, DIAG_OUTDOOR_TEMP_C, DIAG_SKY_TEMP_C);

        // Step 2: add per-surface sky view factor.
        let v_irr_fsky: Vec<ExteriorBoundarySurface> = surfaces
            .iter()
            .map(|s| {
                let mut b = per_surface_boundary(s, wind_speed_h);
                // Keep wind at lumped value.
                b.h_c_ext = h_c_ext_wind_dependent(
                    ExteriorSurfaceDirection::HorizontalRoofWindward,
                    wind_speed_h,
                );
                b
            })
            .collect();
        let (_, t_irr_fsky) =
            aggregate_zone_boundary(&v_irr_fsky, DIAG_OUTDOOR_TEMP_C, DIAG_SKY_TEMP_C);

        // Step 3: add per-surface wind coefficients (full per-surface).
        let v_full: Vec<ExteriorBoundarySurface> = surfaces
            .iter()
            .map(|s| per_surface_boundary(s, wind_speed_h))
            .collect();
        let (h_tr_em_full, t_full) =
            aggregate_zone_boundary(&v_full, DIAG_OUTDOOR_TEMP_C, DIAG_SKY_TEMP_C);

        println!("--- Case {case_id} ({} surfaces) ---", surfaces.len());
        println!("  lumped (current production) : {t_lumped:8.3} °C");
        println!(
            "  + per-orientation irradiance: {t_irr:8.3} °C  (Δ {:+.3})",
            t_irr - t_lumped
        );
        println!(
            "  + per-surface F_sky         : {t_irr_fsky:8.3} °C  (Δ {:+.3})",
            t_irr_fsky - t_irr
        );
        println!(
            "  + per-surface wind (a,b)    : {t_full:8.3} °C  (Δ {:+.3})",
            t_full - t_irr_fsky
        );
        println!(
            "  full per-surface (#4166)    : {t_full:8.3} °C  (Δ {:+.3})",
            t_full - t_lumped
        );
        println!("  h_tr_em (per-surface)       : {h_tr_em_full:8.3} W/K");
        println!();

        // Sanity: the full per-surface value must differ from lumped —
        // otherwise the diagnostic is vacuous.
        assert!(
            (t_full - t_lumped).abs() > 0.05,
            "{case_id}: per-surface sol-air unexpectedly equals lumped"
        );
    }
}
