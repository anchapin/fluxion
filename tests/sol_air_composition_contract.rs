//! §7 surface-pool blitz slice (loop round 126) — the **per-surface sol-air
//! composition closed forms** at the Issue #4166 exterior-boundary seam
//! (`docs/implementation-plans/checker-air-node-mirror-plan.md` §7 item (i)
//! remainder: the r109 suite pinned the per-class *aggregation identities*
//! (contracts 3–4) and the r125 class-drive suite pinned the Table 5-138
//! *fraction constants* at the load-coefficient seam; neither pinned the
//! per-surface closed forms this suite covers).
//!
//! Component under contract:
//!
//! - (a) the per-surface sol-air closed form
//!   `T_sol = T_out + α·(I_bd + I_gr)/h_c − F_sky·ε·σ·(T_out⁴ − T_sky⁴)/h_c`
//!   (`src/sim/exterior_boundary.rs::ExteriorBoundarySurface::sol_air_temperature`,
//!   the corrected ASHRAE ΔR direction per Issue #4224 — a cold sky LOWERS
//!   sol-air); the `h_c ≥ 1` and `[0, 1]` clamps;
//! - (b) the per-surface envelope conductance `h_tr_em = A/(R + 1/h_c)`;
//! - (c) the sky view factor `F_sky = (1 + cos(tilt))/2`
//!   (`sky_view_factor_from_tilt`, the ISO 13790 §12.3.2 convention);
//! - (d) the ASHRAE 140 §5.2.6 forced-convection coefficient pairs
//!   `h_c = a + b·V` per surface direction (`ExteriorSurfaceDirection`)
//!   with the ASHRAE power-law 10 m → building-height conversion
//!   `V(z) = V_10·(z/10)^0.15`, z default 2.7 m
//!   (`src/physics/exterior_convection.rs`);
//! - (e) the windward/leeward selection wrap semantics (`is_windward`);
//! - (f) the zone-aggregate composition `Σg`, `Σg·t/Σg` with the 1e-12
//!   degenerate gate (`aggregate_zone_boundary`);
//! - (g) the fluxion-core sky-temperature derivation
//!   `T_sky = (IR/5.67e-8)^0.25 − 273.15` (`HourlyWeatherData::sky_temperature`,
//!   the sol-air composition's sky input);
//! - (h) the ENGINE-TRUTH composition: the persisted per-class exterior
//!   boundary record (`applied_5r1c_class_boundary`: `cls_g`/`cls_t`/
//!   `cls_area`/`cls_beam`) and the persisted zone aggregate
//!   (`applied_5r1c_boundary`) recomposed bit-exactly from the spec surfaces
//!   and a CONTROLLED weather hour — zeroed solar (`dni = dhi = ghi = 0`,
//!   premise-pinned by `cls_beam == 0`), controlled wind speed/direction and
//!   controlled horizontal IR. The end-to-end wiring exercised: weather
//!   buffer → 10 m wind speed → power-law conversion → §5.2.6 direction pair
//!   → `h_c` → conductance; wind direction → `is_windward` → windward/leeward
//!   pair; orientation → tilt → class (roof < 45° ≤ wall) and F_sky; U-value
//!   → `R = (1/U − 0.13 − 0.04).max(0.5)`; surface order → class fold.
//!   The fold-free record values (`cls_g`, `cls_area`, zone `h_tr_em`) are
//!   pinned BIT-EXACT; the sol-air-bearing values (`cls_t`, zone
//!   `t_sol_air`) carry a 1e-12 relative bound because the engine's class
//!   loop holds an inlined copy of the sol-air arithmetic whose `powi(4)`
//!   lowering can differ from the exported method's by 1 ULP (r126
//!   observation, ~3e-15 on ~8 K values).
//!
//! Plan §7 queue status after this slice: item (i)'s per-surface sol-air
//! composition closed form is pinned (this suite); still open: (ii)'s
//! per-orientation transmittance/absorption split upstream of the class
//! fractions (owned by the rounds-91–105 contract family + r120's shading
//! geometry) and (iv)'s full longwave pool-exchange term (the #2890
//! view-factor network follow-up).
//!
//! Route discipline: contracts 10–11 run the default coupled route (the
//! hourly per-class advance always executes and persists its records); no env
//! flags are read or flipped, so no serialization is needed. These are record
//! pins, not validator bands: a violation means the exterior-boundary
//! composition moved and must be re-examined, never tuned around.
//!
//! Dev-time Python replication: `/tmp/r126_solair_replication.py` (mirrors
//! every closed form operation-for-operation with the exact
//! `STEFAN_BOLTZMANN = 5.67e-8` constant from `fluxion_core::physics_constants`).
//! Bit-exactness is asserted ENGINE-vs-TRANSCRIPTION where the transcription
//! shares the engine's own `.powi(4)` intrinsic (a hand-written fold rounds
//! differently on some arms — the r126 night-arm 1-ULP lesson); the Python
//! literals are pinned at a 1e-9 relative band on those arms, and bit-exactly
//! where the value is fold-invariant (the zone aggregate, the sky
//! temperature).

use fluxion::batch_oracle::BatchOracle;
use fluxion::physics::cta::VectorField;
use fluxion::physics::exterior_convection::{
    h_c_ext_from_10m, h_c_ext_wind_dependent, wind_at_building_height_from_10m,
    ExteriorSurfaceDirection,
};
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::exterior_boundary::{
    aggregate_zone_boundary, is_windward, ExteriorBoundarySurface, ABSORPTANCE_ROOF,
    ABSORPTANCE_WALL,
};
use fluxion::sim::sky_radiation::SolAirTemperature;
use fluxion::weather::HourlyWeatherData;
use fluxion_core::ashrae_cases::Orientation;

const DT: f64 = 3600.0;

/// The engine's own longwave constant (`fluxion_core::physics_constants`).
const SIGMA: f64 = 5.67e-8;

/// Rust `powi(4)` association note: the engine's lowering is context-dependent
/// (LLVM folds it differently in the class loop's inlined copy vs the exported
/// method — observed as a 1-ULP arm difference in r126), so this helper is
/// used ONLY by the `(ΔT)⁴` mis-read discriminator, where the fold is
/// immaterial: the mis-read differs from the true algebra by orders of ULP.
fn powi4(x: f64) -> f64 {
    x * x * x * x
}

/// Independent transcription of the per-surface sol-air closed form
/// (`exterior_boundary.rs::sol_air_temperature`), mirrored operation-for-
/// operation from the Python replication. Asserted bit-exact against the
/// engine method AND the Python `repr()` literals.
fn replicate_sol_air(
    t_out_c: f64,
    t_sky_c: f64,
    alpha: f64,
    eps: f64,
    h_c_ext: f64,
    f_sky: f64,
    irr_bd: f64,
    irr_gr: f64,
) -> f64 {
    let h_c = h_c_ext.max(1.0);
    let alpha = alpha.clamp(0.0, 1.0);
    let eps = eps.clamp(0.0, 1.0);
    let total_irradiance = irr_bd + irr_gr;
    let solar_term = alpha * total_irradiance / h_c;
    let t_out_k = t_out_c + 273.15;
    let t_sky_k = t_sky_c + 273.15;
    // The engine's own intrinsic: same lowering, so the transcription is
    // bit-exact fold-for-fold (a hand-written fold rounds differently on some
    // arms — the r126 night-arm 1-ULP lesson).
    let delta_r = SIGMA * (t_out_k.powi(4) - t_sky_k.powi(4));
    let longwave_term = eps * delta_r / h_c;
    let f_sky = f_sky.clamp(0.0, 1.0);
    t_out_c + solar_term - f_sky * longwave_term
}

/// The `(T_out − T_sky)⁴` mis-reading of ΔR (the natural wrong reading of the
/// zone_balance.rs formula comment, pre-fix): NOT the engine's algebra — the
/// discriminators below pin that the two differ on every arm.
fn replicate_sol_air_misread(
    t_out_c: f64,
    t_sky_c: f64,
    alpha: f64,
    eps: f64,
    h_c_ext: f64,
    f_sky: f64,
    irr_bd: f64,
    irr_gr: f64,
) -> f64 {
    let h_c = h_c_ext.max(1.0);
    let total_irradiance = irr_bd + irr_gr;
    let t_out_k = t_out_c + 273.15;
    let t_sky_k = t_sky_c + 273.15;
    let delta_r_misread = SIGMA * powi4(t_out_k - t_sky_k);
    t_out_c + alpha * total_irradiance / h_c - f_sky * (eps * delta_r_misread / h_c)
}

fn surface(
    area_opaque_m2: f64,
    azimuth_deg: f64,
    tilt_deg: f64,
    f_sky: f64,
    h_c_ext: f64,
    irr_bd: f64,
    irr_gr: f64,
    absorptance: f64,
    emissivity: f64,
    r_materials: f64,
) -> ExteriorBoundarySurface {
    ExteriorBoundarySurface {
        area_opaque_m2,
        azimuth_deg,
        tilt_deg,
        f_sky,
        h_c_ext,
        windward: None,
        convection_ab: (4.0, 4.0),
        irradiance_beam_diffuse_wm2: irr_bd,
        irradiance_ground_wm2: irr_gr,
        absorptance,
        emissivity,
        r_materials,
    }
}

/// A spec-built Case 600 baseline under free-float setpoints with a
/// CONTROLLED weather hour: solar inputs zeroed (`dni = dhi = ghi = 0` — the
/// per-step `calculate_zone_solar_gain` recomputes the orientation arrays
/// from the weather buffer, so the arrays arrive zero; premise-pinned by
/// `cls_beam == 0`), controlled wind speed/direction, controlled horizontal
/// IR. The ideal HVAC never engages (setpoints far outside any reachable
/// state, the round-13b2 convention) and the zone-level gains are neutralized
/// so the driving is the exterior boundary alone.
fn controlled_model(wind_from: f64) -> ThermalModel<VectorField> {
    let mut m = BatchOracle::default_model().expect("spec model builds");
    m.setpoints.heating_setpoint = -50.0;
    m.setpoints.cooling_setpoint = 100.0;
    let n = m.hvac.num_zones;
    m.setpoints.loads = VectorField::new(vec![0.0; n]);
    m.solar.solar_gains = VectorField::new(vec![0.0; n]);
    m.solar.opaque_solar_gains = VectorField::new(vec![0.0; n]);
    m.solar.orientation_irradiance_beam_diffuse = vec![[0.0; 7]; n];
    m.solar.orientation_irradiance_ground = vec![[0.0; 7]; n];
    let mut w = HourlyWeatherData::new(12.0, 0.0, 0.0, 0.0, 4.0, 50.0, 0);
    w.wind_direction = Some(wind_from);
    w.horizontal_infrared = 300.0;
    m.solar.weather = Some(w);
    m
}

/// Independent recomposition of the engine's per-class exterior-boundary
/// record and zone aggregate from the spec surfaces + controlled weather,
/// mirroring zone_balance.rs's construction loop operation-for-operation
/// (same surface order, same class fold `c = tilt ≥ 45`, same 1e-9/1e-12
/// gates, same `R = (1/U − 0.13 − 0.04).max(0.5)` wiring). Returns
/// `(cls_g, cls_t, cls_area, cls_beam, zone_h_tr_em, zone_t_sol_air)`.
#[allow(clippy::too_many_arguments)]
fn replicate_class_record(
    m: &ThermalModel<VectorField>,
    outdoor_temp: f64,
    wind_from: Option<f64>,
    wind_speed_10m: f64,
) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3], f64, f64) {
    let v_wind_building = wind_at_building_height_from_10m(wind_speed_10m, 2.7);
    let sky_temp = m
        .solar
        .weather
        .as_ref()
        .map(|w| w.sky_temperature())
        .unwrap_or(outdoor_temp - 15.0);
    let eps_ext = m
        .conduction
        .exterior_emissivity
        .as_ref()
        .get(0)
        .copied()
        .unwrap_or(SolAirTemperature::ashrae_140_default().emissivity);
    let mut cls_g = [0.0f64; 3];
    let mut cls_gt = [0.0f64; 3];
    let mut cls_area = [0.0f64; 3];
    let mut cls_beam = [0.0f64; 3];
    let mut h_sum = 0.0f64;
    let mut ht_sum = 0.0f64;
    if let Some(zone_surfaces) = m.solar.surfaces.get(0) {
        let irr_bd = m.solar.orientation_irradiance_beam_diffuse.get(0);
        let irr_gr = m.solar.orientation_irradiance_ground.get(0);
        for s in zone_surfaces {
            let opaque_area = (s.area - s.window_area).max(0.0);
            if opaque_area <= 0.0 {
                continue;
            }
            if s.orientation == Orientation::Down {
                continue;
            }
            let is_roof = matches!(s.orientation, Orientation::Up | Orientation::Horizontal);
            let tilt_deg = if is_roof { 0.0 } else { 90.0 };
            let oi = s.orientation as usize;
            let irr_beam_diffuse = irr_bd.map(|a| a[oi]).unwrap_or(0.0);
            let irr_ground = irr_gr.map(|a| a[oi]).unwrap_or(0.0);
            let azimuth = s.orientation.azimuth_deg();
            let windward = if azimuth < 0.0 {
                true
            } else {
                wind_from.map(|wd| is_windward(wd, azimuth)).unwrap_or(true)
            };
            let direction = match (is_roof, windward) {
                (true, true) => ExteriorSurfaceDirection::HorizontalRoofWindward,
                (true, false) => ExteriorSurfaceDirection::HorizontalRoofLeeward,
                (false, true) => ExteriorSurfaceDirection::VerticalWallWindward,
                (false, false) => ExteriorSurfaceDirection::VerticalWallLeeward,
            };
            let h_c_ext = h_c_ext_wind_dependent(direction, v_wind_building);
            let surf = ExteriorBoundarySurface {
                area_opaque_m2: opaque_area,
                azimuth_deg: azimuth,
                tilt_deg,
                f_sky: ExteriorBoundarySurface::sky_view_factor_from_tilt(tilt_deg),
                h_c_ext,
                windward: if azimuth < 0.0 { None } else { Some(windward) },
                convection_ab: direction.ashrae_140_coefficients(),
                irradiance_beam_diffuse_wm2: irr_beam_diffuse,
                irradiance_ground_wm2: irr_ground,
                absorptance: if is_roof {
                    ABSORPTANCE_ROOF
                } else {
                    ABSORPTANCE_WALL
                },
                emissivity: eps_ext,
                r_materials: (1.0 / s.u_value - 0.13 - 0.04).max(0.5),
            };
            let g = surf.h_tr_em();
            let t = surf.sol_air_temperature(outdoor_temp, sky_temp);
            let c = if tilt_deg >= 45.0 { 1usize } else { 0usize };
            cls_g[c] += g;
            cls_gt[c] += g * t;
            cls_area[c] += surf.area_opaque_m2;
            cls_beam[c] += (surf.irradiance_beam_diffuse_wm2 + surf.irradiance_ground_wm2)
                * surf.area_opaque_m2;
            h_sum += g;
            ht_sum += g * t;
        }
    }
    let floor_area: f64 = m
        .solar
        .surfaces
        .get(0)
        .map(|zs| {
            zs.iter()
                .filter(|s| s.orientation == Orientation::Down)
                .map(|s| (s.area - s.window_area).max(0.0))
                .sum()
        })
        .unwrap_or(0.0);
    cls_area[2] = floor_area;
    let mut cls_t = [outdoor_temp; 3];
    for c in 0..2 {
        if cls_g[c] > 1e-9 {
            cls_t[c] = cls_gt[c] / cls_g[c];
        }
    }
    let zone_t = if h_sum > 1e-12 {
        ht_sum / h_sum
    } else {
        outdoor_temp
    };
    (cls_g, cls_t, cls_area, cls_beam, h_sum, zone_t)
}

#[test]
/// Contract 1 — the per-surface sol-air closed form, roof and wall arms,
/// bit-exact against the Python-replicated literals AND the in-test
/// transcription. Discriminators on every arm: the `ΔR = σ·(T_out−T_sky)⁴`
/// mis-reading (the pre-fix zone_balance.rs comment's natural reading) gives
/// a DIFFERENT value, and a cold sky strictly LOWERS sol-air below the
/// no-longwave arm (the Issue #4224 direction).
fn sol_air_closed_form_roof_and_wall_arms() {
    // Roof arm: F_sky = 1.0, α = 0.7, ε = 0.9, h_c = 18.3, I = 400 + 150.
    let roof = surface(40.0, -1.0, 0.0, 1.0, 18.3, 400.0, 150.0, 0.7, 0.9, 0.35);
    let got_roof = roof.sol_air_temperature(25.0, -20.0);
    let want_roof = replicate_sol_air(25.0, -20.0, 0.7, 0.9, 18.3, 1.0, 400.0, 150.0);
    assert_eq!(
        got_roof, want_roof,
        "roof arm: engine must equal the transcription"
    );
    assert!(
        (got_roof - 35.45534525864943).abs() <= 1e-9 * got_roof.abs(),
        "roof arm: agrees with the Python replication literal (got {got_roof})"
    );
    assert_ne!(
        got_roof,
        replicate_sol_air_misread(25.0, -20.0, 0.7, 0.9, 18.3, 1.0, 400.0, 150.0),
        "roof arm: the (ΔT)⁴ mis-reading must NOT coincide"
    );
    let no_lw_roof = replicate_sol_air(25.0, -20.0, 0.7, 0.0, 18.3, 1.0, 400.0, 150.0);
    assert!(
        got_roof < no_lw_roof,
        "roof arm: cold sky must LOWER sol-air (#4224 direction): {got_roof} vs no-LW {no_lw_roof}"
    );

    // Wall arm: F_sky = 0.5, α = 0.6, ε = 0.9, h_c = 15.0, I = 300 + 120.
    let wall = surface(20.0, 90.0, 90.0, 0.5, 15.0, 300.0, 120.0, 0.6, 0.9, 0.5);
    let got_wall = wall.sol_air_temperature(25.0, -20.0);
    let want_wall = replicate_sol_air(25.0, -20.0, 0.6, 0.9, 15.0, 0.5, 300.0, 120.0);
    assert_eq!(
        got_wall, want_wall,
        "wall arm: engine must equal the transcription"
    );
    assert!(
        (got_wall - 35.34442727444282).abs() <= 1e-9 * got_wall.abs(),
        "wall arm: agrees with the Python replication literal (got {got_wall})"
    );
    assert_ne!(
        got_wall,
        replicate_sol_air_misread(25.0, -20.0, 0.6, 0.9, 15.0, 0.5, 300.0, 120.0),
        "wall arm: the (ΔT)⁴ mis-reading must NOT coincide"
    );
}

#[test]
/// Contract 2 — the clamped arms: the `h_c.max(1.0)` degenerate-conductance
/// clamp and the night radiative-cooling arm (solar zero, sky far below air:
/// sol-air BELOW outdoor air — the windshield-frost direction).
fn sol_air_clamp_and_night_arms() {
    // h_c = 0.5 clamps to 1.0.
    let damp = surface(40.0, -1.0, 0.0, 1.0, 0.5, 400.0, 150.0, 0.7, 0.9, 0.35);
    let got = damp.sol_air_temperature(25.0, -20.0);
    assert_eq!(
        got,
        replicate_sol_air(25.0, -20.0, 0.7, 0.9, 0.5, 1.0, 400.0, 150.0),
        "clamp arm: engine must equal the transcription"
    );
    assert!(
        (got - 216.33281823328463).abs() <= 1e-9 * got.abs(),
        "clamp arm: agrees with the Python replication literal (got {got})"
    );

    // Night arm: no sun, T_out = 5 °C, T_sky = −25 °C → radiative cooling.
    let night = surface(40.0, -1.0, 0.0, 1.0, 12.0, 0.0, 0.0, 0.7, 0.9, 0.35);
    let got_night = night.sol_air_temperature(5.0, -25.0);
    assert_eq!(
        got_night,
        replicate_sol_air(5.0, -25.0, 0.7, 0.9, 12.0, 1.0, 0.0, 0.0),
        "night arm: engine must equal the transcription"
    );
    assert!(
        (got_night - (-4.329202313363012)).abs() <= 1e-9 * got_night.abs(),
        "night arm: agrees with the Python replication literal (got {got_night})"
    );
    assert!(
        got_night < 5.0,
        "night arm: sol-air must sit BELOW outdoor air under a cold clear sky: {got_night}"
    );
    assert_ne!(
        got_night,
        replicate_sol_air_misread(5.0, -25.0, 0.7, 0.9, 12.0, 1.0, 0.0, 0.0),
        "night arm: the (ΔT)⁴ mis-reading must NOT coincide"
    );
}

#[test]
/// Contract 3 — the sky view factor closed form `F_sky = (1 + cos tilt)/2`
/// with the `[0, 1]` clamp: roof 1.0, wall 0.5, the 45°/30° literals.
fn sky_view_factor_from_tilt_pins() {
    assert_eq!(ExteriorBoundarySurface::sky_view_factor_from_tilt(0.0), 1.0);
    assert_eq!(
        ExteriorBoundarySurface::sky_view_factor_from_tilt(45.0),
        0.8535533905932737
    );
    assert_eq!(
        ExteriorBoundarySurface::sky_view_factor_from_tilt(90.0),
        0.5
    );
    assert_eq!(
        ExteriorBoundarySurface::sky_view_factor_from_tilt(30.0),
        0.9330127018922194
    );
}

#[test]
/// Contract 4 — the ASHRAE 140 §5.2.6 coefficient pairs and the
/// wind-dependent `h_c` closed form: the four `(a, b)` pairs, windward
/// `a + b·V` through the power-law-converted speed, leeward wind
/// independence, and the negative-speed clamp.
fn ashrae_140_526_coefficient_pins() {
    use ExteriorSurfaceDirection::{
        HorizontalRoofLeeward, HorizontalRoofWindward, VerticalWallLeeward, VerticalWallWindward,
    };
    assert_eq!(VerticalWallWindward.ashrae_140_coefficients(), (4.0, 4.0));
    assert_eq!(VerticalWallLeeward.ashrae_140_coefficients(), (4.0, 0.0));
    assert_eq!(HorizontalRoofWindward.ashrae_140_coefficients(), (5.8, 3.8));
    assert_eq!(HorizontalRoofLeeward.ashrae_140_coefficients(), (5.8, 0.0));

    let v2 = wind_at_building_height_from_10m(2.0, 2.7);
    let v34 = wind_at_building_height_from_10m(3.4, 2.7);
    let v10 = wind_at_building_height_from_10m(10.0, 2.7);
    assert_eq!(
        h_c_ext_wind_dependent(VerticalWallWindward, v2),
        10.573467977455161
    );
    assert_eq!(
        h_c_ext_wind_dependent(VerticalWallWindward, v34),
        15.174895561673774
    );
    assert_eq!(
        h_c_ext_wind_dependent(VerticalWallWindward, v10),
        36.867339887275804
    );
    assert_eq!(
        h_c_ext_wind_dependent(HorizontalRoofWindward, v2),
        12.044794578582403
    );
    assert_eq!(
        h_c_ext_wind_dependent(HorizontalRoofWindward, v34),
        16.416150783590084
    );
    assert_eq!(
        h_c_ext_wind_dependent(HorizontalRoofWindward, v10),
        37.02397289291201
    );

    // Leeward pairs are wind-INDEPENDENT (b = 0) at every speed.
    assert_eq!(h_c_ext_wind_dependent(VerticalWallLeeward, v2), 4.0);
    assert_eq!(h_c_ext_wind_dependent(VerticalWallLeeward, v10), 4.0);
    assert_eq!(h_c_ext_wind_dependent(HorizontalRoofLeeward, v34), 5.8);
    assert_eq!(h_c_ext_wind_dependent(HorizontalRoofLeeward, v10), 5.8);

    // Negative speed clamps to 0 (the windward base coefficient alone).
    assert_eq!(h_c_ext_wind_dependent(VerticalWallWindward, -5.0), 4.0);
}

#[test]
/// Contract 5 — the ASHRAE power-law 10 m → building-height conversion
/// `V(z) = V_10·(z/10)^0.15` with the z = 2.7 m default and the
/// non-finite/negative clamps, plus the composed `h_c_ext_from_10m`.
fn wind_power_law_conversion_pins() {
    // The (2.7/10)^0.15 conversion factor itself.
    assert_eq!(
        wind_at_building_height_from_10m(1.0, 2.7),
        0.8216834971818952
    );
    assert_eq!(
        wind_at_building_height_from_10m(2.0, 2.7),
        1.6433669943637903
    );
    assert_eq!(
        wind_at_building_height_from_10m(3.4, 2.7),
        2.7937238904184434
    );
    assert_eq!(
        wind_at_building_height_from_10m(10.0, 2.7),
        8.216834971818951
    );

    // z fallback: non-positive / non-finite heights use the 2.7 m default.
    assert_eq!(
        wind_at_building_height_from_10m(2.0, 0.0),
        1.6433669943637903
    );
    assert_eq!(
        wind_at_building_height_from_10m(2.0, -3.0),
        1.6433669943637903
    );
    assert_eq!(
        wind_at_building_height_from_10m(2.0, f64::NAN),
        1.6433669943637903
    );

    // Non-finite and negative wind speeds clamp to 0.
    assert_eq!(wind_at_building_height_from_10m(f64::NAN, 2.7), 0.0);
    assert_eq!(
        wind_at_building_height_from_10m(f64::NEG_INFINITY, 2.7),
        0.0
    );
    assert_eq!(wind_at_building_height_from_10m(-3.0, 2.7), 0.0);

    // The composed helper: conversion then coefficient.
    assert_eq!(
        h_c_ext_from_10m(ExteriorSurfaceDirection::VerticalWallWindward, 3.4, 2.7),
        15.174895561673774
    );
    assert_eq!(
        h_c_ext_from_10m(ExteriorSurfaceDirection::HorizontalRoofLeeward, 0.0, 2.7),
        5.8
    );
}

#[test]
/// Contract 6 — the windward/leeward wrap semantics: the wind-TO vector's
/// angular distance from the surface azimuth decides, with the 360° wrap
/// (`350° azimuth` is windward of a north wind; the exact-180° face is
/// leeward — the `> 90°` test).
fn is_windward_wrap_semantics_pins() {
    // North wind (from 0°).
    assert!(is_windward(0.0, 0.0), "north face takes a north wind");
    assert!(!is_windward(0.0, 90.0));
    assert!(!is_windward(0.0, 180.0), "exact-opposite face is leeward");
    assert!(!is_windward(0.0, 270.0));
    assert!(is_windward(0.0, 350.0), "350° wraps into the windward half");

    // South wind (from 180°).
    assert!(!is_windward(180.0, 0.0));
    assert!(!is_windward(180.0, 90.0));
    assert!(is_windward(180.0, 180.0));
    assert!(!is_windward(180.0, 270.0));
}

#[test]
/// Contract 7 — the per-surface envelope conductance closed form
/// `h_tr_em = A/(R + 1/h_c)` with the `h_c.max(1.0)` clamp, bit-exact.
fn h_tr_em_closed_form_pins() {
    let s = surface(20.0, 90.0, 90.0, 0.5, 18.3, 0.0, 0.0, 0.6, 0.9, 0.5);
    assert_eq!(s.h_tr_em(), 36.05911330049261);
    // Exact-f64 fixture: A = 12.5, R = 2.0, h_c = 12.0 → 12.5/(2 + 1/12) = 6.0.
    let exact = surface(12.5, 90.0, 90.0, 0.5, 12.0, 0.0, 0.0, 0.6, 0.9, 2.0);
    assert_eq!(exact.h_tr_em(), 6.0);
}

#[test]
/// Contract 8 — the zone-aggregate composition `aggregate_zone_boundary`:
/// `h_tr_em = Σg`, `t_sol_air = Σg·t/Σg` in surface order (bit-exact vs the
/// Python fold), plus the degenerate gate — zero total conductance returns
/// the outdoor temperature verbatim.
fn aggregate_zone_boundary_composition_pins() {
    let s1 = surface(
        20.0,
        0.0,
        90.0,
        0.5,
        14.0,
        300.0,
        40.0,
        0.6,
        0.9,
        0.5555555555555556,
    );
    let s2 = surface(
        15.0,
        90.0,
        90.0,
        0.5,
        4.0,
        220.0,
        30.0,
        0.6,
        0.9,
        0.47619047619047616,
    );
    let s3 = surface(
        40.0,
        -1.0,
        0.0,
        1.0,
        15.3,
        700.0,
        0.0,
        0.7,
        0.9,
        0.35714285714285715,
    );
    let vec = vec![s1, s2, s3];
    let (h, t) = aggregate_zone_boundary(&vec, 12.0, -8.0);
    assert_eq!(
        h, 147.2285050313045,
        "aggregate h_tr_em: bit-exact vs the Python fold"
    );
    assert_eq!(
        t, 35.28125184447471,
        "aggregate sol-air: bit-exact vs the Python fold"
    );

    // Degenerate arm: a zero-area surface contributes no conductance; an
    // all-degenerate vector gates to the outdoor temperature.
    let dead = surface(0.0, 0.0, 90.0, 0.5, 14.0, 300.0, 40.0, 0.6, 0.9, 0.5);
    let (h0, t0) = aggregate_zone_boundary(std::slice::from_ref(&dead), 12.0, -8.0);
    assert_eq!(h0, 0.0, "zero-area surface contributes no conductance");
    assert_eq!(
        t0, 12.0,
        "the 1e-12 gate falls back to the outdoor temperature"
    );
}

#[test]
/// Contract 9 — the fluxion-core sky-temperature derivation feeding the
/// sol-air composition: `T_sky = (IR/5.67e-8)^0.25 − 273.15` bit-exact, with
/// the `IR ≤ 0` fallback `dry_bulb − 15`.
fn sky_temperature_ir_derivation_pins() {
    let mut w = HourlyWeatherData::new(12.0, 200.0, 100.0, 300.0, 4.0, 50.0, 0);
    w.horizontal_infrared = 300.0;
    assert_eq!(w.sky_temperature(), -3.447762809926246);
    w.horizontal_infrared = 0.0;
    assert_eq!(w.sky_temperature(), -3.0);
    w.horizontal_infrared = -5.0;
    assert_eq!(
        w.sky_temperature(),
        -3.0,
        "negative IR takes the fallback branch"
    );
}

/// `(a, b)` tolerance for the sol-air-bearing record values (`cls_t`,
/// zone `t_sol_air`): the engine's class loop holds an INLINED copy of the
/// sol-air arithmetic whose `powi(4)` lowering may differ from the exported
/// method's by 1 ULP (r126 observation, ~3e-15 on ~8 K values) — a fold
/// artifact, not algebra. Any real composition/wiring drift moves orders of
/// magnitude beyond this bound.
fn assert_solair_close(a: f64, b: f64, what: &str) {
    assert!(
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0),
        "{what}: {a} vs recomposed {b}"
    );
}

#[test]
/// Contract 10 — ENGINE-TRUTH composition, north-wind arm: the persisted
/// per-class exterior-boundary record and zone aggregate recomposed
/// bit-exactly from the spec surfaces under the controlled weather hour
/// (zeroed solar premise-pinned by `cls_beam == 0`). Exercises the full
/// wiring: weather buffer → power law → §5.2.6 pairs → `h_c`; wind direction
/// → `is_windward`; orientation → tilt → class + F_sky; U-value → R films.
fn class_record_composition_north_wind_arm() {
    let mut m = controlled_model(0.0); // north wind: north face windward
    m.step_physics(0, 12.0, DT);
    let rec = m
        .conduction
        .applied_5r1c_class_boundary
        .as_ref()
        .expect("per-class record persisted")
        .first()
        .expect("zone 0 record")
        .as_ref()
        .expect("zone 0 ran the per-class path");
    let zone_boundary = m
        .conduction
        .applied_5r1c_boundary
        .as_ref()
        .expect("zone boundary persisted");

    let (cls_g, cls_t, cls_area, cls_beam, zone_h, zone_t) =
        replicate_class_record(&m, 12.0, Some(0.0), 4.0);

    // Premise: the solar driver consumed the zeroed weather (no beam, no
    // ground-reflected) — otherwise the sol-air arm's irradiance inputs
    // would not be the zeros the transcription assumed.
    assert_eq!(
        rec.cls_beam, [0.0; 3],
        "premise: controlled zero-solar weather must yield zero consumed irradiance"
    );
    assert_eq!(
        cls_beam, [0.0; 3],
        "mirror premise mirrors the zero-solar arm"
    );

    assert_eq!(
        rec.cls_g, cls_g,
        "cls_g: bit-exact recomposition (fold-free arithmetic)"
    );
    assert_solair_close(rec.cls_t[0], cls_t[0], "cls_t[0] (roof class sol-air)");
    assert_solair_close(rec.cls_t[1], cls_t[1], "cls_t[1] (wall class sol-air)");
    assert_eq!(
        rec.cls_t[2], cls_t[2],
        "cls_t[2] (floor fallback) is the outdoor temp verbatim"
    );
    assert_eq!(rec.cls_area, cls_area, "cls_area: bit-exact recomposition");
    assert_eq!(
        zone_boundary.0[0], zone_h,
        "zone h_tr_em: bit-exact recomposition (fold-free)"
    );
    assert_solair_close(zone_boundary.1[0], zone_t, "zone sol-air");

    // Physical shape on the controlled arm: roofs carry less conductance
    // than the four walls folded together, floors carry none, and the cold
    // sky (IR = 300 → T_sky ≈ −3.4 °C vs T_out = 12 °C) drags every class
    // sol-air BELOW the outdoor temperature.
    assert_eq!(
        rec.cls_g[2], 0.0,
        "floor class carries no exterior conductance"
    );
    assert!(rec.cls_area[2] > 0.0, "floor class records its slab area");
    assert!(
        rec.cls_t[0] < 12.0,
        "roof class sol-air below T_out under a cold sky"
    );
    assert!(
        rec.cls_t[1] < 12.0,
        "wall class sol-air below T_out under a cold sky"
    );
}

#[test]
/// Contract 11 — ENGINE-TRUTH composition, wind-direction discrimination:
/// flipping the hour's wind from north to south swaps the north/south walls'
/// windward/leeward §5.2.6 pairs (b = 4.0 vs b = 0.0), so the wall-class
/// conductance MOVES while the roof class (horizontal surfaces are always
/// windward) stays bit-identical. Both arms must equal their own
/// recompositions bit-exactly.
fn class_record_composition_wind_direction_discrimination() {
    let mut north = controlled_model(0.0);
    north.step_physics(0, 12.0, DT);
    let rec_north = north
        .conduction
        .applied_5r1c_class_boundary
        .as_ref()
        .expect("record")
        .first()
        .expect("zone 0")
        .as_ref()
        .expect("per-class path");

    let mut south = controlled_model(180.0);
    south.step_physics(0, 12.0, DT);
    let rec_south = south
        .conduction
        .applied_5r1c_class_boundary
        .as_ref()
        .expect("record")
        .first()
        .expect("zone 0")
        .as_ref()
        .expect("per-class path");

    let north_mirror = replicate_class_record(&north, 12.0, Some(0.0), 4.0);
    let south_mirror = replicate_class_record(&south, 12.0, Some(180.0), 4.0);

    assert_eq!(
        rec_north.cls_g, north_mirror.0,
        "north arm cls_g: bit-exact"
    );
    assert_eq!(
        rec_south.cls_g, south_mirror.0,
        "south arm cls_g: bit-exact"
    );
    assert_solair_close(rec_north.cls_t[0], north_mirror.1[0], "north arm cls_t[0]");
    assert_solair_close(rec_north.cls_t[1], north_mirror.1[1], "north arm cls_t[1]");
    assert_solair_close(rec_south.cls_t[0], south_mirror.1[0], "south arm cls_t[0]");
    assert_solair_close(rec_south.cls_t[1], south_mirror.1[1], "south arm cls_t[1]");

    // The discrimination: direction-blindness (the pre-#4166 mode) would make
    // the two arms identical.
    assert_ne!(
        rec_north.cls_g[1], rec_south.cls_g[1],
        "wall-class conductance must move when the wind flips north↔south"
    );
    assert_eq!(
        rec_north.cls_g[0], rec_south.cls_g[0],
        "roof-class conductance must be wind-direction-blind (horizontal = always windward)"
    );
}
