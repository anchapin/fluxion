//! Shading geometry contracts — the overhang's roof-edge mounting (loop round 119).
//!
//! Premise (verified against the primary sources in `docs/`):
//!
//! - ASHRAE 140-2017a Figure 5-32 (Case 610): the south overhang projects
//!   1.0 m and mounts at the roof edge, **0.5 m above the window head**
//!   (window head at 2.2 m = sill 0.2 + height 2.0, wall/roof at 2.7 m;
//!   Figure 5-1 fixes the Case-600 window geometry: two 3 m × 2 m south
//!   windows, sill 0.2 m, 0.5 m head-to-roof strip).
//! - The standard's own reference inputs (User's Manual, 140-2023, §7.2.2.1.1
//!   worked example): the EnergyPlus file adds `Shading:Zone:Detailed` with
//!   all four vertices at **Z = 2.700000** (the roof line, over the 2.2 m
//!   head), and the CSE file carries `ohDistUp = 1.64042 ft = 0.5 m` — the
//!   overhang-to-window-head distance, verbatim.
//! - Cases 630/930 reuse the same mounting: the E/W windows are "exactly as
//!   in Case 600" other than orientation (§5.2.2.1.2.1), so their overhang
//!   gap is also 0.5 m (Figures 5-43/5-54: 2.0 m window + 0.5 m head strip
//!   in the 2.7 m elevation; fins at the window edges spanning at least the
//!   window height, Figure 5-65).
//!
//! The engine's spec-to-surface conversion mounted the overhang at the window
//! head (`distance_above: 0.0`, loop round 4, 2026-10-09) citing Case-630
//! wording ("at the top of the window") that does not exist in either primary
//! source. Gap 0 shades the head strip at every sub-horizon-positive profile
//! angle — an over-shade of Cases 610/630/910/930 relative to the standard's
//! geometry. Fixed round 119: gap = (mounting_height − (sill + height)).max(0)
//! computed from the spec's own window dims (0.5 m for every 140 shaded case).
//!
//! Known, recorded simplification (NOT re-litigated here): the shaded
//! fraction multiplies the BEAM component only; sky-diffuse passes unshaded.
//! The reference engines reduce diffuse through the shading geometry too (E+
//! shadowing; the CSE input carries the post-shade `wnVfSkyDf = 0.3936`);
//! engine-side diffuse view-factor reduction is the scoped remainder
//! (LOOP119.md, dated).
//!
//! Python replication: `/tmp/r119_replicate.py` (dev-time evidence; predicts
//! every pinned value below, including the pre-fix red gots).
//!
//! env_bool: none — these contracts read the unconditional default route and
//! need no ENV_LOCK.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::shading::{calculate_shaded_fraction, LocalSolarPosition, Overhang, ShadeFin};
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion_core::ashrae_cases::{Orientation, WindowArea};

/// Engine-attached overhang for a spec's surface of `orientation` (zone 0),
/// plus the spec's own window dims for the same orientation — the provenance
/// pair the gap assertions recompute from.
fn attached_shading(
    case: ASHRAE140Case,
    orientation: Orientation,
) -> (Overhang, Vec<ShadeFin>, WindowArea) {
    let spec = case.spec();
    let model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("spec builds");
    let surfaces = model
        .0
        .solar
        .surfaces
        .first()
        .expect("zone 0 surfaces exist");
    let surface = surfaces
        .iter()
        .find(|s| s.orientation == orientation)
        .expect("surface for orientation exists");
    let overhang = surface.overhang.expect("overhang attached on this case");
    let fins = surface.fins.clone();
    let window = spec
        .windows
        .first()
        .and_then(|ws| ws.iter().find(|w| w.orientation == orientation))
        .copied()
        .expect("spec window dims for orientation");
    (overhang, fins, window)
}

/// The spec-true gap recomputed from the spec's own geometry:
/// overhang mounts at the roof edge (`mounting_height`), the window head at
/// `sill + height` — Figure 5-32 / UM E+ vertices Z=2.700 / CSE ohDistUp.
fn spec_true_gap(mounting_height: f64, sill: f64, height: f64) -> f64 {
    (mounting_height - (sill + height)).max(0.0)
}

/// Engine shadow closed form (transcribed from `calculate_shaded_fraction` /
/// `calculate_overhang_shadow_area`): the shaded strip on the window measured
/// from the window top, `depth·tan(altitude)/cos(rel_az)` minus the gap.
fn closed_form_strip(alt_rad: f64, rel_az_rad: f64, gap: f64, window: &WindowArea) -> f64 {
    if rel_az_rad.abs() >= std::f64::consts::FRAC_PI_2 {
        return 0.0; // sun behind the surface
    }
    let tan_profile = alt_rad.tan() / rel_az_rad.cos();
    if tan_profile <= 0.0 {
        return 0.0;
    }
    let shadow_y = 1.0 * tan_profile;
    ((shadow_y - gap).max(0.0)).min(window.height)
}

/// Engine combined closed form for overhang+fins with the overlap correction
/// (transcribed from `calculate_shaded_fraction`; one fin shades — the one on
/// the sun's side — and the overlap is the fin strip ∩ the overhang strip).
fn closed_form_fraction(
    alt_rad: f64,
    rel_az_rad: f64,
    gap: f64,
    window: &WindowArea,
    fin: Option<&ShadeFin>,
) -> f64 {
    if alt_rad <= 0.0 {
        return 1.0; // below-horizon convention (beam is zero there)
    }
    let strip = closed_form_strip(alt_rad, rel_az_rad, gap, window);
    let oh_area = strip * window.width;
    let oh_height = strip;
    let mut fin_area = 0.0;
    let mut overlap = 0.0;
    if let Some(f) = fin {
        let this_fin_shades = match f.side {
            fluxion::sim::shading::Side::Left => rel_az_rad < 0.0,
            fluxion::sim::shading::Side::Right => rel_az_rad > 0.0,
        };
        if this_fin_shades {
            let shadow_x = f.depth * rel_az_rad.abs().tan();
            let width = (shadow_x - f.distance_from_edge).max(0.0).min(window.width);
            fin_area = width * f.height;
            if oh_area > 0.0 {
                overlap = width * f.height.min(oh_height);
            }
        }
    }
    let combined = oh_area + fin_area - overlap;
    (combined / (window.height * window.width)).clamp(0.0, 1.0)
}

/// Case 610: the attached south overhang carries the spec-true roof-edge gap.
///
/// Red-first (pre-fix tree): `got 0.0, want 0.5` — the round-4 wiring.
#[test]
fn case610_overhang_mounts_at_roof_edge_above_window_head() {
    let (overhang, _fins, window) = attached_shading(ASHRAE140Case::Case610, Orientation::South);
    let expected = spec_true_gap(2.7, window.sill_height, window.height);
    assert_eq!(
        expected, 0.5,
        "Case-600 geometry pins the spec gap at 0.5 m"
    );
    assert!(
        (overhang.distance_above - expected).abs() <= 1e-12,
        "attached gap {} != spec-true {} (Figure 5-32: 0.5 m above the 2.2 m head)",
        overhang.distance_above,
        expected
    );
    assert!((overhang.depth - 1.0).abs() <= 1e-12, "1 m projection");
    assert!(
        overhang.extension >= 8.0,
        "overhang spans the full 8 m south wall (\"infinite\" extension)"
    );
}

/// Case 630: the east/west overhangs carry the same roof-edge gap, and the
/// fins sit at the window edges spanning the full window height
/// (Figures 5-54/5-65). The south surface carries no shading device.
#[test]
fn case630_ew_surfaces_carry_roof_edge_overhang_and_edge_fins() {
    for orientation in [Orientation::East, Orientation::West] {
        let (overhang, fins, window) = attached_shading(ASHRAE140Case::Case630, orientation);
        let expected = spec_true_gap(2.7, window.sill_height, window.height);
        assert_eq!(expected, 0.5, "E/W windows are \"exactly as in Case 600\"");
        assert!(
            (overhang.distance_above - expected).abs() <= 1e-12,
            "{:?} attached gap {} != spec-true {}",
            orientation,
            overhang.distance_above,
            expected
        );
        assert!((overhang.depth - 1.0).abs() <= 1e-12);
        assert_eq!(fins.len(), 2, "one fin per window edge");
        for fin in &fins {
            assert!(
                (fin.distance_from_edge - 0.0).abs() <= 1e-12,
                "fins mount at the window edges (Figure 5-54 plan view)"
            );
            assert!(
                (fin.height - window.height).abs() <= 1e-12,
                "fin spans the full window height (Figure 5-65 isometric; the \
                 head-to-roof strip adds no window shading)"
            );
            assert!((fin.depth - 1.0).abs() <= 1e-12, "1 m fin projection");
        }
    }
    let spec = ASHRAE140Case::Case630.spec();
    let model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("spec builds");
    let south = model
        .0
        .solar
        .surfaces
        .first()
        .expect("zone 0 surfaces")
        .iter()
        .find(|s| s.orientation == Orientation::South)
        .expect("south surface exists");
    assert!(
        south.overhang.is_none() && south.fins.is_empty(),
        "Case 630 shades the east/west windows only (§5.2.2.1.3: devices added \
         to the east and west windows)"
    );
}

/// The shaded fraction closes against the transcribed closed form at the
/// spec geometry, including the two profile-angle thresholds the gap creates:
/// shading begins only when `depth·tan(profile) > gap` (26.565…° at gap 0.5)
/// and saturates at `gap + height` (68.198…°). Red-first (pre-fix tree): the
/// 45°/0° pin got 0.5 (head strip shaded), want 0.25.
#[test]
fn shaded_fraction_matches_spec_geometry_closed_form() {
    let (overhang, fins, window) = attached_shading(ASHRAE140Case::Case610, Orientation::South);
    assert!(fins.is_empty(), "Case 610 is overhang-only");
    let combined_window = WindowArea::with_dimensions(
        window.area,
        Orientation::South,
        window.height,
        window.width,
        window.sill_height,
        window.left_offset,
    );
    let gap = overhang.distance_above;

    // Below horizon: full-shade convention (beam is zero there; the gain
    // entry is inert through `irradiance.total_wm2 <= 0`).
    let below = LocalSolarPosition {
        altitude: -0.05,
        relative_azimuth: 0.0,
    };
    let f = calculate_shaded_fraction(&combined_window, Some(&overhang), &[], &below, None);
    assert!(
        (f - 1.0).abs() <= 1e-12,
        "below-horizon convention, got {}",
        f
    );

    // Sun behind the surface: no shading.
    let behind = LocalSolarPosition {
        altitude: 1.0,
        relative_azimuth: 100.0_f64.to_radians(),
    };
    let f = calculate_shaded_fraction(&combined_window, Some(&overhang), &[], &behind, None);
    assert!((f - 0.0).abs() <= 1e-12, "behind-surface guard, got {}", f);

    // The gap threshold: shadow_y == gap exactly → fraction 0 (red-first
    // pre-fix got 0.25 — the head strip shaded at the spec's onset angle).
    let onset = LocalSolarPosition {
        altitude: 0.5_f64.atan(),
        relative_azimuth: 0.0,
    };
    let f = calculate_shaded_fraction(&combined_window, Some(&overhang), &[], &onset, None);
    assert!(
        (f - 0.0).abs() <= 1e-9,
        "shading must begin at depth·tan(profile) = gap (26.565…°), got {}",
        f
    );

    // Saturation: shadow_y == gap + height → the window fully shaded.
    let saturated = LocalSolarPosition {
        altitude: 2.5_f64.atan(),
        relative_azimuth: 0.0,
    };
    let f = calculate_shaded_fraction(&combined_window, Some(&overhang), &[], &saturated, None);
    assert!(
        (f - 1.0).abs() <= 1e-9,
        "saturation at gap+height, got {}",
        f
    );

    // Midpoint: 45° altitude, due south → strip 0.5 m of the 2.0 m window.
    let mid = LocalSolarPosition {
        altitude: std::f64::consts::FRAC_PI_4,
        relative_azimuth: 0.0,
    };
    let f = calculate_shaded_fraction(&combined_window, Some(&overhang), &[], &mid, None);
    assert!(
        (f - 0.25).abs() <= 1e-9,
        "45°/0°: strip = 1.0 − 0.5 gap over 2.0 m height = 0.25, got {}",
        f
    );

    // Sweep grid vs the transcribed closed form (≤1e-12 everywhere).
    for alt_deg in [5.0_f64, 30.0, 45.0, 60.0, 80.0] {
        for az_deg in [-80.0_f64, -45.0, -15.0, 0.0, 15.0, 45.0, 80.0] {
            let sun = LocalSolarPosition {
                altitude: alt_deg.to_radians(),
                relative_azimuth: az_deg.to_radians(),
            };
            let engine =
                calculate_shaded_fraction(&combined_window, Some(&overhang), &[], &sun, None);
            let strip =
                closed_form_strip(sun.altitude, sun.relative_azimuth, gap, &combined_window);
            let want = (strip * combined_window.width
                / (combined_window.height * combined_window.width))
                .clamp(0.0, 1.0);
            assert!(
                (engine - want).abs() <= 1e-12,
                "sweep {}°/{:+}°: engine {} != closed form {}",
                alt_deg,
                az_deg,
                engine,
                want
            );
        }
    }
}

/// Case 630 west surface: overhang + sun-side fin compose through the overlap
/// correction (one fin shades — the one on the sun's side; the overlap is the
/// fin strip ∩ the overhang strip, bounded by min(fin.height, strip)). At
/// 60°/+30° with the spec gap: 0.7981125… (red-first pre-fix got ~1.0).
#[test]
fn case630_overlap_composition_matches_closed_form() {
    let (overhang, fins, window) = attached_shading(ASHRAE140Case::Case630, Orientation::West);
    assert_eq!(fins.len(), 2);
    let combined_window = WindowArea::with_dimensions(
        window.area,
        Orientation::West,
        window.height,
        window.width,
        window.sill_height,
        window.left_offset,
    );
    let sun = LocalSolarPosition {
        altitude: 60.0_f64.to_radians(),
        relative_azimuth: 30.0_f64.to_radians(),
    };
    let engine = calculate_shaded_fraction(&combined_window, Some(&overhang), &fins, &sun, None);
    let want = closed_form_fraction(
        sun.altitude,
        sun.relative_azimuth,
        overhang.distance_above,
        &combined_window,
        Some(&fins[1]),
    );
    assert!(
        (engine - want).abs() <= 1e-12,
        "engine {} != closed form {}",
        engine,
        want
    );
    assert!(
        (engine - 0.7981125224324686).abs() <= 1e-9,
        "60°/+30° spec-geometry value (Python-replicated), got {}",
        engine
    );
}

/// The gap clamps at zero for a mounting height at or below the window head —
/// a low synthetic device must never attach a phantom negative gap (the
/// `.max(0.0)` guard; no 140 case exercises this — it pins the guard).
#[test]
fn overhang_gap_clamps_at_zero_for_low_mounting_height() {
    let spec = ASHRAE140Case::Case600.spec();
    let mut spec = spec;
    spec.shading = Some(fluxion_core::ashrae_cases::ShadingDevice::overhang(
        1.0, 1.5,
    ));
    let model =
        ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default())
            .expect("spec builds");
    let south = model
        .0
        .solar
        .surfaces
        .first()
        .expect("zone 0 surfaces")
        .iter()
        .find(|s| s.orientation == Orientation::South)
        .expect("south surface exists");
    let overhang = south.overhang.expect("overhang attached");
    assert!(
        (overhang.distance_above - 0.0).abs() <= 1e-12,
        "mounting 1.5 m < head 2.2 m must clamp to gap 0, got {}",
        overhang.distance_above
    );
}
