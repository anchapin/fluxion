//! §7 surface-pool blitz slice (loop round 129) — the **per-orientation
//! transmittance/absorption split upstream of the class fractions**
//! (`docs/implementation-plans/checker-air-node-mirror-plan.md` §7 item (ii)
//! remainder — the interior partition DOWNSTREAM of this seam is pinned by
//! the rounds-91–105 contract family plus the r125 class-drive suite; the
//! window seam itself carried only loose lib spot-checks (`> 0`,
//! `<= max · 1.1`), no bit-exact pin and no Python replication).
//!
//! Component under contract: the window solar-gain seam in `src/sim/solar.rs`:
//!
//! - (a) the ASHRAE Handbook Ch. 15 SHGC incidence-ratio table
//!   (`ashrae_140_window_shgc_ratio`, 10° knots, linear interpolation,
//!   `<= 0° → 1.0` / `>= 90° → 0.0` clamps), consumed as
//!   `beam_shgc = shgc · ratio(θ_i)` with a `θ_i ≥ 90° → 0.0` short-circuit
//!   and a `θ_i ≤ 0° → shgc` raw passthrough;
//! - (b) the composition closed forms at `calculate_window_solar_gain`:
//!   `beam = A · E_b·(1−f_sh) · beam_shgc(θ)`,
//!   `diffuse = A · E_d · shgc·0.9·V_sky` (the loop-r120 sky-diffuse view
//!   factor; `1.0` is the exact no-device identity),
//!   `ground = A · E_g · shgc·0.85` (Issue #1271's ground-vs-sky split), and
//!   `total = (beam + diffuse) + ground` (the `algebraic_add` chain, plain
//!   IEEE `+` in default builds);
//! - (c) the edge routes: `total ≤ 0` night zero (exact `SolarGain::zero()`);
//!   sun below horizon (`is_above_horizon == false` → `incidence_cosine == 0`
//!   → `θ = 90°` → beam exactly `0.0` while diffuse/ground stay live);
//! - (d) the per-orientation split (this item's core): the same hour and
//!   irradiance through all four vertical orientations, each recomposed from
//!   its own incidence angle — north beam exactly `0.0` (sun behind the
//!   surface clamps the cosine to 0), east/west bit-equal, south maximal;
//! - (e) the shaded-beam route: `(1 − f_sh)` applies to the BEAM component
//!   only (the r119/r120 record) — diffuse keeps the full `0.9·V_sky` share
//!   and ground keeps `0.85` (the fins sky-diffuse gap stays loop-r128's
//!   scoping item, `docs/investigations/fins-sky-diffuse-isotropic-scoping.md`,
//!   pending Alex's overhang×fins composition call);
//! - (f) ENGINE-TRUTH zone pool: the Case-600 spec model's zone window gain
//!   at a controlled weather hour (`_dhat_calculate_zone_solar_gain` — the
//!   pool input the per-class fractions partition) recomposed bit-exactly
//!   through the same exported path the per-orientation loop drives: the
//!   model's cached sun position, the combined-orientation window area, the
//!   ground-reflection fallback `0.2` (#4269's `albedo.filter(> 0)` route),
//!   the per-surface accumulation fold, and the #4166 per-orientation
//!   irradiance stash.
//!
//! Scope note (not duplicated here): the irradiance formation itself
//! (`calculate_surface_irradiance`: the DNI/DHI/albedo partition) and the
//! opaque-surface absorption route are §6 / exterior-boundary territory —
//! the r126 sol-air suite pins the opaque side at its own seam.
//!
//! Plan §7 queue status after this slice: item (ii) pinned (this suite);
//! still open: (iv)'s full longwave pool-exchange term (the #2890
//! view-factor network follow-up).
//!
//! Route discipline: no env flags are read or flipped, so no serialization is
//! needed. These are record pins, not validator bands: a violation means the
//! window split moved and must be re-examined, never tuned around.
//!
//! Dev-time Python replication: `/tmp/r129_window_split_replication.py` —
//! mirrors the table, the incidence closed form and the composition
//! operation-for-operation (same f64 op order); every literal asserted below
//! is pasted from its output. The in-test transcription recomposes θ through
//! the engine's own `incidence_cosine`, so bit-exactness is structural on the
//! Rust side; the Python literals are the independent cross-language anchor.

use fluxion::batch_oracle::BatchOracle;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::shading::{calculate_shaded_fraction, LocalSolarPosition, Overhang};
use fluxion::sim::solar::{
    calculate_hourly_solar_from_pos, calculate_window_solar_gain, SolarGain, SolarPosition,
    SurfaceIrradiance, WindowProperties,
};
use fluxion::weather::HourlyWeatherData;
use fluxion_core::ashrae_cases::{Orientation, WindowArea};

const DT: f64 = 3600.0;

/// The Case-600 double-clear window SHGC (`WindowProperties::double_clear`,
/// ASHRAE 140 Table B1-5 corrected value, #741).
const SHGC_600: f64 = 0.787;

/// The engine's ratio table + interpolation, transcribed
/// operation-for-operation (`ashrae_140_window_shgc_ratio`, HF Ch. 15).
fn ratio(angle_deg: f64) -> f64 {
    const ANGLES: [f64; 10] = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
    const RATIOS: [f64; 10] = [
        1.000, 0.995, 0.985, 0.970, 0.940, 0.890, 0.810, 0.680, 0.450, 0.000,
    ];
    if angle_deg <= 0.0 {
        return 1.0;
    }
    if angle_deg >= 90.0 {
        return 0.0;
    }
    for i in 0..ANGLES.len() - 1 {
        if angle_deg >= ANGLES[i] && angle_deg <= ANGLES[i + 1] {
            let t = (angle_deg - ANGLES[i]) / (ANGLES[i + 1] - ANGLES[i]);
            return RATIOS[i] * (1.0 - t) + RATIOS[i + 1] * t;
        }
    }
    1.0
}

/// The seam's beam-SHGC route: `θ ≤ 0 → shgc` raw, `θ ≥ 90 → 0.0`, else
/// `shgc · ratio(θ)`.
fn beam_shgc(shgc: f64, theta_deg: f64) -> f64 {
    if theta_deg <= 0.0 {
        shgc
    } else if theta_deg >= 90.0 {
        0.0
    } else {
        shgc * ratio(theta_deg)
    }
}

/// The seam's composition closed forms, operand order identical to
/// `calculate_window_solar_gain` (default-build IEEE `+`; the
/// `algebraic_add` chain resolves to the plain operator).
fn compose(
    area: f64,
    irr: &SurfaceIrradiance,
    shgc: f64,
    theta_deg: f64,
    shaded: f64,
    vf: f64,
) -> SolarGain {
    let b = beam_shgc(shgc, theta_deg);
    let diffuse_shgc = shgc * 0.9 * vf;
    let ground_shgc = shgc * 0.85;
    let effective_beam = irr.beam_wm2 * (1.0 - shaded);
    SolarGain::new(
        area * effective_beam * b,
        area * irr.diffuse_wm2 * diffuse_shgc,
        area * irr.ground_reflected_wm2 * ground_shgc,
    )
}

/// Incidence angle for sun due south over the named vertical orientation —
/// recomposed through the ENGINE's own `incidence_cosine`, so θ is
/// bit-identical to the seam's by construction.
fn theta_deg(sun: &SolarPosition, orientation: Orientation) -> f64 {
    let (tilt, azimuth) = match orientation {
        Orientation::South => (90.0, 180.0),
        Orientation::North => (90.0, 0.0),
        Orientation::East => (90.0, 90.0),
        Orientation::West => (90.0, 270.0),
        _ => unreachable!("suite only drives the four vertical faces"),
    };
    sun.incidence_cosine(tilt, azimuth).acos().to_degrees()
}

/// Sun due south (azimuth 180°) at the given altitude.
fn sun_south(altitude_deg: f64) -> SolarPosition {
    SolarPosition {
        altitude_deg,
        azimuth_deg: 180.0,
        zenith_deg: 90.0 - altitude_deg,
    }
}

#[test]
/// Contract 1 — the ratio table's knots, interpolation and clamps through the
/// seam. Beam-only irradiance isolates `beam_gain`; every scanned altitude
/// (θ ≈ 5°…85° nominal) recomposes bit-exactly; the Up face at zenith pins
/// the `θ ≤ 0` raw-SHGC branch; altitude 0 pins the below-horizon `θ = 90`
/// route at exactly `0.0`.
fn ratio_table_knots_interpolation_and_clamps_pin_bit_exact() {
    let window = WindowProperties::double_clear(12.0);
    let beam_only = SurfaceIrradiance::new(800.0, 0.0, 0.0);
    for altitude in [
        85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0, 45.0, 40.0, 35.0, 30.0, 25.0, 20.0, 15.0,
        10.0, 5.0,
    ] {
        let sun = sun_south(altitude);
        let gain = calculate_window_solar_gain(
            &beam_only,
            &window,
            None,
            None,
            &[],
            &sun,
            Orientation::South,
            1.0,
        );
        let expected = compose(
            12.0,
            &beam_only,
            SHGC_600,
            theta_deg(&sun, Orientation::South),
            0.0,
            1.0,
        );
        assert_eq!(
            gain.beam_gain_w, expected.beam_gain_w,
            "altitude {altitude}: engine beam gain must equal the transcribed composition"
        );
        assert_eq!(gain.diffuse_gain_w, 0.0, "altitude {altitude}");
        assert_eq!(gain.ground_reflected_gain_w, 0.0, "altitude {altitude}");
    }

    // Python anchors (r129 replication, south/beam-only): alt 60 → 6119.712
    // (ratio knot 0.81 after rounding), alt 30 → 7328.544 (knot 0.97),
    // alt 5 → 7536.312 (midpoint 0.9975).
    for (altitude, expected_beam) in [(60.0, 6119.712), (30.0, 7328.544), (5.0, 7536.312)] {
        let sun = sun_south(altitude);
        let gain = calculate_window_solar_gain(
            &beam_only,
            &window,
            None,
            None,
            &[],
            &sun,
            Orientation::South,
            1.0,
        );
        assert_eq!(
            gain.beam_gain_w, expected_beam,
            "python anchor alt {altitude}"
        );
    }

    // `θ ≤ 0` branch: the Up face with the sun at zenith reduces
    // `incidence_cosine` to exactly 1.0 → θ = 0 → the RAW shgc (no table
    // multiply). Python anchor: 7555.200000000001.
    let up_sun = SolarPosition {
        altitude_deg: 90.0,
        azimuth_deg: 180.0,
        zenith_deg: 0.0,
    };
    assert_eq!(up_sun.incidence_cosine(0.0, 0.0), 1.0);
    let up_gain = calculate_window_solar_gain(
        &beam_only,
        &window,
        None,
        None,
        &[],
        &up_sun,
        Orientation::Up,
        1.0,
    );
    assert_eq!(up_gain.beam_gain_w, 7555.200000000001);

    // Below-horizon sun: `incidence_cosine` returns 0.0 (is_above_horizon is
    // false) → θ = 90 → beam exactly 0.0.
    let horizon_sun = sun_south(0.0);
    assert_eq!(horizon_sun.incidence_cosine(90.0, 180.0), 0.0);
    let horizon_gain = calculate_window_solar_gain(
        &beam_only,
        &window,
        None,
        None,
        &[],
        &horizon_sun,
        Orientation::South,
        1.0,
    );
    assert_eq!(horizon_gain.beam_gain_w, 0.0);
}

#[test]
/// Contract 2 — the full composition closed forms at the alt-60 fixture
/// (irradiance 800/100/20, unshaded, no device): every SolarGain field and
/// the `algebraic_add` total chain pinned bit-exactly against the
/// transcription AND the Python literals (6119.712 / 849.96 / 160.548 /
/// 7130.22).
fn composition_closed_forms_pin_bit_exact() {
    let window = WindowProperties::double_clear(12.0);
    let irr = SurfaceIrradiance::new(800.0, 100.0, 20.0);
    let sun = sun_south(60.0);
    let gain = calculate_window_solar_gain(
        &irr,
        &window,
        None,
        None,
        &[],
        &sun,
        Orientation::South,
        1.0,
    );
    let expected = compose(
        12.0,
        &irr,
        SHGC_600,
        theta_deg(&sun, Orientation::South),
        0.0,
        1.0,
    );
    assert_eq!(gain.beam_gain_w, expected.beam_gain_w);
    assert_eq!(gain.diffuse_gain_w, expected.diffuse_gain_w);
    assert_eq!(
        gain.ground_reflected_gain_w,
        expected.ground_reflected_gain_w
    );
    assert_eq!(gain.total_gain_w, expected.total_gain_w);
    // The total chain: (beam + diffuse) + ground, re-associated by hand to
    // prove the fold order (algebraic_add = plain `+` in default builds).
    assert_eq!(
        gain.total_gain_w,
        (gain.beam_gain_w + gain.diffuse_gain_w) + gain.ground_reflected_gain_w
    );
    // Python anchors (r129 replication).
    assert_eq!(gain.beam_gain_w, 6119.712);
    assert_eq!(gain.diffuse_gain_w, 849.96);
    assert_eq!(gain.ground_reflected_gain_w, 160.548);
    assert_eq!(gain.total_gain_w, 7130.22);
}

#[test]
/// Contract 3 — the diffuse (×0.9 × V_sky) and ground (×0.85) factors
/// discriminated with the beam zeroed: the r120 view factor scales ONLY the
/// sky-diffuse share (0.5 is a power of two, so the halving is exact), ground
/// is bit-identical across view factors (its rays never intersect a
/// horizontal device), and the zeroed beam stays exactly 0.0.
fn diffuse_ground_factors_discriminated_pin_bit_exact() {
    let window = WindowProperties::double_clear(12.0);
    let irr_no_beam = SurfaceIrradiance::new(0.0, 100.0, 20.0);
    let sun = sun_south(60.0);

    let gain_vf1 = calculate_window_solar_gain(
        &irr_no_beam,
        &window,
        None,
        None,
        &[],
        &sun,
        Orientation::South,
        1.0,
    );
    assert_eq!(gain_vf1.beam_gain_w, 0.0);
    assert_eq!(gain_vf1.diffuse_gain_w, 849.96);
    assert_eq!(gain_vf1.ground_reflected_gain_w, 160.548);

    let gain_vf_half = calculate_window_solar_gain(
        &irr_no_beam,
        &window,
        None,
        None,
        &[],
        &sun,
        Orientation::South,
        0.5,
    );
    // Exact halving (0.5 scaling is an IEEE-exact operation).
    assert_eq!(gain_vf_half.diffuse_gain_w, 424.98);
    assert_eq!(gain_vf_half.diffuse_gain_w, gain_vf1.diffuse_gain_w / 2.0);
    // Ground does not consume the view factor (loop-r120 record).
    assert_eq!(
        gain_vf_half.ground_reflected_gain_w, gain_vf1.ground_reflected_gain_w,
        "ground share must be identical across sky view factors"
    );
    assert_eq!(gain_vf_half.ground_reflected_gain_w, 160.548);

    // Beam kept live discriminates that the view factor does not touch it:
    // beam gain is bit-identical across the two view factors.
    let irr_full = SurfaceIrradiance::new(800.0, 100.0, 20.0);
    let beam_vf1 = calculate_window_solar_gain(
        &irr_full,
        &window,
        None,
        None,
        &[],
        &sun,
        Orientation::South,
        1.0,
    );
    let beam_vf_half = calculate_window_solar_gain(
        &irr_full,
        &window,
        None,
        None,
        &[],
        &sun,
        Orientation::South,
        0.5,
    );
    assert_eq!(beam_vf1.beam_gain_w, beam_vf_half.beam_gain_w);
    assert_eq!(beam_vf1.beam_gain_w, 6119.712);
}

#[test]
/// Contract 4 — the edge routes: exact-zero irradiance takes the `total ≤ 0`
/// early return (every field exactly 0.0), and a below-horizon sun with live
/// diffuse/ground irradiance pins the `θ = 90` route — beam exactly `0.0`
/// while the diffuse/ground shares compose normally (Python anchors 849.96 /
/// 160.548 / 1010.508).
fn night_and_below_horizon_routes_pin_bit_exact() {
    let window = WindowProperties::double_clear(12.0);
    let day_sun = sun_south(60.0);

    let night = calculate_window_solar_gain(
        &SurfaceIrradiance::zero(),
        &window,
        None,
        None,
        &[],
        &day_sun,
        Orientation::South,
        1.0,
    );
    assert_eq!(night.beam_gain_w, 0.0);
    assert_eq!(night.diffuse_gain_w, 0.0);
    assert_eq!(night.ground_reflected_gain_w, 0.0);
    assert_eq!(night.total_gain_w, 0.0);

    let below = SolarPosition {
        altitude_deg: -10.0,
        azimuth_deg: 180.0,
        zenith_deg: 100.0,
    };
    assert_eq!(below.incidence_cosine(90.0, 180.0), 0.0);
    let irr = SurfaceIrradiance::new(0.0, 100.0, 20.0);
    let gain = calculate_window_solar_gain(
        &irr,
        &window,
        None,
        None,
        &[],
        &below,
        Orientation::South,
        1.0,
    );
    assert_eq!(gain.beam_gain_w, 0.0);
    assert_eq!(gain.diffuse_gain_w, 849.96);
    assert_eq!(gain.ground_reflected_gain_w, 160.548);
    assert_eq!(gain.total_gain_w, 1010.508);
}

#[test]
/// Contract 5 — the per-orientation split (item (ii)'s core): the same sun
/// (alt 40°, due south) and the same irradiance through all four vertical
/// faces, each recomposed from its own incidence chain. South carries the
/// beam (Python anchor 7101.888 at ratio knot 0.94); north/east/west beam is
/// EXACTLY `0.0` (sun behind the surface clamps the cosine to 0 → θ = 90);
/// east and west are bit-equal; diffuse/ground shares are
/// orientation-independent by construction (flat factors).
fn per_orientation_split_pin_bit_exact() {
    let window = WindowProperties::double_clear(12.0);
    let irr = SurfaceIrradiance::new(800.0, 100.0, 20.0);
    let sun = sun_south(40.0);

    let mut gains = Vec::new();
    for orientation in [
        Orientation::North,
        Orientation::East,
        Orientation::South,
        Orientation::West,
    ] {
        let gain =
            calculate_window_solar_gain(&irr, &window, None, None, &[], &sun, orientation, 1.0);
        let expected = compose(12.0, &irr, SHGC_600, theta_deg(&sun, orientation), 0.0, 1.0);
        assert_eq!(gain.beam_gain_w, expected.beam_gain_w, "{orientation:?}");
        assert_eq!(
            gain.diffuse_gain_w, expected.diffuse_gain_w,
            "{orientation:?}"
        );
        assert_eq!(
            gain.ground_reflected_gain_w, expected.ground_reflected_gain_w,
            "{orientation:?}"
        );
        gains.push((orientation, gain));
    }

    let by_ori: std::collections::HashMap<_, _> = gains.iter().map(|(o, g)| (*o, *g)).collect();
    let south = by_ori[&Orientation::South];
    let north = by_ori[&Orientation::North];
    let east = by_ori[&Orientation::East];
    let west = by_ori[&Orientation::West];

    assert_eq!(south.beam_gain_w, 7101.888, "python anchor, south beam");
    assert_eq!(north.beam_gain_w, 0.0, "sun behind the north face");
    assert_eq!(east.beam_gain_w, 0.0);
    assert_eq!(west.beam_gain_w, 0.0);
    assert_eq!(
        east.beam_gain_w, west.beam_gain_w,
        "east/west symmetric geometry must be bit-equal"
    );
    // Diffuse/ground are orientation-independent flat-factor shares here.
    for (orientation, gain) in &gains {
        assert_eq!(gain.diffuse_gain_w, 849.96, "{orientation:?}");
        assert_eq!(gain.ground_reflected_gain_w, 160.548, "{orientation:?}");
    }
}

#[test]
/// Contract 6 — the shaded-beam route: with the r119 Case-610 overhang
/// mounting (depth 1.0 m, 0.5 m above the head) and a 60°-elevation due-south
/// sun, the recomposed shaded fraction reduces ONLY the beam
/// (`A · E_b·(1−f_sh) · beam_shgc`); diffuse keeps the FULL `0.9·V_sky`
/// share (849.96) and ground keeps `0.85` (160.548) — the honest r120 record
/// (beam-side-only shading; the fins/overhang diffuse-composition question
/// stays loop-r128's scoping item for Alex).
fn shaded_beam_route_pin_bit_exact() {
    let window = WindowProperties::double_clear(12.0);
    let irr = SurfaceIrradiance::new(800.0, 100.0, 20.0);
    let sun = sun_south(60.0);

    let geometry = WindowArea {
        area: 12.0,
        orientation: Orientation::South,
        height: 2.0,
        width: 6.0,
        sill_height: 0.2,
        left_offset: 0.0,
    };
    let overhang = Overhang {
        depth: 1.0,
        distance_above: 0.5,
        extension: 1.0,
    };
    // The seam's LocalSolarPosition formation: relative azimuth
    // (180 − 180 wrapped) = 0, altitude in radians.
    let local_solar = LocalSolarPosition {
        altitude: sun.altitude_deg.to_radians(),
        relative_azimuth: 0.0,
    };
    let shaded_fraction =
        calculate_shaded_fraction(&geometry, Some(&overhang), &[], &local_solar, None);
    assert!(
        shaded_fraction > 0.0 && shaded_fraction < 1.0,
        "premise: the fixture must partially shade (got {shaded_fraction})"
    );

    let gain = calculate_window_solar_gain(
        &irr,
        &window,
        Some(&geometry),
        Some(&overhang),
        &[],
        &sun,
        Orientation::South,
        1.0,
    );

    let expected = compose(
        12.0,
        &irr,
        SHGC_600,
        theta_deg(&sun, Orientation::South),
        shaded_fraction,
        1.0,
    );
    assert_eq!(gain.beam_gain_w, expected.beam_gain_w);
    assert_eq!(gain.diffuse_gain_w, expected.diffuse_gain_w);
    assert_eq!(
        gain.ground_reflected_gain_w,
        expected.ground_reflected_gain_w
    );

    // The shading reaches the beam only: diffuse/ground keep their unshaded
    // values bit-exactly.
    assert_eq!(gain.diffuse_gain_w, 849.96, "diffuse must stay unshaded");
    assert_eq!(
        gain.ground_reflected_gain_w, 160.548,
        "ground must stay unshaded"
    );
    assert!(
        gain.beam_gain_w < 6119.712,
        "partial shade must reduce the beam below its unshaded 6119.712 W"
    );
    // And the reduction is exactly the (1 − f_sh) factor on the beam leg.
    let unshaded_beam = 12.0 * 800.0 * beam_shgc(SHGC_600, theta_deg(&sun, Orientation::South));
    assert_eq!(
        gain.beam_gain_w,
        12.0 * (800.0 * (1.0 - shaded_fraction))
            * beam_shgc(SHGC_600, theta_deg(&sun, Orientation::South))
    );
    assert!(unshaded_beam > gain.beam_gain_w);
}

#[test]
/// Contract 7 — ENGINE-TRUTH zone pool: the Case-600 spec model's zone
/// window gain at a controlled weather hour (timestep 12 → Jan 1, sun at
/// 12.5 h mid-hour) recomposed bit-exactly through the same exported path the
/// per-orientation loop drives: the model's cached sun position, the
/// combined-orientation window area, the ground-reflection fallback 0.2
/// (#4269 route), and the per-surface accumulation fold; the #4166
/// per-orientation irradiance stash pinned against the recomposed
/// irradiance. The opaque component of the tuple is the r126 sol-air seam's
/// scope and is not asserted here.
fn zone_pool_engine_truth_pin_bit_exact() {
    let mut model = BatchOracle::default_model().expect("spec model builds");
    let window_props = model.solar.window_properties[0];

    // Premise: zone 0 carries windows on the south orientation only (the
    // Case-600 geometry: two 3 m × 2 m south windows, 12 m² total).
    let surfaces = model.solar.surfaces.get(0).expect("zone 0 surfaces");
    let mut combined_area = 0.0;
    let mut south_surface_count = 0usize;
    for surface in surfaces {
        if surface.orientation == Orientation::South && surface.window_area > 0.0 {
            south_surface_count += 1;
            combined_area += surface.window_area;
        }
        assert!(
            surface.orientation == Orientation::South || surface.window_area == 0.0,
            "premise: only south surfaces carry windows in Case 600"
        );
    }
    assert!(south_surface_count >= 1, "premise: south windows present");
    assert_eq!(combined_area, 12.0, "Case-600 south window area");
    // The spec path carries `WindowSpec::double_clear_glass()` — SHGC 0.77
    // (U 2.10, τ_normal 0.703), the Case-600 declared value — while the
    // analytic-fixture constructor `WindowProperties::double_clear()` carries
    // the #741-corrected 0.787. This arm drives the SPEC path, so the seam's
    // shgc input is the spec's 0.77.
    assert_eq!(window_props.shgc, 0.77, "Case-600 spec window SHGC");

    // Controlled weather hour: dni 500 / dhi 150 / ghi 600, albedo unset so
    // the #4269 fallback (0.2) drives ground reflection.
    let weather = HourlyWeatherData::new(12.0, 500.0, 150.0, 600.0, 4.0, 50.0, 12);
    assert!(
        weather.albedo.map_or(true, |a| a <= 0.0),
        "premise: unset/zero albedo must route the 0.2 fallback"
    );

    let (window_gain, _opaque_gain) = model._dhat_calculate_zone_solar_gain(0, 12, &weather, DT);

    // Recompose through the same exported path: the engine's cached sun
    // position (timestep_to_date(12) = Jan 1 hour 12, mid-hour +0.5), the
    // combined-orientation window properties, no device, view factor 1.0.
    let sun = model.cached_solar_position(12, 2024, 1, 1, 12.5);
    let oriented = WindowProperties {
        area: combined_area,
        ..window_props
    };
    let (irradiance, solar_gain) = calculate_hourly_solar_from_pos(
        &sun,
        2024,
        1,
        1,
        500.0,
        150.0,
        &oriented,
        None,
        None,
        &[],
        Orientation::South,
        Some(0.2),
        1.0,
    );
    assert!(
        irradiance.total_wm2 > 0.0,
        "premise: the controlled hour must be daytime"
    );

    // The per-surface fold: each south surface with windows adds the SAME
    // combined-area gain (mirrored as a loop for bit-exactness).
    let mut expected = 0.0;
    for _ in 0..south_surface_count {
        expected += solar_gain.total_gain_w;
    }
    assert_eq!(
        window_gain, expected,
        "zone pool window component must equal the recomposed per-orientation fold"
    );

    // The #4166 stash: the seam's published per-orientation irradiance.
    let stash_bd = model.solar.orientation_irradiance_beam_diffuse[0][Orientation::South as usize];
    let stash_ground = model.solar.orientation_irradiance_ground[0][Orientation::South as usize];
    assert_eq!(stash_bd, irradiance.beam_wm2 + irradiance.diffuse_wm2);
    assert_eq!(stash_ground, irradiance.ground_reflected_wm2);

    // Non-vacuous premise: the pool actually carries the window gain.
    assert!(window_gain > 0.0);
    // Silence the unused-field lint for the documented-out-of-scope component.
    let _ = _opaque_gain;
}
