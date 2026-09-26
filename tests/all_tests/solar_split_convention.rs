//! Solar split convention specification — Issue #3961.
//!
//! Issue #3961 asked whether the fixed 0.30/0.30 solar split
//! (`solar_distribution_to_air` / `solar_beam_to_mass_fraction`) could be
//! retired to the referee convention (0.0 / 1.0) by defaults alone. Measured
//! against the strict ±15% annual-energy gate, the answer is NO:
//!
//! - `to_air = 0.0` (low-mass): Case 600FF peak T_air moves AWAY from the
//!   reference band (47.29 → 46.01 °C vs 64.9–75.1 °C); Case 600/800
//!   cooling regress ~20 pp each.
//! - `beam_to_mass = 1.0`: high-mass cooling collapses (Case 960:
//!   0.144 → 0.008 MWh) — the capacitance-weighted mass pool swallows
//!   solar into ground-coupled storage.
//! - The 0.30 low-mass air fraction is load-bearing compensation for the
//!   missing air-node capacitance / beam-incident absorption geometry
//!   (#1152-class structural work). See
//!   docs/KNOWN_ISSUES.md §LIMIT-30 and the experiment record in
//!   `thermal_model_core::from_spec_with_selector`.
//!
//! The tests here pin that load-bearing calibration (so it cannot drift
//! silently), pin the fraction *mechanism* invariants that any future
//! structural rewrite must preserve (exact telemetry closure
//! Σ_i solar_absorbed_w + Φ_sol·to_air = Φ_sol, and the legacy partition),
//! and document the `ZoneBoundaryConditions` default semantics.

use fluxion::physics::gauge_zone_solver::{
    GaugeZoneSolver, MultiZoneGaugeSolver, SurfaceType, ZoneBoundaryConditions,
};
use fluxion::physics::units::{FromF64, HeatTransferCoefficient, Temperature};
use fluxion::physics::wall_spec::{LayerSpec, WallSpec};
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

/// Case-600 envelope wall used by the gauge-solver harness tests.
fn envelope_wall() -> WallSpec {
    WallSpec::multi_layer(
        "case600 envelope",
        vec![
            LayerSpec {
                name: "exterior finish".to_string(),
                thickness: 0.009,
                conductivity: 0.514,
                density: 1246.0,
                specific_heat: 2090.0,
            },
            LayerSpec {
                name: "limestone".to_string(),
                thickness: 0.1,
                conductivity: 0.51,
                density: 2133.0,
                specific_heat: 795.0,
            },
            LayerSpec {
                name: "insulation".to_string(),
                thickness: 0.0616,
                conductivity: 0.049,
                density: 43.0,
                specific_heat: 1210.0,
            },
            LayerSpec {
                name: "gypsum".to_string(),
                thickness: 0.012,
                conductivity: 0.16,
                density: 950.0,
                specific_heat: 840.0,
            },
        ],
    )
}

/// Window-transmitted solar at the Case 600FF diagnostic peak: the exact
/// window-area × incident-irradiance product (8,437 W) reported as a
/// floor-area-normalized zone scalar for a 48 m² floor (issue #3961).
const PHI_SOL_PEAK_W: f64 = 8437.0;
const FLOOR_AREA_M2: f64 = 48.0;
const SOLAR_WM2: f64 = PHI_SOL_PEAK_W / FLOOR_AREA_M2; // 175.77 W/m² of floor

fn total_absorbed(telemetry: &[fluxion::physics::gauge_zone_solver::SurfaceTelemetry]) -> f64 {
    telemetry.iter().map(|s| s.solar_absorbed_w).sum()
}

/// Issue #3961: the massiveness-blended solar defaults are load-bearing
/// calibration, not arbitrary constants. Any change to these endpoints
/// WITHOUT the structural air-node capacitance / beam-incident absorption
/// work regresses the strict ±15% annual-energy gate (measured 2026-09;
/// see module docs). This pin makes such a change trip a test that points
/// at the experiment record.
#[test]
fn solar_split_defaults_are_load_bearing_issue_3961() {
    let cases = [
        ("Case600", 0.30, ASHRAE140Case::Case600.spec()),
        ("Case900", 0.0, ASHRAE140Case::Case900.spec()),
    ];
    for (case, expected_to_air, spec) in cases {
        let model = ThermalModel::<fluxion::physics::cta::VectorField>::from_spec_with_selector(
            &spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");

        assert_eq!(
            model.solar.solar_distribution_to_air, expected_to_air,
            "{case}: solar_distribution_to_air is massiveness-blended calibration; \
             see issue #3961 experiment record before changing",
        );
        assert_eq!(
            model.solar.solar_beam_to_mass_fraction, 0.30,
            "{case}: solar_beam_to_mass_fraction is load-bearing calibration; \
             see issue #3961 experiment record before changing",
        );
    }
}

/// `ZoneBoundaryConditions::default()` semantics: zero solar directly to
/// air, and a surface-pool-dominant remainder split
/// (`beam_to_mass = 0.0` ⇒ `st_sol_frac = 1.0`). Production dispatchers
/// always override both fields from the model's load-bearing values; these
/// defaults only apply to direct API callers. Pinned so a silent change to
/// the default split cannot go unnoticed (issue #3961).
#[test]
fn gauge_bc_default_semantics_are_pinned() {
    let bc = ZoneBoundaryConditions::default();
    assert_eq!(bc.solar_distribution_to_air, 0.0);
    assert_eq!(bc.solar_beam_to_mass_fraction, 0.0);
}

/// Mechanism invariant that any future structural rewrite must preserve
/// (issue #3961, multi-zone coupled path): with the convention fractions
/// (to_air = 0, beam = 1) the ENTIRE exact window-area ×
/// incident-irradiance product is absorbed by the interior-surface network,
/// and the telemetry closure Σ_i solar_absorbed_w + Φ_sol·to_air = Φ_sol
/// holds exactly.
#[test]
fn gauge_coupled_path_absorbs_full_window_solar_under_convention() {
    let mut mz = MultiZoneGaugeSolver::new();
    mz.add_zone(0, FLOOR_AREA_M2, 2.7);
    mz.add_opaque_surface_to_zone(0, &envelope_wall(), 12.0, SurfaceType::Wall, 0.0, 90.0)
        .expect("surface 1");
    mz.add_opaque_surface_to_zone(0, &envelope_wall(), 12.0, SurfaceType::Wall, 180.0, 90.0)
        .expect("surface 2");
    mz.add_opaque_surface_to_zone(0, &envelope_wall(), 12.0, SurfaceType::Wall, 90.0, 90.0)
        .expect("surface 3");
    mz.add_opaque_surface_to_zone(0, &envelope_wall(), 12.0, SurfaceType::Wall, 270.0, 90.0)
        .expect("surface 4");
    mz.initialize();

    let bc = ZoneBoundaryConditions {
        solar_irradiance_wm2: SOLAR_WM2,
        solar_distribution_to_air: 0.0,
        solar_beam_to_mass_fraction: 1.0,
        ..ZoneBoundaryConditions::default()
    };
    let mut zones = std::collections::HashMap::new();
    zones.insert(0usize, bc);
    mz.step(3600.0, &zones).expect("step succeeds");

    let absorbed = total_absorbed(&mz.get_zone(0).unwrap().per_surface_telemetry());
    assert!(
        (absorbed - PHI_SOL_PEAK_W).abs() < 1e-6 * PHI_SOL_PEAK_W,
        "convention split must absorb the full window product: got {absorbed} W, \
         want {PHI_SOL_PEAK_W} W",
    );
}

/// Mechanism invariant, single-zone primary path (issue #3961): convention
/// fractions absorb the full window product through `step` as well.
#[test]
fn gauge_primary_path_absorbs_full_window_solar_under_convention() {
    let mut zone = GaugeZoneSolver::new(FLOOR_AREA_M2, 2.7);
    zone.add_opaque_surface(&envelope_wall(), 12.0, SurfaceType::Wall, 0.0, 90.0)
        .expect("surface 1");
    zone.add_opaque_surface(&envelope_wall(), 12.0, SurfaceType::Wall, 180.0, 90.0)
        .expect("surface 2");
    zone.add_opaque_surface(&envelope_wall(), 12.0, SurfaceType::Wall, 90.0, 90.0)
        .expect("surface 3");
    zone.add_opaque_surface(&envelope_wall(), 12.0, SurfaceType::Wall, 270.0, 90.0)
        .expect("surface 4");
    zone.initialize();

    zone.step(
        0,
        3600.0,
        Temperature::from_value(20.0),
        HeatTransferCoefficient::from_value(18.3),
        SOLAR_WM2,
        0.0,   // to_air — ASHRAE 140 §5.2.2 convention
        0.0,   // Q_internal
        0.0,   // Q_infiltration
        5.0,   // t_sky
        5.5,   // h_rad_sky (retained-but-ignored param)
        0.0,   // ventilation_ach
        0.0,   // h_tr_3 (retained-but-ignored)
        0.0,   // cm (retained-but-ignored)
        814.0, // h_tr_is — Case 600 zone star-node conductance
        0.0,   // term_rest_1 (retained-but-ignored)
        0.7,   // convective_fraction
        1.0,   // beam_to_mass — ASHRAE 140 §5.2.2 convention
    )
    .expect("step succeeds");

    let absorbed = total_absorbed(&zone.per_surface_telemetry());
    assert!(
        (absorbed - PHI_SOL_PEAK_W).abs() < 1e-6 * PHI_SOL_PEAK_W,
        "primary path must absorb the full window product: got {absorbed} W, \
         want {PHI_SOL_PEAK_W} W",
    );
}

/// Mechanism preservation (issue #3961): the fraction plumbing stays as
/// explicit configuration — legacy 0.30/0.30 values still partition Φ_sol
/// as 30% direct-to-air + 70% surface/mass pools, with the same exact
/// telemetry closure.
#[test]
fn gauge_split_mechanism_fractions_still_partition() {
    let mut zone = GaugeZoneSolver::new(FLOOR_AREA_M2, 2.7);
    zone.add_opaque_surface(&envelope_wall(), 12.0, SurfaceType::Wall, 0.0, 90.0)
        .expect("surface 1");
    zone.add_opaque_surface(&envelope_wall(), 12.0, SurfaceType::Wall, 180.0, 90.0)
        .expect("surface 2");
    zone.add_opaque_surface(&envelope_wall(), 12.0, SurfaceType::Wall, 90.0, 90.0)
        .expect("surface 3");
    zone.add_opaque_surface(&envelope_wall(), 12.0, SurfaceType::Wall, 270.0, 90.0)
        .expect("surface 4");
    zone.initialize();

    zone.step(
        0,
        3600.0,
        Temperature::from_value(20.0),
        HeatTransferCoefficient::from_value(18.3),
        SOLAR_WM2,
        0.3, // to_air — legacy fraction (mechanism check only)
        0.0,
        0.0,
        5.0,
        5.5,
        0.0,
        0.0,
        0.0,
        814.0,
        0.0,
        0.7,
        0.3, // beam_to_mass — legacy fraction (mechanism check only)
    )
    .expect("step succeeds");

    let absorbed = total_absorbed(&zone.per_surface_telemetry());
    let expected = (1.0 - 0.3) * PHI_SOL_PEAK_W;
    assert!(
        (absorbed - expected).abs() < 1e-6 * expected,
        "legacy fractions must partition 30% to air / 70% to surfaces: got \
         {absorbed} W absorbed, want {expected} W",
    );
}
