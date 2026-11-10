//! Air-node record ROUTE-GATE contracts (loop round 116) — the checker
//! air-node-mirror plan §5's remaining out-of-scope item after increment F:
//! "the R74/R75/R90-filtered archaeology routes (they fall through to the
//! committed-air convention, same as today)".
//!
//! Component under contract: the engine's record-enable gate
//! (`step_5r1c.rs`, dated round-105 comment)
//!
//!   r77_air_record = r77_coupled && !r74_ramp_exact && r90_no_lag_filter
//!
//! and the three mirror consumers' documented committed-air fall-through on
//! the excluded routes. The gate's own comment is the scoping primary source:
//! R74 (exact-exponential ramped forcing — different affine form) and the R90
//! filtered lag route (lag STATE feeds the num within the hour — the
//! state-free response recurrence cannot carry it) are EXCLUDED so they keep
//! the fall-through; "R75 vent-ramp (load-independent constant) and
//! R76-no-refresh (on-branch `M`) are form-invariant and need no exclusion".
//! The round-116 scoping pass therefore PRECISIONS the plan's parenthetical:
//! the committed-air fall-through convention is R74's and R90-filtered's
//! alone; R75 stays record-carrying (the interpolation is the coupled route's
//! own default — `r77_coupled || r75_vent_ramp` — so the flag is inert where
//! the record lives), and these contracts pin both truths.
//!
//! Exclusion necessity (Python replication `/tmp/r116_replication.py`,
//! dev-time evidence, ALL CLAIMS OK): feeding the default-form slots into the
//! response recurrence while the engine runs the R74 exact-exponential
//! mis-mirrors the ramped-hour response by a 0.80 relative gap (zone 0); the
//! R90-filtered lag state mis-mirrors by 0.031. The routes are genuinely
//! engine-differentiating, not merely unmirrorable: the R74 site integrates
//! the ramped forcing from the HOUR-START frozen num (`num_tm_ref +
//! num_rest_ref`) with exact f0→f1 endpoint sampling, while the default
//! coupled route refreshes the num per sub-step with the mutating mass and
//! samples at the midpoint — measured in-contract below, the R74 seed hour
//! is NOT engine-bit-identical to the default run (the round-74 comment's
//! "seed steps stay bit-identical" predates the round-76 refresh). The vent
//! term is probe-independent and additive: the seeded composition closes to
//! 5.3e-15 (hour 1) / 6.4e-14 (hour 2) on the twin with the interpolated
//! vent live, and the recorded slots are identical across vent shapes.
//!
//! Fixture shapes: the round-103/107/108 Case600 + legacy-selector harness
//! (single zone, per-class states present so the t_i route engages under the
//! mass-record surgery) and the round-113 hand-rolled 2-zone lumped 5R1C
//! pattern (ENV_LOCK serialization, free-float setpoints) with a live
//! night-ventilation fan for the R75 pins. These are record pins, not
//! validator bands: a violation means the route gate or a fall-through moved
//! and must be re-examined, never tuned around.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::invariant_checker::InvariantChecker;
use fluxion::sim::thermal_model_core::ThermalModel as CoreThermalModel;
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::validation::NightVentilation;
use std::sync::Mutex;

/// ENV_LOCK (Issue #3453 convention, gate `check_env_setvar_serialized`):
/// the route pins mutate process env, so the mutation is serialized through
/// a file-local mutex and every variable is removed on guard drop (a failing
/// assert must not leak a flag into later tests).
static ENV_LOCK: Mutex<()> = Mutex::new(());

const ROUTE_VARS: [&str; 3] = [
    "FLX_R74_RAMP_EXACT",
    "FLX_R75_VENT_RAMP",
    "FLX_R90_NO_LAG_FILTER",
];

/// RAII guard: remove every route variable on drop, however the test ends.
struct EnvGuard;

impl EnvGuard {
    fn with(k: &'static str, v: &str) -> Self {
        std::env::set_var(k, v);
        EnvGuard
    }
    fn clean() -> Self {
        for k in ROUTE_VARS {
            std::env::remove_var(k);
        }
        EnvGuard
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for k in ROUTE_VARS {
            std::env::remove_var(k);
        }
    }
}

const DT: f64 = 3600.0;

/// Internal-gain driving for the multizone fixture, W/m² — keeps both
/// zones' coupled advances live (the record only persists on a coupled step).
const LOADS_WM2: f64 = 30.0;

/// Symmetric inter-zone coupling for the main fixture, W/K.
const H_IZ: f64 = 2.0;
const H_IZ_RAD: f64 = 1.0;

/// The probe's artificial gain, W (the `check_invariant_with_artificial_gain`
/// argument convention and the twin's probe convention alike).
const GAIN_W: f64 = 1000.0;

/// The hand-rolled 2-zone lumped 5R1C fixture (the round-112/113 pattern,
/// fan removed — this suite adds it back only in `vented_multizone_model`).
/// Asymmetric by construction: zone 0 (48 m²) is the probed zone; zone 1
/// (30 m²) is the neighbor reached only through the inter-zone coupling.
fn multizone_model() -> ThermalModel<VectorField> {
    let mut m = ThermalModel::new(2);
    m.setpoints.zone_area = VectorField::new(vec![48.0, 30.0]);
    m.setpoints.ceiling_height = VectorField::from_scalar(2.7, 2);
    m.setpoints.heating_setpoint = -50.0;
    m.setpoints.cooling_setpoint = 100.0;
    m.setpoints.air_density = VectorField::new(vec![1.2, 1.4]);
    m.setpoints.heat_capacity = VectorField::from_scalar(1005.0, 2);
    m.solar.window_u_value = 3.0;
    m.conduction.h_tr_em = VectorField::new(vec![0.32 * 48.0, 9.0]);
    m.conduction.h_tr_w = VectorField::new(vec![3.0 * 12.0, 24.0]);
    m.conduction.h_tr_floor = VectorField::new(vec![0.15 * 48.0, 6.0]);
    m.conduction.h_ve = VectorField::new(vec![10.0, 8.0]);
    m.conduction.h_tr_ms = VectorField::new(vec![1000.0, 800.0]);
    m.conduction.h_tr_iz = VectorField::from_scalar(H_IZ, 2);
    m.conduction.h_tr_iz_rad = VectorField::from_scalar(H_IZ_RAD, 2);
    m.mass.thermal_capacitance = VectorField::new(vec![48.0 * 150_000.0, 30.0 * 180_000.0]);
    m.setpoints.temperatures = VectorField::from_scalar(21.0, 2);
    m.mass.mass_temperatures = VectorField::from_scalar(21.0, 2);
    m.mass.air_temperatures = VectorField::from_scalar(21.0, 2);
    m.setpoints.loads = VectorField::from_scalar(LOADS_WM2, 2);
    m.solar.solar_gains = VectorField::new(vec![0.0, 0.0]);
    m.solar.opaque_solar_gains = VectorField::new(vec![0.0, 0.0]);
    m.hvac.hvac_heating_capacity = 0.0;
    m.hvac.hvac_cooling_capacity = 0.0;
    m.update_optimization_cache();
    m
}

/// The vented variant: an always-active night-ventilation window (start 0,
/// end 0 — active every hour) with a 1400 m³/h fan, so `h_ve_night =
/// 1400·1.2·1005/3600 = 469.0 W/K` and the zone-0 night-vent routing is live
/// every step. Outdoor steps across hours (−10 → +5) so the persisted vent
/// term and the current one differ and the per-sub-step interpolation
/// (`nv_prev + f·(nv_cur − nv_prev)`, the coupled route's own default) is
/// exercised with real amplitude.
fn vented_multizone_model() -> ThermalModel<VectorField> {
    let mut m = multizone_model();
    m.hvac.night_ventilation = Some(NightVentilation::new(1400.0, 0, 0));
    m
}

/// The probe twin: loads[0] raised by GAIN_W / area[0] (the artificial-gain
/// convention), or the plain fixture when `None`.
fn probed_vented(gain: Option<f64>) -> ThermalModel<VectorField> {
    let mut m = vented_multizone_model();
    if let Some(g) = gain {
        let loads = m.setpoints.loads.as_ref().to_vec();
        m.setpoints.loads = VectorField::new(vec![
            loads[0] + g / m.setpoints.zone_area.as_ref()[0],
            loads[1],
        ]);
    }
    m
}

/// The round-103/107/108 harness: Case 600 (LowMass) under the explicit
/// legacy selector pins the 5R1C network with per-class states present (so
/// the t_i route engages under the mass-record surgery, exactly as the
/// round-107 contract does).
fn case600_legacy() -> CoreThermalModel<VectorField> {
    let spec = ASHRAE140Case::Case600.spec();
    CoreThermalModel::from_spec_with_selector(&spec, &ThermalSelector::legacy())
        .expect("legacy 5R1C selector must initialize")
}

/// Contract 1 — the default route's record premise (the baseline every route
/// pin compares against): a coupled step persists the air record at the
/// round-115 nine-entry layout and the mass record at four entries per
/// sub-step.
#[test]
fn default_route_air_record_premise() {
    let _env = EnvGuard::clean();
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut model = case600_legacy();
    model.step_physics(0, 10.0, DT);

    let air = model
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("the default route must persist the air-node record");
    assert!(
        !air[0].is_empty() && air[0].len() % 9 == 0,
        "default-route air record must carry 9 entries per sub-step (round-115 \
         layout), got len {}",
        air[0].len()
    );
    let mass = model
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("the default route must persist the mass record");
    assert!(
        !mass[0].is_empty() && mass[0].len() % 4 == 0,
        "mass record must carry 4 entries per sub-step, got len {}",
        mass[0].len()
    );
}

/// Contract 2 — the R74 route (`FLX_R74_RAMP_EXACT`, presence-based):
/// the engine must NOT persist the air record (the mass record's gate is
/// `r77_coupled` alone and stays untouched), and every consumer keeps the
/// committed-air convention, composing from the route model's OWN records.
/// Note the route is engine-differentiating on this fixture even at the seed
/// hour (the R74 site freezes the hour-start mass refresh and integrates the
/// ramp exactly, vs the default route's per-sub-step refresh + midpoint
/// sampling), so no cross-route state identity is claimed — the pins are the
/// gate and the fall-through values.
#[test]
fn r74_route_suppresses_the_air_record_and_the_consumers_fall_through() {
    let _env = EnvGuard::clean();
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // The R74 route run — hour 0 of the Case600 legacy fixture.
    let _flag = EnvGuard::with("FLX_R74_RAMP_EXACT", "1");
    let mut model = case600_legacy();
    model.step_physics(0, 10.0, DT);
    drop(_flag);

    // (b) The gate: air record absent, mass record present.
    assert!(
        model.mass.r77_air_substep_coeffs.is_none(),
        "FLX_R74_RAMP_EXACT must suppress the air-node record (the gate's \
         !r74_ramp_exact condition)"
    );
    let mass_rec = model
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("the mass record's gate is r77_coupled alone — R74 must not touch it");
    assert!(
        !mass_rec[0].is_empty() && mass_rec[0].len() % 4 == 0,
        "mass record intact under R74, got len {}",
        mass_rec[0].len()
    );

    // (c) The composite arm falls through to the mass-only replay: the
    // response must equal the mass record's own affine composition (the
    // round-103 direct-share semantics — the same coefficients the arm
    // consumes, replayed in its order; the round-106 expected-value
    // pattern).
    let mut checker = InvariantChecker::default();
    let no_gain = checker.check_invariant(&model, DT, 10.0);
    assert!(
        no_gain.balance.abs() < 1e-9,
        "unperturbed mass-only replay must stay exactly zero (Issue #3297), got {}",
        no_gain.balance
    );
    let with_gain = checker.check_invariant_with_artificial_gain(&model, DT, 10.0, GAIN_W, 0);
    let response = with_gain.balance - no_gain.balance;
    let c = &mass_rec[0];
    let t_mass_prev = model.mass.previous_mass_temperatures.as_ref()[0];
    let t_mass = model.mass.mass_temperatures.as_ref()[0];
    let mut t_comp = t_mass_prev;
    for [a, b, _d, bl] in c.as_chunks::<4>().0 {
        t_comp = *a * t_comp + (*b + *bl * GAIN_W);
    }
    let expected = (t_mass - t_comp) * c[c.len() - 2];
    assert!(
        response.abs() > 1e-6,
        "the mass-only fall-through keeps real probe leverage, got {response} W"
    );
    assert!(
        (response - expected).abs() <= 1e-9 * expected.abs(),
        "R74 composite-arm response must equal the mass-only record \
         composition (the committed-air fall-through): got {response} W, \
         expected {expected} W"
    );

    // (d) The per-class t_i route (round-107 entry-condition surgery): with
    // the mass record cleared the air record is STILL absent (this route
    // suppressed it), so the identity keeps the committed air and the
    // response returns to the round-104 closed form `rad_frac·Δ` verbatim.
    // The no-gain baseline is RE-MEASURED under the per-class arm (the
    // mirror's aggregation identity carries a structural residual on these
    // route models — the class rebuild does not reproduce the lumped
    // committed mass — and it appears in both runs, canceling only against
    // its own arm's baseline; the round-107 contract measures no_gain after
    // the surgery for exactly this reason).
    model
        .mass
        .r77_mass_substep_coeffs
        .as_mut()
        .expect("mass record present before the t_i surgery")[0]
        .clear();
    let no_gain_pc = checker.check_invariant(&model, DT, 10.0);
    let with_gain_no_mass =
        checker.check_invariant_with_artificial_gain(&model, DT, 10.0, GAIN_W, 0);
    let response_no_mass = with_gain_no_mass.balance - no_gain_pc.balance;
    let rad_frac = 1.0 - model.solar.convective_fraction;
    assert!(
        (response_no_mass - rad_frac * GAIN_W).abs() <= 1e-6 * response_no_mass.abs(),
        "R74 t_i route must keep the committed air (fall-through): got \
         {response_no_mass} W, expected {} W",
        rad_frac * GAIN_W
    );

    // (e) The legacy gauge-proxy identity (round-108 surgery): response
    // exactly 0 W — the pre-D blind verdict verbatim, now through a ROUTE
    // suppression instead of a cleared record.
    if let Some(recs) = model.conduction.applied_5r1c_class_boundary.as_mut() {
        if !recs.is_empty() {
            recs[0] = None;
        }
    }
    model.mass.r70_surface_node_temperatures = None;
    model.mass.r70_surface_node_temperatures_prev = None;
    let no_gain_lg = checker.check_invariant(&model, DT, 10.0);
    let with_gain_legacy =
        checker.check_invariant_with_artificial_gain(&model, DT, 10.0, GAIN_W, 0);
    let response_legacy = with_gain_legacy.balance - no_gain_lg.balance;
    assert_eq!(
        response_legacy, 0.0,
        "R74 legacy identity must keep the committed air blend (fall-through): \
         got {response_legacy} W, expected exactly 0 W"
    );

    // (f) Presence-based truthiness: `std::env::var(...).is_ok()` — the
    // falsy token "0" STILL selects the route (an ad-hoc convention, unlike
    // the canonical env_bool sets; pinned so a future migration is a
    // conscious change, per the round-111 truthiness-pinning precedent).
    let _flag = EnvGuard::with("FLX_R74_RAMP_EXACT", "0");
    let mut model_falsy = case600_legacy();
    model_falsy.step_physics(0, 10.0, DT);
    drop(_flag);
    assert!(
        model_falsy.mass.r77_air_substep_coeffs.is_none(),
        "FLX_R74_RAMP_EXACT is presence-based: even \"0\" must select the \
         route and suppress the record"
    );
}

/// Contract 3 — the R90-filtered route (`FLX_R90_NO_LAG_FILTER` falsy): the
/// lag STATE feeds the num within the hour, so the record is suppressed and
/// the consumers keep the committed-air convention. Unlike R74 the route is
/// NOT engine-inert on this fixture (the lag midpoint differs from the direct
/// input even at hour 0 — replication: 0.031 mis-mirror gap), so no inertness
/// is claimed: each consumer composes from the route model's OWN records.
/// The token set pins the canonical env_bool convention (falsy 0/false/no/
/// off/empty; explicit truthy and a typo token behave as unset — fail-closed
/// to the record-carrying default).
#[test]
fn r90_filtered_route_suppresses_the_air_record_and_the_consumers_fall_through() {
    let _env = EnvGuard::clean();
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Token set: every canonical falsy token selects the filtered route
    // (record absent); an explicit truthy token and an unrecognized token
    // stay on the record-carrying default (env_bool fail-closed truthiness,
    // Issue #3751).
    for token in ["0", "false", "no", "off", ""] {
        let _flag = EnvGuard::with("FLX_R90_NO_LAG_FILTER", token);
        let mut m = case600_legacy();
        m.step_physics(0, 10.0, DT);
        drop(_flag);
        assert!(
            m.mass.r77_air_substep_coeffs.is_none(),
            "FLX_R90_NO_LAG_FILTER={token:?} must select the filtered route \
             and suppress the air record"
        );
    }
    for token in ["1", "ture"] {
        let _flag = EnvGuard::with("FLX_R90_NO_LAG_FILTER", token);
        let mut m = case600_legacy();
        m.step_physics(0, 10.0, DT);
        drop(_flag);
        assert!(
            m.mass.r77_air_substep_coeffs.is_some(),
            "FLX_R90_NO_LAG_FILTER={token:?} must behave as unset (record \
             present) — canonical truthy / fail-closed typo handling"
        );
    }

    // The route model for the consumer pins.
    let _flag = EnvGuard::with("FLX_R90_NO_LAG_FILTER", "0");
    let mut model = case600_legacy();
    model.step_physics(0, 10.0, DT);
    drop(_flag);

    // Gate: air record absent, mass record present.
    assert!(model.mass.r77_air_substep_coeffs.is_none());
    let mass_rec = model.mass.r77_mass_substep_coeffs.as_ref().expect(
        "the mass record's gate is r77_coupled alone — the lag route \
                 must not touch it",
    );
    assert!(!mass_rec[0].is_empty() && mass_rec[0].len() % 4 == 0);

    // Composite arm: mass-only fall-through, composed from THIS model's own
    // record (the lag route moves the states, so no cross-route constant is
    // claimed — the round-106 expected-value pattern on the route model).
    let mut checker = InvariantChecker::default();
    let no_gain = checker.check_invariant(&model, DT, 10.0);
    assert!(
        no_gain.balance.abs() < 1e-9,
        "unperturbed mass-only replay must stay exactly zero (Issue #3297), got {}",
        no_gain.balance
    );
    let with_gain = checker.check_invariant_with_artificial_gain(&model, DT, 10.0, GAIN_W, 0);
    let response = with_gain.balance - no_gain.balance;
    let c = &mass_rec[0];
    let t_mass_prev = model.mass.previous_mass_temperatures.as_ref()[0];
    let t_mass = model.mass.mass_temperatures.as_ref()[0];
    let mut t_comp = t_mass_prev;
    for [a, b, _d, bl] in c.as_chunks::<4>().0 {
        t_comp = *a * t_comp + (*b + *bl * GAIN_W);
    }
    let expected = (t_mass - t_comp) * c[c.len() - 2];
    assert!(
        response.abs() > 1e-6,
        "the mass-only fall-through keeps real probe leverage, got {response} W"
    );
    assert!(
        (response - expected).abs() <= 1e-9 * expected.abs(),
        "R90-filtered composite-arm response must equal the mass-only record \
         composition (the committed-air fall-through): got {response} W, \
         expected {expected} W"
    );

    // t_i route fall-through: rad_frac·Δ verbatim. (The no-gain baseline is
    // re-measured under the per-class arm after the surgery — the mirror's
    // aggregation identity carries a structural residual on these route
    // models that cancels only against its own arm's baseline; the
    // round-107 contract measures no_gain after the surgery too.)
    model
        .mass
        .r77_mass_substep_coeffs
        .as_mut()
        .expect("mass record present before the t_i surgery")[0]
        .clear();
    let no_gain_pc = checker.check_invariant(&model, DT, 10.0);
    let with_gain_no_mass =
        checker.check_invariant_with_artificial_gain(&model, DT, 10.0, GAIN_W, 0);
    let response_no_mass = with_gain_no_mass.balance - no_gain_pc.balance;
    let rad_frac = 1.0 - model.solar.convective_fraction;
    assert!(
        (response_no_mass - rad_frac * GAIN_W).abs() <= 1e-6 * response_no_mass.abs(),
        "R90-filtered t_i route must keep the committed air (fall-through): \
         got {response_no_mass} W, expected {} W",
        rad_frac * GAIN_W
    );

    // Legacy identity fall-through: exactly 0 W.
    if let Some(recs) = model.conduction.applied_5r1c_class_boundary.as_mut() {
        if !recs.is_empty() {
            recs[0] = None;
        }
    }
    model.mass.r70_surface_node_temperatures = None;
    model.mass.r70_surface_node_temperatures_prev = None;
    let no_gain_lg = checker.check_invariant(&model, DT, 10.0);
    let with_gain_legacy =
        checker.check_invariant_with_artificial_gain(&model, DT, 10.0, GAIN_W, 0);
    let response_legacy = with_gain_legacy.balance - no_gain_lg.balance;
    assert_eq!(
        response_legacy, 0.0,
        "R90-filtered legacy identity must keep the committed air blend: got \
         {response_legacy} W, expected exactly 0 W"
    );
}

/// Contract 4 — the R75 route stays record-carrying and exact (the
/// round-116 scoping precision of the plan §5 parenthetical): on the
/// recorded (coupled) route the vent-numerator interpolation is the DEFAULT
/// (`r77_coupled || r75_vent_ramp`), so `FLX_R75_VENT_RAMP` is inert there
/// and the records are byte-identical flag on/off at every hour — including
/// the hour where the persisted prev term and the current term differ (the
/// interpolation's real amplitude). The interpolated term is
/// probe-independent and additive, so the composed cross-response vectors
/// (zero-seed hour 1, seeded+NP sustained hour 2) still match the engine
/// counterfactual to <=1e-9 relative with the vent live (replication:
/// 5.3e-15 / 6.4e-14 on the twin).
#[test]
fn r75_route_is_form_invariant_on_the_recorded_route() {
    let _env = EnvGuard::clean();
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Fixture premise: the night-vent routing is actually live (a vented
    // hour must move the zone against the still model).
    let mut vented = vented_multizone_model();
    let mut still = multizone_model();
    vented.step_physics(0, -10.0, DT);
    still.step_physics(0, -10.0, DT);
    assert!(
        vented.setpoints.temperatures.as_ref()[0] != still.setpoints.temperatures.as_ref()[0],
        "fixture premise: the night-vent fan must be live in this fixture"
    );

    // The byte-identity pair: identical vented fixtures, one with the R75
    // flag, one without — stepped in lockstep across the outdoor step
    // (−10 → +5) so hour 1 interpolates between a −4690 W and a +2345 W
    // vent term.
    let mut base = probed_vented(None);
    let mut flagged = probed_vented(None);
    let _flag = EnvGuard::with("FLX_R75_VENT_RAMP", "1");

    base.step_physics(0, -10.0, DT);
    flagged.step_physics(0, -10.0, DT);
    assert_eq!(
        base.mass.r77_air_substep_coeffs, flagged.mass.r77_air_substep_coeffs,
        "hour 0: the R75 flag must be inert on the recorded route (the \
         interpolation is the coupled route's own default) — air records \
         byte-identical"
    );
    assert_eq!(
        base.mass.r77_mass_substep_coeffs, flagged.mass.r77_mass_substep_coeffs,
        "hour 0: mass records byte-identical flag on/off"
    );

    base.step_physics(1, 5.0, DT);
    flagged.step_physics(1, 5.0, DT);
    drop(_flag);
    let base_air_h2 = base
        .mass
        .r77_air_substep_coeffs
        .clone()
        .expect("air record h2 base");
    let flagged_air_h2 = flagged
        .mass
        .r77_air_substep_coeffs
        .clone()
        .expect("air record h2 flagged");
    assert_eq!(
        base_air_h2, flagged_air_h2,
        "hour 1 (nv_prev −4690 W vs nv_cur +2345 W): the interpolated term \
         never enters the slots — records byte-identical flag on/off"
    );

    // The counterfactual: sustained probe on zone 0, both hours, vent live.
    let mut cf = probed_vented(Some(GAIN_W));
    base = probed_vented(None);
    base.step_physics(0, -10.0, DT);
    cf.step_physics(0, -10.0, DT);

    let checker = InvariantChecker::default();
    let zeros = vec![0.0f64; 2];
    let (v1a, v1m) = checker
        .composed_cross_response_vector_seeded(&base, 0, GAIN_W, 0.0, &zeros, &zeros, &zeros);
    assert!(
        v1a[0] != 0.0,
        "hour-1 probe-zone entry carries its own response"
    );
    assert_eq!(
        v1a[1], 0.0,
        "hour-1 neighbor entry exactly zero (seed hour: the neighbor never \
         saw the probe)"
    );

    // Sustained: hour 2 keeps the probe, and the outdoor steps to +5.0 so
    // the vent interpolation is live on the crossed hour too.
    base.step_physics(1, 5.0, DT);
    cf.step_physics(1, 5.0, DT);

    let (v2a, v2m) =
        checker.composed_cross_response_vector_seeded(&base, 0, GAIN_W, GAIN_W, &v1a, &v1a, &v1m);

    let d_t0 = cf.mass.air_temperatures.as_ref()[0] - base.mass.air_temperatures.as_ref()[0];
    let d_m0 = cf.mass.mass_temperatures.as_ref()[0] - base.mass.mass_temperatures.as_ref()[0];
    let d_t1 = cf.mass.air_temperatures.as_ref()[1] - base.mass.air_temperatures.as_ref()[1];
    let d_m1 = cf.mass.mass_temperatures.as_ref()[1] - base.mass.mass_temperatures.as_ref()[1];
    assert!(
        d_t0.abs() > 1e-6 && d_t1.abs() > 1e-9,
        "fixture premise: the sustained probe must move both zones through \
         the iz coupling (air {d_t0}/{d_t1} K)"
    );
    assert!(
        (v2a[0] - d_t0).abs() <= 1e-9 * d_t0.abs(),
        "sustained hour-2 probe-zone AIR entry must equal the engine \
         counterfactual with the interpolated vent live (composed {} vs \
         measured {d_t0})",
        v2a[0]
    );
    assert!(
        (v2m[0] - d_m0).abs() <= 1e-9 * d_m0.abs(),
        "sustained hour-2 probe-zone MASS entry must equal the engine \
         counterfactual with the vent live (composed {} vs measured {d_m0})",
        v2m[0]
    );
    assert!(
        (v2a[1] - d_t1).abs() <= 1e-9 * d_t1.abs(),
        "sustained hour-2 neighbor AIR entry must equal the engine \
         counterfactual through the iz channel with the vent live \
         (composed {} vs measured {d_t1})",
        v2a[1]
    );
    assert!(
        (v2m[1] - d_m1).abs() <= 1e-9 * d_m1.abs(),
        "sustained hour-2 neighbor MASS entry must equal the engine \
         counterfactual with the vent live (composed {} vs measured {d_m1})",
        v2m[1]
    );
}
