//! Inter-zone cross-response contracts (loop round 113) — the checker's
//! NEIGHBOR-side response to a single-zone load probe, the
//! checker-air-node-mirror plan §5's explicitly-bounded-later item
//! ("inter-zone cross-responses (Case 960 family — a single-zone probe's
//! neighbor-side response needs a per-zone response vector, not a scalar)").
//!
//! Component under contract: the recorded inter-zone cross-response slot
//! (the air-node record's EIGHTH entry, round 113) and the checker's
//! `composed_cross_response_vector` over it. The engine's iz block adds
//! `q_iz_net[i] = h_tr_iz[i]·(Σ_j T[j] − N·T[i])` to `phi_ia` from the
//! COMMITTED hour-start air states, and `num_rest` is mul_assign'd by
//! `term_rest_1` (#1585), so a neighbor's committed state enters the num
//! with slope exactly `ramp_f·term_rest_1[i]·h_tr_iz[i]` when the iz block
//! is live (`num_zones > 1`, `h_tr_iz[i] > 0.0`) — the slot records that
//! consumed slope in the same lockstep push, and the checker composes
//! `r_air' = D·r_air + M·r_m + W·Δ_j + C·(Σ_i prev_i − N·prev_j)` per
//! sub-step over the previously committed air-response vector.
//!
//! Scope of the exactness claim (Python replication
//! `/tmp/r113_replication.py`, dev-time evidence, all scenarios PASS):
//! the composition is EXACT for a zone whose own committed states did not
//! respond in the previous hour — the neighbor of a first probe hour —
//! because its own load, lag state and mass are probe-independent there,
//! so the iz source is the only live channel (verified: the iz-source-only
//! perturbation reproduces the full numeric response to 1.4e-13). A zone
//! that DID respond in the previous hour (a sustained probe's own
//! continuation) carries three further channels this record does not
//! capture — the R71 ramp-source persistence response (`num_prev` is the
//! frozen hour-start num_rest, load-dependent; the synthetic engine's gap
//! closes to 8.7e-14 with the seeds + np channel added), the hour-boundary
//! air/mass state seeds, and the R90 lag state on non-record routes — the
//! plan's re-scoped remainder, documented in the plan §5 amendment and
//! quantified by contract 3's divergence pin, never claimed exact.
//!
//! Fixture: the round-112 hand-rolled 2-zone lumped 5R1C pattern
//! (`ThermalModel::new(2)`, CZ3 hand-built, asymmetric zones, symmetric
//! `h_tr_iz = 2.0` / `h_tr_iz_rad = 1.0`), NO night-ventilation fan — the
//! vent routing is round-112's scope and a moving part this suite does
//! not need. Round-105/110/112 suite shape: record pins with bit-exact
//! teeth, file-local ENV_LOCK serialization, free-float setpoints so HVAC
//! cannot mask the air node. These are record pins, not validator bands:
//! a violation means the cross-response routing moved and must be
//! re-examined, never tuned around.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::invariant_checker::InvariantChecker;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

const DT: f64 = 3600.0;

/// Internal-gain driving for the fixture, W/m² — keeps both zones' coupled
/// advances live (the record only persists on a coupled step).
const LOADS_WM2: f64 = 30.0;

/// Symmetric inter-zone coupling for the main fixture, W/K.
const H_IZ: f64 = 2.0;
const H_IZ_RAD: f64 = 1.0;

/// The probe's artificial gain, W (the `check_invariant_with_artificial_gain`
/// argument convention).
const GAIN_W: f64 = 1000.0;

/// The hand-rolled 2-zone lumped 5R1C fixture (the round-112 pattern, fan
/// removed). Asymmetric by construction: zone 0 (48 m²) is the probed zone;
/// zone 1 (30 m², different conductances, mass and air density) is the
/// neighbor reached only through the inter-zone coupling.
fn multizone_model() -> ThermalModel<VectorField> {
    let mut m = ThermalModel::new(2);
    m.setpoints.zone_area = VectorField::new(vec![48.0, 30.0]);
    m.setpoints.ceiling_height = VectorField::from_scalar(2.7, 2);
    m.setpoints.heating_setpoint = -50.0;
    m.setpoints.cooling_setpoint = 100.0;
    // Zone 1's density differs on purpose: any zone-confused coefficient
    // read becomes visible.
    m.setpoints.air_density = VectorField::new(vec![1.2, 1.4]);
    m.setpoints.heat_capacity = VectorField::from_scalar(1005.0, 2);
    m.solar.window_u_value = 3.0;
    m.conduction.h_tr_em = VectorField::new(vec![0.32 * 48.0, 9.0]);
    m.conduction.h_tr_w = VectorField::new(vec![3.0 * 12.0, 24.0]);
    m.conduction.h_tr_floor = VectorField::new(vec![0.15 * 48.0, 6.0]);
    m.conduction.h_ve = VectorField::new(vec![10.0, 8.0]);
    m.conduction.h_tr_ms = VectorField::new(vec![1000.0, 800.0]);
    // h_tr_is keeps the ThermalModel::new default (69.0 per zone), so
    // term_rest_1 is 1069.0 / 869.0 — asserted as a fixture premise below.
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

/// The fixture with the inter-zone coupling overridden (0.0 = independent
/// zones for the cross-response discriminator).
fn with_coupling(iz: f64) -> ThermalModel<VectorField> {
    let mut m = multizone_model();
    m.conduction.h_tr_iz = VectorField::from_scalar(iz, 2);
    m.conduction.h_tr_iz_rad = VectorField::from_scalar(iz, 2);
    m.update_optimization_cache();
    m
}

/// The probe twin: loads[0] raised by GAIN_W / area[0] (the artificial-gain
/// convention), or the plain fixture when `None`.
fn probed(gain: Option<f64>) -> ThermalModel<VectorField> {
    let mut m = multizone_model();
    if let Some(g) = gain {
        let loads = m.setpoints.loads.as_ref().to_vec();
        m.setpoints.loads = VectorField::new(vec![
            loads[0] + g / m.setpoints.zone_area.as_ref()[0],
            loads[1],
        ]);
    }
    m
}

/// The air record for one zone: (sub-step count, the C-slot sequence).
/// EIGHT entries per sub-step `[D, M, W, den, h3od, a, bl, C]` (round-113
/// layout; C at index 7).
fn air_record(m: &ThermalModel<VectorField>, zone: usize) -> (usize, Vec<f64>) {
    let rec = m
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("a live coupled 5R1C step must persist the air-node record");
    let row = &rec[zone];
    assert!(
        !row.is_empty() && row.len() % 8 == 0,
        "air record must carry 8 entries per sub-step, got {}",
        row.len()
    );
    let n = row.len() / 8;
    let c_seq = (0..n).map(|k| row[8 * k + 7]).collect::<Vec<f64>>();
    (n, c_seq)
}

#[test]
/// Contract 1 — the record's EIGHTH entry: shape and the consumed closed
/// form. On every sub-step of a seeded hour the C slot must equal
/// `(1−D)·(term_rest_1[i]·h_tr_iz[i])/den` bit-exactly (air_ramp_f = 1.0),
/// and on every sub-step of a ramped hour `(1−D)·((f_k·term_rest_1)·h_tr_iz)/den`
/// bit-exactly, with `D` and `den` read from the record's own slots 0/3 and
/// `term_rest_1`/`h_tr_iz` from the model — the same operands the engine
/// consumed, in the engine's op order (Z4: no algebra the engine did not
/// execute). The single-zone model consumes no iz term (`num_zones > 1`
/// gate), so its C slot is exactly 0.0.
fn cross_slot_record_shape_and_closed_forms() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut m = multizone_model();
    m.step_physics(0, -10.0, DT);

    let tr1 = m.conduction.derived_term_rest_1.as_ref();
    assert_eq!(tr1[0], 1069.0, "fixture premise: zone 0 term_rest_1");
    assert_eq!(
        tr1[1], 869.0,
        "fixture premise: zone 1 term_rest_1 (asymmetric)"
    );
    // Snapshot the (construction-static) conductances as owned values so the
    // mutable step below can run.
    let (tr1_0, tr1_1, h_iz_0, h_iz_1) = (
        tr1[0],
        tr1[1],
        m.conduction.h_tr_iz.as_ref()[0],
        m.conduction.h_tr_iz.as_ref()[1],
    );
    let h_iz = [h_iz_0, h_iz_1];
    let tr1s = [tr1_0, tr1_1];

    let (n0, c0) = air_record(&m, 0);
    let (n1, c1) = air_record(&m, 1);
    assert_eq!(n0, n1, "same coupled scheme, same sub-step count");
    assert_eq!(n0, 12, "fixture premise: the default coupled N");

    // Seeded hour: air_ramp_f == 1.0, and (1.0 * x) * y == x * y bit-exactly.
    let rec = m.mass.r77_air_substep_coeffs.as_ref().expect("air record");
    for (j, c_seq) in [c0.clone(), c1.clone()].iter().enumerate() {
        for (k, &c) in c_seq.iter().enumerate() {
            let d = rec[j][8 * k];
            let den = rec[j][8 * k + 3];
            let expected = (1.0 - d) * (tr1s[j] * h_iz[j]) / den;
            assert_eq!(
                c, expected,
                "hour 1 zone {j} sub-step {k}: the C slot must be the consumed \
                 iz cross-response coefficient bit-exactly"
            );
            assert!(
                c > 0.0,
                "hour 1 zone {j} sub-step {k}: a live iz coupling must record \
                 a nonzero cross-response slope"
            );
        }
    }

    // Ramped hour: air_ramp_f == f_k, the engine's op order is
    // ((f_k·term_rest_1)·h_tr_iz), then (1−D)·coef/den.
    m.step_physics(1, -10.0, DT);
    let rec2 = m
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record h2");
    let (_, c0_h2) = air_record(&m, 0);
    let (_, c1_h2) = air_record(&m, 1);
    for (j, c_seq) in [c0_h2, c1_h2].iter().enumerate() {
        for (k, &c) in c_seq.iter().enumerate() {
            let f_k = (k as f64 + 0.5) / 12.0;
            let d = rec2[j][8 * k];
            let den = rec2[j][8 * k + 3];
            let aiz = (f_k * tr1s[j]) * h_iz[j];
            let expected = (1.0 - d) * aiz / den;
            assert_eq!(
                c, expected,
                "hour 2 zone {j} sub-step {k}: the C slot must carry the ramped \
                 iz slope bit-exactly"
            );
        }
    }

    // Single-zone discriminator: num_zones == 1 → the iz block never runs,
    // and the slot must be exactly 0.0.
    let mut m1 = ThermalModel::<VectorField>::new(1);
    m1.setpoints.zone_area = VectorField::new(vec![48.0]);
    m1.setpoints.ceiling_height = VectorField::from_scalar(2.7, 1);
    m1.setpoints.heating_setpoint = -50.0;
    m1.setpoints.cooling_setpoint = 100.0;
    m1.setpoints.air_density = VectorField::new(vec![1.2]);
    m1.setpoints.heat_capacity = VectorField::from_scalar(1005.0, 1);
    m1.conduction.h_tr_em = VectorField::new(vec![0.32 * 48.0]);
    m1.conduction.h_tr_w = VectorField::new(vec![36.0]);
    m1.conduction.h_tr_floor = VectorField::new(vec![0.15 * 48.0]);
    m1.conduction.h_ve = VectorField::new(vec![10.0]);
    m1.conduction.h_tr_ms = VectorField::new(vec![1000.0]);
    m1.mass.thermal_capacitance = VectorField::new(vec![48.0 * 150_000.0]);
    m1.setpoints.temperatures = VectorField::from_scalar(21.0, 1);
    m1.mass.mass_temperatures = VectorField::from_scalar(21.0, 1);
    m1.mass.air_temperatures = VectorField::from_scalar(21.0, 1);
    m1.setpoints.loads = VectorField::from_scalar(LOADS_WM2, 1);
    m1.solar.solar_gains = VectorField::new(vec![0.0]);
    m1.solar.opaque_solar_gains = VectorField::new(vec![0.0]);
    m1.hvac.hvac_heating_capacity = 0.0;
    m1.hvac.hvac_cooling_capacity = 0.0;
    m1.update_optimization_cache();
    m1.step_physics(0, -10.0, DT);
    let (n1z, c1z) = air_record(&m1, 0);
    assert_eq!(n1z, 12, "single-zone fixture premise");
    for (k, &c) in c1z.iter().enumerate() {
        assert_eq!(
            c, 0.0,
            "single-zone sub-step {k}: no iz block ran, the C slot must be \
             exactly 0.0 (Z4 — the slot records what the engine consumed)"
        );
    }
}

#[test]
/// Contract 2 — the FIRST probe hour's response vector: the probe zone's
/// entry equals the engine's own committed shift under the artificial gain
/// (the increment-B-class counterfactual agreement, ≤1e-9 relative — the
/// cross source is zero because the committed hour-start states are
/// unshifted), and the neighbor's entry is EXACTLY zero while the neighbor's
/// measured hour-1 shift is also exactly zero — the one-step lag, now
/// pinned at the response-vector level (the round-112 contract-4 state
/// premise, composed). Zero-delta: an all-zero call returns exact zeros.
fn first_probe_hour_vector_self_only_exact() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut base = multizone_model();
    let mut cf = probed(Some(GAIN_W));

    base.step_physics(0, -10.0, DT);
    cf.step_physics(0, -10.0, DT);

    let checker = InvariantChecker::default();
    let zeros = vec![0.0f64; 2];
    let v1 = checker.composed_cross_response_vector(&base, 0, GAIN_W, &zeros);

    let d_t0 = cf.mass.air_temperatures.as_ref()[0] - base.mass.air_temperatures.as_ref()[0];
    let d_t1 = cf.mass.air_temperatures.as_ref()[1] - base.mass.air_temperatures.as_ref()[1];
    assert!(
        d_t0.abs() > 1e-6,
        "fixture premise: the probe zone moves (ΔT_0 = {d_t0})"
    );
    assert_eq!(
        d_t1, 0.0,
        "fixture premise: hour-1 neighbor shift is exactly zero (one-step lag)"
    );
    assert!(
        (v1[0] - d_t0).abs() <= 1e-9 * d_t0.abs(),
        "hour-1 probe-zone entry must equal the engine's committed shift \
         (composed {} vs measured {d_t0})",
        v1[0]
    );
    assert_eq!(
        v1[1], 0.0,
        "hour-1 neighbor entry must be exactly zero — the coupling reaches a \
         zone only through its committed hour-start states"
    );

    let v0 = checker.composed_cross_response_vector(&base, 0, 0.0, &zeros);
    assert!(
        v0.iter().all(|&x| x == 0.0),
        "zero delta with zero prev must return exact zeros (#3297 residual-zero)"
    );
}

#[test]
/// Contract 3 — the FLAGSHIP: the neighbor's first crossed hour. Probe hour
/// 1 only (the counterfactual's load reverts), plain hour 2; the composed
/// hour-2 neighbor entry (from the hour-1 vector as `prev`, zero seeds,
/// delta 0) must equal the engine counterfactual's measured hour-2 neighbor
/// shift to ≤1e-9 relative — the exactness the plan-§5 increment-E scope
/// claims, because the neighbor's own load, lag state and mass are
/// probe-independent at that hour and the iz source is the only live
/// channel. The probe zone's OWN hour-2 entry is NOT claimed exact (the
/// carry-channel remainder: R71 ramp-source persistence, hour-boundary
/// seeds, lag state) — the divergence is pinned, numerically documented,
/// never tuned around.
fn neighbor_first_crossed_hour_matches_engine_counterfactual() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut base = multizone_model();
    let mut cf = probed(Some(GAIN_W));

    base.step_physics(0, -10.0, DT);
    cf.step_physics(0, -10.0, DT);

    let checker = InvariantChecker::default();
    let zeros = vec![0.0f64; 2];
    let v1 = checker.composed_cross_response_vector(&base, 0, GAIN_W, &zeros);

    // The single-hour probe: revert the counterfactual's load for hour 2.
    let cf_loads = cf.setpoints.loads.as_ref().to_vec();
    cf.setpoints.loads = VectorField::new(vec![
        cf_loads[0] - GAIN_W / cf.setpoints.zone_area.as_ref()[0],
        cf_loads[1],
    ]);

    base.step_physics(1, -10.0, DT);
    cf.step_physics(1, -10.0, DT);

    let v2 = checker.composed_cross_response_vector(&base, 0, 0.0, &v1);

    let d_t1 = cf.mass.air_temperatures.as_ref()[1] - base.mass.air_temperatures.as_ref()[1];
    assert!(
        d_t1.abs() > 1e-9,
        "fixture premise: the neighbor must move at hour 2 (ΔT_1 = {d_t1})"
    );
    assert!(
        (v2[1] - d_t1).abs() <= 1e-9 * d_t1.abs(),
        "hour-2 neighbor entry must equal the engine counterfactual exactly \
         to 1e-9 relative (composed {} K vs measured {d_t1} K)",
        v2[1]
    );

    // The documented remainder: the probe zone's own hour-2 response carries
    // the carry channels (R71 ramp-source persistence, hour-boundary seeds),
    // so the records-only composition must NOT match there — pinned as a
    // nonzero, order-visible divergence (replication: composed −0.0022 K/W
    // vs numeric +0.209 K/W on the synthetic twin; the fixture's own numbers
    // are printed for the record).
    let d_t0 = cf.mass.air_temperatures.as_ref()[0] - base.mass.air_temperatures.as_ref()[0];
    let gap = (v2[0] - d_t0).abs();
    println!(
        "remainder at the probe zone (hour 2): composed {} K vs measured {d_t0} K, gap {gap} K",
        v2[0]
    );
    assert!(
        gap > 1e-6,
        "the probe zone's own hour-2 composition is expected to diverge from \
         the counterfactual (the bounded-later carry channels) — a match here \
         would mean the fixture lost its discrimination (composed {} vs {d_t0})",
        v2[0]
    );
}

#[test]
/// Contract 4 — the zero-coupling discriminator: with `h_tr_iz = 0` the iz
/// block never runs, every C slot is exactly 0.0, the hour-2 neighbor entry
/// is exactly zero AND the counterfactual's measured hour-2 neighbor shift
/// is exactly zero (a fully independent zone — the round-112 contract-4
/// state premise, now at the response level).
fn zero_coupling_leaves_neighbor_exactly_zero() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut base = with_coupling(0.0);
    let mut cf = {
        let mut m = with_coupling(0.0);
        let loads = m.setpoints.loads.as_ref().to_vec();
        m.setpoints.loads = VectorField::new(vec![
            loads[0] + GAIN_W / m.setpoints.zone_area.as_ref()[0],
            loads[1],
        ]);
        m
    };

    base.step_physics(0, -10.0, DT);
    cf.step_physics(0, -10.0, DT);

    let (_, c0) = air_record(&base, 0);
    let (_, c1) = air_record(&base, 1);
    for (k, &c) in c0.iter().enumerate() {
        assert_eq!(
            c, 0.0,
            "hour 1 zone 0 sub-step {k}: h_tr_iz = 0 must leave the C slot exactly 0.0"
        );
    }
    for (k, &c) in c1.iter().enumerate() {
        assert_eq!(
            c, 0.0,
            "hour 1 zone 1 sub-step {k}: h_tr_iz = 0 must leave the C slot exactly 0.0"
        );
    }

    let checker = InvariantChecker::default();
    let zeros = vec![0.0f64; 2];
    let v1 = checker.composed_cross_response_vector(&base, 0, GAIN_W, &zeros);
    assert_eq!(v1[1], 0.0, "hour-1 neighbor entry exactly zero");
    assert!(
        v1[0] != 0.0,
        "hour-1 probe-zone entry carries its own load response"
    );

    base.step_physics(1, -10.0, DT);
    cf.step_physics(1, -10.0, DT);
    let v2 = checker.composed_cross_response_vector(&base, 0, 0.0, &v1);
    assert_eq!(
        v2[1], 0.0,
        "hour-2 neighbor entry must be exactly zero without coupling"
    );
    let d_t1 = cf.mass.air_temperatures.as_ref()[1] - base.mass.air_temperatures.as_ref()[1];
    assert_eq!(
        d_t1, 0.0,
        "the counterfactual's hour-2 neighbor shift must be exactly zero \
         without coupling (a fully independent zone)"
    );
}

#[test]
/// Contract 5 — the single-live-channel premise on the REAL engine: after
/// the probe hour, the neighbor's own probe-independent carry channels are
/// byte-identical between the committed and counterfactual runs — its whole
/// air record and mass record (its own load, ramp and coupling coefficients
/// never saw the probe), and its persisted R71 ramp sources
/// (`r71_air_driver_num_prev` / `r71_air_driver_den_prev`) — while its
/// committed states still move at hour 2 (contract 3). These pins are what
/// make the iz source the only live channel for the neighbor's first
/// crossed hour, and they fail loudly if a future engine change routes any
/// other probe-dependent coefficient into a neighbor's record.
fn neighbor_channels_inert_at_first_crossed_hour() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut base = multizone_model();
    let mut cf = probed(Some(GAIN_W));

    base.step_physics(0, -10.0, DT);
    cf.step_physics(0, -10.0, DT);

    let base_air1 = base
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (base h1)")[1]
        .clone();
    let cf_air1 = cf
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (cf h1)")[1]
        .clone();
    assert_eq!(
        base_air1, cf_air1,
        "hour 1: the neighbor's whole air record must be byte-identical \
         probe-on/off — its own coefficients never saw the probe"
    );
    let base_mass1 = base
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("mass record (base h1)")[1]
        .clone();
    let cf_mass1 = cf
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("mass record (cf h1)")[1]
        .clone();
    assert_eq!(
        base_mass1, cf_mass1,
        "hour 1: the neighbor's whole mass record must be byte-identical"
    );
    assert_eq!(
        base.mass.r71_air_driver_num_prev.as_ref()[1],
        cf.mass.r71_air_driver_num_prev.as_ref()[1],
        "hour 1: the neighbor's persisted R71 num ramp source must be \
         byte-identical (its own frozen num_rest never saw the probe)"
    );
    assert_eq!(
        base.mass.r71_air_driver_den_prev.as_ref()[1],
        cf.mass.r71_air_driver_den_prev.as_ref()[1],
        "hour 1: the neighbor's persisted R71 den ramp source must be \
         byte-identical (conductances are probe-independent)"
    );
}
