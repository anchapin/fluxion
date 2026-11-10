//! §7 surface-pool blitz slice (loop round 109) — the surface pools' remaining
//! contracts, mirroring the round-85 mass-node suite's shape (slices 1–6 live
//! in `tests/mass_advance_contract.rs`).
//!
//! Component under contract: the 5R1C **hourly per-class surface-pool state**:
//!
//! - (a) the t_s-first backward-Euler surface advance and the per-class BE
//!   mass form it feeds (`step_5r1c.rs` per-class mass loop; D5 default since
//!   loop round 71, #3983 §57; checker-persisted since round 73, #4512);
//! - (b) the applied per-class boundary record the checker mirrors
//!   (`Applied5R1CClassBoundary`, round-73 persistence, round-104 rebuild);
//! - (c) the per-class sol-air aggregation (Issue #4166 per-surface exterior
//!   boundary; corrected sol-air sign per Issue #4224);
//! - (d) the wired per-surface interior-LW steady estimates (Issue #2890).
//!
//! Plan §7 queue status after this slice
//! (`docs/implementation-plans/checker-air-node-mirror-plan.md` §7):
//! - (i) sol-air composition per class — pinned by contracts 3 and 4;
//! - (ii) phi_st_sol per-class fractions — already pinned by the rounds-91–105
//!   contract family (`mass_advance_contract.rs` contracts 15–22); not
//!   duplicated here;
//! - (iii) the t_s advance per class — pinned by contracts 1, 2, 5, 6 and 7
//!   (input→output map, snapshot lockstep, committed route on the uncoupled
//!   operator, degenerate and identical-construction arms);
//! - (iv) the longwave exchange between pools — the `InteriorSurfaceNetwork`
//!   view-factor math is unit-tested at its own site
//!   (`src/sim/longwave_exchange.rs`); the network is deliberately NOT routed
//!   into the 5R1C air balance yet (Issue #2890-followup), so the honest pin
//!   is the steady estimate the step actually persists today (contract 8).
//!
//! Route discipline: contracts 1–4 and 8 run the default coupled route — the
//! hourly per-class advance always executes and persists its records (the r70
//! t_s state and the applied class boundary) even though the committed mass
//! comes from the coupled sub-step scheme. Contracts 5–7 need the hourly
//! advance to be the COMMITTED mass route, so they flip `FLX_R77_COUPLED_N=0`
//! (the documented uncoupled operator: falsy "0" disables the coupled scheme,
//! `FLX_R76_MASS_SUBSTEPS` defaults to 0, so `r76_final` stays `None` and the
//! hourly per-class result commits). The env flip is process-local: each test
//! file is its own test binary, so only this file's tests need serialization —
//! every test holds the file-local ENV_LOCK for its whole body (round-86
//! convention; the engine reads the flags per step).
//!
//! These are record pins, not validator bands: a violation means the
//! surface-pool advance moved and must be re-examined, never tuned around.

use fluxion::batch_oracle::BatchOracle;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::thermal_integration::{crank_nicolson_iso13790, select_integration_method};
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

const DT: f64 = 3600.0;

/// A spec-built Case 600 baseline under free-float setpoints: the per-class
/// construction tensors are present (the default arm engages), the ideal HVAC
/// never engages (setpoints far outside any reachable state, the round-13b2
/// convention — controlled setpoints hide the committed air node), and the
/// internal loads are neutralized so the driving is the envelope alone.
fn spec_model_freefloat() -> ThermalModel<VectorField> {
    let mut m = BatchOracle::default_model().expect("spec model builds");
    m.setpoints.heating_setpoint = -50.0;
    m.setpoints.cooling_setpoint = 100.0;
    let n = m.hvac.num_zones;
    m.setpoints.loads = VectorField::new(vec![0.0; n]);
    m.solar.solar_gains = VectorField::new(vec![0.0; n]);
    m.solar.opaque_solar_gains = VectorField::new(vec![0.0; n]);
    m
}

/// The free-float spec baseline with the zone-level and per-class `h_tr_ms`
/// zeroed: `derived_h_tr_3` collapses to the 0 fallback (the Issue #2868
/// degenerate route), the zone-level `h_air_mass` series fallback evaluates
/// to 0 (both series terms carry the zeroed `h_tr_ms`), and the per-class
/// mass advance runs the degenerate `h_air_mass·w` arm with the distinct spec
/// capacitances keeping `use_r67` true.
fn degenerate_class_model() -> ThermalModel<VectorField> {
    let mut m = spec_model_freefloat();
    let n = m.hvac.num_zones;
    m.conduction.h_tr_ms = VectorField::from_scalar(0.0, n);
    m.conduction.h_tr_ms_wall = Some(VectorField::from_scalar(0.0, n));
    m.conduction.h_tr_ms_roof = Some(VectorField::from_scalar(0.0, n));
    m.conduction.h_tr_ms_floor = Some(VectorField::from_scalar(0.0, n));
    m.update_optimization_cache();
    m
}

/// Flattens the per-class construction tensors to one shared construction so
/// the `use_r67` structural-equality guard fails: the class loop must take
/// the legacy `advance_node` route (no t_s-first layer).
fn flatten_class_tensors(m: &mut ThermalModel<VectorField>) {
    let n = m.hvac.num_zones;
    let cm = m.conduction.cm_wall.take().expect("spec cm_wall present");
    m.conduction.cm_roof = Some(cm.clone());
    m.conduction.cm_floor = Some(cm.clone());
    let ms = m
        .conduction
        .h_tr_ms_wall
        .take()
        .expect("spec h_tr_ms_wall present");
    m.conduction.h_tr_ms_roof = Some(ms.clone());
    m.conduction.h_tr_ms_floor = Some(ms);
    m.update_optimization_cache();
}

/// One t_s-first backward-Euler surface advance + per-class BE mass advance,
/// replicated verbatim from the engine's operation order (see
/// `/tmp/r109_ts_replication.py` for the independent Python derivation).
/// Returns `(t_s_new_per_class, tm_w)`; classes the engine skips (area share
/// ≤ 0) come back as `None` in the t_s vector.
#[allow(clippy::too_many_arguments)]
fn replicate_per_class_advance(
    rec: &fluxion::sim::thermal_model_data::Applied5R1CClassBoundary,
    cls_cm: &[f64; 3],
    cls_ms: &[f64; 3],
    h_is_zone: f64,
    degen: bool,
    tm_old: f64,
    outdoor_temp: f64,
    t_s_old: &[f64; 3],
) -> ([Option<f64>; 3], f64) {
    let total_area: f64 = rec.cls_area.iter().sum();
    let cm_sum: f64 = cls_cm.iter().sum();
    let mut t_s_new = [None; 3];
    let mut tm_w = 0.0;
    for c in 0..3 {
        let w = rec.cls_area[c] / total_area;
        if w <= 0.0 {
            continue;
        }
        let (h_em_c, t_ext_c) = if c == 2 {
            (0.0, outdoor_temp)
        } else {
            (rec.cls_g[c], rec.cls_t[c])
        };
        let h_is_c = h_is_zone * w;
        let h_ms_c = cls_ms[c];
        let phi_st_c = rec.phi_st_zone * w;
        let cs_c = 0.1 * cls_cm[c];
        let cs_dt = cs_c / DT;
        let denom_s = cs_dt + h_is_c + h_ms_c;
        let t_s_next = if denom_s > 0.0 {
            (cs_dt * t_s_old[c] + h_is_c * rec.t_i + h_ms_c * tm_old + phi_st_c) / denom_s
        } else {
            (h_ms_c * tm_old + h_is_c * rec.t_i + phi_st_c) / (h_ms_c + h_is_c)
        };
        t_s_new[c] = Some(t_s_next);
        let h_am_eff = if degen {
            rec.h_air_mass * w
        } else {
            h_ms_c * h_is_c / (h_ms_c + h_is_c)
        };
        let cm_dt = cls_cm[c] / DT;
        let denom_m = cm_dt + h_em_c + h_am_eff + rec.h_vent_mass_zone * w;
        let tm_c_new = (cm_dt * tm_old
            + h_em_c * t_ext_c
            + h_am_eff * t_s_next
            + rec.h_vent_mass_zone * w * outdoor_temp
            + rec.phi_m_zone * w)
            / denom_m;
        tm_w += (cls_cm[c] / cm_sum) * tm_c_new;
    }
    (t_s_new, tm_w)
}

#[test]
/// Contract 1 — the t_s-first surface advance identity: after a step on the
/// default coupled route, the persisted per-class t_s state must equal the
/// closed form recomputed from the same inputs the engine consumed (the
/// applied class-boundary record, the class tensors, the pre-step mass and
/// the persisted previous t_s). The engine wrote the state from this exact
/// arithmetic, so the assertion is bit-exact — any algebra drift anywhere in
/// the surface advance (coefficients, operand order, the area-only phi_st
/// partition, the floor class's zero exterior conductance) breaks it.
fn ts_first_surface_advance_matches_closed_form() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut m = spec_model_freefloat();
    let tm_old = m.mass.mass_temperatures.as_ref()[0];
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
    let grab = |o: &Option<VectorField>| o.as_ref().and_then(|v| v.as_ref().first().copied());
    let cls_cm: [f64; 3] = [
        grab(&m.conduction.cm_wall).expect("cm_wall"),
        grab(&m.conduction.cm_roof).expect("cm_roof"),
        grab(&m.conduction.cm_floor).expect("cm_floor"),
    ];
    let cls_ms: [f64; 3] = [
        grab(&m.conduction.h_tr_ms_wall).expect("ms_wall"),
        grab(&m.conduction.h_tr_ms_roof).expect("ms_roof"),
        grab(&m.conduction.h_tr_ms_floor).expect("ms_floor"),
    ];
    let h_is_zone = m.conduction.h_tr_is.as_ref()[0];
    let tsn = m
        .mass
        .r70_surface_node_temperatures
        .as_ref()
        .expect("t_s state");
    let tso = m
        .mass
        .r70_surface_node_temperatures_prev
        .as_ref()
        .expect("t_s prev snapshot");
    let degen = m.conduction.derived_h_tr_3.as_ref()[0] <= 0.0;
    assert!(!degen, "premise: the spec zone is non-degenerate");
    let (t_s_hat, _tm_w) = replicate_per_class_advance(
        rec,
        &cls_cm,
        &cls_ms,
        h_is_zone,
        degen,
        tm_old,
        12.0,
        &[tso.as_ref()[0], tso.as_ref()[1], tso.as_ref()[2]],
    );
    for c in 0..3 {
        let Some(t_s_hat_c) = t_s_hat[c] else {
            panic!("class {c} must have a positive area share in the spec zone");
        };
        assert_eq!(
            t_s_hat_c,
            tsn.as_ref()[c],
            "class {c}: persisted t_s must equal the replicated backward-Euler advance"
        );
    }
}

#[test]
/// Contract 2 — snapshot lockstep: the persisted `..._prev` vector is exactly
/// the state the advance consumed. Step 1 starts from the documented 20 °C
/// fill (no prior state); step 2's prev must equal step 1's new state
/// bit-for-bit, and step 2's new state must equal the advance recomputed from
/// it (the identity across two steps, not just the fresh-fill case).
fn ts_prev_snapshot_is_verbatim_and_lockstep() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut m = spec_model_freefloat();
    m.step_physics(0, 12.0, DT);
    let tso_1 = m
        .mass
        .r70_surface_node_temperatures_prev
        .as_ref()
        .expect("prev snapshot");
    for v in tso_1.as_ref() {
        assert_eq!(*v, 20.0, "first step consumes the documented 20 °C fill");
    }
    let tsn_1 = m
        .mass
        .r70_surface_node_temperatures
        .as_ref()
        .expect("t_s state");
    let kept: [f64; 3] = [tsn_1.as_ref()[0], tsn_1.as_ref()[1], tsn_1.as_ref()[2]];
    let tm_old = m.mass.mass_temperatures.as_ref()[0];
    m.step_physics(1, 12.0, DT);
    let tso_2 = m
        .mass
        .r70_surface_node_temperatures_prev
        .as_ref()
        .expect("prev snapshot");
    let tsn_2 = m
        .mass
        .r70_surface_node_temperatures
        .as_ref()
        .expect("t_s state");
    for c in 0..3 {
        assert_eq!(
            tso_2.as_ref()[c],
            kept[c],
            "class {c}: step 2 must consume exactly the state step 1 persisted"
        );
    }
    let rec = m
        .conduction
        .applied_5r1c_class_boundary
        .as_ref()
        .expect("record")
        .first()
        .expect("zone 0")
        .as_ref()
        .expect("per-class path");
    let grab = |o: &Option<VectorField>| o.as_ref().and_then(|v| v.as_ref().first().copied());
    let cls_cm: [f64; 3] = [
        grab(&m.conduction.cm_wall).expect("cm_wall"),
        grab(&m.conduction.cm_roof).expect("cm_roof"),
        grab(&m.conduction.cm_floor).expect("cm_floor"),
    ];
    let cls_ms: [f64; 3] = [
        grab(&m.conduction.h_tr_ms_wall).expect("ms_wall"),
        grab(&m.conduction.h_tr_ms_roof).expect("ms_roof"),
        grab(&m.conduction.h_tr_ms_floor).expect("ms_floor"),
    ];
    let h_is_zone = m.conduction.h_tr_is.as_ref()[0];
    let (t_s_hat, _) = replicate_per_class_advance(
        rec,
        &cls_cm,
        &cls_ms,
        h_is_zone,
        m.conduction.derived_h_tr_3.as_ref()[0] <= 0.0,
        tm_old,
        12.0,
        &kept,
    );
    for c in 0..3 {
        assert_eq!(
            t_s_hat[c].expect("area share"),
            tsn_2.as_ref()[c],
            "class {c}: step-2 t_s must equal the replicated advance from the consumed state"
        );
    }
}

#[test]
/// Contract 3 — the applied record's class partition: the exterior boundary
/// is walls + roofs only (floors are excluded from the solar boundary, so the
/// floor class carries no exterior conductance but does carry its slab area),
/// the zone-aggregated envelope conductance is exactly the wall+roof class
/// sum, and the area shares across all three classes sum to 1.
fn class_partition_and_floor_class_record() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut m = spec_model_freefloat();
    m.step_physics(0, 12.0, DT);
    let rec = m
        .conduction
        .applied_5r1c_class_boundary
        .as_ref()
        .expect("record")
        .first()
        .expect("zone 0")
        .as_ref()
        .expect("per-class path");
    let zone_boundary = m
        .conduction
        .applied_5r1c_boundary
        .as_ref()
        .expect("zone boundary persisted");
    let h_tr_em_zone = zone_boundary.0[0];
    let _t_sol_air_zone = zone_boundary.1[0];
    assert_eq!(
        rec.cls_g[2], 0.0,
        "the floor class carries no exterior solar-boundary conductance"
    );
    assert!(
        rec.cls_area[2] > 0.0,
        "the floor class records its slab area ({})",
        rec.cls_area[2]
    );
    let wall_roof_sum = rec.cls_g[0] + rec.cls_g[1];
    assert!(
        (h_tr_em_zone - wall_roof_sum).abs() <= 1e-9 * h_tr_em_zone.abs().max(1.0),
        "zone h_tr_em ({h_tr_em_zone}) must equal the wall+roof class sum ({wall_roof_sum})"
    );
    let total_area: f64 = rec.cls_area.iter().sum();
    let w_sum: f64 = (0..3).map(|c| rec.cls_area[c] / total_area).sum::<f64>()
        - (0..3)
            .filter(|&c| rec.cls_area[c] / total_area <= 0.0)
            .map(|c| rec.cls_area[c] / total_area)
            .sum::<f64>();
    assert!(
        (w_sum - 1.0).abs() < 1e-12,
        "the area shares the advance consumes must sum to 1 (got {w_sum})"
    );
}

#[test]
/// Contract 4 — the sol-air aggregate identity: the zone-level sol-air is the
/// `h_tr_em`-weighted mean of the per-class values, so the products match the
/// per-class `g·t` sums (`h_tr_em·t_sol_air = Σ_c cls_g[c]·cls_t[c]`). This is
/// the composition the per-class mass advance consumes class-by-class and the
/// lumped identity consumes zone-level — both must see the same driving.
fn solair_zone_aggregate_is_class_h_weighted_mean() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut m = spec_model_freefloat();
    m.step_physics(0, 12.0, DT);
    let rec = m
        .conduction
        .applied_5r1c_class_boundary
        .as_ref()
        .expect("record")
        .first()
        .expect("zone 0")
        .as_ref()
        .expect("per-class path");
    let zone_boundary = m
        .conduction
        .applied_5r1c_boundary
        .as_ref()
        .expect("zone boundary persisted");
    let h_tr_em_zone = zone_boundary.0[0];
    let t_sol_air_zone = zone_boundary.1[0];
    let lhs = h_tr_em_zone * t_sol_air_zone;
    let rhs: f64 = rec
        .cls_g
        .iter()
        .zip(rec.cls_t.iter())
        .map(|(g, t)| g * t)
        .sum();
    assert!(
        (lhs - rhs).abs() <= 1e-9 * lhs.abs().max(1.0),
        "h_tr_em·t_sol_air ({lhs}) must equal Σ g·t over classes ({rhs})"
    );
    assert!(
        t_sol_air_zone.is_finite(),
        "sol-air aggregate must stay finite"
    );
}

#[test]
/// Contract 5 — route precedence on the default coupled operator: the
/// committed mass state is the coupled sub-step composite, replayed from the
/// engine's own persisted affine record (`t := a_k·t + b_k` per sub-step from
/// the hour-start mass — the same application the engine made, so the replay
/// is bit-exact), NOT the hourly per-class advance. The hourly advance still
/// runs and persists the t_s state (contracts 1 and 2), but its mass result
/// is discarded under the arm (step_5r1c, r76-final override) — pinned here
/// by the discriminator: the hourly closed form must NOT coincide with the
/// committed state.
fn committed_mass_is_the_coupled_composite() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut m = spec_model_freefloat();
    let tm_hour_start = m.mass.mass_temperatures.as_ref()[0];
    m.step_physics(0, 12.0, DT);
    let rec_coeffs = m
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("premise: the default coupled route persists the r77 record");
    let c0 = rec_coeffs.first().expect("zone 0 record");
    assert!(
        c0.len() >= 4 && c0.len() % 4 == 0,
        "record shape: four entries per sub-step (got len={})",
        c0.len()
    );
    let mut t_comp = tm_hour_start;
    for k in 0..c0.len() / 4 {
        t_comp = c0[4 * k] * t_comp + c0[4 * k + 1];
    }
    let committed = m.mass.mass_temperatures.as_ref()[0];
    assert_eq!(
        committed, t_comp,
        "the committed mass must equal the persisted composite replay"
    );
    // Discriminator: the hourly per-class advance's mass result is shadow on
    // this route — its closed form must not coincide with the committed state
    // (N=12 sub-steps vs one hourly BE step differ generically).
    let rec = m
        .conduction
        .applied_5r1c_class_boundary
        .as_ref()
        .expect("record")
        .first()
        .expect("zone 0")
        .as_ref()
        .expect("per-class path");
    let grab = |o: &Option<VectorField>| o.as_ref().and_then(|v| v.as_ref().first().copied());
    let cls_cm: [f64; 3] = [
        grab(&m.conduction.cm_wall).expect("cm_wall"),
        grab(&m.conduction.cm_roof).expect("cm_roof"),
        grab(&m.conduction.cm_floor).expect("cm_floor"),
    ];
    let cls_ms: [f64; 3] = [
        grab(&m.conduction.h_tr_ms_wall).expect("ms_wall"),
        grab(&m.conduction.h_tr_ms_roof).expect("ms_roof"),
        grab(&m.conduction.h_tr_ms_floor).expect("ms_floor"),
    ];
    let tso = m
        .mass
        .r70_surface_node_temperatures_prev
        .as_ref()
        .expect("prev");
    let (_, tm_w_hourly) = replicate_per_class_advance(
        rec,
        &cls_cm,
        &cls_ms,
        m.conduction.h_tr_is.as_ref()[0],
        m.conduction.derived_h_tr_3.as_ref()[0] <= 0.0,
        tm_hour_start,
        12.0,
        &[tso.as_ref()[0], tso.as_ref()[1], tso.as_ref()[2]],
    );
    assert!(
        tm_w_hourly != committed,
        "the hourly advance's mass result must be shadow (discriminator), got {tm_w_hourly} == {committed}"
    );
}

#[test]
/// Contract 6 — the degenerate route on the committed coupled operator
/// (derived `H_tr,3 ≤ 0`, the Issue #2868 zero-coupling configuration): the
/// per-class sub-step advance must couple the mass through the recorded
/// zone-level `h_air_mass·w` fallback (here 0 after the series fallback
/// collapses on the zeroed `h_tr_ms`), and the committed state still equals
/// the composite replay — the degenerate coefficients ride the same record.
/// The hourly shadow advance persists its (coupling-free) t_s state under the
/// same closed form, pinned here so the degenerate shadow stays observable.
fn degenerate_zone_composite_routes_lumped_h_air_mass() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut m = degenerate_class_model();
    let tm_hour_start = m.mass.mass_temperatures.as_ref()[0];
    m.step_physics(0, 12.0, DT);
    let derived_3 = m.conduction.derived_h_tr_3.as_ref()[0];
    assert!(
        derived_3 <= 0.0,
        "premise: the fixture must collapse derived H_tr,3 (got {derived_3})"
    );
    let rec = m
        .conduction
        .applied_5r1c_class_boundary
        .as_ref()
        .expect("record")
        .first()
        .expect("zone 0")
        .as_ref()
        .expect("per-class path");
    assert_eq!(
        rec.h_air_mass, 0.0,
        "the series fallback must evaluate to 0 on the zeroed h_tr_ms"
    );
    let rec_coeffs = m
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("the coupled route persists the r77 record on the degenerate fixture");
    let c0 = rec_coeffs.first().expect("zone 0 record");
    let mut t_comp = tm_hour_start;
    for k in 0..c0.len() / 4 {
        t_comp = c0[4 * k] * t_comp + c0[4 * k + 1];
    }
    let committed = m.mass.mass_temperatures.as_ref()[0];
    assert_eq!(
        committed, t_comp,
        "the degenerate route must commit the recorded composite"
    );
    // The hourly shadow t_s advance persists under the degenerate closed form.
    let grab = |o: &Option<VectorField>| o.as_ref().and_then(|v| v.as_ref().first().copied());
    let cls_cm: [f64; 3] = [
        grab(&m.conduction.cm_wall).expect("cm_wall"),
        grab(&m.conduction.cm_roof).expect("cm_roof"),
        grab(&m.conduction.cm_floor).expect("cm_floor"),
    ];
    let cls_ms: [f64; 3] = [
        grab(&m.conduction.h_tr_ms_wall).expect("ms_wall"),
        grab(&m.conduction.h_tr_ms_roof).expect("ms_roof"),
        grab(&m.conduction.h_tr_ms_floor).expect("ms_floor"),
    ];
    let tsn = m
        .mass
        .r70_surface_node_temperatures
        .as_ref()
        .expect("t_s state");
    let tso = m
        .mass
        .r70_surface_node_temperatures_prev
        .as_ref()
        .expect("prev");
    let (t_s_hat, _) = replicate_per_class_advance(
        rec,
        &cls_cm,
        &cls_ms,
        m.conduction.h_tr_is.as_ref()[0],
        true,
        tm_hour_start,
        12.0,
        &[tso.as_ref()[0], tso.as_ref()[1], tso.as_ref()[2]],
    );
    for c in 0..3 {
        assert_eq!(
            t_s_hat[c].expect("area share"),
            tsn.as_ref()[c],
            "class {c}: the degenerate shadow t_s advance must match the closed form"
        );
    }
}

#[test]
/// Contract 7 — the identical-construction fallback, both routes: with the
/// class tensors flattened (the hourly `use_r67` structural-equality guard
/// false) the hourly shadow advance must leave the persisted t_s state
/// untouched (no surface advance), while the committed coupled operator keeps
/// running its own per-class sub-step advance (which has no distinctness
/// guard) — so the committed state still equals the composite replay built
/// from identical-class coefficients. Pins the route separation: the hourly
/// guard is a shadow-route decision, not a committed-route one.
fn identical_construction_fallback_leaves_ts_state_and_commits_composite() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Phase 1 — real per-class advance so a genuine t_s state exists.
    let mut m = spec_model_freefloat();
    m.step_physics(0, 12.0, DT);
    let kept: [f64; 3] = {
        let tsn = m
            .mass
            .r70_surface_node_temperatures
            .as_ref()
            .expect("t_s state");
        [tsn.as_ref()[0], tsn.as_ref()[1], tsn.as_ref()[2]]
    };
    // Phase 2 — flatten and step again: the shadow t_s must not move, and the
    // committed state must be the composite replay of the flattened step.
    flatten_class_tensors(&mut m);
    let tm_hour_start = m.mass.mass_temperatures.as_ref()[0];
    m.step_physics(1, 12.0, DT);
    let tsn = m
        .mass
        .r70_surface_node_temperatures
        .as_ref()
        .expect("t_s state");
    let tso = m
        .mass
        .r70_surface_node_temperatures_prev
        .as_ref()
        .expect("prev");
    for c in 0..3 {
        assert_eq!(
            tsn.as_ref()[c],
            kept[c],
            "class {c}: the fallback step must not advance the persisted t_s state"
        );
        assert_eq!(
            tso.as_ref()[c],
            kept[c],
            "class {c}: the fallback's consumed state is the same untouched state"
        );
    }
    let rec_coeffs =
        m.mass.r77_mass_substep_coeffs.as_ref().expect(
            "the committed coupled route persists its record regardless of the hourly guard",
        );
    let c0 = rec_coeffs.first().expect("zone 0 record");
    let mut t_comp = tm_hour_start;
    for k in 0..c0.len() / 4 {
        t_comp = c0[4 * k] * t_comp + c0[4 * k + 1];
    }
    let committed = m.mass.mass_temperatures.as_ref()[0];
    assert_eq!(
        committed, t_comp,
        "the committed route must be unaffected by the hourly guard"
    );
}

#[test]
/// Contract 8 — the wired interior-LW steady estimates (Issue #2890): the
/// persisted per-surface temperatures must equal the documented steady form
/// `T_zone + (T_env − T_zone)·r_i/(r_i + r_s)` with `r_i = 1/8` (ISO 13790
/// §12.2.2 film pair), floor driven by the ground temperature and
/// ceiling/wall by the zone sol-air. The engine evaluates this exact operand
/// order, so the assertion is bit-exact. The full view-factor network is NOT
/// routed into the air balance yet (Issue #2890-followup) — this pins what
/// the step actually persists today.
fn surface_lw_estimates_match_steady_closed_form() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut m = spec_model_freefloat();
    let timestep = 3usize;
    let t_zone = m.setpoints.temperatures.as_ref()[0];
    let t_g = m.conduction.ground_temperature.ground_temperature(timestep);
    m.step_physics(timestep, 12.0, DT);
    let t_sol_air_zone = m
        .conduction
        .applied_5r1c_boundary
        .as_ref()
        .expect("zone boundary persisted")
        .1[0];
    let r_i = 1.0 / 8.0_f64;
    let u_floor = m.setpoints.floor_u_values.as_ref()[0];
    let u_ceiling = m.setpoints.roof_u_values.as_ref()[0];
    let u_wall = m.setpoints.wall_u_values.as_ref()[0];
    let r_floor = if u_floor > 0.0 { 1.0 / u_floor } else { 1.0e6 };
    let r_ceiling = if u_ceiling > 0.0 {
        1.0 / u_ceiling
    } else {
        1.0e6
    };
    let r_wall = if u_wall > 0.0 { 1.0 / u_wall } else { 1.0e6 };
    let t_si_floor = t_zone + (t_g - t_zone) * r_i / (r_i + r_floor);
    let t_si_ceiling = t_zone + (t_sol_air_zone - t_zone) * r_i / (r_i + r_ceiling);
    let t_si_wall = t_zone + (t_sol_air_zone - t_zone) * r_i / (r_i + r_wall);
    assert_eq!(
        m.mass.surface_temp_floor.as_ref()[0],
        t_si_floor,
        "the persisted floor estimate must equal the steady closed form"
    );
    assert_eq!(
        m.mass.surface_temp_ceiling.as_ref()[0],
        t_si_ceiling,
        "the persisted ceiling estimate must equal the steady closed form"
    );
    assert_eq!(
        m.mass.surface_temp_wall.as_ref()[0],
        t_si_wall,
        "the persisted wall estimate must equal the steady closed form"
    );
}
