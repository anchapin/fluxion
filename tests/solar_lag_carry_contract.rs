//! Solar-lag carry contracts (loop round 110) — the lag state's persisted
//! transitions, the named next blitz slice after round 109's surface-pool
//! suite (`docs/implementation-plans/checker-air-node-mirror-plan.md` §7:
//! "Lag filter: ... its blitz slice shrinks to pinning the persisted-state
//! transitions (the lag state's own carry contract)").
//!
//! Component under contract: the Issue #1860 first-order solar-lag state
//! (`MassState::solar_lag`) on the default route (coupled N=12, R88 share,
//! T5138/R93/R94 defaults, R90 unfiltered default):
//!
//! - (a) the per-sub-step low-pass recurrence
//!   `lag := lag·decay + input·(1−decay)` with `decay = exp(−dt_sub/τ_lag)`,
//!   `τ_lag = sqrt(τ_air·τ_mass)`, `τ_air = c_air/(den_i/term_rest_1)` on the
//!   den the air advance actually consumed (record entry 3), per sub-step;
//! - (b) the hour-boundary carry: the final sub-step state is written back to
//!   `mass.solar_lag` (step_5r1c.rs "Persist final solar-lag state") and the
//!   next hour seeds the scratch pool from it — the carry this suite pins
//!   bit-exactly, hour by hour, off each hour's own record;
//! - (c) the input formation on both live branches (round-93/94 semantics):
//!   T5138-live zones carry the internal-gain §C.3 air share
//!   `h_tr_is·(load·rad_frac)·(1−s_share)/term_rest_1`; lumped zones carry
//!   the full pool `phi_st` at the zone-level share — with solar zeroed the
//!   pool equals the internal-gain component, so the branches discriminate
//!   through their DIFFERENT shares;
//! - (d) the release side: input 0 → pure geometric decay through the same
//!   per-sub-step arithmetic;
//! - (e) the night-vent record identities (the §7 vent remainder): the vent
//!   rides ONLY the air node's consumed denominator (record Δden =
//!   `term_rest_1·h_ve_night`) and never enters the mass record's
//!   `(a, bl, h3od)` — the air↔mass coupling the §7 note names — with the
//!   mass-side path retired (`h_vent_mass_zone = 0`, R86 default) and its
//!   falsy archaeology token restoring it.
//!
//! Round-85/109 suite shape: Python replication first
//! (`/tmp/r110_lag_replication.py`, dev-time evidence), record pins with
//! bit-exact teeth, file-local ENV_LOCK serialization (the engine reads the
//! env flags per step). These are record pins, not validator bands: a
//! violation means the lag/vent routing moved and must be re-examined,
//! never tuned around.

use fluxion::batch_oracle::BatchOracle;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::NightVentilation;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

const DT: f64 = 3600.0;
/// Internal-gain driving for the lag fixtures, W/m² — the only live source
/// (solar and opaque solar zeroed), so the filter input is the internal-gain
/// §C.3 air share alone.
const LOADS_WM2: f64 = 10.0;

/// A spec-built Case 600 baseline (per-class construction tensors present,
/// the T5138-live default arm) under free-float setpoints with constant
/// internal gains: the ideal HVAC never engages and the lag input is the
/// gain-driven §C.3 air share alone.
fn lag_model() -> ThermalModel<VectorField> {
    let mut m = BatchOracle::default_model().expect("spec model builds");
    let n = m.hvac.num_zones;
    m.setpoints.heating_setpoint = -50.0;
    m.setpoints.cooling_setpoint = 100.0;
    m.hvac.hvac_heating_capacity = 0.0;
    m.hvac.hvac_cooling_capacity = 0.0;
    m.setpoints.loads = VectorField::from_scalar(LOADS_WM2, n);
    m.solar.solar_gains = VectorField::new(vec![0.0; n]);
    m.solar.opaque_solar_gains = VectorField::new(vec![0.0; n]);
    m
}

/// A hand-rolled one-zone model (no per-class construction tensors, so the
/// zone is NOT T5138-live: the lag input takes the lumped full-pool branch at
/// the zone-level share). Parameters follow the CZ3 hand-built pattern used
/// by `mass_advance_contract.rs`.
fn lag_model_lumped() -> ThermalModel<VectorField> {
    let mut m = ThermalModel::new(1);
    let floor_area = 48.0;
    let ceiling_height = 2.7;
    m.setpoints.zone_area = VectorField::from_scalar(floor_area, 1);
    m.setpoints.ceiling_height = VectorField::from_scalar(ceiling_height, 1);
    m.setpoints.infiltration_rate = VectorField::from_scalar(0.5, 1);
    m.setpoints.heating_setpoint = -50.0;
    m.setpoints.cooling_setpoint = 100.0;
    m.solar.window_u_value = 3.0;
    m.conduction.h_tr_em = VectorField::from_scalar(0.32 * floor_area, 1);
    m.conduction.h_tr_w = VectorField::from_scalar(3.0 * 12.0, 1);
    m.conduction.h_tr_floor = VectorField::from_scalar(0.15 * floor_area, 1);
    let volume = floor_area * ceiling_height;
    let h_ve = 1.2 * 1000.0 * (0.5 * volume / 3600.0);
    m.conduction.h_ve = VectorField::from_scalar(h_ve, 1);
    m.mass.thermal_capacitance = VectorField::from_scalar(floor_area * 150_000.0, 1);
    m.setpoints.temperatures = VectorField::from_scalar(21.0, 1);
    m.mass.mass_temperatures = VectorField::from_scalar(21.0, 1);
    m.setpoints.loads = VectorField::from_scalar(LOADS_WM2, 1);
    m.solar.solar_gains = VectorField::new(vec![0.0; 1]);
    m.solar.opaque_solar_gains = VectorField::new(vec![0.0; 1]);
    m.hvac.hvac_heating_capacity = 0.0;
    m.hvac.hvac_cooling_capacity = 0.0;
    m.update_optimization_cache();
    m
}

/// The lag fixture with a night-ventilation schedule attached
/// (`capacity_m3_h`, start hour, end hour); `None` removes the fan entirely.
fn vent_model(vent: Option<NightVentilation>) -> ThermalModel<VectorField> {
    let mut m = lag_model();
    m.hvac.night_ventilation = vent;
    m
}

/// The r88 per-class share `Σ_c w_c·s_c`, `s_c = h_ms_c/(h_ms_c + h_is·w_c)`,
/// replicated verbatim from the engine's operand order (the share the lag
/// input scales by on T5138-live zones).
fn s_share_per_class(cls_area: &[f64; 3], cls_ms: &[f64; 3], h_is_i: f64) -> f64 {
    let total: f64 = cls_area.iter().sum();
    let mut share = 0.0_f64;
    for c in 0..3 {
        let w = cls_area[c] / total;
        if w <= 0.0 {
            continue;
        }
        let h_is_c = h_is_i * w;
        let s_c = if cls_ms[c] + h_is_c > 0.0 {
            cls_ms[c] / (cls_ms[c] + h_is_c)
        } else {
            0.0
        };
        share += w * s_c;
    }
    share
}

/// The zone-level lumped share `h_ms/(h_ms + h_is)` (tensor-less zones),
/// engine operand order.
fn s_share_lumped(h_ms_i: f64, h_is_i: f64) -> f64 {
    if h_ms_i + h_is_i > 0.0 {
        h_ms_i / (h_ms_i + h_is_i)
    } else {
        0.0
    }
}

/// The lag input's internal-gain §C.3 air-share closed form,
/// `h_tr_is·(load_w·rad_frac)·(1−s_share)/term_rest_1` — on the T5138-live
/// default this is `h_tr_is·phi_st_int/term_rest_1` scaled by the R88 share;
/// on a lumped zone with solar zeroed `phi_st` is the same
/// `load_w·rad_frac` (the `remaining_sol·st_sol_frac` term is exactly 0), so
/// one builder serves both branches and the SHARE is what discriminates.
fn lag_input_closed_form(
    h_tr_is: f64,
    load_w: f64,
    rad_frac: f64,
    s_share: f64,
    term_rest_1: f64,
) -> f64 {
    let mut x = h_tr_is * (load_w * rad_frac) / term_rest_1;
    x *= 1.0 - s_share;
    x
}

/// One hour of the engine's per-sub-step lag advance, replicated verbatim
/// (Python derivation: `/tmp/r110_lag_replication.py`). `den_seq` is the
/// record's per-sub-step consumed denominator (entry 3 of each 7-tuple —
/// the same `den_i` the lag block reads), so the R71 ramp rides in
/// hour-by-hour.
#[allow(clippy::too_many_arguments)]
fn compose_hour(
    lag_old: f64,
    den_seq: &[f64],
    term_rest_1: f64,
    c_air: f64,
    cm: f64,
    h_tr_3: f64,
    dt_sub: f64,
    lag_input: f64,
) -> f64 {
    let mut lag = lag_old;
    for &den_i in den_seq {
        let den_true = if term_rest_1 > 0.0 {
            den_i / term_rest_1
        } else {
            den_i
        };
        let tau_air = if den_true > 0.0 && c_air > 0.0 {
            c_air / den_true
        } else {
            600.0
        };
        let tau_mass = if h_tr_3 > 0.0 && cm > 0.0 {
            cm / h_tr_3
        } else {
            44000.0
        };
        let tau_lag = (tau_air * tau_mass).sqrt();
        let decay = if tau_lag > 0.0 && dt_sub > 0.0 {
            (-dt_sub / tau_lag).exp()
        } else {
            0.0
        };
        lag = lag * decay + lag_input * (1.0 - decay);
    }
    lag
}

/// The per-zone record accessor: `(n, den_seq, d_first)` — the sub-step count,
/// the consumed denominators, and the first sub-step's consumed air relax.
fn air_record<'a>(m: &'a ThermalModel<VectorField>, zone: usize) -> (usize, Vec<f64>, f64) {
    let rec = m
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("a live coupled 5R1C step must persist the air-node record");
    let row = &rec[zone];
    // Re-pinned 2026-11-13 (loop round 139): stride 7 -> 9 — the round-107
    // seven-entry layout carried [D, M, W, den, h3od, a, bl]; round 113 added
    // slot `C` (inter-zone cross-response) and round 115 slot `np` (ramp-source
    // slope) in the same push, so the live record is 9 per sub-step (the
    // 12x9 = 108-entry failure is proven pre-existing at the base tip, round
    // 138). `den` stays index 3 of each sub-step block.
    assert!(
        !row.is_empty() && row.len() % 9 == 0,
        "air record must carry 9 entries per sub-step, got {}",
        row.len()
    );
    let n = row.len() / 9;
    let den_seq = (0..n).map(|k| row[9 * k + 3]).collect::<Vec<f64>>();
    (n, den_seq, row[0])
}

/// Fixture-level lag quantities for zone 0: `(load_w, rad_frac, h_tr_is,
/// term_rest_1, c_air, cm, h_tr_3)` — every value read exactly where the
/// engine reads it.
fn lag_quantities(m: &ThermalModel<VectorField>) -> (f64, f64, f64, f64, f64, f64, f64) {
    let load_w = m.setpoints.loads.as_ref()[0] * m.setpoints.zone_area.as_ref()[0];
    let rad_frac = 1.0 - m.solar.convective_fraction;
    (
        load_w,
        rad_frac,
        m.conduction.h_tr_is.as_ref()[0],
        m.conduction.derived_term_rest_1.as_ref()[0],
        m.mass.air_thermal_capacitance.as_ref()[0],
        m.mass.thermal_capacitance.as_ref()[0],
        m.conduction.derived_h_tr_3.as_ref()[0],
    )
}

#[test]
/// Contract 1 — the carry contract (flagship): after every hour on the
/// default route, the persisted `mass.solar_lag` equals the per-sub-step
/// low-pass recurrence recomputed from the PREVIOUS hour's persisted state
/// through THAT hour's own record denominators. The engine wrote the state
/// with this exact arithmetic and operand order, so the assertion is
/// bit-exact — any drift in the decay factors, the input formation, the
/// per-sub-step den consumed, or the hour-boundary write-back/seeding breaks
/// it. Discriminating fixture: the per-class share differs from the zone
/// share on this construction, so a branch swap (T5138-live vs lumped input
/// formation) fails the pin.
fn lag_state_composes_bit_exact_and_carries_across_hours() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut m = lag_model();
    let (load_w, rad_frac, h_tr_is, tr1, c_air, cm, h_tr_3) = lag_quantities(&m);
    assert!(
        tr1 > 0.0 && c_air > 0.0 && cm > 0.0 && h_tr_3 > 0.0,
        "fixture must run the real (non-fallback) tau branches"
    );

    let mut lag_prev = m.mass.solar_lag.as_ref()[0];
    assert_eq!(lag_prev, 0.0, "the Issue #1860 lag state must start zeroed");

    for hour in 0..3_usize {
        m.step_physics(hour, 12.0, DT);
        let (n, den_seq, _d0) = air_record(&m, 0);
        let dt_sub = DT / n as f64;
        // The applied record's class areas are the r88 share's own inputs.
        let applied = m
            .conduction
            .applied_5r1c_class_boundary
            .as_ref()
            .expect("per-class record persisted")
            .first()
            .expect("zone 0 record")
            .as_ref()
            .expect("zone 0 ran the per-class path");
        let grab = |o: &Option<VectorField>| o.as_ref().and_then(|v| v.as_ref().first().copied());
        let cls_ms: [f64; 3] = [
            grab(&m.conduction.h_tr_ms_wall).expect("h_tr_ms_wall"),
            grab(&m.conduction.h_tr_ms_roof).expect("h_tr_ms_roof"),
            grab(&m.conduction.h_tr_ms_floor).expect("h_tr_ms_floor"),
        ];
        let s_class = s_share_per_class(&applied.cls_area, &cls_ms, h_tr_is);
        let s_zone = s_share_lumped(m.conduction.h_tr_ms.as_ref()[0], h_tr_is);
        assert!(
            (s_class - s_zone).abs() > 1e-12,
            "fixture premise: the per-class and zone shares must differ so the branch choice discriminates"
        );
        let input = lag_input_closed_form(h_tr_is, load_w, rad_frac, s_class, tr1);
        let lag_next = compose_hour(lag_prev, &den_seq, tr1, c_air, cm, h_tr_3, dt_sub, input);
        let persisted = m.mass.solar_lag.as_ref()[0];
        assert_eq!(
            lag_next, persisted,
            "hour {hour}: the persisted lag state must equal the composed carry \
             from the previous hour's persisted state (recomputed {lag_next}, got {persisted})"
        );
        lag_prev = persisted;
    }
    // The charge must be live and monotone toward the input: the state moved
    // and keeps relaxing in one direction.
    let l1_to_l3 = m.mass.solar_lag.as_ref()[0];
    assert!(
        l1_to_l3 != 0.0,
        "the lag state never charged — the input died"
    );
}

#[test]
/// Contract 2 — the input's physics meaning: the persisted lag state's
/// fixed point under constant driving is the filter input itself, so after
/// 48 hours the state must equal the §C.3 air-share closed form
/// `h_tr_is·(load_w·rad_frac)·(1−s_share)/term_rest_1` to 1e-9 relative.
/// This pins WHAT the state carries (the internal-gain air share through the
/// retained lag site, round-93/94 semantics): re-routing the input to the
/// full pool, dropping the R88 share, or dropping the `term_rest_1` basis
/// moves the fixed point and fails here with the mechanism still
/// bit-consistent.
fn lag_state_converges_to_the_c3_air_share_closed_form() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut m = lag_model();
    let (load_w, rad_frac, h_tr_is, tr1, _c_air, _cm, _h_tr_3) = lag_quantities(&m);
    for hour in 0..48_usize {
        m.step_physics(hour, 12.0, DT);
    }
    let applied = m
        .conduction
        .applied_5r1c_class_boundary
        .as_ref()
        .expect("per-class record persisted")
        .first()
        .expect("zone 0 record")
        .as_ref()
        .expect("zone 0 ran the per-class path");
    let grab = |o: &Option<VectorField>| o.as_ref().and_then(|v| v.as_ref().first().copied());
    let cls_ms: [f64; 3] = [
        grab(&m.conduction.h_tr_ms_wall).expect("h_tr_ms_wall"),
        grab(&m.conduction.h_tr_ms_roof).expect("h_tr_ms_roof"),
        grab(&m.conduction.h_tr_ms_floor).expect("h_tr_ms_floor"),
    ];
    let s_class = s_share_per_class(&applied.cls_area, &cls_ms, h_tr_is);
    let input = lag_input_closed_form(h_tr_is, load_w, rad_frac, s_class, tr1);
    let lag = m.mass.solar_lag.as_ref()[0];
    let rel = (lag - input).abs() / input.abs();
    assert!(
        rel <= 1e-9,
        "the lag state's fixed point must equal the §C.3 air-share input: state {lag} vs input {input} (rel {rel:.3e})"
    );
}

#[test]
/// Contract 3 — the release side and the record pairing: with the gains
/// cut, the lag input is exactly 0 and the state decays through the same
/// per-sub-step arithmetic (bit-exact against the hour's own record), and
/// the record's consumed air relax pairs with the lag route's `tau_air`
/// (both are `c_air·term_rest_1/den_i`): `D_k == exp(−dt_sub/tau_air)` to
/// 1e-12 on every sub-step, charge and release hours alike.
fn zero_input_release_composes_bit_exact_and_pairs_with_the_record() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut m = lag_model();
    let (load_w, rad_frac, h_tr_is, tr1, c_air, cm, h_tr_3) = lag_quantities(&m);
    let cls_area_of = |mm: &ThermalModel<VectorField>| {
        mm.conduction
            .applied_5r1c_class_boundary
            .as_ref()
            .expect("per-class record persisted")
            .first()
            .expect("zone 0 record")
            .as_ref()
            .expect("zone 0 ran the per-class path")
            .cls_area
    };
    let grab = |o: &Option<VectorField>| o.as_ref().and_then(|v| v.as_ref().first().copied());
    let cls_ms_of = |mm: &ThermalModel<VectorField>| -> [f64; 3] {
        [
            grab(&mm.conduction.h_tr_ms_wall).expect("h_tr_ms_wall"),
            grab(&mm.conduction.h_tr_ms_roof).expect("h_tr_ms_roof"),
            grab(&mm.conduction.h_tr_ms_floor).expect("h_tr_ms_floor"),
        ]
    };

    // Charge phase, pinned bit-exact exactly like contract 1 (same machinery).
    let mut lag_prev = 0.0_f64;
    for hour in 0..3_usize {
        m.step_physics(hour, 12.0, DT);
        let (n, den_seq, _) = air_record(&m, 0);
        let dt_sub = DT / n as f64;
        let cls_area = cls_area_of(&m);
        let cls_ms = cls_ms_of(&m);
        let s_class = s_share_per_class(&cls_area, &cls_ms, h_tr_is);
        let input = lag_input_closed_form(h_tr_is, load_w, rad_frac, s_class, tr1);
        let lag_next = compose_hour(lag_prev, &den_seq, tr1, c_air, cm, h_tr_3, dt_sub, input);
        let persisted = m.mass.solar_lag.as_ref()[0];
        assert_eq!(lag_next, persisted, "charge hour {hour} compose");
        lag_prev = persisted;
    }
    let charged = lag_prev;

    // Release phase: gains cut, the input is exactly 0.
    let n_zones = m.hvac.num_zones;
    m.setpoints.loads = VectorField::new(vec![0.0; n_zones]);
    for hour in 3..5_usize {
        m.step_physics(hour, 12.0, DT);
        let (n, den_seq, _) = air_record(&m, 0);
        let dt_sub = DT / n as f64;
        let lag_next = compose_hour(lag_prev, &den_seq, tr1, c_air, cm, h_tr_3, dt_sub, 0.0);
        let persisted = m.mass.solar_lag.as_ref()[0];
        assert_eq!(
            lag_next, persisted,
            "release hour {hour}: with the input exactly 0 the state must decay through the same recurrence"
        );
        assert!(
            persisted < lag_prev,
            "release hour {hour} must decay strictly"
        );
        // Record pairing: D_k == exp(−dt_sub / (c_air/(den_k/tr1))) on every
        // sub-step (the lag's tau_air IS the air advance's).
        let rec = m.mass.r77_air_substep_coeffs.as_ref().expect("air record");
        for k in 0..n {
            let d = rec[0][9 * k];
            let den_k = rec[0][9 * k + 3];
            let tau_air = c_air / (den_k / tr1);
            let d_pred = (-dt_sub / tau_air).exp();
            assert!(
                (d_pred - d).abs() <= 1e-12 * d,
                "record D_k must pair with the lag route's tau_air at sub-step {k}: {d_pred} vs {d}"
            );
        }
        lag_prev = persisted;
    }
    assert!(
        lag_prev < charged * 0.6,
        "two release hours must cut the state substantially (measured tau_lag ≈ 2 h ⇒ ≈ 0.38)"
    );
}

#[test]
/// Contract 4 — the lumped branch: on a tensor-less zone the lag input is
/// the FULL pool at the zone-level share (`phi_st` with solar zeroed equals
/// the internal-gain component; the share is `h_ms/(h_ms + h_is)`), and the
/// same bit-exact carry holds. The lumped share differs from the per-class
/// share of contract 1's fixture, so the two contracts pin DIFFERENT input
/// values through the same recurrence — a branch mix-up fails one of them.
fn lumped_zone_carries_full_pool_with_zone_level_share() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut m = lag_model_lumped();
    let (load_w, rad_frac, h_tr_is, tr1, c_air, cm, h_tr_3) = lag_quantities(&m);
    assert!(
        tr1 > 0.0,
        "lumped fixture must carry a live term_rest_1 basis"
    );
    let s_zone = s_share_lumped(m.conduction.h_tr_ms.as_ref()[0], h_tr_is);
    assert!(s_zone > 0.0, "lumped share must be live");
    let input = lag_input_closed_form(h_tr_is, load_w, rad_frac, s_zone, tr1);
    assert!(input != 0.0, "lumped lag input must be live");

    let mut lag_prev = 0.0_f64;
    for hour in 0..2_usize {
        m.step_physics(hour, 12.0, DT);
        let (n, den_seq, _) = air_record(&m, 0);
        let dt_sub = DT / n as f64;
        let lag_next = compose_hour(lag_prev, &den_seq, tr1, c_air, cm, h_tr_3, dt_sub, input);
        let persisted = m.mass.solar_lag.as_ref()[0];
        assert_eq!(
            lag_next, persisted,
            "lumped hour {hour}: persisted state must equal the composed carry"
        );
        lag_prev = persisted;
    }
    assert!(lag_prev != 0.0);
}

#[test]
/// Contract 5 — the vent record identities (§7 vent remainder): under live
/// night ventilation the fan rides ONLY the air node's consumed denominator
/// — `Δden == term_rest_1·h_ve_night` on every sub-step (the #824 air-side
/// coupling, ISO 13790 §C.10) — while the mass record's `(a, bl, h3od)` —
/// the air↔mass coupling the vent terms must NOT enter under the R86
/// air-side-only default — stay byte-identical to the unventilated twin,
/// and the applied per-class record carries `h_vent_mass_zone = 0`.
fn night_vent_rides_the_air_den_only_record_identities() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const FAN_M3_H: f64 = 1400.0; // the Table 5-160 night-vent fan
    let mut vented = vent_model(Some(NightVentilation::new(FAN_M3_H, 0, 0)));
    let mut still = vent_model(None);

    vented.step_physics(0, -10.0, DT);
    still.step_physics(0, -10.0, DT);

    // h_ve_night with the engine's own reads (setpoints air properties, or
    // the documented 1.2 kg/m³ / 1005 J/kgK fallbacks).
    let rho = vented
        .setpoints
        .air_density
        .as_ref()
        .first()
        .copied()
        .unwrap_or(1.2);
    let cp = vented
        .setpoints
        .heat_capacity
        .as_ref()
        .first()
        .copied()
        .unwrap_or(1005.0);
    let h_ve_night = FAN_M3_H * rho * cp / 3600.0;
    let tr1 = vented.conduction.derived_term_rest_1.as_ref()[0];
    let expected_delta = tr1 * h_ve_night;

    let (n_on, den_on, _) = air_record(&vented, 0);
    let (n_off, den_off, _) = air_record(&still, 0);
    assert_eq!(n_on, n_off, "same coupled scheme, same sub-step count");
    let rec_on = vented
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (vented)");
    let rec_off = still
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (still)");
    for k in 0..n_on {
        let delta = den_on[k] - den_off[k];
        assert!(
            (delta - expected_delta).abs() <= 1e-12 * expected_delta.abs(),
            "sub-step {k}: the vent must shift the air den by exactly term_rest_1·h_ve_night \
             (Δden {delta} vs expected {expected_delta})"
        );
        // The mass route's coefficients are byte-identical: the vent never
        // enters (a, bl, h3od) under the air-side-only default.
        for slot in [4_usize, 5, 6] {
            assert_eq!(
                rec_on[0][9 * k + slot],
                rec_off[0][9 * k + slot],
                "sub-step {k} slot {slot}: the mass-route coefficient moved under vent-on"
            );
        }
    }
    // The applied per-class record carries the retired mass-side path as 0.
    let applied = vented
        .conduction
        .applied_5r1c_class_boundary
        .as_ref()
        .expect("per-class record persisted")
        .first()
        .expect("zone 0 record")
        .as_ref()
        .expect("zone 0 ran the per-class path");
    assert_eq!(
        applied.h_vent_mass_zone, 0.0,
        "the R86 default must keep the mass-side vent path retired"
    );
}

#[test]
/// Contract 6 — the R86 archaeology token: `FLX_R86_VENT_AIR_ONLY=0`
/// restores the mass-side vent path through the real step — the applied
/// record's `h_vent_mass_zone` leaves 0 and the mass record's coefficients
/// move relative to the default run — so the retired route stays reachable
/// and observable for A/B verification.
fn r86_falsy_restores_the_mass_side_vent_record() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut restored = vent_model(Some(NightVentilation::new(1400.0, 0, 0)));
    let mut default = vent_model(Some(NightVentilation::new(1400.0, 0, 0)));

    std::env::set_var("FLX_R86_VENT_AIR_ONLY", "0");
    restored.step_physics(0, -10.0, DT);
    std::env::remove_var("FLX_R86_VENT_AIR_ONLY");
    default.step_physics(0, -10.0, DT);

    let h_vent_mass_restored = restored
        .conduction
        .applied_5r1c_class_boundary
        .as_ref()
        .expect("per-class record persisted")
        .first()
        .expect("zone 0 record")
        .as_ref()
        .expect("zone 0 ran the per-class path")
        .h_vent_mass_zone;
    assert!(
        h_vent_mass_restored > 0.0,
        "the falsy R86 token must restore a live mass-side vent coupling"
    );

    let rec_restored = restored
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (restored)");
    let rec_default = default
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (default)");
    let moved = (0..rec_default[0].len() / 9).any(|k| {
        [5_usize, 6]
            .iter()
            .any(|slot| rec_restored[0][9 * k + slot] != rec_default[0][9 * k + slot])
    });
    assert!(
        moved,
        "the restored mass-side vent must move the mass record's (a, bl) coefficients"
    );
}
