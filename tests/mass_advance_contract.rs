//! Physics TDD blitz — slice 1 (loop round 85): mass-node advance contract
//! suite.
//!
//! Component under contract: the 5R1C coupled sub-step **mass-node advance**
//! (step_5r1c; lumped CN / FE-fallback since round 78, per-class composite
//! arm default since round 85, `FLX_R84_PERCLASS` falsy-token opt-out).
//!
//! Component input/output inventory (blitz slice 1 scope):
//! - inputs: `tm_prev` (mass state), `cm` (mass capacitance; per class
//!   `cm_wall/cm_roof/cm_floor` when present), exterior driving per class
//!   (`t_ext_c` sol-air; floor sees outdoor dry-bulb), interior coupling
//!   (`t_i_sub` from the coupled air sub-step), `h_vent` (R77 folded term),
//!   `phi_m` (mass-side gains snapshot), `dt_m` (sub-step seconds);
//! - outputs: `tm_new` (committed mass state), the r77 affine record
//!   `(a, b, den)` consumed by the invariant-checker composite mirror
//!   (per-class arm: capacitance-weighted composite of the per-class maps);
//! - contracts pinned here (through the public `step_physics` seam, no
//!   internals): equilibrium invariance, relaxation bounds (no overshoot),
//!   conductance monotonicity, and per-class-vs-lumped agreement under
//!   uniform night driving.
//!
//! These are record pins, not validator bands: a violation means the mass
//! advance moved and must be re-examined, never tuned around.

use fluxion::batch_oracle::BatchOracle;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use std::sync::Mutex;

// Issue #3453 convention: this suite flips `FLX_R84_PERCLASS` in-process
// (contract 4's A/B), so every env mutation is serialized through this
// file-local mutex and the mutating test body holds the guard for its
// whole lifetime.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// A spec-built Case 600 baseline (per-class construction tensors present,
/// so the default arm engages). Solar/loads neutralized so the driving is
/// the outdoor dry-bulb alone.
fn spec_model_night() -> ThermalModel<VectorField> {
    let mut m = BatchOracle::default_model().expect("spec model builds");
    let n = m.hvac.num_zones;
    m.setpoints.loads = VectorField::new(vec![0.0; n]);
    m.solar.solar_gains = VectorField::new(vec![0.0; n]);
    m.solar.opaque_solar_gains = VectorField::new(vec![0.0; n]);
    m
}

/// A hand-rolled one-zone model (no per-class construction tensors, so the
/// guard keeps it on the lumped sub-step advance — the pre-85 path).
/// Parameters follow the CZ3 hand-built pattern (48 m², 150 kJ/m²K, roof-only
/// em side, 12 m² window, 0.5 ach vent): the bare `ThermalModel::new`
/// defaults are numerically degenerate (documented divergence, both flag
/// states) and cannot carry a contract fixture.
fn hand_rolled_model() -> ThermalModel<VectorField> {
    let mut m = ThermalModel::new(1);
    let floor_area = 48.0;
    let ceiling_height = 2.7;
    m.setpoints.zone_area = VectorField::from_scalar(floor_area, 1);
    m.setpoints.ceiling_height = VectorField::from_scalar(ceiling_height, 1);
    m.setpoints.infiltration_rate = VectorField::from_scalar(0.5, 1);
    m.setpoints.heating_setpoint = 20.0;
    m.setpoints.cooling_setpoint = 27.0;
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
    m.setpoints.loads = VectorField::new(vec![0.0; 1]);
    m.solar.solar_gains = VectorField::new(vec![0.0; 1]);
    m.solar.opaque_solar_gains = VectorField::new(vec![0.0; 1]);
    m.update_optimization_cache();
    m
}

#[test]
/// Contract 1 — equilibrium invariance: at fixed driving the coupled
/// air+mass relaxation must be monotone-decelerating once the fast air node
/// has led (increments non-increasing from step 3 on) and must converge
/// toward the driving temperature. Any algebra error in the mass advance
/// (lumped or per-class) breaks the affine map's `|a| < 1` through the real
/// path.
fn mass_advance_equilibrium_is_invariant() {
    let dt = 3600.0;
    for (label, make) in [
        ("lumped", hand_rolled_model as fn() -> _),
        ("per-class", spec_model_night),
    ] {
        let mut m = make();
        let t_ext = 10.0;
        let mut incs = Vec::new();
        let mut tm = m.mass.mass_temperatures[0];
        for h in 0..14 {
            m.step_physics(h, t_ext, dt);
            let now = m.mass.mass_temperatures[0];
            incs.push((now - tm).abs());
            tm = now;
        }
        for w in incs.windows(2).skip(2) {
            assert!(
                w[1] <= w[0] + 1e-9,
                "{label}: relaxation accelerates at fixed driving after the air lead (incs tail: {incs:?})"
            );
        }
        assert!(
            incs.iter().all(|d| d.is_finite()),
            "{label}: non-finite increment"
        );
        assert!(
            tm < 21.0 && tm > t_ext,
            "{label}: mass must approach the driving temperature from its initial 21 °C (end={tm})"
        );
        assert!(
            incs[incs.len() - 1] < incs[3],
            "{label}: increments must decay through the window (incs: {incs:?})"
        );
    }
}

#[test]
/// Contract 2 — relaxation bounds: from a warm mass at fixed cold driving,
/// one step must move toward the driving temperature and not past it
/// (CN with den>0 is a convex combination; the FE fallback must respect the
/// same bound at hourly dt). This is the contract the tensor-less
/// replicate-fallback violated by tens of kelvin in the round-85 A/B.
fn mass_advance_relaxation_is_bounded() {
    let dt = 3600.0;
    for (label, make) in [
        ("lumped", hand_rolled_model as fn() -> _),
        ("per-class", spec_model_night),
    ] {
        let mut m = make();
        // Warm start: settle near 24 °C by stepping at 24 °C, then drive at
        // −10 °C and check bounded approach for several steps.
        for h in 0..6 {
            m.step_physics(h, 24.0, dt);
        }
        let warm = m.mass.mass_temperatures[0];
        let mut prev = warm;
        for h in 6..12 {
            m.step_physics(h, -10.0, dt);
            let now = m.mass.mass_temperatures[0];
            assert!(
                now <= prev + 1e-9 && now >= -10.0 - 1e-6,
                "{label}: step {h} left the relaxation hull (prev={prev}, now={now})"
            );
            prev = now;
        }
        assert!(
            warm - prev > 0.0,
            "{label}: mass must relax toward the cold driving (warm={warm}, end={prev})"
        );
    }
}

#[test]
/// Contract 3 — conductance monotonicity: doubling the envelope conductance
/// must not slow the one-step relaxation toward the same driving
/// temperature. Pins that the mass advance consumes `h_tr_em`-side driving
/// with the expected sign.
fn mass_advance_is_monotone_in_conductance() {
    let dt = 3600.0;
    let mut base = spec_model_night();
    for h in 0..4 {
        base.step_physics(h, 24.0, dt);
    }
    let mut leaky = spec_model_night();
    leaky.conduction.h_tr_em = leaky.conduction.h_tr_em.clone() * 2.0;
    leaky.update_optimization_cache();
    for h in 0..4 {
        leaky.step_physics(h, 24.0, dt);
    }
    let t_base = base.mass.mass_temperatures[0];
    let t_leak = leaky.mass.mass_temperatures[0];
    assert!(
        t_leak >= t_base - 1e-9,
        "higher envelope conductance must relax the mass at least as fast toward 24 °C (base={t_base}, leaky={t_leak})"
    );
}

#[test]
/// Contract 4 — per-class arm agrees with the lumped advance on the same
/// spec model under uniform night driving. Both discretize the same
/// continuous relaxation; they may differ by discretization only, not by
/// model. Generous hourly-step tolerance (0.5 K) — this catches
/// structural divergences (the tensor-less path moved states by tens of K),
/// not discretization noise.
fn per_class_arm_agrees_with_lumped_under_uniform_driving() {
    let dt = 3600.0;
    let mut perclass = spec_model_night();
    for h in 0..4 {
        perclass.step_physics(h, 24.0, dt);
    }
    let mut lumped = spec_model_night();
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("FLX_R84_PERCLASS", "0");
    for h in 0..4 {
        lumped.step_physics(h, 24.0, dt);
    }
    std::env::remove_var("FLX_R84_PERCLASS");
    drop(_env_guard);
    let a = perclass.mass.mass_temperatures[0];
    let b = lumped.mass.mass_temperatures[0];
    assert!(
        (a - b).abs() < 0.5,
        "per-class and lumped mass advances diverged under uniform driving: {a} vs {b}"
    );
}
