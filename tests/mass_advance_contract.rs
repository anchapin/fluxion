//! Physics TDD blitz — slice 1 (loop round 85): mass-node advance contract
//! suite.
//!
//! Component under contract: the 5R1C coupled sub-step **mass-node advance**
//! (step_5r1c; lumped CN / FE-fallback since round 78, per-class composite
//! arm default since round 85, `FLX_R84_PERCLASS` falsy-token opt-out).
//!
//! ROUND 87 UPDATE (2026-10-22, Alex-approved "I approve the vent arm",
//! #3983 §66): the mass-side night-vent coupling is now 0 BY DEFAULT on
//! both live sites (the sub-step `h_vent_i` folded into h3, and the legacy
//! hourly `h_vent_mass_zone`; `FLX_R86_VENT_AIR_ONLY=0` restores it for
//! A/B verification only) — night ventilation ships as an air-side
//! exchange (ISO 13790 §C.4 Eq. C.10; 9R4C parity since LIMIT-17). The
//! vent contracts below are state-level and pass unchanged under the
//! air-only default; their pre-round-87 shape pinned the coupled routing
//! (slice 2, loop round 86). The Tmin-band consequence is recorded as
//! §LIMIT-43 (loop round 87).
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
//!   uniform night driving. Slice 2 (loop round 86) adds the night-vent
//!   coupling contracts: vent-off inertness, vent-on monotone approach
//!   toward outdoor, and per-class-vs-lumped agreement under live vent
//!   driving (see the slice-2 header below the slice-1 contracts).
//!
//! Slice 5 (loop round 92): the solar partition site and the retained
//! lag-filter route.
//!
//! Blitz-slice verification against the tree at `ae66d514`:
//! - The Table 5-138 per-class routing (`FLX_R91_T5138`, default since
//!   round 91) is the live surface-pool mass route; the §C.3
//!   conductance-share arm (`FLX_R88_PHIST_MASS_SHARE`) is retained
//!   behind the falsy T5138 token. The route-choice consequence is
//!   pinned by contract 17; the per-class fraction constants live
//!   behind the per-step class-boundary aggregation, so their numeric
//!   content is pinned through the strict-gate baseline, not at this
//!   seam.
//! - The partition site (step_5r1c's Eq. C.5/C.6 composition) carries
//!   two public knobs retired to 0.0 by the round-91 flip-2 spec
//!   default (`FLX_R91_SOL2ST`): `solar_distribution_to_air` and
//!   `solar_beam_to_mass_fraction`. Contract 15 pins the defaults and
//!   the three-way split's conservation, including the Table 5-138
//!   window-loss sink (×0.965, `FLX_R91_RETRANS`).
//! - The R77 solar-lag filter is off the default route since round 90
//!   (`FLX_R90_NO_LAG_FILTER`); contract 18 pins the retained
//!   falsy-token route's transient direction, DC parity, and the
//!   persisted lag state, which advances with identical arithmetic in
//!   both routes.
//!
//! These are record pins, not validator bands: a violation means the mass
//! advance moved and must be re-examined, never tuned around.

use fluxion::batch_oracle::BatchOracle;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use std::sync::Mutex;

// Issue #3453 convention: this suite flips `FLX_R84_PERCLASS` in-process
// (contracts 4 and 7 A/B the flag), and the engine reads the flag PER STEP
// (`step_physics_5r1c`), so every stepping test — not only the mutating
// ones — observes another test's mutation unless the whole suite
// serializes. Every test in this file therefore holds this file-local
// mutex for its entire lifetime (loop round 86: the parallel run could
// interleave a flipping and a stepping test and fail contracts 3/5 with
// cross-contaminated state; single-threaded runs were always green).
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
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dt = 3600.0;
    let mut perclass = spec_model_night();
    for h in 0..4 {
        perclass.step_physics(h, 24.0, dt);
    }
    let mut lumped = spec_model_night();
    std::env::set_var("FLX_R84_PERCLASS", "0");
    for h in 0..4 {
        lumped.step_physics(h, 24.0, dt);
    }
    std::env::remove_var("FLX_R84_PERCLASS");
    let a = perclass.mass.mass_temperatures[0];
    let b = lumped.mass.mass_temperatures[0];
    assert!(
        (a - b).abs() < 0.5,
        "per-class and lumped mass advances diverged under uniform driving: {a} vs {b}"
    );
}

// =============================================================================
// Blitz slice 2 (loop round 86) — night-vent coupling contract.
//
// Component input/output inventory (slice 2 scope): `hvac.night_ventilation`
// (`NightVentilation`, fluxion-core; fan_capacity m³/h, operating_hours) →
// per-step `h_ve_night = fan·ρ·cp/3600` (ρ=1.2, cp=1005 fixed) when
// `is_active_at_hour` fires, routed at TWO sites on the 5R1C step: the
// air-side conductance `h_ext += h_ve_night` (Issue #824, the ISO-13790
// zone-air exchange ASHRAE 140 Table 5-160 specifies) and the mass-side
// `h_vent_mass_zone` added to the sub-step h3 coupling partitioned by area
// share w (Issues #1898/#1693 lineage; #821's own note records that setting
// it to 0 "removes the double-counting"). These tests pin the routing as it
// ships, per class and lumped; the magnitude audit lives in
// docs/investigations (loop round 86). Window (18, 7) is inactive during
// hours 7..=17; window (0, 0) is active every hour (start<=h||h<end).
// =============================================================================

/// Free-floating spec fixture: solar/loads neutralized AND HVAC disabled
/// (setpoints outside any band, capacities 0) so the driving is the outdoor
/// dry-bulb plus the night-vent routing alone — the run_blind_free_float
/// fixture pattern.
fn spec_model_free_float(
    vent: Option<fluxion::validation::NightVentilation>,
) -> ThermalModel<VectorField> {
    let mut m = spec_model_night();
    m.setpoints.heating_setpoint = -999.0;
    m.setpoints.cooling_setpoint = 999.0;
    m.hvac.hvac_heating_capacity = 0.0;
    m.hvac.hvac_cooling_capacity = 0.0;
    m.hvac.night_ventilation = vent;
    m
}

#[test]
/// Contract 5 — vent-off inertness: outside the schedule window (and at zero
/// fan capacity inside it) the night-vent routing must contribute exactly
/// nothing: both live coupling sites add 0, so the step must be numerically
/// identical to a model with no night ventilation configured.
fn night_vent_off_is_inert() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dt = 3600.0;
    // (a) schedule (18,7): hours 8..=14 are inactive — every coupling zero.
    let mut off = spec_model_free_float(None);
    let mut scheduled = spec_model_free_float(Some(fluxion::validation::NightVentilation::new(
        1400.0, 18, 7,
    )));
    for h in 8..=14 {
        off.step_physics(h, 5.0, dt);
        scheduled.step_physics(h, 5.0, dt);
        let a = off.setpoints.temperatures[0];
        let b = scheduled.setpoints.temperatures[0];
        assert!(
            a == b,
            "inactive-window hour {h} must be inert (off={a}, scheduled={b})"
        );
        let ma = off.mass.mass_temperatures[0];
        let mb = scheduled.mass.mass_temperatures[0];
        assert!(
            ma == mb,
            "inactive-window hour {h} must leave the mass advance inert (off={ma}, scheduled={mb})"
        );
    }
    // (b) zero fan capacity under an always-active window: h_ve_night = 0.
    let mut zero_fan =
        spec_model_free_float(Some(fluxion::validation::NightVentilation::new(0.0, 0, 0)));
    let mut base = spec_model_free_float(None);
    for h in 0..=6 {
        base.step_physics(h, 5.0, dt);
        zero_fan.step_physics(h, 5.0, dt);
        assert!(
            base.setpoints.temperatures[0] == zero_fan.setpoints.temperatures[0],
            "zero-capacity hour {h} must be inert"
        );
    }
}

#[test]
/// Contract 6 — vent-on monotone toward outdoor: with the fan running on a
/// cold fixed night, the free-running zone must cool strictly faster than
/// the identical model without ventilation, monotonically, and stay inside
/// the outdoor-temperature hull (ventilation exchanges with outdoor air; no
/// route may overshoot past it). Pins the sign of both live coupling sites
/// through the real path.
fn night_vent_on_pulls_toward_outdoor() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dt = 3600.0;
    let mut vented = spec_model_free_float(Some(fluxion::validation::NightVentilation::new(
        1400.0, 0, 0,
    )));
    let mut still = spec_model_free_float(None);
    for h in 0..=3 {
        vented.step_physics(h, 24.0, dt);
        still.step_physics(h, 24.0, dt);
    }
    let mut prev_zone = vented.setpoints.temperatures[0];
    let mut prev_mass = vented.mass.mass_temperatures[0];
    for h in 4..=11 {
        vented.step_physics(h, -10.0, dt);
        still.step_physics(h, -10.0, dt);
        let zone = vented.setpoints.temperatures[0];
        let mass = vented.mass.mass_temperatures[0];
        assert!(
            zone <= prev_zone + 1e-9,
            "vented zone must cool monotonically at hour {h} (prev={prev_zone}, now={zone})"
        );
        assert!(
            mass <= prev_mass + 1e-9,
            "vented mass must cool monotonically at hour {h} (prev={prev_mass}, now={mass})"
        );
        assert!(
            zone >= -10.0 - 1e-6 && mass >= -10.0 - 1e-6,
            "vent step left the outdoor hull at hour {h} (zone={zone}, mass={mass})"
        );
        let calm = still.setpoints.temperatures[0];
        assert!(
            zone < calm,
            "vented zone must be strictly cooler than unventilated at hour {h} (vented={zone}, still={calm})"
        );
        prev_zone = zone;
        prev_mass = mass;
    }
}

#[test]
/// Contract 7 — per-class vs lumped vent conservation: under live night-vent
/// driving the per-class composite arm and the lumped sub-step advance must
/// track the same state (the w-partition hands each class a share of the
/// same total coupling, Σw = 1, so they discretize the same physics, not two
/// models). Generous 0.5 K hourly tolerance — this catches structural
/// divergence in the vent partition, not discretization noise. The strict
/// Σw conservation identity itself is engine-internal (area shares); through
/// the public seam it is observable as this agreement.
fn per_class_arm_agrees_with_lumped_under_night_vent() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dt = 3600.0;
    let mut perclass = spec_model_free_float(Some(fluxion::validation::NightVentilation::new(
        1400.0, 0, 0,
    )));
    for h in 0..=2 {
        perclass.step_physics(h, 10.0, dt);
    }
    let mut lumped = spec_model_free_float(Some(fluxion::validation::NightVentilation::new(
        1400.0, 0, 0,
    )));
    std::env::set_var("FLX_R84_PERCLASS", "0");
    for h in 0..=2 {
        lumped.step_physics(h, 10.0, dt);
    }
    std::env::remove_var("FLX_R84_PERCLASS");
    let mut worst_mass = 0.0_f64;
    let mut worst_zone = 0.0_f64;
    for h in 3..=8 {
        perclass.step_physics(h, -10.0, dt);
        lumped.step_physics(h, -10.0, dt);
        let dm = (perclass.mass.mass_temperatures[0] - lumped.mass.mass_temperatures[0]).abs();
        let dz = (perclass.setpoints.temperatures[0] - lumped.setpoints.temperatures[0]).abs();
        worst_mass = worst_mass.max(dm);
        worst_zone = worst_zone.max(dz);
    }
    assert!(
        worst_mass < 0.5 && worst_zone < 0.5,
        "per-class and lumped vent routing diverged: worst mass {worst_mass:.4} K, worst zone {worst_zone:.4} K"
    );
}

// ===================================================================
// Slice 3 (loop round 88): surface-pool mass share (FLX_R88_PHIST_MASS_SHARE)
// ===================================================================
//
// Premise (code-verified this round, Item 1): on the live coupled-N path the
// per-class sub-step mass advance consumed phi_c = phi_m_c ONLY — the ISO
// 13790 §C.3 mass share of the surface pool, H_ms/(H_ms + H_is)·phi_st,
// never reached the mass node, while the FULL phi_st entered the air node
// through the R77 solar-lag filter. The arm (flag on) restores the
// conductance-share split at three paired sites: the lag-filter input keeps
// (1 − share_zone)·phi_st, the per-class mass drive adds s_c·phi_st·w_c,
// and the lumped sites add s·phi_st (with the r77 coefficient recorder
// consuming the same value so the checker mirror stays exact).
//
// Contracts pinned here (record pins, not validator bands):
// 1. default-polarity bit-identity — unset vs truthy "1" token step to
//    identical states (both take the share-on code path since the
//    round-89 default flip; the falsy "0" token is the pre-89 legacy
//    route, kept for A/B archaeology);
// 2. flag-on non-inertness with the mass-side direction pin — under pure
//    constant solar driving the mass share deposits energy in the mass, so
//    the flag-on mass is strictly warmer than flag-off, and the air state is
//    not identical (it now sees only the air share through the filter);
// 3. the same pairing on the lumped site for tensor-less zones.

/// Spec-built Case 600 with its solar gains intact (phi_st > 0) and loads
/// neutralized, so the driving is solar + outdoor dry-bulb.
fn spec_model_solar() -> ThermalModel<VectorField> {
    let mut m = BatchOracle::default_model().expect("spec model builds");
    let n = m.hvac.num_zones;
    m.setpoints.loads = VectorField::new(vec![0.0; n]);
    m.solar.opaque_solar_gains = VectorField::new(vec![0.0; n]);
    m
}

/// Constant-magnitude solar gain (W/m²) used by the slice-3 contracts.
const R88_SOLAR_WM2: f64 = 250.0;

#[test]
/// Contract 8 — default-polarity bit-identity: with
/// FLX_R88_PHIST_MASS_SHARE unset (round-89 default: share on) and with its
/// truthy "1" token, the solar-driven spec model steps to bit-identical air
/// and mass states (both runs take the share-on code path; any visible
/// difference means the flag gate leaks arithmetic). The falsy "0" token
/// takes the pre-89 legacy route and is pinned as non-identical under solar
/// driving by contract 9.
fn r88_share_default_is_bit_identical() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dt = 3600.0;
    let mut unset = spec_model_solar();
    unset.solar.solar_gains = VectorField::from_scalar(R88_SOLAR_WM2, unset.hvac.num_zones);
    for h in 0..48 {
        unset.step_physics(h, 12.0, dt);
    }
    std::env::set_var("FLX_R88_PHIST_MASS_SHARE", "1");
    let mut truthy = spec_model_solar();
    truthy.solar.solar_gains = VectorField::from_scalar(R88_SOLAR_WM2, truthy.hvac.num_zones);
    for h in 0..48 {
        truthy.step_physics(h, 12.0, dt);
    }
    std::env::remove_var("FLX_R88_PHIST_MASS_SHARE");
    let n88 = unset.hvac.num_zones;
    for z in 0..n88 {
        assert_eq!(
            unset.mass.air_temperatures[z], truthy.mass.air_temperatures[z],
            "share-on air-node state differs between unset and truthy token (zone {z})"
        );
        assert_eq!(
            unset.mass.mass_temperatures[z], truthy.mass.mass_temperatures[z],
            "share-on mass state differs between unset and truthy token (zone {z})"
        );
    }
}

#[test]
/// Contract 9 — flag-on non-inertness with the mass-side direction pin:
/// under pure constant solar driving the solar-routing default's mass is
/// strictly warmer than the legacy route's (the routed fractions deposit
/// solar in the mass), and the air state is not identical. This is the
/// existence proof that routed solar reaches the mass node on the real
/// path; magnitudes are measurement, not bands. Round-91 flip 3 update:
/// unset now takes the Table 5-138 per-class default (FLX_R91_T5138),
/// under which the R88 share flag is inert for tensor-ready zones — the
/// legacy leg pins BOTH falsy tokens explicitly, mirroring the round-89
/// note pattern.
fn r88_share_reaches_mass() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dt = 3600.0;
    // Round-91 flip-3 default: the legacy leg pins both falsy tokens —
    // unset now takes the Table 5-138 default path, and with it on the
    // R88 share flag no longer selects anything for tensor-ready zones.
    std::env::set_var("FLX_R91_T5138", "0");
    std::env::set_var("FLX_R88_PHIST_MASS_SHARE", "0");
    let mut off = spec_model_solar();
    off.solar.solar_gains = VectorField::from_scalar(R88_SOLAR_WM2, off.hvac.num_zones);
    // Free-float driving: the Case 600 ideal HVAC would clamp the air node
    // to the setpoints under this solar load, masking the air-side share;
    // open the band wide so the zone floats and the air assert is a real
    // observation.
    off.setpoints.heating_setpoint = -50.0;
    off.setpoints.cooling_setpoint = 100.0;
    for h in 0..24 {
        off.step_physics(h, 12.0, dt);
    }
    std::env::remove_var("FLX_R91_T5138");
    std::env::remove_var("FLX_R88_PHIST_MASS_SHARE");
    let mut on = spec_model_solar();
    on.solar.solar_gains = VectorField::from_scalar(R88_SOLAR_WM2, on.hvac.num_zones);
    on.setpoints.heating_setpoint = -50.0;
    on.setpoints.cooling_setpoint = 100.0;
    for h in 0..24 {
        on.step_physics(h, 12.0, dt);
    }
    std::env::remove_var("FLX_R88_PHIST_MASS_SHARE");
    let m_off = off.mass.mass_temperatures[0];
    let m_on = on.mass.mass_temperatures[0];
    assert!(
        m_on > m_off + 1e-9,
        "flag-on mass not warmer under solar driving: off={m_off:.6} on={m_on:.6}"
    );
    let a_off = off.mass.air_temperatures[0];
    let a_on = on.mass.air_temperatures[0];
    assert!(
        (a_on - a_off).abs() > 1e-9,
        "flag-on air-node state identical to flag-off: the arm is inert on the lag-filter path (off={a_off:.6} on={a_on:.6})"
    );
}

#[test]
/// Contract 10 — lumped-site pairing: a tensor-less (hand-rolled) zone on
/// the lumped sub-step advance shows the same flag-on behavior — the mass
/// gains the zone-level share s = h_ms/(h_ms + h_is)·phi_st and the air
/// keeps only (1 − s) through the filter. Pairs with the R85 guard, which
/// keeps tensor-less zones on the lumped advance.
fn r88_share_lumped_site_pairs() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dt = 3600.0;
    // Round-89 default flip: the flag-off leg pins the falsy token
    // explicitly — unset now takes the share-on default path.
    std::env::set_var("FLX_R88_PHIST_MASS_SHARE", "0");
    // The bare ThermalModel defaults send ALL transmitted solar to the air
    // (solar_distribution_to_air = 1), which zeroes phi_st and would make
    // the arm trivially inert — give the fixture a Case-600-like 0.3/0.7
    // split so the surface pool is real. Shared hand_rolled_model() is left
    // untouched (its other consumers zero solar).
    let mut off = hand_rolled_model();
    off.solar.solar_distribution_to_air = 0.3;
    off.solar.solar_gains = VectorField::from_scalar(R88_SOLAR_WM2, 1);
    off.setpoints.heating_setpoint = -50.0;
    off.setpoints.cooling_setpoint = 100.0;
    for h in 0..24 {
        off.step_physics(h, 12.0, dt);
    }
    std::env::set_var("FLX_R88_PHIST_MASS_SHARE", "1");
    let mut on = hand_rolled_model();
    on.solar.solar_distribution_to_air = 0.3;
    on.solar.solar_gains = VectorField::from_scalar(R88_SOLAR_WM2, 1);
    on.setpoints.heating_setpoint = -50.0;
    on.setpoints.cooling_setpoint = 100.0;
    for h in 0..24 {
        on.step_physics(h, 12.0, dt);
    }
    std::env::remove_var("FLX_R88_PHIST_MASS_SHARE");
    let m_off = off.mass.mass_temperatures[0];
    let m_on = on.mass.mass_temperatures[0];
    assert!(
        m_on > m_off + 1e-9,
        "lumped-site flag-on mass not warmer under solar driving: off={m_off:.6} on={m_on:.6}"
    );
    let a_off = off.mass.air_temperatures[0];
    let a_on = on.mass.air_temperatures[0];
    assert!(
        (a_on - a_off).abs() > 1e-9,
        "lumped-site flag-on air-node state identical to flag-off: the arm is inert (off={a_off:.6} on={a_on:.6})"
    );
}

// Slice 4 (loop round 89): mass-node time-coverage alignment
// ==========================================================
//
// Premise (code-verified this round, Item 2a): in the coupled sub-step
// loop the mass advance forms its Crank–Nicolson coefficients at
// dt_m = dt_sub / mass_sub (N=12 ⇒ 25 s) but executes exactly ONE CN
// step per sub-step — there is no inner loop over mass_sub — so the mass
// node integrates N·25 s = 300 s of physics per hour (1/12 of real time)
// while the air node integrates the full 3600 s. The R77 scheme's
// documented N→∞ limit (the continuous-time 5R1C solution) therefore
// does NOT hold for the mass state: its limit depends on N. The
// FLX_R89_MASS_DT_ALIGN now DEFAULTS to the fix (loop round 89 Item-2a
// flip, Alex-directed standing rule): the default runs mass_sub composed
// CN applications at dt_m = dt_sub/mass_sub — full 3600 s/hour coverage
// at the engine's established 25 s step accuracy, with the composed
// affine pair recorded so the checker mirror stays exact by
// construction. Harness values kept for archaeology: "0" = legacy
// 1/12-coverage route, "1" = coarse single-step full coverage,
// "2" = explicit inner-loop default. The round-89 A/B/C decomposition
// (contract 14) showed coarse ≈ inner-loop to ~0.4-1.4% of the movement:
// the fix is coverage physics, not step-size numerics.
//
// Contracts pinned here (record pins, not validator bands):
// 1. flag-off bit-identity — unset vs falsy "0" step to identical states;
// 2. non-inertness with the coverage direction pin — flag-on mass state
//    differs from flag-off under solar + outdoor driving;
// 3. convergence — at N=12 the aligned mass state is strictly closer to
//    the N=96 aligned refinement limit than the unaligned N=12 state is
//    (the unaligned operator's mass coverage shrinks as 3600/N per hour,
//    so its own N→∞ limit is a frozen mass, not the continuous solution).

/// Run the spec solar model at a given coupled N with the R89 alignment
/// flag in a given state, returning the final mass temperatures. Holds
/// ENV_LOCK for the whole run; restores both variables afterwards.
fn r89_mass_state_at(align: Option<&str>, n: usize, hours: usize) -> Vec<f64> {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prev_align = std::env::var("FLX_R89_MASS_DT_ALIGN").ok();
    let prev_n = std::env::var("FLX_R77_COUPLED_N").ok();
    match align {
        Some(v) => std::env::set_var("FLX_R89_MASS_DT_ALIGN", v),
        None => std::env::remove_var("FLX_R89_MASS_DT_ALIGN"),
    }
    std::env::set_var("FLX_R77_COUPLED_N", n.to_string());
    let mut m = spec_model_solar();
    m.solar.solar_gains = VectorField::from_scalar(R88_SOLAR_WM2, m.hvac.num_zones);
    let dt = 3600.0;
    for h in 0..hours {
        m.step_physics(h, 12.0, dt);
    }
    match prev_n {
        Some(v) => std::env::set_var("FLX_R77_COUPLED_N", v),
        None => std::env::remove_var("FLX_R77_COUPLED_N"),
    }
    match prev_align {
        Some(v) => std::env::set_var("FLX_R89_MASS_DT_ALIGN", v),
        None => std::env::remove_var("FLX_R89_MASS_DT_ALIGN"),
    }
    (0..m.hvac.num_zones)
        .map(|z| m.mass.mass_temperatures[z])
        .collect::<Vec<f64>>()
}

#[test]
/// Contract 11 — default polarity bit-identity (re-pinned by the Item-2a
/// flip): with FLX_R89_MASS_DT_ALIGN unset and with its explicit "2"
/// token, the model steps to bit-identical mass states (unset takes the
/// inner-loop full-coverage default; any visible difference means the
/// mode gate leaks arithmetic). The falsy "0" token is the LEGACY
/// 1/12-coverage route and must differ (contract 12).
fn r89_align_off_is_bit_identical() {
    let unset = r89_mass_state_at(None, 12, 48);
    let explicit_default = r89_mass_state_at(Some("2"), 12, 48);
    assert_eq!(
        unset, explicit_default,
        "default mass state differs between unset and the explicit \"2\" token"
    );
}

#[test]
/// Contract 12 — non-inertness with the coverage direction pin: with the
/// arm on, the mass node integrates the full sub-step interval, so under
/// sustained solar + outdoor driving the aligned mass state must differ
/// from the unaligned one (the unaligned mass receives only 1/12 of real
/// time).
fn r89_align_changes_mass_state() {
    let off = r89_mass_state_at(Some("0"), 12, 48);
    let on = r89_mass_state_at(None, 12, 48);
    let max_delta = off
        .iter()
        .zip(on.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_delta > 1e-9,
        "aligned mass state identical to unaligned: the arm is inert (max |ΔT_m| = {max_delta:.6e})"
    );
}

#[test]
/// Contract 13 — convergence (re-anchored to the flipped default): the
/// default (inner-loop full-coverage) N=12 mass state is strictly closer
/// to the default N=96 refinement limit than the legacy ("0")
/// 1/12-coverage N=12 state is. This is the consistency proof the R77
/// scheme's docstring claims: only the full-coverage operator's limit is
/// the continuous system for BOTH nodes.
fn r89_align_converges_to_refined_limit() {
    let align12 = r89_mass_state_at(None, 12, 48);
    let align96 = r89_mass_state_at(None, 96, 48);
    let off12 = r89_mass_state_at(Some("0"), 12, 48);
    let aligned_err = align12
        .iter()
        .zip(align96.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let unaligned_err = off12
        .iter()
        .zip(align96.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        aligned_err < unaligned_err,
        "aligned N=12 not closer to the N=96 limit than unaligned: aligned={aligned_err:.6e} unaligned={unaligned_err:.6e}"
    );
}

#[test]
/// Contract 14 — decomposition pin (round-89 A/B/C experiment): the
/// coarse single-step full-coverage variant ("1") and the default
/// inner-loop full-coverage variant (unset) agree closely relative to
/// their common distance from the legacy 1/12-coverage route — i.e. the
/// measured movement is time-coverage PHYSICS, not step-size numerics.
/// Threshold 10% of the legacy delta: measured ~0.4-1.4% across the
/// strict rows (600 H 1.4%, 600FF Tmin 0.4%).
fn r89_coverage_physics_dominates_step_size() {
    let coarse = r89_mass_state_at(Some("1"), 12, 48);
    let inner = r89_mass_state_at(None, 12, 48);
    let legacy = r89_mass_state_at(Some("0"), 12, 48);
    let bc = coarse
        .iter()
        .zip(inner.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let lc = legacy
        .iter()
        .zip(inner.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        lc > 1e-9,
        "coverage fix inert in this configuration (max |ΔT_m| = {lc:.3e})"
    );
    assert!(
        bc < 0.10 * lc,
        "coarse vs inner-loop gap not small relative to the coverage movement: B-C={bc:.6e} legacy-C={lc:.6e}"
    );
}

// ===================================================================
// Slice 5 (loop round 92): the solar partition site and the retained
// lag-filter route. Contract numbering continues from slice 4.
//
// Route-map note for this slice: under the round-91 defaults the
// surface pool's mass side is the Table 5-138 class fractions (sum 1
// after renormalization) and the pool's air side is the §C.3 lag-site
// term scaled by (1 − s_share); the §C.3 mass-share arm remains the
// falsy-token archaeology route. The fraction constants themselves are
// engine-private (per-step class-boundary aggregation), so their
// numeric content is pinned through the strict-gate baseline, not at
// this seam.

#[test]
/// Contract 15 — partition-site conservation under the round-91
/// defaults. With loads neutralized and the spec model's retired split
/// (both knobs 0.0, pinned below), the whole transmitted flux routes
/// through the surface share, so the Table 5-138 window-loss sink
/// (×0.965, `FLX_R91_RETRANS` default) must act as a pure input
/// scaler: the default route at S and the falsy-token ×1.0 route at
/// 0.965·S step to the same state. (The three-way `last_phi_*`
/// diagnostics are `pr821-diag`-gated, so the split's internal
/// conservation is pinned at this seam only through this scaler
/// identity and the knob defaults.) Any change to the sink constant,
/// its position relative to the split, or the retired knobs breaks
/// this equality.
fn solar_partition_conserves_with_retransmission_sink() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dt = 3600.0;

    let mut sink = spec_model_solar();
    sink.solar.solar_gains = VectorField::from_scalar(R88_SOLAR_WM2, sink.hvac.num_zones);
    sink.setpoints.heating_setpoint = -50.0;
    sink.setpoints.cooling_setpoint = 100.0;
    // Retired-split defaults (round-91 flip 2, spec path): transmitted
    // solar distributes among interior surfaces only.
    assert_eq!(
        sink.solar.solar_distribution_to_air, 0.0,
        "spec model no longer carries a 0.0 solar_distribution_to_air default"
    );
    assert_eq!(
        sink.solar.solar_beam_to_mass_fraction, 0.0,
        "spec model no longer carries a 0.0 solar_beam_to_mass_fraction default"
    );
    for h in 0..24 {
        sink.step_physics(h, 12.0, dt);
    }

    std::env::set_var("FLX_R91_RETRANS", "0");
    let mut scaled = spec_model_solar();
    let s_scaled = R88_SOLAR_WM2 * 0.965;
    scaled.solar.solar_gains = VectorField::from_scalar(s_scaled, scaled.hvac.num_zones);
    scaled.setpoints.heating_setpoint = -50.0;
    scaled.setpoints.cooling_setpoint = 100.0;
    for h in 0..24 {
        scaled.step_physics(h, 12.0, dt);
    }
    std::env::remove_var("FLX_R91_RETRANS");

    let d_air = (sink.mass.air_temperatures[0] - scaled.mass.air_temperatures[0]).abs();
    let d_mass = (sink.mass.mass_temperatures[0] - scaled.mass.mass_temperatures[0]).abs();
    assert!(
        d_air < 1e-9 && d_mass < 1e-9,
        "window-loss sink is not a pure input scaler: d_air={d_air:.3e} d_mass={d_mass:.3e}"
    );
}
#[test]
/// Contract 16 — T5138 default-token gate-leak bit-identity: unset and
/// the truthy "1" token take the same Table 5-138 default route and
/// step to bit-identical air and mass states (contract-8 pattern for
/// the round-91 default; a visible difference means the flag gate
/// leaks arithmetic into the default path).
fn t5138_default_token_is_bit_identical() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dt = 3600.0;
    let mut unset = spec_model_solar();
    unset.solar.solar_gains = VectorField::from_scalar(R88_SOLAR_WM2, unset.hvac.num_zones);
    unset.setpoints.heating_setpoint = -50.0;
    unset.setpoints.cooling_setpoint = 100.0;
    for h in 0..48 {
        unset.step_physics(h, 12.0, dt);
    }
    std::env::set_var("FLX_R91_T5138", "1");
    let mut truthy = spec_model_solar();
    truthy.solar.solar_gains = VectorField::from_scalar(R88_SOLAR_WM2, truthy.hvac.num_zones);
    truthy.setpoints.heating_setpoint = -50.0;
    truthy.setpoints.cooling_setpoint = 100.0;
    for h in 0..48 {
        truthy.step_physics(h, 12.0, dt);
    }
    std::env::remove_var("FLX_R91_T5138");
    for z in 0..unset.hvac.num_zones {
        assert_eq!(
            unset.mass.air_temperatures[z], truthy.mass.air_temperatures[z],
            "T5138 unset vs truthy air states differ (zone {z})"
        );
        assert_eq!(
            unset.mass.mass_temperatures[z], truthy.mass.mass_temperatures[z],
            "T5138 unset vs truthy mass states differ (zone {z})"
        );
    }
}

#[test]
/// Contract 17 — route-choice consequence: the Table 5-138 default
/// deposits the full surface pool in the mass classes (the renormalized
/// fractions sum to 1) while the §C.3 conductance-share arm
/// (`FLX_R91_T5138` falsy, `FLX_R88_PHIST_MASS_SHARE` unset) deposits
/// s = Σ w_c·h_ms_c/(h_ms_c + h_is_c·w) < 1 of the same pool flux.
/// Under identical constant solar, free float, the default-route mass
/// is therefore strictly warmer than the §C.3-arm mass, and the air
/// states differ. Magnitudes are measurement, not bands.
fn t5138_route_deposits_full_surface_share() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dt = 3600.0;
    let mut t5138 = spec_model_solar();
    t5138.solar.solar_gains = VectorField::from_scalar(R88_SOLAR_WM2, t5138.hvac.num_zones);
    t5138.setpoints.heating_setpoint = -50.0;
    t5138.setpoints.cooling_setpoint = 100.0;
    for h in 0..24 {
        t5138.step_physics(h, 12.0, dt);
    }
    std::env::set_var("FLX_R91_T5138", "0");
    let mut c3 = spec_model_solar();
    c3.solar.solar_gains = VectorField::from_scalar(R88_SOLAR_WM2, c3.hvac.num_zones);
    c3.setpoints.heating_setpoint = -50.0;
    c3.setpoints.cooling_setpoint = 100.0;
    for h in 0..24 {
        c3.step_physics(h, 12.0, dt);
    }
    std::env::remove_var("FLX_R91_T5138");
    let m_t = t5138.mass.mass_temperatures[0];
    let m_c = c3.mass.mass_temperatures[0];
    assert!(
        m_t > m_c + 1e-9,
        "T5138 default-route mass not warmer than the §C.3 share arm: t5138={m_t:.6} c3={m_c:.6}"
    );
    let a_t = t5138.mass.air_temperatures[0];
    let a_c = c3.mass.air_temperatures[0];
    assert!(
        (a_t - a_c).abs() > 1e-9,
        "air states identical across the route choice: t5138={a_t:.6} c3={a_c:.6}"
    );
}

#[test]
/// Contract 18 — retained lag-filter route (`FLX_R90_NO_LAG_FILTER`
/// falsy token restores the R77 first-order filter, tau_lag =
/// sqrt(tau_air·tau_mass)). Under a step onset of constant solar, free
/// float: (a) transient direction — the filtered route's air node is
/// cooler during the rise (the filter delays the surface-pool air
/// share); (b) DC parity — the filter's DC gain is 1, so at constant
/// driving both routes converge to the same equilibrium (960 h
/// horizon: the mass-side perturbation decays at the ~30 h mass time
/// constant); (c) state parity — the persisted lag state advances with
/// identical arithmetic in both routes (the round-90 flip changed only
/// the drive site), so `mass.solar_lag` matches exactly.
fn lag_filter_retained_route_direction_dc_and_state_parity() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dt = 3600.0;
    let mut unfiltered = spec_model_solar();
    unfiltered.solar.solar_gains =
        VectorField::from_scalar(R88_SOLAR_WM2, unfiltered.hvac.num_zones);
    unfiltered.setpoints.heating_setpoint = -50.0;
    unfiltered.setpoints.cooling_setpoint = 100.0;
    for h in 0..960 {
        unfiltered.step_physics(h, 12.0, dt);
    }
    std::env::set_var("FLX_R90_NO_LAG_FILTER", "0");
    let mut filtered = spec_model_solar();
    filtered.solar.solar_gains = VectorField::from_scalar(R88_SOLAR_WM2, filtered.hvac.num_zones);
    filtered.setpoints.heating_setpoint = -50.0;
    filtered.setpoints.cooling_setpoint = 100.0;
    let mut peak_gap = 0.0_f64;
    for h in 0..960 {
        filtered.step_physics(h, 12.0, dt);
        if h < 6 {
            let gap = unfiltered.mass.air_temperatures[0] - filtered.mass.air_temperatures[0];
            peak_gap = peak_gap.max(gap);
        }
    }
    std::env::remove_var("FLX_R90_NO_LAG_FILTER");
    assert!(
        peak_gap > 1e-6,
        "filtered route not cooler than the unfiltered default during the rise (peak early gap {peak_gap:.6e} K)"
    );
    let d_air = (unfiltered.mass.air_temperatures[0] - filtered.mass.air_temperatures[0]).abs();
    let d_mass = (unfiltered.mass.mass_temperatures[0] - filtered.mass.mass_temperatures[0]).abs();
    assert!(
        d_air < 1e-6 && d_mass < 1e-6,
        "routes have not converged at constant driving: d_air={d_air:.3e} d_mass={d_mass:.3e}"
    );
    assert_eq!(
        unfiltered.mass.solar_lag.as_ref()[0],
        filtered.mass.solar_lag.as_ref()[0],
        "persisted lag state differs across routes"
    );
}

#[test]
/// Contract 19 — [R93 FIX] the §C.3 lag-site air-share term retires under
/// the T5138 class route: the classes deposit the full transmitted pool on
/// the mass side, so the air node's solar-lag input is identically zero and
/// the lag state never charges. Both falsy tokens restore a live air term:
/// `FLX_R91_T5138=0` (the §C.3 partition route) and
/// `FLX_R93_T5138_AIRSHARE=1` (the pre-r93 composed route, whose
/// bit-identity to the r92 record is proven by the flag-off battery). The
/// default air state differs from both legacy routes.
fn t5138_airshare_retired_default_and_falsy_restore() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dt = 3600.0;
    let run24 = || -> (f64, f64, f64) {
        let mut m = spec_model_solar();
        m.solar.solar_gains = VectorField::from_scalar(R88_SOLAR_WM2, m.hvac.num_zones);
        m.setpoints.heating_setpoint = -50.0;
        m.setpoints.cooling_setpoint = 100.0;
        for h in 0..24 {
            m.step_physics(h, 12.0, dt);
        }
        (
            m.mass.solar_lag.as_ref()[0],
            m.mass.air_temperatures[0],
            m.mass.mass_temperatures[0],
        )
    };
    let (lag_def, air_def, mass_def) = run24();
    assert_eq!(
        lag_def, 0.0,
        "air lag charged under the T5138 default — the §C.3 air-share term is still live"
    );
    std::env::set_var("FLX_R91_T5138", "0");
    let (lag_c3, air_c3, mass_c3) = run24();
    std::env::remove_var("FLX_R91_T5138");
    assert!(
        lag_c3 != 0.0,
        "§C.3 partition route did not charge the air lag"
    );
    assert!(
        (mass_def - mass_c3).abs() > 1e-9,
        "mass states identical across the route choice"
    );
    std::env::set_var("FLX_R93_T5138_AIRSHARE", "1");
    let (lag_leg, air_leg, _mass_leg) = run24();
    std::env::remove_var("FLX_R93_T5138_AIRSHARE");
    assert!(
        lag_leg != 0.0,
        "legacy token did not restore the air-share term"
    );
    assert!(
        air_leg > air_def + 1e-9,
        "legacy composed route air not warmer than the retired route: leg={air_leg:.6} def={air_def:.6}"
    );
    assert!(
        (air_c3 - air_def).abs() > 1e-9,
        "air states identical across §C.3 vs retired route"
    );
}
