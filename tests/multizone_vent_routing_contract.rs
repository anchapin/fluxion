//! Multi-zone night-vent routing contracts (loop round 112) — the disclosed
//! residual from round 110's vent record identities: the night-vent fan
//! supplies outdoor air to **zone 0 only**, and the multi-zone routing (which
//! zones' denominators move, and how the neighbor zone is reached) had no
//! record pin because no multi-zone 5R1C fixture existed (Case 960 is
//! 9R4C-promoted; the r110/r111 disclosures named this a scoping item).
//!
//! Component under contract: the zone-0-only night-vent routing in
//! `step_physics_5r1c` (`src/sim/thermal_model_physics/physics_impl/
//! step_5r1c.rs`). The comment at the `h_ve_night` computation names the
//! single-conditioned-zone premise; the routing itself is zone-0-gated at
//! THREE sites, two of them live on the default route:
//! `scratch.h_ext_owned_zone[0] += h_ve_night` (the den-side add, which also
//! shadows the numerator through `num_rest += h_ext·T_outdoor·term_rest_1`),
//! the LIMIT-17 numerator add `num += h_ve_night·T_outdoor` (also `i == 0`),
//! and the R86-retired mass-side path `h_vent_i` (`night_vent_active_now &&
//! i == 0 && !vent_air_only`, 0 on the default route — the r110/R86
//! archaeology owns that one):
//!
//! - (a) the routing identity at the record level, per sub-step: the vented
//!   zone's consumed air denominator shifts by exactly
//!   `term_rest_1[0]·h_ve_night` (the round-110 identity, now in a 2-zone
//!   fixture), while the neighbor zone's denominator delta is **bit-exactly
//!   zero** — the vent never enters zone 1's den, even though the
//!   vent-active hour rebuilds every zone's den through the
//!   `h_ext + h_iz + h_iz_rad` multi-zone arm (the rebuild reproduces the
//!   cached `derived_den` bit-exactly there — same operand order, same
//!   inputs);
//! - (b) neighbor-zone record truth, both hours of a two-step run: hour 1,
//!   zone 1 is fully inert (whole air record and whole mass record
//!   byte-identical vent-on/off); hour 2, the air record's coefficient slots
//!   and the mass record's load-independent slots (`a`, `denom_m`, `b_load`)
//!   stay byte-identical while the mass record's `b` slot (it reads the
//!   PREVIOUS sub-step's live air state) moves now that zone 1's own air
//!   route diverges through the coupling — the vent reaches a mass-record `b`
//!   only through live air states, never through coefficients;
//! - (c) the vented zone's slot-level mass-record truth: the load-independent
//!   mass slots (`a`, `denom_m`, `b_load`) and the air record's mass-route
//!   trio (`h3od, a, bl` — slots 4, 5, 6) stay byte-identical vent-on/off,
//!   while the mass record's `b` slot (numer_rest carries the PREVIOUS
//!   sub-step's live air temperature) moves from the first sub-step where
//!   zone 0's own vent-shifted air route differs — the vent reaches the mass
//!   record only through that live air state, never through coefficients;
//! - (d) the coupling discrimination and its one-step lag: after the first
//!   hour zone 1 is byte-identical vent-on/off even with `h_tr_iz > 0` (the
//!   inter-zone load `q_iz_net = h_tr_iz·Σ_{j≠i}(T_j − T_i)` is built from
//!   the COMMITTED hour-start temperatures, identical in both runs); after
//!   the second hour zone 1's committed temperatures move (zone 0's hour-1
//!   committed state differs) while its record coefficients stay
//!   byte-identical — state moves through the committed-state load route,
//!   coefficients never; with `h_tr_iz = 0` zone 1 stays byte-identical for
//!   both hours (a fully independent zone);
//! - (e) the construction-time cached multi-zone denominator composition
//!   (`update_optimization_cache`, Issue #351 arm): per zone,
//!   `den = h_ms_is_prod + term_rest_1·(h_ext + h_tr_iz + h_tr_iz_rad)
//!   + ground_coeff` bit-exactly against the Python-replicated closed form —
//!   the composition the vent-active-hour rebuild mirrors for unvented zones;
//! - (f) the num-side routing, both zone-0-gated sites together: the vented
//!   zone's committed free-float air response decomposes as
//!   `air_on ≈ (C + h_ms_is_prod·tm_on + (tr1+1)·h_ve_night·T_out)/den_on`
//!   with `C` fit from the fan-less twin (`air_off·den_off − ms·tm_off`) —
//!   the mass composition reproduces the engine's committed mass state
//!   bit-exactly from the record's own coefficients, and the air composition
//!   carries a −0.0035 K unmodeled residual (the solar-lag state's sub-step
//!   shape under the 600 s τ fallback with `c_air = 0`) in a −16.25 K signal,
//!   pinned at 0.01 K — a moved or dropped num site shifts the state by
//!   ≈7.8–16 K, far past the floor;
//! - (g) zero-capacity inertness: an ACTIVE fan with `fan_capacity = 0.0`
//!   leaves every record slot and every committed temperature byte-identical
//!   to the fan-less run — every vent term is exactly proportional to the
//!   fan capacity, so any future schedule-gated or constant vent addition
//!   breaks this pin.
//!
//! Fixture: a hand-rolled 2-zone lumped 5R1C model (`ThermalModel::new(2)`,
//! the CZ3 hand-built pattern used by `mass_advance_contract.rs` and round
//! 110's `lag_model_lumped`) with asymmetric zones (different conductances,
//! masses and air density — the vent conductance reads zone 0's properties
//! via `.first()`, so zone 1's 1.4 kg/m³ discriminates that read),
//! `NightVentilation::new(1400.0, 0, 0)` (active every hour, the Table 5-160
//! fan) and symmetric inter-zone coupling `h_tr_iz = 2.0`, `h_tr_iz_rad =
//! 1.0`. Python replication first (`/tmp/r112_multizone_replication.py`,
//! dev-time evidence, v2 after the engine probe): `h_ve_night = 469.0`, the
//! expected zone-0 delta `term_rest_1[0]·h_ve_night = 501361.0`, the cached
//! dens `141478.19999999998` / `93262.2`, the first-sub-step record dens
//! `141478.19999999998` (off, == cached) and `642839.2000000001` (on) — the
//! constants asserted below existed with bits before the suite was written.
//!
//! Round-110 suite shape: record pins with bit-exact teeth, file-local
//! ENV_LOCK serialization. These are record pins, not validator bands: a
//! violation means the vent routing moved and must be re-examined, never
//! tuned around.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::NightVentilation;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

const DT: f64 = 3600.0;

/// Internal-gain driving for the fixture, W/m² — keeps both zones' coupled
/// advances live (the record only persists on a coupled step).
const LOADS_WM2: f64 = 30.0;

/// The Table 5-160 night-vent fan, m³/h (the round-110 fixture's value).
const FAN_M3_H: f64 = 1400.0;

/// Symmetric inter-zone coupling for the main fixture, W/K.
const H_IZ: f64 = 2.0;
const H_IZ_RAD: f64 = 1.0;

/// Zone 0's ventilation conductance, W/K — the fixture's own expression
/// (op order mirrored by the Python replication).
fn h_ve_0() -> f64 {
    1.2 * 1000.0 * (0.5 * (48.0 * 2.7) / 3600.0)
}

/// Zone 1's ventilation conductance, W/K.
fn h_ve_1() -> f64 {
    1.2 * 1000.0 * (0.4 * (30.0 * 2.7) / 3600.0)
}

/// The hand-rolled 2-zone lumped 5R1C fixture. Asymmetric by construction:
/// zone 0 (48 m², the r110-style parameters) is the conditioned zone the
/// fan serves; zone 1 (30 m², different conductances, mass and air density)
/// is the neighbor reached only through the inter-zone coupling.
fn multizone_model() -> ThermalModel<VectorField> {
    let mut m = ThermalModel::new(2);
    m.setpoints.zone_area = VectorField::new(vec![48.0, 30.0]);
    m.setpoints.ceiling_height = VectorField::from_scalar(2.7, 2);
    m.setpoints.heating_setpoint = -50.0;
    m.setpoints.cooling_setpoint = 100.0;
    // Zone 1's density differs on purpose: the vent conductance must read
    // zone 0's properties (`.first()`), and this discriminates that read.
    m.setpoints.air_density = VectorField::new(vec![1.2, 1.4]);
    m.setpoints.heat_capacity = VectorField::from_scalar(1005.0, 2);
    m.solar.window_u_value = 3.0;
    m.conduction.h_tr_em = VectorField::new(vec![0.32 * 48.0, 9.0]);
    m.conduction.h_tr_w = VectorField::new(vec![3.0 * 12.0, 24.0]);
    m.conduction.h_tr_floor = VectorField::new(vec![0.15 * 48.0, 6.0]);
    m.conduction.h_ve = VectorField::new(vec![h_ve_0(), h_ve_1()]);
    m.conduction.h_tr_ms = VectorField::new(vec![1000.0, 800.0]);
    // h_tr_is keeps the ThermalModel::new default (69.0 per zone).
    m.conduction.h_tr_iz = VectorField::from_scalar(H_IZ, 2);
    m.conduction.h_tr_iz_rad = VectorField::from_scalar(H_IZ_RAD, 2);
    m.mass.thermal_capacitance = VectorField::new(vec![48.0 * 150_000.0, 30.0 * 180_000.0]);
    m.setpoints.temperatures = VectorField::from_scalar(21.0, 2);
    m.mass.mass_temperatures = VectorField::from_scalar(21.0, 2);
    m.setpoints.loads = VectorField::from_scalar(LOADS_WM2, 2);
    m.solar.solar_gains = VectorField::new(vec![0.0, 0.0]);
    m.solar.opaque_solar_gains = VectorField::new(vec![0.0, 0.0]);
    m.hvac.hvac_heating_capacity = 0.0;
    m.hvac.hvac_cooling_capacity = 0.0;
    m.update_optimization_cache();
    m
}

/// The fixture with a night-ventilation fan attached (`None` removes it).
fn vented(vent: Option<NightVentilation>) -> ThermalModel<VectorField> {
    let mut m = multizone_model();
    m.hvac.night_ventilation = vent;
    m
}

/// The fixture with the inter-zone coupling overridden (0.0 = independent
/// zones for the coupling discrimination).
fn with_coupling(iz: f64) -> ThermalModel<VectorField> {
    let mut m = multizone_model();
    m.conduction.h_tr_iz = VectorField::from_scalar(iz, 2);
    m.conduction.h_tr_iz_rad = VectorField::from_scalar(iz, 2);
    m.update_optimization_cache();
    m
}

/// The air record for one zone: (sub-step count, den sequence). 9 entries per
/// sub-step `[D, M, W, den, h3od, a, bl, C, np]` — the round-107 seven-entry
/// layout extended in lockstep by round 113 (slot `C`, the inter-zone
/// cross-response) and round 115 (slot `np`, the ramp-source slope).
/// Re-pinned 2026-11-13 (loop round 139): the stride assert still counted the
/// round-107 seven-entry layout and failed on the live 12x9 = 108-entry record
/// (proven pre-existing at the base tip in round 138); `den` stays index 3 and
/// the vent trio stays slots 4-6 of each sub-step block.
fn air_record(m: &ThermalModel<VectorField>, zone: usize) -> (usize, Vec<f64>) {
    let rec = m
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("a live coupled 5R1C step must persist the air-node record");
    let row = &rec[zone];
    assert!(
        !row.is_empty() && row.len() % 9 == 0,
        "air record must carry 9 entries per sub-step, got {}",
        row.len()
    );
    let n = row.len() / 9;
    let den_seq = (0..n).map(|k| row[9 * k + 3]).collect::<Vec<f64>>();
    (n, den_seq)
}

/// The engine's own `h_ve_night` reads: the vent conductance takes zone 0's
/// air properties (`setpoints.air_density`/`heat_capacity` `.first()`), with
/// the documented 1.2 / 1005 fallbacks — over the fixture's actual values
/// `1400·1.2·1005/3600 = 469.0` (Python-replicated bits 0x407d500000000000).
fn h_ve_night_engine_read(m: &ThermalModel<VectorField>) -> f64 {
    let rho = m
        .setpoints
        .air_density
        .as_ref()
        .first()
        .copied()
        .unwrap_or(1.2);
    let cp = m
        .setpoints
        .heat_capacity
        .as_ref()
        .first()
        .copied()
        .unwrap_or(1005.0);
    FAN_M3_H * rho * cp / 3600.0
}

#[test]
/// Contract 1 — the zone-0-only routing identity at the record level: on
/// every sub-step the vented zone's consumed den shifts by exactly
/// `term_rest_1[0]·h_ve_night` while the neighbor zone's den delta is
/// bit-exactly zero, even though the vent-active hour rebuilds den for ALL
/// zones through the multi-zone `h_ext + h_iz + h_iz_rad` arm. The first
/// sub-step's absolute dens are pinned bit-exactly against the
/// Python-replicated closed forms (the off-run reads the cached `derived_den`
/// verbatim; the on-run rebuilds through `h_ext[0] + h_ve_night`). The delta
/// uses zone 0's `term_rest_1` (1069.0, not the neighbor's 869.0 — the
/// fixture's asymmetry makes a zone-confused delta visible).
fn vent_shifts_only_zone0s_den_multi_zone() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut on = vented(Some(NightVentilation::new(FAN_M3_H, 0, 0)));
    let mut off = vented(None);

    on.step_physics(0, -10.0, DT);
    off.step_physics(0, -10.0, DT);

    let h_ve_night = h_ve_night_engine_read(&on);
    assert_eq!(
        h_ve_night, 469.0,
        "fixture premise: the vent conductance reads zone 0's 1.2/1005 properties"
    );
    let tr1_0 = on.conduction.derived_term_rest_1.as_ref()[0];
    let tr1_1 = on.conduction.derived_term_rest_1.as_ref()[1];
    assert_eq!(tr1_0, 1069.0, "fixture premise: zone 0 term_rest_1");
    assert_eq!(
        tr1_1, 869.0,
        "fixture premise: zone 1 term_rest_1 (asymmetric)"
    );
    let expected_delta = tr1_0 * h_ve_night;

    // Absolute first-sub-step dens against the engine-order closed forms
    // (Python-replicated bits; the off-run den equals the cached derived_den).
    let (n_on, den_on_0) = air_record(&on, 0);
    let (n_off, den_off_0) = air_record(&off, 0);
    let (_, den_on_1) = air_record(&on, 1);
    let (_, den_off_1) = air_record(&off, 1);
    assert_eq!(n_on, n_off, "same coupled scheme, same sub-step count");
    assert_eq!(
        den_on_1.len(),
        n_on,
        "zone 1 carries its own sub-step sequence"
    );
    let ms_is_0 = 1000.0_f64 * 69.0;
    let den_off_0_closed =
        ms_is_0 + tr1_0 * (36.0 + h_ve_0() + H_IZ + H_IZ_RAD) + tr1_0 * (0.15 * 48.0);
    assert_eq!(
        den_off_0[0], den_off_0_closed,
        "off-run den[0] == cached den closed form"
    );
    assert_eq!(
        den_off_0[0],
        on.conduction.derived_den.as_ref()[0],
        "off-run den[0] == the model's cached derived_den"
    );
    let den_on_0_closed =
        ms_is_0 + tr1_0 * (36.0 + h_ve_0() + h_ve_night + H_IZ + H_IZ_RAD) + tr1_0 * (0.15 * 48.0);
    assert_eq!(
        den_on_0[0], den_on_0_closed,
        "on-run den[0] == vent-shifted closed form"
    );
    assert_eq!(
        den_on_1[0], den_off_1[0],
        "zone 1's first-sub-step den bit-identical"
    );

    for k in 0..n_on {
        let delta_0 = den_on_0[k] - den_off_0[k];
        assert!(
            (delta_0 - expected_delta).abs() <= 1e-12 * expected_delta.abs(),
            "sub-step {k}: the vent must shift ZONE 0's den by exactly \
             term_rest_1[0]·h_ve_night (Δden {delta_0} vs expected {expected_delta})"
        );
        assert_eq!(
            den_on_1[k], den_off_1[k],
            "sub-step {k}: the neighbor zone's den must be bit-identical vent-on/off \
             (the vent-active-hour rebuild reproduces the cached den exactly there)"
        );
    }
}

#[test]
/// Contract 2 — neighbor-zone record truth across BOTH hours of a two-step
/// run. Hour 1: zone 1 is fully inert — its whole air record (7 entries per
/// sub-step `[D, M, W, den, h3od, a, bl]`) and its whole mass record
/// (`[a, b, denom_m, b_load]`) are byte-identical vent-on/off. Hour 2: the
/// air record's coefficient slots stay byte-identical (the inter-zone
/// coupling enters through the air-node LOAD
/// `q_iz_net = h_tr_iz·Σ_{j≠i}(T_j − T_i)`, never through coefficients),
/// while the mass record's load-independent slots (`a`, `denom_m`, `b_load`)
/// stay byte-identical and its `b` slot — which reads the PREVIOUS sub-step's
/// live air state — moves now that zone 1's own air route diverges through
/// the coupling: the vent reaches a mass-record `b` only through live air
/// states, directly for zone 0 and one hour later for zone 1.
fn neighbor_zone_records_byte_identical_vent_on_off() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut on = vented(Some(NightVentilation::new(FAN_M3_H, 0, 0)));
    let mut off = vented(None);

    on.step_physics(0, -10.0, DT);
    off.step_physics(0, -10.0, DT);
    let on_h1_air = on
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (on, h1)")[1]
        .clone();
    let off_h1_air = off
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (off, h1)")[1]
        .clone();
    let on_h1_mass = on
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("mass record (on, h1)")[1]
        .clone();
    let off_h1_mass = off
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("mass record (off, h1)")[1]
        .clone();

    on.step_physics(1, -10.0, DT);
    off.step_physics(1, -10.0, DT);
    let air_on = on
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (on, h2)");
    let air_off = off
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (off, h2)");
    let mass_on = on
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("mass record (on, h2)");
    let mass_off = off
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("mass record (off, h2)");

    assert_eq!(
        on_h1_air, off_h1_air,
        "hour 1: zone 1's whole air record byte-identical vent-on/off"
    );
    assert_eq!(
        on_h1_mass, off_h1_mass,
        "hour 1: zone 1's whole mass record byte-identical"
    );
    assert_eq!(
        air_on[1], air_off[1],
        "hour 2: zone 1's air record (all coefficient slots) byte-identical vent-on/off"
    );
    let nm = mass_on[1].len() / 4;
    assert_eq!(
        mass_on[1].len(),
        mass_off[1].len(),
        "same hour-2 mass-record length"
    );
    let mut b_moved = false;
    for k in 0..nm {
        for slot in [0_usize, 2, 3] {
            assert_eq!(
                mass_on[1][4 * k + slot],
                mass_off[1][4 * k + slot],
                "hour 2 mass sub-step {k} slot {slot}: zone 1's load-independent \
                 mass slot moved under vent-on"
            );
        }
        if mass_on[1][4 * k + 1] != mass_off[1][4 * k + 1] {
            b_moved = true;
        }
    }
    assert!(
        b_moved,
        "hour 2: zone 1's mass-record b slot must move once its own live air \
         route diverges through the coupling (never through coefficients)"
    );
}

#[test]
/// Contract 3 — the vented zone's slot-level mass-record truth: the
/// load-independent slots (`a` at 4k+0, `denom_m` at 4k+2, `b_load` at 4k+3)
/// and the air record's mass-route trio (`h3od`, `a`, `bl` at 7k+4..6) stay
/// byte-identical vent-on/off, while the mass record's `b` slot (4k+1) is
/// identical at the first sub-step (its `numer_rest` reads the PREVIOUS
/// sub-step's air state, which starts at the shared committed hour-start
/// temperature) and moves from the first sub-step where zone 0's own
/// vent-shifted air route differs — the vent reaches the mass record only
/// through that live air state, never through coefficients.
fn zone0_mass_record_slots_identical_except_the_live_air_b() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut on = vented(Some(NightVentilation::new(FAN_M3_H, 0, 0)));
    let mut off = vented(None);

    on.step_physics(0, -10.0, DT);
    off.step_physics(0, -10.0, DT);

    let air_on = on
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (on)");
    let air_off = off
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (off)");
    let n = air_on[0].len() / 9;
    for k in 0..n {
        for slot in [4_usize, 5, 6] {
            assert_eq!(
                air_on[0][9 * k + slot],
                air_off[0][9 * k + slot],
                "sub-step {k} slot {slot}: zone 0's mass-route trio moved under vent-on"
            );
        }
    }

    let mass_on = on
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("mass record (on)");
    let mass_off = off
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("mass record (off)");
    let nm = mass_on[0].len() / 4;
    assert_eq!(
        mass_on[0].len(),
        mass_off[0].len(),
        "same mass-record length"
    );
    let mut b_moved = false;
    for k in 0..nm {
        for slot in [0_usize, 2, 3] {
            assert_eq!(
                mass_on[0][4 * k + slot],
                mass_off[0][4 * k + slot],
                "mass sub-step {k} slot {slot}: zone 0's load-independent mass slot moved"
            );
        }
        if mass_on[0][4 * k + 1] != mass_off[0][4 * k + 1] {
            b_moved = true;
            assert!(
                k >= 1,
                "the first mass sub-step's b must be identical (both runs read the \
                 shared committed hour-start air state)"
            );
        }
    }
    assert!(
        b_moved,
        "zone 0's mass-record b slot must move once its own live air route \
         diverges (the vent reaches the mass record only through the air state)"
    );
}

#[test]
/// Contract 4 — the coupling discrimination and its one-step lag: after hour
/// 1 zone 1 is byte-identical vent-on/off even with `h_tr_iz > 0` (the
/// inter-zone load is built from the committed hour-start temperatures,
/// identical in both runs); after hour 2 zone 1's committed temperatures
/// MOVE (zone 0's hour-1 committed state differs) while its record
/// coefficients stay byte-identical; with `h_tr_iz = 0` zone 1 stays
/// byte-identical for both hours (a fully independent zone) while zone 0
/// moves under the fan.
fn neighbor_state_moves_only_through_coupling() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Pair A — coupled fixture: one-step lag, then movement through the
    // committed-state route.
    let mut b_on = vented(Some(NightVentilation::new(FAN_M3_H, 0, 0)));
    let mut b_off = vented(None);
    b_on.step_physics(0, -10.0, DT);
    b_off.step_physics(0, -10.0, DT);
    assert_eq!(
        b_on.mass.air_temperatures.as_ref()[1],
        b_off.mass.air_temperatures.as_ref()[1],
        "hour 1: with h_tr_iz = 2.0 the neighbor is still byte-identical — \
         the coupling load reads the committed hour-start temperatures"
    );
    let air_on_h1 = b_on
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (on h1)")[1]
        .clone();
    let air_off_h1 = b_off
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (off h1)")[1]
        .clone();
    assert_eq!(
        air_on_h1, air_off_h1,
        "hour 1: zone 1's record byte-identical"
    );

    b_on.step_physics(1, -10.0, DT);
    b_off.step_physics(1, -10.0, DT);
    let d_t1 = b_on.mass.air_temperatures.as_ref()[1] - b_off.mass.air_temperatures.as_ref()[1];
    assert!(
        d_t1.abs() > 1e-9,
        "hour 2: the neighbor must move through the committed-state coupling \
         route (ΔT_1 = {d_t1})"
    );
    let d_t0 = b_on.mass.air_temperatures.as_ref()[0] - b_off.mass.air_temperatures.as_ref()[0];
    assert!(
        d_t0.abs() > 1e-6,
        "the vented zone itself must move under the fan (ΔT_0 = {d_t0})"
    );
    let air_on = b_on
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (on h2)");
    let air_off = b_off
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (off h2)");
    assert_eq!(
        air_on[1], air_off[1],
        "hour 2: the neighbor's record coefficients stay byte-identical while its state moves"
    );

    // Pair B — independent zones: zone 1 never moves, zone 0 does.
    let mut a_on = with_coupling(0.0);
    a_on.hvac.night_ventilation = Some(NightVentilation::new(FAN_M3_H, 0, 0));
    let mut a_off = with_coupling(0.0);
    a_on.step_physics(0, -10.0, DT);
    a_off.step_physics(0, -10.0, DT);
    a_on.step_physics(1, -10.0, DT);
    a_off.step_physics(1, -10.0, DT);
    assert_eq!(
        a_on.mass.air_temperatures.as_ref()[1],
        a_off.mass.air_temperatures.as_ref()[1],
        "with zero coupling the neighbor zone must be byte-identical vent-on/off, both hours"
    );
    assert_eq!(
        a_on.mass.mass_temperatures.as_ref()[1],
        a_off.mass.mass_temperatures.as_ref()[1],
        "with zero coupling the neighbor's mass node must be byte-identical too"
    );
    let a_d_t0 = a_on.mass.air_temperatures.as_ref()[0] - a_off.mass.air_temperatures.as_ref()[0];
    assert!(
        a_d_t0.abs() > 1e-6,
        "the vented zone itself must move (ΔT_0 = {a_d_t0})"
    );
}

#[test]
/// Contract 5 — the construction-time cached multi-zone denominator
/// composition (Issue #351 arm): per zone, `den = h_ms_is_prod +
/// term_rest_1·(h_ext + h_tr_iz + h_tr_iz_rad) + ground_coeff` bit-exactly
/// against the Python-replicated closed form, and the inter-zone terms are
/// actually IN the multi-zone den (the num_zones == 1 arm excludes them —
/// the composition the vent-active-hour rebuild mirrors for unvented zones).
fn cached_multizone_den_matches_closed_form() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let m = multizone_model();
    let c = &m.conduction;

    // Fixture premises (Python-replicated bits, /tmp/r112_multizone_replication.py).
    assert_eq!(
        c.derived_h_ext.as_ref()[0],
        57.6,
        "h_ext_0 = h_tr_w_0 + h_ve_0"
    );
    assert_eq!(
        c.derived_h_ext.as_ref()[1],
        34.8,
        "h_ext_1 = h_tr_w_1 + h_ve_1"
    );
    assert_eq!(
        c.derived_term_rest_1.as_ref()[0],
        1069.0,
        "tr1_0 = (1000 + 69) + 0"
    );
    assert_eq!(
        c.derived_term_rest_1.as_ref()[1],
        869.0,
        "tr1_1 = (800 + 69) + 0"
    );
    assert_eq!(
        c.derived_ground_coeff.as_ref()[0],
        7696.799999999999,
        "gc_0 = tr1_0·h_tr_floor_0"
    );
    assert_eq!(
        c.derived_ground_coeff.as_ref()[1],
        5214.0,
        "gc_1 = tr1_1·h_tr_floor_1"
    );

    // The closed form with the inter-zone terms, bit-exact (engine operand
    // order: ms_is_prod + tr1·h_total + gc with h_total = h_ext + iz + iz_rad).
    let ms_is_0 = 1000.0_f64 * 69.0;
    let ms_is_1 = 800.0_f64 * 69.0;
    let den_0_closed =
        ms_is_0 + 1069.0 * (36.0 + h_ve_0() + H_IZ + H_IZ_RAD) + 1069.0 * (0.15 * 48.0);
    let den_1_closed = ms_is_1 + 869.0 * (24.0 + h_ve_1() + H_IZ + H_IZ_RAD) + 869.0 * 6.0;
    assert_eq!(
        c.derived_den.as_ref()[0],
        den_0_closed,
        "zone 0's cached den must carry the inter-zone composition bit-exactly"
    );
    assert_eq!(
        c.derived_den.as_ref()[1],
        den_1_closed,
        "zone 1's cached den must carry the inter-zone composition bit-exactly"
    );

    // And the discriminator: the cached den is NOT the single-zone form
    // (which excludes h_iz) — the multi-zone arm actually fires for
    // num_zones == 2. Python: 138271.19999999998 / 90655.2.
    let den_0_single_zone = ms_is_0 + 1069.0 * (36.0 + h_ve_0()) + 1069.0 * (0.15 * 48.0);
    let den_1_single_zone = ms_is_1 + 869.0 * (24.0 + h_ve_1()) + 869.0 * 6.0;
    assert_ne!(
        c.derived_den.as_ref()[0],
        den_0_single_zone,
        "the multi-zone den must differ from the single-zone composition"
    );
    assert_ne!(
        c.derived_den.as_ref()[1],
        den_1_single_zone,
        "the multi-zone den must differ from the single-zone composition"
    );
}

#[test]
/// Contract 6 — the num-side routing, both default-route zone-0-gated sites
/// together (the `h_ext[0]` shadow through `num_rest += h_ext·T_out·tr1` and
/// the LIMIT-17 `num += h_ve_night·T_out` add). Model, from the Python
/// replication (`/tmp/r112_multizone_replication.py`): the hour-1 zone-0
/// mass recurrence `tm(k+1) = a·tm(k) + b(k)` composes the engine's committed
/// mass state BIT-EXACTLY from the record's own coefficients, and the air
/// node (instantaneous at the `ThermalModel::new` default `c_air = 0`) obeys
/// `air(11) = (C + h_ms_is_prod·tm(11) + (tr1+1)·h_ve_night·T_out)/den_on`
/// with `C = air_off·den_off − h_ms_is_prod·tm_off` — the lag+load constant
/// (the lag τ falls back to 600 s with `c_air = 0`, identical across runs).
/// The composition carries a −0.0035 K unmodeled residual (the lag state's
/// sub-step shape) in a −16.25 K signal; the 0.01 K floor sits above it and
/// far below any routing change (a moved or dropped num site shifts the
/// state by ≈7.8–16 K). Also pinned: the record's mass-route M slot times
/// the consumed den equals `derived_h_ms_is_prod` exactly, on BOTH runs.
fn num_side_routing_decomposition_into_the_committed_state() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const T_OUT: f64 = -10.0;
    let mut on = vented(Some(NightVentilation::new(FAN_M3_H, 0, 0)));
    let mut off = vented(None);

    on.step_physics(0, T_OUT, DT);
    off.step_physics(0, T_OUT, DT);

    let (_, den_on_0) = air_record(&on, 0);
    let (_, den_off_0) = air_record(&off, 0);
    let den_on = den_on_0[0];
    let den_off = den_off_0[0];
    let air_on = on.mass.air_temperatures.as_ref()[0];
    let air_off = off.mass.air_temperatures.as_ref()[0];
    let tm_on = on.mass.mass_temperatures.as_ref()[0];
    let tm_off = off.mass.mass_temperatures.as_ref()[0];
    let tr1_0 = on.conduction.derived_term_rest_1.as_ref()[0];
    let ms_is = on.conduction.derived_h_ms_is_prod.as_ref()[0];
    let h_ve_night = h_ve_night_engine_read(&on);

    // The record's mass-route M slot times its den is exactly h_ms_is_prod,
    // on both runs (the air advance's mass-refresh coefficient).
    let m_on = on
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (on)")[0][1];
    let m_off = off
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (off)")[0][1];
    assert_eq!(m_on * den_on, ms_is, "M·den == h_ms_is_prod (on run)");
    assert_eq!(m_off * den_off, ms_is, "M·den == h_ms_is_prod (off run)");

    // The two-site num model of the vented zone's committed air.
    let c_fit = air_off * den_off - ms_is * tm_off;
    let nv_total = (tr1_0 + 1.0) * h_ve_night * T_OUT;
    let air_on_model = (c_fit + ms_is * tm_on + nv_total) / den_on;
    assert!(
        (air_on - air_on_model).abs() <= 0.01,
        "the vented zone's committed air must decompose as \
         (C + ms·tm + (tr1+1)·h_ve_night·T_out)/den: measured {air_on}, \
         model {air_on_model} (Δ {} K — the num-side routing moved)",
        air_on - air_on_model
    );
    // And the state actually moved through BOTH sites: the pure-den effect
    // alone cannot explain the signal (a dropped num site leaves ≈−8.4 K).
    let den_only = (air_off * den_off) / den_on - air_off;
    assert!(
        (air_on - air_off - den_only).abs() > 1.0,
        "the num-side vent term must contribute beyond the den shift"
    );
}

#[test]
/// Contract 7 — zero-capacity inertness: an ACTIVE fan with
/// `fan_capacity = 0.0` (`h_ve_night = 0.0`, the schedule firing) must leave
/// every record slot and every committed temperature byte-identical to the
/// fan-less run. Every vent term is exactly proportional to the fan capacity
/// (den add, den rebuild delta, both num sites), so `0.0` capacity is exact
/// inertness — and any future vent term that is not (a constant offset, a
/// schedule-gated add, a per-zone rescale) breaks this pin.
fn zero_capacity_fan_is_state_inert() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut zero = vented(Some(NightVentilation::new(0.0, 0, 0)));
    let mut none = vented(None);

    zero.step_physics(0, -10.0, DT);
    none.step_physics(0, -10.0, DT);

    let zero_air = zero
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (zero)");
    let none_air = none
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("air record (none)");
    assert_eq!(
        zero_air[0], none_air[0],
        "zone 0's air record byte-identical at zero capacity"
    );
    assert_eq!(
        zero_air[1], none_air[1],
        "zone 1's air record byte-identical at zero capacity"
    );
    let zero_mass = zero
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("mass record (zero)");
    let none_mass = none
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("mass record (none)");
    assert_eq!(
        zero_mass[0], none_mass[0],
        "zone 0's mass record byte-identical at zero capacity"
    );
    assert_eq!(
        zero_mass[1], none_mass[1],
        "zone 1's mass record byte-identical at zero capacity"
    );
    for i in 0..2 {
        assert_eq!(
            zero.mass.air_temperatures.as_ref()[i],
            none.mass.air_temperatures.as_ref()[i],
            "zone {i}'s committed air temperature byte-identical at zero capacity"
        );
        assert_eq!(
            zero.mass.mass_temperatures.as_ref()[i],
            none.mass.mass_temperatures.as_ref()[i],
            "zone {i}'s committed mass temperature byte-identical at zero capacity"
        );
    }
}
