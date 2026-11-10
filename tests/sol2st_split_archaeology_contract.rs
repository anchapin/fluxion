//! Solar-split archaeology contracts (loop round 111) — the retired
//! `FLX_R91_SOL2ST` falsy token, the solar-partition queue's remaining
//! unpinned piece (checker-air-node-mirror-plan.md §7 sweep, rounds
//! 91–110; the round-110 disposition named it after the lag-carry and
//! vent-record slices landed).
//!
//! Component under contract: the R91 FLIP 2 construction-time split at
//! `src/sim/thermal_model_core/mod.rs` (`from_spec_with_selector`, the
//! single assignment site of both knobs — every `from_spec` route
//! delegates there):
//!
//! ```text
//! r91_sol2st = env_bool("FLX_R91_SOL2ST", true)
//! w_solar    = massiveness_weight(spec.construction.wall.thermal_capacitance_per_area())
//! (air_frac, beam) = if r91_sol2st { (0.0, 0.0) }            // default: retired
//!                    else          { (0.30*(1.0 - w_solar), 0.30) } // archaeology
//! model.solar.solar_distribution_to_air     = air_frac;
//! model.solar.solar_beam_to_mass_fraction   = beam;
//! ```
//!
//! Premises (verified against the tree; the primary-source record lives in
//! `docs/KNOWN_ISSUES.md` §LIMIT-28 round-91 flip 2 and §LIMIT-44):
//!
//! - ASHRAE 140-2017 §5.2.1.12.1 + Table 5-138 distribute transmitted solar
//!   among interior surfaces only — nothing directly to zone air; ISO 13790
//!   §C.1–C.3 route solar to the surface/mass pair by conductance share, no
//!   fixed beam route. The retired density-driven split is #1216-band-aid
//!   lineage (reduced from 0.70 to 0.30 by #2359); its cited basis
//!   ("ASHRAE 140 Table B-12 F_m") does not exist in 140-2017.
//! - The falsy token is an ARCHAEOLOGY route only (A/B comparison against
//!   the incumbent split), kept per the round-91 flip-2 record — its
//!   presence is deliberate and its exact values are load-bearing for any
//!   future re-adjudication, which is what this suite pins.
//! - `massiveness_weight` = `smoothstep01(κ, 80_000, 100_000)` =
//!   `t²(3−2t)`, `t = clamp((κ−80_000)/20_000, 0, 1)`
//!   (fluxion-core/src/construction.rs). The fixtures pick κ values whose
//!   ρ·δ·c products are exact in f64 (δ = 0.25 = 2⁻²), so w_solar is
//!   exactly {0.0, 0.5, 1.0} and the split's closed form is exact:
//!   to_air = {0.30, 0.15, 0.00} (0.30·0.5 is an exact halving).
//! - `env_bool` canonical sets (src/util/env_bool.rs, case-insensitive,
//!   trimmed): truthy {1, true, yes, on}; falsy {0, false, no, off, empty};
//!   ANY OTHER VALUE warns and behaves as UNSET — fail-closed to the
//!   call-site default (the retired route).
//! - Consumption (the pins' teeth have somewhere to bite): step_5r1c.rs
//!   `sol_to_air = sol_w * solar_distribution_to_air` and the
//!   step_dispatcher's threaded copies — the stored construction-time
//!   values drive every sub-step's three-way solar split.
//!
//! What is pinned ELSEWHERE and deliberately not duplicated: the unset
//! default `(0.0, 0.0)` on a real spec model and the partition-site
//! conservation identity (mass_advance_contract.rs contract 15, round 92).
//! This suite pins the token's RESTORE arm and its truthiness.
//!
//! Suite shape (rounds 85–110): Python replication first
//! (`/tmp/r111_sol2st_replication.py`, dev-time evidence — identical f64
//! arithmetic, bit-level table for every constant below), file-local
//! ENV_LOCK serialization (the engine reads env flags at construction),
//! exact assertions with stated premises. These are record pins, not
//! validator bands: a violation means the archaeology route or its
//! fail-closed semantics moved and must be re-examined, never tuned
//! around.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion_core::ashrae_case_series::series_600::case_600_baseline;
use fluxion_core::ashrae_cases::{
    BuildingType, CaseSpec, ConstructionSpec, ConstructionType, GeometrySpec, HvacSchedule,
    InternalLoads, NightVentilation, Orientation, ShadingDevice, WindowArea, WindowSpec,
};
use fluxion_core::construction::{massiveness_weight, Assemblies, Construction, ConstructionLayer};
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

const DT: f64 = 3600.0;
/// Solar driving for the consumption pin, W/m² — the suite's round-88
/// precedent value (mass_advance_contract.rs `R88_SOLAR_WM2`).
const SOLAR_WM2: f64 = 250.0;

/// One-zone CaseSpec literals whose WALL construction carries an exact
/// f64 κ = ρ·δ·c (δ = 0.25 m so the product is exact); roof/floor are the
/// standard assemblies. Only the wall κ feeds `massiveness_weight` at the
/// flag site.
fn spec_with_wall_kappa(kappa_target: f64, rho: f64, label: &str) -> CaseSpec {
    // Exactness premise: ρ·δ·c must equal kappa_target bit-exactly.
    let wall = Construction::new(vec![ConstructionLayer::new(
        label, 0.5, // W/(m·K) — physical mid-range conductivity
        rho, 1000.0, 0.25, // exact in f64 (2⁻²)
    )]);
    assert_eq!(
        wall.thermal_capacitance_per_area(),
        kappa_target,
        "fixture wall κ must be exact: ρ·δ·c = {rho}·0.25·1000"
    );

    let geometry = GeometrySpec::new(8.0, 6.0, 2.7);
    let mut spec = CaseSpec {
        case_id: format!("r111-sol2st-{label}"),
        description: "FLX_R91_SOL2ST archaeology fixture (round 111)".to_string(),
        geometry: vec![geometry],
        construction_type: ConstructionType::LowMass,
        construction: ConstructionSpec::new(
            wall,
            Assemblies::low_mass_roof(),
            Assemblies::high_mass_floor(),
        ),
        windows: vec![vec![WindowArea::new(12.0, Orientation::South)]],
        window_properties: WindowSpec::double_clear_glass(),
        shading: None::<ShadingDevice>,
        internal_loads: vec![Some(InternalLoads::new(200.0, 0.4, 0.6))],
        hvac: vec![HvacSchedule::constant(20.0, 27.0)],
        night_ventilation: None::<NightVentilation>,
        common_walls: Vec::new(),
        infiltration_ach: 0.5,
        opaque_absorptance: 0.6,
        num_zones: 1,
        zone_origins: None,
        zone_constructions: None,
        weather_data: None,
        door_height: None,
        door_area: None,
        epw_path: None,
        hvac_equipment: None,
        ground_temperature_c: None,
        floor_u_value_override: None,
        window_u_value_override: None,
        building_type: BuildingType::Residential,
    };
    // construction_type does NOT reach the flag site (the split reads the
    // wall κ directly) — assert the premise is harmless either way.
    spec.construction_type = ConstructionType::LowMass;
    spec
}

fn build(spec: &CaseSpec) -> ThermalModel<VectorField> {
    ThermalModel::from_spec(spec)
}

/// Free-float driving for the consumption pin (the ideal HVAC never
/// engages; the solar split is the only live difference between the two
/// models).
fn neutralize(model: &mut ThermalModel<VectorField>) {
    let n = model.hvac.num_zones;
    model.setpoints.heating_setpoint = -50.0;
    model.setpoints.cooling_setpoint = 100.0;
    model.hvac.hvac_heating_capacity = 0.0;
    model.hvac.hvac_cooling_capacity = 0.0;
    model.solar.solar_gains = VectorField::from_scalar(SOLAR_WM2, n);
    model.solar.opaque_solar_gains = VectorField::new(vec![0.0; n]);
}

/// (pin 1) The falsy token restores the retired density-driven split with
/// its EXACT closed form at the massiveness endpoints and midpoint:
/// to_air = 0.30·(1−w_solar), beam = 0.30, w_solar = smoothstep01(κ) ∈
/// {0.0, 0.5, 1.0} for κ ∈ {60_000, 90_000, 120_000} (all exact in f64).
/// Premise asserted per fixture: `massiveness_weight(κ)` equals the
/// expected weight, so the pin pins the flag site's arithmetic, not the
/// smoothstep. Python replication: /tmp/r111_sol2st_replication.py.
#[test]
fn falsy_token_restores_density_driven_split_exact_closed_form() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let fixtures = [
        // (κ target, ρ, expected w_solar, expected to_air)
        (60_000.0, 240.0, 0.0, 0.30), // t = −1 → clamp 0
        (90_000.0, 360.0, 0.5, 0.15), // t = 0.5 → 0.5²·(3−2·0.5) = 0.5; 0.30·0.5 exact halving
        (120_000.0, 480.0, 1.0, 0.0), // t = 2 → clamp 1
    ];
    for (kappa, rho, w_expected, to_air_expected) in fixtures {
        assert_eq!(
            massiveness_weight(kappa),
            w_expected,
            "premise: smoothstep01({kappa}) must be {w_expected}"
        );
        std::env::set_var("FLX_R91_SOL2ST", "0");
        let spec = spec_with_wall_kappa(kappa, rho, &format!("wall-{kappa:.0}"));
        assert_eq!(spec.validate(), Ok(()), "fixture spec must be valid");
        let model = build(&spec);
        std::env::remove_var("FLX_R91_SOL2ST");
        assert_eq!(
            model.solar.solar_distribution_to_air, to_air_expected,
            "falsy restore: to_air must equal 0.30·(1−{w_expected})"
        );
        assert_eq!(
            model.solar.solar_beam_to_mass_fraction, 0.30,
            "falsy restore: beam fraction must equal the retired 0.30 constant"
        );
    }
}

/// (pin 2) The REAL Case 600 spec path carries both arms: the public
/// core factory (`case_600_baseline`) builds a spec whose wall κ is
/// lightweight (premise: `massiveness_weight(κ) == 0.0`), the falsy token
/// restores `(0.30, 0.30)` on it, and the default arm retires to
/// `(0.0, 0.0)` on the same spec — the real-spec witness that the token
/// is not gated behind any case or construction-type check (the flag site
/// reads the wall κ directly; `from_spec` delegates to the single
/// `from_spec_with_selector` site).
#[test]
fn real_case600_spec_carries_both_arms_of_the_token() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let spec = case_600_baseline();
    let kappa = spec.construction.wall.thermal_capacitance_per_area();
    let w = massiveness_weight(kappa);
    assert_eq!(
        w, 0.0,
        "premise: Case 600's wall (κ = {kappa}) is lightweight — w_solar must be 0.0"
    );

    // Falsy arm: the retired split restored on the real spec.
    std::env::set_var("FLX_R91_SOL2ST", "0");
    let restored = build(&spec);
    std::env::remove_var("FLX_R91_SOL2ST");
    assert_eq!(
        restored.solar.solar_distribution_to_air,
        0.30 * (1.0 - w),
        "falsy restore on Case 600: to_air = 0.30·(1−w_solar)"
    );
    assert_eq!(restored.solar.solar_beam_to_mass_fraction, 0.30);

    // Default arm: the same spec retires to (0.0, 0.0) — the unset
    // default also re-pinned here so this witness is self-contained.
    let retired = build(&spec);
    assert_eq!(retired.solar.solar_distribution_to_air, 0.0);
    assert_eq!(retired.solar.solar_beam_to_mass_fraction, 0.0);
}

/// (pin 3) Truthiness fails CLOSED to the retired default: unset, every
/// truthy token ("1", "TRUE", " yes " — trimmed, case-insensitive), and
/// every unrecognized value ("bogus", "2" — the parser warns and falls
/// back to the call-site default) all produce the RETIRED route
/// (0.0, 0.0). Only the falsy set restores. This is the discrimination
/// the round-91 flip-2 record relies on: `FLX_R91_SOL2ST=0` is the ONLY
/// way back to the incumbent split.
#[test]
fn token_truthiness_fails_closed_to_the_retired_default() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let spec = spec_with_wall_kappa(60_000.0, 240.0, "wall-truthiness");
    assert_eq!(spec.validate(), Ok(()));

    let mut arms: Vec<(Option<&str>, &str)> = vec![
        (None, "unset"),
        (Some("1"), "truthy"),
        (Some("TRUE"), "truthy case-insensitive"),
        (Some(" yes "), "truthy trimmed"),
        (Some("bogus"), "unrecognized -> warn + default"),
        (Some("2"), "unrecognized -> warn + default"),
    ];
    for (token, label) in arms.drain(..) {
        match token {
            Some(v) => std::env::set_var("FLX_R91_SOL2ST", v),
            None => std::env::remove_var("FLX_R91_SOL2ST"),
        }
        let model = build(&spec);
        std::env::remove_var("FLX_R91_SOL2ST");
        assert_eq!(
            model.solar.solar_distribution_to_air, 0.0,
            "{label}: non-falsy token must behave as unset (retired route)"
        );
        assert_eq!(
            model.solar.solar_beam_to_mass_fraction, 0.0,
            "{label}: non-falsy token must behave as unset (retired route)"
        );
    }
}

/// (pin 4) The full falsy set restores on this site: "0", "false", "no",
/// "off", and the empty string all rebuild the incumbent split
/// (0.30, 0.30) on the lightweight fixture — the canonical parser's falsy
/// set applied end-to-end at the flag site (env_bool.rs's deliberate
/// empty-string-is-falsy contract included).
#[test]
fn full_falsy_set_restores_the_split_on_this_site() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let spec = spec_with_wall_kappa(60_000.0, 240.0, "wall-falsy-set");
    assert_eq!(spec.validate(), Ok(()));
    for token in ["0", "false", "no", "off", ""] {
        std::env::set_var("FLX_R91_SOL2ST", token);
        let model = build(&spec);
        std::env::remove_var("FLX_R91_SOL2ST");
        assert_eq!(
            model.solar.solar_distribution_to_air, 0.30,
            "falsy token {token:?} must restore to_air = 0.30 (w_solar = 0)"
        );
        assert_eq!(
            model.solar.solar_beam_to_mass_fraction, 0.30,
            "falsy token {token:?} must restore beam = 0.30"
        );
    }
}

/// (pin 5) Construction-time capture + step-time consumption. The env is
/// read ONCE at `from_spec_with_selector`: a model built under the falsy
/// token keeps its restored values after the token is removed, a model
/// built afterwards carries the retired default, and — the teeth — the
/// two models DIVERGE under identical solar driving, because
/// `sol_to_air = sol_w · solar_distribution_to_air` (step_5r1c.rs) routes
/// the restored 0.30 share to air on every step. A restore that lost its
/// consumption path (or a default that started consuming) fails here.
#[test]
fn construction_time_capture_and_step_time_consumption() {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let spec = spec_with_wall_kappa(60_000.0, 240.0, "wall-consumption");
    assert_eq!(spec.validate(), Ok(()));

    // Build under the falsy token, then remove it: the captured fields
    // must not follow the env (construction-time semantics).
    std::env::set_var("FLX_R91_SOL2ST", "0");
    let mut restored = build(&spec);
    assert_eq!(restored.solar.solar_distribution_to_air, 0.30);
    assert_eq!(restored.solar.solar_beam_to_mass_fraction, 0.30);
    std::env::remove_var("FLX_R91_SOL2ST");
    assert_eq!(
        restored.solar.solar_distribution_to_air, 0.30,
        "construction-time capture: fields are fixed at from_spec, not re-read per step"
    );

    // A model built after the flip carries the retired default.
    let mut retired = build(&spec);
    assert_eq!(retired.solar.solar_distribution_to_air, 0.0);
    assert_eq!(retired.solar.solar_beam_to_mass_fraction, 0.0);

    // Consumption teeth: identical driving, divergent states.
    neutralize(&mut restored);
    neutralize(&mut retired);
    let mut d_air = 0.0_f64;
    let mut d_mass = 0.0_f64;
    for h in 0..24 {
        restored.step_physics(h, 12.0, DT);
        retired.step_physics(h, 12.0, DT);
        d_air = (restored.mass.air_temperatures[0] - retired.mass.air_temperatures[0]).abs();
        d_mass = (restored.mass.mass_temperatures[0] - retired.mass.mass_temperatures[0]).abs();
    }
    eprintln!(
        "r111 consumption pin: |Δt_air| = {d_air:.6e} K, |Δt_mass| = {d_mass:.6e} K after 24 h"
    );
    // Measured on the first green run (deterministic fixture): 2.244236e+1 K
    // air, 7.109481e+0 K mass — the floors sit far below those and far above
    // numerical noise, so only a real routing change crosses them.
    assert!(
        d_air > 10.0,
        "restored split must consume at step time: |Δt_air| = {d_air}"
    );
    assert!(
        d_mass > 3.0,
        "restored split must reach the mass state: |Δt_mass| = {d_mass}"
    );
}
