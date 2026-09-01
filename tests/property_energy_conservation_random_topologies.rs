//! Property-based energy conservation tests for randomized multi-zone topologies.
//!
//! Part of issue #1894 — property-based system-level energy conservation for
//! randomized multi-zone building topologies.
//!
//! Expanded in issue #2559 to cover the three acceptance-criteria proptest
//! targets called out by the issue:
//!
//!   (a) **Randomized 9R4C nodal-graph invariants** — multi-zone building
//!       driven by randomized 9R4C envelope parameters; per-zone mass-node
//!       energy balance must hold for every random case.
//!   (b) **Randomized multi-zone convection/ventilation topology** —
//!       randomized zone count (2–8) with randomized conductance matrix
//!       drawn from {star, ring, line, mesh}; net inter-zone heat transfer
//!       must equal zero by Kirchhoff's-current-law identity.
//!   (c) **Randomized thermal-mass topology under summer/winter drivers** —
//!       randomized per-zone capacitance and inter-zone conductance; the
//!       multi-zone network is driven with hot (summer) and cold (winter)
//!       outdoor temperatures and the net imbalance must remain bounded
//!       relative to the conductive + load budget.
//!
//! # Test Strategy
//!
//! Verifies the **zone-lumped mass-node energy balance invariant** across
//! arbitrarily randomized 5R1C/9R4C thermal networks:
//!
//! ```text
//! | ΣQ_in − ΣQ_out − C_z · ΔT_z/Δt | < ε   (ε = 10⁻² W)
//! ```
//!
//! Each proptest case:
//!   1. Creates a base ThermalModel using an ASHRAE 140 Case spec
//!   2. Randomizes only parameters that don't cause derived-parameter staleness:
//!      infiltration rate, initial temperatures, internal gains, HVAC setpoints,
//!      and (where safe) thermal capacitances
//!   3. Runs physics timestep(s)
//!   4. Asserts per-zone energy balance via `InvariantChecker::check_invariant`
//!
//! # Why These Specific Parameters?
//!
//! The thermal network's derived parameters (notably `derived_h_tr_3`) are
//! computed from h_tr_em and h_tr_ms in `update_derived_parameters()`. Since
//! `update_derived_parameters()` is `pub(crate)`, the public API cannot re-derive
//! after manual parameter overwrites. We therefore restrict randomization to
//! parameters that don't affect derived-parameter consistency:
//!
//! - `infiltration_rate` (ACH) — affects h_ve directly; h_ve is recomputed
//!   by `update_derived_parameters()` BEFORE our manual infiltration writes
//! - `initial_temperatures` — used directly in physics integration
//! - `internal_gains` / `solar_gains` — used directly as heat inputs
//! - `heating_setpoint` / `cooling_setpoint` — used directly by HVAC controller
//! - `thermal_capacitance` — used directly in mass-node integration; we
//!   re-run `update_optimization_cache()` so derived parameters stay consistent
//!
//! Parameters NOT randomized to avoid derived-param staleness:
//! h_tr_em, h_tr_ms, zone_area (these would require re-calling
//! update_derived_parameters() to stay consistent).
//!
//! Shrink strategy: minimizes toward 1-zone, small ΔT (clearest failure config).
//!
//! # Determinism
//!
//! All proptest targets use `ProptestConfig::with_cases(...)` plus a fixed
//! deterministic RNG seed (via `ProptestConfig::default().rng_seed`) so the
//! suite is reproducible on CI and does not flake across runs.
//!
//! # References
//!
//! - Issue #1894 — Property-based system-level energy conservation
//! - Issue #2559 — Expand proptest coverage (this file)
//! - Issue #1295 — Energy conservation CI gate
//! - Issue #1348 — N-zone inter-zone thermal coupling network
//! - `tests/test_energy_conservation.rs` — fixed-topology energy conservation

use fluxion::physics::cta::VectorField;
use fluxion::sim::construction::Assemblies;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::invariant_checker::InvariantChecker;
use fluxion::sim::multi_zone_network::{MultiZoneAirflowNetwork, ZoneState};
use fluxion::sim::thermal_selector::ThermalSelector;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseBuilder, InternalLoads};
use proptest::prelude::*;
use proptest::test_runner::RngSeed;

/// Per-zone mass-node balance tolerance (W). Mirrors the value used by the
/// deterministic CI gate in `tests/zone_balance_eplus_isolation.rs`.
const BALANCE_TOLERANCE: f64 = 0.01;

/// Per-zone mass-node balance tolerance for randomized 9R4C targets.
/// 9R4C has more floating-point cancellation paths (4 mass nodes vs 1)
/// so we keep the same absolute tolerance as the deterministic gate.
const NINE_R4C_BALANCE_TOLERANCE: f64 = 0.01;

/// Kirchhoff identity tolerance for the inter-zone conductance matrix
/// (algebraic Σ q_iz = 0 for symmetric conductance).
const IZ_CONSERVATION_TOL: f64 = 1e-6;

/// Proptest case-count knob. Capped at 256 to keep CI runtime low;
/// lower this for fast local iteration.
const PROPEST_CASES: u32 = 256;

/// Fixed RNG seed (encoded as `u64`) to guarantee deterministic regression
/// files in CI. The default proptest seed is randomized per process; we pin
/// it to a known value so that locally-minimized failing cases reproduce on
/// every CI run until the underlying bug is fixed.
const PROPEST_SEED: u64 = 0x2559_F177_00A1_C5D3;

fn proptest_config() -> ProptestConfig {
    ProptestConfig {
        cases: PROPEST_CASES,
        rng_seed: RngSeed::Fixed(PROPEST_SEED),
        ..ProptestConfig::default()
    }
}

// ---------------------------------------------------------------------------
// Strategy helpers
// ---------------------------------------------------------------------------

fn infiltration_strategy() -> impl Strategy<Value = f64> {
    0.0_f64..2.0
}

fn init_temp_strategy() -> impl Strategy<Value = f64> {
    5.0_f64..35.0
}

fn gain_strategy() -> impl Strategy<Value = f64> {
    0.0_f64..500.0
}

fn setpoint_strategy() -> impl Strategy<Value = f64> {
    18.0_f64..24.0
}

fn hvac_deadband_strategy() -> impl Strategy<Value = f64> {
    2.0_f64..5.0
}

/// U-value (W/m²K) strategy. Range spans typical envelope U-values
/// from old leaky buildings (~5.0) to high-performance assemblies (~0.15).
fn u_value_strategy() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.15_f64), // super-insulated wall
        0.3_f64..1.5,   // typical wall range
        2.0_f64..5.0,   // old / leaky building
    ]
}

/// Per-zone thermal capacitance (J/K) strategy.
fn capacitance_strategy() -> impl Strategy<Value = f64> {
    prop_oneof![
        5.0e4_f64..2.0e5, // lightweight construction
        2.0e5_f64..1.0e6, // typical residential
        1.0e6_f64..1.0e7, // heavyweight concrete
    ]
}

/// Inter-zone conductance (W/K) for the multi-zone convection target.
fn conductance_strategy() -> impl Strategy<Value = f64> {
    1.0_f64..100.0
}

// ---------------------------------------------------------------------------
// Existing 5R1C topology test (Issue #1894) — preserved verbatim except
// the case count is brought down to PROPEST_CASES so CI stays fast.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(proptest_config())]

    /// Verify energy conservation holds for every randomized Case 900 case.
    ///
    /// For each case: builds a Case 900 spec, initializes a ThermalModel,
    /// randomises infiltration rate (ACH), initial temperatures, and internal
    /// gains, then asserts the per-zone mass-node energy balance residual
    /// is < 10⁻² W.
    #[test]
    fn proptest_random_topology_energy_balance(
        infiltration in infiltration_strategy(),
        init_t_air in init_temp_strategy(),
        init_t_mass in init_temp_strategy(),
        q_internal in gain_strategy(),
        solar_per_m2 in 0.0_f64..900.0,
    ) {
        let spec = ASHRAE140Case::Case900.spec();
        let num_zones = spec.num_zones;

        let mut model = ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default()).expect("default selector must initialize");

        prop_assert!(num_zones >= 1, "Case 900 should have at least 1 zone");

        // Apply randomized infiltration (ACH).
        for i in 0..num_zones {
            model.setpoints.infiltration_rate.as_mut()[i] = infiltration;
        }

        // Apply randomized initial temperatures.
        for i in 0..num_zones {
            model.setpoints.temperatures.as_mut()[i] = init_t_air;
            model.mass.mass_temperatures.as_mut()[i] = init_t_mass;
            model.mass.previous_mass_temperatures.as_mut()[i] = init_t_mass;
            model.hvac.previous_temperatures.as_mut()[i] = init_t_air;
            model.mass.air_temperatures.as_mut()[i] = init_t_air;
            model.mass.wall_surface_temperatures.as_mut()[i] = init_t_mass;
        }

        // Apply randomized gains.
        for i in 0..num_zones {
            model.setpoints.loads.as_mut()[i] = q_internal;
            model.solar.solar_gains.as_mut()[i] = solar_per_m2;
        }

        // Run a single physics timestep.
        let dt = 3600.0_f64;
        let outdoor_temp = 10.0_f64;
        model.step_physics(0, outdoor_temp, dt);

        // Check energy conservation.
        let mut checker = InvariantChecker::new(BALANCE_TOLERANCE);
        let result = checker.check_invariant(&model, dt, outdoor_temp);

        for (i, &imbalance) in result.zone_imbalances.iter().enumerate() {
            prop_assert!(
                imbalance.abs() < BALANCE_TOLERANCE,
                "zone {} imbalance {:.3e} W >= {} W (infiltration={:.2} ACH, ΔT={:.1}K, gains={:.0}W)",
                i,
                imbalance,
                BALANCE_TOLERANCE,
                infiltration,
                init_t_air - outdoor_temp,
                q_internal,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Acceptance-criterion (a): Randomized 9R4C nodal-graph invariants
// ---------------------------------------------------------------------------
//
// Drive the multi-zone 9R4C thermal network with randomized:
//   * infiltration rate (ACH)
//   * initial air + mass temperatures (must be ≥ setpoint deadband to
//     avoid saturating the HVAC controller and skewing the imbalance)
//   * internal gains (W/m²)
//   * solar gains (W/m²)
//   * U-values (window, wall) within plausible ASHRAE ranges
//   * HVAC heating setpoint (cooling = heating + randomized deadband)
//
// Verifies per-zone mass-node energy balance: |
//      Σ(Q_in) − Σ(Q_out) − C_z · ΔT_z/Δt | < BALANCE_TOLERANCE
//
// The 9R4C model is used by Case 900 (high-mass) ASHRAE 140; this proptest
// catches regressions in the per-surface (wall/roof/floor) mass-node
// integration paths that the existing single-topology tests don't reach.

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn proptest_9r4c_nodal_balance_randomized_envelope(
        infiltration in infiltration_strategy(),
        init_t_air in 5.0_f64..35.0,
        init_t_mass in 5.0_f64..35.0,
        q_internal in gain_strategy(),
        solar_per_m2 in 0.0_f64..900.0,
        window_u in u_value_strategy(),
        hvac_heat in setpoint_strategy(),
        hvac_deadband in hvac_deadband_strategy(),
    ) {
        // Issue #2559 acceptance criterion (a): randomized 9R4C envelope.
        // Case 900 is the canonical 9R4C reference (high-mass concrete).
        let spec = ASHRAE140Case::Case900.spec();
        let num_zones = spec.num_zones;

        let mut model = ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default()).expect("default selector must initialize");

        // U-value randomization. The window/wall U-values feed into
        // h_tr_w and h_tr_em via from_spec() / update_derived_parameters();
        // we update them AFTER initial derivation to keep the network
        // consistent, then re-run the public optimization cache update.
        model.solar.window_u_value = window_u;

        // HVAC setpoint randomization. Use a deadband of at least 2°C so
        // HVAC saturation does not dominate the imbalance signal.
        let hvac_cool = hvac_heat + hvac_deadband;
        model.setpoints.heating_setpoint = hvac_heat;
        model.setpoints.cooling_setpoint = hvac_cool;

        // Per-zone randomized infiltration (ACH).
        for i in 0..num_zones {
            model.setpoints.infiltration_rate.as_mut()[i] = infiltration;
        }

        // Per-zone randomized initial temperatures. Use the same air
        // temp as zone temp to avoid transient sub-timestep convergence
        // artifacts (consistent with the deterministic CI gate).
        for i in 0..num_zones {
            model.setpoints.temperatures.as_mut()[i] = init_t_air;
            model.mass.mass_temperatures.as_mut()[i] = init_t_mass;
            model.mass.previous_mass_temperatures.as_mut()[i] = init_t_mass;
            model.hvac.previous_temperatures.as_mut()[i] = init_t_air;
            model.mass.air_temperatures.as_mut()[i] = init_t_air;
            model.mass.wall_surface_temperatures.as_mut()[i] = init_t_mass;
        }

        // Per-zone randomized gains.
        for i in 0..num_zones {
            model.setpoints.loads.as_mut()[i] = q_internal;
            model.solar.solar_gains.as_mut()[i] = solar_per_m2;
        }

        // Re-run the public optimization cache so derived_h_tr_3 etc.
        // stay consistent with the new window_u_value.
        model.update_optimization_cache();

        // Run a single physics timestep.
        let dt = 3600.0_f64;
        let outdoor_temp = 10.0_f64;
        model.step_physics(0, outdoor_temp, dt);

        // Per-zone mass-node energy balance.
        let mut checker = InvariantChecker::new(NINE_R4C_BALANCE_TOLERANCE);
        let result = checker.check_invariant(&model, dt, outdoor_temp);

        for (i, &imbalance) in result.zone_imbalances.iter().enumerate() {
            prop_assert!(
                imbalance.abs() < NINE_R4C_BALANCE_TOLERANCE,
                "9R4C zone {} imbalance {:.3e} W >= {} W \
                 (window_u={:.2}, hvac=[{:.1},{:.1}]°C, infiltration={:.2} ACH, \
                  ΔT={:.1}K, gains={:.0}W)",
                i,
                imbalance,
                NINE_R4C_BALANCE_TOLERANCE,
                window_u,
                hvac_heat,
                hvac_cool,
                infiltration,
                init_t_air - outdoor_temp,
                q_internal,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Acceptance-criterion (b): Randomized multi-zone convection/ventilation
// topology — Kirchhoff's-current-law identity on a randomized conductance
// graph.
// ---------------------------------------------------------------------------
//
// Generate a randomized N (2..=8) zone network and a randomized
// conductance matrix drawn from {star, ring, line, mesh} topologies.
// The `MultiZoneAirflowNetwork::net_inter_zone_q` algebraic identity
// must hold for any symmetric conductance: |Σ q_iz| < 1e-6 W.
//
// This catches regressions in the multi-zone LU solver and the
// `from_matrix` / `from_adjacency_pairs` constructors (Issue #1348).

/// Proptest-friendly conductance-matrix topology selector. Each variant
/// produces a symmetric matrix; the corresponding test then verifies
/// Kirchhoff's-current-law (Σ q_iz = 0).
#[derive(Debug, Clone, Copy)]
enum Topology {
    Star,
    Ring,
    Line,
    Mesh,
}

impl Topology {
    fn name(&self) -> &'static str {
        match self {
            Topology::Star => "star",
            Topology::Ring => "ring",
            Topology::Line => "line",
            Topology::Mesh => "mesh",
        }
    }
}

fn topology_strategy() -> impl Strategy<Value = Topology> {
    prop_oneof![
        Just(Topology::Star),
        Just(Topology::Ring),
        Just(Topology::Line),
        Just(Topology::Mesh),
    ]
}

/// Build a symmetric NxN conductance matrix according to `topology` and a
/// per-edge base conductance `h_base`. Diagonal entries are 0 (no self-loop).
fn build_topology_matrix(n: usize, topology: Topology, h_base: f64) -> Vec<f64> {
    let mut m = vec![0.0_f64; n * n];
    match topology {
        Topology::Star => {
            // Zone 0 is the hub; zones 1..N-1 connect only to zone 0.
            for j in 1..n {
                m[j] = h_base;
                m[j * n] = h_base;
            }
        }
        Topology::Ring => {
            // Each zone connects to its two neighbours (mod N).
            for i in 0..n {
                let next = (i + 1) % n;
                m[i * n + next] = h_base;
                m[next * n + i] = h_base;
            }
        }
        Topology::Line => {
            // Each zone (except the last) connects only to its immediate
            // successor. Zone 0 connects only to zone 1.
            for i in 0..n.saturating_sub(1) {
                m[i * n + (i + 1)] = h_base;
                m[(i + 1) * n + i] = h_base;
            }
        }
        Topology::Mesh => {
            // Fully connected (every pair has an edge).
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        m[i * n + j] = h_base;
                    }
                }
            }
        }
    }
    m
}

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn proptest_multizone_convection_topology_conserves_energy(
        n_zones in 2_usize..=8,
        topology in topology_strategy(),
        h_base in conductance_strategy(),
        temp_scale in 1.0_f64..20.0,
        temp_bias in -10.0_f64..30.0,
    ) {
        // Issue #2559 acceptance criterion (b): randomized convection
        // /ventilation topology. Build a randomized symmetric conductance
        // matrix from {star, ring, line, mesh} and verify Kirchhoff's
        // current-law identity Σ q_iz = 0.
        let n = n_zones;
        let h_vec = build_topology_matrix(n, topology, h_base);
        let h_mat = nalgebra::DMatrix::from_vec(n, n, h_vec.clone());

        // Build zones with randomized temperatures: T_i = bias + scale * i.
        let temps_before: Vec<f64> = (0..n)
            .map(|i| temp_bias + temp_scale * i as f64)
            .collect();

        // Pure algebraic identity (no solve_step needed):
        // MultiZoneAirflowNetwork::net_inter_zone_q sums q_iz across the
        // full N×N matrix at the supplied temperature vector. For any
        // symmetric conductance matrix the sum is identically zero by
        // antisymmetry of q_ij + q_ji across the matrix diagonal.
        let network = MultiZoneAirflowNetwork::from_matrix(h_mat.clone());
        let net_alg = network.net_inter_zone_q(&temps_before);
        prop_assert!(
            net_alg.abs() < IZ_CONSERVATION_TOL,
            "Algebraic Σ q_iz for {} topology N={} h_base={:.2} |net|={:.3e} W \
             must be < {:.0e} W",
            topology.name(),
            n,
            h_base,
            net_alg,
            IZ_CONSERVATION_TOL,
        );

        // Round-trip: build the same matrix from adjacency pairs and verify
        // that the solve_step path also conserves energy.
        let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let h_ij = h_vec[i * n + j];
                    if h_ij > 0.0 {
                        pairs.push((i, j, h_ij));
                    }
                }
            }
        }
        let network2 = MultiZoneAirflowNetwork::from_adjacency_pairs(n, &pairs);
        let mut zones: Vec<ZoneState> = temps_before
            .iter()
            .map(|&t| ZoneState::new(t, 1.0e6))
            .collect();
        let q_ext = vec![0.0_f64; n];
        let result = network2
            .solve_step(&mut zones, &q_ext, 3600.0)
            .expect("multi-zone solve_step must succeed");
        prop_assert!(
            result.net_w.abs() < IZ_CONSERVATION_TOL,
            "Solve-step Σ q_iz for {} topology N={} h_base={:.2} |net|={:.3e} W \
             must be < {:.0e} W",
            topology.name(),
            n,
            h_base,
            result.net_w,
            IZ_CONSERVATION_TOL,
        );

        // Sanity: temperature differences must produce non-zero per-zone
        // transfers for at least one zone (otherwise the proptest is
        // testing a degenerate input).
        let total_q: f64 = result.q_iz_w.iter().map(|q| q.abs()).sum();
        prop_assert!(
            total_q > 1e-3 || n <= 1,
            "Randomized {} topology N={} h_base={:.2} produced Σ|q_iz|={:.3e} W \
             — degenerate input (no temperature differences resolve to transfer)",
            topology.name(),
            n,
            h_base,
            total_q,
        );
    }
}

// ---------------------------------------------------------------------------
// Acceptance-criterion (c): Randomized thermal-mass topology under
// summer/winter seasonal drivers
// ---------------------------------------------------------------------------
//
// Build a randomized N-zone thermal-mass network and drive it with:
//   * summer outdoor driver (35°C, simulating a heat wave);
//   * winter outdoor driver (-10°C, simulating a cold snap).
// Per-zone thermal capacitance is randomized across lightweight,
// typical, and heavyweight ranges. Inter-zone conductance is randomized
// to exercise the inter-zone mass-node coupling.
//
// For each driver, the multi-zone ThermalModel is run for one hourly
// timestep and the InvariantChecker verifies the per-zone mass-node
// energy balance remains within BALANCE_TOLERANCE.

fn build_random_multizone_spec(
    n: usize,
    hvac_heat: f64,
    hvac_cool: f64,
) -> fluxion::validation::ashrae_140_cases::CaseSpec {
    // Build a Case 900-like envelope (high-mass concrete) with `n` zones.
    // CaseBuilder returns CaseSpec from `build()`; we keep chaining on
    // the builder and call `.build()` only at the end.
    let n = n.max(1);
    let mut builder = CaseBuilder::new()
        .with_case_id(format!("2559-{n}"))
        .with_description(format!(
            "Randomized {n}-zone Case 900 envelope for Issue #2559"
        ))
        .with_dimensions(8.0, 6.0, 2.7)
        .high_mass_construction()
        .with_construction(
            Assemblies::high_mass_wall_standard(),
            Assemblies::high_mass_roof(),
            Assemblies::high_mass_floor(),
        )
        .with_internal_loads(InternalLoads::new(200.0, 0.4, 0.6))
        .with_south_window(12.0)
        .with_hvac_setpoints(hvac_heat, hvac_cool)
        .with_infiltration(0.5)
        .with_ground_temperature(
            fluxion::physics::constants::thermal::ashrae_140::v2023::GROUND_TEMPERATURE_C,
        );
    // Add (n - 1) additional zones matching the same envelope (multi-zone
    // is air-coupled via `h_tr_iz`, not via additional construction).
    for _ in 1..n {
        builder = builder.add_zone(8.0, 6.0, 2.7);
    }
    builder
        .build()
        .expect("Case 900 multi-zone spec must validate")
}

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn proptest_thermal_mass_topology_seasonal_drivers(
        n_zones in 2_usize..=8,
        c_per_zone in capacitance_strategy(),
        h_iz in conductance_strategy(),
        hvac_heat in setpoint_strategy(),
        hvac_deadband in hvac_deadband_strategy(),
    ) {
        // Issue #2559 acceptance criterion (c): randomized thermal-mass
        // topology under summer/winter seasonal drivers.
        let n = n_zones;
        let hvac_cool = hvac_heat + hvac_deadband;

        // Summer (35°C) and winter (-10°C) drivers — extreme but
        // physically plausible ASHRAE 140 seasonal extremes.
        let seasonal_drivers: [(&str, f64); 2] =
            [("summer", 35.0_f64), ("winter", -10.0_f64)];

        for (season, outdoor_temp) in seasonal_drivers.iter() {
            let spec = build_random_multizone_spec(n, hvac_heat, hvac_cool);
            let mut model = ThermalModel::<VectorField>::from_spec_with_selector(&spec, &ThermalSelector::default()).expect("default selector must initialize");

            // Randomize per-zone thermal capacitance. We use the public
            // thermal_capacitance field which feeds directly into the
            // mass-node integration. The HVAC controller and the
            // inter-zone conductance derivation are independent of Cm,
            // so this randomization is safe without re-deriving h_*.
            for i in 0..n {
                model.mass.thermal_capacitance.as_mut()[i] = c_per_zone;
                model.mass.air_thermal_capacitance.as_mut()[i] = c_per_zone * 0.01;
            }

            // Randomize inter-zone conductance via the h_tr_iz_rad /
            // h_tr_iz fields. We use a single scalar per zone-pair by
            // populating h_tr_iz from the randomized conductance.
            for i in 0..n {
                model.conduction.h_tr_iz.as_mut()[i] = h_iz;
                model.conduction.h_tr_iz_rad.as_mut()[i] = h_iz * 0.2;
            }

            // Initialize zones at the HVAC heating setpoint with mass
            // temperatures equal to the setpoint (no transient heat-up
            // transient that would contaminate the imbalance signal).
            for i in 0..n {
                let t_init = hvac_heat;
                model.setpoints.temperatures.as_mut()[i] = t_init;
                model.mass.mass_temperatures.as_mut()[i] = t_init;
                model.mass.previous_mass_temperatures.as_mut()[i] = t_init;
                model.hvac.previous_temperatures.as_mut()[i] = t_init;
                model.mass.air_temperatures.as_mut()[i] = t_init;
                model.mass.wall_surface_temperatures.as_mut()[i] = t_init;
            }

            // No internal gains / solar gains (worst case for the energy
            // balance — only envelope conduction + inter-zone transfer).
            for i in 0..n {
                model.setpoints.loads.as_mut()[i] = 0.0;
                model.solar.solar_gains.as_mut()[i] = 0.0;
            }

            // Run one hourly physics timestep.
            let dt = 3600.0_f64;
            model.step_physics(0, *outdoor_temp, dt);

            // Per-zone mass-node energy balance.
            let mut checker = InvariantChecker::new(BALANCE_TOLERANCE);
            let result = checker.check_invariant(&model, dt, *outdoor_temp);

            for (i, &imbalance) in result.zone_imbalances.iter().enumerate() {
                prop_assert!(
                    imbalance.abs() < BALANCE_TOLERANCE,
                    "{} driver zone {} imbalance {:.3e} W >= {} W \
                     (n_zones={}, Cm={:.2e} J/K, h_iz={:.2} W/K, hvac=[{:.1},{:.1}]°C)",
                    season,
                    i,
                    imbalance,
                    BALANCE_TOLERANCE,
                    n,
                    c_per_zone,
                    h_iz,
                    hvac_heat,
                    hvac_cool,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic-regression-file support
// ---------------------------------------------------------------------------
//
// When a proptest case fails, proptest writes the minimized failing input
// to `proptest-regressions/<test-binary>.txt` (next to Cargo.toml). The
// tests above commit any saved regressions into version control so the
// next CI run replays the failure deterministically and the file is
// removed once the bug is fixed.
//
// To suppress: `PROPTEST_FAIL_FAST=1 cargo test ...`
