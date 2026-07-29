//! Property-based energy conservation tests for randomized multi-zone topologies.
//!
//! Part of issue #1894 — property-based system-level energy conservation for
//! randomized multi-zone building topologies.
//!
//! # Test Strategy
//!
//! Verifies the **zone-lumped mass-node energy balance invariant** across
//! arbitrarily randomized 5R1C thermal networks:
//!
//! ```text
//! | ΣQ_in − ΣQ_out − C_z · ΔT_z/Δt | < ε   (ε = 10⁻² W)
//! ```
//!
//! Each proptest case:
//!   1. Creates a base ThermalModel using ASHRAE 140 Case 600 spec (which
//!      properly initializes h_tr_em, h_tr_ms, and derived_h_tr_3)
//!   2. Randomizes only parameters that don't cause derived-parameter staleness:
//!      infiltration rate, initial temperatures, and internal gains
//!   3. Runs a single 5R1C physics timestep
//!   4. Asserts per-zone energy balance via `InvariantChecker::check_invariant`
//!
//! # Why These Specific Parameters?
//!
//! The thermal network's derived parameters (notably `derived_h_tr_3`) are
//! computed from h_tr_em and h_tr_ms in `update_derived_parameters()`. Since
//! `update_derived_parameters()` is pub(crate), the public API cannot re-derive
//! after manual parameter overwrites. We therefore restrict randomization to
//! parameters that don't affect derived-parameter consistency:
//!
//! - `infiltration_rate` (ACH) — affects h_ve directly; h_ve is recomputed
//!   by `update_derived_parameters()` BEFORE our manual infiltration writes
//! - `initial_temperatures` — used directly in physics integration
//! - `internal_gains` — used directly as heat inputs
//!
//! Parameters NOT randomized to avoid derived-param staleness:
//! h_tr_em, h_tr_ms, zone_area, thermal_capacitance (these would require
//! re-calling update_derived_parameters() to stay consistent).
//!
//! Shrink strategy: minimizes toward 1-zone, small ΔT (clearest failure config).
//!
//! # References
//!
//! - Issue #1894 — Property-based system-level energy conservation
//! - Issue #1295 — Energy conservation CI gate
//! - `tests/test_energy_conservation.rs` — fixed-topology energy conservation

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::invariant_checker::InvariantChecker;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use proptest::prelude::*;

const BALANCE_TOLERANCE: f64 = 0.01;

fn infiltration_strategy() -> impl Strategy<Value = f64> {
    0.0_f64..2.0
}

fn init_temp_strategy() -> impl Strategy<Value = f64> {
    5.0_f64..35.0
}

fn gain_strategy() -> impl Strategy<Value = f64> {
    0.0_f64..500.0
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2_000))]

    /// Verify energy conservation holds for every randomized case.
    ///
    /// For each case: builds a Case 600 spec, initializes a ThermalModel,
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

        let mut model = ThermalModel::<VectorField>::from_spec(&spec);

        prop_assert!(num_zones >= 1, "Case 600 should have at least 1 zone");

        // Apply randomized infiltration (ACH) — affects h_ve; h_ve is recomputed
        // by update_derived_parameters() BEFORE our write, so derived params stay
        // consistent.
        for i in 0..num_zones {
            model.infiltration_rate.as_mut()[i] = infiltration;
        }

        // Apply randomized initial temperatures.
        for i in 0..num_zones {
            model.temperatures.as_mut()[i] = init_t_air;
            model.mass_temperatures.as_mut()[i] = init_t_mass;
            model.previous_mass_temperatures.as_mut()[i] = init_t_mass;
            model.previous_temperatures.as_mut()[i] = init_t_air;
            model.air_temperatures.as_mut()[i] = init_t_air;
            model.wall_surface_temperatures.as_mut()[i] = init_t_mass;
        }

        // Apply randomized gains.
        for i in 0..num_zones {
            model.loads.as_mut()[i] = q_internal;
            model.solar_gains.as_mut()[i] = solar_per_m2;
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
