//! The mass-guard arm of the r137 ramp-persistence fix, in its own test
//! binary because it arms `FLX_R76_NO_NUM_REFRESH` (a process-global env
//! read inside the step — see the main contract file's header for the
//! finding, the verdict, and the fix shape).
//!
//! [Loop round 140 — the LOOP137 finding extended to the remaining
//! above-the-mass-guard persist families: the solar-lag persist, the r77
//! checker records, and the r70 t_s checker pair. The rejection contract
//! below now also pins those families at init; the recovery-twin test
//! proves the engine-facing half (the lag state seeds the next step) and
//! the checker-facing half (no rejected record may be mirrored); the
//! accepted-step pin guards the relocation's inertness with bit-exact
//! literals measured at the unfixed tip.]
//!
//! With the no-refresh diagnostic on, the sub-step air numerator keeps
//! the hour-start mass (`num_tm_ref`) instead of the sub-step mass, so
//! the degenerate-cm recipe (cm near the f64 ceiling, 100 s step) sends
//! the sub-step mass advance to inf WITHOUT poisoning the air node: the
//! air guard passes, #4500 passes (free-float demand is zero), and the
//! step rejects at the #4495 mass-node guard — the site the LOOP136
//! finding named for the R71/R75/R90 block. Pre-fix that rejection left
//! ALL FOUR ramp families advanced with the rejected step's drivers;
//! post-fix they stay at init. The typed error message is asserted to
//! name the Mass-node guard, proving which seam the contract pins.
//!
//! Teeth: T2 (revert only the R71/R75/R90 relocation, keep the R76 move)
//! reds THIS contract alone in the r137 suite — the other contracts'
//! rejections fire at the air guard, above this block.

use fluxion::api::error::FluxionError;
use fluxion::physics::cta::VectorField;
use fluxion::sim::boundary::ConstantGroundTemperature;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::invariant_checker::InvariantChecker;

// [Loop round 140 CI fix — the three R140 sections were separate #[test]
// fns; the process-global env arm races across parallel test threads (one
// test's remove_var pulls the arm out from under another's step, so a
// step that must reject at the mass guard can reject at the air guard
// instead). Merged into the single test fn this binary always required:
// one env set/remove, three sequential sections, identical asserts.]

#[test]
/// C5 — the mass-guard rejection seam: every ramp family stays at init
/// through a mass-node rejection. The env arm is set for this process
/// only (single test per binary, no parallel readers).
fn mass_guard_rejection_sections() {
    std::env::set_var("FLX_R76_NO_NUM_REFRESH", "1");

    let mut m = ThermalModel::<VectorField>::new(1);
    m.setpoints.zone_area = VectorField::from_scalar(48.0, 1);
    m.setpoints.zone_volume = VectorField::from_scalar(129.6, 1);
    m.setpoints.heating_setpoint = -50.0;
    m.setpoints.cooling_setpoint = 100.0;
    m.setpoints.temperatures = VectorField::from_scalar(20.0, 1);
    m.setpoints.air_density = VectorField::from_scalar(1.2, 1);
    m.setpoints.heat_capacity = VectorField::from_scalar(1005.0, 1);
    m.hvac.free_float = true;
    m.conduction.h_tr_em = VectorField::new(vec![0.32 * 48.0]);
    m.conduction.h_tr_w = VectorField::new(vec![3.0 * 12.0]);
    m.conduction.h_tr_floor = VectorField::new(vec![0.15 * 48.0]);
    m.conduction.h_ve = VectorField::new(vec![1.2 * 1000.0 * (0.5 * (48.0 * 2.7) / 3600.0)]);
    m.conduction.h_tr_ms = VectorField::new(vec![1000.0]);
    m.mass.thermal_capacitance = VectorField::new(vec![1.7e308]);
    m.mass.mass_temperatures = VectorField::from_scalar(20.0, 1);
    m.setpoints.loads = VectorField::from_scalar(30.0, 1);
    m.solar.solar_gains = VectorField::new(vec![400.0]);
    m.solar.opaque_solar_gains = VectorField::new(vec![50.0]);
    m.hvac.hvac_heating_capacity = 0.0;
    m.hvac.hvac_cooling_capacity = 0.0;
    m.conduction.ground_temperature = Box::new(ConstantGroundTemperature::new(13.0));
    m.update_optimization_cache();

    let result = m.try_step_physics(0, -10.0, 100.0);
    match result {
        Err(FluxionError::Validation(msg)) => {
            assert!(
                msg.contains("Mass-node"),
                "expected the mass-node guard, got: {msg}"
            );
            assert!(
                msg.contains("issue #4495"),
                "expected the #4495 envelope message, got: {msg}"
            );
        }
        other => panic!(
            "C5: expected Err(FluxionError::Validation) from the mass-node guard, got {:?}",
            other.map(|e| e.to_string())
        ),
    }

    assert_eq!(
        m.mass.r76_sol_air_prev.as_ref(),
        &[0.0],
        "the mass-guard rejection must not persist the sol-air driver"
    );
    assert!(
        !m.mass.r76_has_prev,
        "the mass-guard rejection must not set the sol-air seed flag"
    );
    assert_eq!(
        m.mass.r71_air_driver_num_prev.as_ref(),
        &[0.0],
        "the mass-guard rejection must not persist the num driver"
    );
    assert_eq!(
        m.mass.r71_air_driver_den_prev.as_ref(),
        &[0.0],
        "the mass-guard rejection must not persist the den driver"
    );
    assert_eq!(
        m.mass.r71_num_prev_load_slope.as_ref(),
        &[0.0],
        "the mass-guard rejection must not persist the load slope"
    );
    assert!(
        !m.mass.r75_vent_has_prev,
        "the mass-guard rejection must not set the vent seed flag"
    );
    assert_eq!(
        m.mass.r90_phi_st_prev.as_ref(),
        &[0.0],
        "the mass-guard rejection must not persist the surface gain"
    );
    assert_eq!(
        m.mass.r90_phi_m_prev.as_ref(),
        &[0.0],
        "the mass-guard rejection must not persist the mass gain"
    );
    assert_eq!(
        m.setpoints.temperatures.as_ref()[0],
        20.0,
        "the mass-guard rejection leaves the zone-air commit at init"
    );
    assert_eq!(
        m.mass.mass_temperatures.as_ref()[0],
        20.0,
        "the mass-guard rejection leaves the mass commit at init"
    );

    // [R140 — the LOOP137 finding extended: the solar-lag persist and the
    // r77/r70 checker records also sat above the mass-node guard, so this
    // rejection advanced them with the rejected step's drivers. Post-fix
    // every family below stays at init; pre-fix the lag carries the
    // rejected step's final sub-step value and both R77 records are Some.]
    assert_eq!(
        m.mass.solar_lag.as_ref(),
        &[0.0],
        "the mass-guard rejection must not advance the solar-lag ODE state"
    );
    assert!(
        m.mass.r77_mass_substep_coeffs.is_none(),
        "the mass-guard rejection must leave no R77 mass record for the checker to mirror"
    );
    assert!(
        m.mass.r77_air_substep_coeffs.is_none(),
        "the mass-guard rejection must leave no R77 air record"
    );
    assert!(
        m.conduction.applied_5r1c_class_boundary.is_none(),
        "the mass-guard rejection must leave no per-class boundary record (the Round-78 no-record convention)"
    );
    assert!(
        m.mass.r70_surface_node_temperatures.is_none(),
        "the mass-guard rejection must leave no r70 surface-node state"
    );
    assert!(
        m.mass.r70_surface_node_temperatures_prev.is_none(),
        "the mass-guard rejection must leave no r70 t_s_old snapshot"
    );

    // ---- R140 recovery-twin section ----

    let build = || {
        let mut m = ThermalModel::<VectorField>::new(1);
        m.setpoints.zone_area = VectorField::from_scalar(48.0, 1);
        m.setpoints.zone_volume = VectorField::from_scalar(129.6, 1);
        m.setpoints.heating_setpoint = -50.0;
        m.setpoints.cooling_setpoint = 100.0;
        m.setpoints.temperatures = VectorField::from_scalar(20.0, 1);
        m.setpoints.air_density = VectorField::from_scalar(1.2, 1);
        m.setpoints.heat_capacity = VectorField::from_scalar(1005.0, 1);
        m.hvac.free_float = true;
        m.conduction.h_tr_em = VectorField::new(vec![0.32 * 48.0]);
        m.conduction.h_tr_w = VectorField::new(vec![3.0 * 12.0]);
        m.conduction.h_tr_floor = VectorField::new(vec![0.15 * 48.0]);
        m.conduction.h_ve = VectorField::new(vec![1.2 * 1000.0 * (0.5 * (48.0 * 2.7) / 3600.0)]);
        m.conduction.h_tr_ms = VectorField::new(vec![1000.0]);
        m.mass.thermal_capacitance = VectorField::new(vec![10_000_000.0]);
        m.mass.mass_temperatures = VectorField::from_scalar(20.0, 1);
        m.setpoints.loads = VectorField::from_scalar(30.0, 1);
        m.solar.solar_gains = VectorField::new(vec![400.0]);
        m.solar.opaque_solar_gains = VectorField::new(vec![50.0]);
        m.hvac.hvac_heating_capacity = 0.0;
        m.hvac.hvac_cooling_capacity = 0.0;
        m.conduction.ground_temperature = Box::new(ConstantGroundTemperature::new(13.0));
        m.update_optimization_cache();
        m
    };

    let mut a = build();
    let mut b = build();
    for step in 0..2usize {
        let ra = a.try_step_physics(step, -10.0, 100.0);
        let rb = b.try_step_physics(step, -10.0, 100.0);
        assert!(ra.is_ok() && rb.is_ok(), "steps 0 and 1 must accept");
    }
    // A only: the degenerate-cm rejected step (the C5 recipe).
    a.mass.thermal_capacitance = VectorField::new(vec![1.7e308]);
    let rejected = a.try_step_physics(2, -10.0, 100.0);
    // The fixture's own wiring: the degenerate cm was the rejection
    // TRIGGER, not step state — restore it so A and B present identical
    // inputs to the checker below.
    a.mass.thermal_capacitance = VectorField::new(vec![10_000_000.0]);
    match rejected {
        Err(FluxionError::Validation(msg)) => {
            assert!(
                msg.contains("Mass-node") && msg.contains("issue #4495"),
                "the twin's rejected step must trip the mass-node guard, got: {msg}"
            );
        }
        other => panic!(
            "expected the mass-guard rejection, got {:?}",
            other.map(|e| e.to_string())
        ),
    }

    // Plain-write families: bit-identical between A and B.
    assert_eq!(
        a.mass.solar_lag.as_ref(),
        b.mass.solar_lag.as_ref(),
        "the rejected step must not advance the solar-lag seed"
    );
    assert_eq!(
        a.mass.r70_surface_node_temperatures, b.mass.r70_surface_node_temperatures,
        "the r70 surface-node state must match B's after the rejection"
    );
    assert_eq!(
        a.mass.r70_surface_node_temperatures_prev, b.mass.r70_surface_node_temperatures_prev,
        "the r70 t_s_old snapshot must match B's after the rejection"
    );
    assert_eq!(
        a.mass.air_temperatures.as_ref(),
        b.mass.air_temperatures.as_ref(),
        "committed zone air must match B's"
    );
    assert_eq!(
        a.mass.mass_temperatures.as_ref(),
        b.mass.mass_temperatures.as_ref(),
        "committed mass must match B's"
    );
    assert_eq!(
        a.mass.previous_mass_temperatures.as_ref(),
        b.mass.previous_mass_temperatures.as_ref(),
        "previous mass must match B's"
    );
    assert_eq!(a.mass.r76_has_prev, b.mass.r76_has_prev);
    assert_eq!(a.mass.r75_vent_has_prev, b.mass.r75_vent_has_prev);

    // Take-shaped record fields: the no-record convention. B still holds
    // its step-1 records (its last step accepted and installed); A must
    // present none.
    assert!(
        a.mass.r77_mass_substep_coeffs.is_none(),
        "a rejected step leaves no R77 mass record for the checker to mirror"
    );
    assert!(
        a.mass.r77_air_substep_coeffs.is_none(),
        "a rejected step leaves no R77 air record"
    );
    assert!(
        a.conduction.applied_5r1c_class_boundary.is_none(),
        "a rejected step leaves no per-class record for the checker to mirror"
    );

    // Checker-facing: A must read exactly the documented absent-record
    // fall-through — what B reads with its records surgically absent.
    let mut b_sans_records = b;
    b_sans_records.mass.r77_mass_substep_coeffs = None;
    b_sans_records.mass.r77_air_substep_coeffs = None;
    b_sans_records.conduction.applied_5r1c_class_boundary = None;
    let mut checker_a = InvariantChecker::new(1e-7);
    let mut checker_b = InvariantChecker::new(1e-7);
    let ra = checker_a.check_invariant(&a, 100.0, -10.0);
    let rb = checker_b.check_invariant(&b_sans_records, 100.0, -10.0);
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(
            ra.balance, rb.balance,
            "the checker must not mirror the rejected step's records: A's residual must equal the absent-record fall-through exactly"
        );
    }

    // ---- R140 accepted-step pin section ----

    let mut m = ThermalModel::<VectorField>::new(1);
    m.setpoints.zone_area = VectorField::from_scalar(48.0, 1);
    m.setpoints.zone_volume = VectorField::from_scalar(129.6, 1);
    m.setpoints.heating_setpoint = -50.0;
    m.setpoints.cooling_setpoint = 100.0;
    m.setpoints.temperatures = VectorField::from_scalar(20.0, 1);
    m.setpoints.air_density = VectorField::from_scalar(1.2, 1);
    m.setpoints.heat_capacity = VectorField::from_scalar(1005.0, 1);
    m.hvac.free_float = true;
    m.conduction.h_tr_em = VectorField::new(vec![0.32 * 48.0]);
    m.conduction.h_tr_w = VectorField::new(vec![3.0 * 12.0]);
    m.conduction.h_tr_floor = VectorField::new(vec![0.15 * 48.0]);
    m.conduction.h_ve = VectorField::new(vec![1.2 * 1000.0 * (0.5 * (48.0 * 2.7) / 3600.0)]);
    m.conduction.h_tr_ms = VectorField::new(vec![1000.0]);
    m.mass.thermal_capacitance = VectorField::new(vec![10_000_000.0]);
    m.mass.mass_temperatures = VectorField::from_scalar(20.0, 1);
    m.setpoints.loads = VectorField::from_scalar(30.0, 1);
    m.solar.solar_gains = VectorField::new(vec![400.0]);
    m.solar.opaque_solar_gains = VectorField::new(vec![50.0]);
    m.hvac.hvac_heating_capacity = 0.0;
    m.hvac.hvac_cooling_capacity = 0.0;
    m.conduction.ground_temperature = Box::new(ConstantGroundTemperature::new(13.0));
    m.update_optimization_cache();

    let result = m.try_step_physics(0, -10.0, 100.0);
    assert!(result.is_ok(), "the accepted-step fixture must accept");

    // Bit-exact literals measured at the unfixed tip (a2a7397d, same
    // commit as the contracts) — the relocation must not move any of them.
    assert_eq!(
        m.mass.solar_lag.as_ref(),
        &[0.2847692953618448],
        "accepted step must persist the solar-lag state bit-exactly"
    );
    let mass_rec = m
        .mass
        .r77_mass_substep_coeffs
        .as_ref()
        .expect("the accepted coupled step must persist the R77 mass record");
    assert_eq!(mass_rec[0].len(), 48, "12 sub-steps x 4 entries");
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(mass_rec[0][0], 0.999946423718882, "a_0");
        assert_eq!(mass_rec[0][1], 0.014902961777627507, "b_0");
        assert_eq!(mass_rec[0][2], 14400032.146629825, "d_0");
        assert_eq!(mass_rec[0][3], 0.00000046771431787824033, "bl_0");
        assert_eq!(
            mass_rec[0][46], 14400032.146629825,
            "d_last (the composite residual scaler)"
        );
    }
    let air = m
        .mass
        .r77_air_substep_coeffs
        .as_ref()
        .expect("the accepted step must persist the R77 air record");
    assert_eq!(air[0].len(), 108, "12 sub-steps x 9 slots");
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(air[0][4], 0.00004152921719554665, "h3od slot 0");
    }
    let cls = m
        .conduction
        .applied_5r1c_class_boundary
        .as_ref()
        .expect("the accepted step must persist the per-class boundary record");
    let c0 = cls[0].as_ref().expect("zone 0 must carry its class record");
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(c0.t_i, 25.27103326464767, "committed t_i");
        assert_eq!(c0.phi_st_zone, 7534.080000000001, "committed phi_st_zone");
    }

    std::env::remove_var("FLX_R76_NO_NUM_REFRESH");
}
