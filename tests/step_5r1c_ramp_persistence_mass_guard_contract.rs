//! The mass-guard arm of the r137 ramp-persistence fix, in its own test
//! binary because it arms `FLX_R76_NO_NUM_REFRESH` (a process-global env
//! read inside the step — see the main contract file's header for the
//! finding, the verdict, and the fix shape).
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

#[test]
/// C5 — the mass-guard rejection seam: every ramp family stays at init
/// through a mass-node rejection. The env arm is set for this process
/// only (single test per binary, no parallel readers).
fn mass_guard_rejection_leaves_every_ramp_family_at_init() {
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

    std::env::remove_var("FLX_R76_NO_NUM_REFRESH");
}
