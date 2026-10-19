//! Regression harness for issue #4495 — `step_physics` drove the zone
//! temperature to `-inf` for extreme-but-finite inputs, newly observable
//! after #4490 unmasked deeper fuzz exploration.
//!
//! Crash input (nightly fuzz run 39247689735, artifact
//! `crash-abb71d2b03c62b25f7e5872e4cd5fac025f1eec9`, decoded with the
//! target's `arbitrary` layout):
//!
//! * `num_zones = 1`
//! * `apply_parameters([1.4462980080296809e-290, 0.0, 0.0])`
//!   (U-value absurdly small but finite; setpoints 0/0)
//! * `set_loads([5.0])`
//! * `step_physics(0, -2.8088955232222888e306, 1.5845632502852868e29)`
//!   (outdoor −2.81e306 °C, dt 1.58e29 s — both finite)
//!
//! The fuzz target's Invariant 3 (zone temperatures stay finite after each
//! step) fired at t=0 with `-inf`.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;

const CRASH_U_VALUE: f64 = 1.446298_0080296809e-290;
const CRASH_OUTDOOR: f64 = -2.8088955232222888e306;
const CRASH_DT: f64 = 1.5845632502852868e29;

#[test]
fn test_issue_4495_crash_input_zone_temperature_stays_finite() {
    let mut model = ThermalModel::<VectorField>::new(1);
    model.apply_parameters(&[CRASH_U_VALUE, 0.0, 0.0]);
    model.set_loads(&[5.0]);

    // Issue #4495: this step used to produce a `-inf` zone temperature and
    // commit it into the thermal state. The fix rejects the step with a
    // typed `Validation` error (issue #3638 graceful-degradation contract):
    // the legacy `step_physics` wrapper logs and returns 0.0 kWh, and the
    // thermal state is left untouched (zone air stays at its 20 °C init).
    let energy = model.step_physics(0, CRASH_OUTDOOR, CRASH_DT);
    assert_eq!(energy, 0.0, "rejected step must degrade to 0.0 kWh");

    for (z, &temp) in model.get_temperatures().iter().enumerate() {
        assert!(
            temp.is_finite(),
            "issue #4495: zone {z} temperature is non-finite after the crash step: {temp}"
        );
        assert_eq!(
            temp, 20.0,
            "issue #4495: rejected step must leave the zone state untouched"
        );
    }
}

/// The typed entry point surfaces the rejection as `FluxionError::Validation`
/// (the Issue #3638 contract), not a panic and not a corrupted state.
#[test]
fn test_issue_4495_try_step_physics_returns_typed_validation_error() {
    use fluxion::api::error::FluxionError;

    let mut model = ThermalModel::<VectorField>::new(1);
    model.apply_parameters(&[CRASH_U_VALUE, 0.0, 0.0]);
    model.set_loads(&[5.0]);

    let result = model.try_step_physics(0, CRASH_OUTDOOR, CRASH_DT);
    match result {
        Err(FluxionError::Validation(msg)) => {
            assert!(
                msg.contains("finite envelope"),
                "unexpected validation message: {msg}"
            );
        }
        other => panic!(
            "issue #4495: expected Err(FluxionError::Validation), got {:?}",
            other.map(|e| e.to_string())
        ),
    }
    for &temp in model.get_temperatures().iter() {
        assert!(temp.is_finite() && temp == 20.0, "state must be untouched");
    }
}

/// A tamed variant of the same class (large-but-less-extreme magnitudes)
/// must also stay finite — the guard is an envelope, not a special case
/// for the exact crash bits.
#[test]
fn test_issue_4495_near_crash_inputs_zone_temperature_stays_finite() {
    let mut model = ThermalModel::<VectorField>::new(1);
    model.apply_parameters(&[1e-10, 0.0, 0.0]);
    model.set_loads(&[5.0]);
    let _energy = model.step_physics(0, -1e150, 1e20);
    for (z, &temp) in model.get_temperatures().iter().enumerate() {
        assert!(
            temp.is_finite(),
            "issue #4495: zone {z} temperature is non-finite for large finite inputs: {temp}"
        );
    }
}
