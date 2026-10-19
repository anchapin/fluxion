//! Regression harness for issue #4500 — the `zone_balance_solver` fuzz
//! target's Invariant 2 asserted `step_physics(...) >= 0.0`, but the engine
//! contract is SIGNED net HVAC energy (heating W·dt minus cooling W·dt), so
//! any net-cooling step legitimately returns a negative value. The assert
//! was unreachable until #4490's non-finite-parameter filter let the fuzzer
//! reach real stepping; #4497 then unblocked deeper exploration and a
//! net-cooling first step fired it (nightly run 39317932747).
//!
//! Crash input (artifact `crash-689cb9c2`, decoded with the target's
//! `arbitrary` layout):
//!
//! * `num_zones = 1`
//! * `apply_parameters([1.802222908443235e-307, 1.8048304168711733e-307, 0.0])`
//!   (U-value and setpoints denormal-small but finite)
//! * `set_loads([5.0])`
//! * `step_physics(0, 3.0517578125e-05, 3.0517578125e-05)`
//!   (outdoor ≈ 3.05e-05 °C, dt = 2⁻¹⁵ s — both finite)
//!
//! No engine change is owed: the fix corrects the harness invariant (finite
//! signed output; non-negative per-zone accumulators) and pins the decoded
//! crash constants. Default builds are untouched.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;

const CRASH_U_VALUE: f64 = 1.802222908443235e-307;
const CRASH_HEATING: f64 = 1.8048304168711733e-307;
const CRASH_COOLING: f64 = 0.0;
const CRASH_OUTDOOR: f64 = 3.0517578125e-05;
const CRASH_DT: f64 = 3.0517578125e-05;

/// The exact crash input steps cleanly and returns FINITE SIGNED net energy
/// (negative here: the 5 W internal load drives the zone above the ~0 set
/// points, so the step is net cooling). State stays finite.
#[test]
fn test_issue_4500_crash_input_signed_energy_is_finite() {
    let mut model = ThermalModel::<VectorField>::new(1);
    model.apply_parameters(&[CRASH_U_VALUE, CRASH_HEATING, CRASH_COOLING]);
    model.set_loads(&[5.0]);

    let energy = model.step_physics(0, CRASH_OUTDOOR, CRASH_DT);
    assert!(
        energy.is_finite(),
        "issue #4500: step energy must stay finite, got {energy}"
    );
    assert!(
        energy < 0.0,
        "issue #4500: this step is net cooling; the signed-net contract \
         expects a negative return, got {energy}"
    );

    for (z, &temp) in model.get_temperatures().iter().enumerate() {
        assert!(
            temp.is_finite(),
            "issue #4500: zone {z} temperature non-finite after the crash step: {temp}"
        );
    }
}

/// The non-negativity contract holds where it actually lives: the per-zone
/// heating/cooling accumulators (the same quantities the fuzz target now
/// asserts on, issue #4500).
#[test]
fn test_issue_4500_accumulators_stay_non_negative() {
    let mut model = ThermalModel::<VectorField>::new(1);
    model.apply_parameters(&[CRASH_U_VALUE, CRASH_HEATING, CRASH_COOLING]);
    model.set_loads(&[5.0]);

    let _ = model.step_physics(0, CRASH_OUTDOOR, CRASH_DT);
    for (z, (&h, &c)) in model
        .get_zone_heating_energy_kwh()
        .iter()
        .zip(model.get_zone_cooling_energy_kwh().iter())
        .enumerate()
    {
        assert!(
            h.is_finite() && h >= 0.0,
            "issue #4500: zone {z} heating accumulator {h} must be finite and >= 0"
        );
        assert!(
            c.is_finite() && c >= 0.0,
            "issue #4500: zone {z} cooling accumulator {c} must be finite and >= 0"
        );
    }
}

/// Second crash class from the same night (artifact
/// `crash-398df7346f688942931c78f8b4e4eeaaf5188a39`, 8 zones, all-zero
/// parameters, outdoor bits `427786111` ≈ 2.1e-324 °C, dt bits
/// `9187201950439538559` ≈ 1.38e254 s): the ideal-loads demand itself
/// overflows to ±inf, so the step energy went to −inf. The engine guard
/// (issue #4500) rejects the step at the input envelope: the legacy
/// `step_physics` wrapper degrades to 0.0 kWh and the state is untouched.
#[test]
fn test_issue_4500_overflown_demand_rejected_state_untouched() {
    let mut model = ThermalModel::<VectorField>::new(8);
    model.apply_parameters(&[0.0, 0.0, 0.0]);
    model.set_loads(&vec![5.0_f64; 8]);

    let energy = model.step_physics(
        0,
        f64::from_bits(427786111),
        f64::from_bits(9187201950439538559),
    );
    assert_eq!(
        energy, 0.0,
        "issue #4500: rejected step must degrade to 0.0 kWh"
    );
    for (z, &temp) in model.get_temperatures().iter().enumerate() {
        assert!(
            temp.is_finite(),
            "issue #4500: zone {z} temperature non-finite after the rejected step: {temp}"
        );
    }
    let heating: f64 = model.get_zone_heating_energy_kwh().iter().sum();
    let cooling: f64 = model.get_zone_cooling_energy_kwh().iter().sum();
    assert!(
        heating == 0.0 && cooling == 0.0,
        "issue #4500: rejected step must not touch the energy accumulators          (heating {heating}, cooling {cooling})"
    );
}
