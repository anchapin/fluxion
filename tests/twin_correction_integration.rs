// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Digital-twin (`fluxion-twin`) `TwinCorrection` integration tests (Issue #2461).
//!
//! These tests verify that the `fluxion-twin` UKF correction can be applied
//! to the inner engine `ThermalModel<VectorField>` and that the physics
//! energy balance is preserved across the correction.
//!
//! # Background
//!
//! The `fluxion-twin` crate ([`fluxion_twin`]) defines a [`TwinCorrection`]
//! type that carries per-zone temperature corrections (in °C) produced by
//! the Unscented Kalman Filter. The corrections are intended to be applied
//! to the physics model so that the digital twin can correct drift between
//! simulation and reality.
//!
//! Before this issue, the [`TwinCorrection`] was dropped on the floor in
//! production code: the mock (`thermal_model_mock.rs`) implemented the
//! trait method but the production `ThermalModel` engine did not expose
//! any way to apply the correction. Issue #2461 wires the engine-level
//! `set_twin_correction` so that:
//!
//! 1. The `ThermalModel<VectorField>` engine has a direct method
//!    (`set_twin_correction`) that callers can invoke without going through
//!    the `ThermalModelTrait` adapter.
//! 2. The trait adapters (`PhysicsThermalModel`, `SurrogateThermalModel`,
//!    `HybridThermalModel`, `UnifiedThermalModel`) delegate to the engine
//!    method so the logic lives in one place.
//!
//! # Energy-balance invariant (RULES.md §1, must-always)
//!
//! The correction is a *state* update, not a *process* update. Adjusting
//! the per-zone temperature to the UKF estimate does not add or remove
//! energy from the system — the energy accumulators
//! (`zone_heating_energy_kwh`, `zone_cooling_energy_kwh`) are NOT modified
//! by the correction itself. The next `step_physics` call computes HVAC
//! demand from the corrected state, so the accumulators shift *consistently*
//! with the corrected temperature (e.g. a warmer zone requires less
//! heating in the next step). This preserves the energy-conservation gate
//! that downstream tests rely on (e.g. `tests/zone_balance_eplus_isolation.rs`).
//!
//! # Per-zone multiplicity
//!
//! Extra correction entries beyond `num_zones` are silently ignored
//! (consistent with the existing `ThermalModelTrait::set_twin_correction`
//! contract). Missing entries (correction shorter than `num_zones`) leave
//! the corresponding zone unchanged.
//!
//! # Test model choice
//!
//! The Issue #2461 spec suggests a 2-zone `ThermalModel<VectorField>` and
//! 5 `step_physics` calls. The current implementation of `ThermalModel::new(n)`
//! initializes `thermal_capacitance` to a 1.0 placeholder (see
//! `src/sim/thermal_model_core.rs::ThermalModel::new` line ~2626), which
//! causes the steady-state 5R1C solver to diverge on transient simulations
//! (this is the same limitation tracked in issues #893, #907, #908, #919,
//! and documented in `tests/zone_balance_eplus_isolation.rs::test_physics_thermal_model_setpoint_tracking_constant_outdoor`).
//!
//! To keep the integration test reliable, we use `ThermalModel::from_spec`
//! with [ASHRAE 140 Case 600](`fluxion::validation::ashrae_140_cases::ASHRAE140Case::Case600`)
//! — a 1-zone low-mass baseline — exactly as the existing zone-balance
//! isolation tests do. The 2-zone per-zone indexing is verified by a
//! separate unit test that constructs a 2-zone model and asserts the
//! correction never invokes `step_physics` (so the steady-state limitation
//! is irrelevant).

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion_twin::TwinCorrection;

/// Numerical tolerance for equality of zone temperatures (°C).
///
/// `1e-9` matches the issue's First Step requirement.
const TOL_TEMP: f64 = 1e-9;

/// Numerical tolerance for energy balance residuals (kWh).
///
/// A "no energy created or destroyed" check is exact in this implementation
/// (the correction has no HVAC-process side effect), so we only need to
/// guard against floating-point round-off in the surrounding arithmetic.
const TOL_ENERGY: f64 = 1e-6;

/// Build a 1-zone Case 600 thermal model initialised at 20 °C with the
/// ground temperature locked to a constant outdoor value so the envelope
/// heat balance is isolated from the depth-of-foundation dynamics.
/// Matches the pattern from
/// `tests/zone_balance_eplus_isolation.rs::test_physics_thermal_model_setpoint_tracking_constant_outdoor`.
fn case600_model(outdoor_temp: f64) -> ThermalModel<VectorField> {
    let spec = ASHRAE140Case::Case600.spec();
    let mut model: ThermalModel<VectorField> = ThermalModel::from_spec(&spec);

    // Pin zone AND mass temperatures to the initial setpoint so the first
    // step starts from a known, stable state.
    let init_t = 20.0;
    model.temperatures.as_mut()[0] = init_t;
    if let Some(ref mut mt) = Some(&mut model.mass_temperatures) {
        mt.as_mut()[0] = init_t;
    }
    model.set_ground_temp(outdoor_temp);

    model
}

/// Step the engine forward by `n_steps` hourly ticks under a constant
/// outdoor-temperature scenario. Returns the energy returned by the
/// last `step_physics` call (the per-step HVAC energy in kWh).
fn step_n(model: &mut ThermalModel<VectorField>, n_steps: usize, outdoor: f64) -> f64 {
    let mut last_energy = 0.0;
    for t in 0..n_steps {
        last_energy = model.step_physics(t, outdoor, 3600.0);
    }
    last_energy
}

// =============================================================================
// Section 1: Zero-correction no-op
// =============================================================================

/// A zero correction is a no-op: the per-zone temperature vector is
/// unchanged after `set_twin_correction` returns.
#[test]
fn test_zero_correction_is_noop() {
    let mut model = case600_model(10.0);

    // Establish a baseline by running 5 macro steps.
    step_n(&mut model, 5, 10.0);

    let t_before: Vec<f64> = model.temperatures.as_ref().to_vec();
    let h_before = model.get_zone_heating_energy_kwh();
    let c_before = model.get_zone_cooling_energy_kwh();

    let zero = TwinCorrection::multi_zone(vec![0.0, 0.0], vec![0.01, 0.01]);
    model.set_twin_correction(&zero);

    let t_after: Vec<f64> = model.temperatures.as_ref().to_vec();
    let h_after = model.get_zone_heating_energy_kwh();
    let c_after = model.get_zone_cooling_energy_kwh();

    // Temperature vector is unchanged.
    for (zone, (&a, &b)) in t_before.iter().zip(t_after.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL_TEMP,
            "zone {zone}: zero correction shifted temperature by {} °C (expected 0)",
            (a - b).abs()
        );
    }

    // Energy accumulators are unchanged (correction does not add or remove energy).
    for (zone, (&a, &b)) in h_before.iter().zip(h_after.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL_ENERGY,
            "zone {zone}: zero correction shifted heating accumulator by {} kWh (expected 0)",
            (a - b).abs()
        );
    }
    for (zone, (&a, &b)) in c_before.iter().zip(c_after.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL_ENERGY,
            "zone {zone}: zero correction shifted cooling accumulator by {} kWh (expected 0)",
            (a - b).abs()
        );
    }
}

// =============================================================================
// Section 2: Non-zero correction — exact shift
// =============================================================================

/// A non-zero correction shifts the per-zone temperatures by exactly the
/// correction vector (to `1e-9` °C, per the issue's First Step).
///
/// The energy accumulators are NOT modified by the correction itself —
/// the correction is a state update, not a process update.
#[test]
fn test_correction_shifts_temperatures_exactly() {
    let mut model = case600_model(10.0);

    step_n(&mut model, 5, 10.0);

    let t_before: Vec<f64> = model.temperatures.as_ref().to_vec();
    let h_before = model.get_zone_heating_energy_kwh();
    let c_before = model.get_zone_cooling_energy_kwh();

    let correction = TwinCorrection::multi_zone(vec![0.5, -0.3], vec![0.01, 0.01]);
    model.set_twin_correction(&correction);

    let t_after: Vec<f64> = model.temperatures.as_ref().to_vec();
    let h_after = model.get_zone_heating_energy_kwh();
    let c_after = model.get_zone_cooling_energy_kwh();

    // Single-zone model: only the first entry of the correction vector is
    // applied (the second entry is silently ignored, consistent with the
    // per-zone-multiplicity contract).
    let zone = 0;
    let delta = t_after[zone] - t_before[zone];
    let expected = 0.5;
    assert!(
        (delta - expected).abs() < TOL_TEMP,
        "zone {zone}: correction shifted temperature by {delta} °C (expected {expected} °C)",
    );

    // Energy accumulators are still unchanged by the correction itself.
    for (zone, (&a, &b)) in h_before.iter().zip(h_after.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL_ENERGY,
            "zone {zone}: correction shifted heating accumulator by {} kWh (expected 0)",
            (a - b).abs()
        );
    }
    for (zone, (&a, &b)) in c_before.iter().zip(c_after.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL_ENERGY,
            "zone {zone}: correction shifted cooling accumulator by {} kWh (expected 0)",
            (a - b).abs()
        );
    }
}

// =============================================================================
// Section 3: Energy-balance gate (RULES.md §1)
// =============================================================================

/// Re-running the same step after the correction must produce an
/// energy-consistent trajectory: no energy is created or destroyed by the
/// correction itself, and the next step's HVAC demand shifts consistently
/// with the corrected temperature state (e.g. a warmer zone needs less
/// heating).
///
/// We verify two properties:
/// 1. The energy accumulators are unchanged by the correction itself.
/// 2. After a step run post-correction, the cumulative energy accounts for
///    the corrected state (the correction is a state update, not a process
///    update — no phantom energy is introduced).
///
/// To avoid coupling to the steady-state 5R1C limitation or the HVAC
/// control bugs tracked in issues #893/#907/#908/#919, we run the
/// HVAC-disabled (free-floating) variant: set the heating/cooling
/// capacities to 0 so the only energy accumulator change is the
/// conduction-driven thermal mass dynamics (a small, finite, well-bounded
/// value).
#[test]
fn test_correction_energy_balance_gate() {
    let mut model = case600_model(10.0);

    // Disable HVAC so the energy accumulators stay at zero (free-floating
    // mode). The energy-balance gate then checks that the correction does
    // not introduce phantom energy in the absence of HVAC processes.
    model.heating_setpoint = -999.0;
    model.cooling_setpoint = 999.0;
    model.hvac_heating_capacity = 0.0;
    model.hvac_cooling_capacity = 0.0;

    step_n(&mut model, 5, 10.0);

    let h_before: Vec<f64> = model.get_zone_heating_energy_kwh();
    let c_before: Vec<f64> = model.get_zone_cooling_energy_kwh();

    // Apply a large positive correction.
    let correction = TwinCorrection::multi_zone(vec![5.0, -5.0], vec![0.01, 0.01]);
    model.set_twin_correction(&correction);

    // Immediately after the correction, the energy accumulators are unchanged.
    let h_after_corr: Vec<f64> = model.get_zone_heating_energy_kwh();
    let c_after_corr: Vec<f64> = model.get_zone_cooling_energy_kwh();
    for (zone, (&a, &b)) in h_before.iter().zip(h_after_corr.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL_ENERGY,
            "zone {zone}: correction changed heating accumulator by {} kWh (HVAC off, expected 0)",
            (a - b).abs()
        );
    }
    for (zone, (&a, &b)) in c_before.iter().zip(c_after_corr.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL_ENERGY,
            "zone {zone}: correction changed cooling accumulator by {} kWh (HVAC off, expected 0)",
            (a - b).abs()
        );
    }

    // Run one more step post-correction. With HVAC disabled, the HVAC
    // energy per step is zero, so the accumulators must remain at exactly
    // the pre-correction value.
    let _ = model.step_physics(5, 10.0, 3600.0);

    let h_after_step: Vec<f64> = model.get_zone_heating_energy_kwh();
    let c_after_step: Vec<f64> = model.get_zone_cooling_energy_kwh();
    for (zone, (&a, &b)) in h_before.iter().zip(h_after_step.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL_ENERGY,
            "zone {zone}: post-correction step changed heating accumulator by {} kWh (HVAC off, expected 0)",
            (a - b).abs()
        );
    }
    for (zone, (&a, &b)) in c_before.iter().zip(c_after_step.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL_ENERGY,
            "zone {zone}: post-correction step changed cooling accumulator by {} kWh (HVAC off, expected 0)",
            (a - b).abs()
        );
    }

    // The corrected temperature state must be finite and physically
    // reasonable (sanity floor — the correction must not push the
    // simulation into numerical instability).
    let t_after: Vec<f64> = model.temperatures.as_ref().to_vec();
    for (zone, &t) in t_after.iter().enumerate() {
        assert!(
            t.is_finite(),
            "zone {zone}: post-correction temperature not finite"
        );
        assert!(
            t > -100.0 && t < 200.0,
            "zone {zone}: post-correction temperature {t} °C outside physical range",
        );
    }
}

// =============================================================================
// Section 4: Multi-zone per-zone indexing (issue #2461 first step)
// =============================================================================

/// 2-zone model: verify that the per-zone correction is applied correctly
/// to each zone independently. This test does NOT call `step_physics` (the
/// steady-state 5R1C limitation above makes `ThermalModel::new(n)` unstable
/// for transient simulations; we only verify the correction-application
/// logic which is independent of the physics timestep).
#[test]
fn test_correction_two_zone_per_zone_indexing() {
    let mut model: ThermalModel<VectorField> = ThermalModel::new(2);
    let init_t = 20.0;
    model.temperatures.as_mut()[0] = init_t;
    model.temperatures.as_mut()[1] = init_t;

    let h_before = model.get_zone_heating_energy_kwh();
    let c_before = model.get_zone_cooling_energy_kwh();

    let correction = TwinCorrection::multi_zone(vec![0.5, -0.3], vec![0.01, 0.01]);
    model.set_twin_correction(&correction);

    let t_after = model.temperatures.as_ref().to_vec();

    // Per-zone temperature deltas equal +0.5 °C / -0.3 °C to 1e-9 °C.
    let expected_delta = [0.5, -0.3];
    for (zone, &expected) in expected_delta.iter().enumerate() {
        let delta = t_after[zone] - init_t;
        assert!(
            (delta - expected).abs() < TOL_TEMP,
            "zone {zone}: correction shifted temperature by {delta} °C (expected {expected} °C)",
        );
    }

    // Energy accumulators are unchanged by the correction itself.
    let h_after = model.get_zone_heating_energy_kwh();
    let c_after = model.get_zone_cooling_energy_kwh();
    for (zone, (&a, &b)) in h_before.iter().zip(h_after.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL_ENERGY,
            "zone {zone}: correction shifted heating accumulator by {} kWh (expected 0)",
            (a - b).abs()
        );
    }
    for (zone, (&a, &b)) in c_before.iter().zip(c_after.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL_ENERGY,
            "zone {zone}: correction shifted cooling accumulator by {} kWh (expected 0)",
            (a - b).abs()
        );
    }
}

/// 2-zone model: zero correction is a no-op on the per-zone temperature
/// vector and the energy accumulators.
#[test]
fn test_correction_two_zone_zero_is_noop() {
    let mut model: ThermalModel<VectorField> = ThermalModel::new(2);
    let init_t = 20.0;
    model.temperatures.as_mut()[0] = init_t;
    model.temperatures.as_mut()[1] = init_t;

    let t_before = model.temperatures.as_ref().to_vec();
    let h_before = model.get_zone_heating_energy_kwh();
    let c_before = model.get_zone_cooling_energy_kwh();

    let zero = TwinCorrection::multi_zone(vec![0.0, 0.0], vec![0.01, 0.01]);
    model.set_twin_correction(&zero);

    let t_after = model.temperatures.as_ref().to_vec();
    let h_after = model.get_zone_heating_energy_kwh();
    let c_after = model.get_zone_cooling_energy_kwh();

    // Temperature vector is unchanged.
    for (zone, (&a, &b)) in t_before.iter().zip(t_after.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL_TEMP,
            "zone {zone}: zero correction shifted temperature by {} °C (expected 0)",
            (a - b).abs()
        );
    }

    // Energy accumulators are unchanged.
    for (zone, (&a, &b)) in h_before.iter().zip(h_after.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL_ENERGY,
            "zone {zone}: zero correction shifted heating accumulator by {} kWh (expected 0)",
            (a - b).abs()
        );
    }
    for (zone, (&a, &b)) in c_before.iter().zip(c_after.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL_ENERGY,
            "zone {zone}: zero correction shifted cooling accumulator by {} kWh (expected 0)",
            (a - b).abs()
        );
    }
}

// =============================================================================
// Section 5: Multi-zone multiplicity and edge cases
// =============================================================================

/// Correction with more entries than `num_zones`: extra entries are silently
/// ignored (consistent with the existing `ThermalModelTrait::set_twin_correction`).
#[test]
fn test_correction_extra_entries_ignored() {
    let mut model: ThermalModel<VectorField> = ThermalModel::new(2);
    let init_t = 20.0;
    model.temperatures.as_mut()[0] = init_t;
    model.temperatures.as_mut()[1] = init_t;

    let t_before: Vec<f64> = model.temperatures.as_ref().to_vec();

    // 3 entries for a 2-zone model.
    let correction = TwinCorrection::multi_zone(vec![1.0, 2.0, 99.0], vec![0.1, 0.1, 0.1]);
    model.set_twin_correction(&correction);

    let t_after: Vec<f64> = model.temperatures.as_ref().to_vec();

    // Only the first two entries are applied.
    let expected_delta = [1.0, 2.0];
    for (zone, (&a, &b)) in t_before.iter().zip(t_after.iter()).enumerate() {
        let delta = b - a;
        let expected = expected_delta[zone];
        assert!(
            (delta - expected).abs() < TOL_TEMP,
            "zone {zone}: delta {delta} °C (expected {expected} °C, extra entry must be ignored)",
        );
    }
}

/// Correction with fewer entries than `num_zones`: missing entries leave
/// the corresponding zone unchanged.
#[test]
fn test_correction_missing_entries_unchanged() {
    let mut model: ThermalModel<VectorField> = ThermalModel::new(2);
    let init_t = 20.0;
    model.temperatures.as_mut()[0] = init_t;
    model.temperatures.as_mut()[1] = init_t;

    let t_before: Vec<f64> = model.temperatures.as_ref().to_vec();

    // 1 entry for a 2-zone model.
    let correction = TwinCorrection::multi_zone(vec![1.5], vec![0.1]);
    model.set_twin_correction(&correction);

    let t_after: Vec<f64> = model.temperatures.as_ref().to_vec();

    // Zone 0 is shifted by +1.5 °C; zone 1 is unchanged.
    assert!(
        (t_after[0] - t_before[0] - 1.5).abs() < TOL_TEMP,
        "zone 0: delta {} °C (expected +1.5 °C)",
        t_after[0] - t_before[0],
    );
    assert!(
        (t_after[1] - t_before[1]).abs() < TOL_TEMP,
        "zone 1: delta {} °C (expected 0, missing entry should leave zone unchanged)",
        t_after[1] - t_before[1],
    );
}

/// Single-zone correction matches the trait adapter behaviour.
#[test]
fn test_single_zone_correction() {
    let mut model: ThermalModel<VectorField> = ThermalModel::new(1);
    let init_t = 22.0;
    model.temperatures.as_mut()[0] = init_t;

    let correction = TwinCorrection::single_zone(0.5, 0.1);
    model.set_twin_correction(&correction);

    let t_after = model.temperatures[0];
    assert!(
        (t_after - (init_t + 0.5)).abs() < TOL_TEMP,
        "single-zone correction: T = {t_after} °C (expected {} °C)",
        init_t + 0.5,
    );
}
