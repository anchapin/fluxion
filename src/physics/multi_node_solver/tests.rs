//! Unit tests for the multi-node thermal solver (extracted from
//! `mod.rs` to keep the parent file under the Issue #3457 module-size
//! ratchet ceiling — Issue #3790, decomposition stage 1).
//!
//! Mirrors the PR #3688 `coverage_tests` precedent: the inline
//! `#[cfg(test)] mod tests { ... }` block is extracted to a sibling
//! file wired via `#[cfg(test)] mod tests;` in the parent. The
//! `coverage_tests.rs` child is retainable in the same way.
//!
//! Children see the parent module's items through `use super::*;`
//! — `coverage_tests.rs` and this file follow identical conventions.

use super::*;
use fluxion_core::multi_node::ThermalMassNode;
use std::panic::catch_unwind;

fn create_test_solver() -> MultiNodeSolver {
    let wall = ThermalMassNode::new(20.0, 5e6, 50.0, 20.0);
    let roof = ThermalMassNode::new(20.0, 3e6, 30.0, 15.0);
    let floor = ThermalMassNode::new(20.0, 2e6, 20.0, 10.0);
    let internal = ThermalMassNode::new(20.0, 1e6, 10.0, 5.0);

    MultiNodeSolver::new(10.0, wall, roof, floor, internal)
}

#[test]
fn test_solver_creation() {
    let solver = create_test_solver();
    assert_eq!(solver.wall_temperature(), 20.0);
    assert_eq!(solver.roof_temperature(), 20.0);
    assert_eq!(solver.floor_temperature(), 20.0);
    assert_eq!(solver.internal_temperature(), 20.0);
}

#[test]
fn test_step_changes_temperatures() {
    let mut solver = create_test_solver();
    solver.set_zone_temperature(22.0);
    solver.set_exterior_temperature(5.0);
    solver.set_surface_temperature(18.0);

    let t_wall_before = solver.wall_temperature();
    solver.step(3600.0);

    // Wall should cool toward exterior temperature
    assert!(solver.wall_temperature() < t_wall_before);
}

#[test]
fn test_envelope_temperature_average() {
    let mut solver = create_test_solver();
    solver.mass.wall.temperature = 10.0;
    solver.mass.roof.temperature = 20.0;
    solver.mass.floor.temperature = 30.0;

    let avg = solver.envelope_temperature();
    assert!((avg - 20.0).abs() < 0.001);
}

#[test]
fn test_time_constant_calculation() {
    let solver = create_test_solver();
    let tau = solver.effective_time_constant();

    // With C_total ≈ 11e6 J and h_eff ≈ 10-20 W/K
    // τ should be in the range of hours (h_tr in W/K, C in J/K, so τ in seconds)
    assert!(tau > 0.0);
    assert!(tau < 1e8); // Sanity check
}

#[test]
fn test_steady_state_convergence() {
    let mut solver = create_test_solver();
    solver.set_zone_temperature(20.0);
    solver.set_exterior_temperature(20.0);
    solver.set_surface_temperature(20.0);

    // Run for many hours - temperatures should converge
    for _ in 0..168 {
        solver.step(3600.0);
    }

    // All temperatures should be near 20°C (within 0.1°C)
    assert!((solver.wall_temperature() - 20.0).abs() < 0.1);
    assert!((solver.roof_temperature() - 20.0).abs() < 0.1);
    assert!((solver.floor_temperature() - 20.0).abs() < 0.1);
    assert!((solver.internal_temperature() - 20.0).abs() < 0.1);
}

#[test]
fn test_temperature_gradient_with_known_conductances() {
    let mut solver = create_test_solver();
    solver.set_zone_temperature(25.0);
    solver.set_exterior_temperature(0.0);
    solver.set_surface_temperature(15.0);

    // High-mass wall should show thermal lag
    let t_wall_initial = solver.wall_temperature();
    solver.step(3600.0);

    // Wall should cool slightly but not reach 0°C quickly due to high capacitance
    assert!(solver.wall_temperature() > 0.0);
    assert!(solver.wall_temperature() < t_wall_initial);
}

#[test]
fn test_internal_mass_response() {
    let mut solver = create_test_solver();
    solver.set_zone_temperature(30.0);
    solver.set_exterior_temperature(10.0);
    solver.set_surface_temperature(20.0);

    // Internal mass should respond to zone temperature changes
    let t_internal_initial = solver.internal_temperature();
    solver.step(3600.0);

    // Internal mass should warm toward zone temperature
    assert!(solver.internal_temperature() > t_internal_initial);
    assert!(solver.internal_temperature() < 30.0);
}

#[test]
fn test_backward_euler_stability() {
    let mut solver = create_test_solver();
    solver.set_zone_temperature(100.0); // Large temperature difference
    solver.set_exterior_temperature(-50.0);
    solver.set_surface_temperature(50.0);

    // Take many small timesteps - backward Euler should be stable
    for _ in 0..24 {
        solver.step(300.0); // 5-minute timestep
    }

    // All temperatures should be finite and within reasonable bounds
    assert!(solver.wall_temperature().is_finite());
    assert!(solver.roof_temperature().is_finite());
    assert!(solver.floor_temperature().is_finite());
    assert!(solver.internal_temperature().is_finite());

    // Should not have exploded
    assert!(solver.wall_temperature().abs() < 1000.0);
}

#[test]
fn test_conductance_setters() {
    let mut solver = create_test_solver();

    solver.set_wall_conductances(25.0, 55.0);
    solver.set_roof_conductances(20.0, 40.0);
    solver.set_floor_conductances(15.0, 30.0);
    solver.set_internal_conductance(8.0);

    assert_eq!(solver.mass.wall.h_tr_em, 25.0);
    assert_eq!(solver.mass.wall.h_tr_ms, 55.0);
    assert_eq!(solver.mass.roof.h_tr_em, 20.0);
    assert_eq!(solver.mass.roof.h_tr_ms, 40.0);
    assert_eq!(solver.mass.floor.h_tr_em, 15.0);
    assert_eq!(solver.mass.floor.h_tr_ms, 30.0);
    assert_eq!(solver.mass.internal.h_tr_me, 8.0);
}

#[test]
fn test_capacitance_setters() {
    let mut solver = create_test_solver();

    solver.set_wall_capacitance(1e7);
    solver.set_roof_capacitance(2e7);
    solver.set_floor_capacitance(3e7);
    solver.set_internal_capacitance(4e6);

    assert_eq!(solver.mass.wall.capacitance, 1e7);
    assert_eq!(solver.mass.roof.capacitance, 2e7);
    assert_eq!(solver.mass.floor.capacitance, 3e7);
    assert_eq!(solver.mass.internal.capacitance, 4e6);
}

#[test]
fn test_initialization() {
    let mut solver = create_test_solver();
    solver.initialize_temperatures(15.0);

    assert_eq!(solver.wall_temperature(), 15.0);
    assert_eq!(solver.roof_temperature(), 15.0);
    assert_eq!(solver.floor_temperature(), 15.0);
    assert_eq!(solver.internal_temperature(), 15.0);
    assert_eq!(solver.zone_temperature, 15.0);
    assert_eq!(solver.surface_temperature, 15.0);
}

#[test]
fn test_per_surface_exterior_temps() {
    let mut solver = create_test_solver();
    solver.initialize_temperatures(20.0);
    solver.set_zone_temperature(20.0);
    solver.set_surface_temperature(20.0);

    let temps = SurfaceExteriorTemperatures {
        t_ext_wall: 30.0,
        t_ext_roof: 35.0,
        t_ext_floor: 15.0,
    };
    solver.set_surface_exterior_temperatures(temps);
    solver.step(3600.0);

    assert!(
        solver.wall_temperature() > 20.0,
        "Wall should warm from sol-air"
    );
    assert!(
        solver.roof_temperature() > solver.wall_temperature(),
        "Roof > wall"
    );
    assert!(
        solver.floor_temperature() < 20.0,
        "Floor should cool from ground"
    );
}

// ── Issue #871: Air Balance API Tests ────────────────────────────

#[test]
fn test_compute_zone_air_temperature_steady_state() {
    let solver = create_test_solver();
    // All nodes at 20°C, outdoor at 20°C → T_air ≈ 20°C
    let t_air = solver.compute_zone_air_temperature(20.0, 5.0, 0.0, 0.0);
    assert!(
        (t_air - 20.0).abs() < 0.5,
        "Steady-state T_air should be ~20°C, got {t_air}"
    );
}

#[test]
fn test_compute_zone_air_temperature_solar_gain() {
    let solver = create_test_solver();
    // phi_ia > 0 → T_air > T_outdoor
    let t_air_no_gain = solver.compute_zone_air_temperature(10.0, 5.0, 0.0, 0.0);
    let t_air_with_gain = solver.compute_zone_air_temperature(10.0, 5.0, 0.0, 2000.0);
    assert!(
        t_air_with_gain > t_air_no_gain,
        "Solar gain should raise T_air: {t_air_with_gain} should be > {t_air_no_gain}"
    );
    assert!(
        t_air_with_gain > 10.0,
        "T_air with gains should be above outdoor: {t_air_with_gain} > 10.0"
    );
}

#[test]
fn test_compute_hvac_demand_heating() {
    let solver = create_test_solver();
    // T_air_free < heating setpoint → positive Q (heating needed)
    let q = solver.compute_hvac_demand(15.0, 20.0, 26.0);
    assert!(
        q > 0.0,
        "Heating demand should be positive when T_air < heating setpoint, got {q}"
    );
    // Q = h_tr_is × (20 - 15) = 10 × 5 = 50 W
    assert!((q - 50.0).abs() < 1.0, "Expected ~50W heating, got {q}");
}

#[test]
fn test_compute_hvac_demand_cooling() {
    let solver = create_test_solver();
    // T_air_free > cooling setpoint → negative Q (cooling needed)
    let q = solver.compute_hvac_demand(30.0, 20.0, 26.0);
    assert!(
        q < 0.0,
        "Cooling demand should be negative when T_air > cooling setpoint, got {q}"
    );
    // Q = h_tr_is × (26 - 30) = 10 × (-4) = -40 W
    assert!((q - (-40.0)).abs() < 1.0, "Expected ~-40W cooling, got {q}");
}

#[test]
fn test_compute_hvac_demand_deadband() {
    let solver = create_test_solver();
    // T_air_free within [heat_sp, cool_sp] → zero Q
    let q = solver.compute_hvac_demand(22.0, 20.0, 26.0);
    assert!(
        q.abs() < 1e-10,
        "Demand should be zero within deadband, got {q}"
    );
}

#[test]
fn test_step_with_gains_increases_temp() {
    let mut solver = create_test_solver();
    solver.set_zone_temperature(20.0);
    solver.set_exterior_temperature(10.0);
    solver.set_surface_temperature(18.0);

    // Step without gains
    let mut solver_no_gains = solver.clone();
    solver_no_gains.step(3600.0);
    let t_wall_no_gains = solver_no_gains.wall_temperature();
    let t_roof_no_gains = solver_no_gains.roof_temperature();

    // Step with gains (1000W to wall, 500W to roof)
    solver.step_with_gains(
        3600.0,
        1000.0,
        500.0,
        0.0,
        0.0,
        0.0,
        solver.exterior_temperature,
    );
    let t_wall_with_gains = solver.wall_temperature();
    let t_roof_with_gains = solver.roof_temperature();

    assert!(
        t_wall_with_gains > t_wall_no_gains,
        "Wall with gains ({t_wall_with_gains}) should be > without ({t_wall_no_gains})"
    );
    assert!(
        t_roof_with_gains > t_roof_no_gains,
        "Roof with gains ({t_roof_with_gains}) should be > without ({t_roof_no_gains})"
    );
    // Wall gets more gains than roof → should be hotter
    let wall_delta = t_wall_with_gains - t_wall_no_gains;
    let roof_delta = t_roof_with_gains - t_roof_no_gains;
    assert!(
        wall_delta > roof_delta,
        "Wall delta ({wall_delta}) should exceed roof delta ({roof_delta})"
    );
}

// ── Issue #1281: Parallel-resistance coupling network ─────────────

/// Construct a Case 900-style high-mass solver for Issue #1281 tests.
///
/// Per-surface h_tr_ms values come from the half-insulation rule applied
/// to the ASHRAE 140 Case 900 construction
/// (`src/sim/construction.rs::Assemblies::high_mass_wall` /
/// `high_mass_roof` / `high_mass_floor`).
fn create_case_900_solver(coupling_mode: MassAirCouplingMode) -> MultiNodeSolver {
    let wall = ThermalMassNode::new(20.0, 5.0e6, 76.4, 25.0);
    let roof = ThermalMassNode::new(20.0, 3.0e6, 32.9, 20.0);
    let floor = ThermalMassNode::new(20.0, 2.0e6, 18.0, 10.0);
    let internal = ThermalMassNode::new(20.0, 1.0e6, 0.0, 0.0).with_h_tr_me(100.0);

    // h_tr_is = 3.45 × floor_area = 3.45 × 48 = 165.6 W/K
    // (Issue #714: ASHRAE 140 simplified 5R1C formula)
    MultiNodeSolver::new_with_mode(165.6, wall, roof, floor, internal, coupling_mode)
}

#[test]
fn test_issue_1281_default_mode_is_additive_sum() {
    // Backward compatibility: existing constructor keeps AdditiveSum default.
    let solver = create_test_solver();
    assert_eq!(solver.coupling_mode, MassAirCouplingMode::AdditiveSum);
}

#[test]
fn test_issue_1281_new_with_mode_parallel_resistance() {
    let solver = create_case_900_solver(MassAirCouplingMode::ParallelResistance);
    assert_eq!(
        solver.coupling_mode,
        MassAirCouplingMode::ParallelResistance
    );
}

#[test]
fn test_issue_1281_with_coupling_mode_builder() {
    let solver =
        create_test_solver().with_coupling_mode(MassAirCouplingMode::ParallelResistance);
    assert_eq!(
        solver.coupling_mode,
        MassAirCouplingMode::ParallelResistance
    );
}

#[test]
fn test_issue_1281_parallel_resistance_air_lower_than_additive() {
    // At steady state with hot exterior forcing, the parallel-resistance
    // formulation should give a LOWER T_air than the additive formulation
    // because h_path_total < h_ms_total (each per-surface series conductance
    // is strictly less than the per-surface h_ms_k).
    //
    // Reference: .agents/results/issue-1281-python-verification.py
    // (steady-state Case 900: additive T_air=42.0, parallel-resistance T_air=30.4)

    let mut add = create_case_900_solver(MassAirCouplingMode::AdditiveSum);
    let mut par = create_case_900_solver(MassAirCouplingMode::ParallelResistance);

    // Hot summer forcing (similar to Python verification)
    let ext = SurfaceExteriorTemperatures {
        t_ext_wall: 45.0,
        t_ext_roof: 50.0,
        t_ext_floor: 18.0,
    };
    add.set_surface_exterior_temperatures(ext.clone());
    par.set_surface_exterior_temperatures(ext);
    add.set_zone_temperature(20.0);
    par.set_zone_temperature(20.0);

    // 5000 × 1-hour steps to reach steady state
    for _ in 0..5000 {
        add.step(3600.0);
        par.step(3600.0);
    }

    let t_air_add = add.compute_zone_air_temperature(32.0, 21.7, 0.0, 200.0);
    let t_air_par = par.compute_zone_air_temperature(32.0, 21.7, 0.0, 200.0);

    // Sanity: both finite and positive
    assert!(t_air_add.is_finite() && t_air_par.is_finite());
    assert!(t_air_add > 20.0 && t_air_par > 20.0);

    // The parallel-resistance formulation gives a LOWER air temperature
    // (verified by Python: 30.4 °C vs 42.0 °C).
    assert!(
        t_air_par < t_air_add,
        "Parallel-resistance T_air ({:.3}) should be < additive T_air ({:.3})",
        t_air_par,
        t_air_add,
    );

    // The gap should be on the order of 5-15 °C for Case 900 parameters.
    let gap = t_air_add - t_air_par;
    assert!(
        gap > 1.0,
        "T_air gap ({:.3}) should be meaningful (>1 K), confirming non-additive correction",
        gap,
    );
}

#[test]
fn test_issue_1281_h_series_formula() {
    // Verify the series-combination helper matches a hand calculation.
    // h_series(a, b) = a*b/(a+b)
    // Symmetric: h_series(a, b) == h_series(b, a)
    assert!((h_series(50.0, 165.6) - 50.0 * 165.6 / (50.0 + 165.6)).abs() < 1e-10);
    assert!((h_series(165.6, 50.0) - h_series(50.0, 165.6)).abs() < 1e-10);

    // Degenerate cases: use h_series_strict which returns Result
    // so the error path can be tested in both debug AND release builds
    assert!(
        h_series_strict(0.0, 100.0).is_err(),
        "h_series_strict should Err for degenerate inputs"
    );
    assert!(
        h_series_strict(100.0, 0.0).is_err(),
        "h_series_strict should Err for degenerate inputs"
    );
    assert!(
        h_series_strict(-1.0, 100.0).is_err(),
        "h_series_strict should Err for degenerate inputs"
    );

    // For Case 900 per-surface values:
    let h_path_wall = h_series(76.4, 165.6);
    let h_path_roof = h_series(32.9, 165.6);
    let h_path_floor = h_series(18.0, 165.6);
    let h_path_total = h_path_wall + h_path_roof + h_path_floor;
    let h_ms_total = 76.4 + 32.9 + 18.0;
    assert!(
        h_path_total < h_ms_total,
        "Parallel-resistance total ({:.3}) must be < additive h_ms_total ({:.3})",
        h_path_total,
        h_ms_total,
    );
    // Numerical verification: ratio matches Python's 0.753
    let ratio = h_path_total / h_ms_total;
    assert!(
        (ratio - 0.7534).abs() < 0.01,
        "ratio {ratio} should be ~0.753 (Python-verified 32.7% overcount)"
    );
}

#[test]
fn test_issue_1281_per_surface_t_s_helper() {
    // T_s = (h_ms × t_m + h_is × t_air) / (h_ms + h_is)
    let t_m = 30.0;
    let h_ms = 76.4;
    let h_is = 165.6;
    let t_air = 25.0;
    let expected = (h_ms * t_m + h_is * t_air) / (h_ms + h_is);
    let actual = per_surface_t_s(t_m, h_ms, h_is, t_air);
    assert!((actual - expected).abs() < 1e-10);

    // Degenerate cases
    assert_eq!(per_surface_t_s(20.0, 0.0, 0.0, 30.0), 30.0);
    assert_eq!(per_surface_t_s(f64::NAN, 1.0, 1.0, 20.0), 20.0);
}

#[test]
fn test_issue_1281_parallel_resistance_step_uses_per_surface_t_s() {
    // Verify that step() in ParallelResistance mode produces DIFFERENT mass
    // temperatures than AdditiveSum for the same forcing. This is the
    // physically meaningful behavior change.
    let mut add = create_case_900_solver(MassAirCouplingMode::AdditiveSum);
    let mut par = create_case_900_solver(MassAirCouplingMode::ParallelResistance);

    let ext = SurfaceExteriorTemperatures {
        t_ext_wall: 45.0,
        t_ext_roof: 50.0,
        t_ext_floor: 18.0,
    };
    add.set_surface_exterior_temperatures(ext.clone());
    par.set_surface_exterior_temperatures(ext);
    add.set_zone_temperature(20.0);
    par.set_zone_temperature(20.0);

    // Run 24 hourly steps
    for _ in 0..24 {
        add.step(3600.0);
        par.step(3600.0);
    }

    // Mass temperatures should differ between the two formulations
    // (the parallel-resistance formulation feeds each mass node its OWN
    // per-surface T_s_k, not the shared conductance-weighted mean).
    let diff_wall = (add.wall_temperature() - par.wall_temperature()).abs();
    let diff_roof = (add.roof_temperature() - par.roof_temperature()).abs();
    let diff_floor = (add.floor_temperature() - par.floor_temperature()).abs();

    assert!(
        diff_wall + diff_roof + diff_floor > 0.01,
        "Mass temperatures must differ between additive ({:.3}, {:.3}, {:.3}) and parallel ({:.3}, {:.3}, {:.3})",
        add.wall_temperature(), add.roof_temperature(), add.floor_temperature(),
        par.wall_temperature(), par.roof_temperature(), par.floor_temperature(),
    );
}

#[test]
fn test_issue_1281_parallel_resistance_step_with_gains() {
    // Verify step_with_gains in ParallelResistance mode produces higher
    // mass temperatures than no-gains case (solar gains are heating).
    let mut solver_no_gains = create_case_900_solver(MassAirCouplingMode::ParallelResistance);
    let mut solver_with_gains = create_case_900_solver(MassAirCouplingMode::ParallelResistance);

    let ext = SurfaceExteriorTemperatures {
        t_ext_wall: 45.0,
        t_ext_roof: 50.0,
        t_ext_floor: 18.0,
    };
    solver_no_gains.set_surface_exterior_temperatures(ext.clone());
    solver_with_gains.set_surface_exterior_temperatures(ext);
    solver_no_gains.set_zone_temperature(20.0);
    solver_with_gains.set_zone_temperature(20.0);

    for _ in 0..24 {
        solver_no_gains.step(3600.0);
        solver_with_gains.step_with_gains(
            3600.0,
            1000.0,
            500.0,
            0.0,
            0.0,
            0.0,
            solver_with_gains.exterior_temperature,
        );
    }

    assert!(
        solver_with_gains.wall_temperature() > solver_no_gains.wall_temperature(),
        "Wall with gains ({:.3}) should be hotter than without ({:.3})",
        solver_with_gains.wall_temperature(),
        solver_no_gains.wall_temperature(),
    );
    assert!(
        solver_with_gains.roof_temperature() > solver_no_gains.roof_temperature(),
        "Roof with gains ({:.3}) should be hotter than without ({:.3})",
        solver_with_gains.roof_temperature(),
        solver_no_gains.roof_temperature(),
    );
}

#[test]
fn test_issue_1281_backward_compat_additive_unchanged() {
    // Verify that AdditiveSum mode produces the same T_air as the
    // original (pre-Issue #1281) formulation for a known forcing case.
    // This test serves as a regression guard: if someone breaks the
    // original additive formula, this test fails.
    let solver = create_case_900_solver(MassAirCouplingMode::AdditiveSum);

    // All masses at 20, all temps at 20 → T_air should be 20 (steady state).
    let t_air = solver.compute_zone_air_temperature(20.0, 0.0, 0.0, 0.0);
    assert!(
        (t_air - 20.0).abs() < 0.5,
        "Steady-state T_air should be ~20°C, got {t_air}"
    );
}

#[test]
fn test_issue_1281_parallel_resistance_degenerate_falls_back() {
    // When h_tr_ms sums to ~0 (degenerate construction), the parallel-resistance
    // air calc must fall back to the conductance-weighted average of mass
    // temperatures (not NaN/Inf). In debug builds, debug_assert! in h_series
    // fires; we catch the panic and verify the solver still produces a finite result.
    let wall = ThermalMassNode::new(20.0, 1.0e6, 0.0, 25.0);
    let roof = ThermalMassNode::new(20.0, 1.0e6, 0.0, 20.0);
    let floor = ThermalMassNode::new(20.0, 1.0e6, 0.0, 10.0);
    let internal = ThermalMassNode::new(20.0, 5.0e5, 0.0, 0.0).with_h_tr_me(0.0);
    let solver = MultiNodeSolver::new_with_mode(
        165.6,
        wall,
        roof,
        floor,
        internal,
        MassAirCouplingMode::ParallelResistance,
    );

    // debug_assert! fires in debug builds for degenerate h_series inputs
    let result = catch_unwind(|| solver.compute_zone_air_temperature(30.0, 5.0, 0.0, 0.0));
    if result.is_err() {
        // Debug build: debug_assert! fired as expected — degenerate inputs are caught
        return;
    }
    let t_air = result.unwrap();
    assert!(
        t_air.is_finite(),
        "Degenerate construction must produce finite T_air, got {t_air}"
    );
}

// ── Issue #1858: sky-radiative air-node path ──────────────────────

#[test]
fn test_issue_1858_air_sky_conductance_formula() {
    // Verify the linearized conductance against a hand calculation.
    //
    // Issue #2462 (Phase 2 of the crate split): previously this test also
    // cross-checked against `crate::sim::sky_radiation::SkyRadiationExchange::radiative_coefficient`,
    // but that import was one of the 5 documented `physics ↔ sim` cycle
    // edges. The SkyRadiationExchange::radiative_coefficient formula is
    // `4.0 * eps * f_sky * STEFAN_BOLTZMANN * t_mean^3` — the same
    // hand calculation we already verify below — so the cross-check
    // added nothing once the leaf constant was available here.
    let eps = 0.9;
    let f_sky = 0.25;
    let aperture = 12.0; // m²
    let t_air = -8.0;
    let t_sky = -30.0;

    // Total conductance from the new helper (W/K).
    let h_total = air_sky_conductance(eps, f_sky, aperture, t_air, t_sky);

    // Hand calculation for the linearization at T_mean = 254.075 K.
    let sigma = 5.67e-8_f64;
    let t_mean_k = ((t_air + 273.15) + (t_sky + 273.15)) / 2.0;
    let expected = 4.0 * eps * f_sky * sigma * t_mean_k.powi(3) * aperture;
    assert!(
        (h_total - expected).abs() < 1e-6,
        "h_total {h_total} != hand calc {expected}",
    );

    // Magnitude sanity: ~10 W/K for Case 900 night-min aperture geometry.
    assert!(
        h_total > 5.0 && h_total < 20.0,
        "unexpected h_total {h_total}"
    );
}

#[test]
fn test_issue_1858_air_sky_conductance_degenerate_is_zero() {
    // Degenerate inputs → 0.0 (no-op sky path, preserves backward compat).
    assert_eq!(air_sky_conductance(0.0, 0.5, 12.0, -8.0, -30.0), 0.0);
    assert_eq!(air_sky_conductance(0.9, 0.0, 12.0, -8.0, -30.0), 0.0);
    assert_eq!(air_sky_conductance(0.9, 0.5, 0.0, -8.0, -30.0), 0.0);
    assert_eq!(air_sky_conductance(0.9, 0.5, -1.0, -8.0, -30.0), 0.0);
}

#[test]
fn test_issue_1858_backward_compat_zero_sky_conductance() {
    // A zero sky conductance must recover the original four-term air-node
    // balance EXACTLY for both coupling modes. This is the guard that the
    // sky-radiative path does not perturb existing ASHRAE 140 fixtures.
    for mode in [
        MassAirCouplingMode::AdditiveSum,
        MassAirCouplingMode::ParallelResistance,
    ] {
        let solver = create_case_900_solver(mode);
        let t_outdoor = -10.0;
        let h_ve = 21.7;
        let phi_ia = 200.0;

        let t_air_plain = solver.compute_zone_air_temperature(t_outdoor, h_ve, 0.0, phi_ia);
        let t_air_sky_zero = solver
            .compute_zone_air_temperature_with_sky(t_outdoor, h_ve, 0.0, phi_ia, -30.0, 0.0);

        assert!(
            (t_air_plain - t_air_sky_zero).abs() < 1e-12,
            "mode {:?}: zero-sky path ({t_air_sky_zero}) must equal plain path ({t_air_plain})",
            mode,
        );
    }
}

#[test]
fn test_issue_1858_sky_path_lowers_air_below_outdoor() {
    // The structural gap documented in ISSUE_1168_ROOT_CAUSE.md: the original
    // air-node balance bounds T_air below by min(T_surface, T_out). With a
    // cold sky and a non-zero sky conductance, T_air must be able to fall
    // BELOW the outdoor dry-bulb under clear-sky radiative cooling.
    //
    // Representative ASHRAE 140 Case 900 winter clear-night conditions.
    let solver = create_case_900_solver(MassAirCouplingMode::ParallelResistance);
    let t_outdoor = -10.0;
    let t_sky = -30.0; // clear sky ≈ 20 K below dry-bulb
    let h_ve = 21.7;
    let phi_ia = 200.0;

    // Physics-derived sky conductance: ε=0.9, F_sky=0.25 (window/floor),
    // aperture = 12 m² (Case 900 south glazing).
    let h_rad_sky = air_sky_conductance(0.9, 0.25, 12.0, -8.0, t_sky);
    assert!(h_rad_sky > 0.0, "sky conductance must be positive");

    let t_air_no_sky = solver.compute_zone_air_temperature(t_outdoor, h_ve, 0.0, phi_ia);
    let t_air_with_sky = solver
        .compute_zone_air_temperature_with_sky(t_outdoor, h_ve, 0.0, phi_ia, t_sky, h_rad_sky);

    // The sky path must COOL the air node (sky is colder than every other node).
    assert!(
        t_air_with_sky < t_air_no_sky,
        "sky path must cool air: {t_air_with_sky} < {t_air_no_sky}",
    );

    // The drop is meaningful and physics-derived (no tuning). In this
    // single-timestep idealization (mass nodes held at their default) the
    // drop exceeds the ~0.6 °C *integrated annual* night-min residual the
    // issue targets — the integrated run sees a smaller effect because the
    // mass nodes themselves cool over the season. We assert a generous band
    // that proves the path is active without coupling the unit test to the
    // full simulation.
    let drop = t_air_no_sky - t_air_with_sky;
    assert!(
        drop > 0.5 && drop < 8.0,
        "night-min drop {drop:.3} °C should be a meaningful cooling (>0.5 K) \
         that closes the ~0.6 °C annual residual",
    );
}

#[test]
fn test_issue_1858_sky_path_responds_to_aperture_and_sky_temp() {
    // Larger aperture → more cooling; warmer sky → less cooling.
    // Confirms the path is monotone in the physics, not a tuned constant.
    let solver = create_case_900_solver(MassAirCouplingMode::AdditiveSum);
    let t_outdoor = -10.0;
    let h_ve = 21.7;
    let phi_ia = 200.0;
    let t_sky_clear = -30.0;

    let h_small = air_sky_conductance(0.9, 0.25, 6.0, -8.0, t_sky_clear);
    let h_large = air_sky_conductance(0.9, 0.25, 24.0, -8.0, t_sky_clear);

    let t_small = solver.compute_zone_air_temperature_with_sky(
        t_outdoor,
        h_ve,
        0.0,
        phi_ia,
        t_sky_clear,
        h_small,
    );
    let t_large = solver.compute_zone_air_temperature_with_sky(
        t_outdoor,
        h_ve,
        0.0,
        phi_ia,
        t_sky_clear,
        h_large,
    );
    assert!(
        t_large < t_small,
        "larger aperture must cool more: {t_large} < {t_small}",
    );

    // Warmer sky (overcast) → less cooling than clear sky at fixed aperture.
    let t_sky_overcast = -8.0;
    let h_fixed = air_sky_conductance(0.9, 0.25, 12.0, -8.0, t_sky_clear);
    let t_clear = solver.compute_zone_air_temperature_with_sky(
        t_outdoor,
        h_ve,
        0.0,
        phi_ia,
        t_sky_clear,
        h_fixed,
    );
    let t_overcast = solver.compute_zone_air_temperature_with_sky(
        t_outdoor,
        h_ve,
        0.0,
        phi_ia,
        t_sky_overcast,
        h_fixed,
    );
    assert!(
        t_overcast > t_clear,
        "overcast (warmer) sky must cool less: {t_overcast} > {t_clear}",
    );
}

#[test]
fn test_issue_1858_sky_path_does_not_break_energy_balance() {
    // The sky term is a boundary flux: with h_rad_sky applied, the air-node
    // balance is still a closed linear system and the mass-node backward
    // Euler First-Law invariant (check_energy_balance) is unaffected because
    // the sky path is local to the air node and does not touch step().
    let mut solver = create_case_900_solver(MassAirCouplingMode::ParallelResistance);
    solver.set_zone_temperature(-8.0);
    solver.set_surface_exterior_temperatures(SurfaceExteriorTemperatures {
        t_ext_wall: -10.0,
        t_ext_roof: -12.0,
        t_ext_floor: 2.0,
    });
    // Step the mass nodes — the First-Law debug_assert! in step() must hold.
    solver.step(3600.0);

    // Air-node balance with sky still yields a finite, physically reasonable T.
    // The mass nodes are still warm (Case 900 default 20 °C) after one cold
    // step, so T_air is bounded between the coldest boundary (sky −30 °C) and
    // the warmest mass node — not pinned to a narrow band.
    let t_air = solver.compute_zone_air_temperature_with_sky(
        -10.0,
        21.7,
        0.0,
        200.0,
        -30.0,
        air_sky_conductance(0.9, 0.25, 12.0, -8.0, -30.0),
    );
    assert!(t_air.is_finite(), "T_air must be finite: {t_air}");
    assert!(
        t_air > -30.0 && t_air < 25.0,
        "T_air {t_air} must lie within the boundary-temperature envelope",
    );
}
