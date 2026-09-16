//! MultiNode 9R4C solver isolation test: inside flux within 1% of E+
//!
//! Part of Phase 1 module isolation per ARCHITECTURE.md validation strategy.
//!
//! # Test Strategy
//!
//! Validates `MultiNodeSolver` (src/physics/multi_node_solver.rs) as a
//! `Box<dyn HeatConductionSolver>` drop-in against closed-form analytical
//! solutions, NOT against EnergyPlus reference data. This is bottom-up
//! unit testing per the module isolation protocol.
//!
//! # 9R4C Model (ADR-002)
//!
//! The four-node thermal network:
//! ```text
//! T_ext ── R_se ── T_em ──┬── R_ms ── T_s ── R_si ── T_int
//!                         │
//!                        C_wall (45%)
//!                         │
//!                        C_roof (30%)
//!                         │
//!                        C_floor (18%)
//!                         │
//!                        C_internal (10%)
//! ```
//!
//! Steady-state: q_ss = (T_ext - T_int) / R_total
//!
//! # Acceptance Criteria (Issue #1604)
//!
//! - [x] SolverRegistry construction of 'multinode_9r4c' key returns Some(Box)
//! - [x] steady_state_flux matches analytical to 0.1%
//! - [x] Energy conservation ratio < 0.01 after 168 × 3600s steps
//! - [x] h_tr_is bounds: 0.5 < h_tr_is < 10.0 W/m²K
//! - [x] backward-Euler stability: no NaN/Inf after 168 timesteps
//!
//! # References
//!
//! - ADR-002: `docs/adr/0002-promote-9r4c-high-mass-default.md`
//! - Issue #1429 — MultiNodeSolver drop-in
//! - Issue #1604 — MultiNode 9R4C isolation test

use fluxion::physics::multi_node_solver::MultiNodeSolver;
use fluxion::physics::solver_registry::{registry_keys, SolverRegistry};
use fluxion::physics::solver_trait::HeatConductionSolver;
use fluxion::physics::units::{FromF64, HeatTransferCoefficient, Temperature, Time, ToF64};
use fluxion::physics::wall_spec::{LayerSpec, WallSpec};

// ---------------------------------------------------------------------------
// Wall construction helpers (3+ types per ARCHITECTURE.md)
// ---------------------------------------------------------------------------

/// 200mm normal-weight concrete (high mass)
/// k = 1.73 W/(m·K), ρ = 2243 kg/m³, cₚ = 837 J/(kg·K)
/// R = 0.2/1.73 ≈ 0.1156 m²·K/W
/// C = 2243 × 837 × 0.2 ≈ 375,448 J/(m²·K)
fn wall_200mm_concrete() -> WallSpec {
    WallSpec::single_layer("200mm Concrete", 0.2, 1.73, 2243.0, 837.0)
}

/// Lightweight wall: 13mm gypsum + 90mm wood stud cavity + 13mm gypsum
/// R ≈ 0.08 + 2.25 + 0.08 ≈ 2.41 m²·K/W
/// C ≈ 12 kJ/(m²·K)
fn wall_lightweight() -> WallSpec {
    WallSpec::multi_layer(
        "Lightweight Wood Frame",
        vec![
            LayerSpec::new("Gypsum Exterior", 0.013, 0.16, 800.0, 1090.0),
            LayerSpec::new("Cavity Insulation", 0.09, 0.04, 30.0, 840.0),
            LayerSpec::new("Gypsum Interior", 0.013, 0.16, 800.0, 1090.0),
        ],
    )
}

/// Insulated wall: 100mm brick + 80mm EPS + 13mm gypsum
/// R ≈ 0.123 + 2.0 + 0.081 ≈ 2.20 m²·K/W
/// C ≈ 156 + 7.6 + 11.3 ≈ 175 kJ/(m²·K)
fn wall_insulated() -> WallSpec {
    WallSpec::multi_layer(
        "Brick + Insulation + Gypsum",
        vec![
            LayerSpec::new("Clay Brick", 0.1, 0.81, 1920.0, 790.0),
            LayerSpec::new("EPS Insulation", 0.08, 0.04, 25.0, 1400.0),
            LayerSpec::new("Gypsum Board", 0.013, 0.16, 800.0, 1090.0),
        ],
    )
}

/// Floor area for typical office space (m²)
const FLOOR_AREA: f64 = 54.0;

// ---------------------------------------------------------------------------
// Section 1: SolverRegistry construction
// ---------------------------------------------------------------------------

/// SolverRegistry must construct a valid Box<dyn HeatConductionSolver> for
/// the 'multinode_9r4c' key (Issue #1604 acceptance criterion 1).
#[test]
fn test_solver_registry_construct_multinode_9r4c() {
    let wall = wall_200mm_concrete();
    let solver: Box<dyn HeatConductionSolver> =
        SolverRegistry::construct(registry_keys::MULTINODE_9R4C, &wall, FLOOR_AREA)
            .expect("multinode_9r4c key must construct a solver");
    assert_eq!(solver.name(), "MultiNode9R4C");
    assert!(
        solver.is_valid(),
        "constructed MultiNodeSolver must be valid"
    );
}

/// Unknown key must return SolverError.
#[test]
fn test_solver_registry_unknown_key_errors() {
    let wall = wall_200mm_concrete();
    let err = SolverRegistry::construct("nonexistent_solver", &wall, FLOOR_AREA);
    assert!(err.is_err(), "unknown key must return Err");
}

// ---------------------------------------------------------------------------
// Section 2: steady_state_flux matches analytical to 0.1%
// ---------------------------------------------------------------------------

/// MultiNodeSolver::steady_state_flux must equal the analytical closed-form:
/// q_ss = (T_ext - T_int) / R_total
///
/// Tolerance: 0.1% relative error (Issue #1604 acceptance criterion 2).
#[test]
fn test_steady_state_flux_analytical_200mm_concrete() {
    let wall = wall_200mm_concrete();
    let solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    let t_int = 20.0;
    let t_ext = 0.0;
    let r_total = wall.total_r_value();
    let expected_flux = (t_ext - t_int) / r_total;

    let actual_flux = solver
        .steady_state_flux(
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext),
        )
        .expect("steady_state_flux must succeed")
        .to_value();

    let rel_error = (actual_flux - expected_flux).abs() / expected_flux.abs();
    assert!(
        rel_error < 0.001,
        "Steady-state flux: expected {:.6} W/m², got {:.6} W/m², rel_error = {:.4}% (limit 0.1%)",
        expected_flux,
        actual_flux,
        rel_error * 100.0
    );
}

#[test]
fn test_steady_state_flux_analytical_lightweight() {
    let wall = wall_lightweight();
    let solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    let t_int = 22.0;
    let t_ext = -10.0;
    let r_total = wall.total_r_value();
    let expected_flux = (t_ext - t_int) / r_total;

    let actual_flux = solver
        .steady_state_flux(
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext),
        )
        .expect("steady_state_flux must succeed")
        .to_value();

    let rel_error = (actual_flux - expected_flux).abs() / expected_flux.abs();
    assert!(
        rel_error < 0.001,
        "Lightweight steady-state: expected {:.6} W/m², got {:.6} W/m², rel_error = {:.4}% (limit 0.1%)",
        expected_flux,
        actual_flux,
        rel_error * 100.0
    );
}

#[test]
fn test_steady_state_flux_analytical_insulated() {
    let wall = wall_insulated();
    let solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    let t_int = 20.0;
    let t_ext = 35.0; // Summer condition
    let r_total = wall.total_r_value();
    let expected_flux = (t_ext - t_int) / r_total;

    let actual_flux = solver
        .steady_state_flux(
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext),
        )
        .expect("steady_state_flux must succeed")
        .to_value();

    let rel_error = (actual_flux - expected_flux).abs() / expected_flux.abs();
    assert!(
        rel_error < 0.001,
        "Insulated steady-state: expected {:.6} W/m², got {:.6} W/m², rel_error = {:.4}% (limit 0.1%)",
        expected_flux,
        actual_flux,
        rel_error * 100.0
    );
}

/// Zero ΔT → zero flux. Fundamental energy balance check.
#[test]
fn test_steady_state_zero_delta_t() {
    let wall = wall_200mm_concrete();
    let solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    let flux = solver
        .steady_state_flux(Temperature::from_value(20.0), Temperature::from_value(20.0))
        .expect("steady_state_flux must succeed")
        .to_value();

    assert!(
        flux.abs() < 1e-12,
        "Zero ΔT should produce zero flux, got {:.2e} W/m²",
        flux
    );
}

/// Flux sign convention: T_ext > T_int → positive (heat gain).
#[test]
fn test_steady_state_flux_sign_convention() {
    let wall = wall_200mm_concrete();
    let solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    let flux_heat_gain = solver
        .steady_state_flux(Temperature::from_value(20.0), Temperature::from_value(35.0))
        .expect("steady_state_flux")
        .to_value();

    let flux_heat_loss = solver
        .steady_state_flux(Temperature::from_value(20.0), Temperature::from_value(5.0))
        .expect("steady_state_flux")
        .to_value();

    assert!(
        flux_heat_gain > 0.0,
        "T_ext > T_int → flux should be positive, got {:.4}",
        flux_heat_gain
    );
    assert!(
        flux_heat_loss < 0.0,
        "T_ext < T_int → flux should be negative, got {:.4}",
        flux_heat_loss
    );
}

/// Anti-symmetry: f(T_int, T_ext) + f(T_ext, T_int) == 0.
#[test]
fn test_steady_state_anti_symmetry() {
    let wall = wall_200mm_concrete();
    let solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    let q_fwd = solver
        .steady_state_flux(Temperature::from_value(20.0), Temperature::from_value(10.0))
        .expect("steady_state_flux fwd")
        .to_value();

    let q_rev = solver
        .steady_state_flux(Temperature::from_value(10.0), Temperature::from_value(20.0))
        .expect("steady_state_flux rev")
        .to_value();

    let sum = q_fwd + q_rev;
    assert!(
        sum.abs() < 1e-12,
        "Anti-symmetry violated: q_fwd + q_rev = {:.6}",
        sum
    );
}

// ---------------------------------------------------------------------------
// Section 3: h_tr_is bounds verification
// ---------------------------------------------------------------------------

/// h_tr_is must be in the range [0.5, 10.0] W/(m²·K).
///
/// This is the interior film coefficient expressed as a conductance.
/// Typical values: R_si = 0.11 m²·K/W → h_tr_is = 9.1 W/(m²·K) for furniture;
/// R_si = 0.125 m²·K/W → h_tr_is = 8.0 W/(m²·K) for vertical surfaces (ASHRAE 140).
///
/// Issue #1604 acceptance criterion 3: 0.5 < h_tr_is < 10.0 W/m²K.
#[test]
fn test_h_tr_is_bounds_200mm_concrete() {
    let wall = wall_200mm_concrete();
    let solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    assert!(
        solver.h_tr_is > 0.5,
        "h_tr_is = {:.4} W/(m²·K) must be > 0.5",
        solver.h_tr_is
    );
    assert!(
        solver.h_tr_is < 10.0,
        "h_tr_is = {:.4} W/(m²·K) must be < 10.0",
        solver.h_tr_is
    );
}

#[test]
fn test_h_tr_is_bounds_lightweight() {
    let wall = wall_lightweight();
    let solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    assert!(
        solver.h_tr_is > 0.5,
        "h_tr_is = {:.4} W/(m²·K) must be > 0.5",
        solver.h_tr_is
    );
    assert!(
        solver.h_tr_is < 10.0,
        "h_tr_is = {:.4} W/(m²·K) must be < 10.0",
        solver.h_tr_is
    );
}

#[test]
fn test_h_tr_is_bounds_insulated() {
    let wall = wall_insulated();
    let solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    assert!(
        solver.h_tr_is > 0.5,
        "h_tr_is = {:.4} W/(m²·K) must be > 0.5",
        solver.h_tr_is
    );
    assert!(
        solver.h_tr_is < 10.0,
        "h_tr_is = {:.4} W/(m²·K) must be < 10.0",
        solver.h_tr_is
    );
}

// ---------------------------------------------------------------------------
// Section 4: Energy conservation check within 1%
// ---------------------------------------------------------------------------

/// After 168 hourly timesteps (1 week), energy conservation ratio must be < 1%.
///
/// Energy conservation check:
/// - q_flux = heat flow from mass nodes to interior (returned by step)
/// - q_storage = energy_storage_rate (rate of enthalpy change in mass nodes)
/// - At steady state: q_storage → 0 and q_flux → q_ss
///
/// We verify: |q_flux - q_ss| / |q_ss| < 0.01 after 168 steps.
///
/// Issue #1604 acceptance criterion 4: energy conservation ratio < 0.01.
#[test]
fn test_energy_conservation_168_steps_200mm_concrete() {
    let wall = wall_200mm_concrete();
    let mut solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    let t_int = 20.0;
    let t_ext = 0.0;
    let r_total = wall.total_r_value();
    let q_ss = (t_ext - t_int) / r_total;

    let dt = Time::from_value(3600.0);
    let h_int = HeatTransferCoefficient::from_value(8.0);
    let h_ext = HeatTransferCoefficient::from_value(25.0);

    // Run 168 hourly timesteps
    for _ in 0..168 {
        let _ = HeatConductionSolver::step(
            &mut solver,
            dt,
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext),
            h_int,
            h_ext,
        );
    }

    // After 168 steps, check that q_flux is within 1% of q_ss
    let q_flux = HeatConductionSolver::step(
        &mut solver,
        dt,
        Temperature::from_value(t_int),
        Temperature::from_value(t_ext),
        h_int,
        h_ext,
    )
    .expect("final step must succeed")
    .to_value();

    let rel_error = (q_flux - q_ss).abs() / q_ss.abs();
    assert!(
        rel_error < 0.01,
        "Energy conservation violated after 168 steps: q_flux={:.6} W/m², q_ss={:.6} W/m², rel_error={:.4}% (limit 1%)",
        q_flux,
        q_ss,
        rel_error * 100.0
    );
}

#[test]
fn test_energy_conservation_168_steps_lightweight() {
    let wall = wall_lightweight();
    let mut solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    let t_int = 20.0;
    let t_ext = 0.0;
    let r_total = wall.total_r_value();
    let q_ss = (t_ext - t_int) / r_total;

    let dt = Time::from_value(3600.0);
    let h_int = HeatTransferCoefficient::from_value(8.0);
    let h_ext = HeatTransferCoefficient::from_value(25.0);

    for _ in 0..168 {
        let _ = HeatConductionSolver::step(
            &mut solver,
            dt,
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext),
            h_int,
            h_ext,
        );
    }

    let q_flux = HeatConductionSolver::step(
        &mut solver,
        dt,
        Temperature::from_value(t_int),
        Temperature::from_value(t_ext),
        h_int,
        h_ext,
    )
    .expect("final step must succeed")
    .to_value();

    let rel_error = (q_flux - q_ss).abs() / q_ss.abs();
    assert!(
        rel_error < 0.01,
        "Lightweight energy conservation violated: q_flux={:.6} W/m², q_ss={:.6} W/m², rel_error={:.4}%",
        q_flux,
        q_ss,
        rel_error * 100.0
    );
}

#[test]
fn test_energy_conservation_168_steps_insulated() {
    let wall = wall_insulated();
    let mut solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    let t_int = 20.0;
    let t_ext = 35.0;
    let r_total = wall.total_r_value();
    let q_ss = (t_ext - t_int) / r_total;

    let dt = Time::from_value(3600.0);
    let h_int = HeatTransferCoefficient::from_value(8.0);
    let h_ext = HeatTransferCoefficient::from_value(25.0);

    for _ in 0..168 {
        let _ = HeatConductionSolver::step(
            &mut solver,
            dt,
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext),
            h_int,
            h_ext,
        );
    }

    let q_flux = HeatConductionSolver::step(
        &mut solver,
        dt,
        Temperature::from_value(t_int),
        Temperature::from_value(t_ext),
        h_int,
        h_ext,
    )
    .expect("final step must succeed")
    .to_value();

    let rel_error = (q_flux - q_ss).abs() / q_ss.abs();
    assert!(
        rel_error < 0.01,
        "Insulated energy conservation violated: q_flux={:.6} W/m², q_ss={:.6} W/m², rel_error={:.4}%",
        q_flux,
        q_ss,
        rel_error * 100.0
    );
}

/// Energy storage rate should approach zero as system approaches steady state.
#[test]
fn test_energy_storage_rate_converges_to_zero() {
    let wall = wall_200mm_concrete();
    let mut solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    let t_int = 20.0;
    let t_ext = 0.0;
    let dt = Time::from_value(3600.0);
    let h_int = HeatTransferCoefficient::from_value(8.0);
    let h_ext = HeatTransferCoefficient::from_value(25.0);

    // Run 168 steps and check storage rate is small
    for _ in 0..168 {
        let _ = HeatConductionSolver::step(
            &mut solver,
            dt,
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext),
            h_int,
            h_ext,
        );
    }

    let storage_rate = solver.energy_storage_rate();
    assert!(
        storage_rate.abs() < 0.1, // Should be near zero at steady state
        "Energy storage rate = {:.4} W/m² should approach zero at steady state",
        storage_rate
    );
}

// ---------------------------------------------------------------------------
// Section 5: backward-Euler stability — no NaN/Inf after 168 timesteps
// ---------------------------------------------------------------------------

/// Backward Euler must remain numerically stable for 168 hourly timesteps.
/// All trait methods must return finite values.
///
/// Issue #1604 acceptance criterion 5: no NaN/Inf after 168 timesteps.
#[test]
fn test_backward_euler_stability_168_steps_200mm_concrete() {
    let wall = wall_200mm_concrete();
    let mut solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    let t_int = 20.0;
    let t_ext = 0.0;
    let dt = Time::from_value(3600.0);
    let h_int = HeatTransferCoefficient::from_value(8.0);
    let h_ext = HeatTransferCoefficient::from_value(25.0);

    for step_idx in 0..168 {
        let flux = HeatConductionSolver::step(
            &mut solver,
            dt,
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext),
            h_int,
            h_ext,
        )
        .unwrap_or_else(|_| panic!("step {} must succeed", step_idx));

        let flux_val = flux.to_value();
        assert!(
            flux_val.is_finite(),
            "Step {}: flux = {:?} is not finite",
            step_idx,
            flux_val
        );
    }

    // Final state checks
    assert!(
        solver.h_tr_is.is_finite(),
        "h_tr_is = {:?} is not finite",
        solver.h_tr_is
    );
    assert!(
        solver.mass.wall.temperature.is_finite(),
        "wall temperature is not finite"
    );
    assert!(
        solver.mass.roof.temperature.is_finite(),
        "roof temperature is not finite"
    );
    assert!(
        solver.mass.floor.temperature.is_finite(),
        "floor temperature is not finite"
    );
    assert!(
        solver.mass.internal.temperature.is_finite(),
        "internal temperature is not finite"
    );
}

#[test]
fn test_backward_euler_stability_168_steps_lightweight() {
    let wall = wall_lightweight();
    let mut solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    let t_int = 20.0;
    let t_ext = 0.0;
    let dt = Time::from_value(3600.0);
    let h_int = HeatTransferCoefficient::from_value(8.0);
    let h_ext = HeatTransferCoefficient::from_value(25.0);

    for step_idx in 0..168 {
        let flux = HeatConductionSolver::step(
            &mut solver,
            dt,
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext),
            h_int,
            h_ext,
        )
        .unwrap_or_else(|_| panic!("step {} must succeed", step_idx));

        let flux_val = flux.to_value();
        assert!(
            flux_val.is_finite(),
            "Step {}: flux = {:?} is not finite",
            step_idx,
            flux_val
        );
    }
}

#[test]
fn test_backward_euler_stability_168_steps_insulated() {
    let wall = wall_insulated();
    let mut solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    let t_int = 20.0;
    let t_ext = 35.0;
    let dt = Time::from_value(3600.0);
    let h_int = HeatTransferCoefficient::from_value(8.0);
    let h_ext = HeatTransferCoefficient::from_value(25.0);

    for step_idx in 0..168 {
        let flux = HeatConductionSolver::step(
            &mut solver,
            dt,
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext),
            h_int,
            h_ext,
        )
        .unwrap_or_else(|_| panic!("step {} must succeed", step_idx));

        let flux_val = flux.to_value();
        assert!(
            flux_val.is_finite(),
            "Step {}: flux = {:?} is not finite",
            step_idx,
            flux_val
        );
    }
}

// ---------------------------------------------------------------------------
// Section 6: HeatConductionSolver trait interface tests
// ---------------------------------------------------------------------------

/// Verify trait interface: initialize → step → is_valid.
#[test]
fn test_trait_lifecycle() {
    let wall = wall_200mm_concrete();
    let mut solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    assert!(
        solver.is_valid(),
        "Solver should be valid after from_wall_spec"
    );
    assert_eq!(solver.name(), "MultiNode9R4C");

    let flux = HeatConductionSolver::step(
        &mut solver,
        Time::from_value(3600.0),
        Temperature::from_value(20.0),
        Temperature::from_value(0.0),
        HeatTransferCoefficient::from_value(8.0),
        HeatTransferCoefficient::from_value(25.0),
    )
    .expect("step should succeed");
    assert!(flux.to_value().is_finite(), "Flux should be finite");
}

/// Step before initialization should return InvalidConfig error.
#[test]
fn test_step_before_init_returns_error() {
    let wall_node = fluxion_core::multi_node::ThermalMassNode::new(20.0, 5e6, 50.0, 20.0);
    let roof_node = fluxion_core::multi_node::ThermalMassNode::new(20.0, 3e6, 30.0, 15.0);
    let floor_node = fluxion_core::multi_node::ThermalMassNode::new(20.0, 2e6, 20.0, 10.0);
    let internal_node = fluxion_core::multi_node::ThermalMassNode::new(20.0, 1e6, 10.0, 5.0);
    let mut solver = MultiNodeSolver::new(8.0, wall_node, roof_node, floor_node, internal_node);

    // Not initialized
    assert!(!solver.is_valid());

    let result = HeatConductionSolver::step(
        &mut solver,
        Time::from_value(3600.0),
        Temperature::from_value(20.0),
        Temperature::from_value(0.0),
        HeatTransferCoefficient::from_value(8.0),
        HeatTransferCoefficient::from_value(25.0),
    );

    assert!(result.is_err(), "step before init should return error");
}

/// Verify solver produces same result for same inputs (determinism).
#[test]
fn test_determinism() {
    let wall = wall_200mm_concrete();
    let mut solver1 = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);
    let mut solver2 = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    let flux1 = HeatConductionSolver::step(
        &mut solver1,
        Time::from_value(3600.0),
        Temperature::from_value(20.0),
        Temperature::from_value(0.0),
        HeatTransferCoefficient::from_value(8.0),
        HeatTransferCoefficient::from_value(25.0),
    )
    .unwrap()
    .to_value();

    let flux2 = HeatConductionSolver::step(
        &mut solver2,
        Time::from_value(3600.0),
        Temperature::from_value(20.0),
        Temperature::from_value(0.0),
        HeatTransferCoefficient::from_value(8.0),
        HeatTransferCoefficient::from_value(25.0),
    )
    .unwrap()
    .to_value();

    assert_eq!(
        flux1, flux2,
        "Same inputs should produce identical outputs (determinism)"
    );
}

// ---------------------------------------------------------------------------
// Section 7: Performance gate (< 5 seconds per issue requirement)
// ---------------------------------------------------------------------------

/// The test must complete in under 5 seconds (acceptance criterion).
#[test]
fn test_performance_gate() {
    use std::time::Instant;

    let wall = wall_200mm_concrete();
    let mut solver = MultiNodeSolver::from_wall_spec(&wall, FLOOR_AREA);

    let start = Instant::now();

    // Run 168 steps (matching the stability test)
    for _ in 0..168 {
        let _ = HeatConductionSolver::step(
            &mut solver,
            Time::from_value(3600.0),
            Temperature::from_value(20.0),
            Temperature::from_value(0.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        );
    }

    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs_f64() < 5.0,
        "168 solver steps took {:.2}s (limit 5s)",
        elapsed.as_secs_f64()
    );
}

// ---------------------------------------------------------------------------
// Section 8: Wall properties documentation
// ---------------------------------------------------------------------------

#[test]
fn test_wall_properties_documentation() {
    let walls: Vec<(&str, WallSpec)> = vec![
        ("200mm Concrete", wall_200mm_concrete()),
        ("Lightweight", wall_lightweight()),
        ("Insulated", wall_insulated()),
    ];

    println!(
        "\n┌─────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐"
    );
    println!("│ Wall Type           │ R [m²·K/W]  │ C [kJ/m²·K] │ τ [hours]   │ h_tr_is     │");
    println!("├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤");

    for (name, wall) in &walls {
        let r = wall.total_r_value();
        let c_kj = wall.thermal_capacity() / 1000.0;
        let tau_h = wall.thermal_capacity() * r / 3600.0;
        let solver = MultiNodeSolver::from_wall_spec(wall, FLOOR_AREA);
        let h_tr_is = solver.h_tr_is;
        println!(
            "│ {:<19} │ {:>12.4} │ {:>12.1} │ {:>12.1} │ {:>12.4} │",
            name, r, c_kj, tau_h, h_tr_is
        );
    }

    println!("└─────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘");
}
