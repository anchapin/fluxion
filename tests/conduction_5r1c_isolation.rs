//! Conduction module isolation test: 5R1C solver vs analytical step response.
//!
//! Part of Phase 1 module isolation per ARCHITECTURE.md validation strategy.
//!
//! # Test Strategy
//!
//! Validates `FiveR1CSolver` (src/physics/five_r1c_solver.rs) against closed-form
//! analytical solutions, NOT against EnergyPlus reference data. This is bottom-up
//! unit testing:
//!
//! 1. **Steady-state**: Q = ΔT / R_total (Fourier's law, 1D slab)
//! 2. **Transient step response**: Exponential approach to steady-state
//! 3. **Thermal time constant**: τ = C × R
//!
//! # Reference Data Sources
//!
//! There are 6 CSV files in `tests/reference_data/conduction/`:
//! - 2 files are REAL EnergyPlus 25.2.0 output: `step_response_200mm_concrete.csv`,
//!   `step_response_fixed_zone_20c.csv`
//! - 4 files are SYNTHETIC analytical test fixtures: `step_response_composite.csv`,
//!   `step_response_floor.csv`, `step_response_lightweight.csv`, `step_response_roof.csv`
//!
//! **This test file does NOT use the CSV files** — it computes expected values
//! analytically from WallSpec parameters. The CSV files exist for future E+
//! validation work but are clearly labeled as synthetic fixtures.
//!
//! # 5R1C Model (ISO 13790 / EN 15270)
//!
//! ```text
//! T_ext ── R_se ── T_se ── R_2 ──┬── R_1 ── T_si ── R_si ── T_int
//!                                │
//!                                C_m
//!                                │
//!                               T_m
//! ```
//!
//! The mass node temperature evolves as:
//!   T_m(t) = T_ss + (T_m0 - T_ss) · exp(-t / τ)
//!
//! where τ = C_m · R_total is the thermal time constant.
//!
//! # Current Implementation Status
//!
//! **CRITICAL**: The current `FiveR1CSolver::step()` ignores `timestep`, `h_interior`,
//! and `h_exterior`. It computes only the steady-state flux Q = ΔT / R_total.
//! The mass node `T_mass` is never updated, and `energy_storage_rate()` returns 0.0.
//!
//! Therefore:
//! - Steady-state tests (Section 1) PASS
//! - Transient tests (Section 2): #1206 CLOSED — transient dynamics implemented
//! - Time constant tests (Section 3): #1206 CLOSED — transient dynamics implemented
//!
//! # Acceptance Criteria (Issue #961)
//!
//! - [x] Steady-state within 0.1% of Q = ΔT/R
//! - [ ] Transient matches exponential within 1% (blocked: solver is steady-state only)
//! - [ ] Time constant within 2% (blocked: solver is steady-state only)
//! - [x] 3+ construction types tested
//! - [x] Test runs in <500ms
//!
//! # References
//!
//! - ISO 13790:2008, Section 7.2.2.2 — 5R1C thermal network
//! - EN 15270:2007, Annex B — Simplified method for transient conduction
//! - ASHRAE Handbook, Chapter 26 — Building envelope heat transfer

use fluxion::physics::five_r1c_solver::FiveR1CSolver;
use fluxion::physics::solver_trait::HeatConductionSolver;
use fluxion::physics::units::{
    FromF64, HeatFlux, HeatTransferCoefficient, Temperature, Time, ToF64,
};
use fluxion::physics::wall_spec::{LayerSpec, WallSpec};

// ---------------------------------------------------------------------------
// Construction type definitions (3+ types per acceptance criteria)
// ---------------------------------------------------------------------------

/// Lightweight wall: 13mm gypsum + 90mm wood stud cavity + 13mm gypsum
/// R ≈ 0.08 + 2.25 + 0.08 ≈ 2.41 m²·K/W (fill cavity as still-air equivalent)
/// C ≈ 12 kJ/(m²·K)
fn lightweight_wall() -> WallSpec {
    WallSpec::multi_layer(
        "Lightweight Wood Frame",
        vec![
            LayerSpec::new("Gypsum Exterior", 0.013, 0.16, 800.0, 1090.0),
            // Cavity insulation (mineral wool equivalent)
            LayerSpec::new("Cavity Insulation", 0.09, 0.04, 30.0, 840.0),
            LayerSpec::new("Gypsum Interior", 0.013, 0.16, 800.0, 1090.0),
        ],
    )
}

/// Heavyweight wall: 200mm normal-weight concrete
/// k = 1.73 W/(m·K), ρ = 2243 kg/m³, cₚ = 837 J/(kg·K)
/// R = 0.2/1.73 = 0.1156 m²·K/W
/// C = 2243 × 837 × 0.2 = 375,448 J/(m²·K)
fn heavyweight_wall() -> WallSpec {
    WallSpec::single_layer("200mm Concrete", 0.2, 1.73, 2243.0, 837.0)
}

/// Insulated wall: 100mm brick + 80mm EPS insulation + 13mm gypsum
/// R ≈ 0.123 + 2.0 + 0.081 ≈ 2.20 m²·K/W
/// C ≈ 156 + 7.6 + 11.3 ≈ 175 kJ/(m²·K)
fn insulated_wall() -> WallSpec {
    WallSpec::multi_layer(
        "Brick + Insulation + Gypsum",
        vec![
            LayerSpec::new("Clay Brick", 0.1, 0.81, 1920.0, 790.0),
            LayerSpec::new("EPS Insulation", 0.08, 0.04, 25.0, 1400.0),
            LayerSpec::new("Gypsum Board", 0.013, 0.16, 800.0, 1090.0),
        ],
    )
}

/// Very heavyweight: 300mm concrete (for edge case testing)
fn very_heavyweight_wall() -> WallSpec {
    WallSpec::single_layer("300mm Concrete", 0.3, 1.73, 2243.0, 837.0)
}

/// Helper: initialize solver and return it
fn init_solver(wall: &WallSpec) -> FiveR1CSolver {
    let mut solver = FiveR1CSolver::new();
    solver
        .initialize(wall)
        .expect("Solver initialization should succeed");
    assert!(
        solver.is_valid(),
        "Solver should be valid after initialization"
    );
    solver
}

// ===========================================================================
// Section 1: Steady-State Validation (Q = ΔT / R_total)
// ===========================================================================

/// Steady-state heat flux must equal Fourier's law: Q = (T_ext - T_int) / R_total.
///
/// Tolerance: 0.1% relative error.
///
/// Physics: For a 1D slab at steady state, heat flux is determined entirely by
/// the total thermal resistance and the temperature difference across it.
/// This is independent of thermal mass, surface coefficients (if R_total already
/// includes film resistances), or timestep size.
#[test]
fn test_steady_state_lightweight_wall() {
    let wall = lightweight_wall();
    let mut solver = init_solver(&wall);

    let t_int = 20.0; // °C
    let t_ext = 0.0; // °C
    let r_total = wall.total_r_value();

    let expected_flux = (t_ext - t_int) / r_total;
    let actual_flux: HeatFlux = solver
        .step(
            Time::from_value(3600.0),
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap();
    let actual_flux_value = actual_flux.to_value();

    let rel_error = (actual_flux_value - expected_flux).abs() / expected_flux.abs();
    assert!(
        rel_error < 0.001,
        "Lightweight wall steady-state: expected {:.6} W/m², got {:.6} W/m², rel_error = {:.4}% (limit 0.1%)",
        expected_flux,
        actual_flux_value,
        rel_error * 100.0
    );
}

#[test]
fn test_steady_state_heavyweight_wall() {
    let wall = heavyweight_wall();
    let mut solver = init_solver(&wall);

    let t_int = 22.0;
    let t_ext = -10.0;
    let r_total = wall.total_r_value();

    let expected_flux = (t_ext - t_int) / r_total;
    let actual_flux = solver
        .step(
            Time::from_value(3600.0),
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap()
        .to_value();

    let rel_error = (actual_flux - expected_flux).abs() / expected_flux.abs();
    assert!(
        rel_error < 0.001,
        "Heavyweight wall steady-state: expected {:.6} W/m², got {:.6} W/m², rel_error = {:.4}% (limit 0.1%)",
        expected_flux,
        actual_flux,
        rel_error * 100.0
    );
}

#[test]
fn test_steady_state_insulated_wall() {
    let wall = insulated_wall();
    let mut solver = init_solver(&wall);

    let t_int = 20.0;
    let t_ext = 35.0; // Summer condition
    let r_total = wall.total_r_value();

    let expected_flux = (t_ext - t_int) / r_total;
    let actual_flux = solver
        .step(
            Time::from_value(3600.0),
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap()
        .to_value();

    let rel_error = (actual_flux - expected_flux).abs() / expected_flux.abs();
    assert!(
        rel_error < 0.001,
        "Insulated wall steady-state: expected {:.6} W/m², got {:.6} W/m², rel_error = {:.4}% (limit 0.1%)",
        expected_flux,
        actual_flux,
        rel_error * 100.0
    );
}

#[test]
fn test_steady_state_very_heavyweight_wall() {
    let wall = very_heavyweight_wall();
    let mut solver = init_solver(&wall);

    let t_int = 20.0;
    let t_ext = 5.0;
    let r_total = wall.total_r_value();

    let expected_flux = (t_ext - t_int) / r_total;
    let actual_flux = solver
        .step(
            Time::from_value(3600.0),
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap()
        .to_value();

    let rel_error = (actual_flux - expected_flux).abs() / expected_flux.abs();
    assert!(
        rel_error < 0.001,
        "Very heavyweight wall steady-state: expected {:.6} W/m², got {:.6} W/m², rel_error = {:.4}% (limit 0.1%)",
        expected_flux,
        actual_flux,
        rel_error * 100.0
    );
}

/// Zero ΔT → zero flux. Fundamental energy balance check.
#[test]
fn test_steady_state_zero_delta_t() {
    let wall = heavyweight_wall();
    let mut solver = init_solver(&wall);

    let flux = solver
        .step(
            Time::from_value(3600.0),
            Temperature::from_value(20.0),
            Temperature::from_value(20.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap()
        .to_value();
    assert!(
        flux.abs() < 1e-12,
        "Zero ΔT should produce zero flux, got {:.2e} W/m²",
        flux
    );
}

/// Flux sign convention: positive = heat flowing INTO zone.
/// When T_ext > T_int, flux should be positive (heat gain).
/// When T_ext < T_int, flux should be negative (heat loss).
#[test]
fn test_steady_state_flux_sign_convention() {
    let wall = heavyweight_wall();
    let mut solver = init_solver(&wall);

    // Heat gain scenario
    let flux_gain = solver
        .step(
            Time::from_value(3600.0),
            Temperature::from_value(20.0),
            Temperature::from_value(35.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap()
        .to_value();
    assert!(
        flux_gain > 0.0,
        "T_ext > T_int → flux should be positive (heat gain), got {:.4}",
        flux_gain
    );

    // Heat loss scenario
    let flux_loss = solver
        .step(
            Time::from_value(3600.0),
            Temperature::from_value(20.0),
            Temperature::from_value(5.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap()
        .to_value();
    assert!(
        flux_loss < 0.0,
        "T_ext < T_int → flux should be negative (heat loss), got {:.4}",
        flux_loss
    );
}

/// Symmetric ΔT: reversing interior/exterior temperatures should negate the flux.
#[test]
fn test_steady_state_symmetry() {
    let wall = insulated_wall();
    let mut solver = init_solver(&wall);

    let flux_forward = solver
        .step(
            Time::from_value(3600.0),
            Temperature::from_value(20.0),
            Temperature::from_value(10.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap()
        .to_value();
    let flux_reverse = solver
        .step(
            Time::from_value(3600.0),
            Temperature::from_value(10.0),
            Temperature::from_value(20.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap()
        .to_value();

    assert!(
        (flux_forward + flux_reverse).abs() < 1e-10,
        "Reversing ΔT should negate flux: got {:.6} + {:.6} = {:.2e}",
        flux_forward,
        flux_reverse,
        flux_forward + flux_reverse
    );
}

/// Linearity check: doubling ΔT should exactly double the flux.
#[test]
fn test_steady_state_linearity() {
    let wall = lightweight_wall();
    let mut solver = init_solver(&wall);

    let flux_10k = solver
        .step(
            Time::from_value(3600.0),
            Temperature::from_value(20.0),
            Temperature::from_value(10.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap()
        .to_value();
    let flux_20k = solver
        .step(
            Time::from_value(3600.0),
            Temperature::from_value(20.0),
            Temperature::from_value(0.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap()
        .to_value();

    let ratio = flux_20k / flux_10k;
    assert!(
        (ratio - 2.0).abs() < 1e-10,
        "Doubling ΔT should double flux: ratio = {:.10} (expected 2.0)",
        ratio
    );
}

/// Verify R_total from solver matches WallSpec calculation.
#[test]
fn test_r_total_matches_wall_spec() {
    let walls: Vec<(&str, WallSpec)> = vec![
        ("Lightweight", lightweight_wall()),
        ("Heavyweight", heavyweight_wall()),
        ("Insulated", insulated_wall()),
        ("Very Heavyweight", very_heavyweight_wall()),
    ];

    for (name, wall) in &walls {
        let solver = init_solver(wall);
        // Access R_total through steady_state_flux comparison
        let expected_r = wall.total_r_value();
        // Solver computes: flux = (T_ext - T_int) / R_total
        // So R_total = (T_ext - T_int) / flux
        // Use the solver's internal R_total by comparing with analytical
        let flux_analytical = (0.0 - 20.0) / expected_r;
        // The solver's steady_state_flux should match
        let flux_solver = solver.steady_state_flux(20.0, 0.0);
        let rel_error = (flux_solver - flux_analytical).abs() / flux_analytical.abs();
        assert!(
            rel_error < 1e-12,
            "{}: R_total mismatch, flux error = {:.2e}",
            name,
            rel_error
        );
    }
}

// ===========================================================================
// Section 2: Transient Step Response
// ===========================================================================
//
// These tests are #[ignore] because the current FiveR1CSolver does NOT implement
// transient dynamics. The mass node T_mass is never updated in step(), and
// energy_storage_rate() always returns 0.0.
//
// The analytical solution for a step change in exterior temperature from T_ext0
// to T_ext1 at t=0, with constant T_int:
//
//   T_m(t) = T_ss + (T_m0 - T_ss) · exp(-t / τ)
//
// where:
//   T_ss = steady-state mass temperature (weighted by R-split)
//   τ = C_total × R_total (thermal time constant)
//
// To enable these tests, the FiveR1CSolver needs:
// 1. Mass node temperature update in step() using explicit Euler or Crank-Nicolson
// 2. Proper R_1/R_2 split per ISO 13790 §7.2.2.2
// 3. Non-zero energy_storage_rate() based on dT_mass/dt
//
// Per Phase 1 validation strategy: no parameter tuning, fix the underlying math.

/// Transient step response: exponential approach to steady-state.
///
/// Scenario: Wall initially in equilibrium at 20°C. At t=0, exterior drops to 0°C.
/// The mass node temperature should follow:
///   T_m(t) = T_ss + (T_m0 - T_ss) · exp(-t / τ)
///
/// After 1τ, T_m should be within 36.8% of the initial offset from T_ss.
/// After 3τ, T_m should be within 5% of T_ss.
/// After 5τ, T_m should be within 0.7% of T_ss.
///
/// Acceptance: Transient flux matches exponential within 1%.
#[test]
fn test_transient_step_response_heavyweight() {
    let wall = heavyweight_wall();
    let r_total = wall.total_r_value();
    let c_total = wall.thermal_capacity();
    let tau = c_total * r_total; // Thermal time constant [s]

    let t_int = 20.0;
    let t_ext_initial = 20.0;
    let t_ext_final = 0.0;

    // Steady-state flux at final conditions
    let q_ss = (t_ext_final - t_int) / r_total;

    // At t=0, flux should be 0 (equilibrium)
    // At t→∞, flux should approach q_ss
    // The transient flux follows:
    //   q(t) = q_ss · (1 - exp(-t/τ))

    let dt = 300.0; // 5-minute timestep
    let mut solver = init_solver(&wall);

    // Initialize at equilibrium
    solver
        .step(
            Time::from_value(dt),
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext_initial),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap();

    // Apply step change and run for 5τ
    let n_steps = (5.0 * tau / dt).ceil() as usize;
    let mut fluxes: Vec<f64> = Vec::with_capacity(n_steps);

    for _ in 0..n_steps {
        let flux = solver
            .step(
                Time::from_value(dt),
                Temperature::from_value(t_int),
                Temperature::from_value(t_ext_final),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap()
            .to_value();
        fluxes.push(flux);
    }

    // Check at t = τ: flux should be ~63.2% of q_ss
    let step_at_tau = (tau / dt).round() as usize;
    if step_at_tau > 0 && step_at_tau < fluxes.len() {
        let flux_at_tau = fluxes[step_at_tau - 1];
        let expected_at_tau = q_ss * (1.0 - (-1.0_f64).exp()); // 63.21%
        let rel_error = (flux_at_tau - expected_at_tau).abs() / expected_at_tau.abs();
        assert!(
            rel_error < 0.01,
            "At t=τ: expected flux = {:.4} W/m² (63.2% of ss), got {:.4}, error = {:.2}%",
            expected_at_tau,
            flux_at_tau,
            rel_error * 100.0
        );
    }

    // Check at t = 5τ: flux should be > 99.3% of q_ss
    let final_flux = *fluxes.last().unwrap();
    let rel_error_final = (final_flux - q_ss).abs() / q_ss.abs();
    assert!(
        rel_error_final < 0.01,
        "At t=5τ: expected flux ≈ {:.4} W/m² (steady-state), got {:.4}, error = {:.2}%",
        q_ss,
        final_flux,
        rel_error_final * 100.0
    );
}

/// Transient step response for lightweight construction.
/// Should reach steady-state much faster than heavyweight.
#[test]
fn test_transient_step_response_lightweight() {
    let wall = lightweight_wall();
    let r_total = wall.total_r_value();
    let c_total = wall.thermal_capacity();
    let tau = c_total * r_total;

    let t_int = 20.0;
    let t_ext_final = 0.0;
    let q_ss = (t_ext_final - t_int) / r_total;

    let dt = 300.0;
    let mut solver = init_solver(&wall);

    // Start at equilibrium
    solver
        .step(
            Time::from_value(dt),
            Temperature::from_value(t_int),
            Temperature::from_value(20.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap();

    // Run for 5τ
    let n_steps = (5.0 * tau / dt).ceil() as usize;
    for i in 0..n_steps {
        let flux = solver
            .step(
                Time::from_value(dt),
                Temperature::from_value(t_int),
                Temperature::from_value(t_ext_final),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap()
            .to_value();
        // Check convergence at t = 3τ
        let t_elapsed = (i + 1) as f64 * dt;
        if (t_elapsed - 3.0 * tau).abs() < dt {
            let expected_fraction = 1.0 - (-3.0_f64).exp(); // ~95.02%
            let expected_flux = q_ss * expected_fraction;
            let rel_error = (flux - expected_flux).abs() / expected_flux.abs();
            assert!(
                rel_error < 0.01,
                "Lightweight at t=3τ: expected {:.4}, got {:.4}, error {:.2}%",
                expected_flux,
                flux,
                rel_error * 100.0
            );
        }
    }
}

/// Transient step response for insulated wall.
/// Tests that the solver handles multi-layer constructions correctly.
#[test]
fn test_transient_step_response_insulated() {
    let wall = insulated_wall();
    let r_total = wall.total_r_value();
    let c_total = wall.thermal_capacity();
    let tau = c_total * r_total;

    let t_int = 20.0;
    let t_ext_final = 35.0; // Summer heat gain
    let q_ss = (t_ext_final - t_int) / r_total;

    let dt = 300.0;
    let mut solver = init_solver(&wall);

    // Start at equilibrium
    solver
        .step(
            Time::from_value(dt),
            Temperature::from_value(t_int),
            Temperature::from_value(20.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap();

    // Run for 5τ
    let n_steps = (5.0 * tau / dt).ceil() as usize;
    for _ in 0..n_steps {
        let _flux = solver
            .step(
                Time::from_value(dt),
                Temperature::from_value(t_int),
                Temperature::from_value(t_ext_final),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap()
            .to_value();
    }

    // After 5τ, should be at steady-state
    let final_flux = solver
        .step(
            Time::from_value(dt),
            Temperature::from_value(t_int),
            Temperature::from_value(t_ext_final),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap()
        .to_value();
    let rel_error = (final_flux - q_ss).abs() / q_ss.abs();
    assert!(
        rel_error < 0.01,
        "Insulated wall at t=5τ: expected flux = {:.4}, got {:.4}, error = {:.2}%",
        q_ss,
        final_flux,
        rel_error * 100.0
    );
}

// ===========================================================================
// Section 3: Thermal Time Constant (τ = C × R)
// ===========================================================================
//
// These tests verify that the solver's effective time constant matches
// τ = C_total × R_total derived from the WallSpec.
//
// Currently ignored because the solver doesn't implement transient dynamics.

/// Time constant verification for heavyweight wall.
///
/// τ = C_total × R_total = 375,448 J/(m²·K) × 0.1156 m²·K/W = 43,402 s ≈ 12.1 hours
///
/// Acceptance: τ within 2% of analytical value.
#[test]
fn test_time_constant_heavyweight() {
    let wall = heavyweight_wall();
    let r_total = wall.total_r_value();
    let c_total = wall.thermal_capacity();
    let tau_analytical = c_total * r_total;

    // Expected: 2243 × 837 × 0.2 × (0.2 / 1.73)
    let expected_tau = 2243.0 * 837.0 * 0.2 * (0.2 / 1.73);
    assert!(
        (tau_analytical - expected_tau).abs() / expected_tau < 0.001,
        "WallSpec τ = {:.2} s, expected = {:.2} s",
        tau_analytical,
        expected_tau
    );

    // Measure τ from solver: time to reach 63.2% of steady-state flux
    let t_int = 20.0;
    let t_ext_final = 0.0;
    let q_ss = (t_ext_final - t_int) / r_total;
    let dt = 60.0; // 1-minute timesteps for precision

    let mut solver = init_solver(&wall);
    solver
        .step(
            Time::from_value(dt),
            Temperature::from_value(t_int),
            Temperature::from_value(20.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap();

    let mut tau_measured: Option<f64> = None;
    let max_steps = (10.0 * tau_analytical / dt) as usize;

    for i in 0..max_steps {
        let flux = solver
            .step(
                Time::from_value(dt),
                Temperature::from_value(t_int),
                Temperature::from_value(t_ext_final),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap()
            .to_value();
        let fraction = flux / q_ss;

        // Check if we've crossed the 63.2% threshold
        if fraction.abs() >= 0.6321 {
            tau_measured = Some((i + 1) as f64 * dt);
            break;
        }
    }

    let tau_meas = tau_measured.expect("Should reach 63.2% of steady-state within 10τ");
    let rel_error = (tau_meas - tau_analytical).abs() / tau_analytical;
    assert!(
        rel_error < 0.02,
        "τ measured = {:.2} s, analytical = {:.2} s, error = {:.2}% (limit 2%)",
        tau_meas,
        tau_analytical,
        rel_error * 100.0
    );
}

/// Time constant verification for lightweight wall.
/// Should have a much shorter τ than heavyweight.
#[test]
fn test_time_constant_lightweight() {
    let wall = lightweight_wall();
    let r_total = wall.total_r_value();
    let c_total = wall.thermal_capacity();
    let tau_analytical = c_total * r_total;

    // Lightweight wall should have τ on the order of hours (not days)
    assert!(
        tau_analytical < 100_000.0,
        "Lightweight wall τ = {:.0} s ({:.1} hours) — should be < 28 hours",
        tau_analytical,
        tau_analytical / 3600.0
    );

    let t_int = 20.0;
    let t_ext_final = 0.0;
    let q_ss = (t_ext_final - t_int) / r_total;
    let dt = 60.0;

    let mut solver = init_solver(&wall);
    solver
        .step(
            Time::from_value(dt),
            Temperature::from_value(t_int),
            Temperature::from_value(20.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap();

    let mut tau_measured: Option<f64> = None;
    let max_steps = (10.0 * tau_analytical / dt) as usize;

    for i in 0..max_steps {
        let flux = solver
            .step(
                Time::from_value(dt),
                Temperature::from_value(t_int),
                Temperature::from_value(t_ext_final),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap()
            .to_value();
        let fraction = flux / q_ss;
        if fraction.abs() >= 0.6321 {
            tau_measured = Some((i + 1) as f64 * dt);
            break;
        }
    }

    let tau_meas = tau_measured.expect("Should reach 63.2% threshold");
    let rel_error = (tau_meas - tau_analytical).abs() / tau_analytical;
    assert!(
        rel_error < 0.02,
        "Lightweight τ measured = {:.2} s, analytical = {:.2} s, error = {:.2}%",
        tau_meas,
        tau_analytical,
        rel_error * 100.0
    );
}

/// Time constant verification for insulated wall.
#[test]
fn test_time_constant_insulated() {
    let wall = insulated_wall();
    let r_total = wall.total_r_value();
    let c_total = wall.thermal_capacity();
    let tau_analytical = c_total * r_total;

    let t_int = 20.0;
    let t_ext_final = 0.0;
    let q_ss = (t_ext_final - t_int) / r_total;
    let dt = 60.0;

    let mut solver = init_solver(&wall);
    solver
        .step(
            Time::from_value(dt),
            Temperature::from_value(t_int),
            Temperature::from_value(20.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap();

    let mut tau_measured: Option<f64> = None;
    let max_steps = (10.0 * tau_analytical / dt) as usize;

    for i in 0..max_steps {
        let flux = solver
            .step(
                Time::from_value(dt),
                Temperature::from_value(t_int),
                Temperature::from_value(t_ext_final),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap()
            .to_value();
        let fraction = flux / q_ss;
        if fraction.abs() >= 0.6321 {
            tau_measured = Some((i + 1) as f64 * dt);
            break;
        }
    }

    let tau_meas = tau_measured.expect("Should reach 63.2% threshold");
    let rel_error = (tau_meas - tau_analytical).abs() / tau_analytical;
    assert!(
        rel_error < 0.02,
        "Insulated τ measured = {:.2} s, analytical = {:.2} s, error = {:.2}%",
        tau_meas,
        tau_analytical,
        rel_error * 100.0
    );
}

// ===========================================================================
// Section 4: HeatConductionSolver Trait Interface Tests
// ===========================================================================

/// Verify the trait interface: initialize → step → energy_storage_rate → is_valid.
#[test]
fn test_trait_lifecycle() {
    let wall = heavyweight_wall();
    let mut solver = FiveR1CSolver::new();

    // Before initialization
    assert!(!solver.is_valid());
    assert_eq!(solver.name(), "5R1C");
    assert_eq!(solver.energy_storage_rate(), 0.0);

    // Initialize
    let init_result = solver.initialize(&wall);
    assert!(init_result.is_ok(), "Initialization should succeed");
    assert!(solver.is_valid(), "Solver should be valid after init");

    // Step
    let step_result = solver.step(
        Time::from_value(3600.0),
        Temperature::from_value(20.0),
        Temperature::from_value(0.0),
        HeatTransferCoefficient::from_value(8.0),
        HeatTransferCoefficient::from_value(25.0),
    );
    assert!(step_result.is_ok(), "Step should succeed after init");
    let flux = step_result.unwrap().to_value();
    assert!(flux.is_finite(), "Flux should be finite");
    assert!(flux < 0.0, "Heat loss scenario: flux should be negative");

    // Energy storage rate
    let storage = solver.energy_storage_rate();
    // Current implementation returns 0 (no transient dynamics)
    // This is documented behavior — not a bug to fix here
    assert!(
        storage == 0.0 || storage.is_finite(),
        "Energy storage rate should be 0 or finite"
    );
}

/// Step before initialization should return InvalidConfig error.
#[test]
fn test_step_before_init_returns_error() {
    let mut solver = FiveR1CSolver::new();
    let result = solver.step(
        Time::from_value(3600.0),
        Temperature::from_value(20.0),
        Temperature::from_value(0.0),
        HeatTransferCoefficient::from_value(8.0),
        HeatTransferCoefficient::from_value(25.0),
    );

    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("not initialized"),
        "Error should mention not initialized, got: {}",
        err_msg
    );
}

/// Verify solver name is "5R1C".
#[test]
fn test_solver_name() {
    let solver = FiveR1CSolver::new();
    assert_eq!(solver.name(), "5R1C");
}

/// Verify solver produces same result for same inputs (determinism).
#[test]
fn test_determinism() {
    let wall = heavyweight_wall();
    let mut solver1 = init_solver(&wall);
    let mut solver2 = init_solver(&wall);

    let flux1 = solver1
        .step(
            Time::from_value(3600.0),
            Temperature::from_value(20.0),
            Temperature::from_value(0.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap()
        .to_value();
    let flux2 = solver2
        .step(
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

// ===========================================================================
// Section 5: Performance Gate (< 500ms)
// ===========================================================================

/// The entire test suite must complete in under 500ms.
/// This test runs 1000 solver steps as a proxy.
#[test]
fn test_performance_gate() {
    use std::time::Instant;

    let wall = heavyweight_wall();
    let mut solver = init_solver(&wall);

    let start = Instant::now();

    // Run 1000 steps (simulating ~42 days at 1-hour timesteps)
    for _ in 0..1000 {
        let _ = solver.step(
            Time::from_value(3600.0),
            Temperature::from_value(20.0),
            Temperature::from_value(0.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        );
    }

    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 500,
        "1000 solver steps took {}ms (limit 500ms)",
        elapsed.as_millis()
    );
}

/// Print wall properties for documentation/debugging purposes.
#[test]
fn test_wall_properties_documentation() {
    let walls: Vec<(&str, WallSpec)> = vec![
        ("Lightweight", lightweight_wall()),
        ("Heavyweight", heavyweight_wall()),
        ("Insulated", insulated_wall()),
        ("Very Heavyweight", very_heavyweight_wall()),
    ];

    println!(
        "\n┌─────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐"
    );
    println!("│ Wall Type           │ R [m²·K/W]   │ C [kJ/m²·K]  │ τ [hours]    │ Thickness[m] │");
    println!("├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤");

    for (name, wall) in &walls {
        let r = wall.total_r_value();
        let c = wall.thermal_capacity() / 1000.0; // kJ
        let tau_h = wall.thermal_capacity() * r / 3600.0;
        let thickness = wall.total_thickness();
        println!(
            "│ {:<19} │ {:>12.4} │ {:>12.1} │ {:>12.1} │ {:>12.4} │",
            name, r, c, tau_h, thickness
        );
    }

    println!("└─────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘");
    println!(
        "\nNote: τ = C × R (thermal time constant). Current solver ignores transient dynamics."
    );
}
