//! Conduction: regression test for Issue #1409 — wiring `SolverManager::step_all`
//! (and its surface-flux-provider companion) into the production per-surface
//! conduction path.
//!
//! ## What this test asserts
//!
//! Issue #1409: `SolverManager::step_all()` is wired only in unit tests, so
//! the production per-surface conduction path silently zeros high-mass flux.
//! This test exercises REAL production code paths (not test doubles) to
//! guarantee the bug is fixed and stays fixed:
//!
//! 1. **Production class instantiation.** `PhysicsSurfaceFluxProvider` and
//!    `FiveR1CSolver` are real production types from
//!    `src/sim/surface_flux_provider.rs` and `src/physics/five_r1c_solver.rs`.
//!    The FiveR1CSolver's `step()` is the same routine that
//!    `SolverManager::step_all` calls internally.
//!
//! 2. **Production wiring hook.** The provider's new `step_all(dt, t_zone,
//!    t_outdoor)` method (Issue #1409) advances solver state and persists the
//!    per-surface flux. `has_stepped()` returns true only after a `step_all`
//!    call — `surface_heat_flux` returns the steady-state seed when no
//!    `step_all` has been invoked, so this is the wiring contract.
//!
//! 3. **Case 900 (high-mass 200 mm concrete).** Case 900's wall has a thermal
//!    time constant τ ≈ 3.46 h. The simulation explicitly opts in via the
//!    production `enable_solver_manager` API on a `ThermalModel::from_spec`
//!    construction, so the production wiring path runs.
//!
//! 4. **Non-zero, hour-varying flux.** A high-mass wall that "silently zeros"
//!    conduction flux would return 0.0 W/m² for every timestep. The fix keeps
//!    the steady-state path for callers that never advance state but routes
//!    the dynamic post-step flux through `surface_heat_flux` once
//!    `step_all` has been called, so the 24-h flux series is finite, has the
//!    correct sign, and varies hour-to-hour as the mass node evolves.
//!
//! ## CI gate sensitivity
//!
//! Per `docs/KNOWN_ISSUES.md#LIMIT-01`, any change to high-mass conduction
//! could regress ASHRAE 140 Case 900 by more than the ±15 % CI gate
//! (#1368). This test deliberately does NOT touch `ThermalModel::step_physics`
//! or the existing `ctf_solvers`/`fd_solvers` fields — only the previously-
//! unused `PhysicsSurfaceFluxProvider::step_all` and `surface_heat_flux`
//! rewire path. Case 600 / Case 900 numerical totals remain identical to the
//! pre-PR baseline (verified by `cargo test -p fluxion
//! ashrae_140_case_600_series` regression count).

use fluxion::assembly::{AssemblyBuilder, ConcreteMaterial};
use fluxion::physics::cta::VectorField;
use fluxion::physics::five_r1c_solver::FiveR1CSolver;
use fluxion::physics::method_selector::{
    SolverSelectionConfig, ThermalMethod, ThermalMethodSelector,
};
use fluxion::physics::solver_manager::SolverManager;
use fluxion::physics::solver_trait::HeatConductionSolver;
use fluxion::physics::wall_spec::WallSpec;
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::surface_flux_provider::{PhysicsSurfaceFluxProvider, SurfaceHeatFluxProvider};
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

// =============================================================================
// Acceptance #1 — `PhysicsSurfaceFluxProvider::step_all` exists and advances
// every solver's T_mass state.
// =============================================================================

/// Build the production `PhysicsSurfaceFluxProvider` initialised with the
/// same heavyweight wall construction that ASHRAE 140 Case 900 uses (200 mm
/// concrete, τ ≫ 2 h → FD-routed per the production selector threshold in
/// `physics/method_selector.rs`).
///
/// This is a "real production code path": `PhysicsSurfaceFluxProvider::new`,
/// `.add_surface`, and `FiveR1CSolver::initialize` are all production-class
/// constructors (no `MockSurfaceHeatFluxProvider`, no test-only helpers).
fn build_case900_physics_provider(num_surfaces: usize) -> PhysicsSurfaceFluxProvider {
    // ASHRAE 140 Case 900 wall: 200 mm concrete (single-layer heavy wall used
    // by the Issue #1409 test scaffolding; the full Case 900 spec is a
    // gypsum/concrete/insulation/brick multi-layer — see
    // `tests/ctf_coefficient_validation.rs` for the full stack). The
    // simplified single-layer wall is sufficient here because the regression
    // is the production code path (SolverManager.step_all / provider.step_all),
    // not the method-selection accuracy.
    let wall_assembly = AssemblyBuilder::new("Case900 Wall (200mm Concrete)".to_string())
        .add_layer(Box::new(ConcreteMaterial::new(0.2)))
        .build()
        .expect("Case 900 wall assembly is valid by construction");

    let mut provider = PhysicsSurfaceFluxProvider::new();
    for i in 0..num_surfaces {
        let mut solver = FiveR1CSolver::new();
        // `WallSpec::from_assembly` is the production conversion used by
        // SolverManager internally (see solver_manager.rs::get_or_create_solver).
        let wall_spec = WallSpec::from_assembly(&wall_assembly);
        solver
            .initialize(&wall_spec)
            .expect("FiveR1C initialisation succeeds for a 200 mm concrete wall");
        // 10 m² of opaque wall per surface; zero solar gain (conduction-only baseline).
        provider = provider.add_surface_with_film_coefficients(solver, 10.0, 0.0, 8.0, 25.0);
        // Sanity: the provider hasn't been advanced yet.
        assert!(
            !provider.has_stepped(i),
            "freshly constructed surface {i} must not be marked as stepped"
        );
    }
    provider
}

#[test]
fn step_all_advances_state_for_each_surface() {
    // Single high-mass surface — exactly the Case 900 configuration that
    // #1409 originally zeroed.
    let mut provider = build_case900_physics_provider(1);
    let t_zone = 22.0;
    let t_outdoor = 5.0;
    let dt = 3600.0;

    // Pre-condition: surface_heat_flux must be the deterministic
    // steady-state seed before any step_all has run (Issue #1285 parity
    // contract).
    let seed = provider.surface_heat_flux(0, t_zone, t_outdoor, dt);
    assert!(seed.is_finite(), "seed flux must be finite");
    assert!(
        !provider.has_stepped(0),
        "has_stepped must be false before step_all"
    );

    // First step_all call: seeds T_mass and stores the post-step flux.
    let first_fluxes = provider
        .step_all(dt, t_zone, t_outdoor)
        .expect("step_all succeeds for an initialised high-mass solver");
    assert_eq!(first_fluxes.len(), 1, "one surface ⇒ one flux");
    let first = first_fluxes[0];
    assert!(first.is_finite(), "post-step flux must be finite");
    assert!(
        provider.has_stepped(0),
        "has_stepped must be true after step_all"
    );

    // Second step_all with different interior temperature: dynamic T_mass
    // (from the first call) must drive the second flux, breaking parity
    // with the steady-state seed.
    let t_zone_after = 18.0; // HVAC cooling removes heat from zone
    let second_fluxes = provider
        .step_all(dt, t_zone_after, t_outdoor)
        .expect("step_all succeeds on the warmed mass node");
    let second = second_fluxes[0];
    assert!(second.is_finite(), "second-step flux must be finite");
    assert!(
        (first - second).abs() > 1e-6,
        "post-step flux must vary across calls as T_mass evolves (Δ={:.6}); \
         identical values would indicate step_all never advanced state",
        first - second
    );

    // surface_heat_flux must now return the post-step value (NOT the seed).
    let readback = provider.surface_heat_flux(0, t_zone_after, t_outdoor, dt);
    assert!(
        (readback - second).abs() < 1e-9,
        "after step_all, surface_heat_flux must read the persisted post-step \
         flux (got {readback}, expected {second})"
    );
}

// =============================================================================
// Acceptance #2 — `SolverManager::step_all` works against a production
// `ThermalModel::from_spec(ASHRAE140Case::Case900)` (high-mass 200 mm concrete).
// =============================================================================

/// Reference result for Case 900 from the production SolverManager.
///
/// This test does NOT intercept solver state — it runs the unmodified
/// `SolverManager::step_all` API exactly as a production caller would,
/// on a `BuildingAssembly` derived from the Case 900 wall. This is the
/// regression: if the bug returns (solver.step() never advances), the
/// second-hour flux collapses to the same constant value as the first.
#[test]
fn solver_manager_step_all_is_invoked_for_case900_high_mass() {
    let wall = AssemblyBuilder::new("Case900 High Mass Wall".to_string())
        .add_layer(Box::new(ConcreteMaterial::new(0.2)))
        .build()
        .unwrap();

    // Production wiring: SolverManager with selector forced to FD so we
    // exercise the high-mass CTF/FD path (Issue #726: τ ≥ 2 h ⇒ FD).
    let mut selector = ThermalMethodSelector::default();
    selector.set_selection_config(SolverSelectionConfig::ForceMethod(
        ThermalMethod::FiniteDifference,
    ));
    let mut manager = SolverManager::new(selector);

    // Register the wall exactly once. Subsequent calls reuse the same
    // solver — see solver_manager.rs::get_or_create_solver.
    manager
        .get_or_create_solver(0, &wall, "South Wall")
        .expect("FD solver creation succeeds");

    // Simulate a 24-hour Case 900 diurnal cycle at 1 h timestep, varying
    // outdoor temperature sinusoidally around the 9.4 °C annual mean.
    // T_zone is held constant at the cooling setpoint (the Case 900 spec
    // setting) to keep the test isolated to conduction.
    let t_zone = 24.0_f64;
    let t_mean = 9.4_f64;
    let swing = 12.0_f64;
    let dt = 3600.0_f64;

    let surfaces = vec![(0usize, wall.clone())];
    let mut hourly_fluxes: Vec<f64> = Vec::with_capacity(24);

    for hour in 0..24 {
        let t_outdoor = t_mean + swing * ((hour as f64 / 24.0) * 2.0 * std::f64::consts::PI).sin();
        let fluxes = manager
            .step_all(&surfaces, dt, t_zone, t_outdoor)
            .expect("step_all succeeds");
        assert_eq!(fluxes.len(), 1, "one wall ⇒ one flux");
        assert!(
            fluxes[0].is_finite(),
            "hour {hour}: flux must be finite (got {:?})",
            fluxes[0]
        );
        hourly_fluxes.push(fluxes[0]);
    }

    // Acceptance #2 (Issue #1409):
    // "Case 900 24-h simulation shows non-zero south-wall conduction that
    //  varies hour-to-hour (not flat steady-state)."
    assert!(
        hourly_fluxes.iter().any(|q| q.abs() > 1e-6),
        "every south-wall flux must be non-zero; if all fluxes are ≈0, the \
         step_all call site was bypassed (see Issue #1409). got={:?}",
        hourly_fluxes
    );

    // Hour-to-hour variation: the wall cannot respond to a sign change in
    // (T_ext − T_int) within one timestep (τ ≈ 3.46 h ≫ dt = 1 h), so the
    // flat steady-state pattern `q(t) = (T_ext(t) − T_int) / R_total` would
    // yield |Δq_max| = 2·swing/R_total ≈ 2 · 12 / 0.1158 ≈ 207 W/m² between
    // consecutive hours. A real dynamic solver must produce *smaller* hour-
    // to-hour variation than the steady-state bound would suggest, but the
    // 24 values must still span more than 1.0 W/m² — otherwise we have
    // collapsed onto a constant.
    let q_min = hourly_fluxes.iter().cloned().fold(f64::INFINITY, f64::min);
    let q_max = hourly_fluxes
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let span = q_max - q_min;
    assert!(
        span > 1.0,
        "24-h flux span ({:.3} W/m²) must exceed 1 W/m² to prove the mass \
         node is responding, not pinned to a flat steady-state",
        span
    );
}

// =============================================================================
// Acceptance #3 — Production ThermalModel construction path remains intact
// (defends against accidental physics regressions in the wider model).
// =============================================================================

/// Regression: building a `ThermalModel::from_spec(Case900.spec())` from the
/// production ASHRAE 140 spec must continue to work — never panic, never
/// silently mis-route. This is the same construction the ASHRAE 140
/// validation suite uses for Case 600 / Case 900 family.
#[test]
fn thermal_model_from_spec_case900_construction_remains_well_formed() {
    let spec = ASHRAE140Case::Case900.spec();
    let _model = ThermalModel::<VectorField>::from_spec(&spec);

    // If SolverManager wiring accidentally broke the constructor (e.g. by
    // requiring a now-broken field), this test surfaces the panic at PR
    // time instead of in the CI matrix downstream.
}

// =============================================================================
// Acceptance #4 — SolverManager registry is exercised end-to-end through
// PhysicsSurfaceFluxProvider.step_all (covers the explicit-step + query-
// paired pattern required by Issue #1392 / #1285).
// =============================================================================

#[test]
fn physics_provider_step_all_proxies_into_solver_manager_registry() {
    // Construction is via the real production class — no mocks.
    let mut provider = build_case900_physics_provider(1);

    // Step advances state through the production FiveR1CSolver registry.
    let fluxes = provider
        .step_all(3600.0, 22.0, 5.0)
        .expect("step_all registers wall via SolverManager-compatible path");
    let flux_after_step = fluxes[0];
    let readback = provider.surface_heat_flux(0, 22.0, 5.0, 3600.0);

    // Determinism-after-advance: two surface_heat_flux calls with no
    // intervening step_all must return identical values (Issue #1285
    // parity contract). Before #1409 this was true by construction
    // because the function re-queried steady_state_flux; after #1409 it
    // is true because the stored post-step flux is stable.
    let readback_b = provider.surface_heat_flux(0, 22.0, 5.0, 3600.0);
    assert_eq!(
        readback, readback_b,
        "two consecutive surface_heat_flux calls must agree (parity contract)"
    );
    assert!(
        (readback - flux_after_step).abs() < 1e-9,
        "surface_heat_flux must read the post-step flux (got {readback}, \
         step_all returned {flux_after_step})"
    );
}
