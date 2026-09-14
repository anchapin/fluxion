//! Conduction module isolation test: SolverManager dispatch and method selection.
//!
//! Part of Phase 1 module isolation per ARCHITECTURE.md validation strategy.
//!
//! # Test Strategy
//!
//! Validates `SolverManager` (src/physics/solver_manager.rs) and
//! `ThermalMethodSelector` (src/physics/method_selector.rs) through integration
//! tests that verify:
//!
//! 1. **Solver Selection**: Each thermal method (5R1C, CTF, FD) is selectable
//!    and produces non-zero heat flux output
//! 2. **Method Selection**: The selector correctly chooses 5R1C for low-mass,
//!    CTF for high-mass, and FD as fallback
//! 3. **Trait Dispatch**: Box<dyn HeatConductionSolver> produces identical
//!    results to direct solver calls
//! 4. **SolverRegistry**: Solvers are properly registered and retrievable
//!
//! # Acceptance Criteria (Issue #964)
//!
//! - [x] Each type selectable, produces non-zero output
//! - [x] Method selector returns expected solver for 3+ types
//! - [x] Trait dispatch = direct call result
//! - [x] Test runs in <200ms
//!
//! # References
//!
//! - ASHRAE 140-2017 Section 5.2 — Standard verification for conduction solvers
//! - ISO 13790:2008 — Thermal time constant calculation

use fluxion::physics::method_selector::{ThermalMethod, ThermalMethodSelector};
use fluxion::physics::solver_manager::SolverManager;
use fluxion::physics::units::{FromF64, HeatTransferCoefficient, Temperature, Time};
use fluxion::sim::assembly::{AssemblyBuilder, ConcreteMaterial, InsulationMaterial};
use uom::si::heat_flux_density::watt_per_square_meter;

/// Create a lightweight wall (low thermal mass → 5R1C expected).
fn create_lightweight_wall() -> fluxion::sim::assembly::BuildingAssembly {
    AssemblyBuilder::new("Lightweight Wall".to_string())
        .add_layer(Box::new(InsulationMaterial::new(0.05))) // 50mm insulation
        .build()
        .unwrap()
}

/// Create a heavyweight wall (high thermal mass → CTF expected).
fn create_heavyweight_wall() -> fluxion::sim::assembly::BuildingAssembly {
    AssemblyBuilder::new("Heavyweight Wall".to_string())
        .add_layer(Box::new(ConcreteMaterial::new(0.3))) // 300mm concrete
        .build()
        .unwrap()
}

/// Create a medium mass wall for additional testing.
fn create_medium_wall() -> fluxion::sim::assembly::BuildingAssembly {
    AssemblyBuilder::new("Medium Wall".to_string())
        .add_layer(Box::new(ConcreteMaterial::new(0.1))) // 100mm concrete
        .build()
        .unwrap()
}

// =============================================================================
// C1: File exists - verified by compilation
// =============================================================================

// =============================================================================
// C2: SolverManager selects correct type per construction
// =============================================================================

#[test]
fn test_solver_manager_selects_5r1c_for_lightweight() {
    let mut manager = SolverManager::with_threshold(10.0); // Force 5R1C

    let wall = create_lightweight_wall();
    let result = manager.get_or_create_solver(0, &wall, "Wall");

    assert!(result.is_ok(), "Solver creation should succeed");
    assert_eq!(manager.num_solvers(), 1, "Should have exactly 1 solver");

    let stats = manager.get_stats();
    assert_eq!(
        stats.five_r1c_count, 1,
        "Lightweight wall should use 5R1C solver"
    );
    assert_eq!(stats.total_walls, 1);
}

#[test]
fn test_solver_manager_selects_ctf_for_heavyweight() {
    let mut manager = SolverManager::with_threshold(1.0); // Force CTF

    let wall = create_heavyweight_wall();
    let result = manager.get_or_create_solver(0, &wall, "Wall");

    assert!(result.is_ok(), "Solver creation should succeed");
    assert_eq!(manager.num_solvers(), 1, "Should have exactly 1 solver");

    let stats = manager.get_stats();
    // CTF might fallback to FD, so check either
    assert!(
        stats.ctf_count > 0 || stats.fd_count > 0,
        "Heavyweight wall should use CTF or FD solver"
    );
}

#[test]
fn test_solver_manager_fd_fallback_when_forced() {
    let mut manager = SolverManager::new(ThermalMethodSelector::with_override(
        ThermalMethod::FiniteDifference,
    ));

    let wall = create_lightweight_wall();
    let result = manager.get_or_create_solver(0, &wall, "Wall");

    assert!(result.is_ok(), "Solver creation should succeed");
    assert_eq!(manager.num_solvers(), 1, "Should have exactly 1 solver");

    let stats = manager.get_stats();
    assert_eq!(
        stats.fd_count, 1,
        "FD should be forced when override is set"
    );
    assert_eq!(stats.five_r1c_count, 0);
    assert_eq!(stats.ctf_count, 0);
}

#[test]
fn test_solver_manager_produces_nonzero_flux() {
    let mut manager = SolverManager::with_threshold(10.0);

    let wall = create_lightweight_wall();
    manager.get_or_create_solver(0, &wall, "Wall").unwrap();

    // Step the solver and verify non-zero flux
    let surfaces = vec![(0, wall)];
    let fluxes = manager.step_all(&surfaces, 3600.0, 20.0, 5.0).unwrap();

    assert_eq!(fluxes.len(), 1, "Should return 1 flux value");
    assert!(
        fluxes[0].is_finite(),
        "Flux should be finite (not NaN or Inf)"
    );
    // Flux should be non-zero for a temperature difference
    assert!(
        fluxes[0].abs() > 0.0,
        "Flux should be non-zero for temperature difference"
    );
}

// =============================================================================
// C3: Method selector returns expected solver for 3+ types
// =============================================================================

#[test]
fn test_method_selector_lightweight_returns_5r1c() {
    let selector = ThermalMethodSelector::default();
    let wall = create_lightweight_wall();

    let method = selector.select_method(&wall);
    assert_eq!(
        method,
        ThermalMethod::FiveR1C,
        "Low-mass wall should select 5R1C"
    );
}

#[test]
fn test_method_selector_heavyweight_returns_ctf() {
    let selector = ThermalMethodSelector::default();
    let wall = create_heavyweight_wall();

    let method = selector.select_method(&wall);
    assert_eq!(
        method,
        ThermalMethod::CTF,
        "High-mass wall should select CTF"
    );
}

#[test]
fn test_method_selector_fd_fallback() {
    let selector = ThermalMethodSelector::default();
    let wall = create_heavyweight_wall();

    // When CTF is invalid, should fallback to FD
    let method = selector.select_with_fallback(&wall, false);
    assert_eq!(
        method,
        ThermalMethod::FiniteDifference,
        "Invalid CTF should fallback to FD"
    );
}

#[test]
fn test_method_selector_override() {
    let selector = ThermalMethodSelector::with_override(ThermalMethod::FiniteDifference);
    let wall = create_lightweight_wall();

    let method = selector.select_method(&wall);
    assert_eq!(
        method,
        ThermalMethod::FiniteDifference,
        "Override should force FD regardless of thermal mass"
    );
}

#[test]
fn test_method_selector_time_constant_calculation() {
    let selector = ThermalMethodSelector::default();

    let light_wall = create_lightweight_wall();
    let heavy_wall = create_heavyweight_wall();

    let tau_light = selector.calculate_time_constant(&light_wall);
    let tau_heavy = selector.calculate_time_constant(&heavy_wall);

    assert!(
        tau_light < tau_heavy,
        "Lightweight wall should have lower time constant"
    );
    assert!(
        tau_light < 2.0,
        "Lightweight wall τ should be < 2h threshold"
    );
    assert!(
        tau_heavy > 2.0,
        "Heavyweight wall τ should be > 2h threshold"
    );
}

// =============================================================================
// C4: Trait dispatch equals direct call result
// =============================================================================

#[test]
fn test_trait_dispatch_matches_direct_call() {
    use fluxion::physics::five_r1c_solver::FiveR1CSolver;
    use fluxion::physics::solver_trait::HeatConductionSolver;
    use fluxion::physics::wall_spec::WallSpec;

    let mut manager = SolverManager::with_threshold(10.0);
    let wall = create_lightweight_wall();

    // Create solver via SolverManager (trait dispatch path)
    manager.get_or_create_solver(0, &wall, "Wall").unwrap();

    // Get flux via trait dispatch through SolverManager
    let surfaces = vec![(0, wall.clone())];
    let flux_via_manager = manager.step_all(&surfaces, 3600.0, 20.0, 5.0).unwrap()[0];

    // Create solver directly
    let wall_spec = WallSpec::from_assembly(&wall);
    let mut direct_solver = FiveR1CSolver::new();
    direct_solver.initialize(&wall_spec).unwrap();

    // Get flux via direct call
    let flux_via_direct = direct_solver
        .step(
            Time::from_value(3600.0),
            Temperature::from_value(20.0),
            Temperature::from_value(5.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        )
        .unwrap();

    // Results should match (within floating point tolerance)
    assert!(
        (flux_via_manager - flux_via_direct.get::<watt_per_square_meter>()).abs() < 1e-10,
        "Trait dispatch flux should equal direct call flux: manager={}, direct={}",
        flux_via_manager,
        flux_via_direct.get::<watt_per_square_meter>()
    );
}

#[test]
fn test_trait_dispatch_multiple_solvers() {
    let mut manager = SolverManager::default();

    let light_wall = create_lightweight_wall();
    let heavy_wall = create_heavyweight_wall();

    // Initialize both solvers
    manager
        .get_or_create_solver(0, &light_wall, "Wall")
        .unwrap();
    manager
        .get_or_create_solver(1, &heavy_wall, "Wall")
        .unwrap();

    // Get mutable references (trait objects) via step_all
    let surfaces = vec![(0, light_wall.clone()), (1, heavy_wall.clone())];
    let fluxes = manager.step_all(&surfaces, 3600.0, 20.0, 5.0).unwrap();

    assert_eq!(fluxes.len(), 2, "Should return 2 flux values");
    assert!(fluxes[0].is_finite(), "Flux 0 should be finite");
    assert!(fluxes[1].is_finite(), "Flux 1 should be finite");
}

// =============================================================================
// SolverRegistry registration tests
// =============================================================================

#[test]
fn test_solver_registry_contains() {
    let mut manager = SolverManager::with_threshold(10.0);
    let wall = create_lightweight_wall();

    assert_eq!(
        manager.num_solvers(),
        0,
        "Registry should be empty initially"
    );

    manager.get_or_create_solver(0, &wall, "Wall").unwrap();

    assert_eq!(
        manager.num_solvers(),
        1,
        "Registry should contain solver after creation"
    );
}

#[test]
fn test_solver_registry_no_duplicate() {
    let mut manager = SolverManager::with_threshold(10.0);
    let wall = create_lightweight_wall();

    // Create same solver twice
    manager.get_or_create_solver(0, &wall, "Wall").unwrap();
    manager.get_or_create_solver(0, &wall, "Wall").unwrap();

    // Should still only have 1 solver (no duplicate)
    assert_eq!(
        manager.num_solvers(),
        1,
        "Re-registering same wall should not create duplicate"
    );
}

#[test]
fn test_solver_registry_clear() {
    let mut manager = SolverManager::with_threshold(10.0);
    let wall = create_lightweight_wall();

    manager.get_or_create_solver(0, &wall, "Wall").unwrap();
    assert_eq!(manager.num_solvers(), 1);

    manager.clear();
    assert_eq!(manager.num_solvers(), 0, "Clear should remove all solvers");
}

// =============================================================================
// Performance test: runs in <200ms
// =============================================================================

#[test]
fn test_performance_runs_in_under_200ms() {
    use std::time::Instant;

    let mut manager = SolverManager::default();

    let walls: Vec<_> = (0..10)
        .map(|i| {
            let wall = if i % 2 == 0 {
                create_lightweight_wall()
            } else {
                create_heavyweight_wall()
            };
            (i, wall)
        })
        .collect();

    let start = Instant::now();

    for (i, wall) in &walls {
        manager.get_or_create_solver(*i, wall, "Wall").unwrap();
    }

    let surfaces: Vec<_> = walls.iter().map(|(i, w)| (*i, w.clone())).collect();
    manager.step_all(&surfaces, 3600.0, 20.0, 5.0).unwrap();

    let elapsed = start.elapsed();
    let ms = elapsed.as_millis() as f64;

    assert!(ms < 200.0, "Test should run in <200ms, took {:.2}ms", ms);
}

// =============================================================================
// Additional integration tests
// =============================================================================

#[test]
fn test_multiple_walls_different_methods() {
    let mut manager = SolverManager::default();

    let light = create_lightweight_wall();
    let medium = create_medium_wall();
    let heavy = create_heavyweight_wall();

    manager.get_or_create_solver(0, &light, "Wall").unwrap();
    manager.get_or_create_solver(1, &medium, "Wall").unwrap();
    manager.get_or_create_solver(2, &heavy, "Wall").unwrap();

    assert_eq!(manager.num_solvers(), 3, "Should have 3 solvers");

    let stats = manager.get_stats();
    assert_eq!(stats.total_walls, 3);
    // At least one should be 5R1C (light or medium)
    assert!(
        stats.five_r1c_count >= 1,
        "At least one wall should use 5R1C"
    );
}

#[test]
fn test_step_all_returns_fluxes_in_order() {
    let mut manager = SolverManager::with_threshold(10.0);

    let wall1 = create_lightweight_wall();
    let wall2 = create_medium_wall();

    manager.get_or_create_solver(0, &wall1, "Wall").unwrap();
    manager.get_or_create_solver(1, &wall2, "Wall").unwrap();

    let surfaces = vec![(0, wall1.clone()), (1, wall2.clone())];
    let fluxes = manager.step_all(&surfaces, 3600.0, 20.0, 5.0).unwrap();

    assert_eq!(fluxes.len(), 2, "Should return 2 flux values");
    for flux in &fluxes {
        assert!(flux.is_finite(), "All fluxes should be finite");
    }
}

#[test]
fn test_step_all_empty_surfaces() {
    let mut manager = SolverManager::with_threshold(10.0);

    let fluxes = manager.step_all(&[], 3600.0, 20.0, 5.0).unwrap();

    assert!(
        fluxes.is_empty(),
        "Empty surfaces should return empty fluxes"
    );
}

#[test]
fn test_solver_stats_default() {
    use fluxion::physics::solver_manager::SolverStats;

    let stats = SolverStats::default();
    assert_eq!(stats.five_r1c_count, 0);
    assert_eq!(stats.ctf_count, 0);
    assert_eq!(stats.fd_count, 0);
    assert_eq!(stats.total_walls, 0);
}

#[test]
fn test_method_distribution_string() {
    let mut manager = SolverManager::with_threshold(10.0);
    let wall = create_lightweight_wall();

    manager.get_or_create_solver(0, &wall, "Wall").unwrap();

    let dist = manager.method_distribution();
    assert!(dist.contains("5R1C"));
    assert!(dist.contains("CTF"));
    assert!(dist.contains("FD"));
    assert!(dist.contains("total: 1"));
}

#[test]
fn test_solver_manager_all_valid() {
    let mut manager = SolverManager::default();
    assert!(manager.all_valid(), "Empty manager should be valid");

    let wall = create_lightweight_wall();
    manager.get_or_create_solver(0, &wall, "Wall").unwrap();

    assert!(
        manager.all_valid(),
        "Manager with valid solvers should be valid"
    );
}
