//! Integration tests for physics error handling
//!
//! These tests verify that physics functions return proper errors instead of
//! panicking when given invalid inputs or in invalid states.
//!
//! # Test Coverage
//!
//! - Solver step before initialization returns error
//! - Error propagation through SolverManager
//! - All solver types (FD, CTF, 5R1C) return errors appropriately

use fluxion::physics::ctf_solver_wrapper::CTFSolverWrapper;
use fluxion::physics::fd_solver_wrapper::FDSolverWrapper;
use fluxion::physics::method_selector::ThermalMethodSelector;
use fluxion::physics::solver_manager::SolverManager;
use fluxion::physics::solver_trait::HeatConductionSolver;
use fluxion::physics::units::{FromF64, HeatTransferCoefficient, Temperature, Time, ToF64};
use fluxion::physics::wall_spec::WallSpec;
use fluxion::sim::assembly::{AssemblyBuilder, ConcreteMaterial};

#[test]
fn test_fd_solver_step_before_initialization_returns_error() {
    let mut solver = FDSolverWrapper::new();

    let result = solver.step(
        Time::from_value(3600.0),
        Temperature::from_value(20.0),
        Temperature::from_value(5.0),
        HeatTransferCoefficient::from_value(8.0),
        HeatTransferCoefficient::from_value(25.0),
    );

    assert!(
        result.is_err(),
        "FD solver should return error when step() called before initialize()"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not initialized"),
        "Error should mention initialization: {}",
        err_msg
    );
}

#[test]
fn test_ctf_solver_step_before_initialization_returns_error() {
    let mut solver = CTFSolverWrapper::new();

    let result = solver.step(
        Time::from_value(3600.0),
        Temperature::from_value(20.0),
        Temperature::from_value(5.0),
        HeatTransferCoefficient::from_value(8.0),
        HeatTransferCoefficient::from_value(25.0),
    );

    assert!(
        result.is_err(),
        "CTF solver should return error when step() called before initialize()"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not initialized"),
        "Error should mention initialization: {}",
        err_msg
    );
}

#[test]
fn test_five_r1c_solver_step_before_initialization_returns_error() {
    use fluxion::physics::five_r1c_solver::FiveR1CSolver;

    let mut solver = FiveR1CSolver::new();

    let result = solver.step(
        Time::from_value(3600.0),
        Temperature::from_value(20.0),
        Temperature::from_value(5.0),
        HeatTransferCoefficient::from_value(8.0),
        HeatTransferCoefficient::from_value(25.0),
    );

    assert!(
        result.is_err(),
        "5R1C solver should return error when step() called before initialize()"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not initialized") || err_msg.contains("No solver"),
        "Error should mention initialization or solver: {}",
        err_msg
    );
}

#[test]
fn test_fd_solver_with_valid_wall_succeeds() {
    let wall = AssemblyBuilder::new("Test Wall".to_string())
        .add_layer(Box::new(ConcreteMaterial::new(0.2)))
        .build()
        .unwrap();

    let wall_spec = WallSpec::from_assembly(&wall);
    let mut solver = FDSolverWrapper::new();

    let init_result = solver.initialize(&wall_spec);
    assert!(
        init_result.is_ok(),
        "FD solver should initialize successfully with valid wall"
    );

    let step_result = solver.step(
        Time::from_value(3600.0),
        Temperature::from_value(20.0),
        Temperature::from_value(5.0),
        HeatTransferCoefficient::from_value(8.0),
        HeatTransferCoefficient::from_value(25.0),
    );

    assert!(
        step_result.is_ok(),
        "FD solver step should succeed after initialization"
    );
    let flux = step_result.unwrap();
    assert!(
        flux.to_value().is_finite(),
        "Flux should be finite, got {}",
        flux.to_value()
    );
}

#[test]
fn test_ctf_solver_with_valid_wall_succeeds() {
    let wall = AssemblyBuilder::new("Heavy Wall".to_string())
        .add_layer(Box::new(ConcreteMaterial::new(0.3)))
        .build()
        .unwrap();

    let wall_spec = WallSpec::from_assembly(&wall);
    let mut solver = CTFSolverWrapper::new();

    let init_result = solver.initialize(&wall_spec);
    assert!(
        init_result.is_ok(),
        "CTF solver should initialize successfully with valid wall"
    );

    let step_result = solver.step(
        Time::from_value(3600.0),
        Temperature::from_value(20.0),
        Temperature::from_value(5.0),
        HeatTransferCoefficient::from_value(8.0),
        HeatTransferCoefficient::from_value(25.0),
    );

    assert!(
        step_result.is_ok(),
        "CTF solver step should succeed after initialization"
    );
    let flux = step_result.unwrap();
    assert!(
        flux.to_value().is_finite(),
        "Flux should be finite, got {}",
        flux.to_value()
    );
}

#[test]
fn test_solver_manager_step_all_error_propagation() {
    let mut manager = SolverManager::new(ThermalMethodSelector::default());

    let wall = AssemblyBuilder::new("Test Wall".to_string())
        .add_layer(Box::new(ConcreteMaterial::new(0.1)))
        .build()
        .unwrap();

    let surfaces = [(0, wall)];
    let result = manager.step_all(&surfaces, 3600.0, 20.0, 5.0);

    assert!(result.is_ok(), "step_all should succeed with valid inputs");
    let fluxes = result.unwrap();
    assert_eq!(fluxes.len(), 1, "Should return one flux value");
    assert!(
        fluxes[0].is_finite(),
        "Flux should be finite, got {}",
        fluxes[0]
    );
}

#[test]
fn test_fd_solver_stays_valid_after_successful_step() {
    let wall = AssemblyBuilder::new("Test Wall".to_string())
        .add_layer(Box::new(ConcreteMaterial::new(0.2)))
        .build()
        .unwrap();

    let wall_spec = WallSpec::from_assembly(&wall);
    let mut solver = FDSolverWrapper::new();

    solver
        .initialize(&wall_spec)
        .expect("Initialization should succeed");

    assert!(
        solver.is_valid(),
        "Solver should be valid after initialization"
    );

    let step_result = solver.step(
        Time::from_value(3600.0),
        Temperature::from_value(20.0),
        Temperature::from_value(5.0),
        HeatTransferCoefficient::from_value(8.0),
        HeatTransferCoefficient::from_value(25.0),
    );

    assert!(step_result.is_ok(), "First step should succeed");
    assert!(
        solver.is_valid(),
        "Solver should still be valid after successful step"
    );
}

#[test]
fn test_all_solver_types_return_errors_before_initialization() {
    let mut fd_solver = FDSolverWrapper::new();
    let mut ctf_solver = CTFSolverWrapper::new();
    use fluxion::physics::five_r1c_solver::FiveR1CSolver;
    let mut five_r1c_solver = FiveR1CSolver::new();

    let fd_result = fd_solver.step(
        Time::from_value(3600.0),
        Temperature::from_value(20.0),
        Temperature::from_value(5.0),
        HeatTransferCoefficient::from_value(8.0),
        HeatTransferCoefficient::from_value(25.0),
    );

    let ctf_result = ctf_solver.step(
        Time::from_value(3600.0),
        Temperature::from_value(20.0),
        Temperature::from_value(5.0),
        HeatTransferCoefficient::from_value(8.0),
        HeatTransferCoefficient::from_value(25.0),
    );

    let five_r1c_result = five_r1c_solver.step(
        Time::from_value(3600.0),
        Temperature::from_value(20.0),
        Temperature::from_value(5.0),
        HeatTransferCoefficient::from_value(8.0),
        HeatTransferCoefficient::from_value(25.0),
    );

    assert!(
        fd_result.is_err(),
        "FD solver should return error before initialization"
    );
    assert!(
        ctf_result.is_err(),
        "CTF solver should return error before initialization"
    );
    assert!(
        five_r1c_result.is_err(),
        "5R1C solver should return error before initialization"
    );
}
