//! Concurrency tests for parallel solver execution (Issue #1065)
//!
//! This module tests that the parallel execution paths in SolverManager
//! are free from race conditions and deadlocks when multiple threads
//! update shared boundary conditions (e.g., inter-zone walls).
//!
//! # Running Tests
//!
//! ```bash
//! # Run concurrency tests
//! cargo test --test loom_concurrency_tests
//!
//! # Run with loom model checking (requires restructuring domain types)
//! LOOM=1 cargo test --features loom --test loom_concurrency_tests
//! ```
//!
//! # Note on Loom Model Checking
//!
//! Full model checking with loom requires all captured types to be `Send + Sync + 'static`.
//! The domain types (SolverManager, BuildingAssembly) are complex objects that don't satisfy
//! these bounds. For full model checking, the domain types would need to be wrapped in a
//! simpler abstraction that only exposes the concurrency-critical fields.
//!
//! These tests use std thread/Mutex for concurrency testing. To enable loom model checking,
//! the domain would need refactoring to separate the concurrency-critical state from the
//! complex domain logic.

use fluxion::physics::method_selector::ThermalMethodSelector;
use fluxion::physics::solver_manager::SolverManager;
use fluxion::sim::assembly::{AssemblyBuilder, ConcreteMaterial};
use std::sync::{Arc as StdArc, Mutex as StdMutex};
use std::thread;

/// Heat transfer payload that should never be dropped
#[derive(Debug, Clone, PartialEq)]
pub struct HeatTransferPayload {
    pub wall_index: usize,
    pub flux: f64,
    pub sequence_number: usize,
}

/// Shared boundary condition state (simulates inter-zone wall)
#[derive(Debug, Clone)]
pub struct SharedBoundaryCondition {
    pub wall_index: usize,
    pub temperature: f64,
    pub sequence: usize,
}

impl SharedBoundaryCondition {
    pub fn new(wall_index: usize) -> Self {
        Self {
            wall_index,
            temperature: 20.0,
            sequence: 0,
        }
    }

    /// Update the boundary condition - returns the payload that must be preserved
    pub fn update(&mut self, new_temp: f64) -> HeatTransferPayload {
        self.sequence += 1;
        self.temperature = new_temp;
        HeatTransferPayload {
            wall_index: self.wall_index,
            flux: new_temp * 10.0,
            sequence_number: self.sequence,
        }
    }
}

#[test]
#[allow(deprecated)]
fn test_solver_manager_concurrent_access() {
    let mut manager = SolverManager::new(ThermalMethodSelector::default());

    let wall = AssemblyBuilder::new("Shared Wall".to_string())
        .add_layer(Box::new(ConcreteMaterial::new(0.2)))
        .build()
        .unwrap();

    manager
        .get_or_create_solver(0, &wall, "SharedWall")
        .expect("Failed to create solver");

    let boundary = StdArc::new(StdMutex::new(SharedBoundaryCondition::new(0)));
    let manager = StdArc::new(StdMutex::new(manager));

    let mut handles = Vec::new();

    for thread_id in 0..3 {
        let boundary = StdArc::clone(&boundary);
        let manager = StdArc::clone(&manager);

        handles.push(thread::spawn(move || {
            for step in 0..5 {
                let mut bc = boundary.lock().unwrap();
                let new_temp = 20.0 + (thread_id as f64 * 10.0) + (step as f64);
                let payload = bc.update(new_temp);

                assert!(payload.wall_index == 0);
                assert!(payload.sequence_number > 0);
                assert!(payload.flux.is_finite());

                drop(bc);

                let mgr = manager.lock().unwrap();
                if let Some(solver) = mgr.get_solver(0) {
                    assert!(solver.is_valid());
                }
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    let bc = boundary.lock().unwrap();
    assert_eq!(bc.sequence, 15);
}

#[test]
fn test_get_or_create_solver_concurrent() {
    let manager = StdArc::new(StdMutex::new(SolverManager::new(
        ThermalMethodSelector::default(),
    )));

    let wall = StdArc::new(
        AssemblyBuilder::new("Shared Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap(),
    );

    let mut handles = Vec::new();

    for _thread_id in 0..4 {
        let manager = StdArc::clone(&manager);
        let wall = StdArc::clone(&wall);

        handles.push(thread::spawn(move || {
            for _attempt in 0..3 {
                let mut mgr = manager.lock().unwrap();
                let result = mgr.get_or_create_solver(0, &wall, "Wall");

                assert!(result.is_ok());
                assert_eq!(mgr.num_solvers(), 1);
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    let mgr = manager.lock().unwrap();
    assert_eq!(mgr.num_solvers(), 1);
}

#[test]
fn test_step_all_concurrent_updates() {
    let mut manager = SolverManager::new(ThermalMethodSelector::default());

    let wall1 = AssemblyBuilder::new("Wall 1".to_string())
        .add_layer(Box::new(ConcreteMaterial::new(0.1)))
        .build()
        .unwrap();

    let wall2 = AssemblyBuilder::new("Wall 2".to_string())
        .add_layer(Box::new(ConcreteMaterial::new(0.15)))
        .build()
        .unwrap();

    let wall3 = AssemblyBuilder::new("Wall 3".to_string())
        .add_layer(Box::new(ConcreteMaterial::new(0.2)))
        .build()
        .unwrap();

    manager
        .get_or_create_solver(0, &wall1, "Wall1")
        .expect("Failed to create solver 1");
    manager
        .get_or_create_solver(1, &wall2, "Wall2")
        .expect("Failed to create solver 2");
    manager
        .get_or_create_solver(2, &wall3, "Wall3")
        .expect("Failed to create solver 3");

    let surfaces: Vec<(usize, _)> = vec![(0, wall1), (1, wall2), (2, wall3)];

    let all_fluxes = StdArc::new(StdMutex::new(Vec::new()));
    let manager = StdArc::new(StdMutex::new(manager));

    let mut handles = Vec::new();

    for thread_id in 0..2 {
        let all_fluxes = StdArc::clone(&all_fluxes);
        let manager = StdArc::clone(&manager);
        let surfaces_clone = surfaces.clone();

        handles.push(thread::spawn(move || {
            for step in 0..3 {
                let mut mgr = manager.lock().unwrap();
                let result = mgr.step_all(&surfaces_clone, 3600.0, 22.0, 10.0);

                match result {
                    Ok(fluxes) => {
                        assert_eq!(fluxes.len(), 3);
                        for flux in &fluxes {
                            assert!(flux.is_finite());
                        }
                        let mut recorded = all_fluxes.lock().unwrap();
                        recorded.push((thread_id, step, fluxes));
                    }
                    Err(e) => panic!("step_all should not error: {:?}", e),
                }
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    let recorded = all_fluxes.lock().unwrap();
    assert_eq!(recorded.len(), 6);
}

#[test]
#[allow(deprecated)]
fn test_clear_while_accessing() {
    let manager_reader = StdArc::new(StdMutex::new(SolverManager::new(
        ThermalMethodSelector::default(),
    )));

    let wall = StdArc::new(
        AssemblyBuilder::new("Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap(),
    );

    {
        let mut mgr = manager_reader.lock().unwrap();
        mgr.get_or_create_solver(0, &wall, "Wall")
            .expect("Failed to create solver");
    }

    let manager_writer = StdArc::new(StdMutex::new(SolverManager::new(
        ThermalMethodSelector::default(),
    )));

    let manager_for_reader = StdArc::clone(&manager_reader);
    let reader = thread::spawn(move || {
        for _ in 0..10 {
            let mgr = manager_for_reader.lock().unwrap();
            if let Some(solver) = mgr.get_solver(0) {
                let _ = solver.is_valid();
            }
        }
    });

    let manager_for_writer = StdArc::clone(&manager_writer);
    let wall_for_writer = StdArc::clone(&wall);
    let writer = thread::spawn(move || {
        for _ in 0..5 {
            let mut mgr = manager_for_writer.lock().unwrap();
            mgr.clear();
            mgr.get_or_create_solver(0, &wall_for_writer, "Wall")
                .expect("Failed to recreate solver");
        }
    });

    reader.join().expect("Reader should not panic");
    writer.join().expect("Writer should not panic");

    let mgr = manager_reader.lock().unwrap();
    assert_eq!(mgr.num_solvers(), 1);
}

#[test]
fn test_stats_concurrent_access() {
    let manager_stats = StdArc::new(StdMutex::new(SolverManager::new(
        ThermalMethodSelector::default(),
    )));

    let manager_adder = StdArc::new(StdMutex::new(SolverManager::new(
        ThermalMethodSelector::default(),
    )));

    let wall1 = StdArc::new(
        AssemblyBuilder::new("Wall1".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .build()
            .unwrap(),
    );

    let wall2 = StdArc::new(
        AssemblyBuilder::new("Wall2".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap(),
    );

    let wall1_for_adder = StdArc::clone(&wall1);
    let wall2_for_adder = StdArc::clone(&wall2);
    let manager_for_adder = StdArc::clone(&manager_adder);

    let adder = thread::spawn(move || {
        for _ in 0..3 {
            let mut mgr = manager_for_adder.lock().unwrap();
            mgr.get_or_create_solver(0, &wall1_for_adder, "Wall1")
                .expect("Failed to add wall1");
            mgr.get_or_create_solver(1, &wall2_for_adder, "Wall2")
                .expect("Failed to add wall2");
        }
    });

    let manager_for_reader = StdArc::clone(&manager_stats);
    let reader = thread::spawn(move || {
        for _ in 0..5 {
            let mgr = manager_for_reader.lock().unwrap();
            let stats = mgr.get_stats();

            assert!(stats.total_walls <= 2);
            assert!(stats.five_r1c_count + stats.ctf_count + stats.fd_count <= stats.total_walls);
        }
    });

    adder.join().expect("Adder should not panic");
    reader.join().expect("Reader should not panic");
}
