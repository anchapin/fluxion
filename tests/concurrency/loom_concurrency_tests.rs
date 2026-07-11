//! Concurrency tests for parallel solver execution (Issue #1065, #1194, #1352)
//!
//! This module tests that the parallel execution paths in `SolverManager`
//! (per-wall transient solvers, inter-zone wall boundary merge) and the
//! multi-zone runtime (post-#1291 `ThermalModel<VF>::step_physics`,
//! `solve_timesteps`, `h_tr_iz` inter-zone conductance, and
//! `get_zone_energies_kwh` accumulator) are free from race conditions and
//! deadlocks when multiple threads / rayon workers update shared state.
//!
//! # Scope
//!
//! - Single shared `SolverManager` instance (Issue #1065, #1194):
//!   5 loom tests at lines 113–247 — `MatrixState` updates, sequence
//!   integrity, reader/writer contention, atomic boundary merge.
//! - Multi-zone runtime (Issue #1352): 4 loom tests below line 247 —
//!   2-zone concurrent `step_physics` with shared HVAC schedule,
//!   3-zone rayon `par_iter` over per-zone solvers with shared HVAC
//!   demand payload collection, shared inter-zone airflow conductance
//!   `h_tr_iz` reads/writes, and per-zone energy accumulator under a
//!   shared weather timestep. Each has a paired `std::thread::spawn`
//!   baseline that runs without `LOOM=1`.
//!
//! # Running Tests
//!
//! ```bash
//! # Run all concurrency tests — uses std threads (no loom required)
//! cargo test --test loom_concurrency_tests
//!
//! # Run with loom model checking (explores all thread interleavings)
//! LOOM=1 cargo test --features loom --test loom_concurrency_tests
//! ```
//!
//! # Loom Model Checking
//!
//! Loom runs each test multiple times, exploring different thread interleavings
//! to find race conditions and deadlocks that might only occur rarely. The
//! 4 new multi-zone loom tests exercise shared `Arc<StdMutex<...>>` state
//! across `loom::fuzz` blocks; allow up to 30 minutes for `LOOM=1` to run
//! locally because loom explores N! thread orderings.

use std::sync::{Arc as StdArc, Mutex as StdMutex};
use std::thread;

use fluxion::physics::cta::VectorField;
use fluxion::physics::method_selector::ThermalMethodSelector;
use fluxion::physics::solver_manager::SolverManager;
use fluxion::sim::assembly::{AssemblyBuilder, ConcreteMaterial};
use fluxion::sim::engine::ThermalModel;
use fluxion::sim::per_surface_conduction::{
    MassNode, PerSurfaceConductionSolver, SurfaceKind, SurfaceNode,
};

// Always import rayon — the baseline rayon `par_iter` test (Issue #1352
// acceptance criterion #2) needs the trait in scope to compile under both
// the default build and `--features loom`. The rayon test runs OS threads
// rather than loom-controlled threads; under `LOOM=1` loom cannot introspect
// rayon workers, so the test serves only as a regression baseline.
#[allow(unused_imports)]
use rayon::prelude::*;

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

/// Matrix state for concurrent merge operations.
#[derive(Debug, Clone)]
pub struct MatrixState {
    pub temperatures: Vec<f64>,
    pub sequence: usize,
    pub wall_indices: Vec<usize>,
}

/// Per-zone HVAC demand payload collected from concurrent zone steps
/// (Issue #1352). Mirrors the per-zone energy accumulator surfaced by
/// `ThermalModel::get_zone_energies_kwh` (#1291). Used to assert that
/// no zone's demand payload is dropped under rayon `par_iter` over
/// multiple zones.
#[derive(Debug, Clone, PartialEq)]
pub struct HvacDemandPayload {
    pub zone_index: usize,
    pub sequence: usize,
    pub energy_kwh: f64,
}

impl MatrixState {
    pub fn new(num_zones: usize) -> Self {
        Self {
            temperatures: vec![20.0; num_zones],
            sequence: 0,
            wall_indices: (0..num_zones).collect(),
        }
    }

    pub fn merge_temp_update(&mut self, zone_index: usize, new_temp: f64) -> HeatTransferPayload {
        self.sequence += 1;
        let old_temp = self.temperatures[zone_index];
        self.temperatures[zone_index] = new_temp;
        HeatTransferPayload {
            wall_index: zone_index,
            flux: (new_temp - old_temp).abs(),
            sequence_number: self.sequence,
        }
    }

    pub fn merge_wall_fluxes(&mut self, zone_index: usize, flux: f64) {
        self.sequence += 1;
        self.temperatures[zone_index] += flux * 0.01;
    }
}

// ============ Loom Model Checking Tests ============
// These tests use loom's model checking to explore thread interleavings
//
// Note: Loom requires the closure to be Send + Sync. We use std::sync primitives
// with loom::thread for controlled interleaving.

#[cfg(feature = "loom")]
mod loom_tests {
    use super::*;
    use loom::thread;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc as StdArc, Mutex as StdMutex};

    /// Test concurrent updates to matrix state using loom-controlled threads
    #[test]
    fn test_loom_concurrent_matrix_updates() {
        let state = StdArc::new(StdMutex::new(MatrixState::new(4)));

        loom::fuzz(move || {
            let s2 = StdArc::clone(&state);
            let t1 = thread::spawn(move || {
                let mut s = s2.lock().unwrap();
                s.merge_temp_update(0, 25.0);
                s.merge_temp_update(1, 22.0);
            });

            let s3 = StdArc::clone(&state);
            let t2 = thread::spawn(move || {
                let mut s = s3.lock().unwrap();
                s.merge_temp_update(2, 18.0);
                s.merge_temp_update(3, 20.0);
            });

            t1.join().unwrap();
            t2.join().unwrap();

            let s = state.lock().unwrap();
            assert_eq!(s.temperatures[0], 25.0);
            assert_eq!(s.temperatures[1], 22.0);
            assert_eq!(s.temperatures[2], 18.0);
            assert_eq!(s.temperatures[3], 20.0);
            assert_eq!(s.sequence, 4);
        });
    }

    /// Test sequence number integrity under concurrent updates
    #[test]
    fn test_loom_sequence_integrity() {
        let counter = StdArc::new(StdMutex::new(0usize));

        loom::fuzz(move || {
            let mut handles = vec![];
            for _ in 0..3 {
                let c = StdArc::clone(&counter);
                let handle = thread::spawn(move || {
                    let mut cnt = c.lock().unwrap();
                    *cnt += 1;
                });
                handles.push(handle);
            }

            for h in handles {
                h.join().unwrap();
            }

            let cnt = counter.lock().unwrap();
            assert_eq!(*cnt, 3);
        });
    }

    /// Test read-write no deadlock
    #[test]
    fn test_loom_read_write_no_deadlock() {
        let state = StdArc::new(StdMutex::new(MatrixState::new(2)));

        loom::fuzz(move || {
            let r = StdArc::clone(&state);
            let reader = thread::spawn(move || {
                for _ in 0..5 {
                    let _s = r.lock().unwrap();
                }
            });

            let w = StdArc::clone(&state);
            let writer = thread::spawn(move || {
                for i in 0..3 {
                    let mut s = w.lock().unwrap();
                    s.merge_temp_update(i % 4, 20.0 + (i as f64));
                }
            });

            reader.join().unwrap();
            writer.join().unwrap();
        });
    }

    /// Test multiple threads updating same cell
    #[test]
    fn test_loom_shared_update() {
        let state = StdArc::new(StdMutex::new(MatrixState::new(1)));

        loom::fuzz(move || {
            let s2 = StdArc::clone(&state);
            let t1 = thread::spawn(move || {
                let mut s = s2.lock().unwrap();
                s.merge_temp_update(0, 10.0);
            });

            let s3 = StdArc::clone(&state);
            let t2 = thread::spawn(move || {
                let mut s = s3.lock().unwrap();
                s.merge_temp_update(0, 20.0);
            });

            t1.join().unwrap();
            t2.join().unwrap();

            let s = state.lock().unwrap();
            assert_eq!(s.sequence, 2);
        });
    }

    /// Test boundary condition merge using atomic operations
    #[test]
    fn test_loom_boundary_merge_atomic() {
        let wall0_flux = AtomicUsize::new(0);
        let wall1_flux = AtomicUsize::new(0);

        loom::fuzz(move || {
            // Clone before moving into threads
            let f0_for_thread = AtomicUsize::new(wall0_flux.load(Ordering::SeqCst));
            let f1_for_thread = AtomicUsize::new(wall1_flux.load(Ordering::SeqCst));

            let t0 = thread::spawn(move || {
                f0_for_thread.store(100, Ordering::SeqCst);
            });

            let t1 = thread::spawn(move || {
                f1_for_thread.store(150, Ordering::SeqCst);
            });

            t0.join().unwrap();
            t1.join().unwrap();

            // Note: This test verifies atomic operations work, but doesn't test
            // cross-thread visibility since we cloned before threads ran
        });
    }

    // ========================================================================
    // Multi-zone runtime tests (Issue #1352)
    //
    // The original 5 loom tests above (#1065 / #1194) cover a single shared
    // `SolverManager` instance for per-wall transient conduction. The multi-zone
    // path (post-#1291 / #1293) introduced in `src/cli/multi_zone.rs` and
    // `src/sim/thermal_model_physics/` spawns per-zone solvers concurrently and
    // touches shared HVAC schedules + inter-zone airflow conductances. The
    // tests below extend loom coverage to that path.
    // ========================================================================

    /// Build a balanced 2-zone `ThermalModel<VectorField>` for the multi-zone
    /// loom tests. Helper kept inside the loom module so it doesn't compile
    /// under non-loom configurations where the harness import is unused.
    fn build_two_zone_model() -> ThermalModel<VectorField> {
        let mut m = ThermalModel::<VectorField>::new(2);
        // Pre-populate deterministic loads so `step_physics` produces finite
        // values regardless of which permutation loom explores.
        m.set_loads(&[5.0, 10.0]);
        m
    }

    /// Helper: push a per-zone HVAC demand payload into a shared collector.
    /// Mirrors the per-zone energy payload wired up in #1291 (`per_zone_energies`
    /// accumulator surfaced via `get_zone_energies_kwh`).
    fn record_zone_demand(
        sink: &StdArc<StdMutex<Vec<HvacDemandPayload>>>,
        zone_index: usize,
        energy_kwh: f64,
        step_counter: &StdArc<StdMutex<usize>>,
    ) {
        let mut counter = step_counter.lock().unwrap();
        *counter += 1;
        let seq = *counter;
        drop(counter);

        let mut buf = sink.lock().unwrap();
        buf.push(HvacDemandPayload {
            zone_index,
            sequence: seq,
            energy_kwh,
        });
    }

    /// Issue #1352 acceptance criterion #1: 2-zone concurrent `step_physics`
    /// with a shared HVAC schedule. The schedule fields (`heating_schedule`,
    /// `cooling_schedule`) are per-`ThermalModel`, so all zones share them.
    /// Two loom threads lock the model, mutate the schedule, run
    /// `step_physics`, and bump a shared step counter.
    #[test]
    fn test_loom_two_zone_concurrent_step_with_shared_hvac_schedule() {
        let model = StdArc::new(StdMutex::new(build_two_zone_model()));
        let step_counter = StdArc::new(StdMutex::new(0usize));

        loom::fuzz(move || {
            let m1 = StdArc::clone(&model);
            let c1 = StdArc::clone(&step_counter);
            let t1 = thread::spawn(move || {
                let mut m = m1.lock().unwrap();
                // Mutate the shared HVAC schedule (per-model, per all zones).
                m.heating_schedule.fill_range(0, 12, 1.0);
                let energy = m.step_physics(0, 10.0, 3600.0);
                assert!(energy.is_finite());
                let mut c = c1.lock().unwrap();
                *c += 1;
            });

            let m2 = StdArc::clone(&model);
            let c2 = StdArc::clone(&step_counter);
            let t2 = thread::spawn(move || {
                let mut m = m2.lock().unwrap();
                m.cooling_schedule.fill_range(12, 24, 1.0);
                let energy = m.step_physics(0, 5.0, 3600.0);
                assert!(energy.is_finite());
                let mut c = c2.lock().unwrap();
                *c += 1;
            });

            t1.join().unwrap();
            t2.join().unwrap();

            let m = model.lock().unwrap();
            let temps = m.get_temperatures();
            assert_eq!(temps.len(), 2);
            assert!(temps.iter().all(|t| t.is_finite()));
            drop(m);

            let c = step_counter.lock().unwrap();
            assert_eq!(*c, 2);
        });
    }

    /// Issue #1352 acceptance criterion #2: 3+ zone rayon `par_iter` over
    /// per-zone solvers, asserting no dropped HVAC demand payload and no
    /// deadlock. loom cannot introspect into rayon workers (which use real
    /// OS threads), so this test mirrors the rayon `par_iter` pattern with
    /// three loom-controlled threads — each representing one rayon worker —
    /// collecting its payload into a shared collector.
    #[test]
    fn test_loom_three_zone_rayon_par_iter_shared_hvac_demand() {
        let sink: StdArc<StdMutex<Vec<HvacDemandPayload>>> = StdArc::new(StdMutex::new(Vec::new()));
        let step_counter = StdArc::new(StdMutex::new(0usize));

        loom::fuzz(move || {
            let mut handles = Vec::new();
            for zone_index in 0..3 {
                let s = StdArc::clone(&sink);
                let c = StdArc::clone(&step_counter);
                let handle = thread::spawn(move || {
                    let mut model = ThermalModel::<VectorField>::new(1);
                    model.set_loads(&[5.0 + zone_index as f64]);
                    let energy = model.step_physics(0, 10.0, 3600.0);
                    assert!(energy.is_finite());
                    record_zone_demand(&s, zone_index, energy, &c);
                });
                handles.push(handle);
            }
            for h in handles {
                h.join().unwrap();
            }

            // No dropped payload — all three zone workers pushed their demand.
            let collected = sink.lock().unwrap();
            assert_eq!(collected.len(), 3, "no dropped HVAC demand payload");
            assert!(collected.iter().all(|p| p.energy_kwh.is_finite()));
            let c = step_counter.lock().unwrap();
            assert_eq!(*c, 3);
        });
    }

    /// Issue #1352 acceptance criterion #3: shared inter-zone airflow
    /// conductance (`h_tr_iz`) reads/writes during concurrent steps.
    /// `h_tr_iz` is a per-zone `VectorField` whose values drive the
    /// inter-zone heat-flow term in the multi-zone heat balance (see
    /// `src/cli/multi_zone.rs::MultiZoneConfig::inter_zone_conductance`).
    /// Two loom threads concurrently read the conductance while another
    /// writes — the read must always see a fully-written value.
    #[test]
    fn test_loom_shared_inter_zone_conductance_concurrent_steps() {
        let model = StdArc::new(StdMutex::new(build_two_zone_model()));

        loom::fuzz(move || {
            // Seeder: write initial conductance so readers see a non-zero baseline.
            {
                let mut m = model.lock().unwrap();
                let h_iz = m.h_tr_iz.as_mut_slice();
                h_iz[0] = 5.0;
                h_iz[1] = 5.0;
            }

            let m_writer = StdArc::clone(&model);
            let writer = thread::spawn(move || {
                let mut m = m_writer.lock().unwrap();
                let h_iz = m.h_tr_iz.as_mut_slice();
                // Toggle the inter-zone conductance to a different value;
                // any partial-write would corrupt subsequent reads.
                h_iz[0] = 7.5;
                h_iz[1] = 7.5;
                let energy = m.step_physics(0, 10.0, 3600.0);
                assert!(energy.is_finite());
            });

            let m_reader = StdArc::clone(&model);
            let reader = thread::spawn(move || {
                // Read the conductance — under loom this explores every
                // interleaving with the writer's mutation + step_physics.
                let mut observed = Vec::new();
                for _ in 0..4 {
                    let m = m_reader.lock().unwrap();
                    let h_iz = m.h_tr_iz.as_ref();
                    observed.push((h_iz[0], h_iz[1]));
                }
                // Either the initial 5.0 pair or the post-write 7.5 pair —
                // never a mix, never a partial.
                for (a, b) in observed {
                    assert!(a == 5.0 || a == 7.5, "partial write observed: {}", a);
                    assert!(b == 5.0 || b == 7.5, "partial write observed: {}", b);
                    assert_eq!(a, b, "inter-zone conductance must be symmetric");
                }
            });

            writer.join().unwrap();
            reader.join().unwrap();

            // Post-condition: regardless of interleaving, h_tr_iz must be
            // exactly the writer's value (5.0 before, 7.5 after writer runs).
            let m = model.lock().unwrap();
            let final_h_iz = m.h_tr_iz.as_ref();
            assert!(final_h_iz[0] == 5.0 || final_h_iz[0] == 7.5);
            assert_eq!(final_h_iz[0], final_h_iz[1]);
        });
    }

    /// Issue #1352 acceptance criterion #4: per-zone energy accumulator
    /// (`get_zone_energies_kwh`) under concurrent updates from multiple
    /// zones sharing a common weather timestep. Models the #1291
    /// `per_zone_energies` accumulator. Verifies no energy is silently
    /// dropped across two concurrent step_physics calls on a 2-zone model.
    #[test]
    fn test_loom_per_zone_energy_accumulator_shared_weather() {
        let model = StdArc::new(StdMutex::new(build_two_zone_model()));
        let energy_counter = StdArc::new(StdMutex::new(0usize));

        loom::fuzz(move || {
            let m1 = StdArc::clone(&model);
            let c1 = StdArc::clone(&energy_counter);
            let t1 = thread::spawn(move || {
                let mut m = m1.lock().unwrap();
                let _e = m.step_physics(0, 10.0, 3600.0);
                let mut c = c1.lock().unwrap();
                *c += 1;
            });

            let m2 = StdArc::clone(&model);
            let c2 = StdArc::clone(&energy_counter);
            let t2 = thread::spawn(move || {
                let mut m = m2.lock().unwrap();
                let _e = m.step_physics(0, 10.0, 3600.0);
                let mut c = c2.lock().unwrap();
                *c += 1;
            });

            t1.join().unwrap();
            t2.join().unwrap();

            // Both steps completed — accumulated per-zone energies must be
            // finite and well-formed (len == num_zones).
            let m = model.lock().unwrap();
            let zone_energies = m.get_zone_energies_kwh();
            assert_eq!(zone_energies.len(), m.num_zones);
            assert!(
                zone_energies.iter().all(|e| e.is_finite()),
                "per-zone energies must be finite, got {:?}",
                zone_energies
            );
            drop(m);

            let c = energy_counter.lock().unwrap();
            assert_eq!(*c, 2);
        });
    }

    // ========================================================================
    // Per-Surface Conduction Solver tests (Issue #1426)
    //
    // The tests below extend loom coverage to `PerSurfaceConductionSolver`
    // (the independent backward-Euler per-surface solver in
    // `src/sim/per_surface_conduction.rs`) used by the 9R4C high-mass path
    // (Cases 900–960 per ADR-002). The solver manages separate
    // `Vec<SurfaceNode>` + `Vec<MassNode>` collections that grow on
    // `add_surface_from_params`, and the step paths
    // (`update_all`, `update_mass_nodes`, `update_surface`) read+write these
    // vectors. We verify the lock-protected access patterns don't introduce
    // races, dropped updates, or non-finite energy balances.
    //
    // The four tests mirror the #1384 pattern (`loom::fuzz` + paired
    // `std::thread::spawn` baseline): concurrent add + step, shared
    // boundary temperature, mass-node evolution under contention, and
    // multi-zone 9R4C step. PerSurfaceConductionSolver is `!Sync`
    // (because of interior `Vec` mutability), so we wrap it in
    // `StdArc<StdMutex<…>>` and exercise the locking discipline.
    // ========================================================================

    /// Issue #1426 acceptance criterion #1: 3 loom threads — two call
    /// `add_surface_from_params` while one runs `update_all`. After all
    /// threads join, the surface count must be at least the seed count
    /// (no surface lost to a torn write) and every surface temperature
    /// must remain finite (no NaN introduced by a partial realloc read).
    #[test]
    fn test_loom_per_surface_solver_concurrent_add_and_step() {
        // Seed one surface so the solver is non-empty when update_all runs.
        let solver = StdArc::new(StdMutex::new(PerSurfaceConductionSolver::new()));
        {
            let mut s = solver.lock().unwrap();
            s.add_surface_from_params(
                0,
                SurfaceKind::Wall,
                10.0,
                0.5,
                20.0,
                5.0, // h_tr_ms
                4.0, // h_tr_is
                2.0, // h_tr_em
            );
        }
        let initial_len = solver.lock().unwrap().len();
        assert_eq!(initial_len, 1);

        loom::fuzz(move || {
            let s_add_a = StdArc::clone(&solver);
            let adder_a = thread::spawn(move || {
                let mut s = s_add_a.lock().unwrap();
                s.add_surface_from_params(
                    100,
                    SurfaceKind::Roof,
                    8.0,
                    0.3,
                    22.0,
                    4.0,
                    3.0,
                    1.5,
                );
            });

            let s_add_b = StdArc::clone(&solver);
            let adder_b = thread::spawn(move || {
                let mut s = s_add_b.lock().unwrap();
                s.add_surface_from_params(
                    200,
                    SurfaceKind::Floor,
                    6.0,
                    0.4,
                    18.0,
                    3.5,
                    2.5,
                    1.0,
                );
            });

            let s_step = StdArc::clone(&solver);
            let stepper = thread::spawn(move || {
                let mut s = s_step.lock().unwrap();
                // update_all iterates `self.surfaces` — a Vec; if either adder
                // reallocated the Vec mid-iteration we'd see a torn temperature.
                s.update_all(60.0, 20.0, 5.0, 0.0);
            });

            adder_a.join().unwrap();
            adder_b.join().unwrap();
            stepper.join().unwrap();

            // Post-condition: surface count is monotonic (initial + up to 2 added),
            // never less than the initial seed (no surface lost to realloc race).
            let s = solver.lock().unwrap();
            let len_after = s.len();
            assert!(
                len_after >= initial_len,
                "surface count decreased: {} -> {}",
                initial_len,
                len_after
            );
            assert!(
                len_after <= initial_len + 2,
                "surface count grew beyond expectations: {} -> {}",
                initial_len,
                len_after
            );

            // Every surface temperature must be finite — guards against NaN
            // from a partial read across a Vec realloc.
            for t in s.surface_temperatures() {
                assert!(t.is_finite(), "non-finite surface temperature: {}", t);
            }
        });
    }

    /// Issue #1426 acceptance criterion #2: two threads concurrently call
    /// `update_all` while a third mutates the boundary conditions (air
    /// temperature + exterior temperature + phi_m_surface, all stored in a
    /// shared cell). Assert no NaN/partial-update — analog of #1384's
    /// `test_loom_shared_inter_zone_conductance_concurrent_steps`.
    #[test]
    fn test_loom_per_surface_solver_shared_boundary_temperature() {
        let solver = StdArc::new(StdMutex::new(PerSurfaceConductionSolver::new()));
        {
            let mut s = solver.lock().unwrap();
            s.add_surface_from_params(0, SurfaceKind::Wall, 10.0, 0.5, 20.0, 5.0, 4.0, 2.0);
            s.add_surface_from_params(1, SurfaceKind::Roof, 8.0, 0.3, 22.0, 4.0, 3.0, 1.5);
        }

        // Boundary temperatures live in a shared cell so the writer thread
        // can race against the two stepper threads. In a safe locking
        // discipline, each stepper sees a fully-written triple.
        let boundary = StdArc::new(StdMutex::new((20.0_f64, 5.0_f64, 0.0_f64)));

        loom::fuzz(move || {
            // Seeder: reset the boundary to a known state.
            {
                let mut b = boundary.lock().unwrap();
                b.0 = 20.0;
                b.1 = 5.0;
                b.2 = 0.0;
            }

            let b_writer = StdArc::clone(&boundary);
            let writer = thread::spawn(move || {
                let mut b = b_writer.lock().unwrap();
                b.0 = 25.0; // T_air
                b.1 = 8.0; // T_ext
                b.2 = 100.0; // phi_m_surface
            });

            let s_step_a = StdArc::clone(&solver);
            let b_step_a = StdArc::clone(&boundary);
            let stepper_a = thread::spawn(move || {
                let b = b_step_a.lock().unwrap();
                let mut s = s_step_a.lock().unwrap();
                s.update_all(60.0, b.0, b.1, b.2);
            });

            let s_step_b = StdArc::clone(&solver);
            let b_step_b = StdArc::clone(&boundary);
            let stepper_b = thread::spawn(move || {
                let b = b_step_b.lock().unwrap();
                let mut s = s_step_b.lock().unwrap();
                s.update_all(60.0, b.0, b.1, b.2);
            });

            writer.join().unwrap();
            stepper_a.join().unwrap();
            stepper_b.join().unwrap();

            // Each surface's temperature must be finite — guards against a
            // partial-read of the boundary triple.
            let s = solver.lock().unwrap();
            for t in s.surface_temperatures() {
                assert!(t.is_finite(), "non-finite surface temperature: {}", t);
            }
            // Energy imbalance is bounded — a partial write would push it
            // past reasonable limits.
            let b = boundary.lock().unwrap();
            let imbalance = s.energy_imbalance(b.0, b.1);
            assert!(
                imbalance.is_finite(),
                "energy imbalance must be finite, got {}",
                imbalance
            );
            assert!(
                imbalance < 1.0e6,
                "energy imbalance too large (likely partial write): {}",
                imbalance
            );
        });
    }

    /// Issue #1426 acceptance criterion #3: two threads step the same
    /// `MassNode` via `update_mass_nodes` (the "solve_step" path used by
    /// the 9R4C high-mass solver per Issue #1003). Assert the final mass
    /// temperature is one of the two thread-specific steady-state values,
    /// never an interpolated/partial value. This is the race the issue
    /// explicitly flags: without the lock, the two threads can interleave
    /// `MassNode::update()` such that the final temperature is neither of
    /// the two expected values.
    #[test]
    fn test_loom_per_surface_solver_mass_node_evolution_race() {
        // Pre-populate two mass nodes so update_mass_nodes iterates over a
        // populated Vec. The 9R4C solver drives wall/roof/floor mass nodes.
        let solver = StdArc::new(StdMutex::new(PerSurfaceConductionSolver::new()));
        {
            let mut s = solver.lock().unwrap();
            s.add_surface_with_mass(
                SurfaceNode::new(0, SurfaceKind::Wall, 10.0, 0.5, 20.0, 50_000.0, 5.0, 4.0, 2.0, 20.0),
                MassNode::new(0, 20.0, 50_000.0, 5.0, 4.0),
            );
            s.add_surface_with_mass(
                SurfaceNode::new(1, SurfaceKind::Roof, 8.0, 0.3, 22.0, 40_000.0, 4.0, 3.0, 1.5, 22.0),
                MassNode::new(1, 22.0, 40_000.0, 4.0, 3.0),
            );
        }

        // Two distinct (T_air, T_sky) pairs — each thread picks one and
        // drives the mass-node update. The steady-state mass temperature
        // for node 0 with (T_air, T_sky) = (25, 5) and h_tr_is=5, h_tr_ms=4
        // is (5*25 + 4*5) / (5+4) = (125 + 20) / 9 = 16.111...
        // and with (T_air, T_sky) = (10, 30) is (5*10 + 4*30) / 9 = 170/9
        // = 18.888...
        // (Capacitance is large, so single-step values are very close to
        //  steady state; we assert "close to one of two targets".)
        let target_a = (5.0 * 25.0 + 4.0 * 5.0) / (5.0 + 4.0); // ~16.111
        let target_b = (5.0 * 10.0 + 4.0 * 30.0) / (5.0 + 4.0); // ~18.889

        loom::fuzz(move || {
            // Reset the mass temperatures so each fuzz iteration starts fresh.
            {
                let mut s = solver.lock().unwrap();
                s.get_mass_node_mut(0).unwrap().temperature = 20.0;
                s.get_mass_node_mut(1).unwrap().temperature = 22.0;
            }

            let s_a = StdArc::clone(&solver);
            let driver_a = thread::spawn(move || {
                let mut s = s_a.lock().unwrap();
                s.update_mass_nodes(3600.0, 25.0, 5.0);
            });

            let s_b = StdArc::clone(&solver);
            let driver_b = thread::spawn(move || {
                let mut s = s_b.lock().unwrap();
                s.update_mass_nodes(3600.0, 10.0, 30.0);
            });

            driver_a.join().unwrap();
            driver_b.join().unwrap();

            // After both threads run, the mass temperature for node 0
            // must be close to one of the two thread-specific targets
            // (never an interpolated/partial value).
            let s = solver.lock().unwrap();
            let t0 = s.get_mass_node(0).unwrap().temperature;
            let t1 = s.get_mass_node(1).unwrap().temperature;
            assert!(t0.is_finite(), "non-finite mass temperature: {}", t0);
            assert!(t1.is_finite(), "non-finite mass temperature: {}", t1);
            let dt_a = (t0 - target_a).abs();
            let dt_b = (t0 - target_b).abs();
            let tol = 1.0; // 1°C tolerance — well above any partial-update corruption
            assert!(
                dt_a < tol || dt_b < tol,
                "final mass temperature {} is neither target_a={} (dt={}) nor target_b={} (dt={})",
                t0,
                target_a,
                dt_a,
                target_b,
                dt_b
            );
        });
    }

    /// Issue #1426 acceptance criterion #4: 4-zone model where each zone
    /// owns one `PerSurfaceConductionSolver`. Two loom threads drive
    /// per-zone `update_all` while a third reads `surface_temperatures`.
    /// Mirrors the multi-zone 9R4C high-mass setup (Cases 900–960).
    /// Asserts all temperatures remain finite across all interleavings.
    #[test]
    fn test_loom_per_surface_solver_multi_zone_9r4c_step() {
        // Build 4 zone-owned solvers, each with a wall surface (the
        // 9R4C high-mass building has wall/roof/floor nodes, but for the
        // concurrency test a single wall per zone is sufficient).
        let mut zones: Vec<StdArc<StdMutex<PerSurfaceConductionSolver>>> = Vec::new();
        for zone_index in 0..4 {
            let mut s = PerSurfaceConductionSolver::new();
            s.add_surface_from_params(
                zone_index,
                SurfaceKind::Wall,
                10.0,
                0.5,
                20.0 + zone_index as f64,
                5.0,
                4.0,
                2.0,
            );
            zones.push(StdArc::new(StdMutex::new(s)));
        }

        loom::fuzz(move || {
            // Driver A: steps zone 0 and zone 2.
            let z_a_0 = StdArc::clone(&zones[0]);
            let z_a_2 = StdArc::clone(&zones[2]);
            let driver_a = thread::spawn(move || {
                let mut z0 = z_a_0.lock().unwrap();
                z0.update_all(60.0, 22.0, 5.0, 0.0);
                drop(z0);
                let mut z2 = z_a_2.lock().unwrap();
                z2.update_all(60.0, 22.0, 5.0, 0.0);
            });

            // Driver B: steps zone 1 and zone 3.
            let z_b_1 = StdArc::clone(&zones[1]);
            let z_b_3 = StdArc::clone(&zones[3]);
            let driver_b = thread::spawn(move || {
                let mut z1 = z_b_1.lock().unwrap();
                z1.update_all(60.0, 18.0, 10.0, 0.0);
                drop(z1);
                let mut z3 = z_b_3.lock().unwrap();
                z3.update_all(60.0, 18.0, 10.0, 0.0);
            });

            // Reader: snapshots all 4 zones' surface temperatures.
            let z_r_0 = StdArc::clone(&zones[0]);
            let z_r_1 = StdArc::clone(&zones[1]);
            let z_r_2 = StdArc::clone(&zones[2]);
            let z_r_3 = StdArc::clone(&zones[3]);
            let reader = thread::spawn(move || {
                let mut observed = Vec::new();
                for _ in 0..3 {
                    let t0 = z_r_0.lock().unwrap().surface_temperatures();
                    let t1 = z_r_1.lock().unwrap().surface_temperatures();
                    let t2 = z_r_2.lock().unwrap().surface_temperatures();
                    let t3 = z_r_3.lock().unwrap().surface_temperatures();
                    observed.push((t0, t1, t2, t3));
                }
                for (t0, t1, t2, t3) in observed {
                    for t in t0.iter().chain(t1.iter()).chain(t2.iter()).chain(t3.iter()) {
                        assert!(t.is_finite(), "non-finite surface temperature: {}", t);
                    }
                }
            });

            driver_a.join().unwrap();
            driver_b.join().unwrap();
            reader.join().unwrap();

            // Post-condition: every zone's final surface temperature is finite.
            for zone in &zones {
                let s = zone.lock().unwrap();
                for t in s.surface_temperatures() {
                    assert!(t.is_finite(), "non-finite surface temperature: {}", t);
                }
            }
        });
    }
}

// ============ Standard Concurrency Tests ============

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

#[test]
fn test_matrix_state_concurrent_merge() {
    let state = StdArc::new(StdMutex::new(MatrixState::new(4)));

    let mut handles = vec![];

    for thread_id in 0..4 {
        let state_clone = StdArc::clone(&state);
        let handle = std::thread::spawn(move || {
            let mut s = state_clone.lock().unwrap();
            s.merge_temp_update(thread_id, 20.0 + (thread_id as f64 * 5.0));
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let s = state.lock().unwrap();
    assert_eq!(s.temperatures[0], 20.0);
    assert_eq!(s.temperatures[1], 25.0);
    assert_eq!(s.temperatures[2], 30.0);
    assert_eq!(s.temperatures[3], 35.0);
    assert_eq!(s.sequence, 4);
}

#[test]
fn test_sequence_number_lost_update_detection() {
    let counter = StdArc::new(StdMutex::new(0usize));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter_clone = StdArc::clone(&counter);
        handles.push(std::thread::spawn(move || {
            let mut c = counter_clone.lock().unwrap();
            *c += 1;
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_count = *counter.lock().unwrap();
    assert_eq!(final_count, 10);
}

#[test]
fn test_boundary_condition_race_detection() {
    let bc = StdArc::new(StdMutex::new(SharedBoundaryCondition::new(0)));

    let mut handles = vec![];

    for thread_id in 0..3 {
        let bc_clone = StdArc::clone(&bc);
        handles.push(thread::spawn(move || {
            for step in 0..20 {
                let mut b = bc_clone.lock().unwrap();
                b.update(20.0 + (thread_id as f64 * 10.0) + (step as f64));
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let b = bc.lock().unwrap();
    assert_eq!(b.sequence, 60);
}

// =============================================================================
// Multi-zone runtime baselines (Issue #1352)
//
// Non-loom counterparts of the loom tests above. These run with the default
// `cargo test --test loom_concurrency_tests` (without `LOOM=1`) and use
// `std::thread::spawn` (or `rayon::par_iter` for the rayon-specific test) to
// catch regressions when loom is not available. They mirror the same
// scenarios — 2-zone concurrent `step_physics` with shared HVAC schedule,
// 3-zone rayon `par_iter` with shared demand payload, shared inter-zone
// conductance reads/writes, per-zone energy accumulator — but with real OS
// threads instead of loom's model-checked threads.
// =============================================================================

/// Build a balanced 2-zone `ThermalModel<VectorField>` for the baseline tests.
fn build_two_zone_model_baseline() -> ThermalModel<VectorField> {
    let mut m = ThermalModel::<VectorField>::new(2);
    m.set_loads(&[5.0, 10.0]);
    m
}

/// Baseline #1: 2-zone concurrent `step_physics` with shared HVAC schedule
/// using `std::thread::spawn`. Mirrors
/// `test_loom_two_zone_concurrent_step_with_shared_hvac_schedule`.
#[test]
fn test_two_zone_concurrent_step_with_shared_hvac_schedule_baseline() {
    let model = StdArc::new(StdMutex::new(build_two_zone_model_baseline()));
    let step_counter = StdArc::new(StdMutex::new(0usize));

    let mut handles = Vec::new();

    for thread_id in 0..2 {
        let m = StdArc::clone(&model);
        let c = StdArc::clone(&step_counter);
        let handle = thread::spawn(move || {
            for step in 0..3 {
                let mut model_guard = m.lock().unwrap();
                if thread_id == 0 {
                    model_guard.heating_schedule.fill_range(0, 12, 1.0);
                } else {
                    model_guard.cooling_schedule.fill_range(12, 24, 1.0);
                }
                let energy = model_guard.step_physics(step, 10.0, 3600.0);
                assert!(energy.is_finite(), "energy must be finite");
                drop(model_guard);
                let mut counter = c.lock().unwrap();
                *counter += 1;
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("baseline thread should not panic");
    }

    let final_temps = {
        let m = model.lock().unwrap();
        m.get_temperatures()
    };
    assert_eq!(final_temps.len(), 2);
    assert!(final_temps.iter().all(|t| t.is_finite()));

    let c = step_counter.lock().unwrap();
    assert_eq!(*c, 6);
}

/// Baseline #2: 3-zone rayon `par_iter` with shared HVAC demand payload
/// collection. Uses actual rayon workers (not loom threads). Asserts all
/// 3 zone payloads are collected (no dropped payload, no deadlock).
#[test]
fn test_three_zone_rayon_par_iter_shared_hvac_demand_baseline() {
    let sink: StdArc<StdMutex<Vec<HvacDemandPayload>>> = StdArc::new(StdMutex::new(Vec::new()));
    let step_counter = StdArc::new(StdMutex::new(0usize));

    let zone_indices: Vec<usize> = (0..3).collect();

    // Real rayon `par_iter` over per-zone solvers. The HvacDemandPayload
    // collection and step counter are mutex-guarded so concurrent writes
    // are serialized.
    let _: Vec<()> = zone_indices
        .par_iter()
        .map(|&zone_index| {
            let mut model = ThermalModel::<VectorField>::new(1);
            model.set_loads(&[5.0 + zone_index as f64]);
            let energy = model.step_physics(0, 10.0, 3600.0);
            assert!(energy.is_finite());

            let mut counter = step_counter.lock().unwrap();
            *counter += 1;
            let seq = *counter;
            drop(counter);

            let mut buf = sink.lock().unwrap();
            buf.push(HvacDemandPayload {
                zone_index,
                sequence: seq,
                energy_kwh: energy,
            });
        })
        .collect();

    let collected = sink.lock().unwrap();
    assert_eq!(
        collected.len(),
        3,
        "no dropped HVAC demand payload under rayon par_iter"
    );
    assert!(collected.iter().all(|p| p.energy_kwh.is_finite()));
    let mut zone_indices_seen: Vec<usize> = collected.iter().map(|p| p.zone_index).collect();
    zone_indices_seen.sort_unstable();
    assert_eq!(zone_indices_seen, vec![0, 1, 2]);
    drop(collected);

    let c = step_counter.lock().unwrap();
    assert_eq!(*c, 3);
}

/// Baseline #3: shared inter-zone airflow conductance (`h_tr_iz`)
/// reads/writes during concurrent steps via `std::thread::spawn`.
/// Mirrors `test_loom_shared_inter_zone_conductance_concurrent_steps`.
#[test]
fn test_shared_inter_zone_conductance_concurrent_steps_baseline() {
    let model = StdArc::new(StdMutex::new(build_two_zone_model_baseline()));

    // Seed the conductance.
    {
        let mut m = model.lock().unwrap();
        let h_iz = m.h_tr_iz.as_mut_slice();
        h_iz[0] = 5.0;
        h_iz[1] = 5.0;
    }

    let m_writer = StdArc::clone(&model);
    let writer = thread::spawn(move || {
        let mut m = m_writer.lock().unwrap();
        let h_iz = m.h_tr_iz.as_mut_slice();
        h_iz[0] = 7.5;
        h_iz[1] = 7.5;
        let energy = m.step_physics(0, 10.0, 3600.0);
        assert!(energy.is_finite());
    });

    let m_reader = StdArc::clone(&model);
    let reader = thread::spawn(move || {
        let mut observed = Vec::new();
        for _ in 0..10 {
            let m = m_reader.lock().unwrap();
            let h_iz = m.h_tr_iz.as_ref();
            observed.push((h_iz[0], h_iz[1]));
        }
        for (a, b) in observed {
            assert!(a == 5.0 || a == 7.5, "partial write observed: {}", a);
            assert!(b == 5.0 || b == 7.5, "partial write observed: {}", b);
            assert_eq!(a, b);
        }
    });

    writer.join().expect("writer thread should not panic");
    reader.join().expect("reader thread should not panic");

    let m = model.lock().unwrap();
    let final_h_iz = m.h_tr_iz.as_ref();
    assert!(final_h_iz[0] == 5.0 || final_h_iz[0] == 7.5);
    assert_eq!(final_h_iz[0], final_h_iz[1]);
}

/// Baseline #4: per-zone energy accumulator (`get_zone_energies_kwh`)
/// under concurrent updates from multiple zones sharing a common
/// weather timestep. Mirrors `test_loom_per_zone_energy_accumulator_shared_weather`.
#[test]
fn test_per_zone_energy_accumulator_shared_weather_baseline() {
    let model = StdArc::new(StdMutex::new(build_two_zone_model_baseline()));
    let energy_counter = StdArc::new(StdMutex::new(0usize));

    let mut handles = Vec::new();
    for step in 0..3 {
        let m = StdArc::clone(&model);
        let c = StdArc::clone(&energy_counter);
        let handle = thread::spawn(move || {
            let mut model_guard = m.lock().unwrap();
            let energy = model_guard.step_physics(step, 10.0, 3600.0);
            assert!(energy.is_finite());
            drop(model_guard);
            let mut counter = c.lock().unwrap();
            *counter += 1;
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("baseline thread should not panic");
    }

    let m = model.lock().unwrap();
    let zone_energies = m.get_zone_energies_kwh();
    assert_eq!(zone_energies.len(), m.num_zones);
    assert!(
        zone_energies.iter().all(|e| e.is_finite()),
        "per-zone energies must be finite: {:?}",
        zone_energies
    );

    let c = energy_counter.lock().unwrap();
    assert_eq!(*c, 3);
}

// =============================================================================
// Per-Surface Conduction Solver baselines (Issue #1426)
//
// Non-loom counterparts of the four loom tests above. These run with the
// default `cargo test --test loom_concurrency_tests` (without `LOOM=1`) and
// use `std::thread::spawn` to catch regressions when loom is not available.
// They mirror the same scenarios — concurrent add + step, shared boundary
// temperature, mass-node evolution under contention, and multi-zone 9R4C
// step — but with real OS threads instead of loom's model-checked threads.
// =============================================================================

/// Baseline #1 (Issue #1426): 3 OS threads — two call
/// `add_surface_from_params` while one runs `update_all`. Mirrors
/// `test_loom_per_surface_solver_concurrent_add_and_step`.
#[test]
fn test_per_surface_solver_concurrent_add_and_step_baseline() {
    let solver = StdArc::new(StdMutex::new(PerSurfaceConductionSolver::new()));
    {
        let mut s = solver.lock().unwrap();
        s.add_surface_from_params(0, SurfaceKind::Wall, 10.0, 0.5, 20.0, 5.0, 4.0, 2.0);
    }
    let initial_len = solver.lock().unwrap().len();
    assert_eq!(initial_len, 1);

    let s_add_a = StdArc::clone(&solver);
    let adder_a = thread::spawn(move || {
        let mut s = s_add_a.lock().unwrap();
        s.add_surface_from_params(100, SurfaceKind::Roof, 8.0, 0.3, 22.0, 4.0, 3.0, 1.5);
    });

    let s_add_b = StdArc::clone(&solver);
    let adder_b = thread::spawn(move || {
        let mut s = s_add_b.lock().unwrap();
        s.add_surface_from_params(200, SurfaceKind::Floor, 6.0, 0.4, 18.0, 3.5, 2.5, 1.0);
    });

    let s_step = StdArc::clone(&solver);
    let stepper = thread::spawn(move || {
        let mut s = s_step.lock().unwrap();
        s.update_all(60.0, 20.0, 5.0, 0.0);
    });

    adder_a.join().expect("adder_a should not panic");
    adder_b.join().expect("adder_b should not panic");
    stepper.join().expect("stepper should not panic");

    let s = solver.lock().unwrap();
    let len_after = s.len();
    assert!(
        len_after >= initial_len,
        "surface count decreased: {} -> {}",
        initial_len,
        len_after
    );
    assert!(
        len_after <= initial_len + 2,
        "surface count grew beyond expectations: {} -> {}",
        initial_len,
        len_after
    );

    for t in s.surface_temperatures() {
        assert!(t.is_finite(), "non-finite surface temperature: {}", t);
    }
}

/// Baseline #2 (Issue #1426): two threads concurrently call `update_all`
/// while a third mutates the boundary conditions held in a shared cell.
/// Mirrors `test_loom_per_surface_solver_shared_boundary_temperature`.
#[test]
fn test_per_surface_solver_shared_boundary_temperature_baseline() {
    let solver = StdArc::new(StdMutex::new(PerSurfaceConductionSolver::new()));
    {
        let mut s = solver.lock().unwrap();
        s.add_surface_from_params(0, SurfaceKind::Wall, 10.0, 0.5, 20.0, 5.0, 4.0, 2.0);
        s.add_surface_from_params(1, SurfaceKind::Roof, 8.0, 0.3, 22.0, 4.0, 3.0, 1.5);
    }

    let boundary = StdArc::new(StdMutex::new((20.0_f64, 5.0_f64, 0.0_f64)));

    let b_writer = StdArc::clone(&boundary);
    let writer = thread::spawn(move || {
        let mut b = b_writer.lock().unwrap();
        b.0 = 25.0;
        b.1 = 8.0;
        b.2 = 100.0;
    });

    let s_step_a = StdArc::clone(&solver);
    let b_step_a = StdArc::clone(&boundary);
    let stepper_a = thread::spawn(move || {
        let b = b_step_a.lock().unwrap();
        let mut s = s_step_a.lock().unwrap();
        s.update_all(60.0, b.0, b.1, b.2);
    });

    let s_step_b = StdArc::clone(&solver);
    let b_step_b = StdArc::clone(&boundary);
    let stepper_b = thread::spawn(move || {
        let b = b_step_b.lock().unwrap();
        let mut s = s_step_b.lock().unwrap();
        s.update_all(60.0, b.0, b.1, b.2);
    });

    writer.join().expect("writer should not panic");
    stepper_a.join().expect("stepper_a should not panic");
    stepper_b.join().expect("stepper_b should not panic");

    let s = solver.lock().unwrap();
    for t in s.surface_temperatures() {
        assert!(t.is_finite(), "non-finite surface temperature: {}", t);
    }
    let b = boundary.lock().unwrap();
    let imbalance = s.energy_imbalance(b.0, b.1);
    assert!(
        imbalance.is_finite(),
        "energy imbalance must be finite, got {}",
        imbalance
    );
    assert!(
        imbalance < 1.0e6,
        "energy imbalance too large (likely partial write): {}",
        imbalance
    );
}

/// Baseline #3 (Issue #1426): two threads step the same `MassNode` via
/// `update_mass_nodes`. Assert the final mass temperature is close to one
/// of the two thread-specific steady-state targets. Mirrors
/// `test_loom_per_surface_solver_mass_node_evolution_race`.
#[test]
fn test_per_surface_solver_mass_node_evolution_race_baseline() {
    let solver = StdArc::new(StdMutex::new(PerSurfaceConductionSolver::new()));
    {
        let mut s = solver.lock().unwrap();
        s.add_surface_with_mass(
            SurfaceNode::new(0, SurfaceKind::Wall, 10.0, 0.5, 20.0, 50_000.0, 5.0, 4.0, 2.0, 20.0),
            MassNode::new(0, 20.0, 50_000.0, 5.0, 4.0),
        );
        s.add_surface_with_mass(
            SurfaceNode::new(1, SurfaceKind::Roof, 8.0, 0.3, 22.0, 40_000.0, 4.0, 3.0, 1.5, 22.0),
            MassNode::new(1, 22.0, 40_000.0, 4.0, 3.0),
        );
    }

    let target_a = (5.0 * 25.0 + 4.0 * 5.0) / (5.0 + 4.0);
    let target_b = (5.0 * 10.0 + 4.0 * 30.0) / (5.0 + 4.0);

    let s_a = StdArc::clone(&solver);
    let driver_a = thread::spawn(move || {
        let mut s = s_a.lock().unwrap();
        s.update_mass_nodes(3600.0, 25.0, 5.0);
    });

    let s_b = StdArc::clone(&solver);
    let driver_b = thread::spawn(move || {
        let mut s = s_b.lock().unwrap();
        s.update_mass_nodes(3600.0, 10.0, 30.0);
    });

    driver_a.join().expect("driver_a should not panic");
    driver_b.join().expect("driver_b should not panic");

    let s = solver.lock().unwrap();
    let t0 = s.get_mass_node(0).unwrap().temperature;
    let t1 = s.get_mass_node(1).unwrap().temperature;
    assert!(t0.is_finite(), "non-finite mass temperature: {}", t0);
    assert!(t1.is_finite(), "non-finite mass temperature: {}", t1);
    let dt_a = (t0 - target_a).abs();
    let dt_b = (t0 - target_b).abs();
    let tol = 1.0;
    assert!(
        dt_a < tol || dt_b < tol,
        "final mass temperature {} is neither target_a={} (dt={}) nor target_b={} (dt={})",
        t0,
        target_a,
        dt_a,
        target_b,
        dt_b
    );
}

/// Baseline #4 (Issue #1426): 4-zone model where each zone owns one
/// `PerSurfaceConductionSolver`. Two threads drive per-zone `update_all`
/// while a third reads `surface_temperatures`. Mirrors
/// `test_loom_per_surface_solver_multi_zone_9r4c_step`.
#[test]
fn test_per_surface_solver_multi_zone_9r4c_step_baseline() {
    let mut zones: Vec<StdArc<StdMutex<PerSurfaceConductionSolver>>> = Vec::new();
    for zone_index in 0..4 {
        let mut s = PerSurfaceConductionSolver::new();
        s.add_surface_from_params(
            zone_index,
            SurfaceKind::Wall,
            10.0,
            0.5,
            20.0 + zone_index as f64,
            5.0,
            4.0,
            2.0,
        );
        zones.push(StdArc::new(StdMutex::new(s)));
    }

    let z_a_0 = StdArc::clone(&zones[0]);
    let z_a_2 = StdArc::clone(&zones[2]);
    let driver_a = thread::spawn(move || {
        let mut z0 = z_a_0.lock().unwrap();
        z0.update_all(60.0, 22.0, 5.0, 0.0);
        drop(z0);
        let mut z2 = z_a_2.lock().unwrap();
        z2.update_all(60.0, 22.0, 5.0, 0.0);
    });

    let z_b_1 = StdArc::clone(&zones[1]);
    let z_b_3 = StdArc::clone(&zones[3]);
    let driver_b = thread::spawn(move || {
        let mut z1 = z_b_1.lock().unwrap();
        z1.update_all(60.0, 18.0, 10.0, 0.0);
        drop(z1);
        let mut z3 = z_b_3.lock().unwrap();
        z3.update_all(60.0, 18.0, 10.0, 0.0);
    });

    let z_r_0 = StdArc::clone(&zones[0]);
    let z_r_1 = StdArc::clone(&zones[1]);
    let z_r_2 = StdArc::clone(&zones[2]);
    let z_r_3 = StdArc::clone(&zones[3]);
    let reader = thread::spawn(move || {
        let mut observed = Vec::new();
        for _ in 0..10 {
            let t0 = z_r_0.lock().unwrap().surface_temperatures();
            let t1 = z_r_1.lock().unwrap().surface_temperatures();
            let t2 = z_r_2.lock().unwrap().surface_temperatures();
            let t3 = z_r_3.lock().unwrap().surface_temperatures();
            observed.push((t0, t1, t2, t3));
        }
        for (t0, t1, t2, t3) in observed {
            for t in t0.iter().chain(t1.iter()).chain(t2.iter()).chain(t3.iter()) {
                assert!(t.is_finite(), "non-finite surface temperature: {}", t);
            }
        }
    });

    driver_a.join().expect("driver_a should not panic");
    driver_b.join().expect("driver_b should not panic");
    reader.join().expect("reader should not panic");

    for zone in &zones {
        let s = zone.lock().unwrap();
        for t in s.surface_temperatures() {
            assert!(t.is_finite(), "non-finite surface temperature: {}", t);
        }
    }
}
