//! Solver Registry - Internal ownership of solver HashMap.
//!
//! This module provides the internal registry that owns the HashMap of solver
//! instances. SolverManager uses SolverRegistry as its internal state,
//! enabling lifecycle ownership (pre-warming, cache invalidation, stats aggregation)
//! to be centralized.
//!
//! # Architecture
//!
//! ```text
//! SolverManager (public facade)
//!   ├── selector: ThermalMethodSelector
//!   ├── registry: SolverRegistry (internal ownership)
//!   │     └── solvers: HashMap<usize, Box<dyn HeatConductionSolver>>
//!   └── stats: SolverStats
//! ```

use crate::physics::ctf_solver_wrapper::CTFSolverWrapper;
use crate::physics::fd_solver_wrapper::FDSolverWrapper;
use crate::physics::five_r1c_solver::FiveR1CSolver;
use crate::physics::multi_node_solver::MultiNodeSolver;
use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
use crate::physics::wall_spec::WallSpec;
// Issue #1349 (Phase 2 crate split): `BuildingAssembly` moved to `fluxion_core::assembly`.
use fluxion_core::assembly::BuildingAssembly;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::RwLock;

pub struct SolverRegistry {
    solvers: HashMap<usize, Box<dyn HeatConductionSolver>>,
    wall_assemblies: HashMap<usize, BuildingAssembly>,
    method_counts: HashMap<String, usize>,
}

/// Issue #1429 / Issue #2494 — Registered constructor keys for
/// `SolverRegistry::construct`.
///
/// Each variant maps a stable string key (used by callers like
/// `SolverManager::select`) to a `Box<dyn HeatConductionSolver>` constructor
/// taking a `&WallSpec`. The keys are the public API and must stay stable
/// across releases.
pub mod registry_keys {
    /// Lumped-capacitance 5R1C solver (default for low-mass constructions).
    pub const FIVE_R1C: &str = "5r1c";
    /// Conduction Transfer Function solver (accurate for high-mass; auto-selected
    /// by `SolverManager::select`). Exposed via the registry in Issue #2494.
    pub const CTF: &str = "ctf";
    /// Implicit finite-difference solver (robust fallback / explicit choice).
    /// Exposed via the registry in Issue #2494.
    pub const FD: &str = "fd";
    /// 9R4C four-node envelope solver (default for high-mass Case 900+,
    /// per ADR-002 and Issue #1429).
    pub const MULTINODE_9R4C: &str = "multinode_9r4c";

    /// All built-in constructor keys, in stable order. Used by
    /// [`SolverRegistry::construct`] (fast-path match arms) and to reject
    /// shadowing attempts in [`SolverRegistry::register_solver`].
    pub const BUILTIN_KEYS: &[&str] = &[FIVE_R1C, CTF, FD, MULTINODE_9R4C];
}

/// Issue #2494 — Pluggable solver factory type.
///
/// A factory converts a `&WallSpec` (+ `floor_area`) into a boxed
/// `HeatConductionSolver`, mirroring the signature of
/// [`SolverRegistry::construct`]. External crates (e.g. a fluxion-city surface
/// flux provider, an ML surrogate adapter, or a research solver) can register a
/// factory under a custom key via [`SolverRegistry::register_solver`]; the
/// registry then dispatches that key exactly like a built-in one.
///
/// This is the constructor-level analogue of the
/// `FluxionCitySurfaceFluxProvider` wrapper pattern: rather than wrapping an
/// already-constructed solver, it lets third-party code plug into the
/// *construction* dispatch so the rest of the pipeline (registry insertion,
/// `PhysicsSurfaceFluxProvider::add_surface`, stats aggregation) is reused.
pub type SolverFactory =
    dyn Fn(&WallSpec, f64) -> Result<Box<dyn HeatConductionSolver>, SolverError> + Send + Sync;

/// Issue #2494 — Global, lazily-initialised map of user-registered solver
/// factories. Consulted by [`SolverRegistry::construct`] **only** for keys that
/// are not in [`registry_keys::BUILTIN_KEYS`], so built-in dispatch stays on the
/// lock-free match path and cannot be silently shadowed.
static CUSTOM_FACTORIES: Lazy<RwLock<HashMap<String, Box<SolverFactory>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

impl SolverRegistry {
    pub fn new() -> Self {
        Self {
            solvers: HashMap::new(),
            wall_assemblies: HashMap::new(),
            method_counts: HashMap::new(),
        }
    }

    /// Issue #1429 / Issue #2494 — Construct a `Box<dyn HeatConductionSolver>`
    /// by registry key from a `&WallSpec` and `floor_area`.
    ///
    /// # Built-in keys (see [`registry_keys`])
    ///
    /// - `"5r1c"` → `FiveR1CSolver::new()` + `initialize(wall)` (low-mass
    ///   default)
    /// - `"ctf"` → `CTFSolverWrapper::new()` + `initialize(wall)` — the same
    ///   construction `SolverManager::select` uses for the CTF method (Issue
    ///   #2494). `floor_area` is unused by this solver.
    /// - `"fd"` → `FDSolverWrapper::new()` + `initialize(wall)` — the same
    ///   construction `SolverManager::select` uses for the FD method / CTF
    ///   fallback (Issue #2494). `floor_area` is unused by this solver.
    /// - `"multinode_9r4c"` →
    ///   `MultiNodeSolver::boxed_from_wall_spec(wall, floor_area)`, the 9R4C
    ///   four-node envelope solver (Issue #1429 drop-in, Issue #1593 fix)
    ///
    /// # Pluggable keys (Issue #2494)
    ///
    /// Keys not in [`registry_keys::BUILTIN_KEYS`] are looked up in the
    /// user-registered factory map (see [`SolverRegistry::register_solver`]).
    /// Registered factories are dispatched exactly like built-ins, so the rest
    /// of the pipeline (registry insertion,
    /// `PhysicsSurfaceFluxProvider::add_surface`, stats aggregation) is reused.
    ///
    /// Unknown keys return `SolverError::InvalidConfig`. The returned
    /// `Box<dyn HeatConductionSolver>` can be wrapped in a
    /// `PhysicsSurfaceFluxProvider::add_surface` (Issue #1409 wiring)
    /// the same way `FiveR1CSolver` is wrapped today.
    pub fn construct(
        key: &str,
        wall: &WallSpec,
        floor_area: f64,
    ) -> Result<Box<dyn HeatConductionSolver>, SolverError> {
        match key {
            registry_keys::FIVE_R1C => {
                let mut solver = FiveR1CSolver::new();
                solver.initialize(wall)?;
                Ok(Box::new(solver))
            }
            registry_keys::CTF => {
                let mut solver = CTFSolverWrapper::new();
                solver.initialize(wall)?;
                Ok(Box::new(solver))
            }
            registry_keys::FD => {
                let mut solver = FDSolverWrapper::new();
                solver.initialize(wall)?;
                Ok(Box::new(solver))
            }
            registry_keys::MULTINODE_9R4C => {
                Ok(MultiNodeSolver::boxed_from_wall_spec(wall, floor_area))
            }
            other => {
                // Issue #2494 — consult the pluggable factory map for keys that
                // are not built-in. Built-ins can never be shadowed because they
                // are matched above before this branch runs.
                let map = CUSTOM_FACTORIES.read().unwrap();
                if let Some(factory) = map.get(other) {
                    factory(wall, floor_area)
                } else {
                    Err(SolverError::InvalidConfig(format!(
                        "SolverRegistry::construct: unknown solver key '{other}'. \
                         Built-in keys: {}. Registered keys: {}.",
                        registry_keys::BUILTIN_KEYS.join(", "),
                        Self::registered_keys().join(", "),
                    )))
                }
            }
        }
    }

    /// Issue #2494 — Register a pluggable solver factory under `key`.
    ///
    /// The factory is invoked by [`SolverRegistry::construct`] for any `key`
    /// that is not a built-in (`5r1c` / `ctf` / `fd` / `multinode_9r4c`).
    /// Built-in keys cannot be registered (this returns
    /// `SolverError::InvalidConfig`) so the lock-free match path stays
    /// authoritative and dispatch can never be silently shadowed. Re-registering
    /// an already-registered custom key is also rejected for predictability;
    /// call [`SolverRegistry::unregister_solver`] first to replace one.
    ///
    /// This is the constructor-level analogue of the
    /// `FluxionCitySurfaceFluxProvider` wrapper pattern — it lets third-party
    /// code (e.g. an ML surrogate adapter, a research solver, or a
    /// fluxion-city surface flux provider) plug into construction dispatch so
    /// the rest of the pipeline is reused.
    ///
    /// `factory` must be `Send + Sync + 'static` because constructed solvers
    /// are stored in a registry shared across rayon threads by the
    /// `BatchOracle::evaluate_population` population-level `par_iter()`.
    pub fn register_solver<F>(key: &str, factory: F) -> Result<(), SolverError>
    where
        F: Fn(&WallSpec, f64) -> Result<Box<dyn HeatConductionSolver>, SolverError>
            + Send
            + Sync
            + 'static,
    {
        if registry_keys::BUILTIN_KEYS.contains(&key) {
            return Err(SolverError::InvalidConfig(format!(
                "SolverRegistry::register_solver: '{key}' is a built-in key and \
                 cannot be overridden. Choose a different key."
            )));
        }
        let mut map = CUSTOM_FACTORIES.write().unwrap();
        if map.contains_key(key) {
            return Err(SolverError::InvalidConfig(format!(
                "SolverRegistry::register_solver: '{key}' is already registered. \
                 Call unregister_solver first to replace it."
            )));
        }
        map.insert(key.to_string(), Box::new(factory));
        Ok(())
    }

    /// Issue #2494 — Remove a previously-registered custom solver factory.
    ///
    /// Returns `true` if a factory was present and removed. Built-in keys are
    /// never removable (returns `false`).
    pub fn unregister_solver(key: &str) -> bool {
        if registry_keys::BUILTIN_KEYS.contains(&key) {
            return false;
        }
        CUSTOM_FACTORIES.write().unwrap().remove(key).is_some()
    }

    /// Issue #2494 — Whether `key` resolves in [`construct`], either as a
    /// built-in or as a registered custom key.
    pub fn is_known_key(key: &str) -> bool {
        registry_keys::BUILTIN_KEYS.contains(&key)
            || CUSTOM_FACTORIES.read().unwrap().contains_key(key)
    }

    /// Issue #2494 — Snapshot of currently-registered custom factory keys (in
    /// arbitrary `HashMap` order). Does not include built-in keys; use
    /// [`registry_keys::BUILTIN_KEYS`] for those.
    pub fn registered_keys() -> Vec<String> {
        CUSTOM_FACTORIES.read().unwrap().keys().cloned().collect()
    }

    pub fn get_solver_mut(
        &mut self,
        wall_index: usize,
    ) -> Option<&mut Box<dyn HeatConductionSolver>> {
        self.solvers.get_mut(&wall_index)
    }

    pub fn get_solver(&self, wall_index: usize) -> Option<&dyn HeatConductionSolver> {
        self.solvers.get(&wall_index).map(|v| &**v)
    }

    pub fn contains(&self, wall_index: &usize) -> bool {
        self.solvers.contains_key(wall_index)
    }

    pub fn insert(
        &mut self,
        wall_index: usize,
        solver: Box<dyn HeatConductionSolver>,
        wall_assembly: BuildingAssembly,
        method_name: String,
    ) {
        self.solvers.insert(wall_index, solver);
        self.wall_assemblies.insert(wall_index, wall_assembly);
        *self.method_counts.entry(method_name).or_insert(0) += 1;
    }

    pub fn clear(&mut self) {
        self.solvers.clear();
        self.wall_assemblies.clear();
        self.method_counts.clear();
    }

    pub fn len(&self) -> usize {
        self.solvers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.solvers.is_empty()
    }

    pub fn method_counts(&self) -> &HashMap<String, usize> {
        &self.method_counts
    }

    pub fn wall_assemblies(&self) -> &HashMap<usize, BuildingAssembly> {
        &self.wall_assemblies
    }

    pub fn wall_assemblies_mut(&mut self) -> &mut HashMap<usize, BuildingAssembly> {
        &mut self.wall_assemblies
    }
}

impl Default for SolverRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::wall_spec::WallSpec;

    #[test]
    fn test_solver_registry_new() {
        let registry = SolverRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_solver_registry_insert_and_get() {
        use crate::physics::five_r1c_solver::FiveR1CSolver;
        use fluxion_core::assembly::{AssemblyBuilder, ConcreteMaterial};

        let mut registry = SolverRegistry::new();

        let wall = AssemblyBuilder::new("Test".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .build()
            .unwrap();

        let mut solver = FiveR1CSolver::new();
        solver.initialize(&WallSpec::from_assembly(&wall)).unwrap();
        let boxed: Box<dyn HeatConductionSolver> = Box::new(solver);

        registry.insert(0, boxed, wall.clone(), "5R1C".to_string());

        assert_eq!(registry.len(), 1);
        assert!(registry.contains(&0));
        assert!(registry.get_solver(0).is_some());
    }

    #[test]
    fn test_solver_registry_clear() {
        use crate::physics::five_r1c_solver::FiveR1CSolver;
        use fluxion_core::assembly::{AssemblyBuilder, ConcreteMaterial};

        let mut registry = SolverRegistry::new();

        let wall = AssemblyBuilder::new("Test".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .build()
            .unwrap();

        let mut solver = FiveR1CSolver::new();
        solver.initialize(&WallSpec::from_assembly(&wall)).unwrap();
        let boxed: Box<dyn HeatConductionSolver> = Box::new(solver);

        registry.insert(0, boxed, wall, "5R1C".to_string());
        assert_eq!(registry.len(), 1);

        registry.clear();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_solver_registry_method_counts() {
        use crate::physics::five_r1c_solver::FiveR1CSolver;
        use fluxion_core::assembly::{AssemblyBuilder, ConcreteMaterial};

        let mut registry = SolverRegistry::new();

        let wall = AssemblyBuilder::new("Test".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .build()
            .unwrap();

        let mut solver1 = FiveR1CSolver::new();
        solver1.initialize(&WallSpec::from_assembly(&wall)).unwrap();
        let solver2 = FiveR1CSolver::new();
        let boxed1: Box<dyn HeatConductionSolver> = Box::new(solver1);
        let boxed2: Box<dyn HeatConductionSolver> = Box::new(solver2);

        registry.insert(0, boxed1, wall.clone(), "5R1C".to_string());
        registry.insert(1, boxed2, wall, "5R1C".to_string());

        assert_eq!(registry.method_counts().get("5R1C"), Some(&2));
    }

    // ── Issue #1429 — MultiNodeSolver drop-in via SolverRegistry ──────

    /// Helper: a representative single-layer concrete `WallSpec` used by the
    /// #1429 drop-in parity tests (matches the construction used by the
    /// existing `test_swap_point_provider_parity` baseline).
    fn wall_200mm_concrete() -> WallSpec {
        WallSpec::single_layer("200mm Concrete", 0.2, 1.73, 2243.0, 837.0)
    }

    #[test]
    fn test_issue_1429_construct_multinode_returns_boxed_solver() {
        let wall = wall_200mm_concrete();
        let floor_area = 54.0; // Typical office floor area (m²)
        let solver: Box<dyn HeatConductionSolver> =
            SolverRegistry::construct(registry_keys::MULTINODE_9R4C, &wall, floor_area)
                .expect("multinode_9r4c key must construct a solver");
        assert_eq!(solver.name(), "MultiNode9R4C");
        assert!(
            solver.is_valid(),
            "constructed MultiNodeSolver must be valid"
        );
    }

    #[test]
    fn test_issue_1429_construct_unknown_key_errors() {
        let wall = wall_200mm_concrete();
        let floor_area = 54.0;
        let err = SolverRegistry::construct("nonexistent_solver", &wall, floor_area);
        match err {
            Err(e) => assert!(
                e.to_string().contains("unknown solver key"),
                "expected unknown-key error, got: {e}"
            ),
            Ok(_) => panic!("unknown solver key must return Err"),
        }
    }

    #[test]
    fn test_issue_1429_steady_state_flux_parity_within_2pct() {
        use crate::physics::units::{FromF64, Temperature, ToF64};

        let wall = wall_200mm_concrete();
        let floor_area = 54.0; // Typical office floor area (m²)

        // Construct BOTH solvers via the registry so the trait's
        // `steady_state_flux` (Quantity-typed) is called consistently.
        let r1c: Box<dyn HeatConductionSolver> =
            SolverRegistry::construct(registry_keys::FIVE_R1C, &wall, floor_area)
                .expect("5r1c construct");
        let multi: Box<dyn HeatConductionSolver> =
            SolverRegistry::construct(registry_keys::MULTINODE_9R4C, &wall, floor_area)
                .expect("multinode construct");

        let t_int = Temperature::from_value(22.0);
        let t_ext = Temperature::from_value(5.0);

        let q_r1c = r1c
            .steady_state_flux(t_int, t_ext)
            .expect("5R1C steady_state_flux")
            .to_value();
        let q_multi = multi
            .steady_state_flux(t_int, t_ext)
            .expect("multinode steady_state_flux")
            .to_value();

        // Both use q_ss = (T_ext - T_int) / R_total → identical by construction.
        let rel_err = (q_multi - q_r1c).abs() / q_r1c.abs().max(1e-9);
        assert!(
            rel_err < 0.02,
            "Steady-state flux parity violated: 5R1C={q_r1c:.6}, multinode={q_multi:.6}, rel_err={rel_err:.4} (tol 2%)"
        );
    }

    #[test]
    fn test_issue_1429_transient_flux_converges_within_5pct_after_24h() {
        use crate::physics::units::{FromF64, HeatTransferCoefficient, Temperature, Time, ToF64};

        let wall = wall_200mm_concrete();
        let floor_area = 54.0; // Typical office floor area (m²)

        let r1c: Box<dyn HeatConductionSolver> =
            SolverRegistry::construct(registry_keys::FIVE_R1C, &wall, floor_area)
                .expect("5r1c construct");
        let mut multi: Box<dyn HeatConductionSolver> =
            SolverRegistry::construct(registry_keys::MULTINODE_9R4C, &wall, floor_area)
                .expect("multinode construct");

        let dt = Time::from_value(3600.0);
        let t_int = Temperature::from_value(22.0);
        let t_ext = Temperature::from_value(5.0);
        let h_int = HeatTransferCoefficient::from_value(8.0);
        let h_ext = HeatTransferCoefficient::from_value(25.0);

        // The shared steady-state "envelope" — both solvers use the
        // closed-form q_ss = (T_ext − T_int) / R_total. The MultiNode 9R4C
        // mass nodes have a shorter effective τ (~3 h vs the 5R1C ~12 h for
        // 200 mm concrete), so after 24 h of constant forcing the MultiNode
        // transient must have converged to this shared envelope within 5%.
        let q_envelope = r1c
            .steady_state_flux(t_int, t_ext)
            .expect("5R1C steady_state_flux")
            .to_value();

        // Drive the MultiNode solver for 24 hourly steps under constant BCs.
        let mut last_q_multi = 0.0_f64;
        for _ in 0..24 {
            last_q_multi = multi
                .step(dt, t_int, t_ext, h_int, h_ext)
                .unwrap()
                .to_value();
        }

        let denom = q_envelope.abs().max(1e-9);
        let rel_err = (last_q_multi - q_envelope).abs() / denom;
        assert!(
            rel_err < 0.05,
            "Transient flux did not converge within 5% of the shared steady-state \
             envelope after 24h: multinode={last_q_multi:.6}, envelope(q_ss)={q_envelope:.6}, \
             rel_err={rel_err:.4}"
        );
    }

    #[test]
    fn test_issue_1429_parallel_resistance_preserved_through_trait() {
        use crate::physics::multi_node_solver::MultiNodeSolver;
        use crate::physics::units::{FromF64, HeatTransferCoefficient, Temperature, Time, ToF64};
        use fluxion_core::multi_node::MassAirCouplingMode;

        let wall = wall_200mm_concrete();
        let floor_area = 54.0; // Typical office floor area (m²)

        // Build via the canonical mode-aware constructor, then box it.
        let solver = MultiNodeSolver::from_wall_spec_with_mode(
            &wall,
            floor_area,
            MassAirCouplingMode::ParallelResistance,
        );
        assert_eq!(
            solver.coupling_mode,
            MassAirCouplingMode::ParallelResistance
        );

        // The boxed trait object must still step and produce a finite flux,
        // proving the coupling mode survives the trait boundary.
        let mut boxed: Box<dyn HeatConductionSolver> = Box::new(solver);
        let flux = boxed
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(22.0),
                Temperature::from_value(5.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .expect("ParallelResistance step through trait");
        assert!(
            flux.to_value().is_finite(),
            "ParallelResistance flux must be finite through trait boundary"
        );
    }

    // ── Issue #2494 — expose ctf/fd in SolverRegistry::construct ──────

    /// All four built-in keys must construct a valid `HeatConductionSolver`
    /// with the documented solver name.
    #[test]
    fn test_issue_2494_all_four_builtin_keys_construct() {
        let wall = wall_200mm_concrete();
        let floor_area = 54.0;

        for (key, expected_name) in [
            (registry_keys::FIVE_R1C, "5R1C"),
            (registry_keys::CTF, "CTF"),
            (registry_keys::FD, "FD"),
            (registry_keys::MULTINODE_9R4C, "MultiNode9R4C"),
        ] {
            let solver: Box<dyn HeatConductionSolver> =
                SolverRegistry::construct(key, &wall, floor_area)
                    .unwrap_or_else(|e| panic!("construct('{key}') must succeed: {e}"));
            assert_eq!(
                solver.name(),
                expected_name,
                "construct('{key}') returned wrong solver name"
            );
            assert!(
                solver.is_valid(),
                "construct('{key}') returned an invalid solver"
            );
        }
    }

    /// `ctf` and `fd` keys must be reachable directly (Issue #2494 core fix)
    /// and produce finite flux through the trait boundary.
    #[test]
    fn test_issue_2494_ctf_and_fd_keys_step_to_finite_flux() {
        use crate::physics::units::{FromF64, HeatTransferCoefficient, Temperature, Time, ToF64};

        let wall = wall_200mm_concrete();
        let floor_area = 54.0;

        for key in [registry_keys::CTF, registry_keys::FD] {
            let mut solver: Box<dyn HeatConductionSolver> =
                SolverRegistry::construct(key, &wall, floor_area)
                    .unwrap_or_else(|e| panic!("construct('{key}') must succeed: {e}"));
            let flux = solver
                .step(
                    Time::from_value(3600.0),
                    Temperature::from_value(22.0),
                    Temperature::from_value(5.0),
                    HeatTransferCoefficient::from_value(8.0),
                    HeatTransferCoefficient::from_value(25.0),
                )
                .unwrap_or_else(|e| panic!("step() for '{key}' must succeed: {e}"));
            assert!(
                flux.to_value().is_finite(),
                "construct('{key}') step flux must be finite"
            );
        }
    }

    /// Unknown keys still return `SolverError::InvalidConfig`, now listing both
    /// built-in and registered keys in the message.
    #[test]
    fn test_issue_2494_unknown_key_still_errors() {
        let wall = wall_200mm_concrete();
        let floor_area = 54.0;
        let err = SolverRegistry::construct("definitely_not_a_solver", &wall, floor_area);
        match err {
            Err(e) => assert!(
                e.to_string().contains("unknown solver key"),
                "expected unknown-key error, got: {e}"
            ),
            Ok(_) => panic!("unknown solver key must return Err"),
        }
    }

    /// `BUILTIN_KEYS` must be the four documented keys in stable order — guards
    /// against accidental removal of a key (the original Issue #2494 regression).
    #[test]
    fn test_issue_2494_builtin_keys_are_the_four_documented() {
        assert_eq!(
            registry_keys::BUILTIN_KEYS,
            &["5r1c", "ctf", "fd", "multinode_9r4c"],
            "registry_keys::BUILTIN_KEYS must list all four documented solver keys"
        );
    }

    /// Pluggable registration: a custom factory is dispatched by `construct`
    /// exactly like a built-in. Uses a unique key to avoid collisions with any
    /// parallel test and cleans up afterwards.
    #[test]
    fn test_issue_2494_register_custom_factory_is_dispatched() {
        use crate::physics::units::{
            FromF64, HeatFlux, HeatTransferCoefficient, Temperature, Time,
        };

        const KEY: &str = "test_2494_custom_surrogate";

        struct SurrogateSolver;
        impl HeatConductionSolver for SurrogateSolver {
            fn name(&self) -> &str {
                "Surrogate"
            }
            fn initialize(&mut self, _wall: &WallSpec) -> Result<(), SolverError> {
                Ok(())
            }
            fn step(
                &mut self,
                _dt: Time,
                _t_int: Temperature,
                _t_ext: Temperature,
                _h_int: HeatTransferCoefficient,
                _h_ext: HeatTransferCoefficient,
            ) -> Result<HeatFlux, SolverError> {
                Ok(HeatFlux::from_value(0.0))
            }
            fn energy_storage_rate(&self) -> f64 {
                0.0
            }
            fn is_valid(&self) -> bool {
                true
            }
        }

        // Clean slate for this key (defensive against re-runs).
        let _ = SolverRegistry::unregister_solver(KEY);

        SolverRegistry::register_solver(KEY, |_wall, _floor_area| {
            Ok(Box::new(SurrogateSolver) as Box<dyn HeatConductionSolver>)
        })
        .expect("register_solver must accept a custom factory");

        assert!(
            SolverRegistry::is_known_key(KEY),
            "registered custom key must be known"
        );
        assert!(
            SolverRegistry::registered_keys().iter().any(|k| k == KEY),
            "registered_keys() must list the custom key"
        );

        let wall = wall_200mm_concrete();
        let solver: Box<dyn HeatConductionSolver> = SolverRegistry::construct(KEY, &wall, 54.0)
            .expect("construct must dispatch custom key");
        assert_eq!(solver.name(), "Surrogate");
        assert!(solver.is_valid());

        assert!(
            SolverRegistry::unregister_solver(KEY),
            "unregister_solver must remove the custom key"
        );
        assert!(
            !SolverRegistry::is_known_key(KEY),
            "after unregister the custom key must be unknown"
        );
    }

    /// Built-in keys cannot be shadowed by registration.
    #[test]
    fn test_issue_2494_cannot_register_builtin_key() {
        let factory = |_wall: &WallSpec, _floor_area: f64| {
            Err(SolverError::InvalidConfig("never called".to_string()))
        };
        for key in registry_keys::BUILTIN_KEYS {
            let err = SolverRegistry::register_solver(key, factory)
                .err()
                .unwrap_or_else(|| panic!("register_solver('{key}') must be rejected"));
            assert!(
                err.to_string().contains("built-in key"),
                "expected built-in rejection for '{key}', got: {err}"
            );
        }
    }

    /// Re-registering the same custom key is rejected (must unregister first).
    #[test]
    fn test_issue_2494_cannot_double_register_custom_key() {
        const KEY: &str = "test_2494_double_register";
        let _ = SolverRegistry::unregister_solver(KEY);

        let factory = |_wall: &WallSpec, _floor_area: f64| {
            Err(SolverError::InvalidConfig("never called".to_string()))
        };
        SolverRegistry::register_solver(KEY, factory).expect("first register must succeed");
        let err = SolverRegistry::register_solver(KEY, factory)
            .expect_err("second register must be rejected");
        assert!(
            err.to_string().contains("already registered"),
            "expected already-registered error, got: {err}"
        );

        assert!(SolverRegistry::unregister_solver(KEY));
    }

    /// Built-in keys are never removable via `unregister_solver`.
    #[test]
    fn test_issue_2494_builtin_keys_not_removable() {
        for key in registry_keys::BUILTIN_KEYS {
            assert!(
                !SolverRegistry::unregister_solver(key),
                "built-in key '{key}' must not be removable"
            );
            assert!(
                SolverRegistry::is_known_key(key),
                "built-in key '{key}' must remain known"
            );
        }
    }
}
