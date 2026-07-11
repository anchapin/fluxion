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

use crate::physics::five_r1c_solver::FiveR1CSolver;
use crate::physics::multi_node_solver::MultiNodeSolver;
use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
use crate::physics::wall_spec::WallSpec;
// Issue #1349 (Phase 2 crate split): `BuildingAssembly` moved to `fluxion_core::assembly`.
use fluxion_core::assembly::BuildingAssembly;
use std::collections::HashMap;

pub struct SolverRegistry {
    solvers: HashMap<usize, Box<dyn HeatConductionSolver>>,
    wall_assemblies: HashMap<usize, BuildingAssembly>,
    method_counts: HashMap<String, usize>,
}

/// Issue #1429 — Registered constructor keys for `SolverRegistry::construct`.
///
/// Each variant maps a stable string key (used by callers like
/// `SolverManager::select`) to a `Box<dyn HeatConductionSolver>` constructor
/// taking a `&WallSpec`. The keys are the public API and must stay stable
/// across releases.
pub mod registry_keys {
    /// Lumped-capacitance 5R1C solver (default for low-mass constructions).
    pub const FIVE_R1C: &str = "5r1c";
    /// 9R4C four-node envelope solver (default for high-mass Case 900+,
    /// per ADR-002 and Issue #1429).
    pub const MULTINODE_9R4C: &str = "multinode_9r4c";
}

impl SolverRegistry {
    pub fn new() -> Self {
        Self {
            solvers: HashMap::new(),
            wall_assemblies: HashMap::new(),
            method_counts: HashMap::new(),
        }
    }

    /// Issue #1429 — Construct a `Box<dyn HeatConductionSolver>` by registry
    /// key from a `&WallSpec`.
    ///
    /// Supported keys (see [`registry_keys`]):
    /// - `"5r1c"` → `FiveR1CSolver` initialized on `wall`
    /// - `"multinode_9r4c"` → `MultiNodeSolver::from_wall_spec(wall)`,
    ///   the 9R4C four-node envelope solver (Issue #1429 drop-in)
    ///
    /// Unknown keys return `SolverError::InvalidConfig`. The returned
    /// `Box<dyn HeatConductionSolver>` can be wrapped in a
    /// `PhysicsSurfaceFluxProvider::add_surface` (Issue #1409 wiring)
    /// the same way `FiveR1CSolver` is wrapped today.
    pub fn construct(
        key: &str,
        wall: &WallSpec,
    ) -> Result<Box<dyn HeatConductionSolver>, SolverError> {
        match key {
            registry_keys::FIVE_R1C => {
                let mut solver = FiveR1CSolver::new();
                solver.initialize(wall)?;
                Ok(Box::new(solver))
            }
            registry_keys::MULTINODE_9R4C => Ok(MultiNodeSolver::boxed_from_wall_spec(wall)),
            other => Err(SolverError::InvalidConfig(format!(
                "SolverRegistry::construct: unknown solver key '{other}'. \
                 Supported keys: '{}', '{}'.",
                registry_keys::FIVE_R1C,
                registry_keys::MULTINODE_9R4C
            ))),
        }
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
        let solver: Box<dyn HeatConductionSolver> =
            SolverRegistry::construct(registry_keys::MULTINODE_9R4C, &wall)
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
        let err = SolverRegistry::construct("nonexistent_solver", &wall);
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

        // Construct BOTH solvers via the registry so the trait's
        // `steady_state_flux` (Quantity-typed) is called consistently.
        let mut r1c: Box<dyn HeatConductionSolver> =
            SolverRegistry::construct(registry_keys::FIVE_R1C, &wall).expect("5r1c construct");
        let mut multi: Box<dyn HeatConductionSolver> =
            SolverRegistry::construct(registry_keys::MULTINODE_9R4C, &wall)
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

        let mut r1c: Box<dyn HeatConductionSolver> =
            SolverRegistry::construct(registry_keys::FIVE_R1C, &wall).expect("5r1c construct");
        let mut multi: Box<dyn HeatConductionSolver> =
            SolverRegistry::construct(registry_keys::MULTINODE_9R4C, &wall)
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

        // Build via the canonical mode-aware constructor, then box it.
        let solver = MultiNodeSolver::from_wall_spec_with_mode(
            &wall,
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
}
