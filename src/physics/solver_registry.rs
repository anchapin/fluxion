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

use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
use crate::sim::assembly::BuildingAssembly;
use std::collections::HashMap;

pub struct SolverRegistry {
    solvers: HashMap<usize, Box<dyn HeatConductionSolver>>,
    wall_assemblies: HashMap<usize, BuildingAssembly>,
    method_counts: HashMap<String, usize>,
}

impl SolverRegistry {
    pub fn new() -> Self {
        Self {
            solvers: HashMap::new(),
            wall_assemblies: HashMap::new(),
            method_counts: HashMap::new(),
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

    #[test]
    fn test_solver_registry_new() {
        let registry = SolverRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_solver_registry_insert_and_get() {
        use crate::physics::five_r1c_solver::FiveR1CSolver;
        use crate::sim::assembly::{AssemblyBuilder, ConcreteMaterial};

        let mut registry = SolverRegistry::new();

        let wall = AssemblyBuilder::new("Test".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .build()
            .unwrap();

        let mut solver = FiveR1CSolver::new();
        solver.initialize(&wall).unwrap();
        let boxed: Box<dyn HeatConductionSolver> = Box::new(solver);

        registry.insert(0, boxed, wall.clone(), "5R1C".to_string());

        assert_eq!(registry.len(), 1);
        assert!(registry.contains(&0));
        assert!(registry.get_solver(0).is_some());
    }

    #[test]
    fn test_solver_registry_clear() {
        use crate::physics::five_r1c_solver::FiveR1CSolver;
        use crate::sim::assembly::{AssemblyBuilder, ConcreteMaterial};

        let mut registry = SolverRegistry::new();

        let wall = AssemblyBuilder::new("Test".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .build()
            .unwrap();

        let mut solver = FiveR1CSolver::new();
        solver.initialize(&wall).unwrap();
        let boxed: Box<dyn HeatConductionSolver> = Box::new(solver);

        registry.insert(0, boxed, wall, "5R1C".to_string());
        assert_eq!(registry.len(), 1);

        registry.clear();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_solver_registry_method_counts() {
        use crate::physics::five_r1c_solver::FiveR1CSolver;
        use crate::sim::assembly::{AssemblyBuilder, ConcreteMaterial};

        let mut registry = SolverRegistry::new();

        let wall = AssemblyBuilder::new("Test".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .build()
            .unwrap();

        let mut solver1 = FiveR1CSolver::new();
        solver1.initialize(&wall).unwrap();
        let solver2 = FiveR1CSolver::new();
        let boxed1: Box<dyn HeatConductionSolver> = Box::new(solver1);
        let boxed2: Box<dyn HeatConductionSolver> = Box::new(solver2);

        registry.insert(0, boxed1, wall.clone(), "5R1C".to_string());
        registry.insert(1, boxed2, wall, "5R1C".to_string());

        assert_eq!(registry.method_counts().get("5R1C"), Some(&2));
    }
}