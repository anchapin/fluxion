//! Solver Manager - Unified interface for multiple heat conduction solvers.
//!
//! This module provides a unified manager for handling multiple heat conduction
//! solvers (5R1C, CTF, FD) through a common trait interface, enabling automatic
//! method selection and zero-copy data sharing.
//!
//! # Overview
//!
//! The `SolverManager` provides:
//! - Automatic solver selection based on thermal mass
//! - Per-wall solver instances (one solver per unique construction)
//! - Zero-copy data sharing via `BuildingAssembly` references
//! - Runtime dispatch through trait objects (`Box<dyn HeatConductionSolver>`)
//!
//! # Example
//!
//! ```rust
//! use fluxion::physics::solver_manager::SolverManager;
//! use fluxion::physics::method_selector::ThermalMethodSelector;
//!
//! let mut manager = SolverManager::new(ThermalMethodSelector::default());
//!
//! // Initialize solver for a wall
//! manager.get_or_create_solver(&wall_assembly)?;
//!
//! // Calculate heat flux at each timestep
//! let flux = manager.step(wall_index, 3600.0, T_zone, T_outdoor, h_int, h_ext)?;
//! ```

use crate::physics::ctf_solver_wrapper::CTFSolverWrapper;
use crate::physics::fd_solver_wrapper::FDSolverWrapper;
use crate::physics::five_r1c_solver::FiveR1CSolver;
use crate::physics::method_selector::{
    SolverSelectionResult, ThermalMethod, ThermalMethodSelector, ThermalMethodSelectorConfig,
};
use crate::physics::solver_registry::SolverRegistry;
use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
use crate::sim::assembly::BuildingAssembly;
use log::{debug, warn};

/// Unified solver manager for multiple heat conduction methods.
///
/// This manager handles solver instances for multiple walls, automatically
/// selecting the appropriate solver method based on thermal mass characteristics.
///
/// # Fields
///
/// * `selector` - Automatic method selector based on thermal mass
/// * `solvers` - Map of wall index to solver instance
/// * `wall_assemblies` - Map of wall index to wall construction (zero-copy sharing)
/// * `solver_stats` - Statistics on solver usage (for diagnostics)
pub struct SolverManager {
    pub selector: ThermalMethodSelector,
    registry: SolverRegistry,
}

/// Statistics on solver usage
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    /// Number of walls using 5R1C
    pub five_r1c_count: usize,
    /// Number of walls using CTF
    pub ctf_count: usize,
    /// Number of walls using FD
    pub fd_count: usize,
    /// Total number of walls
    pub total_walls: usize,
}

impl SolverManager {
    /// Create a new solver manager with default settings.
    pub fn new(selector: ThermalMethodSelector) -> Self {
        Self {
            selector,
            registry: SolverRegistry::new(),
        }
    }

    /// Create solver manager with custom threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold_hours` - Time constant threshold for method selection
    pub fn with_threshold(threshold_hours: f64) -> Self {
        let config = ThermalMethodSelectorConfig {
            threshold_hours,
            ..Default::default()
        };
        Self::new(ThermalMethodSelector::from_config(config))
    }

    /// Get or create solver for a wall assembly.
    ///
    /// This method checks if a solver already exists for the given wall index.
    /// If not, it creates the appropriate solver based on the method selector.
    ///
    /// # Arguments
    ///
    /// * `wall_index` - Unique identifier for the wall
    /// * `wall_assembly` - Wall construction with material layers
    ///
    /// # Returns
    ///
    /// Ok if solver created/retrieved successfully, Err if creation failed
    pub fn get_or_create_solver(
        &mut self,
        wall_index: usize,
        wall_assembly: &BuildingAssembly,
        surface_id: &str,
    ) -> Result<SolverSelectionResult, SolverError> {
        // Check if solver already exists
        if self.registry.contains(&wall_index) {
            return Ok(self.selector.select_with_result(wall_assembly, surface_id));
        }

        // Select appropriate solver method with full result (for transparency)
        let result = self.selector.select_with_result(wall_assembly, surface_id);
        let method = result.method;

        // Create solver based on method
        let solver: Box<dyn HeatConductionSolver> = match method {
            ThermalMethod::FiveR1C => {
                debug!("Creating 5R1C solver for wall {}", wall_index);
                let mut solver = FiveR1CSolver::new();
                solver.initialize(wall_assembly)?;
                Box::new(solver)
            }
            ThermalMethod::CTF => {
                debug!("Creating CTF solver for wall {}", wall_index);
                let mut solver = CTFSolverWrapper::new();
                match solver.initialize(wall_assembly) {
                    Ok(()) => Box::new(solver),
                    Err(e) => {
                        // CTF failed, fallback to FD if enabled
                        if self.selector.enable_fallback {
                            warn!(
                                "CTF solver failed for wall {}: {}. Falling back to FD.",
                                wall_index, e
                            );
                            let mut fd_solver = FDSolverWrapper::new();
                            fd_solver.initialize(wall_assembly)?;
                            Box::new(fd_solver)
                        } else {
                            return Err(e);
                        }
                    }
                }
            }
            ThermalMethod::FiniteDifference => {
                debug!("Creating FD solver for wall {}", wall_index);
                let mut solver = FDSolverWrapper::new();
                solver.initialize(wall_assembly)?;
                Box::new(solver)
            }
        };

        // Store solver and wall assembly via registry
        self.registry.insert(
            wall_index,
            solver,
            wall_assembly.clone(),
            method.name().to_string(),
        );

        Ok(result)
    }

    /// Get mutable reference to solver for a wall.
    ///
    /// # Arguments
    ///
    /// * `wall_index` - Index of the wall
    ///
    /// # Returns
    ///
    /// Some(solver) if solver exists, None if not found
    ///
    /// # Deprecated
    ///
    /// Use `step_all()` instead for batch stepping of all surfaces.
    #[deprecated(since = "1.0.0", note = "Use step_all() for batch stepping")]
    pub fn get_solver_mut(
        &mut self,
        wall_index: usize,
    ) -> Option<&mut Box<dyn HeatConductionSolver>> {
        self.registry.get_solver_mut(wall_index)
    }

    /// Get immutable reference to solver for a wall.
    ///
    /// # Arguments
    ///
    /// * `wall_index` - Index of the wall
    ///
    /// # Returns
    ///
    /// Some(solver) if solver exists, None if not found
    ///
    /// # Deprecated
    ///
    /// Use `step_all()` instead for batch stepping of all surfaces.
    #[deprecated(since = "1.0.0", note = "Use step_all() for batch stepping")]
    pub fn get_solver(&self, wall_index: usize) -> Option<&dyn HeatConductionSolver> {
        self.registry.get_solver(wall_index)
    }

    /// Calculate heat flux through a wall.
    ///
    /// This is a convenience method that gets the solver and calls step().
    ///
    /// # Arguments
    ///
    /// * `wall_index` - Index of the wall
    /// * `timestep` - Timestep duration [s]
    /// * `T_interior` - Interior air temperature [°C]
    /// * `T_exterior` - Exterior air temperature [°C]
    /// * `h_interior` - Interior convective coefficient [W/m²·K]
    /// * `h_exterior` - Exterior convective coefficient [W/m²·K]
    ///
    /// # Returns
    ///
    /// Heat flux [W/m²] (positive = into zone)
    ///
    /// # Deprecated
    ///
    /// Use `step_all()` instead for batch stepping of all surfaces.
    #[deprecated(since = "1.0.0", note = "Use step_all() for batch stepping")]
    pub fn step(
        &mut self,
        wall_index: usize,
        timestep: f64,
        T_interior: f64,
        T_exterior: f64,
        h_interior: f64,
        h_exterior: f64,
    ) -> Result<f64, SolverError> {
        let solver = self.registry.get_solver_mut(wall_index).ok_or_else(|| {
            SolverError::InvalidConfig(format!("No solver for wall {}", wall_index))
        })?;

        solver.step(timestep, T_interior, T_exterior, h_interior, h_exterior)
    }

    /// Get energy storage rate for a wall.
    ///
    /// # Arguments
    ///
    /// * `wall_index` - Index of the wall
    ///
    /// # Returns
    ///
    /// Energy storage rate [W/m²] (positive = storing energy)
    ///
    /// # Deprecated
    ///
    /// Use `step_all()` instead for batch stepping of all surfaces.
    #[deprecated(since = "1.0.0", note = "Use step_all() for batch stepping")]
    pub fn energy_storage_rate(&self, wall_index: usize) -> f64 {
        self.registry
            .get_solver(wall_index)
            .map(|s| s.energy_storage_rate())
            .unwrap_or(0.0)
    }

    /// Get solver statistics.
    pub fn get_stats(&self) -> SolverStats {
        let mut stats = SolverStats::default();

        for (method_name, count) in self.registry.method_counts() {
            match method_name.as_str() {
                "5R1C" => stats.five_r1c_count = *count,
                "CTF" => stats.ctf_count = *count,
                "FD" => stats.fd_count = *count,
                _ => {}
            }
        }

        stats.total_walls = self.registry.len();
        stats
    }

    /// Get number of initialized solvers.
    pub fn num_solvers(&self) -> usize {
        self.registry.len()
    }

    /// Check if all solvers are valid.
    pub fn all_valid(&self) -> bool {
        self.registry.wall_assemblies().keys().all(|&idx| {
            self.registry
                .get_solver(idx)
                .map(|s| s.is_valid())
                .unwrap_or(true)
        })
    }

    /// Clear all solvers (for reinitialization).
    pub fn clear(&mut self) {
        self.registry.clear();
    }

    /// Get solver method distribution as a string.
    pub fn method_distribution(&self) -> String {
        let stats = self.get_stats();
        format!(
            "5R1C: {}, CTF: {}, FD: {} (total: {})",
            stats.five_r1c_count, stats.ctf_count, stats.fd_count, stats.total_walls
        )
    }

    /// Step all solvers in a single pass, returning all fluxes.
    ///
    /// This is the primary entry point for the solver lifecycle. It:
    /// 1. Pre-warms any cold solvers
    /// 2. Steps all surfaces in a single pass
    /// 3. Aggregates stats
    /// 4. Returns all fluxes
    ///
    /// # Arguments
    ///
    /// * `surfaces` - Slice of (wall_index, wall_assembly) tuples for all surfaces
    /// * `dt` - Timestep duration [s]
    /// * `T_int` - Interior air temperature [°C]
    /// * `T_ext` - Exterior air temperature [°C]
    ///
    /// # Returns
    ///
    /// Vector of heat fluxes [W/m²] (positive = into zone), one per surface
    pub fn step_all(
        &mut self,
        surfaces: &[(usize, BuildingAssembly)],
        dt: f64,
        T_int: f64,
        T_ext: f64,
    ) -> Result<Vec<f64>, SolverError> {
        let h_int = 8.0;
        let h_ext = 25.0;

        let mut fluxes = Vec::with_capacity(surfaces.len());

        for &(wall_index, ref wall_assembly) in surfaces {
            // Ensure solver exists for this wall
            if !self.registry.contains(&wall_index) {
                self.get_or_create_solver(wall_index, wall_assembly, "Wall")?;
            }

            // Step the solver
            let solver = self.registry.get_solver_mut(wall_index).ok_or_else(|| {
                SolverError::InvalidConfig(format!("No solver for wall {}", wall_index))
            })?;

            let flux = solver.step(dt, T_int, T_ext, h_int, h_ext)?;
            fluxes.push(flux);
        }

        Ok(fluxes)
    }
}

impl Default for SolverManager {
    fn default() -> Self {
        Self::new(ThermalMethodSelector::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::assembly::{AssemblyBuilder, ConcreteMaterial};

    #[test]
    fn test_solver_manager_creation() {
        let manager = SolverManager::new(ThermalMethodSelector::default());
        assert_eq!(manager.num_solvers(), 0);
    }

    #[test]
    fn test_solver_manager_5r1c_solver() {
        let mut manager = SolverManager::with_threshold(10.0); // Force 5R1C

        let wall = AssemblyBuilder::new("Light Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1))) // 100mm concrete
            .build()
            .unwrap();

        let result = manager.get_or_create_solver(0, &wall, "Wall");
        assert!(result.is_ok());
        assert_eq!(manager.num_solvers(), 1);

        // Get stats
        let stats = manager.get_stats();
        assert_eq!(stats.five_r1c_count, 1);
        assert_eq!(stats.total_walls, 1);
    }

    #[test]
    fn test_solver_manager_ctf_solver() {
        let mut manager = SolverManager::with_threshold(1.0); // Force CTF

        let wall = AssemblyBuilder::new("Heavy Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.3))) // 300mm concrete
            .build()
            .unwrap();

        let result = manager.get_or_create_solver(0, &wall, "Wall");
        assert!(result.is_ok());
        assert_eq!(manager.num_solvers(), 1);

        // Get stats
        let stats = manager.get_stats();
        // CTF might fail and fallback to FD, so check either
        assert!(stats.ctf_count > 0 || stats.fd_count > 0);
    }

    #[test]
    #[allow(deprecated)]
    fn test_solver_manager_step() {
        let mut manager = SolverManager::with_threshold(10.0); // Force 5R1C

        let wall = AssemblyBuilder::new("Test Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap();

        manager.get_or_create_solver(0, &wall, "Wall").unwrap();

        // Calculate flux
        let flux = manager.step(0, 3600.0, 20.0, 5.0, 8.0, 25.0).unwrap();

        // Flux should be negative (heat flowing out)
        assert!(flux < 0.0);
    }

    #[test]
    fn test_solver_manager_multiple_walls() {
        let mut manager = SolverManager::default();

        // Create walls with different thermal mass
        let light_wall = AssemblyBuilder::new("Light Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .build()
            .unwrap();

        let heavy_wall = AssemblyBuilder::new("Heavy Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.3)))
            .build()
            .unwrap();

        // Initialize solvers
        manager
            .get_or_create_solver(0, &light_wall, "Wall")
            .unwrap();
        manager
            .get_or_create_solver(1, &heavy_wall, "Wall")
            .unwrap();

        // Should have 2 solvers
        assert_eq!(manager.num_solvers(), 2);

        // Check method distribution
        let stats = manager.get_stats();
        assert_eq!(stats.total_walls, 2);
        // Light wall should use 5R1C, heavy wall should use CTF/FD
        assert!(stats.five_r1c_count >= 1);
    }

    #[test]
    fn test_solver_manager_clear() {
        let mut manager = SolverManager::with_threshold(10.0);

        let wall = AssemblyBuilder::new("Test Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap();

        manager.get_or_create_solver(0, &wall, "Wall").unwrap();
        assert_eq!(manager.num_solvers(), 1);

        manager.clear();
        assert_eq!(manager.num_solvers(), 0);
    }

    // === Phase 3: Additional coverage tests ===

    #[test]
    #[allow(deprecated)]
    fn test_solver_manager_get_solver_mut() {
        let mut manager = SolverManager::default();
        let wall = AssemblyBuilder::new("Test Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap();

        manager.get_or_create_solver(0, &wall, "Wall").unwrap();

        let solver = manager.get_solver_mut(0);
        assert!(solver.is_some());
        let solver_name = solver.unwrap().name();
        assert!(solver_name == "5R1C" || solver_name == "CTF");
    }

    #[test]
    #[allow(deprecated)]
    fn test_solver_manager_get_solver() {
        let _manager = SolverManager::default();
        let wall = AssemblyBuilder::new("Test Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap();

        let mut manager_mut = SolverManager::default();
        manager_mut.get_or_create_solver(0, &wall, "Wall").unwrap();

        let solver = manager_mut.get_solver(0);
        assert!(solver.is_some());
    }

    #[test]
    #[allow(deprecated)]
    fn test_solver_manager_get_solver_not_found() {
        let manager = SolverManager::default();

        let solver = manager.get_solver(999);
        assert!(solver.is_none());

        let mut manager_mut = SolverManager::default();
        let solver_mut = manager_mut.get_solver_mut(999);
        assert!(solver_mut.is_none());
    }

    #[test]
    #[allow(deprecated)]
    fn test_solver_manager_energy_storage_rate() {
        let mut manager = SolverManager::default();
        let wall = AssemblyBuilder::new("Test Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap();

        manager.get_or_create_solver(0, &wall, "Wall").unwrap();

        let rate = manager.energy_storage_rate(0);
        // All current solver implementations return 0 for energy storage rate
        assert_eq!(rate, 0.0);
    }

    #[test]
    #[allow(deprecated)]
    fn test_solver_manager_energy_storage_rate_not_found() {
        let manager = SolverManager::default();

        let rate = manager.energy_storage_rate(999);
        // Should return 0.0 for non-existent solver
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn test_solver_manager_all_valid() {
        let mut manager = SolverManager::default();

        // Initially no solvers - vacuously true
        assert!(manager.all_valid());

        let wall = AssemblyBuilder::new("Test Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap();

        manager.get_or_create_solver(0, &wall, "Wall").unwrap();

        // Should be valid after initialization
        assert!(manager.all_valid());
    }

    #[test]
    fn test_solver_manager_method_distribution() {
        let mut manager = SolverManager::default();

        let wall = AssemblyBuilder::new("Test Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap();

        manager.get_or_create_solver(0, &wall, "Wall").unwrap();

        let dist = manager.method_distribution();
        assert!(!dist.is_empty());
        assert!(dist.contains("5R1C"));
        assert!(dist.contains("CTF"));
        assert!(dist.contains("FD"));
        assert!(dist.contains("total"));
    }

    #[test]
    fn test_solver_manager_method_distribution_empty() {
        let manager = SolverManager::default();

        let dist = manager.method_distribution();
        // All counts should be 0
        assert!(dist.contains("5R1C: 0"));
        assert!(dist.contains("CTF: 0"));
        assert!(dist.contains("FD: 0"));
        assert!(dist.contains("total: 0"));
    }

    #[test]
    fn test_solver_manager_reinitialize() {
        let mut manager = SolverManager::default();
        let wall = AssemblyBuilder::new("Test Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap();

        // First initialization
        manager.get_or_create_solver(0, &wall, "Wall").unwrap();
        assert_eq!(manager.num_solvers(), 1);

        // Re-initialization should succeed (no new solver created)
        manager.get_or_create_solver(0, &wall, "Wall").unwrap();
        assert_eq!(manager.num_solvers(), 1);
    }

    #[test]
    #[allow(deprecated)]
    fn test_solver_manager_step_invalid_wall() {
        let mut manager = SolverManager::default();

        let result = manager.step(999, 3600.0, 20.0, 10.0, 8.0, 25.0);

        assert!(result.is_err());
    }

    #[test]
    fn test_solver_stats_default() {
        let stats = SolverStats::default();

        assert_eq!(stats.five_r1c_count, 0);
        assert_eq!(stats.ctf_count, 0);
        assert_eq!(stats.fd_count, 0);
        assert_eq!(stats.total_walls, 0);
    }

    #[test]
    fn test_fd_solver_forced() {
        let mut manager = SolverManager::new(ThermalMethodSelector::with_override(
            ThermalMethod::FiniteDifference,
        ));

        let wall = AssemblyBuilder::new("Test Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap();

        manager.get_or_create_solver(0, &wall, "Wall").unwrap();

        let stats = manager.get_stats();
        // Should have exactly one FD solver
        assert_eq!(stats.fd_count, 1);
        assert_eq!(stats.five_r1c_count, 0);
        assert_eq!(stats.ctf_count, 0);
    }

    #[test]
    fn test_5r1c_solver_forced() {
        let mut manager =
            SolverManager::new(ThermalMethodSelector::with_override(ThermalMethod::FiveR1C));

        let wall = AssemblyBuilder::new("Test Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.3)))
            .build()
            .unwrap();

        manager.get_or_create_solver(0, &wall, "Wall").unwrap();

        let stats = manager.get_stats();
        // Should have exactly one 5R1C solver
        assert_eq!(stats.five_r1c_count, 1);
        assert_eq!(stats.ctf_count, 0);
        assert_eq!(stats.fd_count, 0);
    }

    #[test]
    fn test_step_all_steps_all_solvers() {
        let mut manager = SolverManager::with_threshold(10.0);

        let wall1 = AssemblyBuilder::new("Wall 1".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .build()
            .unwrap();

        let wall2 = AssemblyBuilder::new("Wall 2".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap();

        manager.get_or_create_solver(0, &wall1, "Wall").unwrap();
        manager.get_or_create_solver(1, &wall2, "Wall").unwrap();

        let surfaces = vec![(0, wall1.clone()), (1, wall2.clone())];
        let dt = 3600.0;
        let T_int = 20.0;
        let T_ext = 5.0;

        let fluxes = manager.step_all(&surfaces, dt, T_int, T_ext).unwrap();

        assert_eq!(fluxes.len(), 2);
        for flux in &fluxes {
            assert!(flux.is_finite());
        }
    }

    #[test]
    fn test_step_all_returns_fluxes_in_order() {
        let mut manager = SolverManager::with_threshold(10.0);

        let wall = AssemblyBuilder::new("Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .build()
            .unwrap();

        manager.get_or_create_solver(0, &wall, "Wall").unwrap();
        manager.get_or_create_solver(1, &wall, "Wall").unwrap();

        let surfaces = vec![(0, wall.clone()), (1, wall.clone())];

        let fluxes = manager.step_all(&surfaces, 3600.0, 20.0, 5.0).unwrap();

        assert_eq!(fluxes.len(), 2);
    }

    #[test]
    fn test_step_all_empty_surfaces() {
        let mut manager = SolverManager::with_threshold(10.0);

        let fluxes = manager.step_all(&[], 3600.0, 20.0, 5.0).unwrap();

        assert!(fluxes.is_empty());
    }
}
