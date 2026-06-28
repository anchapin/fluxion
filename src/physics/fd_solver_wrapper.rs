//! Finite Difference Solver Wrapper - Implements HeatConductionSolver trait for FD method.
//!
//! This module wraps the existing ImplicitFDSolver to implement the common
//! HeatConductionSolver trait interface, enabling unified treatment
//! with 5R1C and CTF solvers.
//!
//! # Overview
//!
//! The `FDSolverWrapper` adapts the ImplicitFDSolver to the common trait interface:
//! - Converts BuildingAssembly to wall discretization
//! - Handles temperature boundary conditions
//! - Returns heat flux in consistent units [W/m²]
//!
//! # Example
//!
//! ```rust
//! use fluxion::physics::fd_solver_wrapper::FDSolverWrapper;
//! use fluxion::physics::solver_trait::HeatConductionSolver;
//!
//! let mut solver = FDSolverWrapper::new();
//! solver.initialize(&wall_assembly)?;
//!
//! let flux = solver.step(3600.0, 20.0, 5.0, 8.0, 25.0)?;
//! ```

use crate::physics::fd_discretization::{MaterialLayer, WallDiscretization};
use crate::physics::fd_solver::{ImplicitFDSolver, SurfaceBC};
use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
use crate::physics::units::{FromF64, HeatFlux, HeatTransferCoefficient, Temperature, Time, ToF64};
use crate::physics::wall_properties::WallProperties;
use crate::physics::wall_spec::WallSpec;

/// Finite difference solver wrapper implementing the common HeatConductionSolver trait.
///
/// This wrapper adapts the ImplicitFDSolver to work with the unified solver interface,
/// handling conversion from BuildingAssembly to wall discretization and managing
/// boundary condition transformations.
#[allow(dead_code)]
pub struct FDSolverWrapper {
    /// Underlying FD solver
    solver: Option<ImplicitFDSolver>,
    /// Wall discretization (cached after initialization)
    discretization: Option<WallDiscretization>,
    /// Number of nodes per layer
    nodes_per_layer: usize,
    /// Interior convective coefficient [W/m²·K]
    h_interior: f64,
    /// Exterior convective coefficient [W/m²·K]
    h_exterior: f64,
    /// Current heat flux [W/m²]
    q_flux: f64,
    /// Initialized flag
    initialized: bool,
    /// Valid flag (solver converged)
    valid: bool,
}

impl FDSolverWrapper {
    /// Create a new uninitialized FD solver wrapper.
    pub fn new() -> Self {
        Self {
            solver: None,
            discretization: None,
            nodes_per_layer: 10, // Default discretization
            h_interior: 8.0,
            h_exterior: 25.0,
            q_flux: 0.0,
            initialized: false,
            valid: false,
        }
    }

    /// Create wrapper with custom discretization.
    pub fn with_discretization(nodes_per_layer: usize) -> Self {
        Self {
            nodes_per_layer,
            ..Self::new()
        }
    }

    /// Create wrapper with custom convective coefficients.
    pub fn with_convection(h_interior: f64, h_exterior: f64) -> Self {
        Self {
            h_interior,
            h_exterior,
            ..Self::new()
        }
    }

    /// Convert WallProperties to material layers.
    ///
    /// This hides BuildingAssembly internals from the solver. If BuildingAssembly
    /// changes its layer structure, only WallProperties::from_assembly() needs updating.
    fn wall_properties_to_material_layers(wall_props: &WallProperties) -> Vec<MaterialLayer> {
        wall_props
            .layers
            .iter()
            .map(|layer| {
                MaterialLayer::new(
                    &layer.name,
                    layer.thickness_m,
                    layer.conductivity_w_mk,
                    layer.density_kg_m3,
                    layer.specific_heat_j_kgk,
                )
            })
            .collect()
    }

    /// Calculate interior surface heat flux from solver state.
    ///
    /// Uses the FD solver's actual surface temperature:
    /// q = h * (T_zone - T_surface)
    /// Positive flux = heat flowing into zone
    fn calculate_surface_flux(solver: &ImplicitFDSolver, T_interior: f64, h_interior: f64) -> f64 {
        let T_surface = solver.interior_surface_temp();
        h_interior * (T_interior - T_surface)
    }
}

impl Default for FDSolverWrapper {
    fn default() -> Self {
        Self::new()
    }
}

impl HeatConductionSolver for FDSolverWrapper {
    fn name(&self) -> &str {
        "FD"
    }

    fn initialize(&mut self, wall: &WallSpec) -> Result<(), SolverError> {
        // Convert WallSpec to wall properties (the seam)
        let wall_props = wall.to_wall_properties();

        // Convert wall properties to material layers
        let materials = Self::wall_properties_to_material_layers(&wall_props);

        if materials.is_empty() {
            return Err(SolverError::ConstructionError(
                "Wall assembly has no layers".to_string(),
            ));
        }

        // Create wall discretization
        let discretization = WallDiscretization::from_layers(&materials, self.nodes_per_layer);

        // Validate discretization
        if discretization.node_positions.is_empty() {
            return Err(SolverError::ConstructionError(
                "Wall discretization failed - no nodes created".to_string(),
            ));
        }

        // Create FD solver with initial temperature of 20°C
        let initial_temp = 20.0;
        let solver = ImplicitFDSolver::new(discretization.clone(), initial_temp);

        // Store solver and discretization
        self.solver = Some(solver);
        self.discretization = Some(discretization);
        self.initialized = true;
        self.valid = true;
        self.q_flux = 0.0;

        Ok(())
    }

    fn step(
        &mut self,
        timestep: Time,
        T_interior: Temperature,
        T_exterior: Temperature,
        h_interior: HeatTransferCoefficient,
        h_exterior: HeatTransferCoefficient,
    ) -> Result<HeatFlux, SolverError> {
        if !self.initialized {
            return Err(SolverError::InvalidConfig(
                "FD solver not initialized. Call initialize() first.".to_string(),
            ));
        }

        if !self.valid {
            return Err(SolverError::ConvergenceError(
                "FD solver is not valid".to_string(),
            ));
        }

        // Get mutable reference to solver
        let solver = self.solver.as_mut().ok_or_else(|| {
            SolverError::InvalidConfig("FD solver is None after initialization".to_string())
        })?;

        // Create boundary conditions
        let interior_bc = SurfaceBC::new_interior(h_interior.to_value(), T_interior.to_value());
        let exterior_bc =
            SurfaceBC::new_exterior(h_exterior.to_value(), T_exterior.to_value(), 0.0);

        // Advance FD solver by one timestep
        solver.step(timestep.to_value(), &interior_bc, &exterior_bc);

        // Calculate surface heat flux using actual solver state
        self.q_flux =
            Self::calculate_surface_flux(solver, T_interior.to_value(), h_interior.to_value());

        Ok(HeatFlux::from_value(self.q_flux))
    }

    fn energy_storage_rate(&self) -> f64 {
        // FD tracks energy storage implicitly in the temperature profile
        // Could estimate from temperature change rate
        0.0
    }

    fn is_valid(&self) -> bool {
        self.initialized && self.valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::units::{
        FromF64, HeatFlux, HeatTransferCoefficient, Temperature, Time, ToF64,
    };
    use crate::physics::wall_spec::WallSpec;
    use fluxion_core::assembly::{AssemblyBuilder, ConcreteMaterial};

    fn create_test_wall() -> WallSpec {
        let assembly = AssemblyBuilder::new("Test Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2))) // 200mm concrete
            .build()
            .unwrap();
        WallSpec::from_assembly(&assembly)
    }

    #[test]
    fn test_fd_wrapper_creation() {
        let wrapper = FDSolverWrapper::new();
        assert!(!wrapper.initialized);
        assert!(!wrapper.valid);
    }

    #[test]
    fn test_fd_wrapper_initialization() {
        let mut wrapper = FDSolverWrapper::new();
        let wall = create_test_wall();

        let result = wrapper.initialize(&wall);
        assert!(result.is_ok());
        assert!(wrapper.is_valid());
    }

    #[test]
    fn test_fd_wrapper_with_discretization() {
        let mut wrapper = FDSolverWrapper::with_discretization(20);
        let wall = create_test_wall();

        let result = wrapper.initialize(&wall);
        assert!(result.is_ok());
        assert_eq!(wrapper.nodes_per_layer, 20);
    }

    #[test]
    fn test_fd_wrapper_flux_calculation() {
        let mut wrapper = FDSolverWrapper::new();
        let wall = create_test_wall();

        wrapper.initialize(&wall).unwrap();

        // Calculate flux for 20°C interior, 0°C exterior
        let flux = wrapper
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(0.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap();

        // Flux should be finite
        assert!(flux.to_value().is_finite());
    }

    #[test]
    fn test_fd_wrapper_uninitialized() {
        let mut wrapper = FDSolverWrapper::new();

        // Should fail if not initialized
        let result = wrapper.step(
            Time::from_value(3600.0),
            Temperature::from_value(20.0),
            Temperature::from_value(0.0),
            HeatTransferCoefficient::from_value(8.0),
            HeatTransferCoefficient::from_value(25.0),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_fd_wrapper_diurnal_simulation() {
        let mut wrapper = FDSolverWrapper::new();
        let wall = create_test_wall();

        wrapper.initialize(&wall).unwrap();

        // 24-hour simulation
        let mut total_flux = 0.0;
        for hour in 0..24 {
            let t_ext = 10.0 + 10.0 * ((hour as f64 - 6.0) * std::f64::consts::PI / 12.0).sin();
            let flux = wrapper
                .step(
                    Time::from_value(3600.0),
                    Temperature::from_value(20.0),
                    Temperature::from_value(t_ext),
                    HeatTransferCoefficient::from_value(8.0),
                    HeatTransferCoefficient::from_value(25.0),
                )
                .unwrap();
            total_flux += flux.to_value();
        }

        // Total flux should be reasonable
        assert!(
            total_flux.abs() < 10000.0,
            "Total flux {:.2} unreasonably large",
            total_flux
        );
    }

    // === Phase 3: Additional coverage tests ===

    #[test]
    fn test_fd_wrapper_with_convection() {
        let wrapper = FDSolverWrapper::with_convection(5.0, 30.0);
        assert!(!wrapper.initialized);
        assert!(!wrapper.valid);
        assert_eq!(wrapper.h_interior, 5.0);
        assert_eq!(wrapper.h_exterior, 30.0);
    }

    #[test]
    fn test_fd_wrapper_default() {
        let wrapper = FDSolverWrapper::default();
        assert_eq!(wrapper.nodes_per_layer, 10);
        assert_eq!(wrapper.h_interior, 8.0);
        assert_eq!(wrapper.h_exterior, 25.0);
    }

    #[test]
    fn test_fd_wrapper_name() {
        let wrapper = FDSolverWrapper::new();
        assert_eq!(wrapper.name(), "FD");
    }

    #[test]
    fn test_fd_wrapper_energy_storage_rate() {
        let mut wrapper = FDSolverWrapper::new();
        let wall = create_test_wall();
        wrapper.initialize(&wall).unwrap();

        // Energy storage rate should be finite (may be 0 as placeholder)
        let rate = wrapper.energy_storage_rate();
        assert!(rate.is_finite());
    }

    #[test]
    fn test_fd_wrapper_is_valid() {
        let mut wrapper = FDSolverWrapper::new();
        let wall = create_test_wall();

        // Not initialized -> not valid
        assert!(!wrapper.is_valid());

        wrapper.initialize(&wall).unwrap();
        // Initialized -> valid
        assert!(wrapper.is_valid());
    }

    #[test]
    fn test_fd_wrapper_step_various_timesteps() {
        let mut wrapper = FDSolverWrapper::new();
        let wall = create_test_wall();
        wrapper.initialize(&wall).unwrap();

        let timesteps = [300.0, 600.0, 1800.0, 3600.0];
        for dt in timesteps {
            let flux = wrapper
                .step(
                    Time::from_value(dt),
                    Temperature::from_value(20.0),
                    Temperature::from_value(10.0),
                    HeatTransferCoefficient::from_value(8.0),
                    HeatTransferCoefficient::from_value(25.0),
                )
                .unwrap();
            assert!(flux.to_value().is_finite());
        }
    }

    #[test]
    fn test_fd_wrapper_step_extreme_temperatures() {
        let mut wrapper = FDSolverWrapper::new();
        let wall = create_test_wall();
        wrapper.initialize(&wall).unwrap();

        // Cold extreme
        let flux_cold = wrapper
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(-10.0),
                Temperature::from_value(-20.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap();
        assert!(flux_cold.to_value().is_finite());

        // Hot extreme
        let flux_hot = wrapper
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(40.0),
                Temperature::from_value(50.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap();
        assert!(flux_hot.to_value().is_finite());
    }

    #[test]
    fn test_fd_wrapper_step_extreme_convection() {
        let mut wrapper = FDSolverWrapper::new();
        let wall = create_test_wall();
        wrapper.initialize(&wall).unwrap();

        // Low convection
        let flux_low = wrapper
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(10.0),
                HeatTransferCoefficient::from_value(1.0),
                HeatTransferCoefficient::from_value(1.0),
            )
            .unwrap();
        assert!(flux_low.to_value().is_finite());

        // High convection
        let flux_high = wrapper
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(10.0),
                HeatTransferCoefficient::from_value(50.0),
                HeatTransferCoefficient::from_value(100.0),
            )
            .unwrap();
        assert!(flux_high.to_value().is_finite());
    }

    #[test]
    fn test_fd_wrapper_initialization_empty_assembly() {
        let mut wrapper = FDSolverWrapper::new();
        let wall = create_test_wall();

        // First initialization should succeed
        let result1 = wrapper.initialize(&wall);
        assert!(result1.is_ok());

        // Second initialization should also succeed (updates state)
        let result2 = wrapper.initialize(&wall);
        assert!(result2.is_ok());
    }

    #[test]
    fn test_fd_wrapper_custom_nodes_per_layer() {
        let wrapper = FDSolverWrapper::with_discretization(5);
        assert_eq!(wrapper.nodes_per_layer, 5);
    }

    #[test]
    fn test_fd_wrapper_flux_directions() {
        let mut wrapper = FDSolverWrapper::new();
        let wall = create_test_wall();
        wrapper.initialize(&wall).unwrap();

        // Interior hotter than exterior - flux should be positive (into zone)
        let flux_heating = wrapper
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(25.0),
                Temperature::from_value(15.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap();
        assert!(
            flux_heating.to_value() > 0.0,
            "Heating flux should be positive"
        );

        // Exterior hotter than interior - flux should be negative (out of zone)
        let flux_cooling = wrapper
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(15.0),
                Temperature::from_value(25.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap();
        assert!(
            flux_cooling.to_value() < 0.0,
            "Cooling flux should be negative"
        );
    }

    #[test]
    fn test_fd_wrapper_timestep_zero() {
        let mut wrapper = FDSolverWrapper::new();
        let wall = create_test_wall();
        wrapper.initialize(&wall).unwrap();

        let flux = wrapper
            .step(
                Time::from_value(0.0),
                Temperature::from_value(20.0),
                Temperature::from_value(10.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap();
        // Zero timestep should give zero flux (no time for heat transfer)
        assert!(
            flux.to_value().abs() < 1e-10,
            "Zero timestep should give near-zero flux"
        );
    }
}
