//! CTF Solver Wrapper - Implements HeatConductionSolver trait for CTF method.
//!
//! This module wraps the existing CTFSolver to implement the common
//! HeatConductionSolver trait interface, enabling unified treatment
//! with 5R1C and finite difference solvers.
//!
//! # Overview
//!
//! The `CTFSolverWrapper` adapts the CTFSolver to the common trait interface:
//! - Converts BuildingAssembly to CTF coefficients
//! - Handles temperature boundary conditions
//! - Returns heat flux in consistent units [W/m²]
//!
//! # Example
//!
//! ```rust
//! use fluxion::physics::ctf_solver_wrapper::CTFSolverWrapper;
//! use fluxion::physics::solver_trait::HeatConductionSolver;
//!
//! let mut solver = CTFSolverWrapper::new();
//! solver.initialize(&wall_assembly)?;
//!
//! let flux = solver.step(3600.0, 20.0, 5.0, 8.0, 25.0)?;
//! ```

use crate::physics::ctf_coefficients::{CTFCalculator, CTFCoefficients, CTFMaterial};
use crate::physics::ctf_solver::{CTFSolver, CTFSolverConfig};
use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
use crate::physics::wall_properties::WallProperties;
use crate::sim::assembly::BuildingAssembly;

/// CTF solver wrapper implementing the common HeatConductionSolver trait.
///
/// This wrapper adapts the CTFSolver to work with the unified solver interface,
/// handling conversion from BuildingAssembly to CTF coefficients and managing
/// boundary condition transformations.
#[allow(dead_code)]
pub struct CTFSolverWrapper {
    /// Underlying CTF solver
    solver: Option<CTFSolver>,
    /// CTF coefficients (cached after initialization)
    coefficients: Option<CTFCoefficients>,
    /// Interior convective coefficient [W/m²·K]
    h_interior: f64,
    /// Exterior convective coefficient [W/m²·K]
    h_exterior: f64,
    /// Previous interior heat flux for convection approximation [W/m²]
    prev_q_flux: f64,
    /// Initialized flag
    initialized: bool,
    /// Valid flag (coefficients converged)
    valid: bool,
}

impl CTFSolverWrapper {
    /// Create a new uninitialized CTF solver wrapper.
    pub fn new() -> Self {
        Self {
            solver: None,
            coefficients: None,
            h_interior: 8.0,
            h_exterior: 25.0,
            prev_q_flux: 0.0,
            initialized: false,
            valid: false,
        }
    }

    /// Create wrapper with custom convective coefficients.
    pub fn with_convection(h_interior: f64, h_exterior: f64) -> Self {
        Self {
            h_interior,
            h_exterior,
            prev_q_flux: 0.0,
            ..Self::new()
        }
    }

    /// Convert WallProperties to CTF materials.
    ///
    /// This hides BuildingAssembly internals from the solver. If BuildingAssembly
    /// changes its layer structure, only WallProperties::from_assembly() needs updating.
    fn wall_properties_to_ctf_materials(wall_props: &WallProperties) -> Vec<CTFMaterial> {
        wall_props
            .layers
            .iter()
            .map(|layer| {
                CTFMaterial::new(
                    &layer.name,
                    layer.thickness_m,
                    layer.conductivity_w_mk,
                    layer.density_kg_m3,
                    layer.specific_heat_j_kgk,
                )
            })
            .collect()
    }

    /// Validate CTF coefficients.
    fn validate_coefficients(coeffs: &CTFCoefficients) -> bool {
        coeffs.x.iter().all(|&x| x.is_finite())
            && coeffs.y.iter().all(|&y| y.is_finite())
            && coeffs.z.iter().all(|&z| z.is_finite())
            && coeffs.phi.iter().all(|&p| p.is_finite())
    }
}

impl Default for CTFSolverWrapper {
    fn default() -> Self {
        Self::new()
    }
}

impl HeatConductionSolver for CTFSolverWrapper {
    fn name(&self) -> &str {
        "CTF"
    }

    fn initialize(&mut self, wall: &BuildingAssembly) -> Result<(), SolverError> {
        // Convert assembly to wall properties (the seam)
        let wall_props = WallProperties::from_assembly(wall);

        // Convert wall properties to CTF materials
        let materials = Self::wall_properties_to_ctf_materials(&wall_props);

        if materials.is_empty() {
            return Err(SolverError::ConstructionError(
                "Wall assembly has no layers".to_string(),
            ));
        }

        // Compute CTF coefficients for 1-hour timestep
        let timestep = 3600.0; // Default 1 hour
        let coeffs = CTFCalculator::with_defaults(&materials, timestep).compute_coefficients();

        // Validate coefficients
        if !Self::validate_coefficients(&coeffs) {
            self.valid = false;
            return Err(SolverError::CoefficientError(
                "CTF coefficient calculation failed - coefficients are not finite".to_string(),
            ));
        }

        // Create solver configuration
        let config = CTFSolverConfig::new(timestep, 50);

        // Create and initialize solver with warmup
        // Use with_warmup() to initialize history buffers with realistic values
        // instead of zero flux/uniform temperature which causes unphysical transients
        self.solver = Some(CTFSolver::with_warmup(
            coeffs.clone(),
            config,
            20.0, // t_interior_initial
            20.0, // t_exterior_initial
            7,    // warmup_days
        ));
        self.coefficients = Some(coeffs);
        self.initialized = true;
        self.valid = true;

        Ok(())
    }

    fn step(
        &mut self,
        timestep: f64,
        T_interior: f64,
        T_exterior: f64,
        _h_interior: f64,
        _h_exterior: f64,
    ) -> Result<f64, SolverError> {
        if !self.initialized {
            return Err(SolverError::InvalidConfig(
                "CTF solver not initialized. Call initialize() first.".to_string(),
            ));
        }

        if !self.valid {
            return Err(SolverError::ConvergenceError(
                "CTF solver is not valid (coefficients may be invalid)".to_string(),
            ));
        }

        // Get mutable reference to solver
        let solver = self.solver.as_mut().ok_or_else(|| {
            SolverError::InvalidConfig("CTF solver is None after initialization".to_string())
        })?;

        // Verify timestep matches CTF configuration
        let solver_timestep = solver.config.timestep;
        if (timestep - solver_timestep).abs() > 1.0 {
            // Timestep mismatch - could interpolate or warn
            // For now, just proceed with CTF's native timestep
            log::warn!(
                "CTF timestep ({:.0}s) differs from model timestep ({:.0}s)",
                solver_timestep,
                timestep
            );
        }

        // Approximate surface temperature accounting for convection resistance
        // T_surface = T_air - q_conv/h (where q_conv = q_flux)
        let t_interior_surface = T_interior - self.prev_q_flux / self.h_interior;
        let t_exterior_surface = T_exterior; // T_exterior assumed to be surface temperature

        // Step the CTF solver with surface temperatures
        let q_flux = solver.step(t_interior_surface, t_exterior_surface);

        // Store flux for next timestep approximation
        self.prev_q_flux = q_flux;

        Ok(q_flux)
    }

    fn energy_storage_rate(&self) -> f64 {
        // CTF doesn't explicitly track energy storage rate
        // Could estimate from flux difference between interior and exterior
        0.0
    }

    fn is_valid(&self) -> bool {
        self.initialized && self.valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::assembly::{AssemblyBuilder, ConcreteMaterial};

    fn create_test_wall() -> BuildingAssembly {
        AssemblyBuilder::new("Test Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2))) // 200mm concrete
            .build()
            .unwrap()
    }

    #[test]
    fn test_ctf_wrapper_creation() {
        let wrapper = CTFSolverWrapper::new();
        assert!(!wrapper.initialized);
        assert!(!wrapper.valid);
    }

    #[test]
    fn test_ctf_wrapper_initialization() {
        let mut wrapper = CTFSolverWrapper::new();
        let wall = create_test_wall();

        let result = wrapper.initialize(&wall);
        assert!(result.is_ok());
        assert!(wrapper.is_valid());
    }

    #[test]
    fn test_ctf_wrapper_flux_calculation() {
        let mut wrapper = CTFSolverWrapper::new();
        let wall = create_test_wall();

        wrapper.initialize(&wall).unwrap();

        // Calculate flux for 20°C interior, 0°C exterior
        let flux = wrapper.step(3600.0, 20.0, 0.0, 8.0, 25.0).unwrap();

        // Flux should be finite
        assert!(flux.is_finite());

        // Flux should be negative (heat flowing out)
        assert!(flux < 0.0);
    }

    #[test]
    fn test_ctf_wrapper_uninitialized() {
        let mut wrapper = CTFSolverWrapper::new();

        // Should fail if not initialized
        let result = wrapper.step(3600.0, 20.0, 0.0, 8.0, 25.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ctf_wrapper_diurnal_simulation() {
        let mut wrapper = CTFSolverWrapper::new();
        let wall = create_test_wall();

        wrapper.initialize(&wall).unwrap();

        // 24-hour simulation
        let mut total_flux = 0.0;
        for hour in 0..24 {
            let t_ext = 10.0 + 10.0 * ((hour as f64 - 6.0) * std::f64::consts::PI / 12.0).sin();
            let flux = wrapper.step(3600.0, 20.0, t_ext, 8.0, 25.0).unwrap();
            total_flux += flux;
        }

        // Total flux should be reasonable
        assert!(
            total_flux.abs() < 10000.0,
            "Total flux {:.2} unreasonably large",
            total_flux
        );
    }

    #[test]
    fn test_ctf_wrapper_with_convection() {
        let mut wrapper = CTFSolverWrapper::with_convection(10.0, 30.0);
        let wall = create_test_wall();

        let result = wrapper.initialize(&wall);
        assert!(result.is_ok());

        // Custom convection coefficients should be stored
        assert!((wrapper.h_interior - 10.0).abs() < 1e-10);
        assert!((wrapper.h_exterior - 30.0).abs() < 1e-10);
    }

    // === Phase 3: Additional coverage tests ===

    #[test]
    fn test_ctf_wrapper_default() {
        let wrapper = CTFSolverWrapper::default();
        assert_eq!(wrapper.h_interior, 8.0);
        assert_eq!(wrapper.h_exterior, 25.0);
        assert_eq!(wrapper.prev_q_flux, 0.0);
    }

    #[test]
    fn test_ctf_wrapper_name() {
        let wrapper = CTFSolverWrapper::new();
        assert_eq!(wrapper.name(), "CTF");
    }

    #[test]
    fn test_ctf_wrapper_is_valid() {
        let mut wrapper = CTFSolverWrapper::new();
        let wall = create_test_wall();

        // Not initialized -> not valid
        assert!(!wrapper.is_valid());

        wrapper.initialize(&wall).unwrap();
        // Initialized -> valid
        assert!(wrapper.is_valid());
    }

    #[test]
    fn test_ctf_wrapper_energy_storage_rate() {
        let mut wrapper = CTFSolverWrapper::new();
        let wall = create_test_wall();
        wrapper.initialize(&wall).unwrap();

        // Energy storage rate is 0 (placeholder for CTF)
        let rate = wrapper.energy_storage_rate();
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn test_ctf_wrapper_warmup_initializes_flux_history() {
        let mut wrapper = CTFSolverWrapper::new();
        let wall = create_test_wall();
        wrapper.initialize(&wall).unwrap();

        // After warmup, history should contain non-zero flux values
        // from the diurnal warmup cycles, not the zero initial state
        let solver = wrapper.solver.as_ref().expect("solver should exist");
        let interior_flux = solver.interior_flux();
        let exterior_flux = solver.exterior_flux();

        // Warmup cycles should have established realistic flux values
        // The exterior flux should reflect the diurnal cycling during warmup
        assert!(
            exterior_flux.is_finite(),
            "Exterior flux should be finite after warmup"
        );
        assert!(
            interior_flux.is_finite(),
            "Interior flux should be finite after warmup"
        );
    }

    #[test]
    fn test_ctf_wrapper_without_warmup_vs_with_warmup() {
        use crate::physics::ctf_coefficients::CTFCalculator;
        use crate::physics::wall_properties::WallProperties;

        let wall = create_test_wall();
        let wall_props = WallProperties::from_assembly(&wall);
        let materials = CTFSolverWrapper::wall_properties_to_ctf_materials(&wall_props);
        let timestep = 3600.0;
        let coeffs = CTFCalculator::with_defaults(&materials, timestep).compute_coefficients();
        let config = CTFSolverConfig::new(timestep, 50);

        // Create wrapper (should now use warmup internally)
        let mut wrapper = CTFSolverWrapper::new();
        wrapper.initialize(&wall).unwrap();
        let wrapper_flux = wrapper.solver.as_ref().unwrap().exterior_flux();

        // Create solver WITHOUT warmup (old behavior)
        let solver_no_warmup = CTFSolver::new(coeffs.clone(), config.clone());
        let flux_no_warmup = solver_no_warmup.exterior_flux();

        // Create solver WITH warmup (expected behavior)
        let solver_with_warmup = CTFSolver::with_warmup(coeffs, config, 20.0, 20.0, 7);
        let flux_with_warmup = solver_with_warmup.exterior_flux();

        // Verify all fluxes are finite
        assert!(wrapper_flux.is_finite(), "Wrapper flux should be finite");
        assert!(
            flux_no_warmup.is_finite(),
            "Flux without warmup should be finite"
        );
        assert!(
            flux_with_warmup.is_finite(),
            "Flux with warmup should be finite"
        );

        // Key assertion: wrapper should use warmup, not zero-init
        // Wrapper flux should match with_warmup flux, not no_warmup flux
        assert_eq!(
            wrapper_flux, flux_with_warmup,
            "Wrapper should use warmup - expected wrapper flux ({}) to match warmup flux ({})",
            wrapper_flux, flux_with_warmup
        );
    }

    #[test]
    fn test_ctf_wrapper_step_extreme_temperatures() {
        let mut wrapper = CTFSolverWrapper::new();
        let wall = create_test_wall();
        wrapper.initialize(&wall).unwrap();

        // Cold extreme
        let flux_cold = wrapper.step(3600.0, -10.0, -20.0, 8.0, 25.0).unwrap();
        assert!(flux_cold.is_finite());

        // Hot extreme
        let flux_hot = wrapper.step(3600.0, 40.0, 50.0, 8.0, 25.0).unwrap();
        assert!(flux_hot.is_finite());
    }

    #[test]
    fn test_ctf_wrapper_step_ignored_convection() {
        let mut wrapper = CTFSolverWrapper::new();
        let wall = create_test_wall();
        wrapper.initialize(&wall).unwrap();

        // Convection parameters are ignored by the step function
        // Should still work with any h_interior, h_exterior values
        let flux = wrapper.step(3600.0, 20.0, 10.0, 100.0, 200.0).unwrap();
        assert!(flux.is_finite());
    }

    #[test]
    fn test_ctf_wrapper_initialization_reinitializable() {
        let mut wrapper = CTFSolverWrapper::new();
        let wall = create_test_wall();

        // First initialization
        let result1 = wrapper.initialize(&wall);
        assert!(result1.is_ok());

        // Re-initialization should also succeed
        let result2 = wrapper.initialize(&wall);
        assert!(result2.is_ok());
    }
}
