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
//! ```rust,no_run
//! use fluxion::physics::ctf_solver_wrapper::CTFSolverWrapper;
//! use fluxion::physics::solver_trait::HeatConductionSolver;
//! use fluxion::physics::units::{FromF64, HeatTransferCoefficient, Temperature, Time};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut solver = CTFSolverWrapper::new();
//!     // Real code constructs a `WallSpec` from a spec / YAML; see
//!     // `WallSpec::from_layers(...)` and the `fluxion::physics::wall_spec` docs.
//!     // solver.initialize(&wall)?;
//!
//!     let _flux = solver.step(
//!         Time::from_value(3600.0),
//!         Temperature::from_value(20.0),
//!         Temperature::from_value(5.0),
//!         HeatTransferCoefficient::from_value(8.0),
//!         HeatTransferCoefficient::from_value(25.0),
//!     )?;
//!     Ok(())
//! }
//! ```

use crate::physics::ctf_coefficients::{CTFCalculator, CTFCoefficients, CTFMaterial};
use crate::physics::ctf_solver::{CTFSolver, CTFSolverConfig};
use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
use crate::physics::units::{FromF64, HeatFlux, HeatTransferCoefficient, Temperature, Time, ToF64};
use crate::physics::wall_properties::WallProperties;
use crate::physics::wall_spec::WallSpec;

/// Default interior convective coefficient [W/m²·K], matching the ASHRAE 140
/// interior film resistance (R_SI=0.125) baked into the CTF coefficients.
const DEFAULT_H_INTERIOR: f64 = 8.0;
/// Default exterior convective coefficient [W/m²·K], matching the ASHRAE 140
/// exterior film resistance (R_SE=0.044). Note this is 1/0.044 ≈ 22.73, *not*
/// 25.0: the old dead `with_convection` default of 25.0 silently selected
/// custom films (R_se=0.040) and diverged from `compute_state_space_ctf`.
const DEFAULT_H_EXTERIOR: f64 = 1.0 / 0.044;

/// CTF solver wrapper implementing the common HeatConductionSolver trait.
///
/// This wrapper adapts the CTFSolver to work with the unified solver interface,
/// handling conversion from BuildingAssembly to CTF coefficients and managing
/// boundary condition transformations.
///
/// # Film Resistance Handling
///
/// CTF coefficients include the effects of surface film resistances (interior
/// and exterior convective heat transfer coefficients). The `h_interior` and
/// `h_exterior` values passed to `step()` are used to compute the film
/// resistances (R = 1/h) that are baked into the CTF coefficients during
/// initialization.
///
/// If the same wrapper is used with different h values across calls, the
/// coefficients will be recomputed with the new values on the first call
/// that differs from the stored values.
pub struct CTFSolverWrapper {
    /// Underlying CTF solver
    solver: Option<CTFSolver>,
    /// CTF coefficients (cached after initialization)
    coefficients: Option<CTFCoefficients>,
    /// Interior convective coefficient [W/m²·K] - used for CTF coefficient computation
    h_interior: f64,
    /// Exterior convective coefficient [W/m²·K] - used for CTF coefficient computation
    h_exterior: f64,
    /// Previous interior heat flux for convection approximation [W/m²]
    prev_q_flux: f64,
    /// Initialized flag
    initialized: bool,
    /// Valid flag (coefficients converged)
    valid: bool,
    /// Wall spec cached for potential coefficient recomputation
    wall_spec: Option<WallSpec>,
}

impl CTFSolverWrapper {
    /// Create a new uninitialized CTF solver wrapper.
    ///
    /// The wrapper uses default film resistances (R_SI=0.125, R_SE=0.044)
    /// until the first `step()` call provides actual h values.
    pub fn new() -> Self {
        Self {
            solver: None,
            coefficients: None,
            h_interior: DEFAULT_H_INTERIOR,
            h_exterior: DEFAULT_H_EXTERIOR,
            prev_q_flux: 0.0,
            initialized: false,
            valid: false,
            wall_spec: None,
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

    fn initialize(&mut self, wall: &WallSpec) -> Result<(), SolverError> {
        // Cache the wall spec for potential coefficient recomputation
        self.wall_spec = Some(wall.clone());

        // Convert WallSpec to wall properties (the seam)
        let wall_props = wall.to_wall_properties();

        // Convert wall properties to CTF materials
        let materials = Self::wall_properties_to_ctf_materials(&wall_props);

        if materials.is_empty() {
            return Err(SolverError::ConstructionError(
                "Wall assembly has no layers".to_string(),
            ));
        }

        // Compute CTF coefficients for 1-hour timestep.
        // Use custom film resistances only when h differs from the defaults
        // (compared in h-space so the default path is bit-identical to
        // `compute_state_space_ctf`).
        let timestep = 3600.0; // Default 1 hour
        let uses_custom_films = (self.h_interior - DEFAULT_H_INTERIOR).abs() > 1e-6
            || (self.h_exterior - DEFAULT_H_EXTERIOR).abs() > 1e-6;

        let coeffs = if uses_custom_films {
            let r_si = 1.0 / self.h_interior;
            let r_se = 1.0 / self.h_exterior;
            CTFCalculator::with_film_resistances(&materials, timestep, 50, r_si, r_se)
                .compute_coefficients()
        } else {
            CTFCalculator::with_defaults(&materials, timestep).compute_coefficients()
        };

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
        timestep: Time,
        T_interior: Temperature,
        T_exterior: Temperature,
        h_interior: HeatTransferCoefficient,
        h_exterior: HeatTransferCoefficient,
    ) -> Result<HeatFlux, SolverError> {
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

        let h_int = h_interior.to_value();
        let h_ext = h_exterior.to_value();

        // Check if film resistances have changed significantly
        // If so, recompute CTF coefficients with new values
        let needs_recompute =
            (h_int - self.h_interior).abs() > 0.01 || (h_ext - self.h_exterior).abs() > 0.01;

        if needs_recompute {
            log::debug!(
                "CTF: Film coefficients changed (h_int: {:.2} -> {:.2}, h_ext: {:.2} -> {:.2}). Recomputing CTF coefficients.",
                self.h_interior, h_int, self.h_exterior, h_ext
            );

            // Update stored h values
            self.h_interior = h_int;
            self.h_exterior = h_ext;

            // Recompute coefficients with new film resistances
            let wall = self.wall_spec.as_ref().ok_or_else(|| {
                SolverError::InvalidConfig("Wall spec not cached for recomputation".to_string())
            })?;

            // Re-run initialization with new h values
            let wall_props = wall.to_wall_properties();
            let materials = Self::wall_properties_to_ctf_materials(&wall_props);

            if materials.is_empty() {
                return Err(SolverError::ConstructionError(
                    "Wall assembly has no layers".to_string(),
                ));
            }

            let solver_timestep = 3600.0;
            let r_si = 1.0 / self.h_interior;
            let r_se = 1.0 / self.h_exterior;

            let coeffs =
                CTFCalculator::with_film_resistances(&materials, solver_timestep, 50, r_si, r_se)
                    .compute_coefficients();

            if !Self::validate_coefficients(&coeffs) {
                self.valid = false;
                return Err(SolverError::CoefficientError(
                    "CTF coefficient recalculation failed".to_string(),
                ));
            }

            // Reinitialize solver with new coefficients
            let config = CTFSolverConfig::new(solver_timestep, 50);
            self.solver = Some(CTFSolver::with_warmup(
                coeffs.clone(),
                config,
                T_interior.to_value(),
                T_exterior.to_value(),
                7, // warmup_days
            ));
            self.coefficients = Some(coeffs);
        }

        // Get mutable reference to solver
        let solver = self.solver.as_mut().ok_or_else(|| {
            SolverError::InvalidConfig("CTF solver is None after initialization".to_string())
        })?;

        // Verify timestep matches CTF configuration
        let solver_timestep = solver.config.timestep;
        let timestep_s = timestep.to_value();
        if (timestep_s - solver_timestep).abs() > 1.0 {
            // Timestep mismatch - could interpolate or warn
            // For now, just proceed with CTF's native timestep
            log::warn!(
                "CTF timestep ({:.0}s) differs from model timestep ({:.0}s)",
                solver_timestep,
                timestep_s
            );
        }

        // The CTF coefficients already include film resistance scaling.
        // Input temperatures should be AIR temperatures, not surface temperatures
        // — applying a surface correction would double-count the film resistance.
        let t_interior_air = T_interior.to_value();
        let t_exterior_air = T_exterior.to_value();

        // Step the CTF solver with air temperatures
        let q_flux = solver.step(t_interior_air, t_exterior_air);

        // Store flux for next timestep approximation
        self.prev_q_flux = q_flux;

        Ok(HeatFlux::from_value(q_flux))
    }

    /// Issue #1418: Pure-query steady-state flux from cached CTF coefficients.
    ///
    /// Returns the steady-state limit of the CTF recurrence
    /// `q = ΣX_j·T_o − ΣY_j·T_i − ΣΦ_j·q_prev` (the recurrence skips Φ_0):
    /// `q·(1 + Σ_{j≥1}Φ_j) = ΣX·T_ext − ΣY·T_int`.
    ///
    /// Note the Φ terms must be included: the film scaling in
    /// `compute_state_space_ctf_with_films` targets `ΣX/(1+ΣΦ) = U_filmed`,
    /// so ΣX alone is *not* the U-value (for film-dominated walls the Φ sum
    /// is far from zero). A ΣX-only query under-reports the flux ~5x for
    /// the test wall with custom films.
    ///
    /// This does NOT advance solver state — it reads only the cached
    /// `coefficients` field populated during `initialize()`.
    fn steady_state_flux(
        &self,
        T_interior: Temperature,
        T_exterior: Temperature,
    ) -> Result<HeatFlux, SolverError> {
        let coeffs = self.coefficients.as_ref().ok_or_else(|| {
            SolverError::InvalidConfig(
                "CTF solver not initialized — no cached coefficients".to_string(),
            )
        })?;
        let sum_x: f64 = coeffs.x.iter().copied().sum();
        let sum_y: f64 = coeffs.y.iter().copied().sum();
        // Mirror the recurrence exactly: it consumes Φ_1.. (phi[j+1]), never Φ_0.
        let sum_phi: f64 = coeffs.phi.iter().skip(1).copied().sum();
        let q = (sum_x * T_exterior.to_value() - sum_y * T_interior.to_value()) / (1.0 + sum_phi);
        Ok(HeatFlux::from_value(q))
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
    use crate::physics::units::{FromF64, HeatTransferCoefficient, Temperature, Time, ToF64};
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

        // Flux should be negative (heat flowing out)
        assert!(flux.to_value() < 0.0);
    }

    #[test]
    fn test_ctf_wrapper_uninitialized() {
        let mut wrapper = CTFSolverWrapper::new();

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
    fn test_ctf_wrapper_diurnal_simulation() {
        let mut wrapper = CTFSolverWrapper::new();
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
    fn test_ctf_wrapper_default() {
        let wrapper = CTFSolverWrapper::default();
        assert_eq!(wrapper.h_interior, DEFAULT_H_INTERIOR);
        assert_eq!(wrapper.h_exterior, DEFAULT_H_EXTERIOR);
        assert_eq!(wrapper.prev_q_flux, 0.0);
        assert!(wrapper.wall_spec.is_none());
        assert!(!wrapper.initialized);
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

        let wall = create_test_wall();
        let wall_props = wall.to_wall_properties();
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
    fn test_ctf_wrapper_step_with_custom_h_values() {
        // Test that CTF honors custom h_interior and h_exterior values:
        // after stepping with a custom h pair, the steady-state flux
        // (U*ΔT from the film-scaled coefficients) must match the analytic
        // U-value for that h pair — not the default-film U-value.
        let mut wrapper = CTFSolverWrapper::new();
        let wall = create_test_wall();
        wrapper.initialize(&wall).unwrap();

        // Step once with custom h values to trigger coefficient recomputation.
        // (h_ext=50 is far from the 1/0.044≈22.73 default, so the custom-film
        //  and default-film U-values are well separated.)
        let h_int = 8.0;
        let h_ext = 50.0;
        let flux = wrapper
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(0.0),
                HeatTransferCoefficient::from_value(h_int),
                HeatTransferCoefficient::from_value(h_ext),
            )
            .unwrap();
        assert!(flux.to_value().is_finite(), "CTF flux should be finite");

        // Analytic U for 200mm concrete (k=1.4 -> R=0.2/1.4) with these films.
        let r_wall = 0.2 / 1.4;
        let u_expected = 1.0 / (1.0 / h_int + r_wall + 1.0 / h_ext);
        let q_expected = u_expected * 20.0;

        let q_ss = wrapper
            .steady_state_flux(Temperature::from_value(20.0), Temperature::from_value(0.0))
            .unwrap()
            .to_value()
            .abs();
        assert!(
            (q_ss - q_expected).abs() / q_expected < 0.01,
            "CTF steady-state flux {:.4} should match analytic U*ΔT {:.4} for h=({:.1},{:.1})",
            q_ss,
            q_expected,
            h_int,
            h_ext
        );

        // And it must NOT match the default-film U-value (proves the custom
        // films were actually used, not the ASHRAE 140 defaults).
        let u_default = 1.0 / (0.125 + r_wall + 0.044);
        assert!(
            (q_ss - u_default * 20.0).abs() / q_expected > 0.01,
            "CTF flux should reflect custom films, not default films"
        );
    }

    #[test]
    fn test_ctf_wrapper_step_recomputes_coefficients_on_h_change() {
        // Test that CTF recomputes coefficients when h values change significantly

        let mut wrapper = CTFSolverWrapper::new();
        let wall = create_test_wall();
        wrapper.initialize(&wall).unwrap();

        // First step with h_interior=8.0
        let flux1 = wrapper
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(0.0),
                HeatTransferCoefficient::from_value(8.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap();

        // Second step with different h_interior (10.0 instead of 8.0)
        // This should trigger coefficient recomputation
        let flux2 = wrapper
            .step(
                Time::from_value(3600.0),
                Temperature::from_value(20.0),
                Temperature::from_value(0.0),
                HeatTransferCoefficient::from_value(10.0),
                HeatTransferCoefficient::from_value(25.0),
            )
            .unwrap();

        // With higher h_interior (lower R_si), the surface resistance decreases,
        // so the wall should respond faster to temperature changes.
        // The exact difference depends on the wall's thermal mass, but there should be a difference.
        // Both fluxes should be finite.
        assert!(flux1.to_value().is_finite(), "First flux should be finite");
        assert!(flux2.to_value().is_finite(), "Second flux should be finite");

        // Verify the wrapper tracked the new h values
        assert!(
            (wrapper.h_interior - 10.0).abs() < 0.01,
            "h_interior should be updated to 10.0"
        );
    }

    #[test]
    fn test_ctf_wrapper_vs_fd_wrapper_same_h() {
        // Test that CTF and FD wrappers converge to the same steady-state
        // interior flux when given identical h values. Comparing
        // steady-state (not transient) fluxes: the two solvers have
        // different transient dynamics, but both must settle to U*ΔT with
        // the films the caller specified. This is the key test for issue
        // #4165: it proves the CTF wrapper actually honors the h values
        // instead of silently using the ASHRAE 140 defaults.
        use crate::physics::fd_solver_wrapper::FDSolverWrapper;

        let wall = create_test_wall();
        let h_int = 8.0;
        let h_ext = 25.0;

        let mut ctf_wrapper = CTFSolverWrapper::new();
        ctf_wrapper.initialize(&wall).unwrap();

        let mut fd_wrapper = FDSolverWrapper::new();
        fd_wrapper.initialize(&wall).unwrap();

        // Drive both to steady state (200mm concrete settles in ~15h;
        // 200 hourly steps is >13 time constants).
        let (mut q_ctf, mut q_fd) = (0.0_f64, 0.0_f64);
        for _ in 0..200 {
            q_ctf = ctf_wrapper
                .step(
                    Time::from_value(3600.0),
                    Temperature::from_value(20.0),
                    Temperature::from_value(0.0),
                    HeatTransferCoefficient::from_value(h_int),
                    HeatTransferCoefficient::from_value(h_ext),
                )
                .unwrap()
                .to_value()
                .abs();

            q_fd = fd_wrapper
                .step(
                    Time::from_value(3600.0),
                    Temperature::from_value(20.0),
                    Temperature::from_value(0.0),
                    HeatTransferCoefficient::from_value(h_int),
                    HeatTransferCoefficient::from_value(h_ext),
                )
                .unwrap()
                .to_value()
                .abs();
        }

        assert!(q_ctf.is_finite(), "CTF flux should be finite");
        assert!(q_fd.is_finite(), "FD flux should be finite");

        // Analytic steady state for 200mm concrete (k=1.4) with these films.
        let r_wall = 0.2 / 1.4;
        let q_expected = 20.0 / (1.0 / h_int + r_wall + 1.0 / h_ext);

        // Sanity bound against the analytic value. 10% is loose on purpose:
        // the FD wrapper's own boundary-condition modeling sits ~6% above
        // analytic here (pre-existing FD behavior, not this issue's scope);
        // the CTF wrapper is verified tightly against analytic in
        // test_ctf_wrapper_step_with_custom_h_values.
        for (name, q) in [("CTF", q_ctf), ("FD", q_fd)] {
            assert!(
                (q - q_expected).abs() / q_expected < 0.10,
                "{} steady-state flux {:.3} should be near analytic U*ΔT {:.3} for h=({:.1},{:.1})",
                name,
                q,
                q_expected,
                h_int,
                h_ext
            );
        }

        // The two solvers must agree with each other, not just the analytic value.
        assert!(
            (q_ctf - q_fd).abs() / q_expected < 0.08,
            "CTF ({:.3}) and FD ({:.3}) steady-state fluxes should agree",
            q_ctf,
            q_fd
        );
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
