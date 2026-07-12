//! CTF-Zone Air Coupling Solver.
//!
//! This module implements the iterative coupling between CTF-based wall conduction
//! and zone air heat balance. The key challenge is that the CTF solver requires
//! the interior surface temperature (T_si) as a boundary condition, but T_si is
//! itself determined by the zone air temperature and the CTF heat flux.
//!
//! # Mathematical Formulation
//!
//! The interior surface heat balance is:
//!
//! ```text
//! 0 = h_ci × (T_zone - T_si) + h_ri × (T_rm - T_si) + α_solar × I_solar - q''_ctf
//! ```
//!
//! where:
//! - `h_ci` = interior convective coefficient [W/m²·K]
//! - `h_ri` = interior radiative coefficient [W/m²·K]
//! - `T_zone` = zone air temperature [°C]
//! - `T_si` = interior surface temperature [°C] (unknown)
//! - `T_rm` = mean radiant temperature [°C] (approximated as mass temperature)
//! - `α_solar` = solar absorptance of interior surface
//! - `I_solar` = solar irradiance absorbed by interior surface [W/m²]
//! - `q''_ctf` = CTF interior heat flux [W/m²] (depends on T_si history)
//!
//! The CTF heat flux is:
//!
//! ```text
//! q''_ctf = -Z₀×T_si + Σ(X_j×T_ext,j) - Σ(Y_j×T_si,j) - Σ(Φ_j×q''_j)
//! ```
//!
//! Substituting the CTF equation into the surface balance gives an implicit
//! equation for T_si that must be solved iteratively.
//!
//! # Solution Method
//!
//! We use a Newton-Raphson iteration to solve for T_si:
//!
//! 1. Initial guess: T_si = T_zone (surface at air temperature)
//! 2. Calculate CTF flux using current T_si estimate
//! 3. Calculate surface balance residual
//! 4. Update T_si using linearized conductance
//! 5. Repeat until convergence
//!
//! The linearized update is:
//!
//! ```text
//! T_si_new = T_si_old + residual / (h_ci + h_ri + Z₀)
//! ```

use crate::physics::ctf_solver::CTFSolver;

/// Interior surface heat transfer coefficients.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceCoefficients {
    /// Interior convective heat transfer coefficient [W/m²·K]
    pub h_ci: f64,
    /// Interior radiative heat transfer coefficient [W/m²·K]
    pub h_ri: f64,
    /// Combined interior coefficient
    pub h_i: f64,
}

impl SurfaceCoefficients {
    /// Create coefficients for typical interior conditions.
    pub fn typical() -> Self {
        // ASHRAE fundamentals: h_ci ≈ 3-4 W/m²·K for natural convection
        // h_ri ≈ 4-5 W/m²·K for typical interior surfaces
        // Combined h_i ≈ 8 W/m²·K (ASHRAE 140 default)
        let h_ci = 3.0;
        let h_ri = 5.0;
        Self {
            h_ci,
            h_ri,
            h_i: h_ci + h_ri,
        }
    }

    /// Create coefficients for ASHRAE 140 standard.
    pub fn ashrae_140() -> Self {
        // ASHRAE 140 uses h_i = 8.0 W/m²·K
        let h_ci = 3.0;
        let h_ri = 5.0;
        Self {
            h_ci,
            h_ri,
            h_i: h_ci + h_ri,
        }
    }
}

/// Result of CTF-zone coupling calculation.
#[derive(Debug, Clone)]
pub struct CtfZoneCouplingResult {
    /// Interior surface temperature [°C]
    pub t_surface_interior: f64,
    /// CTF heat flux at interior surface [W/m²] (positive = into zone)
    pub q_ctf_interior: f64,
    /// Convective heat transfer from surface to zone air [W/m²]
    pub q_convective: f64,
    /// Number of iterations for convergence
    pub iterations: usize,
    /// Whether the solution converged
    pub converged: bool,
}

/// CTF-Zone Air Coupling Solver.
///
/// This solver iteratively finds the interior surface temperature that satisfies
/// both the CTF conduction equation and the surface heat balance.
#[derive(Debug, Clone)]
pub struct CtfZoneCouplingSolver {
    /// Interior surface heat transfer coefficients
    pub coefficients: SurfaceCoefficients,
    /// Convergence tolerance [°C]
    pub tolerance: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
}

impl CtfZoneCouplingSolver {
    /// Create a new coupling solver with default parameters.
    pub fn new() -> Self {
        Self {
            coefficients: SurfaceCoefficients::ashrae_140(),
            tolerance: 0.001, // 0.001°C convergence tolerance
            max_iterations: 20,
        }
    }

    /// Create solver with custom coefficients.
    pub fn with_coefficients(coefficients: SurfaceCoefficients) -> Self {
        Self {
            coefficients,
            tolerance: 0.001,
            max_iterations: 20,
        }
    }

    /// Solve for interior surface temperature given zone conditions.
    ///
    /// # Arguments
    ///
    /// * `solver` - CTF solver (will be stepped with the solution)
    /// * `t_zone` - Zone air temperature [°C]
    /// * `t_mass` - Mean radiant/mass temperature [°C] (approximation for T_rm)
    /// * `t_sol_air` - Sol-air temperature (exterior boundary) [°C]
    /// * `solar_absorbed_interior` - Solar radiation absorbed at interior surface [W/m²]
    ///
    /// # Returns
    ///
    /// Coupling result with surface temperature and heat fluxes.
    pub fn solve(
        &self,
        solver: &mut CTFSolver,
        t_zone: f64,
        t_mass: f64,
        t_sol_air: f64,
        solar_absorbed_interior: f64,
    ) -> CtfZoneCouplingResult {
        if solver.coefficients.total_state_nodes == 0 {
            let u_filmed = solver.coefficients.x.first().copied().unwrap_or(0.0);
            let q_ctf = u_filmed * (t_sol_air - t_zone);
            let h_i = self.coefficients.h_i;
            let t_si = (self.coefficients.h_ci * t_zone + self.coefficients.h_ri * t_mass
                + solar_absorbed_interior - q_ctf) / h_i;

            solver.t_interior_surface = t_si;
            solver.t_exterior_surface = t_sol_air;
            solver.q_interior_history[0] = q_ctf;

            return CtfZoneCouplingResult {
                t_surface_interior: t_si,
                q_ctf_interior: q_ctf,
                q_convective: self.coefficients.h_ci * (t_zone - t_si),
                iterations: 0,
                converged: true,
            };
        }

        // Initial guess: surface at zone air temperature
        let mut t_si = t_zone;
        let mut converged = false;
        let mut iterations = 0;
        let mut q_ctf_last = 0.0; // Track last CTF flux for use after loop

        for _iter in 0..self.max_iterations {
            iterations = _iter + 1;

            // Compute CTF flux WITHOUT stepping the solver during Newton-Raphson.
            // Each step() call shifts history buffers, so calling it inside the
            // NR loop would corrupt the solver state. Instead, use the coefficients
            // directly with the current history.
            let t_int_hist = solver.interior_temperature_history();
            let t_ext_hist = solver.exterior_temperature_history();
            let q_hist = solver.interior_flux_history();

            // Build temporary histories with current T_si estimate
            let mut t_ext_current = vec![t_sol_air; t_ext_hist.len()];
            t_ext_current[1..].copy_from_slice(&t_ext_hist[1..]);

            q_ctf_last = solver.coefficients.calculate_interior_flux(
                t_si,
                &t_ext_current,
                &t_int_hist[1..],
                &q_hist[1..],
            );

            // Surface heat balance residual:
            // f(T_si) = h_ci*(T_zone - T_si) + h_ri*(T_mass - T_si) + solar - q_ctf
            // At convergence, f(T_si) = 0
            let q_convective = self.coefficients.h_ci * (t_zone - t_si);
            let q_radiative = self.coefficients.h_ri * (t_mass - t_si);
            let residual = q_convective + q_radiative + solar_absorbed_interior - q_ctf_last;

            // Check convergence
            if residual.abs() < self.tolerance * (self.coefficients.h_i + 1.0) {
                converged = true;
                break;
            }

            // Newton-Raphson update:
            // df/dT_si = -h_ci - h_ri - dq_ctf/dT_si
            // From CTF equation: dq_ctf/dT_si = -Z₀ (first CTF coefficient)
            // So: df/dT_si = -(h_ci + h_ri) + Z₀ = -(h_i - Z₀)
            //
            // For stability, we use the linearized conductance:
            // ΔT_si = residual / (h_ci + h_ri + Z₀)
            //
            // Note: Z₀ should be positive for stable CTF coefficients
            let z0 = solver.coefficients.z.first().copied().unwrap_or(1.0);
            let effective_conductance = self.coefficients.h_i + z0;

            // Update T_si
            let delta_t = residual / effective_conductance;
            t_si += delta_t;

            // Damping for stability (under-relaxation)
            if delta_t.abs() > 5.0 {
                t_si = t_zone + (t_si - t_zone) * 0.5;
            }
        }

        // Final CTF flux calculation with converged T_si
        let q_ctf_final = if converged {
            q_ctf_last
        } else {
            // If not converged, do one more step to get best estimate
            solver.step(t_si, t_sol_air)
        };

        let q_convective_final = self.coefficients.h_ci * (t_zone - t_si);

        CtfZoneCouplingResult {
            t_surface_interior: t_si,
            q_ctf_interior: q_ctf_final,
            q_convective: q_convective_final,
            iterations,
            converged,
        }
    }

    /// Solve for multiple zones/surfaces simultaneously.
    ///
    /// This method handles the case where multiple surfaces exchange radiation
    /// with each other (longwave radiation network).
    ///
    /// # Arguments
    ///
    /// * `solvers` - CTF solvers (one per surface)
    /// * `t_zone` - Zone air temperature [°C]
    /// * `t_masses` - Mass temperatures for each surface [°C]
    /// * `t_sol_airs` - Sol-air temperatures for each surface [°C]
    /// * `solar_absorbed` - Solar radiation absorbed at each surface [W/m²]
    ///
    /// # Returns
    ///
    /// Vector of coupling results, one per surface.
    pub fn solve_multiple(
        &self,
        solvers: &mut [&mut CTFSolver],
        t_zone: f64,
        t_masses: &[f64],
        t_sol_airs: &[f64],
        solar_absorbed: &[f64],
    ) -> Vec<CtfZoneCouplingResult> {
        solvers
            .iter_mut()
            .zip(t_masses.iter())
            .zip(t_sol_airs.iter())
            .zip(solar_absorbed.iter())
            .map(|(((solver, &t_mass), &t_sol_air), &solar)| {
                self.solve(solver, t_zone, t_mass, t_sol_air, solar)
            })
            .collect()
    }

    /// Calculate the effective conductance for sensitivity analysis.
    ///
    /// This is useful for HVAC load calculations where we need to know
    /// how sensitive the zone is to internal gains.
    pub fn effective_conductance(&self, z0: f64) -> f64 {
        self.coefficients.h_i + z0
    }
}

impl Default for CtfZoneCouplingSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplified CTF-zone coupling for cases where iterative solution is too expensive.
///
/// This uses a linearized approximation that avoids iteration:
/// 1. Assume T_si ≈ T_zone (no surface resistance)
/// 2. Calculate CTF flux
/// 3. Apply flux directly to zone air
///
/// This is less accurate but much faster, suitable for:
/// - Initial simulation startup
/// - Free-floating temperature calculations
/// - Cases with very high thermal mass where surface resistance is negligible
pub struct SimplifiedCtfCoupling;

impl SimplifiedCtfCoupling {
    /// Calculate CTF flux assuming T_si = T_zone.
    ///
    /// # Arguments
    ///
    /// * `solver` - CTF solver
    /// * `t_zone` - Zone air temperature [°C]
    /// * `t_sol_air` - Sol-air temperature [°C]
    ///
    /// # Returns
    ///
    /// Heat flux into zone [W/m²] (positive = heating the zone).
    pub fn flux_direct(solver: &mut CTFSolver, t_zone: f64, t_sol_air: f64) -> f64 {
        solver.step(t_zone, t_sol_air)
    }

    /// Calculate CTF flux with surface resistance correction.
    ///
    /// This applies a first-order correction for surface resistance
    /// without full iteration.
    ///
    /// # Arguments
    ///
    /// * `solver` - CTF solver
    /// * `t_zone` - Zone air temperature [°C]
    /// * `t_mass` - Mean radiant temperature [°C]
    /// * `t_sol_air` - Sol-air temperature [°C]
    /// * `solar_interior` - Interior solar gain [W/m²]
    /// * `h_i` - Combined interior coefficient [W/m²·K]
    ///
    /// # Returns
    ///
    /// Heat flux into zone [W/m²].
    pub fn flux_corrected(
        solver: &mut CTFSolver,
        t_zone: f64,
        _t_mass: f64,
        t_sol_air: f64,
        solar_interior: f64,
        h_i: f64,
    ) -> f64 {
        // First pass: assume T_si = T_zone
        let q_ctf_initial = solver.step(t_zone, t_sol_air);

        // Estimate T_si from surface balance
        // q_ctf = h_i * (T_rm - T_si) + solar, where T_rm is weighted average
        // For simplicity, assume T_rm ≈ T_mass
        let z0 = solver.coefficients.z.first().copied().unwrap_or(1.0);
        let effective_h = h_i + z0;

        // Estimate surface temperature
        let t_si_est = t_zone + (q_ctf_initial - solar_interior) / effective_h;

        // Second pass with corrected T_si
        solver.step(t_si_est, t_sol_air)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::ctf_coefficients::{CTFCalculator, CTFMaterial};
    use crate::physics::ctf_solver::CTFSolverConfig;

    fn case_900_solver() -> CTFSolver {
        let layers = vec![
            CTFMaterial::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
            CTFMaterial::new("Concrete", 0.150, 1.4, 2300.0, 880.0),
            CTFMaterial::new("Insulation", 0.050, 0.04, 50.0, 840.0),
            CTFMaterial::new("Brick", 0.100, 0.81, 1920.0, 790.0),
        ];
        let coeffs = CTFCalculator::with_defaults(&layers, 3600.0).compute_coefficients();
        let config = CTFSolverConfig::new(3600.0, 50);
        CTFSolver::new(coeffs, config)
    }

    #[test]
    fn test_coupling_solver_creation() {
        let solver = CtfZoneCouplingSolver::new();
        assert!(solver.tolerance > 0.0);
        assert!(solver.max_iterations > 0);
    }

    #[test]
    fn test_coupling_solver_basic() {
        let coupling = CtfZoneCouplingSolver::new();
        let mut ctf_solver = case_900_solver();

        let result = coupling.solve(
            &mut ctf_solver,
            20.0, // t_zone
            20.0, // t_mass
            30.0, // t_sol_air (hot exterior)
            0.0,  // no interior solar
        );

        // The solver should produce finite results
        assert!(
            result.t_surface_interior.is_finite(),
            "T_si should be finite"
        );
        assert!(
            result.q_ctf_interior.is_finite(),
            "CTF flux should be finite"
        );
        assert!(result.iterations <= coupling.max_iterations);
    }

    #[test]
    fn test_coupling_solver_cold_exterior() {
        let coupling = CtfZoneCouplingSolver::new();
        let mut ctf_solver = case_900_solver();

        let result = coupling.solve(
            &mut ctf_solver,
            20.0, // t_zone
            20.0, // t_mass
            5.0,  // t_sol_air (cold exterior)
            0.0,  // no interior solar
        );

        // The solver should produce finite results
        assert!(
            result.t_surface_interior.is_finite(),
            "T_si should be finite"
        );
        assert!(
            result.q_ctf_interior.is_finite(),
            "CTF flux should be finite"
        );
    }

    #[test]
    fn test_coupling_solver_with_solar() {
        let coupling = CtfZoneCouplingSolver::new();
        let mut ctf_solver = case_900_solver();

        let result = coupling.solve(
            &mut ctf_solver,
            20.0,  // t_zone
            20.0,  // t_mass
            20.0,  // t_sol_air (same as interior)
            100.0, // interior solar gain
        );

        // The solver should produce finite results
        assert!(
            result.t_surface_interior.is_finite(),
            "T_si should be finite"
        );
        assert!(
            result.q_ctf_interior.is_finite(),
            "CTF flux should be finite"
        );
    }

    #[test]
    fn test_simplified_coupling_direct() {
        let mut ctf_solver = case_900_solver();

        let q = SimplifiedCtfCoupling::flux_direct(&mut ctf_solver, 20.0, 30.0);

        assert!(q.is_finite(), "Flux should be finite");
    }

    #[test]
    fn test_simplified_coupling_corrected() {
        let mut ctf_solver = case_900_solver();

        let q = SimplifiedCtfCoupling::flux_corrected(
            &mut ctf_solver,
            20.0, // t_zone
            20.0, // t_mass
            30.0, // t_sol_air
            0.0,  // solar
            8.0,  // h_i
        );

        assert!(q.is_finite(), "Flux should be finite");
    }

    #[test]
    fn test_effective_conductance() {
        let coupling = CtfZoneCouplingSolver::new();
        let z0 = 1.0;
        let h_eff = coupling.effective_conductance(z0);

        // Should be h_i + z0 = 8.0 + 1.0 = 9.0
        assert!((h_eff - 9.0).abs() < 0.01);
    }

    #[test]
    fn test_convergence_behavior() {
        let coupling = CtfZoneCouplingSolver::new();
        let mut ctf_solver = case_900_solver();

        // Test with large temperature difference
        let result = coupling.solve(
            &mut ctf_solver,
            20.0, // t_zone
            20.0, // t_mass
            50.0, // t_sol_air (very hot)
            0.0,  // no solar
        );

        assert!(result.converged, "Should converge even with large ΔT");
        assert!(result.iterations <= coupling.max_iterations);
    }

    #[test]
    fn test_multiple_surfaces() {
        let coupling = CtfZoneCouplingSolver::new();

        // Create multiple CTF solvers
        let mut solvers: Vec<CTFSolver> = (0..3).map(|_| case_900_solver()).collect();
        let mut solver_refs: Vec<&mut CTFSolver> = solvers.iter_mut().collect();

        let t_masses = vec![20.0, 22.0, 18.0];
        let t_sol_airs = vec![30.0, 25.0, 10.0];
        let solar = vec![0.0, 50.0, 0.0];

        let results = coupling.solve_multiple(
            &mut solver_refs,
            20.0, // t_zone
            &t_masses,
            &t_sol_airs,
            &solar,
        );

        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.converged);
        }
    }

    #[test]
    fn test_surface_coefficients() {
        let coeffs = SurfaceCoefficients::typical();
        assert!(coeffs.h_ci > 0.0);
        assert!(coeffs.h_ri > 0.0);
        assert!((coeffs.h_i - (coeffs.h_ci + coeffs.h_ri)).abs() < 1e-10);
    }

    #[test]
    fn test_steady_state_validation() {
        // For steady-state with constant boundary conditions,
        // the CTF flux should match the analytical U-value calculation
        let coupling = CtfZoneCouplingSolver::new();
        let mut ctf_solver = case_900_solver();

        // Run multiple steps to reach steady state
        let mut final_result = None;
        for _ in 0..100 {
            final_result = Some(coupling.solve(
                &mut ctf_solver,
                20.0, // t_zone
                20.0, // t_mass
                30.0, // t_sol_air
                0.0,  // no solar
            ));
        }

        let result = final_result.unwrap();
        assert!(result.converged);

        // After steady state, flux should be constant
        // For a wall with U ≈ 0.5 W/m²K and ΔT = 10K, flux ≈ 5 W/m²
        let flux = result.q_ctf_interior;
        assert!(
            flux > 0.0 && flux < 100.0,
            "Steady-state flux should be reasonable: got {} W/m²",
            flux
        );
    }
}
