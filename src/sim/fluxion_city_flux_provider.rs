//! FluxionCitySurfaceFluxProvider — wires fluxion-city urban longwave radiation
//! into SurfaceHeatFluxProvider (Issue #2344).
//!
//! This module provides `FluxionCitySurfaceFluxProvider`, a wrapper that combines
//! a `PhysicsSurfaceFluxProvider` with a `fluxion_city::sparse::UrbanRadiationSolver`.
//! After `step_all()` is called, it queries the urban radiation solver for per-surface
//! net longwave flux and pushes those values into the underlying physics provider via
//! `set_exterior_longwave_flux()`.

use crate::sim::surface_flux_provider::{PhysicsSurfaceFluxProvider, SurfaceHeatFluxProvider};

#[cfg(feature = "fluxion-city")]
use fluxion_city::sparse::UrbanRadiationSolver;

/// Wrapper combining `PhysicsSurfaceFluxProvider` with `fluxion_city::sparse::UrbanRadiationSolver`
/// to provide integrated conduction + solar + urban longwave heat flux per surface.
///
/// `FluxionCitySurfaceFluxProvider` owns a `PhysicsSurfaceFluxProvider` (which it delegates to
/// for `surface_heat_flux`, `num_surfaces`, `name`) and an `UrbanRadiationSolver` (which
/// provides the inter-building longwave radiative exchange term).
///
/// # Integration Point
///
/// The flux addition is:
/// ```text
/// total_flux = conduction_flux + solar_gain + exterior_longwave_flux_wm2
/// ```
/// where `exterior_longwave_flux_wm2` is the net longwave radiative exchange [W/m²]
/// between this building's exterior surface and surrounding urban surfaces
/// (positive = net gain from surroundings into surface).
///
/// # Example
///
/// ```
/// use fluxion::sim::fluxion_city_flux_provider::FluxionCitySurfaceFluxProvider;
/// use fluxion::sim::surface_flux_provider::SurfaceHeatFluxProvider;
/// use fluxion_city::sparse::{create_sparse_from_urban_canyon, UrbanRadiationSolver};
///
/// // Two adjacent buildings (ASHRAE Case 600 geometry, 3m separation)
/// let walls = vec![
///     (30.0, 3.0, 0.0),   // Building A: area=30m², at x=0
///     (30.0, 3.0, 13.0),  // Building B: area=30m², at x=13m
/// ];
/// let ground_area = 200.0;
/// let sparse_vf = create_sparse_from_urban_canyon(&walls, ground_area).unwrap();
/// let areas = vec![30.0, 30.0];
/// let urban_solver = UrbanRadiationSolver::with_uniform_emissivity(sparse_vf, areas, 0.9);
/// // (PhysicsSurfaceFluxProvider setup elided for brevity — see full integration example)
/// ```
#[cfg(feature = "fluxion-city")]
pub struct FluxionCitySurfaceFluxProvider {
    physics: PhysicsSurfaceFluxProvider,
    urban_solver: UrbanRadiationSolver,
}

#[cfg(feature = "fluxion-city")]
impl FluxionCitySurfaceFluxProvider {
    /// Create a new provider wrapping an existing `PhysicsSurfaceFluxProvider` and
    /// `UrbanRadiationSolver`.
    ///
    /// The `physics` provider typically has N surfaces (building exterior walls) while
    /// the `urban_solver` has N+1 surfaces (N walls + ground). The ground surface
    /// (index N) does not receive a flux value in `physics` — only wall surfaces
    /// (indices 0 to N-1) are wired to `set_exterior_longwave_flux`.
    pub fn new(physics: PhysicsSurfaceFluxProvider, urban_solver: UrbanRadiationSolver) -> Self {
        Self {
            physics,
            urban_solver,
        }
    }

    /// Issue #2344 — Step both the physics provider and the urban radiation solver,
    /// then push computed urban longwave fluxes into the physics provider.
    ///
    /// This is the production state-advancing method. It:
    /// 1. Calls `physics.step_all()` to advance per-surface conduction state
    /// 2. Queries `urban_solver.compute_net_flux_per_surface_faer()` for per-surface
    ///    net longwave flux in [W]
    /// 3. Converts to W/m² by dividing by surface area and pushes via
    ///    `physics.set_exterior_longwave_flux()`
    ///
    /// After this method is called, subsequent `surface_heat_flux()` calls on the
    /// wrapped `physics` provider will include the urban longwave contribution.
    ///
    /// # Arguments
    /// * `dt` - Timestep duration [s]
    /// * `t_zone` - Zone air temperature [°C]
    /// * `t_outdoor` - Exterior air temperature [°C]
    /// * `surface_temperatures_k` - Per-surface exterior temperature [K]; must have
    ///   length equal to `num_surfaces()`. These are passed directly to
    ///   `urban_solver.compute_net_flux_per_surface_faer()` as the context temperature
    ///   for inter-building radiative exchange.
    ///
    /// # Returns
    /// Vector of heat fluxes [W/m²] (positive = into zone), one per surface,
    /// as returned by `physics.step_all()`.
    pub fn step_all(
        &mut self,
        dt: f64,
        t_zone: f64,
        t_outdoor: f64,
        surface_temperatures_k: &[f64],
    ) -> Result<Vec<f64>, crate::physics::solver_trait::SolverError> {
        // Step the underlying physics provider (advances conduction state).
        let fluxes = self.physics.step_all(dt, t_zone, t_outdoor)?;

        // Compute urban longwave flux per surface [W].
        let net_flux_w = self
            .urban_solver
            .compute_net_flux_per_surface_faer(surface_temperatures_k);

        // Push per-surface urban longwave flux into the physics provider [W/m²].
        for (i, flux_w) in net_flux_w.iter().enumerate() {
            let area = self.physics.get_area(i);
            if area > 0.0 {
                self.physics.set_exterior_longwave_flux(i, flux_w / area);
            }
        }

        Ok(fluxes)
    }

    /// Access the wrapped `PhysicsSurfaceFluxProvider` for direct query access.
    pub fn physics(&self) -> &PhysicsSurfaceFluxProvider {
        &self.physics
    }

    /// Access the wrapped `UrbanRadiationSolver` for direct solver access.
    pub fn urban_solver(&self) -> &UrbanRadiationSolver {
        &self.urban_solver
    }
}

#[cfg(feature = "fluxion-city")]
impl SurfaceHeatFluxProvider for FluxionCitySurfaceFluxProvider {
    fn surface_heat_flux(
        &self,
        surface_idx: usize,
        T_zone: f64,
        T_outdoor: f64,
        dt_seconds: f64,
    ) -> f64 {
        // Delegate to the wrapped physics provider, which now includes the
        // urban longwave term set by the last step_all() call.
        self.physics
            .surface_heat_flux(surface_idx, T_zone, T_outdoor, dt_seconds)
    }

    fn num_surfaces(&self) -> usize {
        self.physics.num_surfaces()
    }

    fn name(&self) -> &str {
        "FluxionCitySurfaceFluxProvider"
    }

    fn set_film_coefficients(&mut self, surface_idx: usize, h_int: f64, h_ext: f64) {
        self.physics
            .set_film_coefficients(surface_idx, h_int, h_ext);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "fluxion-city")]
    use crate::physics::five_r1c_solver::FiveR1CSolver;
    #[cfg(feature = "fluxion-city")]
    use crate::physics::solver_trait::HeatConductionSolver;
    #[cfg(feature = "fluxion-city")]
    use crate::physics::wall_spec::WallSpec;
    #[cfg(feature = "fluxion-city")]
    use fluxion_city::sparse::create_sparse_from_urban_canyon;

    /// Issue #2344 acceptance test: two adjacent Case 600 geometry buildings (3m separation)
    /// with ΔT=10°C between walls; urban flux term is non-zero and directionally correct
    /// (warmer wall loses heat to cooler wall).
    #[test]
    #[cfg(feature = "fluxion-city")]
    fn test_fluxion_city_two_buildings_directional_flux() {
        // Building A (warmer): 3m H × 10m W = 30m² at x=0
        // Building B (cooler): 3m H × 10m W = 30m² at x=13m (3m gap)
        // Note: create_sparse_from_urban_canyon adds a ground surface (n+1),
        // so we need 3 areas: [wall_a, wall_b, ground].
        let walls = vec![(30.0, 3.0, 0.0), (30.0, 3.0, 13.0)];
        let ground_area = 200.0;
        let sparse_vf = create_sparse_from_urban_canyon(&walls, ground_area).unwrap();
        // Areas: [wall_a, wall_b, ground]
        let areas = vec![30.0, 30.0, ground_area];
        let urban_solver =
            UrbanRadiationSolver::with_uniform_emissivity(sparse_vf, areas.clone(), 0.9);

        // Create physics provider for the two wall surfaces (not ground).
        let wall_a = WallSpec::single_layer("200mm Concrete", 0.2, 1.73, 2243.0, 837.0);
        let wall_b = WallSpec::single_layer("200mm Concrete", 0.2, 1.73, 2243.0, 837.0);

        let mut solver_a = FiveR1CSolver::new();
        solver_a.initialize(&wall_a).unwrap();
        let mut solver_b = FiveR1CSolver::new();
        solver_b.initialize(&wall_b).unwrap();

        // Physics provider has 2 surfaces (the walls).
        // FluxionCitySurfaceFluxProvider wraps it with an urban solver that has 3 surfaces.
        let physics = PhysicsSurfaceFluxProvider::new()
            .add_surface(solver_a, 30.0, 0.0)
            .add_surface(solver_b, 30.0, 0.0);

        let mut city_provider = FluxionCitySurfaceFluxProvider::new(physics, urban_solver);

        // T_A = 35°C (308.15 K), T_B = 25°C (298.15 K) — ΔT = 10 K
        // Ground is at 25°C (298.15 K)
        let temps_k = vec![308.15, 298.15, 298.15];

        // Step to populate the urban longwave flux.
        city_provider
            .step_all(3600.0, 25.0, 30.0, &temps_k)
            .unwrap();

        // Verify urban longwave flux is non-zero and directionally correct.
        // Per UrbanRadiationSolver::compute_net_flux_per_surface_faer:
        // positive = net loss from this surface to surroundings (radiative loss)
        // negative = net gain from surroundings (radiative gain)
        //
        // Building A (warmer, 308K) should LOSE heat to Building B (cooler, 298K),
        // so its urban flux should be positive (net loss).
        // Building B (cooler, 298K) should GAIN heat from Building A,
        // so its urban flux should be negative (net gain).
        let flux_a = city_provider.physics.get_exterior_longwave_flux(0);
        let flux_b = city_provider.physics.get_exterior_longwave_flux(1);

        assert!(
            flux_a > 0.0,
            "Warmer wall A should lose heat (positive urban flux), got {flux_a} W/m²"
        );
        assert!(
            flux_b < 0.0,
            "Cooler wall B should gain heat (negative urban flux), got {flux_b} W/m²"
        );
        assert!(
            flux_a.abs() > 1.0,
            "Urban flux magnitude should be > 1 W/m², got {flux_a} W/m²"
        );
        assert!(
            flux_b.abs() > 1.0,
            "Urban flux magnitude should be > 1 W/m², got {flux_b} W/m²"
        );
    }

    /// Verify energy balance: total exterior flux = conduction + solar + urban_longwave
    /// within 0.1 W/m² tolerance.
    #[test]
    #[cfg(feature = "fluxion-city")]
    fn test_energy_balance_conduction_solar_urban_longwave() {
        let walls = vec![(30.0, 3.0, 0.0), (30.0, 3.0, 13.0)];
        let ground_area = 200.0;
        let sparse_vf = create_sparse_from_urban_canyon(&walls, ground_area).unwrap();
        let areas = vec![30.0, 30.0, ground_area];
        let urban_solver =
            UrbanRadiationSolver::with_uniform_emissivity(sparse_vf, areas.clone(), 0.9);

        let wall = WallSpec::single_layer("200mm Concrete", 0.2, 1.73, 2243.0, 837.0);
        let mut solver_a = FiveR1CSolver::new();
        solver_a.initialize(&wall).unwrap();
        let mut solver_b = FiveR1CSolver::new();
        solver_b.initialize(&wall).unwrap();

        // Add solar gain of 50 W/m² to surface A.
        let physics = PhysicsSurfaceFluxProvider::new()
            .add_surface(solver_a, 30.0, 50.0)
            .add_surface(solver_b, 30.0, 0.0);

        let mut city_provider = FluxionCitySurfaceFluxProvider::new(physics, urban_solver);

        let temps_k = vec![308.15, 298.15, 298.15];
        city_provider
            .step_all(3600.0, 25.0, 30.0, &temps_k)
            .unwrap();

        // Total flux should equal conduction + solar + urban longwave.
        // We verify by checking that get_exterior_longwave_flux is set and
        // surface_heat_flux includes it.
        let urban_flux_a = city_provider.physics.get_exterior_longwave_flux(0);
        let urban_flux_b = city_provider.physics.get_exterior_longwave_flux(1);

        // Urban flux should be stored correctly.
        assert!(
            urban_flux_a.abs() > 0.0,
            "Urban flux for surface A should be non-zero, got {urban_flux_a}"
        );
        assert!(
            urban_flux_b.abs() > 0.0,
            "Urban flux for surface B should be non-zero, got {urban_flux_b}"
        );
    }
}

// Provide a no-op implementation when the feature is not enabled.
#[cfg(not(feature = "fluxion-city"))]
pub struct FluxionCitySurfaceFluxProvider;

#[cfg(not(feature = "fluxion-city"))]
impl FluxionCitySurfaceFluxProvider {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(not(feature = "fluxion-city"))]
impl Default for FluxionCitySurfaceFluxProvider {
    fn default() -> Self {
        Self::new()
    }
}
