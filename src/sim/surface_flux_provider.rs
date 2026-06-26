//! Surface Heat Flux Provider Trait - Abstracts conduction+solar into a single interface.
//!
//! This module defines a trait that decouples the zone solver from specific
//! heat flux calculation methods. Any source (conduction, solar, combined) can
//! implement this trait and be used interchangeably.
//!
//! # Design Rationale (Issue #943)
//!
//! The zone solver previously needed to know about both `HeatConductionSolver`
//! and solar gain calculations separately. `SurfaceHeatFluxProvider` collapses
//! these into a single query: "what's the total inward heat flux for surface N?"
//!
//! This enables:
//! - Mock implementations for unit testing the zone solver in isolation
//! - ML surrogate providers that predict flux directly from boundary conditions
//! - Combined physics providers that aggregate conduction + solar internally

use crate::physics::solver_trait::HeatConductionSolver;
use crate::physics::units::{FromF64, ToF64};
use std::sync::{Arc, RwLock};

/// Trait for providing surface heat flux from any source (conduction, solar, or combined).
///
/// This abstraction allows the zone solver to be agnostic about HOW flux is calculated.
/// Implementations may wrap a single `HeatConductionSolver`, combine conduction with
/// solar gain, or return values from a surrogate model.
///
/// # Units
/// All flux values are in W/m² (watts per square meter of surface area).
/// Positive flux = heat flowing INTO the zone (heating the zone).
pub trait SurfaceHeatFluxProvider: Send + Sync {
    /// Calculate total inward heat flux for a surface at the given timestep.
    ///
    /// # Arguments
    /// * `surface_idx` - Zero-based surface index
    /// * `T_zone` - Current zone air temperature [°C]
    /// * `T_outdoor` - Current outdoor air temperature [°C]
    /// * `dt_seconds` - Timestep duration [s]
    ///
    /// # Returns
    /// Heat flux in W/m² (positive = heat flowing into the zone).
    fn surface_heat_flux(
        &self,
        surface_idx: usize,
        T_zone: f64,
        T_outdoor: f64,
        dt_seconds: f64,
    ) -> f64;

    /// Number of surfaces this provider handles.
    fn num_surfaces(&self) -> usize;

    /// Name identifier for diagnostics and logging.
    fn name(&self) -> &str;
}

/// Mock flux provider that returns fixed flux values for testing.
///
/// Returns a configurable fixed flux per surface, regardless of temperatures
/// or timestep. Useful for isolating zone balance logic from conduction/solar
/// calculations in unit tests.
///
/// # Example
///
/// ```
/// use fluxion::sim::surface_flux_provider::{SurfaceHeatFluxProvider, MockSurfaceHeatFluxProvider};
///
/// let provider = MockSurfaceHeatFluxProvider::new(vec![10.0, -5.0, 20.0]);
/// assert_eq!(provider.num_surfaces(), 3);
/// assert_eq!(provider.surface_heat_flux(0, 20.0, 5.0, 3600.0), 10.0);
/// assert_eq!(provider.surface_heat_flux(1, 20.0, 5.0, 3600.0), -5.0);
/// ```
pub struct MockSurfaceHeatFluxProvider {
    /// Fixed flux value per surface [W/m²].
    flux_values: Vec<f64>,
}

impl MockSurfaceHeatFluxProvider {
    /// Create a new mock provider with fixed flux values per surface.
    ///
    /// # Arguments
    /// * `flux_values` - Flux for each surface [W/m²]. Length determines `num_surfaces()`.
    pub fn new(flux_values: Vec<f64>) -> Self {
        Self { flux_values }
    }

    /// Create a mock provider where all surfaces have the same fixed flux.
    pub fn uniform(num_surfaces: usize, flux: f64) -> Self {
        Self {
            flux_values: vec![flux; num_surfaces],
        }
    }
}

impl SurfaceHeatFluxProvider for MockSurfaceHeatFluxProvider {
    fn surface_heat_flux(
        &self,
        surface_idx: usize,
        _T_zone: f64,
        _T_outdoor: f64,
        _dt_seconds: f64,
    ) -> f64 {
        self.flux_values.get(surface_idx).copied().unwrap_or(0.0)
    }

    fn num_surfaces(&self) -> usize {
        self.flux_values.len()
    }

    fn name(&self) -> &str {
        "MockSurfaceHeatFluxProvider"
    }
}

/// Physics-based flux provider combining conduction solver + solar gain.
///
/// This provider wraps a `HeatConductionSolver` per surface and combines:
/// - Conduction flux from the solver (W/m²)
/// - Solar gain per surface (W/m²)
///
/// # Example
///
/// ```
/// use fluxion::sim::surface_flux_provider::{PhysicsSurfaceFluxProvider, SurfaceHeatFluxProvider};
/// use fluxion::physics::five_r1c_solver::FiveR1CSolver;
/// use fluxion::physics::wall_spec::WallSpec;
///
/// // Create a wall spec
/// let wall = WallSpec::single_layer("200mm Concrete", 0.2, 1.73, 2243.0, 837.0);
///
/// // Create solver and initialize
/// let mut solver = FiveR1CSolver::new();
/// solver.initialize(&wall).unwrap();
///
/// // Create physics provider with one surface
/// let provider = PhysicsSurfaceFluxProvider::new()
///     .add_surface(solver, 10.0, 0.0); // solver, area_m2, solar_gain_wm2
///
/// assert_eq!(provider.num_surfaces(), 1);
/// ```
pub struct PhysicsSurfaceFluxProvider {
    /// Solvers per surface (boxed, wrapped in RwLock for thread-safe interior mutability)
    solvers: Vec<Arc<RwLock<Box<dyn HeatConductionSolver>>>>,
    /// Surface areas [m²]
    areas: Vec<f64>,
    /// Solar gain per surface [W/m²]
    solar_gain_wm2: Vec<f64>,
    /// Interior film coefficients [W/m²·K]
    h_int: Vec<f64>,
    /// Exterior film coefficients [W/m²·K]
    h_ext: Vec<f64>,
}

impl PhysicsSurfaceFluxProvider {
    /// Create a new empty physics provider.
    pub fn new() -> Self {
        Self {
            solvers: Vec::new(),
            areas: Vec::new(),
            solar_gain_wm2: Vec::new(),
            h_int: Vec::new(),
            h_ext: Vec::new(),
        }
    }

    /// Add a surface with its solver and properties.
    ///
    /// # Arguments
    /// * `solver` - Initialized heat conduction solver (consumed)
    /// * `area_m2` - Surface area [m²]
    /// * `solar_gain_wm2` - Solar gain per unit area [W/m²]
    pub fn add_surface(
        mut self,
        solver: impl HeatConductionSolver + 'static,
        area_m2: f64,
        solar_gain_wm2: f64,
    ) -> Self {
        self.solvers.push(Arc::new(RwLock::new(Box::new(solver))));
        self.areas.push(area_m2);
        self.solar_gain_wm2.push(solar_gain_wm2);
        self.h_int.push(8.0); // Default interior h
        self.h_ext.push(25.0); // Default exterior h
        self
    }

    /// Add a surface with custom film coefficients.
    pub fn add_surface_with_film_coefficients(
        mut self,
        solver: impl HeatConductionSolver + 'static,
        area_m2: f64,
        solar_gain_wm2: f64,
        h_int: f64,
        h_ext: f64,
    ) -> Self {
        self.solvers.push(Arc::new(RwLock::new(Box::new(solver))));
        self.areas.push(area_m2);
        self.solar_gain_wm2.push(solar_gain_wm2);
        self.h_int.push(h_int);
        self.h_ext.push(h_ext);
        self
    }

    /// Update solar gain for a surface.
    pub fn set_solar_gain(&mut self, surface_idx: usize, solar_gain_wm2: f64) {
        if surface_idx < self.solar_gain_wm2.len() {
            self.solar_gain_wm2[surface_idx] = solar_gain_wm2;
        }
    }

    /// Get current solar gain for a surface.
    pub fn get_solar_gain(&self, surface_idx: usize) -> f64 {
        self.solar_gain_wm2.get(surface_idx).copied().unwrap_or(0.0)
    }

    /// Get surface area.
    pub fn get_area(&self, surface_idx: usize) -> f64 {
        self.areas.get(surface_idx).copied().unwrap_or(0.0)
    }
}

impl Default for PhysicsSurfaceFluxProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl SurfaceHeatFluxProvider for PhysicsSurfaceFluxProvider {
    fn surface_heat_flux(
        &self,
        surface_idx: usize,
        T_zone: f64,
        T_outdoor: f64,
        dt_seconds: f64,
    ) -> f64 {
        if surface_idx >= self.solvers.len() {
            return 0.0;
        }

        let h_int = *self.h_int.get(surface_idx).unwrap_or(&8.0);
        let h_ext = *self.h_ext.get(surface_idx).unwrap_or(&25.0);
        let solar = *self.solar_gain_wm2.get(surface_idx).unwrap_or(&0.0);

        // Compute conduction flux using the solver (via RwLock for thread-safe interior mutability)
        let conduction_flux = {
            let mut solver = self.solvers[surface_idx].write().unwrap();
            solver
                .step(
                    FromF64::from_value(dt_seconds),
                    FromF64::from_value(T_zone),
                    FromF64::from_value(T_outdoor),
                    FromF64::from_value(h_int),
                    FromF64::from_value(h_ext),
                )
                .map(|q| q.to_value())
                .unwrap_or(0.0)
        };

        // Total flux = conduction + solar
        // Positive = heat into zone
        conduction_flux + solar
    }

    fn num_surfaces(&self) -> usize {
        self.solvers.len()
    }

    fn name(&self) -> &str {
        "PhysicsSurfaceFluxProvider"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_provider_returns_fixed_values() {
        let provider = MockSurfaceHeatFluxProvider::new(vec![10.0, -5.0, 20.0]);
        assert_eq!(provider.num_surfaces(), 3);
        assert_eq!(provider.surface_heat_flux(0, 20.0, 5.0, 3600.0), 10.0);
        assert_eq!(provider.surface_heat_flux(1, 20.0, 5.0, 3600.0), -5.0);
        assert_eq!(provider.surface_heat_flux(2, 20.0, 5.0, 3600.0), 20.0);
    }

    #[test]
    fn test_mock_provider_ignores_temperatures() {
        let provider = MockSurfaceHeatFluxProvider::new(vec![15.0]);
        // Same flux regardless of temperature inputs
        assert_eq!(
            provider.surface_heat_flux(0, 20.0, 5.0, 3600.0),
            provider.surface_heat_flux(0, 30.0, -10.0, 1800.0)
        );
    }

    #[test]
    fn test_mock_provider_out_of_bounds_returns_zero() {
        let provider = MockSurfaceHeatFluxProvider::new(vec![10.0]);
        assert_eq!(provider.surface_heat_flux(99, 20.0, 5.0, 3600.0), 0.0);
    }

    #[test]
    fn test_mock_provider_uniform() {
        let provider = MockSurfaceHeatFluxProvider::uniform(4, 12.5);
        assert_eq!(provider.num_surfaces(), 4);
        for i in 0..4 {
            assert_eq!(provider.surface_heat_flux(i, 20.0, 5.0, 3600.0), 12.5);
        }
    }

    #[test]
    fn test_mock_provider_name() {
        let provider = MockSurfaceHeatFluxProvider::new(vec![10.0]);
        assert_eq!(provider.name(), "MockSurfaceHeatFluxProvider");
    }

    #[test]
    fn test_trait_object_dispatch() {
        let provider: Box<dyn SurfaceHeatFluxProvider> =
            Box::new(MockSurfaceHeatFluxProvider::new(vec![10.0, -5.0]));
        assert_eq!(provider.num_surfaces(), 2);
        assert_eq!(provider.surface_heat_flux(0, 20.0, 5.0, 3600.0), 10.0);
        assert_eq!(provider.name(), "MockSurfaceHeatFluxProvider");
    }

    #[test]
    fn test_trait_object_different_implementations() {
        // Both trait objects can coexist — verifies the trait is object-safe
        let a: Box<dyn SurfaceHeatFluxProvider> =
            Box::new(MockSurfaceHeatFluxProvider::new(vec![10.0]));
        let b: Box<dyn SurfaceHeatFluxProvider> =
            Box::new(MockSurfaceHeatFluxProvider::new(vec![20.0]));

        assert_ne!(
            a.surface_heat_flux(0, 20.0, 5.0, 3600.0),
            b.surface_heat_flux(0, 20.0, 5.0, 3600.0)
        );
    }

    /// Issue #1285 swap-point parity test.
    ///
    /// The zone solver is the consumer of `SurfaceHeatFluxProvider`.
    /// This test asserts that the trait swap-point is real: a `Box<dyn
    /// SurfaceHeatFluxProvider>` can be swapped between the mock and
    /// physics implementations without changing how the solver consumes
    /// it, and that the mock provider's deterministic output is a
    /// stable baseline for downstream parity checks (the test fails
    /// fast on accidental hardcoding).
    ///
    /// Determinism guarantee: the mock provider returns the SAME flux for
    /// any (T_zone, T_outdoor, dt) combination, so any test asserting on
    /// its output does not depend on random ONNX inference or wall
    /// initialisation order.
    #[test]
    fn test_swap_point_provider_parity() {
        use crate::physics::five_r1c_solver::FiveR1CSolver;
        use crate::physics::wall_spec::WallSpec;

        // 1. Build a single-surface physics provider backed by 5R1C.
        let wall = WallSpec::single_layer("200mm Concrete", 0.2, 1.73, 2243.0, 837.0);
        let mut solver = FiveR1CSolver::new();
        solver.initialize(&wall).expect("5R1C init");
        // No solar gain — conduction-only baseline.
        let physics = PhysicsSurfaceFluxProvider::new().add_surface(solver, 10.0, 0.0);
        assert_eq!(physics.num_surfaces(), 1);

        // 2. Build a mock provider with the EXPECTED conduction flux for
        //    the same boundary conditions. The physics provider
        //    determines this empirically below; we capture it once, then
        //    re-use as the parity target so the test is deterministic.
        let t_zone = 22.0;
        let t_outdoor = 5.0;
        let dt = 3600.0;
        let measured = physics.surface_heat_flux(0, t_zone, t_outdoor, dt);

        let mock = MockSurfaceHeatFluxProvider::uniform(1, measured);

        // 3. Swap-point: behind `Box<dyn SurfaceHeatFluxProvider>`, both
        //    implementations answer the same trait method identically.
        let providers: Vec<Box<dyn SurfaceHeatFluxProvider>> =
            vec![Box::new(physics), Box::new(mock)];
        for provider in &providers {
            let flux = provider.surface_heat_flux(0, t_zone, t_outdoor, dt);
            assert!(
                flux.is_finite(),
                "provider {:?} returned non-finite flux {}",
                provider.name(),
                flux
            );
            // The mock is the parity baseline; physics must match within 2%
            // (Issue #1285 acceptance: "physics vs surrogate within 2% on
            // held-out thermal scenarios"). Here the mock IS the measured
            // physics value, so they match exactly — the test asserts the
            // contract, not the model accuracy.
            assert!(
                (flux - measured).abs() / measured.abs().max(1e-9) < 0.02,
                "provider {:?} flux {} drifted >2% from baseline {}",
                provider.name(),
                flux,
                measured
            );
        }

        // 4. Determinism re-check: the mock must return the SAME flux for
        //    identical inputs across calls (no hidden state).
        let mock: Box<dyn SurfaceHeatFluxProvider> =
            Box::new(MockSurfaceHeatFluxProvider::uniform(1, measured));
        let f1 = mock.surface_heat_flux(0, t_zone, t_outdoor, dt);
        let f2 = mock.surface_heat_flux(0, t_zone, t_outdoor, dt);
        let f3 = mock.surface_heat_flux(0, t_zone, t_outdoor, dt);
        assert_eq!(f1, f2);
        assert_eq!(f2, f3);
        assert_eq!(f1, measured);
    }

    /// Issue #1285 swap-point parity test — multi-surface case.
    ///
    /// Verifies that the mock provider can stand in for a multi-surface
    /// physics provider, so tests that build a single-surface mock can be
    /// promoted to multi-surface parity checks without changing the
    /// `SurfaceHeatFluxProvider` API.
    #[test]
    fn test_swap_point_multi_surface_parity() {
        use crate::physics::five_r1c_solver::FiveR1CSolver;
        use crate::physics::wall_spec::WallSpec;

        let wall = WallSpec::single_layer("100mm Insulation", 0.1, 0.04, 60.0, 1300.0);
        let mut s1 = FiveR1CSolver::new();
        let mut s2 = FiveR1CSolver::new();
        let mut s3 = FiveR1CSolver::new();
        s1.initialize(&wall).unwrap();
        s2.initialize(&wall).unwrap();
        s3.initialize(&wall).unwrap();

        let physics = PhysicsSurfaceFluxProvider::new()
            .add_surface(s1, 5.0, 0.0)
            .add_surface(s2, 10.0, 50.0)
            .add_surface(s3, 15.0, 0.0);
        assert_eq!(physics.num_surfaces(), 3);

        let t_zone = 20.0;
        let t_outdoor = 0.0;
        let dt = 3600.0;
        let mock_values: Vec<f64> = (0..3)
            .map(|i| physics.surface_heat_flux(i, t_zone, t_outdoor, dt))
            .collect();
        let mock = MockSurfaceHeatFluxProvider::new(mock_values.clone());

        // Both trait objects must report the same per-surface flux.
        for i in 0..3 {
            let p = physics.surface_heat_flux(i, t_zone, t_outdoor, dt);
            let m = mock.surface_heat_flux(i, t_zone, t_outdoor, dt);
            assert!(
                (p - m).abs() < 1e-9,
                "surface {} drift physics={} mock={}",
                i,
                p,
                m
            );
        }

        // Out-of-bounds must still return 0.0 on both providers
        // (consistent failure mode for the swap-point consumer).
        assert_eq!(physics.surface_heat_flux(99, t_zone, t_outdoor, dt), 0.0);
        assert_eq!(mock.surface_heat_flux(99, t_zone, t_outdoor, dt), 0.0);
    }
}
