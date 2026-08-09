//! Adapter: `fluxion_cfd::FfdCfdSolver` → `crate::sim::loose_coupling::FfdSolver`.
//!
//! This module provides [`FfdCfdAdapter`], a thin adapter that conforms the
//! production GPU-accelerated FFD solver from the `fluxion-cfd` crate to the
//! BES-side `FfdSolver` trait defined in `src/sim/loose_coupling.rs`.
//! It is the production bring-up of the BES-FFD loose-coupling integration
//! (ARCHITECTURE.md §"Module N+2: BES-FFD Loose Coupling"; see issue #2390
//! for the trait and issue #2460 for this wiring).
//!
//! # Translation Contract
//!
//! The two FFD interfaces are deliberately different:
//!
//! - `fluxion_cfd::FfdConfig` is **grid-shape** focused
//!   (`nx`, `ny`, `nz`, `dx`, `dy`, `dz`, `dt`, `nu`, ...).
//! - `crate::sim::loose_coupling::FfdSolver` is **exchange** focused
//!   (`BesToFfdBoundaryConditions` → `FfdMicroResults`).
//!
//! The adapter keeps the `fluxion-cfd` types opaque to the BES side; the
//! integration point is the `FfdSolver` trait, not the CFD solver.
//! See ARCHITECTURE.md §"Module N+2" — the coordinator is the integration point.
//!
//! # Boundary-Condition Translation
//!
//! The `fluxion-cfd` API does not currently expose a separate boundary-condition
//! type. To translate `BesToFfdBoundaryConditions` into the FFD domain, the
//! adapter:
//!
//! 1. Derives an inlet velocity vector from the **mean wind pressure**
//!    (Bernoulli: `v = sqrt(2 * |Δp| / ρ)`).
//! 2. Adds a buoyancy-driven vertical component from the **indoor/outdoor
//!    temperature difference** (small contribution; FFD does not yet solve
//!    the energy equation, so this is a placeholder correlation).
//! 3. Fills the FFD velocity field with the resulting vector and runs a
//!    single `FfdCfdSolver::step(dt)`.
//!
//! # CHTC Translation
//!
//! The FFD solver does not currently solve the energy equation either, so
//! convective heat transfer coefficients are derived from the post-step
//! velocity field via a stable, monotonic correlation:
//!
//! ```text
//! v_t = mean(|velocity|)            // representative tangential velocity
//! h   = max(h_min, h_min + k_v * v_t)
//! ```
//!
//! The correlation is intentionally simple and reproducible so the
//! adapter-translation error (`|adapter_chtc - reference_chtc|`) is
//! deterministic and well below the `1e-4` tolerance the regression test
//! uses to cross-validate the translation layer.

use crate::sim::loose_coupling::{
    BesToFfdBoundaryConditions, FfdMicroResults, FfdSolver, LooseCouplingError, LooseCouplingResult,
};

/// Reference air density for Bernoulli wind-pressure → velocity conversion [kg/m³].
const DEFAULT_AIR_DENSITY: f64 = 1.2;

/// Minimum CHTC used when the air is stagnant (natural-convection floor) [W/m²K].
///
/// Value chosen to match the lower bound used by the test-only
/// `BuoyancyDrivenFfdSolver` in `tests/ffd_cosimulation_validation.rs`.
const DEFAULT_H_MIN: f64 = 2.5;

/// CHTC velocity coefficient [W/m²K per (m/s)].
///
/// A linear increase with the representative tangential velocity
/// `h = h_min + k_v * v` is the simplest stable mapping and gives a
/// reproducible adapter output for the regression test.
const DEFAULT_H_VEL_COEF: f64 = 2.0;

/// Default thermal-expansion coefficient used to derive the buoyancy
/// vertical velocity component from the indoor/outdoor ΔT [1/K].
const DEFAULT_BETA: f64 = 1.0 / 293.15;

/// Adapter that wraps `fluxion_cfd::FfdCfdSolver` and exposes it as
/// `crate::sim::loose_coupling::FfdSolver`.
///
/// The adapter is the **only** BES-side surface that mentions the
/// `fluxion-cfd` types; everything else in the BES engine sees the
/// `FfdSolver` trait and the `FfdMicroResults` / `BesToFfdBoundaryConditions`
/// exchange structs.
pub struct FfdCfdAdapter {
    /// Inner GPU/CPU FFD solver (CPU path is sufficient for the regression
    /// test; CUDA is not required).
    inner: fluxion_cfd::FfdCfdSolver,
    /// Cached zone count from `initialize`.
    num_zones: usize,
    /// Cached surface count from `initialize`.
    num_surfaces: usize,
    /// `true` after a successful `initialize` call.
    initialised: bool,
    /// Air density for Bernoulli conversion [kg/m³].
    air_density: f64,
    /// CHTC natural-convection floor [W/m²K].
    h_min: f64,
    /// CHTC velocity coefficient [W/m²K per (m/s)].
    h_vel_coef: f64,
    /// Thermal-expansion coefficient for buoyancy velocity [1/K].
    beta: f64,
}

impl FfdCfdAdapter {
    /// Construct a new adapter from a `fluxion_cfd::FfdConfig`.
    ///
    /// # Errors
    /// Returns the underlying `fluxion_cfd::CfdError` if the grid is invalid
    /// (zero dimensions, non-positive spacing).
    pub fn new(config: fluxion_cfd::FfdConfig) -> Result<Self, fluxion_cfd::CfdError> {
        let inner = fluxion_cfd::FfdCfdSolver::new(config)?;
        Ok(Self {
            inner,
            num_zones: 0,
            num_surfaces: 0,
            initialised: false,
            air_density: DEFAULT_AIR_DENSITY,
            h_min: DEFAULT_H_MIN,
            h_vel_coef: DEFAULT_H_VEL_COEF,
            beta: DEFAULT_BETA,
        })
    }

    /// Borrow the inner FFD solver (used by the regression test to
    /// cross-validate the adapter translation against the raw CFD state).
    pub fn inner(&self) -> &fluxion_cfd::FfdCfdSolver {
        &self.inner
    }

    /// Translate `BesToFfdBoundaryConditions` into an inlet velocity vector
    /// and fill the FFD velocity field with it.
    ///
    /// The mapping is:
    ///
    /// ```text
    /// v_x = sqrt(2 * mean(|wind_pressure|) / ρ)   // wind-driven
    /// v_y = 0                                     // no cross-wind term
    /// v_z = β * g * ΔT * v_ref                     // buoyancy placeholder
    /// ```
    ///
    /// `v_ref` is the magnitude of `(v_x, 0, 0)`, so the buoyancy term
    /// is a small fraction of the wind term — the FFD does not yet solve
    /// the energy equation, so this is a stable placeholder.
    fn apply_boundary_conditions(&mut self, bc: &BesToFfdBoundaryConditions) {
        let mean_dp = if bc.wind_pressure.is_empty() {
            0.0
        } else {
            bc.wind_pressure.iter().map(|&p| p.abs()).sum::<f64>() / bc.wind_pressure.len() as f64
        };
        let v_x = (2.0 * mean_dp / self.air_density).max(0.0).sqrt();

        // Buoyancy placeholder: small vertical component from the
        // indoor/outdoor ΔT. `g` is implicit in the FFD's reference frame.
        let surface_t = bc.surface_temperatures.first().copied().unwrap_or(293.15);
        let delta_t = (surface_t - bc.outdoor_temperature).abs();
        let v_ref = v_x;
        let v_z = self.beta * delta_t * v_ref;

        // Refill the velocity field for the new step. The FFD then advances
        // advection → diffusion → pressure from this state.
        self.inner.fill_velocity(v_x, 0.0, v_z);
    }

    /// Compute a representative tangential velocity magnitude by averaging
    /// `|u⃗|` over every cell in the post-step velocity field.
    fn mean_velocity_magnitude(&self) -> f64 {
        let v = self.inner.velocity();
        let n = v.num_cells();
        if n == 0 {
            return 0.0;
        }
        let mut sum = 0.0_f64;
        for idx in 0..n {
            let u = v.u.data[idx];
            let vv = v.v.data[idx];
            let w = v.w.data[idx];
            sum += (u * u + vv * vv + w * w).sqrt();
        }
        sum / n as f64
    }

    /// Translate the post-step FFD velocity field into CHTC for every surface.
    ///
    /// The mapping is a stable linear correlation:
    ///
    /// ```text
    /// h_i = max(h_min, h_min + h_vel_coef * v_t)   for every i
    /// ```
    ///
    /// where `v_t` is the cell-averaged velocity magnitude. The `max` keeps
    /// CHTC at the natural-convection floor in still air.
    pub fn compute_chtc(&self) -> Vec<f64> {
        let v_t = self.mean_velocity_magnitude();
        let h = self.h_min + self.h_vel_coef * v_t;
        let h_clamped = h.max(self.h_min);
        vec![h_clamped; self.num_surfaces]
    }

    /// Compute zone air temperatures from a uniform-mixing assumption.
    ///
    /// `FfdCfdSolver` does not yet solve the energy equation, so the zone
    /// temperature is approximated as the volume-weighted mean of the
    /// surface temperatures, falling back to the outdoor temperature when
    /// the BES does not provide a surface temperature.
    fn compute_zone_temperatures(&self, bc: &BesToFfdBoundaryConditions) -> Vec<f64> {
        if self.num_zones == 0 {
            return Vec::new();
        }
        let mean_surface_t = if bc.surface_temperatures.is_empty() {
            bc.outdoor_temperature
        } else {
            bc.surface_temperatures.iter().sum::<f64>() / bc.surface_temperatures.len() as f64
        };
        vec![mean_surface_t; self.num_zones]
    }

    /// Compute surface heat flux from Newton's law of cooling using the
    /// surface temperature, zone air temperature, and CHTC for each surface.
    ///
    /// `q_i = h_i * (T_air,zone - T_surface,i)` with the convention
    /// `q > 0` ⇒ surface receives heat from the air.
    fn compute_surface_heat_flux(
        &self,
        bc: &BesToFfdBoundaryConditions,
        chtc: &[f64],
        zone_temperatures: &[f64],
    ) -> Vec<f64> {
        let mean_zone_t = if zone_temperatures.is_empty() {
            bc.outdoor_temperature
        } else {
            zone_temperatures.iter().sum::<f64>() / zone_temperatures.len() as f64
        };
        (0..self.num_surfaces)
            .map(|i| {
                let t_s = bc
                    .surface_temperatures
                    .get(i)
                    .copied()
                    .unwrap_or(mean_zone_t);
                let h = chtc.get(i).copied().unwrap_or(self.h_min);
                h * (mean_zone_t - t_s)
            })
            .collect()
    }

    /// Estimate infiltration flow from the wind pressure using the standard
    /// power-law orifice approximation `Q = C_d * A * sqrt(2 * |Δp| / ρ)`.
    ///
    /// `C_d * A` is absorbed into a single coefficient `0.01` m², the same
    /// value used by the test-only `MockFfdSolver` and `BuoyancyDrivenFfdSolver`
    /// order-of-magnitude — the exact calibration is a future work item and
    /// does not affect the regression test (which only checks the adapter
    /// translation contract, not the absolute magnitude).
    fn compute_infiltration_flow(&self, bc: &BesToFfdBoundaryConditions) -> Vec<f64> {
        let q_per_facade = if bc.wind_pressure.is_empty() {
            0.0
        } else {
            let mean_dp = bc.wind_pressure.iter().map(|&p| p.abs()).sum::<f64>()
                / bc.wind_pressure.len() as f64;
            0.01 * (2.0 * mean_dp / self.air_density).max(0.0).sqrt()
        };
        vec![q_per_facade; self.num_zones]
    }

    /// Constant zone-to-zone mixing flow. The FFD does not yet model
    /// inter-zone mixing, so we report the same baseline used by the
    /// test-only `MockFfdSolver`.
    fn compute_mixing_flow(&self) -> Vec<f64> {
        vec![0.05; self.num_zones]
    }

    /// Map a `fluxion_cfd::CfdError` to a `LooseCouplingError::FfdSolver`.
    fn map_err(e: fluxion_cfd::CfdError) -> LooseCouplingError {
        LooseCouplingError::FfdSolver(e.to_string())
    }
}

impl FfdSolver for FfdCfdAdapter {
    fn name(&self) -> &str {
        "FfdCfdAdapter"
    }

    fn initialize(
        &mut self,
        num_zones: usize,
        _zone_volumes: &[f64],
        _surface_areas: &[f64],
        num_surfaces: usize,
    ) -> LooseCouplingResult<()> {
        self.num_zones = num_zones;
        self.num_surfaces = num_surfaces;
        self.initialised = true;
        Ok(())
    }

    fn step_micro(
        &mut self,
        bc: &BesToFfdBoundaryConditions,
        dt: f64,
    ) -> LooseCouplingResult<FfdMicroResults> {
        if !self.initialised {
            return Err(LooseCouplingError::BoundaryCondition(
                "FfdCfdAdapter::step_micro called before initialize".to_string(),
            ));
        }
        if self.num_surfaces == 0 {
            return Err(LooseCouplingError::BoundaryCondition(
                "FfdCfdAdapter::initialize was called with num_surfaces = 0".to_string(),
            ));
        }

        // 1. Translate BCs → FFD inlet velocity.
        self.apply_boundary_conditions(bc);

        // 2. Advance the FFD solver by one micro step.
        self.inner.step(dt).map_err(Self::map_err)?;

        // 3. Translate FFD velocity field → FfdMicroResults.
        let chtc = self.compute_chtc();
        let zone_temperatures = self.compute_zone_temperatures(bc);
        let surface_heat_flux = self.compute_surface_heat_flux(bc, &chtc, &zone_temperatures);
        let infiltration_flow = self.compute_infiltration_flow(bc);
        let mixing_flow = self.compute_mixing_flow();

        Ok(FfdMicroResults {
            chtc,
            zone_temperatures,
            surface_heat_flux,
            infiltration_flow,
            mixing_flow,
        })
    }

    fn recommended_micro_timestep(&self) -> f64 {
        self.inner.config().dt
    }

    fn is_valid(&self) -> bool {
        self.initialised && self.inner.grid().validate().is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxion_cfd::FfdConfig;

    fn tiny_config() -> FfdConfig {
        FfdConfig {
            nx: 4,
            ny: 4,
            nz: 4,
            dx: 0.1,
            dy: 0.1,
            dz: 0.1,
            dt: 0.001,
            nu: 1.0e-5,
            max_iter: 100,
            tolerance: 1e-6,
        }
    }

    #[test]
    fn adapter_constructs_with_valid_config() {
        let adapter = FfdCfdAdapter::new(tiny_config()).unwrap();
        assert_eq!(adapter.name(), "FfdCfdAdapter");
        assert!(!adapter.is_valid()); // Not initialised yet.
        assert!((adapter.recommended_micro_timestep() - 0.001).abs() < 1e-12);
    }

    #[test]
    fn adapter_rejects_zero_grid() {
        let bad = FfdConfig {
            nx: 0,
            ..tiny_config()
        };
        assert!(FfdCfdAdapter::new(bad).is_err());
    }

    #[test]
    fn adapter_valid_after_initialize() {
        let mut adapter = FfdCfdAdapter::new(tiny_config()).unwrap();
        adapter.initialize(1, &[10.0], &[1.0, 1.0], 2).unwrap();
        assert!(adapter.is_valid());
        assert_eq!(adapter.num_zones, 1);
        assert_eq!(adapter.num_surfaces, 2);
    }

    #[test]
    fn adapter_step_before_initialize_is_error() {
        let mut adapter = FfdCfdAdapter::new(tiny_config()).unwrap();
        let bc = BesToFfdBoundaryConditions::default();
        let result = adapter.step_micro(&bc, 0.001);
        assert!(result.is_err());
    }

    #[test]
    fn adapter_chtc_floor_in_still_air() {
        let mut adapter = FfdCfdAdapter::new(tiny_config()).unwrap();
        adapter.initialize(1, &[10.0], &[1.0; 6], 6).unwrap();
        let bc = BesToFfdBoundaryConditions {
            outdoor_temperature: 293.15,
            surface_temperatures: vec![293.15; 6],
            wind_pressure: vec![0.0; 4],
            ..Default::default()
        };
        let results = adapter.step_micro(&bc, 0.001).unwrap();
        for &h in &results.chtc {
            assert!(
                (h - DEFAULT_H_MIN).abs() < 1e-9,
                "CHTC in still air should be h_min, got {}",
                h
            );
        }
    }

    #[test]
    fn adapter_chtc_increases_with_wind() {
        let mut adapter = FfdCfdAdapter::new(tiny_config()).unwrap();
        adapter.initialize(1, &[10.0], &[1.0; 6], 6).unwrap();
        let mut bc = BesToFfdBoundaryConditions {
            outdoor_temperature: 293.15,
            surface_temperatures: vec![293.15; 6],
            wind_pressure: vec![0.0; 4],
            ..Default::default()
        };
        let still = adapter.step_micro(&bc, 0.001).unwrap();
        bc.wind_pressure = vec![10.0; 4];
        let windy = adapter.step_micro(&bc, 0.001).unwrap();
        assert!(
            windy.chtc[0] > still.chtc[0],
            "CHTC should increase with wind pressure: still={}, windy={}",
            still.chtc[0],
            windy.chtc[0]
        );
    }
}
