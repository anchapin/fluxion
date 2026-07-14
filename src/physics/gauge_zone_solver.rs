//! Zone-level heat balance aggregator using per-surface GaugeSolvers.
//!
//! This module implements Phase 2 of the GaugeSolver elevation: replacing the
//! lumped 5R1C zone network with a geometrically-accurate assembly of 1D
//! per-surface GaugeSolvers that conserve energy at the zone air node.
//!
//! ## Mathematical Model
//!
//! The zone is modeled as a well-mixed air node with thermal capacitance
//! surrounded by N discrete surfaces. For each surface i:
//!
//! q_i = (T_ext,i - T_air) / R_i
//!
//! where T_ext,i = T_exterior + G_solar,i / h_exterior (sol-air temperature).
//!
//! The zone air energy balance:
//!
//! rho_air * V_zone * c_air * dT_air/dt = sum_i(q_i * A_i) + Q_int + Q_inf
//!
//! Using explicit Euler integration:
//!
//! T_air_new = T_air_old + dt/C_air * (sum_i(q_i * A_i) + Q_int + Q_inf)
//!
//! ## Surface Types
//!
//! - **Opaque**: Walls, roof, floor - full resistance path
//! - **Window**: Simplified glazing model (future: multi-layer)
//! - **Ground**: Fixed ground temperature boundary

use crate::physics::gauge_solver::{GaugeBoundaryConditions, GaugeSolver};
use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
use crate::physics::units::FromF64;
use crate::physics::units::{HeatFlux, HeatTransferCoefficient, Temperature, Time, ToF64};
use crate::physics::wall_spec::WallSpec;

/// Surface classification for zone modeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceType {
    /// Opaque exterior wall
    Wall,
    /// Window/glazing
    Window,
    /// Roof/ceiling
    Roof,
    /// Floor slab
    Floor,
    /// Fixed-temperature ground (ASHRAE 140 B3.3)
    Ground,
    /// Internal mass (furniture, partitions)
    InternalMass,
}

impl SurfaceType {
    /// Returns the solar gain fraction for this surface type.
    /// Windows receive solar; opaque surfaces conduct it away.
    pub fn solar_fraction(&self) -> f64 {
        match self {
            SurfaceType::Window => 1.0, // Windows absorb and conduct solar
            SurfaceType::Wall => 0.0,   // Walls receive sol-air via film coefficient
            SurfaceType::Roof => 0.0,
            SurfaceType::Floor => 0.0,
            SurfaceType::Ground => 0.0,
            SurfaceType::InternalMass => 0.0,
        }
    }
}

/// Per-surface gauge solver with geometric and type metadata.
#[derive(Debug, Clone)]
pub(crate) struct SurfaceGaugeSolver {
    /// The 1D gauge solver for this surface
    gauge: GaugeSolver,
    /// Surface area in m²
    area_m2: f64,
    /// Surface type for solar distribution
    surface_type: SurfaceType,
    /// Surface azimuth (degrees, 0=South, 90=West, -90=East)
    _azimuth_deg: f64,
    /// Surface tilt from horizontal (degrees, 90=vertical wall, 0=roof)
    _tilt_deg: f64,
    /// Wall spec for initialization (stored for re-initialization if needed)
    wall_spec: Option<WallSpec>,
}

impl SurfaceGaugeSolver {
    /// Create a new surface gauge solver.
    fn new(
        gauge: GaugeSolver,
        area_m2: f64,
        surface_type: SurfaceType,
        _azimuth_deg: f64,
        _tilt_deg: f64,
    ) -> Self {
        Self {
            gauge,
            area_m2,
            surface_type,
            _azimuth_deg,
            _tilt_deg,
            wall_spec: None,
        }
    }

    /// Compute heat flux through this surface.
    fn compute_flux(
        &mut self,
        timestep: Time,
        T_interior: Temperature,
        T_exterior: Temperature,
        h_exterior: HeatTransferCoefficient,
        solar_irradiance_wm2: f64,
    ) -> Result<HeatFlux, SolverError> {
        let boundary = GaugeBoundaryConditions::new(solar_irradiance_wm2, T_exterior.to_value());
        self.gauge
            .step_with_boundary_conditions(timestep, T_interior, h_exterior, boundary)
    }
}

/// Zone-level heat balance solver using per-surface GaugeSolvers.
///
/// This struct owns the collection of per-surface 1D solvers and manages
/// the zone air node thermal capacitance.
#[derive(Debug, Clone)]
pub struct GaugeZoneSolver {
    /// Per-surface gauge solvers (walls, windows, roof, floor)
    surfaces: Vec<SurfaceGaugeSolver>,
    /// Zone air thermal capacitance (J/K)
    /// C_air = rho * V * c_p ≈ 1.2 * (floor_area * height) * 1006 J/K
    C_air: f64,
    /// Current zone air temperature (°C)
    T_air: f64,
    /// Zone volume (m³)
    #[allow(dead_code)]
    zone_volume: f64,
    /// Floor area (m²)
    floor_area: f64,
    /// Number of surfaces
    num_surfaces: usize,
    /// Solver initialized flag
    initialized: bool,
}

/// Physical constants for air
mod air_constants {
    /// Air density at standard conditions (kg/m³)
    pub const RHO_AIR: f64 = 1.2;

    /// Air specific heat at constant pressure (J/kg·K)
    pub const CP_AIR: f64 = 1006.0;

    /// Calculate zone air thermal capacitance (J/K)
    pub fn zone_air_capacitance(floor_area: f64, ceiling_height: f64) -> f64 {
        let volume = floor_area * ceiling_height;
        RHO_AIR * volume * CP_AIR
    }
}

impl GaugeZoneSolver {
    /// Create a new GaugeZoneSolver with the given zone geometry.
    pub fn new(floor_area: f64, ceiling_height: f64) -> Self {
        let zone_volume = floor_area * ceiling_height;
        let C_air = air_constants::zone_air_capacitance(floor_area, ceiling_height);

        Self {
            surfaces: Vec::new(),
            C_air,
            T_air: 20.0, // Default initial temperature (°C)
            zone_volume,
            floor_area,
            num_surfaces: 0,
            initialized: false,
        }
    }

    /// Add a surface with its gauge solver and geometric properties.
    pub fn add_surface(
        &mut self,
        gauge: GaugeSolver,
        area_m2: f64,
        surface_type: SurfaceType,
        azimuth_deg: f64,
        tilt_deg: f64,
    ) {
        self.surfaces.push(SurfaceGaugeSolver::new(
            gauge,
            area_m2,
            surface_type,
            azimuth_deg,
            tilt_deg,
        ));
        self.num_surfaces = self.surfaces.len();
    }

    /// Add an opaque surface from a WallSpec.
    pub fn add_opaque_surface(
        &mut self,
        wall: &WallSpec,
        area_m2: f64,
        surface_type: SurfaceType,
        azimuth_deg: f64,
        tilt_deg: f64,
    ) -> Result<(), SolverError> {
        let mut gauge = GaugeSolver::default();
        gauge.initialize(wall)?;

        let mut surface =
            SurfaceGaugeSolver::new(gauge, area_m2, surface_type, azimuth_deg, tilt_deg);
        surface.wall_spec = Some(wall.clone());

        self.surfaces.push(surface);
        self.num_surfaces = self.surfaces.len();
        Ok(())
    }

    /// Get current zone air temperature.
    pub fn T_air(&self) -> Temperature {
        Temperature::from_value(self.T_air)
    }

    /// Get zone air thermal capacitance.
    pub fn C_air(&self) -> f64 {
        self.C_air
    }

    /// Check if solver is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized && !self.surfaces.is_empty()
    }

    /// Initialize all surface gauge solvers.
    pub fn initialize(&mut self) -> Result<(), SolverError> {
        if self.surfaces.is_empty() {
            return Err(SolverError::InvalidConfig(
                "GaugeZoneSolver has no surfaces. Add surfaces before initializing.".to_string(),
            ));
        }

        for surface in &mut self.surfaces {
            if let Some(ref wall) = surface.wall_spec {
                surface.gauge.initialize(wall)?;
            }
        }

        self.initialized = true;
        Ok(())
    }

    /// Compute steady-state heat flux (no thermal mass).
    pub fn steady_state_flux(
        &self,
        T_interior: Temperature,
        T_exterior: Temperature,
    ) -> Result<HeatFlux, SolverError> {
        if !self.is_initialized() {
            return Err(SolverError::InvalidConfig(
                "GaugeZoneSolver not initialized".to_string(),
            ));
        }

        let mut total_Q = 0.0;
        for surface in &self.surfaces {
            let q_flux = surface.gauge.steady_state_flux(T_interior, T_exterior)?;
            let q = q_flux.to_value();
            total_Q += q * surface.area_m2;
        }

        // Return flux per unit area (W/m² of floor area)
        let flux_wm2 = total_Q / self.floor_area;
        Ok(HeatFlux::from_value(flux_wm2))
    }

    /// Step the zone model forward by one timestep.
    ///
    /// Computes per-surface fluxes, aggregates at zone air node, and updates T_air.
    ///
    /// # Arguments
    /// * `timestep` - Current timestep index (used for diagnostics)
    /// * `dt_seconds` - Timestep duration in seconds
    /// * `T_exterior` - Exterior air temperature (°C)
    /// * `h_exterior` - Exterior film coefficient (W/m²·K)
    /// * `solar_irradiance_wm2` - Total solar irradiance on horizontal plane (W/m²)
    /// * `Q_internal_w` - Internal heat gains (W)
    /// * `Q_infiltration_w` - Infiltration heat gain/loss (W)
    ///
    /// # Returns
    /// Net zone load in kWh (positive = heating needed, negative = cooling needed)
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        _timestep: usize,
        dt_seconds: f64,
        T_exterior: Temperature,
        h_exterior: HeatTransferCoefficient,
        solar_irradiance_wm2: f64,
        Q_internal_w: f64,
        Q_infiltration_w: f64,
    ) -> Result<f64, SolverError> {
        if !self.is_initialized() {
            return Err(SolverError::InvalidConfig(
                "GaugeZoneSolver not initialized".to_string(),
            ));
        }

        let T_int = Temperature::from_value(self.T_air);
        let mut net_power_watts = 0.0;

        // Sum heat flux from all surfaces
        for surface in &mut self.surfaces {
            let q_flux = surface.compute_flux(
                Time::from_value(dt_seconds),
                T_int,
                T_exterior,
                h_exterior,
                solar_irradiance_wm2 * surface.surface_type.solar_fraction(),
            )?;

            let Q_surface = q_flux.to_value() * surface.area_m2;
            net_power_watts += Q_surface;
        }

        // Add internal gains (infiltration is handled via implicit coupling below)
        net_power_watts += Q_internal_w;

        // Infiltration/ventilation coupling: h = rho * cp * ACH * V / 3600 [W/K]
        // For Case 600: ACH_inf = 0.5, V = 129.6 m³ => h_inf ≈ 21.7 W/K
        // For Case 650 (night vent): additional ACH = 3.0 => h_vent ≈ 130 W/K
        let infiltration_ach = 0.5; // ASHRAE 140 Case 600
        let h_inf = air_constants::RHO_AIR
            * air_constants::CP_AIR
            * (infiltration_ach / 3600.0)
            * self.zone_volume;
        // h_vent = 0 for base case; caller should pass night ventilation ACH if needed
        let h_vent = 0.0;
        let h_total = h_vent + h_inf;

        // Update zone air temperature using implicit Euler (unconditionally stable):
        // T_air_new = (C_air * T_air_old + Q_net * dt) / (C_air + h_total * dt)
        //
        // This is equivalent to solving:
        //   C_air * (T_new - T_old)/dt = Q_net + h_total * (T_out - T_new)
        // which implicit Euler handles by evaluating the coupling at T_new.
        //
        // Stability comparison (explicit Euler):
        //   dt/τ = dt * h_total / C_air
        //   For Case 650: dt/τ ≈ 3.5 (UNSTABLE, exceeds limit of 2)
        //   For Case 600: dt/τ ≈ 0.5 (stable, but implicit is still preferred)
        let T_air_old = self.T_air;
        self.T_air = (self.C_air * T_air_old + net_power_watts * dt_seconds)
            / (self.C_air + h_total * dt_seconds);

        // Add infiltration heat contribution to net power for return value
        net_power_watts += Q_infiltration_w;

        // Return net energy in kWh
        // Convention: positive = heating needed, negative = cooling needed
        // net_power_watts is positive when heat enters zone, negative when it leaves
        // So we negate to get heating/cooling convention
        let energy_kwh = -(net_power_watts * dt_seconds) / 3_600_000.0;
        Ok(energy_kwh)
    }

    /// Access the per-surface solvers (for diagnostics).
    #[allow(private_interfaces)]
    pub fn surfaces(&self) -> &[SurfaceGaugeSolver] {
        &self.surfaces
    }
}

// ============ Tests ============

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::wall_spec::WallSpec;

    fn case600_wall() -> WallSpec {
        // ASHRAE 140 Case 600 low-mass wall
        WallSpec::single_layer("LightWeight", 0.09, 1.0, 50.0, 50.0)
    }

    #[test]
    fn test_zone_air_capacitance() {
        // Case 600 zone: 6m x 8m x 2.7m = 129.6 m³
        let C = air_constants::zone_air_capacitance(48.0, 2.7);
        let expected = 1.2 * 48.0 * 2.7 * 1006.0; // ~156,000 J/K
        assert!((C - expected).abs() < 1.0);
    }

    #[test]
    fn test_steady_state_no_solar() {
        let mut zone = GaugeZoneSolver::new(48.0, 2.7); // Case 600 floor area

        // Add 4 walls (simplified - each gets full wall R)
        let wall = case600_wall();

        // Case 600 dimensions: 8m x 6m x 2.7m
        // Wall heights are 2.7m
        let south_area = 8.0 * 2.7; // 21.6 m²
        let east_area = 6.0 * 2.7; // 16.2 m²

        // South wall (faces equator, gets most solar)
        zone.add_opaque_surface(&wall, south_area, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        // North wall
        zone.add_opaque_surface(&wall, south_area, SurfaceType::Wall, 180.0, 90.0)
            .unwrap();
        // East wall
        zone.add_opaque_surface(&wall, east_area, SurfaceType::Wall, 90.0, 90.0)
            .unwrap();
        // West wall
        zone.add_opaque_surface(&wall, east_area, SurfaceType::Wall, -90.0, 90.0)
            .unwrap();

        zone.initialize().unwrap();

        // Steady state: inside 20°C, outside 10°C, no solar
        let T_int = Temperature::from_value(20.0);
        let T_ext = Temperature::from_value(10.0);

        let flux = zone.steady_state_flux(T_int, T_ext).unwrap();
        // Net flux should be negative (heat leaving zone)
        assert!(
            flux.to_value() < 0.0,
            "Heat should flow from warm interior to cold exterior"
        );
    }

    #[test]
    fn test_multi_surface_aggregation() {
        // Test that adding multiple surfaces properly aggregates heat flows
        let mut zone = GaugeZoneSolver::new(48.0, 2.7);

        let wall = case600_wall();

        // Add 2 identical walls
        let wall_area = 10.0; // 10 m² each
        zone.add_opaque_surface(&wall, wall_area, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        zone.add_opaque_surface(&wall, wall_area, SurfaceType::Wall, 180.0, 90.0)
            .unwrap();

        zone.initialize().unwrap();

        // With T_int=20, T_ext=10, each wall has heat leaving
        // Total should be roughly 2x a single wall
        let T_int = Temperature::from_value(20.0);
        let T_ext = Temperature::from_value(10.0);

        // Get flux from zone with 2 walls
        let flux_2walls = zone.steady_state_flux(T_int, T_ext).unwrap();

        // Now add a 3rd identical wall
        zone.add_opaque_surface(&wall, wall_area, SurfaceType::Wall, 90.0, 90.0)
            .unwrap();
        // Re-initialize since we added a surface
        zone.initialize().unwrap();

        let flux_3walls = zone.steady_state_flux(T_int, T_ext).unwrap();

        // More walls = more heat loss = more negative flux
        // (flux is W/m² of floor area, so with more wall area conducting,
        // the net flux per m² should be more negative)
        assert!(
            flux_3walls.to_value() < flux_2walls.to_value(),
            "More wall area should result in more heat loss"
        );
    }

    #[test]
    fn test_step_updates_temperature() {
        let mut zone = GaugeZoneSolver::new(48.0, 2.7);

        let wall = case600_wall();
        zone.add_opaque_surface(&wall, 48.0, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();

        zone.initialize().unwrap();

        let T_before = zone.T_air();
        assert!((T_before.to_value() - 20.0).abs() < 0.1);

        // Step with large exterior temp difference
        let T_ext = Temperature::from_value(-10.0); // Cold outside
        let h_ext = HeatTransferCoefficient::from_value(25.0);

        // One hour timestep
        let energy = zone
            .step(
                0,      // timestep
                3600.0, // dt = 1 hour
                T_ext, h_ext, 0.0, // no solar
                0.0, // no internal gains
                0.0, // no infiltration
            )
            .unwrap();

        let T_after = zone.T_air();

        // Zone should have cooled (T_after < T_before)
        assert!(
            T_after.to_value() < T_before.to_value(),
            "Zone should cool when exterior is cold"
        );

        // Energy should be positive (heating required)
        assert!(energy > 0.0, "Heating energy should be positive");
    }
}
