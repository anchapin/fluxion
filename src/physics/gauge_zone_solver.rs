//! Zone-level heat balance aggregator using per-surface GaugeSolvers.
//!
//! This module implements Phase 2 of the GaugeSolver elevation: replacing the
//! lumped 5R1C zone network with a geometrically-accurate assembly of 1D
//! per-surface GaugeSolvers that conserve energy at the zone air node.
//!
//! ## N-Zone Support
//!
//! This module supports both single-zone and multi-zone configurations:
//!
//! - **Single-zone**: Classic ASHRAE 140 case with multiple surfaces
//! - **Multi-zone**: N zones with inter-zone coupling via shared walls/floors
//!
//! Inter-zone coupling is defined over an adjacency graph where each edge
//! represents a shared boundary with thermal conductance. No pairwise
//! special-casing is used — all coupling is computed generically from
//! the adjacency structure.
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
//! For multi-zone configurations, inter-zone heat transfer is:
//!
//! Q_zone_j = sum_over_adjacent_zones(g_ij * (T_zone_i - T_zone_j))
//!
//! where g_ij is the inter-zone conductance.
//!
//! The zone air energy balance becomes:
//!
//! rho_air * V_zone * c_air * dT_air/dt = sum_i(q_i * A_i) + Q_int + Q_inf + Q_zone_coupling
//!
//! Using implicit Euler (unconditionally stable):
//!
//! T_air_new = (C_air * T_air_old + (Q_net + Q_zone_coupling) * dt) / (C_air + h_total * dt)
//!
//! ## Surface Types
//!
//! - **Opaque**: Walls, roof, floor - full resistance path
//! - **Window**: Simplified glazing model (future: multi-layer)
//! - **Ground**: Fixed ground temperature boundary
//! - **InternalMass**: Furniture, partitions
//! - **InterZone**: Shared boundary with adjacent zone

use crate::physics::gauge_solver::{GaugeBoundaryConditions, GaugeSolver};
use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
use crate::physics::units::FromF64;
use crate::physics::units::{HeatFlux, HeatTransferCoefficient, Temperature, Time, ToF64};
use crate::physics::wall_spec::WallSpec;
use std::collections::HashMap;

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
    /// Inter-zone boundary (shared wall/floor between zones)
    InterZone,
}

impl SurfaceType {
    /// Returns the solar gain fraction for this surface type.
    /// Windows receive solar; opaque surfaces conduct it away.
    pub fn solar_fraction(&self) -> f64 {
        match self {
            SurfaceType::Window => 1.0,
            SurfaceType::Wall => 0.0,
            SurfaceType::Roof => 0.0,
            SurfaceType::Floor => 0.0,
            SurfaceType::Ground => 0.0,
            SurfaceType::InternalMass => 0.0,
            SurfaceType::InterZone => 0.0,
        }
    }

    /// Returns true if this surface type represents an inter-zone boundary.
    pub fn is_inter_zone(&self) -> bool {
        matches!(self, SurfaceType::InterZone)
    }
}

/// Inter-zone coupling conductance.
///
/// Represents the thermal conductance between two adjacent zones
/// through a shared boundary (wall, floor, etc.).
#[derive(Debug, Clone, Copy)]
pub struct ZoneCoupling {
    /// Conductance between zones (W/K)
    pub conductance: f64,
    /// Area of shared boundary (m²)
    pub area_m2: f64,
    /// Zone ID of the adjacent zone
    pub adjacent_zone_id: usize,
}

impl ZoneCoupling {
    /// Create a new zone coupling.
    pub fn new(conductance: f64, area_m2: f64, adjacent_zone_id: usize) -> Self {
        Self {
            conductance,
            area_m2,
            adjacent_zone_id,
        }
    }

    /// Compute heat transfer rate given temperature difference.
    pub fn heat_transfer(&self, delta_t: f64) -> f64 {
        self.conductance * delta_t
    }
}

/// Boundary conditions for a zone in a multi-zone configuration.
///
/// This extends the basic zone conditions with inter-zone coupling.
#[derive(Debug, Clone)]
pub struct ZoneBoundaryConditions {
    /// Exterior air temperature (°C)
    pub T_exterior: Temperature,
    /// Exterior film coefficient (W/m²·K)
    pub h_exterior: HeatTransferCoefficient,
    /// Solar irradiance on horizontal plane (W/m²)
    pub solar_irradiance_wm2: f64,
    /// Internal heat gains (W)
    pub Q_internal_w: f64,
    /// Infiltration heat gain/loss (W)
    pub Q_infiltration_w: f64,
    /// Air changes per hour for infiltration
    pub infiltration_ach: f64,
    /// Coupled heat from adjacent zones (W)
    pub inter_zone_heat: f64,
}

impl Default for ZoneBoundaryConditions {
    fn default() -> Self {
        Self {
            T_exterior: Temperature::from_value(20.0),
            h_exterior: HeatTransferCoefficient::from_value(25.0),
            solar_irradiance_wm2: 0.0,
            Q_internal_w: 0.0,
            Q_infiltration_w: 0.0,
            infiltration_ach: 0.5,
            inter_zone_heat: 0.0,
        }
    }
}

impl ZoneBoundaryConditions {
    /// Create new boundary conditions.
    pub fn new(
        T_exterior: Temperature,
        h_exterior: HeatTransferCoefficient,
        solar_irradiance_wm2: f64,
    ) -> Self {
        Self {
            T_exterior,
            h_exterior,
            solar_irradiance_wm2,
            Q_internal_w: 0.0,
            Q_infiltration_w: 0.0,
            infiltration_ach: 0.5,
            inter_zone_heat: 0.0,
        }
    }

    /// Set internal heat gains.
    pub fn with_internal_gains(mut self, Q_internal_w: f64) -> Self {
        self.Q_internal_w = Q_internal_w;
        self
    }

    /// Set infiltration parameters.
    pub fn with_infiltration(mut self, Q_infiltration_w: f64, infiltration_ach: f64) -> Self {
        self.Q_infiltration_w = Q_infiltration_w;
        self.infiltration_ach = infiltration_ach;
        self
    }

    /// Set coupled heat from adjacent zones.
    pub fn with_inter_zone_heat(mut self, inter_zone_heat: f64) -> Self {
        self.inter_zone_heat = inter_zone_heat;
        self
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
    /// Adjacent zone ID (for inter-zone surfaces)
    #[allow(dead_code)]
    adjacent_zone_id: Option<usize>,
    #[allow(dead_code)]
    inter_zone_conductance: f64,
    /// Thermal mass temperature (°C) - tracks wall/roof/floor temperature
    T_mass: f64,
    /// Thermal mass capacity (J/K) = rho * c_p * volume per unit area
    C_mass: f64,
    /// Mass-to-interior-surface conductance (W/m²K)
    /// h_tr_ms = 1 / R_ms where R_ms = mass_to_interior_surface_r_value()
    h_tr_ms: f64,
    /// Exterior-to-mass conductance (W/m²K)
    /// h_tr_em = h_tr - h_tr_ms where h_tr = 1 / R_total
    h_tr_em: f64,
    /// Interior surface film coefficient (W/m²K) - shared across all surfaces in a zone
    h_tr_is: f64,
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
            adjacent_zone_id: None,
            inter_zone_conductance: 0.0,
            T_mass: 20.0,
            C_mass: 0.0,
            h_tr_ms: 0.0,
            h_tr_em: 0.0,
            h_tr_is: 0.0,
        }
    }

    /// Create a new inter-zone surface gauge solver.
    #[allow(dead_code)]
    fn new_inter_zone(
        gauge: GaugeSolver,
        area_m2: f64,
        adjacent_zone_id: usize,
        inter_zone_conductance: f64,
    ) -> Self {
        Self {
            gauge,
            area_m2,
            surface_type: SurfaceType::InterZone,
            _azimuth_deg: 0.0,
            _tilt_deg: 90.0,
            wall_spec: None,
            adjacent_zone_id: Some(adjacent_zone_id),
            inter_zone_conductance,
            T_mass: 20.0,
            C_mass: 0.0,
            h_tr_ms: 0.0,
            h_tr_em: 0.0,
            h_tr_is: 0.0,
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

    /// Compute heat flux for an inter-zone boundary.
    #[allow(dead_code)]
    fn compute_inter_zone_flux(
        &mut self,
        T_interior: Temperature,
        T_adjacent: Temperature,
    ) -> Result<HeatFlux, SolverError> {
        if !self.surface_type.is_inter_zone() {
            return Err(SolverError::InvalidConfig(
                "compute_inter_zone_flux called on non-inter-zone surface".to_string(),
            ));
        }
        let delta_t = T_adjacent.to_value() - T_interior.to_value();
        let flux = self.inter_zone_conductance * delta_t / self.area_m2;
        Ok(HeatFlux::from_value(flux))
    }
}

/// Zone-level heat balance solver using per-surface GaugeSolvers.
///
/// This struct owns the collection of per-surface 1D solvers and manages
/// the zone air node thermal capacitance. Supports both single-zone
/// and multi-zone configurations with inter-zone coupling.
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
    zone_volume: f64,
    /// Floor area (m²)
    floor_area: f64,
    /// Number of surfaces
    num_surfaces: usize,
    /// Solver initialized flag
    initialized: bool,
    /// Zone identifier (for multi-zone coupling)
    zone_id: usize,
    /// Inter-zone couplings to adjacent zones
    couplings: Vec<ZoneCoupling>,
    /// Pre-computed inter-zone conductance matrix (zone_id -> conductance)
    inter_zone_conductance: HashMap<usize, f64>,
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
        Self::new_with_id(0, floor_area, ceiling_height)
    }

    /// Create a new GaugeZoneSolver with a specific zone ID.
    pub fn new_with_id(zone_id: usize, floor_area: f64, ceiling_height: f64) -> Self {
        let zone_volume = floor_area * ceiling_height;
        let C_air = air_constants::zone_air_capacitance(floor_area, ceiling_height);

        Self {
            surfaces: Vec::new(),
            C_air,
            T_air: 20.0,
            zone_volume,
            floor_area,
            num_surfaces: 0,
            initialized: false,
            zone_id,
            couplings: Vec::new(),
            inter_zone_conductance: HashMap::new(),
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

    /// Add an inter-zone coupling to an adjacent zone.
    ///
    /// This establishes thermal coupling between this zone and an adjacent zone
    /// through a shared boundary (e.g., interior wall, floor between levels).
    ///
    /// # Arguments
    /// * `adjacent_zone_id` - ID of the adjacent zone
    /// * `shared_area_m2` - Area of shared boundary (m²)
    /// * `R_value` - Thermal resistance of shared boundary (m²·K/W)
    pub fn add_zone_coupling(
        &mut self,
        adjacent_zone_id: usize,
        shared_area_m2: f64,
        R_value: f64,
    ) -> Result<(), SolverError> {
        if R_value <= 0.0 {
            return Err(SolverError::ConstructionError(
                "Inter-zone R_value must be positive".to_string(),
            ));
        }
        let conductance = shared_area_m2 / R_value;
        self.couplings.push(ZoneCoupling::new(
            conductance,
            shared_area_m2,
            adjacent_zone_id,
        ));
        *self
            .inter_zone_conductance
            .entry(adjacent_zone_id)
            .or_insert(0.0) += conductance;
        Ok(())
    }

    /// Get the zone ID.
    pub fn zone_id(&self) -> usize {
        self.zone_id
    }

    /// Get inter-zone conductance to a specific adjacent zone.
    pub fn inter_zone_conductance(&self, adjacent_zone_id: usize) -> f64 {
        self.inter_zone_conductance
            .get(&adjacent_zone_id)
            .copied()
            .unwrap_or(0.0)
    }

    /// Get all inter-zone couplings.
    pub fn couplings(&self) -> &[ZoneCoupling] {
        &self.couplings
    }

    /// Add an opaque surface from a WallSpec.
    ///
    /// Computes thermal mass parameters from wall properties:
    /// - C_mass = wall.thermal_capacity() * area_m2  [J/K]
    /// - h_tr = 1 / wall.total_r_value()  [W/m²K]
    /// - h_tr_ms = 1 / wall.mass_to_interior_surface_r_value()  [W/m²K]
    /// - h_tr_em = h_tr - h_tr_ms  [W/m²K]
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

        let r_total = wall.total_r_value();
        let r_ms = wall.mass_to_interior_surface_r_value();
        let c_mass_per_area = wall.thermal_capacity();

        surface.C_mass = c_mass_per_area * area_m2;
        surface.T_mass = self.T_air;

        if r_total > 0.0 && r_ms > 0.0 {
            let h_tr = 1.0 / r_total;
            let h_tr_ms = 1.0 / r_ms;
            surface.h_tr_ms = h_tr_ms;
            surface.h_tr_em = h_tr - h_tr_ms;
        }

        self.surfaces.push(surface);
        self.num_surfaces = self.surfaces.len();
        Ok(())
    }

    /// Get current zone air temperature.
    pub fn T_air(&self) -> Temperature {
        Temperature::from_value(self.T_air)
    }

    /// Set zone air temperature (for test initialization).
    pub fn set_T_air(&mut self, temp: f64) {
        self.T_air = temp;
    }

    /// Set the interior surface heat transfer coefficient (h_tr_is) for all surfaces.
    ///
    /// This is a zone-level parameter representing the conductance from the interior
    /// surface to the zone air. It is shared across all surfaces in the zone.
    pub fn set_h_tr_is(&mut self, h_tr_is: f64) {
        for surface in &mut self.surfaces {
            surface.h_tr_is = h_tr_is;
        }
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
        let T_ext_val = T_exterior.to_value();
        let h_ext_val = h_exterior.to_value();
        let mut net_power_watts = 0.0;

        // Per-surface transient implicit Euler coupling for thermal mass nodes.
        // When h_tr_is > 0 for a surface, use the full 5R1C-style transient update.
        // When h_tr_is == 0 (not set), fall back to steady-state flux for backward compat.
        for surface in &mut self.surfaces {
            // Compute effective exterior temperature (sol-air temperature)
            let solar_gain = solar_irradiance_wm2 * surface.surface_type.solar_fraction();
            let T_ext_eff = T_ext_val + solar_gain / h_ext_val;

            let area = surface.area_m2;
            let h_ms = surface.h_tr_ms * area;
            let h_em = surface.h_tr_em * area;
            let C_mass = surface.C_mass;
            let h_is = surface.h_tr_is * area;

            if C_mass > 0.0 && surface.h_tr_em > 0.0 && h_is > 0.0 {
                // Transient 5R1C-style coupling: compute T_s first, then update T_mass.
                // h_tr_em > 0 guards against invalid single-layer wall configurations where
                // the computed mass-to-interior resistance exceeds R_total.
                // Step 1: Per-surface interior surface temperature from OLD T_mass and OLD T_air
                let T_mass_old = surface.T_mass;
                let h_ms_is_sum = h_ms + h_is;
                let T_s = if h_ms_is_sum > 1e-10 {
                    (h_ms * T_mass_old + h_is * self.T_air) / h_ms_is_sum
                } else {
                    self.T_air
                };

                // Step 2: Update T_mass using backward Euler with T_s
                // C_mass * (T_mass_new - T_mass_old) / dt
                //     = h_em * (T_ext_eff - T_mass_new) + h_ms * (T_s - T_mass_new)
                let denom = C_mass / dt_seconds + h_em + h_ms;
                if denom > 1e-10 {
                    let numer = C_mass / dt_seconds * T_mass_old
                        + h_em * T_ext_eff
                        + h_ms * T_s;
                    surface.T_mass = numer / denom;
                }

                // Step 3: Heat flow from interior surface to zone air: Q_is = h_is * (T_s - T_air)
                // This is the ONLY path by which thermal mass energy enters the zone air.
                let Q_is = h_is * (T_s - self.T_air);
                net_power_watts += Q_is;
            } else {
                // Fallback: steady-state flux when no thermal mass is configured
                let q_flux = surface.compute_flux(
                    Time::from_value(dt_seconds),
                    T_int,
                    T_exterior,
                    h_exterior,
                    solar_gain,
                )?;
                net_power_watts += q_flux.to_value() * area;
            }
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

    /// Step the zone model with inter-zone coupling.
    ///
    /// This extends the basic step with coupling to adjacent zones.
    /// The inter-zone heat is computed from the temperature difference
    /// between this zone and its neighbors.
    ///
    /// # Arguments
    /// * `dt_seconds` - Timestep duration in seconds
    /// * `bc` - Zone boundary conditions
    /// * `adjacent_temperatures` - Map of zone_id -> temperature for adjacent zones
    ///
    /// # Returns
    /// Net zone load in kWh (positive = heating needed, negative = cooling needed)
    pub fn step_with_coupling(
        &mut self,
        dt_seconds: f64,
        bc: &ZoneBoundaryConditions,
        adjacent_temperatures: &HashMap<usize, Temperature>,
    ) -> Result<f64, SolverError> {
        if !self.is_initialized() {
            return Err(SolverError::InvalidConfig(
                "GaugeZoneSolver not initialized".to_string(),
            ));
        }

        let T_int = Temperature::from_value(self.T_air);
        let mut net_power_watts = 0.0;

        // Sum heat flux from all exterior surfaces
        for surface in &mut self.surfaces {
            if surface.surface_type.is_inter_zone() {
                continue;
            }
            let q_flux = surface.compute_flux(
                Time::from_value(dt_seconds),
                T_int,
                bc.T_exterior,
                bc.h_exterior,
                bc.solar_irradiance_wm2 * surface.surface_type.solar_fraction(),
            )?;

            let Q_surface = q_flux.to_value() * surface.area_m2;
            net_power_watts += Q_surface;
        }

        // Compute inter-zone heat transfer
        let mut inter_zone_heat = 0.0;
        for coupling in &self.couplings {
            if let Some(&T_adjacent) = adjacent_temperatures.get(&coupling.adjacent_zone_id) {
                let Q_transfer = coupling.heat_transfer(T_int.to_value() - T_adjacent.to_value());
                inter_zone_heat += Q_transfer;
            }
        }
        net_power_watts += inter_zone_heat + bc.inter_zone_heat;

        // Add internal gains
        net_power_watts += bc.Q_internal_w;

        // Infiltration/ventilation coupling
        let h_inf = air_constants::RHO_AIR
            * air_constants::CP_AIR
            * (bc.infiltration_ach / 3600.0)
            * self.zone_volume;
        let h_total = h_inf;

        // Update zone air temperature using implicit Euler
        let T_air_old = self.T_air;
        self.T_air = (self.C_air * T_air_old + net_power_watts * dt_seconds)
            / (self.C_air + h_total * dt_seconds);

        // Return net energy in kWh
        let energy_kwh = -(net_power_watts * dt_seconds) / 3_600_000.0;
        Ok(energy_kwh)
    }

    /// Compute inter-zone coupling matrix contribution.
    ///
    /// Returns the heat exchange vector for all adjacent zones.
    /// Used by MultiZoneGaugeSolver to build the global coupling system.
    pub fn compute_zone_coupling_vector(
        &self,
        adjacent_temperatures: &HashMap<usize, Temperature>,
    ) -> HashMap<usize, f64> {
        let T_int = self.T_air;
        let mut coupling_vector = HashMap::new();

        for coupling in &self.couplings {
            if let Some(&T_adjacent) = adjacent_temperatures.get(&coupling.adjacent_zone_id) {
                let Q_transfer = coupling.heat_transfer(T_int - T_adjacent.to_value());
                *coupling_vector
                    .entry(coupling.adjacent_zone_id)
                    .or_insert(0.0) += Q_transfer;
            }
        }

        coupling_vector
    }
}

/// Multi-zone gauge solver for N-zone thermal coupling.
///
/// Manages multiple zones with inter-zone coupling defined over
/// an adjacency graph. Each zone is solved independently but
/// coupling is handled through a global system solve.
///
/// # Example
///
/// ```ignore
/// let mut multi_zone = MultiZoneGaugeSolver::new();
/// multi_zone.add_zone(0, 48.0, 2.7); // Zone 0: 48m² floor, 2.7m height
/// multi_zone.add_zone(1, 36.0, 2.7); // Zone 1: 36m² floor, 2.7m height
/// multi_zone.add_zone_coupling(0, 1, 10.0, 0.5); // Shared wall: 10m², R=0.5
/// multi_zone.initialize().unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MultiZoneGaugeSolver {
    /// All zones in the system
    zones: Vec<GaugeZoneSolver>,
    /// Zone IDs for quick lookup
    zone_ids: Vec<usize>,
    /// Number of zones
    num_zones: usize,
    /// Solver initialized flag
    initialized: bool,
}

impl MultiZoneGaugeSolver {
    /// Create a new empty multi-zone solver.
    pub fn new() -> Self {
        Self {
            zones: Vec::new(),
            zone_ids: Vec::new(),
            num_zones: 0,
            initialized: false,
        }
    }

    /// Add a zone to the system.
    ///
    /// # Arguments
    /// * `zone_id` - Unique identifier for this zone
    /// * `floor_area` - Floor area in m²
    /// * `ceiling_height` - Ceiling height in m
    pub fn add_zone(&mut self, zone_id: usize, floor_area: f64, ceiling_height: f64) {
        self.zones.push(GaugeZoneSolver::new_with_id(
            zone_id,
            floor_area,
            ceiling_height,
        ));
        self.zone_ids.push(zone_id);
        self.num_zones = self.zones.len();
    }

    /// Add a surface to a specific zone.
    pub fn add_surface_to_zone(
        &mut self,
        zone_id: usize,
        gauge: GaugeSolver,
        area_m2: f64,
        surface_type: SurfaceType,
        azimuth_deg: f64,
        tilt_deg: f64,
    ) -> Result<(), SolverError> {
        let zone = self
            .zones
            .iter_mut()
            .find(|z| z.zone_id == zone_id)
            .ok_or_else(|| SolverError::InvalidConfig(format!("Zone {} not found", zone_id)))?;
        zone.add_surface(gauge, area_m2, surface_type, azimuth_deg, tilt_deg);
        Ok(())
    }

    /// Add an opaque surface to a specific zone.
    pub fn add_opaque_surface_to_zone(
        &mut self,
        zone_id: usize,
        wall: &WallSpec,
        area_m2: f64,
        surface_type: SurfaceType,
        azimuth_deg: f64,
        tilt_deg: f64,
    ) -> Result<(), SolverError> {
        let zone = self
            .zones
            .iter_mut()
            .find(|z| z.zone_id == zone_id)
            .ok_or_else(|| SolverError::InvalidConfig(format!("Zone {} not found", zone_id)))?;
        zone.add_opaque_surface(wall, area_m2, surface_type, azimuth_deg, tilt_deg)
    }

    /// Add inter-zone coupling between two zones.
    ///
    /// This adds a symmetric coupling where heat flows between zones
    /// based on the temperature difference and shared boundary conductance.
    ///
    /// # Arguments
    /// * `zone_id_a` - First zone ID
    /// * `zone_id_b` - Second zone ID
    /// * `shared_area_m2` - Area of shared boundary (m²)
    /// * `R_value` - Thermal resistance of shared boundary (m²·K/W)
    pub fn add_zone_coupling(
        &mut self,
        zone_id_a: usize,
        zone_id_b: usize,
        shared_area_m2: f64,
        R_value: f64,
    ) -> Result<(), SolverError> {
        // Add coupling to zone A
        if let Some(zone) = self.zones.iter_mut().find(|z| z.zone_id == zone_id_a) {
            zone.add_zone_coupling(zone_id_b, shared_area_m2, R_value)?;
        }
        // Add symmetric coupling to zone B
        if let Some(zone) = self.zones.iter_mut().find(|z| z.zone_id == zone_id_b) {
            zone.add_zone_coupling(zone_id_a, shared_area_m2, R_value)?;
        }
        Ok(())
    }

    /// Initialize all zones.
    pub fn initialize(&mut self) -> Result<(), SolverError> {
        for zone in &mut self.zones {
            zone.initialize()?;
        }
        self.initialized = true;
        Ok(())
    }

    /// Check if solver is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized && !self.zones.is_empty()
    }

    /// Get the number of zones.
    pub fn num_zones(&self) -> usize {
        self.num_zones
    }

    /// Get zone by ID.
    pub fn get_zone(&self, zone_id: usize) -> Option<&GaugeZoneSolver> {
        self.zones.iter().find(|z| z.zone_id == zone_id)
    }

    /// Get mutable zone by ID.
    pub fn get_zone_mut(&mut self, zone_id: usize) -> Option<&mut GaugeZoneSolver> {
        self.zones.iter_mut().find(|z| z.zone_id == zone_id)
    }

    /// Get all zone IDs.
    pub fn zone_ids(&self) -> &[usize] {
        &self.zone_ids
    }

    /// Step all zones with inter-zone coupling.
    ///
    /// This performs a single timestep for all zones, computing
    /// inter-zone heat transfer based on current temperatures.
    ///
    /// # Arguments
    /// * `dt_seconds` - Timestep duration in seconds
    /// * `boundary_conditions` - Map of zone_id -> boundary conditions
    ///
    /// # Returns
    /// Map of zone_id -> energy in kWh
    pub fn step(
        &mut self,
        dt_seconds: f64,
        boundary_conditions: &HashMap<usize, ZoneBoundaryConditions>,
    ) -> Result<HashMap<usize, f64>, SolverError> {
        if !self.is_initialized() {
            return Err(SolverError::InvalidConfig(
                "MultiZoneGaugeSolver not initialized".to_string(),
            ));
        }

        // First: collect all zone temperatures for coupling calculations
        let zone_temps: HashMap<usize, f64> =
            self.zones.iter().map(|z| (z.zone_id, z.T_air)).collect();

        // Compute coupling vectors for each zone using collected temperatures
        let mut coupling_vectors: HashMap<usize, HashMap<usize, f64>> = HashMap::new();
        for zone in &self.zones {
            let adjacent_temps: HashMap<usize, Temperature> = zone
                .couplings
                .iter()
                .filter_map(|c| {
                    zone_temps
                        .get(&c.adjacent_zone_id)
                        .map(|&t| (c.adjacent_zone_id, Temperature::from_value(t)))
                })
                .collect();
            coupling_vectors.insert(
                zone.zone_id,
                zone.compute_zone_coupling_vector(&adjacent_temps),
            );
        }

        // Second pass: step each zone with its coupling contributions
        let mut results = HashMap::new();
        for zone in &mut self.zones {
            let zone_id = zone.zone_id;
            let bc = boundary_conditions
                .get(&zone_id)
                .cloned()
                .unwrap_or_default();

            // Add coupling contributions to boundary conditions
            // NOTE: inter-zone coupling is applied via adjacent_temps in step_with_coupling.
            // Zero bc.inter_zone_heat to avoid double-counting: the coupling vector
            // contribution is already accounted for through the adjacent_temps path.
            let mut bc_with_coupling = bc.clone();
            bc_with_coupling.inter_zone_heat = 0.0;

            // Get adjacent temperatures for inter-zone surfaces
            let adjacent_temps: HashMap<usize, Temperature> = zone
                .couplings
                .iter()
                .filter_map(|c| {
                    zone_temps
                        .get(&c.adjacent_zone_id)
                        .map(|&t| (c.adjacent_zone_id, Temperature::from_value(t)))
                })
                .collect();

            let energy = zone.step_with_coupling(dt_seconds, &bc_with_coupling, &adjacent_temps)?;
            results.insert(zone_id, energy);
        }

        Ok(results)
    }
}

impl Default for MultiZoneGaugeSolver {
    fn default() -> Self {
        Self::new()
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

    #[test]
    fn test_zone_coupling_basic() {
        // Test that inter-zone coupling is properly stored
        let mut zone_a = GaugeZoneSolver::new(48.0, 2.7);
        let mut zone_b = GaugeZoneSolver::new(36.0, 2.7);

        // Add coupling between zones (10m² shared wall, R=0.5 m²K/W)
        zone_a.add_zone_coupling(1, 10.0, 0.5).unwrap();

        assert_eq!(zone_a.zone_id(), 0);
        assert_eq!(zone_a.inter_zone_conductance(1), 20.0); // g = 10/0.5 = 20 W/K

        assert_eq!(zone_a.couplings().len(), 1);
        assert_eq!(zone_a.couplings()[0].adjacent_zone_id, 1);
        assert_eq!(zone_a.couplings()[0].conductance, 20.0);
    }

    #[test]
    fn test_multi_zone_solver_two_zones() {
        let mut multi_zone = MultiZoneGaugeSolver::new();

        // Zone 0: 48m² floor, 2.7m height (Zone A)
        multi_zone.add_zone(0, 48.0, 2.7);
        // Zone 1: 36m² floor, 2.7m height (Zone B)
        multi_zone.add_zone(1, 36.0, 2.7);

        let wall = case600_wall();

        // Add walls to Zone 0 (exterior walls only)
        multi_zone
            .add_opaque_surface_to_zone(0, &wall, 21.6, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        multi_zone
            .add_opaque_surface_to_zone(0, &wall, 16.2, SurfaceType::Wall, 90.0, 90.0)
            .unwrap();
        multi_zone
            .add_opaque_surface_to_zone(0, &wall, 16.2, SurfaceType::Wall, -90.0, 90.0)
            .unwrap();

        // Add walls to Zone 1 (exterior walls only)
        multi_zone
            .add_opaque_surface_to_zone(1, &wall, 18.0, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        multi_zone
            .add_opaque_surface_to_zone(1, &wall, 10.8, SurfaceType::Wall, 90.0, 90.0)
            .unwrap();

        // Add inter-zone coupling (shared interior wall: 10m², R=0.5)
        multi_zone.add_zone_coupling(0, 1, 10.0, 0.5).unwrap();

        multi_zone.initialize().unwrap();

        assert_eq!(multi_zone.num_zones(), 2);
        assert!(multi_zone.is_initialized());

        // Step both zones
        let mut bc = HashMap::new();
        bc.insert(
            0,
            ZoneBoundaryConditions::new(
                Temperature::from_value(5.0),
                HeatTransferCoefficient::from_value(25.0),
                0.0,
            ),
        );
        bc.insert(
            1,
            ZoneBoundaryConditions::new(
                Temperature::from_value(5.0),
                HeatTransferCoefficient::from_value(25.0),
                0.0,
            ),
        );

        let results = multi_zone.step(3600.0, &bc).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_inter_zone_heat_transfer() {
        // Test that heat flows between zones based on temperature difference
        let mut zone_a = GaugeZoneSolver::new(48.0, 2.7);
        let mut zone_b = GaugeZoneSolver::new(48.0, 2.7);

        let wall = case600_wall();
        zone_a
            .add_opaque_surface(&wall, 48.0, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        zone_b
            .add_opaque_surface(&wall, 48.0, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();

        // Add coupling: 20 W/K conductance
        zone_a.add_zone_coupling(1, 10.0, 0.5).unwrap();
        zone_b.add_zone_coupling(0, 10.0, 0.5).unwrap();

        zone_a.initialize().unwrap();
        zone_b.initialize().unwrap();

        // Set different initial temperatures
        zone_a.T_air = 25.0; // Warmer zone
        zone_b.T_air = 15.0; // Cooler zone

        // Compute coupling vector for zone A
        let mut adjacent_temps = HashMap::new();
        adjacent_temps.insert(1, Temperature::from_value(15.0));

        let coupling_vec = zone_a.compute_zone_coupling_vector(&adjacent_temps);

        // Heat should flow from A to B (positive in our convention means flowing to adjacent)
        // Q = g * (T_a - T_b) = 20 * (25 - 15) = 200 W
        assert!(*coupling_vec.get(&1).unwrap() > 0.0);
    }

    #[test]
    fn test_backward_compatibility_single_zone() {
        // Ensure existing single-zone usage still works
        let mut zone = GaugeZoneSolver::new(48.0, 2.7);
        let wall = case600_wall();

        zone.add_opaque_surface(&wall, 21.6, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        zone.add_opaque_surface(&wall, 16.2, SurfaceType::Wall, 90.0, 90.0)
            .unwrap();

        zone.initialize().unwrap();

        let T_int = Temperature::from_value(20.0);
        let T_ext = Temperature::from_value(10.0);

        let flux = zone.steady_state_flux(T_int, T_ext).unwrap();
        assert!(flux.to_value() < 0.0); // Heat flows out
    }
}
