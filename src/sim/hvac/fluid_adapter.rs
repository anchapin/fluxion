//! Fluid network adapter for HVAC/DAE integration.
//!
//! This module provides [`FluidNetworkAdapter`] which wraps the `fluxion-fluid`
//! component network (chillers, boilers, VAV boxes, pumps, coils) and converts
//! HVAC demand signals into fluid network inputs, producing thermal boundary
//! conditions for zone thermal models.
//!
//! # Architecture
//!
//! The adapter follows ADR-005's design:
//!
//! ```text
//! Zone Thermal Model ←→ FluidNetworkAdapter ←→ fluxion-fluid Network
//!                       (boundary conditions)      (DAE system)
//! ```
//!
//! # Thermal Boundary Conditions
//!
//! The adapter produces [`ThermalBoundaryConditions`] which include:
//! - Supply air temperature to zone (°C)
//! - Supply air humidity ratio (kg/kg)
//! - Supply air mass flow rate (kg/s)
//! - Heating coil load (W)
//! - Cooling coil load (W)
//!
//! # Example: Simple VAV System
//!
//! ```ignore
//! use fluxion::sim::hvac::fluid_adapter::{FluidNetworkAdapter, VavSystemConfig};
//!
//! let config = VavSystemConfig {
//!     chiller_capacity: 100_000.0,  // 100 kW
//!     chiller_cop: 5.0,
//!     boiler_capacity: 80_000.0,    // 80 kW
//!     boiler_efficiency: 0.9,
//!     num_zones: 4,
//!     rated_airflow_per_zone: 0.1,  // 0.1 m³/s per zone
//! };
//!
//! let mut adapter = FluidNetworkAdapter::vav_system(config);
//! adapter.set_cooling_load(zone_id, 5000.0);  // 5 kW cooling
//! adapter.set_heating_load(zone_id, 3000.0);  // 3 kW heating
//! adapter.solve();
//! let bc = adapter.thermal_boundary_conditions(zone_id);
//! ```

#[cfg(feature = "fluid")]
use fluxion_fluid::autodiff::{
    Boiler, Chiller, CoolingCoil, DifferentiableComponent, Pump, VavBox,
};
#[cfg(feature = "fluid")]
#[cfg(feature = "fluid")]
use fluxion_fluid::energy::{ConservationNode, EnthalpyFlow};

use serde::{Deserialize, Serialize};

#[cfg(feature = "fluid")]
const C_P_AIR: f64 = 1006.0;

#[cfg(feature = "fluid")]
const RHO_AIR: f64 = 1.2;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FluidSystemMode {
    Off,
    Heating,
    Cooling,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ThermalBoundaryConditions {
    pub zone_id: usize,
    pub supply_air_temp_c: f64,
    pub supply_humidity_ratio: f64,
    pub supply_mass_flow_kg_s: f64,
    pub heating_load_w: f64,
    pub cooling_load_w: f64,
    pub return_air_temp_c: f64,
    pub mode: FluidSystemMode,
}

impl Default for ThermalBoundaryConditions {
    fn default() -> Self {
        Self {
            zone_id: 0,
            supply_air_temp_c: 22.0,
            supply_humidity_ratio: 0.010,
            supply_mass_flow_kg_s: 0.1,
            heating_load_w: 0.0,
            cooling_load_w: 0.0,
            return_air_temp_c: 24.0,
            mode: FluidSystemMode::Off,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneFluidState {
    pub zone_id: usize,
    pub cooling_load_w: f64,
    pub heating_load_w: f64,
    pub damper_position: f64,
    pub reheat_valve_position: f64,
    pub current_mode: FluidSystemMode,
}

impl Default for ZoneFluidState {
    fn default() -> Self {
        Self {
            zone_id: 0,
            cooling_load_w: 0.0,
            heating_load_w: 0.0,
            damper_position: 0.3,
            reheat_valve_position: 0.0,
            current_mode: FluidSystemMode::Off,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VavSystemConfig {
    pub chiller_capacity: f64,
    pub chiller_cop: f64,
    pub boiler_capacity: f64,
    pub boiler_efficiency: f64,
    pub num_zones: usize,
    pub rated_airflow_per_zone: f64,
    pub static_pressure_setpoint: f64,
    pub chilled_water_supply_temp: f64,
    pub hot_water_supply_temp: f64,
}

impl Default for VavSystemConfig {
    fn default() -> Self {
        Self {
            chiller_capacity: 100_000.0,
            chiller_cop: 5.0,
            boiler_capacity: 80_000.0,
            boiler_efficiency: 0.9,
            num_zones: 4,
            rated_airflow_per_zone: 0.1,
            static_pressure_setpoint: 250.0,
            chilled_water_supply_temp: 7.0,
            hot_water_supply_temp: 60.0,
        }
    }
}

#[cfg(feature = "fluid")]
#[allow(dead_code)]
struct FluidConservationNode {
    id: usize,
    inlet_enthalpy: EnthalpyFlow,
    outlet_enthalpy: EnthalpyFlow,
    energy_transfer_w: f64,
}

#[cfg(feature = "fluid")]
impl ConservationNode for FluidConservationNode {
    fn id(&self) -> usize {
        self.id
    }

    fn mass_balance_residual(&self) -> f64 {
        self.inlet_enthalpy.mass_flow_rate - self.outlet_enthalpy.mass_flow_rate
    }

    fn energy_balance_residual(&self) -> f64 {
        let enthalpy_in = self.inlet_enthalpy.enthalpy_rate();
        let enthalpy_out = self.outlet_enthalpy.enthalpy_rate();
        enthalpy_in + self.energy_transfer_w - enthalpy_out
    }
}

#[cfg(feature = "fluid")]
pub struct FluidNetworkAdapter {
    config: VavSystemConfig,
    chiller: Chiller,
    boiler: Boiler,
    cooling_coil: CoolingCoil,
    #[allow(dead_code)]
    pumps: Vec<Pump>,
    vav_boxes: Vec<VavBox>,
    zone_states: Vec<ZoneFluidState>,
    boundary_conditions: Vec<ThermalBoundaryConditions>,
    mode: FluidSystemMode,
    chiller_state: Vec<f64>,
    boiler_state: Vec<f64>,
    cooling_coil_state: Vec<f64>,
    supply_air_temp_c: f64,
    chilled_water_return_temp_c: f64,
    hot_water_return_temp_c: f64,
}

#[cfg(feature = "fluid")]
impl FluidNetworkAdapter {
    pub fn vav_system(config: VavSystemConfig) -> Self {
        let chiller = Chiller::new(config.chiller_capacity, config.chiller_cop);
        let boiler = Boiler::new(config.boiler_capacity, config.boiler_efficiency);
        let cooling_coil = CoolingCoil::new(config.chiller_capacity * 0.8, 2.0);

        let pumps = vec![
            Pump::new(2.0, 100_000.0, 5000.0),
            Pump::new(1.0, 50_000.0, 2500.0),
        ];

        let vav_boxes: Vec<VavBox> = (0..config.num_zones).map(|_| VavBox::new()).collect();

        let zone_states: Vec<ZoneFluidState> = (0..config.num_zones)
            .map(|i| ZoneFluidState {
                zone_id: i,
                ..Default::default()
            })
            .collect();

        let boundary_conditions: Vec<ThermalBoundaryConditions> = (0..config.num_zones)
            .map(|i| ThermalBoundaryConditions {
                zone_id: i,
                ..Default::default()
            })
            .collect();

        Self {
            config,
            chiller,
            boiler,
            cooling_coil,
            pumps,
            vav_boxes,
            zone_states,
            boundary_conditions,
            mode: FluidSystemMode::Off,
            chiller_state: vec![],
            boiler_state: vec![],
            cooling_coil_state: vec![],
            supply_air_temp_c: 22.0,
            chilled_water_return_temp_c: 12.0,
            hot_water_return_temp_c: 50.0,
        }
    }

    pub fn set_cooling_load(&mut self, zone_id: usize, load_w: f64) {
        if zone_id < self.zone_states.len() {
            self.zone_states[zone_id].cooling_load_w = load_w;
        }
    }

    pub fn set_heating_load(&mut self, zone_id: usize, load_w: f64) {
        if zone_id < self.zone_states.len() {
            self.zone_states[zone_id].heating_load_w = load_w;
        }
    }

    pub fn set_zone_damper(&mut self, zone_id: usize, position: f64) {
        if zone_id < self.zone_states.len() {
            self.zone_states[zone_id].damper_position = position.clamp(0.0, 1.0);
        }
    }

    pub fn thermal_boundary_conditions(&self, zone_id: usize) -> Option<ThermalBoundaryConditions> {
        self.boundary_conditions.get(zone_id).copied()
    }

    pub fn all_boundary_conditions(&self) -> &[ThermalBoundaryConditions] {
        &self.boundary_conditions
    }

    pub fn determine_mode(&mut self) {
        let total_cooling: f64 = self.zone_states.iter().map(|z| z.cooling_load_w).sum();
        let total_heating: f64 = self.zone_states.iter().map(|z| z.heating_load_w).sum();

        self.mode = if total_cooling > total_heating && total_cooling > 100.0 {
            FluidSystemMode::Cooling
        } else if total_heating > total_cooling && total_heating > 100.0 {
            FluidSystemMode::Heating
        } else {
            FluidSystemMode::Off
        };

        for state in &mut self.zone_states {
            state.current_mode = self.mode;
        }
    }

    pub fn solve(&mut self) -> Result<(), FluidSolveError> {
        self.determine_mode();

        match self.mode {
            FluidSystemMode::Off => {
                for bc in &mut self.boundary_conditions {
                    bc.mode = FluidSystemMode::Off;
                    bc.heating_load_w = 0.0;
                    bc.cooling_load_w = 0.0;
                    bc.supply_air_temp_c = 22.0;
                    bc.supply_mass_flow_kg_s = self.config.rated_airflow_per_zone * RHO_AIR * 0.3;
                }
                return Ok(());
            }
            FluidSystemMode::Cooling => {
                self.solve_cooling()?;
            }
            FluidSystemMode::Heating => {
                self.solve_heating()?;
            }
        }

        self.compute_thermal_boundary_conditions();
        Ok(())
    }

    #[cfg(feature = "fluid")]
    fn solve_cooling(&mut self) -> Result<(), FluidSolveError> {
        let t_evap = self.config.chilled_water_supply_temp;
        let t_cond = 35.0;

        let chiller_input = vec![t_evap, t_cond, 0.5];
        let _chiller_output = self.chiller.evaluate(&chiller_input, &self.chiller_state);

        let t_water_in = self.chilled_water_return_temp_c;
        let m_dot_air_cooling = 1.5;

        let coil_input = vec![24.0, m_dot_air_cooling, 0.0, t_water_in];
        let _coil_output = self
            .cooling_coil
            .evaluate(&coil_input, &self.cooling_coil_state);

        for (i, state) in self.zone_states.iter_mut().enumerate() {
            if state.cooling_load_w > 0.0 {
                let zone_cooling = state.cooling_load_w;

                let t_inlet = 24.0;
                let pressure = self.config.static_pressure_setpoint;
                let damper = state.damper_position;

                let vav_input = vec![damper, pressure, t_inlet];
                let vav_state = vec![zone_cooling];
                let vav_output = self.vav_boxes[i].evaluate(&vav_input, &vav_state);

                let m_dot_supply = vav_output[0];
                let t_supply = vav_output[1];

                state.damper_position = damper;
                state.reheat_valve_position = (zone_cooling / 5000.0).clamp(0.0, 1.0);

                self.boundary_conditions[i].supply_air_temp_c = t_supply;
                self.boundary_conditions[i].supply_mass_flow_kg_s = m_dot_supply.max(0.01);
                self.boundary_conditions[i].cooling_load_w = zone_cooling;
                self.boundary_conditions[i].mode = FluidSystemMode::Cooling;
            } else {
                self.boundary_conditions[i].supply_air_temp_c = 22.0;
                self.boundary_conditions[i].supply_mass_flow_kg_s =
                    self.config.rated_airflow_per_zone * RHO_AIR * 0.3;
                self.boundary_conditions[i].cooling_load_w = 0.0;
                self.boundary_conditions[i].mode = FluidSystemMode::Off;
            }
        }

        self.supply_air_temp_c = self.boundary_conditions[0].supply_air_temp_c;
        Ok(())
    }

    #[cfg(not(feature = "fluid"))]
    fn solve_cooling(&mut self) -> Result<(), FluidSolveError> {
        Ok(())
    }

    #[cfg(feature = "fluid")]
    fn solve_heating(&mut self) -> Result<(), FluidSolveError> {
        let t_return = self.hot_water_return_temp_c;
        let t_enter = 55.0;
        let m_dot_hot = 0.5;

        let boiler_input = vec![t_return, m_dot_hot, t_enter];
        let _boiler_output = self.boiler.evaluate(&boiler_input, &self.boiler_state);

        for (i, state) in self.zone_states.iter_mut().enumerate() {
            if state.heating_load_w > 0.0 {
                let zone_heating = state.heating_load_w;

                let reheat = zone_heating;
                let t_supply =
                    40.0 + reheat / (self.config.rated_airflow_per_zone * RHO_AIR * C_P_AIR) * 10.0;

                state.reheat_valve_position = (zone_heating / 5000.0).clamp(0.0, 1.0);

                self.boundary_conditions[i].supply_air_temp_c = t_supply.clamp(22.0, 45.0);
                self.boundary_conditions[i].supply_mass_flow_kg_s =
                    self.config.rated_airflow_per_zone * RHO_AIR;
                self.boundary_conditions[i].heating_load_w = zone_heating;
                self.boundary_conditions[i].mode = FluidSystemMode::Heating;
            } else {
                self.boundary_conditions[i].supply_air_temp_c = 22.0;
                self.boundary_conditions[i].supply_mass_flow_kg_s =
                    self.config.rated_airflow_per_zone * RHO_AIR * 0.3;
                self.boundary_conditions[i].heating_load_w = 0.0;
                self.boundary_conditions[i].mode = FluidSystemMode::Off;
            }
        }

        self.supply_air_temp_c = self.boundary_conditions[0].supply_air_temp_c;
        Ok(())
    }

    #[cfg(not(feature = "fluid"))]
    fn solve_heating(&mut self) -> Result<(), FluidSolveError> {
        Ok(())
    }

    fn compute_thermal_boundary_conditions(&mut self) {
        let avg_return_temp: f64 = if self.mode == FluidSystemMode::Cooling {
            24.0
        } else {
            20.0
        };

        for bc in &mut self.boundary_conditions {
            bc.return_air_temp_c = avg_return_temp;
            bc.supply_humidity_ratio = 0.010;
        }
    }

    pub fn mode(&self) -> FluidSystemMode {
        self.mode
    }

    pub fn zone_state(&self, zone_id: usize) -> Option<&ZoneFluidState> {
        self.zone_states.get(zone_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluidSystemResult {
    pub success: bool,
    pub mode: FluidSystemMode,
    pub zone_boundary_conditions: Vec<ThermalBoundaryConditions>,
    pub total_cooling_load_w: f64,
    pub total_heating_load_w: f64,
    pub iteration_count: usize,
}

pub trait FluidNetworkSolver {
    fn solve_network(&mut self) -> Result<(), FluidSolveError>;
    fn get_thermal_outputs(&self) -> Vec<ThermalBoundaryConditions>;
}

#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
pub enum FluidSolveError {
    #[error("Chiller solve failed: {0}")]
    ChillerError(String),
    #[error("Boiler solve failed: {0}")]
    BoilerError(String),
    #[error("VAV box solve failed: {0}")]
    VavBoxError(String),
    #[error("Cooling coil solve failed: {0}")]
    CoolingCoilError(String),
    #[error("Network did not converge after {0} iterations")]
    ConvergenceFailed(usize),
    #[error("Invalid zone ID: {0}")]
    InvalidZoneId(usize),
    #[error("Fluid system is off")]
    SystemOff,
}

#[cfg(feature = "fluid")]
impl FluidNetworkSolver for FluidNetworkAdapter {
    fn solve_network(&mut self) -> Result<(), FluidSolveError> {
        self.solve()
    }

    fn get_thermal_outputs(&self) -> Vec<ThermalBoundaryConditions> {
        self.boundary_conditions.clone()
    }
}

#[cfg(feature = "fluid")]
impl Default for FluidNetworkAdapter {
    fn default() -> Self {
        Self::vav_system(VavSystemConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "fluid")]
    #[test]
    fn test_vav_system_creation() {
        let config = VavSystemConfig {
            num_zones: 4,
            ..Default::default()
        };
        let adapter = FluidNetworkAdapter::vav_system(config);
        assert_eq!(adapter.zone_states.len(), 4);
        assert_eq!(adapter.boundary_conditions.len(), 4);
    }

    #[cfg(feature = "fluid")]
    #[test]
    fn test_set_cooling_load() {
        let config = VavSystemConfig {
            num_zones: 2,
            ..Default::default()
        };
        let mut adapter = FluidNetworkAdapter::vav_system(config);
        adapter.set_cooling_load(0, 5000.0);
        adapter.set_cooling_load(1, 3000.0);

        adapter.determine_mode();
        assert_eq!(adapter.mode, FluidSystemMode::Cooling);
        assert_eq!(adapter.zone_states[0].cooling_load_w, 5000.0);
        assert_eq!(adapter.zone_states[1].cooling_load_w, 3000.0);
    }

    #[cfg(feature = "fluid")]
    #[test]
    fn test_set_heating_load() {
        let config = VavSystemConfig {
            num_zones: 2,
            ..Default::default()
        };
        let mut adapter = FluidNetworkAdapter::vav_system(config);
        adapter.set_heating_load(0, 3000.0);
        adapter.set_heating_load(1, 2000.0);

        adapter.determine_mode();
        assert_eq!(adapter.mode, FluidSystemMode::Heating);
    }

    #[cfg(feature = "fluid")]
    #[test]
    fn test_solve_cooling() {
        let config = VavSystemConfig {
            num_zones: 2,
            ..Default::default()
        };
        let mut adapter = FluidNetworkAdapter::vav_system(config);
        adapter.set_cooling_load(0, 5000.0);
        adapter.set_cooling_load(1, 3000.0);

        adapter.solve().expect("cooling solve should succeed");

        let bc0 = adapter.thermal_boundary_conditions(0).unwrap();
        assert_eq!(bc0.mode, FluidSystemMode::Cooling);
        assert!(bc0.cooling_load_w > 0.0);
    }

    #[cfg(feature = "fluid")]
    #[test]
    fn test_solve_heating() {
        let config = VavSystemConfig {
            num_zones: 2,
            ..Default::default()
        };
        let mut adapter = FluidNetworkAdapter::vav_system(config);
        adapter.set_heating_load(0, 5000.0);
        adapter.set_heating_load(1, 3000.0);

        adapter.solve().expect("heating solve should succeed");

        let bc0 = adapter.thermal_boundary_conditions(0).unwrap();
        assert_eq!(bc0.mode, FluidSystemMode::Heating);
        assert!(bc0.heating_load_w > 0.0);
    }

    #[cfg(feature = "fluid")]
    #[test]
    fn test_off_mode_no_loads() {
        let config = VavSystemConfig {
            num_zones: 2,
            ..Default::default()
        };
        let mut adapter = FluidNetworkAdapter::vav_system(config);

        adapter.solve().expect("solve should succeed");
        assert_eq!(adapter.mode, FluidSystemMode::Off);

        let bc = adapter.thermal_boundary_conditions(0).unwrap();
        assert_eq!(bc.mode, FluidSystemMode::Off);
        assert_eq!(bc.heating_load_w, 0.0);
        assert_eq!(bc.cooling_load_w, 0.0);
    }

    #[cfg(feature = "fluid")]
    #[test]
    fn test_damper_position_clamping() {
        let config = VavSystemConfig {
            num_zones: 1,
            ..Default::default()
        };
        let mut adapter = FluidNetworkAdapter::vav_system(config);
        adapter.set_zone_damper(0, 1.5);
        assert_eq!(adapter.zone_states[0].damper_position, 1.0);

        adapter.set_zone_damper(0, -0.5);
        assert_eq!(adapter.zone_states[0].damper_position, 0.0);
    }

    #[cfg(not(feature = "fluid"))]
    #[test]
    fn test_adapter_disabled_without_feature() {
        let config = VavSystemConfig {
            num_zones: 2,
            ..Default::default()
        };
        let adapter = FluidNetworkAdapter::vav_system(config);
        assert_eq!(adapter.zone_states.len(), 2);
    }
}
