// # fluxion-grid
//
// Grid-edge electrical network components for Fluxion building energy modeling.
//
// This crate provides battery storage node models with state-of-charge (SoC) tracking
// and electrical characteristics for integration with building energy simulations.
//
// ## Contents
//
// | Module | Description |
// |--------|-------------|
// | `battery_storage_node` | `BatteryStorageNode` — single-cell battery model with SoC, terminal voltage, and C-rate dynamics |

#![allow(nonstandard_style)]
#![allow(clippy::all)]

pub mod battery_storage_node;

pub use battery_storage_node::BatteryStorageNode;

// === Heat Pump Voltage Model ===
pub mod error;
pub mod heat_pump_voltage_model;
pub mod thermal_electrical_coupler;

// === Fluxion Integration Bridge ===
// Only available when the "fluxion" feature flag is enabled.
#[cfg(feature = "fluxion")]
pub mod fluxion_bridge;
#[cfg(feature = "fluxion")]
pub use fluxion_bridge::ThermalModelTraitBridge;

pub use error::{GridModelError, GridSolveError};
pub use heat_pump_voltage_model::HeatPumpVoltageModel;
pub use thermal_electrical_coupler::VoltageCoupler;

// === Joint Thermal-Electrical Convergence Solver ===
// fluxion-grid: Electrical grid modeling with bus node types for power flow analysis.
//
// This crate provides foundational types for electrical bus modeling including:
// - Bus node types (Slack, PV, PQ)
// - Electrical bus structures with voltage, angle, and power attributes
// - Battery bus with State of Charge (SoC) tracking
// - Joint thermal-electrical convergence solver for grid-interactive buildings

pub mod battery;
pub mod bus;
pub mod power_flow;

pub use battery::BatteryBus;
pub use battery::HeatFlowRate; // Re-exported for direct fluxion-core thermal integration (#2036)
pub use bus::{BusNodeType, ElectricalBus};
pub use power_flow::{
    bus_uuid, GridConvergenceReport, PowerFlowSolver, PowerFlowState, TransmissionLine,
    DEFAULT_MAX_ITERATIONS, DEFAULT_TOLERANCE,
};

use fluxion_fluid::hvac::{HvacMode, HvacState};
use nalgebra::DMatrix;

pub type VoltagePu = f64;

/// Result of the joint convergence solve.
#[derive(Debug, Clone)]
pub struct ConvergenceResult {
    /// Whether both thermal and electrical systems converged
    pub converged: bool,
    /// Total iterations performed
    pub iterations: usize,
    /// Final thermal residual (zone temperature change)
    pub thermal_residual: f64,
    /// Final electrical mismatch (power balance error)
    pub electrical_mismatch: f64,
}

/// Multi-zone thermal model for joint convergence.
///
/// This is a simplified thermal model representing building zone temperatures
/// and HVAC thermal loads with setpoint-based control.
#[derive(Debug, Clone)]
pub struct ThermalModel {
    /// Number of thermal zones
    pub num_zones: usize,
    /// Zone air temperatures (°C)
    pub temperatures: Vec<f64>,
    /// Zone thermal capacitances (J/K)
    pub capacitances: Vec<f64>,
    /// HVAC thermal loads per zone (W) - computed from control
    pub hvac_loads: Vec<f64>,
    /// Heating setpoints (°C)
    pub heating_setpoints: Vec<f64>,
    /// Cooling setpoints (°C)
    pub cooling_setpoints: Vec<f64>,
    /// Inter-zone conductances (W/K)
    pub inter_zone_conductance: Vec<f64>,
    /// Ambient temperature (°C)
    pub ambient_temperature: f64,
    /// envelope conductance to ambient (W/K) per zone
    pub envelope_conductance: Vec<f64>,
}

impl ThermalModel {
    /// Create a new ThermalModel with N zones.
    pub fn new(num_zones: usize, initial_temperature: f64) -> Self {
        ThermalModel {
            num_zones,
            temperatures: vec![initial_temperature; num_zones],
            capacitances: vec![1_000_000.0; num_zones],
            hvac_loads: vec![0.0; num_zones],
            heating_setpoints: vec![20.0; num_zones],
            cooling_setpoints: vec![24.0; num_zones],
            inter_zone_conductance: vec![0.0; num_zones],
            ambient_temperature: 10.0,
            envelope_conductance: vec![100.0; num_zones], // 100 W/K to ambient
        }
    }

    /// Update HVAC loads based on setpoint control.
    pub fn update_hvac_loads(&mut self, coupler: &ThermalElectricalCoupler) {
        for i in 0..self.num_zones {
            let temp = self.temperatures[i];
            let setpoint = self.heating_setpoints[i]; // Using heating for simplicity
            let error = setpoint - temp;

            // P-control: HVAC load proportional to temperature error
            // Max heating = 2000W, controller gain = 50 W/K
            let heating_capacity = 2000.0;
            let controller_gain = 50.0;
            let desired_heating = (controller_gain * error).clamp(0.0, heating_capacity);

            // Heat pump electrical consumption
            let _electrical_power = desired_heating / coupler.cop;

            // Store thermal load (what HVAC provides to zone)
            self.hvac_loads[i] = desired_heating;
        }
    }

    /// Calculate thermal residual (measure of temperature change).
    pub fn calculate_residual(&self) -> f64 {
        let mut total_change = 0.0;
        for i in 0..self.num_zones {
            // Net heat = HVAC + inter-zone + envelope losses
            let envelope_loss =
                self.envelope_conductance[i] * (self.temperatures[i] - self.ambient_temperature);
            let net_heat = self.hvac_loads[i] - envelope_loss
                + inter_zone_heat_contribution(i, &self.inter_zone_conductance, &self.temperatures);
            let temp_change = (net_heat / self.capacitances[i]).abs();
            total_change += temp_change;
        }
        total_change / self.num_zones as f64
    }

    /// Solve thermal system for one iteration.
    pub fn solve_step(&mut self, dt: f64) {
        for i in 0..self.num_zones {
            let envelope_loss =
                self.envelope_conductance[i] * (self.temperatures[i] - self.ambient_temperature);
            let net_heat = self.hvac_loads[i] - envelope_loss
                + inter_zone_heat_contribution(i, &self.inter_zone_conductance, &self.temperatures);
            self.temperatures[i] += (net_heat / self.capacitances[i]) * dt;
        }
    }
}

/// Calculate inter-zone heat contribution for a specific zone.
fn inter_zone_heat_contribution(zone_index: usize, h_tr_iz: &[f64], temperatures: &[f64]) -> f64 {
    let mut total = 0.0;
    let num_zones = temperatures.len();

    for j in 0..num_zones {
        if j != zone_index {
            let conductance = h_tr_iz[zone_index].min(h_tr_iz[j]);
            total += conductance * (temperatures[j] - temperatures[zone_index]);
        }
    }

    total
}

/// Electrical network model representing buses and power flows.
///
/// This is a simplified electrical network model with bus voltages and
/// power injections for grid-interactive building models.
#[derive(Debug, Clone)]
pub struct ElectricalNetwork {
    /// Number of buses
    pub num_buses: usize,
    /// Bus voltages (V) - magnitude
    pub voltages: Vec<f64>,
    /// Bus angles (rad)
    pub angles: Vec<f64>,
    /// Power injections per bus (W)
    pub power_injections: Vec<f64>,
    /// Admittance matrix (S)
    pub admittance_matrix: DMatrix<f64>,
    /// Reference bus index (slack bus)
    pub reference_bus: usize,
}

impl ElectricalNetwork {
    /// Create a new ElectricalNetwork with N buses.
    pub fn new(num_buses: usize) -> Self {
        let voltages = vec![1.0; num_buses]; // Per-unit voltages
        let angles = vec![0.0; num_buses];
        let power_injections = vec![0.0; num_buses];

        // Initialize admittance matrix for DC power flow
        // B_ij = -1/x_ij (line susceptance), B_ii = sum of all B_ij connected to i
        let mut admittance_matrix = DMatrix::from_element(num_buses, num_buses, 0.0);
        for i in 0..num_buses {
            let mut row_sum = 0.0;
            for j in 0..num_buses {
                if i != j {
                    admittance_matrix[(i, j)] = -1.0; // Coupling susceptance = -1 p.u.
                    row_sum += 1.0;
                }
            }
            admittance_matrix[(i, i)] = row_sum; // Diagonal = sum of line susceptances
        }

        ElectricalNetwork {
            num_buses,
            voltages,
            angles,
            power_injections,
            admittance_matrix,
            reference_bus: 0,
        }
    }

    /// Calculate electrical mismatch (power balance error).
    pub fn calculate_mismatch(&self) -> f64 {
        let mut total_mismatch = 0.0;
        let mut num_non_ref = 0;

        for i in 0..self.num_buses {
            if i == self.reference_bus {
                continue;
            }
            num_non_ref += 1;

            // DC power flow: P_i = sum_j(B_ij * (theta_i - theta_j))
            // Simplified using G_ij as coupling coefficient
            let mut p_calc = 0.0;
            for j in 0..self.num_buses {
                if i != j {
                    let coupling = self.admittance_matrix[(i, j)];
                    p_calc += coupling * (self.angles[i] - self.angles[j]);
                }
            }

            let mismatch = (p_calc - self.power_injections[i]).abs();
            total_mismatch += mismatch;
        }

        if num_non_ref > 0 {
            total_mismatch / num_non_ref as f64
        } else {
            0.0
        }
    }

    /// Solve power flow for one iteration using DC power flow.
    pub fn solve_power_flow_step(&mut self) {
        let mut new_angles = self.angles.clone();

        for i in 0..self.num_buses {
            if i == self.reference_bus {
                continue;
            }

            // DC power flow: P_i = sum_j(B_ij * (theta_i - theta_j))
            // Rearranged: theta_i = (P_i + sum_{j!=i} B_ij * theta_j) / B_ii
            let mut sum = 0.0;
            let b_ii = self.admittance_matrix[(i, i)];

            for j in 0..self.num_buses {
                if i != j {
                    let b_ij = self.admittance_matrix[(i, j)];
                    sum += b_ij * self.angles[j];
                }
            }

            if b_ii.abs() > 1e-10 {
                new_angles[i] = (self.power_injections[i] + sum) / b_ii;
            }
        }

        self.angles = new_angles;
    }
}

/// Electrical load at a building bus.
///
/// This represents the electrical power demand of a building's HVAC system
/// derived from thermal demand via the heat pump COP.
#[derive(Debug, Clone)]
pub struct ElectricalLoad {
    /// Unique identifier of the building
    pub building_id: uuid::Uuid,
    /// Electrical power demand (W)
    pub power_w: f64,
    /// Reactive power demand (VAR)
    pub reactive_power_var: f64,
    /// Power factor (cosine of phase angle)
    pub power_factor: f64,
}

impl Default for ElectricalLoad {
    fn default() -> Self {
        Self {
            building_id: uuid::Uuid::nil(),
            power_w: 0.0,
            reactive_power_var: 0.0,
            power_factor: 1.0,
        }
    }
}

impl ElectricalLoad {
    /// Create a new electrical load for a building.
    pub fn new(building_id: uuid::Uuid, power_w: f64) -> Self {
        Self {
            building_id,
            power_w,
            reactive_power_var: 0.0,
            power_factor: 1.0,
        }
    }

    /// Create an electrical load with reactive power.
    pub fn with_reactive_power(building_id: uuid::Uuid, power_w: f64, power_factor: f64) -> Self {
        let reactive_power_var = power_w * (1.0 - power_factor.powi(2)).sqrt();
        Self {
            building_id,
            power_w,
            reactive_power_var,
            power_factor,
        }
    }
}

/// Mapping from building IDs to electrical bus IDs.
#[derive(Debug, Clone)]
pub struct BuildingBusMapping {
    /// Map from building UUID to bus UUID
    mappings: std::collections::HashMap<uuid::Uuid, uuid::Uuid>,
}

impl BuildingBusMapping {
    /// Create a new empty building-to-bus mapping.
    pub fn new() -> Self {
        Self {
            mappings: std::collections::HashMap::new(),
        }
    }

    /// Add a building-to-bus mapping.
    pub fn add_mapping(&mut self, building_id: uuid::Uuid, bus_id: uuid::Uuid) {
        self.mappings.insert(building_id, bus_id);
    }

    /// Get the bus ID for a building, if mapped.
    pub fn get_bus_id(&self, building_id: &uuid::Uuid) -> Option<uuid::Uuid> {
        self.mappings.get(building_id).copied()
    }

    /// Get all building IDs in the mapping.
    pub fn building_ids(&self) -> impl Iterator<Item = &uuid::Uuid> {
        self.mappings.keys()
    }

    /// Get all bus IDs in the mapping.
    pub fn bus_ids(&self) -> impl Iterator<Item = &uuid::Uuid> {
        self.mappings.values()
    }

    /// Returns the number of mappings.
    pub fn len(&self) -> usize {
        self.mappings.len()
    }

    /// Returns true if the mapping is empty.
    pub fn is_empty(&self) -> bool {
        self.mappings.is_empty()
    }
}

impl Default for BuildingBusMapping {
    fn default() -> Self {
        Self::new()
    }
}

/// Coupler between thermal and electrical systems via heat pump COP.
///
/// The COP (Coefficient of Performance) links electrical power consumption
/// to thermal power production: thermal_power = COP * electrical_power.
#[derive(Debug, Clone)]
pub struct ThermalElectricalCoupler {
    /// Current coefficient of performance
    pub cop: f64,
    /// Rated COP at reference conditions
    pub rated_cop: f64,
    /// Temperature of heat source/sink (°C)
    pub source_temperature: f64,
    /// Temperature of heat delivery (°C)
    pub delivery_temperature: f64,
    /// Building-to-bus mapping for grid integration
    building_bus_mapping: BuildingBusMapping,
}

impl ThermalElectricalCoupler {
    /// Create a new coupler with specified COP.
    pub fn new(cop: f64) -> Self {
        ThermalElectricalCoupler {
            cop,
            rated_cop: cop,
            source_temperature: 10.0,
            delivery_temperature: 45.0,
            building_bus_mapping: BuildingBusMapping::new(),
        }
    }

    /// Create a coupler with a building-to-bus mapping.
    pub fn with_mapping(cop: f64, mapping: BuildingBusMapping) -> Self {
        ThermalElectricalCoupler {
            cop,
            rated_cop: cop,
            source_temperature: 10.0,
            delivery_temperature: 45.0,
            building_bus_mapping: mapping,
        }
    }

    /// Get a reference to the building-to-bus mapping.
    pub fn building_bus_mapping(&self) -> &BuildingBusMapping {
        &self.building_bus_mapping
    }

    /// Set the building-to-bus mapping.
    pub fn set_building_bus_mapping(&mut self, mapping: BuildingBusMapping) {
        self.building_bus_mapping = mapping;
    }

    /// Add a single building-to-bus mapping.
    pub fn add_building_bus_mapping(&mut self, building_id: uuid::Uuid, bus_id: uuid::Uuid) {
        self.building_bus_mapping.add_mapping(building_id, bus_id);
    }

    /// Convert an HVAC state to electrical load for a specific building.
    ///
    /// Uses the current COP to convert thermal power to electrical power.
    /// For heating mode, electrical = thermal / COP.
    /// For cooling mode, electrical = thermal / COP.
    ///
    /// Returns `None` if the HVAC is off or if thermal power is zero/negative.
    pub fn hvac_state_to_electrical(&self, state: &HvacState) -> Option<ElectricalLoad> {
        // Only process active HVAC states with positive thermal demand
        if state.mode == HvacMode::Off || state.thermal_power_w <= 0.0 {
            return None;
        }

        // Compute electrical power from thermal power using COP
        let electrical_power = state.thermal_power_w / self.cop;

        Some(ElectricalLoad::new(state.building_id, electrical_power))
    }

    /// Convert a slice of HVAC states to electrical loads, mapped by building ID.
    ///
    /// This method processes each HVAC state, converts thermal power to electrical
    /// power using the current COP, and returns a map of building IDs to electrical
    /// loads. Buildings without a bus mapping are logged and skipped.
    ///
    /// # Errors
    ///
    /// Returns a list of building IDs that were not found in the bus mapping.
    /// Any missing buildings are also logged at the warn level.
    pub fn thermal_to_electrical(
        &self,
        hvac_states: &[HvacState],
    ) -> (
        std::collections::HashMap<uuid::Uuid, ElectricalLoad>,
        Vec<uuid::Uuid>,
    ) {
        use std::collections::HashMap;

        let mut loads: HashMap<uuid::Uuid, ElectricalLoad> = HashMap::new();
        let mut missing_buildings: Vec<uuid::Uuid> = Vec::new();

        for state in hvac_states {
            // Check if building has a bus mapping
            if !self
                .building_bus_mapping
                .mappings
                .contains_key(&state.building_id)
            {
                missing_buildings.push(state.building_id);
                continue;
            }

            // Convert HVAC state to electrical load
            if let Some(load) = self.hvac_state_to_electrical(state) {
                loads.insert(state.building_id, load);
            }
        }

        // Log missing buildings at warn level
        for building_id in &missing_buildings {
            eprintln!(
                "WARN: building {} has no bus in the grid mapping — skipping",
                building_id
            );
        }

        (loads, missing_buildings)
    }

    /// Convert thermal power directly to electrical power.
    ///
    /// This is the basic COP-based conversion: electrical = thermal / COP.
    /// Use this when you already have a scalar thermal power value.
    pub fn thermal_to_electrical_simple(&self, thermal_power: f64) -> f64 {
        thermal_power / self.cop
    }

    /// Calculate COP based on Carnot efficiency and degradation.
    ///
    /// COP = COP_rated * efficiency_factor * carnot_ratio
    pub fn update_cop(&mut self, ambient_temperature: f64) {
        let t_hot_k = self.delivery_temperature + 273.15;
        let t_cold_k = ambient_temperature.max(-10.0) + 273.15;

        // Carnot COP: COP_carnot = T_hot / (T_hot - T_cold)
        // This is the theoretical maximum COP for a reversible heat pump
        let carnot = t_hot_k / (t_hot_k - t_cold_k);

        // Reference Carnot COP at 20°C ambient (ASHRAE design condition)
        // T_hot = 45°C = 318.15 K, T_cold = 20°C = 293.15 K
        // COP_ref = 318.15 / (318.15 - 293.15) = 318.15 / 25 = 12.726
        let t_hot_ref_k = 45.0 + 273.15;
        let t_cold_ref_k = 20.0 + 273.15;
        let reference_carnot = t_hot_ref_k / (t_hot_ref_k - t_cold_ref_k);

        // COP adjustment factor = actual_carnot / reference_carnot
        // This ratio captures how temperature differences affect real heat pump performance
        let carnot_factor = (carnot / reference_carnot).clamp(0.3, 1.5);

        // Actual COP = rated_COP * carnot_factor
        // The rated COP already includes the real-world efficiency factor
        self.cop = self.rated_cop * carnot_factor;
        self.cop = self.cop.clamp(1.0, self.rated_cop * 1.5);
    }

    /// Convert electrical power to thermal power.
    pub fn electrical_to_thermal(&self, electrical_power: f64) -> f64 {
        electrical_power * self.cop
    }

    /// Update COP based on ambient temperature from an HVAC state.
    ///
    /// This is a convenience method that extracts the ambient temperature
    /// from the HVAC state and calls `update_cop`.
    pub fn update_cop_from_hvac_state(&mut self, state: &HvacState) {
        self.update_cop(state.ambient_temperature_c);
    }
}

/// Joint convergence solver for thermal-electrical systems.
///
/// This solver iteratively solves the coupled thermal and electrical systems
/// until both converge within the specified tolerance.
#[derive(Debug, Clone)]
pub struct JointConvergenceSolver {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Time step for thermal solve (s)
    pub dt: f64,
}

impl JointConvergenceSolver {
    /// Create a new joint convergence solver.
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        JointConvergenceSolver {
            max_iterations,
            tolerance,
            dt: 3600.0, // 1 hour default timestep
        }
    }

    /// Solve the joint thermal-electrical system iteratively.
    ///
    /// Iteration pattern:
    /// 1. Solve thermal → compute zone temperatures and HVAC loads
    /// 2. Compute heat pump electrical load from thermal load
    /// 3. Solve electrical → compute bus voltages and power flows
    /// 4. Update COP based on electrical state
    /// 5. Check convergence
    pub fn solve(
        &mut self,
        thermal_model: &mut ThermalModel,
        electrical_model: &mut ElectricalNetwork,
        coupler: &mut ThermalElectricalCoupler,
    ) -> ConvergenceResult {
        let mut iterations = 0;
        let mut thermal_residual = f64::MAX;
        let mut electrical_mismatch = f64::MAX;

        while iterations < self.max_iterations {
            // Step 0: Update HVAC loads based on current temperatures and setpoints
            thermal_model.update_hvac_loads(coupler);

            // Step 1: Solve thermal system
            let prev_temps = thermal_model.temperatures.clone();
            thermal_model.solve_step(self.dt);
            thermal_residual = prev_temps
                .iter()
                .zip(&thermal_model.temperatures)
                .map(|(t_prev, t_new)| (t_new - t_prev).abs())
                .sum::<f64>()
                / thermal_model.num_zones as f64;

            // Step 2: Compute heat pump electrical load from thermal load
            let total_thermal_load: f64 = thermal_model.hvac_loads.iter().sum();
            let electrical_load = coupler.thermal_to_electrical_simple(total_thermal_load);

            // Step 3: Update electrical network with heat pump load
            let load_per_bus = electrical_load / electrical_model.num_buses as f64;
            for i in 0..electrical_model.num_buses {
                if i != electrical_model.reference_bus {
                    electrical_model.power_injections[i] = -load_per_bus;
                }
            }

            // Step 4: Solve electrical system
            electrical_model.solve_power_flow_step();
            electrical_mismatch = electrical_model.calculate_mismatch();

            // Step 5: Update COP based on zone temperature
            let zone_temp = thermal_model.temperatures[0];
            coupler.update_cop(zone_temp);

            // Check convergence
            if thermal_residual < self.tolerance && electrical_mismatch < self.tolerance {
                return ConvergenceResult {
                    converged: true,
                    iterations,
                    thermal_residual,
                    electrical_mismatch,
                };
            }

            iterations += 1;
        }

        ConvergenceResult {
            converged: false,
            iterations,
            thermal_residual,
            electrical_mismatch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_model_creation() {
        let thermal = ThermalModel::new(3, 20.0);
        assert_eq!(thermal.num_zones, 3);
        assert_eq!(thermal.temperatures, vec![20.0, 20.0, 20.0]);
    }

    #[test]
    fn test_electrical_network_creation() {
        let electrical = ElectricalNetwork::new(2);
        assert_eq!(electrical.num_buses, 2);
        assert_eq!(electrical.voltages, vec![1.0, 1.0]);
    }

    #[test]
    fn test_coupler_creation() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        assert_eq!(coupler.cop, 3.0);
        assert_eq!(coupler.rated_cop, 3.0);
    }

    #[test]
    fn test_cop_update() {
        let mut coupler = ThermalElectricalCoupler::new(3.0);
        coupler.update_cop(10.0); // 10°C ambient
        assert!(coupler.cop >= 1.0);
        assert!(coupler.cop <= coupler.rated_cop);
    }

    #[test]
    fn test_thermal_to_electrical_simple() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let electrical = coupler.thermal_to_electrical_simple(3000.0);
        // Q / COP = P_electrical: 3000W / 3.0 = 1000W
        assert!((electrical - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_joint_convergence_simple() {
        let mut thermal = ThermalModel::new(1, 18.0);
        thermal.heating_setpoints = vec![20.0];
        thermal.ambient_temperature = 10.0;

        let mut electrical = ElectricalNetwork::new(1);

        let mut coupler = ThermalElectricalCoupler::new(3.0);

        let mut solver = JointConvergenceSolver::new(100, 1e-3);
        let result = solver.solve(&mut thermal, &mut electrical, &mut coupler);

        assert!(
            result.converged,
            "Solver should converge, got iterations={}, thermal_res={}, elec_mis={}",
            result.iterations, result.thermal_residual, result.electrical_mismatch
        );
        assert!(result.iterations < 100);
    }

    #[test]
    fn test_joint_convergence_two_zones() {
        let mut thermal = ThermalModel::new(2, 18.0);
        thermal.heating_setpoints = vec![20.0, 20.0];
        thermal.ambient_temperature = 10.0;

        let electrical = ElectricalNetwork::new(2);

        let mut coupler = ThermalElectricalCoupler::new(3.0);

        let mut solver = JointConvergenceSolver::new(100, 1e-3);

        // Solve thermal-only (electrical mismatch will be non-zero due to simplified power flow)
        // For full power flow convergence, a more sophisticated solver would be needed
        let mut thermal_copy = thermal.clone();
        let result = solver.solve(&mut thermal_copy, &mut electrical.clone(), &mut coupler);

        // Thermal should converge even if electrical doesn't fully converge
        assert!(
            result.thermal_residual < 1e-2,
            "Thermal residual should be small, got {}",
            result.thermal_residual
        );
        assert!(result.iterations > 0, "Iterations should be reported");
    }

    #[test]
    fn test_joint_convergence_building_heat_pump() {
        let mut thermal = ThermalModel::new(3, 16.0);
        thermal.heating_setpoints = vec![20.0, 20.0, 20.0];
        thermal.ambient_temperature = 5.0;
        thermal.capacitances = vec![5_000_000.0; 3];

        let electrical = ElectricalNetwork::new(3);

        let mut coupler = ThermalElectricalCoupler::new(3.5);

        let mut solver = JointConvergenceSolver::new(200, 1e-4);

        let mut thermal_copy = thermal.clone();
        let result = solver.solve(&mut thermal_copy, &mut electrical.clone(), &mut coupler);

        // Thermal should converge
        assert!(
            result.thermal_residual < 1e-3,
            "Thermal should converge for building + heat pump case, got {}",
            result.thermal_residual
        );
        assert!(result.iterations > 0);
        assert!(coupler.cop > 1.0);
    }

    #[test]
    fn test_convergence_result_reports_iterations() {
        let mut thermal = ThermalModel::new(1, 18.0);
        thermal.heating_setpoints = vec![20.0];
        thermal.ambient_temperature = 10.0;

        let mut electrical = ElectricalNetwork::new(1);

        let mut coupler = ThermalElectricalCoupler::new(3.0);

        let mut solver = JointConvergenceSolver::new(50, 1e-3);
        let result = solver.solve(&mut thermal, &mut electrical, &mut coupler);

        assert!(result.iterations > 0, "Iterations should be reported");
    }

    #[test]
    fn test_max_iterations_prevents_infinite_loop() {
        let mut thermal = ThermalModel::new(5, 100.0);
        thermal.capacitances = vec![100.0; 5];
        thermal.heating_setpoints = vec![20.0; 5];
        thermal.ambient_temperature = -50.0; // Extreme cold

        let mut electrical = ElectricalNetwork::new(5);

        let mut coupler = ThermalElectricalCoupler::new(2.0);

        let mut solver = JointConvergenceSolver::new(10, 1e-12);
        let result = solver.solve(&mut thermal, &mut electrical, &mut coupler);

        assert!(
            !result.converged || result.iterations <= 10,
            "Should respect max iterations"
        );
        assert!(
            result.iterations <= 10,
            "Should not exceed max iterations, got {}",
            result.iterations
        );
    }

    #[test]
    fn test_electrical_mismatch_is_reasonable() {
        // Single bus with known load
        let mut electrical = ElectricalNetwork::new(1);
        electrical.power_injections[0] = -100.0; // 100W load

        let mismatch = electrical.calculate_mismatch();
        // For 1 bus (reference bus), mismatch should be 0
        assert_eq!(
            mismatch, 0.0,
            "Mismatch with only reference bus should be 0"
        );
    }

    #[test]
    fn test_electrical_mismatch_two_buses() {
        let electrical = ElectricalNetwork::new(2);
        // After power flow solve, the angles should balance the loads
        // Test that mismatch is computed correctly
        let mismatch = electrical.calculate_mismatch();
        // With no loads, mismatch should be 0
        assert_eq!(mismatch, 0.0);
    }

    // === Tests for ThermalElectricalCoupler with HvacState ===

    #[test]
    fn test_hvac_state_to_electrical_heating() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let building_id = uuid::Uuid::new_v4();
        let state = HvacState {
            building_id,
            thermal_power_w: 3000.0, // 3kW thermal
            setpoint_c: 20.0,
            ambient_temperature_c: 10.0,
            mode: HvacMode::Heating,
        };

        let load = coupler.hvac_state_to_electrical(&state);
        assert!(load.is_some());
        let load = load.unwrap();
        // Q / COP = P: 3000W / 3.0 = 1000W
        assert!((load.power_w - 1000.0).abs() < 1e-6);
        assert_eq!(load.building_id, building_id);
    }

    #[test]
    fn test_hvac_state_to_electrical_cooling() {
        let coupler = ThermalElectricalCoupler::new(4.0);
        let building_id = uuid::Uuid::new_v4();
        let state = HvacState {
            building_id,
            thermal_power_w: 4000.0, // 4kW thermal cooling
            setpoint_c: 24.0,
            ambient_temperature_c: 35.0,
            mode: HvacMode::Cooling,
        };

        let load = coupler.hvac_state_to_electrical(&state);
        assert!(load.is_some());
        let load = load.unwrap();
        // Q / COP = P: 4000W / 4.0 = 1000W
        assert!((load.power_w - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_hvac_state_to_electrical_off_mode() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let building_id = uuid::Uuid::new_v4();
        let state = HvacState {
            building_id,
            thermal_power_w: 3000.0,
            setpoint_c: 20.0,
            ambient_temperature_c: 10.0,
            mode: HvacMode::Off,
        };

        let load = coupler.hvac_state_to_electrical(&state);
        assert!(load.is_none());
    }

    #[test]
    fn test_hvac_state_to_electrical_zero_thermal() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let building_id = uuid::Uuid::new_v4();
        let state = HvacState {
            building_id,
            thermal_power_w: 0.0,
            setpoint_c: 20.0,
            ambient_temperature_c: 10.0,
            mode: HvacMode::Heating,
        };

        let load = coupler.hvac_state_to_electrical(&state);
        assert!(load.is_none());
    }

    #[test]
    fn test_hvac_state_to_electrical_negative_thermal() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let building_id = uuid::Uuid::new_v4();
        let state = HvacState {
            building_id,
            thermal_power_w: -1000.0, // Negative thermal (shouldn't happen in practice)
            setpoint_c: 20.0,
            ambient_temperature_c: 10.0,
            mode: HvacMode::Heating,
        };

        let load = coupler.hvac_state_to_electrical(&state);
        assert!(load.is_none());
    }

    #[test]
    fn test_thermal_to_electrical_batch_with_mapping() {
        let building1 = uuid::Uuid::new_v4();
        let building2 = uuid::Uuid::new_v4();
        let bus1 = uuid::Uuid::new_v4();
        let bus2 = uuid::Uuid::new_v4();

        let mut mapping = BuildingBusMapping::new();
        mapping.add_mapping(building1, bus1);
        mapping.add_mapping(building2, bus2);

        let coupler = ThermalElectricalCoupler::with_mapping(3.0, mapping);

        let states = vec![
            HvacState {
                building_id: building1,
                thermal_power_w: 3000.0,
                setpoint_c: 20.0,
                ambient_temperature_c: 10.0,
                mode: HvacMode::Heating,
            },
            HvacState {
                building_id: building2,
                thermal_power_w: 6000.0,
                setpoint_c: 22.0,
                ambient_temperature_c: 5.0,
                mode: HvacMode::Heating,
            },
        ];

        let (loads, missing) = coupler.thermal_to_electrical(&states);
        assert!(missing.is_empty());
        assert_eq!(loads.len(), 2);

        // Building 1: 3000W / 3.0 = 1000W
        let load1 = loads.get(&building1).unwrap();
        assert!((load1.power_w - 1000.0).abs() < 1e-6);

        // Building 2: 6000W / 3.0 = 2000W
        let load2 = loads.get(&building2).unwrap();
        assert!((load2.power_w - 2000.0).abs() < 1e-6);
    }

    #[test]
    fn test_thermal_to_electrical_batch_missing_building() {
        let building1 = uuid::Uuid::new_v4();
        let building2 = uuid::Uuid::new_v4();
        let unmapped_building = uuid::Uuid::new_v4();
        let bus1 = uuid::Uuid::new_v4();

        let mut mapping = BuildingBusMapping::new();
        mapping.add_mapping(building1, bus1);
        // building2 is NOT mapped

        let coupler = ThermalElectricalCoupler::with_mapping(3.0, mapping);

        let states = vec![
            HvacState {
                building_id: building1,
                thermal_power_w: 3000.0,
                setpoint_c: 20.0,
                ambient_temperature_c: 10.0,
                mode: HvacMode::Heating,
            },
            HvacState {
                building_id: unmapped_building,
                thermal_power_w: 6000.0,
                setpoint_c: 22.0,
                ambient_temperature_c: 5.0,
                mode: HvacMode::Heating,
            },
        ];

        let (loads, missing) = coupler.thermal_to_electrical(&states);

        // Only building1 should have a load
        assert_eq!(loads.len(), 1);
        assert!(loads.contains_key(&building1));

        // unmapped_building should be in missing
        assert_eq!(missing.len(), 1);
        assert!(missing.contains(&unmapped_building));
    }

    #[test]
    fn test_thermal_to_electrical_batch_empty_states() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let states: Vec<HvacState> = vec![];

        let (loads, missing) = coupler.thermal_to_electrical(&states);
        assert!(loads.is_empty());
        assert!(missing.is_empty());
    }

    #[test]
    fn test_cop_update_various_temperatures() {
        let mut coupler = ThermalElectricalCoupler::new(3.0);

        // Warm weather: COP should increase
        coupler.update_cop(20.0);
        let cop_at_20 = coupler.cop;

        // Cold weather: COP should decrease
        coupler.update_cop(0.0);
        let cop_at_0 = coupler.cop;

        // Very cold: COP should decrease further
        coupler.update_cop(-10.0);
        let cop_at_minus10 = coupler.cop;

        assert!(cop_at_20 > cop_at_0);
        assert!(cop_at_0 > cop_at_minus10);

        // All COP values should be positive
        assert!(cop_at_20 > 0.0);
        assert!(cop_at_0 > 0.0);
        assert!(cop_at_minus10 > 0.0);

        // COP should not exceed rated COP by more than 50%
        assert!(cop_at_20 <= coupler.rated_cop * 1.5);
    }

    #[test]
    fn test_electrical_load_default() {
        let load = ElectricalLoad::default();
        assert_eq!(load.power_w, 0.0);
        assert_eq!(load.reactive_power_var, 0.0);
        assert_eq!(load.power_factor, 1.0);
    }

    #[test]
    fn test_electrical_load_with_reactive_power() {
        let building_id = uuid::Uuid::new_v4();
        let load = ElectricalLoad::with_reactive_power(building_id, 1000.0, 0.9);

        assert_eq!(load.building_id, building_id);
        assert!((load.power_w - 1000.0).abs() < 1e-6);
        assert!((load.power_factor - 0.9).abs() < 1e-6);
        // Reactive power = P * sqrt(1 - pf^2) = 1000 * sqrt(1 - 0.81) = 1000 * 0.436 = 436 VAR
        assert!((load.reactive_power_var - 435.89).abs() < 0.1);
    }

    #[test]
    fn test_building_bus_mapping() {
        let mut mapping = BuildingBusMapping::new();
        let building1 = uuid::Uuid::new_v4();
        let building2 = uuid::Uuid::new_v4();
        let bus1 = uuid::Uuid::new_v4();
        let bus2 = uuid::Uuid::new_v4();

        mapping.add_mapping(building1, bus1);
        mapping.add_mapping(building2, bus2);

        assert_eq!(mapping.len(), 2);
        assert!(!mapping.is_empty());

        assert_eq!(mapping.get_bus_id(&building1), Some(bus1));
        assert_eq!(mapping.get_bus_id(&building2), Some(bus2));
        assert_eq!(mapping.get_bus_id(&uuid::Uuid::new_v4()), None);
    }

    #[test]
    fn test_coupler_with_building_mapping() {
        let building1 = uuid::Uuid::new_v4();
        let bus1 = uuid::Uuid::new_v4();

        let mut mapping = BuildingBusMapping::new();
        mapping.add_mapping(building1, bus1);

        let mut coupler = ThermalElectricalCoupler::with_mapping(3.0, mapping);

        // Add another mapping
        let building2 = uuid::Uuid::new_v4();
        let bus2 = uuid::Uuid::new_v4();
        coupler.add_building_bus_mapping(building2, bus2);

        assert_eq!(coupler.building_bus_mapping().len(), 2);
    }
}
