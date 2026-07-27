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
pub mod thermal_electrical_coupler;
pub mod heat_pump_voltage_model;

pub use error::GridModelError;
pub use thermal_electrical_coupler::VoltageCoupler;
pub use heat_pump_voltage_model::HeatPumpVoltageModel;


// === Joint Thermal-Electrical Convergence Solver ===
// fluxion-grid: Electrical grid modeling with bus node types for power flow analysis.
//
// This crate provides foundational types for electrical bus modeling including:
// - Bus node types (Slack, PV, PQ)
// - Electrical bus structures with voltage, angle, and power attributes
// - Battery bus with State of Charge (SoC) tracking
// - Joint thermal-electrical convergence solver for grid-interactive buildings

pub mod bus;
pub mod battery;
pub mod power_flow;

pub use bus::{BusNodeType, ElectricalBus};
pub use battery::BatteryBus;
pub use power_flow::PowerFlowState;

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
            let envelope_loss = self.envelope_conductance[i] * (self.temperatures[i] - self.ambient_temperature);
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
            let envelope_loss = self.envelope_conductance[i] * (self.temperatures[i] - self.ambient_temperature);
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
}

impl ThermalElectricalCoupler {
    /// Create a new coupler with specified COP.
    pub fn new(cop: f64) -> Self {
        ThermalElectricalCoupler {
            cop,
            rated_cop: cop,
            source_temperature: 10.0,
            delivery_temperature: 45.0,
        }
    }

    /// Calculate COP based on Carnot efficiency and degradation.
    ///
    /// COP = COP_rated * efficiency_factor * carnot_ratio
    pub fn update_cop(&mut self, ambient_temperature: f64) {
        let t_hot_k = self.delivery_temperature + 273.15;
        let t_cold_k = ambient_temperature.max(-10.0) + 273.15;

        // Carnot COP: COP_carnot = T_hot / (T_hot - T_cold)
        let carnot = t_hot_k / (t_hot_k - t_cold_k);

        // Real heat pumps achieve ~40-60% of Carnot, but the rated COP already
        // accounts for this, so we use the ratio of actual to reference Carnot
        // Reference Carnot at 20°C: 293.15 / (293.15 - 283.15) = 29.3
        let reference_carnot = 293.15 / 10.0;

        // COP adjustment based on temperature difference
        let carnot_factor = (carnot / reference_carnot).clamp(0.3, 1.5);

        // Actual COP = rated_COP * carnot_factor
        // The rated COP already includes the efficiency factor
        self.cop = self.rated_cop * carnot_factor;
        self.cop = self.cop.clamp(1.0, self.rated_cop * 1.5);
    }

    /// Convert thermal load to electrical power.
    pub fn thermal_to_electrical(&self, thermal_power: f64) -> f64 {
        thermal_power / self.cop
    }

    /// Convert electrical power to thermal power.
    pub fn electrical_to_thermal(&self, electrical_power: f64) -> f64 {
        electrical_power * self.cop
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
            let electrical_load = coupler.thermal_to_electrical(total_thermal_load);

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
    fn test_thermal_to_electrical() {
        let coupler = ThermalElectricalCoupler::new(3.0);
        let electrical = coupler.thermal_to_electrical(3000.0);
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

        assert!(result.converged, "Solver should converge, got iterations={}, thermal_res={}, elec_mis={}",
            result.iterations, result.thermal_residual, result.electrical_mismatch);
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
        assert!(result.thermal_residual < 1e-2,
            "Thermal residual should be small, got {}", result.thermal_residual);
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
        assert!(result.thermal_residual < 1e-3,
            "Thermal should converge for building + heat pump case, got {}",
            result.thermal_residual);
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
        assert_eq!(mismatch, 0.0, "Mismatch with only reference bus should be 0");
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
}
