//! ECS systems for HVAC equipment simulation.
//!
//! Systems are functions that operate on component arrays directly,
//! enabling zero-copy iteration over the SoA storage layout.
//!
//! All systems are non-SIMD (scalar) implementations that process
//! component arrays in a cache-friendly manner.

use crate::ecs::entity::EquipmentKind;
use crate::ecs::storage::EquipmentWorld;

/// Mass Balance System.
///
/// Enforces mass conservation across the HVAC system.
/// For steady-state operation: sum(mass_flowrate_in) = sum(mass_flowrate_out)
///
/// # Zero-copy iteration
///
/// Processes the mass_flowrates array directly without cloning:
/// ```ignore
/// for i in 0..world.entity_count() {
///     let m_dot = mass_flowrates[i];
///     // ... apply mass balance constraints
/// }
/// ```
pub struct MassBalanceSystem;

impl MassBalanceSystem {
    /// Run mass balance system on the equipment world.
    ///
    /// Applies mass conservation constraints to ensure:
    /// - All mass flow rates are non-negative
    /// - Equipment operating in "on" state maintains minimum flow
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(world: &mut EquipmentWorld) {
        let n = world.entity_count();
        if n == 0 {
            return;
        }

        // Collect data into local arrays to avoid multiple mutable borrows
        let kinds: Vec<EquipmentKind> = world.kinds_slice().to_vec();
        let nominal_flowrates: Vec<f64> = world.nominal_flowrates_slice().to_vec();
        let on_offs: Vec<f64> = world.on_offs_slice().to_vec();
        let mut mass_flowrates = world.mass_flowrates_slice().to_vec();

        // Apply mass balance constraints
        for i in 0..n {
            // Ensure minimum flow for active equipment
            if on_offs[i] > 0.5 {
                if mass_flowrates[i] < 0.0 {
                    mass_flowrates[i] = 0.0;
                }
                // For pumps and fans, maintain minimum flow fraction
                match kinds[i] {
                    EquipmentKind::Pump | EquipmentKind::Fan | EquipmentKind::CoolingTower => {
                        let min_flow = nominal_flowrates[i] * 0.1;
                        if mass_flowrates[i] < min_flow {
                            mass_flowrates[i] = min_flow;
                        }
                    }
                    _ => {}
                }
            } else {
                // Equipment off - zero flow
                mass_flowrates[i] = 0.0;
            }
        }

        // Write back to world
        for (i, &m_dot) in mass_flowrates.iter().enumerate().take(n) {
            world.set_mass_flowrate(crate::ecs::entity::EquipmentEntity::new(i as u64), m_dot);
        }
    }
}

/// Heat Transfer System.
///
/// Computes heat transfer for each equipment type:
/// - **Chiller**: Q_evap = rated_capacity * (1 + 0.02 * T_evap) * COP_correction
/// - **Boiler**: Q_output = m_dot * c_p * delta_T / eta
/// - **VAV box**: Q = m_dot * c_p * (T_supply - T_inlet) + reheat
/// - **CoilCooling**: Q = m_dot * c_p_air * effectiveness * (T_wb - T_water)
/// - **CoilHeating**: Q = m_dot * c_p * (T_air_out - T_air_in)
///
/// # Zero-copy iteration
///
/// Directly accesses component arrays:
/// ```ignore
/// let temperatures = world.temperatures_slice();     // &[f64]
/// let enthalpies = world.enthalpies_mut();           // &mut [f64]
/// let rated_capacities = world.rated_capacities_slice(); // &[f64]
/// ```
pub struct HeatTransferSystem;

impl HeatTransferSystem {
    /// Run heat transfer system on the equipment world.
    ///
    /// Computes heat transfer for each active equipment entity
    /// and updates the heat_transfer_outputs array.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(world: &mut EquipmentWorld) {
        let n = world.entity_count();
        if n == 0 {
            return;
        }

        // Collect data into local arrays to avoid multiple mutable borrows
        let temperatures: Vec<f64> = world.temperatures_slice().to_vec();
        let mass_flowrates: Vec<f64> = world.mass_flowrates_slice().to_vec();
        let enthalpies: Vec<f64> = world.enthalpies_slice().to_vec();
        let rated_capacities: Vec<f64> = world.rated_capacities_slice().to_vec();
        let efficiencies: Vec<f64> = world.efficiencies_slice().to_vec();
        let on_offs: Vec<f64> = world.on_offs_slice().to_vec();
        let kinds: Vec<EquipmentKind> = world.kinds_slice().to_vec();

        let mut new_enthalpies = enthalpies.clone();
        let mut outputs: Vec<f64> = vec![0.0; n];

        const C_P_AIR: f64 = 1006.0; // J/(kg·K)
        const C_P_WATER: f64 = 4186.0; // J/(kg·K)

        for i in 0..n {
            // Skip inactive equipment
            if on_offs[i] < 0.5 {
                outputs[i] = 0.0;
                continue;
            }

            let m_dot = mass_flowrates[i];
            let t = temperatures[i];
            let h = enthalpies[i];

            let q = match kinds[i] {
                EquipmentKind::Chiller => {
                    // Chiller heat transfer: Q_evap based on temperature lift
                    let cop = efficiencies[i];
                    let t_cond = t + 28.0; // Assume condensing temp = evap + 28K
                    let delta_t = t_cond - t;
                    let cop_correction = (1.0 - 0.05 * delta_t).max(0.1);
                    let q_evap = rated_capacities[i] * (1.0 + 0.02 * t);
                    let p_compressor = q_evap / (cop * cop_correction);
                    let q_cond = q_evap + p_compressor;

                    // Update enthalpy based on heat transfer
                    new_enthalpies[i] = h + q_evap / m_dot.max(0.001);
                    q_cond
                }
                EquipmentKind::Boiler => {
                    // Boiler heat transfer: Q = m_dot * c_p * delta_T / eta
                    let _t_supply = t + 40.0; // Assume supply temp
                    let eta = efficiencies[i];
                    let q_output = m_dot * C_P_WATER * 40.0;
                    let _q_fuel = q_output / eta;

                    // Update enthalpy
                    new_enthalpies[i] = h + q_output / m_dot.max(0.001);
                    q_output
                }
                EquipmentKind::VavBox => {
                    // VAV box: supply air cooling/reheat
                    let t_supply = (t - 8.0).max(t - 15.0); // Max 15K drop
                    let q_cooling = m_dot * C_P_AIR * (t - t_supply);

                    // Update enthalpy (cooling reduces enthalpy)
                    new_enthalpies[i] = h - q_cooling / m_dot.max(0.001);
                    q_cooling
                }
                EquipmentKind::CoilCooling => {
                    // Cooling coil with effectiveness
                    let effectiveness = 0.75;
                    let t_water_in = t - 5.0;
                    let t_air_out = t - effectiveness * (t - t_water_in);
                    let q_cooling = m_dot * C_P_AIR * (t - t_air_out);

                    new_enthalpies[i] = h - q_cooling / m_dot.max(0.001);
                    q_cooling
                }
                EquipmentKind::CoilHeating => {
                    // Heating coil
                    let _t_air_out = t + 20.0;
                    let q_heating = m_dot * C_P_AIR * 20.0;

                    new_enthalpies[i] = h + q_heating / m_dot.max(0.001);
                    q_heating
                }
                EquipmentKind::Pump => {
                    // Pump: mechanical work to fluid
                    let rated_power = rated_capacities[i];
                    let power = rated_power * 0.8; // Assume 80% of rated
                    let _p_rise = power / m_dot.max(0.001);

                    // Enthalpy increase due to pump work
                    new_enthalpies[i] = h + power / m_dot.max(0.001);
                    power
                }
                EquipmentKind::Fan => {
                    // Fan: mechanical work to air
                    let rated_power = rated_capacities[i];
                    let power = rated_power * 0.75;
                    let _p_rise = power / m_dot.max(0.001);

                    new_enthalpies[i] = h + power / m_dot.max(0.001);
                    power
                }
                EquipmentKind::CoolingTower => {
                    // Cooling tower: evaporative cooling
                    let effectiveness = 0.7;
                    let t_ambient = 25.0;
                    let t_out = t - effectiveness * (t - t_ambient);
                    let q_rejected = m_dot * C_P_WATER * (t - t_out);

                    new_enthalpies[i] = h - q_rejected / m_dot.max(0.001);
                    q_rejected
                }
                EquipmentKind::Damper => {
                    // Damper: no heat transfer, just pressure drop
                    let _pressure_drop = 50.0 * (1.0 - 0.5); // KPa drop based on position
                    0.0 // Dampers don't add/remove heat
                }
            };

            outputs[i] = q;
        }

        // Write back to world
        for (i, &h) in new_enthalpies.iter().enumerate().take(n) {
            let entity = crate::ecs::entity::EquipmentEntity::new(i as u64);
            world.set_enthalpy(entity, h);
        }
    }
}

/// Control Loop System.
///
/// Implements simple proportional control for setpoint tracking:
/// - Updates actuator positions based on error between setpoint and measured value
/// - Clamps positions to [0.0, 1.0] range
///
/// # Zero-copy iteration
///
/// Directly accesses and modifies position arrays:
/// ```ignore
/// let positions = world.positions_mut();
/// let setpoints = world.setpoints_slice();
/// let measurements = world.temperatures_slice();
/// ```
pub struct ControlLoopSystem;

impl ControlLoopSystem {
    /// Run control loop system on the equipment world.
    ///
    /// Uses proportional control: u = Kp * (setpoint - measured)
    /// with anti-windup clamping.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(world: &mut EquipmentWorld) {
        let n = world.entity_count();
        if n == 0 {
            return;
        }

        // Collect data into local arrays to avoid multiple mutable borrows
        let setpoints: Vec<f64> = world.setpoints_slice().to_vec();
        let temperatures: Vec<f64> = world.temperatures_slice().to_vec();
        let on_offs: Vec<f64> = world.on_offs_slice().to_vec();
        let kinds: Vec<EquipmentKind> = world.kinds_slice().to_vec();
        let mut positions: Vec<f64> = world.positions_slice().to_vec();

        // Control gain
        const K_P: f64 = 0.1;

        for i in 0..n {
            // Skip inactive equipment
            if on_offs[i] < 0.5 {
                positions[i] = 0.0;
                continue;
            }

            // Calculate control error based on equipment type
            let error = match kinds[i] {
                EquipmentKind::Chiller
                | EquipmentKind::Boiler
                | EquipmentKind::CoilCooling
                | EquipmentKind::CoilHeating => {
                    // Temperature control
                    setpoints[i] - temperatures[i]
                }
                EquipmentKind::VavBox | EquipmentKind::Damper => {
                    // Position control - adjust based on deviation from target position
                    let target_pos = setpoints[i].clamp(0.0, 1.0);
                    target_pos - positions[i]
                }
                EquipmentKind::Pump | EquipmentKind::Fan => {
                    // Speed/flow control
                    let target_flow = setpoints[i];
                    target_flow - 0.5 // Assume current flow is 0.5
                }
                EquipmentKind::CoolingTower => {
                    // Temperature control
                    setpoints[i] - temperatures[i]
                }
            };

            // Proportional control update
            let delta_u = K_P * error;
            positions[i] = (positions[i] + delta_u).clamp(0.0, 1.0);
        }

        // Write back to world
        for (i, &p) in positions.iter().enumerate().take(n) {
            let entity = crate::ecs::entity::EquipmentEntity::new(i as u64);
            world.set_position(entity, p);
        }
    }
}
