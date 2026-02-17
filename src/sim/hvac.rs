//! HVAC System Models
//!
//! This module provides advanced HVAC system modeling capabilities including
//! Variable Air Volume (VAV), Constant Air Volume (CAV), and heat pump systems.

use serde::{Deserialize, Serialize};

/// HVAC system types supported by the simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HVACSystemType {
    /// Simple on/off HVAC with fixed capacity
    Simple,
    /// Variable Air Volume system with terminal reheat
    VAV,
    /// Constant Air Volume system with fixed airflow
    CAV,
    /// Heat pump system with COP curves
    HeatPump,
}

/// Represents a VAV (Variable Air Volume) terminal unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VAVTerminal {
    /// Terminal unit identifier
    pub id: String,
    /// Zone served by this terminal
    pub zone_id: usize,
    /// Maximum air flow rate (m³/s)
    pub max_airflow: f64,
    /// Minimum air flow rate (m³/s)
    pub min_airflow: f64,
    /// Reheat coil capacity (W)
    pub reheat_capacity: f64,
    /// Current airflow setpoint (m³/s)
    pub airflow_setpoint: f64,
}

impl VAVTerminal {
    /// Create a new VAV terminal unit
    pub fn new(id: String, zone_id: usize, max_airflow: f64) -> Self {
        Self {
            id,
            zone_id,
            max_airflow,
            min_airflow: max_airflow * 0.3, // Minimum 30% of max
            reheat_capacity: 5000.0,        // Default 5kW reheat
            airflow_setpoint: max_airflow,
        }
    }

    /// Calculate heating demand from reheat coil
    pub fn reheat_demand(&self, supply_temp: f64, zone_temp: f64) -> f64 {
        if zone_temp < 20.0 {
            // Need reheat to maintain minimum supply temp
            let temp_diff = (supply_temp - 18.0).max(0.0);
            // Q = ρ * cp * V̇ * ΔT
            let rho = 1.2; // kg/m³
            let cp = 1005.0; // J/kg·K
            let mass_flow = self.airflow_setpoint * rho; // kg/s
            mass_flow * cp * temp_diff
        } else {
            0.0
        }
    }
}

/// Represents a CAV (Constant Air Volume) system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CAVSystem {
    /// System identifier
    pub id: String,
    /// Design air flow rate (m³/s)
    pub design_airflow: f64,
    /// Fan power consumption (W)
    pub fan_power: f64,
    /// Fan efficiency (0-1)
    pub fan_efficiency: f64,
    /// Heating coil capacity (W)
    pub heating_capacity: f64,
    /// Cooling coil capacity (W)
    pub cooling_capacity: f64,
}

impl CAVSystem {
    /// Create a new CAV system
    pub fn new(id: String, design_airflow: f64) -> Self {
        Self {
            id,
            design_airflow,
            fan_power: design_airflow * 500.0, // Default 500 W per m³/s
            fan_efficiency: 0.7,
            heating_capacity: 10000.0, // Default 10kW
            cooling_capacity: 10000.0, // Default 10kW
        }
    }

    /// Calculate fan power consumption
    pub fn fan_power_consumption(&self) -> f64 {
        self.fan_power / self.fan_efficiency
    }
}

/// Heat pump operating mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeatPumpMode {
    /// Heating mode
    Heating,
    /// Cooling mode
    Cooling,
    /// Off
    Off,
}

/// Represents a heat pump system with COP curves
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatPump {
    /// System identifier
    pub id: String,
    /// Rated heating capacity at design conditions (W)
    pub heating_capacity: f64,
    /// Rated cooling capacity at design conditions (W)
    pub cooling_capacity: f64,
    /// Rated heating COP at design conditions
    pub heating_cop: f64,
    /// Rated cooling COP (EER) at design conditions
    pub cooling_cop: f64,
    /// Design outdoor temperature for heating (°C)
    pub design_temp_heating: f64,
    /// Design outdoor temperature for cooling (°C)
    pub design_temp_cooling: f64,
    /// Current operating mode
    pub mode: HeatPumpMode,
}

impl HeatPump {
    /// Create a new heat pump
    pub fn new(
        id: String,
        heating_capacity: f64,
        cooling_capacity: f64,
        heating_cop: f64,
        cooling_cop: f64,
    ) -> Self {
        Self {
            id,
            heating_capacity,
            cooling_capacity,
            heating_cop,
            cooling_cop,
            design_temp_heating: -5.0,  // Design heating temp
            design_temp_cooling: 35.0,   // Design cooling temp
            mode: HeatPumpMode::Off,
        }
    }

    /// Calculate actual COP based on outdoor temperature
    /// Uses a simple linear degradation model
    pub fn heating_cop_at_temperature(&self, outdoor_temp: f64) -> f64 {
        let temp_diff = (self.design_temp_heating - outdoor_temp).abs();
        // COP degrades by about 2% per degree away from design
        let degradation = 1.0 - (temp_diff * 0.02);
        self.heating_cop * degradation.max(0.5) // Minimum 50% of rated COP
    }

    /// Calculate actual COP based on outdoor temperature for cooling
    pub fn cooling_cop_at_temperature(&self, outdoor_temp: f64) -> f64 {
        let temp_diff = (outdoor_temp - self.design_temp_cooling).abs();
        // EER degrades by about 3% per degree away from design
        let degradation = 1.0 - (temp_diff * 0.03);
        self.cooling_cop * degradation.max(0.5)
    }

    /// Calculate heating power consumption
    pub fn heating_power(&self, outdoor_temp: f64) -> f64 {
        if self.mode != HeatPumpMode::Heating {
            return 0.0;
        }
        // Capacity also degrades with temperature
        let temp_diff = (self.design_temp_heating - outdoor_temp).abs();
        let capacity_factor = 1.0 - (temp_diff * 0.01);
        let actual_capacity = self.heating_capacity * capacity_factor.max(0.3);
        
        let cop = self.heating_cop_at_temperature(outdoor_temp);
        actual_capacity / cop
    }

    /// Calculate cooling power consumption
    pub fn cooling_power(&self, outdoor_temp: f64) -> f64 {
        if self.mode != HeatPumpMode::Cooling {
            return 0.0;
        }
        let temp_diff = (outdoor_temp - self.design_temp_cooling).abs();
        let capacity_factor = 1.0 - (temp_diff * 0.015);
        let actual_capacity = self.cooling_capacity * capacity_factor.max(0.3);
        
        let cop = self.cooling_cop_at_temperature(outdoor_temp);
        actual_capacity / cop
    }

    /// Set the operating mode based on zone temperature and setpoints
    pub fn set_mode(&mut self, zone_temp: f64, heating_sp: f64, cooling_sp: f64) {
        self.mode = if zone_temp < heating_sp {
            HeatPumpMode::Heating
        } else if zone_temp > cooling_sp {
            HeatPumpMode::Cooling
        } else {
            HeatPumpMode::Off
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vav_terminal() {
        let vav = VAVTerminal::new("VAV-1".to_string(), 0, 0.5);
        assert_eq!(vav.max_airflow, 0.5);
        assert_eq!(vav.min_airflow, 0.15);
        
        // Test reheat demand: supply_temp (20°C) > zone_temp (18°C), needs reheat
        let demand = vav.reheat_demand(20.0, 18.0);
        assert!(demand > 0.0);
        
        // No reheat needed when zone temp is comfortable
        let no_demand = vav.reheat_demand(20.0, 22.0);
        assert!(no_demand == 0.0);
    }

    #[test]
    fn test_cav_system() {
        let cav = CAVSystem::new("CAV-1".to_string(), 1.0);
        assert_eq!(cav.design_airflow, 1.0);
        assert!(cav.fan_power_consumption() > 0.0);
    }

    #[test]
    fn test_heat_pump_cop() {
        let hp = HeatPump::new(
            "HP-1".to_string(),
            12000.0, // 12kW heating
            10000.0, // 10kW cooling
            3.5,     // COP 3.5
            3.0,     // EER 3.0
        );
        
        // At design temperature, COP should be rated COP
        let cop_at_design = hp.heating_cop_at_temperature(-5.0);
        assert!((cop_at_design - 3.5).abs() < 0.1);
        
        // At colder temperature, COP should degrade
        let cop_cold = hp.heating_cop_at_temperature(-15.0);
        assert!(cop_cold < 3.5);
    }

    #[test]
    fn test_heat_pump_mode() {
        let mut hp = HeatPump::new(
            "HP-1".to_string(),
            12000.0,
            10000.0,
            3.5,
            3.0,
        );
        
        hp.set_mode(18.0, 20.0, 27.0);
        assert_eq!(hp.mode, HeatPumpMode::Heating);
        
        hp.set_mode(28.0, 20.0, 27.0);
        assert_eq!(hp.mode, HeatPumpMode::Cooling);
        
        hp.set_mode(22.0, 20.0, 27.0);
        assert_eq!(hp.mode, HeatPumpMode::Off);
    }
}
