//! Ideal Loads HVAC System
//!
//! This module provides the Ideal Loads concept following EnergyPlus terminology:
//! - **ZoneIdealLoads**: Calculates the sensible and latent thermal energy required to meet
//!   a zone setpoint - assumes 100% efficiency and infinite capacity
//! - **SimpleHVACEquipment**: Converts thermal load to electrical power via COP/efficiency
//! - **IdealLoadsSystem**: Combines both for complete HVAC simulation
//!
//! ASHRAE 140 Standard Values:
//! - Cooling COP: 3.0 (typical for heat pump)
//! - Heating efficiency: 0.9 (electric resistance/furnace)

use serde::{Deserialize, Serialize};

/// Result structure for HVAC energy calculations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HVACEnergyResult {
    /// Thermal load the zone NEEDS (watts) - what equipment must provide
    pub thermal_load_watts: f64,
    /// Electrical power consumed by equipment (kW)
    pub electrical_kw: f64,
    /// Mode of operation
    pub mode: HVACMode,
}

impl HVACEnergyResult {
    /// Create a new HVAC energy result
    pub fn new(thermal_load_watts: f64, electrical_kw: f64, mode: HVACMode) -> Self {
        Self {
            thermal_load_watts,
            electrical_kw,
            mode,
        }
    }
}

/// HVAC operating mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HVACMode {
    /// Heating mode
    Heating,
    /// Cooling mode
    Cooling,
    /// No load / off
    None,
}

impl Default for HVACMode {
    fn default() -> Self {
        HVACMode::None
    }
}

/// Zone Ideal Loads - calculates what the zone NEEDS (100% efficient, infinite capacity)
///
/// This represents the "Ideal Loads Air System" in EnergyPlus terminology:
/// - Calculates sensible and latent thermal energy required to meet setpoint
/// - Assumes 100% efficiency and infinite capacity
/// - Does NOT account for equipment COP/efficiency (that's SimpleHVACEquipment)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ZoneIdealLoads {
    /// Sensible cooling load (watts) - heat to remove
    pub sensible_cooling_watts: f64,
    /// Sensible heating load (watts) - heat to add
    pub sensible_heating_watts: f64,
    /// Latent cooling load (watts) - moisture to remove
    pub latent_cooling_watts: f64,
    /// Latent heating load (watts) - moisture to add (rarely used)
    pub latent_heating_watts: f64,
}

impl Default for ZoneIdealLoads {
    fn default() -> Self {
        Self {
            sensible_cooling_watts: 0.0,
            sensible_heating_watts: 0.0,
            latent_cooling_watts: 0.0,
            latent_heating_watts: 0.0,
        }
    }
}

impl ZoneIdealLoads {
    /// Create a new ZoneIdealLoads with zero values
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate sensible cooling load
    ///
    /// Uses simplified air heat balance: Q = ρ * cp * V̇ * (T_zone - T_supply)
    /// where T_supply < T_zone for cooling
    ///
    /// # Arguments
    /// * `zone_temp` - Current zone air temperature (°C)
    /// * `cooling_setpoint` - Cooling setpoint temperature (°C)
    /// * `supply_air_temp` - Supply air temperature from HVAC (°C), typically 13°C for cooling
    /// * `zone_volume` - Zone volume in cubic meters (m³)
    /// * `air_changes_per_hour` - Ventilation air changes per hour (ACH)
    ///
    /// # Returns
    /// Sensible cooling load in watts (positive when zone needs cooling)
    pub fn calculate_sensible_cooling_load(
        zone_temp: f64,
        cooling_setpoint: f64,
        supply_air_temp: f64,
        zone_volume: f64,
        air_changes_per_hour: f64,
    ) -> f64 {
        if zone_temp <= cooling_setpoint {
            return 0.0;
        }

        let airflow_m3s = zone_volume * air_changes_per_hour / 3600.0;

        let rho = 1.2;
        let cp = 1005.0;

        let mass_flow = airflow_m3s * rho;
        let delta_t = (zone_temp - supply_air_temp).max(0.0);

        mass_flow * cp * delta_t
    }

    /// Calculate sensible heating load
    ///
    /// Uses simplified air heat balance: Q = ρ * cp * V̇ * (T_supply - T_zone)
    /// where T_supply > T_zone for heating
    ///
    /// # Arguments
    /// * `zone_temp` - Current zone air temperature (°C)
    /// * `heating_setpoint` - Heating setpoint temperature (°C)
    /// * `supply_air_temp` - Supply air temperature from HVAC (°C), typically 40-50°C for heating
    /// * `zone_volume` - Zone volume in cubic meters (m³)
    /// * `air_changes_per_hour` - Ventilation air changes per hour (ACH)
    ///
    /// # Returns
    /// Sensible heating load in watts (positive when zone needs heating)
    pub fn calculate_sensible_heating_load(
        zone_temp: f64,
        heating_setpoint: f64,
        supply_air_temp: f64,
        zone_volume: f64,
        air_changes_per_hour: f64,
    ) -> f64 {
        if zone_temp >= heating_setpoint {
            return 0.0;
        }

        let airflow_m3s = zone_volume * air_changes_per_hour / 3600.0;

        let rho = 1.2;
        let cp = 1005.0;

        let mass_flow = airflow_m3s * rho;
        let delta_t = (supply_air_temp - zone_temp).max(0.0);

        mass_flow * cp * delta_t
    }

    /// Calculate latent cooling load (moisture removal)
    ///
    /// # Arguments
    /// * `zone_humidity_ratio` - Zone humidity ratio (kg_water/kg_dry_air)
    /// * `supply_humidity_ratio` - Supply air humidity ratio
    /// * `airflow_m3s` - Supply airflow rate (m³/s)
    ///
    /// # Returns
    /// Latent cooling load in watts
    pub fn calculate_latent_cooling_load(
        zone_humidity_ratio: f64,
        supply_humidity_ratio: f64,
        airflow_m3s: f64,
    ) -> f64 {
        if zone_humidity_ratio <= supply_humidity_ratio {
            return 0.0;
        }

        let rho = 1.2; // kg/m³
        let h_fg = 2501000.0; // J/kg (latent heat of vaporization at 20°C)

        let mass_flow = airflow_m3s * rho;
        let humidity_diff = zone_humidity_ratio - supply_humidity_ratio;

        mass_flow * humidity_diff * h_fg
    }

    /// Determine the required HVAC mode based on loads
    pub fn determine_mode(&self) -> HVACMode {
        // Note: Both positive indicates heating AND cooling (shouldn't happen in normal operation)
        if self.sensible_heating_watts > 0.0 && self.sensible_cooling_watts > 0.0 {
            // This is unusual - pick the larger one
            if self.sensible_heating_watts >= self.sensible_cooling_watts {
                HVACMode::Heating
            } else {
                HVACMode::Cooling
            }
        } else if self.sensible_heating_watts > 0.0 {
            HVACMode::Heating
        } else if self.sensible_cooling_watts > 0.0 {
            HVACMode::Cooling
        } else {
            HVACMode::None
        }
    }
}

/// Simple HVAC Equipment - converts thermal load to electrical consumption
///
/// This represents the equipment-side of HVAC in EnergyPlus:
/// - Takes the ideal loads (what the zone NEEDS)
/// - Applies COP/efficiency to determine electrical consumption
/// - Accounts for real-world equipment limitations
///
/// ASHRAE 140 Standard Values:
/// - Cooling COP: 3.0 (typical for heat pump)
/// - Heating efficiency: 0.9 (electric resistance/furnace)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleHVACEquipment {
    /// Coefficient of Performance for cooling (default 3.0 for ASHRAE 140)
    pub cooling_cop: f64,
    /// Heating efficiency (default 0.9 for electric resistance)
    pub heating_efficiency: f64,
    /// Equipment identifier
    pub equipment_name: String,
}

impl Default for SimpleHVACEquipment {
    fn default() -> Self {
        Self {
            cooling_cop: 3.0,
            heating_efficiency: 0.9,
            equipment_name: "Default".to_string(),
        }
    }
}

impl SimpleHVACEquipment {
    /// Create new SimpleHVACEquipment with default ASHRAE 140 values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create SimpleHVACEquipment with custom COP and efficiency
    ///
    /// # Arguments
    /// * `cop` - Coefficient of Performance for cooling
    /// * `efficiency` - Heating efficiency (0.0 to 1.0)
    pub fn with_custom_cop(cop: f64, efficiency: f64) -> Self {
        Self {
            cooling_cop: cop,
            heating_efficiency: efficiency.clamp(0.1, 1.0), // Clamp to reasonable range
            equipment_name: "Custom".to_string(),
        }
    }

    /// Convert thermal load to electrical consumption
    ///
    /// # Arguments
    /// * `thermal_load_watts` - Thermal energy required (watts)
    /// * `mode` - Operating mode (Heating, Cooling, None)
    ///
    /// # Returns
    /// Electrical power consumption in watts
    pub fn calculate_electrical_consumption(&self, thermal_load_watts: f64, mode: HVACMode) -> f64 {
        match mode {
            HVACMode::Cooling => thermal_load_watts / self.cooling_cop,
            HVACMode::Heating => thermal_load_watts / self.heating_efficiency,
            HVACMode::None => 0.0,
        }
    }

    /// Convert thermal load to electrical consumption (kW output)
    ///
    /// Convenience method that returns kW instead of watts
    pub fn calculate_electrical_kw(&self, thermal_load_watts: f64, mode: HVACMode) -> f64 {
        self.calculate_electrical_consumption(thermal_load_watts, mode) / 1000.0
    }
}

/// Ideal Loads System - combines zone loads and equipment
///
/// This is the main interface for HVAC simulation:
/// - Calculates ideal thermal loads (what zone needs)
/// - Converts to electrical consumption (what equipment uses)
/// - Returns both values for energy accounting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdealLoadsSystem {
    /// Zone thermal loads
    pub zone_loads: ZoneIdealLoads,
    /// HVAC equipment model
    pub equipment: SimpleHVACEquipment,
    /// Supply air temperature for cooling (°C), typically 13°C
    pub supply_cooling_temp: f64,
    /// Supply air temperature for heating (°C), typically 40°C
    pub supply_heating_temp: f64,
    /// Zone volume in cubic meters (m³)
    pub zone_volume: f64,
    /// Ventilation air changes per hour (ACH)
    pub air_changes_per_hour: f64,
}

impl IdealLoadsSystem {
    /// Create a new IdealLoadsSystem with zone properties.
    /// Zone properties are required - no hardcoded defaults.
    ///
    /// # Arguments
    /// * `zone_volume` - Zone volume in cubic meters (m³)
    /// * `air_changes_per_hour` - Ventilation air changes per hour (ACH)
    pub fn new(zone_volume: f64, air_changes_per_hour: f64) -> Self {
        Self {
            zone_loads: ZoneIdealLoads::new(),
            equipment: SimpleHVACEquipment::new(),
            supply_cooling_temp: 13.0,
            supply_heating_temp: 40.0,
            zone_volume,
            air_changes_per_hour,
        }
    }

    /// Create IdealLoadsSystem with custom equipment and zone properties.
    ///
    /// # Arguments
    /// * `equipment` - HVAC equipment model
    /// * `zone_volume` - Zone volume in cubic meters (m³)
    /// * `air_changes_per_hour` - Ventilation air changes per hour (ACH)
    pub fn with_equipment_and_zone_properties(
        equipment: SimpleHVACEquipment,
        zone_volume: f64,
        air_changes_per_hour: f64,
    ) -> Self {
        Self {
            zone_loads: ZoneIdealLoads::new(),
            equipment,
            supply_cooling_temp: 13.0,
            supply_heating_temp: 40.0,
            zone_volume,
            air_changes_per_hour,
        }
    }

    /// Calculate both thermal loads AND electrical consumption
    ///
    /// This is the main method that:
    /// 1. Calculates ideal thermal load (what zone needs)
    /// 2. Converts to electrical consumption (what equipment uses)
    ///
    /// # Arguments
    /// * `zone_temp` - Current zone air temperature (°C)
    /// * `heating_setpoint` - Heating setpoint (°C)
    /// * `cooling_setpoint` - Cooling setpoint (°C)
    ///
    /// # Returns
    /// HVACEnergyResult containing both thermal and electrical values
    pub fn calculate(
        &mut self,
        zone_temp: f64,
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> HVACEnergyResult {
        let cooling_load = ZoneIdealLoads::calculate_sensible_cooling_load(
            zone_temp,
            cooling_setpoint,
            self.supply_cooling_temp,
            self.zone_volume,
            self.air_changes_per_hour,
        );
        let heating_load = ZoneIdealLoads::calculate_sensible_heating_load(
            zone_temp,
            heating_setpoint,
            self.supply_heating_temp,
            self.zone_volume,
            self.air_changes_per_hour,
        );

        self.zone_loads.sensible_cooling_watts = cooling_load;
        self.zone_loads.sensible_heating_watts = heating_load;

        let mode = self.zone_loads.determine_mode();
        let thermal_load = match mode {
            HVACMode::Cooling => cooling_load,
            HVACMode::Heating => heating_load,
            HVACMode::None => 0.0,
        };

        let electrical_kw = self.equipment.calculate_electrical_kw(thermal_load, mode);

        HVACEnergyResult::new(thermal_load, electrical_kw, mode)
    }

    /// Reset loads for new timestep
    pub fn reset(&mut self) {
        self.zone_loads = ZoneIdealLoads::new();
    }

    /// Calculate per-zone HVAC power demand vector using thermodynamic formulas.
    ///
    /// This method replaces the sensitivity-based `hvac_power_demand()` calculation
    /// with proper ideal loads physics: `mass_flow * cp * delta_t`.
    ///
    /// Returns a vector of power values where:
    /// - Positive = heating demand (W)
    /// - Negative = cooling demand (W)
    /// - Zero = no demand or HVAC disabled
    ///
    /// # Arguments
    /// * `zone_temps` - Current zone temperatures (°C)
    /// * `heating_setpoints` - Heating setpoints per zone (°C)
    /// * `cooling_setpoints` - Cooling setpoints per zone (°C)
    /// * `hvac_enabled` - HVAC enabled flag per zone (true = enabled)
    pub fn calculate_power_demand_vector(
        &self,
        zone_temps: &[f64],
        heating_setpoints: &[f64],
        cooling_setpoints: &[f64],
        hvac_enabled: &[f64],
    ) -> Vec<f64> {
        let n = zone_temps.len();
        let mut demand_vec = Vec::with_capacity(n);

        for (i, zone_temp) in zone_temps.iter().enumerate().take(n) {
            let enabled = hvac_enabled.get(i).copied().unwrap_or(1.0);
            if enabled < 0.5 {
                demand_vec.push(0.0);
                continue;
            }
            let heating_sp = heating_setpoints.get(i).copied().unwrap_or(20.0);
            let cooling_sp = cooling_setpoints.get(i).copied().unwrap_or(24.0);

            let cooling_load = ZoneIdealLoads::calculate_sensible_cooling_load(
                *zone_temp,
                cooling_sp,
                self.supply_cooling_temp,
                self.zone_volume,
                self.air_changes_per_hour,
            );
            let heating_load = ZoneIdealLoads::calculate_sensible_heating_load(
                *zone_temp,
                heating_sp,
                self.supply_heating_temp,
                self.zone_volume,
                self.air_changes_per_hour,
            );

            let thermal_load = if cooling_load > 0.0 && cooling_load >= heating_load {
                // Cooling mode: return negative to indicate heat removal
                -cooling_load
            } else if heating_load > 0.0 {
                // Heating mode: return positive
                heating_load
            } else {
                0.0
            };
            demand_vec.push(thermal_load)
        }

        demand_vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== ZoneIdealLoads Tests ====================

    #[test]
    fn test_zone_ideal_loads_default() {
        let loads = ZoneIdealLoads::default();
        assert_eq!(loads.sensible_cooling_watts, 0.0);
        assert_eq!(loads.sensible_heating_watts, 0.0);
    }

    #[test]
    fn test_sensible_cooling_load_above_setpoint() {
        let zone_volume = 129.6; // ASHRAE 140 standard
        let ach = 0.5; // ASHRAE 140 standard
        let load =
            ZoneIdealLoads::calculate_sensible_cooling_load(25.0, 24.0, 13.0, zone_volume, ach);
        assert!(
            load > 0.0,
            "Should have positive cooling load when zone > setpoint"
        );
    }

    #[test]
    fn test_sensible_cooling_load_below_setpoint() {
        let zone_volume = 129.6;
        let ach = 0.5;
        let load =
            ZoneIdealLoads::calculate_sensible_cooling_load(22.0, 24.0, 13.0, zone_volume, ach);
        assert_eq!(load, 0.0, "No cooling load when zone < setpoint");
    }

    #[test]
    fn test_sensible_heating_load_below_setpoint() {
        let zone_volume = 129.6;
        let ach = 0.5;
        let load =
            ZoneIdealLoads::calculate_sensible_heating_load(18.0, 20.0, 40.0, zone_volume, ach);
        assert!(
            load > 0.0,
            "Should have positive heating load when zone < setpoint"
        );
    }

    #[test]
    fn test_sensible_heating_load_above_setpoint() {
        let zone_volume = 129.6;
        let ach = 0.5;
        let load =
            ZoneIdealLoads::calculate_sensible_heating_load(22.0, 20.0, 40.0, zone_volume, ach);
        assert_eq!(load, 0.0, "No heating load when zone > setpoint");
    }

    #[test]
    fn test_determine_mode_cooling() {
        let mut loads = ZoneIdealLoads::new();
        loads.sensible_cooling_watts = 1000.0;
        loads.sensible_heating_watts = 0.0;
        assert_eq!(loads.determine_mode(), HVACMode::Cooling);
    }

    #[test]
    fn test_determine_mode_heating() {
        let mut loads = ZoneIdealLoads::new();
        loads.sensible_cooling_watts = 0.0;
        loads.sensible_heating_watts = 1000.0;
        assert_eq!(loads.determine_mode(), HVACMode::Heating);
    }

    #[test]
    fn test_determine_mode_none() {
        let loads = ZoneIdealLoads::default();
        assert_eq!(loads.determine_mode(), HVACMode::None);
    }

    // ==================== SimpleHVACEquipment Tests ====================

    #[test]
    fn test_simple_hvac_default_values() {
        let equipment = SimpleHVACEquipment::default();
        assert_eq!(equipment.cooling_cop, 3.0);
        assert_eq!(equipment.heating_efficiency, 0.9);
    }

    #[test]
    fn test_simple_hvac_custom_values() {
        let equipment = SimpleHVACEquipment::with_custom_cop(4.0, 0.95);
        assert_eq!(equipment.cooling_cop, 4.0);
        assert_eq!(equipment.heating_efficiency, 0.95);
    }

    #[test]
    fn test_electrical_consumption_cooling() {
        let equipment = SimpleHVACEquipment::default();
        // 3000W thermal load / COP 3.0 = 1000W electrical
        let electrical = equipment.calculate_electrical_consumption(3000.0, HVACMode::Cooling);
        assert!((electrical - 1000.0).abs() < 0.1);
    }

    #[test]
    fn test_electrical_consumption_heating() {
        let equipment = SimpleHVACEquipment::default();
        // 2000W thermal load / efficiency 0.9 = 2222W electrical
        let electrical = equipment.calculate_electrical_consumption(2000.0, HVACMode::Heating);
        assert!((electrical - 2222.22).abs() < 1.0);
    }

    #[test]
    fn test_electrical_consumption_off() {
        let equipment = SimpleHVACEquipment::default();
        let electrical = equipment.calculate_electrical_consumption(1000.0, HVACMode::None);
        assert_eq!(electrical, 0.0);
    }

    #[test]
    fn test_electrical_kw_cooling() {
        let equipment = SimpleHVACEquipment::default();
        // 3000W thermal / COP 3.0 = 1000W = 1.0 kW
        let kw = equipment.calculate_electrical_kw(3000.0, HVACMode::Cooling);
        assert!((kw - 1.0).abs() < 0.01);
    }

    // ==================== IdealLoadsSystem Tests ====================

    #[test]
    fn test_ideal_loads_system_with_ashrae_140_properties() {
        let zone_volume = 129.6; // ASHRAE 140 standard (8m × 6m × 2.7m)
        let ach = 0.5; // ASHRAE 140 standard infiltration
        let system = IdealLoadsSystem::new(zone_volume, ach);
        assert_eq!(system.supply_cooling_temp, 13.0);
        assert_eq!(system.supply_heating_temp, 40.0);
        assert_eq!(system.zone_volume, 129.6);
        assert_eq!(system.air_changes_per_hour, 0.5);
    }

    #[test]
    fn test_ideal_loads_system_with_equipment() {
        let zone_volume = 129.6;
        let ach = 0.5;
        let equipment = SimpleHVACEquipment::with_custom_cop(5.0, 1.0);
        let system =
            IdealLoadsSystem::with_equipment_and_zone_properties(equipment, zone_volume, ach);
        assert_eq!(system.equipment.cooling_cop, 5.0);
        assert_eq!(system.zone_volume, 129.6);
        assert_eq!(system.air_changes_per_hour, 0.5);
    }

    #[test]
    fn test_calculate_cooling_mode() {
        let mut system = IdealLoadsSystem::new(129.6, 0.5);
        let result = system.calculate(25.0, 20.0, 24.0);

        assert_eq!(result.mode, HVACMode::Cooling);
        assert!(result.thermal_load_watts > 0.0);
        assert!(result.electrical_kw > 0.0);
        assert!(result.electrical_kw < result.thermal_load_watts / 1000.0);
    }

    #[test]
    fn test_calculate_heating_mode() {
        let mut system = IdealLoadsSystem::new(129.6, 0.5);
        let result = system.calculate(18.0, 20.0, 24.0);

        assert_eq!(result.mode, HVACMode::Heating);
        assert!(result.thermal_load_watts > 0.0);
        assert!(result.electrical_kw > result.thermal_load_watts / 1000.0);
    }

    #[test]
    fn test_calculate_deadband() {
        let mut system = IdealLoadsSystem::new(129.6, 0.5);
        let result = system.calculate(22.0, 20.0, 24.0);

        assert_eq!(result.mode, HVACMode::None);
        assert_eq!(result.thermal_load_watts, 0.0);
        assert_eq!(result.electrical_kw, 0.0);
    }

    #[test]
    fn test_reset() {
        let mut system = IdealLoadsSystem::new(129.6, 0.5);
        let _ = system.calculate(25.0, 20.0, 24.0);
        assert!(system.zone_loads.sensible_cooling_watts > 0.0);

        system.reset();
        assert_eq!(system.zone_loads.sensible_cooling_watts, 0.0);
    }

    // ==================== Integration Tests ====================

    #[test]
    fn test_separate_loads_and_equipment() {
        let zone_volume = 129.6;
        let ach = 0.5;

        let cooling_load =
            ZoneIdealLoads::calculate_sensible_cooling_load(28.0, 24.0, 13.0, zone_volume, ach);

        let equipment = SimpleHVACEquipment::default();
        let electrical_watts =
            equipment.calculate_electrical_consumption(cooling_load, HVACMode::Cooling);

        let expected = cooling_load / 3.0;
        assert!((electrical_watts - expected).abs() < 0.1);
    }

    #[test]
    fn test_multiple_equipment_types() {
        // Test that different equipment types can be swapped
        let thermal_load = 3000.0; // 3kW thermal

        // Standard equipment (COP=3.0, eff=0.9)
        let standard = SimpleHVACEquipment::default();
        let standard_cooling = standard.calculate_electrical_kw(thermal_load, HVACMode::Cooling);
        let standard_heating = standard.calculate_electrical_kw(thermal_load, HVACMode::Heating);

        // High-efficiency equipment (COP=5.0, eff=1.0)
        let high_eff = SimpleHVACEquipment::with_custom_cop(5.0, 1.0);
        let high_cooling = high_eff.calculate_electrical_kw(thermal_load, HVACMode::Cooling);
        let high_heating = high_eff.calculate_electrical_kw(thermal_load, HVACMode::Heating);

        // Verify high efficiency uses less electricity
        assert!(
            high_cooling < standard_cooling,
            "High COP should use less power"
        );
        assert!(
            high_heating < standard_heating,
            "High efficiency should use less power"
        );
    }

    #[test]
    fn test_ashrae_140_standard_values() {
        let equipment = SimpleHVACEquipment::default();

        assert_eq!(equipment.cooling_cop, 3.0);
        assert_eq!(equipment.heating_efficiency, 0.9);

        let thermal_3000w = 3000.0;
        let cooling_power =
            equipment.calculate_electrical_consumption(thermal_3000w, HVACMode::Cooling);
        let heating_power =
            equipment.calculate_electrical_consumption(thermal_3000w, HVACMode::Heating);

        assert!((cooling_power - 1000.0).abs() < 0.1);
        assert!((heating_power - 3333.33).abs() < 1.0);
    }

    #[test]
    fn test_zone_properties_affect_cooling_load() {
        // ASHRAE 140 Case 900: V=129.6 m³, ACH=0.5
        let load_standard =
            ZoneIdealLoads::calculate_sensible_cooling_load(28.0, 24.0, 13.0, 129.6, 0.5);

        // Larger zone with same ACH: V=300 m³
        let load_large_zone =
            ZoneIdealLoads::calculate_sensible_cooling_load(28.0, 24.0, 13.0, 300.0, 0.5);

        // Same zone volume but higher ACH: ACH=6
        let load_high_ach =
            ZoneIdealLoads::calculate_sensible_cooling_load(28.0, 24.0, 13.0, 129.6, 6.0);

        // Larger zone with higher ACH (old hardcoded values)
        let load_old_hardcoded =
            ZoneIdealLoads::calculate_sensible_cooling_load(28.0, 24.0, 13.0, 300.0, 6.0);

        // Verify that load scales with volume and ACH
        assert!(load_large_zone > load_standard);
        assert!(load_high_ach > load_standard);
        assert!(load_old_hardcoded > load_high_ach);

        // The ratio between old hardcoded and ASHRAE standard should be significant
        let ratio = load_old_hardcoded / load_standard;
        assert!(
            ratio > 10.0,
            "Old hardcoded values should produce >10x load due to wrong volume and ACH"
        );
    }

    #[test]
    fn test_ideal_loads_system_with_custom_zone_properties() {
        let mut system = IdealLoadsSystem::new(200.0, 1.0);
        assert_eq!(system.zone_volume, 200.0);
        assert_eq!(system.air_changes_per_hour, 1.0);

        let result = system.calculate(28.0, 20.0, 24.0);
        assert!(result.thermal_load_watts > 0.0);
    }
}
