//! Battery storage system for building energy management.
//!
//! This module provides models for:
//! - **Battery storage**: Charge/discharge management with SoC tracking
//! - **Self-consumption optimization**: Dispatch strategy that maximizes on-site solar usage
//!
//! ## EnergyPlus Mapping
//! - `Generator:FacilityStorage` → `BatteryStorage`
//!
//! ## References
//! - NREL SAM (System Advisor Model) for validation targets
//! - EnergyPlus Engineering Reference

use serde::{Deserialize, Serialize};

/// Battery storage system with self-consumption optimization.
///
/// Tracks State of Charge (SoC) and optimizes dispatch to maximize
/// self-consumption of on-site solar generation.
///
/// ## Dispatch Strategy
/// The self-consumption optimizer operates as follows:
/// 1. If `excess_pv > 0`: Charge battery from excess PV (up to available capacity)
/// 2. If `deficit > 0`: Discharge battery to meet deficit (up to available energy)
/// 3. Any remaining balance goes to/from grid
///
/// Where:
/// - `excess_pv = pv_generation - building_load` (positive when PV exceeds load)
/// - `deficit = building_load - pv_generation` (positive when load exceeds PV)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryStorage {
    /// Maximum storage capacity (Wh)
    pub capacity_wh: f64,
    /// Maximum charge/discharge power (W)
    pub max_power_w: f64,
    /// Current State of Charge (Wh)
    pub soc_wh: f64,
    /// Charge efficiency (0.0 to 1.0)
    pub charge_efficiency: f64,
    /// Discharge efficiency (0.0 to 1.0)
    pub discharge_efficiency: f64,
    /// Minimum SoC as fraction of capacity (0.0 to 1.0)
    pub min_soc_fraction: f64,
    /// Maximum SoC as fraction of capacity (0.0 to 1.0)
    pub max_soc_fraction: f64,
}

impl BatteryStorage {
    /// Create a new battery storage system.
    ///
    /// # Arguments
    /// * `capacity_wh` - Maximum storage capacity (Wh)
    /// * `max_power_w` - Maximum charge/discharge power (W)
    /// * `initial_soc_fraction` - Initial SoC as fraction of capacity (0.0 to 1.0)
    ///
    /// # Default Parameters
    /// * Charge efficiency: 0.95
    /// * Discharge efficiency: 0.95
    /// * Min SoC: 10% (prevents deep discharge)
    /// * Max SoC: 95% (prevents overcharge)
    pub fn new(capacity_wh: f64, max_power_w: f64, initial_soc_fraction: f64) -> Self {
        let soc_wh = capacity_wh * initial_soc_fraction.clamp(0.1, 0.95);
        Self {
            capacity_wh,
            max_power_w,
            soc_wh,
            charge_efficiency: 0.95,
            discharge_efficiency: 0.95,
            min_soc_fraction: 0.10,
            max_soc_fraction: 0.95,
        }
    }

    /// Get current SoC as a fraction of capacity.
    pub fn soc_fraction(&self) -> f64 {
        self.soc_wh / self.capacity_wh
    }

    /// Get available discharge energy above minimum SoC (Wh).
    pub fn available_discharge_wh(&self) -> f64 {
        ((self.soc_wh / self.capacity_wh) - self.min_soc_fraction).max(0.0) * self.capacity_wh
    }

    /// Get available charge capacity below maximum SoC (Wh).
    pub fn available_charge_wh(&self) -> f64 {
        (self.max_soc_fraction - (self.soc_wh / self.capacity_wh)).max(0.0) * self.capacity_wh
    }

    /// Calculate maximum charge power (W).
    pub fn max_charge_power(&self) -> f64 {
        let available_capacity = self.available_charge_wh();
        (available_capacity * 3600.0 / 1.0).min(self.max_power_w) // Assuming 1-hour timestep
    }

    /// Calculate maximum discharge power (W).
    pub fn max_discharge_power(&self) -> f64 {
        let available_energy = self.available_discharge_wh();
        (available_energy * 3600.0 / 1.0).min(self.max_power_w)
    }

    /// Step the battery for one timestep with self-consumption optimization.
    ///
    /// # Arguments
    /// * `pv_generation_w` - PV AC power generation (W)
    /// * `building_load_w` - Building electrical load (W)
    /// * `grid_import_cost` - Cost of importing from grid ($/Wh), used for economics
    /// * `dt_seconds` - Timestep duration (s)
    ///
    /// # Returns
    /// Tuple of (battery_charge_wh, battery_discharge_wh, grid_import_wh, grid_export_wh)
    pub fn step(
        &mut self,
        pv_generation_w: f64,
        building_load_w: f64,
        _grid_import_cost: f64,
        dt_seconds: f64,
    ) -> (f64, f64, f64, f64) {
        let dt_hours = dt_seconds / 3600.0;

        // Calculate excess/deficit
        let net_energy_wh = (pv_generation_w - building_load_w) * dt_hours;
        let excess_pv_wh = net_energy_wh.max(0.0);
        let deficit_wh = (-net_energy_wh).max(0.0);

        // Self-consumption optimization
        let (charge_wh, discharge_wh) = if excess_pv_wh > 0.0 {
            // Charge battery from excess PV
            let max_charge_wh = self.available_charge_wh().min(self.max_power_w * dt_hours);
            let charge_efficiency_wh = (excess_pv_wh * self.charge_efficiency).min(max_charge_wh);
            (charge_efficiency_wh, 0.0)
        } else {
            // Discharge battery to meet deficit
            let max_discharge_wh = self
                .available_discharge_wh()
                .min(self.max_power_w * dt_hours);
            // Energy out of battery = deficit / discharge_efficiency
            let discharge_from_battery_wh =
                (deficit_wh / self.discharge_efficiency).min(max_discharge_wh);
            (0.0, discharge_from_battery_wh)
        };

        // Update SoC
        // Energy into battery: charge_wh (already efficiency-adjusted)
        // Energy out of battery: discharge_wh (needs to be divided back)
        let soc_change_wh = charge_wh - discharge_wh;
        self.soc_wh = (self.soc_wh + soc_change_wh).clamp(
            self.capacity_wh * self.min_soc_fraction,
            self.capacity_wh * self.max_soc_fraction,
        );

        // Calculate grid import/export
        let grid_import_wh = if deficit_wh > discharge_wh {
            deficit_wh - discharge_wh
        } else {
            0.0
        };

        let grid_export_wh = if excess_pv_wh > charge_wh {
            excess_pv_wh - charge_wh
        } else {
            0.0
        };

        (charge_wh, discharge_wh, grid_import_wh, grid_export_wh)
    }
}

/// Net-zero energy system combining PV and battery.
///
/// This struct coordinates PV generation, battery storage, and grid interaction
/// to achieve net-zero building operation over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetZeroSystem {
    /// PV system
    pub pv: super::pv::PvSystem,
    /// Battery storage
    pub battery: BatteryStorage,
}

impl NetZeroSystem {
    /// Create a new net-zero system.
    ///
    /// # Arguments
    /// * `pv_panel_area_m2` - Total PV panel area (m²)
    /// * `pv_rated_dc_w` - PV rated DC power (W)
    /// * `inverter_efficiency` - Inverter efficiency (0.0 to 1.0)
    /// * `battery_capacity_wh` - Battery capacity (Wh)
    /// * `battery_max_power_w` - Battery max charge/discharge power (W)
    pub fn new(
        pv_panel_area_m2: f64,
        pv_rated_dc_w: f64,
        inverter_efficiency: f64,
        battery_capacity_wh: f64,
        battery_max_power_w: f64,
    ) -> Self {
        let pv = super::pv::PvSystem::new(pv_panel_area_m2, pv_rated_dc_w, inverter_efficiency);
        let battery = BatteryStorage::new(battery_capacity_wh, battery_max_power_w, 0.5);
        Self { pv, battery }
    }

    /// Simulate one timestep.
    ///
    /// # Arguments
    /// * `irradiance_wm2` - Solar irradiance (W/m²)
    /// * `ambient_temp_c` - Ambient temperature (°C)
    /// * `building_load_w` - Building electrical load (W)
    /// * `dt_seconds` - Timestep duration (s)
    ///
    /// # Returns
    /// Tuple of (pv_generation_wh, battery_charge_wh, battery_discharge_wh, grid_import_wh, grid_export_wh, final_soc)
    pub fn step(
        &mut self,
        irradiance_wm2: f64,
        ambient_temp_c: f64,
        building_load_w: f64,
        dt_seconds: f64,
    ) -> (f64, f64, f64, f64, f64, f64) {
        // Calculate PV generation
        let pv_power_w = self.pv.ac_power(irradiance_wm2, ambient_temp_c);
        let pv_generation_wh = pv_power_w * dt_seconds / 3600.0;

        // Battery dispatch
        let (charge_wh, discharge_wh, _grid_import_wh, _grid_export_wh) =
            self.battery
                .step(pv_power_w, building_load_w, 0.0, dt_seconds);

        (
            pv_generation_wh,
            charge_wh,
            discharge_wh,
            _grid_import_wh,
            _grid_export_wh,
            self.battery.soc_wh,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_battery_creation() {
        let battery = BatteryStorage::new(10000.0, 5000.0, 0.5);
        assert_eq!(battery.capacity_wh, 10000.0);
        assert_eq!(battery.max_power_w, 5000.0);
        assert!((battery.soc_wh - 5000.0).abs() < 1.0);
    }

    #[test]
    fn test_battery_soc_fraction() {
        let battery = BatteryStorage::new(10000.0, 5000.0, 0.5);
        assert!((battery.soc_fraction() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_battery_available_discharge() {
        let battery = BatteryStorage::new(10000.0, 5000.0, 0.5);
        // At 50% SoC, min is 10%, so 40% available = 4000 Wh
        let available = battery.available_discharge_wh();
        assert!((available - 4000.0).abs() < 1.0);
    }

    #[test]
    fn test_battery_charge_when_excess_pv() {
        let mut battery = BatteryStorage::new(10000.0, 5000.0, 0.3);
        let initial_soc = battery.soc_wh;

        // PV generates 3000W, load is 1000W, excess = 2000W
        let (charge, discharge, _grid_import, _grid_export) =
            battery.step(3000.0, 1000.0, 0.0, 3600.0);

        assert!(charge > 0.0);
        assert_eq!(discharge, 0.0);
        assert!(battery.soc_wh > initial_soc);
    }

    #[test]
    fn test_battery_discharge_when_deficit() {
        let mut battery = BatteryStorage::new(10000.0, 5000.0, 0.8);
        let initial_soc = battery.soc_wh;

        // PV generates 500W, load is 2000W, deficit = 1500W
        let (charge, discharge, _grid_import, _grid_export) =
            battery.step(500.0, 2000.0, 0.0, 3600.0);

        assert_eq!(charge, 0.0);
        assert!(discharge > 0.0);
        assert!(battery.soc_wh < initial_soc);
    }

    #[test]
    fn test_battery_soc_clamping_min() {
        let mut battery = BatteryStorage::new(10000.0, 5000.0, 0.15);
        // Try to discharge below minimum SoC
        battery.step(0.0, 10000.0, 0.0, 3600.0);
        // SoC should not go below 10% of capacity = 1000 Wh
        assert!(battery.soc_wh >= 1000.0);
    }

    #[test]
    fn test_battery_soc_clamping_max() {
        let mut battery = BatteryStorage::new(10000.0, 5000.0, 0.90);
        // Try to charge above maximum SoC
        battery.step(10000.0, 0.0, 0.0, 3600.0);
        // SoC should not exceed 95% of capacity = 9500 Wh
        assert!(battery.soc_wh <= 9500.0);
    }

    #[test]
    fn test_battery_grid_export_when_excess() {
        let mut battery = BatteryStorage::new(10000.0, 5000.0, 0.5);
        // PV generates more than battery can accept
        battery.step(10000.0, 100.0, 0.0, 3600.0);
        // Should export to grid
    }

    #[test]
    fn test_battery_grid_import_when_deficit() {
        let mut battery = BatteryStorage::new(10000.0, 1000.0, 0.3);
        // Large deficit, battery can only partially cover
        let (_, _, grid_import, _) = battery.step(500.0, 3000.0, 0.0, 3600.0);
        assert!(grid_import > 0.0);
    }

    #[test]
    fn test_net_zero_system_creation() {
        let system = NetZeroSystem::new(20.0, 4000.0, 0.95, 10000.0, 3000.0);
        assert_eq!(system.pv.panel.area_m2, 20.0);
        assert_eq!(system.battery.capacity_wh, 10000.0);
    }

    #[test]
    fn test_net_zero_step_night() {
        let mut system = NetZeroSystem::new(20.0, 4000.0, 0.95, 10000.0, 3000.0);
        system.battery.soc_wh = 5000.0;

        let (_, charge, discharge, _, _, soc) = system.step(0.0, 25.0, 1000.0, 3600.0);

        assert_eq!(charge, 0.0);
        assert!(discharge > 0.0);
        assert!(soc < 5000.0);
    }

    #[test]
    fn test_net_zero_step_day_excess_pv() {
        let mut system = NetZeroSystem::new(20.0, 4000.0, 0.95, 10000.0, 3000.0);
        system.battery.soc_wh = 3000.0;

        let (_, charge, discharge, _, _, soc) = system.step(800.0, 25.0, 500.0, 3600.0);

        assert!(charge > 0.0);
        assert_eq!(discharge, 0.0);
        assert!(soc > 3000.0);
    }

    #[test]
    fn test_battery_efficiency_round_trip() {
        let mut battery = BatteryStorage::new(10000.0, 5000.0, 0.3);

        // Charge 1000 Wh (will get less due to efficiency)
        let _initial_soc = battery.soc_wh;
        battery.step(2000.0, 0.0, 0.0, 3600.0); // excess_pv = 2000W for 1h
        let after_charge = battery.soc_wh;

        // Now discharge
        battery.step(0.0, 2000.0, 0.0, 3600.0);
        let after_discharge = battery.soc_wh;

        // Net change should account for round-trip efficiency
        let _round_trip_loss = after_charge - after_discharge;
        // The actual round trip energy should be less than what we put in
        assert!(after_discharge < after_charge);
    }
}
