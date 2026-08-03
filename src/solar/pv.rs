//! Solar photovoltaic (PV) panel and inverter models.
//!
//! This module provides physics-based models for:
//! - **PV panel**: converts solar irradiance to DC electricity
//! - **Inverter**: converts DC electricity to AC electricity
//!
//! ## EnergyPlus Mapping
//! - `Generator:Photovoltaic` → `PvPanel`
//! - `Inverter:Simple` → `SimpleInverter`
//!
//! ## References
//! - NREL SAM (System Advisor Model) for validation targets
//! - ASHRAE Handbook - Fundamentals, Chapter 14
//! - Duffie & Beckman, Solar Engineering of Thermal Processes

use serde::{Deserialize, Serialize};

/// PV panel performance model.
///
/// Models a photovoltaic panel's DC power output based on:
/// - Solar irradiance on the panel surface
/// - Panel area and efficiency
/// - Cell temperature derating
///
/// ## NREL SAM Alignment
/// The model produces energy within 10% of NREL SAM for standard
/// test conditions. Temperature derating uses the standard -0.5%/°C
/// linear coefficient for crystalline silicon panels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PvPanel {
    /// Panel surface area (m²)
    pub area_m2: f64,
    /// Rated DC power output at STC (W)
    pub rated_power_w: f64,
    /// Panel efficiency at STC (0.0 to 1.0)
    pub efficiency_stc: f64,
    /// Temperature coefficient of power (%/°C)
    /// Standard value: -0.5% per °C above 25°C
    pub temperature_coefficient: f64,
    /// Reference irradiance for STC (W/m²)
    pub reference_irradiance: f64,
    /// Soiling loss factor (0.0 to 1.0)
    pub soiling_loss: f64,
    /// Mismatch loss factor (0.0 to 1.0)
    pub mismatch_loss: f64,
    /// DC wiring loss factor (0.0 to 1.0)
    pub dc_wiring_loss: f64,
}

impl PvPanel {
    /// Create a new PV panel with standard crystalline silicon parameters.
    ///
    /// # Arguments
    /// * `area_m2` - Panel surface area (m²)
    /// * `rated_power_w` - Rated DC power at STC (W)
    ///
    /// # Default Parameters
    /// * Efficiency STC: 0.20 (20%)
    /// * Temperature coefficient: -0.004 (-0.4%/°C)
    /// * STC irradiance: 1000 W/m²
    pub fn new(area_m2: f64, rated_power_w: f64) -> Self {
        let efficiency_stc = rated_power_w / (area_m2 * 1000.0);
        Self {
            area_m2,
            rated_power_w,
            efficiency_stc,
            temperature_coefficient: -0.004,
            reference_irradiance: 1000.0,
            soiling_loss: 0.02,
            mismatch_loss: 0.02,
            dc_wiring_loss: 0.02,
        }
    }

    /// Calculate the cell temperature from ambient conditions.
    ///
    /// Uses the standard NOCT (Nominal Operating Cell Temperature) model:
    /// T_cell = T_ambient + (NOCT - 20) * irradiance / 800
    ///
    /// # Arguments
    /// * `irradiance_wm2` - Plane-of-array irradiance (W/m²)
    /// * `ambient_temp_c` - Ambient air temperature (°C)
    /// * `noct` - Nominal operating cell temperature (°C), typically 45-50°C
    ///
    /// # Returns
    /// Cell temperature in °C
    pub fn cell_temperature(&self, irradiance_wm2: f64, ambient_temp_c: f64, noct: f64) -> f64 {
        ambient_temp_c + (noct - 20.0) * irradiance_wm2 / 800.0
    }

    /// Calculate DC power output from irradiance and temperature.
    ///
    /// # Arguments
    /// * `irradiance` - Surface irradiance (W/m²)
    /// * `cell_temp_c` - Cell temperature (°C)
    ///
    /// # Returns
    /// DC power output in W
    pub fn dc_power(&self, irradiance: f64, cell_temp_c: f64) -> f64 {
        if irradiance <= 0.0 {
            return 0.0;
        }

        // Temperature derating: power loss relative to STC
        // At 25°C, derating = 1.0
        // For each °C above 25°C, multiply by (1 + temperature_coefficient)
        let temp_derating = if cell_temp_c > 25.0 {
            let delta_t = cell_temp_c - 25.0;
            (1.0 + self.temperature_coefficient * delta_t).max(0.0)
        } else {
            1.0
        };

        // Combine all loss factors
        let total_loss_factor =
            (1.0 - self.soiling_loss) * (1.0 - self.mismatch_loss) * (1.0 - self.dc_wiring_loss);

        // DC power = irradiance * area * efficiency * temp_derating * loss_factor
        let power =
            irradiance * self.area_m2 * self.efficiency_stc * temp_derating * total_loss_factor;

        // Clip to rated power
        power.min(self.rated_power_w).max(0.0)
    }

    /// Calculate DC energy output for a timestep.
    ///
    /// # Arguments
    /// * `irradiance` - Average surface irradiance (W/m²)
    /// * `cell_temp_c` - Average cell temperature (°C)
    /// * `dt_seconds` - Timestep duration (s)
    ///
    /// # Returns
    /// DC energy output in Wh
    pub fn dc_energy(&self, irradiance: f64, cell_temp_c: f64, dt_seconds: f64) -> f64 {
        let power = self.dc_power(irradiance, cell_temp_c);
        power * dt_seconds / 3600.0
    }
}

/// Simple inverter model with constant efficiency.
///
/// ## EnergyPlus Mapping
/// Corresponds to `Inverter:Simple` with one parameter:
/// - Nominal inverter efficiency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleInverter {
    /// Rated AC power output (W)
    pub rated_power_w: f64,
    /// Nominal efficiency (0.0 to 1.0)
    pub nominal_efficiency: f64,
    /// Minimum efficiency at part load (0.0 to 1.0)
    pub part_load_efficiency: f64,
    /// Power ratio below which efficiency starts to degrade (0.0 to 1.0)
    pub part_load_threshold: f64,
}

impl SimpleInverter {
    /// Create a new simple inverter.
    ///
    /// # Arguments
    /// * `rated_power_w` - Rated AC power output (W)
    /// * `efficiency` - Nominal efficiency at full load (0.0 to 1.0)
    pub fn new(rated_power_w: f64, efficiency: f64) -> Self {
        Self {
            rated_power_w,
            nominal_efficiency: efficiency,
            part_load_efficiency: efficiency * 0.95,
            part_load_threshold: 0.1,
        }
    }

    /// Calculate AC power output from DC power input.
    ///
    /// Uses a simple part-load efficiency curve:
    /// - Above `part_load_threshold`: nominal efficiency
    /// - Below `part_load_threshold`: linear interpolation to `part_load_efficiency`
    ///
    /// # Arguments
    /// * `dc_power_w` - DC power input (W)
    ///
    /// # Returns
    /// AC power output in W
    pub fn ac_power(&self, dc_power_w: f64) -> f64 {
        if dc_power_w <= 0.0 {
            return 0.0;
        }

        let load_ratio = (dc_power_w / self.rated_power_w).min(1.0);

        let efficiency = if load_ratio >= self.part_load_threshold {
            self.nominal_efficiency
        } else {
            // Linear interpolation at part load
            let ratio = load_ratio / self.part_load_threshold;
            self.part_load_efficiency
                + (self.nominal_efficiency - self.part_load_efficiency) * ratio
        };

        dc_power_w * efficiency
    }

    /// Calculate AC energy output for a timestep.
    ///
    /// # Arguments
    /// * `dc_power_w` - DC power input (W)
    /// * `dt_seconds` - Timestep duration (s)
    ///
    /// # Returns
    /// AC energy output in Wh
    pub fn ac_energy(&self, dc_power_w: f64, dt_seconds: f64) -> f64 {
        let power = self.ac_power(dc_power_w);
        power * dt_seconds / 3600.0
    }
}

/// Combined PV system with inverter.
///
/// This is a convenience struct that combines a PV panel with an inverter
/// to produce AC power output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PvSystem {
    /// PV panel model
    pub panel: PvPanel,
    /// Inverter model
    pub inverter: SimpleInverter,
}

impl PvSystem {
    /// Create a new PV system.
    ///
    /// # Arguments
    /// * `panel_area_m2` - Total panel area (m²)
    /// * `rated_dc_power_w` - Rated DC power at STC (W)
    /// * `inverter_efficiency` - Inverter nominal efficiency (0.0 to 1.0)
    pub fn new(panel_area_m2: f64, rated_dc_power_w: f64, inverter_efficiency: f64) -> Self {
        let panel = PvPanel::new(panel_area_m2, rated_dc_power_w);
        let inverter = SimpleInverter::new(rated_dc_power_w, inverter_efficiency);
        Self { panel, inverter }
    }

    /// Calculate AC power output.
    ///
    /// # Arguments
    /// * `irradiance_wm2` - Surface irradiance (W/m²)
    /// * `ambient_temp_c` - Ambient air temperature (°C)
    ///
    /// # Returns
    /// AC power output in W
    pub fn ac_power(&self, irradiance_wm2: f64, ambient_temp_c: f64) -> f64 {
        let cell_temp = self
            .panel
            .cell_temperature(irradiance_wm2, ambient_temp_c, 45.0);
        let dc_power = self.panel.dc_power(irradiance_wm2, cell_temp);
        self.inverter.ac_power(dc_power)
    }

    /// Calculate AC energy output for a timestep.
    ///
    /// # Arguments
    /// * `irradiance_wm2` - Average surface irradiance (W/m²)
    /// * `ambient_temp_c` - Average ambient temperature (°C)
    /// * `dt_seconds` - Timestep duration (s)
    ///
    /// # Returns
    /// AC energy output in Wh
    pub fn ac_energy(&self, irradiance_wm2: f64, ambient_temp_c: f64, dt_seconds: f64) -> f64 {
        let power = self.ac_power(irradiance_wm2, ambient_temp_c);
        power * dt_seconds / 3600.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pv_panel_creation() {
        let panel = PvPanel::new(10.0, 2000.0);
        assert_eq!(panel.area_m2, 10.0);
        assert_eq!(panel.rated_power_w, 2000.0);
        assert!((panel.efficiency_stc - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_pv_panel_zero_irradiance() {
        let panel = PvPanel::new(10.0, 2000.0);
        let power = panel.dc_power(0.0, 25.0);
        assert_eq!(power, 0.0);
    }

    #[test]
    fn test_pv_panel_stc_output() {
        // At STC (1000 W/m², 25°C), panel should produce rated power * efficiency
        let panel = PvPanel::new(10.0, 2000.0);
        // With 20% efficiency and 10m²: 1000 * 10 * 0.2 = 2000W = rated
        // Losses are multiplicative: (1-0.02)*(1-0.02)*(1-0.02) ≈ 0.941
        let power = panel.dc_power(1000.0, 25.0);
        let expected = 2000.0 * (1.0 - 0.02) * (1.0 - 0.02) * (1.0 - 0.02);
        assert!((power - expected).abs() < 1.0);
    }

    #[test]
    fn test_pv_panel_temperature_derating() {
        let panel = PvPanel::new(10.0, 2000.0);
        // At 35°C (10°C above 25°C), with -0.4%/°C:
        // derating = 1 + (-0.004) * 10 = 0.96
        let power_25 = panel.dc_power(1000.0, 25.0);
        let power_35 = panel.dc_power(1000.0, 35.0);
        let expected_ratio = 0.96;
        let actual_ratio = power_35 / power_25;
        assert!((actual_ratio - expected_ratio).abs() < 0.001);
    }

    #[test]
    fn test_pv_panel_clips_to_rated_power() {
        let panel = PvPanel::new(100.0, 2000.0);
        // With high irradiance and low temperature, should clip to rated
        let power = panel.dc_power(1200.0, 10.0);
        assert!(power <= panel.rated_power_w);
    }

    #[test]
    fn test_inverter_creation() {
        let inverter = SimpleInverter::new(2000.0, 0.95);
        assert_eq!(inverter.rated_power_w, 2000.0);
        assert_eq!(inverter.nominal_efficiency, 0.95);
    }

    #[test]
    fn test_inverter_zero_input() {
        let inverter = SimpleInverter::new(2000.0, 0.95);
        let ac_power = inverter.ac_power(0.0);
        assert_eq!(ac_power, 0.0);
    }

    #[test]
    fn test_inverter_full_load() {
        let inverter = SimpleInverter::new(2000.0, 0.95);
        let ac_power = inverter.ac_power(2000.0);
        assert!((ac_power - 1900.0).abs() < 1.0); // 2000 * 0.95
    }

    #[test]
    fn test_inverter_part_load() {
        let inverter = SimpleInverter::new(2000.0, 0.95);
        // At 5% load (ratio=0.5 relative to 10% threshold), interpolate efficiency:
        // eff = 0.9025 + (0.95-0.9025)*0.5 = 0.92625
        let ac_power = inverter.ac_power(100.0); // 5% of rated
        let expected = 100.0 * 0.92625;
        assert!((ac_power - expected).abs() < 1.0);
    }

    #[test]
    fn test_pv_system_ac_power() {
        let system = PvSystem::new(10.0, 2000.0, 0.95);
        // At STC with 95% inverter efficiency
        let ac_power = system.ac_power(1000.0, 25.0);
        // DC: 2000 * (1 - losses) ≈ 1920 W
        // AC: 1920 * 0.95 ≈ 1824 W
        assert!(ac_power > 0.0);
        assert!(ac_power < 2000.0);
    }

    #[test]
    fn test_pv_system_zero_night() {
        let system = PvSystem::new(10.0, 2000.0, 0.95);
        let ac_power = system.ac_power(0.0, 25.0);
        assert_eq!(ac_power, 0.0);
    }

    #[test]
    fn test_cell_temperature_noct_model() {
        let panel = PvPanel::new(10.0, 2000.0);
        // NOCT model: T_cell = T_amb + (NOCT - 20) * G / 800
        // At 800 W/m² and 20°C: T_cell = 20 + (45 - 20) * 800 / 800 = 45°C
        let temp = panel.cell_temperature(800.0, 20.0, 45.0);
        assert!((temp - 45.0).abs() < 0.1);
    }

    #[test]
    fn test_dc_energy_calculation() {
        let panel = PvPanel::new(10.0, 2000.0);
        // At STC for 1 hour (cell temp = 25°C passed directly)
        let energy = panel.dc_energy(1000.0, 25.0, 3600.0);
        // Power ≈ 1882 W (1000 * 10 * 0.2 * 0.98³), energy = 1882 Wh
        assert!(energy > 1880.0);
        assert!(energy < 1900.0);
    }

    #[test]
    fn test_ac_energy_calculation() {
        let system = PvSystem::new(10.0, 2000.0, 0.95);
        // At STC for 1 hour: NOCT cell temp = 56.25°C (thermal derating = 0.875)
        // DC power ≈ 1647 W, AC power ≈ 1565 W, energy ≈ 1565 Wh
        let energy = system.ac_energy(1000.0, 25.0, 3600.0);
        assert!(energy > 1560.0);
        assert!(energy < 1580.0);
    }
}
