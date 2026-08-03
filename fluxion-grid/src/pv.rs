//! Photovoltaic (PV) panel and inverter models for grid-edge systems.
//!
//! This module provides simplified PV models for integration with
//! battery storage and grid-export calculations.
//!
//! ## EnergyPlus Mapping
//! - `Generator:Photovoltaic` → `PvPanel`
//! - `Inverter:Simple` → `SimpleInverter`

use serde::{Deserialize, Serialize};

/// PV panel performance model for grid-edge systems.
///
/// Simplified model that converts solar irradiance to DC power.
/// Temperature derating uses a standard -0.5%/°C coefficient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PvPanel {
    /// Panel surface area (m²)
    pub area_m2: f64,
    /// Rated DC power output at STC (W)
    pub rated_power_w: f64,
    /// Panel efficiency at STC (0.0 to 1.0)
    pub efficiency_stc: f64,
    /// Temperature coefficient of power (fraction/°C)
    pub temperature_coefficient: f64,
    /// Reference irradiance for STC (W/m²)
    pub reference_irradiance: f64,
    /// Combined loss factor (soiling + mismatch + wiring) (0.0 to 1.0)
    pub combined_loss_factor: f64,
}

impl PvPanel {
    /// Create a new PV panel.
    ///
    /// # Arguments
    /// * `area_m2` - Panel surface area (m²)
    /// * `rated_power_w` - Rated DC power at STC (W)
    pub fn new(area_m2: f64, rated_power_w: f64) -> Self {
        let efficiency_stc = rated_power_w / (area_m2 * 1000.0);
        Self {
            area_m2,
            rated_power_w,
            efficiency_stc,
            temperature_coefficient: -0.004,
            reference_irradiance: 1000.0,
            combined_loss_factor: 0.94, // ~6% total losses
        }
    }

    /// Calculate cell temperature using NOCT model.
    ///
    /// # Arguments
    /// * `irradiance_wm2` - Irradiance (W/m²)
    /// * `ambient_temp_c` - Ambient temperature (°C)
    pub fn cell_temperature(&self, irradiance_wm2: f64, ambient_temp_c: f64) -> f64 {
        let noct = 45.0; // Standard NOCT
        ambient_temp_c + (noct - 20.0) * irradiance_wm2 / 800.0
    }

    /// Calculate DC power output.
    ///
    /// # Arguments
    /// * `irradiance_wm2` - Surface irradiance (W/m²)
    /// * `cell_temp_c` - Cell temperature (°C)
    pub fn dc_power(&self, irradiance_wm2: f64, cell_temp_c: f64) -> f64 {
        if irradiance_wm2 <= 0.0 {
            return 0.0;
        }

        let temp_derating = if cell_temp_c > 25.0 {
            let delta_t = cell_temp_c - 25.0;
            (1.0 + self.temperature_coefficient * delta_t).max(0.0)
        } else {
            1.0
        };

        let power = irradiance_wm2
            * self.area_m2
            * self.efficiency_stc
            * temp_derating
            * self.combined_loss_factor;

        power.min(self.rated_power_w).max(0.0)
    }
}

/// Simple inverter with constant efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleInverter {
    /// Rated AC power output (W)
    pub rated_power_w: f64,
    /// Nominal efficiency (0.0 to 1.0)
    pub efficiency: f64,
}

impl SimpleInverter {
    /// Create a new simple inverter.
    ///
    /// # Arguments
    /// * `rated_power_w` - Rated AC power (W)
    /// * `efficiency` - Nominal efficiency (0.0 to 1.0)
    pub fn new(rated_power_w: f64, efficiency: f64) -> Self {
        Self {
            rated_power_w,
            efficiency,
        }
    }

    /// Calculate AC power output from DC input.
    ///
    /// # Arguments
    /// * `dc_power_w` - DC power input (W)
    pub fn ac_power(&self, dc_power_w: f64) -> f64 {
        if dc_power_w <= 0.0 {
            return 0.0;
        }
        (dc_power_w * self.efficiency).min(self.rated_power_w)
    }
}

/// Combined PV system with inverter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PvSystem {
    /// PV panel
    pub panel: PvPanel,
    /// Inverter
    pub inverter: SimpleInverter,
}

impl PvSystem {
    /// Create a new PV system.
    ///
    /// # Arguments
    /// * `panel_area_m2` - Total panel area (m²)
    /// * `rated_dc_power_w` - Rated DC power (W)
    /// * `inverter_efficiency` - Inverter efficiency (0.0 to 1.0)
    pub fn new(panel_area_m2: f64, rated_dc_power_w: f64, inverter_efficiency: f64) -> Self {
        let panel = PvPanel::new(panel_area_m2, rated_dc_power_w);
        let inverter = SimpleInverter::new(rated_dc_power_w, inverter_efficiency);
        Self { panel, inverter }
    }

    /// Calculate AC power output.
    ///
    /// # Arguments
    /// * `irradiance_wm2` - Surface irradiance (W/m²)
    /// * `ambient_temp_c` - Ambient temperature (°C)
    pub fn ac_power(&self, irradiance_wm2: f64, ambient_temp_c: f64) -> f64 {
        let cell_temp = self.panel.cell_temperature(irradiance_wm2, ambient_temp_c);
        let dc_power = self.panel.dc_power(irradiance_wm2, cell_temp);
        self.inverter.ac_power(dc_power)
    }

    /// Calculate AC energy output for a timestep.
    ///
    /// # Arguments
    /// * `irradiance_wm2` - Average irradiance (W/m²)
    /// * `ambient_temp_c` - Average ambient temperature (°C)
    /// * `dt_seconds` - Timestep duration (s)
    pub fn ac_energy(&self, irradiance_wm2: f64, ambient_temp_c: f64, dt_seconds: f64) -> f64 {
        let power = self.ac_power(irradiance_wm2, ambient_temp_c);
        power * dt_seconds / 3600.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pv_panel_stc() {
        let panel = PvPanel::new(10.0, 2000.0);
        let power = panel.dc_power(1000.0, 25.0);
        // Expected: 1000 * 10 * 0.2 * 0.94 = 1880 W (before clipping)
        assert!(power > 1800.0 && power <= 2000.0);
    }

    #[test]
    fn test_pv_panel_zero_irradiance() {
        let panel = PvPanel::new(10.0, 2000.0);
        assert_eq!(panel.dc_power(0.0, 25.0), 0.0);
    }

    #[test]
    fn test_inverter() {
        let inverter = SimpleInverter::new(2000.0, 0.95);
        let ac = inverter.ac_power(2000.0);
        assert!((ac - 1900.0).abs() < 1.0);
    }

    #[test]
    fn test_pv_system_ac_power() {
        let system = PvSystem::new(10.0, 2000.0, 0.95);
        let ac = system.ac_power(1000.0, 25.0);
        // At 1000 W/m² with ambient 25°C, NOCT model gives cell temp ~56°C
        // Temperature derating: 1 + (-0.004) * (56-25) = 0.875
        // DC: 1000 * 10 * 0.2 * 0.875 * 0.94 = 1645 W
        // AC: 1645 * 0.95 = 1563 W
        assert!(ac > 1500.0 && ac < 1700.0);
    }
}
