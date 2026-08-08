//! Domestic Hot Water (DHW) Mixed-Tank Model
//!
//! This module provides the `DHWTank` struct for modeling domestic hot water systems
//! with standby losses and draw-based heating energy consumption.
//!
//! # Model Description
//!
//! DHW systems consume electricity or gas to heat water and lose heat through tank walls
//! (standby losses) into the mechanical room. The draw profile drives the heating load.
//!
//! # Energy Calculation
//!
//! Water heating energy: `draw_L * 4.18 kJ/(kg·K) * (T_draw - T_supply) / 3600` kWh
//! Standby loss: `standby_loss_W * dt_hours` always injected as internal gain to zone
//!
//! # References
//!
//! - ASHRAE Handbook - HVAC Systems and Equipment (Ch. 50: Water Heating)
//! - EnergyPlus Engineering Reference: Water Heaters

use serde::{Deserialize, Serialize};

use crate::sim::schedule::DailySchedule;

/// Heating source for DHW tank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeatingSource {
    /// Electric resistance heating.
    Electric,
    /// Gas-fired heating.
    Gas,
}

impl Default for HeatingSource {
    fn default() -> Self {
        HeatingSource::Electric
    }
}

/// Result of a DHW tank step, representing energy consumption and heat losses.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DHWResult {
    /// Water heating energy consumed (kWh) for this timestep.
    pub heating_energy_kwh: f64,
    /// Standby heat loss injected to the tank location zone (W).
    pub standby_loss_w: f64,
    /// Volume of hot water drawn (L) for this timestep.
    pub draw_liters: f64,
    /// Hot water supply temperature (°C).
    pub supply_temp_c: f64,
    /// Total DHW energy consumed to date (kWh).
    pub total_dhw_energy_kwh: f64,
}

impl Default for DHWResult {
    fn default() -> Self {
        Self {
            heating_energy_kwh: 0.0,
            standby_loss_w: 0.0,
            draw_liters: 0.0,
            supply_temp_c: 10.0,
            total_dhw_energy_kwh: 0.0,
        }
    }
}

/// Domestic Hot Water tank model with standby losses and draw-based heating.
///
/// The DHWTank model calculates:
/// - Water heating energy based on draw volume and temperature rise
/// - Standby heat losses injected into the zone where the tank is located
///
/// # Energy Balance
///
/// For each timestep:
/// - Heating energy: `draw_L * 4.18 kJ/(kg·K) * (60°C - supply_temp) / 3600` kWh
/// - Standby loss: `standby_loss_W * dt_hours` W continuously injected to zone
///
/// # Example
///
/// ```
/// use fluxion::sim::hvac::dhw::{DHWTank, HeatingSource};
/// use fluxion::sim::schedule::DailySchedule;
///
/// let mut tank = DHWTank::new(
///     "DHW-1".to_string(),
///     200.0,  // 200L tank
///     60.0,   // 60°C setpoint
///     50.0,   // 50W standby loss
///     0,      // zone 0
///     HeatingSource::Electric,
/// );
///
/// let result = tank.step(10.0, 3600.0);
/// assert!(result.standby_loss_w > 0.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DHWTank {
    /// Tank identifier.
    pub id: String,
    /// Tank volume (L).
    pub volume_L: f64,
    /// Hot water setpoint temperature (°C).
    pub setpoint_C: f64,
    /// Standby heat loss rate (W).
    pub standby_loss_W: f64,
    /// Zone ID where the tank is located (for standby loss injection).
    pub tank_location_zone_id: usize,
    /// Heating source (electric or gas).
    pub heating_source: HeatingSource,
    /// Draw profile schedule (L/hr of hot water at 60°C).
    pub draw_profile: DailySchedule,
    /// Supply water temperature (°C) - cold water entering the tank.
    pub supply_temp_C: f64,
    /// Accumulated total DHW energy (kWh).
    total_dhw_energy_kwh: f64,
    /// Water density (kg/L).
    water_density_kg_per_L: f64,
    /// Specific heat of water (kJ/kg·K).
    water_cp_kj_per_kg_K: f64,
}

impl DHWTank {
    /// Create a new DHW tank.
    ///
    /// # Arguments
    ///
    /// * `id` - Tank identifier
    /// * `volume_L` - Tank volume in liters
    /// * `setpoint_C` - Hot water setpoint temperature in °C
    /// * `standby_loss_W` - Standby heat loss rate in watts
    /// * `tank_location_zone_id` - Zone ID where tank is located
    /// * `heating_source` - Heating source (Electric or Gas)
    ///
    /// # Example
    ///
    /// ```
    /// use fluxion::sim::hvac::dhw::{DHWTank, HeatingSource};
    ///
    /// let tank = DHWTank::new(
    ///     "DHW-1".to_string(),
    ///     200.0,
    ///     60.0,
    ///     50.0,
    ///     0,
    ///     HeatingSource::Electric,
    /// );
    /// ```
    pub fn new(
        id: String,
        volume_L: f64,
        setpoint_C: f64,
        standby_loss_W: f64,
        tank_location_zone_id: usize,
        heating_source: HeatingSource,
    ) -> Self {
        Self {
            id,
            volume_L,
            setpoint_C,
            standby_loss_W,
            tank_location_zone_id,
            heating_source,
            draw_profile: DailySchedule::new(),
            supply_temp_C: 10.0,
            total_dhw_energy_kwh: 0.0,
            water_density_kg_per_L: 1.0,
            water_cp_kj_per_kg_K: 4.186,
        }
    }

    /// Create a new DHW tank with a custom draw profile.
    ///
    /// # Arguments
    ///
    /// * `id` - Tank identifier
    /// * `volume_L` - Tank volume in liters
    /// * `setpoint_C` - Hot water setpoint temperature in °C
    /// * `standby_loss_W` - Standby heat loss rate in watts
    /// * `tank_location_zone_id` - Zone ID where tank is located
    /// * `heating_source` - Heating source (Electric or Gas)
    /// * `draw_profile` - Daily schedule of hot water draw (L/hr)
    ///
    pub fn with_draw_profile(
        id: String,
        volume_L: f64,
        setpoint_C: f64,
        standby_loss_W: f64,
        tank_location_zone_id: usize,
        heating_source: HeatingSource,
        draw_profile: DailySchedule,
    ) -> Self {
        Self {
            id,
            volume_L,
            setpoint_C,
            standby_loss_W,
            tank_location_zone_id,
            heating_source,
            draw_profile,
            supply_temp_C: 10.0,
            total_dhw_energy_kwh: 0.0,
            water_density_kg_per_L: 1.0,
            water_cp_kj_per_kg_K: 4.186,
        }
    }

    /// Set the supply water temperature (cold water entering the tank).
    pub fn with_supply_temp(mut self, supply_temp_C: f64) -> Self {
        self.supply_temp_C = supply_temp_C;
        self
    }

    /// Execute one simulation timestep for the DHW tank.
    ///
    /// # Arguments
    ///
    /// * `hour` - Current hour of the day (0-23)
    /// * `dt` - Timestep duration in seconds
    ///
    /// # Returns
    ///
    /// `DHWResult` containing heating energy, standby loss, and draw volume
    ///
    /// # Energy Calculation
    ///
    /// The heating energy is calculated as:
    /// `draw_L * water_density * water_cp * (setpoint_C - supply_temp_C) / 3600` kWh
    ///
    /// Where:
    /// - `draw_L` is the hot water draw in liters for this timestep
    /// - `water_density` is approximately 1 kg/L
    /// - `water_cp` is 4.186 kJ/(kg·K)
    /// - Temperature difference is (setpoint_C - supply_temp_C)
    /// - Division by 3600 converts from kJ to kWh
    ///
    /// Standby loss is calculated as:
    /// `standby_loss_W * (dt / 3600)` kWh → reported as W (instantaneous)
    ///
    pub fn step(&mut self, hour: usize, dt: f64) -> DHWResult {
        let dt_hours = dt / 3600.0;

        let draw_rate_L_hr = self.draw_profile.value(hour);

        let draw_L = draw_rate_L_hr * dt_hours;

        let temp_rise_K = (self.setpoint_C - self.supply_temp_C).max(0.0);

        let heating_energy_kwh =
            draw_L * self.water_density_kg_per_L * self.water_cp_kj_per_kg_K * temp_rise_K / 3600.0;

        self.total_dhw_energy_kwh += heating_energy_kwh;

        DHWResult {
            heating_energy_kwh,
            standby_loss_w: self.standby_loss_W,
            draw_liters: draw_L,
            supply_temp_c: self.supply_temp_C,
            total_dhw_energy_kwh: self.total_dhw_energy_kwh,
        }
    }

    /// Execute one simulation timestep using day type.
    ///
    /// # Arguments
    ///
    /// * `day_type` - Day type for schedule lookup
    /// * `hour` - Current hour of the day (0-23)
    /// * `dt` - Timestep duration in seconds
    ///
    pub fn step_for_day(
        &mut self,
        day_type: crate::sim::schedule::DayType,
        hour: usize,
        dt: f64,
    ) -> DHWResult {
        let dt_hours = dt / 3600.0;

        let draw_rate_L_hr = self.draw_profile.value_for_day(day_type, hour);

        let draw_L = draw_rate_L_hr * dt_hours;

        let temp_rise_K = (self.setpoint_C - self.supply_temp_C).max(0.0);

        let heating_energy_kwh =
            draw_L * self.water_density_kg_per_L * self.water_cp_kj_per_kg_K * temp_rise_K / 3600.0;

        self.total_dhw_energy_kwh += heating_energy_kwh;

        DHWResult {
            heating_energy_kwh,
            standby_loss_w: self.standby_loss_W,
            draw_liters: draw_L,
            supply_temp_c: self.supply_temp_C,
            total_dhw_energy_kwh: self.total_dhw_energy_kwh,
        }
    }

    /// Get the total DHW energy consumed to date (kWh).
    pub fn total_dhw_energy(&self) -> f64 {
        self.total_dhw_energy_kwh
    }

    /// Reset the DHW tank to its initial state.
    pub fn reset(&mut self) {
        self.total_dhw_energy_kwh = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dhw_tank_creation() {
        let tank = DHWTank::new(
            "DHW-1".to_string(),
            200.0,
            60.0,
            50.0,
            0,
            HeatingSource::Electric,
        );

        assert_eq!(tank.id, "DHW-1");
        assert_eq!(tank.volume_L, 200.0);
        assert_eq!(tank.setpoint_C, 60.0);
        assert_eq!(tank.standby_loss_W, 50.0);
        assert_eq!(tank.tank_location_zone_id, 0);
        assert_eq!(tank.heating_source, HeatingSource::Electric);
    }

    #[test]
    fn test_dhw_tank_standby_loss() {
        let mut tank = DHWTank::new(
            "DHW-1".to_string(),
            200.0,
            60.0,
            50.0,
            0,
            HeatingSource::Electric,
        );

        let result = tank.step(12, 3600.0);

        assert_eq!(
            result.standby_loss_w, 50.0,
            "200L tank with 50W standby loss should have 50W continuous zone gain"
        );
    }

    #[test]
    fn test_dhw_tank_heating_energy() {
        let mut tank = DHWTank::new(
            "DHW-1".to_string(),
            100.0,
            60.0,
            0.0,
            0,
            HeatingSource::Electric,
        );
        tank.supply_temp_C = 10.0;

        let mut draw_profile = DailySchedule::new();
        draw_profile.fill_range(0, 24, 100.0);
        tank.draw_profile = draw_profile;

        let result = tank.step(12, 3600.0);

        let expected_kwh =
            100.0 * 1.0 * 4.186 * (60.0 - 10.0) / 1.0;

        assert!(
            (result.heating_energy_kwh - expected_kwh).abs() < 0.01,
            "100L draw at 60°C with 10°C supply water should be ~{:.1} kWh, got {:.2}",
            expected_kwh,
            result.heating_energy_kwh
        );
    }

    #[test]
    fn test_dhw_tank_100l_draw_approx_5_8_kwh() {
        let mut tank = DHWTank::new(
            "DHW-1".to_string(),
            200.0,
            60.0,
            0.0,
            0,
            HeatingSource::Electric,
        );
        tank.supply_temp_C = 10.0;

        let mut draw_profile = DailySchedule::new();
        draw_profile.fill_range(0, 24, 100.0);
        tank.draw_profile = draw_profile;

        let result = tank.step(12, 3600.0);

        let expected = 5.8;
        assert!(
            (result.heating_energy_kwh - expected).abs() < 0.2,
            "100L draw at 60°C with 10°C supply water should be ~5.8 kWh, got {:.2}",
            result.heating_energy_kwh
        );
    }

    #[test]
    fn test_dhw_tank_total_energy_accumulation() {
        let mut tank = DHWTank::new(
            "DHW-1".to_string(),
            200.0,
            60.0,
            0.0,
            0,
            HeatingSource::Electric,
        );
        tank.supply_temp_C = 10.0;

        let mut draw_profile = DailySchedule::new();
        draw_profile.fill_range(0, 24, 100.0);
        tank.draw_profile = draw_profile;

        tank.step(12, 3600.0);
        let result2 = tank.step(13, 3600.0);

        assert!(
            result2.total_dhw_energy_kwh > tank.step(12, 3600.0).heating_energy_kwh,
            "Total DHW energy should accumulate across timesteps"
        );
    }

    #[test]
    fn test_dhw_tank_gas_heating_source() {
        let tank = DHWTank::new(
            "DHW-Gas-1".to_string(),
            200.0,
            60.0,
            50.0,
            0,
            HeatingSource::Gas,
        );

        assert_eq!(tank.heating_source, HeatingSource::Gas);
    }

    #[test]
    fn test_dhw_tank_reset() {
        let mut tank = DHWTank::new(
            "DHW-1".to_string(),
            200.0,
            60.0,
            0.0,
            0,
            HeatingSource::Electric,
        );
        tank.supply_temp_C = 10.0;

        let mut draw_profile = DailySchedule::new();
        draw_profile.fill_range(0, 24, 100.0);
        tank.draw_profile = draw_profile;

        tank.step(12, 3600.0);
        assert!(tank.total_dhw_energy_kwh > 0.0);

        tank.reset();
        assert_eq!(tank.total_dhw_energy_kwh, 0.0);
    }

    #[test]
    fn test_dhw_tank_with_supply_temp() {
        let tank = DHWTank::new(
            "DHW-1".to_string(),
            200.0,
            60.0,
            50.0,
            0,
            HeatingSource::Electric,
        )
        .with_supply_temp(15.0);

        assert_eq!(tank.supply_temp_C, 15.0);
    }

    #[test]
    fn test_dhw_tank_draw_profile_schedule() {
        let mut tank = DHWTank::new(
            "DHW-1".to_string(),
            200.0,
            60.0,
            0.0,
            0,
            HeatingSource::Electric,
        );

        let mut draw_profile = DailySchedule::new();
        draw_profile.set_hour(7, 50.0);
        draw_profile.set_hour(8, 100.0);
        draw_profile.set_hour(9, 80.0);
        tank.draw_profile = draw_profile;

        assert_eq!(tank.draw_profile.value(7), 50.0);
        assert_eq!(tank.draw_profile.value(8), 100.0);
        assert_eq!(tank.draw_profile.value(9), 80.0);
    }

    #[test]
    fn test_dhw_result_default() {
        let result = DHWResult::default();
        assert_eq!(result.heating_energy_kwh, 0.0);
        assert_eq!(result.standby_loss_w, 0.0);
        assert_eq!(result.draw_liters, 0.0);
    }

    #[test]
    fn test_heating_source_default() {
        assert_eq!(HeatingSource::default(), HeatingSource::Electric);
    }
}
