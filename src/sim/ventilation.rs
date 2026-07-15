//! Ventilation and infiltration modeling.
//!
//! This module provides tools for defining ventilation schedules and calculating
//! time-varying air change rates.
//!
//! # Ventilation Modes
//!
//! - **Constant**: Fixed ACH regardless of conditions
//! - **Scheduled**: Time-based ACH changes (e.g., night ventilation)
//! - **Weather-Responsive**: ACH varies with outdoor temperature and wind speed

use crate::physics::units::{FromF64, ThermalConductance};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Stack effect coefficient for buoyancy-driven ventilation.
/// Based on ASHRAE 140 natural ventilation model.
pub const STACK_COEFFICIENT: f64 = 0.025;

/// Air density at standard conditions (kg/m³).
pub const AIR_DENSITY: f64 = 1.2;

/// Air specific heat capacity (J/kg·K).
pub const AIR_SPECIFIC_HEAT: f64 = 1000.0;

/// Calculates infiltration ACH from wind speed using the ASHRAE simple method.
///
/// # Arguments
/// * `wind_speed` - Wind speed in m/s
/// * `building_height` - Building height in meters (affects shielding coefficient)
/// * `shielding_factor` - Shielding factor (0 = very sheltered, 1 = no shielding)
///
/// # Returns
/// Wind-driven infiltration rate in ACH
pub fn calculate_wind_infiltration_ach(
    wind_speed: f64,
    building_height: f64,
    shielding_factor: f64,
) -> f64 {
    let shelter_coefficient = 0.0 + (1.0 - shielding_factor) * 0.4;
    let height_factor = (building_height / 3.0).powf(0.5);
    let base_wind_speed = 3.0;
    let n_factor = shelter_coefficient * height_factor;
    n_factor * (wind_speed / base_wind_speed)
}

/// Calculates stack effect ACH for temperature-driven natural ventilation.
///
/// # Arguments
/// * `indoor_temp` - Indoor air temperature (°C)
/// * `outdoor_temp` - Outdoor air temperature (°C)
/// * `height_diff` - Vertical distance between inlet and outlet (m)
/// * `opening_area` - Opening area (m²)
/// * `zone_volume` - Zone volume (m³)
///
/// # Returns
/// Stack-driven infiltration rate in ACH
pub fn calculate_stack_infiltration_ach(
    indoor_temp: f64,
    outdoor_temp: f64,
    height_diff: f64,
    opening_area: f64,
    zone_volume: f64,
) -> f64 {
    if zone_volume <= 0.0 || height_diff <= 0.0 {
        return 0.0;
    }
    let delta_t = (indoor_temp - outdoor_temp).abs();
    if delta_t < 0.5 {
        return 0.0;
    }
    let flow_arg = delta_t / height_diff;
    let flow_sqrt = if flow_arg > 0.0 { flow_arg.sqrt() } else { 0.0 };
    let q_vent = STACK_COEFFICIENT * opening_area * flow_sqrt;
    q_vent / zone_volume
}

/// Calculates combined infiltration ACH using simplified ASHRAE method.
///
/// Combines wind-driven and stack-driven effects for realistic infiltration modeling.
/// Used for free-floating and mixed-mode buildings without mechanical ventilation.
///
/// # Arguments
/// * `outdoor_temp` - Outdoor air temperature (°C)
/// * `indoor_temp` - Indoor air temperature (°C)
/// * `wind_speed` - Wind speed in m/s
/// * `height_diff` - Building height in meters
/// * `opening_area` - Effective opening area for stack effect (m²)
/// * `zone_volume` - Zone volume (m³)
/// * `shielding_factor` - Shielding factor (0-1)
pub fn calculate_combined_infiltration_ach(
    outdoor_temp: f64,
    indoor_temp: f64,
    wind_speed: f64,
    height_diff: f64,
    opening_area: f64,
    zone_volume: f64,
    shielding_factor: f64,
) -> f64 {
    let wind_ach = calculate_wind_infiltration_ach(wind_speed, height_diff, shielding_factor);
    let stack_ach = calculate_stack_infiltration_ach(
        indoor_temp,
        outdoor_temp,
        height_diff,
        opening_area,
        zone_volume,
    );
    let total_ach = wind_ach + stack_ach;
    total_ach.max(0.0)
}

/// Trait for defining air change rate (ACH) schedules.
///
/// All implementations should return weather-dependent ACH when applicable,
/// using the provided weather parameters.
pub trait VentilationSchedule: Debug + Send + Sync {
    /// Returns the air change rate (ACH) for a given hour.
    ///
    /// # Arguments
    /// * `hour` - Hour of day (0-23)
    /// * `T_outdoor` - Outdoor temperature [C]
    /// * `T_indoor` - Indoor temperature [C]
    /// * `wind_speed` - Wind speed [m/s]
    /// * `volume` - Zone volume [m³]
    fn get_ach(
        &self,
        hour: usize,
        T_outdoor: f64,
        T_indoor: f64,
        wind_speed: f64,
        volume: f64,
    ) -> f64;
    /// Clones the schedule into a boxed trait object.
    fn clone_box(&self) -> Box<dyn VentilationSchedule>;
}

/// Computes the forced-convection multiplier for h_tr_is based on ACH.
///
/// Uses the ASHRAE/EnergyPlus empirical correlation for interior forced convection:
/// `h_c = h_c_still + 0.84 * ACH^0.8` [W/m²K]
///
/// Where:
/// - `h_c_still = 3.45 W/m²K` (ASHRAE 140 simplified 5R1C still-air value)
/// - ACH is in air changes per hour
///
/// This gives approximately:
/// - ACH=0.5: ratio ≈ 1.14× (daytime baseline)
/// - ACH=3:   ratio ≈ 1.59× (Case 950 night vent threshold)
/// - ACH=13.14: ratio ≈ 2.91× (Case 650/950 spec night vent ACH=13.14)
/// - ACH=40:  ratio ≈ 5.66× (theoretical high-ACH night vent)
///
/// Reference: ASHRAE Handbook — Fundamentals (ch. 4), EnergyPlus Engineering Reference.
/// Issue #1279, Issue #1624.
pub fn h_tr_is_ach_multiplier(ach: f64) -> f64 {
    const H_C_STILL: f64 = 3.45; // W/m²K - ASHRAE 140 simplified still-air value
    if ach <= 0.0 {
        1.0
    } else {
        // h_c_forced = h_c_still + 0.84 * ACH^0.8
        let h_c_forced = H_C_STILL + 0.84 * ach.powf(0.8);
        h_c_forced / H_C_STILL
    }
}

/// A constant ventilation schedule.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConstantVentilation {
    pub ach: f64,
}

impl ConstantVentilation {
    pub fn new(ach: f64) -> Self {
        Self { ach }
    }
}

impl VentilationSchedule for ConstantVentilation {
    fn get_ach(
        &self,
        _hour: usize,
        _T_outdoor: f64,
        _T_indoor: f64,
        _wind_speed: f64,
        _volume: f64,
    ) -> f64 {
        self.ach
    }
    fn clone_box(&self) -> Box<dyn VentilationSchedule> {
        Box::new(*self)
    }
}

/// A scheduled ventilation system with base infiltration and a timed fan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledVentilation {
    /// Base infiltration rate (ACH) always present.
    pub base_ach: f64,
    /// Additional ACH when fan is ON.
    pub fan_ach: f64,
    /// 24-hour binary schedule (true = fan ON, false = fan OFF).
    pub schedule: [bool; 24],
}

impl ScheduledVentilation {
    /// Creates a new scheduled ventilation.
    pub fn new(base_ach: f64, fan_ach: f64) -> Self {
        Self {
            base_ach,
            fan_ach,
            schedule: [false; 24],
        }
    }

    /// Creates a night ventilation schedule (ON during specified range).
    pub fn night_ventilation(
        base_ach: f64,
        fan_ach: f64,
        start_hour: usize,
        end_hour: usize,
    ) -> Self {
        let mut vent = Self::new(base_ach, fan_ach);
        if start_hour == end_hour {
            vent.schedule = [true; 24];
        } else if start_hour < end_hour {
            for i in start_hour..end_hour {
                vent.schedule[i] = true;
            }
        } else {
            for i in start_hour..24 {
                vent.schedule[i] = true;
            }
            for i in 0..end_hour {
                vent.schedule[i] = true;
            }
        }
        vent
    }
}

impl VentilationSchedule for ScheduledVentilation {
    fn get_ach(
        &self,
        hour: usize,
        _T_outdoor: f64,
        _T_indoor: f64,
        _wind_speed: f64,
        _volume: f64,
    ) -> f64 {
        if self.schedule[hour] {
            self.base_ach + self.fan_ach
        } else {
            self.base_ach
        }
    }
    fn clone_box(&self) -> Box<dyn VentilationSchedule> {
        Box::new(self.clone())
    }
}

/// Weather-dependent ventilation that responds to outdoor temperature.
///
/// This ventilation mode is particularly important for mixed-mode buildings
/// where natural ventilation (windows, vents) responds to weather conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherDependentVentilation {
    /// Base infiltration rate (ACH) always present.
    pub base_ach: f64,
    /// Minimum ACH when ventilation is OFF (natural ventilation closed).
    pub min_ach: f64,
    /// Maximum ACH when ventilation is fully ON (windows open).
    pub max_ach: f64,
    /// Outdoor temperature threshold to START ventilation (°C).
    pub start_temp: f64,
    /// Outdoor temperature threshold to fully OPEN ventilation (°C).
    pub full_open_temp: f64,
    /// Indoor temperature threshold for cooling mode (°C).
    pub indoor_cooling_setpoint: f64,
    /// Building height for wind calculation (m).
    pub building_height: f64,
    /// Effective opening area fraction (0-1).
    pub opening_fraction: f64,
}

impl WeatherDependentVentilation {
    pub fn new(
        base_ach: f64,
        min_ach: f64,
        max_ach: f64,
        start_temp: f64,
        full_open_temp: f64,
    ) -> Self {
        Self {
            base_ach,
            min_ach,
            max_ach,
            start_temp,
            full_open_temp: if full_open_temp <= start_temp {
                start_temp + 5.0
            } else {
                full_open_temp
            },
            indoor_cooling_setpoint: 26.0,
            building_height: 2.7,
            opening_fraction: 0.3,
        }
    }

    pub fn mixed_mode(
        base_ach: f64,
        max_ach: f64,
        start_temp: f64,
        full_open_temp: f64,
        indoor_cooling_setpoint: f64,
    ) -> Self {
        Self {
            base_ach,
            min_ach: base_ach,
            max_ach,
            start_temp,
            full_open_temp: if full_open_temp <= start_temp {
                start_temp + 5.0
            } else {
                full_open_temp
            },
            indoor_cooling_setpoint,
            building_height: 2.7,
            opening_fraction: 0.3,
        }
    }

    /// Calculate ventilation ACH based on weather and indoor conditions.
    ///
    /// Returns the actual ACH considering both outdoor temperature and wind.
    ///
    /// # ASHRAE 140-2023 §5.5.3.6 default-infiltration lock-in
    ///
    /// ASHRAE Standard 140 (BESTEST) specifies a default infiltration rate of
    /// **0.5 ACH** for the Case 900 / 920 / 940 / 950 / 960 reference models.
    /// When this struct is constructed with `min_ach == max_ach == 0.5` and the
    /// spec inputs (T_in = 20 °C, building height = 2.7 m, shielding = 0.5,
    /// volume = 129.6 m³, Denver TMY3 wind), this method returns exactly
    /// `0.5 ACH` for every of the 8760 hours — i.e. the wind/temperature
    /// blending does not perturb the spec value.
    ///
    /// See `tests/ventilation_isolation.rs::test_ashrae_140_0p5_ach_default`
    /// for the lock-in assertion (Issue #1327), and PRs #1278/#1279 for the
    /// upstream wind + forced-convection fixes that this lock-in guards.
    pub fn get_ach_weather(
        &self,
        outdoor_temp: f64,
        indoor_temp: f64,
        wind_speed: f64,
        zone_volume: f64,
    ) -> f64 {
        let temp_benefit = self.outdoor_temp_benefit(outdoor_temp, indoor_temp);
        let wind_benefit = self.wind_benefit(wind_speed, outdoor_temp, indoor_temp, zone_volume);
        let combined = (temp_benefit + wind_benefit) / 2.0;
        (self.min_ach + (self.max_ach - self.min_ach) * combined).max(self.min_ach)
    }

    /// Calculate outdoor temperature benefit (0-1).
    ///
    /// Returns 0 when outdoor temp <= start_temp (no benefit),
    /// Returns 1 when outdoor temp >= full_open_temp (full benefit).
    fn outdoor_temp_benefit(&self, outdoor_temp: f64, indoor_temp: f64) -> f64 {
        if outdoor_temp <= self.start_temp {
            return 0.0;
        }
        if outdoor_temp >= self.full_open_temp {
            return 1.0;
        }
        if indoor_temp <= self.indoor_cooling_setpoint {
            return 0.0;
        }
        let delta_t_out = self.full_open_temp - self.start_temp;
        if delta_t_out <= 0.0 {
            return 0.0;
        }
        ((outdoor_temp - self.start_temp) / delta_t_out).clamp(0.0, 1.0)
    }

    /// Calculate wind benefit (0-1) using combined wind + stack infiltration.
    ///
    /// Uses `calculate_combined_infiltration_ach` which properly accounts for:
    /// - Wind-driven infiltration via `calculate_wind_infiltration_ach`
    /// - Stack-driven infiltration via `calculate_stack_infiltration_ach`
    fn wind_benefit(
        &self,
        wind_speed: f64,
        outdoor_temp: f64,
        indoor_temp: f64,
        zone_volume: f64,
    ) -> f64 {
        let opening_area = self.opening_fraction * 2.0 * (self.building_height * 3.0);
        // Shielding factor 0.5 is appropriate for weather-responsive ventilation
        // (windows/vents have partial shielding vs. whole-building tightness)
        let shielding_factor = 0.5;
        let ach = calculate_combined_infiltration_ach(
            outdoor_temp,
            indoor_temp,
            wind_speed,
            self.building_height,
            opening_area,
            zone_volume,
            shielding_factor,
        );
        (ach / self.max_ach).clamp(0.0, 1.0)
    }
}

impl VentilationSchedule for WeatherDependentVentilation {
    fn get_ach(
        &self,
        _hour: usize,
        T_outdoor: f64,
        T_indoor: f64,
        wind_speed: f64,
        volume: f64,
    ) -> f64 {
        self.get_ach_weather(T_outdoor, T_indoor, wind_speed, volume)
    }
    fn clone_box(&self) -> Box<dyn VentilationSchedule> {
        Box::new(self.clone())
    }
}

/// Utility to calculate thermal conductance (W/K) from air change rate (ACH).
///
/// This computes the h_ve ventilation heat transfer coefficient, which represents
/// the rate of heat gain/loss due to ventilation air exchange.
///
/// # Validation
/// The h_ve calculation is validated against EnergyPlus reference data.
/// For a reference case with ACH=0.5, volume=129.6 m³, rho=1.2 kg/m³, cp=1005 J/kg·K:
/// - Fluxion result: ~21.71 W/K
/// - EnergyPlus result: 21.6 W/K
/// - Difference: 0.5% (within acceptable tolerance)
///
/// See GitHub Issue #918 for the full validation study.
/// Issue concluded that h_ve values are correct and the ventilation module is working as expected.
///
/// # Arguments
/// * `ach` - Air changes per hour (1/h)
/// * `volume` - Zone volume (m³)
/// * `rho` - Air density (kg/m³), typically 1.2
/// * `cp` - Specific heat capacity of air (J/kg·K), typically 1005
pub fn ach_to_conductance(ach: f64, volume: f64, rho: f64, cp: f64) -> ThermalConductance {
    ThermalConductance::from_value((ach * volume * rho * cp) / 3600.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_ventilation() {
        let vent = ConstantVentilation::new(0.5);
        assert_eq!(vent.ach, 0.5);
        assert_eq!(vent.get_ach(0, 20.0, 22.0, 2.0, 100.0), 0.5);
        assert_eq!(vent.get_ach(12, 20.0, 22.0, 2.0, 100.0), 0.5);
        assert_eq!(vent.get_ach(23, 20.0, 22.0, 2.0, 100.0), 0.5);
    }

    #[test]
    fn test_constant_ventilation_clone() {
        let vent = ConstantVentilation::new(1.0);
        let cloned = vent.clone_box();
        assert_eq!(cloned.get_ach(5, 20.0, 22.0, 2.0, 100.0), 1.0);
    }

    #[test]
    fn test_scheduled_ventilation_default() {
        let vent = ScheduledVentilation::new(0.3, 2.0);
        assert_eq!(vent.base_ach, 0.3);
        assert_eq!(vent.fan_ach, 2.0);
        assert!(!vent.schedule.iter().any(|&x| x)); // all false
                                                    // Should return base_ach for all hours
        for hour in 0..24 {
            assert_eq!(vent.get_ach(hour, 20.0, 22.0, 2.0, 100.0), 0.3);
        }
    }

    #[test]
    fn test_night_ventilation_normal_range() {
        let vent = ScheduledVentilation::night_ventilation(0.3, 2.0, 22, 6);
        // Fan ON from hour 22 to 23, 0 to 5
        assert_eq!(vent.get_ach(21, 20.0, 22.0, 2.0, 100.0), 0.3); // before start
        assert_eq!(vent.get_ach(22, 20.0, 22.0, 2.0, 100.0), 2.3); // fan on
        assert_eq!(vent.get_ach(23, 20.0, 22.0, 2.0, 100.0), 2.3); // fan on
        assert_eq!(vent.get_ach(0, 20.0, 22.0, 2.0, 100.0), 2.3); // fan on (next day)
        assert_eq!(vent.get_ach(5, 20.0, 22.0, 2.0, 100.0), 2.3); // fan on
        assert_eq!(vent.get_ach(6, 20.0, 22.0, 2.0, 100.0), 0.3); // fan off
        assert_eq!(vent.get_ach(12, 20.0, 22.0, 2.0, 100.0), 0.3); // fan off
    }

    #[test]
    fn test_night_ventilation_same_start_end() {
        let vent = ScheduledVentilation::night_ventilation(0.3, 2.0, 10, 10);
        // When start == end, fan is on all 24 hours
        for hour in 0..24 {
            assert_eq!(vent.get_ach(hour, 20.0, 22.0, 2.0, 100.0), 2.3);
        }
    }

    #[test]
    fn test_night_ventilation_single_hour() {
        let vent = ScheduledVentilation::night_ventilation(0.5, 1.5, 14, 15);
        assert_eq!(vent.get_ach(13, 20.0, 22.0, 2.0, 100.0), 0.5);
        assert_eq!(vent.get_ach(14, 20.0, 22.0, 2.0, 100.0), 2.0); // fan on
        assert_eq!(vent.get_ach(15, 20.0, 22.0, 2.0, 100.0), 0.5); // fan off
    }

    #[test]
    fn test_scheduled_ventilation_clone() {
        let vent = ScheduledVentilation::night_ventilation(0.3, 2.0, 20, 8);
        let cloned = vent.clone_box();
        assert_eq!(cloned.get_ach(21, 20.0, 22.0, 2.0, 100.0), 2.3);
        assert_eq!(cloned.get_ach(10, 20.0, 22.0, 2.0, 100.0), 0.3);
    }

    #[test]
    fn test_ach_to_conductance() {
        use crate::physics::units::ToF64;
        // Standard values: ach=1.0, volume=100m³, rho=1.2, cp=1005
        let conductance = ach_to_conductance(1.0, 100.0, 1.2, 1005.0);
        assert!((conductance.to_value() - 33.5).abs() < 0.01); // (1*100*1.2*1005)/3600 = 33.5
    }

    #[test]
    fn test_ach_to_conductance_zero() {
        use crate::physics::units::ToF64;
        assert_eq!(ach_to_conductance(0.0, 100.0, 1.2, 1005.0).to_value(), 0.0);
    }

    #[test]
    fn test_ach_to_conductance_scaling() {
        use crate::physics::units::ToF64;
        // Doubling ACH should double conductance
        let c1 = ach_to_conductance(0.5, 100.0, 1.2, 1005.0);
        let c2 = ach_to_conductance(1.0, 100.0, 1.2, 1005.0);
        assert!((c2.to_value() - 2.0 * c1.to_value()).abs() < 0.001);
    }

    #[test]
    fn test_ventilation_schedule_trait_object() {
        let vent1: Box<dyn VentilationSchedule> = Box::new(ConstantVentilation::new(0.5));
        let vent2: Box<dyn VentilationSchedule> =
            Box::new(ScheduledVentilation::night_ventilation(0.3, 2.0, 22, 6));

        assert_eq!(vent1.get_ach(10, 20.0, 22.0, 2.0, 100.0), 0.5);
        assert_eq!(vent2.get_ach(23, 20.0, 22.0, 2.0, 100.0), 2.3);
    }

    #[test]
    fn test_wind_infiltration_basic() {
        let ach = calculate_wind_infiltration_ach(3.0, 3.0, 0.0);
        assert!(ach >= 0.0);
        assert!(ach < 1.0);
    }

    #[test]
    fn test_wind_infiltration_high_wind() {
        let ach_calm = calculate_wind_infiltration_ach(2.0, 3.0, 0.0);
        let ach_windy = calculate_wind_infiltration_ach(6.0, 3.0, 0.0);
        assert!(ach_windy > ach_calm);
    }

    #[test]
    fn test_stack_infiltration_no_delta_t() {
        let ach = calculate_stack_infiltration_ach(25.0, 25.0, 2.0, 1.0, 100.0);
        assert_eq!(ach, 0.0);
    }

    #[test]
    fn test_stack_infiltration_with_delta_t() {
        let ach = calculate_stack_infiltration_ach(25.0, 20.0, 2.0, 1.0, 100.0);
        assert!(ach > 0.0);
    }

    #[test]
    fn test_combined_infiltration_ach() {
        let ach = calculate_combined_infiltration_ach(
            25.0,  // outdoor
            28.0,  // indoor
            3.0,   // wind
            2.7,   // height
            1.0,   // opening
            129.6, // volume
            0.3,   // shielding
        );
        assert!(ach >= 0.0);
    }

    #[test]
    fn test_weather_dependent_ventilation_creation() {
        let vent = WeatherDependentVentilation::new(0.3, 0.3, 2.0, 18.0, 26.0);
        assert_eq!(vent.base_ach, 0.3);
        assert_eq!(vent.min_ach, 0.3);
        assert_eq!(vent.max_ach, 2.0);
        assert_eq!(vent.start_temp, 18.0);
    }

    #[test]
    fn test_weather_dependent_ventilation_outdoor_temp_benefit() {
        let vent = WeatherDependentVentilation::new(0.3, 0.3, 2.0, 18.0, 26.0);
        assert_eq!(vent.outdoor_temp_benefit(15.0, 28.0), 0.0);
        assert_eq!(vent.outdoor_temp_benefit(18.0, 28.0), 0.0);
        assert!(vent.outdoor_temp_benefit(23.0, 28.0) > 0.0);
        assert_eq!(vent.outdoor_temp_benefit(26.0, 28.0), 1.0);
    }

    #[test]
    fn test_weather_dependent_ventilation_indoor_not_cooling() {
        let vent = WeatherDependentVentilation::new(0.3, 0.3, 2.0, 18.0, 26.0);
        assert_eq!(vent.outdoor_temp_benefit(25.0, 24.0), 0.0);
    }

    #[test]
    fn test_weather_dependent_ventilation_fallback_full_open() {
        let vent = WeatherDependentVentilation::new(0.3, 0.3, 2.0, 26.0, 18.0);
        assert_eq!(vent.full_open_temp, 31.0);
    }

    // =============================================================================
    // Issue #1624: Forced-convection h_tr_is boost during high ACH night flush
    // =============================================================================

    #[test]
    fn test_h_tr_is_ach_multiplier_zero_ach() {
        // Zero or negative ACH should return 1.0 (no boost)
        assert_eq!(h_tr_is_ach_multiplier(0.0), 1.0);
        assert_eq!(h_tr_is_ach_multiplier(-1.0), 1.0);
    }

    #[test]
    fn test_h_tr_is_ach_multiplier_baseline() {
        // ACH=0.5: ratio ≈ 1.14× (daytime baseline, below night flush threshold)
        let multiplier = h_tr_is_ach_multiplier(0.5);
        assert!((multiplier - 1.14).abs() < 0.01);
    }

    #[test]
    fn test_h_tr_is_ach_multiplier_night_flush_threshold() {
        // ACH=3.0: ratio ≈ 1.59× (Case 950 night vent threshold)
        let multiplier = h_tr_is_ach_multiplier(3.0);
        assert!((multiplier - 1.59).abs() < 0.01);
    }

    #[test]
    fn test_h_tr_is_ach_multiplier_high_ach() {
        // ACH=13.14: ratio ≈ 2.91× (Case 650/950 spec night vent ACH=13.14)
        let multiplier = h_tr_is_ach_multiplier(13.14);
        assert!((multiplier - 2.91).abs() < 0.02);
    }

    #[test]
    fn test_h_tr_is_ach_multiplier_very_high_ach() {
        // ACH=40: ratio ≈ 5.66× (theoretical high-ACH night vent)
        let multiplier = h_tr_is_ach_multiplier(40.0);
        assert!((multiplier - 5.66).abs() < 0.02);
    }

    #[test]
    fn test_h_tr_is_ach_multiplier_monotonic_increase() {
        // Multiplier should increase monotonically with ACH
        let m1 = h_tr_is_ach_multiplier(1.0);
        let m2 = h_tr_is_ach_multiplier(5.0);
        let m3 = h_tr_is_ach_multiplier(10.0);
        let m4 = h_tr_is_ach_multiplier(20.0);
        assert!(m1 < m2);
        assert!(m2 < m3);
        assert!(m3 < m4);
    }

    #[test]
    fn test_h_tr_is_ach_multiplier_ach_3_threshold_boost() {
        // Verify boost activates at ACH >= 3.0 (night flush threshold)
        let multiplier_below = h_tr_is_ach_multiplier(2.9);
        let multiplier_at = h_tr_is_ach_multiplier(3.0);
        let multiplier_above = h_tr_is_ach_multiplier(4.0);

        // Below threshold: multiplier < 1.59
        assert!(multiplier_below < 1.59);
        // At threshold: multiplier ≈ 1.59
        assert!((multiplier_at - 1.59).abs() < 0.01);
        // Above threshold: multiplier > 1.59
        assert!(multiplier_above > 1.59);
    }
}
