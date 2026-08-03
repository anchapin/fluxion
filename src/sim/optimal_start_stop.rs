//! Optimal start/stop HVAC control strategies.
//!
//! This module implements advanced HVAC control strategies that leverage
//! thermal mass and predictive algorithms to minimize energy consumption
//! while maintaining occupant comfort.
//!
//! # Control Strategies
//!
//! - **Optimal Start**: Compute earliest HVAC start time to reach setpoint by occupancy
//! - **Optimal Stop**: Determine latest HVAC shutdown while maintaining setpoint until departure
//! - **Predictive Control**: Use thermal model to anticipate load needs
//!
//! # Energy Savings
//!
//! These strategies typically reduce annual HVAC energy by 5-15% compared to
//! fixed schedules, with the exact savings depending on:
//! - Building thermal mass
//! - Occupancy pattern regularity
//! - Climate conditions
//! - Setpoint setbacks implemented

use crate::physics::cta::VectorField;
use serde::{Deserialize, Serialize};

/// Configuration for optimal start/stop control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalStartStopConfig {
    /// Enable optimal start control (default: true)
    pub optimal_start_enabled: bool,
    /// Enable optimal stop control (default: true)
    pub optimal_stop_enabled: bool,
    /// Maximum time before occupancy to start HVAC (hours)
    pub max_start_time_hours: f64,
    /// Maximum time before departure to stop HVAC (hours)
    pub max_stop_time_hours: f64,
    /// Temperature tolerance for reaching setpoint (°C)
    pub setpoint_tolerance: f64,
    /// Safety margin multiplier for start time calculation
    pub start_margin_multiplier: f64,
    /// Safety margin multiplier for stop time calculation
    pub stop_margin_multiplier: f64,
}

impl Default for OptimalStartStopConfig {
    fn default() -> Self {
        Self {
            optimal_start_enabled: true,
            optimal_stop_enabled: true,
            max_start_time_hours: 3.0,
            max_stop_time_hours: 2.0,
            setpoint_tolerance: 0.5,
            start_margin_multiplier: 1.2,
            stop_margin_multiplier: 1.2,
        }
    }
}

/// Result of an optimal start calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalStartResult {
    /// Whether HVAC should be running
    pub should_run: bool,
    /// Recommended start time (hours before occupancy)
    pub start_time_hours: f64,
    /// Estimated temperature at occupancy without pre-start
    pub estimated_temp_without_prestart: f64,
    /// Estimated temperature at occupancy with pre-start
    pub estimated_temp_with_prestart: f64,
    /// Confidence score (0-1) in the prediction
    pub confidence: f64,
}

/// Result of an optimal stop calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalStopResult {
    /// Whether HVAC should be running
    pub should_run: bool,
    /// Recommended stop time (hours before departure)
    pub stop_time_hours: f64,
    /// Estimated temperature at departure if stopped at recommended time
    pub estimated_temp_at_departure: f64,
    /// Estimated temperature at departure if continued running
    pub estimated_temp_if_continued: f64,
    /// Energy saved by stopping early (Wh)
    pub energy_saved_wh: f64,
    /// Confidence score (0-1) in the prediction
    pub confidence: f64,
}

/// Thermal characteristics needed for optimal start/stop prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalCharacteristics {
    /// Zone volume (m³)
    pub zone_volume: f64,
    /// Zone thermal mass (J/K)
    pub thermal_mass_jk: f64,
    /// Heating capacity (W)
    pub heating_capacity_w: f64,
    /// Cooling capacity (W)
    pub cooling_capacity_w: f64,
    /// Heating setpoint (°C)
    pub heating_setpoint: f64,
    /// Cooling setpoint (°C)
    pub cooling_setpoint: f64,
    /// Current zone temperature (°C)
    pub current_temp: f64,
    /// Outdoor temperature (°C)
    pub outdoor_temp: f64,
    /// Supply air temperature for heating (°C)
    pub supply_heating_temp: f64,
    /// Supply air temperature for cooling (°C)
    pub supply_cooling_temp: f64,
}

impl ThermalCharacteristics {
    /// Calculate heating rate to the zone.
    ///
    /// Uses the ideal loads formula: Q = ṁ·cp·ΔT
    pub fn heating_rate(&self, supply_temp: f64) -> f64 {
        let delta_t = supply_temp - self.current_temp;
        if delta_t <= 0.0 {
            return 0.0;
        }
        // Airflow rate (m³/s) = ACH * volume / 3600
        // For simplicity, using 0.5 ACH as baseline
        let airflow_m3s = 0.5 * self.zone_volume / 3600.0;
        let mass_flow_kgs = airflow_m3s * 1.2; // air density kg/m³
        let cp = 1005.0; // J/kg·K
        mass_flow_kgs * cp * delta_t
    }

    /// Calculate time to reach setpoint from current temperature (hours).
    pub fn time_to_heating_setpoint(&self) -> f64 {
        let delta_t = self.heating_setpoint - self.current_temp;
        if delta_t <= 0.0 {
            return 0.0;
        }
        let rate_per_hour = self.heating_rate(self.supply_heating_temp);
        if rate_per_hour <= 0.0 {
            return f64::INFINITY;
        }
        // Energy needed: delta_t * thermal_mass
        let energy_needed_j = delta_t * self.thermal_mass_jk;
        // Rate in W (J/s), so divide by 3600 for Wh
        let rate_wh = rate_per_hour / 3600.0;
        if rate_wh <= 0.0 {
            return f64::INFINITY;
        }
        energy_needed_j / rate_per_hour / 3600.0 // hours
    }

    /// Calculate time to reach setpoint from current temperature for cooling (hours).
    pub fn time_to_cooling_setpoint(&self) -> f64 {
        let delta_t = self.current_temp - self.cooling_setpoint;
        if delta_t <= 0.0 {
            return 0.0;
        }
        let rate_per_hour = self.cooling_rate(self.supply_cooling_temp);
        if rate_per_hour <= 0.0 {
            return f64::INFINITY;
        }
        let energy_needed_j = delta_t * self.thermal_mass_jk;
        energy_needed_j / rate_per_hour / 3600.0 // hours
    }

    /// Calculate cooling rate from the zone.
    pub fn cooling_rate(&self, supply_temp: f64) -> f64 {
        let delta_t = self.current_temp - supply_temp;
        if delta_t <= 0.0 {
            return 0.0;
        }
        let airflow_m3s = 0.5 * self.zone_volume / 3600.0;
        let mass_flow_kgs = airflow_m3s * 1.2;
        let cp = 1005.0;
        mass_flow_kgs * cp * delta_t
    }
}

/// Optimal start/stop controller that uses thermal characteristics to predict HVAC needs.
#[derive(Debug, Clone)]
pub struct OptimalStartStopController {
    config: OptimalStartStopConfig,
    /// Previous zone temperatures for rate estimation
    prev_temps: VectorField,
    /// Previous outdoor temperatures for rate estimation
    prev_outdoor_temps: VectorField,
    /// Number of zones
    num_zones: usize,
}

impl OptimalStartStopController {
    /// Create a new optimal start/stop controller.
    pub fn new(num_zones: usize) -> Self {
        Self {
            config: OptimalStartStopConfig::default(),
            prev_temps: VectorField::from_scalar(20.0, num_zones),
            prev_outdoor_temps: VectorField::from_scalar(10.0, num_zones),
            num_zones,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(num_zones: usize, config: OptimalStartStopConfig) -> Self {
        Self {
            config,
            prev_temps: VectorField::from_scalar(20.0, num_zones),
            prev_outdoor_temps: VectorField::from_scalar(10.0, num_zones),
            num_zones,
        }
    }

    /// Calculate optimal start time for a zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zone index
    /// * `characteristics` - Thermal characteristics of the zone
    /// * `hours_until_occupied` - Hours until next occupancy period
    ///
    /// # Returns
    /// OptimalStartResult with recommended start time and predictions
    pub fn calculate_optimal_start(
        &self,
        _zone_id: usize,
        characteristics: &ThermalCharacteristics,
        hours_until_occupied: f64,
    ) -> OptimalStartResult {
        if !self.config.optimal_start_enabled {
            return OptimalStartResult {
                should_run: true,
                start_time_hours: 0.0,
                estimated_temp_without_prestart: characteristics.current_temp,
                estimated_temp_with_prestart: characteristics.current_temp,
                confidence: 1.0,
            };
        }

        // Determine mode: heating or cooling needed
        let mode = if characteristics.current_temp < characteristics.heating_setpoint {
            OptimalMode::Heating
        } else if characteristics.current_temp > characteristics.cooling_setpoint {
            OptimalMode::Cooling
        } else {
            return OptimalStartResult {
                should_run: false,
                start_time_hours: 0.0,
                estimated_temp_without_prestart: characteristics.current_temp,
                estimated_temp_with_prestart: characteristics.current_temp,
                confidence: 1.0,
            };
        };

        // Calculate time needed to reach setpoint
        let time_needed = match mode {
            OptimalMode::Heating => characteristics.time_to_heating_setpoint(),
            OptimalMode::Cooling => characteristics.time_to_cooling_setpoint(),
        };

        // Apply margin for safety
        let adjusted_time_needed = time_needed * self.config.start_margin_multiplier;

        // Calculate recommended start time
        let start_time = adjusted_time_needed
            .min(hours_until_occupied)
            .min(self.config.max_start_time_hours);

        // Estimate temperature at occupancy
        // "without" = free-float (no HVAC) until occupancy
        let estimated_without =
            self.estimate_free_float_temp(characteristics, hours_until_occupied);
        // "with" = HVAC running for start_time hours, then free-float for remaining time
        let preheat_hours = start_time;
        let free_float_hours = hours_until_occupied - start_time;
        let temp_after_preheat = if preheat_hours > 0.0 {
            self.estimate_temp_at_time(characteristics, preheat_hours, mode)
        } else {
            characteristics.current_temp
        };
        let estimated_with = if free_float_hours > 0.0 {
            // Account for cooling after HVAC stops - use the preheat temp as starting point
            let chars_after_preheat = ThermalCharacteristics {
                current_temp: temp_after_preheat,
                ..*characteristics
            };
            self.estimate_free_float_temp(&chars_after_preheat, free_float_hours)
        } else {
            temp_after_preheat
        };

        // Calculate confidence based on how well we understand the thermal mass
        let confidence = self.calculate_confidence(characteristics, time_needed);

        OptimalStartResult {
            should_run: start_time > 0.0 && hours_until_occupied > 0.0,
            start_time_hours: start_time,
            estimated_temp_without_prestart: estimated_without,
            estimated_temp_with_prestart: estimated_with,
            confidence,
        }
    }

    /// Calculate optimal stop time for a zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zone index
    /// * `characteristics` - Thermal characteristics of the zone
    /// * `hours_until_departure` - Hours until end of occupancy period
    ///
    /// # Returns
    /// OptimalStopResult with recommended stop time and energy savings
    pub fn calculate_optimal_stop(
        &self,
        _zone_id: usize,
        characteristics: &ThermalCharacteristics,
        hours_until_departure: f64,
    ) -> OptimalStopResult {
        if !self.config.optimal_stop_enabled {
            return OptimalStopResult {
                should_run: true,
                stop_time_hours: 0.0,
                estimated_temp_at_departure: characteristics.current_temp,
                estimated_temp_if_continued: characteristics.current_temp,
                energy_saved_wh: 0.0,
                confidence: 1.0,
            };
        }

        // Determine current mode
        let mode = if characteristics.current_temp < characteristics.heating_setpoint {
            OptimalMode::Heating
        } else if characteristics.current_temp > characteristics.cooling_setpoint {
            OptimalMode::Cooling
        } else {
            // Already at setpoint, can stop immediately
            return OptimalStopResult {
                should_run: false,
                stop_time_hours: 0.0,
                estimated_temp_at_departure: characteristics.current_temp,
                estimated_temp_if_continued: characteristics.current_temp,
                energy_saved_wh: 0.0,
                confidence: 1.0,
            };
        };

        // Calculate how long we can run before temperature drifts out of band
        let allowable_drift = self.config.setpoint_tolerance;
        let (time_available, _) = self.calculate_drift_time(
            characteristics,
            hours_until_departure,
            allowable_drift,
            mode,
        );

        // Apply margin for safety
        let adjusted_time_available = time_available / self.config.stop_margin_multiplier;

        // Calculate recommended stop time
        let stop_time = adjusted_time_available
            .min(hours_until_departure)
            .min(self.config.max_stop_time_hours);

        // Calculate energy savings
        let power = match mode {
            OptimalMode::Heating => {
                characteristics.heating_rate(characteristics.supply_heating_temp)
            }
            OptimalMode::Cooling => {
                characteristics.cooling_rate(characteristics.supply_cooling_temp)
            }
        };
        let energy_saved_wh = (power / 1000.0) * stop_time; // kW * h = kWh -> Wh

        // Estimate temperatures
        let estimated_at_departure =
            self.estimate_temp_at_time(characteristics, hours_until_departure - stop_time, mode);
        let estimated_if_continued =
            self.estimate_temp_at_time(characteristics, hours_until_departure, mode);

        let confidence = self.calculate_confidence(characteristics, time_available);

        OptimalStopResult {
            should_run: stop_time < hours_until_departure,
            stop_time_hours: stop_time,
            estimated_temp_at_departure: estimated_at_departure,
            estimated_temp_if_continued: estimated_if_continued,
            energy_saved_wh,
            confidence,
        }
    }

    /// Estimate temperature at a future time given current conditions.
    fn estimate_temp_at_time(
        &self,
        characteristics: &ThermalCharacteristics,
        hours: f64,
        mode: OptimalMode,
    ) -> f64 {
        if hours <= 0.0 {
            return characteristics.current_temp;
        }

        let rate = match mode {
            OptimalMode::Heating => {
                characteristics.heating_rate(characteristics.supply_heating_temp)
            }
            OptimalMode::Cooling => {
                characteristics.cooling_rate(characteristics.supply_cooling_temp)
            }
        };

        // Simple linear approximation: delta_t = rate * time / thermal_mass
        let delta_t = (rate * hours * 3600.0) / characteristics.thermal_mass_jk;

        match mode {
            OptimalMode::Heating => characteristics.current_temp + delta_t,
            OptimalMode::Cooling => characteristics.current_temp - delta_t,
        }
    }

    /// Estimate free-float temperature at a future time (no HVAC).
    ///
    /// In free-float mode, the zone temperature drifts toward the outdoor temperature
    /// based on the effective heat transfer coefficient and thermal mass.
    fn estimate_free_float_temp(
        &self,
        characteristics: &ThermalCharacteristics,
        hours: f64,
    ) -> f64 {
        if hours <= 0.0 {
            return characteristics.current_temp;
        }

        // Heat transfer coefficient (W/K) - approximates building envelope conductance
        // Use a simplified model:UA = ventilation_conductance + envelope_conductance
        // For a typical office: UA ≈ 50-200 W/K
        // Use 100 W/K as a reasonable default
        let ua = 100.0; // W/K

        // Thermal time constant (hours)
        // tau = thermal_mass / ua
        let tau_hours = characteristics.thermal_mass_jk / (ua * 3600.0);

        // Outdoor temperature
        let t_out = characteristics.outdoor_temp;

        // Free-float equation: T(t) = T_out + (T_0 - T_out) * exp(-t / tau)
        let t0 = characteristics.current_temp;
        let t_diff = t0 - t_out;

        if t_diff.abs() < 0.01 {
            return t0; // Already at outdoor temperature
        }

        // Exponential approach to outdoor temperature
        let time_constant_hours = tau_hours.max(1.0); // Avoid division by very small numbers
        let decay = (-hours / time_constant_hours).exp();
        t_out + t_diff * decay
    }

    /// Calculate how long until temperature drifts out of acceptable range.
    fn calculate_drift_time(
        &self,
        characteristics: &ThermalCharacteristics,
        max_time: f64,
        allowable_drift: f64,
        mode: OptimalMode,
    ) -> (f64, f64) {
        let rate = match mode {
            OptimalMode::Heating => {
                characteristics.heating_rate(characteristics.supply_heating_temp)
            }
            OptimalMode::Cooling => {
                characteristics.cooling_rate(characteristics.supply_cooling_temp)
            }
        };

        if rate <= 0.0 {
            return (max_time, 0.0);
        }

        // Calculate drift rate (°C per hour)
        let drift_rate_per_hour = (rate * 3600.0) / characteristics.thermal_mass_jk;

        if drift_rate_per_hour <= 0.0 {
            return (max_time, 0.0);
        }

        // Time until drift equals allowable
        let time_to_drift = allowable_drift / drift_rate_per_hour;
        (time_to_drift.min(max_time), drift_rate_per_hour)
    }

    /// Calculate confidence in the prediction.
    fn calculate_confidence(
        &self,
        characteristics: &ThermalCharacteristics,
        time_estimate: f64,
    ) -> f64 {
        // Higher confidence for:
        // - Larger thermal mass (slower response, more predictable)
        // - Reasonable time estimates (not too fast, not too slow)
        let thermal_mass_factor =
            (characteristics.thermal_mass_jk / 1e6_f64).clamp(0.1, 10.0) / 10.0;
        let time_factor = if time_estimate.is_infinite() {
            0.5
        } else {
            (time_estimate / 4.0).clamp(0.1, 1.0)
        };

        (thermal_mass_factor * 0.6 + time_factor * 0.4).clamp(0.0, 1.0)
    }

    /// Update thermal tracking with current observations.
    pub fn update_thermal_tracking(&mut self, zone_id: usize, zone_temp: f64, outdoor_temp: f64) {
        if zone_id < self.num_zones {
            self.prev_temps.as_mut_slice()[zone_id] = zone_temp;
            self.prev_outdoor_temps.as_mut_slice()[zone_id] = outdoor_temp;
        }
    }

    /// Get configuration reference.
    pub fn config(&self) -> &OptimalStartStopConfig {
        &self.config
    }

    /// Mutably get configuration reference.
    pub fn config_mut(&mut self) -> &mut OptimalStartStopConfig {
        &mut self.config
    }
}

/// Mode of HVAC operation for optimal control.
#[derive(Debug, Clone, Copy, PartialEq)]
enum OptimalMode {
    Heating,
    Cooling,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_characteristics() -> ThermalCharacteristics {
        ThermalCharacteristics {
            zone_volume: 129.6,           // 8m x 6m x 2.7m
            thermal_mass_jk: 1_000_000.0, // J/K
            heating_capacity_w: 5000.0,
            cooling_capacity_w: 5000.0,
            heating_setpoint: 22.0,
            cooling_setpoint: 26.0,
            current_temp: 18.0,
            outdoor_temp: 10.0,
            supply_heating_temp: 40.0,
            supply_cooling_temp: 13.0,
        }
    }

    #[test]
    fn test_optimal_start_heating_needed() {
        let controller = OptimalStartStopController::new(1);
        let chars = create_test_characteristics();

        let result = controller.calculate_optimal_start(0, &chars, 2.0);

        assert!(result.should_run);
        assert!(result.start_time_hours > 0.0);
        assert!(result.estimated_temp_with_prestart > result.estimated_temp_without_prestart);
        assert!(result.confidence > 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_optimal_start_already_at_setpoint() {
        let mut chars = create_test_characteristics();
        chars.current_temp = 24.0; // Between heating and cooling setpoints

        let controller = OptimalStartStopController::new(1);
        let result = controller.calculate_optimal_start(0, &chars, 2.0);

        assert!(!result.should_run);
    }

    #[test]
    fn test_optimal_stop_heating() {
        let mut chars = create_test_characteristics();
        chars.current_temp = 22.0; // At heating setpoint

        let controller = OptimalStartStopController::new(1);
        let result = controller.calculate_optimal_stop(0, &chars, 1.0);

        // Should recommend stopping since we're already at setpoint
        assert!(result.stop_time_hours >= 0.0);
        assert!(result.energy_saved_wh >= 0.0);
    }

    #[test]
    fn test_thermal_characteristics_heating_rate() {
        let chars = create_test_characteristics();
        let rate = chars.heating_rate(40.0);

        assert!(rate > 0.0);
    }

    #[test]
    fn test_thermal_characteristics_cooling_rate() {
        let mut chars = create_test_characteristics();
        chars.current_temp = 30.0;
        let rate = chars.cooling_rate(13.0);

        assert!(rate > 0.0);
    }

    #[test]
    fn test_time_to_heating_setpoint() {
        let chars = create_test_characteristics();
        let time = chars.time_to_heating_setpoint();

        assert!(time > 0.0);
        assert!(!time.is_infinite());
    }

    #[test]
    fn test_time_to_cooling_setpoint_already_cool() {
        let mut chars = create_test_characteristics();
        chars.current_temp = 24.0; // Already below cooling setpoint
        let time = chars.time_to_cooling_setpoint();

        assert_eq!(time, 0.0);
    }

    #[test]
    fn test_optimal_start_no_pre_start_needed() {
        let mut chars = create_test_characteristics();
        chars.current_temp = 21.5; // Very close to heating setpoint

        let controller = OptimalStartStopController::new(1);
        let result = controller.calculate_optimal_start(0, &chars, 0.5);

        // Should run but with minimal pre-start
        assert!(result.should_run);
        assert!(result.start_time_hours < 0.5);
    }

    #[test]
    fn test_optimal_stop_early_stopping() {
        let mut chars = create_test_characteristics();
        chars.current_temp = 21.5;
        chars.thermal_mass_jk = 5_000_000.0; // High thermal mass

        let controller = OptimalStartStopController::new(1);
        let result = controller.calculate_optimal_stop(0, &chars, 3.0);

        // With high thermal mass, should be able to stop earlier
        assert!(result.stop_time_hours > 0.0);
        assert!(result.energy_saved_wh > 0.0);
    }

    #[test]
    fn test_confidence_lower_for_small_thermal_mass() {
        let mut chars = create_test_characteristics();
        chars.thermal_mass_jk = 100_000.0; // Low thermal mass

        let controller = OptimalStartStopController::new(1);
        let result = controller.calculate_optimal_start(0, &chars, 2.0);

        // Lower confidence for small thermal mass
        assert!(result.confidence < 0.7);
    }

    #[test]
    fn test_update_thermal_tracking() {
        let mut controller = OptimalStartStopController::new(2);
        controller.update_thermal_tracking(0, 20.0, 10.0);
        controller.update_thermal_tracking(1, 22.0, 12.0);

        assert_eq!(controller.prev_temps.as_slice()[0], 20.0);
        assert_eq!(controller.prev_temps.as_slice()[1], 22.0);
        assert_eq!(controller.prev_outdoor_temps.as_slice()[0], 10.0);
        assert_eq!(controller.prev_outdoor_temps.as_slice()[1], 12.0);
    }

    #[test]
    fn test_disabled_optimal_start() {
        let mut config = OptimalStartStopConfig::default();
        config.optimal_start_enabled = false;

        let controller = OptimalStartStopController::with_config(1, config);
        let chars = create_test_characteristics();

        let result = controller.calculate_optimal_start(0, &chars, 2.0);

        // Should always run when disabled
        assert!(result.should_run);
        assert_eq!(result.start_time_hours, 0.0);
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn test_disabled_optimal_stop() {
        let mut config = OptimalStartStopConfig::default();
        config.optimal_stop_enabled = false;

        let controller = OptimalStartStopController::with_config(1, config);
        let chars = create_test_characteristics();

        let result = controller.calculate_optimal_stop(0, &chars, 2.0);

        // Should always run when disabled
        assert!(result.should_run);
        assert_eq!(result.stop_time_hours, 0.0);
        assert_eq!(result.energy_saved_wh, 0.0);
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn test_max_start_time_limit() {
        let chars = create_test_characteristics();
        let mut config = OptimalStartStopConfig::default();
        config.max_start_time_hours = 1.0;

        let controller = OptimalStartStopController::with_config(1, config);
        let result = controller.calculate_optimal_start(0, &chars, 5.0);

        // Should be capped at max_start_time_hours
        assert!(result.start_time_hours <= 1.0);
    }

    #[test]
    fn test_max_stop_time_limit() {
        let chars = create_test_characteristics();
        let mut config = OptimalStartStopConfig::default();
        config.max_stop_time_hours = 0.5;

        let controller = OptimalStartStopController::with_config(1, config);
        let result = controller.calculate_optimal_stop(0, &chars, 5.0);

        // Should be capped at max_stop_time_hours
        assert!(result.stop_time_hours <= 0.5);
    }
}
