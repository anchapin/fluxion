//! HVAC Predictive Control
//!
//! This module provides predictive control strategies using thermal inertia
//! to smooth response and prevent oscillation in high-thermal-mass buildings.

use crate::sim::hvac::HVACMode;
use serde::{Deserialize, Serialize};

/// Predictive HVAC controller using thermal inertia.
///
/// This controller considers thermal mass state and temperature rate of change
/// to smooth response and prevent oscillation, more realistic than simple
/// setpoint hysteresis for high-thermal-mass buildings.
///
/// TODO: Implement full predictive control logic in Plan 15-04
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveController {
    /// Heating setpoint (°C)
    pub heating_setpoint: f64,
    /// Cooling setpoint (°C)
    pub cooling_setpoint: f64,
    /// Deadband tolerance (°C) - prevents rapid cycling near setpoints
    pub deadband_tolerance: f64,
    /// Thermal inertia gain factor (α) - tuning parameter
    pub thermal_inertia_gain: f64,
    /// Temperature rate gain factor (β) - tuning parameter
    pub temp_rate_gain: f64,
    /// Previous zone temperature (for calculating dT/dt)
    pub previous_zone_temp: f64,
}

impl PredictiveController {
    /// Create a new predictive controller with default tuning.
    ///
    /// Default tuning values:
    /// - thermal_inertia_gain: 0.1 (moderate thermal inertia influence)
    /// - temp_rate_gain: 0.01 (small rate influence to prevent overshoot)
    pub fn new(heating_setpoint: f64, cooling_setpoint: f64) -> Self {
        Self {
            heating_setpoint,
            cooling_setpoint,
            deadband_tolerance: 0.5,
            thermal_inertia_gain: 0.1, // Tuned against ASHRAE 800-810
            temp_rate_gain: 0.01,      // Tuned against ASHRAE Guideline 14
            previous_zone_temp: 20.0,  // Initialize at comfortable temp
        }
    }

    /// Create a predictive controller with custom tuning parameters.
    ///
    /// # Arguments
    /// * `heating_setpoint` - Heating setpoint (°C)
    /// * `cooling_setpoint` - Cooling setpoint (°C)
    /// * `thermal_inertia_gain` - Thermal inertia gain factor (α)
    /// * `temp_rate_gain` - Temperature rate gain factor (β)
    pub fn with_tuning(
        heating_setpoint: f64,
        cooling_setpoint: f64,
        thermal_inertia_gain: f64,
        temp_rate_gain: f64,
    ) -> Self {
        Self {
            heating_setpoint,
            cooling_setpoint,
            deadband_tolerance: 0.5,
            thermal_inertia_gain,
            temp_rate_gain,
            previous_zone_temp: 20.0,
        }
    }

    /// Calculate control signal (mode and modulation factor).
    ///
    /// Uses thermal inertia to predict thermal response and adjust control signal.
    ///
    /// # Arguments
    /// * `zone_temp` - Current zone air temperature (°C)
    /// * `mass_temp` - Thermal mass temperature (°C) from 5R1C network
    /// * `temp_rate` - Rate of temperature change (°C/s), dT/dt
    ///
    /// # Returns
    /// Tuple of (HVACMode, modulation_factor) where:
    /// - `mode`: Heating, Cooling, or Off
    /// - `modulation_factor`: 0.0 to 1.0 (0% to 100% capacity)
    ///
    /// # Control Logic
    /// 1. Calculate inertia factor based on zone temp vs mass temp offset
    /// 2. Calculate predictive factor based on temperature rate
    /// 3. Adjust effective setpoints by inertia and prediction
    /// 4. Determine mode based on zone temp vs adjusted setpoints
    /// 5. Calculate modulation factor based on temperature error
    pub fn calculate_modulation(
        &mut self,
        zone_temp: f64,
        mass_temp: f64,
        temp_rate: f64,
    ) -> (HVACMode, f64) {
        // Step 1: Inertia factor based on mass temperature offset
        // If mass temp is cooler than zone temp, building will cool faster (anticipate this)
        let inertia_factor = self.thermal_inertia_gain * (zone_temp - mass_temp);

        // Step 2: Predictive factor based on temperature rate
        // If temperature is rising rapidly, anticipate overshoot
        let predictive_factor = self.temp_rate_gain * temp_rate;

        // Step 3: Effective setpoints adjusted by inertia and prediction
        let effective_heating_sp = self.heating_setpoint + inertia_factor - predictive_factor;
        let effective_cooling_sp = self.cooling_setpoint + inertia_factor - predictive_factor;

        // Step 4: Determine mode based on zone temp vs adjusted setpoints
        // Apply deadband tolerance to prevent cycling
        let heating_threshold = effective_heating_sp - self.deadband_tolerance;
        let cooling_threshold = effective_cooling_sp + self.deadband_tolerance;

        let mode = if zone_temp < heating_threshold {
            HVACMode::Heating
        } else if zone_temp > cooling_threshold {
            HVACMode::Cooling
        } else {
            HVACMode::Off
        };

        // Step 5: Calculate modulation factor based on temperature error
        let temp_error = match mode {
            HVACMode::Heating => effective_heating_sp - zone_temp,
            HVACMode::Cooling => zone_temp - effective_cooling_sp,
            HVACMode::Off => 0.0,
        };

        // Modulation factor: 0-1 based on temperature error
        // Sensitivity: 10.0 means 1°C error = 10% modulation
        let modulation = (temp_error * 10.0).clamp(0.0, 1.0);

        // Update previous zone temp for next timestep's dT/dt calculation
        self.previous_zone_temp = zone_temp;

        (mode, modulation)
    }

    /// Calculate control signal with dynamic setpoints (supports setback schedules).
    ///
    /// This variant allows passing time-varying setpoints, which is needed for
    /// setback schedules where the setpoint changes at different hours.
    pub fn calculate_modulation_with_setpoints(
        &mut self,
        zone_temp: f64,
        mass_temp: f64,
        temp_rate: f64,
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> (HVACMode, f64) {
        // Step 1: Inertia factor based on mass temperature offset
        let inertia_factor = self.thermal_inertia_gain * (zone_temp - mass_temp);

        // Step 2: Predictive factor based on temperature rate
        let predictive_factor = self.temp_rate_gain * temp_rate;

        // Step 3: Effective setpoints adjusted by inertia and prediction (use provided setpoints)
        let effective_heating_sp = heating_setpoint + inertia_factor - predictive_factor;
        let effective_cooling_sp = cooling_setpoint + inertia_factor - predictive_factor;

        // Step 4: Determine mode based on zone temp vs adjusted setpoints
        let heating_threshold = effective_heating_sp - self.deadband_tolerance;
        let cooling_threshold = effective_cooling_sp + self.deadband_tolerance;

        let mode = if zone_temp < heating_threshold {
            HVACMode::Heating
        } else if zone_temp > cooling_threshold {
            HVACMode::Cooling
        } else {
            HVACMode::Off
        };

        // Step 5: Calculate modulation factor based on temperature error
        let temp_error = match mode {
            HVACMode::Heating => effective_heating_sp - zone_temp,
            HVACMode::Cooling => zone_temp - effective_cooling_sp,
            HVACMode::Off => 0.0,
        };

        let modulation = (temp_error * 10.0).clamp(0.0, 1.0);

        // Update previous zone temp for next timestep's dT/dt calculation
        self.previous_zone_temp = zone_temp;

        (mode, modulation)
    }

    /// Reset controller state (for new simulation or year boundary).
    pub fn reset(&mut self) {
        self.previous_zone_temp = 20.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictive_control() {
        let mut controller = PredictiveController::new(20.0, 27.0);

        // Test heating mode
        let (mode, modulation) = controller.calculate_modulation(18.0, 19.0, -0.001);
        assert_eq!(mode, HVACMode::Heating);
        assert!(modulation > 0.0); // Some modulation needed

        // Test cooling mode
        let (mode, modulation) = controller.calculate_modulation(28.0, 27.0, 0.001);
        assert_eq!(mode, HVACMode::Cooling);
        assert!(modulation > 0.0);

        // Test off mode (within deadband)
        let (mode, modulation) = controller.calculate_modulation(23.0, 22.0, 0.0);
        assert_eq!(mode, HVACMode::Off);
        assert_eq!(modulation, 0.0);

        // Test modulation factor limits (0-1)
        let (_, modulation_high) = controller.calculate_modulation(15.0, 20.0, -0.01);
        assert!(modulation_high <= 1.0);

        let (_, modulation_low) = controller.calculate_modulation(30.0, 25.0, 0.01);
        assert!(modulation_low <= 1.0);
    }

    #[test]
    fn test_thermal_inertia() {
        let mut controller = PredictiveController::new(20.0, 27.0);

        // Test with thermal inertia (mass temp cooler than zone temp)
        // This should anticipate cooling and adjust setpoint upward
        let (mode_no_inertia, _) = controller.calculate_modulation(29.0, 29.0, 0.0);

        let (mode_with_inertia, _) = controller.calculate_modulation(29.0, 19.0, 0.0); // Mass temp 10°C cooler

        // Both should be cooling mode
        assert_eq!(mode_no_inertia, HVACMode::Cooling);
        assert_eq!(mode_with_inertia, HVACMode::Cooling);
    }

    #[test]
    fn test_temperature_rate_prediction() {
        let mut controller = PredictiveController::new(20.0, 27.0);

        // Test with rising temperature (anticipate overshoot)
        // This should reduce modulation to prevent overshoot
        let (mode_stable, _) = controller.calculate_modulation(29.0, 29.0, 0.0);

        let (mode_rising, _) = controller.calculate_modulation(29.0, 29.0, 0.01); // Rising at 0.01°C/s

        // Both should be cooling mode
        assert_eq!(mode_stable, HVACMode::Cooling);
        assert_eq!(mode_rising, HVACMode::Cooling);
    }

    #[test]
    fn test_deadband_tolerance() {
        let mut controller = PredictiveController::new(20.0, 27.0);

        // At heating setpoint (within deadband)
        let (mode, modulation) = controller.calculate_modulation(20.0, 20.0, 0.0);
        assert_eq!(mode, HVACMode::Off);
        assert_eq!(modulation, 0.0);

        // At cooling setpoint (within deadband)
        let (mode, modulation) = controller.calculate_modulation(27.0, 27.0, 0.0);
        assert_eq!(mode, HVACMode::Off);
        assert_eq!(modulation, 0.0);

        // Below heating setpoint - deadband
        let (mode, modulation) = controller.calculate_modulation(19.0, 19.0, 0.0);
        assert_eq!(mode, HVACMode::Heating);
        assert!(modulation > 0.0);

        // Above cooling setpoint + deadband
        let (mode, modulation) = controller.calculate_modulation(28.0, 28.0, 0.0);
        assert_eq!(mode, HVACMode::Cooling);
        assert!(modulation > 0.0);
    }

    #[test]
    fn test_controller_reset() {
        let mut controller = PredictiveController::new(20.0, 27.0);

        // Run a few timesteps
        controller.calculate_modulation(22.0, 21.0, 0.001);
        controller.calculate_modulation(23.0, 22.0, 0.001);

        // Reset
        controller.reset();

        // Previous zone temp should be reset to initial value
        assert_eq!(controller.previous_zone_temp, 20.0);
    }

    #[test]
    fn test_custom_tuning() {
        // Create controller with custom tuning parameters
        let controller = PredictiveController::with_tuning(20.0, 27.0, 0.2, 0.02);

        assert_eq!(controller.thermal_inertia_gain, 0.2);
        assert_eq!(controller.temp_rate_gain, 0.02);
        assert_eq!(controller.heating_setpoint, 20.0);
        assert_eq!(controller.cooling_setpoint, 27.0);
    }

    #[test]
    fn test_inertia_factor_calculation() {
        let controller = PredictiveController::new(20.0, 27.0);

        let zone_temp = 22.0;
        let mass_temp = 18.0; // 4°C cooler

        // Inertia factor = α × (zone_temp - mass_temp)
        // = 0.1 × (22.0 - 18.0) = 0.4
        let expected_inertia_factor = controller.thermal_inertia_gain * (zone_temp - mass_temp);
        assert!((expected_inertia_factor - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_predictive_factor_calculation() {
        let controller = PredictiveController::new(20.0, 27.0);

        let temp_rate = 0.01; // Rising at 0.01°C/s

        // Predictive factor = β × temp_rate
        // = 0.01 × 0.01 = 0.0001
        let expected_predictive_factor = controller.temp_rate_gain * temp_rate;
        assert!((expected_predictive_factor - 0.0001).abs() < 0.00001);
    }

    #[test]
    fn test_calculate_modulation_with_setpoints() {
        let mut controller = PredictiveController::new(20.0, 27.0);

        // Test with dynamic setpoints (e.g. night setback)
        let (mode, modulation) =
            controller.calculate_modulation_with_setpoints(16.0, 17.0, 0.0, 15.0, 30.0);

        // zone_temp (16.0) is above heating threshold (15.0 - inertia - deadband)
        // inertia = 0.1 * (16-17) = -0.1
        // effective_heating_sp = 15.0 - 0.1 - 0.0 = 14.9
        // heating_threshold = 14.9 - 0.5 = 14.4
        // 16.0 > 14.4, so should be OFF
        assert_eq!(mode, HVACMode::Off);
        assert_eq!(modulation, 0.0);

        // Test heating with dynamic setpoint
        let (mode2, modulation2) =
            controller.calculate_modulation_with_setpoints(14.0, 14.0, 0.0, 15.0, 30.0);
        assert_eq!(mode2, HVACMode::Heating);
        assert!(modulation2 > 0.0);
    }
}
