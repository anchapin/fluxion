//! HVAC controller module
//!
//! HVAC mode state machine and IdealHVACController implementation.

/// Determines whether HVAC is actively controlling temperature or just tracking it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HvacSystemMode {
    /// Normal HVAC operation with heating/cooling based on setpoints
    #[default]
    Controlled,
    /// Free-floating mode: no HVAC, track temperatures only
    /// Used for ASHRAE 140 FF cases (600FF, 900FF, 650FF, 950FF)
    FreeFloat,
}

/// HVAC operation mode for dual setpoint control.
///
/// The HVAC system operates in three modes based on zone temperature:
/// - `Heating`: Zone temperature is below heating setpoint
/// - `Cooling`: Zone temperature is above cooling setpoint
/// - `Off`: Zone temperature is within the deadband (between heating and cooling setpoints)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HVACMode {
    Heating,
    Cooling,
    Off,
}

/// Ideal HVAC controller with deadband and staging support.
///
/// This controller implements ASHRAE 140 compliant HVAC control with:
/// - Dual setpoint control (heating and cooling)
/// - Deadband between heating and cooling setpoints
/// - Optional staging for multi-stage systems
/// - Proportional control near setpoints to prevent cycling
#[derive(Clone, Debug)]
pub struct IdealHVACController {
    /// Heating setpoint (°C)
    pub heating_setpoint: f64,
    /// Cooling setpoint (°C)
    pub cooling_setpoint: f64,
    /// Deadband tolerance (°C) - prevents rapid cycling near setpoints
    pub deadband_tolerance: f64,
    /// Number of heating stages (1 = single stage, 2+ = multi-stage)
    pub heating_stages: u8,
    /// Number of cooling stages (1 = single stage, 2+ = multi-stage)
    pub cooling_stages: u8,
    /// Maximum heating capacity per stage (W)
    pub heating_capacity_per_stage: f64,
    /// Maximum cooling capacity per stage (W)
    pub cooling_capacity_per_stage: f64,
}

impl IdealHVACController {
    /// Creates a new ideal HVAC controller with specified setpoints.
    pub fn new(heating_setpoint: f64, cooling_setpoint: f64) -> Self {
        Self {
            heating_setpoint,
            cooling_setpoint,
            deadband_tolerance: 0.5,
            heating_stages: 1,
            cooling_stages: 1,
            heating_capacity_per_stage: 100_000.0,
            cooling_capacity_per_stage: 100_000.0,
        }
    }

    /// Creates a controller with staging support.
    pub fn with_stages(
        heating_setpoint: f64,
        cooling_setpoint: f64,
        heating_stages: u8,
        cooling_stages: u8,
        heating_capacity_per_stage: f64,
        cooling_capacity_per_stage: f64,
    ) -> Self {
        Self {
            heating_setpoint,
            cooling_setpoint,
            deadband_tolerance: 0.5,
            heating_stages,
            cooling_stages,
            heating_capacity_per_stage,
            cooling_capacity_per_stage,
        }
    }

    /// Returns the current HVAC mode based on zone temperature.
    pub fn determine_mode(&self, zone_temp: f64) -> HVACMode {
        let heating_threshold = self.heating_setpoint - self.deadband_tolerance;
        let cooling_threshold = self.cooling_setpoint + self.deadband_tolerance;

        if zone_temp < heating_threshold {
            HVACMode::Heating
        } else if zone_temp > cooling_threshold {
            HVACMode::Cooling
        } else {
            HVACMode::Off
        }
    }

    /// Calculates the required HVAC power (W) to maintain setpoint.
    pub fn calculate_power(&self, _zone_temp: f64, free_float_temp: f64, sensitivity: f64) -> f64 {
        let mode = self.determine_mode(free_float_temp);

        match mode {
            HVACMode::Heating => {
                let target_temp = self.heating_setpoint + self.deadband_tolerance;
                let temp_deficit = target_temp - free_float_temp;
                let power_needed = temp_deficit / sensitivity;
                let max_power = self.heating_capacity_per_stage * self.heating_stages as f64;
                power_needed.clamp(0.0, max_power)
            }
            HVACMode::Cooling => {
                let target_temp = self.cooling_setpoint - self.deadband_tolerance;
                let temp_excess = free_float_temp - target_temp;
                let power_needed = temp_excess / sensitivity;
                let max_power = self.cooling_capacity_per_stage * self.cooling_stages as f64;
                (-power_needed).clamp(-max_power, 0.0)
            }
            HVACMode::Off => 0.0,
        }
    }

    /// Returns the number of active heating stages for the given power output.
    pub fn active_heating_stages(&self, power_watts: f64) -> u8 {
        if power_watts <= 0.0 || self.heating_stages == 0 {
            return 0;
        }
        let stages_needed = (power_watts / self.heating_capacity_per_stage).ceil() as u8;
        stages_needed.min(self.heating_stages)
    }

    /// Returns the number of active cooling stages for the given power output.
    pub fn active_cooling_stages(&self, power_watts: f64) -> u8 {
        if power_watts >= 0.0 || self.cooling_stages == 0 {
            return 0;
        }
        let stages_needed = (power_watts.abs() / self.cooling_capacity_per_stage).ceil() as u8;
        stages_needed.min(self.cooling_stages)
    }

    /// Validates that the setpoints form a valid deadband.
    pub fn validate(&self) -> Result<(), String> {
        let deadband = self.cooling_setpoint - self.heating_setpoint;
        if deadband < 2.0 * self.deadband_tolerance {
            return Err(format!(
                "Invalid deadband: cooling setpoint ({:.1}°C) must be at least {:.1}°C above heating setpoint ({:.1}°C)",
                self.cooling_setpoint,
                2.0 * self.deadband_tolerance,
                self.heating_setpoint
            ));
        }
        Ok(())
    }
}

impl Default for IdealHVACController {
    fn default() -> Self {
        Self::new(20.0, 27.0)
    }
}
