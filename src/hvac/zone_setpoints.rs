//! Zone-specific HVAC setpoints management.
//!
//! This module provides ZoneSetpoints struct for managing heating/cooling
//! setpoints and deadband configuration for multiple thermal zones.

use crate::physics::cta::VectorField;

/// Zone-specific HVAC setpoints and deadband configuration.
#[derive(Debug, Clone)]
pub struct ZoneSetpoints {
    /// Number of thermal zones
    num_zones: usize,

    /// Heating setpoints for each zone (°C)
    heating_setpoints: VectorField,

    /// Cooling setpoints for each zone (°C)
    cooling_setpoints: VectorField,

    /// Deadband values for each zone (°C)
    deadbands: VectorField,
}

impl ZoneSetpoints {
    /// Create a new ZoneSetpoints instance with default values.
    ///
    /// # Arguments
    /// * `num_zones` - Number of thermal zones
    ///
    /// # Returns
    /// A new ZoneSetpoints instance with default setpoints (20°C heating, 24°C cooling, 2°C deadband)
    pub fn new(num_zones: usize) -> Self {
        ZoneSetpoints {
            num_zones,
            heating_setpoints: VectorField::from_scalar(20.0, num_zones),
            cooling_setpoints: VectorField::from_scalar(24.0, num_zones),
            deadbands: VectorField::from_scalar(2.0, num_zones),
        }
    }

    /// Set heating setpoint for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zone index (0-based)
    /// * `temperature` - Heating setpoint temperature (°C)
    ///
    /// # Returns
    /// Result indicating success or validation error
    pub fn set_heating_setpoint(&mut self, zone_id: usize, temperature: f64) -> Result<(), String> {
        self.validate_temperature(temperature)?;
        self.validate_zone_id(zone_id)?;
        self.heating_setpoints.set(zone_id, temperature);
        Ok(())
    }

    /// Set cooling setpoint for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zone index (0-based)
    /// * `temperature` - Cooling setpoint temperature (°C)
    ///
    /// # Returns
    /// Result indicating success or validation error
    pub fn set_cooling_setpoint(&mut self, zone_id: usize, temperature: f64) -> Result<(), String> {
        self.validate_temperature(temperature)?;
        self.validate_zone_id(zone_id)?;
        self.cooling_setpoints.set(zone_id, temperature);
        Ok(())
    }

    /// Set deadband for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zone index (0-based)
    /// * `deadband` - Deadband value (°C)
    ///
    /// # Returns
    /// Result indicating success or validation error
    pub fn set_deadband(&mut self, zone_id: usize, deadband: f64) -> Result<(), String> {
        self.validate_deadband(deadband)?;
        self.validate_zone_id(zone_id)?;
        self.deadbands.set(zone_id, deadband);
        Ok(())
    }

    /// Get heating setpoint for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zone index (0-based)
    ///
    /// # Returns
    /// Heating setpoint temperature (°C)
    pub fn get_heating_setpoint(&self, zone_id: usize) -> f64 {
        self.heating_setpoints.get(zone_id)
    }

    /// Get cooling setpoint for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zone index (0-based)
    ///
    /// # Returns
    /// Cooling setpoint temperature (°C)
    pub fn get_cooling_setpoint(&self, zone_id: usize) -> f64 {
        self.cooling_setpoints.get(zone_id)
    }

    /// Get deadband for a specific zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zone index (0-based)
    ///
    /// # Returns
    /// Deadband value (°C)
    pub fn get_deadband(&self, zone_id: usize) -> f64 {
        self.deadbands.get(zone_id)
    }

    /// Validate all setpoints and deadbands.
    ///
    /// # Returns
    /// Result indicating validation success or error message
    pub fn validate_setpoints(&self) -> Result<(), String> {
        for zone_id in 0..self.num_zones {
            let heating = self.get_heating_setpoint(zone_id);
            let cooling = self.get_cooling_setpoint(zone_id);
            let deadband = self.get_deadband(zone_id);

            self.validate_temperature(heating)?;
            self.validate_temperature(cooling)?;
            self.validate_deadband(deadband)?;

            // Ensure heating setpoint is below cooling setpoint
            if heating >= cooling {
                return Err(format!(
                    "Zone {}: Heating setpoint ({}°C) must be below cooling setpoint ({}°C)",
                    zone_id, heating, cooling
                ));
            }

            // Ensure deadband is reasonable relative to setpoint difference
            let setpoint_diff = cooling - heating;
            if deadband > setpoint_diff {
                return Err(format!(
                    "Zone {}: Deadband ({}°C) cannot be larger than setpoint difference ({}°C)",
                    zone_id, deadband, setpoint_diff
                ));
            }
        }
        Ok(())
    }

    /// Validate temperature value.
    fn validate_temperature(&self, temperature: f64) -> Result<(), String> {
        if temperature < 10.0 || temperature > 40.0 {
            Err(format!(
                "Temperature {}°C is out of valid range (10.0°C to 40.0°C)",
                temperature
            ))
        } else {
            Ok(())
        }
    }

    /// Validate deadband value.
    fn validate_deadband(&self, deadband: f64) -> Result<(), String> {
        if deadband <= 0.0 || deadband > 5.0 {
            Err(format!(
                "Deadband {}°C is out of valid range (0.0°C to 5.0°C)",
                deadband
            ))
        } else {
            Ok(())
        }
    }

    /// Validate zone ID.
    fn validate_zone_id(&self, zone_id: usize) -> Result<(), String> {
        if zone_id >= self.num_zones {
            Err(format!(
                "Zone ID {} is out of range (0 to {})",
                zone_id,
                self.num_zones - 1
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_zone_setpoints() {
        let setpoints = ZoneSetpoints::new(3);
        assert_eq!(setpoints.num_zones, 3);
        assert_eq!(setpoints.get_heating_setpoint(0), 20.0);
        assert_eq!(setpoints.get_cooling_setpoint(0), 24.0);
        assert_eq!(setpoints.get_deadband(0), 2.0);
    }

    #[test]
    fn test_set_heating_setpoint() {
        let mut setpoints = ZoneSetpoints::new(2);
        assert!(setpoints.set_heating_setpoint(0, 22.0).is_ok());
        assert_eq!(setpoints.get_heating_setpoint(0), 22.0);
    }

    #[test]
    fn test_set_cooling_setpoint() {
        let mut setpoints = ZoneSetpoints::new(2);
        assert!(setpoints.set_cooling_setpoint(1, 26.0).is_ok());
        assert_eq!(setpoints.get_cooling_setpoint(1), 26.0);
    }

    #[test]
    fn test_set_deadband() {
        let mut setpoints = ZoneSetpoints::new(2);
        assert!(setpoints.set_deadband(0, 3.0).is_ok());
        assert_eq!(setpoints.get_deadband(0), 3.0);
    }

    #[test]
    fn test_validate_setpoints() {
        let setpoints = ZoneSetpoints::new(2);
        assert!(setpoints.validate_setpoints().is_ok());
    }

    #[test]
    fn test_invalid_temperature() {
        let mut setpoints = ZoneSetpoints::new(1);
        assert!(setpoints.set_heating_setpoint(0, 5.0).is_err());
        assert!(setpoints.set_heating_setpoint(0, 45.0).is_err());
    }

    #[test]
    fn test_invalid_deadband() {
        let mut setpoints = ZoneSetpoints::new(1);
        assert!(setpoints.set_deadband(0, 0.0).is_err());
        assert!(setpoints.set_deadband(0, 6.0).is_err());
    }

    #[test]
    fn test_invalid_zone_id() {
        let mut setpoints = ZoneSetpoints::new(2);
        assert!(setpoints.set_heating_setpoint(5, 22.0).is_err());
    }

    #[test]
    fn test_setpoint_order_validation() {
        let mut setpoints = ZoneSetpoints::new(1);
        setpoints.set_heating_setpoint(0, 25.0).unwrap();
        setpoints.set_cooling_setpoint(0, 23.0).unwrap();
        assert!(setpoints.validate_setpoints().is_err());
    }

    #[test]
    fn test_deadband_larger_than_setpoint_diff() {
        let mut setpoints = ZoneSetpoints::new(1);
        setpoints.set_heating_setpoint(0, 20.0).unwrap();
        setpoints.set_cooling_setpoint(0, 24.0).unwrap();
        setpoints.set_deadband(0, 5.0).unwrap();
        assert!(setpoints.validate_setpoints().is_err());
    }
}
