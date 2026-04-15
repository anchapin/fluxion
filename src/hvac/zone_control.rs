//! Zone-level HVAC control logic.
//!
//! This module implements independent HVAC control for each thermal zone
//! based on current temperatures and configured setpoints.

use std::sync::Arc;

use crate::physics::cta::VectorField;
use crate::thermal::thermal_model::ThermalModel;

/// HVAC system status for a zone.
#[derive(Debug, Clone, PartialEq)]
pub enum HVACStatus {
    Heating,
    Cooling,
    Off,
}

/// Zone-level HVAC control system.
#[derive(Debug)]
pub struct ZoneControl {
    /// Reference to the thermal model
    pub thermal_model: Arc<ThermalModel>,

    /// Zone setpoints configuration
    setpoints: crate::hvac::zone_setpoints::ZoneSetpoints,

    /// Current HVAC status for each zone
    zone_status: VectorField,
}

impl ZoneControl {
    /// Create a new ZoneControl instance.
    ///
    /// # Arguments
    /// * `thermal_model` - Arc-wrapped thermal model
    /// * `setpoints` - Zone setpoints configuration
    ///
    /// # Returns
    /// A new ZoneControl instance
    pub fn new(
        thermal_model: Arc<ThermalModel>,
        setpoints: crate::hvac::zone_setpoints::ZoneSetpoints,
    ) -> Self {
        let num_zones = thermal_model.num_zones;
        ZoneControl {
            thermal_model,
            setpoints,
            zone_status: VectorField::from_scalar(0.0, num_zones), // 0.0 = Off, 1.0 = Heating, -1.0 = Cooling
        }
    }

    /// Update HVAC controls for all zones based on current temperatures.
    ///
    /// # Arguments
    /// * `current_temperatures` - Current zone temperatures
    ///
    /// # Returns
    /// Energy input vector (Watts) for each zone
    pub fn update_zone_controls(&mut self, current_temperatures: &VectorField) -> VectorField {
        let mut energy_input = VectorField::from_scalar(0.0, self.thermal_model.num_zones);

        for zone_id in 0..self.thermal_model.num_zones {
            let current_temp = current_temperatures.as_slice()[zone_id];
            let heating_setpoint = self.setpoints.get_heating_setpoint(zone_id);
            let cooling_setpoint = self.setpoints.get_cooling_setpoint(zone_id);
            let deadband = self.setpoints.get_deadband(zone_id);

            let status = self.determine_hvac_status(
                zone_id,
                current_temp,
                heating_setpoint,
                cooling_setpoint,
                deadband,
            );

            // Update status tracking
            self.zone_status.as_mut_slice()[zone_id] = match status {
                HVACStatus::Heating => 1.0,
                HVACStatus::Cooling => -1.0,
                HVACStatus::Off => 0.0,
            };

            // Calculate energy input based on status
            let energy = self.calculate_energy_input(zone_id, current_temp, &status);
            energy_input.as_mut_slice()[zone_id] = energy;
        }

        energy_input
    }

    /// Determine HVAC status for a zone.
    fn determine_hvac_status(
        &self,
        _zone_id: usize,
        current_temp: f64,
        heating_setpoint: f64,
        cooling_setpoint: f64,
        deadband: f64,
    ) -> HVACStatus {
        let heating_threshold = heating_setpoint - deadband / 2.0;
        let cooling_threshold = cooling_setpoint + deadband / 2.0;

        if current_temp < heating_threshold {
            HVACStatus::Heating
        } else if current_temp > cooling_threshold {
            HVACStatus::Cooling
        } else {
            HVACStatus::Off
        }
    }

    /// Get current HVAC status for a zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zone index (0-based)
    ///
    /// # Returns
    /// HVACStatus enum value
    pub fn get_zone_hvac_status(&self, zone_id: usize) -> HVACStatus {
        let status_value = self.zone_status.as_slice()[zone_id];
        if status_value > 0.0 {
            HVACStatus::Heating
        } else if status_value < 0.0 {
            HVACStatus::Cooling
        } else {
            HVACStatus::Off
        }
    }

    /// Calculate energy input for a zone.
    ///
    /// # Arguments
    /// * `zone_id` - Zone index (0-based)
    /// * `current_temp` - Current zone temperature (°C)
    /// * `status` - Current HVAC status
    ///
    /// # Returns
    /// Energy input in Watts
    pub fn calculate_energy_input(
        &self,
        zone_id: usize,
        current_temp: f64,
        status: &HVACStatus,
    ) -> f64 {
        match status {
            HVACStatus::Heating => {
                let heating_setpoint = self.setpoints.get_heating_setpoint(zone_id);
                let temp_diff = heating_setpoint - current_temp;
                // Simple proportional control: 1000W per °C difference
                1000.0 * temp_diff.max(0.0)
            }
            HVACStatus::Cooling => {
                let cooling_setpoint = self.setpoints.get_cooling_setpoint(zone_id);
                let temp_diff = current_temp - cooling_setpoint;
                // Simple proportional control: 1000W per °C difference
                1000.0 * temp_diff.max(0.0)
            }
            HVACStatus::Off => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thermal::thermal_model::ThermalModel;

    #[test]
    fn test_zone_control_creation() {
        let thermal_model = Arc::new(ThermalModel::new(3, 20.0));
        let setpoints = crate::hvac::zone_setpoints::ZoneSetpoints::new(3);
        let zone_control = ZoneControl::new(thermal_model, setpoints);

        // Initial status should be Off for all zones
        for zone_id in 0..3 {
            assert_eq!(zone_control.get_zone_hvac_status(zone_id), HVACStatus::Off);
        }
    }

    #[test]
    fn test_heating_control() {
        let thermal_model = Arc::new(ThermalModel::new(1, 18.0));
        let mut setpoints = crate::hvac::zone_setpoints::ZoneSetpoints::new(1);
        setpoints.set_heating_setpoint(0, 22.0).unwrap();
        setpoints.set_cooling_setpoint(0, 26.0).unwrap();

        let mut zone_control = ZoneControl::new(thermal_model.clone(), setpoints);
        let current_temps = VectorField::from_scalar(18.0, 1);

        let energy_input = zone_control.update_zone_controls(&current_temps);

        assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Heating);
        assert!(energy_input.as_slice()[0] > 0.0);
    }

    #[test]
    fn test_cooling_control() {
        let thermal_model = Arc::new(ThermalModel::new(1, 20.0));
        let mut setpoints = crate::hvac::zone_setpoints::ZoneSetpoints::new(1);
        setpoints.set_heating_setpoint(0, 22.0).unwrap();
        setpoints.set_cooling_setpoint(0, 26.0).unwrap();

        let mut zone_control = ZoneControl::new(thermal_model.clone(), setpoints);
        let current_temps = VectorField::from_scalar(28.0, 1);

        let energy_input = zone_control.update_zone_controls(&current_temps);

        assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Cooling);
        assert!(energy_input.as_slice()[0] > 0.0);
    }

    #[test]
    fn test_deadband_control() {
        let thermal_model = Arc::new(ThermalModel::new(1, 20.0));
        let mut setpoints = crate::hvac::zone_setpoints::ZoneSetpoints::new(1);
        setpoints.set_heating_setpoint(0, 22.0).unwrap();
        setpoints.set_cooling_setpoint(0, 26.0).unwrap();
        setpoints.set_deadband(0, 2.0).unwrap();

        let mut zone_control = ZoneControl::new(thermal_model.clone(), setpoints);
        let current_temps = VectorField::from_scalar(23.0, 1);

        let energy_input = zone_control.update_zone_controls(&current_temps);

        assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Off);
        assert_eq!(energy_input.as_slice()[0], 0.0);
    }

    #[test]
    fn test_independent_zone_control() {
        let thermal_model = Arc::new(ThermalModel::new(2, 20.0));
        let mut setpoints = crate::hvac::zone_setpoints::ZoneSetpoints::new(2);
        setpoints.set_heating_setpoint(0, 22.0).unwrap();
        setpoints.set_cooling_setpoint(0, 26.0).unwrap();
        setpoints.set_heating_setpoint(1, 18.0).unwrap();
        setpoints.set_cooling_setpoint(1, 22.0).unwrap();

        let mut zone_control = ZoneControl::new(thermal_model.clone(), setpoints);
        let current_temps = VectorField::new(vec![19.0, 25.0]); // Zone 0: heating, Zone 1: cooling

        let energy_input = zone_control.update_zone_controls(&current_temps);

        assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Heating);
        assert_eq!(zone_control.get_zone_hvac_status(1), HVACStatus::Cooling);
        assert!(energy_input.as_slice()[0] > 0.0);
        assert!(energy_input.as_slice()[1] > 0.0);
    }

    #[test]
    fn test_energy_calculation() {
        let thermal_model = Arc::new(ThermalModel::new(1, 20.0));
        let mut setpoints = crate::hvac::zone_setpoints::ZoneSetpoints::new(1);
        setpoints.set_heating_setpoint(0, 22.0).unwrap();

        let mut zone_control = ZoneControl::new(thermal_model.clone(), setpoints);
        let current_temps = VectorField::from_scalar(18.0, 1);

        let energy_input = zone_control.update_zone_controls(&current_temps);

        // 4°C difference (22°C setpoint - 18°C current) * 1000W/°C = 4000W
        assert_eq!(energy_input.as_slice()[0], 4000.0);
    }

    #[test]
    fn test_hvac_status_transitions() {
        let thermal_model = Arc::new(ThermalModel::new(1, 20.0));
        let mut setpoints = crate::hvac::zone_setpoints::ZoneSetpoints::new(1);
        setpoints.set_heating_setpoint(0, 22.0).unwrap();
        setpoints.set_cooling_setpoint(0, 26.0).unwrap();

        let mut zone_control = ZoneControl::new(thermal_model.clone(), setpoints);

        // Start with heating
        let mut current_temps = VectorField::from_scalar(20.0, 1);
        zone_control.update_zone_controls(&current_temps);
        assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Heating);

        // Transition to deadband
        current_temps = VectorField::from_scalar(23.0, 1);
        zone_control.update_zone_controls(&current_temps);
        assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Off);

        // Transition to cooling (above cooling threshold of 27.0°C)
        current_temps = VectorField::from_scalar(27.1, 1);
        zone_control.update_zone_controls(&current_temps);
        assert_eq!(zone_control.get_zone_hvac_status(0), HVACStatus::Cooling);
    }
}
