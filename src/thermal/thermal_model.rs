//! Multi-zone thermal model implementation.
//!
//! This module extends the single-zone thermal model to support N zones
//! using the N×5R1C thermal network pattern.

use crate::physics::cta::VectorField;

/// Multi-zone thermal model supporting N zones.
#[derive(Debug, Clone)]
pub struct ThermalModel {
    /// Number of thermal zones
    pub num_zones: usize,

    /// Zone air temperatures (°C)
    pub temperatures: VectorField,

    /// Zone mass temperatures (°C)
    pub mass_temperatures: VectorField,

    /// Zone-specific heating setpoints (°C)
    pub heating_setpoints: VectorField,

    /// Zone-specific cooling setpoints (°C)
    pub cooling_setpoints: VectorField,

    /// Inter-zone conductance values (W/K)
    pub h_tr_iz: VectorField,

    /// Zone thermal capacitances (J/K)
    pub thermal_capacitances: VectorField,
}

impl ThermalModel {
    /// Create a new ThermalModel with N zones.
    ///
    /// # Arguments
    /// * `num_zones` - Number of thermal zones
    /// * `initial_temperature` - Initial temperature for all zones (°C)
    ///
    /// # Returns
    /// A new ThermalModel instance with all zones initialized
    pub fn new(num_zones: usize, initial_temperature: f64) -> Self {
        ThermalModel {
            num_zones,
            temperatures: VectorField::from_scalar(initial_temperature, num_zones),
            mass_temperatures: VectorField::from_scalar(initial_temperature, num_zones),
            heating_setpoints: VectorField::from_scalar(20.0, num_zones), // Default heating setpoint
            cooling_setpoints: VectorField::from_scalar(24.0, num_zones), // Default cooling setpoint
            h_tr_iz: VectorField::from_scalar(0.0, num_zones), // Initialize with zero conductance
            thermal_capacitances: VectorField::from_scalar(1000000.0, num_zones), // Default capacitance
        }
    }

    /// Create a new ThermalModel with specified zone properties.
    ///
    /// # Arguments
    /// * `num_zones` - Number of thermal zones
    /// * `temperatures` - Initial zone temperatures
    /// * `mass_temperatures` - Initial zone mass temperatures
    /// * `heating_setpoints` - Zone heating setpoints
    /// * `cooling_setpoints` - Zone cooling setpoints
    /// * `h_tr_iz` - Inter-zone conductance values
    /// * `thermal_capacitances` - Zone thermal capacitances
    pub fn with_properties(
        num_zones: usize,
        temperatures: Vec<f64>,
        mass_temperatures: Vec<f64>,
        heating_setpoints: Vec<f64>,
        cooling_setpoints: Vec<f64>,
        h_tr_iz: Vec<f64>,
        thermal_capacitances: Vec<f64>,
    ) -> Self {
        ThermalModel {
            num_zones,
            temperatures: VectorField::new(temperatures),
            mass_temperatures: VectorField::new(mass_temperatures),
            heating_setpoints: VectorField::new(heating_setpoints),
            cooling_setpoints: VectorField::new(cooling_setpoints),
            h_tr_iz: VectorField::new(h_tr_iz),
            thermal_capacitances: VectorField::new(thermal_capacitances),
        }
    }

    /// Get current zone temperatures.
    pub fn get_temperatures(&self) -> Vec<f64> {
        self.temperatures.as_slice().to_vec()
    }

    /// Set zone temperatures.
    pub fn set_temperatures(&mut self, temperatures: Vec<f64>) {
        self.temperatures = VectorField::new(temperatures);
    }

    /// Get zone thermal capacitances.
    pub fn get_thermal_capacitances(&self) -> Vec<f64> {
        self.thermal_capacitances.as_slice().to_vec()
    }

    /// Set zone thermal capacitances.
    pub fn set_thermal_capacitances(&mut self, capacitances: Vec<f64>) {
        self.thermal_capacitances = VectorField::new(capacitances);
    }

    /// Get inter-zone conductance values.
    pub fn get_inter_zone_conductance(&self) -> Vec<f64> {
        self.h_tr_iz.as_slice().to_vec()
    }

    /// Set inter-zone conductance values.
    pub fn set_inter_zone_conductance(&mut self, h_tr_iz: Vec<f64>) {
        self.h_tr_iz = VectorField::new(h_tr_iz);
    }

    /// Get zones as a vector (for compatibility with performance metrics)
    pub fn zones(&self) -> Vec<usize> {
        (0..self.num_zones).collect()
    }

    /// Step the thermal model physics (placeholder implementation)
    pub fn step_physics(&mut self, _zone_index: usize, _outdoor_temp: f64, _timestep: f64) {
        // Placeholder: In a real implementation, this would update the thermal state
        // For now, we'll just leave temperatures unchanged
    }

    /// Step the thermal model (placeholder implementation)
    pub fn step(&mut self, _timestep: f64, _outdoor_temp: f64, _heating: f64, _cooling: f64) {
        // Placeholder: In a real implementation, this would update the thermal state
        // For now, we'll just leave temperatures unchanged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_model_creation() {
        let model = ThermalModel::new(3, 20.0);
        assert_eq!(model.num_zones, 3);
        assert_eq!(model.get_temperatures(), vec![20.0, 20.0, 20.0]);
        assert_eq!(
            model.get_thermal_capacitances(),
            vec![1000000.0, 1000000.0, 1000000.0]
        );
    }

    #[test]
    fn test_thermal_model_with_properties() {
        let model = ThermalModel::with_properties(
            2,
            vec![20.0, 22.0],
            vec![19.0, 21.0],
            vec![20.0, 21.0],
            vec![24.0, 25.0],
            vec![50.0, 60.0],
            vec![800000.0, 900000.0],
        );
        assert_eq!(model.num_zones, 2);
        assert_eq!(model.get_temperatures(), vec![20.0, 22.0]);
        assert_eq!(model.get_inter_zone_conductance(), vec![50.0, 60.0]);
    }

    #[test]
    fn test_set_temperatures() {
        let mut model = ThermalModel::new(2, 20.0);
        model.set_temperatures(vec![25.0, 26.0]);
        assert_eq!(model.get_temperatures(), vec![25.0, 26.0]);
    }

    #[test]
    fn test_set_thermal_capacitances() {
        let mut model = ThermalModel::new(2, 20.0);
        model.set_thermal_capacitances(vec![500000.0, 600000.0]);
        assert_eq!(model.get_thermal_capacitances(), vec![500000.0, 600000.0]);
    }

    #[test]
    fn test_set_inter_zone_conductance() {
        let mut model = ThermalModel::new(2, 20.0);
        model.set_inter_zone_conductance(vec![30.0, 40.0]);
        assert_eq!(model.get_inter_zone_conductance(), vec![30.0, 40.0]);
    }
}
