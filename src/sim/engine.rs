use crate::ai::surrogate::SurrogateManager;

/// Represents a simplified thermal network (RC Network) for building energy modeling.
///
/// This is the core physics engine. It models heat transfer through building zones using
/// resistor-capacitor network approximations. The struct is cloneable to enable batch processing
/// where each parallel thread gets its own instance with independent parameters.
///
/// # Fields
/// * `num_zones` - Number of thermal zones in the building
/// * `temperatures` - Current temperature of each zone (°C)
/// * `loads` - Current thermal loads (W/m²) from environment and internal sources
/// * `window_u_value` - Thermal transmittance of windows (W/m²K) - optimization variable
/// * `hvac_setpoint` - HVAC system setpoint temperature (°C) - optimization variable
#[derive(Clone)]
pub struct ThermalModel {
    pub num_zones: usize,
    pub temperatures: Vec<f64>,
    pub loads: Vec<f64>,
    // Simulation parameters that might be optimized
    pub window_u_value: f64,
    pub hvac_setpoint: f64,
}

impl ThermalModel {
    /// Create a new ThermalModel with specified number of thermal zones.
    ///
    /// # Arguments
    /// * `num_zones` - Number of thermal zones to model
    ///
    /// # Defaults
    /// - All zones initialized to 20°C
    /// - Window U-value: 2.5 W/m²K (typical for double-glazed windows)
    /// - HVAC setpoint: 21°C
    pub fn new(num_zones: usize) -> Self {
        ThermalModel {
            num_zones,
            temperatures: vec![20.0; num_zones], // Initialize at 20°C
            loads: vec![0.0; num_zones],
            window_u_value: 2.5, // Default U-value
            hvac_setpoint: 21.0, // Default setpoint
        }
    }

    /// Updates model parameters based on a gene vector from an optimizer.
    ///
    /// This method maps optimization variables (genes) to physical parameters of the thermal model.
    ///
    /// # Arguments
    /// * `params` - Parameter vector from optimizer:
    ///   - `params[0]`: Window U-value (W/m²K, range: 0.5-3.0)
    ///   - `params[1]`: HVAC setpoint (°C, range: 19-24)
    pub fn apply_parameters(&mut self, params: &[f64]) {
        if !params.is_empty() {
            self.window_u_value = params[0];
        }
        if params.len() >= 2 {
            self.hvac_setpoint = params[1];
        }
    }

    /// Core physics simulation loop for annual building energy performance.
    ///
    /// Simulates hourly thermal dynamics of the building, computing cumulative energy consumption.
    /// Can use either analytical load calculations (exact) or neural network surrogates (fast).
    ///
    /// # Arguments
    /// * `steps` - Number of hourly timesteps (typically 8760 for 1 year)
    /// * `surrogates` - Reference to SurrogateManager for load predictions
    /// * `use_ai` - If true, use neural surrogates; if false, use analytical calculations
    ///
    /// # Returns
    /// Cumulative annual energy use intensity (dimensionless, normalized)
    pub fn solve_timesteps(
        &mut self,
        steps: usize,
        surrogates: &SurrogateManager,
        use_ai: bool,
    ) -> f64 {
        let mut total_energy = 0.0;

        for _t in 0..steps {
            // 1. Calculate Loads
            if use_ai {
                self.loads = surrogates.predict_loads(&self.temperatures);
            } else {
                self.calc_analytical_loads();
            }

            // 2. Solve Thermal Network (State Update)
            // In a batch scenario, this inner loop runs inside a single thread of the Rayon pool
            // We avoid nested parallelism here to prevent thread-pool exhaustion
            for i in 0..self.num_zones {
                let load = self.loads[i];
                // Heat transfer logic influenced by U-value
                let conduction_loss = (self.temperatures[i] - 0.0) * self.window_u_value * 0.1;
                self.temperatures[i] += (load - conduction_loss) * 0.1;
            }

            // Energy calculation (simplified)
            // Energy = proportional to gap between temp and setpoint
            let energy_step: f64 = self
                .temperatures
                .iter()
                .map(|t| (t - self.hvac_setpoint).abs())
                .sum();

            total_energy += energy_step;
        }

        total_energy
    }

    /// Calculate analytical thermal loads without neural surrogates.
    ///
    /// This is a simplified analytical model for baseline load prediction.
    /// In production, this would incorporate weather data, solar radiation, infiltration, etc.
    fn calc_analytical_loads(&mut self) {
        for load in self.loads.iter_mut() {
            *load = 0.5;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ThermalModel;
    use crate::ai::surrogate::SurrogateManager;

    #[test]
    fn test_thermal_model_creation() {
        let model = ThermalModel::new(10);
        assert_eq!(model.num_zones, 10);
        assert_eq!(model.temperatures.len(), 10);
        const EPSILON: f64 = 1e-9;
        assert!(model.temperatures.iter().all(|&t| (t - 20.0).abs() < EPSILON));
    }

    #[test]
    fn test_apply_parameters_updates_model() {
        let mut model = ThermalModel::new(10);
        let params = vec![1.5, 22.0];

        model.apply_parameters(&params);
        assert_eq!(model.window_u_value, 1.5);
        assert_eq!(model.hvac_setpoint, 22.0);
    }

    #[test]
    fn test_apply_parameters_partial() {
        let mut model = ThermalModel::new(10);
        let params = vec![1.5];

        model.apply_parameters(&params);
        assert_eq!(model.window_u_value, 1.5);
        assert_eq!(model.hvac_setpoint, 21.0); // Should remain default
    }

    #[test]
    fn test_solve_timesteps_energy_conservation() {
        let model = ThermalModel::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        // Analytical baseline (no AI)
        let energy_analytical = model.clone().solve_timesteps(8760, &surrogates, false);

        // Should produce non-zero energy
        assert!(energy_analytical > 0.0, "Energy should be non-zero");
    }

    #[test]
    fn test_solve_timesteps_with_surrogates() {
        let model = ThermalModel::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        // Surrogate-based prediction
        let energy_surrogate = model.clone().solve_timesteps(8760, &surrogates, true);

        // Should produce non-zero energy
        assert!(energy_surrogate > 0.0, "Energy should be non-zero");
    }

    #[test]
    fn test_calc_analytical_loads() {
        let mut model = ThermalModel::new(5);
        model.calc_analytical_loads();

        // All loads should be 0.5
        const EPSILON: f64 = 1e-9;
        assert!(model.loads.iter().all(|&l| (l - 0.5).abs() < EPSILON));
    }
}
