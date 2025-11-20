use rayon::prelude::*;
use crate::ai::surrogate::SurrogateManager;

/// Represents a simplified Thermal Network (RC Network).
/// We derive Clone so we can replicate the base model for batch processing.
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
    pub fn new(num_zones: usize) -> Self {
        ThermalModel {
            num_zones,
            temperatures: vec![20.0; num_zones], // Initialize at 20°C
            loads: vec![0.0; num_zones],
            window_u_value: 2.5, // Default U-value
            hvac_setpoint: 21.0, // Default setpoint
        }
    }

    /// Updates the model parameters based on a "gene" vector from the optimizer.
    /// params[0] -> Window U-Value
    /// params[1] -> HVAC Setpoint
    pub fn apply_parameters(&mut self, params: &[f64]) {
        if params.len() >= 1 { self.window_u_value = params[0]; }
        if params.len() >= 2 { self.hvac_setpoint = params[1]; }
    }

    /// The core physics loop.
    pub fn solve_timesteps(&mut self, steps: usize, surrogates: &SurrogateManager, use_ai: bool) -> f64 {
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
            let energy_step: f64 = self.temperatures.iter()
                .map(|t| (t - self.hvac_setpoint).abs())
                .sum();
            
            total_energy += energy_step;
        }
        
        total_energy
    }

    fn calc_analytical_loads(&mut self) {
        for load in self.loads.iter_mut() {
            *load = 0.5; 
        }
    }
}
