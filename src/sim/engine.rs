use crate::ai::surrogate::SurrogateManager;

#[derive(Clone)]
pub struct ThermalModel {
    pub num_zones: usize,
    pub temperatures: Vec<f64>,
    pub loads: Vec<f64>,
    pub window_u_value: f64,
    pub hvac_setpoint: f64,
}

impl ThermalModel {
    pub fn new(num_zones: usize) -> Self {
        ThermalModel {
            num_zones,
            temperatures: vec![21.0; num_zones],
            loads: vec![0.0; num_zones],
            window_u_value: 2.0,
            hvac_setpoint: 21.0,
        }
    }

    pub fn apply_parameters(&mut self, params: &[f64]) {
        if params.len() >= 1 {
            self.window_u_value = params[0];
        }
        if params.len() >= 2 {
            self.hvac_setpoint = params[1];
        }
    }

    pub fn solve_timesteps(&mut self, steps: usize, use_surrogates: bool) -> f64 {
        let surrogates = SurrogateManager::new();
        let mut total_energy = 0.0;

        for _ in 0..steps {
            if use_surrogates {
                self.loads = surrogates.predict_loads(&self.temperatures);
            } else {
                self.compute_analytical_loads();
            }
            total_energy += self.loads.iter().sum::<f64>();
        }

        total_energy
    }

    fn compute_analytical_loads(&mut self) {
        for i in 0..self.num_zones {
            let temp_diff = self.hvac_setpoint - self.temperatures[i];
            self.loads[i] = temp_diff * self.window_u_value * 10.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_model_creation() {
        let model = ThermalModel::new(5);
        assert_eq!(model.num_zones, 5);
        assert_eq!(model.temperatures.len(), 5);
    }

    #[test]
    fn test_apply_parameters() {
        let mut model = ThermalModel::new(5);
        model.apply_parameters(&vec![1.5, 22.0]);
        assert_eq!(model.window_u_value, 1.5);
        assert_eq!(model.hvac_setpoint, 22.0);
    }

    #[test]
    fn test_solve_timesteps() {
        let mut model = ThermalModel::new(5);
        let energy = model.solve_timesteps(100, false);
        assert!(energy > 0.0);
    }
}
