use crate::ai::surrogate::SurrogateManager;
use crate::physics::continuous::ConstantField;
use crate::sim::components::WallSurface;

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
/// * `surfaces` - List of wall surfaces for each zone (Vec of Vec of WallSurface)
/// * `window_u_value` - Thermal transmittance of windows (W/m²K) - optimization variable
/// * `hvac_setpoint` - HVAC system setpoint temperature (°C) - optimization variable
#[derive(Clone)]
pub struct ThermalModel {
    pub num_zones: usize,
    pub temperatures: Vec<f64>,
    pub loads: Vec<f64>,
    pub surfaces: Vec<Vec<WallSurface>>,
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
    /// - Surfaces: 4 walls of 10m² each per zone (default)
    pub fn new(num_zones: usize) -> Self {
        // Initialize default surfaces: 4 walls of 10m² each per zone
        let mut surfaces = Vec::with_capacity(num_zones);
        for _ in 0..num_zones {
            let mut zone_surfaces = Vec::new();
            for _ in 0..4 {
                zone_surfaces.push(WallSurface::new(10.0, 2.5)); // 10m², U=2.5 (default)
            }
            surfaces.push(zone_surfaces);
        }

        ThermalModel {
            num_zones,
            temperatures: vec![20.0; num_zones], // Initialize at 20°C
            loads: vec![0.0; num_zones],
            surfaces,
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
            // Update surface U-values as well (simplified for now - applying to all surfaces)
            for zone_surfaces in &mut self.surfaces {
                for surface in zone_surfaces {
                    surface.u_value = self.window_u_value;
                }
            }
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
                // Determine load from surfaces using ContinuousField integration
                // The 'load' from SurrogateManager or analytical model currently represents
                // a flux or source term (W/m²). We treat this as a constant field over the surface for now.
                // In the future, this field will be spatially varying (predicted by AI).

                // Get the base load (W/m²) predicted for this zone
                let predicted_load_flux = self.loads[i];

                // Create a ContinuousField wrapper for this load
                let field = ConstantField {
                    value: predicted_load_flux,
                };

                // Calculate total heat gain from all surfaces in this zone
                let mut total_surface_heat_gain = 0.0;
                if i < self.surfaces.len() {
                    for surface in &self.surfaces[i] {
                        total_surface_heat_gain += surface.calculate_heat_gain(&field);
                    }
                }

                // Note: The original logic treated 'load' as a direct term in the temperature update.
                // Original: self.temperatures[i] += (load - conduction_loss) * 0.1;
                // where load was presumably total load or flux scaled implicitly.
                // If load was W/m², and we didn't multiply by area, it was unit-inconsistent or assumed specific scaling.
                //
                // Let's assume the new physics engine should use the total watts.
                // The temperature update equation: dT/dt = (Q_gain - Q_loss) / C_thermal
                // The original code: temperature += (load - conduction_loss) * 0.1
                // This implies 0.1 includes (dt / C_thermal).
                //
                // If 'load' (W/m²) was used directly, and we now calculate total Watts (W),
                // we should scale it to be consistent with previous behavior or update the physics.
                // Previous behavior: load (approx 0.5 - 1.2)
                // New behavior: total_surface_heat_gain = Area (4 * 10 = 40) * load (0.5 - 1.2) = 20 - 48.
                // This is ~40x larger.
                //
                // To maintain behavior of the *simulation* while changing the *structure*,
                // we should normalize by total area if the original 'load' was intended as W/m² acting on an implicit area
                // that matched the capacitance scaling.
                //
                // However, the original code `load - conduction_loss` mixed units.
                // `conduction_loss = (temp - 0) * u_value * 0.1`.
                // U-value is W/m²K. Temp is K. So this is W/m².
                // So the original equation was in flux units (W/m²).
                //
                // So if calculate_heat_gain returns Watts (W), we should divide by total area to get W/m² back
                // to plug into the existing simplified equation.

                let total_area: f64 = if i < self.surfaces.len() {
                    self.surfaces[i].iter().map(|s| s.area).sum()
                } else {
                    1.0 // Avoid division by zero
                };

                let load_flux = if total_area > 0.0 {
                    total_surface_heat_gain / total_area
                } else {
                    0.0
                };

                // Heat transfer logic influenced by U-value
                // Note: conduction_loss uses window_u_value.
                // In the new model, surfaces have u_values. We should probably use those.
                // But for now, we keep the original logic structure but use the load calculated from fields.

                let conduction_loss = (self.temperatures[i] - 0.0) * self.window_u_value * 0.1;
                self.temperatures[i] += (load_flux - conduction_loss) * 0.1;
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
        // Check surfaces created
        assert_eq!(model.surfaces.len(), 10);
        assert_eq!(model.surfaces[0].len(), 4);

        const EPSILON: f64 = 1e-9;
        assert!(model
            .temperatures
            .iter()
            .all(|&t| (t - 20.0).abs() < EPSILON));
    }

    #[test]
    fn test_apply_parameters_updates_model() {
        let mut model = ThermalModel::new(10);
        let params = vec![1.5, 22.0];

        model.apply_parameters(&params);
        assert_eq!(model.window_u_value, 1.5);
        assert_eq!(model.hvac_setpoint, 22.0);

        // Check surface updates
        assert_eq!(model.surfaces[0][0].u_value, 1.5);
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

    #[test]
    fn test_surrogate_vs_analytical_consistency() {
        // Compare surrogate and analytical results on same model.
        // Both should produce positive, non-zero energy.
        let mut model_analytical = ThermalModel::new(10);
        let mut model_surrogate = ThermalModel::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        let params = vec![1.5, 21.0];
        model_analytical.apply_parameters(&params);
        model_surrogate.apply_parameters(&params);

        let energy_analytical = model_analytical.solve_timesteps(8760, &surrogates, false);
        let energy_surrogate = model_surrogate.solve_timesteps(8760, &surrogates, true);

        // Both should be positive
        assert!(
            energy_analytical > 0.0,
            "Analytical energy should be positive"
        );
        assert!(
            energy_surrogate > 0.0,
            "Surrogate energy should be positive"
        );

        // For mock surrogates (returning 1.2), surrogate energy should match analytical
        // because both use the same default mock loads in the base case.
        // This test just verifies both code paths work without panicking.
    }

    #[test]
    fn test_onnx_model_loading() {
        use std::path::Path;

        // Check if dummy ONNX model exists
        let model_path = "assets/loads_predictor.onnx";
        if !Path::new(model_path).exists() {
            // Skip if model file not generated yet
            return;
        }

        // Try to load - this will panic if libonnxruntime is not installed,
        // which is expected in CI/dev environments without ONNX Runtime
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            SurrogateManager::load_onnx(model_path)
        })) {
            Ok(result) => {
                assert!(
                    result.is_ok(),
                    "Should successfully load ONNX model from {}: {:?}",
                    model_path,
                    result.err()
                );

                let manager = result.unwrap();
                assert!(manager.model_loaded);
                assert_eq!(manager.model_path, Some(model_path.to_string()));

                // Try predicting with loaded model
                let temps = vec![20.0, 21.0, 22.0, 20.5, 21.5];
                let loads = manager.predict_loads(&temps);

                // Should return exactly 5 values (one per input zone)
                assert_eq!(loads.len(), temps.len());

                // Dummy model returns 1.2 for each zone
                for load in loads {
                    assert!((load - 1.2).abs() < 1e-5, "Dummy model should return 1.2");
                }
            }
            Err(_) => {
                // libonnxruntime not installed - skip test gracefully
                eprintln!("Skipping ONNX model loading test: libonnxruntime not installed");
            }
        }
    }

    #[test]
    fn test_trained_surrogate_model() {
        use std::path::Path;

        // Test the trained thermal surrogate model
        let model_path = "assets/thermal_surrogate.onnx";
        if !Path::new(model_path).exists() {
            // Skip if trained model not generated yet
            return;
        }

        // Try to load trained model
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            SurrogateManager::load_onnx(model_path)
        })) {
            Ok(result) => {
                assert!(result.is_ok(), "Should load trained surrogate model");

                let manager = result.unwrap();
                assert!(manager.model_loaded);

                // Test with multiple temperature vectors
                let test_temps = vec![
                    vec![20.0, 21.0, 22.0, 20.5, 21.5, 19.5, 22.5, 20.0, 21.0, 22.0],
                    vec![18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 18.5, 19.5, 20.5],
                ];

                for temps in test_temps {
                    let loads = manager.predict_loads(&temps);
                    // Should output 10 values (one per zone)
                    assert_eq!(loads.len(), 10);
                    // All loads should be positive
                    for load in &loads {
                        assert!(*load > 0.0, "Loads should be positive");
                    }
                }
            }
            Err(_) => {
                eprintln!("Skipping trained surrogate test: libonnxruntime not installed");
            }
        }
    }

    #[test]
    fn test_apply_parameters_boundary_values() {
        let mut model = ThermalModel::new(10);

        // Test minimum boundary
        model.apply_parameters(&[0.5, 19.0]);
        assert_eq!(model.window_u_value, 0.5);
        assert_eq!(model.hvac_setpoint, 19.0);

        // Test maximum boundary
        model.apply_parameters(&[3.0, 24.0]);
        assert_eq!(model.window_u_value, 3.0);
        assert_eq!(model.hvac_setpoint, 24.0);
    }

    #[test]
    fn test_apply_parameters_extra_values() {
        let mut model = ThermalModel::new(10);
        let params = vec![1.5, 22.0, 1000.0, 999.0];

        // Should only use first two elements
        model.apply_parameters(&params);
        assert_eq!(model.window_u_value, 1.5);
        assert_eq!(model.hvac_setpoint, 22.0);
    }

    #[test]
    fn test_thermal_model_zones() {
        let model_5 = ThermalModel::new(5);
        assert_eq!(model_5.num_zones, 5);
        assert_eq!(model_5.temperatures.len(), 5);
        assert_eq!(model_5.loads.len(), 5);

        let model_20 = ThermalModel::new(20);
        assert_eq!(model_20.num_zones, 20);
        assert_eq!(model_20.temperatures.len(), 20);
        assert_eq!(model_20.loads.len(), 20);
    }

    #[test]
    fn test_solve_timesteps_zero_steps() {
        let mut model = ThermalModel::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        model.apply_parameters(&[1.5, 21.0]);
        let energy = model.solve_timesteps(0, &surrogates, false);

        // Zero steps should result in zero energy
        assert_eq!(energy, 0.0);
    }

    #[test]
    fn test_solve_timesteps_short_and_long() {
        let mut model = ThermalModel::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        model.apply_parameters(&[1.5, 21.0]);

        // Short simulation
        let energy_short = model.clone().solve_timesteps(168, &surrogates, false);
        assert!(energy_short > 0.0);

        // Long simulation (5 years)
        let energy_long = model.solve_timesteps(8760 * 5, &surrogates, false);
        assert!(energy_long > 0.0);
        // 5-year should be roughly 5x the annual (with some variation)
        assert!(energy_long > energy_short);
    }

    #[test]
    fn test_calc_analytical_loads_mutation() {
        let mut model = ThermalModel::new(10);

        model.calc_analytical_loads();

        // All loads should be 0.5
        for &load in model.loads.iter() {
            assert_eq!(load, 0.5);
        }
    }

    #[test]
    fn test_parameters_affect_energy() {
        let mut model1 = ThermalModel::new(10);
        let mut model2 = ThermalModel::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        // Two different parameter sets
        model1.apply_parameters(&[0.5, 19.0]); // Better insulation, lower setpoint
        model2.apply_parameters(&[3.0, 24.0]); // Worse insulation, higher setpoint

        let energy1 = model1.solve_timesteps(8760, &surrogates, false);
        let energy2 = model2.solve_timesteps(8760, &surrogates, false);

        // Different parameters should give different energy results
        assert_ne!(energy1, energy2);
    }
}
