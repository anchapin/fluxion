use crate::ai::surrogate::SurrogateManager;
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
    // New fields for 5R1C model
    pub thermal_capacitance: Vec<f64>, // J/K
    pub hvac_cooling_capacity: Vec<f64>, // Watts
    pub hvac_heating_capacity: Vec<f64>, // Watts
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
            thermal_capacitance: vec![10e6; num_zones], // Default capacitance: 10 MJ/K per zone
            hvac_cooling_capacity: vec![5000.0; num_zones], // Default: 5kW cooling per zone
            hvac_heating_capacity: vec![5000.0; num_zones], // Default: 5kW heating per zone
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
        let dt = 3600.0; // Timestep in seconds (1 hour)

        for t in 0..steps {
            // 1. Calculate External Loads (Solar, etc.)
            if use_ai {
                self.loads = surrogates.predict_loads(&self.temperatures);
            } else {
                self.calc_analytical_loads(t);
            }

            // 2. Solve Thermal Network (5R1C)
            // Pre-calculate outdoor temperature for this timestep, as it's the same for all zones
            let hour_of_day = t % 24;
            let daily_cycle = (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
            let outdoor_temp = 10.0 + 10.0 * daily_cycle;

            for i in 0..self.num_zones {
                // A) Calculate Total Conductive Heat Gain/Loss (from surfaces)
                // This represents heat transfer through walls, windows, etc.
                let mut total_conductive_gain = 0.0;
                if i < self.surfaces.len() {
                    for surface in &self.surfaces[i] {
                        // Q = U * A * (T_out - T_in)
                        total_conductive_gain +=
                            surface.u_value * surface.area * (outdoor_temp - self.temperatures[i]);
                    }
                }

                // B) Get Solar & Internal Gains
                // The `loads` field now represents Q_solar + Q_internal (in Watts)
                let solar_and_internal_gains = self.loads[i];

                // C) Calculate HVAC System Output
                let temp_diff = self.hvac_setpoint - self.temperatures[i];
                let mut hvac_output = 0.0;
                if temp_diff > 0.0 {
                    // Heating needed
                    hvac_output = temp_diff.abs().min(self.hvac_heating_capacity[i]);
                } else if temp_diff < 0.0 {
                    // Cooling needed
                    hvac_output = -temp_diff.abs().min(self.hvac_cooling_capacity[i]);
                }

                // D) Total Heat Flow (Watts)
                // Sum of all heat gains and losses
                let total_heat_flow =
                    total_conductive_gain + solar_and_internal_gains + hvac_output;

                // E) Update Zone Temperature using Euler method
                // dT/dt = Q_total / C
                // T_new = T_old + (dT/dt) * dt
                let temp_delta = (total_heat_flow / self.thermal_capacitance[i]) * dt;
                self.temperatures[i] += temp_delta;

                // F) Accumulate HVAC energy consumption for this step
                total_energy += hvac_output.abs() * dt; // Energy in Joules
            }
        }

        // Normalize energy to kWh for easier interpretation, though the original was dimensionless
        total_energy / 3.6e6
    }

    /// Calculate analytical thermal loads without neural surrogates.
    ///
    /// This is a simplified analytical model for baseline load prediction.
    /// In production, this would incorporate weather data, solar radiation, infiltration, etc.
    fn calc_analytical_loads(&mut self, timestep: usize) {
        // Simulate a simple daily solar gain and outdoor temperature cycle
        let hour_of_day = timestep % 24;
        let daily_cycle = (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();

        // Solar Gain (Watts) - Peaking in the afternoon
        let solar_gain = (1000.0 * daily_cycle).max(0.0);

        for i in 0..self.num_zones {
            // Internal gains (e.g., from equipment, occupants) - constant for now
            let internal_gains = 100.0; // 100W per zone

            // Total external + internal load (in Watts)
            self.loads[i] = solar_gain + internal_gains;

            // In a more complex model, we would also update outdoor air infiltration here,
            // but for now, we handle conduction directly in the solver loop.
            // The solver loop uses a fixed 0°C for conduction, so we'll adjust that next.
            // For now, this `outdoor_temp` is for context and future use.
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
        model.calc_analytical_loads(12); // noon

        // Check if loads are calculated
        assert!(model.loads.iter().all(|&l| l > 0.0));

        // Check against expected values for noon
        let hour_of_day = 12;
        let daily_cycle = (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
        let solar_gain = (1000.0 * daily_cycle).max(0.0);
        let internal_gains = 100.0;
        let expected_load = solar_gain + internal_gains;

        const EPSILON: f64 = 1e-9;
        assert!((model.loads[0] - expected_load).abs() < EPSILON);
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

        model.calc_analytical_loads(0);

        // All loads should be calculated
        for &load in model.loads.iter() {
            assert!(load >= 0.0);
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

    #[test]
    fn test_thermal_lag() {
        let mut model = ThermalModel::new(1);
        model.hvac_setpoint = 999.0; // effectively disable HVAC
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        let mut outdoor_temps = Vec::new();
        let mut indoor_temps = Vec::new();

        for t in 0..48 {
            model.solve_timesteps(1, &surrogates, false);
            indoor_temps.push(model.temperatures[0]);

            let hour_of_day = t % 24;
            let daily_cycle = (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
            outdoor_temps.push(10.0 + 10.0 * daily_cycle);
        }

        let (max_outdoor_hour, _) = outdoor_temps
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let (max_indoor_hour, _) = indoor_temps
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        assert!(
            max_indoor_hour > max_outdoor_hour,
            "Indoor temp peak ({}) should lag outdoor temp peak ({})",
            max_indoor_hour,
            max_outdoor_hour
        );
    }
}
