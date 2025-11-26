use crate::ai::surrogate::SurrogateManager;
use crate::physics::cta::VectorField;
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
    pub temperatures: VectorField,
    pub loads: VectorField,
    pub surfaces: Vec<Vec<WallSurface>>,
    // Simulation parameters that might be optimized
    pub window_u_value: f64,
    pub hvac_setpoint: f64,
    // New fields for 5R1C model
    pub mass_temperatures: VectorField,   // Tm (Mass temperature)
    pub thermal_capacitance: VectorField, // Cm (J/K)
    pub hvac_cooling_capacity: VectorField, // Watts
    pub hvac_heating_capacity: VectorField, // Watts

    // 5R1C Conductances (W/K)
    pub h_tr_em: VectorField, // Transmission: Exterior -> Mass
    pub h_tr_ms: VectorField, // Transmission: Mass -> Surface
    pub h_tr_is: VectorField, // Transmission: Surface -> Interior
    pub h_tr_w: VectorField,  // Transmission: Exterior -> Interior (Windows)
    pub h_ve: VectorField,    // Ventilation: Exterior -> Interior
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
            temperatures: VectorField::from_scalar(20.0, num_zones), // Initialize at 20°C
            mass_temperatures: VectorField::from_scalar(20.0, num_zones), // Initialize Tm at 20°C
            loads: VectorField::from_scalar(0.0, num_zones),
            surfaces,
            window_u_value: 2.5, // Default U-value
            hvac_setpoint: 21.0, // Default setpoint
            thermal_capacitance: VectorField::from_scalar(10e6, num_zones), // Default capacitance: 10 MJ/K per zone
            hvac_cooling_capacity: VectorField::from_scalar(5000.0, num_zones), // Default: 5kW cooling per zone
            hvac_heating_capacity: VectorField::from_scalar(5000.0, num_zones), // Default: 5kW heating per zone

            // Initialize 5R1C conductances (Defaults for 40m² surface/zone)
            // H_tr_w: Window transmission (approx 2.5 * 10m²)
            h_tr_w: VectorField::from_scalar(25.0, num_zones),
            // H_tr_em: Opaque transmission (Exterior -> Mass)
            h_tr_em: VectorField::from_scalar(50.0, num_zones),
            // H_tr_ms: Mass -> Surface (High coupling)
            h_tr_ms: VectorField::from_scalar(1000.0, num_zones),
            // H_tr_is: Surface -> Interior (High coupling)
            h_tr_is: VectorField::from_scalar(200.0, num_zones),
            // H_ve: Ventilation (Exterior -> Interior)
            h_ve: VectorField::from_scalar(10.0, num_zones),
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

            // Update 5R1C conductances based on new U-value
            // Assuming 10m² window and 30m² opaque wall per zone (Total 40m²)
            // CTA: Use scalar broadcast instead of loop
            self.h_tr_w = VectorField::from_scalar(self.window_u_value * 10.0, self.num_zones);
            self.h_tr_em = VectorField::from_scalar(self.window_u_value * 30.0, self.num_zones);
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
        for t in 0..steps {
            let hour_of_day = t % 24;
            let daily_cycle = (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
            let outdoor_temp = 10.0 + 10.0 * daily_cycle;
            total_energy += self.solve_single_step(t, outdoor_temp, use_ai, surrogates, true);
        }
        total_energy
    }

    /// Solves a single timestep of the thermal simulation.
    ///
    /// # Returns
    /// HVAC energy consumption for the timestep in kWh.
    pub fn solve_single_step(
        &mut self,
        timestep: usize,
        outdoor_temp: f64,
        use_ai: bool,
        surrogates: &SurrogateManager,
        use_analytical_gains: bool,
    ) -> f64 {
        let dt = 3600.0; // Timestep in seconds (1 hour)

        // 1. Calculate External Loads (Solar, etc.)
        if use_ai {
            let pred = surrogates.predict_loads(self.temperatures.as_slice());
            self.loads = VectorField::new(pred);
        } else {
            self.calc_analytical_loads(timestep, use_analytical_gains);
        }

        // 2. Solve Thermal Network (5R1C) with CTA
        let t_e = VectorField::from_scalar(outdoor_temp, self.num_zones);
        let phi_ia = self.loads.clone() * 0.5;
        let phi_st = self.loads.clone() * 0.5;
        let t_m_prev = self.mass_temperatures.clone();

        let h_ext = self.h_tr_w.clone() + self.h_ve.clone();
        let den = (self.h_tr_ms.clone() * self.h_tr_is.clone())
            + (self.h_tr_ms.clone() * h_ext.clone())
            + (self.h_tr_is.clone() * h_ext.clone());
        let term_rest_1 = self.h_tr_ms.clone() + self.h_tr_is.clone();

        let num_tm = self.h_tr_ms.clone() * self.h_tr_is.clone() * t_m_prev.clone();
        let num_phi_st = self.h_tr_is.clone() * phi_st.clone();
        let num_rest = term_rest_1.clone() * ((h_ext.clone() * t_e.clone()) + phi_ia.clone());
        let t_i_free = (num_tm.clone() + num_phi_st.clone() + num_rest) / den.clone();

        // 3. HVAC Calculation
        let t_set = self.hvac_setpoint;
        let sensitivity = (self.h_tr_ms.clone() + self.h_tr_is.clone()) / den.clone();
        let mut hvac_output_data = Vec::with_capacity(self.num_zones);

        for i in 0..self.num_zones {
            let val_free = t_i_free[i];
            let sens = sensitivity[i];
            let mut out = 0.0;

            if val_free < t_set {
                let q_req = (t_set - val_free) / sens;
                out = q_req.min(self.hvac_heating_capacity[i]);
            } else if val_free > t_set {
                let q_req = (val_free - t_set) / sens;
                out = -q_req.min(self.hvac_cooling_capacity[i]);
            }
            hvac_output_data.push(out);
        }
        let hvac_output = VectorField::new(hvac_output_data);
        let hvac_energy_for_step = hvac_output.as_slice().iter().map(|o| o.abs()).sum::<f64>() * dt;

        // 4. Update Temperatures
        let phi_ia_act = phi_ia + hvac_output;
        let num_rest_act = term_rest_1 * ((h_ext.clone() * t_e.clone()) + phi_ia_act);
        let t_i_act = (num_tm + num_phi_st + num_rest_act) / den.clone();

        let ts_num = (self.h_tr_ms.clone() * t_m_prev.clone())
            + (self.h_tr_is.clone() * t_i_act.clone())
            + phi_st;
        let ts_den = self.h_tr_ms.clone() + self.h_tr_is.clone();
        let t_s_act = ts_num / ts_den;

        let term_1 = self.h_tr_em.clone() * (t_e - t_m_prev.clone());
        let term_2 = self.h_tr_ms.clone() * (t_s_act - t_m_prev.clone());
        let q_m_net = term_1 + term_2;
        let dt_m = (q_m_net / self.thermal_capacitance.clone()) * dt;
        let t_m_new = t_m_prev + dt_m;

        self.temperatures = t_i_act;
        self.mass_temperatures = t_m_new;

        hvac_energy_for_step / 3.6e6 // Return kWh
    }

    /// Calculate analytical thermal loads without neural surrogates.
    fn calc_analytical_loads(&mut self, timestep: usize, use_analytical_gains: bool) {
        let mut total_gain = 0.0;
        if use_analytical_gains {
            let hour_of_day = timestep % 24;
            let daily_cycle = (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
            let solar_gain = (1000.0 * daily_cycle).max(0.0);
            let internal_gains = 100.0;
            total_gain = solar_gain + internal_gains;
        }
        self.loads = VectorField::from_scalar(total_gain, self.num_zones);
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
        model.calc_analytical_loads(12, true); // noon

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

        model.calc_analytical_loads(0, true);

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

    mod validation {
        use super::*;
        use crate::ai::surrogate::SurrogateManager;
        use crate::physics::cta::VectorField;

        #[test]
        fn steady_state_heat_transfer_matches_analytical() {
            // --- Common setup ---
            let mut model = ThermalModel::new(1);
            let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

            let h_tr_em = model.h_tr_em[0];
            let h_tr_ms = model.h_tr_ms[0];
            let h_tr_is = model.h_tr_is[0];
            let h_tr_w = model.h_tr_w[0];
            let h_ve = model.h_ve[0];

            // U_opaque is the equivalent conductance for the opaque envelope components (3 resistors in series)
            let u_opaque = 1.0 / (1.0 / h_tr_em + 1.0 / h_tr_ms + 1.0 / h_tr_is);
            let h_total = u_opaque + h_tr_w + h_ve;

            // --- Test Heating ---
            let outdoor_temp_heating = 10.0; // °C
            let setpoint_heating = 20.0; // °C

            // To achieve steady-state, mass temp must be at its equilibrium value, not the air temp
            // H_ms_is is the equivalent conductance of the mass-to-surface and surface-to-air resistors
            let h_ms_is = 1.0 / (1.0 / h_tr_ms + 1.0 / h_tr_is);
            let t_m_steady_state_heating =
                (h_tr_em * outdoor_temp_heating + h_ms_is * setpoint_heating) / (h_tr_em + h_ms_is);

            model.hvac_setpoint = setpoint_heating;
            model.temperatures = VectorField::from_scalar(setpoint_heating, 1);
            model.mass_temperatures = VectorField::from_scalar(t_m_steady_state_heating, 1);

            let energy_kwh =
                model.solve_single_step(0, outdoor_temp_heating, false, &surrogates, false);
            let energy_watts = energy_kwh * 1000.0;

            let analytical_load = h_total * (setpoint_heating - outdoor_temp_heating);

            let relative_error = (energy_watts - analytical_load).abs() / analytical_load;
            assert!(
                relative_error < 0.00001,
                "Heating: Analytical vs. Simulated load mismatch. Analytical: {:.2}, Simulated: {:.2}, Rel Error: {:.5}%",
                analytical_load,
                energy_watts,
                relative_error * 100.0
            );

            // --- Test Cooling ---
            let outdoor_temp_cooling = 30.0; // °C
            let setpoint_cooling = 22.0; // °C

            // Calculate steady-state mass temp for cooling scenario
            let t_m_steady_state_cooling =
                (h_tr_em * outdoor_temp_cooling + h_ms_is * setpoint_cooling) / (h_tr_em + h_ms_is);

            model.hvac_setpoint = setpoint_cooling;
            model.temperatures = VectorField::from_scalar(setpoint_cooling, 1);
            model.mass_temperatures = VectorField::from_scalar(t_m_steady_state_cooling, 1);

            let energy_kwh_cool =
                model.solve_single_step(0, outdoor_temp_cooling, false, &surrogates, false);
            let energy_watts_cool = energy_kwh_cool * 1000.0;
            let analytical_load_cool = h_total * (outdoor_temp_cooling - setpoint_cooling);

            let relative_error_cool =
                (energy_watts_cool - analytical_load_cool).abs() / analytical_load_cool;
            assert!(
                relative_error_cool < 0.01,
                "Cooling: Analytical vs. Simulated load mismatch. Analytical: {:.2}, Simulated: {:.2}, Rel Error: {:.5}%",
                analytical_load_cool,
                energy_watts_cool,
                relative_error_cool * 100.0
            );
        }

        #[test]
        fn zero_load_when_no_temperature_difference() {
            let mut model = ThermalModel::new(1);
            let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

            let outdoor_temp = 20.0;
            let setpoint = 20.0;
            model.hvac_setpoint = setpoint;
            model.temperatures = VectorField::from_scalar(setpoint, 1);
            model.mass_temperatures = VectorField::from_scalar(setpoint, 1);

            let energy_kwh = model.solve_single_step(0, outdoor_temp, false, &surrogates, false);

            assert!(
                energy_kwh.abs() < 1e-9,
                "HVAC load should be zero when outdoor and setpoint temperatures are equal."
            );
        }
    }
}
