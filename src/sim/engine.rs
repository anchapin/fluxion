use crate::ai::surrogate::SurrogateManager;
use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::physics::nd_array::NDArrayField;
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
pub struct ThermalModel<T: ContinuousTensor<f64>> {
    pub num_zones: usize,
    pub temperatures: T,
    pub loads: T,
    pub surfaces: Vec<Vec<WallSurface>>,
    // Simulation parameters that might be optimized
    pub window_u_value: f64,
    pub hvac_setpoint: f64,

    // Physical Constants (Per Zone)
    pub zone_area: T,         // Floor Area (m²)
    pub ceiling_height: T,    // Ceiling Height (m)
    pub air_density: T,       // Air Density (kg/m³)
    pub heat_capacity: T,     // Specific Heat Capacity of Air (J/kg·K)
    pub window_ratio: T,      // Window-to-Wall Ratio (0.0-1.0)
    pub aspect_ratio: T,      // Zone Aspect Ratio (Length/Width)
    pub infiltration_rate: T, // Infiltration Rate (ACH)

    // New fields for 5R1C model
    pub mass_temperatures: T,     // Tm (Mass temperature)
    pub thermal_capacitance: T,   // Cm (J/K) - Includes Air + Structure
    pub hvac_cooling_capacity: T, // Watts
    pub hvac_heating_capacity: T, // Watts

    // 5R1C Conductances (W/K)
    pub h_tr_em: T, // Transmission: Exterior -> Mass
    pub h_tr_ms: T, // Transmission: Mass -> Surface
    pub h_tr_is: T, // Transmission: Surface -> Interior
    pub h_tr_w: T,  // Transmission: Exterior -> Interior (Windows)
    pub h_ve: T,    // Ventilation: Exterior -> Interior
}

impl ThermalModel<VectorField> {
    /// Create a new ThermalModel with specified number of thermal zones.
    ///
    /// # Arguments
    /// * `num_zones` - Number of thermal zones to model
    ///
    /// # Defaults
    /// - All zones initialized to 20°C
    /// - Window U-value: 2.5 W/m²K (typical for double-glazed windows)
    /// - HVAC setpoint: 21°C
    /// - Zone Area: 20 m²
    /// - Ceiling Height: 3.0 m
    /// - Window Ratio: 0.15
    pub fn new(num_zones: usize) -> Self {
        Self::init(num_zones, VectorField::from_scalar)
    }
}

impl ThermalModel<NDArrayField> {
    /// Create a new ThermalModel using NDArray backend.
    pub fn new_ndarray(num_zones: usize) -> Self {
        Self::init(num_zones, |v, n| {
            NDArrayField::from_shape_vec(vec![n], vec![v; n])
        })
    }

    /// Create a new ThermalModel using NDArray backend with a specific N-dimensional shape.
    pub fn new_ndarray_with_shape(shape: Vec<usize>) -> Self {
        let num_zones = shape.iter().product();
        Self::init(num_zones, move |v, _n| {
            NDArrayField::from_shape_vec(shape.clone(), vec![v; num_zones])
        })
    }
}

impl<T: ContinuousTensor<f64> + AsRef<[f64]>> ThermalModel<T> {
    /// Internal helper to initialize a ThermalModel using a tensor factory function.
    fn init(num_zones: usize, from_scalar: impl Fn(f64, usize) -> T) -> Self {
        // Initialize default physical parameters
        let zone_area_val: f64 = 20.0;
        let ceiling_height_val: f64 = 3.0;
        let aspect_ratio_val: f64 = 1.0;
        let window_ratio_val: f64 = 0.15;

        // Calculate geometry for initial surfaces
        let width = (zone_area_val * aspect_ratio_val).sqrt();
        let depth = zone_area_val / width;
        let perimeter = 2.0 * (width + depth);
        let gross_wall_area = perimeter * ceiling_height_val;
        let window_area = gross_wall_area * window_ratio_val;
        // Divide by 4 for per-wall properties in surfaces list
        let win_area_per_side = window_area / 4.0;

        // Initialize default surfaces: 4 walls
        let mut surfaces = Vec::with_capacity(num_zones);
        for _ in 0..num_zones {
            let mut zone_surfaces = Vec::new();
            for _ in 0..4 {
                zone_surfaces.push(WallSurface::new(win_area_per_side, 2.5));
            }
            surfaces.push(zone_surfaces);
        }

        let mut model = ThermalModel {
            num_zones,
            temperatures: from_scalar(20.0, num_zones), // Initialize at 20°C
            mass_temperatures: from_scalar(20.0, num_zones), // Initialize Tm at 20°C
            loads: from_scalar(0.0, num_zones),
            surfaces,
            window_u_value: 2.5, // Default U-value
            hvac_setpoint: 21.0, // Default setpoint

            // Physical Constants Defaults
            zone_area: from_scalar(zone_area_val, num_zones),
            ceiling_height: from_scalar(ceiling_height_val, num_zones),
            air_density: from_scalar(1.2, num_zones),
            heat_capacity: from_scalar(1005.0, num_zones),
            window_ratio: from_scalar(window_ratio_val, num_zones),
            aspect_ratio: from_scalar(aspect_ratio_val, num_zones),
            infiltration_rate: from_scalar(0.5, num_zones), // 0.5 ACH

            // Placeholders (will be updated by update_derived_parameters)
            thermal_capacitance: from_scalar(1.0, num_zones),
            hvac_cooling_capacity: from_scalar(5000.0, num_zones), // Default: 5kW cooling per zone
            hvac_heating_capacity: from_scalar(5000.0, num_zones), // Default: 5kW heating per zone

            h_tr_w: from_scalar(0.0, num_zones),
            h_tr_em: from_scalar(0.0, num_zones),
            h_tr_ms: from_scalar(1000.0, num_zones), // Fixed coupling
            h_tr_is: from_scalar(200.0, num_zones),  // Fixed coupling
            h_ve: from_scalar(0.0, num_zones),
        };

        model.update_derived_parameters();
        model
    }

    /// Updates derived physical parameters based on geometry and constants.
    fn update_derived_parameters(&mut self) {
        // Geometry Calculations
        let width = self
            .zone_area
            .zip_with(&self.aspect_ratio, |a, ar| (a * ar).sqrt());
        let depth = self.zone_area.zip_with(&width, |a, w| a / w);
        let perimeter = (width.clone() + depth) * 2.0;
        let gross_wall_area = perimeter * self.ceiling_height.clone();

        let window_area = gross_wall_area.clone() * self.window_ratio.clone();
        // Opaque area: Gross - Window + Roof (Assume Roof = Floor Area)
        // Note: For 5R1C, h_tr_em typically represents the opaque envelope conductance.
        let opaque_wall_area = gross_wall_area.zip_with(&window_area, |g, w| g - w);
        let total_opaque_area = opaque_wall_area + self.zone_area.clone();

        let volume = self.zone_area.clone() * self.ceiling_height.clone();

        // Update Conductances
        // h_tr_w = U_win * Window Area
        self.h_tr_w = window_area * self.window_u_value;

        // h_tr_em = U_opaque * Opaque Area
        // We use a fixed reference U-value for opaque surfaces (e.g. 0.5 W/m²K)
        // In a full model this might be another parameter.
        self.h_tr_em = total_opaque_area * 0.5;

        // Ventilation
        // h_ve = (infiltration_rate * volume * density * cp) / 3600
        // infiltration_rate is in ACH (1/hr)
        let air_cap = volume * self.air_density.clone() * self.heat_capacity.clone();
        self.h_ve = (air_cap.clone() * self.infiltration_rate.clone()) / 3600.0;

        // Thermal Capacitance (Air + Structure)
        // Structure assumption: 200 kJ/m²K per m² floor area
        let structure_cap = self.zone_area.clone() * 200_000.0;
        self.thermal_capacitance = air_cap + structure_cap;
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
            // Surfaces update for metadata/consistency
            for zone_surfaces in &mut self.surfaces {
                for surface in zone_surfaces {
                    surface.u_value = self.window_u_value;
                }
            }
        }
        if params.len() >= 2 {
            self.hvac_setpoint = params[1];
        }

        // Recalculate derived conductances (h_tr_w, etc.) using new U-values and fixed geometry
        self.update_derived_parameters();
    }

    /// Calculates HVAC power demand based on free-floating temperature and setpoint.
    ///
    /// This function implements the core logic for HVAC power calculation using CTA,
    /// making it reusable and simplifying the main simulation loop.
    ///
    /// # Arguments
    /// * `t_i_free` - The free-floating indoor temperature tensor (i.e., without HVAC).
    /// * `sensitivity` - A tensor representing how much 1W of HVAC power changes the indoor temperature.
    ///
    /// # Returns
    /// A tensor representing the HVAC power (heating is positive, cooling is negative).
    fn hvac_power_demand(&self, t_i_free: &T, sensitivity: &T) -> T {
        let t_set = self.hvac_setpoint;
        let t_err = t_i_free.map(|t| t - t_set); // Error from setpoint

        // Required power to correct the error.
        let q_req = t_err.zip_with(sensitivity, |err, sens| -err / sens);

        // Apply heating/cooling capacities
        q_req
            .zip_with(&self.hvac_heating_capacity, |q, cap| q.min(cap))
            .zip_with(&self.hvac_cooling_capacity, |q, cap| q.max(-cap))
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
    /// Cumulative annual energy use intensity (EUI) in kWh/m²/year.
    pub fn solve_timesteps(
        &mut self,
        steps: usize,
        surrogates: &SurrogateManager,
        use_ai: bool,
    ) -> f64 {
        let total_energy_kwh: f64 = (0..steps)
            .map(|t| {
                let hour_of_day = t % 24;
                let daily_cycle = (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
                let outdoor_temp = 10.0 + 10.0 * daily_cycle;
                self.solve_single_step(t, outdoor_temp, use_ai, surrogates, true)
            })
            .sum();

        // Normalize by total floor area to get EUI
        let total_area = self.zone_area.integrate();
        if total_area > 0.0 {
            total_energy_kwh / total_area
        } else {
            0.0
        }
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
        // 1. Calculate External Loads
        if use_ai {
            let pred = surrogates.predict_loads(self.temperatures.as_ref());
            self.loads = self.temperatures.new_with_data(pred);
        } else {
            self.calc_analytical_loads(timestep, use_analytical_gains);
        }

        self.step_physics(outdoor_temp)
    }

    /// Performs the physics update for a single timestep using current loads and state.
    ///
    /// # Arguments
    /// * `outdoor_temp` - Outdoor temperature in °C.
    ///
    /// # Returns
    /// HVAC energy consumption for the timestep in kWh.
    pub fn step_physics(&mut self, outdoor_temp: f64) -> f64 {
        let dt = 3600.0; // Timestep in seconds (1 hour)

        // Convert loads (W/m²) to Watts
        let loads_watts = self.loads.clone() * self.zone_area.clone();

        // 2. Solve Thermal Network
        let t_e = self.temperatures.constant_like(outdoor_temp);
        // Distribute internal gains (50% air, 50% surface)
        let phi_ia = loads_watts.clone() * 0.5;
        let phi_st = loads_watts.clone() * 0.5;

        // Simplified 5R1C calculation using CTA
        let h_ext = self.h_tr_w.clone() + self.h_ve.clone();
        let den = self.h_tr_ms.clone() * self.h_tr_is.clone()
            + self.h_tr_ms.clone() * h_ext.clone()
            + self.h_tr_is.clone() * h_ext.clone();
        let term_rest_1 = self.h_tr_ms.clone() + self.h_tr_is.clone();

        let num_tm = self.h_tr_ms.clone() * self.h_tr_is.clone() * self.mass_temperatures.clone();
        let num_phi_st = self.h_tr_is.clone() * phi_st.clone();
        let num_rest = term_rest_1.clone() * (h_ext.clone() * t_e.clone() + phi_ia.clone());
        let t_i_free = (num_tm.clone() + num_phi_st.clone() + num_rest.clone()) / den.clone();

        // 3. HVAC Calculation
        let sensitivity = (self.h_tr_ms.clone() + self.h_tr_is.clone()) / den.clone();
        let hvac_output = self.hvac_power_demand(&t_i_free, &sensitivity);
        let hvac_energy_for_step = hvac_output.map(|o| o.abs()).integrate() * dt;

        // 4. Update Temperatures
        let phi_ia_act = phi_ia + hvac_output;
        let num_rest_act = term_rest_1 * (h_ext * t_e.clone() + phi_ia_act);
        let t_i_act = (num_tm + num_phi_st.clone() + num_rest_act) / den.clone();

        let ts_num = self.h_tr_ms.clone() * self.mass_temperatures.clone()
            + self.h_tr_is.clone() * t_i_act.clone()
            + phi_st;
        let ts_den = self.h_tr_ms.clone() + self.h_tr_is.clone();
        let t_s_act = ts_num / ts_den;

        let q_m_net = self.h_tr_em.clone() * (t_e - self.mass_temperatures.clone())
            + self.h_tr_ms.clone() * (t_s_act - self.mass_temperatures.clone());
        let dt_m = (q_m_net / self.thermal_capacitance.clone()) * dt;
        self.mass_temperatures = self.mass_temperatures.clone() + dt_m;
        self.temperatures = t_i_act;

        hvac_energy_for_step / 3.6e6 // Return kWh
    }

    /// Calculate analytical thermal loads without neural surrogates.
    pub fn calc_analytical_loads(&mut self, timestep: usize, use_analytical_gains: bool) {
        let total_gain = if use_analytical_gains {
            let hour_of_day = timestep % 24;
            let daily_cycle = (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
            (50.0 * daily_cycle).max(0.0) + 10.0 // Adjusted for W/m² (lower values than Watts)
        } else {
            0.0
        };
        self.loads = self.temperatures.constant_like(total_gain);
    }
}

#[cfg(test)]
mod tests {
    use super::ThermalModel;
    use crate::ai::surrogate::SurrogateManager;
    use crate::physics::cta::VectorField;

    #[test]
    fn test_thermal_model_creation() {
        let model = ThermalModel::<VectorField>::new(10);
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

        // Check derived constants
        // Zone Area 20m2.
        assert!((model.zone_area[0] - 20.0).abs() < EPSILON);
        // h_tr_w should be derived.
        // Gross Wall = P * H. P = 4*sqrt(20) = 17.888. H=3. Gross=53.66.
        // Win Area = 53.66 * 0.15 = 8.05.
        // h_tr_w = 2.5 * 8.05 = 20.125.
        assert!(model.h_tr_w[0] > 19.0 && model.h_tr_w[0] < 21.0);
    }

    #[test]
    fn test_apply_parameters_updates_model() {
        let mut model = ThermalModel::<VectorField>::new(10);
        let params = vec![1.5, 22.0];

        model.apply_parameters(&params);
        assert_eq!(model.window_u_value, 1.5);
        assert_eq!(model.hvac_setpoint, 22.0);

        // Check surface updates
        assert_eq!(model.surfaces[0][0].u_value, 1.5);

        // Check conductance update
        // With U=1.5, h_tr_w should be lower than initial U=2.5.
        // Approx 1.5/2.5 * 20.125 = 12.075
        assert!(model.h_tr_w[0] > 11.0 && model.h_tr_w[0] < 13.0);
    }

    #[test]
    fn test_apply_parameters_partial() {
        let mut model = ThermalModel::<VectorField>::new(10);
        let params = vec![1.5];

        model.apply_parameters(&params);
        assert_eq!(model.window_u_value, 1.5);
        assert_eq!(model.hvac_setpoint, 21.0); // Should remain default
    }

    #[test]
    fn test_solve_timesteps_with_surrogates() {
        let model = ThermalModel::<VectorField>::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        // Surrogate-based prediction
        let energy_surrogate = model.clone().solve_timesteps(8760, &surrogates, true);

        // Should produce non-zero energy
        assert!(energy_surrogate > 0.0, "Energy should be non-zero");
    }

    #[test]
    fn test_calc_analytical_loads() {
        let mut model = ThermalModel::<VectorField>::new(5);
        model.calc_analytical_loads(12, true); // noon

        // Check if loads are calculated
        assert!(model.loads.iter().all(|&l| l > 0.0));

        // Check against expected values for noon
        let hour_of_day = 12;
        let daily_cycle = (hour_of_day as f64 / 24.0 * 2.0 * std::f64::consts::PI).sin();
        let solar_gain = (50.0 * daily_cycle).max(0.0);
        let internal_gains = 10.0;
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
        let mut model = ThermalModel::<VectorField>::new(10);

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
        let mut model = ThermalModel::<VectorField>::new(10);
        let params = vec![1.5, 22.0, 1000.0, 999.0];

        // Should only use first two elements
        model.apply_parameters(&params);
        assert_eq!(model.window_u_value, 1.5);
        assert_eq!(model.hvac_setpoint, 22.0);
    }

    #[test]
    fn test_thermal_model_zones() {
        let model_5 = ThermalModel::<VectorField>::new(5);
        assert_eq!(model_5.num_zones, 5);
        assert_eq!(model_5.temperatures.len(), 5);
        assert_eq!(model_5.loads.len(), 5);

        let model_20 = ThermalModel::<VectorField>::new(20);
        assert_eq!(model_20.num_zones, 20);
        assert_eq!(model_20.temperatures.len(), 20);
        assert_eq!(model_20.loads.len(), 20);
    }

    #[test]
    fn test_thermal_model_ndarray() {
        use crate::physics::nd_array::NDArrayField;

        // Test creation with NDArray backend
        let model = ThermalModel::<NDArrayField>::new_ndarray(5);
        assert_eq!(model.num_zones, 5);
        assert_eq!(model.temperatures.len(), 5);

        // Test basic property access to verify struct layout and logic
        assert!((model.temperatures.as_slice()[0] - 20.0).abs() < 1e-9);

        // Test parameter application
        let mut model = model;
        model.apply_parameters(&[1.5, 22.0]);
        assert_eq!(model.window_u_value, 1.5);
    }

    #[test]
    fn test_solve_timesteps_zero_steps() {
        let mut model = ThermalModel::<VectorField>::new(10);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        model.apply_parameters(&[1.5, 21.0]);
        let energy = model.solve_timesteps(0, &surrogates, false);

        // Zero steps should result in zero energy
        assert_eq!(energy, 0.0);
    }

    #[test]
    fn test_solve_timesteps_short_and_long() {
        let mut model = ThermalModel::<VectorField>::new(10);
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
        let mut model = ThermalModel::<VectorField>::new(10);

        model.calc_analytical_loads(0, true);

        // All loads should be calculated
        for &load in model.loads.iter() {
            assert!(load >= 0.0);
        }
    }

    #[test]
    fn test_parameters_affect_energy() {
        let mut model1 = ThermalModel::<VectorField>::new(10);
        let mut model2 = ThermalModel::<VectorField>::new(10);
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
        let mut model = ThermalModel::<VectorField>::new(1);
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
            let mut model = ThermalModel::<VectorField>::new(1);
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
            let mut model = ThermalModel::<VectorField>::new(1);
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
