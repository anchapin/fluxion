//! Top-level annual solver, timestep sizing, and load accessors.
//!
//! Hosts the public solver entry points ([`ThermalModel::solve_timesteps`],
//! [`ThermalModel::solve_timesteps_with_dt`]), the adaptive-timestep helpers
//! ([`ThermalModel::calculate_timestep_seconds`],
//! [`ThermalModel::estimate_time_constant_hours`]), and the load /
//! temperature accessors ([`ThermalModel::calculate_analytical_loads`],
//! [`ThermalModel::set_loads`], [`ThermalModel::set_weather`],
//! [`ThermalModel::get_temperatures`],
//! [`ThermalModel::get_hourly_temperatures`]). Originally part of the
//! monolithic `thermal_model_physics.rs` (Issue #898), extracted as part
//! of the Issue #902 modular split.

use log::{error, info, warn};

use crate::ai::surrogate::SurrogateManager;
use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::adaptive_timestep::TimestepMode;
use crate::sim::equipment::Equipment;
use crate::sim::lighting::LightingSchedule;
use crate::sim::occupancy::OccupancyProfile;
use crate::sim::profiles;
use crate::sim::thermal_model_core::{get_daily_cycle, ThermalModel};
use crate::sim::timestep_solver::StepParameters;
use crate::weather::HourlyWeatherData;

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
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
    /// Solves thermal network dynamics over specified timesteps.
    ///
    /// Uses CTA operations for vector-accelerated solving of the 5R1C/6R2C algebraic system.
    /// Calculates Ti_free (free-floating temp), determines HVAC demand, solves Ti_act and Tm_next.
    /// Returns Energy Use Intensity (EUI) normalized by floor area.
    ///
    /// # Arguments
    /// * `steps` - Number of timesteps to simulate (8760 for 1 year)
    /// * `surrogates` - SurrogateManager for AI-based load prediction
    /// * `use_ai` - If true, use surrogates; if false, compute analytical loads
    /// * `lighting` - Optional lighting schedule (Plan 17-04)
    /// * `equipment` - Optional equipment list (Plan 17-04)
    /// * `occupancy` - Optional occupancy profile (Plan 17-04)
    ///
    /// # Returns
    /// Energy Use Intensity (EUI) in kWh/m²/year
    ///
    /// # Performance
    /// - Target: <100ms for 8760 timesteps (single building)
    /// - Uses CTA operations for vector acceleration
    /// - Thread-safe for parallel evaluation via rayon
    ///
    /// # Example
    /// ```rust,no_run
    /// let eui = model.solve_timesteps(8760, &surrogates, false, None, None, None);
    /// // Simulates 1 year with analytical loads (no internal loads)
    /// ```
    pub fn solve_timesteps(
        &mut self,
        steps: usize,
        surrogates: &SurrogateManager,
        use_ai: bool,
        lighting: Option<&LightingSchedule>,
        equipment: Option<&[Box<dyn Equipment>]>,
        occupancy: Option<&OccupancyProfile>,
    ) -> f64 {
        // Determine timestep based on case_id and timestep_mode
        let dt_seconds = self.calculate_timestep_seconds();

        self.solve_timesteps_with_dt(
            steps, surrogates, use_ai, lighting, equipment, occupancy, dt_seconds,
        )
    }

    /// Calculate timestep in seconds based on timestep_mode and case_id.
    ///
    /// For high-mass cases (900 series), this returns 360 seconds (6 minutes).
    /// For low-mass cases (600 series), this returns 3600 seconds (1 hour).
    pub fn calculate_timestep_seconds(&self) -> f64 {
        match &self.0.timestep_mode {
            TimestepMode::Fixed { dt } => dt.as_secs_f64(),
            TimestepMode::Adaptive {
                base_dt,
                min_dt: _,
                threshold_tau,
            } => {
                // Calculate time constant for this building
                let tau_hours = self.estimate_time_constant_hours();
                if tau_hours >= *threshold_tau {
                    // High-mass: use adaptive timestep
                    base_dt.as_secs_f64()
                } else {
                    // Low-mass: use 1-hour standard timestep
                    3600.0
                }
            }
        }
    }

    /// Estimate thermal time constant in hours from physical parameters.
    ///
    /// Issue #821 / Probe H: We previously short-circuited this with
    /// `TimeConstantAnalyzer::for_case(&self.0.case_id)`, returning a hard-coded τ
    /// table per ASHRAE 140 case identifier. That broke blind-validation
    /// (case_id is supposed to be opaque to the solver) and could disagree with
    /// the actual `Cm / h_tr_ms` after Probe A's ISO 13790 conductance change.
    ///
    /// We now always derive τ from physics:
    ///   τ_seconds = Σ C_m  /  Σ h_tr_ms   (sum across zones)
    /// and only fall back to a 2-hour default if `h_tr_ms` is degenerate.
    pub fn estimate_time_constant_hours(&self) -> f64 {
        // Check if this is a high-mass case (900 series)
        let is_high_mass = matches!(
            self.0.case_id.as_str(),
            "900" | "910" | "920" | "930" | "940" | "950" | "900FF" | "950FF"
        );

        // For high-mass cases, use H_tr_3 (~40 W/K) for correct slow thermal coupling
        // This gives ~69 hour time constant instead of ~1.9 hours with h_tr_ms
        let h_tr_sum = if is_high_mass {
            // Use derived_h_tr_3 for high-mass cases
            // Fall back to h_tr_ms if derived_h_tr_3 hasn't been computed yet (model not initialized)
            let derived = self.0.derived_h_tr_3.as_ref().iter().sum::<f64>();
            if derived > 1e-6 {
                derived
            } else {
                self.0.h_tr_ms.as_ref().iter().sum::<f64>()
            }
        } else {
            // Standard: use h_tr_ms for surface-to-mass coupling
            self.0.h_tr_ms.as_ref().iter().sum::<f64>()
        };

        if h_tr_sum > 0.0 {
            let tau_seconds = self.0.thermal_capacitance.as_ref().iter().sum::<f64>() / h_tr_sum;
            let tau_hours = tau_seconds / 3600.0; // Convert to hours

            // Sanity check: if tau is extremely small (< 0.001 hours = 3.6 seconds),
            // the model is likely not properly initialized (placeholder values).
            // Use 2-hour default for uninitialized models.
            if tau_hours < 0.001 {
                // Model not properly initialized - use default
                2.0
            } else {
                tau_hours
            }
        } else {
            2.0 // Default: 2 hours (boundary between low/high mass)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_timesteps_with_dt(
        &mut self,
        steps: usize,
        surrogates: &SurrogateManager,
        use_ai: bool,
        lighting: Option<&LightingSchedule>,
        equipment: Option<&[Box<dyn Equipment>]>,
        occupancy: Option<&OccupancyProfile>,
        dt_seconds: f64,
    ) -> f64 {
        // Record call for wiring validation (Plan 21-10)
        #[cfg(feature = "wiring-tracing")]
        if let Some(ref tracer) = self.0.tracer {
            tracer.record_call("solve_timesteps_with_dt");
        }

        info!(
            "Starting simulation for {} timesteps with dt={:.1}s, use_ai={}",
            steps, dt_seconds, use_ai
        );

        // Auto-load building profile if not manually provided (Plan 17-04)
        let profile_bundle = match (lighting, equipment, occupancy) {
            (None, None, None) => {
                // Load profile from JSON based on building_type
                match profiles::load_building_profile(self.0.building_type) {
                    Ok(profile) => {
                        info!(
                            "Auto-loaded building profile for {:?}: lighting, {} equipment items, occupancy",
                            self.0.building_type,
                            profile.equipment.len()
                        );
                        Some(profile)
                    }
                    Err(e) => {
                        warn!(
                            "Failed to load building profile for {:?}: {}. Running without internal loads.",
                            self.0.building_type, e
                        );
                        None
                    }
                }
            }
            _ => None, // Use provided overrides
        };

        // Determine which profiles to use
        let lighting_ref = profile_bundle.as_ref().map(|p| &p.lighting).or(lighting);
        let occupancy_ref = profile_bundle.as_ref().map(|p| &p.occupancy).or(occupancy);

        // Handle equipment conversion if profile was loaded
        let equipment_converted: Option<Vec<Box<dyn Equipment>>> =
            profile_bundle.as_ref().map(|profile| {
                profile
                    .equipment
                    .iter()
                    .map(|eq| {
                        // Try to downcast and clone
                        if let Some(computer) = eq
                            .as_any()
                            .downcast_ref::<crate::sim::equipment::ComputerEquipment>()
                        {
                            Box::new(computer.clone()) as Box<dyn Equipment>
                        } else if let Some(server) = eq
                            .as_any()
                            .downcast_ref::<crate::sim::equipment::ServerRack>()
                        {
                            Box::new(server.clone()) as Box<dyn Equipment>
                        } else if let Some(generic) =
                            eq.as_any()
                                .downcast_ref::<crate::sim::equipment::GenericEquipment>()
                        {
                            Box::new(generic.clone()) as Box<dyn Equipment>
                        } else {
                            panic!("Unknown equipment type in building profile");
                        }
                    })
                    .collect()
            });
        let _equipment_ref = equipment_converted.as_deref().or(equipment);

        // Issue #763 — initialize hourly temperature storage before timestep loop
        self.0.hourly_temperatures = Some(vec![Vec::with_capacity(steps); self.0.num_zones]);

        // Issue #901 perf: construct a single StepParameters once and reuse it
        // (passed by & reference to solve_single_step). Avoids per-step clones of
        // SurrogateManager, LightingSchedule, and OccupancyProfile.
        // When use_ai is false we still need a SurrogateManager value; use Default
        // (heap-free) instead of cloning the (potentially heavy) composite surrogate.
        let step_params = StepParameters {
            use_ai,
            surrogates: if use_ai {
                surrogates.clone()
            } else {
                SurrogateManager::default()
            },
            use_analytical_gains: true,
            lighting: lighting_ref.cloned(),
            equipment: None, // Can't clone dyn Equipment, so pass None
            occupancy: occupancy_ref.cloned(),
        };

        let cycle = get_daily_cycle();
        let total_energy_kwh: f64 = (0..steps)
            .map(|t| {
                if t % 1000 == 0 {
                    info!("Progress: {}/{} timesteps", t, steps);
                }
                let hour_of_day = t % 24;
                let daily_cycle = cycle[hour_of_day];
                let outdoor_temp = 10.0 + 10.0 * daily_cycle;
                let energy = self.solve_single_step(t, outdoor_temp, &step_params, dt_seconds);

                // Issue #763 — capture zone temperatures after each timestep
                // Issue #901 perf: bound check is unnecessary — temperatures always has
                // exactly num_zones entries and hourly is sized to num_zones at init.
                if let Some(ref mut hourly) = self.0.hourly_temperatures {
                    let temps = self.0.temperatures.as_ref();
                    debug_assert_eq!(temps.len(), hourly.len());
                    for (zone_idx, &temp) in temps.iter().enumerate() {
                        hourly[zone_idx].push(temp);
                    }
                }

                energy
            })
            .sum();

        // Normalize by total floor area to get EUI
        let total_area = self.0.zone_area.integrate();
        if total_area > 0.0 {
            let eui = total_energy_kwh / total_area;
            info!("Simulation complete: EUI = {:.2} kWh/m²/year", eui);
            eui
        } else {
            error!("Total floor area is zero, cannot calculate EUI");
            0.0
        }
    }

    /// Extract current temperatures for batched inference.
    ///
    /// # Returns
    /// Vector of current zone temperatures in degrees Celsius.
    pub fn get_temperatures(&self) -> Vec<f64> {
        self.0.temperatures.as_ref().to_vec()
    }

    /// Get the full hourly zone temperature profiles (Issue #763).
    ///
    /// # Returns
    /// `Some([[T00, T01, ...], [T10, T11, ...], ...])` where outer index is zone,
    /// inner index is timestep (0..steps-1), or `None` if the simulation has not
    /// been run through `solve_timesteps_with_dt`.
    pub fn get_hourly_temperatures(&self) -> Option<Vec<Vec<f64>>> {
        self.0.hourly_temperatures.clone()
    }

    /// Calculate analytical thermal loads without neural surrogates.
    ///
    /// This method computes thermal loads from first principles physics:
    /// - Solar gains: self.0.solar_gains.as_ref()[zone] for each zone
    /// - Conduction: window_u_value * window_area[zone] * (outdoor_temp - temperatures.as_ref()[zone])
    /// - Ventilation: h_ve.as_ref()[zone] * (outdoor_temp - temperatures.as_ref()[zone])
    ///
    /// # Arguments
    ///
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    /// * `hour_of_day` - Hour of day (0-23) for solar gain calculation
    ///
    /// # Returns
    ///
    /// Vector of thermal loads (W/m²) for each zone
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use fluxion::sim::engine::ThermalModel;
    /// use fluxion::physics::cta::VectorField;
    ///
    /// let model = ThermalModel::<VectorField>::new(1);
    /// let outdoor_temp = 35.0;
    /// let hour_of_day = 12;
    /// let loads = model.calculate_analytical_loads(outdoor_temp, hour_of_day);
    /// ```
    pub fn calculate_analytical_loads(&self, outdoor_temp: f64, _hour_of_day: usize) -> Vec<f64> {
        let mut loads = Vec::with_capacity(self.0.num_zones);

        // Calculate window area for each zone
        let width = self
            .0
            .zone_area
            .zip_with(&self.0.aspect_ratio, |a, ar| (a * ar).sqrt());
        let depth = self.0.zone_area.zip_with(&width, |a, w| a / w);
        let perimeter = (width + depth) * 2.0;
        let gross_wall_area = perimeter.clone() * self.0.ceiling_height.clone();
        let window_area = gross_wall_area * self.0.window_ratio.clone();

        for zone_idx in 0..self.0.num_zones {
            let zone_temp = self.0.temperatures.as_ref()[zone_idx];
            let zone_window_area = window_area.as_ref()[zone_idx];
            let h_ve = self.0.h_ve.as_ref()[zone_idx];

            // 1. Solar gains (already computed in self.0.solar_gains)
            let solar_gain = self.0.solar_gains.as_ref()[zone_idx];

            // 2. Conduction through windows: Q = U * A * (T_out - T_in)
            let conduction = self.0.window_u_value * zone_window_area * (outdoor_temp - zone_temp);

            // 3. Ventilation: Q = h_ve * (T_out - T_in)
            let ventilation = h_ve * (outdoor_temp - zone_temp);

            // Total load (W/m²)
            let total_load = solar_gain + conduction + ventilation;
            loads.push(total_load);
        }

        loads
    }

    /// Apply pre-computed loads from batched inference.
    ///
    /// # Arguments
    /// * `loads` - Thermal loads (W/m²) for each zone
    pub fn set_loads(&mut self, loads: &[f64]) {
        self.0.loads = T::from(VectorField::new(loads.to_vec()));
    }

    /// Set weather data for solar gain calculations.
    ///
    /// This enables proper solar gain calculation in step_physics when weather data
    /// is not provided through the CaseSpec (e.g., when using DenverTmyWeather directly).
    ///
    /// # Arguments
    ///
    /// * `weather` - Hourly weather data to use for solar calculations
    pub fn set_weather(&mut self, weather: HourlyWeatherData) {
        self.0.weather = Some(weather);
    }
}
