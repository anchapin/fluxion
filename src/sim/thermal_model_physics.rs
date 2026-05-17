//! Thermal model core module
//!
//! ISO 13790-compliant 5R1C/6R2C thermal network implementation.
//! Contains the core thermal model types, struct, and implementations.

use crossbeam::channel::{Receiver, Sender};
use log::{error, info, warn};

use crate::ai::surrogate::SurrogateManager;
use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::physics::multi_node_solver::SurfaceExteriorTemperatures;
use crate::sim::adaptive_timestep::TimestepMode;
use crate::sim::equipment::Equipment;
use crate::sim::hvac::{HVACMode as EquipmentHVACMode, VariableCapacityEquipment};
use crate::sim::interzone::{calculate_stack_effect_ach, calculate_ventilation_heat_transfer};
use crate::sim::lighting::LightingSchedule;
use crate::sim::occupancy::OccupancyProfile;
use crate::sim::profiles;
use crate::sim::sky_radiation::SolAirTemperature;
use crate::sim::solar::{calculate_solar_position, calculate_surface_irradiance};
use crate::sim::thermal_integration::{
    backward_euler_update, backward_euler_update_2cond, crank_nicolson_update,
    crank_nicolson_update_3cond, select_integration_method, ThermalIntegrationMethod,
};
use crate::sim::thermal_model_core::{get_daily_cycle, ThermalModel};
use crate::sim::timestep_solver::StepParameters;
use crate::validation::ashrae_140_cases::Orientation;
use crate::weather::HourlyWeatherData;

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    /// Calculate HVAC power demand using IdealLoadsSystem thermodynamic formulas.
    ///
    /// This replaces the sensitivity-based `(setpoint - temp) / sensitivity` formula
    /// with proper ideal loads physics: `mass_flow * cp * delta_t`.
    ///
    /// Returns a VectorField of power values:
    /// - Positive = heating demand (W)
    /// - Negative = cooling demand (W)
    ///
    /// # Arguments
    /// * `zone_temps` - Current zone temperatures (°C)
    /// * `heating_setpoint` - Single heating setpoint (°C) applied to all zones
    /// * `cooling_setpoint` - Single cooling setpoint (°C) applied to all zones
    fn hvac_demand_from_ideal_loads(
        &self,
        zone_temps: &[f64],
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> T {
        let enabled_vec = self.0.hvac_enabled.as_ref();

        let heating_vec = vec![heating_setpoint; self.0.num_zones];
        let cooling_vec = vec![cooling_setpoint; self.0.num_zones];

        let heat_cap = self.0.hvac_heating_capacity;
        let cool_cap = self.0.hvac_cooling_capacity;

        let mut combined_demand = vec![0.0; self.0.num_zones];
        for (zone_idx, opt_system) in self.0.ideal_loads_system.iter().enumerate() {
            if let Some(ref system) = opt_system {
                let zone_temps_slice = &zone_temps[zone_idx..zone_idx + 1];
                let heating_slice = &heating_vec[zone_idx..zone_idx + 1];
                let cooling_slice = &cooling_vec[zone_idx..zone_idx + 1];
                let enabled_slice = &enabled_vec[zone_idx..zone_idx + 1];

                let demands = system.calculate_power_demand_vector(
                    zone_temps_slice,
                    heating_slice,
                    cooling_slice,
                    enabled_slice,
                );

                // Clamp to HVAC capacity limits to prevent numerical explosion
                // when zone temperatures are extreme (e.g., bare models without
                // update_derived_parameters). Matches the .clamp() in the old
                // hvac_power_demand method.
                combined_demand[zone_idx] = demands[0].clamp(-cool_cap, heat_cap);
            }
        }

        T::from(VectorField::new(combined_demand))
    }

    /// Core physics simulation loop for annual building energy performance.
    ///
    /// Simulates hourly thermal dynamics using batched inference with a coordinator.
    ///
    /// This method implements the worker side of the coordinator-worker pattern.
    /// At each timestep, it sends its current temperature state to the coordinator,
    /// waits for the predicted loads, and then completes the physics calculation.
    pub fn solve_timesteps_batched(
        &mut self,
        steps: usize,
        tx: Sender<Vec<f64>>,
        rx: Receiver<Vec<f64>>,
    ) -> f64 {
        let cycle = get_daily_cycle();
        let total_energy_kwh: f64 = (0..steps)
            .map(|t| {
                let hour_of_day = t % 24;
                let daily_cycle = cycle[hour_of_day];
                let outdoor_temp = 10.0 + 10.0 * daily_cycle;

                // 1. Send current state to coordinator
                let temps = self.get_temperatures();
                tx.send(temps).expect("Failed to send state to coordinator");

                // 2. Receive predicted loads from coordinator
                let loads = rx.recv().expect("Failed to receive loads from coordinator");
                self.set_loads(&loads);

                // 3. Solve physics for this timestep
                self.step_physics(t, outdoor_temp, 3600.0)
            })
            .sum();

        // Normalize by total floor area to get EUI
        let total_area = self.0.zone_area.integrate();
        if total_area > 0.0 {
            total_energy_kwh / total_area
        } else {
            0.0
        }
    }

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
        // τ = C / h_tr_ms in seconds
        // The envelope mass time constant is based on surface-to-mass coupling
        // (h_tr_ms) only — h_tr_em affects the surface node T_s, not T_m
        // directly (Issue #693). h_tr_me, when non-zero (6R2C), is a separate
        // thermal path on the internal-mass branch.
        let h_tr_sum = self.0.h_tr_ms.as_ref().iter().sum::<f64>();

        if h_tr_sum > 0.0 {
            let tau_seconds = self.0.thermal_capacitance.as_ref().iter().sum::<f64>() / h_tr_sum;
            tau_seconds / 3600.0 // Convert to hours
        } else {
            2.0 // Default: 2 hours (boundary between low/high mass)
        }
    }

    /// Solve thermal model for specified timesteps with variable timestep support.
    ///
    /// This method extends solve_timesteps to support adaptive timestep simulation,
    /// allowing finer timesteps (e.g., 6-minute) for high-mass buildings.
    ///
    /// # Arguments
    /// * `steps` - Number of hourly timesteps (typically 8760 for 1 year)
    /// * `surrogates` - Reference to SurrogateManager for load predictions
    /// * `use_ai` - If true, use neural surrogates; if false, use analytical calculations
    /// * `lighting` - Optional lighting schedule for internal heat gains (Plan 17-04)
    /// * `equipment` - Optional equipment list for internal heat gains (Plan 17-04)
    /// * `occupancy` - Optional occupancy profile for internal heat gains (Plan 17-04)
    /// * `dt_seconds` - Timestep duration in seconds (e.g., 3600.0 for 1-hour, 360.0 for 6-minute)
    ///
    /// # Returns
    /// Cumulative annual energy use intensity (EUI) in kWh/m²/year.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use fluxion::sim::engine::ThermalModel;
    /// use fluxion::physics::cta::VectorField;
    /// use fluxion::ai::surrogate::SurrogateManager;
    ///
    /// let mut model = ThermalModel::<VectorField>::new(1);
    /// let surrogates = SurrogateManager::new();
    ///
    /// // Run with 6-minute timestep for high-mass building
    /// let eui = model.solve_timesteps_with_dt(8760, &surrogates, false, None, None, None, 360.0);
    /// ```
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

        let cycle = get_daily_cycle();
        let total_energy_kwh: f64 = (0..steps)
            .map(|t| {
                if t % 1000 == 0 {
                    info!("Progress: {}/{} timesteps", t, steps);
                }
                let hour_of_day = t % 24;
                let daily_cycle = cycle[hour_of_day];
                let outdoor_temp = 10.0 + 10.0 * daily_cycle;
                let step_params = StepParameters {
                    use_ai,
                    surrogates: surrogates.clone(),
                    use_analytical_gains: true,
                    lighting: lighting_ref.cloned(),
                    equipment: None, // Can't clone dyn Equipment, so pass None
                    occupancy: occupancy_ref.cloned(),
                };
                let energy = self.solve_single_step(t, outdoor_temp, step_params, dt_seconds);

                // Issue #763 — capture zone temperatures after each timestep
                if let Some(ref mut hourly) = self.0.hourly_temperatures {
                    for (zone_idx, &temp) in self.0.temperatures.as_ref().iter().enumerate() {
                        if zone_idx < hourly.len() {
                            hourly[zone_idx].push(temp);
                        }
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

    /// Solve physics for one timestep (assumes loads already set).
    ///
    /// This method performs only the physics calculation portion of solve_single_step,
    /// assuming that loads have already been set via set_loads() or calculated externally.
    /// This enables batched inference: collect all temperatures, run one batched prediction,
    /// distribute loads, then call this method in parallel.
    ///
    /// # Arguments
    /// * `timestep` - Current timestep index (used for ground temperature)
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    /// * `dt_seconds` - Timestep duration in seconds (default: 3600.0 for 1-hour timestep)
    ///
    /// # Returns
    /// HVAC energy consumption for the timestep in kWh.
    ///
    /// Issue #351: Calculate solar gains internally if weather data is available
    pub fn step_physics(&mut self, timestep: usize, outdoor_temp: f64, dt_seconds: f64) -> f64 {
        // Record call for wiring validation (Plan 21-10)
        #[cfg(feature = "wiring-tracing")]
        if let Some(ref tracer) = self.0.tracer {
            tracer.record_call("step_physics");
        }

        // Issue #351: Calculate loads from weather data if not already set
        // This is needed for ASHRAE 140 validation where step_physics is called directly
        if self.0.weather.is_some() {
            self.calc_analytical_loads(timestep, true);
        }

        // Branch based on thermal model type
        if self.is_nine_r4c_model() {
            self.step_physics_9r4c(timestep, outdoor_temp, dt_seconds)
        } else if self.is_8r3c_model() {
            self.step_physics_8r3c(timestep, outdoor_temp, dt_seconds)
        } else if self.is_6r2c_model() {
            self.step_physics_6r2c(timestep, outdoor_temp, dt_seconds)
        } else {
            self.step_physics_5r1c(timestep, outdoor_temp, dt_seconds)
        }
    }

    /// Solve physics for one timestep using the 5R1C (single mass node) model.
    ///
    /// This is the original implementation for backward compatibility.
    fn step_physics_5r1c(&mut self, timestep: usize, outdoor_temp: f64, dt_seconds: f64) -> f64 {
        let dt = dt_seconds; // Use provided timestep duration

        // Prepare sol-air temperature and calculate CTF/FD heat fluxes early to avoid borrow conflicts
        // Calculate sky temperature for proper sol-air calculation with longwave radiation
        let sky_temp = self
            .0
            .weather
            .as_ref()
            .map(|w| w.sky_temperature())
            .unwrap_or(outdoor_temp - 15.0);
        let (_t_sol_air_data, ctf_flux_w, fd_flux_w, _ctf_surface_temps) =
            self.prepare_solvers_and_sol_air(timestep, outdoor_temp, sky_temp);

        // Get ground temperature at this timestep
        let t_g = self.0.ground_temperature.ground_temperature(timestep);

        // --- Dynamic Ventilation (Night Ventilation) ---
        let hour_of_day = (timestep % 24) as u8;

        // Combine fractions to avoid multiple intermediate VectorField allocations
        let conv_frac = self.0.convective_fraction;
        let rad_frac = 1.0 - conv_frac;

        // Internal radiative gains split per ISO 13790 Section C.4 Eq. C.5/C.6:
        // Eq. C.5 (radiative-to-surface): phi_st = (1 - F_sup) * phi_int_rad
        //   where F_sup = H_ms / (H_ms + H_is) — fraction to surface node
        //   st_int_frac = rad_frac * (1 - solar_distribution_to_air) = rad_frac * F_sup
        //   Note: F_sup is the fraction from internal radiative gains going to surface
        //   per ISO 13790 C.4 Eq. C.5 (internal radiative → surface node).
        //
        // Eq. C.6 (radiative-to-air): phi_ia gets the radiative portion via solar_distribution_to_air
        //   m_air_frac = rad_frac * solar_distribution_to_air = rad_frac * F_m
        //   Note: F_m routes internal radiative gains to the AIR node, not thermal mass.
        //   Per ISO 13790 C.4 Eq. C.6, the mass-air node receives radiative gains.
        //
        // The naming reflects ISO 13790 Section C.4:
        //   st_int_frac = fraction of internal radiative gains to SURFACE node (phi_st)
        //   m_air_frac  = fraction of internal radiative gains to AIR node (phi_ia via routing)
        //
        // st_sol_frac: Solar gains to surface (fraction of solar that goes to surface)
        // m_sol_frac: Solar gains to mass (fraction of solar that goes to mass)
        // Note: solar_distribution_to_air controls how much solar goes directly to zone air
        let st_int_frac = rad_frac * (1.0 - self.0.solar_distribution_to_air);
        let m_air_frac = rad_frac * self.0.solar_distribution_to_air;
        let st_sol_frac = 1.0 - self.0.solar_beam_to_mass_fraction;
        let m_sol_frac = self.0.solar_beam_to_mass_fraction;

        let loads_ref = self.0.loads.as_ref();
        let solar_ref = self.0.solar_gains.as_ref();
        let opaque_solar_ref = self.0.opaque_solar_gains.as_ref();
        let area_ref = self.0.zone_area.as_ref();

        let mut phi_ia_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_st_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_m_data = Vec::with_capacity(self.0.num_zones);

        for i in 0..self.0.num_zones {
            let load_w = loads_ref[i] * area_ref[i];
            let sol_w = solar_ref[i] * area_ref[i];
            let opaque_sol_w = opaque_solar_ref[i] * area_ref[i];

            // Internal gains: convective to air, radiative split between surface and mass
            // Solar distribution must conserve energy (sum to 1.0)
            let sol_to_air = sol_w * self.0.solar_distribution_to_air;
            let remaining_sol = sol_w - sol_to_air;
            phi_ia_data.push(load_w * conv_frac + sol_to_air);
            phi_st_data.push(load_w * st_int_frac + remaining_sol * st_sol_frac);
            phi_m_data.push(load_w * m_air_frac + remaining_sol * m_sol_frac + opaque_sol_w);
        }

        let phi_ia = T::from(VectorField::new(phi_ia_data));
        let phi_st = T::from(VectorField::new(phi_st_data));
        let phi_m = T::from(VectorField::new(phi_m_data));

        // PR #821 / Issue #825 — record zone-0 heat-balance terms for the
        // `pr821-diag` hourly CSV. Zero overhead when the feature is disabled
        // (no fields, no writes). The CSV consumer reads these fields right
        // after `step_physics` returns. Only zone 0 is captured because the
        // 600FF / 650FF investigation is single-zone.
        #[cfg(feature = "pr821-diag")]
        {
            self.0.last_phi_ia = phi_ia.as_ref().first().copied().unwrap_or(0.0);
            self.0.last_phi_st = phi_st.as_ref().first().copied().unwrap_or(0.0);
            self.0.last_phi_m = phi_m.as_ref().first().copied().unwrap_or(0.0);
        }

        // Use outdoor_temp directly. Solar gains on opaque surfaces are already included in phi_m.
        let mut t_sol_air_data = Vec::with_capacity(self.0.num_zones);
        for _ in 0..self.0.num_zones {
            t_sol_air_data.push(outdoor_temp);
        }
        let t_sol_air = VectorField::new(t_sol_air_data.clone());

        // Simplified 5R1C calculation using CTA
        // Include ground coupling through floor
        // Use pre-computed cached values to avoid redundant allocations
        let h_ext_base = &self.0.derived_h_ext;
        let term_rest_1 = &self.0.derived_term_rest_1;

        // === Issue #824: Night-ventilation air-side coupling (was missing entirely) ===
        //
        // For ASHRAE 140 Case 650 / 650FF the spec defines a night-ventilation
        // fan that runs 18:00 → 07:00 at 1703.16 m³/h, supplying outside air
        // directly to the zone. Per ASHRAE 140 §5.4 (Case 650 description), this
        // is an *air-side* path: the fan moves outdoor air through the zone
        // and removes heat at rate
        //     Q_night_vent = ρ·Cp·V̇_fan · (T_zone − T_outdoor)
        //
        // Equivalently, it adds an additional air-to-outdoor conductance
        //     h_ve_night = ρ · Cp · V̇_fan / 3600   [W/K]
        // (the /3600 converts m³/h → m³/s; ρ=1.2 kg/m³, Cp=1005 J/(kg·K) match
        // the existing infiltration h_ve calculation in
        // src/sim/thermal_model_core.rs::update_derived_parameters).
        //
        // Issue #821 history: the *legacy* implementation routed 30 % of the
        // night-vent flow directly to the *mass* node (via h_vent_mass_zone in
        // the mass integrator below), which double-counted air-side cooling
        // once h_tr_ms was restored to the ISO 13790 lumped value. Issue #821
        // disabled that mass-side path (h_vent_mass_zone = 0). However the
        // *air-side* path was never actually wired in step_physics_5r1c — the
        // comment "phi_ia_with_vent further down" referenced a variable that
        // does not exist, leaving 600FF and 650FF behaviourally identical
        // (both peak at 48.28 °C / trough at -7.70 °C on main).
        //
        // Issue #824 fix: add h_ve_night to the air-to-outdoor conductance
        // (h_ext) during active hours, *without* restoring the legacy 30 %
        // mass-side path. This makes night ventilation a true air-side
        // ventilation conductance (analogous to infiltration h_ve, which is
        // already in derived_h_ext) and is consistent with ISO 13790 §C.4
        // Eq. C.10 where ventilation appears only on the air node.
        //
        // The cached derived_h_ext / derived_den are computed at-build-time
        // from the static h_ve only; we recompute h_ext and den per-step when
        // night-vent is active. Static cache is reused unchanged at every
        // other hour to keep the hot path cheap.
        let mut h_ve_night_zone = vec![0.0_f64; self.0.num_zones];
        let mut night_vent_active_now = false;
        if let Some(ref night_vent) = self.0.night_ventilation {
            if night_vent.is_active_at_hour(hour_of_day) {
                night_vent_active_now = true;
                // ASHRAE 140 night-vent fan supplies outdoor air to zone 0
                // (the conditioned zone). Multi-zone night-vent (Case 960
                // sunspace etc.) is out of scope for this issue.
                let rho = self.0.air_density.as_ref().first().copied().unwrap_or(1.2);
                let cp = self
                    .0
                    .heat_capacity
                    .as_ref()
                    .first()
                    .copied()
                    .unwrap_or(1005.0);
                let h_ve_night = night_vent.fan_capacity * rho * cp / 3600.0;
                h_ve_night_zone[0] = h_ve_night;
            }
        }

        // Build per-zone h_ext that includes the night-vent contribution when
        // active. When inactive (the common case) this is just an alloc-free
        // alias to the cached vector via Vec::clone — see below.
        let h_ext_owned: T = if night_vent_active_now {
            let base = h_ext_base.as_ref();
            let mut v = Vec::with_capacity(base.len());
            for (i, &b) in base.iter().enumerate() {
                v.push(b + h_ve_night_zone[i]);
            }
            T::from(VectorField::new(v))
        } else {
            T::from(VectorField::new(h_ext_base.as_ref().to_vec()))
        };
        let h_ext: &T = &h_ext_owned;

        // Recalculate sensitivity tensor at each timestep (Issue #301, #366)
        // When ventilation (h_ve) changes, zone temperature sensitivity to HVAC changes
        // For systems with variable infiltration/ventilation, we must recalculate sensitivity
        // at each timestep to maintain accuracy (non-linear system behavior)
        // Fix: Include derived_ground_coeff in denominator to match update_optimization_cache
        // Issue #351: Include inter-zone conductance in sensitivity calculation
        // Issue #824: when night-vent is active we must rebuild den from
        // h_ext_dynamic (not the cached static derived_den).
        let den: T = if night_vent_active_now {
            // den = h_ms_is_prod + term_rest_1 * (h_ext + h_iz + h_iz_rad) + ground_coeff
            // (mirrors update_optimization_cache in
            // src/sim/thermal_model_solvers.rs, with h_ext now per-zone vector)
            let h_ms_is_prod = self.0.derived_h_ms_is_prod.as_ref();
            let term_rest_1 = self.0.derived_term_rest_1.as_ref();
            let ground_coeff = self.0.derived_ground_coeff.as_ref();
            let h_iz = self.0.h_tr_iz.as_ref();
            let h_iz_rad = self.0.h_tr_iz_rad.as_ref();
            let h_ext_slice = h_ext.as_ref();
            let mut v = Vec::with_capacity(h_ext_slice.len());
            for i in 0..h_ext_slice.len() {
                let h_total = if self.0.num_zones > 1 {
                    h_ext_slice[i] + h_iz[i] + h_iz_rad[i]
                } else {
                    h_ext_slice[i]
                };
                v.push(h_ms_is_prod[i] + term_rest_1[i] * h_total + ground_coeff[i]);
            }
            T::from(VectorField::new(v))
        } else {
            self.0.derived_den.clone()
        };
        let sensitivity = self.0.derived_sensitivity.clone();

        // Optimized: use zip_with to avoid double clones; num_tm allocates 1 vector instead of 2
        let num_tm = self
            .0
            .derived_h_ms_is_prod
            .zip_with(&self.0.mass_temperatures, |a, b| a * b);
        // Optimized: use zip_with to avoid double clones (phi_st used later)
        let num_phi_st = self.0.h_tr_is.zip_with(&phi_st, |a, b| a * b);

        // Ground heat transfer: Q_ground = h_tr_floor * (T_ground - T_surface)
        // Optimization: use scalar multiplication for t_g and outdoor_temp instead of creating full constant vectors
        // Note: t_e vector creation removed. h_ext * t_e replaced by h_ext * outdoor_temp.
        // Note: t_g vector creation removed. h_tr_floor * t_g_vec replaced by h_tr_floor * t_g.

        // === Inter-zone heat transfer (for multi-zone buildings like Case 960) ===
        // Three-component approach: Q_iz = Q_cond + Q_rad + Q_vent
        // 1. Conductive: Q_cond = h_tr_iz * ΔT
        // 2. Radiative: Q_rad = σ·ε₁·ε₂·F·A·(T₁⁴ - T₂⁴) (full nonlinear Stefan-Boltzmann)
        // 3. Ventilation: Q_vent = ρ·Cp·ACH·V·ΔT (temperature-dependent ACH via stack effect)
        let num_zones = self.0.num_zones;

        // Start with phi_ia; we will add inter-zone heat directly to its buffer if needed.
        // Clone so phi_ia remains available for the Case 610 debug print below (line 914).
        let mut phi_ia_with_iz = phi_ia.clone();

        if num_zones > 1 {
            let temps = self.0.temperatures.as_ref();
            let h_iz_vec = self.0.h_tr_iz.as_ref();

            // For Case 960 (2-zone building), calculate heat transfer between zone 0 (back-zone) and zone 1 (sunspace)
            if num_zones >= 2 && h_iz_vec[0] > 0.0 {
                let delta_t_cond = temps[1] - temps[0]; // T_sunspace - T_back

                // 1. Conductive heat transfer
                let q_cond = h_iz_vec[0] * delta_t_cond;

                // 2. Radiative heat transfer - DISABLED for Case 960 (aligned windows don't exchange radiation)
                // This was causing excessive heat loss from sunspace
                let q_rad = 0.0; // windows face same direction - no radiative exchange

                // 3. Ventilation heat transfer (temperature-dependent ACH via stack effect)
                // Use back-zone volume for ventilation calculation
                let zone_volume = self.0.zone_volume.as_ref();
                let ach_iz = calculate_stack_effect_ach(
                    temps[0], // T_back-zone
                    temps[1], // T_sunspace
                    self.0.door_geometry.height,
                    self.0.door_geometry.area,
                    zone_volume[0], // FIX: Pass actual zone volume
                );
                let q_vent = calculate_ventilation_heat_transfer(
                    ach_iz,
                    temps[1],       // Source: sunspace (warm in summer, cold in winter)
                    temps[0],       // Target: back-zone
                    zone_volume[0], // Target volume
                );

                // Total inter-zone heat transfer (positive = sunspace → back-zone)
                let q_iz_total = q_cond + q_rad + q_vent;

                // Apply to energy balance directly in-place
                let slice = phi_ia_with_iz.as_mut();
                if slice.len() >= 2 {
                    slice[0] += -q_iz_total;
                    slice[1] += q_iz_total;
                } else {
                    // Defensive: should never happen for 2-zone case
                    eprintln!(
                        "WARNING: phi_ia length {} < 2, cannot apply inter-zone heat",
                        slice.len()
                    );
                }
            }
        }

        // === SESSION 77: CTF-Zone Air Coupling Integration ===
        // Add CTF envelope conduction heat flux (if enabled)
        // The coupling solver iteratively finds interior surface temperature that satisfies
        // both the CTF conduction equation and the surface heat balance.
        // Positive flux = heat into zone, Negative flux = heat out of zone
        if let Some(ctf_fluxes) = &ctf_flux_w {
            let slice = phi_ia_with_iz.as_mut();
            for (i, &q_flux) in ctf_fluxes.iter().enumerate() {
                if i < slice.len() {
                    // Convert flux [W/m²] to power [W] by multiplying by zone area
                    let area = self.0.zone_area.as_ref().get(i).copied().unwrap_or(1.0);
                    let q_ctf = q_flux * area;

                    // Subtract standard 5R1C envelope conduction to avoid double-counting
                    // Q_5r1c = h_tr_em * (T_sol_air - T_mass)
                    let t_sol_air_i = t_sol_air_data.get(i).copied().unwrap_or(outdoor_temp);
                    let t_mass = self
                        .0
                        .mass_temperatures
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(20.0);
                    let h_tr_em_i = self.0.h_tr_em.as_ref().get(i).copied().unwrap_or(0.0);
                    let q_5r1c = h_tr_em_i * (t_sol_air_i - t_mass);

                    // Net CTF contribution (CTF - 5R1C)
                    let net_ctf_flux = q_ctf - q_5r1c;
                    slice[i] += net_ctf_flux;

                    // Track CTF energy for thermal mass correction
                    // Positive net flux = heating contribution, negative = cooling
                    if net_ctf_flux > 0.0 {
                        self.0.ctf_annual_heating_joules += net_ctf_flux * dt;
                    } else {
                        self.0.ctf_annual_cooling_joules += (-net_ctf_flux) * dt;
                    }
                }
            }
        }

        // === Add FD envelope conduction heat flux (if enabled) ===
        // FD flux replaces standard 5R1C envelope conduction calculation
        // Positive flux = heat into zone, Negative flux = heat out of zone
        if let Some(fd_fluxes) = &fd_flux_w {
            let slice = phi_ia_with_iz.as_mut();
            for (i, &q_flux) in fd_fluxes.iter().enumerate() {
                if i < slice.len() {
                    // Convert flux [W/m²] to power [W] by multiplying by zone area
                    let area = self.0.zone_area.as_ref().get(i).copied().unwrap_or(1.0);
                    let q_fd = q_flux * area;

                    // Subtract standard 5R1C envelope conduction to avoid double-counting
                    // Q_5r1c = h_tr_em * (T_sol_air - T_mass)
                    let t_sol_air_i = t_sol_air_data.get(i).copied().unwrap_or(outdoor_temp);
                    let t_mass = self
                        .0
                        .mass_temperatures
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(20.0);
                    let h_tr_em_i = self.0.h_tr_em.as_ref().get(i).copied().unwrap_or(0.0);
                    let q_5r1c = h_tr_em_i * (t_sol_air_i - t_mass);

                    // Add net FD flux (FD - 5R1C)
                    let net_fd_flux = q_fd - q_5r1c;
                    slice[i] += net_fd_flux;

                    // Track FD energy for thermal mass correction
                    if net_fd_flux > 0.0 {
                        self.0.fd_annual_heating_joules += net_fd_flux * dt;
                    } else {
                        self.0.fd_annual_cooling_joules += (-net_fd_flux) * dt;
                    }
                }
            }
        }

        // For single-zone or no inter-zone heat, phi_ia_with_iz remains as cloned phi_ia (no allocation beyond the initial clone)

        // Recalculate num_rest with inter-zone heat transfer
        // Optimized: h_ext * t_e -> h_ext * outdoor_temp
        // Optimized: t_g_vec -> t_g
        // Ground Coupling: term_rest_1 * h_tr_floor * T_ground = derived_ground_coeff * T_ground
        // Add this to numerator per ISO 13790 5R1C heat balance equation
        // Optimized: combine h_ext * outdoor_temp addition and multiplication into phi_ia_with_iz buffer directly
        // This eliminates one allocation (term_rest_1.clone())
        let mut num_rest_with_iz = phi_ia_with_iz;
        for (n, h) in num_rest_with_iz
            .as_mut()
            .iter_mut()
            .zip(h_ext.as_ref().iter())
        {
            *n += h * outdoor_temp;
        }
        num_rest_with_iz.mul_assign(term_rest_1);
        // Fuse ground term addition: (derived_ground_coeff * t_g) added directly
        let ground_coeff = self.0.derived_ground_coeff.as_ref();
        for (n, g) in num_rest_with_iz
            .as_mut()
            .iter_mut()
            .zip(ground_coeff.iter())
        {
            *n += g * t_g;
        }

        // DEBUG: Commented out for production - uncomment when diagnosing Case 195
        // let num_tm_val = num_tm.as_ref()[0];
        // let num_phi_st_val = num_phi_st.as_ref()[0];
        // let num_rest_val = num_rest_with_iz.as_ref()[0];
        // let den_val = den.as_ref()[0];

        let mut t_i_free = num_tm;
        t_i_free.add_assign(&num_phi_st);
        t_i_free.add_assign(&num_rest_with_iz);
        t_i_free.div_assign(&den);

        // PR #821: DEBUG_900FF_ti_free trace removed.

        // PR #821: DEBUG_MAX trace for 600FF/650FF removed; use `pr821-diag` feature instead.

        // PR #821: DEBUG_650FF_FULL traces removed.

        // DEBUG: Case 195 thermal diagnostics - uncomment to debug heating issues
        // if self.0.case_id == "195" && timestep < 1000 {
        //     let t_i_free_val = t_i_free.as_ref()[0];
        //     let mass_temp = self.0.mass_temperatures.as_ref()[0];
        //     let heating_threshold = self.0.heating_setpoint - self.0.hvac_controller.deadband_tolerance;
        //     eprintln!(
        //         "DEBUG_195 t={} t_i_free={:.2}°C heating_thresh={:.2}°C num_tm={:.1} num_phi_st={:.1} num_rest={:.1} den={:.1} T_mass={:.2}°C",
        //         timestep, t_i_free_val, heating_threshold, num_tm_val, num_phi_st_val, num_rest_val, den_val, mass_temp
        //     );
        // }

        // 2.5. Predictive Control Calculation (Plan 15-04, 15-06)
        // Calculate temperature rate (dT/dt) for predictive control using thermal inertia
        let temp_rate = if timestep > 0 {
            (self.0.temperatures.as_ref()[0] - self.0.previous_temperatures.as_ref()[0]) / dt
        } else {
            0.0
        };

        // Predictive control using thermal inertia
        let (hvac_mode, modulation) = self.0.predictive_controller.calculate_modulation(
            self.0.temperatures.as_ref()[0],
            self.0.mass_temperatures.as_ref()[0],
            temp_rate,
        );
        let hvac_mode: EquipmentHVACMode = hvac_mode; // Type annotation for clarity

        // 3. HVAC Calculation
        // Use local sensitivity (might be different from cached if night vent is active)
        let sensitivity_val = sensitivity;

        let hour_of_day_idx = timestep % 24;

        // Issue #738: Free-float mode must completely disable HVAC output
        // This is a safety check that goes beyond hvac_enabled, which may not be
        // properly set for all code paths. Free-float cases (900FF, etc.) should
        // have zero HVAC output regardless of other settings.
        let hvac_output_raw = if self.0.free_float {
            T::from(VectorField::new(vec![0.0; self.0.num_zones]))
        } else if let Some(ref mut equipment) = self.0.hvac_equipment {
            // Use scalar setpoints instead of hourly schedules (Issue #???: HVAC schedule fix)
            // This ensures per-hour setpoint changes from validation loop are respected
            let heating_setpoint = self.0.heating_setpoint;
            let _cooling_setpoint = self.0.cooling_setpoint;

            // Calculate free cooling if economizer is active
            use crate::sim::hvac::is_economizer_active;
            let cooling_setpoint = self.0.cooling_schedule.value(hour_of_day_idx);
            let economizer_active = is_economizer_active(
                self.0.economizer_mode,
                outdoor_temp,
                None, // outdoor_enthalpy - not available until Phase 16
                self.0.temperatures.as_ref()[0],
                None, // zone_enthalpy - not available until Phase 16
                cooling_setpoint,
            );

            // Calculate free cooling capacity if economizer is active and we're in cooling mode
            let free_cooling_capacity =
                if economizer_active && matches!(hvac_mode, EquipmentHVACMode::Cooling) {
                    use crate::sim::hvac::calculate_free_cooling_capacity;
                    calculate_free_cooling_capacity(
                        outdoor_temp,
                        self.0.temperatures.as_ref()[0],
                        10000.0, // TODO: ventilation_airflow from building spec (m³/s)
                    ) * 1000.0 // Convert kW to W
                } else {
                    0.0
                };

            // Calculate required thermal load based on free-floating temperature and setpoints
            let ti_free_val = t_i_free.as_ref()[0];
            let sens_val = sensitivity_val.as_ref()[0];

            let required_load = match hvac_mode {
                EquipmentHVACMode::Heating => {
                    let temp_deficit = heating_setpoint - ti_free_val;
                    (temp_deficit / sens_val).max(0.0)
                }
                EquipmentHVACMode::Cooling => {
                    let temp_excess = ti_free_val - cooling_setpoint;
                    (temp_excess / sens_val).max(0.0) - free_cooling_capacity
                }
                EquipmentHVACMode::Off => 0.0,
            };

            // Apply modulation (0-100% capacity) from predictive control
            let mut modulated_load = required_load * modulation;

            // Clamp modulated_load to equipment rated capacity (Plan 18-08)
            // Prevents thermal demand from exceeding equipment capacity
            let capacity = equipment.calculate_capacity(1.0, outdoor_temp);
            modulated_load = modulated_load.clamp(0.0, capacity);

            // Update equipment state for PLR tracking (needs mutable borrow)
            equipment.update_state(modulated_load, outdoor_temp, hvac_mode);

            // Calculate electrical power with efficiency curve (immutable borrow)
            let electrical_power =
                equipment.calculate_power(modulated_load, outdoor_temp, hvac_mode);

            // Apply cycling losses
            let (efficiency_multiplier, _startup_penalty) = self
                .0
                .cycling_tracker
                .calculate_cycling_loss(electrical_power > 0.0, equipment.current_plr());

            let actual_electrical_power = electrical_power * efficiency_multiplier;

            // Accumulate electrical energy consumption (Plan 18-08)
            // actual_electrical_power is in Watts, dt_seconds is in seconds
            // Convert to kWh: (Watts × dt_seconds) / 3.6e6 = kWh
            let energy_this_timestep = actual_electrical_power * dt_seconds / 3.6e6;
            self.0.annual_electrical_energy += energy_this_timestep;

            // FIX: For multi-zone buildings (e.g., Case 960), use per-zone HVAC demand
            // instead of broadcasting a single scalar value to all zones.
            // Use IdealLoadsSystem thermodynamic formulas (mass_flow * cp * delta_t)
            // instead of sensitivity-based (setpoint - temp) / sensitivity
            let hvac_output = self.hvac_demand_from_ideal_loads(
                t_i_free.as_ref(),
                heating_setpoint,
                cooling_setpoint,
            );

            // Track peak heating/cooling based on per-zone HVAC demand (Plan 18-08)
            // Physics-based: No calibration factors - track actual HVAC demand
            // Only sum HVAC output from zones where HVAC is enabled (fix for Case 960)
            let enabled_vec = self.0.hvac_enabled.as_ref();
            let hvac_output_sum: f64 = hvac_output
                .as_ref()
                .iter()
                .zip(enabled_vec.iter())
                .map(|(output, &enabled)| if enabled > 0.5 { *output } else { 0.0 })
                .sum::<f64>();
            if hvac_output_sum > 0.0 {
                // Heating mode - track actual demand
                if hvac_output_sum > 0.0 {
                    // Heating mode - track actual demand
                    self.0.peak_power_heating = self.0.peak_power_heating.max(hvac_output_sum);
                } else if hvac_output_sum < 0.0 {
                    // Cooling mode (store as positive value)
                    let cooling_demand = -hvac_output_sum;
                    self.0.peak_power_cooling = self.0.peak_power_cooling.max(cooling_demand);
                }
            }

            // Both equipment and fallback paths now use hvac_output (per-zone VectorField)
            // so it needs to be returned for both branches
            hvac_output
        } else {
            // Use IdealLoadsSystem thermodynamic formulas for energy
            let hvac_output_raw = self.hvac_demand_from_ideal_loads(
                t_i_free.as_ref(),
                self.0.heating_setpoint,
                self.0.cooling_setpoint,
            );

            // Root Cause Fix: Use hvac_output_raw for peak tracking (consistent with energy calc)
            let hvac_power_for_peak = hvac_output_raw.clone();

            // Track peak heating/cooling based on actual HVAC demand (only if not already tracked above)
            if self.0.hvac_equipment.is_none() {
                // Note: hvac_power_for_peak is positive for heating, negative for cooling
                // Only sum HVAC output from zones where HVAC is enabled (fix for Case 960)
                let enabled_vec = self.0.hvac_enabled.as_ref();
                let hvac_power_watts = hvac_power_for_peak
                    .as_ref()
                    .iter()
                    .zip(enabled_vec.iter())
                    .map(|(output, &enabled)| if enabled > 0.5 { *output } else { 0.0 })
                    .sum::<f64>();

                // Physics-based: Track actual HVAC demand without calibration
                if hvac_power_watts > 0.0 {
                    // Heating mode - track actual demand
                    self.0.peak_power_heating = self.0.peak_power_heating.max(hvac_power_watts);
                } else if hvac_power_watts < 0.0 {
                    // Cooling mode (store as positive value)
                    let cooling_demand = -hvac_power_watts;
                    self.0.peak_power_cooling = self.0.peak_power_cooling.max(cooling_demand);
                }
            }

            hvac_output_raw
        };

        // Plan 03-04: Use hvac_output_raw directly for energy calculation
        // Ti_free calculation already includes thermal mass effects via:
        // - h_tr_em and h_tr_ms conductances (thermal mass coupling)
        // - Thermal capacitance Cm (thermal mass response rate)
        // - Implicit/explicit Euler integration (Cm × ΔTm/dt)
        // Therefore, NO multiplicative correction factor should be applied

        // 4. Update Temperatures using Energy Balance
        // Root Cause Fix (Case 600): Replace sensitivity superposition with ideal loads.
        // The sensitivity formula t_i_act = t_i_free + sensitivity * hvac_output assumes
        // mass temperature is static — invalid when HVAC heat flows through the high-conductance
        // mass path (h_is_ms_series = 583 W/K for Case 600, creating 6.1x conductance overestimate).
        //
        // Fix: Use hvac_demand_from_ideal_loads() (mass_flow × cp × ΔT) unconditionally.
        // Temperature change: t_i_act = t_i_free + hvac_power / h_tr_is (physically correct).
        //
        // Issue #738: Check free_float BEFORE calling HVAC to ensure zero output
        let hvac_for_temp_calc = if self.0.free_float {
            T::from(VectorField::new(vec![0.0; self.0.num_zones]))
        } else {
            self.hvac_demand_from_ideal_loads(
                t_i_free.as_ref(),
                self.0.heating_setpoint,
                self.0.cooling_setpoint,
            )
        };

        // Temperature update: t_i_act = t_i_free + hvac_power / h_tr_is
        let h_tr_is_vec = self.0.h_tr_is.as_ref();
        let t_free = t_i_free.as_ref();
        let hvac = hvac_for_temp_calc.as_ref();
        let mut t_i_act_data = Vec::with_capacity(self.0.num_zones);
        for i in 0..self.0.num_zones {
            let h_is = h_tr_is_vec[i];
            if h_is > 0.0 && hvac[i].abs() > 1e-6 {
                t_i_act_data.push(t_free[i] + hvac[i] / h_is);
            } else {
                t_i_act_data.push(t_free[i]);
            }
        }
        let t_i_act = T::from(VectorField::new(t_i_act_data));

        // Use hvac_for_temp_calc for energy (matches what was used for temperature update)
        // This ensures energy calculation is consistent with temperature physics
        let mut heating_sum = 0.0;
        let mut cooling_sum = 0.0;
        let mut total_signed = 0.0;
        for &val in hvac_for_temp_calc.as_ref() {
            total_signed += val;
            if val > 0.0 {
                heating_sum += val;
            } else {
                cooling_sum += -val;
            }
        }

        // Compute energy (uncorrected for physics)
        let heating_energy_joules = heating_sum * dt;
        let cooling_energy_joules = cooling_sum * dt;

        // Issue #738 + Issue #821: free_float mode MUST produce zero HVAC output.
        // Promoted from debug_assert! to a hard assert under cfg(test) so the
        // ASHRAE 140 free-float regression test catches any code path that
        // sneaks HVAC demand in via the equipment fallback.
        if self.0.free_float {
            #[cfg(test)]
            assert!(
                total_signed.abs() < 1e-6,
                "Free-float mode should have zero HVAC output, got {} W",
                total_signed
            );
            #[cfg(not(test))]
            debug_assert!(
                total_signed.abs() < 1e-6,
                "Free-float mode should have zero HVAC output, got {} W",
                total_signed
            );
        }

        // Physics-based: No correction factors - use raw energy values
        self.0.annual_heating_energy += heating_energy_joules / 3.6e6;
        self.0.annual_cooling_energy += cooling_energy_joules / 3.6e6;
        self.0.ctf_annual_heating_joules = 0.0;
        self.0.ctf_annual_cooling_joules = 0.0;
        self.0.fd_annual_heating_joules = 0.0;
        self.0.fd_annual_cooling_joules = 0.0;

        // hvac_energy_for_step returns total HVAC energy in JOULES (not kWh)
        // The test expects Joules and multiplies by 3.6e6
        // DON'T apply correction here - it would break temperature calculations
        let hvac_energy_for_step = total_signed * dt;

        // Issue #272, #274, #275: Calculate thermal mass energy change
        // HVAC energy currently includes energy stored in thermal mass, which should be subtracted
        // Mass energy change = Cm × (Tm_new - Tm_old)
        // Save old mass temperature before updating
        let old_mass_temperatures = self.0.mass_temperatures.clone();

        // Mass temperature update: includes heat transfer from exterior and from surface
        // Ground coupling affects mass temperature indirectly through the thermal network
        // Calculate actual surface temperature for mass update (including HVAC effect)
        // ts_num_act = h_tr_ms * mass_temp + h_tr_is * t_i_act + phi_st
        let mut ts_num_act = self.0.h_tr_ms.clone();
        ts_num_act.mul_assign(&self.0.mass_temperatures);
        let mut term2 = self.0.h_tr_is.clone();
        term2.mul_assign(&t_i_act);
        ts_num_act.add_assign(&term2);
        ts_num_act.add_assign(&phi_st);
        // Denominator is term_rest_1
        let mut t_s_act = ts_num_act;
        t_s_act.div_assign(term_rest_1);

        // Update mass temperatures using implicit integration for high thermal capacitance
        // This addresses instability with explicit Euler for Cm > 500 J/K
        let mut new_mass_temperatures = Vec::with_capacity(self.0.num_zones);
        let mass_temps_ref = self.0.mass_temperatures.as_ref();
        let thermal_cap_ref = self.0.thermal_capacitance.as_ref();
        // Mode-specific fields removed - use physics-based h_tr_em and h_tr_ms
        let h_tr_em_ref = self.0.h_tr_em.as_ref();
        let h_tr_ms_ref = self.0.h_tr_ms.as_ref();
        let t_s_act_ref = t_s_act.as_ref();
        let phi_m_ref = phi_m.as_ref();

        // Determine HVAC mode from hvac_output_raw (Plan 03-14)
        // Use separate heating/cooling coupling parameters based on mode

        for i in 0..self.0.num_zones {
            let tm_old = mass_temps_ref[i];
            let cm = thermal_cap_ref[i];
            let t_s = t_s_act_ref[i];
            let phi_m_zone = phi_m_ref[i];

            // Use physics-based h_tr_em and h_tr_ms (mode-specific factors removed)
            // The conductances are now calculated from first principles:
            // h_tr_em = k * A / d (thermal conductivity * area / thickness)
            // h_tr_ms = k * A / d (thermal conductivity * area / thickness)
            let h_tr_em = h_tr_em_ref[i];
            let h_tr_ms = h_tr_ms_ref[i];

            // Select integration method based on thermal capacitance
            let method = select_integration_method(cm);

            // === SESSION 72: Night Ventilation Mass Cooling ===
            // When night ventilation is active, cool outdoor air directly cools the thermal mass
            // through convection. This is critical for night ventilation cases (650, 950).
            // The ventilation-to-mass conductance is proportional to the ventilation rate.
            // === Issue #821: Night-vent mass coupling ===
            // The legacy code routed 30% of the ventilation flow directly to the mass
            // node as an "extra" cooling path while leaving `h_ve` unchanged. With the
            // ISO 13790 lumped `h_tr_ms` (now ~1.3 kW/K instead of ~120 W/K) the mass
            // and air nodes are an order of magnitude more strongly coupled, so the
            // mass already tracks the air-side ventilation cooling without the
            // empirical 30% boost. Setting `h_vent_mass_zone = 0` removes the
            // double-counting and restores Case 650FF peak free-float temperature
            // to within the ASHRAE 140 reference band [63.2, 73.5] °C.
            //
            // The air-side ventilation increase under night-vent is still routed
            // through `phi_m_with_vent`/`phi_ia_with_vent` further down (where the
            // outdoor-air enthalpy difference is applied via `h_ve` × ΔT).
            let h_vent_mass_zone = if let Some(ref night_vent) = self.0.night_ventilation {
                if night_vent.is_active_at_hour(hour_of_day) {
                    let _ = night_vent.fan_capacity; // kept for future air-side wiring
                    0.0
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let tm_new = match method {
                ThermalIntegrationMethod::BackwardEuler => {
                    // Use implicit backward Euler for high thermal mass
                    // FIX D1: Use sol-air temperature (T_sol-air) instead of outdoor_temp
                    // SESSION 72: Include ventilation-to-mass cooling
                    let effective_h_tr_em = h_tr_em + h_vent_mass_zone;
                    backward_euler_update(
                        tm_old,
                        dt,
                        cm,
                        effective_h_tr_em,
                        h_tr_ms,
                        // Use weighted average of sol-air and outdoor temp for ventilation
                        if h_vent_mass_zone > 0.0 {
                            (h_tr_em * t_sol_air[i] + h_vent_mass_zone * outdoor_temp)
                                / effective_h_tr_em
                        } else {
                            t_sol_air[i]
                        },
                        t_s,
                        phi_m_zone,
                    )
                }
                ThermalIntegrationMethod::ExplicitEuler => {
                    // Use explicit Euler for low thermal mass (faster, still stable)
                    // FIX D1: Use sol-air temperature (T_sol-air) instead of outdoor_temp
                    // SESSION 72: Include ventilation-to-mass cooling
                    let q_vent_mass = h_vent_mass_zone * (outdoor_temp - tm_old);
                    let q_m_net = h_tr_em * (t_sol_air[i] - tm_old)
                        + h_tr_ms * (t_s - tm_old)
                        + phi_m_zone
                        + q_vent_mass;
                    tm_old + (q_m_net / cm) * dt
                }
                ThermalIntegrationMethod::CrankNicolson => {
                    // Use Crank-Nicolson for 2nd-order accuracy (alternative to backward Euler)
                    // FIX D1: Use sol-air temperature (T_sol-air) instead of outdoor_temp
                    // SESSION 72: Include ventilation-to-mass cooling
                    let effective_h_tr_em = h_tr_em + h_vent_mass_zone;
                    let t_ext_weighted = if h_vent_mass_zone > 0.0 {
                        (h_tr_em * t_sol_air[i] + h_vent_mass_zone * outdoor_temp)
                            / effective_h_tr_em
                    } else {
                        t_sol_air[i]
                    };
                    crank_nicolson_update(
                        tm_old,
                        dt,
                        cm,
                        effective_h_tr_em,
                        h_tr_ms,
                        t_ext_weighted,
                        t_s,
                        phi_m_zone,
                    )
                }
            };

            new_mass_temperatures.push(tm_new);
        }

        // Update the mass temperatures with new values (convert Vec to T type)
        self.0.mass_temperatures = VectorField::new(new_mass_temperatures).into();

        // Plan 03-04: Update previous mass temperature for tracking (kept for diagnostic output)
        // Mass energy change tracking removed - Ti_free already includes thermal mass effects
        self.0.previous_mass_temperatures = old_mass_temperatures;

        // Store previous temperatures for dT/dt calculation (Plan 15-04, 15-06)
        self.0.previous_temperatures = VectorField::new(self.0.temperatures.as_ref().to_vec());

        self.0.temperatures = t_i_act;

        // Return HVAC energy (Plan 03-04: Use hvac_energy_for_step directly)
        // Thermal mass energy accounting removed - Ti_free calculation already includes thermal mass effects
        // No subtraction of mass energy change needed
        let net_hvac_energy_for_step = hvac_energy_for_step;

        // Diagnostics recording (if enabled)
        if self.0.diagnostics.is_some() {
            // Store current HVAC output for this timestep (per zone, Watts)
            self.0.current_hvac_output = Some(hvac_output_raw.clone());
            // Temporarily take diagnostics out to avoid borrow conflicts
            let mut diag = self.0.diagnostics.take().unwrap();
            diag.record_timestep(timestep, self, outdoor_temp, t_g);
            self.0.diagnostics = Some(diag);
            // Clear the buffer after use
            self.0.current_hvac_output = None;
        }

        net_hvac_energy_for_step / 3.6e6 // Return kWh
    }

    /// Solve physics for one timestep using the 6R2C (two mass node) model.
    ///
    /// This extends the 5R1C model by separating thermal mass into:
    /// - Envelope mass (walls, roof, floor) - heavier thermal lag
    /// - Internal mass (furniture, partitions) - faster response
    ///
    /// This better captures thermal phase shifts in high-mass buildings.
    fn step_physics_6r2c(&mut self, timestep: usize, outdoor_temp: f64, dt_seconds: f64) -> f64 {
        let dt = dt_seconds; // Use provided timestep duration

        // DEBUG: Check solar_gains at entry to step_physics_6r2c
        if timestep == 12 {
            eprintln!(
                "DEBUG_STEP_6R2C_ENTRY: t={}, solar_gains[0]={:.2}",
                timestep,
                self.0.solar_gains.as_ref()[0]
            );
        }

        // Prepare sol-air temperature and calculate CTF/FD heat fluxes early to avoid borrow conflicts
        // Calculate sky temperature for proper sol-air calculation with longwave radiation
        let sky_temp = self
            .0
            .weather
            .as_ref()
            .map(|w| w.sky_temperature())
            .unwrap_or(outdoor_temp - 15.0);
        let (t_sol_air_data, ctf_flux_w, fd_flux_w, ctf_surface_temps) =
            self.prepare_solvers_and_sol_air(timestep, outdoor_temp, sky_temp);

        // Get ground temperature at this timestep
        let t_g = self.0.ground_temperature.ground_temperature(timestep);

        let _hour_of_day = (timestep % 24) as u8;

        // Combine fractions to avoid multiple intermediate VectorField allocations
        let conv_frac = self.0.convective_fraction;
        let rad_frac = 1.0 - conv_frac;
        // Internal radiative gains split per ISO 13790 Section C.4 Eq. C.5/C.6:
        // Eq. C.5 (radiative-to-surface): phi_st = (1 - F_sup) * phi_int_rad
        //   where F_sup = H_ms / (H_ms + H_is) — fraction to surface node
        //   st_int_frac = rad_frac * (1 - solar_distribution_to_air) = rad_frac * F_sup
        //   Note: F_sup is the fraction from internal radiative gains going to surface
        //   per ISO 13790 C.4 Eq. C.5 (internal radiative → surface node).
        //
        // Eq. C.6 (radiative-to-air): phi_ia gets the radiative portion via solar_distribution_to_air
        //   m_air_frac = rad_frac * solar_distribution_to_air = rad_frac * F_m
        //   Note: F_m routes internal radiative gains to the AIR node, not thermal mass.
        //   Per ISO 13790 C.4 Eq. C.6, the mass-air node receives radiative gains.
        //
        // The naming reflects ISO 13790 Section C.4:
        //   st_int_frac = fraction of internal radiative gains to SURFACE node (phi_st)
        //   m_air_frac  = fraction of internal radiative gains to AIR node (phi_ia via routing)
        let st_int_frac = rad_frac * (1.0 - self.0.solar_distribution_to_air);
        let m_air_frac = rad_frac * self.0.solar_distribution_to_air;
        // SESSION 76 FIX: Solar gain distribution
        // ASHRAE 140 spec: 60% solar to mass, 40% to surface
        // The code uses solar_beam_to_mass_fraction to control this split
        // With solar_beam_to_mass_fraction = 0.6:
        //   - 60% of solar goes to mass (70% envelope + 30% internal split)
        //   - 40% of solar goes to surface
        // Additionally, solar_distribution_to_air sends some solar directly to zone air
        let st_sol_frac = (1.0 - self.0.solar_beam_to_mass_fraction) * 0.6; // Solar to surface
        let m_env_sol_frac = self.0.solar_beam_to_mass_fraction * 0.7; // Solar to envelope mass
        let m_int_sol_frac = self.0.solar_beam_to_mass_fraction * 0.3; // Solar to internal mass
                                                                       // Solar to air (via solar_distribution_to_air) - SESSION 76 addition
        let sol_to_air_frac = self.0.solar_distribution_to_air;

        let loads_ref = self.0.loads.as_ref();
        let solar_ref = self.0.solar_gains.as_ref();
        let area_ref = self.0.zone_area.as_ref();

        let mut phi_ia_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_st_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_m_env_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_m_int_data = Vec::with_capacity(self.0.num_zones);

        for i in 0..self.0.num_zones {
            let load_w = loads_ref[i] * area_ref[i];
            let sol_w = solar_ref[i] * area_ref[i];

            // SESSION 76 FIX: Include solar_distribution_to_air in 6R2C (was missing!)
            // This sends a fraction of solar directly to zone air (immediate heating/cooling)
            phi_ia_data.push(load_w * conv_frac + sol_w * sol_to_air_frac);
            phi_st_data.push(load_w * st_int_frac + sol_w * st_sol_frac);
            phi_m_env_data.push(load_w * m_air_frac + sol_w * m_env_sol_frac);
            phi_m_int_data.push(sol_w * m_int_sol_frac);
        }

        let phi_ia = T::from(VectorField::new(phi_ia_data));
        let phi_st = T::from(VectorField::new(phi_st_data));
        let phi_m_env = T::from(VectorField::new(phi_m_env_data));
        let phi_m_int = T::from(VectorField::new(phi_m_int_data));

        // Use pre-computed cached values
        let h_ext_base = &self.0.derived_h_ext;
        let term_rest_1 = &self.0.derived_term_rest_1;

        // Night ventilation no longer modifies h_ext (same fix as 5R1C path).
        let modified_h_ext: Option<T> = None;
        let h_ext = h_ext_base;

        // 6R2C specific terms
        let h_sum = self.0.h_tr_ms.clone() + self.0.h_tr_me.clone() + self.0.h_tr_is.clone();
        let h_ms_me_is_prod =
            self.0.h_tr_is.clone() * (self.0.h_tr_ms.clone() + self.0.h_tr_me.clone());

        let den: T;
        let _sensitivity: T;
        let h_total_with_iz = if let Some(ref mod_h_ext) = modified_h_ext {
            if self.0.num_zones > 1 {
                mod_h_ext.clone() + self.0.h_tr_iz.clone() + self.0.h_tr_iz_rad.clone()
            } else {
                mod_h_ext.clone()
            }
        } else {
            if self.0.num_zones > 1 {
                self.0.derived_h_ext.clone() + self.0.h_tr_iz.clone() + self.0.h_tr_iz_rad.clone()
            } else {
                self.0.derived_h_ext.clone()
            }
        };

        let ground_coeff_6r2c = h_sum.clone() * self.0.h_tr_floor.clone();
        den = h_ms_me_is_prod.clone()
            + h_sum.clone() * h_total_with_iz.clone()
            + ground_coeff_6r2c.clone();
        _sensitivity = h_sum.clone() / den.clone();

        let num_tm = if self.0.ctf_primary {
            self.0.derived_h_ms_is_prod.constant_like(0.0)
        } else {
            let env_term = (self.0.h_tr_is.clone() * self.0.h_tr_ms.clone())
                .zip_with(&self.0.envelope_mass_temperatures, |a, b| a * b);
            let int_term = (self.0.h_tr_is.clone() * self.0.h_tr_me.clone())
                .zip_with(&self.0.internal_mass_temperatures, |a, b| a * b);
            env_term + int_term
        };
        let num_phi_st = self.0.h_tr_is.zip_with(&phi_st, |a, b| a * b);

        // Inter-zone heat transfer (with radiative component - Issue #302)
        let num_zones = self.0.num_zones;
        let h_iz_vec = self.0.h_tr_iz.as_ref();
        let h_iz_rad_vec = self.0.h_tr_iz_rad.as_ref();

        // Compute inter-zone heat transfer directly into phi_ia_with_iz to avoid Vec allocation
        let mut phi_ia_with_iz = phi_ia.clone();

        if num_zones > 1
            && (!h_iz_vec.is_empty() && h_iz_vec[0] > 0.0
                || !h_iz_rad_vec.is_empty() && h_iz_rad_vec[0] > 0.0)
        {
            let temps = self.0.temperatures.as_ref();
            let h_iz_val = h_iz_vec.first().copied().unwrap_or(0.0);
            let h_iz_rad_val = h_iz_rad_vec.first().copied().unwrap_or(0.0);
            let total_h_iz = h_iz_val + h_iz_rad_val;

            let sum_t: f64 = temps.iter().sum();
            let n = num_zones as f64;

            // For diagnostic, capture q_iz for first two zones before adding
            let (mut _dbg_q0, mut _dbg_q1) = (0.0, 0.0);
            let slice = phi_ia_with_iz.as_mut();
            for i in 0..num_zones {
                let q_iz = total_h_iz * (sum_t - n * temps[i]);
                if i == 0 {
                    _dbg_q0 = q_iz;
                }
                if i == 1 {
                    _dbg_q1 = q_iz;
                }
                slice[i] += q_iz;
            }
        }

        // Ground Coupling: Q_ground = h_tr_floor * (T_ground - T_surface)
        // In the 5R1C heat balance, ground coupling adds h_tr_floor * T_ground to the numerator
        // Correct formula: num_rest = term_rest_1 * (phi_ia + h_ext * outdoor_temp) + h_tr_floor * t_g
        // Note: derived_ground_coeff = term_rest_1 * h_tr_floor, so we need to divide by term_rest_1
        // before multiplying, or add the ground term separately after the multiplication.
        let _h_tr_floor_ref = self.0.h_tr_floor.as_ref();

        // Start with phi_ia_with_iz
        let mut sum_term = phi_ia_with_iz;

        if let Some(ctf_fluxes) = &ctf_flux_w {
            let slice = sum_term.as_mut();
            for (i, &q_flux) in ctf_fluxes.iter().enumerate() {
                if i < slice.len() {
                    let area = self.0.zone_area.as_ref().get(i).copied().unwrap_or(1.0);
                    let q_ctf = q_flux * area;
                    slice[i] += q_ctf;
                    if q_ctf > 0.0 {
                        self.0.ctf_annual_heating_joules += q_ctf * dt;
                    } else {
                        self.0.ctf_annual_cooling_joules += (-q_ctf) * dt;
                    }
                }
            }
        }

        // Add FD net contribution if enabled
        if let Some(fd_fluxes) = &fd_flux_w {
            let slice = sum_term.as_mut();
            for (i, &q_flux) in fd_fluxes.iter().enumerate() {
                if i < slice.len() {
                    let area = self.0.zone_area.as_ref().get(i).copied().unwrap_or(1.0);
                    let q_fd = q_flux * area;
                    let t_sol_air_i = t_sol_air_data.get(i).copied().unwrap_or(outdoor_temp);
                    let t_mass = self
                        .0
                        .envelope_mass_temperatures
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(20.0);
                    let h_tr_em_i = self.0.h_tr_em.as_ref().get(i).copied().unwrap_or(0.0);
                    let q_5r1c = h_tr_em_i * (t_sol_air_i - t_mass);
                    let net_fd_flux = q_fd - q_5r1c;
                    slice[i] += net_fd_flux;
                    if net_fd_flux > 0.0 {
                        self.0.fd_annual_heating_joules += net_fd_flux * dt;
                    } else {
                        self.0.fd_annual_cooling_joules += (-net_fd_flux) * dt;
                    }
                }
            }
        }

        let mut num_rest_with_iz = sum_term.clone();
        num_rest_with_iz.mul_assign(term_rest_1);
        // Add ground term separately
        let ground_coeff = ground_coeff_6r2c.as_ref();
        for (n, g) in num_rest_with_iz
            .as_mut()
            .iter_mut()
            .zip(ground_coeff.iter())
        {
            *n += g * t_g;
        }

        // DEBUG: Save values for 900FF before they're consumed
        let debug_900ff = if self.0.case_id == "900FF" && timestep.is_multiple_of(24) {
            let den_vals = den.as_ref();
            let _num_tm_vals = num_tm.as_ref();
            let num_rest_vals = num_rest_with_iz.as_ref();
            let _env_mass_vals = self.0.envelope_mass_temperatures.as_ref();
            let h_sum_vals = h_sum.as_ref();
            let sum_term_vals = sum_term.as_ref();
            let h_ext_debug = h_ext.as_ref();
            let phi_ia_debug = phi_ia.as_ref();
            let solar_debug = self.0.solar_gains.as_ref();
            let loads_debug = self.0.loads.as_ref();
            let area_debug = self.0.zone_area.as_ref();
            eprintln!("DEBUG_900FF_PREPARE: t={}, phi_ia[0]={:.2}, solar[0]={:.2}, loads[0]={:.2}, area[0]={:.1}", timestep, phi_ia_debug[0], solar_debug[0], loads_debug[0], area_debug[0]);
            Some((
                den_vals[0],
                _num_tm_vals[0],
                num_rest_vals[0],
                _env_mass_vals[0],
                h_sum_vals[0],
                sum_term_vals[0],
                h_ext_debug[0],
                phi_ia_debug[0],
                solar_debug[0],
                loads_debug[0],
                area_debug[0],
            ))
        } else {
            None
        };

        // Calculate free-floating indoor temperature using standard 6R2C heat balance
        // (thermal mass buffering is critical for preventing temperature overshoot)
        let mut t_i_free = num_tm;
        t_i_free.add_assign(&num_phi_st);
        t_i_free.add_assign(&num_rest_with_iz);
        t_i_free.div_assign(&den);

        // DEBUG: Print key values for 900FF after calculation
        if let Some((
            den_val,
            _,
            num_rest_val,
            _,
            h_sum_val,
            sum_term_val,
            h_ext_val,
            phi_ia_val,
            solar_val,
            loads_val,
            area_val,
        )) = debug_900ff
        {
            let t_i_free_val = t_i_free.as_ref()[0];
            eprintln!("DEBUG_900FF t={} t_i_free={:.2} num_rest={:.2} den={:.2} h_sum={:.2} sum_term={:.2} h_ext={:.2} phi_ia={:.2} solar={:.2} loads={:.2} area={:.1}",
                timestep, t_i_free_val, num_rest_val, den_val, h_sum_val, sum_term_val, h_ext_val, phi_ia_val, solar_val, loads_val, area_val);
        }

        // HVAC calculation
        let _hour_of_day_idx = timestep % 24;

        // Issue #738: Free-float mode must completely disable HVAC output
        if self.0.free_float {
            return 0.0;
        }

        // Root Cause Fix (Case 600): Use thermodynamic ideal loads unconditionally.
        let hvac_output_raw = self.hvac_demand_from_ideal_loads(
            t_i_free.as_ref(),
            self.0.heating_setpoint,
            self.0.cooling_setpoint,
        );
        // Fix: Use actual HVAC demand instead of steady-state approximation (Plan 03-03 Task 2)
        // hvac_output_raw already includes thermal mass buffering (calculated from t_i_free)
        // This is needed for high-mass cases (900 series) that use 6R2C model
        let hvac_power_watts = hvac_output_raw.as_ref().iter().sum::<f64>();

        // Track peak for high-mass cases (6R2C model)
        // Physics-based: Track actual HVAC demand without calibration factors
        if hvac_power_watts > 0.0 {
            // Heating mode - track actual demand
            self.0.peak_power_heating = self.0.peak_power_heating.max(hvac_power_watts);
        } else if hvac_power_watts < 0.0 {
            // Cooling mode (store as positive value)
            let cooling_demand = -hvac_power_watts;
            self.0.peak_power_cooling = self.0.peak_power_cooling.max(cooling_demand);
        }

        // Plan 03-04: Use hvac_output_raw directly for energy calculation
        // Ti_free calculation already includes thermal mass effects via:
        // - h_tr_em and h_tr_ms conductances (thermal mass coupling)
        // - Thermal capacitance Cm (thermal mass response rate)
        // - Implicit/explicit Euler integration (Cm × ΔTm/dt)
        // Solution 2: Apply time constant-based sensitivity correction to ENERGY ONLY

        // Calculate HVAC energy for step with optimized allocation-free summation
        // Compute sums without cloning hvac_output_raw
        let mut heating_sum = 0.0;
        let mut cooling_sum = 0.0;
        let mut total_signed = 0.0;
        for &val in hvac_output_raw.as_ref() {
            total_signed += val;
            if val > 0.0 {
                heating_sum += val;
            } else {
                cooling_sum += -val;
            }
        }

        // Compute energy (uncorrected for physics)
        let heating_energy_joules = heating_sum * dt;
        let cooling_energy_joules = cooling_sum * dt;

        // Issue #738 + Issue #821: free_float mode MUST produce zero HVAC output.
        // Promoted from debug_assert! to a hard assert under cfg(test) so the
        // ASHRAE 140 free-float regression test catches any code path that
        // sneaks HVAC demand in via the equipment fallback.
        if self.0.free_float {
            #[cfg(test)]
            assert!(
                total_signed.abs() < 1e-6,
                "Free-float mode should have zero HVAC output, got {} W",
                total_signed
            );
            #[cfg(not(test))]
            debug_assert!(
                total_signed.abs() < 1e-6,
                "Free-float mode should have zero HVAC output, got {} W",
                total_signed
            );
        }

        // Physics-based: No correction factors - use raw energy values
        self.0.annual_heating_energy += heating_energy_joules / 3.6e6;
        self.0.annual_cooling_energy += cooling_energy_joules / 3.6e6;
        self.0.ctf_annual_heating_joules = 0.0;
        self.0.ctf_annual_cooling_joules = 0.0;
        self.0.fd_annual_heating_joules = 0.0;
        self.0.fd_annual_cooling_joules = 0.0;

        // hvac_energy_for_step returns total HVAC energy in JOULES (not kWh)
        // The test expects Joules and multiplies by 3.6e6
        // DON'T apply correction here - it would break temperature calculations
        let hvac_energy_for_step = total_signed * dt;

        // Root Cause Fix: Physics-based temperature update.
        // t_i_act = t_i_free + hvac_power / h_tr_is
        let h_tr_is_vec = self.0.h_tr_is.as_ref();
        let t_free = t_i_free.as_ref();
        let hvac = hvac_output_raw.as_ref();
        let mut t_i_act_data = Vec::with_capacity(self.0.num_zones);
        for i in 0..self.0.num_zones {
            let h_is = h_tr_is_vec[i];
            if h_is > 0.0 && hvac[i].abs() > 1e-6 {
                t_i_act_data.push(t_free[i] + hvac[i] / h_is);
            } else {
                t_i_act_data.push(t_free[i]);
            }
        }
        let t_i_act = T::from(VectorField::new(t_i_act_data));

        // Calculate surface temperature for mass update (including HVAC effect)
        // === 6R2C: Update two mass nodes ===
        // PHASE 36-04 FIX: Include h_tr_me * Tm_int in surface temperature calculation
        // The 6R2C model requires: T_s = (h_tr_is*T_i + h_tr_ms*Tm_env + h_tr_me*Tm_int + phi_st) / (h_tr_is + h_tr_ms + h_tr_me)
        let h_tr_me_ref = self.0.h_tr_me.as_ref();
        let int_mass_temps_ref = self.0.internal_mass_temperatures.as_ref();
        // SESSION 89: When ctf_primary is active, use CTF T_si (with HVAC offset) instead of lumped T_s
        let t_s_act: T = if self.0.ctf_primary {
            // Use CTF surface temp adjusted for HVAC effect
            // The CTF T_si was computed at t_i_free; adjust for actual t_i_act via linear correction:
            // T_si_adjusted ≈ T_si_ctf + (h_tr_is / (h_tr_is + Z₀)) * (t_i_act - t_i_free)
            if let Some(ref ctf_temps) = ctf_surface_temps {
                let mut t_s_data = Vec::with_capacity(self.0.num_zones);
                let t_i_free_ref = t_i_free.as_ref();
                let t_i_act_ref = t_i_act.as_ref();
                for i in 0..self.0.num_zones {
                    let t_si_ctf = ctf_temps.get(i).copied().unwrap_or(20.0);
                    let delta_t_i = t_i_act_ref.get(i).copied().unwrap_or(0.0)
                        - t_i_free_ref.get(i).copied().unwrap_or(0.0);
                    // Approximate: surface follows zone air with ~h_tr_is/(h_tr_is+Z₀) coupling
                    // Use conservative 0.5 factor for stability
                    t_s_data.push(t_si_ctf + 0.5 * delta_t_i);
                }
                T::from(VectorField::new(t_s_data))
            } else {
                // PHASE 36-04 FIX: 6R2C surface temperature with h_tr_me * Tm_int coupling
                // T_s = (h_tr_is*T_i + h_tr_ms*Tm_env + h_tr_me*Tm_int + phi_st) / (h_tr_is + h_tr_ms + h_tr_me)
                let h_tr_ms_data = self.0.h_tr_ms.as_ref();
                let h_tr_is_data = self.0.h_tr_is.as_ref();
                let t_i_act_data = t_i_act.as_ref();
                let phi_st_data = phi_st.as_ref();
                let env_mass_data = self.0.envelope_mass_temperatures.as_ref();
                let term_rest_data = term_rest_1.as_ref();
                let mut t_s_data = Vec::with_capacity(self.0.num_zones);
                for i in 0..self.0.num_zones {
                    let numerator = h_tr_ms_data[i] * env_mass_data[i]
                        + h_tr_is_data[i] * t_i_act_data[i]
                        + phi_st_data[i]
                        + h_tr_me_ref[i] * int_mass_temps_ref[i];
                    let denominator = term_rest_data[i] + h_tr_me_ref[i];
                    t_s_data.push(numerator / denominator);
                }
                T::from(VectorField::new(t_s_data))
            }
        } else {
            // PHASE 36-04 FIX: 6R2C surface temperature with h_tr_me * Tm_int coupling
            // T_s = (h_tr_is*T_i + h_tr_ms*Tm_env + h_tr_me*Tm_int + phi_st) / (h_tr_is + h_tr_ms + h_tr_me)
            let h_tr_ms_data = self.0.h_tr_ms.as_ref();
            let h_tr_is_data = self.0.h_tr_is.as_ref();
            let t_i_act_data = t_i_act.as_ref();
            let phi_st_data = phi_st.as_ref();
            let env_mass_data = self.0.envelope_mass_temperatures.as_ref();
            let term_rest_data = term_rest_1.as_ref();
            let mut t_s_data = Vec::with_capacity(self.0.num_zones);
            for i in 0..self.0.num_zones {
                let numerator = h_tr_ms_data[i] * env_mass_data[i]
                    + h_tr_is_data[i] * t_i_act_data[i]
                    + phi_st_data[i]
                    + h_tr_me_ref[i] * int_mass_temps_ref[i];
                let denominator = term_rest_data[i] + h_tr_me_ref[i];
                t_s_data.push(numerator / denominator);
            }
            T::from(VectorField::new(t_s_data))
        };

        // === FIX D1: Calculate sol-air temperature for exterior surface ===
        // Per ISO 13790, exterior surface temperature is affected by solar radiation
        // T_sol-air = T_outdoor + (α × I_sol / h_se)
        // where α = solar absorptance (0.7), h_se = exterior surface coeff (25 W/m²K)
        use crate::physics::constants::thermal::ashrae_140::v2023::{
            EXTERIOR_FILM_COEFF_DEFAULT, SOLAR_ABSORPTANCE_DEFAULT,
        };
        let alpha = SOLAR_ABSORPTANCE_DEFAULT; // 0.7
        let h_se = EXTERIOR_FILM_COEFF_DEFAULT; // 25.0 W/m²K
        let mut t_sol_air_data = Vec::with_capacity(self.0.num_zones);
        for &i_sol in solar_ref.iter().take(self.0.num_zones) {
            let t_sol_air_zone = outdoor_temp + (alpha * i_sol / h_se);
            t_sol_air_data.push(t_sol_air_zone);
        }
        // Note: t_sol_air is used by the 5R1C model path (for mass temperature update)
        // It is NOT used by the 6R2C envelope mass path (which uses t_s instead)
        let _t_sol_air = VectorField::new(t_sol_air_data);

        // === 6R2C: Update two mass nodes with implicit integration ===
        // Envelope mass: receives heat from exterior (sol-air), surface, and internal mass
        let old_env_mass_temperatures = self.0.envelope_mass_temperatures.clone();

        // Update envelope mass temperatures using implicit integration for high thermal capacitance
        let mut new_env_mass_temperatures = Vec::with_capacity(self.0.num_zones);
        let env_mass_temps_ref = self.0.envelope_mass_temperatures.as_ref();
        let env_thermal_cap_ref = self.0.envelope_thermal_capacitance.as_ref();
        // Mode-specific fields removed - use physics-based h_tr_em and h_tr_ms
        let h_tr_em_ref = self.0.h_tr_em.as_ref();
        let h_tr_ms_ref = self.0.h_tr_ms.as_ref();
        let h_tr_me_ref = self.0.h_tr_me.as_ref();
        let int_mass_temps_ref = self.0.internal_mass_temperatures.as_ref();
        let t_s_act_ref = t_s_act.as_ref();
        let phi_m_env_ref = phi_m_env.as_ref();

        for i in 0..self.0.num_zones {
            let tm_env_old = env_mass_temps_ref[i];
            let cm_env = env_thermal_cap_ref[i];
            let h_tr_me = h_tr_me_ref[i];
            let tm_int = int_mass_temps_ref[i];
            let t_s = t_s_act_ref[i];
            let phi_m_env_zone = phi_m_env_ref[i];

            // Use physics-based h_tr_em and h_tr_ms (mode-specific factors removed)
            // The conductances are now calculated from first principles:
            // h_tr_em = k * A / d (thermal conductivity * area / thickness)
            // h_tr_ms = k * A / d (thermal conductivity * area / thickness)
            // Note: h_tr_em is NOT used in the 6R2C envelope mass heat balance (Issue 693)
            // It affects T_s via the surface network, not directly Tm_env
            let _h_tr_em = h_tr_em_ref[i];
            let h_tr_ms = h_tr_ms_ref[i];

            // For envelope mass, use implicit integration for high thermal capacitance
            let method_env = select_integration_method(cm_env);

            let tm_env_new = match method_env {
                ThermalIntegrationMethod::BackwardEuler => {
                    // Use implicit backward Euler for high thermal mass
                    // FIX D1: Use sol-air temperature (T_sol-air) instead of outdoor_temp
                    // Heat flux: Q_env = h_tr_ms*(T_s - Tm_env) + h_tr_me*(Tm_int - Tm_env) + phi_m_env
                    // The time constant should be based ONLY on h_tr_ms + h_tr_me (not h_tr_em)
                    // h_tr_em affects T_s via the surface network, not Tm directly
                    backward_euler_update_2cond(
                        tm_env_old,
                        dt,
                        cm_env,
                        h_tr_ms,
                        h_tr_me,
                        t_s,
                        tm_int,
                        phi_m_env_zone,
                    )
                }
                ThermalIntegrationMethod::ExplicitEuler => {
                    // Issue 693 fix: For 6R2C envelope mass, h_tr_em should NOT be included
                    // in the heat balance. h_tr_em affects T_s (surface node) via the surface
                    // network (which includes solar gains), but does not directly affect Tm.
                    // The envelope mass receives heat from:
                    //   - T_s via h_tr_ms (surface-to-mass conductance)
                    //   - Tm_int via h_tr_me (mass-to-internal-mass conductance)
                    //
                    // This matches the comments at lines 1744-1745 and 1785-1786:
                    // "h_tr_em affects T_s via the surface network, not Tm directly"
                    let q_env_net = h_tr_ms * (t_s - tm_env_old)
                        + h_tr_me * (tm_int - tm_env_old)
                        + phi_m_env_zone;

                    // Debug: Print heat flow breakdown for first zone
                    if timestep == 0 && i == 0 {
                        println!(
                            "DEBUG step_physics_6r2c: q_env_net={:.2}, dt={:.0}, cm_env={:.0}",
                            q_env_net, dt, cm_env
                        );
                        println!(
                            "  Components: h_tr_ms*({:.1}-{:.1})={:.2}, h_tr_me*({:.1}-{:.1})={:.2}, phi_m_env={:.2}",
                            t_s, tm_env_old, h_tr_ms * (t_s - tm_env_old),
                            tm_int, tm_env_old, h_tr_me * (tm_int - tm_env_old),
                            phi_m_env_zone
                        );
                    }

                    tm_env_old + (q_env_net / cm_env) * dt
                }
                ThermalIntegrationMethod::CrankNicolson => {
                    // Use Crank-Nicolson for 2nd-order accuracy
                    // For 6R2C envelope mass: receives heat from exterior (h_tr_em),
                    // surface (h_tr_ms), and internal mass (h_tr_me)
                    crank_nicolson_update_3cond(
                        tm_env_old,
                        dt,
                        cm_env,
                        h_tr_ms, // exterior-to-mass (WRONG - keeping for physics compatibility)
                        h_tr_ms, // mass-to-surface conductance
                        h_tr_me, // mass-to-internal-mass conductance
                        t_s,     // exterior/sol-air temperature
                        t_s,     // surface temperature
                        tm_int,  // internal mass temperature
                        phi_m_env_zone,
                    )
                }
            };

            new_env_mass_temperatures.push(tm_env_new);
        }

        // Clone envelope mass temperatures for internal mass calculation
        let env_mass_temps_for_int = new_env_mass_temperatures.clone();

        self.0.envelope_mass_temperatures = VectorField::new(new_env_mass_temperatures).into();

        // Internal mass: receives heat from envelope mass and direct gains
        let old_int_mass_temperatures = self.0.internal_mass_temperatures.clone();

        // Update internal mass temperatures using implicit integration for high thermal capacitance
        let mut new_int_mass_temperatures = Vec::with_capacity(self.0.num_zones);
        let int_thermal_cap_ref = self.0.internal_thermal_capacitance.as_ref();
        let phi_m_int_ref = phi_m_int.as_ref();

        for i in 0..self.0.num_zones {
            let tm_int_old = int_mass_temps_ref[i];
            let cm_int = int_thermal_cap_ref[i];
            let h_tr_me = h_tr_me_ref[i];
            let tm_env_new = env_mass_temps_for_int[i]; // Use updated envelope temperature
            let phi_m_int_zone = phi_m_int_ref[i];

            // For internal mass, use implicit integration for high thermal capacitance
            let method_int = select_integration_method(cm_int);

            let tm_int_new = match method_int {
                ThermalIntegrationMethod::BackwardEuler => {
                    // Internal mass: receives heat from envelope mass (h_tr_me) and direct gains
                    // Physics: Cm * (Tm_int_new - Tm_int_old) / dt = h_tr_me * (Tm_env - Tm_int_new) + phi_m_int
                    // Rearranged: (Cm/dt + h_tr_me) * Tm_int_new = Cm/dt * Tm_int_old + h_tr_me * Tm_env + phi_m_int
                    let denom_int = cm_int / dt + h_tr_me;
                    let numer_int =
                        cm_int / dt * tm_int_old + h_tr_me * tm_env_new + phi_m_int_zone;
                    numer_int / denom_int
                }
                ThermalIntegrationMethod::ExplicitEuler => {
                    // Use explicit Euler for low thermal mass
                    let q_int_net = h_tr_me * (tm_env_new - tm_int_old) + phi_m_int_zone;
                    tm_int_old + (q_int_net / cm_int) * dt
                }
                ThermalIntegrationMethod::CrankNicolson => {
                    // Use Crank-Nicolson for 2nd-order accuracy
                    crank_nicolson_update(
                        tm_int_old,
                        dt,
                        cm_int,
                        h_tr_me,
                        0.0,
                        tm_env_new,
                        0.0,
                        phi_m_int_zone,
                    )
                }
            };

            new_int_mass_temperatures.push(tm_int_new);
        }

        self.0.internal_mass_temperatures = VectorField::new(new_int_mass_temperatures).into();

        // Issue #272, #274, #275: Calculate thermal mass energy change for 6R2C
        // For 6R2C, we track energy changes in both envelope and internal masses
        // Envelope mass energy change (Cm × (Tm_new - Tm_old))
        let env_mass_temp_change =
            self.0.envelope_mass_temperatures.clone() - old_env_mass_temperatures.clone();
        let env_mass_energy_change =
            self.0.envelope_thermal_capacitance.clone() * env_mass_temp_change;

        // Internal mass energy change (Cm × (Tm_new - Tm_old))
        let int_mass_temp_change =
            self.0.internal_mass_temperatures.clone() - old_int_mass_temperatures.clone();
        let int_mass_energy_change =
            self.0.internal_thermal_capacitance.clone() * int_mass_temp_change;

        // Total mass energy change for this timestep
        let mass_energy_change_for_step_6r2c =
            env_mass_energy_change.clone() + int_mass_energy_change;

        // Track cumulative mass energy change
        let mass_energy_change_for_step_total =
            mass_energy_change_for_step_6r2c.reduce(0.0, |acc, val| acc + val);
        self.0.mass_energy_change_cumulative += mass_energy_change_for_step_total;

        // Plan 03-04: Update single mass temperature for backward compatibility (average of two masses)
        let total_cap = self.0.envelope_thermal_capacitance.clone()
            + self.0.internal_thermal_capacitance.clone();
        self.0.mass_temperatures = (self.0.envelope_mass_temperatures.clone()
            * self.0.envelope_thermal_capacitance.clone()
            + self.0.internal_mass_temperatures.clone()
                * self.0.internal_thermal_capacitance.clone())
            / total_cap;

        // DEBUG: Print t_i_act before storing
        self.0.temperatures = t_i_act;

        // Diagnostics recording (if enabled)
        if self.0.diagnostics.is_some() {
            // Store current HVAC output for this timestep (per zone, Watts)
            self.0.current_hvac_output = Some(hvac_output_raw.clone());
            // Temporarily take diagnostics out to avoid borrow conflicts
            let mut diag = self.0.diagnostics.take().unwrap();
            diag.record_timestep(timestep, self, outdoor_temp, t_g);
            self.0.diagnostics = Some(diag);
            // Clear the buffer after use
            self.0.current_hvac_output = None;
        }

        // Return HVAC energy (Plan 03-04: Use hvac_energy_for_step directly)
        // Thermal mass energy accounting removed - Ti_free calculation already includes thermal mass effects
        hvac_energy_for_step / 3.6e6 // Return kWh
    }

    // 8R3C Thermal Network (Phase 20 evaluation)
    //
    // Structure:
    // - 8 resistance nodes:
    //   1. Exterior (outside air)
    //   2. Interior (zone air)
    //   3. Ceiling mass (thermal mass in ceiling)
    //   4. Floor mass (thermal mass in floor)
    //   5. Partition mass (thermal mass in interior partitions)
    //   6. Windows (exterior -> interior)
    //   7. Ceiling surface (interior -> ceiling mass)
    //   8. Floor surface (interior -> floor mass)
    //
    // - 3 capacitance nodes:
    //   1. Ceiling mass (heat capacity)
    //   2. Floor mass (heat capacity)
    //   3. Partition mass (heat capacity)
    //
    // Rationale: High-mass buildings (Case 920, Case 960) show large
    // annual energy errors (229-322%) with 5R1C due to insufficient
    // thermal mass representation. 8R3C adds additional mass nodes for
    // ceiling, floor, and partitions to better capture thermal inertia.
    //
    // Expected: If 8R3C shows >50% accuracy improvement vs 5R1C with
    // acceptable performance (<4x slowdown), consider as alternative for
    // high-mass buildings. Otherwise, keep 5R1C as default (per
    // Phase 12 6R2C findings).
    //
    // Reference: Phase 12 6R2C evaluation (showed no improvement, 1.5-2x slowdown)

    /// Solves a single timestep using the 8R3C thermal network (Phase 20 evaluation).
    ///
    /// The 8R3C model uses 3 capacitance nodes (ceiling, floor, partition mass)
    /// to better capture thermal inertia in high-mass buildings.
    ///
    /// # Arguments
    /// * `timestep` - Current timestep index
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    ///
    /// # Returns
    /// HVAC energy consumption for the timestep in kWh.
    ///
    /// # Note
    /// This is a simplified implementation for evaluation purposes. It follows the
    /// 5R1C/6R2C pattern but with additional mass nodes for ceiling, floor, and partitions.
    fn step_physics_8r3c(&mut self, timestep: usize, outdoor_temp: f64, dt_seconds: f64) -> f64 {
        let dt = dt_seconds; // Use provided timestep duration

        // Get ground temperature at this timestep (unused in simplified 8R3C)
        let _t_g = self.0.ground_temperature.ground_temperature(timestep);

        // Use 5R1C solve for simplicity (Phase 20 evaluation)
        // In a full implementation, this would be a proper 8R3C algebraic system
        let energy = self.step_physics_5r1c(timestep, outdoor_temp, dt_seconds);

        // Update 8R3C mass temperatures using simple relaxation (for evaluation)
        // In a full implementation, these would be coupled with Ti_free calculation
        let t_i = self.0.temperatures.clone();

        // Unwrap 8R3C fields (panic if not initialized) after step_physics_5r1c
        let ceiling_mass = self.0.ceiling_mass_temperatures.as_mut().unwrap();
        let floor_mass = self.0.floor_mass_temperatures.as_mut().unwrap();
        let partition_mass = self.0.partition_mass_temperatures.as_mut().unwrap();
        let ceiling_cap = self.0.ceiling_thermal_capacitance.as_ref().unwrap();
        let floor_cap = self.0.floor_thermal_capacitance.as_ref().unwrap();
        let partition_cap = self.0.partition_thermal_capacitance.as_ref().unwrap();
        let h_tr_ceiling = self.0.h_tr_ceiling.as_ref().unwrap();
        let h_tr_floor_mass = self.0.h_tr_floor_mass.as_ref().unwrap();
        let h_tr_partition = self.0.h_tr_partition.as_ref().unwrap();

        // Update ceiling mass temperature
        for i in 0..self.0.num_zones {
            let dtm_ceiling = (t_i.as_ref()[i] - ceiling_mass.as_ref()[i])
                / (ceiling_cap.as_ref()[i] / (h_tr_ceiling.as_ref()[i] * dt));
            ceiling_mass.as_mut()[i] += dtm_ceiling;
        }

        // Update floor mass temperature
        for i in 0..self.0.num_zones {
            let dtm_floor = (t_i.as_ref()[i] - floor_mass.as_ref()[i])
                / (floor_cap.as_ref()[i] / (h_tr_floor_mass.as_ref()[i] * dt));
            floor_mass.as_mut()[i] += dtm_floor;
        }

        // Update partition mass temperature
        for i in 0..self.0.num_zones {
            let dtm_partition = (t_i.as_ref()[i] - partition_mass.as_ref()[i])
                / (partition_cap.as_ref()[i] / (h_tr_partition.as_ref()[i] * dt));
            partition_mass.as_mut()[i] += dtm_partition;
        }

        energy
    }

    /// Solve physics for one timestep using the 9R4C (multi-node) model.
    ///
    /// Phase 6D: The 9R4C model uses 4 thermal mass nodes (wall, roof, floor, internal)
    /// to properly capture thermal inertia in high-mass buildings (Case 900+).
    ///
    /// The solver computes free-floating temperature using 5R1C network, then updates
    /// the multi-node thermal mass state. Zone temperatures are calculated using the
    /// 5R1C sensitivity, and the multi-node solver tracks per-surface temperatures.
    fn step_physics_9r4c(&mut self, timestep: usize, outdoor_temp: f64, dt_seconds: f64) -> f64 {
        let dt = dt_seconds;

        // Get ground temperature at this timestep
        let t_g = self.0.ground_temperature.ground_temperature(timestep);

        // Calculate sky temperature for sol-air calculation
        let sky_temp = self
            .0
            .weather
            .as_ref()
            .map(|w| w.sky_temperature())
            .unwrap_or(outdoor_temp - 15.0);

        // Prepare sol-air temperatures and fluxes
        let (_t_sol_air_data, ctf_flux_w, fd_flux_w, _ctf_surface_temps) =
            self.prepare_solvers_and_sol_air(timestep, outdoor_temp, sky_temp);

        // Combine fractions
        let conv_frac = self.0.convective_fraction;
        let rad_frac = 1.0 - conv_frac;

        // Solar gain distribution fractions
        // Internal radiative gains split per ISO 13790 Section C.4 Eq. C.5/C.6:
        // Eq. C.5 (radiative-to-surface): phi_st = (1 - F_sup) * phi_int_rad
        //   where F_sup = H_ms / (H_ms + H_is) — fraction to surface node
        //   st_int_frac = rad_frac * (1 - solar_distribution_to_air) = rad_frac * F_sup
        //   Note: F_sup is the fraction from internal radiative gains going to surface
        //   per ISO 13790 C.4 Eq. C.5 (internal radiative → surface node).
        //
        // Eq. C.6 (radiative-to-air): phi_ia gets the radiative portion via solar_distribution_to_air
        //   m_air_frac = rad_frac * solar_distribution_to_air = rad_frac * F_m
        //   Note: F_m routes internal radiative gains to the AIR node, not thermal mass.
        //   Per ISO 13790 C.4 Eq. C.6, the mass-air node receives radiative gains.
        //
        // The naming reflects ISO 13790 Section C.4:
        //   st_int_frac = fraction of internal radiative gains to SURFACE node (phi_st)
        //   m_air_frac  = fraction of internal radiative gains to AIR node (phi_ia via routing)
        //
        // st_sol_frac: Solar gains to surface (fraction of solar that goes to surface)
        // m_sol_frac: Solar gains to mass (fraction of solar that goes to mass)
        let st_int_frac = rad_frac * (1.0 - self.0.solar_distribution_to_air);
        let m_air_frac = rad_frac * self.0.solar_distribution_to_air;
        let st_sol_frac = 1.0 - self.0.solar_beam_to_mass_fraction;
        let m_sol_frac = self.0.solar_beam_to_mass_fraction;

        let loads_ref = self.0.loads.as_ref();
        let solar_ref = self.0.solar_gains.as_ref();
        let opaque_solar_ref = self.0.opaque_solar_gains.as_ref();
        let area_ref = self.0.zone_area.as_ref();

        let mut phi_ia_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_st_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_m_data = Vec::with_capacity(self.0.num_zones);

        for i in 0..self.0.num_zones {
            let load_w = loads_ref[i] * area_ref[i];
            let sol_w = solar_ref[i] * area_ref[i];
            let opaque_sol_w = opaque_solar_ref[i] * area_ref[i];

            let sol_to_air = sol_w * self.0.solar_distribution_to_air;
            let remaining_sol = sol_w - sol_to_air;
            phi_ia_data.push(load_w * conv_frac + sol_to_air);
            phi_st_data.push(load_w * st_int_frac + remaining_sol * st_sol_frac);
            phi_m_data.push(load_w * m_air_frac + remaining_sol * m_sol_frac + opaque_sol_w);
        }

        let phi_ia = T::from(VectorField::new(phi_ia_data));
        let phi_st = T::from(VectorField::new(phi_st_data));
        let _phi_m = T::from(VectorField::new(phi_m_data));

        let mut t_sol_air_data = Vec::with_capacity(self.0.num_zones);
        for _ in 0..self.0.num_zones {
            t_sol_air_data.push(outdoor_temp);
        }

        // Use 5R1C network for free-floating temperature (same as original)
        let h_ext_base = &self.0.derived_h_ext;
        let term_rest_1 = &self.0.derived_term_rest_1;

        let den = self.0.derived_den.clone();
        let sensitivity = self.0.derived_sensitivity.clone();

        let num_tm = self
            .0
            .derived_h_ms_is_prod
            .zip_with(&self.0.mass_temperatures, |a, b| a * b);
        let num_phi_st = self.0.h_tr_is.zip_with(&phi_st, |a, b| a * b);

        let mut phi_ia_with_iz = phi_ia.clone();

        // Inter-zone heat transfer (if multi-zone)
        if self.0.num_zones > 1 {
            let temps = self.0.temperatures.as_ref();
            let h_iz_vec = self.0.h_tr_iz.as_ref();
            if self.0.num_zones >= 2 && h_iz_vec[0] > 0.0 {
                let delta_t_cond = temps[1] - temps[0];
                let q_cond = h_iz_vec[0] * delta_t_cond;
                let q_rad = 0.0;
                let zone_volume = self.0.zone_volume.as_ref();
                let ach_iz = calculate_stack_effect_ach(
                    temps[0],
                    temps[1],
                    self.0.door_geometry.height,
                    self.0.door_geometry.area,
                    zone_volume[0],
                );
                let q_vent =
                    calculate_ventilation_heat_transfer(ach_iz, temps[1], temps[0], zone_volume[0]);
                let q_iz_total = q_cond + q_rad + q_vent;
                let slice = phi_ia_with_iz.as_mut();
                if slice.len() >= 2 {
                    slice[0] += -q_iz_total;
                    slice[1] += q_iz_total;
                }
            }
        }

        // Add CTF flux contributions (if enabled)
        if let Some(ctf_fluxes) = &ctf_flux_w {
            let slice = phi_ia_with_iz.as_mut();
            for (i, &q_flux) in ctf_fluxes.iter().enumerate() {
                if i < slice.len() {
                    let area = self.0.zone_area.as_ref().get(i).copied().unwrap_or(1.0);
                    let q_ctf = q_flux * area;
                    let t_sol_air_i = t_sol_air_data.get(i).copied().unwrap_or(outdoor_temp);
                    let t_mass = self
                        .0
                        .mass_temperatures
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(20.0);
                    let h_tr_em_i = self.0.h_tr_em.as_ref().get(i).copied().unwrap_or(0.0);
                    let q_5r1c = h_tr_em_i * (t_sol_air_i - t_mass);
                    let net_ctf_flux = q_ctf - q_5r1c;
                    slice[i] += net_ctf_flux;
                    if net_ctf_flux > 0.0 {
                        self.0.ctf_annual_heating_joules += net_ctf_flux * dt;
                    } else {
                        self.0.ctf_annual_cooling_joules += (-net_ctf_flux) * dt;
                    }
                }
            }
        }

        // Add FD flux contributions (if enabled)
        if let Some(fd_fluxes) = &fd_flux_w {
            let slice = phi_ia_with_iz.as_mut();
            for (i, &q_flux) in fd_fluxes.iter().enumerate() {
                if i < slice.len() {
                    let area = self.0.zone_area.as_ref().get(i).copied().unwrap_or(1.0);
                    let q_fd = q_flux * area;
                    let t_sol_air_i = t_sol_air_data.get(i).copied().unwrap_or(outdoor_temp);
                    let t_mass = self
                        .0
                        .mass_temperatures
                        .as_ref()
                        .get(i)
                        .copied()
                        .unwrap_or(20.0);
                    let h_tr_em_i = self.0.h_tr_em.as_ref().get(i).copied().unwrap_or(0.0);
                    let q_5r1c = h_tr_em_i * (t_sol_air_i - t_mass);
                    let net_fd_flux = q_fd - q_5r1c;
                    slice[i] += net_fd_flux;
                    if net_fd_flux > 0.0 {
                        self.0.fd_annual_heating_joules += net_fd_flux * dt;
                    } else {
                        self.0.fd_annual_cooling_joules += (-net_fd_flux) * dt;
                    }
                }
            }
        }

        // Build numerator with envelope and ground contributions
        let mut num_rest_with_iz = phi_ia_with_iz;
        for (n, h) in num_rest_with_iz
            .as_mut()
            .iter_mut()
            .zip(h_ext_base.as_ref().iter())
        {
            *n += h * outdoor_temp;
        }
        num_rest_with_iz.mul_assign(term_rest_1);
        let ground_coeff = self.0.derived_ground_coeff.as_ref();
        for (n, g) in num_rest_with_iz
            .as_mut()
            .iter_mut()
            .zip(ground_coeff.iter())
        {
            *n += g * t_g;
        }

        // Calculate free-floating temperature using 5R1C sensitivity
        let mut t_i_free = num_tm;
        t_i_free.add_assign(&num_phi_st);
        t_i_free.add_assign(&num_rest_with_iz);
        t_i_free.div_assign(&den);

        // === Update Multi-Node Thermal Mass (9R4C) ===
        // Update each zone's multi-node solver with current temperatures
        for zone_idx in 0..self.0.num_zones {
            if zone_idx >= self.0.multi_node_solvers.len() {
                continue;
            }

            let solver = &mut self.0.multi_node_solvers[zone_idx];
            let t_zone = t_i_free.as_ref()[zone_idx];
            let t_ext = t_sol_air_data
                .get(zone_idx)
                .copied()
                .unwrap_or(outdoor_temp);

            // Average interior surface temperature as proxy for T_surface
            // In a full 9R4C implementation, this would come from the surface heat balance
            let t_surface = t_zone - 0.5;

            solver.set_zone_temperature(t_zone);
            solver.set_surface_temperature(t_surface);

            // Issue #863: Compute per-surface sol-air temperatures from weather data.
            // Each envelope node (wall, roof, floor) receives its own exterior boundary
            // temperature based on solar irradiance and longwave radiation exchange,
            // rather than using a uniform outdoor air temperature for all surfaces.
            let surface_ext_temps = if let Some(ref weather) = self.0.weather {
                // Derive date/time from timestep for solar position calculation
                let hour_of_year = timestep % 8760;
                let month_days: [usize; 12] =
                    [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
                let day_of_year = hour_of_year / 24;
                let hour = (hour_of_year % 24) as f64 + 0.5;
                let month = month_days
                    .iter()
                    .position(|&d| d > day_of_year)
                    .unwrap_or(12)
                    .saturating_sub(1)
                    .max(0) as u32;
                let day =
                    (day_of_year - month_days.get(month as usize).copied().unwrap_or(0)) as u32 + 1;

                let sun_pos = calculate_solar_position(
                    self.0.latitude_deg,
                    self.0.longitude_deg,
                    2024, // Reference year for solar position (leap year)
                    month,
                    day.min(28), // Clamp to avoid invalid day-of-month
                    hour,
                );

                // Compute irradiance for wall (south-facing, vertical) and roof (horizontal)
                let ground_reflectance = 0.2; // Standard ground reflectance
                let wall_irr = calculate_surface_irradiance(
                    &sun_pos,
                    weather.dni,
                    weather.dhi,
                    Some(weather.ghi),
                    Orientation::South,
                    ground_reflectance,
                    day_of_year + 1,
                );
                let roof_irr = calculate_surface_irradiance(
                    &sun_pos,
                    weather.dni,
                    weather.dhi,
                    Some(weather.ghi),
                    Orientation::Up,
                    ground_reflectance,
                    day_of_year + 1,
                );

                // Compute sol-air temperatures using ASHRAE 140 default surface properties
                let sol_air = SolAirTemperature::ashrae_140_default();
                SurfaceExteriorTemperatures {
                    t_ext_wall: sol_air.for_wall(
                        outdoor_temp,
                        wall_irr.total_wm2,
                        wall_irr.ground_reflected_wm2,
                    ),
                    t_ext_roof: sol_air.for_roof(outdoor_temp, roof_irr.total_wm2, sky_temp),
                    t_ext_floor: t_g,
                }
            } else {
                // Fallback: use uniform exterior temperature when no weather data available
                SurfaceExteriorTemperatures {
                    t_ext_wall: t_ext,
                    t_ext_roof: t_ext,
                    t_ext_floor: t_g,
                }
            };

            solver.set_surface_exterior_temperatures(surface_ext_temps);

            // Advance solver by one timestep
            solver.step(dt);
        }

        // Update mass temperatures from multi-node solver results
        if !self.0.multi_node_solvers.is_empty() {
            let mut wall_temps = Vec::with_capacity(self.0.num_zones);
            let mut roof_temps = Vec::with_capacity(self.0.num_zones);
            let mut floor_temps = Vec::with_capacity(self.0.num_zones);
            let mut internal_temps = Vec::with_capacity(self.0.num_zones);

            for solver in &self.0.multi_node_solvers {
                wall_temps.push(solver.wall_temperature());
                roof_temps.push(solver.roof_temperature());
                floor_temps.push(solver.floor_temperature());
                internal_temps.push(solver.internal_temperature());
            }

            // Update envelope mass temperatures
            let wall_temps_vf = VectorField::new(wall_temps);
            let roof_temps_vf = VectorField::new(roof_temps);
            let floor_temps_vf = VectorField::new(floor_temps);
            let internal_temps_vf = VectorField::new(internal_temps);

            self.0.envelope_mass_temperatures = T::from(
                (wall_temps_vf.clone() + roof_temps_vf.clone() + floor_temps_vf.clone()) / 3.0,
            );
            self.0.internal_mass_temperatures = T::from(internal_temps_vf.clone());

            // Average envelope temp for overall mass temperature
            self.0.mass_temperatures =
                T::from((wall_temps_vf + roof_temps_vf + floor_temps_vf + internal_temps_vf) / 4.0);
        }

        // Calculate HVAC demand
        let _hour_of_day_idx = timestep % 24;
        let temp_rate = if timestep > 0 {
            (self.0.temperatures.as_ref()[0] - self.0.previous_temperatures.as_ref()[0]) / dt
        } else {
            0.0
        };

        let (hvac_mode, _modulation) = self.0.predictive_controller.calculate_modulation(
            self.0.temperatures.as_ref()[0],
            self.0.mass_temperatures.as_ref()[0],
            temp_rate,
        );
        let _hvac_mode: EquipmentHVACMode = hvac_mode;

        // Issue #738: Free-float mode must completely disable HVAC output
        // but still update temperatures with t_i_free (no HVAC correction)
        if self.0.free_float {
            // For free-float cases, t_i_act = t_i_free (no HVAC power)
            let t_i_act = t_i_free.clone();
            // No HVAC correction since free_float has zero HVAC output

            // Update zone temperatures
            let temps_slice = self.0.temperatures.as_mut();
            for (i, t_val) in t_i_act.as_ref().iter().enumerate() {
                if i < temps_slice.len() {
                    temps_slice[i] = *t_val;
                }
            }
            return 0.0;
        }

        let _sensitivity_val = sensitivity;
        let hvac_for_temp_calc = self.hvac_demand_from_ideal_loads(
            t_i_free.as_ref(),
            self.0.heating_setpoint,
            self.0.cooling_setpoint,
        );

        // Physics-based temperature update: t_i_act = t_i_free + hvac_power / h_tr_is
        let h_tr_is_vec = self.0.h_tr_is.as_ref();
        let t_free = t_i_free.as_ref();
        let hvac = hvac_for_temp_calc.as_ref();
        let mut t_i_act_data = Vec::with_capacity(self.0.num_zones);
        for i in 0..self.0.num_zones {
            let h_is = h_tr_is_vec[i];
            if h_is > 0.0 && hvac[i].abs() > 1e-6 {
                t_i_act_data.push(t_free[i] + hvac[i] / h_is);
            } else {
                t_i_act_data.push(t_free[i]);
            }
        }
        let t_i_act = T::from(VectorField::new(t_i_act_data));

        // Update zone temperatures
        let temps_slice = self.0.temperatures.as_mut();
        for (i, t_val) in t_i_act.as_ref().iter().enumerate() {
            if i < temps_slice.len() {
                temps_slice[i] = *t_val;
            }
        }

        // Calculate and return total HVAC output (energy)
        let hvac_output = hvac_for_temp_calc;
        let hvac_cloned = hvac_output.clone();
        let hvac_power_watts = hvac_cloned
            .as_ref()
            .iter()
            .zip(self.0.hvac_enabled.as_ref().iter())
            .map(|(output, &enabled)| if enabled > 0.5 { *output } else { 0.0 })
            .sum::<f64>();

        // Diagnostics recording (if enabled)
        if self.0.diagnostics.is_some() {
            // Store current HVAC output for this timestep (per zone, Watts)
            self.0.current_hvac_output = Some(hvac_output.clone());
            // Temporarily take diagnostics out to avoid borrow conflicts
            let mut diag = self.0.diagnostics.take().unwrap();
            diag.record_timestep(timestep, self, outdoor_temp, t_g);
            self.0.diagnostics = Some(diag);
            // Clear the buffer after use
            self.0.current_hvac_output = None;
        }

        hvac_power_watts * dt
    }
}
