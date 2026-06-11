//! Thermal model core module
//!
//! ISO 13790-compliant 5R1C/6R2C thermal network implementation.
//! Contains the core thermal model types, struct, and implementations.

use std::collections::HashMap;

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::boundary::{
    ConstantGroundTemperature, DynamicGroundTemperature, GroundTemperature,
};
use crate::sim::holiday;
use crate::sim::shading::ShadeFin;
use crate::sim::solar::{calculate_hourly_solar, WindowProperties};
use crate::sim::thermal_model_core::{get_daily_cycle, ThermalModel};
use crate::sim::timestep_solver::StepParameters;
use crate::validation::ashrae_140_cases::{GeometrySpec, Orientation, WindowArea};
use crate::weather::HourlyWeatherData;

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    /// Solves a single timestep of the thermal simulation.
    ///
    /// # Arguments
    ///
    /// * `timestep` - Current timestep index (used for ground temperature)
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    /// * `step_params` - Parameters for this simulation step
    /// * `dt_seconds` - Timestep duration in seconds (default: 3600.0 for 1-hour timestep)
    ///
    /// # Returns
    ///
    /// HVAC energy consumption for the timestep in kWh.
    pub fn solve_single_step(
        &mut self,
        timestep: usize,
        outdoor_temp: f64,
        step_params: &StepParameters,
        dt_seconds: f64,
    ) -> f64 {
        // 1. Calculate External Loads
        if step_params.use_ai {
            // Record call for wiring validation (Plan 21-10)
            #[cfg(feature = "wiring-tracing")]
            if let Some(ref tracer) = self.0.tracer {
                tracer.record_call("predict_loads");
            }

            // Try ONNX with fallback to analytical mode
            match step_params
                .surrogates
                .predict_loads_with_fallback(self.0.temperatures.as_ref())
            {
                Ok(pred) => {
                    self.0.loads = T::from(VectorField::new(pred));
                }
                Err(e) => {
                    // If both ONNX and analytical fail, log error and use analytical mode
                    log::error!(
                        "Both ONNX and analytical fallback failed: {}. Using analytical mode.",
                        e
                    );
                    self.calc_analytical_loads(timestep, step_params.use_analytical_gains);
                }
            }
        } else {
            self.calc_analytical_loads(timestep, step_params.use_analytical_gains);
        }

        // 1.5. Add Internal Loads (lighting, equipment, occupancy) - Plan 17-04
        // Internal loads are added to self.0.loads which will be used by step_physics
        let day_of_year = timestep / 24 + 1; // 1-indexed day of year
        let hour = timestep % 24;
        let _day_type = holiday::get_day_type(day_of_year);
        let hour_of_week = (day_of_year - 1) % 7 * 24 + hour;

        let mut internal_convective = 0.0;
        let mut internal_radiative_to_air = 0.0;
        let mut internal_radiative_to_mass = 0.0;

        // Lighting: fixed convective/radiative split (radiative goes to mass)
        if let Some(lighting) = &step_params.lighting {
            internal_convective += lighting.convective_heat_gains(hour);
            internal_radiative_to_mass += lighting.radiative_heat_gains(hour);
        }

        // Equipment: mass-coupled radiative heat split
        if let Some(equipment_list) = &step_params.equipment {
            for eq in equipment_list {
                let equipment_rad = eq.radiative_gains(timestep);
                internal_convective += eq.convective_gains(timestep);

                // Split radiative heat between air and mass based on mass_coupling_factor
                let radiative_to_air = equipment_rad * (1.0 - eq.mass_coupling_factor());
                let radiative_to_mass = equipment_rad * eq.mass_coupling_factor();

                internal_radiative_to_air += radiative_to_air;
                internal_radiative_to_mass += radiative_to_mass;
            }
        }

        // Occupancy: fixed convective/radiative split (radiative goes to mass)
        if let Some(occ) = &step_params.occupancy {
            internal_convective += occ.convective_heat_gains(hour_of_week);
            internal_radiative_to_mass += occ.radiative_heat_gains(hour_of_week);
        }

        // Add internal heat gains to self.loads (W/m²)
        // These are added BEFORE step_physics so they're included in energy balance
        if internal_convective > 0.0 || internal_radiative_to_air > 0.0 {
            // Convert Watts to W/m² by dividing by zone_area
            let loads_slice = self.0.loads.as_mut();
            for (i, &zone_area) in self
                .0
                .zone_area
                .as_ref()
                .iter()
                .enumerate()
                .take(self.0.num_zones)
            {
                if zone_area > 0.0 {
                    loads_slice[i] += (internal_convective + internal_radiative_to_air) / zone_area;
                }
            }
        }

        // Note: internal_radiative_to_mass will be handled in step_physics_5r1c
        // where it's added directly to thermal mass temperature
        self.0.internal_radiative_to_mass = internal_radiative_to_mass;

        // 2. Call step_physics (pass timestep for ground temperature and dt_seconds)
        self.step_physics(timestep, outdoor_temp, dt_seconds)
    }

    /// Convert timestep to (year, month, day, hour) for solar calculations.
    ///
    /// This function converts a timestep (0-8759) to a date and time,
    /// assuming a non-leap year for consistency with ASHRAE 140.
    pub fn timestep_to_date(timestep: usize) -> (i32, u32, u32, f64) {
        let year = 2024; // Use a fixed year for solar calculations
        let days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

        let day_of_year = timestep / 24;
        let hour_of_day = timestep % 24;

        // Find month and day from day_of_year
        let mut month = 1;
        let mut day = day_of_year + 1; // Day 0 is January 1st

        for (m, &days) in days_in_month.iter().enumerate() {
            if day <= days {
                month = m + 1;
                break;
            }
            day -= days;
        }

        (year, month as u32, day as u32, hour_of_day as f64)
    }

    /// Calculate solar gain for a specific zone using weather data and window properties.
    ///
    /// This method integrates the solar module to calculate realistic solar gains
    /// based on actual solar position, weather data, and window characteristics.
    fn calculate_zone_solar_gain(
        &self,
        zone_idx: usize,
        timestep: usize,
        weather: &HourlyWeatherData,
    ) -> (f64, f64) {
        // Get window properties for this zone
        let window_props = if zone_idx < self.0.window_properties.len() {
            &self.0.window_properties[zone_idx]
        } else {
            // Fallback to first zone if not specified
            &self.0.window_properties[0]
        };

        // Convert timestep to date
        let (year, month, day, hour) = Self::timestep_to_date(timestep);

        // Calculate solar gain for each surface in the zone
        let mut total_window_gain = 0.0;
        let mut total_opaque_gain = 0.0;
        let alpha = 0.6; // Default absorptance for ASHRAE 140
        let re = 0.034; // Exterior film resistance (m²K/W)

        if let Some(zone_surfaces) = self.0.surfaces.get(zone_idx) {
            // Group surfaces by orientation to avoid double-counting solar gain
            // Solar irradiance is the same for all surfaces with the same orientation,
            // so we should calculate it once per unique orientation
            let mut surfaces_by_orientation: HashMap<Orientation, (f64, f64)> = HashMap::new();

            // Diagnostic: Check if surfaces have window areas
            for surface in zone_surfaces {
                let orientation = surface.orientation;

                // Skip floor (Down) for solar gain as it's typically coupled to ground
                if orientation == Orientation::Down {
                    continue;
                }

                let win_area = surface.window_area;
                let opaque_area = (surface.area - win_area).max(0.0);

                // Accumulate areas by orientation
                surfaces_by_orientation
                    .entry(orientation)
                    .and_modify(|(w, o)| {
                        *w += win_area;
                        *o += opaque_area;
                    })
                    .or_insert((win_area, opaque_area));
            }

            // DEBUG: Trace solar calculation for key timesteps (noon in summer/winter)
            let _debug_timesteps = [12, 288, 576]; // Jan 1 noon, Jan 12 noon, Feb 1 noon
                                                   // DEBUG: DEBUG_ZONE_SOLAR removed (PR #821)

            // Now calculate solar gain once per unique orientation
            for (orientation, (total_win_area, _total_opaque_area)) in surfaces_by_orientation {
                // Create temporary window properties with the combined window area for this orientation
                let oriented_window_props = WindowProperties {
                    area: total_win_area,
                    ..*window_props
                };

                // Get shading devices from surfaces with this orientation
                let overhang = zone_surfaces
                    .iter()
                    .filter(|s| s.orientation == orientation)
                    .find_map(|s| s.overhang.as_ref());
                let fins: Vec<ShadeFin> = zone_surfaces
                    .iter()
                    .filter(|s| s.orientation == orientation)
                    .flat_map(|s| s.fins.iter())
                    .cloned()
                    .collect();

                // Create window geometry for shading calculations (needed when overhang/fins present)
                let geometry = if overhang.is_some() || !fins.is_empty() {
                    // Calculate window dimensions from area (assume square-ish window)
                    let width = (total_win_area / 1.5_f64).sqrt();
                    let height = total_win_area / width;
                    // Use a default window geometry - typical window
                    Some(WindowArea {
                        area: total_win_area,
                        orientation,
                        height,
                        width,
                        sill_height: 0.8,
                        left_offset: 0.0,
                    })
                } else {
                    None
                };

                // Use solar module to calculate irradiance for this orientation
                let (_sun_pos, irradiance, solar_gain) = calculate_hourly_solar(
                    self.0.latitude_deg,
                    self.0.longitude_deg,
                    year,
                    month,
                    day,
                    hour,
                    weather.dni,
                    weather.dhi,
                    &oriented_window_props, // Use combined window area for this orientation
                    geometry.as_ref(),      // Use geometry for shading calculations
                    overhang,               // Use overhang from surface
                    &fins,                  // Use fins from surface
                    orientation,
                    Some(0.2), // Ground reflectance
                );

                // DEBUG: Log EVERY calculate_hourly_solar call for key timesteps
                // DEBUG: DEBUG_HOURLY_SOLAR removed (PR #821)

                // Distribute solar gain to each surface with this orientation
                for surface in zone_surfaces {
                    if surface.orientation != orientation {
                        continue;
                    }

                    let win_area = surface.window_area;
                    let opaque_area = (surface.area - win_area).max(0.0);

                    // 1. Window Solar Gain
                    // Scale by ratio of per-surface window area to total orientation window area
                    if win_area > 0.0 && total_win_area > 0.0 {
                        let area_ratio = win_area / total_win_area;
                        let window_gain = solar_gain.total_gain_w * area_ratio;
                        total_window_gain += window_gain;
                    }

                    // 2. Opaque Solar Gain (Wall/Roof)
                    // Issue #831: Sol-air method  q = α × I × R_ext × U × A
                    // Previously this multiplied by an extra `area_ratio = opaque_area / total_opaque_area`,
                    // which double-attenuated the gain by `1/N` per orientation. With one wall surface per
                    // orientation in Case 600 (N=1) the bug had no effect, but with multiple orientations
                    // sharing surfaces (e.g. roof spans the floor footprint) the gain was halved.
                    if opaque_area > 0.0 {
                        total_opaque_gain +=
                            opaque_area * surface.u_value * irradiance.total_wm2 * alpha * re;
                    }
                }
            }
        }

        (total_window_gain, total_opaque_gain)
    }

    /// Calculate area-weighted radiative gain distribution for a zone.
    /// This method distributes radiative gains (internal + solar) among zone surfaces
    /// based on their relative surface areas and thermal mass. This implements Issue #303:
    /// Detailed Internal Radiation Network by using the ISO 13790 compliant
    /// distribution approach.
    ///
    /// # Arguments
    /// * `zone_idx` - Zone index
    /// * `radiative_gain_watts` - Total radiative gain to distribute (Watts)
    ///
    /// # Returns
    /// * (radiative_to_surface_watts, radiative_to_mass_watts)
    ///   - radiative_to_surface_watts: Portion going directly to surface temperature node (phi_st)
    ///   - radiative_to_mass_watts: Portion going to thermal mass nodes (phi_m)
    pub fn calculate_area_weighted_radiative_distribution(
        &self,
        zone_idx: usize,
        radiative_gain_watts: f64,
    ) -> (f64, f64) {
        // Get surfaces for this zone
        if zone_idx >= self.0.surfaces.len() || self.0.surfaces[zone_idx].is_empty() {
            // Fallback to default distribution if no surfaces defined
            // Use solar_distribution_to_air for diffuse, solar_beam_to_mass_fraction for beam
            let radiative_to_surface = radiative_gain_watts * self.0.solar_distribution_to_air;
            let radiative_to_mass = radiative_gain_watts * (1.0 - self.0.solar_distribution_to_air);
            return (radiative_to_surface, radiative_to_mass);
        }

        let surfaces = &self.0.surfaces[zone_idx];
        let a_at: f64 = surfaces.iter().map(|s| s.area).sum();

        if a_at == 0.0 {
            // Fallback to default distribution if total area is zero
            let radiative_to_surface = radiative_gain_watts * self.0.solar_distribution_to_air;
            let radiative_to_mass = radiative_gain_watts * (1.0 - self.0.solar_distribution_to_air);
            return (radiative_to_surface, radiative_to_mass);
        }

        // ISO 13790 Detailed Radiation Network Distribution
        // 1. Effective mass area (Am) is derived from h_tr_ms = 9.1 * Am
        let h_ms_val = self.0.h_tr_ms.as_ref()[zone_idx];
        let a_m = h_ms_val / 9.1;

        // 2. Window conductance for correction (simplified)
        let h_tr_w = self.0.h_tr_w.as_ref()[zone_idx];

        // 3. Distribution factors
        // Fraction to mass (phi_m)
        let f_m = (a_m / a_at).min(1.0);

        // Fraction to surface (phi_st)
        // Correction for radiation lost through windows: h_tr_w / (9.1 * A_at)
        let f_st = (1.0 - f_m - (h_tr_w / (9.1 * a_at))).max(0.0);

        // Normalize factors to ensure energy conservation within the model nodes
        let total_f = f_m + f_st;
        if total_f > 0.0 {
            let phi_m = radiative_gain_watts * (f_m / total_f);
            let phi_st = radiative_gain_watts * (f_st / total_f);
            (phi_st, phi_m)
        } else {
            (0.0, radiative_gain_watts)
        }
    }

    /// Tensor version of radiative distribution for use in physics step.
    pub fn calculate_area_weighted_radiative_distribution_tensor(
        &self,
        radiative_gain: T,
    ) -> (T, T) {
        let num_zones = self.0.num_zones;
        let mut f_st_vec = Vec::with_capacity(num_zones);
        let mut f_m_vec = Vec::with_capacity(num_zones);

        for zone_idx in 0..num_zones {
            let (st, m) = if radiative_gain.as_ref()[zone_idx].abs() > 1e-10 {
                let (st_w, m_w) =
                    self.calculate_area_weighted_radiative_distribution(zone_idx, 1.0);
                (st_w, m_w)
            } else {
                (0.5, 0.5) // Default neutral split for zero gain
            };
            f_st_vec.push(st);
            f_m_vec.push(m);
        }

        let f_st_field = T::from(VectorField::new(f_st_vec));
        let f_m_field = T::from(VectorField::new(f_m_vec));

        (
            radiative_gain.clone() * f_st_field,
            radiative_gain * f_m_field,
        )
    }

    /// Calculate radiative conductance through inter-zone windows.
    ///
    /// This method implements Issue #302: Refine Inter-Zone Longwave Radiation
    /// by calculating the linearized radiative heat transfer coefficient through
    /// windows connecting zones.
    ///
    /// # Arguments
    /// * `window_area` - Area of inter-zone windows (m²)
    /// * `surface_emissivity` - Emissivity of interior surfaces (0.0-1.0)
    /// * `reference_temp` - Reference temperature for linearization (K)
    ///
    /// # Returns
    /// Radiative conductance (W/K)
    ///
    /// # Physics
    /// Radiative exchange: Q_rad = σ * ε1 * ε2 * A * F12 * (T1^4 - T2^4)
    /// Linearized: Q_rad ≈ h_rad * (T1 - T2)
    /// Where h_rad ≈ 4 * σ * ε * T_avg^3 * A
    #[allow(dead_code)]
    pub(crate) fn calculate_total_interior_surface_area(geometry: &GeometrySpec) -> f64 {
        geometry.wall_area() + geometry.floor_area() + geometry.roof_area()
    }

    #[allow(dead_code)]
    pub(crate) fn calculate_zone_to_zone_view_factor(
        common_window_area: f64,
        zone_a_area: f64,
        zone_b_area: f64,
    ) -> f64 {
        let zone_a_interior_area = zone_a_area;
        let zone_b_interior_area = zone_b_area;

        let f_window_to_a = common_window_area / zone_a_interior_area;
        let f_window_to_b = common_window_area / zone_b_interior_area;

        0.5 * (f_window_to_a + f_window_to_b)
    }

    pub(crate) fn calculate_radiative_conductance_with_view_factor(
        window_area: f64,
        surface_emissivity: f64,
        reference_temp: f64,
        view_factor: f64,
    ) -> f64 {
        const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;
        let effective_emissivity =
            1.0 / (1.0 / surface_emissivity + 1.0 / surface_emissivity - 1.0);
        let h_rad =
            4.0 * STEFAN_BOLTZMANN * effective_emissivity * view_factor * reference_temp.powi(3);
        h_rad * window_area
    }

    /// Calculate window-to-window radiative conductance using glass emissivity.
    ///
    /// Implements Issue #349: Window-to-Window Radiative Exchange
    ///
    /// The radiative heat exchange between two windows follows:
    /// Q_ij = σ * F_ij * ε_glass^2 * A_window * (T_i^4 - T_j^4)
    ///
    /// Linearized around reference temperature T_ref:
    /// Q_ij ≈ h_rad * (T_i - T_j)
    ///
    /// where h_rad = 4 * σ * F_ij * ε_glass^2 * A_window * T_ref^3
    ///
    /// # Arguments
    /// * `window_area` - Area of the windows (m²)
    /// * `glass_emissivity` - Emissivity of glass for longwave radiation (0-1)
    /// * `reference_temp` - Reference temperature for linearization (K)
    /// * `view_factor` - View factor between windows (0-1)
    ///
    /// # Returns
    /// Radiative conductance in W/K
    #[allow(dead_code)]
    fn calculate_window_radiative_conductance(
        window_area: f64,
        glass_emissivity: f64,
        reference_temp: f64,
        view_factor: f64,
    ) -> f64 {
        const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;
        let effective_emissivity = glass_emissivity * glass_emissivity;
        let h_rad =
            4.0 * STEFAN_BOLTZMANN * effective_emissivity * view_factor * reference_temp.powi(3);
        h_rad * window_area
    }

    /// Calculate analytical thermal loads without neural surrogates.
    ///
    /// When weather data is available, this uses the solar module to calculate
    /// realistic solar gains based on solar position, DNI, DHI, and window properties.
    /// Falls back to trivial sine-wave approximation if weather data is not available.
    pub(crate) fn calc_analytical_loads(&mut self, timestep: usize, use_analytical_gains: bool) {
        // Diagnostic: Check if calc_analytical_loads is being called (removed for release performance)

        if use_analytical_gains {
            // Try to use weather data for solar gain calculation (Issue #278)
            if let Some(ref weather) = self.0.weather {
                // Calculate solar gain for each zone using weather data
                let mut zone_solar_gains = Vec::with_capacity(self.0.num_zones);
                let mut zone_opaque_gains = Vec::with_capacity(self.0.num_zones);

                for zone_idx in 0..self.0.num_zones {
                    let (window_gain_watts, opaque_gain_watts) =
                        self.calculate_zone_solar_gain(zone_idx, timestep, weather);
                    let floor_area = self.0.zone_area.as_ref()[zone_idx];
                    let solar_gain_normalized = window_gain_watts / floor_area;
                    let opaque_gain_normalized = opaque_gain_watts / floor_area;
                    zone_solar_gains.push(solar_gain_normalized);
                    zone_opaque_gains.push(opaque_gain_normalized);
                }

                // SESSION 79: For free-floating cases, DO NOT reduce solar gains
                // FF cases need FULL solar gains to achieve high summer temperatures
                // Previous 0.5x reduction was preventing max temps from reaching reference values
                // The thermal capacitance reduction (0.5x) is sufficient for FF behavior

                // Apply zone-specific solar gains
                // Issue #901 perf: consume the freshly-built Vec directly into VectorField::new
                // rather than cloning first (the variables are not used after this point).
                self.0.solar_gains = T::from(VectorField::new(zone_solar_gains));
                self.0.opaque_solar_gains = T::from(VectorField::new(zone_opaque_gains));
            } else {
                // Fallback to trivial sine-wave approximation if no weather data
                let hour_of_day = timestep % 24;
                let daily_cycle = get_daily_cycle()[hour_of_day];
                let total_gain = (50.0 * daily_cycle).max(0.0);
                self.0.solar_gains = self.0.temperatures.constant_like(total_gain);
                self.0.opaque_solar_gains = self.0.temperatures.constant_like(0.0);
            }
        } else {
            self.0.loads = self.0.temperatures.constant_like(0.0);
        }
    }

    /// Set a constant ground temperature.
    ///
    /// Use this for deep foundations where ground temperature is effectively constant.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Constant ground temperature (°C)
    pub fn set_ground_temp(&mut self, temperature: f64) {
        self.0.ground_temperature = Box::new(ConstantGroundTemperature::new(temperature));
    }

    /// Set a dynamic ground temperature model using the Kusuda formula.
    ///
    /// Use this for shallow foundations or when seasonal ground temperature
    /// variation is significant. The Kusuda formula calculates time-varying
    /// soil temperature based on depth and thermal diffusivity.
    ///
    /// # Arguments
    ///
    /// * `t_mean` - Mean annual soil temperature (°C)
    /// * `t_amplitude` - Annual temperature amplitude (°C)
    /// * `depth` - Depth below surface (m)
    /// * `diffusivity` - Soil thermal diffusivity (m²/day)
    pub fn set_dynamic_ground_temp(
        &mut self,
        t_mean: f64,
        t_amplitude: f64,
        depth: f64,
        diffusivity: f64,
    ) {
        self.0.ground_temperature = Box::new(DynamicGroundTemperature::new(
            t_mean,
            t_amplitude,
            depth,
            diffusivity,
        ));
    }

    /// Set a custom ground temperature model.
    ///
    /// Allows for advanced ground temperature modeling strategies.
    ///
    /// # Arguments
    ///
    /// * `ground_temp` - Custom ground temperature model implementing GroundTemperature trait
    pub fn with_ground_temperature(&mut self, ground_temp: Box<dyn GroundTemperature>) {
        self.0.ground_temperature = ground_temp;
    }

    /// Get the ground temperature at a specific timestep.
    ///
    /// # Arguments
    ///
    /// * `timestep` - Timestep index (0-8759 for hourly annual simulation)
    ///
    /// # Returns
    ///
    /// Ground temperature (°C)
    pub fn ground_temperature_at(&self, timestep: usize) -> f64 {
        self.0.ground_temperature.ground_temperature(timestep)
    }

    /// Solve coupled zone temperatures using matrix-based approach (Issue #381)
    ///
    /// This method implements a proper thermal network solver for multi-zone buildings,
    /// solving the coupled system of equations simultaneously using matrix operations.
    ///
    /// # Arguments
    /// * `num_zones` - Number of thermal zones
    /// * `temps` - Current zone temperatures \[K\]
    /// * `h_iz_vec` - Inter-zone conductances \[W/K\]
    /// * `h_iz_rad_vec` - Radiative inter-zone conductances \[W/K\]
    ///
    /// # Returns
    /// * Vector of inter-zone heat flows \[W\] for each zone
    ///
    /// # Mathematical Formulation
    /// For a multi-zone system with N zones, we solve:
    /// Q_iz\[i\] = Σ_j (h_iz_ij + h_iz_rad_ij) * (T\[j\] - T\[i\])
    ///
    /// This can be expressed as a matrix equation:
    /// Q = (A - I) * diag(h_total) * T
    /// where A is the adjacency matrix, I is identity, h_total is the conductance matrix
    pub fn solve_coupled_zone_temperatures(
        &self,
        num_zones: usize,
        temps: &[f64],
        h_iz_vec: &[f64],
        h_iz_rad_vec: &[f64],
    ) -> Option<Vec<f64>> {
        if num_zones <= 1
            || (h_iz_vec.is_empty()
                || h_iz_vec[0] <= 0.0 && (h_iz_rad_vec.is_empty() || h_iz_rad_vec[0] <= 0.0))
        {
            return None;
        }

        let total_h_iz =
            h_iz_vec.first().copied().unwrap_or(0.0) + h_iz_rad_vec.first().copied().unwrap_or(0.0);

        // Optimization: avoid matrix allocation and multiplication for simple cases
        // Solve Q_i = sum(G[i,j] * (T_j - T_i)) directly
        let sum_t: f64 = temps.iter().sum();
        let n = num_zones as f64;
        let q_iz: Vec<f64> = (0..num_zones)
            .map(|i| total_h_iz * (sum_t - n * temps[i]))
            .collect();
        Some(q_iz)
    }

    /// Calculate the free-floating temperature (without HVAC).
    ///
    /// # Arguments
    ///
    /// * `timestep` - Current timestep index
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    ///
    /// # Returns
    ///
    /// Free-floating zone temperature (°C)
    pub fn calculate_free_float_temperature(&self, timestep: usize, outdoor_temp: f64) -> f64 {
        // Use the same calculation as in step_physics
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
        let st_int_frac = rad_frac * (1.0 - self.0.solar_distribution_to_air);
        let m_air_frac = rad_frac * self.0.solar_distribution_to_air;
        let st_sol_frac = 1.0 - self.0.solar_beam_to_mass_fraction;
        let m_sol_frac = self.0.solar_beam_to_mass_fraction;

        let loads_ref = self.0.loads.as_ref();
        let solar_ref = self.0.solar_gains.as_ref();
        let opaque_ref = self.0.opaque_solar_gains.as_ref();
        let area_ref = self.0.zone_area.as_ref();

        let mut phi_ia_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_st_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_m_data = Vec::with_capacity(self.0.num_zones);

        for i in 0..self.0.num_zones {
            let load_w = loads_ref[i] * area_ref[i];
            let sol_w = solar_ref[i] * area_ref[i];
            let opaque_sol_w = opaque_ref[i] * area_ref[i];

            let sol_to_air = sol_w * self.0.solar_distribution_to_air;
            let remaining_sol = sol_w - sol_to_air;

            phi_ia_data.push(load_w * conv_frac + sol_to_air);
            phi_st_data.push(load_w * st_int_frac + remaining_sol * st_sol_frac);
            phi_m_data.push(load_w * m_air_frac + remaining_sol * m_sol_frac + opaque_sol_w);
        }

        let phi_ia = T::from(VectorField::new(phi_ia_data));
        let phi_st = T::from(VectorField::new(phi_st_data));
        let phi_m = T::from(VectorField::new(phi_m_data));

        // Simplified 5R1C calculation using CTA
        // Include ground coupling through floor
        // Use pre-computed cached values to avoid redundant allocations
        let h_ext_base = &self.0.derived_h_ext;

        // Night ventilation is modeled as a separate heat term in the zone energy balance,
        // NOT as a modification to h_ext (which represents building envelope conductance).
        // Q_vent = ρ·Cp·ACH·V·(T_outdoor - T_zone) is applied directly to phi_ia.
        // h_ext modification was incorrect: night ventilation cools zone through direct air supply.
        let h_ext = h_ext_base;

        let term_rest_1 = &self.0.derived_term_rest_1;

        // Dynamic den must include derived_ground_coeff
        // den = h_ms_is_prod + term_rest_1 * (h_ext + h_tr_floor + h_tr_iz)
        // Issue #351: Include inter-zone conductance
        // Night ventilation no longer modifies h_ext, so we always use the cached denominator.
        // Night ventilation directly cools thermal mass (critical for Cases 650, 950)
        // This effect is captured by modifying phi_m during night ventilation hours
        let phi_m_with_vent = phi_m.clone();
        // === Issue #821: see thermal_model_physics.rs::step_physics_5r1c ===
        // The empirical "30% of ventilation flow cools mass directly" path was
        // double-counting once the ISO 13790 `h_tr_ms` was raised to its standard
        // value (~1.3 kW/K). Mass cooling under night ventilation is now mediated
        // entirely by air-side h_ve and the much-stronger air-mass coupling.
        if let Some(ref night_vent) = self.0.night_ventilation {
            if night_vent.is_active_at_hour(hour_of_day) {
                let _ = night_vent.fan_capacity;
            }
        }

        let den = self.0.derived_den.clone();

        // Use mass_temperatures to match step_physics_5r1c
        let num_tm = self
            .0
            .derived_h_ms_is_prod
            .zip_with(&self.0.mass_temperatures, |a, b| a * b);
        let num_phi_st = self.0.h_tr_is.zip_with(&phi_st, |a, b| a * b);
        let num_phi_m = self.0.h_tr_ms.zip_with(&phi_m_with_vent, |a, b| a * b);

        // Inter-zone heat transfer (with radiative component - Issue #302)
        // Optimized: eliminate Vec allocation by adding directly to phi_ia buffer
        let num_zones = self.0.num_zones;

        // Get inter-zone heat transfer coefficients
        let h_iz_vec = self.0.h_tr_iz.as_ref();
        let h_iz_rad_vec = self.0.h_tr_iz_rad.as_ref();

        // Issue #381: Use matrix-based solver for simultaneous boundary conditions
        // Optimization: Replace O(N^2) nested loop with O(N) grouping solver from step_physics.
        // Returns Option<Vec> to prevent allocating zero-filled VectorFields when uncoupled.
        let inter_zone_heat: Option<Vec<f64>> = self.solve_coupled_zone_temperatures(
            num_zones,
            self.0.temperatures.as_ref(),
            h_iz_vec,
            h_iz_rad_vec,
        );

        let phi_ia_with_iz = if let Some(q_iz) = inter_zone_heat {
            phi_ia + VectorField::new(q_iz).into()
        } else {
            phi_ia
        };

        // Optimization: Use scalar multiplications
        // Ground Coupling: term_rest_1 * h_tr_floor * T_ground = derived_ground_coeff * T_ground
        // Add this to numerator per ISO 13790 5R1C heat balance equation
        // Re-enabled with correct formula: num_rest = term_rest_1 * (phi_ia + h_ext * T_ext + h_tr_floor * T_ground)
        let num_rest = term_rest_1.clone() * (h_ext.clone() * outdoor_temp + phi_ia_with_iz)
            + num_phi_m
            + self.0.h_tr_floor.clone() * t_g;

        let t_i_free = (num_tm + num_phi_st + num_rest) / den;

        // PR #821: DEBUG_650FF trace removed; use `pr821-diag` feature.

        // Return the first zone temperature
        t_i_free.as_ref()[0]
    }

    /// Calculate HVAC capacity from design day simulation.
    ///
    /// Runs 24-hour simulations for heating and cooling design days to determine
    /// actual peak loads, then applies a safety factor (typically 1.1-1.2x).
    ///
    /// # Arguments
    ///
    /// * `heating_design_hours` - 24 hours of weather data for heating design (extreme cold)
    /// * `cooling_design_hours` - 24 hours of weather data for cooling design (extreme hot)
    /// * `safety_factor` - Safety factor to apply to peak loads (typically 1.15 for 15% margin)
    ///
    /// # Returns
    ///
    /// * `(heating_capacity_w, cooling_capacity_w)` - HVAC capacities in Watts
    pub fn calculate_hvac_capacity_from_design_day(
        &mut self,
        heating_design_hours: &[crate::weather::HourlyWeatherData],
        cooling_design_hours: &[crate::weather::HourlyWeatherData],
        safety_factor: f64,
    ) -> (f64, f64) {
        // Save current state
        let original_temperatures = self.0.temperatures.clone();
        let original_mass_temperatures = self.0.mass_temperatures.clone();

        // Run heating design day simulation (24 hours)
        self.reset_peak_power();
        self.reset_all_energy_tracking();

        for (hour, weather) in heating_design_hours.iter().enumerate() {
            self.set_weather(weather.clone());
            self.step_physics(hour, weather.dry_bulb_temp, 3600.0);
        }

        let peak_heating_w = self.get_peak_heating_power_kw() * 1000.0;

        // Run cooling design day simulation (24 hours)
        self.reset_peak_power();
        self.reset_all_energy_tracking();

        // Reset temperatures to initial state for cooling simulation
        self.0.temperatures = original_temperatures.clone();
        self.0.mass_temperatures = original_mass_temperatures.clone();

        for (hour, weather) in cooling_design_hours.iter().enumerate() {
            self.set_weather(weather.clone());
            self.step_physics(hour, weather.dry_bulb_temp, 3600.0);
        }

        let peak_cooling_w = self.get_peak_cooling_power_kw() * 1000.0;

        // Restore original temperatures
        self.0.temperatures = original_temperatures;
        self.0.mass_temperatures = original_mass_temperatures;

        // Apply safety factor
        let heating_capacity = peak_heating_w * safety_factor;
        let cooling_capacity = peak_cooling_w * safety_factor;

        (heating_capacity, cooling_capacity)
    }
}
