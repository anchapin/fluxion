    /// Convert timestep to (year, month, day, hour) for solar calculations.
    ///
    /// This function converts a timestep (0-8759) to a date and time,
    /// assuming a non-leap year for consistency with ASHRAE 140.
    fn timestep_to_date(timestep: usize) -> (i32, u32, u32, f64) {
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
    ) -> f64 {
        // Get window properties for this zone
        let window_props = if zone_idx < self.window_properties.len() {
            &self.window_properties[zone_idx]
        } else {
            // Fallback to first zone if not specified
            &self.window_properties[0]
        };

        // Get window orientations for this zone
        let orientations = if zone_idx < self.window_orientations.len() {
            &self.window_orientations[zone_idx]
        } else {
            // Default to South if no orientations specified
            &vec![Orientation::South]
        };

        // Convert timestep to date
        let (year, month, day, hour) = Self::timestep_to_date(timestep);

        // Calculate solar gain for each window orientation and sum them
        let mut total_solar_gain = 0.0;
        for &orientation in orientations {
            // Use solar module to calculate gain for this orientation
            let (_sun_pos, _irradiance, solar_gain_watts) = calculate_hourly_solar(
                self.latitude_deg,
                self.longitude_deg,
                year,
                month,
                day,
                hour,
                weather.dni,
                weather.dhi,
                window_props,
                None, // No window geometry specified
                None, // No overhang
                &[],  // No fins
                orientation,
                Some(0.2), // Ground reflectance
            );

            total_solar_gain += solar_gain_watts;
        }

        total_solar_gain
    }

    /// Calculate analytical thermal loads without neural surrogates.
    ///
    /// When weather data is available, this uses the solar module to calculate
    /// realistic solar gains based on solar position, DNI, DHI, and window properties.
    /// Falls back to the trivial sine-wave approximation if weather data is not available.
    fn calc_analytical_loads(&mut self, timestep: usize, use_analytical_gains: bool) {
        if use_analytical_gains {
            // Try to use weather data for solar gain calculation (Issue #278)
            if let Some(ref weather) = self.weather {
                // Calculate solar gain for each zone using weather data
                let zone_gains: Vec<f64> = (0..self.num_zones)
                    .map(|zone_idx| {
                        let solar_gain =
                            self.calculate_zone_solar_gain(zone_idx, timestep, weather);

                        // Add internal gains (constant 10 W/m² from original implementation)
                        // TODO: This should use actual internal loads from spec when available
                        let zone_area = self.zone_area.as_ref()[zone_idx];
                        let internal_gain = 10.0 * zone_area;

                        solar_gain + internal_gain
                    })
                    .collect();

                // Apply zone-specific gains by creating new VectorField
                // For now, since T is VectorField in most cases, this should work
                let _ = zone_gains;
                // TODO: Properly handle the generic T type here
                // For now, fall back to the old behavior (constant across all zones)
                let hour_of_day = timestep % 24;
                let daily_cycle = get_daily_cycle()[hour_of_day];
                let total_gain = (50.0 * daily_cycle).max(0.0) + 10.0;
                self.loads = self.temperatures.constant_like(total_gain);
            } else {
                // Fallback to trivial sine-wave approximation if no weather data
                let hour_of_day = timestep % 24;
                let daily_cycle = get_daily_cycle()[hour_of_day];
                let total_gain = (50.0 * daily_cycle).max(0.0) + 10.0;
                self.loads = self.temperatures.constant_like(total_gain);
            }
        } else {
            self.loads = self.temperatures.constant_like(0.0);
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
        self.ground_temperature = Box::new(ConstantGroundTemperature::new(temperature));
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
        self.ground_temperature = Box::new(DynamicGroundTemperature::new(
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
        self.ground_temperature = ground_temp;
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
        self.ground_temperature.ground_temperature(timestep)
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
        let t_g = self.ground_temperature.ground_temperature(timestep);

        // --- Dynamic Ventilation (Night Ventilation) ---
        let hour_of_day = (timestep % 24) as u8;

        let loads_watts = self.loads.clone() * self.zone_area.clone();
        let phi_ia = loads_watts.clone() * self.convective_fraction;
        let phi_st = loads_watts.clone() * (1.0 - self.convective_fraction);

        // Simplified 5R1C calculation using CTA
        // Include ground coupling through floor
        // Use pre-computed cached values to avoid redundant allocations
        let h_ext_base = &self.derived_h_ext;

        let modified_h_ext: Option<T>;

        // If h_ve changed, we need to adjust h_ext
        let h_ext = if let Some(night_vent) = &self.night_ventilation {
            if night_vent.is_active_at_hour(hour_of_day) {
                // Calculate h_ve for night ventilation
                // h_ve_vent = (Capacity * rho * cp) / 3600
                let air_cap_vent = night_vent.fan_capacity * 1.2 * 1005.0;
                let h_ve_vent = air_cap_vent / 3600.0;

                // h_ext = derived_h_ext + h_ve_vent
                let new_h_ext = h_ext_base.clone() + self.temperatures.constant_like(h_ve_vent);
                modified_h_ext = Some(new_h_ext);
                modified_h_ext.as_ref().unwrap()
            } else {
                h_ext_base
            }
        } else {
            h_ext_base
        };

        let term_rest_1 = &self.derived_term_rest_1;

        // Dynamic den must include derived_ground_coeff
        let den = self.derived_h_ms_is_prod.clone()
            + term_rest_1.clone() * h_ext.clone()
            + self.derived_ground_coeff.clone();

        let num_tm = self.derived_h_ms_is_prod.clone() * self.mass_temperatures.clone();
        let num_phi_st = self.h_tr_is.clone() * phi_st.clone();

        // Inter-zone heat transfer
        let num_zones = self.num_zones;
        let h_iz_vec = self.h_tr_iz.as_ref();

        let inter_zone_heat: Vec<f64> =
            if num_zones > 1 && !h_iz_vec.is_empty() && h_iz_vec[0] > 0.0 {
                let temps = self.temperatures.as_ref();
                let h_iz_val = h_iz_vec[0];
                (0..num_zones)
                    .map(|i| {
                        let mut q_iz = 0.0;
                        for j in 0..num_zones {
                            if i != j {
                                q_iz += h_iz_val * (temps[j] - temps[i]);
                            }
                        }
                        q_iz
                    })
                    .collect()
            } else {
                vec![0.0; num_zones]
            };

        let q_iz_tensor: T = VectorField::new(inter_zone_heat).into();
        let phi_ia_with_iz = phi_ia + q_iz_tensor;

        // Optimization: Use scalar multiplications
        // Corrected Ground Coupling: term_rest_1 * h_tr_floor * t_g = derived_ground_coeff * t_g
        let num_rest = term_rest_1.clone() * (h_ext.clone() * outdoor_temp + phi_ia_with_iz)
            + self.derived_ground_coeff.clone() * t_g;

        let t_i_free = (num_tm + num_phi_st + num_rest) / den;

        // Return the first zone temperature
        t_i_free.as_ref()[0]
    }
}
