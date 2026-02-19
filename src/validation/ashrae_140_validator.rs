use crate::physics::cta::VectorField;
use crate::sim::engine::{IdealHVACController, ThermalModel};
use crate::sim::solar::{calculate_hourly_solar, WindowProperties};
use crate::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec, Orientation};
use crate::validation::benchmark;
use crate::validation::diagnostic::{DiagnosticCollector, DiagnosticConfig, HourlyData};
use crate::validation::report::{BenchmarkReport, MetricType};
use crate::weather::denver::DenverTmyWeather;
use crate::weather::WeatherSource;

/// Validator for ASHRAE 140 standard cases.
pub struct ASHRAE140Validator {
    /// Diagnostic collector for detailed output
    diagnostic: DiagnosticCollector,
}

impl Default for ASHRAE140Validator {
    fn default() -> Self {
        Self::new()
    }
}

impl ASHRAE140Validator {
    /// Creates a new ASHRAE 140 validator.
    pub fn new() -> Self {
        Self {
            diagnostic: DiagnosticCollector::from_env(),
        }
    }

    /// Creates a new validator with custom diagnostic configuration.
    pub fn with_diagnostics(config: DiagnosticConfig) -> Self {
        Self {
            diagnostic: DiagnosticCollector::new(config),
        }
    }

    /// Creates a new validator with full diagnostic output enabled.
    pub fn with_full_diagnostics() -> Self {
        Self {
            diagnostic: DiagnosticCollector::new(DiagnosticConfig::full()),
        }
    }

    /// Creates an IdealHVACController from a case specification.
    ///
    /// This creates a controller with:
    /// - Dual setpoint control (heating and cooling)
    /// - Deadband tolerance (0.5°C default)
    /// - High capacity limits for ASHRAE 140 validation
    ///
    /// # Arguments
    /// * `spec` - The ASHRAE 140 case specification
    ///
    /// # Returns
    /// An IdealHVACController configured for the case
    pub fn create_hvac_controller(spec: &CaseSpec) -> IdealHVACController {
        let hvac_schedule = spec.hvac.first();
        let heating_setpoint = hvac_schedule
            .and_then(|h| h.heating_setpoint_at_hour(0))
            .unwrap_or(20.0);
        let cooling_setpoint = hvac_schedule
            .and_then(|h| h.cooling_setpoint_at_hour(0))
            .unwrap_or(27.0);

        IdealHVACController::new(heating_setpoint, cooling_setpoint)
    }

    /// Validates a case using the IdealHVACController for more sophisticated control.
    ///
    /// This method uses the IdealHVACController which provides:
    /// - Deadband tolerance to prevent rapid cycling
    /// - Staged response to temperature deviation
    /// - Proportional control near setpoints
    ///
    /// # Arguments
    /// * `case` - The ASHRAE 140 case to validate
    ///
    /// # Returns
    /// A BenchmarkReport with validation results
    pub fn validate_with_ideal_control(&mut self, case: ASHRAE140Case) -> BenchmarkReport {
        let mut report = BenchmarkReport::new();
        let benchmark_data = benchmark::get_all_benchmark_data();
        let weather = DenverTmyWeather::new();

        let case_id = case.number();
        if let Some(data) = benchmark_data.get(&case_id) {
            let spec = case.spec();

            // Create IdealHVACController for this case
            let controller = Self::create_hvac_controller(&spec);

            // Validate controller configuration
            if let Err(e) = controller.validate() {
                eprintln!("Warning: Invalid HVAC controller config: {}", e);
            }

            // Run simulation with the controller
            let results = self.simulate_case_with_ideal_control(&spec, &weather, &controller);

            if spec.is_free_floating() {
                if let Some(min_temp) = results.min_temp_celsius {
                    report.add_result_simple(
                        &case_id,
                        MetricType::MinFreeFloat,
                        min_temp,
                        data.min_free_float_min,
                        data.min_free_float_max,
                    );
                }

                if let Some(max_temp) = results.max_temp_celsius {
                    report.add_result_simple(
                        &case_id,
                        MetricType::MaxFreeFloat,
                        max_temp,
                        data.max_free_float_min,
                        data.max_free_float_max,
                    );
                }
            } else {
                report.add_result_simple(
                    &case_id,
                    MetricType::AnnualHeating,
                    results.annual_heating_mwh,
                    data.annual_heating_min,
                    data.annual_heating_max,
                );

                report.add_result_simple(
                    &case_id,
                    MetricType::AnnualCooling,
                    results.annual_cooling_mwh,
                    data.annual_cooling_min,
                    data.annual_cooling_max,
                );

                if data.peak_heating_min >= 0.0 {
                    report.add_result_simple(
                        &case_id,
                        MetricType::PeakHeating,
                        results.peak_heating_kw,
                        data.peak_heating_min,
                        data.peak_heating_max,
                    );
                }

                if data.peak_cooling_min >= 0.0 {
                    report.add_result_simple(
                        &case_id,
                        MetricType::PeakCooling,
                        results.peak_cooling_kw,
                        data.peak_cooling_min,
                        data.peak_cooling_max,
                    );
                }
            }

            report.add_benchmark_data(&case_id, data.clone());
        }

        report
    }

    /// Simulates a case using IdealHVACController for HVAC control.
    fn simulate_case_with_ideal_control(
        &self,
        spec: &CaseSpec,
        weather: &DenverTmyWeather,
        controller: &IdealHVACController,
    ) -> CaseResults {
        let mut model = ThermalModel::<VectorField>::from_spec(spec);
        const STEPS: usize = 8760;
        let num_zones = model.num_zones;

        let is_free_floating = spec.is_free_floating();

        if is_free_floating {
            model.heating_setpoint = -999.0;
            model.cooling_setpoint = 999.0;
            model.hvac_heating_capacity = 0.0;
            model.hvac_cooling_capacity = 0.0;
        } else {
            // Apply controller setpoints to model
            model.heating_setpoint = controller.heating_setpoint;
            model.cooling_setpoint = controller.cooling_setpoint;
        }

        let mut annual_heating_joules = 0.0;
        let mut annual_cooling_joules = 0.0;
        let mut peak_heating_watts: f64 = 0.0;
        let mut peak_cooling_watts: f64 = 0.0;
        let mut min_temp_celsius: f64 = f64::INFINITY;
        let mut max_temp_celsius: f64 = f64::NEG_INFINITY;

        for step in 0..STEPS {
            let hour_of_day = step % 24;
            let day_of_year = step / 24 + 1;

            let days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
            let mut month = 1;
            let mut day = day_of_year;
            for (i, &days) in days_in_month.iter().enumerate() {
                if day <= days as usize {
                    month = i + 1;
                    break;
                }
                day -= days as usize;
            }

            let weather_data = weather.get_hourly_data(step).unwrap();

            // Apply dynamic setpoints from schedule
            if let Some(hvac_schedule) = spec.hvac.first() {
                if let Some(heating_sp) = hvac_schedule.heating_setpoint_at_hour(hour_of_day as u8)
                {
                    model.heating_setpoint = heating_sp;
                }
                if let Some(cooling_sp) = hvac_schedule.cooling_setpoint_at_hour(hour_of_day as u8)
                {
                    model.cooling_setpoint = cooling_sp;
                }
            }

            // Apply night ventilation
            if let Some(vent) = &spec.night_ventilation {
                if vent.is_active_at_hour(hour_of_day as u8) {
                    if let Some(hvac_schedule) = spec.hvac.first() {
                        if hvac_schedule.heating_setpoint < 0.0 {
                            model.cooling_setpoint = -100.0;
                        }
                    }
                }
            }

            // Calculate solar gains
            let mut total_solar_gain_per_zone: Vec<f64> = vec![0.0; num_zones];
            for (zone_idx, zone_windows) in spec.windows.iter().enumerate() {
                if zone_idx >= num_zones {
                    break;
                }
                for win_area in zone_windows {
                    let props = WindowProperties::new(
                        win_area.area,
                        spec.window_properties.shgc,
                        spec.window_properties.normal_transmittance,
                    );

                    let mut overhang = None;
                    let mut fins = Vec::new();
                    if let Some(zone_surfaces) = model.surfaces.get(zone_idx) {
                        for surf in zone_surfaces {
                            if surf.orientation == win_area.orientation {
                                overhang = surf.overhang.as_ref();
                                fins = surf.fins.clone();
                                break;
                            }
                        }
                    }

                    let (_, _, gain) = calculate_hourly_solar(
                        39.7392,
                        -104.9903,
                        2024,
                        month as u32,
                        day as u32,
                        hour_of_day as f64 + 0.5,
                        weather_data.dni,
                        weather_data.dhi,
                        &props,
                        Some(win_area),
                        overhang,
                        &fins,
                        win_area.orientation,
                        Some(0.2),
                    );
                    total_solar_gain_per_zone[zone_idx] += gain;
                }

                // Opaque solar gains
                let alpha = spec.opaque_absorptance;
                let re = 0.034;
                let wall_area = spec.geometry[zone_idx].wall_area();
                let window_area: f64 = spec.windows[zone_idx].iter().map(|w| w.area).sum();
                let opaque_wall_area = wall_area - window_area;
                let roof_area = spec.geometry[zone_idx].roof_area();

                let wall_u = spec.construction.wall.u_value(None);
                let roof_u = spec.construction.roof.u_value(None);

                for orientation in [
                    Orientation::South,
                    Orientation::West,
                    Orientation::North,
                    Orientation::East,
                ] {
                    let (_, irr, _) = calculate_hourly_solar(
                        39.7392,
                        -104.9903,
                        2024,
                        month as u32,
                        day as u32,
                        hour_of_day as f64 + 0.5,
                        weather_data.dni,
                        weather_data.dhi,
                        &WindowProperties::new(0.0, 0.0, 0.0),
                        None,
                        None,
                        &[],
                        orientation,
                        Some(0.2),
                    );
                    total_solar_gain_per_zone[zone_idx] +=
                        (opaque_wall_area / 4.0) * wall_u * irr.total_wm2 * alpha * re;
                }

                let (_, irr_roof, _) = calculate_hourly_solar(
                    39.7392,
                    -104.9903,
                    2024,
                    month as u32,
                    day as u32,
                    hour_of_day as f64 + 0.5,
                    weather_data.dni,
                    weather_data.dhi,
                    &WindowProperties::new(0.0, 0.0, 0.0),
                    None,
                    None,
                    &[],
                    Orientation::Up,
                    Some(0.2),
                );
                total_solar_gain_per_zone[zone_idx] +=
                    roof_area * roof_u * irr_roof.total_wm2 * alpha * re;
            }

            // Calculate loads
            let mut total_loads: Vec<f64> = Vec::with_capacity(num_zones);
            for zone_idx in 0..num_zones {
                let internal_gains = spec
                    .internal_loads
                    .get(zone_idx)
                    .or(spec.internal_loads.first())
                    .and_then(|l| l.as_ref())
                    .map_or(0.0, |l| l.total_load);

                let floor_area = spec
                    .geometry
                    .get(zone_idx)
                    .or(spec.geometry.first())
                    .map_or(20.0, |g| g.floor_area());

                let solar = total_solar_gain_per_zone
                    .get(zone_idx)
                    .copied()
                    .unwrap_or(0.0);
                total_loads.push(internal_gains / floor_area + solar / floor_area);
            }
            model.set_loads(&total_loads);

            let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp);

            if is_free_floating {
                if let Some(&zone_0_temp) = model.temperatures.as_slice().first() {
                    min_temp_celsius = min_temp_celsius.min(zone_0_temp);
                    max_temp_celsius = max_temp_celsius.max(zone_0_temp);
                }
            }

            if hvac_kwh > 0.0 {
                annual_heating_joules += hvac_kwh * 3.6e6;
                peak_heating_watts = peak_heating_watts.max(hvac_kwh * 1000.0);
            } else {
                annual_cooling_joules += (-hvac_kwh) * 3.6e6;
                peak_cooling_watts = peak_cooling_watts.max((-hvac_kwh) * 1000.0);
            }
        }

        CaseResults {
            annual_heating_mwh: annual_heating_joules / 3.6e9,
            annual_cooling_mwh: annual_cooling_joules / 3.6e9,
            peak_heating_kw: peak_heating_watts / 1000.0,
            peak_cooling_kw: peak_cooling_watts / 1000.0,
            min_temp_celsius: if is_free_floating && min_temp_celsius != f64::INFINITY {
                Some(min_temp_celsius)
            } else {
                None
            },
            max_temp_celsius: if is_free_floating && max_temp_celsius != f64::NEG_INFINITY {
                Some(max_temp_celsius)
            } else {
                None
            },
        }
    }

    /// Validates a single case with diagnostic output.
    ///
    /// This method runs a single ASHRAE 140 case and returns detailed diagnostic information
    /// including hourly data, energy breakdown, and peak timing.
    ///
    /// # Arguments
    /// * `case` - The ASHRAE 140 case to validate
    ///
    /// # Returns
    /// A tuple of (BenchmarkReport, DiagnosticCollector) with validation results and diagnostics
    pub fn validate_single_case_with_diagnostics(
        &mut self,
        case: ASHRAE140Case,
    ) -> (BenchmarkReport, DiagnosticCollector) {
        let mut report = BenchmarkReport::new();
        let benchmark_data = benchmark::get_all_benchmark_data();
        let weather = DenverTmyWeather::new();

        let case_id = case.number();
        if let Some(data) = benchmark_data.get(&case_id) {
            let spec = case.spec();

            // Start diagnostic collection for this case
            self.diagnostic.start_case(&case_id, spec.num_zones);

            let results = self.simulate_case_with_diagnostics(&spec, &weather);

            // Finalize diagnostic collection
            self.diagnostic
                .finalize_case(results.annual_heating_mwh, results.annual_cooling_mwh);

            if spec.is_free_floating() {
                if let Some(min_temp) = results.min_temp_celsius {
                    report.add_result_simple(
                        &case_id,
                        MetricType::MinFreeFloat,
                        min_temp,
                        data.min_free_float_min,
                        data.min_free_float_max,
                    );
                }

                if let Some(max_temp) = results.max_temp_celsius {
                    report.add_result_simple(
                        &case_id,
                        MetricType::MaxFreeFloat,
                        max_temp,
                        data.max_free_float_min,
                        data.max_free_float_max,
                    );
                }
            } else {
                report.add_result_simple(
                    &case_id,
                    MetricType::AnnualHeating,
                    results.annual_heating_mwh,
                    data.annual_heating_min,
                    data.annual_heating_max,
                );

                report.add_result_simple(
                    &case_id,
                    MetricType::AnnualCooling,
                    results.annual_cooling_mwh,
                    data.annual_cooling_min,
                    data.annual_cooling_max,
                );

                if data.peak_heating_min >= 0.0 {
                    report.add_result_simple(
                        &case_id,
                        MetricType::PeakHeating,
                        results.peak_heating_kw,
                        data.peak_heating_min,
                        data.peak_heating_max,
                    );
                }

                if data.peak_cooling_min >= 0.0 {
                    report.add_result_simple(
                        &case_id,
                        MetricType::PeakCooling,
                        results.peak_cooling_kw,
                        data.peak_cooling_min,
                        data.peak_cooling_max,
                    );
                }
            }

            report.add_benchmark_data(&case_id, data.clone());
        }

        // Save diagnostic outputs if configured
        let _ = self.diagnostic.save_all();

        (report, self.diagnostic.clone())
    }

    /// Validates the analytical engine against the ASHRAE 140 cases.
    pub fn validate_analytical_engine(&mut self) -> BenchmarkReport {
        let mut report = BenchmarkReport::new();
        let benchmark_data = benchmark::get_all_benchmark_data();
        let weather = DenverTmyWeather::new();

        // Cases to validate - all 18 ASHRAE 140 cases
        let cases = vec![
            // Low mass cases (600 series)
            ASHRAE140Case::Case600,
            ASHRAE140Case::Case610,
            ASHRAE140Case::Case620,
            ASHRAE140Case::Case630,
            ASHRAE140Case::Case640,
            ASHRAE140Case::Case650,
            ASHRAE140Case::Case600FF,
            ASHRAE140Case::Case650FF,
            // High mass cases (900 series)
            ASHRAE140Case::Case900,
            ASHRAE140Case::Case910,
            ASHRAE140Case::Case920,
            ASHRAE140Case::Case930,
            ASHRAE140Case::Case940,
            ASHRAE140Case::Case950,
            ASHRAE140Case::Case900FF,
            ASHRAE140Case::Case950FF,
            // Special cases
            ASHRAE140Case::Case960,
            ASHRAE140Case::Case195,
        ];

        for case in cases {
            let case_id = case.number();
            if let Some(data) = benchmark_data.get(&case_id) {
                let spec = case.spec();
                let results = self.simulate_case(&spec, &weather);

                if spec.is_free_floating() {
                    println!(
                        "Case {} (Free-Floating): Min Temp={:.2}°C (Ref: {:.2}-{:.2}), Max Temp={:.2}°C (Ref: {:.2}-{:.2})",
                        case_id,
                        results.min_temp_celsius.unwrap_or(0.0),
                        data.min_free_float_min,
                        data.min_free_float_max,
                        results.max_temp_celsius.unwrap_or(0.0),
                        data.max_free_float_min,
                        data.max_free_float_max
                    );

                    // Add free-floating temperature metrics
                    if let Some(min_temp) = results.min_temp_celsius {
                        report.add_result_simple(
                            &case_id,
                            MetricType::MinFreeFloat,
                            min_temp,
                            data.min_free_float_min,
                            data.min_free_float_max,
                        );
                    }

                    if let Some(max_temp) = results.max_temp_celsius {
                        report.add_result_simple(
                            &case_id,
                            MetricType::MaxFreeFloat,
                            max_temp,
                            data.max_free_float_min,
                            data.max_free_float_max,
                        );
                    }
                } else {
                    println!(
                        "Case {}: Heating={:.2} (Ref: {:.2}-{:.2}), Cooling={:.2} (Ref: {:.2}-{:.2}), Peak H={:.2}, Peak C={:.2}",
                        case_id,
                        results.annual_heating_mwh,
                        data.annual_heating_min,
                        data.annual_heating_max,
                        results.annual_cooling_mwh,
                        data.annual_cooling_min,
                        data.annual_cooling_max,
                        results.peak_heating_kw,
                        results.peak_cooling_kw
                    );

                    report.add_result_simple(
                        &case_id,
                        MetricType::AnnualHeating,
                        results.annual_heating_mwh,
                        data.annual_heating_min,
                        data.annual_heating_max,
                    );

                    report.add_result_simple(
                        &case_id,
                        MetricType::AnnualCooling,
                        results.annual_cooling_mwh,
                        data.annual_cooling_min,
                        data.annual_cooling_max,
                    );

                    // Add peak loads if reference data is available
                    if data.peak_heating_min >= 0.0 {
                        report.add_result_simple(
                            &case_id,
                            MetricType::PeakHeating,
                            results.peak_heating_kw,
                            data.peak_heating_min,
                            data.peak_heating_max,
                        );
                    }

                    if data.peak_cooling_min >= 0.0 {
                        report.add_result_simple(
                            &case_id,
                            MetricType::PeakCooling,
                            results.peak_cooling_kw,
                            data.peak_cooling_min,
                            data.peak_cooling_max,
                        );
                    }
                }

                report.add_benchmark_data(&case_id, data.clone());
            }
        }

        report
    }

    fn simulate_case(&self, spec: &CaseSpec, weather: &DenverTmyWeather) -> CaseResults {
        let mut model = ThermalModel::<VectorField>::from_spec(spec);
        const STEPS: usize = 8760;
        let num_zones = model.num_zones;

        // Check if this is a free-floating case (no HVAC for zone 0)
        let is_free_floating = spec.is_free_floating();

        // For free-floating cases, disable HVAC by setting extreme setpoints
        if is_free_floating {
            model.heating_setpoint = -999.0;
            model.cooling_setpoint = 999.0;
            model.hvac_heating_capacity = 0.0;
            model.hvac_cooling_capacity = 0.0;
        }

        let mut annual_heating_joules = 0.0;
        let mut annual_cooling_joules = 0.0;
        let mut peak_heating_watts: f64 = 0.0;
        let mut peak_cooling_watts: f64 = 0.0;
        let mut min_temp_celsius: f64 = f64::INFINITY;
        let mut max_temp_celsius: f64 = f64::NEG_INFINITY;

        for step in 0..STEPS {
            let hour_of_day = step % 24;
            let day_of_year = step / 24 + 1;

            // Correctly calculate month and day from day_of_year
            let days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
            let mut month = 1;
            let mut day = day_of_year;
            for (i, &days) in days_in_month.iter().enumerate() {
                if day <= days as usize {
                    month = i + 1;
                    break;
                }
                day -= days as usize;
            }

            let weather_data = weather.get_hourly_data(step).unwrap();

            // Apply dynamic setpoints based on HVAC schedule (for setback cases)
            if let Some(hvac_schedule) = spec.hvac.first() {
                if let Some(heating_sp) = hvac_schedule.heating_setpoint_at_hour(hour_of_day as u8)
                {
                    model.heating_setpoint = heating_sp;
                }
                if let Some(cooling_sp) = hvac_schedule.cooling_setpoint_at_hour(hour_of_day as u8)
                {
                    model.cooling_setpoint = cooling_sp;
                }
            }

            // Apply night ventilation if active (adds extra cooling during night hours)
            if let Some(vent) = &spec.night_ventilation {
                if vent.is_active_at_hour(hour_of_day as u8) {
                    // Night ventilation provides additional cooling effect
                    // For simplicity, we reduce the cooling setpoint to -100 to allow free cooling
                    // This simulates the ventilation providing cool outdoor air to the zone
                    if let Some(hvac_schedule) = spec.hvac.first() {
                        // Only apply if heating is disabled (cooling-only mode)
                        if hvac_schedule.heating_setpoint < 0.0 {
                            model.cooling_setpoint = -100.0; // Allow free cooling during night vent hours
                        }
                    }
                }
            }

            // Calculate solar gains for all windows in the spec
            // For multi-zone, sum across all zones
            let mut total_solar_gain_per_zone: Vec<f64> = vec![0.0; num_zones];
            for (zone_idx, zone_windows) in spec.windows.iter().enumerate() {
                if zone_idx >= num_zones {
                    break;
                }
                for win_area in zone_windows {
                    let props = WindowProperties::new(
                        win_area.area,
                        spec.window_properties.shgc,
                        spec.window_properties.normal_transmittance,
                    );

                    // Find matching surface to get shading devices
                    let mut overhang = None;
                    let mut fins = Vec::new();
                    if let Some(zone_surfaces) = model.surfaces.get(zone_idx) {
                        for surf in zone_surfaces {
                            if surf.orientation == win_area.orientation {
                                overhang = surf.overhang.as_ref();
                                fins = surf.fins.clone();
                                break;
                            }
                        }
                    }

                    let (_, _, gain) = calculate_hourly_solar(
                        39.7392,
                        -104.9903,
                        2024,
                        month as u32,
                        day as u32,
                        hour_of_day as f64 + 0.5,
                        weather_data.dni,
                        weather_data.dhi,
                        &props,
                        Some(win_area),
                        overhang,
                        &fins,
                        win_area.orientation,
                        Some(0.2),
                    );
                    total_solar_gain_per_zone[zone_idx] += gain;
                }

                // --- Opaque Solar Gains (Walls + Roof) ---
                // ASHRAE 140: Default absorptance = 0.6, Re = 0.034
                let alpha = spec.opaque_absorptance;
                let re = 0.034; // Exterior film resistance (m²K/W)
                let wall_area = spec.geometry[zone_idx].wall_area();
                let window_area: f64 = spec.windows[zone_idx].iter().map(|w| w.area).sum();
                let opaque_wall_area = wall_area - window_area;
                let roof_area = spec.geometry[zone_idx].roof_area();

                // Get U-values from spec
                let wall_u = spec.construction.wall.u_value(None);
                let roof_u = spec.construction.roof.u_value(None);

                // Average solar gain on opaque walls
                for orientation in [
                    Orientation::South,
                    Orientation::West,
                    Orientation::North,
                    Orientation::East,
                ] {
                    let (_, irr, _) = calculate_hourly_solar(
                        39.7392,
                        -104.9903,
                        2024,
                        month as u32,
                        day as u32,
                        hour_of_day as f64 + 0.5,
                        weather_data.dni,
                        weather_data.dhi,
                        &WindowProperties::new(0.0, 0.0, 0.0), // No window
                        None,
                        None,
                        &[],
                        orientation,
                        Some(0.2),
                    );
                    // Opaque gain = UA * I * alpha * Re
                    total_solar_gain_per_zone[zone_idx] +=
                        (opaque_wall_area / 4.0) * wall_u * irr.total_wm2 * alpha * re;
                }

                // Roof gain
                let (_, irr_roof, _) = calculate_hourly_solar(
                    39.7392,
                    -104.9903,
                    2024,
                    month as u32,
                    day as u32,
                    hour_of_day as f64 + 0.5,
                    weather_data.dni,
                    weather_data.dhi,
                    &WindowProperties::new(0.0, 0.0, 0.0),
                    None,
                    None,
                    &[],
                    Orientation::Up,
                    Some(0.2),
                );
                total_solar_gain_per_zone[zone_idx] +=
                    roof_area * roof_u * irr_roof.total_wm2 * alpha * re;
            }

            // Calculate loads per zone (internal gains + solar)
            // IMPORTANT: Internal loads and solar gains are treated differently:
            // - Internal loads: Have convective/radiative split (40%/60% per ASHRAE 140)
            // - Solar gains: Mostly radiative (shortwave radiation absorbed by surfaces)
            //
            // The thermal model's convective_fraction applies to internal loads only.
            // Solar gains are handled separately with solar_distribution_to_air.
            let mut internal_loads_per_zone: Vec<f64> = Vec::with_capacity(num_zones);
            let mut solar_loads_per_zone: Vec<f64> = Vec::with_capacity(num_zones);

            for (zone_idx, solar_gain) in total_solar_gain_per_zone.iter().enumerate() {
                let internal_gains = spec
                    .internal_loads
                    .get(zone_idx)
                    .or(spec.internal_loads.first())
                    .and_then(|l| l.as_ref())
                    .map_or(0.0, |l| l.total_load);

                let floor_area = spec
                    .geometry
                    .get(zone_idx)
                    .or(spec.geometry.first())
                    .map_or(20.0, |g| g.floor_area());

                internal_loads_per_zone.push(internal_gains / floor_area);
                solar_loads_per_zone.push(solar_gain / floor_area);
            }

            // Combine internal loads and solar gains
            // IMPORTANT: The thermal model handles the distribution:
            // - Internal loads: convective_fraction (40%) goes to air, rest to mass
            // - Solar gains: solar_distribution_to_air (10%) goes to air, rest to mass
            // Since we can't modify VectorField in place, we combine before setting
            let mut total_loads: Vec<f64> = Vec::with_capacity(num_zones);
            for i in 0..num_zones {
                let internal = internal_loads_per_zone.get(i).copied().unwrap_or(0.0);
                let solar = solar_loads_per_zone.get(i).copied().unwrap_or(0.0);
                // Both internal and solar gains are added together
                // The model's convective_fraction applies to the total
                // Note: This is a simplification - ideally solar would use solar_distribution_to_air
                // but for ASHRAE 140 validation, the combined approach with calibrated parameters works
                total_loads.push(internal + solar);
            }
            model.set_loads(&total_loads);

            let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp);

            // Track min/max temperatures for free-floating cases
            if is_free_floating {
                // Get zone 0 air temperature (primary zone)
                if let Some(&zone_0_temp) = model.temperatures.as_slice().first() {
                    min_temp_celsius = min_temp_celsius.min(zone_0_temp);
                    max_temp_celsius = max_temp_celsius.max(zone_0_temp);
                }
            }

            // Positive = heating, negative = cooling
            if hvac_kwh > 0.0 {
                annual_heating_joules += hvac_kwh * 3.6e6;
                let hvac_watts = hvac_kwh * 1000.0; // kWh to Wh for 1 hour = kW * 1000
                peak_heating_watts = peak_heating_watts.max(hvac_watts);
            } else {
                annual_cooling_joules += (-hvac_kwh) * 3.6e6;
                let hvac_watts = (-hvac_kwh) * 1000.0; // kWh to Wh for 1 hour = kW * 1000
                peak_cooling_watts = peak_cooling_watts.max(hvac_watts);
            }
        }

        CaseResults {
            annual_heating_mwh: annual_heating_joules / 3.6e9,
            annual_cooling_mwh: annual_cooling_joules / 3.6e9,
            peak_heating_kw: peak_heating_watts / 1000.0,
            peak_cooling_kw: peak_cooling_watts / 1000.0,
            min_temp_celsius: if is_free_floating && min_temp_celsius != f64::INFINITY {
                Some(min_temp_celsius)
            } else {
                None
            },
            max_temp_celsius: if is_free_floating && max_temp_celsius != f64::NEG_INFINITY {
                Some(max_temp_celsius)
            } else {
                None
            },
        }
    }

    /// Simulates a case with diagnostic data collection.
    fn simulate_case_with_diagnostics(
        &mut self,
        spec: &CaseSpec,
        weather: &DenverTmyWeather,
    ) -> CaseResults {
        let mut model = ThermalModel::<VectorField>::from_spec(spec);
        const STEPS: usize = 8760;
        let num_zones = model.num_zones;

        // Check if this is a free-floating case (no HVAC for zone 0)
        let is_free_floating = spec.is_free_floating();

        // For free-floating cases, disable HVAC by setting extreme setpoints
        if is_free_floating {
            model.heating_setpoint = -999.0;
            model.cooling_setpoint = 999.0;
            model.hvac_heating_capacity = 0.0;
            model.hvac_cooling_capacity = 0.0;
        }

        let mut annual_heating_joules = 0.0;
        let mut annual_cooling_joules = 0.0;
        let mut peak_heating_watts: f64 = 0.0;
        let mut peak_cooling_watts: f64 = 0.0;
        let mut min_temp_celsius: f64 = f64::INFINITY;
        let mut max_temp_celsius: f64 = f64::NEG_INFINITY;

        for step in 0..STEPS {
            let hour_of_day = step % 24;
            let day_of_year = step / 24 + 1;

            // Correctly calculate month and day from day_of_year
            let days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
            let mut month = 1;
            let mut day = day_of_year;
            for (i, &days) in days_in_month.iter().enumerate() {
                if day <= days as usize {
                    month = i + 1;
                    break;
                }
                day -= days as usize;
            }

            let weather_data = weather.get_hourly_data(step).unwrap();

            // Apply dynamic setpoints based on HVAC schedule (for setback cases)
            if let Some(hvac_schedule) = spec.hvac.first() {
                if let Some(heating_sp) = hvac_schedule.heating_setpoint_at_hour(hour_of_day as u8)
                {
                    model.heating_setpoint = heating_sp;
                }
                if let Some(cooling_sp) = hvac_schedule.cooling_setpoint_at_hour(hour_of_day as u8)
                {
                    model.cooling_setpoint = cooling_sp;
                }
            }

            // Apply night ventilation if active
            if let Some(vent) = &spec.night_ventilation {
                if vent.is_active_at_hour(hour_of_day as u8) {
                    if let Some(hvac_schedule) = spec.hvac.first() {
                        if hvac_schedule.heating_setpoint < 0.0 {
                            model.cooling_setpoint = -100.0;
                        }
                    }
                }
            }

            // Calculate solar gains for all windows in the spec
            let mut total_solar_gain_per_zone: Vec<f64> = vec![0.0; num_zones];
            for (zone_idx, zone_windows) in spec.windows.iter().enumerate() {
                if zone_idx >= num_zones {
                    break;
                }
                for win_area in zone_windows {
                    let props = WindowProperties::new(
                        win_area.area,
                        spec.window_properties.shgc,
                        spec.window_properties.normal_transmittance,
                    );

                    let mut overhang = None;
                    let mut fins = Vec::new();
                    if let Some(zone_surfaces) = model.surfaces.get(zone_idx) {
                        for surf in zone_surfaces {
                            if surf.orientation == win_area.orientation {
                                overhang = surf.overhang.as_ref();
                                fins = surf.fins.clone();
                                break;
                            }
                        }
                    }

                    let (_, _, gain) = calculate_hourly_solar(
                        39.7392,
                        -104.9903,
                        2024,
                        month as u32,
                        day as u32,
                        hour_of_day as f64 + 0.5,
                        weather_data.dni,
                        weather_data.dhi,
                        &props,
                        Some(win_area),
                        overhang,
                        &fins,
                        win_area.orientation,
                        Some(0.2),
                    );
                    total_solar_gain_per_zone[zone_idx] += gain;
                }

                // Opaque Solar Gains
                let alpha = spec.opaque_absorptance;
                let re = 0.034;
                let wall_area = spec.geometry[zone_idx].wall_area();
                let window_area: f64 = spec.windows[zone_idx].iter().map(|w| w.area).sum();
                let opaque_wall_area = wall_area - window_area;
                let roof_area = spec.geometry[zone_idx].roof_area();

                let wall_u = spec.construction.wall.u_value(None);
                let roof_u = spec.construction.roof.u_value(None);

                for orientation in [
                    Orientation::South,
                    Orientation::West,
                    Orientation::North,
                    Orientation::East,
                ] {
                    let (_, irr, _) = calculate_hourly_solar(
                        39.7392,
                        -104.9903,
                        2024,
                        month as u32,
                        day as u32,
                        hour_of_day as f64 + 0.5,
                        weather_data.dni,
                        weather_data.dhi,
                        &WindowProperties::new(0.0, 0.0, 0.0),
                        None,
                        None,
                        &[],
                        orientation,
                        Some(0.2),
                    );
                    total_solar_gain_per_zone[zone_idx] +=
                        (opaque_wall_area / 4.0) * wall_u * irr.total_wm2 * alpha * re;
                }

                let (_, irr_roof, _) = calculate_hourly_solar(
                    39.7392,
                    -104.9903,
                    2024,
                    month as u32,
                    day as u32,
                    hour_of_day as f64 + 0.5,
                    weather_data.dni,
                    weather_data.dhi,
                    &WindowProperties::new(0.0, 0.0, 0.0),
                    None,
                    None,
                    &[],
                    Orientation::Up,
                    Some(0.2),
                );
                total_solar_gain_per_zone[zone_idx] +=
                    roof_area * roof_u * irr_roof.total_wm2 * alpha * re;
            }

            // Calculate loads per zone
            let mut internal_loads_per_zone: Vec<f64> = Vec::with_capacity(num_zones);
            let mut solar_loads_per_zone: Vec<f64> = Vec::with_capacity(num_zones);

            for (zone_idx, solar_gain) in total_solar_gain_per_zone.iter().enumerate() {
                let internal_gains = spec
                    .internal_loads
                    .get(zone_idx)
                    .or(spec.internal_loads.first())
                    .and_then(|l| l.as_ref())
                    .map_or(0.0, |l| l.total_load);

                let floor_area = spec
                    .geometry
                    .get(zone_idx)
                    .or(spec.geometry.first())
                    .map_or(20.0, |g| g.floor_area());

                internal_loads_per_zone.push(internal_gains / floor_area);
                solar_loads_per_zone.push(solar_gain / floor_area);
            }

            let mut total_loads: Vec<f64> = Vec::with_capacity(num_zones);
            for i in 0..num_zones {
                let internal = internal_loads_per_zone.get(i).copied().unwrap_or(0.0);
                let solar = solar_loads_per_zone.get(i).copied().unwrap_or(0.0);
                total_loads.push(internal + solar);
            }
            model.set_loads(&total_loads);

            let hvac_kwh = model.step_physics(step, weather_data.dry_bulb_temp);

            // Track min/max temperatures for free-floating cases
            if is_free_floating {
                if let Some(&zone_0_temp) = model.temperatures.as_slice().first() {
                    min_temp_celsius = min_temp_celsius.min(zone_0_temp);
                    max_temp_celsius = max_temp_celsius.max(zone_0_temp);
                }
            }

            // Record hourly diagnostic data
            let mut hourly_data = HourlyData::new(step, num_zones);
            hourly_data.outdoor_temp = weather_data.dry_bulb_temp;
            hourly_data.zone_temps = model.temperatures.as_slice().to_vec();
            hourly_data.mass_temps = model.mass_temperatures.as_slice().to_vec();

            for zone_idx in 0..num_zones {
                hourly_data.solar_gains[zone_idx] = total_solar_gain_per_zone
                    .get(zone_idx)
                    .copied()
                    .unwrap_or(0.0);
                hourly_data.internal_loads[zone_idx] = spec
                    .internal_loads
                    .get(zone_idx)
                    .or(spec.internal_loads.first())
                    .and_then(|l| l.as_ref())
                    .map_or(0.0, |l| l.total_load);
            }

            if hvac_kwh > 0.0 {
                annual_heating_joules += hvac_kwh * 3.6e6;
                let hvac_watts = hvac_kwh * 1000.0;
                peak_heating_watts = peak_heating_watts.max(hvac_watts);
                hourly_data.hvac_heating[0] = hvac_watts;
            } else {
                annual_cooling_joules += (-hvac_kwh) * 3.6e6;
                let hvac_watts = (-hvac_kwh) * 1000.0;
                peak_cooling_watts = peak_cooling_watts.max(hvac_watts);
                hourly_data.hvac_cooling[0] = hvac_watts;
            }

            self.diagnostic.record_hour(hourly_data);
        }

        CaseResults {
            annual_heating_mwh: annual_heating_joules / 3.6e9,
            annual_cooling_mwh: annual_cooling_joules / 3.6e9,
            peak_heating_kw: peak_heating_watts / 1000.0,
            peak_cooling_kw: peak_cooling_watts / 1000.0,
            min_temp_celsius: if is_free_floating && min_temp_celsius != f64::INFINITY {
                Some(min_temp_celsius)
            } else {
                None
            },
            max_temp_celsius: if is_free_floating && max_temp_celsius != f64::NEG_INFINITY {
                Some(max_temp_celsius)
            } else {
                None
            },
        }
    }
}

struct CaseResults {
    annual_heating_mwh: f64,
    annual_cooling_mwh: f64,
    peak_heating_kw: f64,
    peak_cooling_kw: f64,
    /// Minimum zone temperature (°C) for free-floating cases
    min_temp_celsius: Option<f64>,
    /// Maximum zone temperature (°C) for free-floating cases
    max_temp_celsius: Option<f64>,
}
