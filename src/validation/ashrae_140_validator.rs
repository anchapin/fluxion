use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;
use crate::sim::solar::{calculate_hourly_solar, WindowProperties};
use crate::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec};
use crate::validation::benchmark;
use crate::validation::report::{BenchmarkReport, MetricType};
use crate::weather::denver::DenverTmyWeather;
use crate::weather::WeatherSource;

/// Validator for ASHRAE 140 standard cases.
pub struct ASHRAE140Validator;

impl Default for ASHRAE140Validator {
    fn default() -> Self {
        Self::new()
    }
}

impl ASHRAE140Validator {
    /// Creates a new ASHRAE 140 validator.
    pub fn new() -> Self {
        Self {}
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

        for step in 0..STEPS {
            let hour_of_day = step % 24;
            let day_of_year = step / 24 + 1;

            let weather_data = weather.get_hourly_data(step).unwrap();

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
                        (day_of_year as u32) / 30 + 1,
                        day_of_year as u32,
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
            }

            // Calculate loads per zone (internal gains + solar)
            // For zones without internal loads specified, use first zone's value
            let mut loads_per_zone: Vec<f64> = Vec::with_capacity(num_zones);
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

                let total_load = internal_gains + solar_gain;
                loads_per_zone.push(total_load / floor_area);
            }

            model.set_loads(&loads_per_zone);

            let hvac_energy_kwh = model.step_physics(step, weather_data.dry_bulb_temp);
            let hvac_energy_joules = hvac_energy_kwh * 3.6e6;

            // For non-free-floating cases, categorize HVAC energy based on free-floating temperature
            if !is_free_floating {
                // Get the free-floating temperature BEFORE HVAC is applied
                // This tells us whether heating or cooling is needed
                let t_i_free = model.calculate_free_float_temperature(weather_data.dry_bulb_temp);

                // Determine HVAC mode based on FREE-FLOATING temperature
                if t_i_free < model.heating_setpoint {
                    // Free-floating temp is below heating setpoint - HVAC was heating
                    annual_heating_joules += hvac_energy_joules;
                } else if t_i_free > model.cooling_setpoint {
                    // Free-floating temp is above cooling setpoint - HVAC was cooling
                    annual_cooling_joules += hvac_energy_joules;
                }
                // If free-floating temp is in deadband, no HVAC energy used
            }
            // For free-floating cases, HVAC is disabled so no energy is added
        }

        CaseResults {
            annual_heating_mwh: annual_heating_joules / 3.6e9,
            annual_cooling_mwh: annual_cooling_joules / 3.6e9,
        }
    }
}

struct CaseResults {
    annual_heating_mwh: f64,
    annual_cooling_mwh: f64,
}
