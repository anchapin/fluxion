//! ASHRAE Standard 140 Case 600-CZ7 Climate Zone 7/8 Model
//!
//! Case 600-CZ7 is a variant of Case 600 using Minneapolis (Climate Zone 7/8 - very cold)
//! weather data instead of Denver (Climate Zone 5).
//!
//! This validates that fluxion correctly handles very cold climate conditions
//! where heating loads dominate and cooling loads are minimal.

use crate::ai::surrogate::SurrogateManager;
use crate::physics::cta::VectorField;
use crate::sim::construction::Assemblies;
use crate::sim::engine::ThermalModel;
use crate::sim::solar::{calculate_hourly_solar, WindowProperties};
use crate::validation::ashrae_140_cases::Orientation;
use crate::weather::minneapolis::MinneapolisTmyWeather;
use crate::weather::WeatherSource;

#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub annual_heating_mwh: f64,
    pub annual_cooling_mwh: f64,
    pub peak_heating_kw: f64,
    pub peak_cooling_kw: f64,
    pub hourly_temperatures: Vec<f64>,
    pub hourly_solar_gains: Vec<f64>,
}

/// Case 600-CZ7: Low-mass building in very cold climate (Minneapolis).
///
/// # Building Specifications
///
/// Same as Case 600 baseline:
/// - Single-zone, 48 m² floor area
/// - Low-mass construction
/// - Heating setpoint: 20°C
/// - Cooling setpoint: 27°C
///
/// # Climate
///
/// Minneapolis, MN (Climate Zone 7/8 - very cold):
/// - Very cold winters
/// - Warm summers
/// - High heating loads
/// - Moderate cooling loads
pub struct Case600CZ7Model {
    pub model: ThermalModel<VectorField>,
    weather: MinneapolisTmyWeather,
}

impl Case600CZ7Model {
    pub fn new() -> Self {
        let mut model = ThermalModel::<VectorField>::new(1);

        let floor_area = 48.0;
        let ceiling_height = 2.7;

        model.setpoints.zone_area = VectorField::from_scalar(floor_area, 1);
        model.setpoints.ceiling_height = VectorField::from_scalar(ceiling_height, 1);
        model.setpoints.infiltration_rate = VectorField::from_scalar(0.5, 1);
        model.setpoints.heating_setpoint = 20.0;
        model.setpoints.cooling_setpoint = 27.0;
        model.solar.window_u_value = 3.0;

        let _wall_assembly = Assemblies::low_mass_wall();
        let roof_assembly = Assemblies::low_mass_roof();
        let floor_assembly = Assemblies::insulated_floor();

        let u_roof = roof_assembly.u_value(None, None);

        let h_roof = u_roof * floor_area;
        model.conduction.h_tr_em = VectorField::from_scalar(h_roof, 1);

        let window_area = 12.0;
        let h_window = model.solar.window_u_value * window_area;
        model.conduction.h_tr_w = VectorField::from_scalar(h_window, 1);

        let h_floor = floor_assembly.u_value(None, None) * floor_area;
        model.conduction.h_tr_floor = VectorField::from_scalar(h_floor, 1);

        let volume = floor_area * ceiling_height;
        let q_vent = 0.5 * volume / 3600.0;
        let air_density = 1.2;
        let cp_air = 1000.0;
        let h_ve = air_density * cp_air * q_vent;
        model.conduction.h_ve = VectorField::from_scalar(h_ve, 1);

        let thermal_capacitance = floor_area * 150000.0;
        model.mass.thermal_capacitance = VectorField::from_scalar(thermal_capacitance, 1);

        model.setpoints.temperatures = VectorField::from_scalar(10.0, 1);
        model.mass.mass_temperatures = VectorField::from_scalar(10.0, 1);

        model.update_optimization_cache();

        let weather = MinneapolisTmyWeather::new();

        Case600CZ7Model { model, weather }
    }

    pub fn simulate_year(&mut self) -> SimulationResult {
        const STEPS: usize = 8760;
        const HOURS_PER_DAY: usize = 24;

        let mut annual_heating_joules = 0.0;
        let mut annual_cooling_joules = 0.0;
        let mut peak_heating_watts: f64 = 0.0;
        let mut peak_cooling_watts: f64 = 0.0;

        let mut hourly_temps = Vec::with_capacity(STEPS);
        let mut hourly_solar = Vec::with_capacity(STEPS);

        let _surrogates = SurrogateManager::new().unwrap();

        let window_area = 12.0;
        let window = WindowProperties::double_clear(window_area);
        let window_orientation = Orientation::South;

        for step in 0..STEPS {
            let hour_of_day = step % HOURS_PER_DAY;
            let day_of_year = step / HOURS_PER_DAY + 1;

            let weather_data = self.weather.get_hourly_data(step).unwrap();
            let dry_bulb = weather_data.dry_bulb_temp;
            let dni = weather_data.dni;
            let dhi = weather_data.dhi;

            let (_, _, solar_gain_watts) = calculate_hourly_solar(
                44.88,
                -93.22,
                2024,
                (day_of_year as u32) / 30 + 1,
                day_of_year as u32,
                hour_of_day as f64 + 0.5,
                dni,
                dhi,
                &window,
                None,
                None,
                &[],
                window_orientation,
                Some(0.2),
                None,
            );

            let internal_gains = 200.0;
            let total_loads = internal_gains + solar_gain_watts.total_gain_w;

            hourly_solar.push(solar_gain_watts.total_gain_w);

            let load_per_area = total_loads / 48.0;
            self.model.set_loads(&[load_per_area]);

            let hvac_kwh = self.model.step_physics(step, dry_bulb, 3600.0);

            if hvac_kwh > 0.0 {
                annual_heating_joules += hvac_kwh * 3.6e6;
                let hvac_power_watts = hvac_kwh * 1000.0;
                peak_heating_watts = peak_heating_watts.max(hvac_power_watts);
            } else {
                annual_cooling_joules += (-hvac_kwh) * 3.6e6;
                let hvac_power_watts = (-hvac_kwh) * 1000.0;
                peak_cooling_watts = peak_cooling_watts.max(hvac_power_watts);
            }

            let indoor_temp = self.model.get_temperatures()[0];
            hourly_temps.push(indoor_temp);
        }

        let annual_heating_mwh = annual_heating_joules / 3.6e9;
        let annual_cooling_mwh = annual_cooling_joules / 3.6e9;
        let peak_heating_kw = peak_heating_watts / 1000.0;
        let peak_cooling_kw = peak_cooling_watts / 1000.0;

        SimulationResult {
            annual_heating_mwh,
            annual_cooling_mwh,
            peak_heating_kw,
            peak_cooling_kw,
            hourly_temperatures: hourly_temps,
            hourly_solar_gains: hourly_solar,
        }
    }
}

impl Default for Case600CZ7Model {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_600_cz7_creation() {
        let model = Case600CZ7Model::new();
        assert_eq!(model.model.hvac.num_zones, 1);
        assert_eq!(model.model.setpoints.heating_setpoint, 20.0);
        assert_eq!(model.model.setpoints.cooling_setpoint, 27.0);
        assert_eq!(model.model.solar.window_u_value, 3.0);
    }

    #[test]
    fn test_case_600_cz7_simulation() {
        let mut model = Case600CZ7Model::new();
        let result = model.simulate_year();

        assert_eq!(result.hourly_temperatures.len(), 8760);
        assert_eq!(result.hourly_solar_gains.len(), 8760);

        assert!(result.annual_heating_mwh >= 0.0);
        assert!(result.annual_cooling_mwh >= 0.0);
        assert!(result.peak_heating_kw >= 0.0);
        assert!(result.peak_cooling_kw >= 0.0);

        let min_temp = result
            .hourly_temperatures
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_temp = result
            .hourly_temperatures
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_temp > 5.0,
            "Max temp should be reasonable for cold climate"
        );
        assert!(min_temp < 35.0, "Min temp should be reasonable");
    }

    #[test]
    fn test_case_600_cz7_higher_heating_than_denver() {
        let mut model_cz7 = Case600CZ7Model::new();
        let result_cz7 = model_cz7.simulate_year();

        let mut model_denver = crate::validation::ashrae_140::Case600Model::new();
        let result_denver = model_denver.simulate_year();

        println!(
            "CZ7 Annual Heating: {:.2} MWh, Denver Annual Heating: {:.2} MWh",
            result_cz7.annual_heating_mwh, result_denver.annual_heating_mwh
        );
        println!(
            "CZ7 Annual Cooling: {:.2} MWh, Denver Annual Cooling: {:.2} MWh",
            result_cz7.annual_cooling_mwh, result_denver.annual_cooling_mwh
        );

        assert!(
            result_cz7.annual_heating_mwh > result_denver.annual_heating_mwh,
            "CZ7 heating {:.2} MWh should be > Denver heating {:.2} MWh",
            result_cz7.annual_heating_mwh,
            result_denver.annual_heating_mwh
        );
    }
}
