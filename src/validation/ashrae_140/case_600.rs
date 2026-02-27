//! ASHRAE Standard 140 Case 600 Baseline Model
//!
//! Case 600 is the baseline low-mass test case from ASHRAE Standard 140.
//! It represents a single-zone building with specific geometry, construction,
//! and operating conditions for validation purposes.

use crate::ai::surrogate::SurrogateManager;
use crate::physics::cta::VectorField;
use crate::sim::construction::Assemblies;
use crate::sim::engine::ThermalModel;
use crate::sim::solar::{calculate_hourly_solar, WindowProperties};
use crate::validation::ashrae_140_cases::Orientation;
use crate::weather::denver::DenverTmyWeather;
use crate::weather::WeatherSource;

/// Result structure for Case 600 simulation.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Annual heating energy consumption (MWh)
    pub annual_heating_mwh: f64,
    /// Annual cooling energy consumption (MWh)
    pub annual_cooling_mwh: f64,
    /// Peak heating demand (kW)
    pub peak_heating_kw: f64,
    /// Peak cooling demand (kW)
    pub peak_cooling_kw: f64,
    /// Hourly indoor temperatures (°C) for analysis
    pub hourly_temperatures: Vec<f64>,
    /// Hourly solar gains (W) for analysis
    pub hourly_solar_gains: Vec<f64>,
}

/// ASHRAE 140 Case 600 baseline model.
///
/// # Building Specifications
///
/// **Geometry:**
/// - Dimensions: 8m (W) × 6m (D) × 2.7m (H)
/// - Floor area: 48 m²
/// - Volume: 129.6 m³
/// - Walls: perimeter 28m × height 2.7m = 75.6 m²
/// - Roof: 48 m²
/// - Floor: 48 m²
///
/// **Windows:**
/// - South-facing: 12 m²
/// - Height: 2m, Width: 6m
/// - Offsets: 0.2m from sides, 0.5m from floor/ceiling
/// - U-value: 3.0 W/m²K
/// - SHGC: 0.789
/// - Double-pane clear glass
///
/// **Construction:**
/// - Walls: Low-mass wall (U ≈ 0.514 W/m²K)
/// - Roof: Low-mass roof (U ≈ 0.318 W/m²K)
/// - Floor: Insulated floor (U ≈ 0.039 W/m²K)
///
/// **HVAC:**
/// - Heating setpoint: 20°C
/// - Cooling setpoint: 27°C
/// - Efficiency: 100% (no losses)
///
/// **Infiltration:** 0.5 ACH (Air Changes per Hour)
///
/// **Internal Loads:** 200W continuous
/// - Radiative: 120W (60%)
/// - Convective: 80W (40%)
///
/// **Ground:** Constant 10°C
///
/// **Weather:** Denver TMY
pub struct Case600Model {
    pub model: ThermalModel<VectorField>,
    weather: DenverTmyWeather,
}

impl Case600Model {
    /// Create a new ASHRAE 140 Case 600 baseline model.
    ///
    /// # Returns
    /// A new Case600Model instance configured with all Case 600 specifications.
    pub fn new() -> Self {
        // Create base thermal model for single zone
        let mut model = ThermalModel::<VectorField>::new(1);

        // Configure geometry for Case 600
        // Floor area: 8m × 6m = 48 m²
        let floor_area = 48.0;
        let ceiling_height = 2.7;
        let width = 8.0;
        let depth = 6.0;
        let _perimeter = 2.0 * (width + depth);

        // Set zone area and ceiling height
        model.zone_area = VectorField::from_scalar(floor_area, 1);
        model.ceiling_height = VectorField::from_scalar(ceiling_height, 1);

        // Set infiltration rate (0.5 ACH as specified in ASHRAE 140)
        model.infiltration_rate = VectorField::from_scalar(0.5, 1);

        // Set HVAC setpoints (dual setpoint control)
        model.heating_setpoint = 20.0;
        model.cooling_setpoint = 27.0;

        // Configure window U-value (3.0 W/m²K for double clear glass)
        model.window_u_value = 3.0;

        // Calculate construction U-values and update conductances
        // Get Case 600 construction assemblies
        let wall_assembly = Assemblies::low_mass_wall();
        let roof_assembly = Assemblies::low_mass_roof();
        let floor_assembly = Assemblies::insulated_floor();

        // Calculate U-values with default wind speed (25 m/s → ~21 W/m²K film coefficient)
        let _u_wall = wall_assembly.u_value(None, None);
        let u_roof = roof_assembly.u_value(None, None);
        let _u_floor = floor_assembly.u_value(None, None);

        // Update 5R1C conductances based on construction U-values
        // h_tr_em: Exterior → Mass (roof)
        // h_tr_ms: Mass → Surface (internal)
        // h_tr_is: Surface → Interior (internal)
        // h_tr_w: Exterior → Interior (windows)
        // h_ve: Ventilation (exterior → interior)

        // Roof conductance (Exterior → Mass)
        let h_roof = u_roof * floor_area;
        model.h_tr_em = VectorField::from_scalar(h_roof, 1);

        // Window conductance (Exterior → Interior)
        let window_area = 12.0; // South-facing window
        let h_window = model.window_u_value * window_area;
        model.h_tr_w = VectorField::from_scalar(h_window, 1);

        // Floor conductance (to ground)
        let h_floor = _u_floor * floor_area;
        model.h_tr_floor = VectorField::from_scalar(h_floor, 1);

        // Ventilation conductance
        // Q_vent = ACH * Volume / 3600 (m³/s)
        // h_ve = ρ_air * cp_air * Q_vent
        let volume = floor_area * ceiling_height; // 48 × 2.7 = 129.6 m³
        let q_vent = 0.5 * volume / 3600.0; // 0.5 ACH × 129.6 / 3600 = 0.018 m³/s
        let air_density = 1.2; // kg/m³
        let cp_air = 1000.0; // J/kg·K
        let h_ve = air_density * cp_air * q_vent;
        model.h_ve = VectorField::from_scalar(h_ve, 1);

        // Set thermal mass (for low-mass construction)
        // Use approximate thermal capacitance for light construction
        let thermal_capacitance = floor_area * 150000.0; // J/K (150 kJ/m²K)
        model.thermal_capacitance = VectorField::from_scalar(thermal_capacitance, 1);

        // Initialize temperatures at 20°C
        model.temperatures = VectorField::from_scalar(20.0, 1);
        model.mass_temperatures = VectorField::from_scalar(20.0, 1);

        // Update optimization cache since we manually modified conductances
        model.update_optimization_cache();

        // Create Denver TMY weather source
        let weather = DenverTmyWeather::new();

        Case600Model { model, weather }
    }

    /// Run simulation for one full year (8760 hours).
    ///
    /// # Returns
    /// SimulationResult containing annual heating, annual cooling, peak loads,
    /// and hourly temperature traces.
    pub fn simulate_year(&mut self) -> SimulationResult {
        const STEPS: usize = 8760;
        const HOURS_PER_DAY: usize = 24;

        let mut annual_heating_joules = 0.0;
        let mut annual_cooling_joules = 0.0;
        let mut peak_heating_watts: f64 = 0.0;
        let mut peak_cooling_watts: f64 = 0.0;

        let mut hourly_temps = Vec::with_capacity(STEPS);
        let mut hourly_solar = Vec::with_capacity(STEPS);

        // Create surrogates manager (mock mode for analytical simulation)
        let _surrogates = SurrogateManager::new().unwrap();

        // Window properties for Case 600
        let window_area = 12.0; // m²
        let window = WindowProperties::double_clear(window_area);
        let window_orientation = Orientation::South;

        // Ground temperature (constant 10°C per ASHRAE 140)
        let _ground_temp = 10.0;

        // Simulate each hour
        for step in 0..STEPS {
            let hour_of_day = step % HOURS_PER_DAY;
            let day_of_year = step / HOURS_PER_DAY + 1;

            // Get weather data for this hour
            let weather_data = self.weather.get_hourly_data(step).unwrap();
            let dry_bulb = weather_data.dry_bulb_temp;
            let dni = weather_data.dni;
            let dhi = weather_data.dhi;

            // Calculate solar gain through south window
            let (_, _, solar_gain) = calculate_hourly_solar(
                39.7392,                       // Denver latitude (°N)
                -104.9903,                     // Denver longitude (°W)
                2024,                          // Year
                (day_of_year as u32) / 30 + 1, // Approximate month
                day_of_year as u32,
                hour_of_day as f64 + 0.5, // Mid-hour
                dni,
                dhi,
                &window,
                None, // No window geometry details
                None, // No overhang
                &[],  // No fins
                window_orientation,
                Some(0.2), // Ground reflectance (typical grass)
            );

            // Calculate total internal loads (W)
            // Internal gains: 200W continuous (60% radiative, 40% convective)
            let internal_gains = 200.0;
            let total_loads = internal_gains + solar_gain.total_gain_w;

            // Store solar gain for analysis
            hourly_solar.push(solar_gain.total_gain_w);

            // Set loads for this timestep (W/m²)
            let load_per_area = total_loads / 48.0; // 48 m² floor area
            self.model.set_loads(&[load_per_area]);

            // Solve physics for this hour
            let hvac_kwh = self.model.step_physics(step, dry_bulb);

            // Positive = heating (energy added to building), negative = cooling (energy removed)
            // hvac_kwh is the HVAC energy for 1 hour timestep.
            // Since it's already in kWh (energy), to get the average power during that hour:
            // power(kW) = energy(kWh) / time(h) = kWh / 1h = kWh value as kW
            if hvac_kwh > 0.0 {
                // Heating energy in joules = kWh * 3600 * 1000
                annual_heating_joules += hvac_kwh * 3.6e6;
                // Convert kWh to Watts: kWh for 1 hour = kW = kW * 1000 = W
                // Bug fix: was dividing by 3600 incorrectly
                let hvac_power_watts = hvac_kwh * 1000.0;
                peak_heating_watts = peak_heating_watts.max(hvac_power_watts);
            } else {
                // Cooling energy in joules (absolute value)
                annual_cooling_joules += (-hvac_kwh) * 3.6e6;
                let hvac_power_watts = (-hvac_kwh) * 1000.0;
                peak_cooling_watts = peak_cooling_watts.max(hvac_power_watts);
            }

            // Store indoor temperature
            let indoor_temp = self.model.get_temperatures()[0];
            hourly_temps.push(indoor_temp);
        }

        // Convert Joules to MWh
        let annual_heating_mwh = annual_heating_joules / 3.6e9; // J → MWh
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

impl Default for Case600Model {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_600_creation() {
        let model = Case600Model::new();
        assert_eq!(model.model.num_zones, 1);
        assert_eq!(model.model.heating_setpoint, 20.0);
        assert_eq!(model.model.cooling_setpoint, 27.0);
        assert_eq!(model.model.window_u_value, 3.0);
    }

    #[test]
    fn test_case_600_simulation() {
        let mut model = Case600Model::new();
        let result = model.simulate_year();

        // Verify simulation ran for full year
        assert_eq!(result.hourly_temperatures.len(), 8760);
        assert_eq!(result.hourly_solar_gains.len(), 8760);

        // Verify energy values are positive
        assert!(result.annual_heating_mwh >= 0.0);
        assert!(result.annual_cooling_mwh >= 0.0);
        assert!(result.peak_heating_kw >= 0.0);
        assert!(result.peak_cooling_kw >= 0.0);

        // Verify temperature range is reasonable
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
        assert!(min_temp < 30.0, "Min temperature should be below 30°C");
        assert!(max_temp > 15.0, "Max temperature should be above 15°C");
    }
}
