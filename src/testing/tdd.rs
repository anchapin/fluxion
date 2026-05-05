//! TDD (Test-Driven Development) Framework Functions
//!
//! This module provides the core functions used in TDD tests, including:
//! - `get_test_climate()`: Returns weather data for ASHRAE 140 testing
//! - `simulate_blind()`: Runs a blind simulation with given parameters
//! - `BlindTestSpec`: Specification for blind test cases
//! - `assert_in_range()`: Assertion helper for validation ranges

use crate::weather::epw::EpwWeatherSource;
use crate::weather::WeatherSource;
use crate::MassClass;

/// Schedule for night setpoint setbacks
#[derive(Debug, Clone)]
pub struct SetbackSchedule {
    /// Heating setpoint reduction during setback (°C)
    pub heating_setback_c: f64,
    /// Hour when setback starts (0-23)
    pub start_hour: u8,
    /// Hour when setback ends (0-23)
    pub end_hour: u8,
}

/// Specification for a blind test case (ASHRAE 140 style)
#[derive(Debug, Clone)]
pub struct BlindTestSpec {
    /// Thermal mass classification
    pub mass_class: MassClass,
    /// South-facing window area (m²)
    pub south_window: f64,
    /// Heating setpoint (°C)
    pub heating_setpoint: f64,
    /// Cooling setpoint (°C)
    pub cooling_setpoint: f64,
    /// Night setback schedule (if any)
    pub setback: Option<SetbackSchedule>,
}

impl BlindTestSpec {
    /// Create a light mass building specification
    pub fn light_mass() -> Self {
        BlindTestSpec {
            mass_class: MassClass::Light,
            south_window: 12.0,
            heating_setpoint: 20.0,
            cooling_setpoint: 27.0,
            setback: None,
        }
    }

    /// Create a heavy mass building specification
    pub fn heavy_mass() -> Self {
        BlindTestSpec {
            mass_class: MassClass::Heavy,
            south_window: 12.0,
            heating_setpoint: 20.0,
            cooling_setpoint: 27.0,
            setback: None,
        }
    }

    pub fn with_south_window(mut self, area: f64) -> Self {
        self.south_window = area;
        self
    }

    pub fn with_setpoints(mut self, heating: f64, cooling: f64) -> Self {
        self.heating_setpoint = heating;
        self.cooling_setpoint = cooling;
        self
    }

    pub fn with_night_setback(mut self, schedule: SetbackSchedule) -> Self {
        self.setback = Some(schedule);
        self
    }

    pub fn build(&self) -> Self {
        self.clone()
    }
}

/// Result from a blind simulation
#[derive(Debug, Clone)]
pub struct BlindSimulationResult {
    /// Annual heating energy (MWh)
    pub annual_heating_mwh: f64,
    /// Annual cooling energy (MWh)
    pub annual_cooling_mwh: f64,
    /// Peak heating load (kW)
    pub peak_heating_kw: f64,
    /// Peak cooling load (kW)
    pub peak_cooling_kw: f64,
}

/// Returns the test climate using real EPW weather data.
///
/// This function loads actual Denver TMY weather data from the EPW file
/// rather than using synthetic approximations. This provides more accurate
/// ASHRAE 140 validation results.
///
/// # Returns
///
/// `EpwWeatherSource` loaded from `tests/test_data/denver.epw`
///
/// # Panics
///
/// Panics if the EPW file cannot be loaded (which should never happen
/// in normal testing scenarios as the file is checked into the repository).
pub fn get_test_climate() -> EpwWeatherSource {
    EpwWeatherSource::from_file("tests/test_data/denver.epw")
        .expect("Failed to load test climate EPW file - this should never happen")
}

/// Runs a blind simulation with the given specification and weather data.
///
/// This function performs a simplified building energy simulation following
/// ASHRAE 140 procedures with the given parameters.
///
/// # Arguments
///
/// * `spec` - The blind test specification
/// * `weather` - The weather data source
///
/// # Returns
///
/// `BlindSimulationResult` containing annual and peak energy values
pub fn simulate_blind(spec: &BlindTestSpec, weather: &EpwWeatherSource) -> BlindSimulationResult {
    let mut annual_heating_kwh = 0.0;
    let mut annual_cooling_kwh = 0.0;
    let mut max_heating_load = 0.0;
    let mut max_cooling_load = 0.0;

    // Base load calculation factors based on mass class
    let load_factor = match spec.mass_class {
        MassClass::VeryLight => 2.5,
        MassClass::Light => 2.2,
        MassClass::Medium => 1.8,
        MassClass::Heavy => 1.4,
        MassClass::VeryHeavy => 1.0,
    };

    // Window solar gain factor (SHGC approximation)
    let window_factor = spec.south_window * 0.6; // Approximate solar heat gain coefficient

    for hour in 0..8760 {
        let data = weather.get_hourly_data(hour).unwrap();
        let hour_of_day = hour % 24;

        // Check if setback is active
        let setback_active = if let Some(ref sb) = spec.setback {
            let start: usize = sb.start_hour.into();
            let end: usize = sb.end_hour.into();
            if start < end {
                hour_of_day >= start || hour_of_day < end
            } else {
                hour_of_day >= start || hour_of_day < end
            }
        } else {
            false
        };

        // Calculate effective setpoints during setback
        let effective_heating = if setback_active {
            spec.heating_setpoint - spec.setback.as_ref().unwrap().heating_setback_c
        } else {
            spec.heating_setpoint
        };
        let effective_cooling = spec.cooling_setpoint;

        // Temperature difference from setpoints
        let heating_diff = (effective_heating - data.dry_bulb_temp).max(0.0);
        let cooling_diff = (data.dry_bulb_temp - effective_cooling).max(0.0);

        // Simple load calculation (U-value * area * deltaT / 1000 for kW)
        let envelope_factor = match spec.mass_class {
            MassClass::VeryLight => 0.8,
            MassClass::Light => 0.6,
            MassClass::Medium => 0.5,
            MassClass::Heavy => 0.4,
            MassClass::VeryHeavy => 0.3,
        };

        // Conduction load
        let conduction_load = envelope_factor * heating_diff * 50.0; // kW per K
        let cooling_load_base = envelope_factor * cooling_diff * 40.0;

        // Solar gain from window
        let solar_gain_kw = if data.ghi > 0.0 {
            let sky_factor = (data.dry_bulb_temp / 30.0).max(0.2).min(1.0);
            window_factor * data.ghi / 1000.0 * sky_factor
        } else {
            0.0
        };

        // Internal gain estimate (simplified)
        let internal_gain = 1.5; // kW

        // Total loads
        let total_heating =
            (conduction_load + internal_gain - solar_gain_kw).max(0.0) * load_factor;
        let total_cooling = (cooling_load_base + solar_gain_kw * 0.5).max(0.0) * load_factor;

        // Track maximums
        if total_heating > max_heating_load {
            max_heating_load = total_heating;
        }
        if total_cooling > max_cooling_load {
            max_cooling_load = total_cooling;
        }

        // Integrate energy
        annual_heating_kwh += total_heating;
        annual_cooling_kwh += total_cooling;
    }

    BlindSimulationResult {
        annual_heating_mwh: annual_heating_kwh / 1000.0,
        annual_cooling_mwh: annual_cooling_kwh / 1000.0,
        peak_heating_kw: max_heating_load,
        peak_cooling_kw: max_cooling_load,
    }
}

/// Asserts that a value is within the specified range.
///
/// # Arguments
///
/// * `value` - The value to check
/// * `min` - Minimum of the acceptable range
/// * `max` - Maximum of the acceptable range
/// * `message` - Description of what is being checked
///
/// # Panics
///
/// Panics if the value is outside the range
pub fn assert_in_range(value: f64, min: f64, max: f64, message: &str) {
    assert!(
        value >= min && value <= max,
        "{}: expected {:.2} to be in range [{:.2}, {:.2}]",
        message,
        value,
        min,
        max
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_test_climate() {
        let weather = get_test_climate();
        assert!(weather.location().is_some());

        // Verify we can get data for all hours
        for hour in [0, 100, 1000, 5000, 8759] {
            let data = weather.get_hourly_data(hour);
            assert!(data.is_ok());
        }
    }

    #[test]
    fn test_simulate_blind_light_mass() {
        let weather = get_test_climate();
        let spec = BlindTestSpec::light_mass()
            .with_south_window(12.0)
            .with_setpoints(20.0, 27.0)
            .build();

        let result = simulate_blind(&spec, &weather);

        // Sanity check - values should be positive and reasonable
        assert!(result.annual_heating_mwh > 0.0);
        assert!(result.annual_cooling_mwh > 0.0);
        assert!(result.peak_heating_kw > 0.0);
        assert!(result.peak_cooling_kw > 0.0);
    }

    #[test]
    fn test_simulate_blind_heavy_mass() {
        let weather = get_test_climate();
        let spec = BlindTestSpec::heavy_mass()
            .with_south_window(12.0)
            .with_setpoints(20.0, 27.0)
            .build();

        let result = simulate_blind(&spec, &weather);

        // Sanity check - values should be positive and reasonable
        assert!(result.annual_heating_mwh > 0.0);
        assert!(result.annual_cooling_mwh > 0.0);
        assert!(result.peak_heating_kw > 0.0);
        assert!(result.peak_cooling_kw > 0.0);
    }

    #[test]
    fn test_blind_spec_builder() {
        let spec = BlindTestSpec::light_mass()
            .with_south_window(20.0)
            .with_setpoints(18.0, 28.0)
            .with_night_setback(SetbackSchedule {
                heating_setback_c: 5.0,
                start_hour: 22,
                end_hour: 6,
            })
            .build();

        assert_eq!(spec.mass_class, MassClass::Light);
        assert_eq!(spec.south_window, 20.0);
        assert_eq!(spec.heating_setpoint, 18.0);
        assert_eq!(spec.cooling_setpoint, 28.0);
        assert!(spec.setback.is_some());
    }

    #[test]
    fn test_assert_in_range_pass() {
        assert_in_range(5.0, 0.0, 10.0, "Test");
        assert_in_range(0.0, 0.0, 10.0, "Test boundary");
        assert_in_range(10.0, 0.0, 10.0, "Test boundary");
    }

    #[test]
    #[should_panic(expected = "expected 15.00 to be in range")]
    fn test_assert_in_range_fail_low() {
        assert_in_range(15.0, 0.0, 10.0, "Test");
    }

    #[test]
    #[should_panic(expected = "expected -5.00 to be in range")]
    fn test_assert_in_range_fail_high() {
        assert_in_range(-5.0, 0.0, 10.0, "Test");
    }
}
