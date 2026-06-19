//! Psychrometric calculations for building energy modeling.
//!
//! This module provides ASHRAE-compliant psychrometric calculations following
//! the methodology in ASHRAE Handbook of Fundamentals, Chapter 1.
//!
//! # Calculations Provided
//!
//! - **Saturation vapor pressure**: Magnus-Tetens (≥0°C) + ASHRAE Hyland-Wexler ice (<0°C)
//! - **Dew point temperature**: Newton-Raphson iteration
//! - **Wet-bulb temperature**: Psychrometric equation solving
//! - **Humidity ratio**: kg_water_vapor / kg_dry_air
//! - **Enthalpy**: kJ/kg of moist air
//!
//! # Units
//!
//! - Temperature: °C
//! - Pressure: Pa
//! - Humidity ratio: kg/kg (kg_water_vapor / kg_dry_air)
//! - Enthalpy: kJ/kg
//!
//! # References
//!
//! - ASHRAE Handbook of Fundamentals, Chapter 1: Psychrometrics
//! - ASHRAE Standard 140: Standard Method of Test for the Evaluation of Building Energy Analysis Computer Programs

/// Standard atmospheric pressure at sea level (Pa).
///
/// This constant is used as the default pressure for psychrometric calculations
/// when altitude-specific pressure is not provided.
pub const STANDARD_ATMOSPHERIC_PRESSURE_Pa: f64 = 101325.0;

/// Calculates saturation vapor pressure at a given temperature.
///
/// Matches the ASHRAE Handbook of Fundamentals (Chapter 1) saturation table
/// within 0.1% across the building operating range (-40°C to 60°C).
///
/// - For `T >= 0°C`, uses the Magnus-Tetens approximation, which reproduces the
///   ASHRAE saturation-over-water values to within ~0.1%.
/// - For `T < 0°C`, uses the ASHRAE Hyland-Wexler saturation-over-ice equation
///   (ASHRAE HoF Eq. 6), since Tetens diverges by >20% for sub-zero conditions
///   (e.g. ASHRAE p_ws(-20°C) = 103 Pa; Tetens gives 125 Pa).
///
/// # Formula (T >= 0°C)
///
/// ```text
/// p_sat = A × exp((B × T) / (T + C))
/// ```
///
/// # Formula (T < 0°C, Hyland-Wexler ice, T_K in Kelvin)
///
/// ```text
/// ln(p_sat) = C1/T_K + C2 + C3·T_K + C4·T_K² + C5·T_K³ + C6·T_K⁴ + C7·ln(T_K)
/// ```
///
/// Where:
/// - `p_sat` = saturation vapor pressure (Pa)
/// - `T` = temperature (°C), `T_K` = temperature (K)
/// - `A = 610.78 Pa`, `B = 17.27`, `C = 237.3°C` (Tetens)
///
/// # Arguments
///
/// * `temperature` - Temperature in °C
///
/// # Returns
///
/// Saturation vapor pressure in Pa
///
/// # Example
///
/// ```
/// use fluxion::weather::psychrometrics::saturation_vapor_pressure;
///
/// let p_sat = saturation_vapor_pressure(20.0);
/// assert!((p_sat - 2339.0).abs() < 5.0); // ~2339 Pa at 20°C
/// // ASHRAE saturation-over-ice value at -20°C is ~103 Pa
/// assert!((saturation_vapor_pressure(-20.0) - 103.0).abs() < 1.0);
/// ```
pub fn saturation_vapor_pressure(temperature: f64) -> f64 {
    if temperature < 0.0 {
        const C1: f64 = -5674.5359;
        const C2: f64 = 6.3925247;
        const C3: f64 = -9.677843e-3;
        const C4: f64 = 6.2215701e-7;
        const C5: f64 = 2.0747825e-9;
        const C6: f64 = -9.484024e-13;
        const C7: f64 = 4.1635019;

        let tk = temperature + 273.15;
        (C1 / tk
            + C2
            + C3 * tk
            + C4 * tk.powi(2)
            + C5 * tk.powi(3)
            + C6 * tk.powi(4)
            + C7 * tk.ln())
        .exp()
    } else {
        const A: f64 = 610.78;
        const B: f64 = 17.27;
        const C: f64 = 237.3;

        A * ((B * temperature) / (temperature + C)).exp()
    }
}

/// Calculates dew point temperature from dry bulb and relative humidity.
///
/// Uses Newton-Raphson iteration to solve for the temperature at which
/// the saturation vapor pressure equals the actual water vapor pressure.
///
/// # Algorithm
///
/// 1. Calculate water vapor pressure: p_water = p_sat(T) × (RH/100)
/// 2. Newton-Raphson iteration: Td_{n+1} = Td_n - (p_sat(Td_n) - p_water) / (dp_sat/dT)
/// 3. Derivative: central finite difference of `saturation_vapor_pressure` (branch-consistent
///    for both the Tetens water curve and the Hyland-Wexler ice curve below 0°C)
/// 4. Initial guess: Td = dry_bulb
/// 5. Convergence tolerance: 1e-6
/// 6. Max iterations: 20 (prevent infinite loops)
/// 7. Physical constraint: dew_point ≤ dry_bulb (clamp after iteration)
///
/// # Arguments
///
/// * `dry_bulb` - Dry bulb temperature in °C
/// * `relative_humidity` - Relative humidity (0-100)
/// * `pressure` - Atmospheric pressure in Pa (unused in calculation, for API consistency)
///
/// # Returns
///
/// Dew point temperature in °C, always ≤ dry_bulb temperature
///
/// # Example
///
/// ```
/// use fluxion::weather::psychrometrics::calculate_dew_point;
///
/// let dp = calculate_dew_point(25.0, 50.0, 101325.0);
/// assert!((dp - 13.9).abs() < 0.5); // ~13.9°C at 25°C, 50% RH
/// ```
pub fn calculate_dew_point(dry_bulb: f64, relative_humidity: f64, _pressure: f64) -> f64 {
    const MAX_ITERATIONS: usize = 20;
    const TOLERANCE: f64 = 1e-6;
    const DERIV_EPSILON: f64 = 1e-4;

    // Calculate water vapor pressure
    let p_sat = saturation_vapor_pressure(dry_bulb);
    let p_water = p_sat * (relative_humidity / 100.0);

    // Newton-Raphson iteration
    let mut td = dry_bulb; // Initial guess

    for _ in 0..MAX_ITERATIONS {
        let p_sat_td = saturation_vapor_pressure(td);
        let delta_p = p_sat_td - p_water;

        if delta_p.abs() < TOLERANCE {
            break;
        }

        // Central finite-difference derivative of saturation_vapor_pressure
        // (consistent with whichever saturation branch is active).
        let derivative = (saturation_vapor_pressure(td + DERIV_EPSILON)
            - saturation_vapor_pressure(td - DERIV_EPSILON))
            / (2.0 * DERIV_EPSILON);

        // Prevent division by zero
        if derivative.abs() < 1e-10 {
            break;
        }

        td -= delta_p / derivative;
    }

    // Physical constraint: dew point cannot exceed dry bulb temperature
    td.min(dry_bulb)
}

/// Calculates humidity ratio (kg_water_vapor / kg_dry_air).
///
/// # Formula
///
/// ```text
/// ω = (0.62198 × p_sat(T) × RH/100) / (P - p_sat(T) × RH/100)
/// ```
///
/// Where:
/// - `ω` = humidity ratio (kg_water_vapor / kg_dry_air)
/// - `p_sat(T)` = saturation vapor pressure at temperature T (Pa)
/// - `RH` = relative humidity (0-100)
/// - `P` = atmospheric pressure (Pa)
/// - `0.62198` = ratio of molecular weights (H2O / dry_air)
///
/// # Arguments
///
/// * `dry_bulb` - Dry bulb temperature in °C
/// * `relative_humidity` - Relative humidity (0-100)
/// * `pressure` - Atmospheric pressure in Pa
///
/// # Returns
///
/// Humidity ratio in kg_water_vapor / kg_dry_air
///
/// # Example
///
/// ```
/// use fluxion::weather::psychrometrics::calculate_humidity_ratio;
///
/// let omega = calculate_humidity_ratio(25.0, 50.0, 101325.0);
/// assert!((omega - 0.0099).abs() < 0.0001); // ~0.0099 kg/kg at 25°C, 50% RH
/// ```
pub fn calculate_humidity_ratio(dry_bulb: f64, relative_humidity: f64, pressure: f64) -> f64 {
    const RATIO_MW: f64 = 0.62198; // H2O / dry_air molecular weight ratio

    let p_sat = saturation_vapor_pressure(dry_bulb);
    let p_water = p_sat * (relative_humidity / 100.0);

    (RATIO_MW * p_water) / (pressure - p_water)
}

/// Calculates enthalpy of moist air (kJ/kg).
///
/// Uses the exact ASHRAE formula accounting for both dry air and water vapor.
///
/// # Formula
///
/// ```text
/// h = 1.006 × T + ω × (2501 + 1.86 × T)
/// ```
///
/// Where:
/// - `h` = enthalpy of moist air (kJ/kg)
/// - `T` = temperature (°C)
/// - `ω` = humidity ratio (kg_water_vapor / kg_dry_air)
/// - `1.006` = specific heat of dry air (kJ/(kg·K))
/// - `2501` = latent heat of vaporization at 0°C (kJ/kg)
/// - `1.86` = specific heat of water vapor (kJ/(kg·K))
///
/// # Arguments
///
/// * `dry_bulb` - Dry bulb temperature in °C
/// * `relative_humidity` - Relative humidity (0-100)
/// * `pressure` - Atmospheric pressure in Pa
///
/// # Returns
///
/// Enthalpy of moist air in kJ/kg
///
/// # Example
///
/// ```
/// use fluxion::weather::psychrometrics::calculate_enthalpy;
///
/// let h = calculate_enthalpy(25.0, 50.0, 101325.0);
/// assert!((h - 50.4).abs() < 0.5); // ~50.4 kJ/kg at 25°C, 50% RH
/// ```
pub fn calculate_enthalpy(dry_bulb: f64, relative_humidity: f64, pressure: f64) -> f64 {
    const CP_DRY_AIR: f64 = 1.006; // kJ/(kg·K)
    const LATENT_HEAT: f64 = 2501.0; // kJ/kg
    const CP_WATER_VAPOR: f64 = 1.86; // kJ/(kg·K)

    let omega = calculate_humidity_ratio(dry_bulb, relative_humidity, pressure);

    CP_DRY_AIR * dry_bulb + omega * (LATENT_HEAT + CP_WATER_VAPOR * dry_bulb)
}

/// Calculates wet-bulb temperature.
///
/// Solves the psychrometric equation iteratively for the temperature at which
/// air becomes saturated (100% RH) while maintaining the same enthalpy.
///
/// # Algorithm
///
/// 1. Enthalpy balance: h(Tw, RH=100%) = h(T, RH)
/// 2. Use Newton-Raphson with initial guess: Tw = (dry_bulb + dew_point) / 2
/// 3. Convergence tolerance: 1e-6
/// 4. Max iterations: 20
///
/// # Arguments
///
/// * `dry_bulb` - Dry bulb temperature in °C
/// * `relative_humidity` - Relative humidity (0-100)
/// * `pressure` - Atmospheric pressure in Pa
///
/// # Returns
///
/// Wet-bulb temperature in °C
///
/// # Example
///
/// ```
/// use fluxion::weather::psychrometrics::calculate_wet_bulb;
///
/// let wb = calculate_wet_bulb(25.0, 50.0, 101325.0);
/// // Wet-bulb is between dew point and dry bulb
/// assert!(wb > 13.0 && wb < 25.0);
/// ```
pub fn calculate_wet_bulb(dry_bulb: f64, relative_humidity: f64, pressure: f64) -> f64 {
    const MAX_ITERATIONS: usize = 20;
    const TOLERANCE: f64 = 1e-6;

    // Calculate target enthalpy
    let target_enthalpy = calculate_enthalpy(dry_bulb, relative_humidity, pressure);

    // Initial guess: average of dry bulb and dew point
    let dp = calculate_dew_point(dry_bulb, relative_humidity, pressure);
    let mut tw = (dry_bulb + dp) / 2.0;

    // Newton-Raphson iteration
    for _ in 0..MAX_ITERATIONS {
        let current_enthalpy = calculate_enthalpy(tw, 100.0, pressure);
        let delta_h = current_enthalpy - target_enthalpy;

        if delta_h.abs() < TOLERANCE {
            break;
        }

        // Approximate derivative using enthalpy at slightly different temperature
        const EPSILON: f64 = 0.001;
        let enthalpy_epsilon = calculate_enthalpy(tw + EPSILON, 100.0, pressure);
        let derivative = (enthalpy_epsilon - current_enthalpy) / EPSILON;

        // Prevent division by zero
        if derivative.abs() < 1e-10 {
            break;
        }

        tw -= delta_h / derivative;
    }

    // Clamp to physical bounds
    tw.clamp(dp, dry_bulb)
}

/// Convenience structure for psychrometric calculation inputs.
///
/// This struct provides a unified interface for psychrometric inputs,
/// enabling future extensibility for custom pressure values or alternative
/// input types (e.g., zone air conditions, duct conditions).
///
/// # Note
///
/// Currently, trait implementations for `HourlyWeatherData` use direct
/// function calls rather than `PsychrometricInputs`. This struct is provided
/// for future use cases where a structured input approach is beneficial.
#[derive(Debug, Clone, PartialEq)]
pub struct PsychrometricInputs {
    /// Dry bulb temperature (°C)
    pub temperature: f64,
    /// Relative humidity (0-100)
    pub relative_humidity: f64,
    /// Atmospheric pressure (Pa)
    pub pressure: f64,
}

/// Trait for types that support psychrometric property calculations.
///
/// This trait provides ASHRAE-compliant psychrometric calculations
/// for building energy modeling, including dew point, wet-bulb,
/// humidity ratio, and enthalpy.
///
/// # Example
///
/// ```
/// use fluxion::weather::{HourlyWeatherData, PsychrometricCalculations};
///
/// let weather = HourlyWeatherData::new(25.0, 800.0, 100.0, 900.0, 3.5, 50.0, 0);
/// let dp = weather.dew_point();  // ~13.9°C at 50% RH
/// let h = weather.enthalpy();    // ~50.4 kJ/kg at 50% RH
/// ```
pub trait PsychrometricCalculations {
    /// Calculates dew point temperature (°C) from dry bulb and relative humidity.
    fn dew_point(&self) -> f64;

    /// Calculates wet-bulb temperature (°C).
    fn wet_bulb(&self) -> f64;

    /// Calculates humidity ratio (kg_water_vapor / kg_dry_air).
    fn humidity_ratio(&self) -> f64;

    /// Calculates enthalpy of moist air (kJ/kg).
    fn enthalpy(&self) -> f64;
}

use crate::weather::HourlyWeatherData;

impl PsychrometricCalculations for HourlyWeatherData {
    fn dew_point(&self) -> f64 {
        calculate_dew_point(
            self.dry_bulb_temp,
            self.humidity,
            STANDARD_ATMOSPHERIC_PRESSURE_Pa,
        )
    }

    fn wet_bulb(&self) -> f64 {
        calculate_wet_bulb(
            self.dry_bulb_temp,
            self.humidity,
            STANDARD_ATMOSPHERIC_PRESSURE_Pa,
        )
    }

    fn humidity_ratio(&self) -> f64 {
        calculate_humidity_ratio(
            self.dry_bulb_temp,
            self.humidity,
            STANDARD_ATMOSPHERIC_PRESSURE_Pa,
        )
    }

    fn enthalpy(&self) -> f64 {
        calculate_enthalpy(
            self.dry_bulb_temp,
            self.humidity,
            STANDARD_ATMOSPHERIC_PRESSURE_Pa,
        )
    }
}

/// Extracts psychrometric inputs from hourly weather data.
///
/// # Arguments
///
/// * `weather` - Hourly weather data
///
/// # Returns
///
/// `PsychrometricInputs` structure with temperature, RH, and standard atmospheric pressure
///
/// # Example
///
/// ```
/// use fluxion::weather::{HourlyWeatherData, from_weather_data};
///
/// let weather = HourlyWeatherData::new(25.0, 800.0, 100.0, 900.0, 3.5, 50.0, 0);
/// let inputs = from_weather_data(&weather);
/// assert_eq!(inputs.temperature, 25.0);
/// assert_eq!(inputs.relative_humidity, 50.0);
/// ```
pub fn from_weather_data(weather: &HourlyWeatherData) -> PsychrometricInputs {
    PsychrometricInputs {
        temperature: weather.dry_bulb_temp,
        relative_humidity: weather.humidity,
        pressure: STANDARD_ATMOSPHERIC_PRESSURE_Pa,
    }
}

/// Calculates enthalpy directly from hourly weather data.
///
/// Convenience function that uses the `PsychrometricCalculations` trait
/// to provide enthalpy without requiring explicit trait import.
///
/// # Arguments
///
/// * `weather` - Hourly weather data
///
/// # Returns
///
/// Enthalpy of moist air (kJ/kg)
///
/// # Example
///
/// ```
/// use fluxion::weather::{HourlyWeatherData, enthalpy_from_weather};
///
/// let weather = HourlyWeatherData::new(25.0, 800.0, 100.0, 900.0, 3.5, 50.0, 0);
/// let h = enthalpy_from_weather(&weather);  // ~50.4 kJ/kg at 50% RH
/// ```
pub fn enthalpy_from_weather(weather: &HourlyWeatherData) -> f64 {
    weather.enthalpy()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_saturation_vapor_pressure_reference_values() {
        // ASHRAE reference values (tolerance: ±5 Pa)
        let p_sat_0 = saturation_vapor_pressure(0.0);
        assert!((p_sat_0 - 611.2).abs() < 5.0, "p_sat(0°C) ≈ 611.2 Pa");

        let p_sat_20 = saturation_vapor_pressure(20.0);
        assert!((p_sat_20 - 2339.0).abs() < 5.0, "p_sat(20°C) ≈ 2339 Pa");

        let p_sat_30 = saturation_vapor_pressure(30.0);
        assert!((p_sat_30 - 4246.0).abs() < 5.0, "p_sat(30°C) ≈ 4246 Pa");
    }

    #[test]
    fn test_dew_point_reference_values() {
        // ASHRAE reference values (tolerance: ±0.5°C)
        let dp = calculate_dew_point(25.0, 50.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!((dp - 13.9).abs() < 0.5, "dew_point(25°C, 50%) ≈ 13.9°C");

        let dp = calculate_dew_point(20.0, 80.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!((dp - 16.4).abs() < 0.5, "dew_point(20°C, 80%) ≈ 16.4°C");

        let dp = calculate_dew_point(30.0, 20.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!((dp - 5.0).abs() < 0.5, "dew_point(30°C, 20%) ≈ 5.0°C");
    }

    #[test]
    fn test_dew_point_le_dry_bulb() {
        // Property test: dew point never exceeds dry bulb
        for temp in (-10..=40).step_by(2) {
            for rh in [10, 30, 50, 70, 90] {
                let dry_bulb = temp as f64;
                let dp = calculate_dew_point(dry_bulb, rh as f64, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
                assert!(
                    dp <= dry_bulb + 0.01,
                    "dew_point({dry_bulb}°C, {rh}%) ≤ {dry_bulb}°C"
                );
            }
        }
    }

    #[test]
    fn test_humidity_ratio_reference_values() {
        // Reference values (tolerance: ±1%)
        let omega = calculate_humidity_ratio(25.0, 50.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!(
            (omega - 0.0099).abs() < 0.0001,
            "humidity_ratio(25°C, 50%) ≈ 0.0099 kg/kg"
        );
    }

    #[test]
    fn test_enthalpy_reference_values() {
        // Reference values validated against ASHRAE psychrometric calculations
        // Tolerance: ±1.0 kJ/kg to account for formula variations

        let h = calculate_enthalpy(25.0, 50.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!(
            (h - 50.4).abs() < 1.0,
            "enthalpy(25°C, 50%) ≈ 50.4 kJ/kg, got {}",
            h
        );

        let h = calculate_enthalpy(20.0, 80.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!(
            (h - 49.0).abs() < 1.0,
            "enthalpy(20°C, 80%) ≈ 49.0 kJ/kg, got {}",
            h
        );

        let h = calculate_enthalpy(30.0, 20.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        // Note: Calculated value is ~43.6 kJ/kg using ASHRAE formula
        // Reference may be from different psychrometric chart or approximation
        assert!(
            (h - 43.6).abs() < 1.0,
            "enthalpy(30°C, 20%) ≈ 43.6 kJ/kg, got {}",
            h
        );
    }

    #[test]
    fn test_enthalpy_monotonic_with_temperature() {
        // Property test: enthalpy increases with temperature at fixed RH
        for rh in [10, 30, 50, 70, 90] {
            let mut prev_enthalpy = f64::NEG_INFINITY;
            for temp in (-10..=40).step_by(2) {
                let h =
                    calculate_enthalpy(temp as f64, rh as f64, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
                assert!(
                    h > prev_enthalpy - 0.01,
                    "enthalpy increases with T at {rh}% RH"
                );
                prev_enthalpy = h;
            }
        }
    }

    #[test]
    fn test_enthalpy_monotonic_with_rh() {
        // Property test: enthalpy increases with RH at fixed temperature
        for temp in [0.0, 10.0, 20.0, 30.0] {
            let mut prev_enthalpy = f64::NEG_INFINITY;
            for rh in [10, 30, 50, 70, 90] {
                let h = calculate_enthalpy(temp, rh as f64, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
                assert!(
                    h > prev_enthalpy - 0.01,
                    "enthalpy increases with RH at {temp}°C"
                );
                prev_enthalpy = h;
            }
        }
    }

    #[test]
    fn test_from_weather_data() {
        let weather = HourlyWeatherData::new(25.0, 800.0, 100.0, 900.0, 3.5, 50.0, 0);
        let inputs = from_weather_data(&weather);

        assert_eq!(inputs.temperature, 25.0);
        assert_eq!(inputs.relative_humidity, 50.0);
        assert_eq!(inputs.pressure, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
    }

    #[test]
    fn test_enthalpy_from_weather_matches_trait() {
        let weather = HourlyWeatherData::new(25.0, 800.0, 100.0, 900.0, 3.5, 50.0, 0);
        let h1 = enthalpy_from_weather(&weather);
        let h2 = weather.enthalpy();

        assert!(
            (h1 - h2).abs() < 0.001,
            "helper function matches trait method"
        );
    }

    #[test]
    fn test_trait_methods_match_functions() {
        // Test 25°C/50% RH
        let weather1 = HourlyWeatherData::new(25.0, 800.0, 100.0, 900.0, 3.5, 50.0, 0);
        assert!(
            (weather1.dew_point()
                - calculate_dew_point(25.0, 50.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa))
            .abs()
                < 0.01
        );
        assert!(
            (weather1.wet_bulb()
                - calculate_wet_bulb(25.0, 50.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa))
            .abs()
                < 0.01
        );
        assert!(
            (weather1.humidity_ratio()
                - calculate_humidity_ratio(25.0, 50.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa))
            .abs()
                < 0.0001
        );
        assert!(
            (weather1.enthalpy()
                - calculate_enthalpy(25.0, 50.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa))
            .abs()
                < 0.01
        );

        // Test 20°C/80% RH
        let weather2 = HourlyWeatherData::new(20.0, 800.0, 100.0, 900.0, 3.5, 80.0, 0);
        assert!(
            (weather2.dew_point()
                - calculate_dew_point(20.0, 80.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa))
            .abs()
                < 0.01
        );
        assert!(
            (weather2.wet_bulb()
                - calculate_wet_bulb(20.0, 80.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa))
            .abs()
                < 0.01
        );
        assert!(
            (weather2.humidity_ratio()
                - calculate_humidity_ratio(20.0, 80.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa))
            .abs()
                < 0.0001
        );
        assert!(
            (weather2.enthalpy()
                - calculate_enthalpy(20.0, 80.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa))
            .abs()
                < 0.01
        );

        // Test 30°C/20% RH
        let weather3 = HourlyWeatherData::new(30.0, 800.0, 100.0, 900.0, 3.5, 20.0, 0);
        assert!(
            (weather3.dew_point()
                - calculate_dew_point(30.0, 20.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa))
            .abs()
                < 0.01
        );
        assert!(
            (weather3.wet_bulb()
                - calculate_wet_bulb(30.0, 20.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa))
            .abs()
                < 0.01
        );
        assert!(
            (weather3.humidity_ratio()
                - calculate_humidity_ratio(30.0, 20.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa))
            .abs()
                < 0.0001
        );
        assert!(
            (weather3.enthalpy()
                - calculate_enthalpy(30.0, 20.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa))
            .abs()
                < 0.01
        );
    }

    #[test]
    fn test_wet_bulb_convergence() {
        // Verify convergence across full T/RH range
        for temp in (-10..=40).step_by(10) {
            for rh in [10, 30, 50, 70, 90] {
                let wb =
                    calculate_wet_bulb(temp as f64, rh as f64, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
                let dp =
                    calculate_dew_point(temp as f64, rh as f64, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
                let dry_bulb = temp as f64;

                // Wet-bulb should be between dew point and dry bulb
                assert!(wb >= dp - 0.01, "wet_bulb ≥ dew_point");
                assert!(wb <= dry_bulb + 0.01, "wet_bulb ≤ dry_bulb");
                assert!(wb.is_finite(), "wet_bulb is finite");
            }
        }
    }

    // === FINE GRID TESTS (130 points each: 26 temps × 5 RH levels) ===

    #[test]
    fn test_dew_point_fine_grid() {
        // Fine grid test: 26 temperatures × 5 RH levels = 130 test points
        for t in (-10i32..=40).step_by(2) {
            for rh in [10.0, 30.0, 50.0, 70.0, 90.0] {
                let dp = calculate_dew_point(t as f64, rh, STANDARD_ATMOSPHERIC_PRESSURE_Pa);

                // Property: dew point must be ≤ dry bulb
                assert!(
                    dp <= t as f64 + 0.01,
                    "Dew point {} exceeded dry bulb {} at RH {}",
                    dp,
                    t,
                    rh
                );

                // Reasonable range check (not strict ASHRAE reference, but sanity check)
                // At RH=10%, dew point is much lower than dry bulb
                // At RH=90%, dew point approaches dry bulb
                let max_dp = t as f64 - (10.0 - rh) * 0.2; // Heuristic
                assert!(
                    dp <= max_dp + 2.0,
                    "Dew point {} outside reasonable range at {}°C, {}% RH",
                    dp,
                    t,
                    rh
                );

                assert!(
                    dp.is_finite(),
                    "Dew point is infinite at {}°C, {}% RH",
                    t,
                    rh
                );
                assert!(!dp.is_nan(), "Dew point is NaN at {}°C, {}% RH", t, rh);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Property-Based Tests (proptest)
    // Issue #1062: Property-based testing for core math & parsers
    //
    // These tests verify physical invariants across random inputs:
    // - Temperature range: -50°C to 60°C (building operating range)
    // - Pressure range: 70-110 kPa (altitude range)
    // - Humidity range: 0-100% RH
    // -------------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10_000))]

        #[test]
        fn prop_saturation_vapor_pressure_bounds(temperature in -50.0_f64..60.0) {
            let p_sat = saturation_vapor_pressure(temperature);
            prop_assert!(p_sat > 0.0, "Saturation pressure must be positive");
            prop_assert!(p_sat.is_finite(), "Saturation pressure must be finite");
            // At -50°C, p_sat ≈ 1 Pa; at 60°C, p_sat ≈ 19 kPa
            prop_assert!(p_sat < 30_000.0, "Saturation pressure unreasonable at {}", temperature);
        }

        #[test]
        fn prop_dew_point_never_exceeds_dry_bulb(
            dry_bulb in -50.0_f64..60.0,
            relative_humidity in 0.0_f64..100.0,
            pressure_pa in 70_000.0_f64..110_000.0,
        ) {
            let dp = calculate_dew_point(dry_bulb, relative_humidity, pressure_pa);
            prop_assert!(dp <= dry_bulb + 1e-6, "Dew point {} exceeds dry bulb {}", dp, dry_bulb);
            prop_assert!(dp.is_finite(), "Dew point must be finite");
        }

        #[test]
        fn prop_wet_bulb_between_dew_point_and_dry_bulb(
            dry_bulb in -50.0_f64..60.0,
            relative_humidity in 0.0_f64..100.0,
            pressure_pa in 70_000.0_f64..110_000.0,
        ) {
            let wb = calculate_wet_bulb(dry_bulb, relative_humidity, pressure_pa);
            let dp = calculate_dew_point(dry_bulb, relative_humidity, pressure_pa);
            prop_assert!(wb >= dp - 1e-4, "Wet bulb {} below dew point {}", wb, dp);
            prop_assert!(wb <= dry_bulb + 1e-4, "Wet bulb {} exceeds dry bulb {}", wb, dry_bulb);
            prop_assert!(wb.is_finite(), "Wet bulb must be finite");
        }

        #[test]
        fn prop_humidity_ratio_positive_and_bounded(
            dry_bulb in -50.0_f64..60.0,
            relative_humidity in 0.0_f64..100.0,
            pressure_pa in 70_000.0_f64..110_000.0,
        ) {
            let omega = calculate_humidity_ratio(dry_bulb, relative_humidity, pressure_pa);
            prop_assert!(omega >= 0.0, "Humidity ratio must be non-negative");
            prop_assert!(omega < 0.30, "Humidity ratio {} unreasonably high", omega);
            prop_assert!(omega.is_finite(), "Humidity ratio must be finite");
        }

        #[test]
        fn prop_enthalpy_increases_with_temperature(
            temp1 in -50.0_f64..60.0,
            temp2 in -50.0_f64..60.0,
            relative_humidity in 0.0_f64..100.0,
            pressure_pa in 70_000.0_f64..110_000.0,
        ) {
            let h1 = calculate_enthalpy(temp1, relative_humidity, pressure_pa);
            let h2 = calculate_enthalpy(temp2, relative_humidity, pressure_pa);
            if temp2 > temp1 {
                prop_assert!(h2 >= h1 - 1e-10, "Enthalpy must increase with temperature");
            }
        }

        #[test]
        fn prop_enthalpy_increases_with_humidity(
            temperature in -50.0_f64..60.0,
            rh1 in 0.0_f64..100.0,
            rh2 in 0.0_f64..100.0,
            pressure_pa in 70_000.0_f64..110_000.0,
        ) {
            let h1 = calculate_enthalpy(temperature, rh1, pressure_pa);
            let h2 = calculate_enthalpy(temperature, rh2, pressure_pa);
            if rh2 > rh1 {
                prop_assert!(h2 >= h1 - 1e-10, "Enthalpy must increase with RH");
            }
        }

        #[test]
        fn prop_saturation_pressure_monotonically_increasing(temperature in -50.0_f64..60.0) {
            let p_sat = saturation_vapor_pressure(temperature);
            let p_sat_higher = saturation_vapor_pressure(temperature + 0.1);
            prop_assert!(p_sat_higher > p_sat, "Saturation pressure must increase with temperature");
        }
    }

    #[test]
    fn test_wet_bulb_fine_grid() {
        // Fine grid test: 26 temperatures × 5 RH levels = 130 test points
        for t in (-10i32..=40).step_by(2) {
            for rh in [10.0, 30.0, 50.0, 70.0, 90.0] {
                let wb = calculate_wet_bulb(t as f64, rh, STANDARD_ATMOSPHERIC_PRESSURE_Pa);

                // Property: wet bulb is between dew point and dry bulb
                let dp = calculate_dew_point(t as f64, rh, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
                assert!(
                    wb >= dp - 0.1 && wb <= t as f64 + 0.1,
                    "Wet bulb {} outside [{}, {}] range at {}°C, {}% RH",
                    wb,
                    dp,
                    t,
                    t,
                    rh
                );

                // Reasonable range check
                assert!(
                    wb.is_finite(),
                    "Wet bulb is infinite at {}°C, {}% RH",
                    t,
                    rh
                );
                assert!(!wb.is_nan(), "Wet bulb is NaN at {}°C, {}% RH", t, rh);
            }
        }
    }

    #[test]
    fn test_enthalpy_fine_grid() {
        // Fine grid test: 26 temperatures × 5 RH levels = 130 test points
        for t in (-10i32..=40).step_by(2) {
            for rh in [10.0, 30.0, 50.0, 70.0, 90.0] {
                let h = calculate_enthalpy(t as f64, rh, STANDARD_ATMOSPHERIC_PRESSURE_Pa);

                // Reasonable enthalpy range for building HVAC
                // Cold/dry: ~0 kJ/kg, Hot/humid: ~160 kJ/kg
                assert!(
                    h > -10.0 && h < 200.0,
                    "Enthalpy {} outside reasonable range at {}°C, {}% RH",
                    h,
                    t,
                    rh
                );

                assert!(h.is_finite(), "Enthalpy is infinite at {}°C, {}% RH", t, rh);
                assert!(!h.is_nan(), "Enthalpy is NaN at {}°C, {}% RH", t, rh);
            }
        }
    }
}
