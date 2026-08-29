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
/// use fluxion_core::weather::psychrometrics::saturation_vapor_pressure;
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
/// use fluxion_core::weather::psychrometrics::calculate_dew_point;
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
/// use fluxion_core::weather::psychrometrics::calculate_humidity_ratio;
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
/// use fluxion_core::weather::psychrometrics::calculate_enthalpy;
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

/// Calculates the partial pressure of water vapor from humidity ratio and total pressure.
///
/// This is the algebraic inverse of [`calculate_humidity_ratio`]. Given a humidity ratio
/// `W` and total pressure `P`, it returns the partial vapor pressure `p_w` that produced
/// `W` (per ASHRAE Handbook of Fundamentals, Chapter 1, Eq. 22 rearranged).
///
/// # Formula
///
/// ```text
/// p_w = W · P / (W + 0.62198)
/// ```
///
/// Where:
/// - `p_w` = partial pressure of water vapor (Pa)
/// - `W` = humidity ratio (kg_water_vapor / kg_dry_air)
/// - `P` = total atmospheric pressure (Pa)
/// - `0.62198` = ratio of molar masses M_w / M_da (water / dry air)
///
/// # Arguments
///
/// * `humidity_ratio` - Humidity ratio W (kg_water_vapor / kg_dry_air)
/// * `pressure` - Total atmospheric pressure (Pa)
///
/// # Returns
///
/// Partial pressure of water vapor in Pa
///
/// # Notes
///
/// - At saturation (RH = 100%), `partial_vapor_pressure(W_sat, P)` returns `p_ws(T)`,
///   the saturation vapor pressure at the dry-bulb temperature.
/// - `partial_vapor_pressure` is the algebraic inverse of `calculate_humidity_ratio`:
///   `partial_vapor_pressure(calculate_humidity_ratio(T, rh, P), P) ≈ (rh/100) · p_ws(T)`.
/// - This routine is dependency-free (only `std` math) and lives in `fluxion-core`
///   to respect the cycle-breaking rule (#1255, #1349, #1441).
///
/// # Example
///
/// ```
/// use fluxion_core::weather::psychrometrics::{calculate_humidity_ratio, partial_vapor_pressure};
///
/// let p = 101325.0_f64;
/// let w = calculate_humidity_ratio(20.0, 50.0, p); // ≈ 0.00726 kg/kg
/// let pw = partial_vapor_pressure(w, p);            // ≈ 1170 Pa
/// assert!((pw - 1170.0).abs() < 5.0);
/// ```
pub fn partial_vapor_pressure(humidity_ratio: f64, pressure: f64) -> f64 {
    const RATIO_MW: f64 = 0.62198; // H2O / dry_air molar mass ratio
    humidity_ratio * pressure / (humidity_ratio + RATIO_MW)
}

/// Calculates moist-air density at given dry-bulb temperature, humidity ratio, and pressure.
///
/// Implements the ASHRAE Handbook of Fundamentals, Chapter 1 form of the ideal-gas
/// moist-air density equation. The formula combines Dalton's law of partial pressures
/// with the ideal-gas law for dry air and water vapor.
///
/// # Formula (ASHRAE HoF Ch.1 Eq. 28, rearranged)
///
/// ```text
/// ρ = P · (1 + W) / (R_da · T_K · (1 + 1.6078·W))
/// ```
///
/// Equivalently, using partial pressures:
///
/// ```text
/// ρ = (P − p_w) · M_da / (R_u · T_K)  +  p_w · M_w / (R_u · T_K)
/// ```
///
/// Where:
/// - `ρ` = moist-air density (kg/m³)
/// - `P` = total atmospheric pressure (Pa)
/// - `W` = humidity ratio (kg_water_vapor / kg_dry_air)
/// - `R_da` = specific gas constant for dry air = 287.055 J/(kg·K)
/// - `T_K` = absolute temperature (K) = `T_dry_bulb + 273.15`
/// - `1.6078` = `R_v / R_da` = `M_da / M_w` (ratio of specific gas constants / molar masses)
/// - `R_u` = 8.314 J/(mol·K) universal gas constant
/// - `M_da` = 28.965 g/mol, `M_w` = 18.015 g/mol
///
/// # Arguments
///
/// * `dry_bulb` - Dry-bulb temperature in °C
/// * `humidity_ratio` - Humidity ratio W (kg_water_vapor / kg_dry_air)
/// * `pressure` - Total atmospheric pressure in Pa
///
/// # Returns
///
/// Moist-air density in kg/m³
///
/// # Reference Values (ASHRAE HoF 2021 Ch.1, 101.325 kPa)
///
/// | T (°C) | RH (%) | ρ (kg/m³) |
/// |--------|--------|-----------|
/// | 0      | 50     | 1.290     |
/// | 20     | 50     | 1.199     |
/// | 20     | 100    | 1.194     |
/// | 30     | 50     | 1.155     |
/// | 40     | 50     | 1.112     |
///
/// Source: ASHRAE Handbook of Fundamentals 2021, Chapter 1, Table 2
/// (Thermodynamic Properties of Moist Air at Standard Atmospheric Pressure).
///
/// # Example
///
/// ```
/// use fluxion_core::weather::psychrometrics::{calculate_humidity_ratio, moist_air_density};
///
/// let p = 101325.0_f64;
/// let w = calculate_humidity_ratio(20.0, 50.0, p);
/// let rho = moist_air_density(20.0, w, p); // ≈ 1.199 kg/m³
/// assert!((rho - 1.199).abs() < 0.01);
/// ```
pub fn moist_air_density(dry_bulb: f64, humidity_ratio: f64, pressure: f64) -> f64 {
    const R_DA: f64 = 287.055; // J/(kg·K) — specific gas constant for dry air (ASHRAE)
    const INV_RATIO_MW: f64 = 1.6078; // R_v / R_da = M_da / M_w

    let t_kelvin = dry_bulb + 273.15;
    pressure * (1.0 + humidity_ratio) / (R_DA * t_kelvin * (1.0 + INV_RATIO_MW * humidity_ratio))
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
/// use fluxion_core::weather::psychrometrics::calculate_wet_bulb;
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
/// use fluxion_core::weather::{HourlyWeatherData, PsychrometricCalculations};
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
/// use fluxion_core::weather::{HourlyWeatherData, from_weather_data};
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
/// use fluxion_core::weather::{HourlyWeatherData, enthalpy_from_weather};
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

    // =====================================================================
    // Issue #1760 — psychrometrics library (ASHRAE Ch.1, SI units)
    //
    // Round-trip and ASHRAE-reference tests for the two new functions:
    //   - `moist_air_density`     (ASHRAE HoF Ch.1 Eq. 28)
    //   - `partial_vapor_pressure` (inverse of humidity ratio, ASHRAE Ch.1 Eq. 22)
    //
    // Reference data sourced from:
    //   - ASHRAE Handbook of Fundamentals 2021, Chapter 1, Table 2
    //     (Thermodynamic Properties of Moist Air at Standard Atmospheric
    //     Pressure 101.325 kPa). Specific volume is reported per kg of
    //     dry air; density is derived as ρ = (1 + W) / v.
    //   - ASHRAE HoF 2021 Ch.1, Table 1 (Saturation pressure of water vapor).
    //   - NIST Webbook spot-checks for moist-air density at sea-level standard
    //     conditions (https://webbook.nist.gov/chemistry/fluid/).
    //
    // Tolerance: 1 % relative error against reference values, per the
    // acceptance criterion for issue #1760.
    // =====================================================================

    /// Test points from ASHRAE HoF 2021 Ch.1 Table 2 (101.325 kPa).
    /// Density is derived from published specific volume: ρ = (1 + W) / v.
    const ASHRAE_DENSITY_REFERENCES: &[(f64, f64, f64)] = &[
        // (T_dry_bulb_°C, RH_%, expected_ρ_kg_m3)
        (0.0, 50.0, 1.290),   // ASHRAE HoF 2021 Ch.1 Table 2, 0°C 50% RH
        (10.0, 50.0, 1.244),  // ASHRAE HoF 2021 Ch.1 Table 2, 10°C 50% RH
        (20.0, 50.0, 1.199),  // ASHRAE HoF 2021 Ch.1 Table 2, 20°C 50% RH
        (20.0, 100.0, 1.194), // ASHRAE HoF 2021 Ch.1 Table 2, 20°C 100% RH
        (25.0, 50.0, 1.177),  // ASHRAE HoF 2021 Ch.1 Table 2, 25°C 50% RH
        (30.0, 50.0, 1.155),  // ASHRAE HoF 2021 Ch.1 Table 2, 30°C 50% RH
        (40.0, 50.0, 1.112),  // ASHRAE HoF 2021 Ch.1 Table 2, 40°C 50% RH
    ];

    #[test]
    fn test_moist_air_density_ashrae_reference_values() {
        // Acceptance criterion: 1% tolerance against ASHRAE HoF 2021 Ch.1 Table 2.
        for &(t_c, rh, rho_ref) in ASHRAE_DENSITY_REFERENCES {
            let w = calculate_humidity_ratio(t_c, rh, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
            let rho = moist_air_density(t_c, w, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
            let rel_err = ((rho - rho_ref) / rho_ref).abs();
            assert!(
                rel_err < 0.01,
                "moist_air_density({}, {}%, 101325 Pa) = {} kg/m³, ASHRAE ref = {} \
                 (rel_err = {:.4}%, must be < 1%)",
                t_c,
                rh,
                rho,
                rho_ref,
                rel_err * 100.0
            );
            assert!(rho.is_finite(), "density is not finite");
            assert!(rho > 0.0, "density must be positive");
        }
    }

    #[test]
    fn test_moist_air_density_consistency_with_humidity_ratio() {
        // ρ should depend on T, W, and P only — not on RH directly.
        // Cross-check: computing ρ via two (T, RH, P) inputs that yield the
        // same W must produce identical density.
        let cases = [(20.0, 30.0), (20.0, 50.0), (20.0, 80.0), (30.0, 50.0)];
        for (t1, rh1) in cases {
            let w1 = calculate_humidity_ratio(t1, rh1, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
            let rho1 = moist_air_density(t1, w1, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
            // Now feed the same W back at a different T — density must change with T
            // (ideal gas law), but the (W, P) component is preserved.
            let t2 = t1 + 10.0;
            let rho2 = moist_air_density(t2, w1, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
            // Ideal-gas ratio check: rho2 / rho1 = T_K1 / T_K2 (since P, W constant)
            let t_k1 = t1 + 273.15;
            let t_k2 = t2 + 273.15;
            let expected_ratio = t_k1 / t_k2;
            let actual_ratio = rho2 / rho1;
            assert!(
                (actual_ratio - expected_ratio).abs() < 1e-10,
                "rho ratio does not match ideal-gas T scaling at T={}, W={}",
                t1,
                w1
            );
        }
    }

    #[test]
    fn test_moist_air_density_ideal_gas_law_limit() {
        // At W = 0 (no water vapor), moist-air density must reduce to the
        // dry-air ideal-gas law: ρ = P / (R_da · T_K).
        for t_c in [-20.0, 0.0, 20.0, 40.0, 60.0] {
            let t_k = t_c + 273.15;
            let rho_dry = STANDARD_ATMOSPHERIC_PRESSURE_Pa / (287.055 * t_k);
            let rho_moist = moist_air_density(t_c, 0.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
            assert!(
                (rho_moist - rho_dry).abs() / rho_dry < 1e-12,
                "moist_air_density at W=0 must equal dry-air density at T={}°C",
                t_c
            );
        }
    }

    #[test]
    fn test_moist_air_density_altitude_effect() {
        // Higher altitude → lower P → lower ρ at same (T, W).
        // Test at 1500 m elevation (~ Denver, CO): P ≈ 84.0 kPa.
        let p_denver = 84000.0_f64; // Pa, representative of Denver elevation
        let t_sea = 20.0_f64;
        let rh = 50.0_f64;
        let w = calculate_humidity_ratio(t_sea, rh, STANDARD_ATMOSPHERIC_PRESSURE_Pa);

        let rho_sea = moist_air_density(t_sea, w, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        let rho_denver = moist_air_density(t_sea, w, p_denver);

        // At lower P, density should scale roughly linearly: rho_denver / rho_sea ≈ P_denver / P_sea
        let expected_ratio = p_denver / STANDARD_ATMOSPHERIC_PRESSURE_Pa;
        let actual_ratio = rho_denver / rho_sea;
        assert!(
            (actual_ratio - expected_ratio).abs() < 1e-10,
            "Density must scale linearly with P at fixed (T, W)"
        );
        assert!(rho_denver < rho_sea, "Higher altitude → lower density");
    }

    #[test]
    fn test_moist_air_density_fine_grid_bounds() {
        // Sanity check: density must stay within reasonable bounds across the
        // building HVAC operating envelope. Reference: ASHRAE Handbook Ch.1,
        // building HVAC typical range is 0.9–1.4 kg/m³ at sea level.
        for t_c in (-20i32..=50).step_by(5) {
            for rh in [10.0, 30.0, 50.0, 70.0, 100.0] {
                let w = calculate_humidity_ratio(t_c as f64, rh, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
                let rho = moist_air_density(t_c as f64, w, STANDARD_ATMOSPHERIC_PRESSURE_Pa);

                assert!(
                    rho > 0.9 && rho < 1.5,
                    "Density {} kg/m³ outside building HVAC range at {}°C, {}% RH",
                    rho,
                    t_c,
                    rh
                );
                assert!(rho.is_finite(), "Density non-finite");
                assert!(!rho.is_nan(), "Density NaN");
            }
        }
    }

    #[test]
    fn test_partial_vapor_pressure_at_saturation() {
        // At saturation (RH = 100%), the partial vapor pressure must equal
        // the saturation vapor pressure at the dry-bulb temperature.
        // This cross-validates `partial_vapor_pressure` against `saturation_vapor_pressure`
        // and against ASHRAE HoF 2021 Ch.1 Table 1.
        for t_c in [-20.0, -10.0, 0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0] {
            let w = calculate_humidity_ratio(t_c, 100.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
            let pw = partial_vapor_pressure(w, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
            let p_ws = saturation_vapor_pressure(t_c);
            let rel_err = ((pw - p_ws) / p_ws).abs();
            assert!(
                rel_err < 1e-10,
                "At RH=100%, p_w must equal p_ws(T={}°C); got p_w={}, p_ws={}",
                t_c,
                pw,
                p_ws
            );
        }
    }

    #[test]
    fn test_partial_vapor_pressure_round_trip() {
        // p_w → W → p_w must round-trip exactly (algebraic inverse).
        // Reference humidity ratios taken from ASHRAE HoF 2021 Ch.1 Table 2.
        let test_points = [
            (20.0, 50.0, 0.00726_f64), // 20°C 50% RH
            (30.0, 50.0, 0.01321_f64), // 30°C 50% RH
            (25.0, 80.0, 0.01607_f64), // 25°C 80% RH
            (10.0, 30.0, 0.00228_f64), // 10°C 30% RH
        ];
        for (t_c, rh, _w_ref) in test_points {
            // Compute W from (T, RH, P), then recover p_w, then check it
            // matches (RH/100) * p_ws(T) — i.e., it reproduces the input.
            let w = calculate_humidity_ratio(t_c, rh, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
            let pw = partial_vapor_pressure(w, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
            let p_ws = saturation_vapor_pressure(t_c);
            let p_w_expected = (rh / 100.0) * p_ws;
            let rel_err = ((pw - p_w_expected) / p_w_expected).abs();
            assert!(
                rel_err < 1e-12,
                "Partial vapor pressure round-trip failed at T={}°C, RH={}%: \
                 p_w={}, expected={} (rel_err = {})",
                t_c,
                rh,
                pw,
                p_w_expected,
                rel_err
            );
        }
    }

    #[test]
    fn test_partial_vapor_pressure_ashrae_reference_values() {
        // Validate p_w against ASHRAE HoF 2021 Ch.1 Table 1 (saturation pressure).
        // At saturation (RH = 100%), p_w = p_ws(T), which is tabulated.
        // Tolerance: 1% relative error (acceptance criterion for #1760).
        // Sub-zero values use the ASHRAE Hyland-Wexler ice equation (Eq. 6),
        // which is also the formula implemented in `saturation_vapor_pressure`,
        // so agreement is exact to the formula — but we use published table
        // values here to lock the behavior.
        let ashrae_p_ws_ref: &[(f64, f64)] = &[
            // (T_°C, p_ws_Pa from ASHRAE HoF 2021 Ch.1 Table 1)
            (-20.0, 103.24),
            (-10.0, 260.64),
            (0.0, 611.21),
            (5.0, 872.6),
            (10.0, 1228.5),
            (15.0, 1705.9),
            (20.0, 2339.0),
            (25.0, 3170.3),
            (30.0, 4246.0),
            (35.0, 5629.7),
            (40.0, 7386.6),
            (50.0, 12355.0),
        ];
        for &(t_c, p_ws_ref) in ashrae_p_ws_ref {
            // Saturated humidity ratio → partial vapor pressure
            let w_sat = calculate_humidity_ratio(t_c, 100.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
            let p_w_calc = partial_vapor_pressure(w_sat, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
            let rel_err = ((p_w_calc - p_ws_ref) / p_ws_ref).abs();
            assert!(
                rel_err < 0.01,
                "p_w at saturation must match ASHRAE HoF 2021 Ch.1 Table 1 \
                 within 1% at T={}°C: got {} Pa, ref = {} Pa (rel_err = {:.3}%)",
                t_c,
                p_w_calc,
                p_ws_ref,
                rel_err * 100.0
            );
        }
    }

    #[test]
    fn test_partial_vapor_pressure_monotonic_with_humidity_ratio() {
        // p_w must increase monotonically with W (at fixed P) and must equal 0 at W = 0.
        let p_pa = STANDARD_ATMOSPHERIC_PRESSURE_Pa;
        let mut prev = partial_vapor_pressure(0.0, p_pa);
        assert_eq!(prev, 0.0, "p_w must be 0 at W = 0");
        for i in 1..=20 {
            let w = i as f64 * 0.005; // 0.005 .. 0.100
            let pw = partial_vapor_pressure(w, p_pa);
            assert!(
                pw > prev,
                "p_w must increase with W: pw({}) = {} ≤ prev = {}",
                w,
                pw,
                prev
            );
            // p_w must be less than total pressure (water vapor is a fraction of P)
            assert!(pw < p_pa, "p_w must be < P at W={}", w);
            prev = pw;
        }
    }

    #[test]
    fn test_partial_vapor_pressure_altitude_effect() {
        // At constant W, lower pressure → lower p_w (linear scaling).
        let w = 0.01_f64; // kg/kg
        let pw_sea = partial_vapor_pressure(w, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        let p_denver = 84000.0_f64;
        let pw_denver = partial_vapor_pressure(w, p_denver);
        let expected_ratio = p_denver / STANDARD_ATMOSPHERIC_PRESSURE_Pa;
        let actual_ratio = pw_denver / pw_sea;
        assert!(
            (actual_ratio - expected_ratio).abs() < 1e-10,
            "p_w must scale linearly with P at fixed W"
        );
    }

    // === Property-based tests for the new functions ===

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2_000))]

        #[test]
        fn prop_moist_air_density_positive_and_finite(
            t_c in -50.0_f64..60.0,
            w in 0.0_f64..0.05,
            p_pa in 70_000.0_f64..110_000.0,
        ) {
            let rho = moist_air_density(t_c, w, p_pa);
            prop_assert!(rho > 0.0, "Density must be positive: {}", rho);
            prop_assert!(rho.is_finite(), "Density must be finite: {}", rho);
            // Physical envelope for the input domain
            // (-50°C..60°C, 70..110 kPa, W ≤ 0.05):
            //   T=-50, P=110 kPa, W=0   -> ρ ≈ 1.72 kg/m³ (upper bound)
            //   T=60,  P=70  kPa, W=0   -> ρ ≈ 0.73 kg/m³ (lower bound)
            // Use 0.6..1.8 to give margin while still catching gross errors.
            prop_assert!(
                rho > 0.6 && rho < 1.8,
                "Density {} out of physical envelope",
                rho
            );
        }

        #[test]
        fn prop_moist_air_density_increases_with_pressure(
            t_c in -30.0_f64..50.0,
            w in 0.0_f64..0.03,
            p1 in 70_000.0_f64..110_000.0,
            p2 in 70_000.0_f64..110_000.0,
        ) {
            let rho1 = moist_air_density(t_c, w, p1);
            let rho2 = moist_air_density(t_c, w, p2);
            if p2 > p1 {
                prop_assert!(rho2 > rho1, "Density must increase with pressure");
            }
        }

        #[test]
        fn prop_moist_air_density_decreases_with_temperature(
            t1 in -30.0_f64..50.0,
            t2 in -30.0_f64..50.0,
            w in 0.0_f64..0.03,
            p_pa in 70_000.0_f64..110_000.0,
        ) {
            let rho1 = moist_air_density(t1, w, p_pa);
            let rho2 = moist_air_density(t2, w, p_pa);
            if t2 > t1 {
                prop_assert!(rho2 < rho1, "Density must decrease with temperature");
            }
        }

        #[test]
        fn prop_partial_vapor_pressure_in_range(
            w in 0.0_f64..0.05,
            p_pa in 70_000.0_f64..110_000.0,
        ) {
            let pw = partial_vapor_pressure(w, p_pa);
            prop_assert!(pw >= 0.0, "p_w must be non-negative: {}", pw);
            prop_assert!(pw < p_pa, "p_w must be less than total pressure");
            prop_assert!(pw.is_finite(), "p_w must be finite");
        }

        #[test]
        fn prop_partial_vapor_pressure_round_trip_inverse(
            t_c in -20.0_f64..50.0,
            rh in 1.0_f64..99.0,
            p_pa in 70_000.0_f64..110_000.0,
        ) {
            // Forward: T, RH → W → p_w
            let w = calculate_humidity_ratio(t_c, rh, p_pa);
            let pw = partial_vapor_pressure(w, p_pa);
            // Expected: p_w = (rh/100) * p_ws(t_c)
            let p_ws = saturation_vapor_pressure(t_c);
            let pw_expected = (rh / 100.0) * p_ws;
            let rel_err = ((pw - pw_expected) / pw_expected.max(1.0)).abs();
            prop_assert!(rel_err < 1e-10, "Round-trip failed: rel_err = {}", rel_err);
        }
    }

    #[test]
    fn test_saturation_vapor_pressure_freezing_point() {
        let p_pos = saturation_vapor_pressure(0.0);
        let p_neg = saturation_vapor_pressure(-0.001);
        assert!(p_pos > 600.0 && p_pos < 620.0);
        assert!(p_neg > 600.0 && p_neg < 620.0);
    }

    #[test]
    fn test_saturation_vapor_pressure_extreme_cold() {
        let p_neg40 = saturation_vapor_pressure(-40.0);
        assert!(p_neg40 > 0.0 && p_neg40 < 500.0);
        let p_neg50 = saturation_vapor_pressure(-50.0);
        assert!(p_neg50 > 0.0 && p_neg50 < 200.0);
    }

    #[test]
    fn test_saturation_vapor_pressure_extreme_hot() {
        let p_60 = saturation_vapor_pressure(60.0);
        assert!(p_60 > 10_000.0 && p_60 < 30_000.0);
        let p_100 = saturation_vapor_pressure(100.0);
        assert!(p_100 > 80_000.0 && p_100 < 200_000.0);
    }

    #[test]
    fn test_dew_point_at_saturation() {
        let dp = calculate_dew_point(25.0, 100.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!((dp - 25.0).abs() < 0.01, "At RH=100%, dew point must equal dry bulb");
    }

    #[test]
    fn test_dew_point_at_zero_rh() {
        let dp = calculate_dew_point(25.0, 0.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!(dp < -30.0, "At RH=0%, dew point must be very low");
    }

    #[test]
    fn test_humidity_ratio_at_saturation() {
        let w = calculate_humidity_ratio(25.0, 100.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!(w > 0.01 && w < 0.03);
        assert!(w.is_finite());
    }

    #[test]
    fn test_humidity_ratio_at_zero_rh() {
        let w = calculate_humidity_ratio(25.0, 0.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert_eq!(w, 0.0);
    }

    #[test]
    fn test_humidity_ratio_at_extreme_temp() {
        let w_hot = calculate_humidity_ratio(50.0, 80.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!(w_hot > 0.05 && w_hot < 0.15);
        let w_cold = calculate_humidity_ratio(-20.0, 80.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!(w_cold > 0.0 && w_cold < 0.005);
    }

    #[test]
    fn test_wet_bulb_at_saturation() {
        let wb = calculate_wet_bulb(25.0, 100.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!((wb - 25.0).abs() < 0.5);
    }

    #[test]
    fn test_wet_bulb_at_zero_rh() {
        let wb = calculate_wet_bulb(30.0, 0.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!(wb.is_finite());
        assert!(wb <= 30.0);
    }

    #[test]
    fn test_psychrometric_inputs_struct() {
        let inputs = PsychrometricInputs {
            temperature: 25.0,
            relative_humidity: 60.0,
            pressure: STANDARD_ATMOSPHERIC_PRESSURE_Pa,
        };
        assert_eq!(inputs.temperature, 25.0);
        assert_eq!(inputs.relative_humidity, 60.0);
        assert_eq!(inputs.pressure, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
    }

    #[test]
    fn test_psychrometric_inputs_debug() {
        let inputs = PsychrometricInputs {
            temperature: 20.0,
            relative_humidity: 50.0,
            pressure: STANDARD_ATMOSPHERIC_PRESSURE_Pa,
        };
        let d = format!("{:?}", inputs);
        assert!(d.contains("20"));
        assert!(d.contains("50"));
    }

    #[test]
    fn test_psychrometric_inputs_clone() {
        let inputs = PsychrometricInputs {
            temperature: 20.0,
            relative_humidity: 50.0,
            pressure: 101325.0,
        };
        let cloned = inputs.clone();
        assert_eq!(cloned.temperature, inputs.temperature);
        assert_eq!(cloned.relative_humidity, inputs.relative_humidity);
    }

    #[test]
    fn test_psychrometric_inputs_partialeq() {
        let i1 = PsychrometricInputs { temperature: 20.0, relative_humidity: 50.0, pressure: 101325.0 };
        let i2 = PsychrometricInputs { temperature: 20.0, relative_humidity: 50.0, pressure: 101325.0 };
        let i3 = PsychrometricInputs { temperature: 21.0, relative_humidity: 50.0, pressure: 101325.0 };
        assert_eq!(i1, i2);
        assert_ne!(i1, i3);
    }

    #[test]
    fn test_psychrometric_calculations_trait_dry_bulb_zero() {
        let weather = HourlyWeatherData::new(0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0);
        assert!(weather.dew_point().is_finite());
        assert!(weather.wet_bulb().is_finite());
        assert!(weather.humidity_ratio().is_finite());
        assert!(weather.enthalpy().is_finite());
    }

    #[test]
    fn test_psychrometric_calculations_trait_high_humidity() {
        let weather = HourlyWeatherData::new(30.0, 0.0, 0.0, 0.0, 0.0, 95.0, 0);
        let dp = weather.dew_point();
        let wb = weather.wet_bulb();
        assert!(dp <= 30.0);
        assert!(wb <= 30.0);
        assert!(dp < wb);
    }

    #[test]
    fn test_from_weather_data_pressure() {
        let weather = HourlyWeatherData::new(20.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0);
        let inputs = from_weather_data(&weather);
        assert_eq!(inputs.pressure, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
    }

    #[test]
    fn test_enthalpy_from_weather_different_temps() {
        let cold = HourlyWeatherData::new(-10.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0);
        let hot = HourlyWeatherData::new(40.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0);
        let h_cold = enthalpy_from_weather(&cold);
        let h_hot = enthalpy_from_weather(&hot);
        assert!(h_hot > h_cold);
        assert!(h_cold.is_finite());
        assert!(h_hot < 200.0);
    }

    #[test]
    fn test_standard_atmospheric_pressure_constant() {
        assert_eq!(STANDARD_ATMOSPHERIC_PRESSURE_Pa, 101325.0);
        assert!(STANDARD_ATMOSPHERIC_PRESSURE_Pa.is_finite());
    }

    #[test]
    fn test_moist_air_density_zero_humidity_ratio() {
        let rho = moist_air_density(20.0, 0.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!(rho > 1.0 && rho < 1.3);
        assert!(rho.is_finite());
    }

    #[test]
    fn test_partial_vapor_pressure_zero_humidity_ratio() {
        let pw = partial_vapor_pressure(0.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert_eq!(pw, 0.0);
    }

    #[test]
    fn test_enthalpy_at_extreme_humidity() {
        let h_dry = calculate_enthalpy(25.0, 5.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        let h_humid = calculate_enthalpy(25.0, 95.0, STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!(h_humid > h_dry);
    }
}
