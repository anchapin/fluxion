//! HVAC Economizer Mode
//!
//! This module provides economizer control for free cooling when outdoor
//! conditions are favorable, reducing mechanical cooling energy.

use serde::{Deserialize, Serialize};

/// Calculates zone air enthalpy from zone temperature and outdoor humidity.
///
/// This helper function estimates zone enthalpy for economizer control,
/// assuming zone humidity is similar to outdoor humidity (valid approximation
/// for many economizer applications).
///
/// # Arguments
/// * `zone_temp` - Zone air temperature (°C)
/// * `outdoor_humidity` - Outdoor relative humidity (%) - used as proxy for zone humidity
///
/// # Returns
/// Zone air enthalpy (kJ/kg)
///
/// # Note
/// For more accurate zone enthalpy, use HourlyWeatherData for zone conditions
/// if available. This helper is for economizer control where only zone_temp
/// is typically known.
#[allow(dead_code)]
fn zone_enthalpy_from_temp(zone_temp: f64, outdoor_humidity: f64) -> f64 {
    use crate::weather::calculate_enthalpy;
    calculate_enthalpy(
        zone_temp,
        outdoor_humidity,
        crate::weather::STANDARD_ATMOSPHERIC_PRESSURE_Pa,
    )
}

/// Economizer operating mode.
///
/// Economizers provide free cooling by using outdoor air when conditions
/// are favorable, reducing mechanical cooling energy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EconomizerMode {
    /// Economizer disabled (mechanical cooling only)
    Disabled,
    /// Dry bulb temperature control (free cooling when outdoor air is cooler)
    DryBulb,
    /// Enthalpy control (free cooling when outdoor air is cooler AND drier)
    Enthalpy,
}

/// Check if economizer should be active for free cooling.
///
/// Economizer is active when outdoor conditions are favorable:
/// - Dry bulb mode: Outdoor temperature < zone temperature AND outdoor temperature < zone setpoint
/// - Enthalpy mode: Outdoor temperature < zone temperature AND outdoor enthalpy < zone enthalpy
///
/// # Arguments
/// * `mode` - Economizer operating mode
/// * `outdoor_temp` - Outdoor air temperature (°C)
/// * `outdoor_enthalpy` - Outdoor air enthalpy (kJ/kg) - optional, used in Enthalpy mode
/// * `zone_temp` - Zone air temperature (°C)
/// * `zone_enthalpy` - Zone air enthalpy (kJ/kg) - optional, used in Enthalpy mode
/// * `cooling_setpoint` - Zone cooling setpoint (°C)
///
/// # Returns
/// true if economizer should provide free cooling, false otherwise
///
/// # Note
/// Enthalpy mode requires HourlyWeatherData to calculate outdoor enthalpy.
/// If HourlyWeatherData is not available, Enthalpy mode falls back to Disabled
/// (safe default: no free cooling).
///
/// For calculating zone enthalpy from zone temperature and outdoor humidity,
/// use the helper function `zone_enthalpy_from_temp()`.
pub fn is_economizer_active(
    mode: EconomizerMode,
    outdoor_temp: f64,
    outdoor_enthalpy: Option<f64>,
    zone_temp: f64,
    zone_enthalpy: Option<f64>,
    cooling_setpoint: f64,
) -> bool {
    match mode {
        EconomizerMode::Disabled => false,

        EconomizerMode::DryBulb => {
            // Free cooling when outdoor air is cooler than zone AND below setpoint
            outdoor_temp < zone_temp && outdoor_temp < cooling_setpoint
        }

        EconomizerMode::Enthalpy => {
            // Free cooling when outdoor air is cooler AND has lower enthalpy
            // Phase 16: Psychrometrics module provides accurate enthalpy calculations

            // Check outdoor air enthalpy (requires HourlyWeatherData)
            let outdoor_h = match outdoor_enthalpy {
                Some(h) => h,
                None => {
                    // If HourlyWeatherData not available, cannot calculate enthalpy
                    // Fall back to dry bulb mode as safe default
                    return false;
                }
            };

            // Check zone air enthalpy (calculate if not provided)
            let zone_h = match zone_enthalpy {
                Some(h) => h,
                None => {
                    // If zone enthalpy not provided, cannot make comparison
                    // Return false (safe default: no free cooling)
                    return false;
                }
            };

            // Economizer active if: outdoor air is cooler AND has lower enthalpy
            outdoor_temp < zone_temp && outdoor_h < zone_h
        }
    }
}

/// Calculate free cooling capacity from economizer (kW).
///
/// Economizer capacity is based on cooling potential of outdoor air.
///
/// # Arguments
/// * `outdoor_temp` - Outdoor air temperature (°C)
/// * `zone_temp` - Zone air temperature (°C)
/// * `airflow_rate` - Economizer airflow rate (m³/s)
///
/// # Returns
/// Free cooling capacity in kilowatts
///
/// # Formula
/// Q = ρ × cp × V̇ × ΔT
/// where:
/// - ρ = air density (1.2 kg/m³)
/// - cp = specific heat of air (1005 J/kg·K)
/// - V̇ = airflow rate (m³/s)
/// - ΔT = temperature difference (zone_temp - outdoor_temp)
pub fn calculate_free_cooling_capacity(
    outdoor_temp: f64,
    zone_temp: f64,
    airflow_rate: f64,
) -> f64 {
    if outdoor_temp >= zone_temp {
        return 0.0; // No cooling potential
    }

    let rho = 1.2; // kg/m³
    let cp = 1005.0; // J/kg·K
    let delta_t = zone_temp - outdoor_temp; // K

    // Q = ρ × cp × V̇ × ΔT (Watts)
    let capacity_watts = rho * cp * airflow_rate * delta_t;

    capacity_watts / 1000.0 // Convert to kW
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_economizer_disabled() {
        let active = is_economizer_active(
            EconomizerMode::Disabled,
            15.0, // Outdoor temp
            None,
            25.0, // Zone temp
            None,
            24.0, // Cooling setpoint
        );
        assert_eq!(active, false);
    }

    #[test]
    fn test_dry_bulb_active() {
        // Dry bulb mode: outdoor cooler than zone and below setpoint
        let active = is_economizer_active(
            EconomizerMode::DryBulb,
            15.0, // Outdoor temp
            None,
            25.0, // Zone temp
            None,
            24.0, // Cooling setpoint
        );
        assert_eq!(active, true);
    }

    #[test]
    fn test_dry_bulb_above_setpoint() {
        // Dry bulb mode: outdoor cooler but above setpoint
        let active = is_economizer_active(
            EconomizerMode::DryBulb,
            25.0, // Outdoor temp (above setpoint)
            None,
            26.0, // Zone temp
            None,
            24.0, // Cooling setpoint
        );
        assert_eq!(active, false);
    }

    #[test]
    fn test_economizer_enthalpy_mode_active() {
        // Enthalpy mode: outdoor (15°C, 60 kJ/kg) is cooler than zone (25°C, 70 kJ/kg)
        // Economizer should be active
        let active = is_economizer_active(
            EconomizerMode::Enthalpy,
            15.0,       // Outdoor temp (cooler than zone)
            Some(60.0), // Outdoor enthalpy (lower than zone)
            25.0,       // Zone temp
            Some(70.0), // Zone enthalpy
            24.0,       // Cooling setpoint
        );
        assert!(
            active,
            "Economizer should be active when outdoor is cooler and has lower enthalpy"
        );
    }

    #[test]
    fn test_economizer_enthalpy_mode_inactive_same_enthalpy() {
        // Enthalpy mode: outdoor enthalpy equals zone enthalpy
        // Economizer should be inactive (no benefit to free cooling)
        let active = is_economizer_active(
            EconomizerMode::Enthalpy,
            15.0,       // Outdoor temp (cooler than zone)
            Some(65.0), // Outdoor enthalpy (same as zone)
            25.0,       // Zone temp
            Some(65.0), // Zone enthalpy
            24.0,       // Cooling setpoint
        );
        assert!(
            !active,
            "Economizer should be inactive when enthalpies are equal"
        );
    }

    #[test]
    fn test_economizer_enthalpy_mode_inactive_hotter() {
        // Enthalpy mode: outdoor is hotter than zone
        // Economizer should be inactive
        let active = is_economizer_active(
            EconomizerMode::Enthalpy,
            30.0,       // Outdoor temp (hotter than zone)
            Some(75.0), // Outdoor enthalpy (higher than zone)
            25.0,       // Zone temp
            Some(65.0), // Zone enthalpy
            24.0,       // Cooling setpoint
        );
        assert!(
            !active,
            "Economizer should be inactive when outdoor is hotter"
        );
    }

    #[test]
    fn test_economizer_enthalpy_mode_inactive_higher_enthalpy() {
        // Enthalpy mode: outdoor is cooler but has higher enthalpy (humid outdoor air)
        // Economizer should be inactive (would increase cooling load)
        let active = is_economizer_active(
            EconomizerMode::Enthalpy,
            15.0,       // Outdoor temp (cooler than zone)
            Some(80.0), // Outdoor enthalpy (higher than zone - humid air)
            25.0,       // Zone temp
            Some(65.0), // Zone enthalpy
            24.0,       // Cooling setpoint
        );
        assert!(
            !active,
            "Economizer should be inactive when outdoor enthalpy is higher"
        );
    }

    #[test]
    fn test_economizer_enthalpy_mode_missing_data() {
        // Enthalpy mode: enthalpy data missing (backward compatibility test)
        // Economizer should be inactive (safe default)
        let active_no_outdoor = is_economizer_active(
            EconomizerMode::Enthalpy,
            15.0,
            None, // Missing outdoor enthalpy
            25.0,
            Some(70.0),
            24.0,
        );
        assert_eq!(
            active_no_outdoor, false,
            "Enthalpy mode should be inactive without outdoor enthalpy"
        );

        let active_no_zone = is_economizer_active(
            EconomizerMode::Enthalpy,
            15.0,
            Some(60.0),
            25.0,
            None, // Missing zone enthalpy
            24.0,
        );
        assert_eq!(
            active_no_zone, false,
            "Enthalpy mode should be inactive without zone enthalpy"
        );

        let active_none = is_economizer_active(
            EconomizerMode::Enthalpy,
            15.0,
            None, // Missing outdoor enthalpy
            25.0,
            None, // Missing zone enthalpy
            24.0,
        );
        assert_eq!(
            active_none, false,
            "Enthalpy mode should be inactive without enthalpy data"
        );
    }

    #[test]
    fn test_free_cooling_capacity() {
        // Free cooling capacity proportional to temperature difference
        let capacity_1 = calculate_free_cooling_capacity(20.0, 25.0, 0.5);
        let capacity_2 = calculate_free_cooling_capacity(15.0, 25.0, 0.5);

        // Larger temperature difference = more free cooling
        assert!(capacity_2 > capacity_1);

        // No cooling when outdoor temp >= zone temp
        let capacity_no_cooling = calculate_free_cooling_capacity(25.0, 25.0, 0.5);
        assert_eq!(capacity_no_cooling, 0.0);

        // Calculate expected capacity for verification
        // Q = ρ × cp × V̇ × ΔT
        let rho = 1.2; // kg/m³
        let cp = 1005.0; // J/kg·K
        let delta_t = 5.0; // K
        let airflow = 0.5; // m³/s
        let expected_watts = rho * cp * airflow * delta_t; // ~3015 W
        let expected_kw = expected_watts / 1000.0; // ~3.015 kW

        let capacity_test = calculate_free_cooling_capacity(20.0, 25.0, 0.5);
        assert!((capacity_test - expected_kw).abs() < 0.1);
    }

    #[test]
    fn test_zone_enthalpy_from_temp() {
        // Verify helper function calculates correct enthalpy
        use crate::weather::calculate_enthalpy;

        let zone_h = zone_enthalpy_from_temp(25.0, 50.0);
        let expected_h =
            calculate_enthalpy(25.0, 50.0, crate::weather::STANDARD_ATMOSPHERIC_PRESSURE_Pa);
        assert!(
            (zone_h - expected_h).abs() < 0.01,
            "Zone enthalpy calculation mismatch"
        );
    }
}
