//! Atmospheric constants.
//!
//! This module provides atmospheric constants from ISO 2533:1975 Standard Atmosphere
//! and ASHRAE Handbook of Fundamentals for psychrometric calculations.

/// Standard atmospheric pressure at sea level.
///
/// **Value:** 101,325 Pa
/// **Units:** Pa (pascals) or kPa (101.325 kPa)
/// **Source:** ISO 2533:1975, Standard Atmosphere
/// **Reference:** ASHRAE Handbook of Fundamentals, Chapter 1, Psychrometrics
/// **Uncertainty:** ±10 Pa (±0.01%, atmospheric pressure variation)
/// **Validity:** Valid at sea level (0 m altitude), 15°C, standard conditions
/// **Assumptions:** Dry air, static atmosphere, no weather variations
/// **Notes:** Pressure decreases with altitude at ~11.3 Pa/m near sea level (lapse rate 0.0113 kPa/m). Used for ventilation, infiltration, and psychrometric calculations.
pub const STANDARD_ATMOSPHERIC_PRESSURE: f64 = 101325.0;

/// Air density at sea level and 15°C (dry air).
///
/// **Value:** 1.225 kg/m³
/// **Units:** kg/m³ (kilograms per cubic meter)
/// **Source:** ISO 2533:1975, Standard Atmosphere
/// **Reference:** ASHRAE Handbook of Fundamentals, Chapter 1, Psychrometrics
/// **Uncertainty:** ±0.01 kg/m³ (±0.8%, temperature/humidity variation ±5°C, ±10% RH)
/// **Validity:** Valid at sea level (0 m altitude), 15°C, 101.325 kPa pressure, dry air conditions
/// **Assumptions:** Dry air, ideal gas behavior, standard atmospheric pressure
/// **Notes:** Density decreases with temperature: ρ = P / (R_specific × T), where R_specific = 287.05 J/kgK for dry air. Used for ventilation and infiltration mass flow calculations. Humid air is less dense (~2% lower at 25°C, 50% RH).
pub const AIR_DENSITY_SEA_LEVEL: f64 = 1.225;

/// Specific heat capacity of dry air at constant pressure.
///
/// **Value:** 1005.0 J/kgK
/// **Units:** J/kgK (joules per kilogram Kelvin)
/// **Source:** ASHRAE Handbook of Fundamentals, Chapter 1, Psychrometrics
/// **Reference:** ISO 2533:1975, Standard Atmosphere
/// **Uncertainty:** ±5.0 J/kgK (±0.5%, temperature variation 0-50°C)
/// **Validity:** Valid for dry air at 0-50°C, standard pressure
/// **Assumptions:** Constant specific heat over temperature range, dry air composition
/// **Notes:** Specific heat increases slightly with temperature (1005 J/kgK at 15°C, 1009 J/kgK at 50°C). Used for ventilation and infiltration thermal capacity calculations: Q = ρ × cp × V × ΔT.
pub const AIR_SPECIFIC_HEAT: f64 = 1005.0;

/// Specific gas constant for dry air.
///
/// **Value:** 287.05 J/kgK
/// **Units:** J/kgK (joules per kilogram Kelvin)
/// **Source:** ISO 2533:1975, Standard Atmosphere
/// **Reference:** ASHRAE Handbook of Fundamentals, Chapter 1, Psychrometrics
/// **Uncertainty:** ±0.01 J/kgK (negligible for simulation purposes)
/// **Validity:** Valid for dry air composition (78.08% N₂, 20.95% O₂, 0.93% Ar, 0.04% CO₂)
/// **Assumptions:** Constant air composition, ideal gas behavior
/// **Notes:** Used in ideal gas law: P = ρ × R_specific × T. Specific gas constant = R_universal / M_air, where R_universal = 8314.32 J/kgmolK and M_air = 28.9644 kg/kmol.
pub const SPECIFIC_GAS_CONSTANT_DRY_AIR: f64 = 287.05;

/// Specific gas constant for water vapor.
///
/// **Value:** 461.52 J/kgK
/// **Units:** J/kgK (joules per kilogram Kelvin)
/// **Source:** ISO 2533:1975, Standard Atmosphere
/// **Reference:** ASHRAE Handbook of Fundamentals, Chapter 1, Psychrometrics
/// **Uncertainty:** ±0.01 J/kgK (negligible for simulation purposes)
/// **Validity:** Valid for water vapor composition (H₂O)
/// **Assumptions:** Constant molecular weight, ideal gas behavior
/// **Notes:** Used in psychrometric calculations for humid air. Specific gas constant = R_universal / M_water, where M_water = 18.01528 kg/kmol. Used to calculate humidity ratio: W = 0.622 × (P_v / (P - P_v)).
pub const SPECIFIC_GAS_CONSTANT_WATER_VAPOR: f64 = 461.52;

/// Standard atmospheric temperature lapse rate.
///
/// **Value:** 0.0065 K/m (6.5 K/km)
/// **Units:** K/m (Kelvins per meter) or K/km
/// **Source:** ISO 2533:1975, Standard Atmosphere
/// **Reference:** ASHRAE Handbook of Fundamentals, Chapter 1, Psychrometrics
/// **Uncertainty:** ±0.0001 K/m (±0.01 K/km, depends on weather conditions)
/// **Validity:** Valid in troposphere (0-11 km altitude), mid-latitude conditions
/// **Assumptions:** Standard atmosphere, no temperature inversions
/// **Notes:** Temperature decreases with altitude: T_altitude = T_sea_level - lapse_rate × altitude. Used for altitude corrections to air density and pressure. Lapse rate varies in stratosphere (11-20 km: 0 K/m, constant temperature).
pub const ATMOSPHERIC_LAPSE_RATE: f64 = 0.0065;

/// Gravity acceleration at sea level.
///
/// **Value:** 9.80665 m/s²
/// **Units:** m/s² (meters per second squared)
/// **Source:** ISO 2533:1975, Standard Atmosphere
/// **Reference:** ASHRAE Handbook of Fundamentals, Chapter 1, Psychrometrics
/// **Uncertainty:** ±0.00001 m/s² (±0.0001%, varies with latitude/altitude)
/// **Validity:** Valid at sea level, standard latitude (45°)
/// **Assumptions:** Earth is a perfect oblate spheroid, standard gravity model
/// **Notes:** Gravity decreases with altitude: g_altitude = g_sea_level × (R / (R + altitude))², where R = 6,371,000 m (Earth radius). Used for stack effect calculations in natural ventilation.
pub const GRAVITY_ACCELERATION: f64 = 9.80665;

/// Standard temperature at sea level.
///
/// **Value:** 288.15 K (15°C)
/// **Units:** K (Kelvin) or °C (Celsius)
/// **Source:** ISO 2533:1975, Standard Atmosphere
/// **Reference:** ASHRAE Handbook of Fundamentals, Chapter 1, Psychrometrics
/// **Uncertainty:** ±0.5 K (±0.5°C, seasonal variation)
/// **Validity:** Valid at sea level (0 m altitude), standard conditions
/// **Assumptions:** Standard atmosphere, no weather variations
/// **Notes:** Used as reference temperature for altitude corrections: T_altitude = T_sea_level - lapse_rate × altitude. Also used as reference for psychrometric calculations (e.g., 15°C standard conditions for ASHRAE 140).
pub const STANDARD_TEMPERATURE_SEA_LEVEL: f64 = 288.15;
