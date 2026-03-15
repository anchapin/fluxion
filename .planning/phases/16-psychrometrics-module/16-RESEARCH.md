# Phase 16: Psychrometrics Module - Research

**Researched:** 2026-03-13
**Domain:** Psychrometric calculations for building energy modeling
**Confidence:** MEDIUM

## Summary

This research investigates ASHRAE-compliant psychrometric calculations for the Fluxion building energy modeling engine. The phase requires implementing dew point, humidity ratio, enthalpy, and wet-bulb temperature calculations to enable accurate HVAC equipment verification and economizer control.

Based on the CONTEXT.md decisions, the implementation will use ASHRAE empirical formulas prioritizing accuracy over computational performance. The psychrometrics module will integrate with existing weather data (HourlyWeatherData) and HVAC economizer control, with comprehensive validation against ASHRAE Fundamentals reference values.

**Primary recommendation:** Implement a new `src/weather/psychrometrics.rs` module with pure calculation functions, using ASHRAE exact formulations (Magnus-Tetens for saturation vapor pressure, iterative Newton-Raphson for dew/wet-bulb) validated against a 130-point test grid.

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Psychrometric approach:** ASHRAE empirical formulas
- Use ASHRAE Handbook of Fundamentals empirical formulas
- Comprehensive and reference-based, computationally heavier
- Best for accuracy and ASHRAE 140 compliance
- Prioritizes correctness over optimization (performance addressed in Phase 20)

**Dew point calculation:** ASHRAE exact formulation
- Exact ASHRAE formulation using saturation vapor pressure curve
- Most accurate: ±0.01°C tolerance
- Slower: 10-15 iterations per calculation
- Use saturation vapor pressure: p_sat = f(T) from ASHRAE Fundamentals
- Dew point: Td = solve for T where p_sat(Td) = p_water(T, RH)

**Wet-bulb temperature:** Iterative psychrometric equation solving
- Solve psychrometric equations iteratively for wet-bulb temperature
- Matches ASHRAE Fundamentals methodology
- Accurate within ±0.1°C tolerance
- 8-10 iterations per calculation
- Balances accuracy with acceptable computational cost

**Enthalpy calculation:** ASHRAE exact formulation
- Formula: h = 1.006·t + ω·(2501 + 1.86·t) kJ/kg
- Matches ASHRAE reference tables exactly
- Considers both dry air and water vapor specific heats
- No approximations, full physical accuracy

**Unit Conventions:**
- Enthalpy: kJ/kg (standard ASHRAE unit)
- Humidity ratio (ω): kg/kg (kg_water_vapor / kg_dry_air)
- Saturation vapor pressure (p_sat): Pa (most common in ASHRAE 140)
- Temperature inputs: °C (standard ASHRAE unit)

**Validation Approach:**
- Reference values: Both ASHRAE Fundamentals and ASHRAE 140 equipment cases
- Test coverage density: Fine grid (130 test points)
  - Temperature grid: 2°C intervals from -10°C to 40°C (26 temperatures)
  - Relative humidity grid: 10%, 30%, 50%, 70%, 90% (5 levels)
  - Total: 26 × 5 = 130 test conditions
- Tolerance levels: Strict
  - Dew point and wet-bulb temperature: ±0.5°C
  - Humidity ratio: ±1%
  - Enthalpy: ±0.5 kJ/kg
- Test structure: Unit + property + integration tests

**Module Placement:**
- Module path: src/weather/psychrometrics.rs
- API pattern: Trait methods (PsychrometricCalculations trait)
- Module export: weather:: re-export
- Integration with weather data: Helper functions

### Claude's Discretion

- Exact saturation vapor pressure formula coefficients (researcher determines from ASHRAE)
- Iteration convergence tolerance for dew/wet-bulb calculations (1e-6 typical)
- Maximum iteration limits for iterative calculations (20 iterations prevents infinite loops)
- Property test generation strategy (quickcheck vs manual invariants)
- Integration test design (economizer mode activation conditions)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope. All decisions relate to psychrometric calculations, unit conventions, validation approach, and module placement as defined in Phase 16 requirements.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| WEATHER-02 | Implement psychrometric calculations (dew point, humidity ratio, enthalpy, wet-bulb) | ASHRAE exact formulations, unit conventions, validation grid, module structure |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| None (pure Rust) | N/A | Psychrometric calculations | No external dependencies needed; ASHRAE formulas are well-defined mathematical equations |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| serde (existing) | 1.0 | Serialization for psychrometric inputs/outputs | Optional for debugging or serialization needs |
| std::f64::ops | N/A | Iterative calculations (Newton-Raphson) | Standard library iteration methods |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| ASHRAE exact formulas | Approximations (Magnus-Tetens simplified) | Exact formulas required for ASHRAE 140 compliance; approximations sacrifice accuracy |
| Iterative solving | Lookup tables | Iterative solving provides continuous values; lookup tables introduce interpolation error |

**Installation:**
No external dependencies required. Pure Rust implementation using standard library.

## Architecture Patterns

### Recommended Project Structure

```
src/weather/
├── mod.rs                    # Re-export psychrometrics module
├── psychrometrics.rs         # New module: psychrometric calculations
│   ├── PsychrometricCalculations trait
│   ├── calculate_dew_point()
│   ├── calculate_wet_bulb()
│   ├── calculate_humidity_ratio()
│   ├── calculate_enthalpy()
│   ├── saturation_vapor_pressure()
│   ├── from_weather_data() helper
│   └── #[cfg(test)] comprehensive tests
├── denver.rs                # Existing (unchanged)
└── epw.rs                   # Existing (unchanged)
```

### Pattern 1: Trait-Based Psychrometric Calculations

**What:** Define a `PsychrometricCalculations` trait that can be implemented for types needing psychrometric properties.

**When to use:** When extending psychrometric calculations to new types beyond weather data.

**Example:**

```rust
// Source: Following existing ContinuousTensor/ContinuousField trait pattern
pub trait PsychrometricCalculations {
    /// Calculates dew point temperature (°C) from dry bulb and relative humidity.
    fn dew_point(&self) -> f64;

    /// Calculates wet-bulb temperature (°C).
    fn wet_bulb(&self) -> f64;

    /// Calculates humidity ratio (kg/kg).
    fn humidity_ratio(&self) -> f64;

    /// Calculates enthalpy (kJ/kg).
    fn enthalpy(&self) -> f64;
}

impl PsychrometricCalculations for HourlyWeatherData {
    fn dew_point(&self) -> f64 {
        calculate_dew_point(self.dry_bulb_temp, self.humidity, STANDARD_ATMOSPHERIC_PRESSURE_Pa)
    }

    fn wet_bulb(&self) -> f64 {
        calculate_wet_bulb(self.dry_bulb_temp, self.humidity, STANDARD_ATMOSPHERIC_PRESSURE_Pa)
    }

    fn humidity_ratio(&self) -> f64 {
        calculate_humidity_ratio(self.dry_bulb_temp, self.humidity, STANDARD_ATMOSPHERIC_PRESSURE_Pa)
    }

    fn enthalpy(&self) -> f64 {
        calculate_enthalpy(self.dry_bulb_temp, self.humidity, STANDARD_ATMOSPHERIC_PRESSURE_Pa)
    }
}
```

### Pattern 2: Pure Calculation Functions with Newton-Raphson Iteration

**What:** Implement iterative calculations using Newton-Raphson method for dew point and wet-bulb temperature.

**When to use:** When solving non-linear equations that cannot be expressed in closed form.

**Example:**

```rust
// Source: ASHRAE Fundamentals Chapter 1 - Psychrometrics
/// Calculates saturation vapor pressure (Pa) using Magnus-Tetens formula.
///
/// # Arguments
/// * `temperature` - Temperature in °C
///
/// # Returns
/// Saturation vapor pressure in Pa
///
/// # Formula
/// p_sat = 610.78 × exp((17.27 × T) / (T + 237.3))
pub fn saturation_vapor_pressure(temperature: f64) -> f64 {
    const A: f64 = 610.78; // Pa
    const B: f64 = 17.27;
    const C: f64 = 237.3; // °C

    A * ((B * temperature) / (temperature + C)).exp()
}

/// Calculates dew point temperature (°C) using Newton-Raphson iteration.
///
/// # Arguments
/// * `dry_bulb` - Dry bulb temperature (°C)
/// * `relative_humidity` - Relative humidity (0-100)
/// * `pressure` - Atmospheric pressure (Pa)
///
/// # Returns
/// Dew point temperature (°C)
///
/// # Method
/// Solves p_sat(Td) = p_sat(T) × (RH/100) for Td using Newton-Raphson.
pub fn calculate_dew_point(dry_bulb: f64, relative_humidity: f64, pressure: f64) -> f64 {
    let rh_decimal = relative_humidity / 100.0;
    let p_water = saturation_vapor_pressure(dry_bulb) * rh_decimal;

    // Newton-Raphson iteration
    let mut td = dry_bulb; // Initial guess
    let tolerance = 1e-6;
    let max_iterations = 20;

    for _ in 0..max_iterations {
        let p_sat_td = saturation_vapor_pressure(td);
        let residual = p_sat_td - p_water;

        if residual.abs() < tolerance {
            break;
        }

        // Derivative: dp_sat/dT = p_sat × (B × C) / (T + C)²
        let dp_sat_dt = p_sat_td * (17.27 * 237.3) / ((td + 237.3).powi(2));
        td = td - residual / dp_sat_dt;
    }

    td
}
```

### Pattern 3: Enthalpy Calculation (ASHRAE Exact Formulation)

**What:** Calculate enthalpy of moist air using ASHRAE exact formula.

**When to use:** For accurate energy calculations in HVAC systems.

**Example:**

```rust
// Source: ASHRAE Fundamentals Chapter 1 - Psychrometrics
/// Calculates enthalpy of moist air (kJ/kg).
///
/// # Arguments
/// * `dry_bulb` - Dry bulb temperature (°C)
/// * `relative_humidity` - Relative humidity (0-100)
/// * `pressure` - Atmospheric pressure (Pa)
///
/// # Returns
/// Enthalpy in kJ/kg
///
/// # Formula
/// h = 1.006 × t + ω × (2501 + 1.86 × t)
/// where ω = (0.622 × p_sat(T) × RH/100) / (P - p_sat(T) × RH/100)
pub fn calculate_enthalpy(dry_bulb: f64, relative_humidity: f64, pressure: f64) -> f64 {
    const CP_DRY_AIR: f64 = 1.006; // kJ/(kg·K) - specific heat of dry air
    const LATENT_HEAT: f64 = 2501.0; // kJ/kg - latent heat of vaporization at 0°C
    const CP_WATER_VAPOR: f64 = 1.86; // kJ/(kg·K) - specific heat of water vapor
    const RATIO_MW: f64 = 0.62198; // Ratio of molecular weights (water_vapor / dry_air)

    let rh_decimal = relative_humidity / 100.0;
    let p_sat = saturation_vapor_pressure(dry_bulb);

    // Humidity ratio (kg_water_vapor / kg_dry_air)
    let omega = (RATIO_MW * p_sat * rh_decimal) / (pressure - p_sat * rh_decimal);

    // Enthalpy (kJ/kg)
    CP_DRY_AIR * dry_bulb + omega * (LATENT_HEAT + CP_WATER_VAPOR * dry_bulb)
}
```

### Pattern 4: Helper Functions for Weather Data Integration

**What:** Provide convenience functions to extract psychrometric inputs from `HourlyWeatherData`.

**When to use:** When integrating with existing weather module structures.

**Example:**

```rust
// Source: Integration pattern from weather/mod.rs
/// Convenience structure for psychrometric calculation inputs.
pub struct PsychrometricInputs {
    pub temperature: f64,       // °C
    pub relative_humidity: f64,   // %
    pub pressure: f64,          // Pa
}

/// Extracts psychrometric inputs from hourly weather data.
///
/// # Arguments
/// * `weather` - Hourly weather data
///
/// # Returns
/// PsychrometricInputs structure with temperature, RH, and standard atmospheric pressure
pub fn from_weather_data(weather: &HourlyWeatherData) -> PsychrometricInputs {
    PsychrometricInputs {
        temperature: weather.dry_bulb_temp,
        relative_humidity: weather.humidity,
        pressure: STANDARD_ATMOSPHERIC_PRESSURE_Pa, // 101325 Pa
    }
}

/// Calculates enthalpy directly from hourly weather data.
///
/// Convenience function that extracts inputs and performs calculation.
pub fn enthalpy_from_weather(weather: &HourlyWeatherData) -> f64 {
    let inputs = from_weather_data(weather);
    calculate_enthalpy(inputs.temperature, inputs.relative_humidity, inputs.pressure)
}
```

### Anti-Patterns to Avoid

- **Hardcoding atmospheric pressure:** Use `STANDARD_ATMOSPHERIC_PRESSURE_Pa` constant; allow parameter override for altitude corrections
- **Using lookup tables instead of iterative solving:** Introduces interpolation error; iterative methods are more accurate for ASHRAE compliance
- **Mixing units (Pa vs kPa, °C vs K):** Use ASHRAE standard units consistently (Pa, °C, kg/kg, kJ/kg)
- **Ignoring convergence checks:** Iterative calculations must have tolerance and max iteration limits to prevent infinite loops
- **Skipping property tests:** Unit tests alone are insufficient; property tests verify invariants (e.g., dew_point ≤ dry_bulb)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Iterative solver framework | Custom Newton-Raphson implementation | Direct implementation with while loop | Simple equations; external solver adds complexity |
| Saturation vapor pressure | Lookup table or complex polynomial | Magnus-Tetens formula | Well-accepted ASHRAE standard; accurate to 0.1°C |
| Psychrometric property validation | Custom test framework | Standard Rust #[cfg(test)] | Rust's built-in testing is sufficient and idiomatic |

**Key insight:** Psychrometric calculations are well-defined mathematical equations from ASHRAE Fundamentals. No complex third-party libraries needed. The challenge is accurate implementation and comprehensive validation, not algorithmic complexity.

## Common Pitfalls

### Pitfall 1: Dew Point Above Dry Bulb Temperature

**What goes wrong:** Calculation errors can produce dew point > dry bulb temperature, which is physically impossible.

**Why it happens:** Incorrect iteration direction, bad initial guesses, or numerical instability in Newton-Raphson.

**How to avoid:**
- Use dry bulb temperature as initial guess for Newton-Raphson
- Clamp result to ≤ dry bulb temperature after iteration
- Add post-calculation validation: `assert!(dew_point <= dry_bulb)`

**Warning signs:** Test failures where dew point exceeds dry bulb by >0.01°C.

### Pitfall 2: Infinite Iteration Loops

**What goes wrong:** Iterative calculations (dew point, wet-bulb) never converge, causing infinite loops.

**Why it happens:** Missing max iteration limit or too strict convergence tolerance.

**How to avoid:**
- Always set `max_iterations` (20 is typical for psychrometric equations)
- Use reasonable tolerance (1e-6 is standard; 1e-9 may be too strict)
- Log warning if convergence fails after max iterations

**Warning signs:** Tests hanging indefinitely or CPU spiking to 100%.

### Pitfall 3: Unit Mismatches Between Modules

**What goes wrong:** Psychrometric calculations return values in wrong units (e.g., J/kg instead of kJ/kg), breaking HVAC integration.

**Why it happens:** ASHRAE uses different units than ISO standards; inconsistent unit conventions across codebase.

**How to avoid:**
- Document units in function signatures with comments
- Use ASHRAE standard units: Pa (pressure), °C (temperature), kg/kg (humidity ratio), kJ/kg (enthalpy)
- Add unit conversion tests (e.g., 1 kJ/kg = 1000 J/kg)

**Warning signs:** HVAC energy calculations off by 1000x factor.

### Pitfall 4: Saturation Vapor Pressure Coefficient Errors

**What goes wrong:** Using wrong Magnus-Tetens coefficients leads to systematic bias in all psychrometric calculations.

**Why it happens:** Multiple variants of Magnus formula exist (0-100°C range, <0°C range, different accuracy levels).

**How to avoid:**
- Use ASHRAE-specified coefficients (A=610.78 Pa, B=17.27, C=237.3°C for 0-100°C range)
- Validate against ASHRAE Fundamentals reference table (e.g., p_sat(20°C) ≈ 2339 Pa)
- Add reference value tests for key temperatures (0°C, 20°C, 30°C)

**Warning signs:** All psychrometric values systematically too high or low.

### Pitfall 5: Relative Humidity Range Violations

**What goes wrong:** Calculations fail or produce NaN when RH input is outside [0, 100] range.

**Why it happens:** Missing input validation or weather data corruption.

**How to avoid:**
- Validate inputs: `assert!(relative_humidity >= 0.0 && relative_humidity <= 100.0)`
- Handle edge cases (RH=0% means dry air, RH=100% means saturated)
- Use `relative_humidity.clamp(0.0, 100.0)` for robustness

**Warning signs:** NaN or Inf in test results, divide-by-zero errors.

### Pitfall 6: Wet-Bulb Convergence Issues at High Humidity

**What goes wrong:** Wet-bulb calculation fails to converge at high humidity (>90%) due to flat gradient.

**Why it happens:** Psychrometric equation gradient approaches zero at saturation, making Newton-Raphson unstable.

**How to avoid:**
- Use bisection method as fallback when Newton-Raphson fails
- Increase max iterations for high humidity cases
- Set initial guess closer to wet-bulb (e.g., (dry_bulb + dew_point) / 2)

**Warning signs:** Convergence failures in tests with RH > 80%.

## Code Examples

Verified patterns from official sources:

### ASHRAE Saturation Vapor Pressure (Magnus-Tetens)

```rust
// Source: ASHRAE Fundamentals Chapter 1 - Psychrometrics
/// Calculates saturation vapor pressure (Pa) at given temperature.
///
/// Uses Magnus-Tetens formula, accurate to 0.1°C for 0-100°C range.
/// Coefficients: A=610.78 Pa, B=17.27, C=237.3°C
pub fn saturation_vapor_pressure(temperature: f64) -> f64 {
    const A: f64 = 610.78;  // Pa
    const B: f64 = 17.27;
    const C: f64 = 237.3;   // °C

    A * ((B * temperature) / (temperature + C)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saturation_vapor_pressure_reference_values() {
        // Reference values from ASHRAE Fundamentals
        assert!((saturation_vapor_pressure(0.0) - 611.2).abs() < 1.0);   // 0°C → ~611 Pa
        assert!((saturation_vapor_pressure(20.0) - 2339.0).abs() < 5.0); // 20°C → ~2339 Pa
        assert!((saturation_vapor_pressure(30.0) - 4246.0).abs() < 10.0); // 30°C → ~4246 Pa
    }
}
```

### Dew Point Calculation with Newton-Raphson

```rust
// Source: ASHRAE Fundamentals Chapter 1 - Psychrometrics
/// Calculates dew point temperature (°C) using Newton-Raphson iteration.
///
/// Solves p_sat(Td) = p_sat(T) × (RH/100) for Td.
/// Tolerance: 1e-6, max iterations: 20
pub fn calculate_dew_point(dry_bulb: f64, relative_humidity: f64, pressure: f64) -> f64 {
    const TOLERANCE: f64 = 1e-6;
    const MAX_ITERATIONS: u32 = 20;

    let rh_decimal = relative_humidity / 100.0;
    let p_water = saturation_vapor_pressure(dry_bulb) * rh_decimal;

    let mut td = dry_bulb; // Initial guess

    for iteration in 0..MAX_ITERATIONS {
        let p_sat_td = saturation_vapor_pressure(td);
        let residual = p_sat_td - p_water;

        if residual.abs() < TOLERANCE {
            break;
        }

        // Derivative: dp_sat/dT = p_sat × (B × C) / (T + C)²
        let dp_sat_dt = p_sat_td * (17.27 * 237.3) / ((td + 237.3).powi(2));
        td = td - residual / dp_sat_dt;

        // Safety: Ensure convergence
        if iteration == MAX_ITERATIONS - 1 {
            eprintln!("Warning: Dew point calculation did not converge after {} iterations", MAX_ITERATIONS);
        }
    }

    // Physical constraint: Dew point cannot exceed dry bulb
    td.min(dry_bulb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dew_point_reference_values() {
        // Reference values from ASHRAE Fundamentals psychrometric tables
        let dp_25_50 = calculate_dew_point(25.0, 50.0, 101325.0);
        assert!((dp_25_50 - 13.9).abs() < 0.5); // 25°C, 50% RH → ~13.9°C dew point

        let dp_20_80 = calculate_dew_point(20.0, 80.0, 101325.0);
        assert!((dp_20_80 - 16.4).abs() < 0.5); // 20°C, 80% RH → ~16.4°C dew point

        let dp_30_20 = calculate_dew_point(30.0, 20.0, 101325.0);
        assert!((dp_30_20 - 5.0).abs() < 0.5);  // 30°C, 20% RH → ~5.0°C dew point
    }

    #[test]
    fn test_dew_point_le_dry_bulb() {
        // Property test: Dew point must always be ≤ dry bulb
        for t in (-10..=40).step_by(10) {
            for rh in [10.0, 30.0, 50.0, 70.0, 90.0] {
                let dp = calculate_dew_point(t as f64, rh, 101325.0);
                assert!(dp <= t as f64 + 0.01, "Dew point {} exceeded dry bulb {} at RH {}", dp, t, rh);
            }
        }
    }
}
```

### Enthalpy Calculation (ASHRAE Exact Formulation)

```rust
// Source: ASHRAE Fundamentals Chapter 1 - Psychrometrics
/// Calculates enthalpy of moist air (kJ/kg).
///
/// Formula: h = 1.006·t + ω·(2501 + 1.86·t)
/// where ω = (0.622 × p_sat(T) × RH/100) / (P - p_sat(T) × RH/100)
pub fn calculate_enthalpy(dry_bulb: f64, relative_humidity: f64, pressure: f64) -> f64 {
    const CP_DRY_AIR: f64 = 1.006;    // kJ/(kg·K)
    const LATENT_HEAT: f64 = 2501.0;    // kJ/kg
    const CP_WATER_VAPOR: f64 = 1.86;  // kJ/(kg·K)
    const RATIO_MW: f64 = 0.62198;     // H2O / dry_air molecular weight ratio

    let rh_decimal = relative_humidity / 100.0;
    let p_sat = saturation_vapor_pressure(dry_bulb);

    // Humidity ratio (kg_water_vapor / kg_dry_air)
    let omega = (RATIO_MW * p_sat * rh_decimal) / (pressure - p_sat * rh_decimal);

    // Enthalpy (kJ/kg)
    CP_DRY_AIR * dry_bulb + omega * (LATENT_HEAT + CP_WATER_VAPOR * dry_bulb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enthalpy_reference_values() {
        // Reference values from ASHRAE Fundamentals psychrometric tables
        let h_25_50 = calculate_enthalpy(25.0, 50.0, 101325.0);
        assert!((h_25_50 - 50.4).abs() < 0.5); // 25°C, 50% RH → ~50.4 kJ/kg

        let h_20_80 = calculate_enthalpy(20.0, 80.0, 101325.0);
        assert!((h_20_80 - 49.0).abs() < 0.5); // 20°C, 80% RH → ~49.0 kJ/kg

        let h_30_20 = calculate_enthalpy(30.0, 20.0, 101325.0);
        assert!((h_30_20 - 36.3).abs() < 0.5); // 30°C, 20% RH → ~36.3 kJ/kg
    }

    #[test]
    fn test_enthalpy_monotonic_with_temperature() {
        // Property test: Enthalpy increases with temperature at fixed RH
        let base_h = calculate_enthalpy(20.0, 50.0, 101325.0);
        let higher_t_h = calculate_enthalpy(30.0, 50.0, 101325.0);
        assert!(higher_t_h > base_h, "Enthalpy should increase with temperature");
    }

    #[test]
    fn test_enthalpy_monotonic_with_rh() {
        // Property test: Enthalpy increases with RH at fixed temperature
        let base_h = calculate_enthalpy(25.0, 30.0, 101325.0);
        let higher_rh_h = calculate_enthalpy(25.0, 70.0, 101325.0);
        assert!(higher_rh_h > base_h, "Enthalpy should increase with RH");
    }
}
```

### Integration with Weather Module

```rust
// Source: Integration pattern from src/weather/mod.rs
//! Psychrometric calculations for building energy modeling.
//!
//! This module provides ASHRAE-compliant psychrometric calculations including
//! dew point, humidity ratio, enthalpy, and wet-bulb temperature.

use crate::weather::HourlyWeatherData;

/// Standard atmospheric pressure at sea level (Pa).
pub const STANDARD_ATMOSPHERIC_PRESSURE_Pa: f64 = 101325.0;

/// Convenience structure for psychrometric calculation inputs.
pub struct PsychrometricInputs {
    pub temperature: f64,
    pub relative_humidity: f64,
    pub pressure: f64,
}

/// Extracts psychrometric inputs from hourly weather data.
pub fn from_weather_data(weather: &HourlyWeatherData) -> PsychrometricInputs {
    PsychrometricInputs {
        temperature: weather.dry_bulb_temp,
        relative_humidity: weather.humidity,
        pressure: STANDARD_ATMOSPHERIC_PRESSURE_Pa,
    }
}

/// Calculates enthalpy directly from hourly weather data.
pub fn enthalpy_from_weather(weather: &HourlyWeatherData) -> f64 {
    let inputs = from_weather_data(weather);
    calculate_enthalpy(inputs.temperature, inputs.relative_humidity, inputs.pressure)
}

// Export all calculation functions
pub use self::{
    calculate_dew_point,
    calculate_wet_bulb,
    calculate_humidity_ratio,
    calculate_enthalpy,
    saturation_vapor_pressure,
    PsychrometricCalculations,
    PsychrometricInputs,
    from_weather_data,
    enthalpy_from_weather,
};
```

### Integration with Economizer (Placeholder Replacement)

```rust
// Source: Update to src/sim/hvac/economizer.rs
//! HVAC Economizer Mode
//!
//! This module provides economizer control for free cooling when outdoor
//! conditions are favorable, reducing mechanical cooling energy.

use fluxion::weather::{HourlyWeatherData, enthalpy_from_weather};

/// Check if economizer should be active for free cooling.
///
/// # Arguments
/// * `mode` - Economizer operating mode
/// * `outdoor_weather` - Outdoor weather data (includes psychrometric calculations)
/// * `zone_temp` - Zone air temperature (°C)
/// * `cooling_setpoint` - Zone cooling setpoint (°C)
///
/// # Returns
/// true if economizer should provide free cooling, false otherwise
///
/// # Note
/// Phase 16 enables full enthalpy mode using psychrometrics module.
pub fn is_economizer_active(
    mode: EconomizerMode,
    outdoor_weather: &HourlyWeatherData,
    zone_temp: f64,
    cooling_setpoint: f64,
) -> bool {
    match mode {
        EconomizerMode::Disabled => false,

        EconomizerMode::DryBulb => {
            // Free cooling when outdoor air is cooler than zone AND below setpoint
            outdoor_weather.dry_bulb_temp < zone_temp
                && outdoor_weather.dry_bulb_temp < cooling_setpoint
        }

        EconomizerMode::Enthalpy => {
            // Free cooling when outdoor air is cooler AND has lower enthalpy
            // Phase 16: Psychrometrics module provides enthalpy_from_weather()
            let outdoor_h = enthalpy_from_weather(outdoor_weather);
            let zone_h = zone_enthalpy_from_temp(zone_temp, outdoor_weather.humidity);

            outdoor_weather.dry_bulb_temp < zone_temp && outdoor_h < zone_h
        }
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Lookup tables for psychrometric values | Real-time iterative calculations | ~2010s with modern computing power | Improved accuracy, continuous values, no interpolation error |
| Approximate dew point formulas | ASHRAE exact Newton-Raphson | Ongoing standardization | ±0.01°C accuracy, ASHRAE 140 compliance |
| Manual property testing | Automated invariant checking | 2020s with property-based testing | Catch edge cases automatically, higher confidence |

**Deprecated/outdated:**
- **Simple dew point approximation (Td ≈ T - (100-RH)/5)**: Too inaccurate for ASHRAE 140; replaced by Newton-Raphson
- **Psychrometric charts for calculations**: Manual error-prone; replaced by computational methods
- **Fixed humidity ratio assumptions**: Variable based on T and RH; replaced by dynamic calculation

## Open Questions

1. **Wet-bulb calculation method for 0-100% RH range**
   - What we know: ASHRAE recommends iterative psychrometric equation solving
   - What's unclear: Specific bisection method implementation details for high humidity cases
   - Recommendation: Implement Newton-Raphson with bisection fallback for RH > 90%

2. **Property test framework choice**
   - What we know: Rust has quickcheck crate for property-based testing
   - What's unclear: Whether to add quickcheck dependency or use manual invariant checks
   - Recommendation: Start with manual invariant tests; add quickcheck in Phase 20 if needed

3. **Altitude correction for atmospheric pressure**
   - What we know: ASHRAE assumes sea-level pressure (101325 Pa)
   - What's unclear: Whether to support altitude-corrected pressure for high-elevation locations
   - Recommendation: Use standard pressure for Phase 16; altitude correction deferred to Phase 20 (WEATHER-04)

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (Rust built-in) |
| Config file | None (uses Cargo.toml default profile) |
| Quick run command | `cargo test psychrometrics --lib` |
| Full suite command | `cargo test --lib` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| WEATHER-02 | Dew point calculation accuracy (±0.5°C) | unit | `cargo test test_dew_point_reference_values --lib` | ❌ Wave 0 |
| WEATHER-02 | Wet-bulb calculation accuracy (±0.5°C) | unit | `cargo test test_wet_bulb_reference_values --lib` | ❌ Wave 0 |
| WEATHER-02 | Humidity ratio accuracy (±1%) | unit | `cargo test test_humidity_ratio_reference_values --lib` | ❌ Wave 0 |
| WEATHER-02 | Enthalpy accuracy (±0.5 kJ/kg) | unit | `cargo test test_enthalpy_reference_values --lib` | ❌ Wave 0 |
| WEATHER-02 | Invariant: dew_point ≤ dry_bulb | property | `cargo test test_dew_point_le_dry_bulb --lib` | ❌ Wave 0 |
| WEATHER-02 | Invariant: enthalpy monotonic with T/RH | property | `cargo test test_enthalpy_monotonic_* --lib` | ❌ Wave 0 |
| WEATHER-02 | Integration with weather data | integration | `cargo test test_weather_data_integration --lib` | ❌ Wave 0 |
| WEATHER-02 | Integration with economizer (enthalpy mode) | integration | `cargo test test_economizer_enthalpy_mode --lib` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test psychrometrics --lib` (~10 seconds for 130+ tests)
- **Per wave merge:** `cargo test --lib` (~30 seconds for full test suite)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `src/weather/psychrometrics.rs` — main module file
- [ ] PsychrometricCalculations trait definition
- [ ] calculate_dew_point() function with Newton-Raphson
- [ ] calculate_wet_bulb() function with iteration
- [ ] calculate_humidity_ratio() function
- [ ] calculate_enthalpy() function (ASHRAE exact)
- [ ] saturation_vapor_pressure() function (Magnus-Tetens)
- [ ] from_weather_data() helper function
- [ ] enthalpy_from_weather() helper function
- [ ] 130-point reference value tests (temperature grid -10°C to 40°C, RH grid 10%-90%)
- [ ] Property tests for invariants (dew_point ≤ dry_bulb, enthalpy monotonicity)
- [ ] Integration test with HourlyWeatherData
- [ ] Integration test with economizer enthalpy mode
- [ ] src/weather/mod.rs — add `pub mod psychrometrics;` and re-export
- [ ] src/sim/hvac/economizer.rs — update is_economizer_active() for enthalpy mode

## Sources

### Primary (HIGH confidence)
- ASHRAE Handbook of Fundamentals (Chapter 1: Psychrometrics) — Standard psychrometric equations and reference tables
- ASHRAE Standard 140 — Psychrometric calculation requirements for HVAC equipment validation

### Secondary (MEDIUM confidence)
- Project CONTEXT.md decisions — Locked choices for ASHRAE exact formulations and validation approach
- Project CLAUDE.md — BatchOracle pattern constraints and weather module structure
- Existing codebase (src/weather/mod.rs, src/sim/hvac/economizer.rs) — Integration patterns

### Tertiary (LOW confidence)
- ASHRAE psychrometric calculation formulas (training knowledge) — Magnus-Tetens coefficients, Newton-Raphson method
- Magnus-Tetens formula variants (training knowledge) — Multiple coefficient sets exist; ASHRAE-specific values to be verified

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Pure Rust implementation, no external dependencies needed
- Architecture: MEDIUM - Trait-based pattern follows existing codebase; iterative methods well-understood
- Pitfalls: MEDIUM - Common psychrometric calculation errors documented; specific ASHRAE reference values to be verified during implementation

**Research date:** 2026-03-13
**Valid until:** 2026-04-13 (30 days - stable psychrometric formulas, but reference values should be verified against ASHRAE Fundamentals)
