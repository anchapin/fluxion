// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! NAPI bindings for BuildingParameters - type-safe building parameter wrapper.
//!
//! Provides a JavaScript class for building design parameters with validation,
//! improving type safety and reducing misuse compared to raw number arrays.

use crate::api::parameters::BuildingParameters as CoreBuildingParameters;

/// JavaScript-accessible BuildingParameters with validation.
///
/// This class provides named properties for building design parameters, improving
/// type safety and reducing misuse compared to raw number arrays.
///
/// # TypeScript Example
/// ```typescript
/// import { BuildingParameters } from '@fluxion/native';
///
/// // Create parameters with validation
/// const params = new BuildingParameters(1.5, 20.0, 24.0);
///
/// // Access properties
/// console.log(`Window U-value: ${params.windowUValue} W/m²K`);
/// console.log(`Heating setpoint: ${params.heatingSetpoint}°C`);
/// console.log(`Cooling setpoint: ${params.coolingSetpoint}°C`);
///
/// // Convert to array for backward compatibility
/// const array = params.toVec();
/// console.log(`As array: ${array}`); // [1.5, 20.0, 24.0]
/// ```
///
/// # Field Constraints
/// - `windowUValue`: 0.1–5.0 W/m²K
/// - `heatingSetpoint`: 15.0–25.0 °C
/// - `coolingSetpoint`: 22.0–32.0 °C
/// - Heating setpoint must be less than cooling setpoint
///
/// # Typical Values
/// - Window U-value: 1.5 (double-glazed low-E) to 5.0 (single glass) W/m²K
/// - Heating setpoint: 20.0 °C for office buildings
/// - Cooling setpoint: 24.0 °C for office buildings
#[napi_derive::napi]
pub struct BuildingParameters {
    inner: CoreBuildingParameters,
}

#[napi_derive::napi]
impl BuildingParameters {
    /// Create new BuildingParameters with validation.
    ///
    /// # TypeScript Example
    /// ```typescript
    /// import { BuildingParameters } from '@fluxion/native';
    ///
    /// // Valid parameters
    /// const params = new BuildingParameters(1.5, 20.0, 24.0);
    ///
    /// // Invalid parameters - will throw ValidationError
    /// const invalid = new BuildingParameters(6.0, 20.0, 24.0); // U-value too high
    /// ```
    ///
    /// # Arguments
    /// * `windowUValue` - Window U-value (thermal transmittance) in W/m²K (range: 0.1-5.0)
    /// * `heatingSetpoint` - Heating setpoint temperature in °C (range: 15.0-25.0)
    /// * `coolingSetpoint` - Cooling setpoint temperature in °C (range: 22.0-32.0)
    ///
    /// # Throws
    /// - `ValidationError` if parameters are out of valid ranges
    #[napi(constructor)]
    pub fn new(
        window_u_value: f64,
        heating_setpoint: f64,
        cooling_setpoint: f64,
    ) -> napi::bindgen_prelude::Result<Self> {
        CoreBuildingParameters::new(window_u_value, heating_setpoint, cooling_setpoint)
            .map(|inner| BuildingParameters { inner })
            .map_err(|e| napi::bindgen_prelude::Error::from_reason(format!("Validation error: {}", e)))
    }

    /// Get window U-value (thermal transmittance) in W/m²K.
    ///
    /// Range: 0.1–5.0 W/m²K
    /// Typical values: Single glass (5.0) to triple-pane low-E (0.1)
    #[napi(getter)]
    pub fn window_u_value(&self) -> f64 {
        self.inner.window_u_value
    }

    /// Get heating setpoint temperature in °C.
    ///
    /// Range: 15.0–25.0 °C
    /// Typical value: 20.0 °C for office buildings
    #[napi(getter)]
    pub fn heating_setpoint(&self) -> f64 {
        self.inner.heating_setpoint
    }

    /// Get cooling setpoint temperature in °C.
    ///
    /// Range: 22.0–32.0 °C
    /// Typical value: 24.0 °C for office buildings
    #[napi(getter)]
    pub fn cooling_setpoint(&self) -> f64 {
        self.inner.cooling_setpoint
    }

    /// Convert parameters to array for backward compatibility.
    ///
    /// Returns array in format: `[window_u_value, heating_setpoint, cooling_setpoint]`
    ///
    /// # TypeScript Example
    /// ```typescript
    /// const params = new BuildingParameters(1.5, 20.0, 24.0);
    /// const array = params.toVec();
    ///
    /// // Use with BatchOracle
    /// const oracle = new BatchOracle();
    /// const population = [array, [2.0, 20.0, 24.0]];
    /// const results = oracle.evaluatePopulation(population, false);
    /// ```
    #[napi]
    pub fn to_vec(&self) -> Vec<f64> {
        self.inner.to_vec()
    }
}

/// NAPI constructor wrapper for BuildingParameters.
#[allow(non_snake_case)]
#[doc(hidden)]
pub fn js_constructor(
    env: napi::bindgen_prelude::Env,
    _this: napi::bindgen_prelude::CallbackInfo<void>,
) -> napi::bindgen_prelude::Result<BuildingParameters> {
    BuildingParameters::new(0.0, 0.0, 0.0) // Placeholder - actual values come from JS
}
