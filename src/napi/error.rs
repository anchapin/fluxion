// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Error type definitions for NAPI bindings.
//!
//! Provides JavaScript-accessible error classes that map to Fluxion's
//! error types, enabling proper error handling in JavaScript/TypeScript code.

use napi::bindgen_prelude::{Error as NapiError, Object};

/// Base error class for all Fluxion errors.
///
/// This is the parent class for all specific Fluxion error types and can be used
/// for catch-all error handling.
///
/// # TypeScript Example
/// ```typescript
/// try {
///   const oracle = new BatchOracle();
///   const results = oracle.evaluatePopulation(population, false);
/// } catch (error) {
///   if (error instanceof FluxionError) {
///     console.error('Fluxion error:', error.message);
///   } else {
///     throw error; // Re-throw non-Fluxion errors
///   }
/// }
/// ```
#[napi_derive::napi]
pub struct FluxionError {
    message: String,
}

#[napi_derive::napi]
impl FluxionError {
    /// Create a new FluxionError with a message.
    #[napi(constructor)]
    pub fn new(message: String) -> Self {
        FluxionError { message }
    }

    /// Get the error message.
    #[napi(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }
}

/// Error thrown when simulation parameters are invalid.
///
/// This error is thrown when building parameters violate physical constraints
/// or are outside valid ranges.
///
/// # TypeScript Example
/// ```typescript
/// try {
///   const oracle = new BatchOracle();
///
///   // Invalid: U-value too high
///   const invalidParams = [6.0, 20.0, 24.0]; // U-value > 5.0
///   oracle.validateParameters(invalidParams);
/// } catch (error) {
///   if (error instanceof ValidationError) {
///     console.error('Invalid parameters:', error.message);
///     // Output: "Window U-value (index 0, 6.00 W/m²K) out of range [0.1, 5.0] W/m²K"
///   }
/// }
/// ```
#[napi_derive::napi]
pub struct ValidationError {
    message: String,
}

#[napi_derive::napi]
impl ValidationError {
    /// Create a new ValidationError with a message.
    #[napi(constructor)]
    pub fn new(message: String) -> Self {
        ValidationError { message }
    }

    /// Get the error message.
    #[napi(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }
}

/// Error thrown when physics simulation fails.
///
/// This error is thrown when the thermal simulation encounters a numerical
/// issue or physical impossibility during execution.
///
/// # TypeScript Example
/// ```typescript
/// try {
///   const oracle = new BatchOracle();
///   const results = oracle.evaluatePopulation(population, false);
/// } catch (error) {
///   if (error instanceof SimulationError) {
///     console.error('Simulation failed:', error.message);
///     // Handle simulation failure (e.g., retry with different parameters)
///   }
/// }
/// ```
#[napi_derive::napi]
pub struct SimulationError {
    message: String,
}

#[napi_derive::napi]
impl SimulationError {
    /// Create a new SimulationError with a message.
    #[napi(constructor)]
    pub fn new(message: String) -> Self {
        SimulationError { message }
    }

    /// Get the error message.
    #[napi(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }
}

/// Error thrown when AI surrogate model evaluation fails.
///
/// This error is thrown when neural network surrogate models encounter issues,
/// such as model loading failures or inference errors.
///
/// # TypeScript Example
/// ```typescript
/// try {
///   const oracle = new BatchOracle();
///
///   // Use AI surrogates for fast evaluation
///   const results = oracle.evaluatePopulation(population, true);
/// } catch (error) {
///   if (error instanceof SurrogateError) {
///     console.error('AI surrogate failed:', error.message);
///     // Fallback to physics-based evaluation
///     const fallbackResults = oracle.evaluatePopulation(population, false);
///   }
/// }
/// ```
#[napi_derive::napi]
pub struct SurrogateError {
    message: String,
}

#[napi_derive::napi]
impl SurrogateError {
    /// Create a new SurrogateError with a message.
    #[napi(constructor)]
    pub fn new(message: String) -> Self {
        SurrogateError { message }
    }

    /// Get the error message.
    #[napi(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }
}

/// NAPI constructor wrappers for error classes.
#[allow(non_snake_case)]
#[doc(hidden)]
pub fn fluxion_error_constructor(
    env: napi::bindgen_prelude::Env,
    _this: napi::bindgen_prelude::CallbackInfo<void>,
) -> napi::bindgen_prelude::Result<FluxionError> {
    Ok(FluxionError::new("Fluxion error".to_string()))
}

#[allow(non_snake_case)]
#[doc(hidden)]
pub fn validation_error_constructor(
    env: napi::bindgen_prelude::Env,
    _this: napi::bindgen_prelude::CallbackInfo<void>,
) -> napi::bindgen_prelude::Result<ValidationError> {
    Ok(ValidationError::new("Validation error".to_string()))
}

#[allow(non_snake_case)]
#[doc(hidden)]
pub fn simulation_error_constructor(
    env: napi::bindgen_prelude::Env,
    _this: napi::bindgen_prelude::CallbackInfo<void>,
) -> napi::bindgen_prelude::Result<SimulationError> {
    Ok(SimulationError::new("Simulation error".to_string()))
}

#[allow(non_snake_case)]
#[doc(hidden)]
pub fn surrogate_error_constructor(
    env: napi::bindgen_prelude::Env,
    _this: napi::bindgen_prelude::CallbackInfo<void>,
) -> napi::bindgen_prelude::Result<SurrogateError> {
    Ok(SurrogateError::new("Surrogate error".to_string()))
}
