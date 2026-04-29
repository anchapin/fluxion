// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Error type definitions for NAPI bindings.
//!
//! Provides JavaScript-accessible error classes that map to Fluxion's
//! error types, enabling proper error handling in JavaScript/TypeScript code.

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
///     throw error;
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
