// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Node.js/NAPI bindings for Fluxion Building Energy Modeling engine.
//!
//! This module provides high-performance native Node.js bindings using napi-rs,
//! enabling JavaScript/TypeScript consumers to leverage Fluxion's 10,000+ configs/sec
//! throughput for building energy optimization workflows.
//!
//! # Architecture
//! - **NAPI-RS Framework**: Used for type-safe, zero-cost bindings
//! - **FFI-friendly API**: Direct mapping to Rust core functions
//! - **TypeScript Generation**: Automatic type definitions via napi-rs
//! - **Cross-platform**: Supports macOS (x64 + ARM), Linux, Windows
//!
//! # Performance
//! - ~2x faster than Python bindings for ONNX workloads
//! - Zero-copy data transfer where possible
//! - Multi-threaded execution preserved from Rust core
//!
//! # TypeScript Example
//! ```typescript
//! import { BatchOracle, BuildingParameters } from '@fluxion/native';
//!
//! // Create oracle instance
//! const oracle = new BatchOracle();
//!
//! // Define building parameters
//! const params = new BuildingParameters(1.5, 20.0, 24.0);
//!
//! // Evaluate population (high-throughput optimization)
//! const population = [
//!   [1.5, 20.0, 24.0],
//!   [2.0, 20.0, 24.0],
//!   [2.5, 20.0, 24.0]
//! ];
//!
//! const results = oracle.evaluatePopulation(population, false);
//! console.log(`EUI values: ${results}`); // [120.5, 115.2, 110.8]
//! ```

#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
mod batch_oracle;
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
mod building_parameters;
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
mod error;

#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
pub use batch_oracle::BatchOracle;
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
pub use building_parameters::BuildingParameters;
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
pub use error::{FluxionError, SimulationError, SurrogateError, ValidationError};

/// Register NAPI module with Node.js.
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
#[napi_derive::napi]
pub fn register(js_exports: &napi::bindgen_prelude::Object) -> napi::bindgen_prelude::Result<()> {
    let env = js_exports.env();

    // Register BatchOracle class
    let mut batch_oracle_class = env.define_class(
        "BatchOracle",
        batch_oracle::js_constructor,
        &[
            napi::bindgen_prelude::Property::new(&env, "evaluatePopulation")
                .with_method(batch_oracle::evaluate_population),
            napi::bindgen_prelude::Property::new(&env, "validateParameters")
                .with_method(batch_oracle::validate_parameters),
        ],
    )?;

    // Register BuildingParameters class
    let mut params_class = env.define_class(
        "BuildingParameters",
        building_parameters::js_constructor,
        &[
            napi::bindgen_prelude::Property::new(&env, "windowUValue")
                .with_getter(building_parameters::get_window_u_value),
            napi::bindgen_prelude::Property::new(&env, "heatingSetpoint")
                .with_getter(building_parameters::get_heating_setpoint),
            napi::bindgen_prelude::Property::new(&env, "coolingSetpoint")
                .with_getter(building_parameters::get_cooling_setpoint),
            napi::bindgen_prelude::Property::new(&env, "toVec")
                .with_method(building_parameters::to_vec),
        ],
    )?;

    // Register error classes
    env.define_class("FluxionError", error::fluxion_error_constructor, &[])?;
    env.define_class("SimulationError", error::simulation_error_constructor, &[])?;
    env.define_class("SurrogateError", error::surrogate_error_constructor, &[])?;
    env.define_class("ValidationError", error::validation_error_constructor, &[])?;

    // Export classes
    js_exports.set_named_property("BatchOracle", batch_oracle_class)?;
    js_exports.set_named_property("BuildingParameters", params_class)?;

    Ok(())
}
