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
mod fmi_exporter;
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
mod gbxml_exporter;
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
mod nine_r4c_config;
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
mod osm_exporter;
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
mod state_extractor;

#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
pub use batch_oracle::BatchOracle;
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
pub use building_parameters::BuildingParameters;
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
pub use error::{FluxionError, SimulationError, SurrogateError, ValidationError};
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
pub use fmi_exporter::FmiExporter;
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
pub use gbxml_exporter::GbXmlExporter;
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
pub use nine_r4c_config::NineR4CConfig;
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
pub use osm_exporter::OsmExporter;
#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
pub use state_extractor::{StateExtractor, StateMatrices};

#[cfg(all(feature = "napi-bindings", not(target_arch = "wasm32")))]
#[napi_derive::napi]
pub fn register() -> napi::bindgen_prelude::Result<()> {
    Ok(())
}
#[cfg(not(feature = "napi-bindings"))]
pub fn register() -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
