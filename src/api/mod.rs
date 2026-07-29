// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Python API support modules.
//!
//! This module contains components specifically for the Python API,
//! including parameter types, error definitions, and the unified
//! simulation schema.

pub mod email_notification;
pub mod error;
pub mod metrics;
pub mod parameters;
pub mod schema;
pub mod server;

// Re-export commonly used types
pub use error::FluxionError;
pub use parameters::BuildingParameters;
pub use schema::{
    ConstructionSet, ControlConfig, ControlSet, Geometry, SchemaMetadata, SchemaVersion,
    SimulationOutput, SimulationSchema, SimulationSchemaV1, SurfaceConstruction, WeatherData,
    WindowSpec, ZoneGeometry,
};
// Re-export REST server entrypoints (Issue #1342)
pub use server::{
    run_simulation, AppState, CampaignSpec, CampaignState, CampaignStatus,
    InMemorySimulationStateStore, SimulationStateStore,
};

#[cfg(feature = "python-bindings")]
pub use error::{FluxionErrorPy, SimulationError, SurrogateError, ValidationError};
