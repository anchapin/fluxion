// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! FMI (Functional Mock-up Interface) interoperability for Fluxion.
//!
//! This module provides FMI export and co-simulation capabilities for
//! the Fluxion building energy modeling engine.
//!
//! # Scope
//!
//! This is a constrained initial implementation (IO-01 spike) with the
//! following known limitations:
//!
//! ## Export Mode (Fluxion → FMU)
//! - Single-zone thermal network only
//! - Fixed timestep (3600s = 1 hour)
//! - Outputs: zone temperature, heating/cooling load
//! - Inputs: outdoor temperature, solar gains, internal gains
//!
//! ## Co-Simulation Mode
//! - Master algorithm: first-order Euler
//! - Communication timestep: 1 hour
//! - Requires external FMU for weather data or controls
//!
//! # FMI Standard
//!
//! Implements FMI 2.0 for Co-Simulation (export) and Model Exchange (import).
//! See: https://fmi-standard.org/
//!
//! # Example
//!
//! ```ignore
//! use fluxion::interop::fmi::{FmiExporter, FmiConfig};
//!
//! let config = FmiConfig::default();
//! let exporter = FmiExporter::new(config);
//! exporter.export_fmu("fluxion_building.fmu")?;
//! ```

pub mod fmi;
pub mod gbxml;
// osm module is temporarily disabled due to compilation errors
// pub mod osm;

pub use fmi::{FmiConfig, FmiExporter, FmiMode};
pub use gbxml::{export_gbxml, import_gbxml, GbXmlError, GbXmlReader, GbXmlWriter};
