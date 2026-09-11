// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Shared FMI types: the crate-wide error enum, the export/import mode
//! selector, and the (currently informational) FMI XML namespace constant.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// FMI 2.0 model-description XML namespace.
///
/// Note: the FMI 2.0 XSD set declares no `targetNamespace`, so we do
/// NOT emit an `xmlns` attribute on the generated XML.  This constant
/// is kept for documentation / future-proofing (FMI 3.0 does use a
/// namespace) and is not currently used.
#[allow(dead_code)]
const FMI_XMLNS: &str = "http://fmi-standard.org/";

/// Errors that can occur during FMI operations.
#[derive(Debug, Error)]
pub enum FmiError {
    #[error("FMU export failed: {0}")]
    ExportFailed(String),

    #[error("FMU import failed: {0}")]
    ImportFailed(String),

    #[error("Simulation error: {0}")]
    Simulation(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("ZIP archive error: {0}")]
    ZipError(String),
}

/// FMI execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FmiMode {
    /// Export Fluxion as FMU (Co-Simulation)
    Export,
    /// Import external FMU for co-simulation
    Import,
    /// Co-simulation with Fluxion as master
    Cosimulation,
}

impl Default for FmiMode {
    fn default() -> Self {
        FmiMode::Cosimulation
    }
}
