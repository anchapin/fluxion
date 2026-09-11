// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! FMI 2.0 Co-Simulation export **and** import for Fluxion.
//!
//! ## Export (Fluxion → FMU)
//!
//! This module generates a valid FMI 2.0 [`modelDescription.xml`]
//! for one or more thermal zones and packages it into a Functional
//! Mock-up Unit (`.fmu`) ZIP archive, ready to be loaded by an
//! FMI 2.0 master such as FMPy or PyFMI.
//!
//! ## Import (FMU → Fluxion, `FmiMode::Import`)
//!
//! [`FmiImporter`] reads an exported `.fmu` archive, parses its
//! [`modelDescription.xml`] with `quick-xml`, and rebuilds a
//! [`ThermalModel`](crate::sim::engine::ThermalModel) with the correct zone count and communication
//! timestep.  A co-simulation master ([`FmuCoSimulationMaster`])
//! drives the re-imported model via [`FmuCoSimulationMaster::do_step`],
//! which is the Fluxion equivalent of the FMI 2.0 `fmi2DoStep` C
//! callback: it forwards the per-timestep weather inputs to
//! [`ThermalModel::step_physics`](crate::sim::engine::ThermalModel::step_physics) and reports zone temperature +
//! heating/cooling loads back to the master (issue #1708).
//!
//! # Design
//!
//! * **Per-zone variables** — every zone contributes 4 inputs
//!   (`outdoor_temperature`, `direct_normal_solar`,
//!   `diffuse_horizontal_solar`, `internal_gains`) and 3 outputs
//!   (`zone_temperature`, `heating_load`, `cooling_load`) — i.e.
//!   `7 × N` [`fmi2ScalarVariable`]s in total.
//! * **Configurable timestep** — the FMU's `<DefaultExperiment stepSize="…">`
//!   element is taken from [`FmiConfig::communication_timestep`]; the value
//!   is validated to be positive and is forwarded verbatim to the
//!   master.  The master is also told it can use a variable step
//!   (`canHandleVariableCommunicationStepSize="true"`).
//! * **Standalone FMU** — the FMU declares
//!   `needsExecutionTool="true"` so the master drives the simulation.
//!   No platform binary is shipped; the master tool is expected to
//!   call into Fluxion for each [`doStep`].
//!
//! [`modelDescription.xml`]: https://fmi-standard.org/docs/2.0.4/#fmi-model-description
//! [`fmi2ScalarVariable`]: https://fmi-standard.org/docs/2.0.4/#fmi2-scalarvariable
//! [`doStep`]: https://fmi-standard.org/docs/2.0.4/#fmi2-dostep

mod lifecycle;
mod marshaling;
mod model_description;

use thiserror::Error;

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

pub use lifecycle::{
    ffd_fmu2DoStep, ffd_fmu2EnterInitializationMode, ffd_fmu2ExitInitializationMode,
    ffd_fmu2FreeInstance, ffd_fmu2GetReal, ffd_fmu2GetTypesPlatform, ffd_fmu2GetVersion,
    ffd_fmu2Instantiate, ffd_fmu2Reset, ffd_fmu2SetReal, ffd_fmu2SetupExperiment, fmi2Boolean,
    fmi2Status, FfdFmuCApi, Fmi2Component, Fmi2ComponentEnvironment, Fmi2Logger, Fmi2Status,
    FmuCoSimulationMaster, FMI2_MAX_VALUE_REFERENCES,
};
pub use marshaling::{FfdFmuInputs, FfdFmuOutputs, FfdFmuState, FmuInputs, FmuOutputs};
pub use model_description::{
    import_fmu, FfdFmuConfig, FfdFmuExporter, FfdFmuVariables, FmiConfig, FmiExporter, FmiImporter,
    FmiMode, FmiVariables, ImportedDefaultExperiment, ImportedFmu, ImportedModelDescription,
    ImportedScalarVariable, ZoneVariables, FFD_MAX_SURFACES, FFD_STRATIFICATION_LEVELS,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmi_error_display() {
        let err = FmiError::ExportFailed("test error".to_string());
        assert_eq!(format!("{}", err), "FMU export failed: test error");
    }
}
