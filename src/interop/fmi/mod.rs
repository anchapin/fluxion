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

mod common;
mod cosim;
mod export;
mod ffd;
mod import;
mod xml;

#[cfg(test)]
mod ffd_tests;
#[cfg(test)]
mod tests;

pub use common::{FmiError, FmiMode};
pub use cosim::{FmuCoSimulationMaster, FmuInputs, FmuOutputs};
pub use export::{FmiConfig, FmiExporter, FmiVariables, ZoneVariables};
pub use ffd::{
    ffd_fmu2DoStep, ffd_fmu2EnterInitializationMode, ffd_fmu2ExitInitializationMode,
    ffd_fmu2FreeInstance, ffd_fmu2GetReal, ffd_fmu2GetTypesPlatform, ffd_fmu2GetVersion,
    ffd_fmu2Instantiate, ffd_fmu2Reset, ffd_fmu2SetReal, ffd_fmu2SetupExperiment, fmi2Boolean,
    fmi2Status, FfdFmuCApi, FfdFmuConfig, FfdFmuExporter, FfdFmuInputs, FfdFmuOutputs, FfdFmuState,
    FfdFmuVariables, Fmi2Component, Fmi2ComponentEnvironment, Fmi2Logger, Fmi2Status,
    FFD_MAX_SURFACES, FFD_STRATIFICATION_LEVELS, FMI2_MAX_VALUE_REFERENCES,
};
pub use import::{
    import_fmu, FmiImporter, ImportedDefaultExperiment, ImportedFmu, ImportedModelDescription,
    ImportedScalarVariable,
};
