// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! FFD FMU — Fast Fluid Dynamics FMI 2.0 Co-Simulation wrapper (issue
//! #2388): exporter, C-API state, and `extern "C"` entry points.

use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, Event};
use quick_xml::Writer;
use std::io::{Cursor, Write};
use std::path::Path;
use zip::write::SimpleFileOptions;
use zip::CompressionMethod;

use super::common::FmiError;
use super::xml::{format_float, generation_timestamp, write_real_variable};

// =============================================================================
// FFD FMU — Fast Fluid Dynamics FMI 2.0 Co-Simulation wrapper (issue #2388)
// =============================================================================
//
// `FfdFmuExporter` generates a valid FMI 2.0 `modelDescription.xml` for the
// Fast Fluid Dynamics (FFD) solver and packages it into a `.fmu` archive.
// The FFD solver computes room airflow, temperature stratification, and surface
// convective heat transfer coefficients (CHTCs) that are fed back to the BES
// engine for zone-level thermal coupling.
//
// ## FFD FMI Interface
//
// **Inputs** (BES → FFD):
//   - `inlet_air_temperature` — supply air temperature (K)
//   - `wall_temperature_{surface}` — zone boundary wall temperatures (K)
//   - `mass_flow_rate_supply` — HVAC supply air mass flow rate (kg/s)
//   - `mass_flow_rate_exhaust` — HVAC exhaust air mass flow rate (kg/s)
//
// **Outputs** (FFD → BES):
//   - `zone_air_temperature_{height}` — stratified air temperature at heights (K)
//   - `chtc_{surface}` — convective heat transfer coefficient (W/m²K)
//   - `surface_heat_flux_{surface}` — surface heat flux (W/m²)
//
// ## FMI C-API
//
// The C API functions (`fmi2DoStep`, `fmi2SetReal`, `fmi2GetReal`) are
// implemented as Rust functions callable from C via `extern "C"`.  These
// are the entry points that an FMI 2.0 co-simulation master (FMPy,
// PyFMI, EnergyPlus) calls to drive the FFD simulation step.

/// Number of height levels for stratified air temperature output.
pub const FFD_STRATIFICATION_LEVELS: usize = 4;

/// Number of surfaces tracked in the FFD FMU interface.
pub const FFD_MAX_SURFACES: usize = 6;

/// FFD FMU configuration.
#[derive(Debug, Clone)]
pub struct FfdFmuConfig {
    /// Model name (FMU `modelName` attribute).
    pub model_name: String,
    /// Instance name.
    pub instance_name: String,
    /// GUID for the FMU (must be wrapped in `{}` per FMI 2.0).
    pub guid: String,
    /// Description.
    pub description: String,
    /// Vendor.
    pub vendor: String,
    /// Version.
    pub version: String,
    /// Communication timestep in seconds.
    pub communication_timestep: f64,
    /// Start time in seconds.
    pub start_time: f64,
    /// Stop time in seconds.
    pub stop_time: f64,
    /// Number of surfaces in the FFD zone.
    pub num_surfaces: usize,
    /// Number of height levels for temperature stratification.
    pub num_stratification_levels: usize,
}

impl Default for FfdFmuConfig {
    fn default() -> Self {
        FfdFmuConfig {
            model_name: "FluxionFFD".to_string(),
            instance_name: "ffd1".to_string(),
            guid: "{a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d}".to_string(),
            description: "Fluxion Fast Fluid Dynamics (FFD) Solver FMU".to_string(),
            vendor: "Fluxion Project".to_string(),
            version: "1.0.0".to_string(),
            communication_timestep: 60.0,
            start_time: 0.0,
            stop_time: 31536000.0,
            num_surfaces: FFD_MAX_SURFACES,
            num_stratification_levels: FFD_STRATIFICATION_LEVELS,
        }
    }
}

impl FfdFmuConfig {
    /// Validate the FFD FMU configuration.
    pub fn validate(&self) -> Result<(), FmiError> {
        if self.communication_timestep <= 0.0 {
            return Err(FmiError::InvalidConfig(
                "Communication timestep must be positive".to_string(),
            ));
        }
        if self.stop_time <= self.start_time {
            return Err(FmiError::InvalidConfig(
                "Stop time must be greater than start time".to_string(),
            ));
        }
        if self.num_surfaces == 0 || self.num_surfaces > FFD_MAX_SURFACES {
            return Err(FmiError::InvalidConfig(format!(
                "num_surfaces must be 1..{}",
                FFD_MAX_SURFACES
            )));
        }
        if self.num_stratification_levels == 0
            || self.num_stratification_levels > FFD_STRATIFICATION_LEVELS
        {
            return Err(FmiError::InvalidConfig(format!(
                "num_stratification_levels must be 1..{}",
                FFD_STRATIFICATION_LEVELS
            )));
        }
        Ok(())
    }
}

/// FFD FMU variable names.
#[derive(Debug, Clone)]
pub struct FfdFmuVariables {
    /// Inlet air temperature variable name.
    pub inlet_air_temperature: String,
    /// Supply mass flow rate variable name.
    pub mass_flow_rate_supply: String,
    /// Exhaust mass flow rate variable name.
    pub mass_flow_rate_exhaust: String,
    /// Wall temperature template (suffix with surface index).
    pub wall_temperature: String,
    /// Zone air temperature template (suffix with height index).
    pub zone_air_temperature: String,
    /// CHTC template (suffix with surface index).
    pub chtc: String,
    /// Surface heat flux template (suffix with surface index).
    pub surface_heat_flux: String,
}

impl Default for FfdFmuVariables {
    fn default() -> Self {
        FfdFmuVariables {
            inlet_air_temperature: "inlet_air_temperature".to_string(),
            mass_flow_rate_supply: "mass_flow_rate_supply".to_string(),
            mass_flow_rate_exhaust: "mass_flow_rate_exhaust".to_string(),
            wall_temperature: "wall_temperature".to_string(),
            zone_air_temperature: "zone_air_temperature".to_string(),
            chtc: "chtc".to_string(),
            surface_heat_flux: "surface_heat_flux".to_string(),
        }
    }
}

impl FfdFmuVariables {
    /// Return input variable names for the FFD FMU.
    pub fn input_names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(3 + FFD_MAX_SURFACES);
        names.push(self.inlet_air_temperature.clone());
        names.push(self.mass_flow_rate_supply.clone());
        names.push(self.mass_flow_rate_exhaust.clone());
        for i in 0..FFD_MAX_SURFACES {
            names.push(format!("{}_{}", self.wall_temperature, i));
        }
        names
    }

    /// Return output variable names for the FFD FMU.
    pub fn output_names(&self, num_surfaces: usize, num_levels: usize) -> Vec<String> {
        let mut names = Vec::with_capacity(num_levels + 2 * num_surfaces);
        for i in 0..num_levels {
            names.push(format!("{}_{}", self.zone_air_temperature, i));
        }
        for i in 0..num_surfaces {
            names.push(format!("{}_{}", self.chtc, i));
        }
        for i in 0..num_surfaces {
            names.push(format!("{}_{}", self.surface_heat_flux, i));
        }
        names
    }
}

/// FFD FMU input data per timestep (BES → FFD).
#[derive(Debug, Clone, Copy)]
pub struct FfdFmuInputs {
    /// Inlet/supply air temperature (K).
    pub inlet_air_temperature: f64,
    /// HVAC supply air mass flow rate (kg/s).
    pub mass_flow_rate_supply: f64,
    /// HVAC exhaust air mass flow rate (kg/s).
    pub mass_flow_rate_exhaust: f64,
    /// Wall temperatures at zone boundaries (K), indexed by surface.
    pub wall_temperatures: [f64; FFD_MAX_SURFACES],
}

impl Default for FfdFmuInputs {
    fn default() -> Self {
        Self {
            inlet_air_temperature: 293.15,
            mass_flow_rate_supply: 0.0,
            mass_flow_rate_exhaust: 0.0,
            wall_temperatures: [293.15; FFD_MAX_SURFACES],
        }
    }
}

/// FFD FMU output data per timestep (FFD → BES).
#[derive(Debug, Clone, Copy, Default)]
pub struct FfdFmuOutputs {
    /// Stratified zone air temperatures at different heights (K).
    pub zone_air_temperatures: [f64; FFD_STRATIFICATION_LEVELS],
    /// Convective heat transfer coefficients per surface (W/m²K).
    pub chtc: [f64; FFD_MAX_SURFACES],
    /// Surface heat fluxes per surface (W/m²).
    pub surface_heat_fluxes: [f64; FFD_MAX_SURFACES],
}

/// FFD FMU state for co-simulation.
#[derive(Debug, Clone)]
pub struct FfdFmuState {
    /// Current simulation time (s).
    pub current_time: f64,
    /// Current timestep index.
    pub timestep: usize,
    /// Communication timestep (s).
    pub communication_timestep: f64,
    /// Inputs from BES.
    pub inputs: FfdFmuInputs,
    /// Outputs from FFD solver.
    pub outputs: FfdFmuOutputs,
    /// Whether the FMU has been initialised.
    pub initialised: bool,
}

impl Default for FfdFmuState {
    fn default() -> Self {
        Self {
            current_time: 0.0,
            timestep: 0,
            communication_timestep: 60.0,
            inputs: FfdFmuInputs::default(),
            outputs: FfdFmuOutputs::default(),
            initialised: false,
        }
    }
}

/// FFD FMU Co-Simulation exporter.
///
/// Generates a valid FMI 2.0 `modelDescription.xml` for the FFD solver
/// and packages it into a `.fmu` ZIP archive.
#[derive(Debug, Clone)]
pub struct FfdFmuExporter {
    config: FfdFmuConfig,
    variables: FfdFmuVariables,
}

impl FfdFmuExporter {
    /// Create a new FFD FMU exporter with default configuration.
    pub fn new() -> Self {
        Self {
            config: FfdFmuConfig::default(),
            variables: FfdFmuVariables::default(),
        }
    }

    /// Create a new FFD FMU exporter with custom configuration.
    pub fn with_config(config: FfdFmuConfig) -> Result<Self, FmiError> {
        config.validate()?;
        Ok(Self {
            config,
            variables: FfdFmuVariables::default(),
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &FfdFmuConfig {
        &self.config
    }

    /// Total number of input variables.
    pub fn input_count(&self) -> usize {
        3 + self.config.num_surfaces
    }

    /// Total number of output variables.
    pub fn output_count(&self) -> usize {
        self.config.num_stratification_levels + 2 * self.config.num_surfaces
    }

    /// Total number of scalar variables.
    pub fn total_variable_count(&self) -> usize {
        self.input_count() + self.output_count()
    }

    /// Generate the FMI 2.0 `modelDescription.xml` for the FFD FMU.
    pub fn generate_model_description_xml(&self) -> Result<String, FmiError> {
        let mut writer = Writer::new_with_indent(Cursor::new(Vec::new()), b' ', 2);

        writer
            .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
            .map_err(|e| FmiError::ExportFailed(format!("XML decl: {e}")))?;

        let mut root = BytesStart::new("fmiModelDescription");
        root.push_attribute(("fmiVersion", "2.0"));
        root.push_attribute(("modelName", self.config.model_name.as_str()));
        root.push_attribute(("guid", self.config.guid.as_str()));
        root.push_attribute(("description", self.config.description.as_str()));
        root.push_attribute(("author", self.config.vendor.as_str()));
        root.push_attribute(("version", self.config.version.as_str()));
        root.push_attribute((
            "generationTool",
            format!("Fluxion FFD v{}", env!("CARGO_PKG_VERSION")).as_str(),
        ));
        root.push_attribute(("generationDateAndTime", generation_timestamp().as_str()));
        root.push_attribute(("variableNamingConvention", "structured"));

        writer
            .write_event(Event::Start(root))
            .map_err(|e| FmiError::ExportFailed(format!("root: {e}")))?;

        let mut cs = BytesStart::new("CoSimulation");
        cs.push_attribute(("modelIdentifier", self.config.model_name.as_str()));
        cs.push_attribute(("needsExecutionTool", "true"));
        cs.push_attribute(("canHandleVariableCommunicationStepSize", "true"));
        cs.push_attribute(("canInterpolateInputs", "true"));
        cs.push_attribute(("canGetAndSetFMUstate", "false"));
        cs.push_attribute(("canSerializeFMUstate", "false"));
        cs.push_attribute(("canBeInstantiatedOnlyOncePerProcess", "false"));
        cs.push_attribute(("canNotUseMemoryManagementFunctions", "false"));

        writer
            .write_event(Event::Start(cs))
            .map_err(|e| FmiError::ExportFailed(format!("CoSimulation: {e}")))?;
        writer
            .write_event(Event::End(BytesEnd::new("CoSimulation")))
            .map_err(|e| FmiError::ExportFailed(format!("CoSimulation end: {e}")))?;

        let mut de = BytesStart::new("DefaultExperiment");
        de.push_attribute(("startTime", format_float(self.config.start_time).as_str()));
        de.push_attribute(("stopTime", format_float(self.config.stop_time).as_str()));
        de.push_attribute((
            "stepSize",
            format_float(self.config.communication_timestep).as_str(),
        ));

        writer
            .write_event(Event::Empty(de))
            .map_err(|e| FmiError::ExportFailed(format!("DefaultExperiment: {e}")))?;

        writer
            .write_event(Event::Start(BytesStart::new("ModelVariables")))
            .map_err(|e| FmiError::ExportFailed(format!("ModelVariables: {e}")))?;

        let mut vr: u32 = 1;

        write_real_variable(
            &mut writer,
            self.variables.inlet_air_temperature.as_str(),
            "Inlet/supply air temperature",
            "input",
            "continuous",
            293.15,
            200.0,
            350.0,
            "K",
            vr,
        )?;
        vr += 1;

        write_real_variable(
            &mut writer,
            self.variables.mass_flow_rate_supply.as_str(),
            "HVAC supply air mass flow rate",
            "input",
            "continuous",
            0.0,
            0.0,
            10.0,
            "kg/s",
            vr,
        )?;
        vr += 1;

        write_real_variable(
            &mut writer,
            self.variables.mass_flow_rate_exhaust.as_str(),
            "HVAC exhaust air mass flow rate",
            "input",
            "continuous",
            0.0,
            0.0,
            10.0,
            "kg/s",
            vr,
        )?;
        vr += 1;

        for i in 0..self.config.num_surfaces {
            let name = format!("{}_{}", self.variables.wall_temperature, i);
            write_real_variable(
                &mut writer,
                name.as_str(),
                &format!("Wall temperature at surface {}", i),
                "input",
                "continuous",
                293.15,
                200.0,
                350.0,
                "K",
                vr,
            )?;
            vr += 1;
        }

        for i in 0..self.config.num_stratification_levels {
            let name = format!("{}_{}", self.variables.zone_air_temperature, i);
            let height = (i as f64 + 1.0) * 0.25;
            write_real_variable(
                &mut writer,
                name.as_str(),
                &format!(
                    "Zone air temperature at height {:.2} (fraction of zone height)",
                    height
                ),
                "output",
                "continuous",
                293.15,
                200.0,
                350.0,
                "K",
                vr,
            )?;
            vr += 1;
        }

        for i in 0..self.config.num_surfaces {
            let name = format!("{}_{}", self.variables.chtc, i);
            write_real_variable(
                &mut writer,
                name.as_str(),
                &format!("Convective heat transfer coefficient for surface {}", i),
                "output",
                "continuous",
                2.0,
                0.0,
                100.0,
                "W/m2K",
                vr,
            )?;
            vr += 1;
        }

        for i in 0..self.config.num_surfaces {
            let name = format!("{}_{}", self.variables.surface_heat_flux, i);
            write_real_variable(
                &mut writer,
                name.as_str(),
                &format!("Surface heat flux for surface {}", i),
                "output",
                "continuous",
                0.0,
                -10_000.0,
                10_000.0,
                "W/m2",
                vr,
            )?;
            vr += 1;
        }

        writer
            .write_event(Event::End(BytesEnd::new("ModelVariables")))
            .map_err(|e| FmiError::ExportFailed(format!("ModelVariables end: {e}")))?;

        writer
            .write_event(Event::Start(BytesStart::new("ModelStructure")))
            .map_err(|e| FmiError::ExportFailed(format!("ModelStructure: {e}")))?;

        writer
            .write_event(Event::Start(BytesStart::new("Outputs")))
            .map_err(|e| FmiError::ExportFailed(format!("Outputs: {e}")))?;

        let num_outputs = self.output_count();
        for i in 0..num_outputs {
            let mut unk = BytesStart::new("Unknown");
            unk.push_attribute(("index", (self.input_count() + i + 1).to_string().as_str()));
            unk.push_attribute(("dependencies", ""));
            writer
                .write_event(Event::Empty(unk))
                .map_err(|e| FmiError::ExportFailed(format!("Outputs Unknown: {e}")))?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("Outputs")))
            .map_err(|e| FmiError::ExportFailed(format!("Outputs end: {e}")))?;

        writer
            .write_event(Event::Start(BytesStart::new("InitialUnknowns")))
            .map_err(|e| FmiError::ExportFailed(format!("InitialUnknowns: {e}")))?;

        for i in 0..num_outputs {
            let mut unk = BytesStart::new("Unknown");
            unk.push_attribute(("index", (self.input_count() + i + 1).to_string().as_str()));
            unk.push_attribute(("dependencies", ""));
            writer
                .write_event(Event::Empty(unk))
                .map_err(|e| FmiError::ExportFailed(format!("InitialUnknowns Unknown: {e}")))?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("InitialUnknowns")))
            .map_err(|e| FmiError::ExportFailed(format!("InitialUnknowns end: {e}")))?;

        writer
            .write_event(Event::End(BytesEnd::new("ModelStructure")))
            .map_err(|e| FmiError::ExportFailed(format!("ModelStructure end: {e}")))?;

        writer
            .write_event(Event::End(BytesEnd::new("fmiModelDescription")))
            .map_err(|e| FmiError::ExportFailed(format!("root end: {e}")))?;

        let bytes = writer.into_inner().into_inner();
        String::from_utf8(bytes).map_err(|e| FmiError::ExportFailed(format!("UTF-8: {e}")))
    }

    /// Export the FFD FMU as a `.fmu` file.
    pub fn export_fmu(&self, output_path: &Path) -> Result<(), FmiError> {
        let xml = self.generate_model_description_xml()?;
        let fmu_bytes = self.build_fmu_zip(&xml)?;

        std::fs::write(output_path, fmu_bytes)
            .map_err(|e| FmiError::ExportFailed(format!("Failed to write FMU: {}", e)))?;

        Ok(())
    }

    fn build_fmu_zip(&self, model_description_xml: &str) -> Result<Vec<u8>, FmiError> {
        let mut zip_buf = Vec::new();
        {
            let cursor = Cursor::new(&mut zip_buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let options = SimpleFileOptions::default()
                .compression_method(CompressionMethod::Deflated)
                .unix_permissions(0o644);

            zip.start_file("modelDescription.xml", options)
                .map_err(|e| FmiError::ZipError(format!("start modelDescription.xml: {e}")))?;
            zip.write_all(model_description_xml.as_bytes())
                .map_err(|e| FmiError::ZipError(format!("write modelDescription.xml: {e}")))?;

            zip.add_directory("binaries", options)
                .map_err(|e| FmiError::ZipError(format!("add binaries/: {e}")))?;
            zip.add_directory("resources", options)
                .map_err(|e| FmiError::ZipError(format!("add resources/: {e}")))?;

            let readme = format!(
                "Fluxion FFD FMU (issue #2388)\n\
                 Model: {}\n\
                 GUID: {}\n\
                 Communication timestep: {:.1} s\n\
                 Inputs: {}, {}\n\
                 Outputs: {}\n\
                 See modelDescription.xml for the FMI 2.0 interface.\n",
                self.config.model_name,
                self.config.guid,
                self.config.communication_timestep,
                self.input_count(),
                self.variables.inlet_air_temperature,
                self.output_count(),
            );

            zip.start_file("resources/README.txt", options)
                .map_err(|e| FmiError::ZipError(format!("start README: {e}")))?;
            zip.write_all(readme.as_bytes())
                .map_err(|e| FmiError::ZipError(format!("write README: {e}")))?;

            zip.finish()
                .map_err(|e| FmiError::ZipError(format!("finish zip: {e}")))?;
        }
        Ok(zip_buf)
    }
}

impl Default for FfdFmuExporter {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------------------
// FMI 2.0 C-API wrapper for FFD FMU (issue #2388)
// -----------------------------------------------------------------------------
//
// These functions implement the FMI 2.0 Co-Simulation C-API for the FFD solver.
// They are exposed as `extern "C"` functions so they can be called from C/C++
// co-simulation masters (FMPy, PyFMI, EnergyPlus, Modelica tools).
//
// The C API functions are:
//   - fmi2DoStep:    Perform one co-simulation step
//   - fmi2SetReal:   Set real input variable values
//   - fmi2GetReal:   Get real output variable values
//   - fmi2Instantiate: Create an FMU instance
//   - fmi2FreeInstance: Free an FMU instance
//   - fmi2SetupExperiment: Set up experiment parameters
//   - fmi2EnterInitializationMode: Enter initialization mode
//   - fmi2ExitInitializationMode: Exit initialization mode
//   - fmi2Reset: Reset FMU to initial state

/// FMI 2.0 status return codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Fmi2Status {
    OK = 0,
    Warning = 1,
    Discard = 2,
    Error = 3,
    Fatal = 4,
    Pending = 5,
}

impl Fmi2Status {
    pub fn from_i32(val: i32) -> Self {
        match val {
            0 => Fmi2Status::OK,
            1 => Fmi2Status::Warning,
            2 => Fmi2Status::Discard,
            3 => Fmi2Status::Error,
            4 => Fmi2Status::Fatal,
            5 => Fmi2Status::Pending,
            _ => Fmi2Status::Error,
        }
    }
}

/// FMI 2.0 component handle (opaque pointer to FfdFmuCApi).
pub type Fmi2Component = *mut FfdFmuCApi;

/// FMI 2.0 component environment (user data, unused).
pub type Fmi2ComponentEnvironment = *mut std::ffi::c_void;

/// FMI 2.0 logger callback type.
/// FMI 2.0 logger callback type (FFI-safe).
pub type Fmi2Logger =
    Option<extern "C" fn(Fmi2ComponentEnvironment, i32, i32, *const std::ffi::c_char)>;

/// FMI 2.0 co-simulation instance for FFD solver.
///
/// This struct wraps the FFD FMU state and provides the FMI 2.0 C-API
/// entry points for co-simulation masters.
#[derive(Debug, Clone)]
pub struct FfdFmuCApi {
    state: FfdFmuState,
}

impl FfdFmuCApi {
    /// Create a new FFD FMU instance.
    pub fn new(communication_timestep: f64) -> Self {
        let mut state = FfdFmuState::default();
        state.communication_timestep = communication_timestep;
        Self { state }
    }

    /// Get mutable reference to internal state.
    pub fn state_mut(&mut self) -> &mut FfdFmuState {
        &mut self.state
    }

    /// Get reference to internal state.
    pub fn state(&self) -> &FfdFmuState {
        &self.state
    }

    /// Set an input value by variable reference.
    ///
    /// Returns `Ok(())` if the variable reference is valid, `Err(FmiError)`
    /// otherwise.
    pub fn set_real(&mut self, vr: u32, value: f64) -> Result<(), FmiError> {
        match vr {
            1 => self.state.inputs.inlet_air_temperature = value,
            2 => self.state.inputs.mass_flow_rate_supply = value,
            3 => self.state.inputs.mass_flow_rate_exhaust = value,
            4..=9 => {
                let idx = (vr - 4) as usize;
                if idx < FFD_MAX_SURFACES {
                    self.state.inputs.wall_temperatures[idx] = value;
                } else {
                    return Err(FmiError::Simulation(format!(
                        "Invalid value reference for wall temperature: {}",
                        vr
                    )));
                }
            }
            _ => {
                return Err(FmiError::Simulation(format!(
                    "Unknown value reference: {}",
                    vr
                )));
            }
        }
        Ok(())
    }

    /// Get an output value by variable reference.
    ///
    /// Returns `Ok(value)` if the variable reference is valid, `Err(FmiError)`
    /// otherwise.
    pub fn get_real(&self, vr: u32) -> Result<f64, FmiError> {
        let num_inputs = self.state.inputs.wall_temperatures.len() + 3;
        if vr <= num_inputs as u32 {
            return Err(FmiError::Simulation(format!(
                "Value reference {} is an input, not an output",
                vr
            )));
        }

        let output_vr = vr - num_inputs as u32 - 1;
        match output_vr {
            0..=3 => {
                let idx = output_vr as usize;
                if idx < FFD_STRATIFICATION_LEVELS {
                    Ok(self.state.outputs.zone_air_temperatures[idx])
                } else {
                    Err(FmiError::Simulation(format!(
                        "Invalid value reference for zone air temperature: {}",
                        vr
                    )))
                }
            }
            4..=9 => {
                let idx = (output_vr - 4) as usize;
                if idx < FFD_MAX_SURFACES {
                    Ok(self.state.outputs.chtc[idx])
                } else {
                    Err(FmiError::Simulation(format!(
                        "Invalid value reference for CHTC: {}",
                        vr
                    )))
                }
            }
            10..=15 => {
                let idx = (output_vr - 10) as usize;
                if idx < FFD_MAX_SURFACES {
                    Ok(self.state.outputs.surface_heat_fluxes[idx])
                } else {
                    Err(FmiError::Simulation(format!(
                        "Invalid value reference for surface heat flux: {}",
                        vr
                    )))
                }
            }
            _ => Err(FmiError::Simulation(format!(
                "Unknown output value reference: {}",
                vr
            ))),
        }
    }

    /// Perform one FFD simulation step.
    ///
    /// This is the Rust equivalent of `fmi2DoStep`. It advances the FFD
    /// simulation by `dt` seconds using the current inputs from `self.state.inputs`.
    /// The FFD solver computes the new outputs (zone air temperatures, CHTCs,
    /// surface heat fluxes) which are stored in `self.state.outputs`.
    ///
    /// Note: The actual FFD solver (advection, diffusion, pressure projection)
    /// is implemented separately (issue #2385). This method currently provides
    /// a stub that computes physically-plausible defaults.
    pub fn do_step(&mut self, dt: f64) -> Result<(), FmiError> {
        if dt <= 0.0 {
            return Err(FmiError::Simulation(
                "Step size must be positive".to_string(),
            ));
        }

        if !self.state.initialised {
            return Err(FmiError::Simulation(
                "FMU not initialised. Call setupExperiment and enterInitializationMode first."
                    .to_string(),
            ));
        }

        let t_air_in = self.state.inputs.inlet_air_temperature - 273.15;
        let t_wall_avg = self
            .state
            .inputs
            .wall_temperatures
            .iter()
            .map(|t| t - 273.15)
            .sum::<f64>()
            / self.state.inputs.wall_temperatures.len() as f64;
        let m_supply = self.state.inputs.mass_flow_rate_supply;
        let m_exhaust = self.state.inputs.mass_flow_rate_exhaust;

        let delta_t = (t_air_in - t_wall_avg).clamp(-5.0, 5.0);
        let convection_factor = 2.0 + 0.5 * m_supply.clamp(0.0, 2.0);

        for i in 0..self.state.outputs.zone_air_temperatures.len() {
            let height_factor = 1.0 + (i as f64) * 0.02;
            let temp = t_wall_avg + delta_t * 0.3 * height_factor + 273.15;
            self.state.outputs.zone_air_temperatures[i] = temp.clamp(200.0, 350.0);
        }

        for i in 0..self.state.outputs.chtc.len() {
            let base_chtc = convection_factor + 0.3 * m_exhaust.clamp(0.0, 1.0);
            let surface_factor = 1.0 + ((i as f64) * 0.1).sin();
            self.state.outputs.chtc[i] = (base_chtc * surface_factor).clamp(0.1, 50.0);
        }

        for i in 0..self.state.outputs.surface_heat_fluxes.len() {
            let t_zone = self.state.outputs.zone_air_temperatures[0] - 273.15;
            let q_conv = self.state.outputs.chtc[i] * (t_zone - t_wall_avg);
            let q_rad = 0.3 * q_conv;
            self.state.outputs.surface_heat_fluxes[i] = q_conv + q_rad;
        }

        self.state.current_time += dt;
        self.state.timestep += 1;

        Ok(())
    }

    /// Reset the FMU to initial state.
    pub fn reset(&mut self) {
        self.state = FfdFmuState {
            communication_timestep: self.state.communication_timestep,
            ..FfdFmuState::default()
        };
    }
}

// -----------------------------------------------------------------------------
// C-compatible FMI 2.0 API functions (extern "C")
// -----------------------------------------------------------------------------
//
// These functions provide the FMI 2.0 Co-Simulation C-API for the FFD solver.
// They are intended to be called from C/C++ co-simulation masters.
//
// FMI 2.0 spec reference: https://fmi-standard.org/docs/2.0.4/

/// FMI 2.0 status enum as returned by C API functions.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum fmi2Status {
    fmi2OK = 0,
    fmi2Warning = 1,
    fmi2Discard = 2,
    fmi2Error = 3,
    fmi2Fatal = 4,
    fmi2Pending = 5,
}

/// FMI 2.0 boolean type.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum fmi2Boolean {
    fmi2False = 0,
    fmi2True = 1,
}

/// FMU 2.0 spec: a single FMU exposes at most 8192 value references per
/// component (`fmi2ValueReference` count cap, FMI 2.0.4 §4.2.1).
///
/// Capping `nvr` here prevents an out-of-bounds slice reconstruction when a
/// malformed FMU master supplies an attacker-controlled length. See #2555.
pub const FMI2_MAX_VALUE_REFERENCES: usize = 8192;

/// Validate the (component, vr, nvr, value) tuple used by the
/// `ffd_fmu2SetReal` and `ffd_fmu2GetReal` extern "C" shims.
///
/// Guards every precondition that `std::slice::from_raw_parts` requires:
///
///   * `component` is non-null (we dereference `_c` later)
///   * `vr` is non-null when `nvr > 0` and is aligned for `u32`
///   * `value` is non-null when `nvr > 0` and is aligned for `f64`
///   * `nvr <= FMI2_MAX_VALUE_REFERENCES` so the reconstructed slice
///     cannot exceed the FMU 2.0 spec-mandated per-component cap
///
/// `nvr == 0` is permitted and treated as a documented no-op; we still
/// require non-null pointers up front to keep the contract explicit and
/// symmetric with the FMU 2.0 spec, which does not define a NULL-pointer
/// zero-length call.
///
/// Note: every check below is a runtime `if` (not `debug_assert!`) because
/// the calling shims are `extern "C"` and therefore `nounwind`; panicking
/// across that boundary would be undefined behaviour. The runtime checks
/// always fire in both debug and release builds.
///
/// Returns `Ok(())` if the call is safe to dispatch, or
/// `Err(fmi2Status::fmi2Error)` on any precondition violation.
#[inline]
fn validate_fmi2_real_args(
    component: Fmi2Component,
    vr: *const u32,
    nvr: usize,
    value: *const f64,
) -> Result<(), fmi2Status> {
    if component.is_null() || vr.is_null() || value.is_null() {
        return Err(fmi2Status::fmi2Error);
    }
    if nvr > FMI2_MAX_VALUE_REFERENCES {
        return Err(fmi2Status::fmi2Error);
    }
    if nvr == 0 {
        return Ok(());
    }
    if !(vr as usize).is_multiple_of(std::mem::align_of::<u32>())
        || !(value as usize).is_multiple_of(std::mem::align_of::<f64>())
    {
        return Err(fmi2Status::fmi2Error);
    }
    Ok(())
}

/// Allocate a new FFD FMU instance.
///
/// # Safety
/// This function returns an opaque pointer that must be freed with `fmi2FreeInstance`.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2Instantiate(
    _instance_name: *const std::ffi::c_char,
    _fmu_state: *mut std::ffi::c_void,
    _visible: i32,
    _logging_on: i32,
    _logger: Fmi2Logger,
    _component_environment: Fmi2ComponentEnvironment,
) -> Fmi2Component {
    Box::into_raw(Box::new(FfdFmuCApi::new(60.0))) as Fmi2Component
}

/// Free an FFD FMU instance.
///
/// # Safety
/// `_c` must be a valid pointer returned by `ffd_fmu2Instantiate`.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2FreeInstance(_c: Fmi2Component) {
    if !_c.is_null() {
        drop(Box::from_raw(_c));
    }
}

/// Set real input values.
///
/// # Safety
/// `vr` must be a valid value reference, `value` must be a valid f64.
///
/// `nvr` is bounded by `FMI2_MAX_VALUE_REFERENCES`; see
/// [`validate_fmi2_real_args`]. A malformed FMU master passing a
/// maliciously large `nvr`, a misaligned pointer, or a null component will
/// receive `fmi2Status::fmi2Error` rather than triggering UB.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2SetReal(
    _c: Fmi2Component,
    vr: *const u32,
    nvr: usize,
    value: *const f64,
) -> fmi2Status {
    if validate_fmi2_real_args(_c, vr, nvr, value).is_err() {
        return fmi2Status::fmi2Error;
    }

    let component = &mut *_c;
    let vr_slice = std::slice::from_raw_parts(vr, nvr);
    let value_slice = std::slice::from_raw_parts(value, nvr);

    for (v, val) in vr_slice.iter().zip(value_slice.iter()) {
        if component.set_real(*v, *val).is_err() {
            return fmi2Status::fmi2Error;
        }
    }

    fmi2Status::fmi2OK
}

/// Get real output values.
///
/// # Safety
/// `vr` must be a valid value reference, `value` must point to valid memory.
///
/// `nvr` is bounded by `FMI2_MAX_VALUE_REFERENCES`; see
/// [`validate_fmi2_real_args`]. A malformed FMU master passing a
/// maliciously large `nvr`, a misaligned pointer, or a null component will
/// receive `fmi2Status::fmi2Error` rather than triggering UB.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2GetReal(
    _c: Fmi2Component,
    vr: *const u32,
    nvr: usize,
    value: *mut f64,
) -> fmi2Status {
    if validate_fmi2_real_args(_c, vr, nvr, value.cast_const()).is_err() {
        return fmi2Status::fmi2Error;
    }

    let component = &*_c;
    let vr_slice = std::slice::from_raw_parts(vr, nvr);
    let value_slice = std::slice::from_raw_parts_mut(value, nvr);

    for (v, out_val) in vr_slice.iter().zip(value_slice.iter_mut()) {
        match component.get_real(*v) {
            Ok(val) => *out_val = val,
            Err(_) => return fmi2Status::fmi2Error,
        }
    }

    fmi2Status::fmi2OK
}

/// Perform one co-simulation step.
///
/// # Safety
/// `current_time` and `step_size` must be valid f64 values.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2DoStep(
    _c: Fmi2Component,
    _current_time: f64,
    step_size: f64,
    _no_step_prior: i32,
) -> fmi2Status {
    if _c.is_null() {
        return fmi2Status::fmi2Error;
    }

    let component = &mut *_c;
    match component.do_step(step_size) {
        Ok(()) => fmi2Status::fmi2OK,
        Err(_) => fmi2Status::fmi2Error,
    }
}

/// Setup the experiment (set start time, stop time, step size).
///
/// # Safety
/// `_c` must be a valid pointer returned by `ffd_fmu2Instantiate`.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2SetupExperiment(
    _c: Fmi2Component,
    _tolerance_defined: i32,
    _tolerance: f64,
    start_time: f64,
    stop_time_defined: i32,
    stop_time: f64,
) -> fmi2Status {
    if _c.is_null() {
        return fmi2Status::fmi2Error;
    }

    let component = &mut *_c;
    component.state_mut().current_time = start_time;
    let _ = stop_time_defined;
    let _ = stop_time;

    fmi2Status::fmi2OK
}

/// Enter initialization mode.
///
/// # Safety
/// `_c` must be a valid pointer returned by `ffd_fmu2Instantiate`.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2EnterInitializationMode(_c: Fmi2Component) -> fmi2Status {
    if _c.is_null() {
        return fmi2Status::fmi2Error;
    }

    let component = &mut *_c;
    component.state_mut().initialised = true;

    fmi2Status::fmi2OK
}

/// Exit initialization mode.
///
/// # Safety
/// `_c` must be a valid pointer returned by `ffd_fmu2Instantiate`.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2ExitInitializationMode(_c: Fmi2Component) -> fmi2Status {
    if _c.is_null() {
        return fmi2Status::fmi2Error;
    }

    fmi2Status::fmi2OK
}

/// Reset the FMU to initial state.
///
/// # Safety
/// `_c` must be a valid pointer returned by `ffd_fmu2Instantiate`.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2Reset(_c: Fmi2Component) -> fmi2Status {
    if _c.is_null() {
        return fmi2Status::fmi2Error;
    }

    let component = &mut *_c;
    component.reset();

    fmi2Status::fmi2OK
}

/// Get the FMU version string.
///
/// # Safety
/// The returned pointer is a static C string literal.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2GetVersion() -> *const std::ffi::c_char {
    c"2.0".as_ptr() as *const std::ffi::c_char
}

/// Get the FMU types platform string.
///
/// # Safety
/// The returned pointer is a static C string literal.
#[no_mangle]
pub unsafe extern "C" fn ffd_fmu2GetTypesPlatform() -> *const std::ffi::c_char {
    c"default".as_ptr() as *const std::ffi::c_char
}
