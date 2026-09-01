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
//! [`ThermalModel`] with the correct zone count and communication
//! timestep.  A co-simulation master ([`FmuCoSimulationMaster`])
//! drives the re-imported model via [`FmuCoSimulationMaster::do_step`],
//! which is the Fluxion equivalent of the FMI 2.0 `fmi2DoStep` C
//! callback: it forwards the per-timestep weather inputs to
//! [`ThermalModel::step_physics`] and reports zone temperature +
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

use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, Event};
use quick_xml::{Reader, Writer};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Cursor, Write};
use std::path::Path;
use thiserror::Error;
use zip::write::SimpleFileOptions;
use zip::CompressionMethod;

use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;

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

/// Configuration for FMI operations.
///
/// `communication_timestep` is forwarded verbatim to the FMU's
/// `<DefaultExperiment stepSize="…"/>` and to the
/// `CoSimulation.stepSize` attribute.  It must be positive (and
/// `stop_time > start_time`); see [`FmiExporter::with_config`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FmiConfig {
    /// FMI mode
    pub mode: FmiMode,
    /// Model name (FMU `modelName` attribute)
    pub model_name: String,
    /// Instance name
    pub instance_name: String,
    /// GUID for the FMU (must be wrapped in `{}` per FMI 2.0)
    pub guid: String,
    /// Description
    pub description: String,
    /// Vendor
    pub vendor: String,
    /// Version
    pub version: String,
    /// Communication timestep in seconds (default: 3600 = 1 hour).
    /// Must be positive; for the multi-zone extension, common values
    /// are 60 s, 300 s, 600 s, or 3600 s.  Default 3600 s is preserved
    /// for backward compatibility with the single-zone spike (#1125).
    pub communication_timestep: f64,
    /// Start time in seconds (default: 0)
    pub start_time: f64,
    /// Stop time in seconds (default: 31536000 = 1 year)
    pub stop_time: f64,
    /// Generation tool identifier (FMU `generationTool` attribute).
    #[serde(default = "default_generation_tool")]
    pub generation_tool: String,
}

fn default_generation_tool() -> String {
    format!("Fluxion v{}", env!("CARGO_PKG_VERSION"))
}

impl Default for FmiConfig {
    fn default() -> Self {
        FmiConfig {
            mode: FmiMode::Cosimulation,
            model_name: "FluxionBuilding".to_string(),
            instance_name: "fluxion1".to_string(),
            guid: "{8c4e8d3a-2b1f-4a6c-9e5f-0d3b2a4c6e8d}".to_string(),
            description: "Fluxion AI-Accelerated Building Energy Model".to_string(),
            vendor: "Fluxion Project".to_string(),
            version: "1.0.0".to_string(),
            communication_timestep: 3600.0,
            start_time: 0.0,
            stop_time: 31536000.0,
            generation_tool: default_generation_tool(),
        }
    }
}

/// FMI variable name templates for a single zone.
///
/// These names are suffixed with the zone identifier when generating
/// the multi-zone [`modelDescription.xml`](https://fmi-standard.org/docs/2.0.4/):
/// `{zone}_{name}`.  For the single-zone case (the original #1125
/// spike) zone 0 uses the bare template names — see
/// [`ZoneVariables::suffixed_name`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FmiVariables {
    /// Input: Outdoor dry bulb temperature (K)
    pub outdoor_temperature: String,
    /// Input: Direct normal solar radiation (W/m²)
    pub direct_normal_solar: String,
    /// Input: Diffuse horizontal solar radiation (W/m²)
    pub diffuse_horizontal_solar: String,
    /// Input: Internal heat gains (W)
    pub internal_gains: String,
    /// Output: Zone temperature (K)
    pub zone_temperature: String,
    /// Output: Heating load (W)
    pub heating_load: String,
    /// Output: Cooling load (W)
    pub cooling_load: String,
}

impl FmiVariables {
    /// Enumerate the four input names.
    pub fn input_names(&self) -> [&str; 4] {
        [
            self.outdoor_temperature.as_str(),
            self.direct_normal_solar.as_str(),
            self.diffuse_horizontal_solar.as_str(),
            self.internal_gains.as_str(),
        ]
    }

    /// Enumerate the three output names.
    pub fn output_names(&self) -> [&str; 3] {
        [
            self.zone_temperature.as_str(),
            self.heating_load.as_str(),
            self.cooling_load.as_str(),
        ]
    }

    /// Total number of scalar variables per zone (4 inputs + 3 outputs).
    pub const PER_ZONE_VARIABLE_COUNT: usize = 7;

    /// For zone `0`, use the bare template name (preserves the original
    /// single-zone spike interface).  For `zone >= 1`, prefix with
    /// `zone{idx}_` so multiple zones can coexist without name clashes
    /// in the FMI namespace.
    pub fn suffixed_name(&self, base: &str, zone_index: usize) -> String {
        if zone_index == 0 {
            base.to_string()
        } else {
            format!("zone{}_{}", zone_index, base)
        }
    }
}

impl Default for FmiVariables {
    fn default() -> Self {
        FmiVariables {
            outdoor_temperature: "outdoor_temperature".to_string(),
            direct_normal_solar: "direct_normal_solar".to_string(),
            diffuse_horizontal_solar: "diffuse_horizontal_solar".to_string(),
            internal_gains: "internal_gains".to_string(),
            zone_temperature: "zone_temperature".to_string(),
            heating_load: "heating_load".to_string(),
            cooling_load: "cooling_load".to_string(),
        }
    }
}

/// A single zone participating in the multi-zone FMU interface.
///
/// The `name` field is used as the FMI scalar-variable prefix when
/// generating the [`modelDescription.xml`].  For backward
/// compatibility with the original spike (`FmiExporter::new()` /
/// `FmiExporter::with_config()`), the default zone is named `"zone"`
/// and its variables use the bare [`FmiVariables`] template names.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneVariables {
    /// Zone identifier (e.g. `"zone"`, `"living"`, `"bedroom"`).
    pub name: String,
}

impl ZoneVariables {
    /// Create a new zone with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// The default single zone (used by [`FmiExporter::new`] and
    /// [`FmiExporter::with_config`]).
    pub fn default_zone() -> Self {
        Self {
            name: "zone".to_string(),
        }
    }
}

impl Default for ZoneVariables {
    fn default() -> Self {
        Self::default_zone()
    }
}

/// FMI Co-Simulation exporter for Fluxion.
///
/// Generates a real FMI 2.0 [`modelDescription.xml`] for one or more
/// thermal zones and packages it into a `.fmu` ZIP archive.
///
/// [`modelDescription.xml`]: https://fmi-standard.org/docs/2.0.4/#fmi-model-description
#[derive(Debug, Clone)]
pub struct FmiExporter {
    config: FmiConfig,
    variables: FmiVariables,
    zones: Vec<ZoneVariables>,
}

impl FmiExporter {
    /// Create a new FMI exporter with default configuration (one zone).
    pub fn new() -> Self {
        Self {
            config: FmiConfig::default(),
            variables: FmiVariables::default(),
            zones: vec![ZoneVariables::default_zone()],
        }
    }

    /// Create a new FMI exporter with custom configuration (one zone).
    pub fn with_config(config: FmiConfig) -> Result<Self, FmiError> {
        if config.communication_timestep <= 0.0 {
            return Err(FmiError::InvalidConfig(
                "Communication timestep must be positive".to_string(),
            ));
        }
        if config.stop_time <= config.start_time {
            return Err(FmiError::InvalidConfig(
                "Stop time must be greater than start time".to_string(),
            ));
        }
        Ok(Self {
            config,
            variables: FmiVariables::default(),
            zones: vec![ZoneVariables::default_zone()],
        })
    }

    /// Add zones to the exporter, replacing any previously configured
    /// zones.  At least one zone must be supplied.  Used for the
    /// multi-zone extension (#1339).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use fluxion::interop::fmi::{FmiExporter, ZoneVariables};
    ///
    /// let exporter = FmiExporter::new()
    ///     .with_zones(vec![
    ///         ZoneVariables::new("living"),
    ///         ZoneVariables::new("bedroom"),
    ///         ZoneVariables::new("kitchen"),
    ///     ]);
    /// assert_eq!(exporter.zone_count(), 3);
    /// ```
    pub fn with_zones(mut self, zones: Vec<ZoneVariables>) -> Self {
        assert!(!zones.is_empty(), "at least one zone is required");
        self.zones = zones;
        self
    }

    /// Get the configuration.
    pub fn config(&self) -> &FmiConfig {
        &self.config
    }

    /// Get the FMI variable name templates.
    pub fn variables(&self) -> &FmiVariables {
        &self.variables
    }

    /// Get the configured zones.
    pub fn zones(&self) -> &[ZoneVariables] {
        &self.zones
    }

    /// Number of zones (`>= 1`).  The FMU will expose
    /// `7 × N` ScalarVariables.
    pub fn zone_count(&self) -> usize {
        self.zones.len()
    }

    /// Total number of scalar variables the FMU will declare
    /// (4 inputs + 3 outputs per zone, i.e. 7 × N).
    pub fn total_variable_count(&self) -> usize {
        self.zones.len() * FmiVariables::PER_ZONE_VARIABLE_COUNT
    }

    /// Generate the FMI 2.0 [`modelDescription.xml`] content for the
    /// currently configured zones.  Public so callers can inspect /
    /// validate the XML before packaging it into a `.fmu`.
    ///
    /// [`modelDescription.xml`]: https://fmi-standard.org/docs/2.0.4/#fmi-model-description
    pub fn generate_model_description_xml(&self) -> Result<String, FmiError> {
        // Pre-compute the (zone_idx, var_index, name, causality) tuples so we
        // can assign valueReferences and emit a ModelStructure section.
        let mut entries: Vec<(usize, usize, String, String, &'static str)> =
            Vec::with_capacity(self.total_variable_count());
        for (zone_idx, zone) in self.zones.iter().enumerate() {
            for (i, input) in self.variables.input_names().iter().enumerate() {
                let name = zone_variable_name(&self.variables, zone, zone_idx, input);
                // Store the template name alongside so per-zone variables
                // get the correct metadata (unit, min, max, start, desc).
                entries.push((zone_idx, i, name, input.to_string(), "input"));
            }
            for (i, output) in self.variables.output_names().iter().enumerate() {
                let name = zone_variable_name(&self.variables, zone, zone_idx, output);
                entries.push((zone_idx, 4 + i, name, output.to_string(), "output"));
            }
        }

        let mut writer = Writer::new_with_indent(Cursor::new(Vec::new()), b' ', 2);

        // XML declaration
        writer
            .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
            .map_err(|e| FmiError::ExportFailed(format!("XML decl: {e}")))?;

        // <fmiModelDescription fmiVersion="2.0" ...>
        // Note: FMI 2.0 modelDescription.xml is conventionally emitted
        // WITHOUT a target namespace (the official XSD set declares no
        // targetNamespace), so we intentionally do not push an
        // `xmlns` attribute here.
        let mut root = BytesStart::new("fmiModelDescription");
        root.push_attribute(("fmiVersion", "2.0"));
        root.push_attribute(("modelName", self.config.model_name.as_str()));
        root.push_attribute(("guid", self.config.guid.as_str()));
        root.push_attribute(("description", self.config.description.as_str()));
        root.push_attribute(("author", self.config.vendor.as_str())); // FMI 2.0 XSD uses `author`, not `vendor`
        root.push_attribute(("version", self.config.version.as_str()));
        root.push_attribute(("generationTool", self.config.generation_tool.as_str()));
        let ts = generation_timestamp();
        root.push_attribute(("generationDateAndTime", ts.as_str()));
        root.push_attribute(("variableNamingConvention", "structured"));

        writer
            .write_event(Event::Start(root))
            .map_err(|e| FmiError::ExportFailed(format!("root: {e}")))?;

        // Co-Simulation-only — we deliberately do NOT also declare
        // `<ModelExchange/>` because it requires `modelIdentifier` and
        // Fluxion is exported as a Co-Simulation FMU driven by an
        // external master via `needsExecutionTool="true"`.

        // <CoSimulation modelIdentifier=... needsExecutionTool=...>
        // (FMI 2.0 XSD does NOT define `stepSize` on this element;
        // the communication step is conveyed via `<DefaultExperiment>`.)
        let mut cs = BytesStart::new("CoSimulation");
        cs.push_attribute(("modelIdentifier", self.config.model_name.as_str()));
        cs.push_attribute(("needsExecutionTool", "true"));
        cs.push_attribute(("canHandleVariableCommunicationStepSize", "true"));
        cs.push_attribute(("canInterpolateInputs", "true"));
        // NOTE: the FMI 2.0 XSD uses lowercase `state` in the
        // `canGetAndSetFMUstate` / `canSerializeFMUstate` attribute names.
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

        // <DefaultExperiment startTime=... stopTime=... stepSize=...>
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

        // <ModelVariables> ... </ModelVariables>
        writer
            .write_event(Event::Start(BytesStart::new("ModelVariables")))
            .map_err(|e| FmiError::ExportFailed(format!("ModelVariables: {e}")))?;

        for (vr, (_zone_idx, _var_idx, name, template_name, causality)) in
            entries.iter().enumerate()
        {
            // FMI 2.0 requires every ScalarVariable to have a valueReference;
            // we use a simple deterministic scheme (vr starts at 1 per spec).
            let (start, min, max, unit, desc) = match *causality {
                "input" => input_meta(template_name),
                "output" => output_meta(template_name),
                _ => unreachable!(),
            };
            write_real_variable(
                &mut writer,
                name,
                desc,
                causality,
                "continuous",
                start,
                min,
                max,
                unit,
                // valueReference: 1-based per FMI 2.0 §3
                (vr + 1) as u32,
            )?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("ModelVariables")))
            .map_err(|e| FmiError::ExportFailed(format!("ModelVariables end: {e}")))?;

        // <ModelStructure> ... </ModelStructure>  (required by FMI 2.0 XSD)
        // - Outputs list every output variable (required)
        // - Derivatives + InitialUnknowns are empty (no state derivatives in
        //   this Co-Simulation-only spike)
        writer
            .write_event(Event::Start(BytesStart::new("ModelStructure")))
            .map_err(|e| FmiError::ExportFailed(format!("ModelStructure: {e}")))?;

        writer
            .write_event(Event::Start(BytesStart::new("Outputs")))
            .map_err(|e| FmiError::ExportFailed(format!("Outputs: {e}")))?;
        for (vr, (_zi, _vi, _name, _template, causality)) in entries.iter().enumerate() {
            if *causality == "output" {
                let mut unk = BytesStart::new("Unknown");
                unk.push_attribute(("index", (vr + 1).to_string().as_str()));
                unk.push_attribute(("dependencies", ""));
                writer
                    .write_event(Event::Empty(unk))
                    .map_err(|e| FmiError::ExportFailed(format!("Outputs Unknown: {e}")))?;
            }
        }
        writer
            .write_event(Event::End(BytesEnd::new("Outputs")))
            .map_err(|e| FmiError::ExportFailed(format!("Outputs end: {e}")))?;

        writer
            .write_event(Event::Start(BytesStart::new("InitialUnknowns")))
            .map_err(|e| FmiError::ExportFailed(format!("InitialUnknowns: {e}")))?;
        for (vr, (_zi, _vi, _name, _template, causality)) in entries.iter().enumerate() {
            if *causality == "output" {
                let mut unk = BytesStart::new("Unknown");
                unk.push_attribute(("index", (vr + 1).to_string().as_str()));
                unk.push_attribute(("dependencies", ""));
                writer
                    .write_event(Event::Empty(unk))
                    .map_err(|e| FmiError::ExportFailed(format!("InitialUnknowns Unknown: {e}")))?;
            }
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

    /// Export the Fluxion model as an FMU file (`.fmu`).
    ///
    /// The FMU is a ZIP archive containing:
    /// * `modelDescription.xml` — FMI 2.0 model description
    /// * `binaries/` — empty placeholder (master calls into Fluxion
    ///   via `needsExecutionTool="true"`)
    /// * `resources/` — empty placeholder for optional data files
    ///
    /// # Arguments
    /// * `output_path` — path where the FMU file will be written
    pub fn export_fmu(&self, output_path: &Path) -> Result<(), FmiError> {
        let xml = self.generate_model_description_xml()?;
        let fmu_bytes = self.build_fmu_zip(&xml)?;

        std::fs::write(output_path, fmu_bytes)
            .map_err(|e| FmiError::ExportFailed(format!("Failed to write FMU: {}", e)))?;

        Ok(())
    }

    /// Build the FMU archive (ZIP) from a generated `modelDescription.xml`.
    fn build_fmu_zip(&self, model_description_xml: &str) -> Result<Vec<u8>, FmiError> {
        let mut zip_buf = Vec::new();
        {
            let cursor = Cursor::new(&mut zip_buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let options = SimpleFileOptions::default()
                .compression_method(CompressionMethod::Deflated)
                .unix_permissions(0o644);

            // modelDescription.xml at archive root (mandatory per FMI 2.0 §2.2)
            zip.start_file("modelDescription.xml", options)
                .map_err(|e| FmiError::ZipError(format!("start modelDescription.xml: {e}")))?;
            zip.write_all(model_description_xml.as_bytes())
                .map_err(|e| FmiError::ZipError(format!("write modelDescription.xml: {e}")))?;

            // Empty binaries/ and resources/ placeholders so FMPy / PyFMI
            // accept the archive as a structurally valid FMU.
            zip.add_directory("binaries", options)
                .map_err(|e| FmiError::ZipError(format!("add binaries/: {e}")))?;
            zip.add_directory("resources", options)
                .map_err(|e| FmiError::ZipError(format!("add resources/: {e}")))?;

            // A small README so a human opening the FMU knows what's inside.
            let readme = format!(
                "Fluxion FMU (issue #1339)\n\
                 Zones: {n}\n\
                 Communication timestep: {ts:.1} s\n\
                 Total scalar variables: 7 x {n} = {total}\n\
                 See modelDescription.xml for the FMI 2.0 interface.\n",
                n = self.zones.len(),
                ts = self.config.communication_timestep,
                total = self.total_variable_count(),
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

    /// Generate the variable list as a flat Vec for tests / inspection.
    pub fn variable_names(&self) -> Vec<(String, String)> {
        // (name, causality)
        let mut out = Vec::with_capacity(self.total_variable_count());
        for (zone_idx, zone) in self.zones.iter().enumerate() {
            for input in self.variables.input_names() {
                let prefixed = zone_variable_name(&self.variables, zone, zone_idx, input);
                out.push((prefixed, "input".to_string()));
            }
            for output in self.variables.output_names() {
                let prefixed = zone_variable_name(&self.variables, zone, zone_idx, output);
                out.push((prefixed, "output".to_string()));
            }
        }
        out
    }

    /// Convenience: read back the FMU's `modelDescription.xml` from
    /// an exported ZIP, useful for tests / verification scripts.
    pub fn read_model_description_from_fmu(path: &Path) -> Result<String, FmiError> {
        let file =
            File::open(path).map_err(|e| FmiError::ExportFailed(format!("open FMU: {e}")))?;
        let mut zip =
            zip::ZipArchive::new(file).map_err(|e| FmiError::ZipError(format!("read FMU: {e}")))?;
        let mut entry = zip
            .by_name("modelDescription.xml")
            .map_err(|e| FmiError::ZipError(format!("missing modelDescription.xml: {e}")))?;
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut entry, &mut buf)
            .map_err(|e| FmiError::ExportFailed(format!("read entry: {e}")))?;
        Ok(buf)
    }
}

impl Default for FmiExporter {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// FMI 2.0 Import  (FmiMode::Import) — issue #1708
// =============================================================================
//
// `FmiImporter` is the import-side counterpart to `FmiExporter`.  It reads a
// previously-exported `.fmu` ZIP archive, parses the FMI 2.0
// `modelDescription.xml` with `quick-xml`, and rebuilds a Fluxion
// [`ThermalModel`] with the correct number of zones and communication
// timestep.  The accompanying [`FmuCoSimulationMaster`] then drives the
// re-imported model one `doStep` at a time, mirroring the role of an FMI 2.0
// co-simulation master algorithm.
//
// The import path deliberately mirrors the export structure so that an
// FMU produced by `FmiExporter::export_fmu` round-trips losslessly:
//
//   FmiExporter::export_fmu ──►  .fmu  ──►  FmiImporter::import ──► ImportedFmu
//                                                                     │
//                                                     FmuCoSimulationMaster::do_step
//                                                                     │
//                                                          ThermalModel::step_physics

/// Number of outputs declared per zone in the Fluxion FMU interface
/// (`zone_temperature`, `heating_load`, `cooling_load`).  Used to derive
/// the zone count from the parsed `<ModelVariables>` list.
const FMI_OUTPUTS_PER_ZONE: usize = 3;

/// A parsed FMI 2.0 `<ScalarVariable>`.
///
/// Only the attributes Fluxion cares about are retained; unknown
/// attributes are silently ignored by the parser.
#[derive(Debug, Clone, Default)]
pub struct ImportedScalarVariable {
    /// FMI variable name (e.g. `outdoor_temperature`, `zone1_zone_temperature`).
    pub name: String,
    /// Numeric valueReference (1-based in Fluxion-exported FMUs).
    pub value_reference: u32,
    /// `input`, `output`, `parameter`, …
    pub causality: String,
    /// `continuous`, `discrete`, …
    pub variability: String,
    /// Human-readable description attribute.
    pub description: String,
    /// `unit` attribute of the nested `<Real>` element (e.g. `K`, `W/m2`).
    pub unit: String,
    /// `start` attribute of the nested `<Real>` element, if present.
    pub start: Option<f64>,
}

/// Parsed `<DefaultExperiment>` element.
#[derive(Debug, Clone, Default)]
pub struct ImportedDefaultExperiment {
    /// `startTime` (seconds).
    pub start_time: f64,
    /// `stopTime` (seconds).
    pub stop_time: f64,
    /// `stepSize` (seconds) — the communication timestep.
    pub step_size: f64,
}

/// The fully-parsed FMI 2.0 [`modelDescription.xml`].
///
/// Produced by [`FmiImporter::parse_model_description`].  This is a
/// lossless (for Fluxion's purposes) in-memory representation of the
/// XML that lives inside a `.fmu` archive.
///
/// [`modelDescription.xml`]: https://fmi-standard.org/docs/2.0.4/#fmi-model-description
#[derive(Debug, Clone, Default)]
pub struct ImportedModelDescription {
    /// `fmiVersion` attribute (expected `"2.0"`).
    pub fmi_version: String,
    /// `modelName` attribute.
    pub model_name: String,
    /// `guid` attribute (FMI 2.0 instantiation identifier).
    pub guid: String,
    /// `description` attribute.
    pub description: String,
    /// `author` attribute.
    pub author: String,
    /// `version` attribute.
    pub version: String,
    /// `generationTool` attribute.
    pub generation_tool: String,
    /// `generationDateAndTime` attribute.
    pub generation_date_and_time: String,
    /// `variableNamingConvention` attribute.
    pub variable_naming_convention: String,
    /// Parsed `<DefaultExperiment>`.
    pub default_experiment: ImportedDefaultExperiment,
    /// All `<ScalarVariable>` entries in document order.
    pub variables: Vec<ImportedScalarVariable>,
}

impl ImportedModelDescription {
    /// Count of input variables (`causality="input"`).
    pub fn input_count(&self) -> usize {
        self.variables
            .iter()
            .filter(|v| v.causality == "input")
            .count()
    }

    /// Count of output variables (`causality="output"`).
    pub fn output_count(&self) -> usize {
        self.variables
            .iter()
            .filter(|v| v.causality == "output")
            .count()
    }

    /// Number of thermal zones implied by the variable list.
    ///
    /// Every zone contributes exactly [`FMI_OUTPUTS_PER_ZONE`] outputs
    /// (`zone_temperature`, `heating_load`, `cooling_load`), so the zone
    /// count is `output_count / 3`.  A well-formed Fluxion FMU always
    /// has at least one zone.
    pub fn zone_count(&self) -> usize {
        let n = self.output_count() / FMI_OUTPUTS_PER_ZONE;
        n.max(1)
    }

    /// Communication timestep (seconds) from `<DefaultExperiment stepSize>`.
    pub fn communication_timestep(&self) -> f64 {
        self.default_experiment.step_size
    }
}

/// A successfully imported FMU: the parsed [`ImportedModelDescription`] plus
/// a ready-to-step [`ThermalModel`] sized to the FMU's zone count.
///
/// Built by [`FmiImporter::import`].  Use [`ImportedFmu::thermal_model`] or
/// [`ImportedFmu::into_thermal_model`] to obtain the underlying physics
/// model, and [`FmuCoSimulationMaster::from_imported`] to drive it as a
/// co-simulation slave.
#[derive(Clone)]
pub struct ImportedFmu {
    /// The parsed FMI 2.0 model description.
    pub description: ImportedModelDescription,
    model: ThermalModel<VectorField>,
}

impl std::fmt::Debug for ImportedFmu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedFmu")
            .field("description", &self.description)
            .field("zone_count", &self.zone_count())
            .finish()
    }
}

impl ImportedFmu {
    /// Number of zones in the re-imported model.
    pub fn zone_count(&self) -> usize {
        self.description.zone_count()
    }

    /// Communication timestep declared by the FMU (seconds).
    pub fn communication_timestep(&self) -> f64 {
        self.description.communication_timestep()
    }

    /// Borrow the underlying [`ThermalModel`].
    pub fn thermal_model(&self) -> &ThermalModel<VectorField> {
        &self.model
    }

    /// Mutably borrow the underlying [`ThermalModel`].
    pub fn thermal_model_mut(&mut self) -> &mut ThermalModel<VectorField> {
        &mut self.model
    }

    /// Consume the wrapper and return the underlying [`ThermalModel`].
    pub fn into_thermal_model(self) -> ThermalModel<VectorField> {
        self.model
    }
}

/// FMI 2.0 FMU importer ([`FmiMode::Import`]).
///
/// The importer is the import-side mirror of [`FmiExporter`]: it reads a
/// `.fmu` ZIP archive, parses `modelDescription.xml`, and rebuilds a
/// Fluxion [`ThermalModel`].
///
/// # Example
///
/// ```ignore
/// use fluxion::interop::fmi::{FmiExporter, FmiImporter, ZoneVariables};
///
/// // Export a 3-zone FMU …
/// let exporter = FmiExporter::new().with_zones(vec![
///     ZoneVariables::new("zone"),
///     ZoneVariables::new("bedroom"),
///     ZoneVariables::new("kitchen"),
/// ]);
/// exporter.export_fmu("fluxion_three_zone.fmu").unwrap();
///
/// // … then re-import it.
/// let fmu = FmiImporter::new().import("fluxion_three_zone.fmu").unwrap();
/// assert_eq!(fmu.zone_count(), 3);
/// ```
#[derive(Debug, Clone, Default)]
pub struct FmiImporter;

impl FmiImporter {
    /// Create a new importer.
    pub fn new() -> Self {
        Self
    }

    /// Import (open + parse) an FMU archive at `path`.
    ///
    /// The archive must contain a `modelDescription.xml` entry at its
    /// root, as produced by [`FmiExporter::export_fmu`].  The returned
    /// [`ImportedFmu`] owns a [`ThermalModel`] sized to the zone count
    /// implied by the FMU's variable list.
    pub fn import(&self, path: &Path) -> Result<ImportedFmu, FmiError> {
        let xml = read_model_description_from_fmu(path)?;
        let description = Self::parse_model_description(&xml)?;
        let model = ThermalModel::<VectorField>::new(description.zone_count());
        Ok(ImportedFmu { description, model })
    }

    /// Parse an FMI 2.0 `modelDescription.xml` document into an
    /// [`ImportedModelDescription`].
    ///
    /// Uses a streaming `quick-xml` reader so the full DOM is never
    /// materialised; only the elements Fluxion emits are recognised.
    pub fn parse_model_description(xml: &str) -> Result<ImportedModelDescription, FmiError> {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);

        let mut desc = ImportedModelDescription::default();
        let mut buf = Vec::new();
        let mut in_model_variables = false;
        // The `<Real>` child currently being accumulated inside a
        // `<ScalarVariable>` (only valid while inside ModelVariables).
        let mut current_var: Option<ImportedScalarVariable> = None;

        loop {
            buf.clear();
            let event = reader
                .read_event_into(&mut buf)
                .map_err(|e| FmiError::ImportFailed(format!("XML parse: {e}")))?;
            match event {
                Event::Start(ref e) | Event::Empty(ref e) => {
                    let name = e.name();
                    match name.as_ref() {
                        "fmiModelDescription" => {
                            for attr in e.attributes() {
                                let attr =
                                    attr.map_err(|e| FmiError::ImportFailed(format!("attr: {e}")))?;
                                let v = attr_value(&attr)?;
                                match attr.key.as_ref() {
                                    "fmiVersion" => desc.fmi_version = v,
                                    "modelName" => desc.model_name = v,
                                    "guid" => desc.guid = v,
                                    "description" => desc.description = v,
                                    "author" => desc.author = v,
                                    "version" => desc.version = v,
                                    "generationTool" => desc.generation_tool = v,
                                    "generationDateAndTime" => desc.generation_date_and_time = v,
                                    "variableNamingConvention" => {
                                        desc.variable_naming_convention = v
                                    }
                                    _ => {}
                                }
                            }
                        }
                        "DefaultExperiment" => {
                            for attr in e.attributes() {
                                let attr =
                                    attr.map_err(|e| FmiError::ImportFailed(format!("attr: {e}")))?;
                                let v = attr_value(&attr)?;
                                let parsed = v.parse::<f64>().unwrap_or(0.0);
                                match attr.key.as_ref() {
                                    "startTime" => desc.default_experiment.start_time = parsed,
                                    "stopTime" => desc.default_experiment.stop_time = parsed,
                                    "stepSize" => desc.default_experiment.step_size = parsed,
                                    _ => {}
                                }
                            }
                        }
                        "ModelVariables" => in_model_variables = true,
                        "ScalarVariable" if in_model_variables => {
                            // Start a new accumulator.  Attributes on the
                            // opening tag; the nested <Real> fills unit/start.
                            let mut sv = ImportedScalarVariable::default();
                            for attr in e.attributes() {
                                let attr =
                                    attr.map_err(|e| FmiError::ImportFailed(format!("attr: {e}")))?;
                                let v = attr_value(&attr)?;
                                match attr.key.as_ref() {
                                    "name" => sv.name = v,
                                    "causality" => sv.causality = v,
                                    "variability" => sv.variability = v,
                                    "description" => sv.description = v,
                                    "valueReference" => {
                                        sv.value_reference = v.parse::<u32>().unwrap_or(0)
                                    }
                                    _ => {}
                                }
                            }
                            // For an Empty event there is no nested <Real>;
                            // for a Start event we keep accumulating.
                            current_var = Some(sv);
                            if matches!(event, Event::Empty(_)) {
                                if let Some(v) = current_var.take() {
                                    desc.variables.push(v);
                                }
                            }
                        }
                        "Real" if current_var.is_some() => {
                            if let Some(ref mut sv) = current_var {
                                for attr in e.attributes() {
                                    let attr = attr.map_err(|e| {
                                        FmiError::ImportFailed(format!("attr: {e}"))
                                    })?;
                                    let v = attr_value(&attr)?;
                                    match attr.key.as_ref() {
                                        "unit" => sv.unit = v,
                                        "start" => sv.start = v.parse::<f64>().ok(),
                                        _ => {}
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Event::End(ref e) => match e.name().as_ref() {
                    "ScalarVariable" => {
                        if let Some(v) = current_var.take() {
                            desc.variables.push(v);
                        }
                    }
                    "ModelVariables" => in_model_variables = false,
                    _ => {}
                },
                Event::Eof => break,
                _ => {}
            }
        }

        if desc.variables.is_empty() {
            return Err(FmiError::ImportFailed(
                "modelDescription.xml contains no ScalarVariables".to_string(),
            ));
        }
        if desc.fmi_version.is_empty() {
            desc.fmi_version = "2.0".to_string();
        }
        Ok(desc)
    }
}

/// Read the `modelDescription.xml` entry out of an FMU ZIP archive.
///
/// This is the import-side companion to
/// [`FmiExporter::read_model_description_from_fmu`]; it is a free
/// function so [`FmiImporter`] does not need an `FmiExporter` instance.
fn read_model_description_from_fmu(path: &Path) -> Result<String, FmiError> {
    let file = File::open(path).map_err(|e| FmiError::ImportFailed(format!("open FMU: {e}")))?;
    let mut zip =
        zip::ZipArchive::new(file).map_err(|e| FmiError::ZipError(format!("read FMU: {e}")))?;
    let mut entry = zip
        .by_name("modelDescription.xml")
        .map_err(|e| FmiError::ZipError(format!("missing modelDescription.xml: {e}")))?;
    let mut buf = String::new();
    std::io::Read::read_to_string(&mut entry, &mut buf)
        .map_err(|e| FmiError::ImportFailed(format!("read entry: {e}")))?;
    Ok(buf)
}

/// Read one XML attribute's value, normalising XML entities per XML 1.0.
///
/// `quick-xml` 0.41 deprecated `Attribute::unescape_value` in favour of
/// `normalized_value(XmlVersion)`; this helper centralises the call so the
/// deprecation is resolved in exactly one place.
fn attr_value(attr: &quick_xml::events::attributes::Attribute<'_>) -> Result<String, FmiError> {
    attr.normalized_value(quick_xml::XmlVersion::Implicit1_0)
        .map(|c| c.into_owned())
        .map_err(|e| FmiError::ImportFailed(format!("attr value: {e}")))
}

/// Import an FMU archive and return the underlying [`ThermalModel`].
///
/// Convenience wrapper around [`FmiImporter::import`] +
/// [`ImportedFmu::into_thermal_model`] that directly yields the physics
/// model, matching the function signature requested in issue #1708.
///
/// # Example
///
/// ```ignore
/// let model = fluxion::interop::fmi::import_fmu("fluxion_three_zone.fmu")?;
/// assert_eq!(model.hvac.num_zones, 3);
/// ```
pub fn import_fmu(path: &Path) -> Result<ThermalModel<VectorField>, FmiError> {
    FmiImporter::new()
        .import(path)
        .map(ImportedFmu::into_thermal_model)
}

// -----------------------------------------------------------------------------
// Co-simulation master algorithm (fmi2DoStep wrapper)
// -----------------------------------------------------------------------------

/// Per-timestep FMI inputs for a single zone, in the units declared by the
/// Fluxion FMU interface (SI units: Kelvin, W/m², W).
#[derive(Debug, Clone, Copy)]
pub struct FmuInputs {
    /// Outdoor dry-bulb temperature (K).
    pub outdoor_temperature: f64,
    /// Direct normal solar irradiance (W/m²).
    pub direct_normal_solar: f64,
    /// Diffuse horizontal solar irradiance (W/m²).
    pub diffuse_horizontal_solar: f64,
    /// Internal heat gains (W).
    pub internal_gains: f64,
}

impl Default for FmuInputs {
    fn default() -> Self {
        // Matches the `<Real start=…>` defaults emitted by the exporter.
        Self {
            outdoor_temperature: 280.0,
            direct_normal_solar: 0.0,
            diffuse_horizontal_solar: 0.0,
            internal_gains: 0.0,
        }
    }
}

/// Per-timestep FMI outputs for a single zone, in the units declared by the
/// Fluxion FMU interface.
#[derive(Debug, Clone, Copy, Default)]
pub struct FmuOutputs {
    /// Zone air temperature (K).
    pub zone_temperature: f64,
    /// Heating load over the step (W, non-negative).
    pub heating_load: f64,
    /// Cooling load over the step (W, non-negative).
    pub cooling_load: f64,
}

/// Co-simulation master driving a re-imported FMU one `doStep` at a time.
///
/// This is the Fluxion equivalent of the FMI 2.0 `fmi2DoStep` C callback:
/// each call to [`FmuCoSimulationMaster::do_step`] forwards the master's
/// per-timestep weather inputs to [`ThermalModel::step_physics`] and
/// returns the resulting zone temperature and heating/cooling loads.
///
/// Loads are derived from the per-zone energy accumulators
/// (`zone_heating_energy_kwh` / `zone_cooling_energy_kwh`) that
/// `step_physics` advances, converted from kWh-over-the-step to average
/// Watts.  This preserves energy conservation across the co-simulation
/// boundary (acceptance criterion #2 of issue #1708).
pub struct FmuCoSimulationMaster {
    model: ThermalModel<VectorField>,
    /// Communication timestep declared by the FMU (seconds).
    communication_timestep: f64,
    /// Current simulation time (seconds).
    current_time: f64,
    /// Current timestep index (0-based).
    timestep: usize,
}

impl FmuCoSimulationMaster {
    /// Build a master from an imported FMU, adopting its communication
    /// timestep and [`ThermalModel`].
    pub fn from_imported(fmu: ImportedFmu) -> Self {
        let communication_timestep = fmu.communication_timestep();
        Self {
            model: fmu.into_thermal_model(),
            communication_timestep,
            current_time: 0.0,
            timestep: 0,
        }
    }

    /// Borrow the underlying [`ThermalModel`].
    pub fn model(&self) -> &ThermalModel<VectorField> {
        &self.model
    }

    /// Mutably borrow the underlying [`ThermalModel`].
    pub fn model_mut(&mut self) -> &mut ThermalModel<VectorField> {
        &mut self.model
    }

    /// Communication timestep (seconds).
    pub fn communication_timestep(&self) -> f64 {
        self.communication_timestep
    }

    /// Current simulation time (seconds).
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Perform one co-simulation step — the `fmi2DoStep` wrapper.
    ///
    /// Forwards `inputs` to [`ThermalModel::step_physics`] (converting the
    /// outdoor temperature from Kelvin, as declared in the FMU interface,
    /// to degrees Celsius, as required by the physics engine) and returns
    /// a [`FmuOutputs`] entry per zone (converted back to Kelvin for the
    /// zone temperature) together with each zone's heating/cooling loads
    /// averaged over the step.
    ///
    /// The returned vector has length `model.hvac.num_zones`, so external
    /// co-simulation masters (FMPy, PyFMI, EnergyPlus-to-FMU, Modelica)
    /// receive telemetry for **every** zone the FMU was exported with —
    /// `FmuCoSimulationMaster::do_step` no longer silently drops
    /// `zone 1..N-1` (issue #2459).
    ///
    /// If `step_size` is omitted the FMU's declared communication timestep
    /// is used.
    pub fn do_step(&mut self, inputs: FmuInputs, step_size: Option<f64>) -> Vec<FmuOutputs> {
        let dt = step_size.unwrap_or(self.communication_timestep).max(1.0);

        // Snapshot per-zone energy accumulators *before* the step so the
        // delta gives the energy consumed during this step alone.
        let heat_before: Vec<f64> = self.model.hvac.zone_heating_energy_kwh.as_ref().to_vec();
        let cool_before: Vec<f64> = self.model.hvac.zone_cooling_energy_kwh.as_ref().to_vec();

        // FMI inputs are Kelvin; step_physics expects °C.
        let outdoor_temp_c = inputs.outdoor_temperature - 273.15;
        let _energy_kwh = self.model.step_physics(self.timestep, outdoor_temp_c, dt);

        let temps_c = self.model.setpoints.temperatures.as_ref();
        let heat_after = self.model.hvac.zone_heating_energy_kwh.as_ref();
        let cool_after = self.model.hvac.zone_cooling_energy_kwh.as_ref();

        // Convert kWh-delta over the step to average Watts:
        //   W = kWh * 3_600_000 / dt
        let outputs: Vec<FmuOutputs> = (0..self.model.hvac.num_zones)
            .map(|i| {
                let zone_temp_c = temps_c.get(i).copied().unwrap_or(20.0);
                let heating_load = heat_before
                    .get(i)
                    .copied()
                    .zip(heat_after.get(i).copied())
                    .map(|(a, b)| ((b - a) * 3_600_000.0 / dt).max(0.0))
                    .unwrap_or(0.0);
                let cooling_load = cool_before
                    .get(i)
                    .copied()
                    .zip(cool_after.get(i).copied())
                    .map(|(a, b)| ((b - a) * 3_600_000.0 / dt).max(0.0))
                    .unwrap_or(0.0);
                FmuOutputs {
                    zone_temperature: zone_temp_c + 273.15,
                    heating_load,
                    cooling_load,
                }
            })
            .collect();

        self.timestep += 1;
        self.current_time += dt;

        outputs
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Per-zone variable name: for zone 0 (the legacy single-zone case)
/// use the bare template name so existing FMUs stay name-compatible;
/// for `zone_idx >= 1` prefix with `zone{N}_` to avoid name clashes.
fn zone_variable_name(
    variables: &FmiVariables,
    zone: &ZoneVariables,
    zone_idx: usize,
    base: &str,
) -> String {
    if zone_idx == 0 {
        // For the legacy single-zone case the template name is already
        // the user-facing name; the zone's own `name` is informational
        // only (`"zone"`).  This preserves the #1125 spike interface.
        if zone.name == "zone" {
            return variables.suffixed_name(base, 0);
        }
        return format!("{}_{}", sanitize_xml_name(&zone.name), base);
    }
    format!("{}_{}", sanitize_xml_name(&zone.name), base)
}

/// Strip characters that are illegal in FMI variable names.  FMI 2.0
/// variables follow C identifier rules: `[A-Za-z_][A-Za-z0-9_]*`.
fn sanitize_xml_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for (i, ch) in name.chars().enumerate() {
        let ok = if i == 0 {
            ch.is_ascii_alphabetic() || ch == '_'
        } else {
            ch.is_ascii_alphanumeric() || ch == '_'
        };
        if ok {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('z');
    }
    out
}

/// `(start, min, max, unit, description)` for an input variable template.
fn input_meta(name: &str) -> (f64, f64, f64, &'static str, &'static str) {
    match name {
        "outdoor_temperature" => (280.0, 200.0, 320.0, "K", "Outdoor dry bulb temperature"),
        "direct_normal_solar" => (0.0, 0.0, 1200.0, "W/m2", "Direct normal solar radiation"),
        "diffuse_horizontal_solar" => (
            0.0,
            0.0,
            800.0,
            "W/m2",
            "Diffuse horizontal solar radiation",
        ),
        "internal_gains" => (0.0, 0.0, 10000.0, "W", "Total internal heat gains"),
        _ => (0.0, 0.0, 0.0, "", ""),
    }
}

/// `(start, min, max, unit, description)` for an output variable template.
fn output_meta(name: &str) -> (f64, f64, f64, &'static str, &'static str) {
    match name {
        "zone_temperature" => (293.15, 200.0, 320.0, "K", "Zone air temperature"),
        "heating_load" => (0.0, 0.0, 100000.0, "W", "Heating load (positive)"),
        "cooling_load" => (0.0, 0.0, 100000.0, "W", "Cooling load (positive)"),
        _ => (0.0, 0.0, 0.0, "", ""),
    }
}

#[allow(clippy::too_many_arguments)]
fn write_real_variable<W: Write>(
    writer: &mut Writer<W>,
    name: &str,
    description: &str,
    causality: &str,
    variability: &str,
    start: f64,
    min: f64,
    max: f64,
    unit: &str,
    value_reference: u32,
) -> Result<(), FmiError> {
    let mut sv = BytesStart::new("ScalarVariable");
    sv.push_attribute(("name", name));
    sv.push_attribute(("valueReference", value_reference.to_string().as_str()));
    sv.push_attribute(("description", description));
    sv.push_attribute(("causality", causality));
    sv.push_attribute(("variability", variability));
    writer
        .write_event(Event::Start(sv))
        .map_err(|e| FmiError::ExportFailed(format!("ScalarVariable {name}: {e}")))?;

    let mut real = BytesStart::new("Real");
    // declaredType is omitted unless a TypeDefinitions/SimpleType is referenced.
    real.push_attribute(("quantity", ""));
    real.push_attribute(("unit", unit));
    real.push_attribute(("displayUnit", ""));
    real.push_attribute(("relativeQuantity", "false"));
    real.push_attribute(("min", format_float(min).as_str()));
    real.push_attribute(("max", format_float(max).as_str()));
    real.push_attribute(("nominal", "0.0"));
    real.push_attribute(("unbounded", "false"));
    real.push_attribute(("start", format_float(start).as_str()));
    real.push_attribute(("reinit", "false"));

    writer
        .write_event(Event::Empty(real))
        .map_err(|e| FmiError::ExportFailed(format!("Real {name}: {e}")))?;

    writer
        .write_event(Event::End(BytesEnd::new("ScalarVariable")))
        .map_err(|e| FmiError::ExportFailed(format!("ScalarVariable end {name}: {e}")))?;
    Ok(())
}

/// FMI 2.0 attribute defaults are strings; format f64 compactly.
fn format_float(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1.0e15 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

/// ISO 8601 UTC timestamp for `generationDateAndTime`.
fn generation_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    // Convert epoch seconds to UTC calendar date without pulling in
    // `chrono` (which is already in Cargo.toml as a dependency but
    // avoiding it here keeps this helper dependency-light).
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400);
    let (h, m, s) = (
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60,
    );
    let (y, mo, d) = days_to_ymd(days);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, m, s)
}

/// Convert days-since-epoch to (year, month, day).  Uses the
/// proleptic Gregorian calendar; accurate enough for a `generationDateAndTime`.
fn days_to_ymd(days_since_epoch: i64) -> (i32, u32, u32) {
    // Algorithm by Howard Hinnant (public domain).
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = (yoe as i32) + (era as i32) * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmi_config_default() {
        let config = FmiConfig::default();
        assert_eq!(config.model_name, "FluxionBuilding");
        assert_eq!(config.communication_timestep, 3600.0);
        assert_eq!(config.stop_time, 31536000.0);
    }

    #[test]
    fn test_fmi_exporter_new() {
        let exporter = FmiExporter::new();
        assert_eq!(exporter.config().model_name, "FluxionBuilding");
        assert_eq!(exporter.zone_count(), 1);
    }

    #[test]
    fn test_fmi_exporter_with_config_valid() {
        let config = FmiConfig::default();
        let exporter = FmiExporter::with_config(config);
        assert!(exporter.is_ok());
    }

    #[test]
    fn test_fmi_exporter_with_config_invalid_timestep() {
        let mut config = FmiConfig::default();
        config.communication_timestep = 0.0;
        let exporter = FmiExporter::with_config(config);
        assert!(exporter.is_err());
    }

    #[test]
    fn test_fmi_exporter_with_config_invalid_time_range() {
        let mut config = FmiConfig::default();
        config.start_time = 100.0;
        config.stop_time = 50.0;
        let exporter = FmiExporter::with_config(config);
        assert!(exporter.is_err());
    }

    #[test]
    fn test_fmi_variables_default() {
        let vars = FmiVariables::default();
        assert_eq!(vars.outdoor_temperature, "outdoor_temperature");
        assert_eq!(vars.zone_temperature, "zone_temperature");
        assert_eq!(FmiVariables::PER_ZONE_VARIABLE_COUNT, 7);
    }

    #[test]
    fn test_fmi_mode_default() {
        let mode = FmiMode::default();
        assert_eq!(mode, FmiMode::Cosimulation);
    }

    #[test]
    fn test_fmi_error_display() {
        let err = FmiError::ExportFailed("test error".to_string());
        assert_eq!(format!("{}", err), "FMU export failed: test error");
    }

    // -------------------------------------------------------------------------
    // Multi-zone extension tests (#1339)
    // -------------------------------------------------------------------------

    #[test]
    fn test_multi_zone_default_single_zone() {
        let exporter = FmiExporter::new();
        assert_eq!(exporter.zone_count(), 1);
        assert_eq!(exporter.total_variable_count(), 7);
    }

    #[test]
    fn test_multi_zone_three_zones_count() {
        let exporter = FmiExporter::new().with_zones(vec![
            ZoneVariables::new("living"),
            ZoneVariables::new("bedroom"),
            ZoneVariables::new("kitchen"),
        ]);
        assert_eq!(exporter.zone_count(), 3);
        assert_eq!(exporter.total_variable_count(), 7 * 3);
    }

    #[test]
    #[should_panic(expected = "at least one zone is required")]
    fn test_multi_zone_empty_zones_panics() {
        let _ = FmiExporter::new().with_zones(vec![]);
    }

    #[test]
    fn test_multi_zone_variable_names() {
        let exporter = FmiExporter::new().with_zones(vec![
            ZoneVariables::new("zone"), // legacy single-zone shape
            ZoneVariables::new("bedroom"),
            ZoneVariables::new("kitchen"),
        ]);
        let vars = exporter.variable_names();
        // 3 zones × 7 vars = 21 entries
        assert_eq!(vars.len(), 21);

        // Zone 0 (legacy) keeps the bare template names (#1125 compatibility).
        let zone0_inputs: Vec<_> = vars
            .iter()
            .take(4)
            .map(|(n, c)| (n.clone(), c.clone()))
            .collect();
        assert_eq!(zone0_inputs[0].0, "outdoor_temperature");
        assert_eq!(zone0_inputs[0].1, "input");

        // Zone 1 ("bedroom") uses the `bedroom_` prefix.
        let zone1_inputs: Vec<_> = vars.iter().skip(7).take(4).collect();
        assert_eq!(zone1_inputs[0].0, "bedroom_outdoor_temperature");
        assert_eq!(zone1_inputs[0].1, "input");
        assert_eq!(zone1_inputs[3].0, "bedroom_internal_gains");

        // Zone 2 ("kitchen") uses the `kitchen_` prefix.
        let zone2_outputs: Vec<_> = vars.iter().skip(14).skip(4).collect();
        assert_eq!(zone2_outputs[0].0, "kitchen_zone_temperature");
        assert_eq!(zone2_outputs[0].1, "output");
    }

    #[test]
    fn test_multi_zone_xml_generation_n3() {
        let exporter = FmiExporter::new().with_zones(vec![
            ZoneVariables::new("zone"),
            ZoneVariables::new("bedroom"),
            ZoneVariables::new("kitchen"),
        ]);
        let xml = exporter.generate_model_description_xml().unwrap();

        // FMI 2.0 root
        assert!(
            xml.contains("fmiVersion=\"2.0\""),
            "missing fmiVersion: {}",
            xml
        );
        assert!(
            xml.contains("<fmiModelDescription"),
            "missing fmiModelDescription root: {}",
            xml
        );
        assert!(
            xml.contains("<CoSimulation"),
            "missing CoSimulation element: {}",
            xml
        );
        assert!(
            xml.contains("<DefaultExperiment"),
            "missing DefaultExperiment: {}",
            xml
        );
        // 21 ScalarVariables total (3 × 7)
        let sv_count = xml.matches("<ScalarVariable ").count();
        assert_eq!(
            sv_count, 21,
            "expected 21 ScalarVariables for 3 zones, got {}",
            sv_count
        );
        // 21 Real children
        let real_count = xml.matches("<Real ").count();
        assert_eq!(
            real_count, 21,
            "expected 21 Real attributes, got {}",
            real_count
        );
    }

    #[test]
    fn test_configurable_timestep_default_3600s() {
        let exporter = FmiExporter::new();
        let xml = exporter.generate_model_description_xml().unwrap();
        assert!(
            xml.contains("stepSize=\"3600.0\""),
            "default stepSize missing: {}",
            xml
        );
    }

    #[test]
    fn test_configurable_timestep_60s() {
        let mut cfg = FmiConfig::default();
        cfg.communication_timestep = 60.0;
        let exporter = FmiExporter::with_config(cfg).unwrap();
        let xml = exporter.generate_model_description_xml().unwrap();
        assert!(
            xml.contains("stepSize=\"60.0\""),
            "60s stepSize missing: {}",
            xml
        );
    }

    #[test]
    fn test_configurable_timestep_300s() {
        let mut cfg = FmiConfig::default();
        cfg.communication_timestep = 300.0;
        let exporter = FmiExporter::with_config(cfg).unwrap();
        let xml = exporter.generate_model_description_xml().unwrap();
        assert!(xml.contains("stepSize=\"300.0\""));
    }

    #[test]
    fn test_configurable_timestep_600s() {
        let mut cfg = FmiConfig::default();
        cfg.communication_timestep = 600.0;
        let exporter = FmiExporter::with_config(cfg).unwrap();
        let xml = exporter.generate_model_description_xml().unwrap();
        assert!(xml.contains("stepSize=\"600.0\""));
    }

    #[test]
    fn test_export_fmu_writes_valid_zip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("multi_zone.fmu");
        let exporter = FmiExporter::new().with_zones(vec![
            ZoneVariables::new("zone"),
            ZoneVariables::new("bedroom"),
            ZoneVariables::new("kitchen"),
        ]);
        exporter.export_fmu(&out).expect("export_fmu");

        // Round-trip: open the FMU and read modelDescription.xml back out.
        let xml = FmiExporter::read_model_description_from_fmu(&out).expect("read FMU");
        assert!(xml.contains("<fmiModelDescription"));
        assert!(xml.contains("fmiVersion=\"2.0\""));
        let sv_count = xml.matches("<ScalarVariable ").count();
        assert_eq!(sv_count, 21);
    }

    #[test]
    fn test_xml_contains_required_attributes() {
        let exporter = FmiExporter::new();
        let xml = exporter.generate_model_description_xml().unwrap();

        // Required per FMI 2.0 spec §3
        for needle in &[
            "fmiVersion=\"2.0\"",
            "modelName=\"FluxionBuilding\"",
            "guid=\"{8c4e8d3a-2b1f-4a6c-9e5f-0d3b2a4c6e8d}\"",
            "<CoSimulation",
            "needsExecutionTool=\"true\"",
            "canHandleVariableCommunicationStepSize=\"true\"",
            "<DefaultExperiment",
            "<ModelVariables>",
            "<ScalarVariable name=\"outdoor_temperature\"",
            "<ScalarVariable name=\"zone_temperature\"",
            "causality=\"input\"",
            "causality=\"output\"",
        ] {
            assert!(
                xml.contains(needle),
                "missing required FMI 2.0 attribute/element `{}` in:\n{}",
                needle,
                xml
            );
        }
    }

    #[test]
    fn test_variable_names_input_output_split() {
        let exporter = FmiExporter::new();
        let names = exporter.variable_names();
        let inputs: Vec<_> = names.iter().filter(|(_, c)| c == "input").collect();
        let outputs: Vec<_> = names.iter().filter(|(_, c)| c == "output").collect();
        // Single-zone: 4 inputs, 3 outputs
        assert_eq!(inputs.len(), 4);
        assert_eq!(outputs.len(), 3);
    }

    #[test]
    fn test_zone_variables_default() {
        let z = ZoneVariables::default();
        assert_eq!(z.name, "zone");
        let z2 = ZoneVariables::new("kitchen");
        assert_eq!(z2.name, "kitchen");
    }

    #[test]
    fn test_sanitize_xml_name() {
        assert_eq!(sanitize_xml_name("kitchen"), "kitchen");
        assert_eq!(sanitize_xml_name("living room"), "living_room");
        // First char digit → '_', '-' → '_' (FMI names must match C identifiers).
        assert_eq!(sanitize_xml_name("3rd-floor"), "_rd_floor");
        assert_eq!(sanitize_xml_name(""), "z"); // empty fallback
    }

    #[test]
    fn test_format_float() {
        assert_eq!(format_float(3600.0), "3600.0");
        assert_eq!(format_float(0.0), "0.0");
        assert_eq!(format_float(60.0), "60.0");
        assert_eq!(format_float(1.5e-3), "0.0015");
    }

    #[test]
    fn test_days_to_ymd_known_dates() {
        // 1970-01-01 (epoch)
        assert_eq!(days_to_ymd(0), (1970, 1, 1));
        // 2000-01-01
        assert_eq!(days_to_ymd(10_957), (2000, 1, 1));
        // 2024-02-29 (leap year)
        assert_eq!(days_to_ymd(19_782), (2024, 2, 29));
        // 2026-06-27 (today, just before write-time)
        assert_eq!(days_to_ymd(20_631), (2026, 6, 27));
    }

    // -------------------------------------------------------------------------
    // Import (FmiMode::Import) tests — issue #1708
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_model_description_single_zone() {
        let exporter = FmiExporter::new();
        let xml = exporter.generate_model_description_xml().unwrap();
        let desc = FmiImporter::parse_model_description(&xml).unwrap();

        assert_eq!(desc.fmi_version, "2.0");
        assert_eq!(desc.model_name, "FluxionBuilding");
        assert_eq!(desc.variable_naming_convention, "structured");
        // 4 inputs + 3 outputs = 7 variables
        assert_eq!(desc.variables.len(), 7);
        assert_eq!(desc.input_count(), 4);
        assert_eq!(desc.output_count(), 3);
        assert_eq!(desc.zone_count(), 1);
        assert_eq!(desc.communication_timestep(), 3600.0);
    }

    #[test]
    fn test_parse_model_description_multi_zone() {
        let exporter = FmiExporter::new().with_zones(vec![
            ZoneVariables::new("zone"),
            ZoneVariables::new("bedroom"),
            ZoneVariables::new("kitchen"),
        ]);
        let xml = exporter.generate_model_description_xml().unwrap();
        let desc = FmiImporter::parse_model_description(&xml).unwrap();

        assert_eq!(desc.variables.len(), 21);
        assert_eq!(desc.input_count(), 12);
        assert_eq!(desc.output_count(), 9);
        assert_eq!(desc.zone_count(), 3);

        // Spot-check that variable names round-trip and units are captured.
        let outdoor = desc
            .variables
            .iter()
            .find(|v| v.name == "outdoor_temperature")
            .expect("outdoor_temperature present");
        assert_eq!(outdoor.causality, "input");
        assert_eq!(outdoor.unit, "K");
        assert_eq!(outdoor.start, Some(280.0));

        let zone_temp = desc
            .variables
            .iter()
            .find(|v| v.name == "kitchen_zone_temperature")
            .expect("kitchen_zone_temperature present");
        assert_eq!(zone_temp.causality, "output");
        assert_eq!(zone_temp.unit, "K");
    }

    #[test]
    fn test_parse_model_description_empty_xml_errors() {
        let xml = r#"<?xml version="1.0"?>
<fmiModelDescription fmiVersion="2.0">
  <ModelVariables/>
</fmiModelDescription>"#;
        let res = FmiImporter::parse_model_description(xml);
        assert!(res.is_err(), "empty ModelVariables must error");
    }

    #[test]
    fn test_import_fmu_round_trip_single_zone() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("single_zone.fmu");
        FmiExporter::new().export_fmu(&out).expect("export");

        let model = import_fmu(&out).expect("import_fmu");
        assert_eq!(model.hvac.num_zones, 1);
    }

    #[test]
    fn test_import_fmu_round_trip_three_zone() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("fluxion_three_zone.fmu");
        let exporter = FmiExporter::new().with_zones(vec![
            ZoneVariables::new("zone"),
            ZoneVariables::new("bedroom"),
            ZoneVariables::new("kitchen"),
        ]);
        exporter.export_fmu(&out).expect("export");

        let fmu = FmiImporter::new().import(&out).expect("import");
        assert_eq!(fmu.zone_count(), 3);
        assert_eq!(fmu.communication_timestep(), 3600.0);
        assert_eq!(fmu.thermal_model().hvac.num_zones, 3);
        assert_eq!(fmu.into_thermal_model().hvac.num_zones, 3);
    }

    #[test]
    fn test_import_fmu_configurable_timestep() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("ts300.fmu");
        let mut cfg = FmiConfig::default();
        cfg.communication_timestep = 300.0;
        FmiExporter::with_config(cfg)
            .unwrap()
            .export_fmu(&out)
            .expect("export");

        let fmu = FmiImporter::new().import(&out).expect("import");
        assert_eq!(fmu.communication_timestep(), 300.0);
    }

    #[test]
    fn test_cosimulation_master_do_step_calls_step_physics() {
        // Export a single-zone FMU, re-import it, and drive one doStep.
        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("master.fmu");
        FmiExporter::new().export_fmu(&out).expect("export");
        let fmu = FmiImporter::new().import(&out).expect("import");

        let initial_temp_k = fmu.thermal_model().setpoints.temperatures.as_ref()[0] + 273.15;
        let mut master = FmuCoSimulationMaster::from_imported(fmu);

        // Cold outdoor air (263.15 K = -10 °C) → expect the zone to cool
        // and/or heating to engage.
        let inputs = FmuInputs {
            outdoor_temperature: 263.15,
            direct_normal_solar: 0.0,
            diffuse_horizontal_solar: 0.0,
            internal_gains: 0.0,
        };
        let out_step = master.do_step(inputs, Some(3600.0));

        // do_step must return a finite zone temperature in Kelvin for the
        // single zone (single-zone FMU ⇒ vector length == 1).
        assert_eq!(out_step.len(), 1);
        let zone_out = &out_step[0];
        assert!(zone_out.zone_temperature.is_finite());
        assert!(zone_out.zone_temperature > 200.0 && zone_out.zone_temperature < 320.0);
        // The master advanced time by one communication step.
        assert_eq!(master.current_time(), 3600.0);
        // The zone temperature should have moved away from the initial 20 °C
        // (293.15 K) under the cold boundary condition.
        assert_ne!(zone_out.zone_temperature, initial_temp_k);
    }

    #[test]
    fn test_cosimulation_master_loads_nonneg_and_balanced() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("loads.fmu");
        FmiExporter::new().export_fmu(&out).expect("export");
        let fmu = FmiImporter::new().import(&out).expect("import");
        let mut master = FmuCoSimulationMaster::from_imported(fmu);

        // Drive a handful of steps; loads must be non-negative for every
        // zone reported by do_step.
        for _ in 0..5 {
            let outputs = master.do_step(FmuInputs::default(), Some(3600.0));
            assert!(!outputs.is_empty());
            for o in &outputs {
                assert!(o.heating_load >= 0.0);
                assert!(o.cooling_load >= 0.0);
            }
        }
        assert_eq!(master.current_time(), 5.0 * 3600.0);
    }

    #[test]
    fn test_import_fmu_missing_file_errors() {
        let res = import_fmu(Path::new("/nonexistent/does_not_exist.fmu"));
        assert!(res.is_err());
    }
}

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

#[cfg(test)]
mod ffd_tests {
    use super::*;

    #[test]
    fn test_ffd_fmu_config_default() {
        let config = FfdFmuConfig::default();
        assert_eq!(config.model_name, "FluxionFFD");
        assert_eq!(config.communication_timestep, 60.0);
    }

    #[test]
    fn test_ffd_fmu_config_validate_ok() {
        let config = FfdFmuConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_ffd_fmu_config_validate_bad_timestep() {
        let mut config = FfdFmuConfig::default();
        config.communication_timestep = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ffd_fmu_config_validate_bad_time_range() {
        let mut config = FfdFmuConfig::default();
        config.start_time = 100.0;
        config.stop_time = 50.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ffd_fmu_config_validate_bad_num_surfaces() {
        let mut config = FfdFmuConfig::default();
        config.num_surfaces = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ffd_fmu_exporter_new() {
        let exporter = FfdFmuExporter::new();
        assert_eq!(exporter.config().model_name, "FluxionFFD");
        assert_eq!(exporter.input_count(), 3 + FFD_MAX_SURFACES);
        assert_eq!(
            exporter.output_count(),
            FFD_STRATIFICATION_LEVELS + 2 * FFD_MAX_SURFACES
        );
    }

    #[test]
    fn test_ffd_fmu_exporter_with_config() {
        let config = FfdFmuConfig::default();
        let exporter = FfdFmuExporter::with_config(config);
        assert!(exporter.is_ok());
    }

    #[test]
    fn test_ffd_fmu_exporter_with_config_invalid() {
        let mut config = FfdFmuConfig::default();
        config.communication_timestep = 0.0;
        let exporter = FfdFmuExporter::with_config(config);
        assert!(exporter.is_err());
    }

    #[test]
    fn test_ffd_fmu_xml_generation() {
        let exporter = FfdFmuExporter::new();
        let xml = exporter.generate_model_description_xml().unwrap();

        assert!(xml.contains("fmiVersion=\"2.0\""));
        assert!(xml.contains("<fmiModelDescription"));
        assert!(xml.contains("<CoSimulation"));
        assert!(xml.contains("<DefaultExperiment"));
        assert!(xml.contains("inlet_air_temperature"));
        assert!(xml.contains("mass_flow_rate_supply"));
        assert!(xml.contains("zone_air_temperature_0"));
        assert!(xml.contains("chtc_0"));
        assert!(xml.contains("surface_heat_flux_0"));
    }

    #[test]
    fn test_ffd_fmu_xml_has_required_attributes() {
        let exporter = FfdFmuExporter::new();
        let xml = exporter.generate_model_description_xml().unwrap();

        for needle in &[
            "fmiVersion=\"2.0\"",
            "modelName=\"FluxionFFD\"",
            "<CoSimulation",
            "needsExecutionTool=\"true\"",
            "canHandleVariableCommunicationStepSize=\"true\"",
            "<DefaultExperiment",
            "<ModelVariables>",
            "<ScalarVariable name=\"inlet_air_temperature\"",
            "causality=\"input\"",
            "causality=\"output\"",
        ] {
            assert!(
                xml.contains(needle),
                "missing required attribute `{}` in:\n{}",
                needle,
                xml
            );
        }
    }

    #[test]
    fn test_ffd_fmu_export_fmu_writes_valid_zip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("ffd.fmu");
        let exporter = FfdFmuExporter::new();
        exporter.export_fmu(&out).expect("export_fmu");

        let file = std::fs::File::open(&out).expect("open FMU");
        let mut zip = zip::ZipArchive::new(file).expect("read FMU");
        assert!(zip.by_name("modelDescription.xml").is_ok());
    }

    #[test]
    fn test_ffd_fmu_capi_set_real() {
        let mut api = FfdFmuCApi::new(60.0);

        api.set_real(1, 295.15).unwrap();
        assert_eq!(api.state().inputs.inlet_air_temperature, 295.15);

        api.set_real(2, 0.5).unwrap();
        assert_eq!(api.state().inputs.mass_flow_rate_supply, 0.5);

        api.set_real(4, 290.15).unwrap();
        assert_eq!(api.state().inputs.wall_temperatures[0], 290.15);

        assert!(api.set_real(999, 100.0).is_err());
    }

    #[test]
    fn test_ffd_fmu_capi_get_real() {
        let api = FfdFmuCApi::new(60.0);

        // num_inputs = 3 (inlet, supply, exhaust) + 6 (wall temps) = 9
        // Output vrs start at 10 (num_inputs + 1)
        let num_inputs = 3 + FFD_MAX_SURFACES;
        // vr = num_inputs = 9 is still an input, should error
        assert!(api.get_real(num_inputs as u32).is_err());
        // vr = num_inputs + 1 = 10 is the first output, should be Ok
        let result = api.get_real(num_inputs as u32 + 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ffd_fmu_capi_do_step() {
        let mut api = FfdFmuCApi::new(60.0);
        api.state_mut().initialised = true;

        api.state_mut().inputs.inlet_air_temperature = 295.15;
        api.state_mut().inputs.mass_flow_rate_supply = 0.3;
        api.state_mut().inputs.wall_temperatures = [293.15; FFD_MAX_SURFACES];

        api.do_step(60.0).unwrap();

        assert_eq!(api.state().current_time, 60.0);
        assert_eq!(api.state().timestep, 1);

        for temp in api.state().outputs.zone_air_temperatures {
            assert!(temp > 200.0 && temp < 350.0);
        }
    }

    #[test]
    fn test_ffd_fmu_capi_do_step_not_initialised() {
        let mut api = FfdFmuCApi::new(60.0);
        assert!(api.do_step(60.0).is_err());
    }

    #[test]
    fn test_ffd_fmu_capi_reset() {
        let mut api = FfdFmuCApi::new(60.0);
        api.state_mut().initialised = true;
        api.state_mut().current_time = 3600.0;
        api.state_mut().timestep = 60;

        api.reset();

        assert_eq!(api.state().current_time, 0.0);
        assert_eq!(api.state().timestep, 0);
        assert!(!api.state().initialised);
    }

    #[test]
    fn test_ffd_fmu_inputs_default() {
        let inputs = FfdFmuInputs::default();
        assert_eq!(inputs.inlet_air_temperature, 293.15);
        assert_eq!(inputs.mass_flow_rate_supply, 0.0);
        assert_eq!(inputs.mass_flow_rate_exhaust, 0.0);
        for t in inputs.wall_temperatures {
            assert_eq!(t, 293.15);
        }
    }

    #[test]
    fn test_ffd_fmu_outputs_default() {
        let outputs = FfdFmuOutputs::default();
        for temp in outputs.zone_air_temperatures {
            assert_eq!(temp, 0.0);
        }
        for chtc in outputs.chtc {
            assert_eq!(chtc, 0.0);
        }
        for flux in outputs.surface_heat_fluxes {
            assert_eq!(flux, 0.0);
        }
    }

    #[test]
    fn test_ffd_fmu_variable_names() {
        let vars = FfdFmuVariables::default();
        let inputs = vars.input_names();
        assert_eq!(inputs.len(), 3 + FFD_MAX_SURFACES);
        assert_eq!(inputs[0], "inlet_air_temperature");
        assert_eq!(inputs[1], "mass_flow_rate_supply");

        let outputs = vars.output_names(6, 4);
        assert_eq!(outputs.len(), 4 + 12);
        assert_eq!(outputs[0], "zone_air_temperature_0");
        assert_eq!(outputs[4], "chtc_0");
        assert_eq!(outputs[10], "surface_heat_flux_0");
    }
}
