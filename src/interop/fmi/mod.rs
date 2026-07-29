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
use zip::write::FileOptions;
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
            let options = FileOptions::default()
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
                        b"fmiModelDescription" => {
                            for attr in e.attributes() {
                                let attr =
                                    attr.map_err(|e| FmiError::ImportFailed(format!("attr: {e}")))?;
                                let v = attr_value(&attr)?;
                                match attr.key.as_ref() {
                                    b"fmiVersion" => desc.fmi_version = v,
                                    b"modelName" => desc.model_name = v,
                                    b"guid" => desc.guid = v,
                                    b"description" => desc.description = v,
                                    b"author" => desc.author = v,
                                    b"version" => desc.version = v,
                                    b"generationTool" => desc.generation_tool = v,
                                    b"generationDateAndTime" => desc.generation_date_and_time = v,
                                    b"variableNamingConvention" => {
                                        desc.variable_naming_convention = v
                                    }
                                    _ => {}
                                }
                            }
                        }
                        b"DefaultExperiment" => {
                            for attr in e.attributes() {
                                let attr =
                                    attr.map_err(|e| FmiError::ImportFailed(format!("attr: {e}")))?;
                                let v = attr_value(&attr)?;
                                let parsed = v.parse::<f64>().unwrap_or(0.0);
                                match attr.key.as_ref() {
                                    b"startTime" => desc.default_experiment.start_time = parsed,
                                    b"stopTime" => desc.default_experiment.stop_time = parsed,
                                    b"stepSize" => desc.default_experiment.step_size = parsed,
                                    _ => {}
                                }
                            }
                        }
                        b"ModelVariables" => in_model_variables = true,
                        b"ScalarVariable" if in_model_variables => {
                            // Start a new accumulator.  Attributes on the
                            // opening tag; the nested <Real> fills unit/start.
                            let mut sv = ImportedScalarVariable::default();
                            for attr in e.attributes() {
                                let attr =
                                    attr.map_err(|e| FmiError::ImportFailed(format!("attr: {e}")))?;
                                let v = attr_value(&attr)?;
                                match attr.key.as_ref() {
                                    b"name" => sv.name = v,
                                    b"causality" => sv.causality = v,
                                    b"variability" => sv.variability = v,
                                    b"description" => sv.description = v,
                                    b"valueReference" => {
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
                        b"Real" if current_var.is_some() => {
                            if let Some(ref mut sv) = current_var {
                                for attr in e.attributes() {
                                    let attr = attr.map_err(|e| {
                                        FmiError::ImportFailed(format!("attr: {e}"))
                                    })?;
                                    let v = attr_value(&attr)?;
                                    match attr.key.as_ref() {
                                        b"unit" => sv.unit = v,
                                        b"start" => sv.start = v.parse::<f64>().ok(),
                                        _ => {}
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Event::End(ref e) => match e.name().as_ref() {
                    b"ScalarVariable" => {
                        if let Some(v) = current_var.take() {
                            desc.variables.push(v);
                        }
                    }
                    b"ModelVariables" => in_model_variables = false,
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
/// assert_eq!(model.num_zones, 3);
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
    /// the zone temperature (converted back to Kelvin) together with the
    /// heating/cooling loads averaged over the step.
    ///
    /// If `step_size` is omitted the FMU's declared communication timestep
    /// is used.
    pub fn do_step(&mut self, inputs: FmuInputs, step_size: Option<f64>) -> FmuOutputs {
        let dt = step_size.unwrap_or(self.communication_timestep).max(1.0);

        // Snapshot per-zone energy accumulators *before* the step so the
        // delta gives the energy consumed during this step alone.
        let heat_before = self.model.zone_heating_energy_kwh.as_ref().first().copied();
        let cool_before = self.model.zone_cooling_energy_kwh.as_ref().first().copied();

        // FMI inputs are Kelvin; step_physics expects °C.
        let outdoor_temp_c = inputs.outdoor_temperature - 273.15;
        let _energy_kwh = self.model.step_physics(self.timestep, outdoor_temp_c, dt);

        let zone_temp_c = self
            .model
            .temperatures
            .as_ref()
            .first()
            .copied()
            .unwrap_or(20.0);

        // Convert kWh-delta over the step to average Watts:
        //   W = kWh * 3_600_000 / dt
        let heating_load = heat_before
            .zip(self.model.zone_heating_energy_kwh.as_ref().first().copied())
            .map(|(a, b)| ((b - a) * 3_600_000.0 / dt).max(0.0))
            .unwrap_or(0.0);
        let cooling_load = cool_before
            .zip(self.model.zone_cooling_energy_kwh.as_ref().first().copied())
            .map(|(a, b)| ((b - a) * 3_600_000.0 / dt).max(0.0))
            .unwrap_or(0.0);

        self.timestep += 1;
        self.current_time += dt;

        FmuOutputs {
            zone_temperature: zone_temp_c + 273.15,
            heating_load,
            cooling_load,
        }
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
        assert_eq!(model.num_zones, 1);
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
        assert_eq!(fmu.thermal_model().num_zones, 3);
        assert_eq!(fmu.into_thermal_model().num_zones, 3);
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

        let initial_temp_k = fmu.thermal_model().temperatures.as_ref()[0] + 273.15;
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

        // do_step must return a finite zone temperature in Kelvin.
        assert!(out_step.zone_temperature.is_finite());
        assert!(out_step.zone_temperature > 200.0 && out_step.zone_temperature < 320.0);
        // The master advanced time by one communication step.
        assert_eq!(master.current_time(), 3600.0);
        // The zone temperature should have moved away from the initial 20 °C
        // (293.15 K) under the cold boundary condition.
        assert_ne!(out_step.zone_temperature, initial_temp_k);
    }

    #[test]
    fn test_cosimulation_master_loads_nonneg_and_balanced() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("loads.fmu");
        FmiExporter::new().export_fmu(&out).expect("export");
        let fmu = FmiImporter::new().import(&out).expect("import");
        let mut master = FmuCoSimulationMaster::from_imported(fmu);

        // Drive a handful of steps; loads must be non-negative.
        for _ in 0..5 {
            let o = master.do_step(FmuInputs::default(), Some(3600.0));
            assert!(o.heating_load >= 0.0);
            assert!(o.cooling_load >= 0.0);
        }
        assert_eq!(master.current_time(), 5.0 * 3600.0);
    }

    #[test]
    fn test_import_fmu_missing_file_errors() {
        let res = import_fmu(Path::new("/nonexistent/does_not_exist.fmu"));
        assert!(res.is_err());
    }
}
