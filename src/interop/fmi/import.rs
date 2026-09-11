// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! FMI 2.0 import path (FMU → Fluxion, `FmiMode::Import`): parse a `.fmu`
//! archive's `modelDescription.xml` and rebuild a [`ThermalModel`].

use quick_xml::events::Event;
use quick_xml::Reader;
use std::fs::File;
use std::path::Path;

use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;

use super::common::FmiError;

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
/// model, and [`FmuCoSimulationMaster::from_imported`](crate::interop::fmi::FmuCoSimulationMaster::from_imported) to drive it as a
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

/// FMI 2.0 FMU importer ([`FmiMode::Import`](crate::interop::fmi::FmiMode::Import)).
///
/// The importer is the import-side mirror of [`FmiExporter`](crate::interop::fmi::FmiExporter): it reads a
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
    /// root, as produced by [`FmiExporter::export_fmu`](crate::interop::fmi::FmiExporter::export_fmu).  The returned
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
    ///
    /// # Security (issue #3591)
    ///
    /// The reader is hardened against XML External Entity (XXE) attacks:
    ///
    /// * `Config::expand_empty_elements` is explicitly disabled so a
    ///   `<tag attr="…"/> ` self-closing tag can never silently expand to
    ///   `<tag attr="…"></tag>` carrying injected content.
    /// * Any `<!DOCTYPE …>` carrying a `SYSTEM` or `PUBLIC` keyword — or
    ///   any DTD internal subset declaring an `<!ENTITY … SYSTEM "…">`
    ///   or `<!NOTATION … SYSTEM "…">` external reference — is rejected
    ///   with [`FmiError::ImportFailed`].
    /// * Any general entity reference (`&name;`) other than the five
    ///   XML-predefined entities (`&amp;`, `&lt;`, `&gt;`, `&quot;`,
    ///   `&apos;`) or a numeric character reference (`&#10;`, `&#xA;`)
    ///   is rejected — a custom entity could only originate from an
    ///   external DTD that was already blocked above, but the check is
    ///   kept as defense-in-depth.
    pub fn parse_model_description(xml: &str) -> Result<ImportedModelDescription, FmiError> {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);
        // XXE guard (issue #3591): refuse to silently expand self-closing
        // tags into open/close pairs.
        reader.config_mut().expand_empty_elements = false;

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
                Event::DocType(ref dt) => {
                    // XXE guard (issue #3591): reject any DTD that
                    // references an external resource.  This covers
                    // `<!DOCTYPE foo SYSTEM "…">`, `<!DOCTYPE foo PUBLIC
                    // "…">`, `<!ENTITY x SYSTEM "…">`,
                    // `<!ENTITY x PUBLIC "…">`, and `<!NOTATION y SYSTEM
                    // "…">` because the SYSTEM/PUBLIC keywords are
                    // reserved by the XML 1.0 spec and have no other
                    // legitimate use inside a `modelDescription.xml`
                    // (FMI 2.0 does not declare or require a DTD).
                    let content = dt.as_ref();
                    let upper = content.to_ascii_uppercase();
                    if upper.contains("SYSTEM") || upper.contains("PUBLIC") {
                        return Err(FmiError::ImportFailed(format!(
                            "modelDescription.xml declares an external DTD or entity \
                             reference (XXE guard, issue #3591): {content}"
                        )));
                    }
                }
                Event::GeneralRef(ref r) => {
                    // XXE guard (issue #3591): only the five XML-predefined
                    // entities and numeric character references are safe.
                    let name = r.as_ref();
                    let is_predefined = matches!(name, "amp" | "lt" | "gt" | "quot" | "apos");
                    let is_numeric_ref = name.starts_with('#');
                    if !is_predefined && !is_numeric_ref {
                        return Err(FmiError::ImportFailed(format!(
                            "modelDescription.xml references custom entity &{name}; \
                             (XXE guard, issue #3591)"
                        )));
                    }
                }
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
/// [`FmiExporter::read_model_description_from_fmu`](crate::interop::fmi::FmiExporter::read_model_description_from_fmu); it is a free
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
