// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Parser for EnergyPlus IDF (Input Data File) text.
//!
//! This module consumes the [`RawObject`] stream produced by
//! [`super::lexer::tokenize`] and turns each raw object into a typed
//! [`IdfObject`]. Object-type dispatch is case-insensitive (per EnergyPlus
//! convention) and only the **10 MVP object types** listed in design §4.1
//! are explicitly classified; everything else is still captured as a
//! generic [`IdfObject`] so callers can inspect or forward unknown
//! objects without the parser rejecting the file.
//!
//! # Scope (MVP — issue #1341 / design §4.1)
//!
//! The parser covers:
//! - `Version`
//! - `Timestep`
//! - `RunPeriod`
//! - `Building`
//! - `Zone`
//! - `Material`
//! - `Construction`
//! - `BuildingSurface:Detailed`
//! - `GlobalGeometryRules`
//! - `Site:GroundTemperature:BuildingSurface`
//!
//! Out of scope (design §10): HVAC, Schedule, Window/Door,
//! `FenestrationSurface:Detailed`, IDF export, and the
//! `TryFrom<IdfFile> for SimulationSchema` conversion (design §4.3).

use std::str::FromStr;

use fluxion_core::parser_limits::ParserLimits;

use super::error::IdfError;
use super::lexer::{tokenize, RawObject};

/// A single field value after quote and comment processing.
///
/// EnergyPlus distinguishes alphabetic (`A1`) and numeric (`N1`/`N2`)
/// fields at the IDD level, but for the MVP we only need the parsed
/// payload. The `Empty` variant covers the `, ,` shorthand used for
/// optional / defaulted fields.
#[derive(Debug, Clone, PartialEq)]
pub enum IdfValue {
    /// Quoted or unquoted string field.
    String(String),
    /// Parsed real number (EnergyPlus `N1`/`N2`).
    Real(f64),
    /// Parsed integer (subset of numeric fields).
    Integer(i64),
    /// Field omitted (`, ,` in source).
    Empty,
}

impl IdfValue {
    /// Render the value back to a string suitable for diagnostics and
    /// round-tripping. Empty values render as the empty string so a
    /// future IDF writer can faithfully reproduce `, ,` shorthand.
    pub fn as_str(&self) -> &str {
        match self {
            IdfValue::String(s) => s.as_str(),
            // Numeric variants carry no string form on this struct;
            // callers that need the textual form should use
            // [`IdfValue::to_display_string`] (allocates) or keep the
            // original raw token from the lexer.
            IdfValue::Real(_) | IdfValue::Integer(_) => "",
            IdfValue::Empty => "",
        }
    }

    /// Lossless textual rendering for any variant — useful when the
    /// caller doesn't care whether the source field was numeric or a
    /// quoted string. Returns `None` for [`IdfValue::Empty`].
    pub fn to_display_string(&self) -> Option<String> {
        match self {
            IdfValue::String(s) => Some(s.clone()),
            IdfValue::Real(f) => Some(f.to_string()),
            IdfValue::Integer(i) => Some(i.to_string()),
            IdfValue::Empty => None,
        }
    }
}

/// A single parsed IDF object.
///
/// `name` is the first field after the type name (for objects that have
/// one). For objects without a name field (`Version`, `Timestep`,
/// `GlobalGeometryRules`, `Site:GroundTemperature:BuildingSurface`),
/// `name` is `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct IdfObject {
    pub object_type: String,
    pub name: Option<String>,
    pub fields: Vec<IdfValue>,
    /// 1-indexed source line where the object starts.
    pub line: usize,
}

/// Parsed in-memory representation of an IDF file.
///
/// The 10 MVP object types are *also* exposed via typed accessors
/// ([`IdfFile::versions`], [`IdfFile::zones`], etc.) so downstream
/// consumers can iterate by category without re-filtering `objects`.
#[derive(Debug, Clone, Default)]
pub struct IdfFile {
    pub version: Option<String>,
    pub objects: Vec<IdfObject>,
}

impl IdfFile {
    /// All `Version` objects (typically 0 or 1 per file).
    pub fn versions(&self) -> impl Iterator<Item = &IdfObject> {
        self.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("Version"))
    }

    /// All `Zone` objects.
    pub fn zones(&self) -> impl Iterator<Item = &IdfObject> {
        self.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("Zone"))
    }

    /// All `Material` objects.
    pub fn materials(&self) -> impl Iterator<Item = &IdfObject> {
        self.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("Material"))
    }

    /// All `Construction` objects.
    pub fn constructions(&self) -> impl Iterator<Item = &IdfObject> {
        self.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("Construction"))
    }

    /// All `BuildingSurface:Detailed` objects.
    pub fn building_surfaces(&self) -> impl Iterator<Item = &IdfObject> {
        self.objects.iter().filter(|o| {
            o.object_type
                .eq_ignore_ascii_case("BuildingSurface:Detailed")
        })
    }

    /// All `Site:GroundTemperature:BuildingSurface` objects.
    pub fn ground_temperatures(&self) -> impl Iterator<Item = &IdfObject> {
        self.objects.iter().filter(|o| {
            o.object_type
                .eq_ignore_ascii_case("Site:GroundTemperature:BuildingSurface")
        })
    }

    /// All `FenestrationSurface:Detailed` objects (windows and doors).
    pub fn fenestration_surfaces(&self) -> impl Iterator<Item = &IdfObject> {
        self.objects.iter().filter(|o| {
            o.object_type
                .eq_ignore_ascii_case("FenestrationSurface:Detailed")
        })
    }

    /// All `Schedule:Compact` objects.
    pub fn schedules(&self) -> impl Iterator<Item = &IdfObject> {
        self.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("Schedule:Compact"))
    }

    /// All `ZoneHVAC:IdealLoadsAirSystem` objects.
    pub fn zone_hvac_ideal_loads(&self) -> impl Iterator<Item = &IdfObject> {
        self.objects.iter().filter(|o| {
            o.object_type
                .eq_ignore_ascii_case("ZoneHVAC:IdealLoadsAirSystem")
        })
    }

    /// All `ZoneHVAC:EquipmentConnections` objects.
    pub fn zone_hvac_equipment_connections(&self) -> impl Iterator<Item = &IdfObject> {
        self.objects.iter().filter(|o| {
            o.object_type
                .eq_ignore_ascii_case("ZoneHVAC:EquipmentConnections")
        })
    }

    /// All `ThermostatSetpoint:DualSetpoint` objects.
    pub fn thermostat_setpoint_dual(&self) -> impl Iterator<Item = &IdfObject> {
        self.objects.iter().filter(|o| {
            o.object_type
                .eq_ignore_ascii_case("ThermostatSetpoint:DualSetpoint")
        })
    }

    /// All `ZoneControl:Thermostat` objects.
    pub fn zone_control_thermostat(&self) -> impl Iterator<Item = &IdfObject> {
        self.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("ZoneControl:Thermostat"))
    }

    /// All `WindowMaterial:Glazing` objects.
    pub fn window_materials(&self) -> impl Iterator<Item = &IdfObject> {
        self.objects
            .iter()
            .filter(|o| o.object_type.eq_ignore_ascii_case("WindowMaterial:Glazing"))
    }

    /// All `ZoneInfiltration:DesignFlowRate` objects.
    pub fn infiltration(&self) -> impl Iterator<Item = &IdfObject> {
        self.objects.iter().filter(|o| {
            o.object_type
                .eq_ignore_ascii_case("ZoneInfiltration:DesignFlowRate")
        })
    }
}

/// Entry point for parsing IDF documents.
///
/// `IdfParser` is intentionally a zero-sized struct so callers can write
/// `IdfParser::from_str(...)` / `IdfParser::from_path(...)` per design §9
/// without managing state. All parsing logic lives in associated functions.
pub struct IdfParser;

impl IdfParser {
    /// Parse an in-memory IDF document with the strict default parser
    /// limits (64 MiB / 1M lines — issue #2527).
    #[allow(clippy::should_implement_trait)] // design §9 mandates `from_str` signature
    pub fn from_str(input: &str) -> Result<IdfFile, IdfError> {
        Self::from_str_with_limits(input, &ParserLimits::default())
    }

    /// Parse an in-memory IDF document with explicit [`ParserLimits`].
    ///
    /// Enforces `max_file_bytes` (on `input.len()`, checked in O(1)
    /// before the lexer allocates a `Vec<char>` of the whole source)
    /// and `max_lines` (counted without allocating).
    pub fn from_str_with_limits(input: &str, limits: &ParserLimits) -> Result<IdfFile, IdfError> {
        limits.check_file_bytes(input.len())?;
        limits.check_lines(input.lines().count())?;

        let raw_objects = tokenize(input)?;
        let mut idf = IdfFile::default();
        for raw in raw_objects {
            let obj = parse_raw_object(raw)?;
            if obj.object_type.eq_ignore_ascii_case("Version") {
                // Cache the first Version object's payload — used by the
                // SimulationSchema conversion in Phase 3.
                if idf.version.is_none() {
                    idf.version = obj.fields.first().and_then(|v| v.to_display_string());
                }
            }
            idf.objects.push(obj);
        }
        Ok(idf)
    }

    /// Parse an IDF document from a filesystem path.
    pub fn from_path(path: &std::path::Path) -> Result<IdfFile, IdfError> {
        Self::from_path_with_limits(path, &ParserLimits::default())
    }

    /// Parse an IDF document from a filesystem path with explicit
    /// [`ParserLimits`]. The on-disk size is checked before the file is
    /// read (issue #2527).
    pub fn from_path_with_limits(
        path: &std::path::Path,
        limits: &ParserLimits,
    ) -> Result<IdfFile, IdfError> {
        let file_len = std::fs::metadata(path).map_err(IdfError::from)?.len() as usize;
        limits.check_file_bytes(file_len)?;
        let content = std::fs::read_to_string(path)?;
        Self::from_str_with_limits(&content, limits)
    }
}

/// [`FromStr`] implementation so `"25.2".parse::<IdfFile>()` also works.
/// Internally delegates to [`IdfParser::from_str`] (the design §9 API).
impl FromStr for IdfFile {
    type Err = IdfError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        IdfParser::from_str(s)
    }
}

impl TryFrom<&std::path::Path> for IdfFile {
    type Error = IdfError;

    fn try_from(path: &std::path::Path) -> Result<Self, Self::Error> {
        IdfParser::from_path(path)
    }
}

/// Convert a single [`RawObject`] from the lexer into a typed
/// [`IdfObject`]. Splits the body on field commas while respecting
/// quoted strings.
fn parse_raw_object(raw: RawObject) -> Result<IdfObject, IdfError> {
    let fields = split_fields(&raw.body, raw.line)?;
    let object_type = raw.object_type.clone();

    // Extract the name (first non-empty field) for objects that have one.
    // We don't try to know *which* types have names — that mapping lives
    // in the Phase 3 converter. For now we just expose the first field as
    // a best-effort name so downstream code can label objects in
    // diagnostics and tests.
    let name = fields.first().and_then(|v| match v {
        IdfValue::String(s) if !s.is_empty() => Some(s.clone()),
        _ => None,
    });

    Ok(IdfObject {
        object_type,
        name,
        fields,
        line: raw.line,
    })
}

/// Split a raw object body into typed [`IdfValue`]s by walking
/// character-by-character and tracking quote state. Commas inside quoted
/// strings do not split fields; doubled `""` inside a quoted string is an
/// escaped quote.
fn split_fields(body: &str, line: usize) -> Result<Vec<IdfValue>, IdfError> {
    let mut fields: Vec<IdfValue> = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut saw_quote = false; // distinguishes "" from an empty field

    let chars: Vec<char> = body.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '"' if in_quotes => {
                // Doubled quote "" inside a quoted string is an escaped
                // literal quote (EnergyPlus convention). The lexer
                // preserves both `"` characters so we can detect the
                // pair here.
                if i + 1 < chars.len() && chars[i + 1] == '"' {
                    current.push('"');
                    i += 2;
                    continue;
                }
                in_quotes = false;
                // Do not push the closing quote.
            }
            '"' => {
                in_quotes = true;
                saw_quote = true;
            }
            ',' if !in_quotes => {
                fields.push(finalize_field(&current, saw_quote));
                current.clear();
                saw_quote = false;
            }
            _ => current.push(c),
        }
        i += 1;
    }

    if in_quotes {
        return Err(IdfError::parse_error(
            line,
            "Unterminated quoted string inside object body",
        ));
    }

    fields.push(finalize_field(&current, saw_quote));
    Ok(fields)
}

/// Convert the collected raw text for a single field into a typed
/// [`IdfValue`]. Numeric coercion is attempted for non-quoted fields
/// since EnergyPlus `, 25.2` and `, 25` are the common shapes; quoted
/// fields always remain [`IdfValue::String`].
fn finalize_field(raw: &str, was_quoted: bool) -> IdfValue {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return IdfValue::Empty;
    }
    if was_quoted {
        // Strip surrounding quotes that the lexer preserved.
        // (The lexer keeps the quote characters on `current`; the
        // `was_quoted` flag tells us the field was a quoted string.)
        let unquoted = trimmed.trim_matches('"').to_string();
        return IdfValue::String(unquoted);
    }
    // Try integer first (most restrictive), then float, else string.
    if let Ok(i) = trimmed.parse::<i64>() {
        return IdfValue::Integer(i);
    }
    if let Ok(f) = trimmed.parse::<f64>() {
        return IdfValue::Real(f);
    }
    IdfValue::String(trimmed.to_string())
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_version() {
        let src = "Version, 25.2;";
        let idf = IdfParser::from_str(src).unwrap();
        assert_eq!(idf.version.as_deref(), Some("25.2"));
        assert_eq!(idf.objects.len(), 1);
        let v = &idf.objects[0];
        assert_eq!(v.object_type, "Version");
        // "25.2" parses cleanly as an f64, so the parser stores it as
        // Real (consistent with EnergyPlus' untyped numeric fields).
        assert_eq!(v.fields, vec![IdfValue::Real(25.2)]);
    }

    #[test]
    fn parses_object_with_integer_and_float_fields() {
        let src = "Timestep, 4;";
        let idf = IdfParser::from_str(src).unwrap();
        let ts = &idf.objects[0];
        assert_eq!(ts.object_type, "Timestep");
        assert_eq!(ts.fields, vec![IdfValue::Integer(4)]);
    }

    #[test]
    fn parses_quoted_comma_field_keeps_inner_comma() {
        let src = r#"Material, "Hello, World!", 1.0;"#;
        let idf = IdfParser::from_str(src).unwrap();
        let m = &idf.objects[0];
        assert_eq!(m.object_type, "Material");
        assert_eq!(m.fields.len(), 2);
        assert_eq!(m.fields[0], IdfValue::String("Hello, World!".to_string()));
        assert_eq!(m.fields[1], IdfValue::Real(1.0));
    }

    #[test]
    fn missing_fields_become_empty() {
        let src = "RunPeriod, AnnualRun, 1, 1, , 12, 31;";
        let idf = IdfParser::from_str(src).unwrap();
        let rp = &idf.objects[0];
        // 6 fields: name, begin_month, begin_day, <empty begin_year>,
        // end_month, end_day.
        assert_eq!(rp.fields.len(), 6);
        assert_eq!(rp.fields[3], IdfValue::Empty);
    }

    #[test]
    fn case_insensitive_object_matching() {
        // The lexer preserves case but the parser should still classify
        // "version" and "VERSION" correctly when filtering.
        let src = "version, 25.2;\n";
        let idf = IdfParser::from_str(src).unwrap();
        assert!(idf.versions().next().is_some());
    }

    #[test]
    fn unknown_object_types_are_captured_not_rejected() {
        let src = "TotallyMadeUpObject, foo, bar;\n";
        let idf = IdfParser::from_str(src).unwrap();
        assert_eq!(idf.objects.len(), 1);
        assert_eq!(idf.objects[0].object_type, "TotallyMadeUpObject");
    }

    // ----- Issue #2527: parser DoS limits -----------------------------------

    fn tiny_limits() -> fluxion_core::parser_limits::ParserLimits {
        fluxion_core::parser_limits::ParserLimits {
            max_file_bytes: 4 * 1024,
            max_lines: 20,
            max_recursion_depth: 256,
            max_array_elements: 1_000_000,
        }
    }

    #[test]
    fn idf_parses_with_limits() {
        let src = "Version, 25.2;\nTimestep, 4;\n";
        let idf = IdfParser::from_str_with_limits(src, &tiny_limits()).unwrap();
        assert_eq!(idf.objects.len(), 2);
    }

    #[test]
    fn idf_rejects_too_many_lines() {
        // 25 lines, limit is 20.
        let mut src = String::from("Version, 25.2;\n");
        for _ in 0..24 {
            src.push_str("! comment\n");
        }
        let err = IdfParser::from_str_with_limits(&src, &tiny_limits()).unwrap_err();
        assert!(
            matches!(err, IdfError::SizeLimitExceeded(_)),
            "expected SizeLimitExceeded, got {:?}",
            err
        );
        assert!(err.to_string().to_lowercase().contains("line"));
    }

    #[test]
    fn idf_rejects_oversized_bytes() {
        // Build a >4KiB single-line IDF so only the byte cap fires.
        let padding = "a".repeat(5 * 1024);
        let src = format!("Version, {};\n", padding);
        let err = IdfParser::from_str_with_limits(&src, &tiny_limits()).unwrap_err();
        assert!(
            matches!(err, IdfError::SizeLimitExceeded(_)),
            "expected SizeLimitExceeded, got {:?}",
            err
        );
        assert!(err.to_string().to_lowercase().contains("file size"));
    }

    #[test]
    fn idf_default_limits_match_issue_acceptance() {
        let d = fluxion_core::parser_limits::ParserLimits::default();
        assert_eq!(d.max_file_bytes, 64 * 1024 * 1024);
        assert_eq!(d.max_lines, 1_000_000);
    }
}
