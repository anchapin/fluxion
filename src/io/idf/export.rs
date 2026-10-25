// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! EnergyPlus IDF (Input Data File) **export** — issue #3991 (slice 1).
//!
//! This module is the counterpart of [`super::parser`]: it serialises an
//! [`IdfFile`] (or a [`CaseSpec`] envelope) back into IDF text that the
//! existing parser can read again.
//!
//! # Entry points
//!
//! - [`export_idf`] — full-fidelity export of every object in an
//!   [`IdfFile`]. This is the primary entry point: the emitted text
//!   re-parses into an [`IdfFile`] whose objects and fields are equal to
//!   the input's.
//! - [`export_idf_to_writer`] — same content, streamed to a [`std::io::Write`].
//! - [`export_case_spec_envelope`] — envelope-only export of a
//!   [`CaseSpec`] (the "`.flux`" model): Version, Timestep, RunPeriod,
//!   Building, Zone, Material, Construction, BuildingSurface:Detailed,
//!   GlobalGeometryRules and Site:GroundTemperature:BuildingSurface.
//!
//! # Determinism / idempotence
//!
//! `export(parse(export(x))) == export(x)` **byte for byte**. The
//! guarantees that make this hold are:
//!
//! 1. **Object order is the input order.** Objects are emitted in the
//!    order they appear in [`IdfFile::objects`]; the exporter never
//!    re-sorts or groups them. (A canonical per-type ordering is *not*
//!    applied, because the importer's canonical order is not defined and
//!    re-ordering would break the byte-identity property for files whose
//!    objects are not already sorted.)
//! 2. **One object per line.** Every object is written as a single line
//!    `Type, field, field, ...;` — no pretty-printing, no line wrapping,
//!    so the output is a pure function of the parsed values.
//! 3. **Numbers use a round-trip-exact textual form.** [`IdfValue::Real`]
//!    is rendered with Rust's shortest round-trip `Display` form and is
//!    forced to contain a `.` (or an exponent) so that re-parsing yields
//!    a `Real` again and never an `Integer`. [`IdfValue::Integer`] is
//!    rendered as a plain integer.
//! 4. **Strings keep their type.** A string that would otherwise
//!    re-parse as a number (or that contains a delimiter, a comment
//!    character, or leading/trailing whitespace) is emitted **quoted**;
//!    embedded `"` are escaped as `""` per the EnergyPlus convention.
//!    Quoting is therefore never lost on re-parse.
//! 5. **No header comment.** The parser drops comments, so a header
//!    comment would break the idempotence property (the second export
//!    would not contain it). Callers that want a header should prepend
//!    it themselves.
//!
//! # Object-type coverage
//!
//! The exporter is *type-agnostic*: it renders the raw
//! [`IdfObject::fields`] vector of every object it is given, so **every**
//! object type the importer captures is round-tripped verbatim — there
//! are no silent drops. Concretely:
//!
//! | Object type | Export path |
//! |-------------|-------------|
//! | `Version`, `Timestep`, `RunPeriod`, `Building`, `Zone` | verbatim fields |
//! | `Material`, `Construction` | verbatim fields |
//! | `BuildingSurface:Detailed` | verbatim fields |
//! | `GlobalGeometryRules`, `Site:GroundTemperature:BuildingSurface` | verbatim fields |
//! | `Schedule:*`, `FenestrationSurface:Detailed`, `ZoneHVAC:*`, `WindowMaterial:*`, `ZoneInfiltration:*`, … (captured but not typed) | verbatim fields |
//!
//! The typed accessors on [`IdfFile`] (e.g. [`IdfFile::zones`]) are
//! convenience filters over the same [`IdfObject::fields`] vectors, so
//! "typed" and "captured" objects follow the very same export path and
//! the exporter's field order always mirrors the order the importer
//! reads (field `i` of the exported text is field `i` of the parsed
//! object).
//!
//! # `CaseSpec` (envelope) export — scope
//!
//! [`export_case_spec_envelope`] writes the envelope of a [`CaseSpec`]:
//! geometry, constructions and ground temperature. It **excludes**, and
//! documents, the following (they are out of scope for this slice and
//! are not representable without inventing data):
//!
//! - **Glazing** (`FenestrationSurface:Detailed`, `WindowMaterial:*`) —
//!   a [`CaseSpec`] carries a window *area* and a glazing U-value/SHGC
//!   pair, but no thickness/optical-property set that maps onto the
//!   EnergyPlus window IDD without fabrication.
//! - **Systems** (HVAC, schedules, thermostats, infiltration, internal
//!   loads) — those are the subject of a later export slice; the
//!   `CaseSpec` fields that carry them are not written out here.
//!
//! # Surface field layout
//!
//! `BuildingSurface:Detailed` is emitted with the **same field layout
//! that the importer reads** (see
//! `super::convert::parse_surface_polygon`, which probes field index 10
//! for the vertex count and index 11 for the first vertex). That layout
//! matches the reference model
//! `tests/reference_data/energyplus_models/ashrae_140_case_600.idf`.
//! It is one field wider than the strict EnergyPlus IDD ordering
//! (where `Number of Vertices` sits at index 9); aligning the importer
//! with the strict IDD ordering is out of scope for this slice.
//!
//! # Example
//!
//! ```ignore
//! use fluxion::io::idf::{export_idf, IdfParser};
//!
//! let src = "Version, 25.2;\nTimestep, 1;\n";
//! let idf = IdfParser::from_str(src).expect("parses");
//! let text = export_idf(&idf);
//! assert!(text.contains("Version, 25.2;"));
//! ```

use std::io::{self, Write};

use crate::ashrae_cases::CaseSpec;

use super::parser::{IdfFile, IdfObject, IdfValue};

/// Version string written by [`export_case_spec_envelope`].
const EXPORT_VERSION: &str = "25.2";

/// Timestep (per hour) written by [`export_case_spec_envelope`].
const EXPORT_TIMESTEP: i64 = 1;

/// Export an [`IdfFile`] to IDF text.
///
/// Every object of the file is emitted, in input order, as a single line
/// `Type, field, field, ...;`. The result re-parses into an equal
/// [`IdfFile`] (see the module header for the determinism guarantees).
pub fn export_idf(idf: &IdfFile) -> String {
    let mut out = String::new();
    for object in &idf.objects {
        write_object(&mut out, object);
    }
    out
}

/// Export an [`IdfFile`] to a [`std::io::Write`] sink.
///
/// Byte-for-byte identical to [`export_idf`]'s output.
pub fn export_idf_to_writer<W: Write>(idf: &IdfFile, out: &mut W) -> io::Result<()> {
    for object in &idf.objects {
        write_object_to_writer(out, object)?;
    }
    Ok(())
}

/// Export the **envelope** of a [`CaseSpec`] to IDF text.
///
/// The emitted file contains only the object types the importer covers
/// for the envelope (see the module header for the exact list and for
/// the documented exclusions: glazing and systems).
///
/// The result re-parses into a [`CaseSpec`] (via
/// [`super::convert::case_spec_from_idf`]) whose geometry, constructions
/// and ground temperature match the input — the "IDF → .flux → IDF"
/// envelope round-trip.
pub fn export_case_spec_envelope(case: &CaseSpec) -> String {
    let objects = case_spec_objects(case);
    let mut out = String::new();
    for object in &objects {
        write_object(&mut out, object);
    }
    out
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn write_object(out: &mut String, object: &IdfObject) {
    out.push_str(&object.object_type);
    for field in &object.fields {
        out.push_str(", ");
        out.push_str(&format_value(field));
    }
    out.push_str(";\n");
}

fn write_object_to_writer<W: Write>(out: &mut W, object: &IdfObject) -> io::Result<()> {
    out.write_all(object.object_type.as_bytes())?;
    for field in &object.fields {
        out.write_all(b", ")?;
        out.write_all(format_value(field).as_bytes())?;
    }
    out.write_all(b";\n")?;
    Ok(())
}

/// Render a single field value.
///
/// * [`IdfValue::Empty`] → the empty string (the `, ,` shorthand).
/// * [`IdfValue::Integer`] → a plain integer.
/// * [`IdfValue::Real`] → the shortest round-trip decimal form, always
///   carrying a `.` (or an exponent) so that it re-parses as a `Real`.
/// * [`IdfValue::String`] → the raw text, quoted when quoting is needed
///   to preserve the value (see [`format_string`]).
pub fn format_value(value: &IdfValue) -> String {
    match value {
        IdfValue::Empty => String::new(),
        IdfValue::Integer(i) => i.to_string(),
        IdfValue::Real(f) => format_real(*f),
        IdfValue::String(s) => format_string(s),
    }
}

/// Render a real number so that re-parsing yields the very same `f64`
/// **and** the very same [`IdfValue`] variant.
///
/// Rust's `Display` for `f64` prints the shortest decimal string that
/// round-trips exactly. Integral values would lose the `Real` variant
/// on re-parse (`18.0` → `18` → `Integer`), so a `.0` suffix is added
/// when the rendered text carries neither a decimal point nor an
/// exponent.
fn format_real(value: f64) -> String {
    if !value.is_finite() {
        // NaN / ±inf — not valid IDF, but do not corrupt the text.
        return format!("{value}");
    }
    let text = format!("{value}");
    if text.contains('.') || text.contains('e') || text.contains('E') {
        text
    } else {
        format!("{text}.0")
    }
}

/// Render a string field.
///
/// The text is quoted when an unquoted rendering would not survive a
/// round-trip:
/// * it is empty or has leading/trailing whitespace,
/// * it contains a field/object delimiter (`,` `;`), a comment
///   character (`!`), a quote or a line break,
/// * it would re-parse as a number (`25.2`, `1e5`, `NaN`, …).
///
/// Embedded `"` are escaped as `""` (the EnergyPlus convention).
pub fn format_string(value: &str) -> String {
    if needs_quoting(value) {
        let escaped = value.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        value.to_string()
    }
}

fn needs_quoting(value: &str) -> bool {
    if value.is_empty() {
        return true;
    }
    if value.trim() != value {
        return true;
    }
    if value.chars().any(|c| {
        matches!(
            c,
            ',' | ';' | '"' | '!' | '\n' | '\r' | '\t' | '{' | '}' | '=' | '*'
        )
    }) {
        return true;
    }
    // Would re-parse as a number → quote it so the field stays a string.
    value.parse::<f64>().is_ok() || value.parse::<i64>().is_ok()
}

// ---------------------------------------------------------------------------
// CaseSpec → IdfObject list
// ---------------------------------------------------------------------------

fn case_spec_objects(case: &CaseSpec) -> Vec<IdfObject> {
    let mut objects: Vec<IdfObject> = Vec::new();

    objects.push(object("Version", vec![value_string(EXPORT_VERSION)]));
    objects.push(object("Timestep", vec![value_int(EXPORT_TIMESTEP)]));
    objects.push(object(
        "RunPeriod",
        vec![
            value_string("AnnualRun"),
            value_int(1),  // begin month
            value_int(1),  // begin day
            value_empty(), // begin year
            value_int(12), // end month
            value_int(31), // end day
            value_empty(), // end year
            value_string("Tuesday"),
            value_string("No"),
            value_string("No"),
            value_string("No"),
            value_string("No"),
            value_string("No"),
        ],
    ));
    objects.push(object(
        "Building",
        vec![
            value_string(&case.case_id),
            value_real(0.0), // north axis
            value_string("City"),
            value_real(0.04),
            value_real(0.4),
            value_string("FullExterior"),
            value_int(25),
        ],
    ));

    // Materials / Constructions — deduplicated by material name.
    let mut materials: Vec<IdfObject> = Vec::new();
    let mut seen_materials: Vec<String> = Vec::new();
    for (_ctor_name, construction) in construction_assemblies(case) {
        for layer in &construction.layers {
            if seen_materials.iter().any(|m| m == &layer.name) {
                continue;
            }
            seen_materials.push(layer.name.clone());
            materials.push(object(
                "Material",
                vec![
                    value_string(&layer.name),
                    value_string("MediumRough"),
                    value_real(layer.thickness),
                    value_real(layer.conductivity),
                    value_real(layer.density),
                    value_real(layer.specific_heat),
                ],
            ));
        }
    }
    objects.extend(materials);

    // Constructions — one per envelope surface type (wall / roof / floor).
    for (ctor_name, construction) in construction_assemblies(case) {
        let mut fields = vec![value_string(&ctor_name)];
        for layer in &construction.layers {
            fields.push(value_string(&layer.name));
        }
        objects.push(object("Construction", fields));
    }

    // Zones and their surfaces.
    for (index, geometry) in case.geometry.iter().enumerate() {
        let zone_name = geometry
            .name
            .clone()
            .unwrap_or_else(|| format!("ZONE{}", index + 1));
        let origin = case
            .zone_origins
            .as_ref()
            .and_then(|origins| origins.get(index))
            .map(|o| (o.x, o.y))
            .unwrap_or((0.0, 0.0));

        objects.push(object(
            "Zone",
            vec![
                value_string(&zone_name),
                value_real(0.0), // direction of relative north
                value_real(origin.0),
                value_real(origin.1),
                value_real(0.0),                   // z origin
                value_int(1),                      // type
                value_int(1),                      // multiplier
                value_real(geometry.height),       // ceiling height
                value_real(geometry.volume()),     // volume
                value_real(geometry.floor_area()), // floor area
            ],
        ));

        let (x0, y0) = origin;
        let x1 = x0 + geometry.width;
        let y1 = y0 + geometry.depth;
        let h = geometry.height;

        // Floor (ground-coupled).
        objects.push(surface(
            &format!("{zone_name}-Floor"),
            "Floor",
            "FluxionFloor",
            &zone_name,
            "Ground",
            "NoSun",
            "NoWind",
            &[(x0, y0, 0.0), (x0, y1, 0.0), (x1, y1, 0.0), (x1, y0, 0.0)],
        ));

        // Roof.
        objects.push(surface(
            &format!("{zone_name}-Roof"),
            "Roof",
            "FluxionRoof",
            &zone_name,
            "Outdoors",
            "SunExposed",
            "WindExposed",
            &[(x0, y0, h), (x0, y1, h), (x1, y1, h), (x1, y0, h)],
        ));

        // Four walls.
        objects.push(surface(
            &format!("{zone_name}-Wall-South"),
            "Wall",
            "FluxionWall",
            &zone_name,
            "Outdoors",
            "SunExposed",
            "WindExposed",
            &[(x0, y0, h), (x0, y0, 0.0), (x1, y0, 0.0), (x1, y0, h)],
        ));
        objects.push(surface(
            &format!("{zone_name}-Wall-North"),
            "Wall",
            "FluxionWall",
            &zone_name,
            "Outdoors",
            "SunExposed",
            "WindExposed",
            &[(x0, y1, h), (x0, y1, 0.0), (x1, y1, 0.0), (x1, y1, h)],
        ));
        objects.push(surface(
            &format!("{zone_name}-Wall-East"),
            "Wall",
            "FluxionWall",
            &zone_name,
            "Outdoors",
            "SunExposed",
            "WindExposed",
            &[(x1, y0, h), (x1, y0, 0.0), (x1, y1, 0.0), (x1, y1, h)],
        ));
        objects.push(surface(
            &format!("{zone_name}-Wall-West"),
            "Wall",
            "FluxionWall",
            &zone_name,
            "Outdoors",
            "SunExposed",
            "WindExposed",
            &[(x0, y1, h), (x0, y1, 0.0), (x0, y0, 0.0), (x0, y0, h)],
        ));
    }

    objects.push(object(
        "GlobalGeometryRules",
        vec![
            value_string("UpperLeftCorner"),
            value_string("ClockWise"),
            value_string("World"),
        ],
    ));

    let ground = case.ground_temperature_c.unwrap_or(10.0);
    objects.push(object(
        "Site:GroundTemperature:BuildingSurface",
        (0..12).map(|_| value_real(ground)).collect(),
    ));

    objects
}

/// The envelope construction assemblies of a [`CaseSpec`], as
/// `(construction name, assembly)` pairs, in the order wall → roof →
/// floor. Per-zone assemblies (`zone_constructions`) take precedence
/// over the global ones when present.
fn construction_assemblies(
    case: &CaseSpec,
) -> Vec<(String, crate::sim::construction::Construction)> {
    let zone_construction = case.zone_constructions.as_ref().and_then(|c| c.first());
    let wall = zone_construction
        .map(|c| c.wall.clone())
        .unwrap_or_else(|| case.construction.wall.clone());
    let roof = zone_construction
        .map(|c| c.roof.clone())
        .unwrap_or_else(|| case.construction.roof.clone());
    let floor = zone_construction
        .map(|c| c.floor.clone())
        .unwrap_or_else(|| case.construction.floor.clone());
    vec![
        ("FluxionWall".to_string(), wall),
        ("FluxionRoof".to_string(), roof),
        ("FluxionFloor".to_string(), floor),
    ]
}

/// Build a `BuildingSurface:Detailed` object using the field layout the
/// importer reads (vertex count at field index 10, vertices from 11) —
/// see the module header.
#[allow(clippy::too_many_arguments)]
fn surface(
    name: &str,
    surface_type: &str,
    construction: &str,
    zone: &str,
    outside_boundary_condition: &str,
    sun_exposure: &str,
    wind_exposure: &str,
    vertices: &[(f64, f64, f64)],
) -> IdfObject {
    let mut fields = vec![
        value_string(name),
        value_string(surface_type),
        value_string(construction),
        value_string(zone),
        value_empty(), // outside boundary condition object
        value_string(outside_boundary_condition),
        value_empty(), // (field 7 in the reference model layout)
        value_string(sun_exposure),
        value_string(wind_exposure),
        value_empty(), // view factor to ground
        value_int(vertices.len() as i64),
    ];
    for (x, y, z) in vertices {
        fields.push(value_real(*x));
        fields.push(value_real(*y));
        fields.push(value_real(*z));
    }
    object("BuildingSurface:Detailed", fields)
}

// ---------------------------------------------------------------------------
// Small constructors
// ---------------------------------------------------------------------------

fn object(object_type: &str, fields: Vec<IdfValue>) -> IdfObject {
    let name = fields.first().and_then(|v| match v {
        IdfValue::String(s) if !s.is_empty() => Some(s.clone()),
        _ => None,
    });
    IdfObject {
        object_type: object_type.to_string(),
        name,
        fields,
        line: 0,
    }
}

fn value_string(value: &str) -> IdfValue {
    IdfValue::String(value.to_string())
}

fn value_real(value: f64) -> IdfValue {
    IdfValue::Real(value)
}

fn value_int(value: i64) -> IdfValue {
    IdfValue::Integer(value)
}

fn value_empty() -> IdfValue {
    IdfValue::Empty
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::idf::parser::IdfParser;

    #[test]
    fn empty_file_exports_to_empty_string() {
        let idf = IdfFile::default();
        assert_eq!(export_idf(&idf), "");
    }

    #[test]
    fn version_object_round_trips() {
        let idf = IdfParser::from_str("Version, 25.2;\n").unwrap();
        let text = export_idf(&idf);
        assert_eq!(text, "Version, 25.2;\n");
        let reparsed = IdfParser::from_str(&text).unwrap();
        assert_eq!(reparsed.objects, idf.objects);
    }

    #[test]
    fn real_fields_keep_their_variant() {
        let idf = IdfParser::from_str("Timestep, 4;\nMaterial, M, R, 18.0, 0.5;\n").unwrap();
        let text = export_idf(&idf);
        assert!(text.contains("18.0"), "text was: {text}");
        let reparsed = IdfParser::from_str(&text).unwrap();
        assert_eq!(reparsed.objects, idf.objects);
    }

    #[test]
    fn numeric_strings_are_quoted() {
        let idf = IdfParser::from_str("Material, \"25.2\", R;\n").unwrap();
        let text = export_idf(&idf);
        assert_eq!(text, "Material, \"25.2\", R;\n");
        let reparsed = IdfParser::from_str(&text).unwrap();
        assert_eq!(reparsed.objects, idf.objects);
    }

    #[test]
    fn strings_with_delimiters_are_quoted_and_escaped() {
        let idf = IdfParser::from_str("Material, \"a, b\", \"say \"\"hi\"\"\";\n").unwrap();
        let text = export_idf(&idf);
        let reparsed = IdfParser::from_str(&text).unwrap();
        assert_eq!(reparsed.objects, idf.objects);
    }

    #[test]
    fn writer_output_matches_string_output() {
        let idf = IdfParser::from_str("Version, 25.2;\nTimestep, 1;\n").unwrap();
        let mut buffer: Vec<u8> = Vec::new();
        export_idf_to_writer(&idf, &mut buffer).unwrap();
        assert_eq!(String::from_utf8(buffer).unwrap(), export_idf(&idf));
    }
}
