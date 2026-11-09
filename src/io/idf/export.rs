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
//! - [`export_schema`] — full standalone export of a
//!   [`crate::api::schema::SimulationSchemaV1`] model (issue #4702):
//!   zones, envelope surfaces, glazing
//!   (`FenestrationSurface:Detailed` + `WindowMaterial:Glazing` /
//!   `WindowMaterial:SimpleGlazingSystem`) and — for the `Ideal`
//!   template — the `ZoneHVAC` equipment chain. Anything the export
//!   cannot represent is reported as an explicit `! LOSS: …` note in
//!   the exported text.
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
//! The exclusions above are scoped to [`export_case_spec_envelope`]
//! only. The **schema** export ([`export_schema`]) *does* write glazing
//! and — for the `Ideal` template — the HVAC equipment chain, because the
//! [`crate::api::schema::SimulationSchemaV1`] model carries the data
//! those objects need.
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

use crate::api::schema::{SimulationSchemaV1, WindowSpec};
use crate::ashrae_cases::CaseSpec;
use crate::sim::hvac::HVACSystemType;
use crate::sim::per_surface_conduction::SurfaceKind;
use crate::validation::ashrae_140_cases::Orientation;

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
// Schema (SimulationSchemaV1) export — issue #4702
// ---------------------------------------------------------------------------

/// Prefix of every loss note written by [`export_schema`].
///
/// A loss note is a **comment line** in the exported IDF text:
///
/// ```text
/// ! LOSS: <object/feature dropped> — <reason>
/// ```
///
/// Exactly one note is written per dropped feature, so a reader of the
/// exported file can see what the export could not represent. Nothing is
/// ever dropped silently.
const LOSS_NOTE_PREFIX: &str = "! LOSS: ";

/// Conductivity (W/m·K) written into the emitted
/// `WindowMaterial:SimpleGlazingSystem` objects.
const GLAZING_CONDUCTIVITY: f64 = 0.9;

/// Ground temperature (°C) written by [`export_schema`].
///
/// The simulation schema carries no ground-temperature field, so a flat
/// ASHRAE 140 default is written for every month.
const SCHEMA_GROUND_TEMPERATURE_C: f64 = 18.0;

/// Export a [`SimulationSchemaV1`] model to a **standalone** IDF file.
///
/// The emitted text is a complete EnergyPlus input file — it carries its
/// own `Version`, `Timestep`, `RunPeriod`, `Building`, `Zone`,
/// `Material` / `Construction`, `BuildingSurface:Detailed` envelope,
/// glazing and HVAC objects. It is *not* a set of fragments to append to
/// another file.
///
/// # What is emitted
///
/// - **Envelope** — one `BuildingSurface:Detailed` per zone surface.
///   When the model declares per-surface geometry
///   ([`crate::api::schema::Geometry::surfaces`]) those surfaces are
///   exported verbatim (their vertex loop when present); otherwise a
///   rectangular box is synthesised from the zone's floor area, volume
///   and height.
/// - **Glazing** — every [`WindowSpec`] becomes a
///   `FenestrationSurface:Detailed` whose parent is a
///   `BuildingSurface:Detailed` **emitted by the same export** (the
///   importer silently skips a window whose parent cannot be resolved,
///   so an exportable window always has a parent). The glazing
///   properties are written as a `WindowMaterial:SimpleGlazingSystem`
///   (Name, U-Factor, SHGC, Conductivity) — the only EnergyPlus window
///   material that carries an explicit U-factor and SHGC — plus a
///   `WindowMaterial:Glazing` object holding the spectral material data.
///   The window `Construction` references the
///   `WindowMaterial:SimpleGlazingSystem` layer, because a construction
///   built only from `WindowMaterial:Glazing` yields no U-value at all
///   (see `super::convert::window_spec_from_construction`).
/// - **HVAC** — when [`crate::api::schema::ControlSet::hvac_system`] is
///   [`HVACSystemType::Ideal`], a
///   `ZoneHVAC:EquipmentConnections` → `ZoneHVAC:EquipmentList` →
///   `ZoneHVAC:IdealLoadsAirSystem` chain is emitted for **every** zone,
///   so the re-imported schema recovers
///   `controls.hvac_system == Some(HVACSystemType::Ideal)`. Any other
///   template has no IDF equipment-chain mapping and is reported as a
///   loss note.
///
/// # Loss notes
///
/// A feature that cannot be represented in IDF is **never dropped
/// silently**: one comment line of the form
///
/// ```text
/// ! LOSS: <object/feature dropped> — <reason>
/// ```
///
/// is written into the exported text (at the end of the file), one per
/// dropped feature. The currently documented loss notes are:
///
/// | Dropped feature | Note text (after the `! LOSS: ` prefix) |
/// |---|---|
/// | a non-`Ideal` HVAC template | `HVAC template {name} — only the Ideal template maps to an IDF ZoneHVAC equipment chain; no HVAC objects were written` |
/// | a window with an undetermined glazing area | `window on {parent} — the glazing area is zero or undetermined (window_area <= 0 and no window_to_wall_ratio)` |
/// | a window larger than its host surface | `window on {parent} — the glazing area {area} m² exceeds the host surface area {host} m²` |
/// | a window with a non-positive U-value | `window on {parent} — the glazing U-value {u} W/m²K is not positive, so no glazing system was written` |
/// | a window whose host surface was not exported | `window on surface {id} — the host surface was not exported, so the window has no parent` |
/// | a surface with an unknown area | `surface {id} — the area is unknown (neither an area nor a vertex loop was given)` |
/// | glazing on the floor construction | `window on the floor construction — floor glazing is not representable in an IDF` |
///
/// # Round-trip guarantees
///
/// The exported text re-imports through the merged importers:
///
/// - the **window area and U-value** through the `CaseSpec` importer
///   path (`super::convert::case_spec_from_idf`, which derives the area
///   from the fenestration polygon and the U-value from the
///   construction's `WindowMaterial:SimpleGlazingSystem`), and
/// - the **HVAC assignment** through the schema importer path
///   (`TryFrom<&IdfFile> for SimulationSchemaV1`), which reads the
///   `ZoneHVAC:IdealLoadsAirSystem` chain of the first zone.
pub fn export_schema(schema: &SimulationSchemaV1) -> String {
    let mut out = String::new();
    let mut losses: Vec<String> = Vec::new();

    // --- Site-level objects --------------------------------------------
    write_object(
        &mut out,
        &object("Version", vec![value_string(EXPORT_VERSION)]),
    );
    write_object(
        &mut out,
        &object("Timestep", vec![value_int(EXPORT_TIMESTEP)]),
    );
    write_object(
        &mut out,
        &object(
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
        ),
    );
    let building_name = if schema.metadata.name.trim().is_empty() {
        "FluxionModel".to_string()
    } else {
        schema.metadata.name.clone()
    };
    write_object(
        &mut out,
        &object(
            "Building",
            vec![
                value_string(&building_name),
                value_real(0.0), // north axis
                value_string("City"),
                value_real(0.04),
                value_real(0.4),
                value_string("FullExterior"),
                value_int(25),
            ],
        ),
    );

    // --- Zones -----------------------------------------------------------
    for (index, zone) in schema.geometry.zones.iter().enumerate() {
        write_object(&mut out, &zone_object(index, zone));
    }

    // --- Materials and constructions ------------------------------------
    let mut materials: Vec<IdfObject> = Vec::new();
    let mut material_names: Vec<String> = Vec::new();
    for (name, construction) in schema_constructions(schema) {
        for layer in &construction.layers {
            if material_names.iter().any(|m| m == &layer.name) {
                continue;
            }
            material_names.push(layer.name.clone());
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
        let mut fields = vec![value_string(&name)];
        for layer in &construction.layers {
            fields.push(value_string(&layer.name));
        }
        write_object(&mut out, &object("Construction", fields));
    }
    for material in materials {
        write_object(&mut out, &material);
    }

    // --- Envelope surfaces ------------------------------------------------
    let mut surfaces: Vec<ExportedSurface> = Vec::new();
    for (index, zone) in schema.geometry.zones.iter().enumerate() {
        let zone_name = zone_name(index, zone);
        let declared: Vec<&crate::api::schema::SurfaceSpec> = schema
            .geometry
            .surfaces
            .iter()
            .filter(|s| s.zone_index == index)
            .collect();
        let built: Vec<ExportedSurface> = if declared.is_empty() {
            synthesized_box(index, zone, &mut losses)
        } else {
            declared
                .iter()
                .filter_map(|spec| build_declared_surface(spec, zone, &zone_name, &mut losses))
                .collect()
        };
        for surface in &built {
            write_object(
                &mut out,
                &surface_object(surface, &construction_name(schema, surface.kind)),
            );
        }
        surfaces.extend(built);
    }

    // --- Glazing -----------------------------------------------------------
    let mut window_index = 0usize;
    for spec in &schema.geometry.surfaces {
        let Some(window) = &spec.window else {
            continue;
        };
        let Some(parent) = surfaces.iter().find(|s| s.name == spec.id) else {
            losses.push(format!(
                "window on surface {} — the host surface was not exported, so the window has no parent",
                spec.id
            ));
            continue;
        };
        emit_window(
            &mut out,
            &mut losses,
            &mut window_index,
            window,
            parent,
            &parent.name,
        );
    }
    // The aggregate (construction-level) glazing: the schema's wall
    // construction carries the model's wall glazing, the roof
    // construction the skylights, and the floor construction nothing
    // representable.
    if let Some(window) = &schema.constructions.wall.window {
        match aggregate_host(&surfaces, SurfaceKind::Wall) {
            Some(parent) => emit_window(
                &mut out,
                &mut losses,
                &mut window_index,
                window,
                parent,
                &parent.name,
            ),
            None => losses.push(
                "aggregate wall glazing — no exported wall surface is available to host it"
                    .to_string(),
            ),
        }
    }
    if let Some(window) = &schema.constructions.roof.window {
        match aggregate_host(&surfaces, SurfaceKind::Roof) {
            Some(parent) => emit_window(
                &mut out,
                &mut losses,
                &mut window_index,
                window,
                parent,
                &parent.name,
            ),
            None => losses.push(
                "aggregate roof glazing — no exported roof surface is available to host it"
                    .to_string(),
            ),
        }
    }
    if schema.constructions.floor.window.is_some() {
        losses.push(
            "window on the floor construction — floor glazing is not representable in an IDF"
                .to_string(),
        );
    }

    // --- HVAC ---------------------------------------------------------------
    match schema.controls.hvac_system {
        Some(HVACSystemType::Ideal) => {
            for (index, zone) in schema.geometry.zones.iter().enumerate() {
                write_hvac_chain(&mut out, &zone_name(index, zone));
            }
        }
        Some(other) => {
            losses.push(format!(
                "HVAC template {} — only the Ideal template maps to an IDF ZoneHVAC equipment chain; no HVAC objects were written",
                format!("{:?}", other)
            ));
        }
        None => {}
    }

    // --- Geometry rules and ground temperature ------------------------------
    write_object(
        &mut out,
        &object(
            "GlobalGeometryRules",
            vec![
                value_string("UpperLeftCorner"),
                value_string("ClockWise"),
                value_string("World"),
            ],
        ),
    );
    write_object(
        &mut out,
        &object(
            "Site:GroundTemperature:BuildingSurface",
            (0..12)
                .map(|_| value_real(SCHEMA_GROUND_TEMPERATURE_C))
                .collect(),
        ),
    );

    // --- Loss notes ----------------------------------------------------------
    for loss in &losses {
        out.push_str(LOSS_NOTE_PREFIX);
        out.push_str(loss);
        out.push('\n');
    }

    out
}

/// The axis a surface polygon is constant in.
#[derive(Debug, Clone, Copy)]
enum PlaneAxis {
    X,
    Y,
    Z,
}

/// The plane an exported surface polygon lies in.
///
/// The `u` / `v` coordinates are the two free axes of the plane; the
/// `constant` coordinate is the axis the plane is orthogonal to. Fenestration
/// is placed in the same plane as its host surface.
#[derive(Debug, Clone, Copy)]
struct Plane {
    axis: PlaneAxis,
    constant: f64,
    u0: f64,
    u1: f64,
    v0: f64,
    v1: f64,
}

impl Plane {
    /// The 3D point at the plane coordinates `(u, v)`.
    fn point(&self, u: f64, v: f64) -> (f64, f64, f64) {
        match self.axis {
            PlaneAxis::X => (self.constant, u, v),
            PlaneAxis::Y => (u, self.constant, v),
            PlaneAxis::Z => (u, v, self.constant),
        }
    }

    /// The plane's axis-aligned extent (m²).
    fn area(&self) -> f64 {
        (self.u1 - self.u0) * (self.v1 - self.v0)
    }

    /// The rectangle covering the plane, as four vertices.
    fn vertices(&self) -> Vec<(f64, f64, f64)> {
        vec![
            self.point(self.u0, self.v1),
            self.point(self.u1, self.v1),
            self.point(self.u1, self.v0),
            self.point(self.u0, self.v0),
        ]
    }
}

/// One exported envelope surface.
#[derive(Debug, Clone)]
struct ExportedSurface {
    /// The `BuildingSurface:Detailed` object name.
    name: String,
    /// Wall / roof / floor classification.
    kind: SurfaceKind,
    /// The zone the surface belongs to.
    zone: String,
    /// The plane the surface polygon lies in.
    plane: Plane,
    /// True when the surface already carries an explicit per-surface window.
    has_window: bool,
}

/// The name of the `Zone` object for a zone of the schema.
fn zone_name(index: usize, zone: &crate::api::schema::ZoneGeometry) -> String {
    if zone.name.trim().is_empty() {
        format!("ZONE{}", index + 1)
    } else {
        zone.name.clone()
    }
}

/// The `Zone` object of one zone.
///
/// The field layout is the one the **importer** reads (see
/// `super::convert::build_geometry`, which reads the ceiling height, the
/// volume and the floor area from fields 6, 7 and 8) — the same choice
/// the `BuildingSurface:Detailed` layout makes (see the module header).
fn zone_object(index: usize, zone: &crate::api::schema::ZoneGeometry) -> IdfObject {
    object(
        "Zone",
        vec![
            value_string(&zone_name(index, zone)),
            value_real(0.0),             // direction of relative north
            value_real(0.0),             // x origin
            value_real(0.0),             // y origin
            value_real(0.0),             // z origin
            value_int(1),                // type
            value_real(zone.height),     // ceiling height
            value_real(zone.volume),     // volume
            value_real(zone.floor_area), // floor area
        ],
    )
}

/// The `(name, construction)` pairs of the schema's construction set.
fn schema_constructions(
    schema: &SimulationSchemaV1,
) -> Vec<(String, &crate::api::schema::SurfaceConstruction)> {
    vec![
        (
            construction_name(schema, SurfaceKind::Wall),
            &schema.constructions.wall,
        ),
        (
            construction_name(schema, SurfaceKind::Roof),
            &schema.constructions.roof,
        ),
        (
            construction_name(schema, SurfaceKind::Floor),
            &schema.constructions.floor,
        ),
    ]
}

/// The `Construction` object name used for a surface kind.
fn construction_name(schema: &SimulationSchemaV1, kind: SurfaceKind) -> String {
    let (name, fallback) = match kind {
        SurfaceKind::Wall => (&schema.constructions.wall.name, "FluxionWall"),
        SurfaceKind::Roof => (&schema.constructions.roof.name, "FluxionRoof"),
        SurfaceKind::Floor => (&schema.constructions.floor.name, "FluxionFloor"),
    };
    if name.trim().is_empty() {
        fallback.to_string()
    } else {
        name.clone()
    }
}

/// The rectangular box synthesised for a zone without declared surfaces.
fn synthesized_box(
    index: usize,
    zone: &crate::api::schema::ZoneGeometry,
    losses: &mut Vec<String>,
) -> Vec<ExportedSurface> {
    let name = zone_name(index, zone);
    let width = zone.floor_area.max(0.0).sqrt();
    let depth = width;
    let height = zone.height.max(0.0);
    if width <= 0.0 || height <= 0.0 {
        losses.push(format!(
            "zone {name} — the floor area and the height are not positive, so no envelope surfaces were exported"
        ));
        return Vec::new();
    }
    let wall = |kind: SurfaceKind, surface_name: String, plane: Plane| ExportedSurface {
        name: surface_name,
        kind,
        zone: name.clone(),
        plane,
        has_window: false,
    };
    vec![
        wall(
            SurfaceKind::Floor,
            format!("{name}-Floor"),
            Plane {
                axis: PlaneAxis::Z,
                constant: 0.0,
                u0: 0.0,
                u1: width,
                v0: 0.0,
                v1: depth,
            },
        ),
        wall(
            SurfaceKind::Roof,
            format!("{name}-Roof"),
            Plane {
                axis: PlaneAxis::Z,
                constant: height,
                u0: 0.0,
                u1: width,
                v0: 0.0,
                v1: depth,
            },
        ),
        wall(
            SurfaceKind::Wall,
            format!("{name}-Wall-South"),
            Plane {
                axis: PlaneAxis::Y,
                constant: 0.0,
                u0: 0.0,
                u1: width,
                v0: 0.0,
                v1: height,
            },
        ),
        wall(
            SurfaceKind::Wall,
            format!("{name}-Wall-North"),
            Plane {
                axis: PlaneAxis::Y,
                constant: depth,
                u0: 0.0,
                u1: width,
                v0: 0.0,
                v1: height,
            },
        ),
        wall(
            SurfaceKind::Wall,
            format!("{name}-Wall-East"),
            Plane {
                axis: PlaneAxis::X,
                constant: width,
                u0: 0.0,
                u1: depth,
                v0: 0.0,
                v1: height,
            },
        ),
        wall(
            SurfaceKind::Wall,
            format!("{name}-Wall-West"),
            Plane {
                axis: PlaneAxis::X,
                constant: 0.0,
                u0: 0.0,
                u1: depth,
                v0: 0.0,
                v1: height,
            },
        ),
    ]
}

/// Build the exported surface of a declared
/// [`crate::api::schema::SurfaceSpec`].
fn build_declared_surface(
    spec: &crate::api::schema::SurfaceSpec,
    zone: &crate::api::schema::ZoneGeometry,
    zone_name: &str,
    losses: &mut Vec<String>,
) -> Option<ExportedSurface> {
    let Some(area) = spec.effective_area() else {
        losses.push(format!(
            "surface {} — the area is unknown (neither an area nor a vertex loop was given)",
            spec.id
        ));
        return None;
    };
    if area <= 0.0 {
        losses.push(format!(
            "surface {} — the area is not positive, so the surface was not exported",
            spec.id
        ));
        return None;
    }
    let plane = match &spec.vertices {
        Some(vertices) if vertices.len() >= 3 => {
            let points: Vec<(f64, f64, f64)> = vertices.iter().map(|v| (v.x, v.y, v.z)).collect();
            plane_from_vertices(&points)
        }
        Some(_) => {
            losses.push(format!(
                "surface {} — the vertex loop has fewer than three vertices, so the surface was not exported",
                spec.id
            ));
            return None;
        }
        None => Some(synthesized_plane(spec, zone, area)),
    };
    let plane = plane?;
    Some(ExportedSurface {
        name: spec.id.clone(),
        kind: spec.kind,
        zone: zone_name.to_string(),
        plane,
        has_window: spec.window.is_some(),
    })
}

/// The plane of a declared surface that has no vertex loop.
fn synthesized_plane(
    spec: &crate::api::schema::SurfaceSpec,
    zone: &crate::api::schema::ZoneGeometry,
    area: f64,
) -> Plane {
    let width = zone.floor_area.max(0.0).sqrt();
    let depth = width;
    let height = zone.height.max(0.0);
    match spec.kind {
        SurfaceKind::Wall => {
            let wall_width = if height > 0.0 { area / height } else { area };
            match spec.orientation.unwrap_or(Orientation::South) {
                Orientation::North => Plane {
                    axis: PlaneAxis::Y,
                    constant: depth,
                    u0: 0.0,
                    u1: wall_width,
                    v0: 0.0,
                    v1: height,
                },
                Orientation::East => Plane {
                    axis: PlaneAxis::X,
                    constant: width,
                    u0: 0.0,
                    u1: wall_width,
                    v0: 0.0,
                    v1: height,
                },
                Orientation::West => Plane {
                    axis: PlaneAxis::X,
                    constant: 0.0,
                    u0: 0.0,
                    u1: wall_width,
                    v0: 0.0,
                    v1: height,
                },
                _ => Plane {
                    axis: PlaneAxis::Y,
                    constant: 0.0,
                    u0: 0.0,
                    u1: wall_width,
                    v0: 0.0,
                    v1: height,
                },
            }
        }
        SurfaceKind::Roof | SurfaceKind::Floor => {
            let span = if width > 0.0 { area / width } else { area };
            Plane {
                axis: PlaneAxis::Z,
                constant: if spec.kind == SurfaceKind::Roof {
                    height
                } else {
                    0.0
                },
                u0: 0.0,
                u1: width,
                v0: 0.0,
                v1: span,
            }
        }
    }
}

/// The plane of an arbitrary polygon: the axis it is most constant in.
fn plane_from_vertices(points: &[(f64, f64, f64)]) -> Option<Plane> {
    if points.len() < 3 {
        return None;
    }
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_z, mut max_z) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in points {
        min_x = min_x.min(p.0);
        max_x = max_x.max(p.0);
        min_y = min_y.min(p.1);
        max_y = max_y.max(p.1);
        min_z = min_z.min(p.2);
        max_z = max_z.max(p.2);
    }
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let dz = max_z - min_z;
    if dz <= dy && dz <= dx {
        Some(Plane {
            axis: PlaneAxis::Z,
            constant: min_z,
            u0: min_x,
            u1: max_x,
            v0: min_y,
            v1: max_y,
        })
    } else if dy <= dx && dy <= dz {
        Some(Plane {
            axis: PlaneAxis::Y,
            constant: min_y,
            u0: min_x,
            u1: max_x,
            v0: min_z,
            v1: max_z,
        })
    } else {
        Some(Plane {
            axis: PlaneAxis::X,
            constant: min_x,
            u0: min_y,
            u1: max_y,
            v0: min_z,
            v1: max_z,
        })
    }
}

/// The `BuildingSurface:Detailed` object of an exported surface.
fn surface_object(surface: &ExportedSurface, construction: &str) -> IdfObject {
    let (surface_type, boundary, sun, wind) = match surface.kind {
        SurfaceKind::Wall => ("Wall", "Outdoors", "SunExposed", "WindExposed"),
        SurfaceKind::Roof => ("Roof", "Outdoors", "SunExposed", "WindExposed"),
        SurfaceKind::Floor => ("Floor", "Ground", "NoSun", "NoWind"),
    };
    let mut fields = vec![
        value_string(&surface.name),
        value_string(surface_type),
        value_string(construction),
        value_string(&surface.zone),
        value_empty(), // outside boundary condition object
        value_string(boundary),
        value_empty(),
        value_string(sun),
        value_string(wind),
        value_empty(), // view factor to ground
        value_int(surface.plane.vertices().len() as i64),
    ];
    for (x, y, z) in surface.plane.vertices() {
        fields.push(value_real(x));
        fields.push(value_real(y));
        fields.push(value_real(z));
    }
    object("BuildingSurface:Detailed", fields)
}

/// The host surface of an aggregate (construction-level) window: the first
/// surface of the requested kind that carries no explicit window.
fn aggregate_host<'a>(
    surfaces: &'a [ExportedSurface],
    kind: SurfaceKind,
) -> Option<&'a ExportedSurface> {
    surfaces.iter().find(|s| s.kind == kind && !s.has_window)
}

/// The rectangle of a window of `area` m² placed on `host`.
///
/// The window spans the full width of the host surface and is centred
/// vertically; its height is `area / width`, so the polygon's area equals
/// the requested area exactly. `None` when the window does not fit.
fn window_vertices(host: &Plane, area: f64) -> Option<Vec<(f64, f64, f64)>> {
    let width = host.u1 - host.u0;
    let span = host.v1 - host.v0;
    if width <= 0.0 || span <= 0.0 {
        return None;
    }
    let height = area / width;
    if height > span + 1e-9 {
        return None;
    }
    let v0 = host.v0 + (span - height) / 2.0;
    let v1 = v0 + height;
    Some(vec![
        host.point(host.u0, v0),
        host.point(host.u0, v1),
        host.point(host.u1, v1),
        host.point(host.u1, v0),
    ])
}

/// Emit the glazing objects of one window.
fn emit_window(
    out: &mut String,
    losses: &mut Vec<String>,
    index: &mut usize,
    window: &WindowSpec,
    host: &ExportedSurface,
    host_name: &str,
) {
    let area = window.effective_area(host.plane.area());
    if area <= 0.0 {
        losses.push(format!(
            "window on {host_name} — the glazing area is zero or undetermined (window_area <= 0 and no window_to_wall_ratio)"
        ));
        return;
    }
    if window.window_u_value <= 0.0 {
        losses.push(format!(
            "window on {host_name} — the glazing U-value {} W/m²K is not positive, so no glazing system was written",
            window.window_u_value
        ));
        return;
    }
    let Some(vertices) = window_vertices(&host.plane, area) else {
        losses.push(format!(
            "window on {host_name} — the glazing area {area} m² exceeds the host surface area {} m²",
            host.plane.area()
        ));
        return;
    };

    let n = *index;
    *index += 1;
    let glazing_name = format!("FluxionGlazing{n}");
    let system_name = format!("FluxionGlazingSystem{n}");
    let construction_name = format!("FluxionWindowConstruction{n}");
    let window_name = format!("FluxionWindow{n}");

    // Material data of the glazing (spectral properties).
    write_object(
        out,
        &object(
            "WindowMaterial:Glazing",
            vec![
                value_string(&glazing_name),
                value_string("SpectralAverage"),
                value_real(0.003), // thickness
                value_real(0.837), // solar transmittance at normal incidence
                value_real(0.075), // front side solar reflectance
                value_real(0.075), // back side solar reflectance
                value_real(0.837), // visible transmittance
                value_real(0.075), // front side visible reflectance
                value_real(0.075), // back side visible reflectance
                value_real(0.0),   // infrared transmittance
                value_real(0.84),  // front side infrared hemispherical emissivity
                value_real(0.84),  // back side infrared hemispherical emissivity
                value_real(GLAZING_CONDUCTIVITY),
            ],
        ),
    );
    // The U-factor / SHGC carrier: `WindowMaterial:Glazing` carries
    // neither, so the construction must reference the simple glazing
    // system for the U-value to survive the round-trip.
    write_object(
        out,
        &object(
            "WindowMaterial:SimpleGlazingSystem",
            vec![
                value_string(&system_name),
                value_real(window.window_u_value),
                value_real(window.window_shgc),
                value_real(GLAZING_CONDUCTIVITY),
            ],
        ),
    );
    write_object(
        out,
        &object(
            "Construction",
            vec![value_string(&construction_name), value_string(&system_name)],
        ),
    );

    // `FenestrationSurface:Detailed` — the field layout is the one the
    // importer reads (number of vertices at field 8, vertices from 9).
    let mut fields = vec![
        value_string(&window_name),
        value_string("Window"),
        value_string(&construction_name),
        value_string(host_name),
        value_empty(), // outside boundary condition object
        value_empty(), // view factor to ground
        value_empty(), // frame and divider name
        value_empty(), // multiplier
        value_int(vertices.len() as i64),
    ];
    for (x, y, z) in &vertices {
        fields.push(value_real(*x));
        fields.push(value_real(*y));
        fields.push(value_real(*z));
    }
    write_object(out, &object("FenestrationSurface:Detailed", fields));
}

/// The `ZoneHVAC` equipment chain of one zone (ideal loads).
fn write_hvac_chain(out: &mut String, zone: &str) {
    write_object(
        out,
        &object(
            "ZoneHVAC:EquipmentConnections",
            vec![
                value_string(&format!("{zone}_EquipmentConnections")),
                value_string(zone),
                value_string(&format!("{zone}_EquipmentList")),
                value_string(&format!("{zone}_Inlet Node")),
                value_string(&format!("{zone}_Exhaust Node")),
                value_string(&format!("{zone}_Zone Air Node")),
                value_string(&format!("{zone}_Return Air Node")),
            ],
        ),
    );
    write_object(
        out,
        &object(
            "ZoneHVAC:EquipmentList",
            vec![
                value_string(&format!("{zone}_EquipmentList")),
                value_string("SequentialLoad"),
                value_string("ZoneHVAC:IdealLoadsAirSystem"),
                value_string(&format!("{zone}_IdealLoads")),
                value_int(1), // cooling sequence
                value_int(1), // heating sequence
            ],
        ),
    );
    write_object(
        out,
        &object(
            "ZoneHVAC:IdealLoadsAirSystem",
            vec![
                value_string(&format!("{zone}_IdealLoads")),
                value_string(&format!("{zone}_Supply Inlet Node")),
                value_string(&format!("{zone}_Exhaust Node")),
                value_string(&format!("{zone}_Supply Air Node")),
                value_real(50.0),        // maximum heating supply air temperature
                value_real(13.0),        // minimum cooling supply air temperature
                value_real(0.0156),      // maximum heating supply air humidity ratio
                value_real(0.0077),      // minimum cooling supply air humidity ratio
                value_string("NoLimit"), // heating limit
                value_string("NoLimit"), // cooling limit
                value_string("ConstantSupplyHumidityRatio"),
                value_real(0.7), // cooling sensible heat ratio
                value_empty(),
                value_empty(),
                value_string("None"), // outdoor economizer type
            ],
        ),
    );
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
