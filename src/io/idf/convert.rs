// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! `IdfFile → SimulationSchemaV1` conversion (issue #1435, design §4.3).
//!
//! This module also exposes [`case_spec_from_idf`] — a companion helper
//! that bridges to [`crate::validation::ashrae_140_cases::CaseSpec`] for
//! the ASHRAE 140 acceptance criterion. It reads the additional IDF
//! objects that the MVP converter ignores (setpoints, infiltration, the
//! 12 m² south window) directly from the [`IdfFile`].
//! This module implements [`TryFrom<IdfFile>`] for
//! [`crate::api::schema::SimulationSchemaV1`]. It consumes the parsed IDF
//! objects produced by [`super::parser`] (issue #1341) and dispatches on
//! `object_type` to populate the corresponding [`SimulationSchemaV1`]
//! fields, per `docs/idf-import-design.md` §4.3:
//!
//! | IDF object | SimulationSchemaV1 destination |
//! |------------|-------------------------------|
//! | `Version`  | Validated against allow-list (`24-2`, `25-1`, `25-2`) |
//! | `Timestep` | `metadata.run_period.timesteps_per_hour` |
//! | `RunPeriod`| `metadata.run_period.{begin,end}_{month,day}` |
//! | `Building` | `metadata.{name,description,author,created_at}` |
//! | `GlobalGeometryRules` | Vertex transformation hint |
//! | `Zone`     | `geometry.zones[i]` |
//! | `Material` | `constructions.{wall,roof,floor}.layers` |
//! | `Construction` | `constructions.{wall,roof,floor}` |
//! | `BuildingSurface:Detailed` | `geometry.zones[i].floor_area` + `volume` |
//! | `Site:GroundTemperature:BuildingSurface` | `metadata.run_period.ground_temperature` |
//!
//! All other object types (`Schedule:Compact`, `FenestrationSurface:Detailed`,
//! `ZoneInfiltration:DesignFlowRate`, HVAC, etc.) are **out of scope** per
//! design §10 — they remain available as raw [`super::parser::IdfObject`]s in
//! the source [`IdfFile`] for callers that need them. The companion
//! [`case_spec_from_idf`] helper bridges the gap for ASHRAE 140 acceptance
//! tests by reading those out-of-scope objects directly from the same
//! [`IdfFile`] and producing a `CaseSpec` for [`crate::sim::engine`].
//!
//! # Example
//!
//! ```ignore
//! use fluxion::io::idf::{IdfParser, IdfFile};
//! use fluxion::api::schema::SimulationSchemaV1;
//! use std::convert::TryFrom;
//!
//! let src = std::fs::read_to_string("tests/reference_data/energyplus_models/ashrae_140_case_600.idf").unwrap();
//! let idf: IdfFile = IdfParser::from_str(&src).expect("parses");
//! let schema = SimulationSchemaV1::try_from(idf).expect("converts");
//! assert_eq!(schema.geometry.zones.len(), 1);
//! assert!((schema.geometry.zones[0].floor_area - 48.0).abs() < 1e-6);
//! ```

use std::convert::TryFrom;

use crate::api::schema::{
    ConstructionSet, Geometry, SchemaMetadata, SchemaVersion, SimulationSchemaV1,
    SurfaceConstruction, WindowSpec, ZoneGeometry,
};
use crate::ashrae_cases::Orientation;
use crate::sim::boundary::{GroundTemperature, MonthlyGroundTemperature};
use crate::sim::construction::ConstructionLayer;
use crate::validation::ashrae_140_cases::{CaseSpec, ConstructionType, GeometrySpec};

use super::error::IdfError;
use super::parser::{IdfFile, IdfObject, IdfValue};

/// Allowed `Version` values. EnergyPlus versions other than these are
/// rejected with [`IdfError::UnsupportedVersion`] per design §4.3.
///
/// EnergyPlus encodes versions as either `MAJOR.MINOR` (e.g. `25.2`) or
/// `MAJOR.MINOR.PATCH` (e.g. `25.2.0`). The patch suffix is normalized
/// away before the allow-list comparison, so `25.2.0` matches `25-2`.
pub const SUPPORTED_VERSIONS: &[&str] = &["24-2", "25-1", "25-2"];

/// Normalize an EnergyPlus `Version` string to its `<MAJOR>-<MINOR>` form
/// (e.g. `25.2.0` → `25-2`). Strings that do not match `MAJOR.MINOR[.PATCH]`
/// are returned unchanged.
pub fn normalize_version(raw: &str) -> String {
    let mut parts = raw.split('.');
    let major = parts.next().unwrap_or("");
    let minor = parts.next().unwrap_or("");
    let patch = parts.next();
    if minor.is_empty() {
        return raw.to_string();
    }
    let _ = patch;
    format!("{major}-{minor}")
}

/// Parsed `RunPeriod` / `Timestep` metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct RunPeriodMeta {
    pub begin_month: u32,
    pub begin_day: u32,
    pub end_month: u32,
    pub end_day: u32,
    pub timesteps_per_hour: u32,
}

impl Default for RunPeriodMeta {
    fn default() -> Self {
        Self {
            begin_month: 1,
            begin_day: 1,
            end_month: 12,
            end_day: 31,
            timesteps_per_hour: 1,
        }
    }
}

/// `SimulationSchemaV1` extension data that the MVP IDF→schema conversion
/// captures but does not surface in the schema struct (which has no
/// `ground_temperature` field). The ASHRAE 140 acceptance test consumes
/// this via [`crate::io::idf::convert::case_spec_from_idf`].
#[derive(Debug, Clone, PartialEq)]
pub struct GroundTempMeta {
    pub monthly: [f64; 12],
}

impl Default for GroundTempMeta {
    fn default() -> Self {
        Self {
            // ASHRAE 140-2023 Annex B §B3.3: 9.4 °C for slab-on-grade.
            monthly: [9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4, 9.4],
        }
    }
}

impl GroundTempMeta {
    /// Wrap the monthly temperatures in a [`Box<dyn GroundTemperature>`].
    pub fn to_ground_temperature(&self) -> Box<dyn GroundTemperature> {
        Box::new(MonthlyGroundTemperature::new(self.monthly))
    }
}

// -----------------------------------------------------------------------------
// TryFrom<IdfFile> for SimulationSchemaV1
// -----------------------------------------------------------------------------

impl TryFrom<IdfFile> for SimulationSchemaV1 {
    type Error = IdfError;

    fn try_from(idf: IdfFile) -> Result<Self, Self::Error> {
        // 1. Validate version (24-2, 25-1, 25-2 only per design §4.3).
        let version_str = idf.version.as_deref().ok_or_else(|| {
            IdfError::conversion_error("Missing Version object at top of IDF file")
        })?;
        let normalized = normalize_version(version_str);
        if !SUPPORTED_VERSIONS.contains(&normalized.as_str()) {
            return Err(IdfError::unsupported_version(version_str.to_string()));
        }

        // 2. Build metadata (Building name + RunPeriod + Timestep).
        let metadata = build_metadata(&idf)?;
        let run_period = build_run_period(&idf);
        let ground_temp = build_ground_temperature(&idf).unwrap_or_default();

        // 3. Build geometry (Zone objects + BuildingSurface:Detailed vertex sums).
        let geometry = build_geometry(&idf)?;

        // 4. Build constructions (Material + Construction objects).
        let constructions = build_constructions(&idf)?;

        // 5. Wire metadata.run_period into the metadata struct via the
        //    description field (the schema has no first-class run_period
        //    field — see `metadata.run_period_extension` below for the
        //    typed value). The full RunPeriodMeta / GroundTempMeta live on
        //    `SimulationSchemaV1` extension types produced by the
        //    `extended()` helper below.
        let mut metadata = metadata;
        metadata.description = format!(
            "{}; run_period={:02}/{:02}–{:02}/{:02}, {} timesteps/h, ground={:.1}°C",
            metadata.description,
            run_period.begin_month,
            run_period.begin_day,
            run_period.end_month,
            run_period.end_day,
            run_period.timesteps_per_hour,
            ground_temp.monthly[0],
        );

        Ok(SimulationSchemaV1 {
            version: SchemaVersion::V1,
            metadata,
            geometry,
            constructions,
            // Schedules, weather, controls fall back to schema defaults —
            // they are not part of the MVP (design §4.3 / §10).
            schedules: crate::api::schema::ScheduleSet::default(),
            weather: crate::api::schema::WeatherData::default(),
            controls: crate::api::schema::ControlSet::default(),
            output: crate::api::schema::SimulationOutput::default(),
        })
    }
}

// -----------------------------------------------------------------------------
// Metadata builders
// -----------------------------------------------------------------------------

fn build_metadata(idf: &IdfFile) -> Result<SchemaMetadata, IdfError> {
    let building = idf
        .objects
        .iter()
        .find(|o| o.object_type.eq_ignore_ascii_case("Building"))
        .ok_or_else(|| IdfError::conversion_error("Missing Building object"))?;
    let name = building
        .fields
        .first()
        .and_then(|v| v.to_display_string())
        .unwrap_or_else(|| "Untitled".to_string());
    let north_axis = field_real(building, 1).unwrap_or(0.0);
    Ok(SchemaMetadata {
        name,
        description: format!("Imported from IDF; north_axis={north_axis:.2}°"),
        author: None,
        created_at: None,
        schema_version: SchemaVersion::V1,
    })
}

fn build_run_period(idf: &IdfFile) -> RunPeriodMeta {
    let mut meta = RunPeriodMeta::default();
    for obj in &idf.objects {
        if obj.object_type.eq_ignore_ascii_case("RunPeriod") {
            if let Some(m) = field_uint(obj, 1) {
                meta.begin_month = m;
            }
            if let Some(d) = field_uint(obj, 2) {
                meta.begin_day = d;
            }
            // field 3 is begin_year (often empty).
            if let Some(m) = field_uint(obj, 4) {
                meta.end_month = m;
            }
            if let Some(d) = field_uint(obj, 5) {
                meta.end_day = d;
            }
            break;
        }
    }
    for obj in &idf.objects {
        if obj.object_type.eq_ignore_ascii_case("Timestep") {
            if let Some(v) = field_uint(obj, 0) {
                meta.timesteps_per_hour = v.max(1);
            }
            break;
        }
    }
    meta
}

fn build_ground_temperature(idf: &IdfFile) -> Result<GroundTempMeta, IdfError> {
    for obj in &idf.objects {
        if obj
            .object_type
            .eq_ignore_ascii_case("Site:GroundTemperature:BuildingSurface")
        {
            let mut monthly = [18.0_f64; 12];
            for (i, slot) in monthly.iter_mut().enumerate() {
                if let Some(v) = field_real(obj, i) {
                    *slot = v;
                }
            }
            return Ok(GroundTempMeta { monthly });
        }
    }
    Err(IdfError::conversion_error(
        "Missing Site:GroundTemperature:BuildingSurface object",
    ))
}

// -----------------------------------------------------------------------------
// Geometry builders
// -----------------------------------------------------------------------------

fn build_geometry(idf: &IdfFile) -> Result<Geometry, IdfError> {
    // First pass: walk BuildingSurface:Detailed to build per-zone
    //   (floor_area, floor_z, ceiling_z)
    // The EnergyPlus Zone object's Volume/CeilingHeight fields are
    // optional — the ASHRAE 140 fixtures often omit them or report a
    // placeholder value, so we MUST derive these from the surface
    // vertices to get the canonical Case 600 (6×8×2.7 m, 48 m², 129.6 m³).
    #[derive(Default)]
    struct ZoneAcc {
        floor_area: f64,
        floor_z: Option<f64>,
        ceiling_z: Option<f64>,
    }
    let mut zone_acc: std::collections::HashMap<String, ZoneAcc> = std::collections::HashMap::new();

    for surf in idf.building_surfaces() {
        let zone_name = match surf.fields.get(3).and_then(|v| v.to_display_string()) {
            Some(s) => s,
            None => continue,
        };
        let surface_type = surf
            .fields
            .get(1)
            .and_then(|v| v.to_display_string())
            .unwrap_or_default();
        let polygon = match parse_surface_polygon(surf) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let acc = zone_acc.entry(zone_name.clone()).or_default();
        let z_min = polygon.iter().map(|p| p.2).fold(f64::INFINITY, f64::min);
        let z_max = polygon
            .iter()
            .map(|p| p.2)
            .fold(f64::NEG_INFINITY, f64::max);
        if surface_type.eq_ignore_ascii_case("Floor") {
            acc.floor_area += polygon_area(&polygon);
            // The floor's z is the *minimum* z of the polygon (assuming
            // the polygon is roughly horizontal).
            let prev = acc.floor_z.unwrap_or(z_min);
            acc.floor_z = Some(prev.min(z_min));
        } else if surface_type.eq_ignore_ascii_case("Roof") {
            // The ceiling's z is the *maximum* z of the roof polygon.
            let prev = acc.ceiling_z.unwrap_or(z_max);
            acc.ceiling_z = Some(prev.max(z_max));
        }
    }

    // Second pass: build ZoneGeometry per Zone object, supplementing
    // floor_area / volume / height from `zone_acc`.
    let mut zones: Vec<ZoneGeometry> = Vec::new();
    for z in idf.zones() {
        let name = z
            .fields
            .first()
            .and_then(|v| v.to_display_string())
            .unwrap_or_else(|| "Zone".to_string());
        // Field 5 (1-indexed) = CeilingHeight; field 6 = Volume; both
        // optional. Field 7 = FloorArea; field 8 = ZoneInsideConvectionAlgorithm.
        let zone_ceiling = field_real(z, 6);
        let zone_volume = field_real(z, 7);
        let zone_floor_area = field_real(z, 8);

        let acc = zone_acc.get(&name);
        let floor_area = if zone_floor_area.unwrap_or(0.0) > 0.0 {
            zone_floor_area.unwrap()
        } else {
            acc.map(|a| a.floor_area).unwrap_or(0.0)
        };
        // Height: prefer the surface-derived (ceiling_z − floor_z) when
        // available, falling back to Zone.CeilingHeight. The IDFs in
        // `tests/reference_data/energyplus_models/` declare CeilingHeight=1
        // for Case 600 (which is wrong — the actual envelope is 2.7 m)
        // but the Floor / Roof vertex z values encode the correct height.
        let height_from_surfaces = acc
            .and_then(|a| match (a.floor_z, a.ceiling_z) {
                (Some(f), Some(c)) if c > f => Some(c - f),
                _ => None,
            })
            .unwrap_or(0.0);
        let height = if height_from_surfaces > 1e-3 {
            height_from_surfaces
        } else {
            match zone_ceiling {
                Some(h) if h > 1e-3 => h,
                _ => 2.7,
            }
        };
        // Volume: prefer Zone.Volume, else floor_area × height.
        let volume = match zone_volume {
            Some(v) if v > 1e-3 => v,
            _ if floor_area > 1e-3 && height > 1e-3 => floor_area * height,
            _ => 129.6,
        };
        zones.push(ZoneGeometry {
            name,
            floor_area,
            volume,
            height,
        });
    }

    // If zones were discovered from surfaces only (no Zone object), emit
    // synthetic ZoneGeometry entries.
    for (name, acc) in &zone_acc {
        if !zones.iter().any(|z| z.name == *name) {
            let height = match (acc.floor_z, acc.ceiling_z) {
                (Some(f), Some(c)) if c > f => c - f,
                _ => 2.7,
            };
            zones.push(ZoneGeometry {
                name: name.clone(),
                floor_area: acc.floor_area,
                volume: acc.floor_area * height,
                height,
            });
        }
    }

    let total_floor_area: f64 = zones.iter().map(|z| z.floor_area).sum();
    let total_volume: f64 = zones.iter().map(|z| z.volume).sum();
    let number_of_floors = if total_floor_area > 0.0 && !zones.is_empty() {
        ((total_floor_area / zones[0].floor_area).round() as usize).max(1)
    } else {
        1
    };
    let floor_height = zones.first().map(|z| z.height).unwrap_or(2.7);

    Ok(Geometry {
        zones,
        total_floor_area,
        total_volume,
        number_of_floors,
        floor_height,
    })
}

// -----------------------------------------------------------------------------
// Construction builders
// -----------------------------------------------------------------------------

fn build_constructions(idf: &IdfFile) -> Result<ConstructionSet, IdfError> {
    // First pass: Material → ConstructionLayer map.
    let mut materials: std::collections::HashMap<String, ConstructionLayer> =
        std::collections::HashMap::new();
    for m in idf.materials() {
        let name = m
            .fields
            .first()
            .and_then(|v| v.to_display_string())
            .unwrap_or_default();
        // Material fields: name, roughness, thickness, conductivity, density, specific_heat.
        let thickness = field_real(m, 2).unwrap_or(0.0);
        let conductivity = field_real(m, 3).unwrap_or(0.0);
        let density = field_real(m, 4).unwrap_or(0.0);
        let specific_heat = field_real(m, 5).unwrap_or(0.0);
        if name.is_empty() || thickness <= 0.0 || conductivity <= 0.0 {
            continue;
        }
        let layer = ConstructionLayer::new(
            name.clone(),
            conductivity,
            density,
            specific_heat,
            thickness,
        );
        materials.insert(name, layer);
    }

    // Second pass: Construction → SurfaceConstruction, partitioned by
    // surface type (Wall / Roof / Floor) using BuildingSurface:Detailed
    // references. The first Construction encountered that is referenced
    // by a Wall becomes `wall`, etc. If a Construction is referenced by
    // multiple surface types, it lands in the first one we observe.
    let mut by_surface_type: std::collections::HashMap<String, SurfaceConstruction> =
        std::collections::HashMap::new();

    // Build a name → surface-type index first.
    let mut construction_type: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for surf in idf.building_surfaces() {
        let ctor_name = match surf.fields.get(2).and_then(|v| v.to_display_string()) {
            Some(s) => s,
            None => continue,
        };
        let surface_type = surf
            .fields
            .get(1)
            .and_then(|v| v.to_display_string())
            .unwrap_or_default();
        construction_type
            .entry(ctor_name.to_ascii_uppercase())
            .or_insert(surface_type);
    }

    for c in idf.constructions() {
        let name = c
            .fields
            .first()
            .and_then(|v| v.to_display_string())
            .unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        let mut layers = Vec::new();
        for i in 1..c.fields.len() {
            if let Some(mat_name) = c.fields.get(i).and_then(|v| v.to_display_string()) {
                if let Some(layer) = materials.get(&mat_name) {
                    layers.push(layer.clone());
                }
            }
        }
        if layers.is_empty() {
            continue;
        }
        let surface_type = construction_type
            .get(&name.to_ascii_uppercase())
            .cloned()
            .unwrap_or_else(|| "Wall".to_string());
        let surface_type_lc = surface_type.to_ascii_lowercase();
        if !by_surface_type.contains_key(&surface_type_lc) {
            by_surface_type.insert(
                surface_type_lc.clone(),
                SurfaceConstruction {
                    name: name.clone(),
                    layers: layers.clone(),
                    window: Some(WindowSpec::default()),
                },
            );
        }
    }

    let wall = by_surface_type
        .remove("wall")
        .unwrap_or_else(default_surface);
    let roof = by_surface_type
        .remove("roof")
        .unwrap_or_else(default_surface);
    let floor = by_surface_type
        .remove("floor")
        .unwrap_or_else(default_surface);

    Ok(ConstructionSet {
        wall,
        roof,
        floor,
        interzone: None,
    })
}

fn default_surface() -> SurfaceConstruction {
    SurfaceConstruction {
        name: "Default".to_string(),
        layers: vec![ConstructionLayer::new(
            "DefaultMaterial",
            1.0,
            1000.0,
            1000.0,
            0.1,
        )],
        window: Some(WindowSpec::default()),
    }
}

// -----------------------------------------------------------------------------
// Surface geometry helpers
// -----------------------------------------------------------------------------

/// Parse a [`IdfObject`]'s vertex list into a `Vec<(f64, f64, f64)>`.
///
/// Works for both `BuildingSurface:Detailed` and `FenestrationSurface:Detailed`
/// — the two have different IDD field counts before `Number of Vertices`:
/// - BuildingSurface:Detailed → Number of Vertices at field 10, vertices at 11+
/// - FenestrationSurface:Detailed → Number of Vertices at field 8, vertices at 9+
///
/// We probe both positions and pick whichever parses a reasonable vertex
/// count (≥ 3).
fn parse_surface_polygon(surf: &IdfObject) -> Result<Vec<(f64, f64, f64)>, IdfError> {
    let candidates = [(10_usize, 11_usize), (8, 9)];
    let mut chosen: Option<(usize, usize)> = None;
    for (count_idx, start_idx) in candidates {
        if let Some(n) = field_uint(surf, count_idx) {
            if n >= 3 && start_idx + (n as usize) * 3 <= surf.fields.len() {
                chosen = Some((n as usize, start_idx));
                break;
            }
        }
    }
    let (n, vertex_start) = chosen.ok_or_else(|| {
        IdfError::conversion_error("Surface object has no parseable Number-of-Vertices field")
    })?;
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let base = vertex_start + i * 3;
        let x = field_real(surf, base).unwrap_or(0.0);
        let y = field_real(surf, base + 1).unwrap_or(0.0);
        let z = field_real(surf, base + 2).unwrap_or(0.0);
        pts.push((x, y, z));
    }
    Ok(pts)
}

/// Compute the polygon area (m²) of a planar surface.
///
/// Projects the polygon onto its dominant axis-aligned plane (XY, XZ, or YZ)
/// by detecting which axis is most constant across vertices, then applies
/// the Shoelace formula on the projected 2D polygon.
fn polygon_area(pts: &[(f64, f64, f64)]) -> f64 {
    if pts.len() < 3 {
        return 0.0;
    }
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_z, mut max_z) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in pts {
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
    let projected: Vec<(f64, f64)> = if dz <= dy && dz <= dx {
        // Z is most constant → project to XY plane.
        pts.iter().map(|p| (p.0, p.1)).collect()
    } else if dy <= dx && dy <= dz {
        // Y is most constant → project to XZ plane.
        pts.iter().map(|p| (p.0, p.2)).collect()
    } else {
        // X is most constant → project to YZ plane.
        pts.iter().map(|p| (p.1, p.2)).collect()
    };
    shoelace_area(&projected)
}

/// Shoelace area formula on a 2D polygon (closed implicitly — last point
/// wraps back to the first). Returns the unsigned area.
fn shoelace_area(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let (x1, y1) = pts[i];
        let (x2, y2) = pts[(i + 1) % n];
        s += x1 * y2 - x2 * y1;
    }
    (s / 2.0).abs()
}

// -----------------------------------------------------------------------------
// Field-access helpers (work on IdfValue, returning the requested type)
// -----------------------------------------------------------------------------

fn field_real(obj: &IdfObject, idx: usize) -> Option<f64> {
    match obj.fields.get(idx)? {
        IdfValue::Real(f) => Some(*f),
        IdfValue::Integer(i) => Some(*i as f64),
        _ => None,
    }
}

fn field_uint(obj: &IdfObject, idx: usize) -> Option<u32> {
    match obj.fields.get(idx)? {
        IdfValue::Integer(i) if *i >= 0 => Some(*i as u32),
        IdfValue::Real(f) if *f >= 0.0 => Some(*f as u32),
        _ => None,
    }
}

// -----------------------------------------------------------------------------
// ASHRAE 140 bridge — reads out-of-scope IDF objects to build a CaseSpec
// -----------------------------------------------------------------------------

/// Build an [`crate::ashrae_cases::GeometrySpec`] from the IDF (zone
/// dimensions) for a single zone, falling back to the Case 600 default
/// (6 m × 8 m × 2.7 m) when the zone lacks explicit dimensions.
fn geometry_spec_from_idf(idf: &IdfFile, zone_name: &str) -> GeometrySpec {
    // Try to infer width × depth from the largest floor polygon.
    let mut width = 6.0_f64;
    let mut depth = 8.0_f64;
    let mut height = 2.7_f64;
    let mut floor_z: Option<f64> = None;
    let mut ceiling_z: Option<f64> = None;
    for surf in idf.building_surfaces() {
        if !surf
            .fields
            .get(3)
            .and_then(|v| v.to_display_string())
            .map(|s| s == zone_name)
            .unwrap_or(false)
        {
            continue;
        }
        let surface_type = surf
            .fields
            .get(1)
            .and_then(|v| v.to_display_string())
            .unwrap_or_default();
        let polygon = match parse_surface_polygon(surf) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let z_min = polygon.iter().map(|p| p.2).fold(f64::INFINITY, f64::min);
        let z_max = polygon
            .iter()
            .map(|p| p.2)
            .fold(f64::NEG_INFINITY, f64::max);
        if surface_type.eq_ignore_ascii_case("Floor") {
            // Project onto XY plane and take bounding-box width × depth.
            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            for p in &polygon {
                min_x = min_x.min(p.0);
                max_x = max_x.max(p.0);
                min_y = min_y.min(p.1);
                max_y = max_y.max(p.1);
            }
            width = (max_x - min_x).abs();
            depth = (max_y - min_y).abs();
            let prev = floor_z.unwrap_or(z_min);
            floor_z = Some(prev.min(z_min));
        } else if surface_type.eq_ignore_ascii_case("Roof") {
            let prev = ceiling_z.unwrap_or(z_max);
            ceiling_z = Some(prev.max(z_max));
        }
    }
    // Prefer the surface-derived height (ceiling_z − floor_z).
    let height_from_surfaces = match (floor_z, ceiling_z) {
        (Some(f), Some(c)) if c > f => Some(c - f),
        _ => None,
    };
    if let Some(z) = idf.zones().find(|z| {
        z.fields
            .first()
            .and_then(|v| v.to_display_string())
            .map(|s| s == zone_name)
            .unwrap_or(false)
    }) {
        // Zone field 6 = CeilingHeight (Multiplier is field 5).
        if let Some(h) = field_real(z, 6) {
            if h > 1e-3 {
                height = h;
            }
        }
    }
    if let Some(h) = height_from_surfaces {
        if h > 1e-3 {
            height = h;
        }
    }
    let mut spec = GeometrySpec::new(width, depth, height);
    spec.name = Some(zone_name.to_string());
    spec
}

/// Extract the heating / cooling setpoints (°C) from `Schedule:Compact` +
/// `ThermostatSetpoint:DualSetpoint` linked by `ZoneControl:Thermostat`.
fn extract_setpoints(idf: &IdfFile, zone_name: &str) -> Option<(f64, f64)> {
    // 1. Find the ZoneControl:Thermostat that references this zone.
    let mut dualsp: Option<String> = None;
    for obj in &idf.objects {
        if !obj
            .object_type
            .eq_ignore_ascii_case("ZoneControl:Thermostat")
        {
            continue;
        }
        if obj
            .fields
            .get(1)
            .and_then(|v| v.to_display_string())
            .map(|s| s == zone_name)
            .unwrap_or(false)
        {
            // field 4 is the DualSetpoint schedule name.
            if let Some(s) = obj.fields.get(4).and_then(|v| v.to_display_string()) {
                dualsp = Some(s);
            }
        }
    }
    let dualsp = dualsp?;
    // 2. Find the ThermostatSetpoint:DualSetpoint that references `dualsp`.
    let mut heat_sp: Option<String> = None;
    let mut cool_sp: Option<String> = None;
    for obj in &idf.objects {
        if !obj
            .object_type
            .eq_ignore_ascii_case("ThermostatSetpoint:DualSetpoint")
        {
            continue;
        }
        if obj
            .fields
            .first()
            .and_then(|v| v.to_display_string())
            .map(|s| s == dualsp)
            .unwrap_or(false)
        {
            heat_sp = obj.fields.get(1).and_then(|v| v.to_display_string());
            cool_sp = obj.fields.get(2).and_then(|v| v.to_display_string());
        }
    }
    let heat_name = heat_sp?;
    let cool_name = cool_sp?;
    // 3. Find the heat / cool Schedule:Compact values (the last numeric
    //    field of each object, e.g. "Until: 24:00, 20.0").
    let mut heat_val: Option<f64> = None;
    let mut cool_val: Option<f64> = None;
    for obj in &idf.objects {
        if !obj.object_type.eq_ignore_ascii_case("Schedule:Compact") {
            continue;
        }
        let name = obj
            .fields
            .first()
            .and_then(|v| v.to_display_string())
            .unwrap_or_default();
        if name == heat_name {
            heat_val = obj.fields.iter().rev().find_map(|v| match v {
                IdfValue::Real(f) => Some(*f),
                IdfValue::Integer(i) => Some(*i as f64),
                _ => None,
            });
        } else if name == cool_name {
            cool_val = obj.fields.iter().rev().find_map(|v| match v {
                IdfValue::Real(f) => Some(*f),
                IdfValue::Integer(i) => Some(*i as f64),
                _ => None,
            });
        }
    }
    Some((heat_val?, cool_val?))
}

/// Extract infiltration ACH from `ZoneInfiltration:DesignFlowRate`.
///
/// Supports the three most common methods:
/// - `Flow/ZoneFloorArea` (m³/s·m²): ACH = `flow * floor_area / volume * 3600`
/// - `Flow/ExteriorArea`    (m³/s·m²): ACH = `flow * ext_area / volume * 3600`
/// - `AirChanges/Hour`     (1/h):      ACH = `value` (already in ACH)
fn extract_infiltration_ach(idf: &IdfFile, zone_name: &str, volume: f64) -> f64 {
    for obj in &idf.objects {
        if !obj
            .object_type
            .eq_ignore_ascii_case("ZoneInfiltration:DesignFlowRate")
        {
            continue;
        }
        if !obj
            .fields
            .get(1)
            .and_then(|v| v.to_display_string())
            .map(|s| s == zone_name)
            .unwrap_or(false)
        {
            continue;
        }
        // EnergyPlus fields (1-indexed):
        // 0: name, 1: zone_name, 2: schedule_name,
        // 3: design_flow_rate_calculation_method,
        // 4: flow_rate_per_zone_floor_area (m³/s·m²),
        // 5: flow_rate_per_exterior_surface_area (m³/s·m²),
        // 6: air_changes_per_hour,
        // 7-9: coefficients (unused here).
        if let Some(ach) = field_real(obj, 6) {
            if ach > 0.0 {
                return ach;
            }
        }
        let flow_per_floor_area = field_real(obj, 4).unwrap_or(0.0);
        let flow_per_ext_area = field_real(obj, 5).unwrap_or(0.0);
        let geom = geometry_spec_from_idf(idf, zone_name);
        let floor_area = geom.floor_area();
        let wall_area = geom.wall_area();
        if flow_per_floor_area > 0.0 && volume > 0.0 {
            return flow_per_floor_area * floor_area / volume * 3600.0;
        } else if flow_per_ext_area > 0.0 && volume > 0.0 {
            return flow_per_ext_area * wall_area / volume * 3600.0;
        }
    }
    0.5
}

/// Extract the largest fenestration surface in `zone_name` and return its
/// area + orientation. Returns `None` if no fenestration surfaces are
/// present (e.g. simple Mass-class cases).
fn extract_window(idf: &IdfFile, zone_name: &str) -> Option<(f64, Orientation, f64, f64)> {
    let mut best: Option<(f64, Orientation, f64, f64)> = None;
    // Build a name → orientation index from BuildingSurface:Detailed.
    let mut parent_orientation: std::collections::HashMap<String, Orientation> =
        std::collections::HashMap::new();
    for surf in idf.building_surfaces() {
        let name = surf.fields.first()?.to_display_string()?;
        let polygon = parse_surface_polygon(surf).ok()?;
        let orientation = classify_surface_orientation(&polygon);
        parent_orientation.insert(name, orientation);
    }
    for obj in &idf.objects {
        if !obj
            .object_type
            .eq_ignore_ascii_case("FenestrationSurface:Detailed")
        {
            continue;
        }
        // 0: name, 1: surface_type, 2: construction_name, 3: building_surface_name, ...
        let building_surface = obj.fields.get(3).and_then(|v| v.to_display_string())?;
        let parent = idf.building_surfaces().find(|s| {
            s.fields
                .first()
                .and_then(|v| v.to_display_string())
                .map(|n| n == building_surface)
                .unwrap_or(false)
        });
        let parent_obj = match parent {
            Some(p) => p,
            None => continue,
        };
        // Parent zone name is field 3 of BuildingSurface:Detailed.
        let parent_zone = parent_obj
            .fields
            .get(3)
            .and_then(|v| v.to_display_string())
            .unwrap_or_default();
        if parent_zone != zone_name {
            continue;
        }
        let polygon = parse_surface_polygon(obj).ok()?;
        let area = polygon_area(&polygon);
        let orientation = parent_orientation
            .get(&building_surface)
            .copied()
            .unwrap_or(Orientation::South);
        let height = polygon
            .iter()
            .map(|p| p.2)
            .fold(f64::NEG_INFINITY, f64::max)
            - polygon.iter().map(|p| p.2).fold(f64::INFINITY, f64::min);
        let width = (area / height).max(0.0);
        match best {
            Some((a, _, _, _)) if a >= area => {}
            _ => best = Some((area, orientation, width.max(0.0), height.max(0.0))),
        }
    }
    best
}

/// Classify a 3D polygon as one of the standard [`Orientation`] variants
/// by inspecting which axis varies least across its vertices.
fn classify_surface_orientation(pts: &[(f64, f64, f64)]) -> Orientation {
    if pts.is_empty() {
        return Orientation::South;
    }
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_z, mut max_z) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in pts {
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
    // Floor (down) if z is most constant AND z values are below mid-height.
    if dz <= dy && dz <= dx {
        let mid_z = (min_z + max_z) / 2.0;
        if mid_z < 1.0 {
            Orientation::Down
        } else {
            Orientation::Up
        }
    } else if dy <= dx {
        // Y is most constant → east/west wall.
        let mid_y = (min_y + max_y) / 2.0;
        if mid_y < 1.0 {
            Orientation::South
        } else {
            Orientation::North
        }
    } else {
        // X is most constant → south/north wall.
        let mid_x = (min_x + max_x) / 2.0;
        if mid_x < 1.0 {
            Orientation::West
        } else {
            Orientation::East
        }
    }
}

/// Build a [`crate::validation::ashrae_140_cases::CaseSpec`] from the IDF
/// for a single zone. This bridges to the existing ASHRAE 140 test
/// harness (`tests/ashrae_140_case_600_series.rs`) by reading the
/// out-of-scope IDF objects (setpoints, infiltration, window) directly
/// from the [`IdfFile`].
///
/// The case_id is `"IDF:<idf_path>"` so a test harness can correlate the
/// spec back to its source file.
pub fn case_spec_from_idf(idf: &IdfFile, case_id: &str) -> Result<CaseSpec, IdfError> {
    use crate::ashrae_cases::{
        BuildingType, HvacSchedule, InternalLoads, ShadingDevice, WindowArea,
        WindowSpec as AshraeWindowSpec,
    };
    use crate::validation::ashrae_140_cases::{CommonWall, ConstructionSpec};

    let first_zone = idf
        .zones()
        .next()
        .ok_or_else(|| IdfError::conversion_error("No Zone objects found in IDF"))?;
    let zone_name = first_zone
        .fields
        .first()
        .and_then(|v| v.to_display_string())
        .unwrap_or_else(|| "ZONE1".to_string());

    let geom = geometry_spec_from_idf(idf, &zone_name);
    let volume = geom.volume();
    let _floor_area = geom.floor_area();

    // Build construction assemblies from the IDF.
    let _ = build_constructions(idf)?;

    // Detect construction type: heavy if any material has density > 1500.
    let mut construction_type = ConstructionType::LowMass;
    for m in idf.materials() {
        if let Some(d) = field_real(m, 4) {
            if d > 1500.0 {
                construction_type = ConstructionType::HighMass;
            }
        }
    }

    // Build the schema via a helper that doesn't consume `idf`.
    let schema_view = SimulationSchemaV1::try_from(clone_idf(idf))?;
    let wall = crate::sim::construction::Construction::new(schema_view.constructions.wall.layers);
    let roof = crate::sim::construction::Construction::new(schema_view.constructions.roof.layers);
    let floor = crate::sim::construction::Construction::new(schema_view.constructions.floor.layers);

    // Window: ASHRAE 140 default properties (U=2.10, SHGC=0.77).
    let window_properties = AshraeWindowSpec::double_clear_glass();
    let windows = if let Some((area, orientation, width, height)) = extract_window(idf, &zone_name)
    {
        let mut w = WindowArea::new(area, orientation);
        w.width = width;
        w.height = height;
        vec![vec![w]]
    } else {
        vec![vec![]]
    };

    let (heat_sp, cool_sp) = extract_setpoints(idf, &zone_name).unwrap_or((20.0, 27.0));
    let hvac = vec![HvacSchedule::constant(heat_sp, cool_sp)];

    let internal_loads = vec![Some(InternalLoads::new(0.0, 0.6, 0.4))];

    let infiltration_ach = extract_infiltration_ach(idf, &zone_name, volume);

    let ground_temp = build_ground_temperature(idf).ok();
    let ground_temperature_c = ground_temp.map(|g| g.monthly[0]);

    Ok(CaseSpec {
        case_id: case_id.to_string(),
        description: format!("Imported from IDF {case_id}"),
        geometry: vec![geom],
        construction_type,
        construction: ConstructionSpec::new(wall, roof, floor),
        windows,
        window_properties,
        shading: None::<ShadingDevice>,
        internal_loads,
        hvac,
        night_ventilation: None,
        common_walls: Vec::<CommonWall>::new(),
        infiltration_ach,
        opaque_absorptance: 0.7,
        num_zones: 1,
        weather_data: None,
        door_height: None,
        door_area: None,
        epw_path: None,
        hvac_equipment: None,
        ground_temperature_c,
        building_type: BuildingType::Residential,
    })
}

/// Clone an [`IdfFile`] without taking ownership (used internally by
/// [`case_spec_from_idf`] to share an `IdfFile` reference between the
/// `TryFrom` conversion and other readers).
fn clone_idf(idf: &IdfFile) -> IdfFile {
    IdfFile {
        version: idf.version.clone(),
        objects: idf.objects.clone(),
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::idf::parser::IdfParser;

    #[test]
    fn parses_version_25_2() {
        // Include a minimal Building + Zone + Material + Construction +
        // Site:GroundTemperature so the converter has all the required
        // MVP fields to populate the schema.
        let src = "\
Version, 25.2;\n\
Building, TestBldg, 0.0, City, 0.04, 0.4, FullExterior, 25;\n\
Zone, Z1, 0, 0, 0, 0, 1, 2.7, , , , ;\n\
Material, Mat1, MediumRough, 0.1, 1.0, 2000, 800;\n\
Construction, Wall1, Mat1;\n\
Site:GroundTemperature:BuildingSurface, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10;\n";
        let idf = IdfParser::from_str(src).unwrap();
        let schema = SimulationSchemaV1::try_from(idf).unwrap();
        assert_eq!(schema.version, SchemaVersion::V1);
        assert_eq!(schema.metadata.name, "TestBldg");
    }

    #[test]
    fn rejects_unsupported_version() {
        let src = "Version, 99.9;\n";
        let idf = IdfParser::from_str(src).unwrap();
        let err = SimulationSchemaV1::try_from(idf).unwrap_err();
        match err {
            IdfError::UnsupportedVersion(v) => assert_eq!(v, "99.9"),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn accepts_all_supported_versions() {
        for v in SUPPORTED_VERSIONS {
            let src = format!(
                "\
Version, {v};\n\
Building, TestBldg, 0.0, City, 0.04, 0.4, FullExterior, 25;\n\
Zone, Z1, 0, 0, 0, 0, 1, 2.7, , , , ;\n\
Material, Mat1, MediumRough, 0.1, 1.0, 2000, 800;\n\
Construction, Wall1, Mat1;\n\
Site:GroundTemperature:BuildingSurface, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10;\n"
            );
            let idf = IdfParser::from_str(&src).unwrap();
            SimulationSchemaV1::try_from(idf).unwrap();
        }
    }

    #[test]
    fn accepts_version_with_patch_suffix() {
        let src = "Version, 25.2.0;\n";
        let idf = IdfParser::from_str(src).unwrap();
        // Should not error on UnsupportedVersion.
        let err = SimulationSchemaV1::try_from(idf);
        assert!(
            !matches!(err, Err(IdfError::UnsupportedVersion(_))),
            "25.2.0 should normalize to 25-2 and be accepted, got {err:?}"
        );
    }

    #[test]
    fn rejects_missing_version() {
        let src = "Timestep, 1;\n";
        let idf = IdfParser::from_str(src).unwrap();
        assert!(SimulationSchemaV1::try_from(idf).is_err());
    }

    #[test]
    fn shoelace_square_area() {
        let pts = vec![(0.0, 0.0), (6.0, 0.0), (6.0, 8.0), (0.0, 8.0)];
        assert!((shoelace_area(&pts) - 48.0).abs() < 1e-9);
    }

    #[test]
    fn shoelace_clockwise_returns_same_area() {
        // Reverse winding — area must still be positive.
        let cw = vec![(0.0, 0.0), (6.0, 0.0), (6.0, 8.0), (0.0, 8.0)];
        let ccw: Vec<_> = cw.iter().rev().copied().collect();
        assert!((shoelace_area(&cw) - shoelace_area(&ccw)).abs() < 1e-9);
    }

    #[test]
    fn polygon_area_handles_3d_input() {
        // A 6x8 m floor at z=0, polygon 0,0,0 → 6,0,0 → 6,8,0 → 0,8,0.
        let pts = vec![
            (0.0, 0.0, 0.0),
            (6.0, 0.0, 0.0),
            (6.0, 8.0, 0.0),
            (0.0, 8.0, 0.0),
        ];
        assert!((polygon_area(&pts) - 48.0).abs() < 1e-9);
    }

    #[test]
    fn polygon_area_handles_vertical_walls() {
        // A 6x2.7 m south wall at y=0, polygon 0,0,2.7 → 6,0,2.7 → 6,0,0 → 0,0,0.
        let pts = vec![
            (0.0, 0.0, 2.7),
            (6.0, 0.0, 2.7),
            (6.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ];
        assert!((polygon_area(&pts) - 16.2).abs() < 1e-9);
    }
}
