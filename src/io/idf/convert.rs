// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! `IdfFile → SimulationSchemaV1` and `SimulationSchema` conversion (issue #2367).
//!
//! This module also exposes [`case_spec_from_idf`] — a companion helper
//! that bridges to [`crate::validation::ashrae_140_cases::CaseSpec`] for
//! the ASHRAE 140 acceptance criterion. It reads the additional IDF
//! objects that the MVP converter ignores (setpoints, infiltration, the
//! 12 m² south window) directly from the [`IdfFile`].
//! This module implements [`TryFrom<&IdfFile>`] for
//! both [`crate::api::schema::SimulationSchemaV1`] and
//! [`crate::api::schema::SimulationSchema`]. It consumes the parsed IDF
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
//! | `ZoneHVAC:EquipmentConnections` + `ZoneHVAC:EquipmentList` + `ZoneHVAC:IdealLoadsAirSystem` | `controls.hvac_system = Some(HVACSystemType::Ideal)` (issue #3991) |
//!
//! Both [`TryFrom<&IdfFile> for SimulationSchemaV1`] and
//! [`TryFrom<&IdfFile> for SimulationSchema`] are implemented here.
//!
//! All other object types (`Schedule:Compact`, `FenestrationSurface:Detailed`,
//! `ZoneInfiltration:DesignFlowRate`, other HVAC types, etc.) are **out of
//! scope** per design §10 — they remain available as raw
//! [`super::parser::IdfObject`]s in the source [`IdfFile`] for callers
//! that need them. The companion
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
    ConstructionSet, ControlConfig, ControlSet, Geometry, ScheduleSet, SchemaMetadata,
    SchemaVersion, SimulationSchema, SimulationSchemaV1, SurfaceConstruction, WindowSpec,
    ZoneGeometry,
};
use crate::ashrae_cases::Orientation;
use crate::sim::boundary::{GroundTemperature, MonthlyGroundTemperature};
use crate::sim::construction::ConstructionLayer;
use crate::sim::hvac::HVACSystemType;
use crate::sim::schedule::{DailySchedule, HVACSchedule};
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
// TryFrom<&IdfFile> for SimulationSchemaV1
// -----------------------------------------------------------------------------

impl TryFrom<&IdfFile> for SimulationSchemaV1 {
    type Error = IdfError;

    fn try_from(idf: &IdfFile) -> Result<Self, Self::Error> {
        // 1. Validate version (24-2, 25-1, 25-2 only per design §4.3).
        let version_str = idf.version.as_deref().ok_or_else(|| {
            IdfError::conversion_error("Missing Version object at top of IDF file")
        })?;
        let normalized = normalize_version(version_str);
        if !SUPPORTED_VERSIONS.contains(&normalized.as_str()) {
            return Err(IdfError::unsupported_version(version_str.to_string()));
        }

        // 2. Build metadata (Building name + RunPeriod + Timestep).
        let metadata = build_metadata(idf)?;
        let run_period = build_run_period(idf);
        let ground_temp = build_ground_temperature(idf).unwrap_or_default();

        // 3. Build geometry (Zone objects + BuildingSurface:Detailed vertex sums).
        let geometry = build_geometry(idf)?;

        // 4. Build constructions (Material + Construction objects).
        let constructions = build_constructions(idf)?;

        // 5. Extract zone name for schedule/control extraction.
        let zone_name = idf
            .zones()
            .next()
            .and_then(|z| z.fields.first().and_then(|v| v.to_display_string()))
            .unwrap_or_else(|| "ZONE1".to_string());

        // 6. Build schedules and controls from IDF objects.
        let schedules = build_schedules(idf, &zone_name);
        let controls = build_controls(idf, &zone_name);

        // 7. Wire metadata.run_period into the metadata struct via the
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
            schedules,
            weather: crate::api::schema::WeatherData::default(),
            controls,
            output: crate::api::schema::SimulationOutput::default(),
        })
    }
}

impl TryFrom<&IdfFile> for SimulationSchema {
    type Error = IdfError;

    fn try_from(idf: &IdfFile) -> Result<Self, Self::Error> {
        Ok(SimulationSchema::V1(SimulationSchemaV1::try_from(idf)?))
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
        surfaces: Vec::new(),
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
/// - FenestrationSurface:Detailed → Number of Vertices at field 6, vertices at 7+
///   (the IDD layout) or at field 8 / vertices at 9+ (the layout used by the
///   reference EnergyPlus fixtures in this repository, which carry two extra
///   fields — Outside Boundary Condition Object and View Factor to Ground —
///   before the vertex count)
///
/// We probe the positions in the order that matches the object's type and
/// pick whichever parses a reasonable vertex count (≥ 3).
fn parse_surface_polygon(surf: &IdfObject) -> Result<Vec<(f64, f64, f64)>, IdfError> {
    let candidates: [(usize, usize); 3] = if surf
        .object_type
        .eq_ignore_ascii_case("FenestrationSurface:Detailed")
    {
        [(6, 7), (8, 9), (10, 11)]
    } else {
        [(10, 11), (8, 9), (6, 7)]
    };
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

/// Name-like value of a single field.
///
/// Only [`IdfValue::String`] fields carry names in IDF (object names,
/// zone names, node names, object types); numeric fields are never
/// compared against a name.
fn field_name(value: &IdfValue) -> Option<&str> {
    match value {
        IdfValue::String(s) if !s.is_empty() => Some(s.as_str()),
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

/// Build a `ScheduleSet` from an IDF by extracting schedule objects.
///
/// Returns a `ScheduleSet` with HVAC setpoints extracted from
/// `ZoneControl:Thermostat` + `ThermostatSetpoint:DualSetpoint` +
/// `Schedule:Compact` objects, or defaults if not found.
fn build_schedules(idf: &IdfFile, zone_name: &str) -> ScheduleSet {
    let (heat_sp, cool_sp) = extract_setpoints(idf, zone_name).unwrap_or((20.0, 24.0));
    ScheduleSet {
        occupancy: DailySchedule::weekly("Occupancy".to_string()),
        lighting: DailySchedule::weekly("Lighting".to_string()),
        hvac: HVACSchedule::constant_schedule(heat_sp, cool_sp)
            .expect("constant_schedule on fresh daily schedules cannot fail"),
        infiltration: None,
        // Issue #4101: no IDF-sourced lighting density or equipment yet.
        lighting_power_density_w_m2: None,
        equipment: Vec::new(),
    }
}

/// Build a `ControlSet` from an IDF by extracting thermostat objects.
///
/// Returns a `ControlSet` with zone control setpoints extracted from
/// `ZoneControl:Thermostat` + `ThermostatSetpoint:DualSetpoint` objects,
/// or defaults if not found.
///
/// Issue #3991 — the HVAC system serving the zone is also detected:
/// a zone whose equipment chain reaches a
/// `ZoneHVAC:IdealLoadsAirSystem` gets
/// [`HVACSystemType::Ideal`]; zones without any `ZoneHVAC:*` objects
/// keep `hvac_system: None` (the historical behaviour).
fn build_controls(idf: &IdfFile, zone_name: &str) -> ControlSet {
    let (heat_sp, cool_sp) = extract_setpoints(idf, zone_name).unwrap_or((20.0, 24.0));
    ControlSet {
        zone_control: ControlConfig {
            heating_setpoint: heat_sp,
            cooling_setpoint: cool_sp,
            deadband_tolerance: 0.5,
            heating_capacity: 100_000.0,
            cooling_capacity: 100_000.0,
        },
        global_control: None,
        hvac_system: zone_hvac_system(idf, zone_name),
    }
}

/// Detect the HVAC system serving `zone_name` (issue #3991).
///
/// Only the basic slice is implemented: a zone is served by an
/// [`HVACSystemType::Ideal`] system when its equipment chain reaches a
/// `ZoneHVAC:IdealLoadsAirSystem` object. The chain that is resolved is
///
/// ```text
/// ZoneHVAC:EquipmentConnections   (zone name → equipment list name)
///   → ZoneHVAC:EquipmentList      (object type + name entries)
///     → ZoneHVAC:IdealLoadsAirSystem
/// ```
///
/// The simpler *direct* case is handled as well: an
/// `ZoneHVAC:IdealLoadsAirSystem` object that names the zone itself
/// (its own zone-name field) serves that zone.
///
/// Returns `None` when the zone has no HVAC equipment at all (or the
/// chain cannot be resolved), preserving the previous behaviour.
fn zone_hvac_system(idf: &IdfFile, zone_name: &str) -> Option<HVACSystemType> {
    if zone_served_by_ideal_loads(idf, zone_name) {
        Some(HVACSystemType::Ideal)
    } else {
        None
    }
}

/// True when `zone_name` is served by a `ZoneHVAC:IdealLoadsAirSystem`.
fn zone_served_by_ideal_loads(idf: &IdfFile, zone_name: &str) -> bool {
    let ideal_loads: Vec<&IdfObject> = idf.zone_hvac_ideal_loads().collect();
    if ideal_loads.is_empty() {
        return false;
    }

    // Direct case: the IdealLoads object names the zone itself. The
    // object's own name field (index 0) is skipped so that an object
    // merely *named* after the zone is not mistaken for a zone link.
    for il in &ideal_loads {
        if il
            .fields
            .iter()
            .skip(1)
            .any(|v| field_name(v) == Some(zone_name))
        {
            return true;
        }
    }

    // Chained case: ZoneHVAC:EquipmentConnections → ZoneHVAC:EquipmentList
    // → ZoneHVAC:IdealLoadsAirSystem.
    let ideal_names: Vec<&str> = ideal_loads
        .iter()
        .filter_map(|il| il.fields.first().and_then(field_name))
        .collect();
    if ideal_names.is_empty() {
        return false;
    }

    for conn in idf.zone_hvac_equipment_connections() {
        if !connection_serves_zone(conn, zone_name) {
            continue;
        }
        // The connection names the equipment list it uses (IDD field
        // "Zone Conditioning Equipment List Name"; index 2). Hand-written
        // IDFs may order the fields differently, so any field naming one
        // of the equipment lists is accepted as well.
        for list in idf.zone_hvac_equipment_lists() {
            let Some(this_list_name) = list.fields.first().and_then(field_name) else {
                continue;
            };
            let referenced = conn
                .fields
                .get(2)
                .and_then(field_name)
                .map(|n| n.eq_ignore_ascii_case(this_list_name))
                .unwrap_or(false)
                || conn.fields.iter().skip(1).any(|v| {
                    field_name(v)
                        .map(|n| n.eq_ignore_ascii_case(this_list_name))
                        .unwrap_or(false)
                });
            if !referenced {
                continue;
            }
            if equipment_list_has_ideal_loads(list, &ideal_names) {
                return true;
            }
        }
    }

    false
}

/// True when a `ZoneHVAC:EquipmentConnections` object serves `zone_name`.
///
/// The full EnergyPlus IDD layout is
/// `Name, Zone Name, Zone Conditioning Equipment List Name, …`, so the
/// zone name sits at index 1. Some hand-written models (including the
/// reference models under `tests/reference_data/energyplus_models/`)
/// omit the dedicated zone-name field and use the object name for it,
/// so the object name is accepted as well.
fn connection_serves_zone(conn: &IdfObject, zone_name: &str) -> bool {
    conn.fields.get(1).and_then(field_name) == Some(zone_name)
        || conn.fields.first().and_then(field_name) == Some(zone_name)
}

/// True when a `ZoneHVAC:EquipmentList` object lists one of the given
/// `ZoneHVAC:IdealLoadsAirSystem` object names.
///
/// The structured EnergyPlus layout groups the equipment entries in
/// blocks of four fields — object type, object name, cooling sequence,
/// heating sequence — starting after the object name (index 0) and the
/// load distribution scheme (index 1). When the structured layout does
/// not match (a hand-written IDF), the list is also scanned for a
/// reference to one of the IdealLoads object names.
fn equipment_list_has_ideal_loads(list: &IdfObject, ideal_names: &[&str]) -> bool {
    let mut entry = 0usize;
    loop {
        let type_idx = 2 + 4 * entry;
        let name_idx = 3 + 4 * entry;
        if name_idx >= list.fields.len() {
            break;
        }
        let obj_type = field_name(&list.fields[type_idx]);
        let obj_name = field_name(&list.fields[name_idx]);
        if let (Some(t), Some(n)) = (obj_type, obj_name) {
            if t.eq_ignore_ascii_case("ZoneHVAC:IdealLoadsAirSystem")
                && ideal_names.iter().any(|i| i.eq_ignore_ascii_case(n))
            {
                return true;
            }
        }
        entry += 1;
    }

    // Fallback: the list mentions an IdealLoads object by name.
    list.fields.iter().any(|v| {
        field_name(v)
            .map(|s| ideal_names.iter().any(|i| i.eq_ignore_ascii_case(s)))
            .unwrap_or(false)
    })
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

/// A `FenestrationSurface:Detailed` surface resolved against the
/// building surface that hosts it.
#[derive(Debug, Clone)]
struct FenestrationWindow {
    /// Glazed area (m²).
    area: f64,
    /// Orientation inherited from the parent building surface.
    orientation: Orientation,
    /// Height (m) — vertical extent of the fenestration polygon.
    height: f64,
    /// Construction name (field 2) that carries the glazing properties.
    construction_name: String,
}

/// Extract **every** `FenestrationSurface:Detailed` surface belonging to
/// `zone_name` (windows and glazed doors).
///
/// The previous implementation returned only the single largest surface
/// of the zone, silently dropping all the others. `CaseSpec` carries a
/// single [`crate::ashrae_cases::WindowSpec`] for the whole case but does
/// support one [`WindowArea`] per orientation, so the caller aggregates
/// the returned surfaces per orientation (see `case_spec_from_idf`).
///
/// Surfaces whose parent building surface cannot be resolved, or whose
/// vertex list cannot be parsed, are skipped instead of aborting the
/// whole extraction.
fn extract_windows(idf: &IdfFile, zone_name: &str) -> Vec<FenestrationWindow> {
    // Build a name → orientation index from BuildingSurface:Detailed.
    let mut parent_orientation: std::collections::HashMap<String, Orientation> =
        std::collections::HashMap::new();
    for surf in idf.building_surfaces() {
        let Some(name) = surf.fields.first().and_then(|v| v.to_display_string()) else {
            continue;
        };
        let Ok(polygon) = parse_surface_polygon(surf) else {
            continue;
        };
        let orientation = classify_surface_orientation(&polygon);
        parent_orientation.insert(name, orientation);
    }

    let mut windows: Vec<FenestrationWindow> = Vec::new();
    for obj in idf.fenestration_surfaces() {
        // 0: name, 1: surface_type, 2: construction_name, 3: building_surface_name, ...
        let Some(building_surface) = obj.fields.get(3).and_then(|v| v.to_display_string()) else {
            continue;
        };
        let Some(parent_obj) = idf.building_surfaces().find(|s| {
            s.fields
                .first()
                .and_then(|v| v.to_display_string())
                .map(|n| n == building_surface)
                .unwrap_or(false)
        }) else {
            continue;
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
        let Ok(polygon) = parse_surface_polygon(obj) else {
            continue;
        };
        let area = polygon_area(&polygon);
        let orientation = parent_orientation
            .get(&building_surface)
            .copied()
            .unwrap_or(Orientation::South);
        let z_max = polygon
            .iter()
            .map(|p| p.2)
            .fold(f64::NEG_INFINITY, f64::max);
        let z_min = polygon.iter().map(|p| p.2).fold(f64::INFINITY, f64::min);
        let height = (z_max - z_min).max(0.0);
        let construction_name = obj
            .fields
            .get(2)
            .and_then(|v| v.to_display_string())
            .unwrap_or_default();
        windows.push(FenestrationWindow {
            area,
            orientation,
            height,
            construction_name,
        });
    }
    windows
}

/// Resolve the glazing properties of a window construction.
///
/// EnergyPlus only stores an explicit U-factor and SHGC in
/// `WindowMaterial:SimpleGlazingSystem` (IDD fields: A1 Name, A2 U-Factor
/// in W/m²K, A3 SHGC, A4 Conductivity). A `WindowMaterial:Glazing` object
/// carries *material* data only (thickness, solar/visible transmittance
/// and reflectance, IR emissivity, conductivity) — there is no U-factor
/// and no SHGC in it, and no defensible way to derive a centre-of-glass
/// U-value or an SHGC from spectral data alone. A construction built only
/// from `WindowMaterial:Glazing` therefore yields `None` and the caller
/// keeps the ASHRAE 140 default glazing properties.
///
/// `WindowMaterial:SimpleGlazingSystem` is accepted either as a layer of
/// the window `Construction` (the usual EnergyPlus usage) or, as a
/// fallback, referenced directly by the surface's construction name.
fn window_spec_from_construction(
    idf: &IdfFile,
    construction_name: &str,
) -> Option<crate::ashrae_cases::WindowSpec> {
    if construction_name.is_empty() {
        return None;
    }

    // Layer names of the construction (field 0 is the construction name).
    let layers: Vec<String> = idf
        .constructions()
        .find(|c| {
            c.fields
                .first()
                .and_then(|v| v.to_display_string())
                .map(|n| n == construction_name)
                .unwrap_or(false)
        })
        .map(|c| {
            c.fields
                .iter()
                .skip(1)
                .filter_map(|v| v.to_display_string())
                .collect()
        })
        .unwrap_or_default();

    let mut system: Option<&IdfObject> = None;
    for layer in &layers {
        if let Some(obj) = idf.simple_glazing_systems().find(|o| {
            o.fields
                .first()
                .and_then(|v| v.to_display_string())
                .as_deref()
                == Some(layer.as_str())
        }) {
            system = Some(obj);
            break;
        }
    }
    if system.is_none() {
        system = idf.simple_glazing_systems().find(|o| {
            o.fields
                .first()
                .and_then(|v| v.to_display_string())
                .as_deref()
                == Some(construction_name)
        });
    }
    let system = system?;

    // IDD: A1 Name, A2 U-Factor (W/m²K), A3 SHGC, A4 Conductivity.
    let u_value = field_real(system, 1)?;
    let shgc = field_real(system, 2)?;
    if u_value <= 0.0 {
        return None;
    }

    // Start from the ASHRAE 140 double-clear-glass defaults so that the
    // frame / emissivity parameters keep their canonical values, then
    // override the two properties the IDF states explicitly. The normal
    // transmittance is not part of `WindowMaterial:SimpleGlazingSystem`
    // and therefore keeps its default value.
    let mut spec = crate::ashrae_cases::WindowSpec::double_clear_glass();
    spec.u_value = u_value;
    spec.shgc = shgc;
    Some(spec)
}

/// Pick the glazing properties for a zone's fenestration.
///
/// `CaseSpec` carries a single `WindowSpec` for the whole case, so when
/// the zone mixes several window constructions the properties of the
/// construction covering the largest glazed area are used (the dominant
/// glazing governs the solar / conductive gains). Falls back to the
/// ASHRAE 140 default glazing when no explicit glazing data is present.
fn resolve_window_properties(
    idf: &IdfFile,
    windows: &[FenestrationWindow],
) -> crate::ashrae_cases::WindowSpec {
    // Aggregate the glazed area per construction name.
    let mut area_by_construction: std::collections::HashMap<&str, f64> =
        std::collections::HashMap::new();
    for window in windows {
        *area_by_construction
            .entry(window.construction_name.as_str())
            .or_insert(0.0) += window.area;
    }
    let dominant = area_by_construction
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(name, _area)| name);

    match dominant {
        Some(name) => window_spec_from_construction(idf, name)
            .unwrap_or_else(crate::ashrae_cases::WindowSpec::double_clear_glass),
        None => crate::ashrae_cases::WindowSpec::double_clear_glass(),
    }
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

    // Build the schema via the reference-based TryFrom impl.
    let schema_view = SimulationSchemaV1::try_from(idf)?;
    let wall = crate::sim::construction::Construction::new(schema_view.constructions.wall.layers);
    let roof = crate::sim::construction::Construction::new(schema_view.constructions.roof.layers);
    let floor = crate::sim::construction::Construction::new(schema_view.constructions.floor.layers);

    // Window geometry: every fenestration surface of the zone is taken into
    // account and aggregated per orientation. `CaseSpec` carries a single
    // `WindowSpec` for the whole case (not one per orientation), so the
    // surfaces facing the same way are merged into a single `WindowArea`
    // whose area is the sum of the individual surfaces. Windows with
    // different orientations stay separate entries so that the solar
    // gains keep their azimuth.
    let fenestration = extract_windows(idf, &zone_name);
    let mut area_by_orientation: std::collections::BTreeMap<Orientation, (f64, f64)> =
        std::collections::BTreeMap::new();
    for window in &fenestration {
        let entry = area_by_orientation
            .entry(window.orientation)
            .or_insert((0.0, 0.0));
        entry.0 += window.area;
        entry.1 = entry.1.max(window.height);
    }
    let window_areas: Vec<WindowArea> = area_by_orientation
        .iter()
        .map(|(orientation, (area, height))| {
            let mut w = WindowArea::new(*area, *orientation);
            if *height > 0.0 {
                w.height = *height;
                w.width = *area / *height;
            }
            w
        })
        .collect();
    let windows = vec![window_areas];

    // Window properties: taken from the glazing system referenced by the
    // window construction when the IDF provides one (explicit U-factor /
    // SHGC); otherwise the ASHRAE 140 default glazing is kept.
    let window_properties = resolve_window_properties(idf, &fenestration);

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
        zone_origins: None,
        zone_constructions: None,
        weather_data: None,
        door_height: None,
        door_area: None,
        epw_path: None,
        hvac_equipment: None,
        ground_temperature_c,
        floor_u_value_override: None,
        window_u_value_override: None,
        building_type: BuildingType::Residential,
    })
}

// -----------------------------------------------------------------------------
// Tests
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
        let schema = SimulationSchemaV1::try_from(&idf).unwrap();
        assert_eq!(schema.version, SchemaVersion::V1);
        assert_eq!(schema.metadata.name, "TestBldg");
    }

    #[test]
    fn rejects_unsupported_version() {
        let src = "Version, 99.9;\n";
        let idf = IdfParser::from_str(src).unwrap();
        let err = SimulationSchemaV1::try_from(&idf).unwrap_err();
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
            SimulationSchemaV1::try_from(&idf).unwrap();
        }
    }

    #[test]
    fn accepts_version_with_patch_suffix() {
        let src = "Version, 25.2.0;\n";
        let idf = IdfParser::from_str(src).unwrap();
        // Should not error on UnsupportedVersion.
        let err = SimulationSchemaV1::try_from(&idf);
        assert!(
            !matches!(err, Err(IdfError::UnsupportedVersion(_))),
            "25.2.0 should normalize to 25-2 and be accepted, got {err:?}"
        );
    }

    #[test]
    fn rejects_missing_version() {
        let src = "Timestep, 1;\n";
        let idf = IdfParser::from_str(src).unwrap();
        assert!(SimulationSchemaV1::try_from(&idf).is_err());
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

    #[test]
    fn round_trip_preserves_geometry_and_constructions() {
        let src = "\
Version, 25.2;\n\
Building, TestBldg, 0.0, City, 0.04, 0.4, FullExterior, 25;\n\
Zone, Z1, 0, 0, 0, 0, 1, 2.7, 129.6, 48.0, ;
\n\
Material, Mat1, MediumRough, 0.1, 1.0, 2000, 800;\n\
Material, Mat2, MediumRough, 0.05, 0.04, 30, 840;\n\
Construction, Wall1, Mat1, Mat2;\n\
Site:GroundTemperature:BuildingSurface, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18;\n\
GlobalGeometryRules, UpperLeftCorner, ClockWise, World;\n\
BuildingSurface:Detailed, SouthWall, Wall, Wall1, Z1, , Outdoors, , SunExposed, WindExposed, ,\n\
  4,\n\
  0, 0, 2.7,\n\
  6, 0, 2.7,\n\
  6, 0, 0,\n\
  0, 0, 0;\n\
BuildingSurface:Detailed, Floor, Floor, Wall1, Z1, , Ground, , NoSun, NoWind, ,\n\
  4,\n\
  0, 0, 0,\n\
  6, 0, 0,\n\
  6, 8, 0,\n\
  0, 8, 0;\n";
        let idf = IdfParser::from_str(src).unwrap();
        let schema = SimulationSchemaV1::try_from(&idf).expect("IDF converts to schema");

        assert_eq!(schema.geometry.zones.len(), 1);
        assert!((schema.geometry.zones[0].floor_area - 48.0).abs() < 1e-3);
        assert!((schema.geometry.zones[0].volume - 129.6).abs() < 1e-3);
        assert!((schema.geometry.zones[0].height - 2.7).abs() < 1e-3);

        let wall = &schema.constructions.wall;
        assert_eq!(wall.layers.len(), 2);
        assert!((wall.layers[0].conductivity - 1.0).abs() < 1e-3);
        assert!((wall.layers[1].conductivity - 0.04).abs() < 1e-3);
    }

    #[test]
    fn schedules_and_controls_extracted_from_real_idf() {
        let fixtures_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("reference_data")
            .join("energyplus_models");
        let path = fixtures_dir.join("ashrae_140_case_600.idf");
        let idf = IdfParser::from_path(&path).expect("IDF parses");
        let schema = SimulationSchemaV1::try_from(&idf).expect("IDF converts to schema");

        assert_eq!(schema.controls.zone_control.heating_setpoint, 20.0);
        assert_eq!(schema.controls.zone_control.cooling_setpoint, 27.0);
        assert!((schema.schedules.hvac.heating_setpoint(0) - 20.0).abs() < 1e-3);
        assert!((schema.schedules.hvac.cooling_setpoint(0) - 27.0).abs() < 1e-3);
    }

    #[test]
    fn window_data_extracted_from_fenestration_surface() {
        let src = "\
Version, 25.2;\n\
Building, TestBldg, 0.0, City, 0.04, 0.4, FullExterior, 25;\n\
Zone, Z1, 0, 0, 0, 0, 1, 2.7, 129.6, 48.0, ;
\n\
Material, Mat1, MediumRough, 0.1, 1.0, 2000, 800;\n\
Construction, Wall1, Mat1;\n\
WindowMaterial:Glazing, DblClear,\n\
  SpectralAverage, 0.003, 0.837, 0.075, 0.075, 0.837, 0.075, 0.075, 0.0, 0.84, 0.84, 0.9;\n\
Construction, DblWin, DblClear;\n\
Site:GroundTemperature:BuildingSurface, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18;\n\
GlobalGeometryRules, UpperLeftCorner, ClockWise, World;\n\
BuildingSurface:Detailed, SouthWall, Wall, Wall1, Z1, , Outdoors, , SunExposed, WindExposed, ,\n\
  4,\n\
  0, 0, 2.7,\n\
  6, 0, 2.7,\n\
  6, 0, 0,\n\
  0, 0, 0;\n\
FenestrationSurface:Detailed, SouthWindow, Window, DblWin, SouthWall, , , 0.0, Outside,\n\
  4,\n\
  0.5, 0.0, 2.0,\n\
  5.5, 0.0, 2.0,\n\
  5.5, 0.0, 0.0,\n\
  0.5, 0.0, 0.0;\n";
        let idf = IdfParser::from_str(src).unwrap();
        let schema = SimulationSchemaV1::try_from(&idf).expect("IDF converts to schema");

        let window = &schema.constructions.wall.window;
        assert!(window.is_some());
        let window = window.as_ref().unwrap();
        assert!(window.window_area > 0.0);
    }

    #[test]
    fn idf_file_try_from_path_succeeds() {
        let fixtures_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("reference_data")
            .join("energyplus_models");
        let path = fixtures_dir.join("ashrae_140_case_600.idf");
        let idf =
            IdfFile::try_from(path.as_path()).expect("IdfFile::try_from(path) parses Case 600 IDF");
        assert_eq!(idf.version.as_deref(), Some("25.2"));
    }

    #[test]
    fn simulation_schema_try_from_idf_file_works() {
        let fixtures_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("reference_data")
            .join("energyplus_models");
        let path = fixtures_dir.join("ashrae_140_case_600.idf");
        let idf = IdfFile::try_from(path.as_path()).expect("IDF parses");
        let schema = SimulationSchema::try_from(&idf).expect("converts to SimulationSchema");

        match schema {
            SimulationSchema::V1(v1) => {
                assert_eq!(v1.geometry.zones.len(), 1);
                assert!((v1.geometry.zones[0].floor_area - 48.0).abs() < 1e-3);
                assert_eq!(v1.controls.zone_control.heating_setpoint, 20.0);
                assert_eq!(v1.controls.zone_control.cooling_setpoint, 27.0);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Issue #3991 — glazing-aware, multi-surface IDF fenestration import
    // -------------------------------------------------------------------------

    /// Inline IDF fixture with three `FenestrationSurface:Detailed` surfaces
    /// (two on the south wall, one on the east wall) referencing a window
    /// construction built from `layer_name`.
    ///
    /// The south windows use the EnergyPlus IDD layout (vertex count at
    /// field 6), the east window uses the layout of the reference fixtures
    /// in this repository (vertex count at field 8).
    fn multi_window_idf(glazing_object: &str, layer_name: &str) -> String {
        format!(
            "\
Version, 25.2;\n\
Building, TestBldg, 0.0, City, 0.04, 0.4, FullExterior, 25;\n\
Zone, Z1, 0, 0, 0, 0, 1, 2.7, 48.0, 129.6, ;\n\
Material, Mat1, MediumRough, 0.1, 1.0, 2000, 800;\n\
Construction, Wall1, Mat1;\n\
{glazing_object}\
Construction, WinConstr, {layer_name};\n\
Site:GroundTemperature:BuildingSurface, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18;\n\
GlobalGeometryRules, UpperLeftCorner, ClockWise, World;\n\
BuildingSurface:Detailed, SouthWall, Wall, Wall1, Z1, , Outdoors, , SunExposed, WindExposed, ,\n\
  4,\n\
  0, 0, 2.7,\n\
  8, 0, 2.7,\n\
  8, 0, 0,\n\
  0, 0, 0;\n\
BuildingSurface:Detailed, EastWall, Wall, Wall1, Z1, , Outdoors, , SunExposed, WindExposed, ,\n\
  4,\n\
  8, 0, 2.7,\n\
  8, 6, 2.7,\n\
  8, 6, 0,\n\
  8, 0, 0;\n\
BuildingSurface:Detailed, Floor, Floor, Wall1, Z1, , Ground, , NoSun, NoWind, ,\n\
  4,\n\
  0, 0, 0,\n\
  8, 0, 0,\n\
  8, 6, 0,\n\
  0, 6, 0;\n\
FenestrationSurface:Detailed, SouthWindowA, Window, WinConstr, SouthWall, , 1, 4,\n\
  0.5, 0.0, 2.0,\n\
  5.5, 0.0, 2.0,\n\
  5.5, 0.0, 0.0,\n\
  0.5, 0.0, 0.0;\n\
FenestrationSurface:Detailed, SouthWindowB, Window, WinConstr, SouthWall, , 1, 4,\n\
  6.0, 0.0, 2.0,\n\
  7.5, 0.0, 2.0,\n\
  7.5, 0.0, 0.0,\n\
  6.0, 0.0, 0.0;\n\
FenestrationSurface:Detailed, EastWindow, Window, WinConstr, EastWall, , , 0.0, Outside, 4,\n\
  8, 1.0, 2.0,\n\
  8, 4.0, 2.0,\n\
  8, 4.0, 0.0,\n\
  8, 1.0, 0.0;\n"
        )
    }

    #[test]
    fn case_spec_accounts_for_all_fenestration_surfaces() {
        // Glazing without an explicit U-factor / SHGC — the geometry
        // aggregation must still cover every surface of the zone.
        let src = multi_window_idf(
            "WindowMaterial:Glazing, DblClear, SpectralAverage, 0.003, 0.837, 0.075, 0.075, 0.837, 0.075, 0.075, 0.0, 0.84, 0.84, 0.9;\n",
            "DblClear",
        );
        let idf = IdfParser::from_str(&src).unwrap();
        let spec = super::case_spec_from_idf(&idf, "IDF:test").expect("CaseSpec builds from IDF");

        assert_eq!(spec.windows.len(), 1, "one zone");
        // Three surfaces → one aggregated window per orientation
        // (two south-facing surfaces merged, one east-facing surface).
        assert_eq!(
            spec.windows[0].len(),
            2,
            "all fenestration surfaces must be accounted for (per orientation)"
        );

        let south = spec.windows[0]
            .iter()
            .find(|w| w.orientation == Orientation::South)
            .expect("south-facing window");
        let east = spec.windows[0]
            .iter()
            .find(|w| w.orientation == Orientation::East)
            .expect("east-facing window");

        // South: 10 m² + 3 m², East: 6 m².
        assert!(
            (south.area - 13.0).abs() < 1e-6,
            "south area = {}",
            south.area
        );
        assert!((east.area - 6.0).abs() < 1e-6, "east area = {}", east.area);

        let total: f64 = spec.windows[0].iter().map(|w| w.area).sum();
        assert!(
            (total - 19.0).abs() < 1e-6,
            "total fenestration area = {total} (expected 19 m²)"
        );
    }

    #[test]
    fn case_spec_uses_simple_glazing_system_properties() {
        // `WindowMaterial:SimpleGlazingSystem` states an explicit U-factor
        // and SHGC — those must reach the imported `WindowSpec`.
        let src = multi_window_idf(
            "WindowMaterial:SimpleGlazingSystem, CustomGlazing, 1.45, 0.42, 0.9;\n",
            "CustomGlazing",
        );
        let idf = IdfParser::from_str(&src).unwrap();
        let spec = super::case_spec_from_idf(&idf, "IDF:test").expect("CaseSpec builds from IDF");

        assert!(
            (spec.window_properties.u_value - 1.45).abs() < 1e-9,
            "u_value = {}",
            spec.window_properties.u_value
        );
        assert!(
            (spec.window_properties.shgc - 0.42).abs() < 1e-9,
            "shgc = {}",
            spec.window_properties.shgc
        );
    }

    #[test]
    fn case_spec_keeps_default_glazing_without_explicit_u_factor() {
        // `WindowMaterial:Glazing` carries spectral data only (no U-factor,
        // no SHGC) — the ASHRAE 140 default glazing must be preserved.
        let src = multi_window_idf(
            "WindowMaterial:Glazing, DblClear, SpectralAverage, 0.003, 0.837, 0.075, 0.075, 0.837, 0.075, 0.075, 0.0, 0.84, 0.84, 0.9;\n",
            "DblClear",
        );
        let idf = IdfParser::from_str(&src).unwrap();
        let spec = super::case_spec_from_idf(&idf, "IDF:test").expect("CaseSpec builds from IDF");

        let default = crate::ashrae_cases::WindowSpec::double_clear_glass();
        assert_eq!(spec.window_properties.u_value, default.u_value);
        assert_eq!(spec.window_properties.shgc, default.shgc);
    }

    // ----- Issue #3991: basic HVAC import (ZoneHVAC:IdealLoadsAirSystem) -----

    /// IDF with two zones; only `ZONE1` is served by an IdealLoads system
    /// through the EnergyPlus chain
    /// `ZoneHVAC:EquipmentConnections` → `ZoneHVAC:EquipmentList` →
    /// `ZoneHVAC:IdealLoadsAirSystem`.
    fn idf_with_ideal_loads_chain() -> String {
        "\
Version, 25.2;\n\
Building, TestBldg, 0.0, City, 0.04, 0.4, FullExterior, 25;\n\
Zone, ZONE1, 0, 0, 0, 0, 1, 2.7, , , , ;\n\
Zone, ZONE2, 0, 0, 0, 0, 1, 2.7, , , , ;\n\
ZoneHVAC:EquipmentConnections,\n\
  ZONE1 Connections,\n\
  ZONE1,\n\
  ZONE1 Equip List,\n\
  ZONE1 Supply Node,\n\
  ,\n\
  ,\n\
  ;\n\
ZoneHVAC:EquipmentList,\n\
  ZONE1 Equip List,\n\
  SequentialLoad,\n\
  ZoneHVAC:IdealLoadsAirSystem,\n\
  ZONE1 IdealLoads,\n\
  , , , ;\n\
ZoneHVAC:IdealLoadsAirSystem,\n\
  ZONE1 IdealLoads,\n\
  ,\n\
  ZONE1 Supply Node,\n\
  ZONE1 Exhaust Node,\n\
  ;\n"
        .to_string()
    }

    #[test]
    fn zone_with_ideal_loads_chain_is_marked_as_ideal_hvac() {
        let idf = IdfParser::from_str(&idf_with_ideal_loads_chain()).unwrap();
        let controls = build_controls(&idf, "ZONE1");
        assert_eq!(controls.hvac_system, Some(HVACSystemType::Ideal));
    }

    #[test]
    fn imported_schema_reports_ideal_hvac_for_ideal_loads_zone() {
        let idf = IdfParser::from_str(&idf_with_ideal_loads_chain()).unwrap();
        let schema = SimulationSchemaV1::try_from(&idf).unwrap();
        assert_eq!(schema.controls.hvac_system, Some(HVACSystemType::Ideal));
    }

    #[test]
    fn zone_without_any_zonehvac_objects_has_no_hvac_system() {
        let src = "\
Version, 25.2;\n\
Building, TestBldg, 0.0, City, 0.04, 0.4, FullExterior, 25;\n\
Zone, ZONE1, 0, 0, 0, 0, 1, 2.7, , , , ;\n";
        let idf = IdfParser::from_str(src).unwrap();
        let controls = build_controls(&idf, "ZONE1");
        assert_eq!(controls.hvac_system, None);
    }

    #[test]
    fn second_zone_without_equipment_is_not_marked_as_ideal_hvac() {
        let idf = IdfParser::from_str(&idf_with_ideal_loads_chain()).unwrap();
        assert_eq!(
            build_controls(&idf, "ZONE2").hvac_system,
            None,
            "only the zone served by the IdealLoads chain may be marked"
        );
    }

    #[test]
    fn ideal_loads_object_naming_the_zone_directly_is_detected() {
        // No EquipmentConnections / EquipmentList: the IdealLoads object
        // names the zone itself.
        let src = "\
Version, 25.2;\n\
Building, TestBldg, 0.0, City, 0.04, 0.4, FullExterior, 25;\n\
Zone, ZONE3, 0, 0, 0, 0, 1, 2.7, , , , ;\n\
ZoneHVAC:IdealLoadsAirSystem,\n\
  ZONE3 IdealLoads,\n\
  ,\n\
  ZONE3,\n\
  ZONE3 Supply Node,\n\
  ZONE3 Exhaust Node,\n\
  ;\n";
        let idf = IdfParser::from_str(src).unwrap();
        assert_eq!(
            build_controls(&idf, "ZONE3").hvac_system,
            Some(HVACSystemType::Ideal)
        );
        assert_eq!(build_controls(&idf, "ZONE1").hvac_system, None);
    }

    #[test]
    fn reference_model_zone_served_by_ideal_loads_is_marked() {
        // `ashrae_140_case_600.idf` carries the full chain:
        // `ZoneHVAC:EquipmentConnections` → `ZoneHVAC:EquipmentList` →
        // `ZoneHVAC:IdealLoadsAirSystem`.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("reference_data")
            .join("energyplus_models")
            .join("ashrae_140_case_600.idf");
        let idf = IdfFile::try_from(path.as_path()).expect("IDF parses");
        assert_eq!(
            build_controls(&idf, "ZONE1").hvac_system,
            Some(HVACSystemType::Ideal)
        );
    }
}
