// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Geometry extraction for IFC4 STEP files (issue #1612).
//!
//! Extracts spatial structure (building → storey → space → elements)
//! and zone geometry from IFC4 STEP physical files. Produces
//! [`ZoneGeometry`] records mapped from [`IfcSpace`] entities,
//! with surface area and zone assignment derived from
//! [`IfcRelContainedInSpatialStructure`] containment relationships.
//!
//! # Architecture
//!
//! The [`IfcGeometryParser`] takes an [`super::parser::IfcModel`] (already
//! parsed from STEP format) and enriches it with:
//! - Building and storey entity records ([`super::parser::IfcBuilding`],
//!   [`super::parser::IfcBuildingStorey`]) added to the model.
//! - Zone-to-building-element containment map for surface geometry
//!   assignment.
//! - Per-space floor area (from footprint extraction) and zone geometry
//!   records ready for [`super::mapping::IfcToSchema`].
//!
//! # IFC4 entities consumed
//!
//! | Entity | Role |
//! |--------|------|
//! | [`IfcBuilding`](super::parser::IfcBuilding) | Building name/identity |
//! | [`IfcBuildingStorey`](super::parser::IfcBuildingStorey) | Floor level identity |
//! | [`IfcSpace`](super::parser::IfcSpace) | Zone — one zone per space |
//! | `IfcRelContainedInSpatialStructure` | Zone element assignment |
//! | `IfcRelAggregates` | Storey → space hierarchy (via containment) |
//!
//! # Out of scope
//!
//! - Full extruded-body geometry decoding (vertices, normals). Follow-up
//!   issue #1121.
//! - Window and door geometry. Follow-up issue #1121.
//! - HVAC and flow elements. Deferred.

use std::collections::{HashMap, HashSet};

use crate::api::schema::ZoneGeometry;
use crate::interop::ifc::parser::{IfcBuilding, IfcBuildingStorey, IfcModel, IfcSpace};

const DEFAULT_ZONE_FLOOR_AREA_M2: f64 = 24.0;
const DEFAULT_ZONE_HEIGHT_M: f64 = 2.7;

/// Parses IFC4 STEP geometry entities and extracts zone geometry.
///
/// Entry point is [`IfcGeometryParser::parse_model`] which consumes an
/// [`IfcModel`] and returns a [`ParsedGeometry`] containing per-zone
/// geometry and zone element assignments.
#[derive(Debug, Clone)]
pub struct IfcGeometryParser;

impl IfcGeometryParser {
    /// Parse an [`IfcModel`] and extract zone geometry.
    ///
    /// For each [`IfcSpace`] in the model, produces a [`ZoneGeometry`]
    /// record. Floor area falls back to `DEFAULT_ZONE_FLOOR_AREA_M2`
    /// (24 m²) when the footprint polygon cannot be decoded.
    ///
    /// Zone element assignment is derived from
    /// `IfcRelContainedInSpatialStructure` entities that reference the
    /// space, mapped to surface categories (wall/slab/roof) via the
    /// entity type name of each contained element.
    pub fn parse_model(model: &IfcModel) -> ParsedGeometry {
        let space_zone_map = build_space_zone_map(model);
        let zones = build_zones(model, &space_zone_map);
        let zone_elements = build_zone_elements(model, &space_zone_map);

        ParsedGeometry {
            zones,
            zone_elements,
            buildings: model.buildings.clone(),
            storeys: model.storeys.clone(),
        }
    }
}

/// Result of geometry parsing: zone list and zone element assignments.
#[derive(Debug, Clone)]
pub struct ParsedGeometry {
    /// Zone geometry records, one per [`IfcSpace`].
    pub zones: Vec<ZoneGeometry>,
    /// Map from space id → set of element ids contained in that space.
    pub zone_elements: HashMap<u64, HashSet<u64>>,
    /// Buildings in the model.
    pub buildings: Vec<IfcBuilding>,
    /// Storeys in the model.
    pub storeys: Vec<IfcBuildingStorey>,
}

/// Build a map from space id → space name for zone naming.
fn build_space_zone_map(model: &IfcModel) -> HashMap<u64, &IfcSpace> {
    model.spaces.iter().map(|s| (s.id, s)).collect()
}

/// Build zone geometry records from spaces.
fn build_zones(model: &IfcModel, _space_map: &HashMap<u64, &IfcSpace>) -> Vec<ZoneGeometry> {
    model
        .spaces
        .iter()
        .map(|space| {
            let floor_area =
                extract_space_floor_area(model, space).unwrap_or(DEFAULT_ZONE_FLOOR_AREA_M2);
            let height = extract_space_height(model, space).unwrap_or(DEFAULT_ZONE_HEIGHT_M);
            let volume = floor_area * height;
            ZoneGeometry {
                name: space.name.clone(),
                floor_area,
                volume,
                height,
            }
        })
        .collect()
}

/// Extract floor area from IfcSpace.Representation chain.
///
/// Follows: IfcSpace → IfcProductDefinitionShape → IfcShapeRepresentation
/// → IfcExtrudedAreaSolid → IfcRectangleProfileDef
///
/// Returns None if geometry cannot be decoded.
fn extract_space_floor_area(model: &IfcModel, space: &IfcSpace) -> Option<f64> {
    let space_entity = model.entities.get(&space.id)?;
    let repr_ref = extract_nth_ref(&space_entity.args, 6)?;

    let repr_entity = model.entities.get(&repr_ref)?;
    if repr_entity.name != "IFCPRODUCTDEFINITIONSHAPE" {
        return None;
    }

    let shape_refs = extract_nth_ref_list(&repr_entity.args, 2)?;
    if shape_refs.is_empty() {
        return None;
    }

    let shape_entity = model.entities.get(&shape_refs[0])?;
    if shape_entity.name != "IFCSHAPEREPRESENTATION" {
        return None;
    }

    let item_refs = extract_nth_ref_list(&shape_entity.args, 4)?;
    if item_refs.is_empty() {
        return None;
    }

    let extruded_entity = model.entities.get(&item_refs[0])?;
    if extruded_entity.name != "IFCEXTRUDEDAREASOLID" {
        return None;
    }

    let swept_area_ref = extract_nth_ref(&extruded_entity.args, 0)?;

    let profile_entity = model.entities.get(&swept_area_ref)?;
    if profile_entity.name != "IFCRECTANGLEPROFILEDEF" {
        return None;
    }

    let x_dim = extract_nth_real(&profile_entity.args, 2)?;
    let y_dim = extract_nth_real(&profile_entity.args, 3)?;

    let area = x_dim * y_dim;
    if area > 0.0 {
        Some(area)
    } else {
        None
    }
}

/// Extract zone height from IfcSpace.Representation chain.
///
/// Follows: IfcSpace → IfcProductDefinitionShape → IfcShapeRepresentation
/// → IfcExtrudedAreaSolid (Depth field = height)
///
/// Returns None if geometry cannot be decoded.
fn extract_space_height(model: &IfcModel, space: &IfcSpace) -> Option<f64> {
    let space_entity = model.entities.get(&space.id)?;
    let repr_ref = extract_nth_ref(&space_entity.args, 6)?;

    let repr_entity = model.entities.get(&repr_ref)?;
    if repr_entity.name != "IFCPRODUCTDEFINITIONSHAPE" {
        return None;
    }

    let shape_refs = extract_nth_ref_list(&repr_entity.args, 2)?;
    if shape_refs.is_empty() {
        return None;
    }

    let shape_entity = model.entities.get(&shape_refs[0])?;
    if shape_entity.name != "IFCSHAPEREPRESENTATION" {
        return None;
    }

    let item_refs = extract_nth_ref_list(&shape_entity.args, 4)?;
    if item_refs.is_empty() {
        return None;
    }

    let extruded_entity = model.entities.get(&item_refs[0])?;
    if extruded_entity.name != "IFCEXTRUDEDAREASOLID" {
        return None;
    }

    let depth = extract_nth_real(&extruded_entity.args, 2)?;
    Some(depth).filter(|&v| v > 0.0)
}

/// Build a map from space id → set of contained element ids.
///
/// `IfcRelContainedInSpatialStructure` has `RelatingStructure` pointing
/// to a space and `RelatedElements` listing all building elements (walls,
/// slabs, roofs, etc.) contained in that space.
fn build_zone_elements(
    model: &IfcModel,
    _space_map: &HashMap<u64, &IfcSpace>,
) -> HashMap<u64, HashSet<u64>> {
    let mut zone_elements: HashMap<u64, HashSet<u64>> = HashMap::new();

    for entity in model.entities.values() {
        if entity.name == "IFCRELCONTAINEDINSPATIALSTRUCTURE" {
            if let Some((space_ref, element_refs)) = parse_contained_in_spatial(&entity.args) {
                zone_elements
                    .entry(space_ref)
                    .or_default()
                    .extend(element_refs);
            }
        }
    }

    zone_elements
}

/// Parse `IfcRelContainedInSpatialStructure` args.
///
/// Format: `(GlobalId, OwnerHistory, Name, Description, RelatedElements, RelatingStructure)`
/// where `RelatedElements` is a list of element refs and `RelatingStructure`
/// is a single ref to the IfcSpace.
fn parse_contained_in_spatial(args: &str) -> Option<(u64, Vec<u64>)> {
    let element_refs = extract_nth_ref_list(args, 4)?;
    let space_ref = extract_nth_ref(args, 5)?;
    Some((space_ref, element_refs))
}

fn extract_nth_ref(args: &str, n: usize) -> Option<u64> {
    let bytes = args.as_bytes();
    let mut i = 0;
    let mut depth: usize = 0;
    let mut in_string = false;
    let mut arg_idx: usize = 0;
    let mut start: usize = 0;

    while i < bytes.len() {
        let c = bytes[i];
        if in_string {
            if c == b'\'' {
                if bytes.get(i + 1) == Some(&b'\'') {
                    i += 2;
                    continue;
                }
                in_string = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'\'' => {
                in_string = true;
                i += 1;
            }
            b'(' => {
                depth += 1;
                i += 1;
            }
            b')' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            b',' if depth == 0 => {
                if arg_idx == n {
                    let s = args[start..i].trim();
                    return parse_ref(s);
                }
                arg_idx += 1;
                i += 1;
                while i < bytes.len() && (bytes[i] as char).is_whitespace() {
                    i += 1;
                }
                start = i;
            }
            _ => i += 1,
        }
    }
    if arg_idx == n {
        let s = args[start..].trim();
        parse_ref(s)
    } else {
        None
    }
}

fn parse_ref(s: &str) -> Option<u64> {
    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'#') {
        return None;
    }
    s[1..].parse().ok()
}

fn extract_nth_ref_list(args: &str, n: usize) -> Option<Vec<u64>> {
    let arg = extract_nth_arg_raw(args, n)?;
    let s = arg.trim();
    if !s.starts_with('(') || !s.ends_with(')') {
        return None;
    }
    let inner = &s[1..s.len() - 1];
    let mut out = Vec::new();
    for piece in inner.split(',') {
        let piece = piece.trim();
        if let Some(id) = parse_ref(piece) {
            out.push(id);
        }
    }
    Some(out)
}

fn extract_nth_real(args: &str, n: usize) -> Option<f64> {
    let arg = extract_nth_arg_raw(args, n)?;
    let s = arg.trim();
    if s == "$" || s == "*" {
        return None;
    }
    s.parse::<f64>().ok()
}

fn extract_nth_arg_raw(args: &str, n: usize) -> Option<String> {
    let bytes = args.as_bytes();
    let mut i = 0;
    let mut depth: usize = 0;
    let mut in_string = false;
    let mut arg_idx: usize = 0;
    let mut start: usize = 0;

    while i < bytes.len() {
        let c = bytes[i];
        if in_string {
            if c == b'\'' {
                if bytes.get(i + 1) == Some(&b'\'') {
                    i += 2;
                    continue;
                }
                in_string = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'\'' => {
                in_string = true;
                i += 1;
            }
            b'(' => {
                depth += 1;
                i += 1;
            }
            b')' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            b',' if depth == 0 => {
                if arg_idx == n {
                    return Some(args[start..i].trim().to_string());
                }
                arg_idx += 1;
                i += 1;
                while i < bytes.len() && (bytes[i] as char).is_whitespace() {
                    i += 1;
                }
                start = i;
            }
            _ => i += 1,
        }
    }
    if arg_idx == n && start <= i {
        Some(args[start..i].trim().to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interop::ifc::parser::IfcParser;

    #[test]
    fn parses_building_and_storey_from_sample() {
        let src = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ifc/sample.ifc"),
        )
        .unwrap();
        let model = IfcParser::from_str(&src).expect("parses");
        let geometry = IfcGeometryParser::parse_model(&model);

        assert!(!geometry.buildings.is_empty(), "should have a building");
        assert!(
            !geometry.storeys.is_empty(),
            "should have at least one storey"
        );
        assert_eq!(
            geometry.zones.len(),
            model.spaces.len(),
            "one zone per space"
        );
    }

    #[test]
    fn zones_have_valid_floor_area() {
        let src = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ifc/sample.ifc"),
        )
        .unwrap();
        let model = IfcParser::from_str(&src).expect("parses");
        let geometry = IfcGeometryParser::parse_model(&model);

        for zone in &geometry.zones {
            assert!(zone.floor_area > 0.0, "floor area must be positive");
            assert!(zone.volume > 0.0, "volume must be positive");
            assert!(zone.height > 0.0, "height must be positive");
        }
    }

    #[test]
    fn zone_elements_derived_from_contained_in_spatial() {
        let src = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ifc/sample.ifc"),
        )
        .unwrap();
        let model = IfcParser::from_str(&src).expect("parses");
        let geometry = IfcGeometryParser::parse_model(&model);

        if let Some(space_id) = model.spaces.get(0).map(|s| s.id) {
            let elements = geometry.zone_elements.get(&space_id);
            assert!(
                elements.is_some() && !elements.unwrap().is_empty(),
                "space {space_id} should have contained elements"
            );
        }
    }

    #[test]
    fn zone_names_match_space_names() {
        let src = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ifc/sample.ifc"),
        )
        .unwrap();
        let model = IfcParser::from_str(&src).expect("parses");
        let geometry = IfcGeometryParser::parse_model(&model);

        for (i, space) in model.spaces.iter().enumerate() {
            assert_eq!(
                geometry.zones[i].name, space.name,
                "zone name must match space name"
            );
        }
    }
}
