// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! gbXML schema types for import/export.
//!
//! This module defines Rust structs that map to the gbXML schema elements.
//! Used for parsing and serializing gbXML files.
//!
//! # gbXML Schema Elements
//!
//! - [`GbXmlDocument`] - Top-level gbXML document
//! - [`Campus`] - Building site with location
//! - [`Building`] - Building with storeys
//! - [`BuildingStorey`] - Floor level
//! - [`Space`] - Thermal zone
//! - [`Surface`] - Wall/roof/floor with geometry
//! - [`Construction`] - Layer assembly
//! - [`Layer`] - Material layer
//! - [`Material`] - Thermal material properties
//!
//! # References
//!
//! See: <https://www.gbxml.org/schema_doc/6.01/GreenBuildingXML_Ver6.01.html>

use serde::{Deserialize, Serialize};

/// gbXML document root
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbXmlDocument {
    #[serde(rename = "xmlns", alias = "xmlns")]
    pub xmlns: String,
    #[serde(rename = "version")]
    pub version: String,
    #[serde(rename = "Campus")]
    pub campus: Campus,
    #[serde(rename = "Construction", default)]
    pub constructions: Vec<Construction>,
    #[serde(rename = "Layer", default)]
    pub layers: Vec<Layer>,
    #[serde(rename = "Material", default)]
    pub materials: Vec<Material>,
}

impl Default for GbXmlDocument {
    fn default() -> Self {
        GbXmlDocument {
            xmlns: "http://www.gbxml.org/schema".to_string(),
            version: "8.01".to_string(),
            campus: Campus::default(),
            constructions: Vec::new(),
            layers: Vec::new(),
            materials: Vec::new(),
        }
    }
}

/// Campus - top-level container containing a building
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Campus {
    #[serde(rename = "id", alias = "id")]
    pub id: String,
    #[serde(rename = "name", alias = "name", default)]
    pub name: String,
    #[serde(rename = "Location")]
    pub location: Location,
    #[serde(rename = "Building")]
    pub building: Building,
}

/// Building - contains building storeys
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Building {
    #[serde(rename = "id", alias = "id")]
    pub id: String,
    #[serde(rename = "name", alias = "name", default)]
    pub name: String,
    #[serde(rename = "BuildingStorey", default)]
    pub building_storeys: Vec<BuildingStorey>,
}

/// Building storey - floor level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingStorey {
    #[serde(rename = "id", alias = "id")]
    pub id: String,
    #[serde(rename = "name", alias = "name", default)]
    pub name: String,
    #[serde(rename = "level", alias = "level", default)]
    pub level: f64,
    #[serde(rename = "Space", default)]
    pub spaces: Vec<Space>,
}

/// Space - thermal zone
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Space {
    #[serde(rename = "id", alias = "id")]
    pub id: String,
    #[serde(rename = "name", alias = "name", default)]
    pub name: String,
    #[serde(rename = "Area", alias = "Area", default)]
    pub area: Option<f64>,
    #[serde(rename = "Volume", alias = "Volume", default)]
    pub volume: Option<f64>,
    #[serde(rename = "Surface", default)]
    pub surfaces: Vec<Surface>,
}

/// Surface - wall/roof/floor with geometry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Surface {
    #[serde(rename = "id", alias = "id")]
    pub id: String,
    #[serde(rename = "name", alias = "name", default)]
    pub name: String,
    #[serde(rename = "surfaceType", alias = "surfaceType")]
    pub surface_type: String,
    #[serde(rename = "Area", alias = "Area", default)]
    pub area: Option<f64>,
    #[serde(rename = "ConstructionIdRef", alias = "ConstructionIdRef", default)]
    pub construction_id_ref: Option<String>,
    #[serde(rename = "RectangularGeometry")]
    pub rectangular_geometry: RectangularGeometry,
    #[serde(rename = "AdjacentSpaceId", alias = "AdjacentSpaceId", default)]
    pub adjacent_space_ids: Vec<AdjacentSpaceId>,
}

/// Adjacent space reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjacentSpaceId {
    #[serde(rename = "spaceIdRef", alias = "spaceIdRef")]
    pub space_id_ref: String,
}

/// Rectangular geometry - planar surface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RectangularGeometry {
    #[serde(rename = "Azimuth", alias = "Azimuth", default)]
    pub azimuth: Option<f64>,
    #[serde(rename = "Tilt", alias = "Tilt", default)]
    pub tilt: Option<f64>,
    #[serde(rename = "CartesianPoint")]
    pub cartesian_point: CartesianPoint,
}

/// Cartesian point - 3D coordinate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartesianPoint {
    #[serde(rename = "Coordinate", alias = "Coordinate", default)]
    pub coordinates: Vec<f64>,
}

/// Location with latitude/longitude
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    #[serde(rename = "Name", alias = "Name", default)]
    pub name: String,
    #[serde(rename = "Latitude", alias = "Latitude", default)]
    pub latitude: Option<f64>,
    #[serde(rename = "Longitude", alias = "Longitude", default)]
    pub longitude: Option<f64>,
}

/// Construction - assembly of layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Construction {
    #[serde(rename = "id", alias = "id")]
    pub id: String,
    #[serde(rename = "name", alias = "name", default)]
    pub name: String,
    #[serde(rename = "layerCount", alias = "layerCount", default)]
    pub layer_count: Option<usize>,
    #[serde(rename = "LayerIdRef", alias = "LayerIdRef", default)]
    pub layer_id_refs: Vec<String>,
}

/// Layer - contains material references
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    #[serde(rename = "id", alias = "id")]
    pub id: String,
    #[serde(rename = "MaterialIdRef", alias = "MaterialIdRef", default)]
    pub material_id_refs: Vec<String>,
}

/// Material - thermal properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    #[serde(rename = "id", alias = "id")]
    pub id: String,
    #[serde(rename = "name", alias = "name", default)]
    pub name: String,
    #[serde(rename = "Thickness", alias = "Thickness", default)]
    pub thickness: Option<f64>,
    #[serde(rename = "Conductivity", alias = "Conductivity", default)]
    pub conductivity: Option<f64>,
    #[serde(rename = "Density", alias = "Density", default)]
    pub density: Option<f64>,
    #[serde(rename = "SpecificHeat", alias = "SpecificHeat", default)]
    pub specific_heat: Option<f64>,
    #[serde(rename = "Absorptance", alias = "Absorptance", default)]
    pub absorptance: Option<f64>,
    #[serde(rename = "Emissivity", alias = "Emissivity", default)]
    pub emissivity: Option<f64>,
}

/// CAD building surface types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CADBuildingSurfaceType {
    InteriorWall,
    ExteriorWall,
    Roof,
    Floor,
    Ceiling,
    InteriorFloor,
    UndergroundWall,
    UndergroundSlab,
    SlabOnGrade,
    FreestandingColumn,
    EmbeddedColumn,
    UndergroundCeiling,
    RaisedFloor,
    Undefined,
}

impl Default for Campus {
    fn default() -> Self {
        Campus {
            id: "campus1".to_string(),
            name: "Main Campus".to_string(),
            location: Location::default(),
            building: Building::default(),
        }
    }
}

impl Default for Building {
    fn default() -> Self {
        Building {
            id: "building1".to_string(),
            name: "Main Building".to_string(),
            building_storeys: Vec::new(),
        }
    }
}

impl Default for BuildingStorey {
    fn default() -> Self {
        BuildingStorey {
            id: "storey1".to_string(),
            name: "Floor 1".to_string(),
            level: 0.0,
            spaces: Vec::new(),
        }
    }
}

impl Default for Space {
    fn default() -> Self {
        Space {
            id: "space1".to_string(),
            name: "Zone 1".to_string(),
            area: Some(48.0),
            volume: Some(129.6),
            surfaces: Vec::new(),
        }
    }
}

impl Default for Surface {
    fn default() -> Self {
        Surface {
            id: "surface1".to_string(),
            name: "Wall 1".to_string(),
            surface_type: "ExteriorWall".to_string(),
            area: Some(12.0),
            construction_id_ref: None,
            rectangular_geometry: RectangularGeometry::default(),
            adjacent_space_ids: Vec::new(),
        }
    }
}

impl Default for RectangularGeometry {
    fn default() -> Self {
        RectangularGeometry {
            azimuth: Some(180.0),
            tilt: Some(90.0),
            cartesian_point: CartesianPoint::default(),
        }
    }
}

impl Default for CartesianPoint {
    fn default() -> Self {
        CartesianPoint {
            coordinates: vec![0.0, 0.0, 0.0],
        }
    }
}

impl Default for Location {
    fn default() -> Self {
        Location {
            name: "Unknown".to_string(),
            latitude: Some(39.739),
            longitude: Some(-104.984),
        }
    }
}

impl CADBuildingSurfaceType {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "InteriorWall" => CADBuildingSurfaceType::InteriorWall,
            "ExteriorWall" => CADBuildingSurfaceType::ExteriorWall,
            "Roof" => CADBuildingSurfaceType::Roof,
            "Floor" => CADBuildingSurfaceType::Floor,
            "Ceiling" => CADBuildingSurfaceType::Ceiling,
            "InteriorFloor" => CADBuildingSurfaceType::InteriorFloor,
            "UndergroundWall" => CADBuildingSurfaceType::UndergroundWall,
            "UndergroundSlab" => CADBuildingSurfaceType::UndergroundSlab,
            "SlabOnGrade" => CADBuildingSurfaceType::SlabOnGrade,
            "FreestandingColumn" => CADBuildingSurfaceType::FreestandingColumn,
            "EmbeddedColumn" => CADBuildingSurfaceType::EmbeddedColumn,
            "UndergroundCeiling" => CADBuildingSurfaceType::UndergroundCeiling,
            "RaisedFloor" => CADBuildingSurfaceType::RaisedFloor,
            _ => CADBuildingSurfaceType::Undefined,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            CADBuildingSurfaceType::InteriorWall => "InteriorWall",
            CADBuildingSurfaceType::ExteriorWall => "ExteriorWall",
            CADBuildingSurfaceType::Roof => "Roof",
            CADBuildingSurfaceType::Floor => "Floor",
            CADBuildingSurfaceType::Ceiling => "Ceiling",
            CADBuildingSurfaceType::InteriorFloor => "InteriorFloor",
            CADBuildingSurfaceType::UndergroundWall => "UndergroundWall",
            CADBuildingSurfaceType::UndergroundSlab => "UndergroundSlab",
            CADBuildingSurfaceType::SlabOnGrade => "SlabOnGrade",
            CADBuildingSurfaceType::FreestandingColumn => "FreestandingColumn",
            CADBuildingSurfaceType::EmbeddedColumn => "EmbeddedColumn",
            CADBuildingSurfaceType::UndergroundCeiling => "UndergroundCeiling",
            CADBuildingSurfaceType::RaisedFloor => "RaisedFloor",
            CADBuildingSurfaceType::Undefined => "Undefined",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cad_surface_type_from_str() {
        assert_eq!(
            CADBuildingSurfaceType::from_str("ExteriorWall"),
            CADBuildingSurfaceType::ExteriorWall
        );
        assert_eq!(
            CADBuildingSurfaceType::from_str("Roof"),
            CADBuildingSurfaceType::Roof
        );
        assert_eq!(
            CADBuildingSurfaceType::from_str("Unknown"),
            CADBuildingSurfaceType::Undefined
        );
    }

    #[test]
    fn test_cad_surface_type_as_str() {
        assert_eq!(
            CADBuildingSurfaceType::ExteriorWall.as_str(),
            "ExteriorWall"
        );
        assert_eq!(CADBuildingSurfaceType::Roof.as_str(), "Roof");
    }

    #[test]
    fn test_default_gbxml_document() {
        let doc = GbXmlDocument::default();
        assert_eq!(doc.version, "8.01");
        assert_eq!(doc.campus.name, "Main Campus");
    }

    #[test]
    fn test_default_space() {
        let space = Space::default();
        assert_eq!(space.name, "Zone 1");
        assert_eq!(space.area, Some(48.0));
        assert_eq!(space.volume, Some(129.6));
    }
}
