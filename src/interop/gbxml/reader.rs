// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! gbXML reader - imports gbXML files into fluxion schema.
//!
//! This module provides functionality to parse gbXML files and convert
//! them into fluxion's [`SimulationSchema`] format.
//!
//! # Example
//!
//! ```ignore
//! use fluxion::interop::gbxml::{import_gbxml, GbXmlReader};
//!
//! let schema = import_gbxml("building.xml")?;
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use fluxion_core::parser_limits::ParserLimits;
use quick_xml::events::Event;
use quick_xml::Reader;
use quick_xml::XmlVersion;

use crate::api::schema::{
    ConstructionSet, ControlSet, Geometry, SchemaMetadata, SimulationOutput, SimulationSchemaV1,
    SurfaceConstruction, WeatherData, ZoneGeometry,
};
use crate::interop::gbxml::error::GbXmlError;
use crate::interop::gbxml::types::*;

/// Import a gbXML file with the strict default parser limits (64 MiB —
/// issue #2527).
pub fn import_gbxml(path: impl AsRef<Path>) -> Result<SimulationSchemaV1, GbXmlError> {
    import_gbxml_with_limits(path, &ParserLimits::default())
}

/// Import a gbXML file with explicit [`ParserLimits`] (issue #2527).
/// The on-disk size is checked via `fs::metadata` before the file is
/// read into memory.
pub fn import_gbxml_with_limits(
    path: impl AsRef<Path>,
    limits: &ParserLimits,
) -> Result<SimulationSchemaV1, GbXmlError> {
    let path = path.as_ref();
    let file_len = fs::metadata(path)
        .map_err(|e| GbXmlError::io_error(path, e.to_string()))?
        .len() as usize;
    limits.check_file_bytes(file_len)?;

    let content =
        fs::read_to_string(path).map_err(|e| GbXmlError::io_error(path, e.to_string()))?;

    let reader = GbXmlReader::new();
    reader.parse_with_limits(&content, limits)
}

/// Parse gbXML content into a GbXmlDocument with the strict default
/// parser limits (issue #2527).
pub fn parse_gbxml(content: &str) -> Result<GbXmlDocument, GbXmlError> {
    parse_gbxml_with_limits(content, &ParserLimits::default())
}

/// Parse gbXML content into a GbXmlDocument with explicit
/// [`ParserLimits`] (issue #2527). Enforces `max_file_bytes` before the
/// XML event loop runs.
pub fn parse_gbxml_with_limits(
    content: &str,
    limits: &ParserLimits,
) -> Result<GbXmlDocument, GbXmlError> {
    limits.check_file_bytes(content.len())?;
    let mut reader = Reader::from_str(content);
    reader.config_mut().trim_text(true);

    let mut doc = GbXmlDocument::default();
    let mut stack: Vec<String> = Vec::new();
    let mut _current_element = String::new();
    let mut text_content = String::new();
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                _current_element = e.name().as_ref().to_string();
                stack.push(_current_element.clone());
                text_content.clear();

                parse_start_element(&mut doc, &_current_element, &e)?;
            }
            Ok(Event::Text(e)) => {
                text_content = e.xml10_content().into_owned();
            }
            Ok(Event::End(e)) => {
                let end_element = e.name().as_ref().to_string();
                if let Some(pop) = stack.pop() {
                    if pop != end_element {
                        return Err(GbXmlError::invalid_structure(format!(
                            "Mismatched tags: {} vs {}",
                            pop, end_element
                        )));
                    }
                }
                parse_end_element(&mut doc, &end_element, &text_content)?;
                text_content.clear();
            }
            Ok(Event::Empty(e)) => {
                _current_element = e.name().as_ref().to_string();
                parse_start_element(&mut doc, &_current_element, &e)?;
            }
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(e) => return Err(GbXmlError::XmlParseError(e.to_string())),
        }
    }

    Ok(doc)
}

fn parse_start_element(
    doc: &mut GbXmlDocument,
    element: &str,
    e: &quick_xml::events::BytesStart,
) -> Result<(), GbXmlError> {
    match element {
        "gbXML" => {
            for attr in e.attributes().flatten() {
                let key = attr.key.as_ref().to_string();
                if key == "version" {
                    doc.version = attr
                        .normalized_value(XmlVersion::Implicit1_0)
                        .unwrap_or_default()
                        .to_string();
                }
            }
        }
        "Campus" => {
            doc.campus.id = get_attribute(e, "id").unwrap_or_else(|| "campus1".to_string());
            doc.campus.name = get_attribute(e, "name").unwrap_or_default();
        }
        "Location" => {}
        "Building" => {
            doc.campus.building.id =
                get_attribute(e, "id").unwrap_or_else(|| "building1".to_string());
            doc.campus.building.name = get_attribute(e, "name").unwrap_or_default();
        }
        "BuildingStorey" => {
            let storey = BuildingStorey {
                id: get_attribute(e, "id").unwrap_or_else(|| {
                    format!("storey{}", doc.campus.building.building_storeys.len() + 1)
                }),
                name: get_attribute(e, "name").unwrap_or_default(),
                level: get_attribute(e, "level")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0),
                spaces: Vec::new(),
            };
            doc.campus.building.building_storeys.push(storey);
        }
        "Space" => {
            let space = Space {
                id: get_attribute(e, "id")
                    .unwrap_or_else(|| format!("space{}", get_space_count(doc))),
                name: get_attribute(e, "name").unwrap_or_default(),
                area: None,
                volume: None,
                surfaces: Vec::new(),
            };
            if let Some(storey) = doc.campus.building.building_storeys.last_mut() {
                storey.spaces.push(space);
            }
        }
        "Surface" => {
            let surface = Surface {
                id: get_attribute(e, "id")
                    .unwrap_or_else(|| format!("surface{}", get_surface_count(doc))),
                name: get_attribute(e, "name").unwrap_or_default(),
                surface_type: get_attribute(e, "surfaceType")
                    .unwrap_or_else(|| "Undefined".to_string()),
                area: None,
                construction_id_ref: get_attribute(e, "constructionIdRef"),
                rectangular_geometry: RectangularGeometry::default(),
                adjacent_space_ids: Vec::new(),
            };
            if let Some(storey) = doc.campus.building.building_storeys.last_mut() {
                if let Some(space) = storey.spaces.last_mut() {
                    space.surfaces.push(surface);
                }
            }
        }
        "Construction" => {
            let construction = Construction {
                id: get_attribute(e, "id")
                    .unwrap_or_else(|| format!("construction{}", doc.constructions.len() + 1)),
                name: get_attribute(e, "name").unwrap_or_default(),
                layer_count: get_attribute(e, "layerCount").and_then(|s| s.parse::<usize>().ok()),
                layer_id_refs: Vec::new(),
            };
            doc.constructions.push(construction);
        }
        "Layer" => {
            let layer = Layer {
                id: get_attribute(e, "id")
                    .unwrap_or_else(|| format!("layer{}", doc.layers.len() + 1)),
                material_id_refs: Vec::new(),
            };
            doc.layers.push(layer);
        }
        "Material" => {
            let material = Material {
                id: get_attribute(e, "id")
                    .unwrap_or_else(|| format!("material{}", doc.materials.len() + 1)),
                name: get_attribute(e, "name").unwrap_or_default(),
                thickness: None,
                conductivity: None,
                density: None,
                specific_heat: None,
                absorptance: None,
                emissivity: None,
            };
            doc.materials.push(material);
        }
        "RectangularGeometry" => {
            if let Some(storey) = doc.campus.building.building_storeys.last_mut() {
                if let Some(space) = storey.spaces.last_mut() {
                    if let Some(surface) = space.surfaces.last_mut() {
                        surface.rectangular_geometry.azimuth =
                            get_attribute(e, "Azimuth").and_then(|s| s.parse::<f64>().ok());
                        surface.rectangular_geometry.tilt =
                            get_attribute(e, "Tilt").and_then(|s| s.parse::<f64>().ok());
                    }
                }
            }
        }
        "CartesianPoint" => {}
        "AdjacentSpaceId" => {
            if let Some(space_id_ref) = get_attribute(e, "spaceIdRef") {
                if let Some(storey) = doc.campus.building.building_storeys.last_mut() {
                    if let Some(space) = storey.spaces.last_mut() {
                        if let Some(surface) = space.surfaces.last_mut() {
                            surface
                                .adjacent_space_ids
                                .push(AdjacentSpaceId { space_id_ref });
                        }
                    }
                }
            }
        }
        "LayerIdRef" => {
            if let Some(id_ref) = get_attribute(e, "layerIdRef") {
                if let Some(construction) = doc.constructions.last_mut() {
                    construction.layer_id_refs.push(id_ref);
                }
            }
        }
        "MaterialIdRef" => {
            if let Some(id_ref) = get_attribute(e, "materialIdRef") {
                if let Some(layer) = doc.layers.last_mut() {
                    layer.material_id_refs.push(id_ref);
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn parse_end_element(doc: &mut GbXmlDocument, element: &str, text: &str) -> Result<(), GbXmlError> {
    match element {
        "Name" => {
            let storey_count = doc.campus.building.building_storeys.len();
            let in_space_context = doc
                .campus
                .building
                .building_storeys
                .last()
                .map(|s| !s.spaces.is_empty())
                .unwrap_or(false);

            if !in_space_context && storey_count == 0 {
                // No storeys yet - Name belongs to location
                if doc.campus.location.name.is_empty() || doc.campus.location.name == "Unknown" {
                    doc.campus.location.name = text.trim().to_string();
                }
            }
        }
        "Latitude" => {
            if let Ok(lat) = text.trim().parse::<f64>() {
                doc.campus.location.latitude = Some(lat);
            }
        }
        "Longitude" => {
            if let Ok(lon) = text.trim().parse::<f64>() {
                doc.campus.location.longitude = Some(lon);
            }
        }
        "Area" => {
            if let Ok(area) = text.trim().parse::<f64>() {
                if let Some(storey) = doc.campus.building.building_storeys.last_mut() {
                    if let Some(space) = storey.spaces.last_mut() {
                        if space.area.is_none() {
                            space.area = Some(area);
                        } else if let Some(surface) = space.surfaces.last_mut() {
                            if surface.area.is_none() {
                                surface.area = Some(area);
                            }
                        }
                    }
                }
            }
        }
        "Volume" => {
            if let Ok(vol) = text.trim().parse::<f64>() {
                if let Some(storey) = doc.campus.building.building_storeys.last_mut() {
                    if let Some(space) = storey.spaces.last_mut() {
                        if space.volume.is_none() {
                            space.volume = Some(vol);
                        }
                    }
                }
            }
        }
        "Thickness" => {
            if let Ok(thickness) = text.trim().parse::<f64>() {
                if let Some(material) = doc.materials.last_mut() {
                    material.thickness = Some(thickness);
                }
            }
        }
        "Conductivity" => {
            if let Ok(cond) = text.trim().parse::<f64>() {
                if let Some(material) = doc.materials.last_mut() {
                    material.conductivity = Some(cond);
                }
            }
        }
        "Density" => {
            if let Ok(dens) = text.trim().parse::<f64>() {
                if let Some(material) = doc.materials.last_mut() {
                    material.density = Some(dens);
                }
            }
        }
        "SpecificHeat" => {
            if let Ok(sh) = text.trim().parse::<f64>() {
                if let Some(material) = doc.materials.last_mut() {
                    material.specific_heat = Some(sh);
                }
            }
        }
        "Absorptance" => {
            if let Ok(abs) = text.trim().parse::<f64>() {
                if let Some(material) = doc.materials.last_mut() {
                    material.absorptance = Some(abs);
                }
            }
        }
        "Emissivity" => {
            if let Ok(emi) = text.trim().parse::<f64>() {
                if let Some(material) = doc.materials.last_mut() {
                    material.emissivity = Some(emi);
                }
            }
        }
        "Coordinate" => {
            if let Ok(coord) = text.trim().parse::<f64>() {
                if let Some(storey) = doc.campus.building.building_storeys.last_mut() {
                    if let Some(space) = storey.spaces.last_mut() {
                        if let Some(surface) = space.surfaces.last_mut() {
                            surface
                                .rectangular_geometry
                                .cartesian_point
                                .coordinates
                                .push(coord);
                        }
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn get_attribute(e: &quick_xml::events::BytesStart, key: &str) -> Option<String> {
    for attr in e.attributes().flatten() {
        if attr.key.as_ref() == key {
            return Some(
                attr.normalized_value(XmlVersion::Implicit1_0)
                    .unwrap_or_default()
                    .to_string(),
            );
        }
    }
    None
}

fn get_space_count(doc: &GbXmlDocument) -> usize {
    doc.campus
        .building
        .building_storeys
        .iter()
        .map(|s| s.spaces.len())
        .sum()
}

fn get_surface_count(doc: &GbXmlDocument) -> usize {
    doc.campus
        .building
        .building_storeys
        .iter()
        .map(|s| s.spaces.iter().map(|sp| sp.surfaces.len()).sum::<usize>())
        .sum()
}

/// GbXmlReader for parsing gbXML files.
#[allow(dead_code)]
pub struct GbXmlReader {
    construction_map: HashMap<String, Construction>,
    layer_map: HashMap<String, Layer>,
    material_map: HashMap<String, Material>,
}

impl GbXmlReader {
    /// Create a new GbXmlReader.
    pub fn new() -> Self {
        GbXmlReader {
            construction_map: HashMap::new(),
            layer_map: HashMap::new(),
            material_map: HashMap::new(),
        }
    }

    /// Parse gbXML content into fluxion SimulationSchema.
    pub fn parse(&self, content: &str) -> Result<SimulationSchemaV1, GbXmlError> {
        self.parse_with_limits(content, &ParserLimits::default())
    }

    /// Parse gbXML content with explicit [`ParserLimits`] (issue #2527).
    pub fn parse_with_limits(
        &self,
        content: &str,
        limits: &ParserLimits,
    ) -> Result<SimulationSchemaV1, GbXmlError> {
        let doc = parse_gbxml_with_limits(content, limits)?;

        // Build lookup maps
        let mut construction_map: HashMap<String, &Construction> = HashMap::new();
        for c in &doc.constructions {
            construction_map.insert(c.id.clone(), c);
        }

        let mut layer_map: HashMap<String, &Layer> = HashMap::new();
        for l in &doc.layers {
            layer_map.insert(l.id.clone(), l);
        }

        let mut material_map: HashMap<String, &Material> = HashMap::new();
        for m in &doc.materials {
            material_map.insert(m.id.clone(), m);
        }

        // Extract location for weather
        let location_name = doc.campus.location.name.clone();

        // Convert zones
        let mut zones: Vec<ZoneGeometry> = Vec::new();
        let mut total_floor_area = 0.0;
        let mut total_volume = 0.0;
        let mut number_of_floors = doc.campus.building.building_storeys.len();

        for storey in &doc.campus.building.building_storeys {
            for space in &storey.spaces {
                let zone = ZoneGeometry {
                    name: space.name.clone(),
                    floor_area: space.area.unwrap_or(48.0),
                    volume: space.volume.unwrap_or(129.6),
                    height: if space.area.unwrap_or(48.0) > 0.0 {
                        space.volume.unwrap_or(129.6) / space.area.unwrap_or(48.0)
                    } else {
                        2.7
                    },
                };
                total_floor_area += zone.floor_area;
                total_volume += zone.volume;
                zones.push(zone);
            }
        }

        // If no zones found, create a default one
        if zones.is_empty() {
            zones.push(ZoneGeometry::default());
            total_floor_area = 48.0;
            total_volume = 129.6;
            number_of_floors = 1;
        }

        let geometry = Geometry {
            zones,
            total_floor_area,
            total_volume,
            number_of_floors,
            floor_height: if number_of_floors > 0 {
                total_volume / total_floor_area
            } else {
                2.7
            },
        };

        // Create default constructions (wall, roof, floor)
        let constructions = ConstructionSet {
            wall: SurfaceConstruction::default(),
            roof: SurfaceConstruction::default(),
            floor: SurfaceConstruction::default(),
            interzone: None,
        };

        let metadata = SchemaMetadata {
            name: doc.campus.building.name.clone(),
            description: format!("Imported from gbXML. Location: {}", location_name),
            author: None,
            created_at: Some(chrono::Utc::now().format("%Y-%m-%d").to_string()),
            schema_version: crate::api::schema::SchemaVersion::V1,
        };

        let weather = if location_name != "Unknown" && location_name.is_empty() {
            WeatherData::TmyLocation {
                location: location_name,
            }
        } else {
            WeatherData::TmyLocation {
                location: "Denver, CO".to_string(),
            }
        };

        Ok(SimulationSchemaV1 {
            version: crate::api::schema::SchemaVersion::V1,
            metadata,
            geometry,
            constructions,
            schedules: crate::api::schema::ScheduleSet::default(),
            weather,
            controls: ControlSet::default(),
            output: SimulationOutput::default(),
        })
    }
}

impl Default for GbXmlReader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_GBXML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<gbXML xmlns="http://www.gbxml.org/schema" version="8.01">
  <Campus id="c1" name="Main Campus">
    <Location>
      <Name>Denver, CO</Name>
      <Latitude>39.739</Latitude>
      <Longitude>-104.984</Longitude>
    </Location>
    <Building id="b1" name="Office Building">
      <BuildingStorey id="bs1" name="Floor 1" level="0">
        <Space id="s1" name="Zone 1">
          <Area>48.0</Area>
          <Volume>129.6</Volume>
          <Surface id="surf1" surfaceType="ExteriorWall" constructionIdRef="c1">
            <Name>West Wall</Name>
            <Area>12.0</Area>
            <RectangularGeometry Azimuth="180" Tilt="90">
              <CartesianPoint>
                <Coordinate>0.0</Coordinate>
                <Coordinate>0.0</Coordinate>
                <Coordinate>0.0</Coordinate>
              </CartesianPoint>
            </RectangularGeometry>
            <AdjacentSpaceId spaceIdRef="s1"/>
          </Surface>
        </Space>
      </BuildingStorey>
    </Building>
  </Campus>
  <Construction id="c1" name="ExtWall" layerCount="2">
    <LayerIdRef layerIdRef="layer1"/>
    <LayerIdRef layerIdRef="layer2"/>
  </Construction>
  <Layer id="layer1">
    <MaterialIdRef materialIdRef="mat1"/>
  </Layer>
  <Layer id="layer2">
    <MaterialIdRef materialIdRef="mat2"/>
  </Layer>
  <Material id="mat1" name="Concrete">
    <Thickness>0.1</Thickness>
    <Conductivity>1.4</Conductivity>
    <Density>2300</Density>
    <SpecificHeat>840</SpecificHeat>
  </Material>
  <Material id="mat2" name="Insulation">
    <Thickness>0.05</Thickness>
    <Conductivity>0.04</Conductivity>
    <Density>50</Density>
    <SpecificHeat>840</SpecificHeat>
  </Material>
</gbXML>"#;

    #[test]
    fn test_parse_gbxml() {
        let doc = parse_gbxml(SAMPLE_GBXML).expect("Should parse gbXML");
        assert_eq!(doc.version, "8.01");
        assert_eq!(doc.campus.name, "Main Campus");
        assert_eq!(doc.campus.location.name, "Denver, CO");
    }

    #[test]
    fn test_import_gbxml() {
        let reader = GbXmlReader::new();
        let schema = reader
            .parse(SAMPLE_GBXML)
            .expect("Should convert to schema");
        assert_eq!(schema.geometry.zones.len(), 1);
        assert_eq!(schema.geometry.zones[0].name, "Zone 1");
        assert_eq!(schema.geometry.total_floor_area, 48.0);
    }

    #[test]
    fn test_missing_optional_elements() {
        let minimal = r#"<?xml version="1.0" encoding="UTF-8"?>
<gbXML xmlns="http://www.gbxml.org/schema" version="8.01">
  <Campus id="c1"/>
</gbXML>"#;
        let doc = parse_gbxml(minimal).expect("Should parse minimal gbXML");
        assert_eq!(doc.campus.id, "c1");
    }

    // ----- Issue #2527: parser DoS limits -----------------------------------

    fn tiny_limits() -> fluxion_core::parser_limits::ParserLimits {
        fluxion_core::parser_limits::ParserLimits {
            max_file_bytes: 512,
            max_lines: 1_000_000,
            max_recursion_depth: 256,
            max_array_elements: 1_000_000,
        }
    }

    #[test]
    fn gbxml_rejects_oversized_bytes() {
        // Build a gbXML doc larger than 512 bytes.
        let mut big = String::from(
            "<?xml version=\"1.0\"?><gbXML xmlns=\"http://www.gbxml.org/schema\" version=\"8.01\">",
        );
        big.push_str(&"<Campus id=\"c1\"/>".repeat(40));
        big.push_str("</gbXML>");
        let err = parse_gbxml_with_limits(&big, &tiny_limits()).unwrap_err();
        assert!(
            matches!(err, GbXmlError::SizeLimitExceeded(_)),
            "expected SizeLimitExceeded, got {:?}",
            err
        );
        assert!(err.to_string().to_lowercase().contains("file size"));
    }

    #[test]
    fn gbxml_normal_parses_with_default_limits() {
        let reader = GbXmlReader::new();
        let schema = reader.parse(SAMPLE_GBXML).expect("default limits parse");
        assert_eq!(schema.geometry.zones.len(), 1);
    }
}
