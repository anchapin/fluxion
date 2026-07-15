// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! OSM file parser - parses OpenStudio Model files into intermediate representations.
//!
//! This module provides the core parsing functionality for OSM (OpenStudio Model) files,
//! separating the tokenization and object parsing from the schema conversion logic in
//! [`reader.rs`](super::reader).
//!
//! # OSM Format
//!
//! OSM files use a line-oriented key-value format similar to IDF but with
//! OpenStudio-specific IDD schema. Objects are defined with their type name
//! followed by comma-separated fields.

use std::fs;
use std::path::Path;

use super::error::OsmError;
use super::types::*;

/// OSM Parser - parses OpenStudio Model files into OsmDocument.
pub struct OsmParser {
    current_line: usize,
}

impl OsmParser {
    pub fn new() -> Self {
        OsmParser { current_line: 0 }
    }

    pub fn parse_file(&mut self, path: impl AsRef<Path>) -> Result<OsmDocument, OsmError> {
        let content = fs::read_to_string(path.as_ref()).map_err(OsmError::IoError)?;
        self.parse_content(&content)
    }

    pub fn parse_content(&mut self, content: &str) -> Result<OsmDocument, OsmError> {
        let mut objects = Vec::new();
        let mut lines: Vec<&str> = content.lines().collect();
        let mut version = String::new();

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();
            self.current_line = i + 1;

            if line.is_empty() || line.starts_with('!') || line.starts_with('#') {
                i += 1;
                continue;
            }

            if line.starts_with("OS:") || line.starts_with("OSM:") {
                let obj = self.parse_object(&mut lines, &mut i)?;
                if obj.object_type == "OS:Version" {
                    version = obj
                        .fields
                        .get("version_identifier")
                        .cloned()
                        .unwrap_or_default();
                }
                objects.push(obj);
            }
            i += 1;
        }

        Ok(OsmDocument { version, objects })
    }

    fn parse_object(&self, lines: &mut [&str], idx: &mut usize) -> Result<OsmObject, OsmError> {
        let first_line = lines[*idx].trim();
        let object_type = self.extract_object_type(first_line)?;

        let mut obj = OsmObject::new(&object_type);
        let mut field_idx = 0;

        while *idx < lines.len() {
            *idx += 1;
            if *idx >= lines.len() {
                break;
            }

            let line = lines[*idx].trim();

            if line.is_empty() {
                continue;
            }

            let line_has_terminator = line.contains(';') || line == ";";

            if line == ";" {
                break;
            }

            if line.starts_with("OS:") || line.starts_with("OSM:") {
                if *idx > 0 {
                    *idx -= 1;
                }
                break;
            }

            if line.starts_with('!') {
                obj.comments
                    .push(line.strip_prefix('!').unwrap().trim().to_string());
                continue;
            }

            let field_value = self.extract_field_value(line, field_idx)?;
            let field_name = self.get_field_name(&object_type, field_idx);
            obj.fields.insert(field_name, field_value);
            field_idx += 1;

            if line_has_terminator {
                break;
            }
        }

        Ok(obj)
    }

    fn extract_object_type(&self, line: &str) -> Result<String, OsmError> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.is_empty() {
            return Err(OsmError::parse_error(
                self.current_line,
                "Empty object type",
            ));
        }

        let obj_type = parts[0].trim();
        if obj_type.starts_with("OS:") || obj_type.starts_with("OSM:") {
            Ok(obj_type.to_string())
        } else {
            Ok(format!("OS:{}", obj_type.trim_start_matches("OS:")))
        }
    }

    fn extract_field_value(&self, line: &str, _field_idx: usize) -> Result<String, OsmError> {
        let line = line.trim_end_matches(',').trim();

        let line = if let Some(idx) = line.rfind(';') {
            &line[..idx]
        } else {
            line
        };

        if let Some(idx) = line.find(", !-") {
            Ok(line[..idx].trim().to_string())
        } else {
            Ok(line.trim().to_string())
        }
    }

    fn get_field_name(&self, object_type: &str, field_idx: usize) -> String {
        match object_type {
            "OS:Material" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "roughness".to_string(),
                3 => "thickness".to_string(),
                4 => "conductivity".to_string(),
                5 => "density".to_string(),
                6 => "specific_heat".to_string(),
                7 => "emissivity".to_string(),
                8 => "vapor_transmission".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Construction" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                n => format!("layer_{}", n - 2),
            },
            "OS:BuildingStory" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "level".to_string(),
                3 => "height".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Space" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "zone_handle".to_string(),
                3 => "building_story_handle".to_string(),
                4 => "floor_area".to_string(),
                5 => "volume".to_string(),
                6 => "x_origin".to_string(),
                7 => "y_origin".to_string(),
                8 => "z_origin".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Surface" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "surface_type".to_string(),
                3 => "construction_handle".to_string(),
                4 => "building_boundary".to_string(),
                5 => "outside_boundary_condition".to_string(),
                6 => "sun_exposure".to_string(),
                7 => "wind_exposure".to_string(),
                _ => "extra".to_string(),
            },
            "OS:SubSurface" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "surface_handle".to_string(),
                3 => "construction_handle".to_string(),
                4 => "window_type".to_string(),
                _ => "extra".to_string(),
            },
            "OS:ThermalZone" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "thermostat_handle".to_string(),
                3 => "multiplier".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Site" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "latitude".to_string(),
                3 => "longitude".to_string(),
                4 => "elevation".to_string(),
                5 => "time_zone".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Building" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "building_story_handles".to_string(),
                3 => "zone_handles".to_string(),
                4 => "area".to_string(),
                5 => "number_of_floors".to_string(),
                6 => "floor_height".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Thermostat" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "heating_setpoint".to_string(),
                3 => "cooling_setpoint".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Lights" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "zone_handle".to_string(),
                3 => "design_level".to_string(),
                _ => "extra".to_string(),
            },
            "OS:ElectricEquipment" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "zone_handle".to_string(),
                3 => "design_level".to_string(),
                _ => "extra".to_string(),
            },
            "OS:People" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "zone_handle".to_string(),
                3 => "number_of_people".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Schedule:Constant" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "schedule_type".to_string(),
                3 => "value".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Version" => match field_idx {
                0 => "handle".to_string(),
                1 => "version_identifier".to_string(),
                _ => "extra".to_string(),
            },
            _ => "field".to_string(),
        }
    }
}

impl Default for OsmParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_OSM: &str = r#"
! Sample OpenStudio Model
OS:Version,
  {abc123},!- Handle
  3.2.0; !- Version Identifier

OS:Site,
  {site1}, !- Handle
  Denver CO, !- Name
  39.739, !- Latitude
  -104.984, !- Longitude
  1609; !- Elevation

OS:Building,
  {bldg1}, !- Handle
  Small Office, !- Name
  , !- Building Story Names
  , !- Thermal Zone Names
  1393.5, !- Floor Area {m2}
  1, !- Number of Floors
  3.0; !- Floor Height {m}

OS:Material,
  {mat1}, !- Handle
  Concrete, !- Name
  MediumRough, !- Roughness
  0.1, !- Thickness {m}
  1.4, !- Conductivity {W/m-K}
  2300, !- Density {kg/m3}
  840; !- Specific Heat {J/kg-K}

OS:Construction,
  {cons1}, !- Handle
  ExtWall, !- Name
  {mat1}; !- Outside Layer

OS:ThermalZone,
  {zone1}, !- Handle
  Zone 1, !- Name
  , !- Thermostat Handle
  1; !- Multiplier

OS:Space,
  {space1}, !- Handle
  Zone 1 Space, !- Name
  {zone1}, !- Zone Handle
  , !- Building Story Handle
  0, !- X Origin {m}
  0, !- Y Origin {m}
  0; !- Z Origin {m}

OS:Surface,
  {surf1}, !- Handle
  West Wall, !- Name
  Wall, !- Surface Type
  {cons1}, !- Construction Handle
  , !- Building Boundary Type
  Outdoors, !- Outside Boundary Condition
  , !- Sun Exposure
  ; !- Wind Exposure
"#;

    #[test]
    fn test_parse_version() {
        let mut parser = OsmParser::new();
        let result = parser.parse_content(SAMPLE_OSM);
        assert!(result.is_ok(), "Should parse OSM without error");
        let doc = result.unwrap();

        let version_obj = doc.objects.iter().find(|o| o.object_type == "OS:Version");
        assert!(version_obj.is_some(), "Should have OS:Version object");

        let version_field = version_obj.unwrap().fields.get("version_identifier");
        assert_eq!(
            version_field.as_ref().map(|s| s.as_str()),
            Some("3.2.0"),
            "Version should be 3.2.0"
        );
    }

    #[test]
    fn test_parse_material() {
        let mut parser = OsmParser::new();
        let result = parser.parse_content(SAMPLE_OSM);
        assert!(result.is_ok());
        let doc = result.unwrap();

        let materials: Vec<_> = doc
            .objects
            .iter()
            .filter(|o| o.object_type == "OS:Material")
            .collect();
        assert_eq!(materials.len(), 1, "Should have 1 material");

        let mat = &materials[0];
        assert_eq!(
            mat.fields.get("name").as_ref().map(|s| s.as_str()),
            Some("Concrete"),
            "Material name should be Concrete"
        );
        assert_eq!(
            mat.fields.get("thickness").as_ref().map(|s| s.as_str()),
            Some("0.1"),
            "Thickness should be 0.1"
        );
    }

    #[test]
    fn test_parse_construction() {
        let mut parser = OsmParser::new();
        let result = parser.parse_content(SAMPLE_OSM);
        assert!(result.is_ok());
        let doc = result.unwrap();

        let constructions: Vec<_> = doc
            .objects
            .iter()
            .filter(|o| o.object_type == "OS:Construction")
            .collect();
        assert_eq!(constructions.len(), 1, "Should have 1 construction");

        let cons = &constructions[0];
        assert_eq!(
            cons.fields.get("name").as_ref().map(|s| s.as_str()),
            Some("ExtWall"),
            "Construction name should be ExtWall"
        );
    }

    #[test]
    fn test_parse_thermal_zone() {
        let mut parser = OsmParser::new();
        let result = parser.parse_content(SAMPLE_OSM);
        assert!(result.is_ok());
        let doc = result.unwrap();

        let zones: Vec<_> = doc
            .objects
            .iter()
            .filter(|o| o.object_type == "OS:ThermalZone")
            .collect();
        assert_eq!(zones.len(), 1, "Should have 1 thermal zone");
        assert_eq!(
            zones[0].fields.get("name").as_ref().map(|s| s.as_str()),
            Some("Zone 1"),
            "Zone name should be Zone 1"
        );
    }

    #[test]
    fn test_parse_space() {
        let mut parser = OsmParser::new();
        let result = parser.parse_content(SAMPLE_OSM);
        assert!(result.is_ok());
        let doc = result.unwrap();

        let spaces: Vec<_> = doc
            .objects
            .iter()
            .filter(|o| o.object_type == "OS:Space")
            .collect();
        assert_eq!(spaces.len(), 1, "Should have 1 space");
        assert_eq!(
            spaces[0].fields.get("name").as_ref().map(|s| s.as_str()),
            Some("Zone 1 Space"),
            "Space name should be Zone 1 Space"
        );
    }

    #[test]
    fn test_parse_surface() {
        let mut parser = OsmParser::new();
        let result = parser.parse_content(SAMPLE_OSM);
        assert!(result.is_ok());
        let doc = result.unwrap();

        let surfaces: Vec<_> = doc
            .objects
            .iter()
            .filter(|o| o.object_type == "OS:Surface")
            .collect();
        assert_eq!(surfaces.len(), 1, "Should have 1 surface");
        assert_eq!(
            surfaces[0].fields.get("name").as_ref().map(|s| s.as_str()),
            Some("West Wall"),
            "Surface name should be West Wall"
        );
        assert_eq!(
            surfaces[0]
                .fields
                .get("surface_type")
                .as_ref()
                .map(|s| s.as_str()),
            Some("Wall"),
            "Surface type should be Wall"
        );
    }

    #[test]
    fn test_parse_building() {
        let mut parser = OsmParser::new();
        let result = parser.parse_content(SAMPLE_OSM);
        assert!(result.is_ok());
        let doc = result.unwrap();

        let buildings: Vec<_> = doc
            .objects
            .iter()
            .filter(|o| o.object_type == "OS:Building")
            .collect();
        assert_eq!(buildings.len(), 1, "Should have 1 building");
        assert_eq!(
            buildings[0].fields.get("name").as_ref().map(|s| s.as_str()),
            Some("Small Office"),
            "Building name should be Small Office"
        );
    }

    #[test]
    fn test_parse_site() {
        let mut parser = OsmParser::new();
        let result = parser.parse_content(SAMPLE_OSM);
        assert!(result.is_ok());
        let doc = result.unwrap();

        let sites: Vec<_> = doc
            .objects
            .iter()
            .filter(|o| o.object_type == "OS:Site")
            .collect();
        assert_eq!(sites.len(), 1, "Should have 1 site");
        assert_eq!(
            sites[0].fields.get("name").as_ref().map(|s| s.as_str()),
            Some("Denver CO"),
            "Site name should be Denver CO"
        );
        assert_eq!(
            sites[0].fields.get("latitude").as_ref().map(|s| s.as_str()),
            Some("39.739"),
            "Latitude should be 39.739"
        );
    }

    #[test]
    fn test_parse_building_with_handle() {
        let osm_with_handle = r#"
OS:Building,
  {bldg-1}, !- Handle
  Test Building, !- Name
  , !- Building Story Names
  , !- Thermal Zone Names
  100.0, !- Floor Area {m2}
  1, !- Number of Floors
  3.0; !- Floor Height {m}
"#;
        let mut parser = OsmParser::new();
        let result = parser.parse_content(osm_with_handle);
        assert!(result.is_ok());
        let doc = result.unwrap();

        let building = doc.objects.first().unwrap();
        assert_eq!(
            building.fields.get("handle").as_ref().map(|s| s.as_str()),
            Some("{bldg-1}"),
            "Handle should be {{bldg-1}}"
        );
    }

    #[test]
    fn test_parse_multilayer_construction() {
        let osm_multilayer = r#"
OS:Construction,
  {cons-w0}, !- Handle
  ExtWall, !- Name
  {mat-w0}, !- Layer 1
  {mat-w1}, !- Layer 2
  {mat-w2}; !- Layer 3
"#;
        let mut parser = OsmParser::new();
        let result = parser.parse_content(osm_multilayer);
        assert!(result.is_ok());
        let doc = result.unwrap();

        let cons = doc.objects.first().unwrap();
        assert_eq!(
            cons.fields.get("layer_1").as_ref().map(|s| s.as_str()),
            Some("{mat-w1}"),
            "Layer 1 should be {{mat-w1}}"
        );
    }
}
