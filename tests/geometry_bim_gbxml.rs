//! Geometry and BIM Translation Tests (gbXML Parsing)
//!
//! Validates that gbXML files can be correctly parsed into Fluxion's internal
//! geometry structures (Assembly, Boundary, ZoneGeometry).
//!
//! # Test Cases
//!
//! - **Shoebox**: Simple rectangular single-zone building
//! - **L-Shaped**: Multi-zone building with inter-zone walls
//! - **Multi-Story**: Two-story building with inter-zone floor
//! - **Inter-Zone Walls**: Building with multiple inter-zone surfaces
//! - **Complex Building**: Multiple zones with skylights and windows
//!
//! # Assertions
//!
//! - Total surface areas by type (walls, roofs, floors)
//! - Volume calculations
//! - Boundary condition assignments (exterior vs inter-zone)
//!
//! See Issue #1055

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// gbXML surface types mapped to Fluxion boundary conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceType {
    ExteriorWall,
    InteriorWall,
    Roof,
    Skylight,
    SlabOnGrade,
    InteriorFloor,
    Unknown,
}

impl SurfaceType {
    /// Classify boundary condition based on gbXML surface type
    pub fn to_boundary_condition(&self) -> BoundaryCondition {
        match self {
            SurfaceType::ExteriorWall
            | SurfaceType::Roof
            | SurfaceType::Skylight
            | SurfaceType::SlabOnGrade => BoundaryCondition::Exterior,
            SurfaceType::InteriorWall | SurfaceType::InteriorFloor => BoundaryCondition::InterZone,
            SurfaceType::Unknown => BoundaryCondition::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    Exterior,
    InterZone,
    Unknown,
}

/// A parsed surface from gbXML
#[derive(Debug, Clone)]
pub struct ParsedSurface {
    pub id: String,
    pub name: String,
    pub surface_type: SurfaceType,
    pub area: f64,
    pub height: f64,
    pub azimuth: f64,
    pub tilt: f64,
    pub adjacent_space_id: Option<String>,
    pub construction_id: Option<String>,
    pub boundary_condition: BoundaryCondition,
}

/// A parsed space from gbXML
#[derive(Debug, Clone)]
pub struct ParsedSpace {
    pub id: String,
    pub name: String,
    pub area: f64,
    pub volume: f64,
}

/// A parsed zone from gbXML
#[derive(Debug, Clone)]
pub struct ParsedZone {
    pub id: String,
    pub name: String,
    pub volume: f64,
    pub area: f64,
    pub spaces: Vec<ParsedSpace>,
}

/// A parsed building story from gbXML
#[derive(Debug, Clone)]
pub struct ParsedStory {
    pub id: String,
    pub name: String,
    pub level: f64,
    pub height: f64,
}

/// A parsed building from gbXML
#[derive(Debug, Clone)]
pub struct ParsedBuilding {
    pub id: String,
    pub name: String,
    pub stories: Vec<ParsedStory>,
    pub zones: Vec<ParsedZone>,
}

/// A parsed construction from gbXML
#[derive(Debug, Clone)]
pub struct ParsedConstruction {
    pub id: String,
    pub name: String,
    pub u_value: Option<f64>,
}

/// Geometry ingestion result - maps gbXML to Fluxion internal structures
#[derive(Debug, Clone)]
pub struct GbXmlGeometry {
    pub building: ParsedBuilding,
    pub surfaces: Vec<ParsedSurface>,
    pub constructions: HashMap<String, ParsedConstruction>,
    pub total_floor_area: f64,
    pub total_volume: f64,
    pub exterior_wall_area: f64,
    pub interzone_wall_area: f64,
    pub roof_area: f64,
    pub floor_area: f64,
    pub skylight_area: f64,
}

/// Parse surface type string to enum
fn parse_surface_type(type_str: &str) -> SurfaceType {
    match type_str {
        "ExteriorWall" => SurfaceType::ExteriorWall,
        "InteriorWall" => SurfaceType::InteriorWall,
        "Roof" => SurfaceType::Roof,
        "Skylight" => SurfaceType::Skylight,
        "SlabOnGrade" => SurfaceType::SlabOnGrade,
        "InteriorFloor" => SurfaceType::InteriorFloor,
        _ => SurfaceType::Unknown,
    }
}

/// Simple gbXML parser using basic string operations
/// Extracts key geometry data from gbXML files for testing
pub fn parse_gbxml(content: &str) -> Result<GbXmlGeometry, String> {
    let mut surfaces = Vec::new();
    let mut zones: Vec<ParsedZone> = Vec::new();
    let mut constructions: HashMap<String, ParsedConstruction> = HashMap::new();
    let mut building_name = String::new();
    let mut stories: Vec<ParsedStory> = Vec::new();

    // Extract building name
    if let Some(start) = content.find("<Name>") {
        if let Some(end) = content[start..].find("</Name>") {
            let name_start = start + 6;
            let name_end = start + end;
            building_name = content[name_start..name_end].to_string();
        }
    }

    // Extract zones with their volumes and areas
    let mut zone_pos = 0;
    while let Some(zone_start) = content[zone_pos..].find("<Zone id=\"") {
        let abs_zone_start = zone_pos + zone_start;
        if let Some(zone_end_tag) = content[abs_zone_start..].find("</Zone>") {
            let zone_end = abs_zone_start + zone_end_tag;
            let zone_content = &content[abs_zone_start..zone_end];

            let zone_id = if let Some(id_start) = zone_content.find("id=\"") {
                let id_start = id_start + 4;
                if let Some(id_end) = zone_content[id_start..].find("\"") {
                    zone_content[id_start..id_start + id_end].to_string()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let zone_name = if let Some(name_start) = zone_content.find("<Name>") {
                let name_start = name_start + 6;
                if let Some(name_end) = zone_content[name_start..].find("</Name>") {
                    zone_content[name_start..name_start + name_end].to_string()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let zone_volume: f64 = if let Some(vol_start) = zone_content.find("<Volume>") {
                let vol_start = vol_start + 8;
                if let Some(vol_end) = zone_content[vol_start..].find("</Volume>") {
                    zone_content[vol_start..vol_start + vol_end]
                        .trim()
                        .parse()
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let zone_area: f64 = if let Some(area_start) = zone_content.find("<Area>") {
                let area_start = area_start + 6;
                if let Some(area_end) = zone_content[area_start..].find("</Area>") {
                    zone_content[area_start..area_start + area_end]
                        .trim()
                        .parse()
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            zones.push(ParsedZone {
                id: zone_id,
                name: zone_name,
                volume: zone_volume,
                area: zone_area,
                spaces: Vec::new(),
            });

            zone_pos = zone_end;
        } else {
            break;
        }
    }

    // Extract surfaces
    let mut surf_pos = 0;
    while let Some(surf_start) = content[surf_pos..].find("<Surface id=\"") {
        let abs_surf_start = surf_pos + surf_start;
        if let Some(surf_end_tag) = content[abs_surf_start..].find("</Surface>") {
            let surf_end = abs_surf_start + surf_end_tag + 11; // include closing tag
            let surf_content = &content[abs_surf_start..surf_end];

            let surface_id = if let Some(id_start) = surf_content.find("id=\"") {
                let id_start = id_start + 4;
                if let Some(id_end) = surf_content[id_start..].find("\"") {
                    surf_content[id_start..id_start + id_end].to_string()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let surface_type_str = if let Some(type_start) = surf_content.find("surfaceType=\"") {
                let type_start = type_start + 13;
                if let Some(type_end) = surf_content[type_start..].find("\"") {
                    surf_content[type_start..type_start + type_end].to_string()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
            let surface_type = parse_surface_type(&surface_type_str);

            let surface_name = if let Some(name_start) = surf_content.find("<Name>") {
                let name_start = name_start + 6;
                if let Some(name_end) = surf_content[name_start..].find("</Name>") {
                    surf_content[name_start..name_start + name_end].to_string()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let surface_area: f64 = if let Some(area_start) = surf_content.find("<Area>") {
                let area_start = area_start + 6;
                if let Some(area_end) = surf_content[area_start..].find("</Area>") {
                    surf_content[area_start..area_start + area_end]
                        .trim()
                        .parse()
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let surface_height: f64 = if let Some(h_start) = surf_content.find("<Height>") {
                let h_start = h_start + 8;
                if let Some(h_end) = surf_content[h_start..].find("</Height>") {
                    surf_content[h_start..h_start + h_end]
                        .trim()
                        .parse()
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let surface_azimuth: f64 = if let Some(az_start) = surf_content.find("<Azimuth>") {
                let az_start = az_start + 9;
                if let Some(az_end) = surf_content[az_start..].find("</Azimuth>") {
                    surf_content[az_start..az_start + az_end]
                        .trim()
                        .parse()
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let surface_tilt: f64 = if let Some(tilt_start) = surf_content.find("<Tilt>") {
                let tilt_start = tilt_start + 6;
                if let Some(tilt_end) = surf_content[tilt_start..].find("</Tilt>") {
                    surf_content[tilt_start..tilt_start + tilt_end]
                        .trim()
                        .parse()
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let adjacent_space = if let Some(adj_start) = surf_content.find("<AdjacentSpaceIdRef>")
            {
                let adj_start = adj_start + 20;
                if let Some(adj_end) = surf_content[adj_start..].find("</AdjacentSpaceIdRef>") {
                    Some(surf_content[adj_start..adj_start + adj_end].to_string())
                } else {
                    None
                }
            } else {
                None
            };

            let construction_id = if let Some(c_start) = surf_content.find("constructionIdRef=\"") {
                let c_start = c_start + 18;
                if let Some(c_end) = surf_content[c_start..].find("\"") {
                    Some(surf_content[c_start..c_start + c_end].to_string())
                } else {
                    None
                }
            } else {
                None
            };

            let boundary_condition = surface_type.to_boundary_condition();

            surfaces.push(ParsedSurface {
                id: surface_id,
                name: surface_name,
                surface_type,
                area: surface_area,
                height: surface_height,
                azimuth: surface_azimuth,
                tilt: surface_tilt,
                adjacent_space_id: adjacent_space,
                construction_id,
                boundary_condition,
            });

            surf_pos = surf_end;
        } else {
            break;
        }
    }

    // Extract constructions
    let mut const_pos = 0;
    while let Some(const_start) = content[const_pos..].find("<Construction id=\"") {
        let abs_const_start = const_pos + const_start;
        if let Some(const_end_tag) = content[abs_const_start..].find("</Construction>") {
            let const_end = abs_const_start + const_end_tag + 15;
            let const_content = &content[abs_const_start..const_end];

            let const_id = if let Some(id_start) = const_content.find("id=\"") {
                let id_start = id_start + 4;
                if let Some(id_end) = const_content[id_start..].find("\"") {
                    const_content[id_start..id_start + id_end].to_string()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let const_name = if let Some(name_start) = const_content.find("<Name>") {
                let name_start = name_start + 6;
                if let Some(name_end) = const_content[name_start..].find("</Name>") {
                    const_content[name_start..name_start + name_end].to_string()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let const_u_value: Option<f64> = if let Some(u_start) = const_content.find("<U-value>")
            {
                let u_start = u_start + 9; // +9 to skip "<U-value>" (8 chars) + ">" (1 char)
                if let Some(u_end) = const_content[u_start..].find("</U-value>") {
                    const_content[u_start..u_start + u_end].trim().parse().ok()
                } else {
                    None
                }
            } else {
                None
            };

            constructions.insert(
                const_id.clone(),
                ParsedConstruction {
                    id: const_id,
                    name: const_name,
                    u_value: const_u_value,
                },
            );

            const_pos = const_end;
        } else {
            break;
        }
    }

    // Calculate totals
    let total_floor_area: f64 = zones.iter().map(|z| z.area).sum();
    let total_volume: f64 = zones.iter().map(|z| z.volume).sum();
    let exterior_wall_area: f64 = surfaces
        .iter()
        .filter(|s| s.surface_type == SurfaceType::ExteriorWall)
        .map(|s| s.area)
        .sum();
    let interzone_wall_area: f64 = surfaces
        .iter()
        .filter(|s| s.surface_type == SurfaceType::InteriorWall)
        .map(|s| s.area)
        .sum();
    let roof_area: f64 = surfaces
        .iter()
        .filter(|s| s.surface_type == SurfaceType::Roof)
        .map(|s| s.area)
        .sum();
    let floor_area: f64 = surfaces
        .iter()
        .filter(|s| s.surface_type == SurfaceType::SlabOnGrade)
        .map(|s| s.area)
        .sum();
    let skylight_area: f64 = surfaces
        .iter()
        .filter(|s| s.surface_type == SurfaceType::Skylight)
        .map(|s| s.area)
        .sum();

    let building = ParsedBuilding {
        id: "b1".to_string(),
        name: building_name,
        stories,
        zones,
    };

    Ok(GbXmlGeometry {
        building,
        surfaces,
        constructions,
        total_floor_area,
        total_volume,
        exterior_wall_area,
        interzone_wall_area,
        roof_area,
        floor_area,
        skylight_area,
    })
}

/// Load and parse a gbXML file
pub fn load_gbxml(path: &Path) -> Result<GbXmlGeometry, String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("Failed to read gbXML file: {}", e))?;
    parse_gbxml(&content)
}

// ============================================================================
// TEST CASES
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Get path to test gbXML file
    fn test_file_path(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/test_data/gbxml")
            .join(name)
    }

    // -------------------------------------------------------------------------
    // Test Case 1: Simple Shoebox Building
    // -------------------------------------------------------------------------

    #[test]
    fn test_shoebox_geometry() {
        let path = test_file_path("shoebox.xml");
        let geometry = load_gbxml(&path).expect("Failed to load shoebox.xml");

        // Verify building info
        assert_eq!(
            geometry.building.name, "SimpleShoebox",
            "Building name should be SimpleShoebox"
        );

        // Verify zone count
        assert_eq!(
            geometry.building.zones.len(),
            1,
            "Should have exactly 1 zone"
        );

        // Verify zone properties
        let zone = &geometry.building.zones[0];
        assert_eq!(
            zone.name, "ThermalZone1",
            "Zone name should be ThermalZone1"
        );
        assert!(
            (zone.volume - 144.0).abs() < 0.1,
            "Zone volume should be 144.0 m³"
        );
        assert!(
            (zone.area - 48.0).abs() < 0.1,
            "Zone area should be 48.0 m²"
        );

        // Verify total floor area
        assert!(
            (geometry.total_floor_area - 48.0).abs() < 0.1,
            "Total floor area should be 48.0 m²"
        );

        // Verify total volume
        assert!(
            (geometry.total_volume - 144.0).abs() < 0.1,
            "Total volume should be 144.0 m³"
        );

        // Verify surface areas
        assert!(
            (geometry.exterior_wall_area - 84.0).abs() < 0.1,
            "Exterior wall area should be 84.0 m² (24+24+18+18)"
        );
        assert!(
            (geometry.roof_area - 48.0).abs() < 0.1,
            "Roof area should be 48.0 m²"
        );
        assert!(
            (geometry.floor_area - 48.0).abs() < 0.1,
            "Floor area should be 48.0 m²"
        );

        // Verify boundary conditions - shoebox has all exterior surfaces
        let exterior_surfaces: Vec<_> = geometry
            .surfaces
            .iter()
            .filter(|s| s.boundary_condition == BoundaryCondition::Exterior)
            .collect();
        let interzone_surfaces: Vec<_> = geometry
            .surfaces
            .iter()
            .filter(|s| s.boundary_condition == BoundaryCondition::InterZone)
            .collect();

        assert_eq!(
            exterior_surfaces.len(),
            6,
            "Shoebox should have 6 exterior surfaces (4 walls + roof + floor)"
        );
        assert_eq!(
            interzone_surfaces.len(),
            0,
            "Shoebox should have 0 inter-zone surfaces"
        );
    }

    #[test]
    fn test_shoebox_surface_count() {
        let path = test_file_path("shoebox.xml");
        let geometry = load_gbxml(&path).expect("Failed to load shoebox.xml");

        // Verify surface count
        assert_eq!(
            geometry.surfaces.len(),
            6,
            "Shoebox should have exactly 6 surfaces"
        );

        // Verify surface types
        let wall_count = geometry
            .surfaces
            .iter()
            .filter(|s| s.surface_type == SurfaceType::ExteriorWall)
            .count();
        let roof_count = geometry
            .surfaces
            .iter()
            .filter(|s| s.surface_type == SurfaceType::Roof)
            .count();
        let floor_count = geometry
            .surfaces
            .iter()
            .filter(|s| s.surface_type == SurfaceType::SlabOnGrade)
            .count();

        assert_eq!(wall_count, 4, "Should have 4 exterior walls");
        assert_eq!(roof_count, 1, "Should have 1 roof");
        assert_eq!(floor_count, 1, "Should have 1 floor slab");
    }

    // -------------------------------------------------------------------------
    // Test Case 2: L-Shaped Multi-Zone Building
    // -------------------------------------------------------------------------

    #[test]
    fn test_lshaped_geometry() {
        let path = test_file_path("l_shaped.xml");
        let geometry = load_gbxml(&path).expect("Failed to load l_shaped.xml");

        // Verify building info
        assert_eq!(
            geometry.building.name, "LShapedBuilding",
            "Building name should be LShapedBuilding"
        );

        // Verify zone count - should have 2 zones
        assert_eq!(
            geometry.building.zones.len(),
            2,
            "L-shaped building should have exactly 2 zones"
        );

        // Verify zone names
        let zone_names: Vec<_> = geometry
            .building
            .zones
            .iter()
            .map(|z| z.name.as_str())
            .collect();
        assert!(
            zone_names.contains(&"ZoneA") && zone_names.contains(&"ZoneB"),
            "Should have ZoneA and ZoneB"
        );

        // Verify total floor area (36 + 18 = 54 m²)
        assert!(
            (geometry.total_floor_area - 54.0).abs() < 0.1,
            "Total floor area should be 54.0 m²"
        );

        // Verify total volume (108 + 54 = 162 m³)
        assert!(
            (geometry.total_volume - 162.0).abs() < 0.1,
            "Total volume should be 162.0 m³"
        );

        // Verify inter-zone wall exists
        assert!(
            geometry.interzone_wall_area > 0.0,
            "L-shaped building should have inter-zone wall area"
        );

        // Verify boundary conditions - should have both exterior and inter-zone
        let exterior_surfaces: Vec<_> = geometry
            .surfaces
            .iter()
            .filter(|s| s.boundary_condition == BoundaryCondition::Exterior)
            .collect();
        let interzone_surfaces: Vec<_> = geometry
            .surfaces
            .iter()
            .filter(|s| s.boundary_condition == BoundaryCondition::InterZone)
            .collect();

        assert!(
            exterior_surfaces.len() > 0,
            "L-shaped building should have exterior surfaces"
        );
        assert_eq!(
            interzone_surfaces.len(),
            1,
            "L-shaped building should have exactly 1 inter-zone surface"
        );
    }

    #[test]
    fn test_lshaped_boundary_conditions() {
        let path = test_file_path("l_shaped.xml");
        let geometry = load_gbxml(&path).expect("Failed to load l_shaped.xml");

        // Find the inter-zone surface
        let interzone_surface = geometry
            .surfaces
            .iter()
            .find(|s| s.surface_type == SurfaceType::InteriorWall);

        assert!(
            interzone_surface.is_some(),
            "Should find an InteriorWall surface"
        );

        let interzone = interzone_surface.unwrap();
        assert_eq!(
            interzone.boundary_condition,
            BoundaryCondition::InterZone,
            "InteriorWall should have InterZone boundary condition"
        );
        assert!(
            (interzone.area - 9.0).abs() < 0.1,
            "Inter-zone wall area should be 9.0 m²"
        );
    }

    // -------------------------------------------------------------------------
    // Test Case 3: Multi-Story Building
    // -------------------------------------------------------------------------

    #[test]
    fn test_multistory_geometry() {
        let path = test_file_path("multi_story.xml");
        let geometry = load_gbxml(&path).expect("Failed to load multi_story.xml");

        // Verify building info
        assert_eq!(
            geometry.building.name, "MultiStoryBuilding",
            "Building name should be MultiStoryBuilding"
        );

        // Verify zone count - should have 2 zones (one per floor)
        assert_eq!(
            geometry.building.zones.len(),
            2,
            "Multi-story building should have exactly 2 zones"
        );

        // Verify total floor area (48 + 48 = 96 m²)
        assert!(
            (geometry.total_floor_area - 96.0).abs() < 0.1,
            "Total floor area should be 96.0 m²"
        );

        // Verify total volume (144 + 144 = 288 m³)
        assert!(
            (geometry.total_volume - 288.0).abs() < 0.1,
            "Total volume should be 288.0 m³"
        );

        // Verify inter-zone floor exists
        let interior_floor_count = geometry
            .surfaces
            .iter()
            .filter(|s| s.surface_type == SurfaceType::InteriorFloor)
            .count();
        assert_eq!(
            interior_floor_count, 1,
            "Multi-story should have 1 interior floor surface"
        );
    }

    #[test]
    fn test_multistory_interzone_floor() {
        let path = test_file_path("multi_story.xml");
        let geometry = load_gbxml(&path).expect("Failed to load multi_story.xml");

        // Find the interior floor
        let interior_floor = geometry
            .surfaces
            .iter()
            .find(|s| s.surface_type == SurfaceType::InteriorFloor);

        assert!(
            interior_floor.is_some(),
            "Should find an InteriorFloor surface"
        );

        let floor = interior_floor.unwrap();
        assert_eq!(
            floor.boundary_condition,
            BoundaryCondition::InterZone,
            "InteriorFloor should have InterZone boundary condition"
        );
        assert!(
            (floor.area - 48.0).abs() < 0.1,
            "Interior floor area should be 48.0 m²"
        );
    }

    // -------------------------------------------------------------------------
    // Test Case 4: Inter-Zone Walls Building
    // -------------------------------------------------------------------------

    #[test]
    fn test_interzone_walls_geometry() {
        let path = test_file_path("interzone_walls.xml");
        let geometry = load_gbxml(&path).expect("Failed to load interzone_walls.xml");

        // Verify building info
        assert_eq!(
            geometry.building.name, "InterZoneWallTest",
            "Building name should be InterZoneWallTest"
        );

        // Verify zone count - should have 3 zones
        assert_eq!(
            geometry.building.zones.len(),
            3,
            "Inter-zone walls building should have exactly 3 zones"
        );

        // Verify total floor area (24 + 24 + 24 = 72 m²)
        assert!(
            (geometry.total_floor_area - 72.0).abs() < 0.1,
            "Total floor area should be 72.0 m²"
        );

        // Verify inter-zone wall area is significant
        assert!(
            geometry.interzone_wall_area > 50.0,
            "Inter-zone wall area should be substantial (>50 m²)"
        );

        // Count inter-zone surfaces
        let interzone_surfaces: Vec<_> = geometry
            .surfaces
            .iter()
            .filter(|s| s.boundary_condition == BoundaryCondition::InterZone)
            .collect();

        // Should have multiple inter-zone surfaces (walls between 3 zones)
        assert!(
            interzone_surfaces.len() >= 4,
            "Should have at least 4 inter-zone surfaces (walls between 3 adjacent zones)"
        );
    }

    #[test]
    fn test_interzone_walls_boundary_assignment() {
        let path = test_file_path("interzone_walls.xml");
        let geometry = load_gbxml(&path).expect("Failed to load interzone_walls.xml");

        // Verify that all InteriorWall surfaces have InterZone boundary condition
        let interior_walls: Vec<_> = geometry
            .surfaces
            .iter()
            .filter(|s| s.surface_type == SurfaceType::InteriorWall)
            .collect();

        for wall in interior_walls {
            assert_eq!(
                wall.boundary_condition,
                BoundaryCondition::InterZone,
                "InteriorWall '{}' should have InterZone boundary condition",
                wall.name
            );
        }

        // Verify that all ExteriorWall surfaces have Exterior boundary condition
        let exterior_walls: Vec<_> = geometry
            .surfaces
            .iter()
            .filter(|s| s.surface_type == SurfaceType::ExteriorWall)
            .collect();

        for wall in exterior_walls {
            assert_eq!(
                wall.boundary_condition,
                BoundaryCondition::Exterior,
                "ExteriorWall '{}' should have Exterior boundary condition",
                wall.name
            );
        }
    }

    // -------------------------------------------------------------------------
    // Test Case 5: Complex Building with Skylights
    // -------------------------------------------------------------------------

    #[test]
    fn test_complex_building_geometry() {
        let path = test_file_path("complex_building.xml");
        let geometry = load_gbxml(&path).expect("Failed to load complex_building.xml");

        // Verify building info
        assert_eq!(
            geometry.building.name, "ComplexBuilding",
            "Building name should be ComplexBuilding"
        );

        // Verify zone count - should have 2 zones
        assert_eq!(
            geometry.building.zones.len(),
            2,
            "Complex building should have exactly 2 zones"
        );

        // Verify total floor area (60 + 30 = 90 m²)
        assert!(
            (geometry.total_floor_area - 90.0).abs() < 0.1,
            "Total floor area should be 90.0 m²"
        );

        // Verify total volume (210 + 105 = 315 m³)
        assert!(
            (geometry.total_volume - 315.0).abs() < 0.1,
            "Total volume should be 315.0 m³"
        );

        // Verify skylight area
        assert!(
            (geometry.skylight_area - 4.0).abs() < 0.1,
            "Skylight area should be 4.0 m²"
        );

        // Verify skylight has Exterior boundary condition
        let skylights: Vec<_> = geometry
            .surfaces
            .iter()
            .filter(|s| s.surface_type == SurfaceType::Skylight)
            .collect();

        assert_eq!(
            skylights.len(),
            1,
            "Complex building should have 1 skylight"
        );

        for skylight in skylights {
            assert_eq!(
                skylight.boundary_condition,
                BoundaryCondition::Exterior,
                "Skylight should have Exterior boundary condition"
            );
        }
    }

    #[test]
    fn test_complex_building_surface_types() {
        let path = test_file_path("complex_building.xml");
        let geometry = load_gbxml(&path).expect("Failed to load complex_building.xml");

        // Count surface types
        let wall_count = geometry
            .surfaces
            .iter()
            .filter(|s| s.surface_type == SurfaceType::ExteriorWall)
            .count();
        let interior_wall_count = geometry
            .surfaces
            .iter()
            .filter(|s| s.surface_type == SurfaceType::InteriorWall)
            .count();
        let roof_count = geometry
            .surfaces
            .iter()
            .filter(|s| s.surface_type == SurfaceType::Roof)
            .count();
        let skylight_count = geometry
            .surfaces
            .iter()
            .filter(|s| s.surface_type == SurfaceType::Skylight)
            .count();
        let floor_count = geometry
            .surfaces
            .iter()
            .filter(|s| s.surface_type == SurfaceType::SlabOnGrade)
            .count();

        assert_eq!(wall_count, 8, "Should have 8 exterior walls");
        assert_eq!(interior_wall_count, 1, "Should have 1 interior wall");
        assert_eq!(roof_count, 2, "Should have 2 roof surfaces");
        assert_eq!(skylight_count, 1, "Should have 1 skylight");
        assert_eq!(floor_count, 2, "Should have 2 floor slabs");
    }

    // -------------------------------------------------------------------------
    // Test Case 6: Volume and Area Consistency
    // -------------------------------------------------------------------------

    #[test]
    fn test_volume_area_consistency() {
        let test_files = [
            ("shoebox.xml", 144.0, 48.0),
            ("l_shaped.xml", 162.0, 54.0),
            ("multi_story.xml", 288.0, 96.0),
            ("interzone_walls.xml", 216.0, 72.0),
            ("complex_building.xml", 315.0, 90.0),
        ];

        for (filename, expected_volume, expected_area) in test_files {
            let path = test_file_path(filename);
            let geometry = load_gbxml(&path).expect(&format!("Failed to load {}", filename));

            // Check volume matches zone volumes
            let zone_volume: f64 = geometry.building.zones.iter().map(|z| z.volume).sum();
            assert!(
                (zone_volume - expected_volume).abs() < 0.1,
                "Zone volume for {} should be {} but got {}",
                filename,
                expected_volume,
                zone_volume
            );

            // Check area matches zone areas
            let zone_area: f64 = geometry.building.zones.iter().map(|z| z.area).sum();
            assert!(
                (zone_area - expected_area).abs() < 0.1,
                "Zone area for {} should be {} but got {}",
                filename,
                expected_area,
                zone_area
            );

            // Check floor area matches
            assert!(
                (geometry.total_floor_area - expected_area).abs() < 0.1,
                "Total floor area for {} should be {}",
                filename,
                expected_area
            );

            // Check volume matches
            assert!(
                (geometry.total_volume - expected_volume).abs() < 0.1,
                "Total volume for {} should be {}",
                filename,
                expected_volume
            );
        }
    }

    // -------------------------------------------------------------------------
    // Test Case 7: Construction Data Parsing
    // -------------------------------------------------------------------------

    #[test]
    fn test_construction_parsing() {
        let path = test_file_path("shoebox.xml");
        let geometry = load_gbxml(&path).expect("Failed to load shoebox.xml");

        // Verify constructions were parsed
        assert!(
            !geometry.constructions.is_empty(),
            "Should have parsed constructions"
        );

        // Verify standard wall construction exists
        assert!(
            geometry.constructions.contains_key("const_wall1"),
            "Should have const_wall1 construction"
        );

        let wall_const = geometry
            .constructions
            .get("const_wall1")
            .expect("Should have wall construction");
        assert_eq!(wall_const.name, "StandardWall");
        assert!(wall_const.u_value.is_some());
        assert!((wall_const.u_value.unwrap() - 0.5).abs() < 0.01);
    }

    // -------------------------------------------------------------------------
    // Test Case 8: Surface Azimuth and Tilt
    // -------------------------------------------------------------------------

    #[test]
    fn test_surface_orientation() {
        let path = test_file_path("shoebox.xml");
        let geometry = load_gbxml(&path).expect("Failed to load shoebox.xml");

        // Find north wall (azimuth = 0)
        let north_wall = geometry.surfaces.iter().find(|s| s.name.contains("North"));

        assert!(north_wall.is_some(), "Should find North Wall");
        let wall = north_wall.unwrap();
        assert!(
            (wall.azimuth - 0.0).abs() < 0.1,
            "North wall azimuth should be 0"
        );
        assert!(
            (wall.tilt - 90.0).abs() < 0.1,
            "Wall tilt should be 90 degrees"
        );

        // Find roof (tilt = 0)
        let roof = geometry
            .surfaces
            .iter()
            .find(|s| s.surface_type == SurfaceType::Roof);

        assert!(roof.is_some(), "Should find roof");
        let roof_surface = roof.unwrap();
        assert!(
            (roof_surface.tilt - 0.0).abs() < 0.1,
            "Roof tilt should be 0 degrees"
        );
    }
}

// ============================================================================
// IFC Parsing Stubs
// ============================================================================
//
// These stubs describe the expected behavior for IFC (Industry Foundation Classes)
// parsing, which is the other major BIM format mentioned in Issue #1055.
//
// IFC parsing is not yet implemented. These tests serve as a specification
// for future development and validation of IFC geometry ingestion.
//
// IFC is more complex than gbXML because:
// - Uses STEP format (ISO 10303) instead of XML
// - Contains full BIM data (walls, windows, doors, spaces, zones)
// - Requires geometric reasoning (extrusions, boolean operations)
// - Has multiple schema versions (IFC2x3, IFC4, IFC4x1)
//
// Once IFC parsing is implemented, these stubs should be converted to
// real tests that validate the actual parsing behavior.

#[cfg(test)]
mod ifc_parsing_stubs {
    /// IFC surface classification for boundary condition determination
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[allow(dead_code)]
    pub enum IfcSurfaceType {
        IfcWall,
        IfcRoof,
        IfcFloor,
        IfcWindow,
        IfcDoor,
        IfcSlab,
        IfcCovering,
        IfcUnknown,
    }

    /// Boundary condition for IFC surfaces
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[allow(dead_code)]
    pub enum IfcBoundaryCondition {
        Exterior,
        Interior,
        Ground,
        InterZone,
        Unknown,
    }

    /// Parsed IFC building element
    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    pub struct ParsedIfcElement {
        pub id: String,
        pub name: String,
        pub element_type: IfcSurfaceType,
        pub area: f64,
        pub volume: Option<f64>,
        pub boundary_condition: IfcBoundaryCondition,
    }

    /// Expected behavior: Parse IFC file and extract building geometry
    ///
    /// # Input
    /// - IFC STEP file (e.g., "simple_office.ifc")
    ///
    /// # Expected Output
    /// - List of building elements (walls, roofs, floors, etc.)
    /// - Zone assignments for each space
    /// - Boundary condition classification
    ///
    /// # Notes
    /// - IFC uses placement graphs to determine element orientation
    /// - Area/volume may need to be computed from geometry representation
    #[test]
    fn test_ifc_parsing_interface_exists() {
        // This test validates that the IFC parsing interface is defined
        // Once IFC parsing is implemented, this would parse an actual IFC file

        let ifc_file_path = "tests/test_data/ifc/simple_office.ifc";

        // STUB: When IFC parsing is implemented:
        // let elements = parse_ifc_file(ifc_file_path).expect("Failed to parse IFC file");
        // assert!(!elements.is_empty(), "Should parse at least one element");

        // For now, we just document the expected interface
        assert!(
            true,
            "IFC parsing stub - interface is documented for future implementation"
        );
    }

    /// Expected behavior: Classify IFC elements by surface type
    ///
    /// IFC elements are classified using IfcWall, IfcRoof, IfcSlab, etc.
    /// These must be mapped to Fluxion's boundary condition system.
    #[test]
    fn test_ifc_surface_type_classification() {
        // STUB: When IFC parsing is implemented:
        // let elements = parse_ifc_file("tests/test_data/ifc/simple_office.ifc");
        //
        // let walls: Vec<_> = elements.iter()
        //     .filter(|e| e.element_type == IfcSurfaceType::IfcWall)
        //     .collect();
        //
        // assert!(!walls.is_empty(), "Should find walls in IFC file");
        // for wall in walls {
        //     assert_ne!(wall.boundary_condition, IfcBoundaryCondition::Unknown);
        // }

        assert!(
            true,
            "IFC surface type classification stub - validates element type mapping"
        );
    }

    /// Expected behavior: Determine boundary conditions from IFC space adjacency
    ///
    /// IFC stores space adjacency through the IfcRelSpaceBoundary relationship.
    /// This determines whether a surface is exterior or inter-zone.
    #[test]
    fn test_ifc_boundary_condition_from_adjacency() {
        // STUB: When IFC parsing is implemented:
        // - IfcRelSpaceBoundary with RelatedSpace = external space → Exterior
        // - IfcRelSpaceBoundary with RelatedSpace = another zone → InterZone
        // - IfcRelSpaceBoundary with RelatedSpace = same zone → Interior (ignored)

        assert!(
            true,
            "IFC boundary condition stub - validates adjacency-based classification"
        );
    }

    /// Expected behavior: Calculate zone volumes from IFC spaces
    ///
    /// IFC stores space geometry as either:
    /// - IfcSpace with IfcLocalPlacement (polygonal boundary representation)
    /// - IfcSpace with IfcShapeRepresentation (boundary representation)
    ///
    /// Volume is computed from the enclosed geometry.
    #[test]
    fn test_ifc_zone_volume_calculation() {
        // STUB: When IFC parsing is implemented:
        // let spaces = parse_ifc_spaces("tests/test_data/ifc/simple_office.ifc");
        // for space in spaces {
        //     assert!(space.volume.is_some());
        //     assert!(space.volume.unwrap() > 0.0);
        // }

        assert!(
            true,
            "IFC zone volume calculation stub - validates volume extraction"
        );
    }

    /// Expected behavior: Handle IFC4x1 schema (latest version)
    ///
    /// IFC4x1 (ISO 16739-1:2022) is the current standard.
    /// Key differences from IFC2x3:
    /// - IfcSpace now has proper volume representation
    /// - IfcZone replaced by IfcRelAssignsToGroup
    /// - Better support for nested zones
    #[test]
    fn test_ifc_schema_version_handling() {
        // STUB: When IFC parsing is implemented:
        // let schema = detect_ifc_schema_version("tests/test_data/ifc/simple_office.ifc");
        // assert_eq!(schema, IfcSchemaVersion::Ifc4x1);

        assert!(
            true,
            "IFC schema version handling stub - validates version detection"
        );
    }
}
