use crate::geometry::{BuildingGeometry, BuildingLevel, Space, Vertex};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometrySummary {
    pub building_id: String,
    pub building_name: String,
    pub level_count: usize,
    pub space_count: usize,
    pub zone_count: usize,
    pub total_floor_area: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryToZoneMapping {
    pub space_id: String,
    pub zone_id: Option<String>,
    pub zone_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneGeometryInfo {
    pub zone_id: String,
    pub zone_name: String,
    pub space_ids: Vec<String>,
    pub bounding_box: crate::geometry::BoundingBox,
    pub total_area: f64,
}

fn compute_geometry_summary(geometry: &BuildingGeometry) -> GeometrySummary {
    let space_count = geometry.levels.iter().map(|l| l.spaces.len()).sum();
    let total_floor_area: f64 = geometry
        .levels
        .iter()
        .flat_map(|l| &l.spaces)
        .map(|s| {
            s.surfaces
                .iter()
                .find(|sf| sf.surface_type == "Floor")
                .map(|sf| sf.area)
                .unwrap_or(0.0)
        })
        .sum();

    GeometrySummary {
        building_id: geometry.id.clone(),
        building_name: geometry.name.clone(),
        level_count: geometry.levels.len(),
        space_count,
        zone_count: geometry.zones.len(),
        total_floor_area,
    }
}

#[tauri::command]
pub fn load_geometry() -> Result<BuildingGeometry, String> {
    Ok(BuildingGeometry::sample())
}

#[tauri::command]
pub fn get_geometry_summary(geometry: BuildingGeometry) -> Result<GeometrySummary, String> {
    Ok(compute_geometry_summary(&geometry))
}

#[tauri::command]
pub fn get_geometry_to_zone_mapping(
    geometry: BuildingGeometry,
) -> Result<Vec<GeometryToZoneMapping>, String> {
    let mut mappings = Vec::new();

    for level in &geometry.levels {
        for space in &level.spaces {
            let zone_info = geometry
                .zones
                .iter()
                .find(|z| z.space_ids.contains(&space.id));
            mappings.push(GeometryToZoneMapping {
                space_id: space.id.clone(),
                zone_id: space.zone_id.clone(),
                zone_name: zone_info.map(|z| z.name.clone()),
            });
        }
    }

    Ok(mappings)
}

#[tauri::command]
pub fn get_zone_geometry_info(
    geometry: BuildingGeometry,
    zone_id: String,
) -> Result<Option<ZoneGeometryInfo>, String> {
    let zone = geometry.zones.iter().find(|z| z.id == zone_id);

    match zone {
        Some(z) => {
            let spaces: Vec<Space> = geometry
                .levels
                .iter()
                .flat_map(|l| &l.spaces)
                .filter(|s| z.space_ids.contains(&s.id))
                .cloned()
                .collect();

            let all_vertices: Vec<&Vertex> = spaces
                .iter()
                .flat_map(|s| &s.surfaces)
                .flat_map(|sf| &sf.vertices)
                .collect();

            let (min_x, max_x) = all_vertices
                .iter()
                .fold((f64::MAX, f64::MIN), |(mn, mx), v| {
                    (mn.min(v.x), mx.max(v.x))
                });
            let (min_y, max_y) = all_vertices
                .iter()
                .fold((f64::MAX, f64::MIN), |(mn, mx), v| {
                    (mn.min(v.y), mx.max(v.y))
                });
            let (min_z, max_z) = all_vertices
                .iter()
                .fold((f64::MAX, f64::MIN), |(mn, mx), v| {
                    (mn.min(v.z), mx.max(v.z))
                });

            let total_area: f64 = spaces
                .iter()
                .flat_map(|s| &s.surfaces)
                .filter(|sf| sf.surface_type == "Floor")
                .map(|sf| sf.area)
                .sum();

            Ok(Some(ZoneGeometryInfo {
                zone_id: z.id.clone(),
                zone_name: z.name.clone(),
                space_ids: z.space_ids.clone(),
                bounding_box: crate::geometry::BoundingBox {
                    min: Vertex {
                        x: min_x,
                        y: min_y,
                        z: min_z,
                    },
                    max: Vertex {
                        x: max_x,
                        y: max_y,
                        z: max_z,
                    },
                },
                total_area,
            }))
        }
        None => Ok(None),
    }
}

#[tauri::command]
pub fn get_building_levels(geometry: BuildingGeometry) -> Result<Vec<BuildingLevel>, String> {
    Ok(geometry.levels.clone())
}
