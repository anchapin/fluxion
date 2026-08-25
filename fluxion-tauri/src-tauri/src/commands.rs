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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParameters {
    pub zone_id: Option<String>,
    pub heating_setpoint: Option<f64>,
    pub cooling_setpoint: Option<f64>,
    pub lighting_load: Option<f64>,
    pub equipment_load: Option<f64>,
    pub occupancy: Option<f64>,
    pub ventilation_rate: Option<f64>,
    pub wall_u_value: Option<f64>,
    pub roof_u_value: Option<f64>,
}

static SIM_PARAMS: std::sync::RwLock<SimulationParameters> = std::sync::RwLock::new(SimulationParameters {
    zone_id: None,
    heating_setpoint: Some(20.0),
    cooling_setpoint: Some(26.0),
    lighting_load: Some(5.0),
    equipment_load: Some(10.0),
    occupancy: Some(0.1),
    ventilation_rate: Some(0.5),
    wall_u_value: Some(0.5),
    roof_u_value: Some(0.3),
});

#[tauri::command]
pub fn get_simulation_parameters() -> Result<SimulationParameters, String> {
    let params = SIM_PARAMS.read().map_err(|e| e.to_string())?;
    Ok(params.clone())
}

#[tauri::command]
pub fn update_simulation_parameters(params: SimulationParameters) -> Result<SimulationParameters, String> {
    let mut current = SIM_PARAMS.write().map_err(|e| e.to_string())?;
    if let Some(zone_id) = params.zone_id {
        current.zone_id = Some(zone_id);
    }
    if let Some(v) = params.heating_setpoint { current.heating_setpoint = Some(v); }
    if let Some(v) = params.cooling_setpoint { current.cooling_setpoint = Some(v); }
    if let Some(v) = params.lighting_load { current.lighting_load = Some(v); }
    if let Some(v) = params.equipment_load { current.equipment_load = Some(v); }
    if let Some(v) = params.occupancy { current.occupancy = Some(v); }
    if let Some(v) = params.ventilation_rate { current.ventilation_rate = Some(v); }
    if let Some(v) = params.wall_u_value { current.wall_u_value = Some(v); }
    if let Some(v) = params.roof_u_value { current.roof_u_value = Some(v); }
    Ok(current.clone())
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
