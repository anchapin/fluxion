use crate::geometry::{BuildingGeometry, BuildingLevel, ThermalZone};

#[derive(Debug, serde::Serialize)]
pub struct GeometrySummary {
    pub name: String,
    pub total_floor_area: f64,
    pub num_levels: usize,
    pub num_zones: usize,
    pub num_spaces: usize,
    pub num_surfaces: usize,
    pub bounding_box: BoundingBoxInfo,
}

#[derive(Debug, serde::Serialize)]
pub struct BoundingBoxInfo {
    pub width: f64,
    pub height: f64,
    pub depth: f64,
    pub center_x: f64,
    pub center_y: f64,
    pub center_z: f64,
}

#[tauri::command]
pub async fn load_geometry_file(file_path: String) -> Result<BuildingGeometry, String> {
    tracing::info!("Loading geometry file: {}", file_path);

    let extension = std::path::Path::new(&file_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let content = tokio::fs::read_to_string(&file_path)
        .await
        .map_err(|e| format!("Failed to read file: {}", e))?;

    match extension.as_str() {
        "xml" | "gbxml" => crate::geometry::parse_gbxml_content(&content),
        "ifc" => crate::geometry::parse_ifc_content(&content),
        "json" => serde_json::from_str(&content).map_err(|e| format!("JSON parse error: {}", e)),
        _ => {
            tracing::warn!("Unknown file extension '{}', returning sample geometry", extension);
            Ok(crate::geometry::create_sample_geometry())
        }
    }
}

#[tauri::command]
pub async fn get_geometry_summary(geometry: BuildingGeometry) -> Result<GeometrySummary, String> {
    let mut num_spaces = 0;
    let mut num_surfaces = 0;

    for level in &geometry.levels {
        num_spaces += level.spaces.len();
        for space in &level.spaces {
            num_surfaces += space.surfaces.len();
        }
    }

    let size = geometry.bounding_box.size();
    let center = geometry.bounding_box.center();

    Ok(GeometrySummary {
        name: geometry.name.clone(),
        total_floor_area: geometry.total_floor_area,
        num_levels: geometry.levels.len(),
        num_zones: geometry.zones.len(),
        num_spaces,
        num_surfaces,
        bounding_box: BoundingBoxInfo {
            width: size.x,
            height: size.z,
            depth: size.y,
            center_x: center.x,
            center_y: center.y,
            center_z: center.z,
        },
    })
}

#[tauri::command]
pub fn get_zone_ids(geometry: BuildingGeometry) -> Vec<String> {
    geometry.zones.iter().map(|z| z.id.clone()).collect()
}

#[tauri::command]
pub fn get_level_ids(geometry: BuildingGeometry) -> Vec<String> {
    geometry.levels.iter().map(|l| l.id.clone()).collect()
}
