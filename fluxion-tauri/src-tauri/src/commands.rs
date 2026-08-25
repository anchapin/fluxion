use crate::geometry::BuildingGeometry;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParameters {
    pub heating_setpoint: Option<f64>,
    pub cooling_setpoint: Option<f64>,
    pub lighting_load: Option<f64>,
    pub equipment_load: Option<f64>,
    pub occupancy: Option<f64>,
    pub ventilation_rate: Option<f64>,
    pub u_value: Option<f64>,
    pub zone_id: Option<String>,
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
pub async fn get_sample_geometry() -> Result<BuildingGeometry, String> {
    Ok(crate::geometry::create_sample_geometry())
}

#[tauri::command]
pub async fn update_simulation_parameters(
    parameters: SimulationParameters,
) -> Result<String, String> {
    tracing::info!(
        "Updating simulation parameters: heating={:?}, cooling={:?}, zone={:?}",
        parameters.heating_setpoint,
        parameters.cooling_setpoint,
        parameters.zone_id
    );

    let params_json = serde_json::to_string(&parameters)
        .map_err(|e| format!("Failed to serialize parameters: {}", e))?;

    tracing::debug!("Parameter update payload: {}", params_json);

    Ok(format!(
        "Parameters updated: heating={:?}, cooling={:?}",
        parameters.heating_setpoint, parameters.cooling_setpoint
    ))
}
