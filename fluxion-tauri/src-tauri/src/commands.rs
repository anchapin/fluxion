use crate::geometry::BuildingGeometry;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParameters {
    pub heating_setpoint: f64,
    pub cooling_setpoint: f64,
    pub lighting_load: f64,
    pub equipment_load: f64,
    pub occupancy: f64,
    pub ventilation_rate: f64,
    pub wall_u_value: f64,
    pub roof_u_value: f64,
}

static SIM_PARAMS: std::sync::OnceLock<SimulationParameters> = std::sync::OnceLock::new();

fn get_sim_params() -> &'static SimulationParameters {
    SIM_PARAMS.get_or_init(|| SimulationParameters {
        heating_setpoint: 20.0,
        cooling_setpoint: 26.0,
        lighting_load: 12.0,
        equipment_load: 15.0,
        occupancy: 0.1,
        ventilation_rate: 0.5,
        wall_u_value: 0.5,
        roof_u_value: 0.3,
    })
}

#[tauri::command]
pub async fn update_simulation_parameters(
    params: SimulationParameters,
) -> Result<SimulationParameters, String> {
    tracing::info!(
        "Updating simulation parameters: heating={}, cooling={}, lighting={}, equipment={}, occupancy={}, ventilation={}, wall_u={}, roof_u={}",
        params.heating_setpoint,
        params.cooling_setpoint,
        params.lighting_load,
        params.equipment_load,
        params.occupancy,
        params.ventilation_rate,
        params.wall_u_value,
        params.roof_u_value,
    );
    SIM_PARAMS
        .set(params.clone())
        .map_err(|_| "Parameters already initialized".to_string())?;
    Ok(params)
}

#[tauri::command]
pub async fn get_simulation_parameters() -> Result<SimulationParameters, String> {
    Ok(get_sim_params().clone())
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
            tracing::warn!(
                "Unknown file extension '{}', returning sample geometry",
                extension
            );
            Ok(crate::geometry::create_sample_geometry())
        }
    }
}

#[tauri::command]
pub async fn get_sample_geometry() -> Result<BuildingGeometry, String> {
    Ok(crate::geometry::create_sample_geometry())
}
