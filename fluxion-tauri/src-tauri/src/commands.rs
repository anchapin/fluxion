use crate::geometry::BuildingGeometry;

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
