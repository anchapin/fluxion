#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod geometry;

use tracing::info;
use tracing_subscriber::EnvFilter;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("Starting Fluxion Tauri application");

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            commands::load_geometry_file,
            commands::get_sample_geometry,
            commands::update_simulation_parameters,
            commands::get_simulation_parameters,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
