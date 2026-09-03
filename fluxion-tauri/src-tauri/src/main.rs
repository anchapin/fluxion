#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod geometry;

use commands::{
    get_building_levels, get_geometry_summary, get_geometry_to_zone_mapping,
    get_simulation_parameters, get_zone_geometry_info, load_geometry, update_simulation_parameters,
};
use log::info;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!("Starting Fluxion Tauri application");

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            load_geometry,
            get_geometry_summary,
            get_geometry_to_zone_mapping,
            get_zone_geometry_info,
            get_building_levels,
            get_simulation_parameters,
            update_simulation_parameters,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
