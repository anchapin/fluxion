//! Inter-zone heat transfer modules for multi-zone building energy modeling.
//!
//! This module provides tools for calculating heat transfer between different thermal zones,
//! including conductive coupling through common walls and radiative exchange between surfaces.

use crate::sim::construction::SurfaceType;

/// Calculates the radiative view factor between two surfaces.
/// 
/// For now, implements a simplified analytical solution for two rectangular zones 
/// separated by a common wall.
pub fn calculate_zone_to_zone_view_factor(
    common_wall_area: f64,
    total_area_zone_1: f64,
    total_area_zone_2: f64,
) -> f64 {
    // Simplified view factor based on area ratio
    // F_12 = (A_common / A_total_1) * (A_common / A_total_2)
    // This is a rough approximation for Case 960
    (common_wall_area / total_area_zone_1) * (common_wall_area / total_area_zone_2)
}

/// Calculates the radiative conductance between two zones.
///
/// # Arguments
/// * `area` - Shared area between zones (m²)
/// * `emissivity` - Surface emissivity (typically 0.9)
/// * `mean_temp_k` - Mean radiant temperature in Kelvin (typically 293.15 K)
/// * `view_factor` - Radiative view factor between the zones
pub fn calculate_radiative_conductance(
    area: f64,
    emissivity: f64,
    mean_temp_k: f64,
    view_factor: f64,
) -> f64 {
    const SIGMA: f64 = 5.670374419e-8; // Stefan-Boltzmann constant
    // linearized radiative exchange: h_rad = 4 * sigma * epsilon * F * T^3 * Area
    4.0 * SIGMA * emissivity * emissivity * view_factor * mean_temp_k.powi(3) * area
}

/// Calculates window-to-window radiative exchange conductance.
/// 
/// Specifically for Case 960 where the sunspace window exchanges radiation 
/// with the back-zone through the common glazing.
pub fn calculate_window_radiative_conductance(
    window_area: f64,
    glass_emissivity: f64,
    mean_temp_k: f64,
    view_factor: f64,
) -> f64 {
    calculate_radiative_conductance(window_area, glass_emissivity, mean_temp_k, view_factor)
}
