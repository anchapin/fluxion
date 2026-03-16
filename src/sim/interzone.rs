//! Inter-zone heat transfer modules for multi-zone building energy modeling.
//!
//! This module provides tools for calculating heat transfer between different thermal zones,
//! including conductive coupling through common walls, radiative exchange between surfaces,
//! and temperature-dependent air exchange through door openings (stack effect).

use crate::sim::construction::Construction;

/// Stack effect coefficient for buoyancy-driven ventilation.
/// Based on ASHRAE 140 natural ventilation model.
pub const STACK_COEFFICIENT: f64 = 0.025;

/// Air density at standard conditions (kg/m³).
pub const AIR_DENSITY: f64 = 1.2;

/// Air specific heat capacity (J/kg·K).
pub const AIR_SPECIFIC_HEAT: f64 = 1000.0;

/// Calculates directional inter-zone conductance accounting for asymmetric insulation.
///
/// For walls with insulation on one side only, heat flow differs in each direction.
/// Example: Insulation on zone 0 side reduces h_tr_iz_0_to_1 more than h_tr_iz_1_to_0.
///
/// # Arguments
/// * `common_wall_area` - Area of common wall between zones (m²)
/// * `construction` - Base wall construction (concrete, brick, etc.)
/// * `insulation_r_side_a` - Additional insulation R-value on side A (m²K/W)
/// * `insulation_r_side_b` - Additional insulation R-value on side B (m²K/W)
///
/// # Returns
/// Tuple (h_a_to_b, h_b_to_a) - Directional conductances (W/K)
///
/// # Formula
/// h_a_to_b = A_common / (R_base + R_insulation_a)
/// h_b_to_a = A_common / (R_base + R_insulation_b)
///
/// # Note
/// Uses materials-only R-value (excludes film coefficients) since both surfaces
/// are interior for inter-zone walls.
pub fn calculate_directional_interzone_conductance(
    common_wall_area: f64,
    construction: &Construction,
    insulation_r_side_a: f64,
    insulation_r_side_b: f64,
) -> (f64, f64) {
    let base_r = construction.r_value_materials();

    let h_a_to_b = common_wall_area / (base_r + insulation_r_side_a);
    let h_b_to_a = common_wall_area / (base_r + insulation_r_side_b);

    (h_a_to_b, h_b_to_a)
}

/// Calculates inter-zone conductance from first principles (no directionality).
///
/// For symmetric insulation or no additional insulation, use this simpler form.
/// h_tr_iz = A_common / R_common_wall
///
/// # Arguments
/// * `common_wall_area` - Area of common wall (m²)
/// * `construction` - Wall construction (layers with R-values)
///
/// # Returns
/// Conductance h_tr_iz (W/K)
///
/// # Note
/// Uses materials-only R-value (excludes film coefficients) since both surfaces
/// are interior for inter-zone walls.
pub fn calculate_interzone_conductance(common_wall_area: f64, construction: &Construction) -> f64 {
    let r_value = construction.r_value_materials();
    common_wall_area / r_value
}

/// Calculates the radiative view factor between two surfaces.
///
/// For now, implements a simplified analytical solution for two rectangular zones
/// separated by a common wall.
#[allow(dead_code)]
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

/// Calculates air exchange rate (ACH) using stack effect for temperature-dependent ventilation.
///
/// Stack effect ventilation is driven by thermal buoyancy: warm air rises through door openings,
/// creating natural airflow between zones at different temperatures.
///
/// # Arguments
/// * `temp_a` - Temperature in zone A (°C)
/// * `temp_b` - Temperature in zone B (°C)
/// * `door_height` - Door opening height (m)
/// * `door_area` - Door opening area (m²)
///
/// # Returns
/// Air exchange rate ACH (air changes per hour, 1/hr)
///
/// # Formula
/// Q_vent = C·A·√(ΔT/h)  (volumetric flow rate, m³/hr)
/// ACH = Q_vent / V_zone (air changes per hour)
///
/// Where:
/// - C = STACK_COEFFICIENT = 0.025 (empirical coefficient)
/// - A = door opening area (m²)
/// - ΔT = |T_A - T_B| (temperature difference, °C)
/// - h = door opening height (m)
/// - V_zone = zone volume (m³), approximated as door_area × door_height
///
/// # Note
/// This formula captures thermal buoyancy dynamics critical for sunspace buildings,
/// where temperature differences can be 20-40°C between zones.
pub fn calculate_stack_effect_ach(
    temp_a: f64,
    temp_b: f64,
    door_height: f64,
    door_area: f64,
) -> f64 {
    // Temperature difference (absolute value for magnitude)
    let delta_t = (temp_a - temp_b).abs();

    // Stack effect volumetric flow rate: Q = C·A·√(ΔT/h)
    let q_vent = STACK_COEFFICIENT * door_area * (delta_t / door_height).sqrt();

    // ACH = Q_vent / V_zone (assuming door height represents zone height)
    let zone_volume = door_area * door_height;
    q_vent / zone_volume // Units: 1/hr (if Q in m³/hr)
}

/// Calculates ventilation heat transfer using air enthalpy method.
///
/// # Arguments
/// * `ach` - Air exchange rate (1/hr)
/// * `temp_source` - Source zone temperature (°C)
/// * `temp_target` - Target zone temperature (°C)
/// * `volume_target` - Target zone volume (m³)
///
/// # Returns
/// Ventilation heat transfer Q_vent (Watts). Positive if heat flows from source to target.
///
/// # Formula
/// Q_vent = ρ·Cp·ACH·V·(T_source - T_target)
///
/// Where:
/// - ρ = AIR_DENSITY = 1.2 kg/m³ (air density)
/// - Cp = AIR_SPECIFIC_HEAT = 1000.0 J/kg·K (specific heat capacity)
/// - ACH = air exchange rate (1/hr)
/// - V = target zone volume (m³)
/// - T_source, T_target = zone temperatures (°C)
///
/// # Note
/// Air enthalpy method includes air density and specific heat for thermodynamic rigor.
/// Omitting ρ·Cp gives 1200× error in heat transfer calculation.
pub fn calculate_ventilation_heat_transfer(
    ach: f64,
    temp_source: f64,
    temp_target: f64,
    volume_target: f64,
) -> f64 {
    let delta_t = temp_source - temp_target;
    // Air enthalpy method: Q = ρ·Cp·ACH·V·(T_source - T_target)
    // Note: Units: (kg/m³)·(J/kg·K)·(1/hr)·(m³)·K = W/hr
    // Need to convert to Watts (divide by 3600 for hours to seconds)
    AIR_DENSITY * AIR_SPECIFIC_HEAT * ach * volume_target * delta_t / 3600.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::construction::Assemblies;

    #[test]
    fn test_interzone_conductance_case_960() {
        // Case 960: 0.200m concrete wall
        let wall = Assemblies::concrete_wall(0.200);
        let area = 21.6; // Common wall area

        // Expected: h = A / R = 21.6 / 0.177 = 122.0 W/K
        // (R = thickness / k = 0.200 / 1.13 = 0.177 m²K/W for concrete)
        let h = calculate_interzone_conductance(area, &wall);
        assert!((h - 122.0).abs() < 1.0, "Expected ~122.0 W/K, got {}", h);
    }

    #[test]
    fn test_directional_interzone_conductance_asymmetric() {
        let wall = Assemblies::concrete_wall(0.200);
        let area = 21.6;

        // Asymmetric insulation: R_side_a = 2.0, R_side_b = 0.0
        let (h_a_to_b, h_b_to_a) =
            calculate_directional_interzone_conductance(area, &wall, 2.0, 0.0);

        // Expected: h_a_to_b = 21.6 / (0.177 + 2.0) = 9.92 W/K
        //           h_b_to_a = 21.6 / (0.177 + 0.0) = 122.0 W/K
        assert!(
            (h_a_to_b - 9.92).abs() < 0.5,
            "Expected ~9.92 W/K, got {}",
            h_a_to_b
        );
        assert!(
            (h_b_to_a - 122.0).abs() < 1.0,
            "Expected ~122.0 W/K, got {}",
            h_b_to_a
        );
    }

    #[test]
    fn test_directional_interzone_conductance_symmetric() {
        let wall = Assemblies::concrete_wall(0.200);
        let area = 21.6;

        // Symmetric insulation: R_side_a = 2.0, R_side_b = 2.0
        let (h_a_to_b, h_b_to_a) =
            calculate_directional_interzone_conductance(area, &wall, 2.0, 2.0);

        // Expected: h_a_to_b = h_b_to_a = 21.6 / (0.177 + 2.0) = 9.92 W/K
        assert!(
            (h_a_to_b - 9.92).abs() < 0.5,
            "Expected ~9.92 W/K, got {}",
            h_a_to_b
        );
        assert!(
            (h_b_to_a - 9.92).abs() < 0.5,
            "Expected ~9.92 W/K, got {}",
            h_b_to_a
        );
        assert!(
            (h_a_to_b - h_b_to_a).abs() < 0.1,
            "Expected equal conductances"
        );
    }

    #[test]
    fn test_interzone_conductance_zero_insulation() {
        let wall = Assemblies::concrete_wall(0.200);
        let area = 21.6;

        // No additional insulation: R_side_a = 0.0, R_side_b = 0.0
        let (h_a_to_b, h_b_to_a) =
            calculate_directional_interzone_conductance(area, &wall, 0.0, 0.0);

        // Expected: h_a_to_b = h_b_to_a = 21.6 / 0.177 = 122.0 W/K
        assert!(
            (h_a_to_b - 122.0).abs() < 1.0,
            "Expected ~122.0 W/K, got {}",
            h_a_to_b
        );
        assert!(
            (h_b_to_a - 122.0).abs() < 1.0,
            "Expected ~122.0 W/K, got {}",
            h_b_to_a
        );
    }
}
