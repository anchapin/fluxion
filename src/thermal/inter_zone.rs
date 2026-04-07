//! Inter-zone conductance calculation module.
//!
//! This module provides functions for calculating heat transfer conductance
//! between thermal zones in multi-zone buildings.

/// Calculate inter-zone conductance between two zones.
///
/// # Arguments
/// * `common_wall_area` - Area of common wall between zones (m²)
/// * `wall_u_value` - U-value of the wall (W/m²·K)
///
/// # Returns
/// Conductance in W/K
///
/// # Formula
/// h_tr_iz = A_common × U_wall
pub fn calculate_inter_zone_conductance(common_wall_area: f64, wall_u_value: f64) -> f64 {
    common_wall_area * wall_u_value
}

/// Calculate heat flow between two zones using inter-zone conductance.
///
/// # Arguments
/// * `h_tr_ij` - Inter-zone conductance (W/K)
/// * `ti` - Temperature of zone i (°C)
/// * `tj` - Temperature of zone j (°C)
///
/// # Returns
/// Heat flow from zone i to zone j in Watts
///
/// # Note
/// Follows sign convention: Q_ij = -Q_ji
pub fn inter_zone_heat_flow(h_tr_ij: f64, ti: f64, tj: f64) -> f64 {
    h_tr_ij * (ti - tj)
}

/// Build inter-zone conductance matrix for N zones.
///
/// # Arguments
/// * `num_zones` - Number of thermal zones
/// * `zone_properties` - Properties for each zone including wall areas and U-values
///
/// # Returns
/// Vector of inter-zone conductance values
///
/// # Note
/// This is a simplified version that assumes symmetric conductance.
/// For asymmetric cases, use directional conductance calculation.
pub fn build_inter_zone_matrix(num_zones: usize, zone_properties: &ZoneProperties) -> Vec<f64> {
    let mut conductances = Vec::with_capacity(num_zones);

    // For now, implement a simple symmetric matrix
    // In a real implementation, this would use the zone properties
    for i in 0..num_zones {
        // Placeholder: calculate conductance for zone i
        // In practice, this would use wall areas, U-values, etc.
        conductances.push(zone_properties.default_conductance);
    }

    conductances
}

/// Properties for inter-zone conductance calculation.
#[derive(Debug, Clone)]
pub struct ZoneProperties {
    /// Default conductance value for testing
    pub default_conductance: f64,
    // Additional properties would be added here in a full implementation
    // pub wall_areas: Vec<f64>,
    // pub u_values: Vec<f64>,
    // etc.
}

impl Default for ZoneProperties {
    fn default() -> Self {
        ZoneProperties {
            default_conductance: 100.0, // Default value for testing
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inter_zone_conductance() {
        // Test with typical values: 20 m² wall, U=0.5 W/m²·K
        let conductance = calculate_inter_zone_conductance(20.0, 0.5);
        assert_eq!(conductance, 10.0); // 20 × 0.5 = 10 W/K
    }

    #[test]
    fn test_inter_zone_conductance_zero_area() {
        let conductance = calculate_inter_zone_conductance(0.0, 0.5);
        assert_eq!(conductance, 0.0);
    }

    #[test]
    fn test_inter_zone_conductance_zero_u_value() {
        let conductance = calculate_inter_zone_conductance(20.0, 0.0);
        assert_eq!(conductance, 0.0);
    }

    #[test]
    fn test_inter_zone_heat_flow() {
        // Q = h_tr_ij × (ti - tj)
        let heat_flow = inter_zone_heat_flow(10.0, 25.0, 20.0);
        assert_eq!(heat_flow, 50.0); // 10 × (25 - 20) = 50 W
    }

    #[test]
    fn test_inter_zone_heat_flow_negative() {
        // Test sign convention: Q_ij = -Q_ji
        let heat_flow_ij = inter_zone_heat_flow(10.0, 20.0, 25.0);
        let heat_flow_ji = inter_zone_heat_flow(10.0, 25.0, 20.0);
        assert_eq!(heat_flow_ij, -50.0);
        assert_eq!(heat_flow_ji, 50.0);
        assert_eq!(heat_flow_ij, -heat_flow_ji);
    }

    #[test]
    fn test_inter_zone_heat_flow_zero_diff() {
        let heat_flow = inter_zone_heat_flow(10.0, 20.0, 20.0);
        assert_eq!(heat_flow, 0.0);
    }

    #[test]
    fn test_build_inter_zone_matrix() {
        let zone_properties = ZoneProperties::default();
        let matrix = build_inter_zone_matrix(3, &zone_properties);
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix, vec![100.0, 100.0, 100.0]);
    }

    #[test]
    fn test_zone_properties_default() {
        let properties = ZoneProperties::default();
        assert_eq!(properties.default_conductance, 100.0);
    }
}
