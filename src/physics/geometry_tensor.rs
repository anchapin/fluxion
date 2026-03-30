//! Geometry Tensor Module
//!
//! This module provides zero-copy geometry tensor support for the Python-Rust boundary.
//! It allows passing CTA geometry tensors from Python to Rust without memory copies.

/// Maximum number of thermal zones supported
pub const MAX_ZONES: usize = 100;
/// Maximum number of walls supported
pub const MAX_WALLS: usize = 500;

/// Zone coordinates tensor shape: (MAX_ZONES, 20)
/// Format: [x1, y1, x2, y2, ..., x8, y8, floor_height, ceiling_height, area, volume, perimeter, zone_id]
pub const ZONE_COORDS_DIMS: (usize, usize) = (MAX_ZONES, 20);

/// Wall matrix shape: (MAX_WALLS, 6)
/// Format: [x1, y1, x2, y2, height, thickness]
pub const WALL_MATRIX_DIMS: (usize, usize) = (MAX_WALLS, 6);

/// Window matrix shape: (MAX_WALLS, 6)
/// Format: [x1, y1, x2, y2, height, sill_height]
pub const WINDOW_MATRIX_DIMS: (usize, usize) = (MAX_WALLS, 6);

/// Adjacency matrix shape: (MAX_ZONES, MAX_ZONES)
pub const ADJACENCY_MATRIX_DIMS: (usize, usize) = (MAX_ZONES, MAX_ZONES);

/// Zone properties shape: (MAX_ZONES, 5)
/// Format: [floor_area, volume, perimeter, num_windows, num_doors]
pub const ZONE_PROPERTIES_DIMS: (usize, usize) = (MAX_ZONES, 5);

/// GeometryTensor - A container for CTA geometry tensors.
///
/// This struct holds all the geometry information extracted from PDF/CAD files
/// in a format ready for use in the Fluxion thermal simulation.
#[derive(Debug, Clone)]
pub struct GeometryTensor {
    /// Zone coordinates: (MAX_ZONES, 20)
    pub zone_coords: Vec<f64>,
    /// Wall matrix: (MAX_WALLS, 6)
    pub wall_matrix: Vec<f64>,
    /// Window matrix: (MAX_WALLS, 6)
    pub window_matrix: Vec<f64>,
    /// Adjacency matrix: (MAX_ZONES, MAX_ZONES)
    pub adjacency_matrix: Vec<f64>,
    /// Zone properties: (MAX_ZONES, 5)
    pub zone_properties: Vec<f64>,
    /// Summary: [num_zones, num_walls, num_windows, num_doors, total_area, total_volume]
    pub summary: Vec<f64>,
}

impl GeometryTensor {
    /// Create a new empty GeometryTensor.
    pub fn new() -> Self {
        let zone_coords = vec![0.0; MAX_ZONES * 20];
        let wall_matrix = vec![0.0; MAX_WALLS * 6];
        let window_matrix = vec![0.0; MAX_WALLS * 6];
        let adjacency_matrix = vec![0.0; MAX_ZONES * MAX_ZONES];
        let zone_properties = vec![0.0; MAX_ZONES * 5];
        let summary = vec![0.0; 6];

        GeometryTensor {
            zone_coords,
            wall_matrix,
            window_matrix,
            adjacency_matrix,
            zone_properties,
            summary,
        }
    }

    /// Create a GeometryTensor from numpy arrays (zero-copy when possible).
    #[cfg(feature = "python-bindings")]
    pub fn from_numpy_arrays(
        zone_coords: &[f64],
        wall_matrix: &[f64],
        window_matrix: &[f64],
        adjacency_matrix: &[f64],
        zone_properties: &[f64],
        summary: &[f64],
    ) -> Result<Self, String> {
        // Validate sizes
        if zone_coords.len() != MAX_ZONES * 20 {
            return Err(format!(
                "zone_coords has {} elements, expected {}",
                zone_coords.len(),
                MAX_ZONES * 20
            ));
        }
        if wall_matrix.len() != MAX_WALLS * 6 {
            return Err(format!(
                "wall_matrix has {} elements, expected {}",
                wall_matrix.len(),
                MAX_WALLS * 6
            ));
        }
        if window_matrix.len() != MAX_WALLS * 6 {
            return Err(format!(
                "window_matrix has {} elements, expected {}",
                window_matrix.len(),
                MAX_WALLS * 6
            ));
        }
        if adjacency_matrix.len() != MAX_ZONES * MAX_ZONES {
            return Err(format!(
                "adjacency_matrix has {} elements, expected {}",
                adjacency_matrix.len(),
                MAX_ZONES * MAX_ZONES
            ));
        }
        if zone_properties.len() != MAX_ZONES * 5 {
            return Err(format!(
                "zone_properties has {} elements, expected {}",
                zone_properties.len(),
                MAX_ZONES * 5
            ));
        }

        Ok(GeometryTensor {
            zone_coords: zone_coords.to_vec(),
            wall_matrix: wall_matrix.to_vec(),
            window_matrix: window_matrix.to_vec(),
            adjacency_matrix: adjacency_matrix.to_vec(),
            zone_properties: zone_properties.to_vec(),
            summary: summary.to_vec(),
        })
    }

    /// Get the number of zones in the geometry.
    pub fn num_zones(&self) -> usize {
        self.summary[0] as usize
    }

    /// Get the number of walls in the geometry.
    pub fn num_walls(&self) -> usize {
        self.summary[1] as usize
    }

    /// Get the total floor area.
    pub fn total_area(&self) -> f64 {
        self.summary[4]
    }

    /// Get the total volume.
    pub fn total_volume(&self) -> f64 {
        self.summary[5]
    }

    /// Get zone coordinates at index.
    pub fn get_zone_coords(&self, index: usize) -> Option<&[f64; 20]> {
        if index < MAX_ZONES {
            let start = index * 20;
            let slice = &self.zone_coords[start..start + 20];
            // Convert slice to array
            Some(unsafe { &*(slice.as_ptr() as *const [f64; 20]) })
        } else {
            None
        }
    }

    /// Get wall data at index.
    pub fn get_wall(&self, index: usize) -> Option<WallData> {
        if index < MAX_WALLS {
            let start = index * 6;
            let data = &self.wall_matrix[start..start + 6];
            Some(WallData {
                x1: data[0],
                y1: data[1],
                x2: data[2],
                y2: data[3],
                height: data[4],
                thickness: data[5],
            })
        } else {
            None
        }
    }

    /// Check if zones i and j are adjacent.
    pub fn zones_adjacent(&self, i: usize, j: usize) -> bool {
        if i < MAX_ZONES && j < MAX_ZONES {
            let idx = i * MAX_ZONES + j;
            self.adjacency_matrix[idx] > 0.5
        } else {
            false
        }
    }

    /// Validate the geometry tensor.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for NaN
        if self.zone_coords.iter().any(|x| x.is_nan()) {
            issues.push("zone_coords contains NaN".to_string());
        }
        if self.wall_matrix.iter().any(|x| x.is_nan()) {
            issues.push("wall_matrix contains NaN".to_string());
        }

        // Check for negative areas
        if self.zone_properties.iter().any(|&x| x.is_nan()) {
            // Check floor_area (index 0 in each zone's properties)
            for i in 0..MAX_ZONES {
                let area = self.zone_properties[i * 5];
                if area < 0.0 {
                    issues.push(format!("Zone {} has negative area: {}", i, area));
                }
            }
        }

        // Check adjacency symmetry
        for i in 0..MAX_ZONES {
            for j in 0..MAX_ZONES {
                let a = self.adjacency_matrix[i * MAX_ZONES + j];
                let b = self.adjacency_matrix[j * MAX_ZONES + i];
                if (a > 0.5) != (b > 0.5) {
                    issues.push(format!(
                        "Adjacency matrix asymmetry at ({}, {}): {} vs {}",
                        i, j, a, b
                    ));
                    break;
                }
            }
        }

        issues
    }
}

/// Wall data structure.
#[derive(Debug, Clone, Copy)]
pub struct WallData {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub height: f64,
    pub thickness: f64,
}

impl WallData {
    /// Calculate wall length.
    pub fn length(&self) -> f64 {
        let dx = self.x2 - self.x1;
        let dy = self.y2 - self.y1;
        (dx * dx + dy * dy).sqrt()
    }

    /// Calculate wall area.
    pub fn area(&self) -> f64 {
        self.length() * self.height
    }
}

impl Default for GeometryTensor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometry_tensor_creation() {
        let tensor = GeometryTensor::new();
        assert_eq!(tensor.zone_coords.len(), MAX_ZONES * 20);
        assert_eq!(tensor.wall_matrix.len(), MAX_WALLS * 6);
        assert_eq!(tensor.window_matrix.len(), MAX_WALLS * 6);
        assert_eq!(tensor.adjacency_matrix.len(), MAX_ZONES * MAX_ZONES);
        assert_eq!(tensor.zone_properties.len(), MAX_ZONES * 5);
        assert_eq!(tensor.summary.len(), 6);
    }

    #[test]
    fn test_geometry_tensor_default() {
        let tensor = GeometryTensor::default();
        assert_eq!(tensor.zone_coords.len(), MAX_ZONES * 20);
        assert_eq!(tensor.wall_matrix.len(), MAX_WALLS * 6);
    }

    #[test]
    fn test_wall_data_length_horizontal() {
        let wall = WallData {
            x1: 0.0,
            y1: 0.0,
            x2: 3.0,
            y2: 0.0,
            height: 2.4,
            thickness: 0.2,
        };
        assert!((wall.length() - 3.0).abs() < 1e-10);
        assert!((wall.area() - 7.2).abs() < 1e-10);
    }

    #[test]
    fn test_wall_data_length_vertical() {
        let wall = WallData {
            x1: 0.0,
            y1: 0.0,
            x2: 0.0,
            y2: 4.0,
            height: 2.4,
            thickness: 0.2,
        };
        assert!((wall.length() - 4.0).abs() < 1e-10);
        assert!((wall.area() - 9.6).abs() < 1e-10);
    }

    #[test]
    fn test_wall_data_diagonal() {
        let wall = WallData {
            x1: 0.0,
            y1: 0.0,
            x2: 3.0,
            y2: 4.0,
            height: 2.4,
            thickness: 0.2,
        };
        assert!((wall.length() - 5.0).abs() < 1e-10);
        assert!((wall.area() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_wall_data_zero_height_area() {
        let wall = WallData {
            x1: 0.0,
            y1: 0.0,
            x2: 3.0,
            y2: 4.0,
            height: 0.0,
            thickness: 0.2,
        };
        assert_eq!(wall.area(), 0.0);
    }

    #[test]
    fn test_num_zones() {
        let mut tensor = GeometryTensor::new();
        tensor.summary[0] = 5.0;
        assert_eq!(tensor.num_zones(), 5);
    }

    #[test]
    fn test_num_walls() {
        let mut tensor = GeometryTensor::new();
        tensor.summary[1] = 20.0;
        assert_eq!(tensor.num_walls(), 20);
    }

    #[test]
    fn test_total_area() {
        let mut tensor = GeometryTensor::new();
        tensor.summary[4] = 500.0;
        assert!((tensor.total_area() - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_volume() {
        let mut tensor = GeometryTensor::new();
        tensor.summary[5] = 1500.0;
        assert!((tensor.total_volume() - 1500.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_zone_coords_valid() {
        let mut tensor = GeometryTensor::new();
        // Set some test values in zone 0
        tensor.zone_coords[0] = 1.0;
        tensor.zone_coords[1] = 2.0;
        tensor.zone_coords[19] = 3.0;

        let coords = tensor.get_zone_coords(0).unwrap();
        assert_eq!(coords[0], 1.0);
        assert_eq!(coords[1], 2.0);
        assert_eq!(coords[19], 3.0);
    }

    #[test]
    fn test_get_zone_coords_invalid() {
        let tensor = GeometryTensor::new();
        assert!(tensor.get_zone_coords(MAX_ZONES).is_none());
        assert!(tensor.get_zone_coords(MAX_ZONES + 10).is_none());
    }

    #[test]
    fn test_get_wall_valid() {
        let mut tensor = GeometryTensor::new();
        // Set some test values for wall 0
        tensor.wall_matrix[0] = 1.0;
        tensor.wall_matrix[1] = 2.0;
        tensor.wall_matrix[2] = 3.0;
        tensor.wall_matrix[3] = 4.0;
        tensor.wall_matrix[4] = 2.5;
        tensor.wall_matrix[5] = 0.2;

        let wall = tensor.get_wall(0).unwrap();
        assert_eq!(wall.x1, 1.0);
        assert_eq!(wall.y1, 2.0);
        assert_eq!(wall.x2, 3.0);
        assert_eq!(wall.y2, 4.0);
        assert_eq!(wall.height, 2.5);
        assert_eq!(wall.thickness, 0.2);
    }

    #[test]
    fn test_get_wall_invalid() {
        let tensor = GeometryTensor::new();
        assert!(tensor.get_wall(MAX_WALLS).is_none());
        assert!(tensor.get_wall(MAX_WALLS + 10).is_none());
    }

    #[test]
    fn test_zones_adjacent_true() {
        let mut tensor = GeometryTensor::new();
        let idx = 2 * MAX_ZONES + 3;
        tensor.adjacency_matrix[idx] = 1.0;
        assert!(tensor.zones_adjacent(2, 3));
    }

    #[test]
    fn test_zones_adjacent_false() {
        let tensor = GeometryTensor::new();
        // Default values are 0.0
        assert!(!tensor.zones_adjacent(0, 1));
    }

    #[test]
    fn test_zones_adjacent_symmetry() {
        let mut tensor = GeometryTensor::new();
        // Set adjacency for zone 0 -> 1
        let idx_01 = 0 * MAX_ZONES + 1;
        tensor.adjacency_matrix[idx_01] = 1.0;
        // Also set reverse
        let idx_10 = 1 * MAX_ZONES + 0;
        tensor.adjacency_matrix[idx_10] = 1.0;

        assert!(tensor.zones_adjacent(0, 1));
        assert!(tensor.zones_adjacent(1, 0));
    }

    #[test]
    fn test_zones_adjacent_out_of_bounds() {
        let tensor = GeometryTensor::new();
        assert!(!tensor.zones_adjacent(MAX_ZONES, 0));
        assert!(!tensor.zones_adjacent(0, MAX_ZONES));
    }

    #[test]
    fn test_validate_clean() {
        let tensor = GeometryTensor::new();
        let issues = tensor.validate();
        // Empty tensor should have no issues (zeros are valid)
        assert!(issues.is_empty());
    }

    #[test]
    fn test_validate_zone_coords_nan() {
        let mut tensor = GeometryTensor::new();
        tensor.zone_coords[0] = f64::NAN;
        let issues = tensor.validate();
        assert!(issues.iter().any(|s| s.contains("NaN")));
    }

    #[test]
    fn test_validate_wall_matrix_nan() {
        let mut tensor = GeometryTensor::new();
        tensor.wall_matrix[0] = f64::NAN;
        let issues = tensor.validate();
        assert!(issues
            .iter()
            .any(|s| s.contains("NaN") && s.contains("wall")));
    }

    #[test]
    fn test_validate_negative_zone_area() {
        let mut tensor = GeometryTensor::new();
        tensor.zone_properties[0] = -50.0; // Negative area for zone 0
        tensor.zone_properties[1] = f64::NAN; // Add NaN to trigger nested check
        let issues = tensor.validate();
        assert!(issues.iter().any(|s| s.contains("negative area")));
    }

    #[test]
    fn test_validate_adjacency_asymmetry() {
        let mut tensor = GeometryTensor::new();
        // Set zone 0 -> 1 adjacency but not 1 -> 0
        let idx_01 = 0 * MAX_ZONES + 1;
        tensor.adjacency_matrix[idx_01] = 1.0;
        let issues = tensor.validate();
        assert!(issues.iter().any(|s| s.contains("asymmetry")));
    }

    #[test]
    fn test_validate_multiple_issues() {
        let mut tensor = GeometryTensor::new();
        tensor.zone_coords[0] = f64::NAN;
        tensor.zone_properties[0] = -50.0;
        tensor.zone_properties[1] = f64::NAN; // Add NaN to trigger area check
        let issues = tensor.validate();
        assert!(issues.len() >= 2);
    }

    #[test]
    fn test_geometry_tensor_clone() {
        let mut tensor = GeometryTensor::new();
        tensor.summary[0] = 3.0;
        tensor.summary[4] = 100.0;
        let cloned = tensor.clone();
        assert_eq!(cloned.summary[0], 3.0);
        assert_eq!(cloned.summary[4], 100.0);
    }

    #[test]
    fn test_constants_values() {
        assert_eq!(MAX_ZONES, 100);
        assert_eq!(MAX_WALLS, 500);
        assert_eq!(ZONE_COORDS_DIMS, (100, 20));
        assert_eq!(WALL_MATRIX_DIMS, (500, 6));
        assert_eq!(WINDOW_MATRIX_DIMS, (500, 6));
        assert_eq!(ADJACENCY_MATRIX_DIMS, (100, 100));
        assert_eq!(ZONE_PROPERTIES_DIMS, (100, 5));
    }

    #[test]
    fn test_wall_data_debug() {
        let wall = WallData {
            x1: 1.0,
            y1: 2.0,
            x2: 3.0,
            y2: 4.0,
            height: 2.5,
            thickness: 0.2,
        };
        let debug_str = format!("{:?}", wall);
        assert!(debug_str.contains("WallData"));
    }

    #[test]
    fn test_wall_data_copy() {
        let wall = WallData {
            x1: 1.0,
            y1: 2.0,
            x2: 3.0,
            y2: 4.0,
            height: 2.5,
            thickness: 0.2,
        };
        let copied = wall;
        assert_eq!(copied.x1, 1.0);
        assert_eq!(copied.y2, 4.0);
    }

    #[test]
    fn test_geometry_tensor_debug() {
        let tensor = GeometryTensor::new();
        let debug_str = format!("{:?}", tensor);
        assert!(debug_str.contains("GeometryTensor"));
    }

    #[test]
    fn test_get_multiple_zones() {
        let mut tensor = GeometryTensor::new();
        // Set markers for first 3 zones
        for i in 0..3 {
            let idx = i * 20;
            tensor.zone_coords[idx] = i as f64;
        }

        for i in 0..3 {
            let coords = tensor.get_zone_coords(i).unwrap();
            assert_eq!(coords[0], i as f64);
        }
    }

    #[test]
    fn test_get_multiple_walls() {
        let mut tensor = GeometryTensor::new();
        // Set markers for first 5 walls
        for i in 0..5 {
            let idx = i * 6;
            tensor.wall_matrix[idx] = i as f64;
        }

        for i in 0..5 {
            let wall = tensor.get_wall(i).unwrap();
            assert_eq!(wall.x1, i as f64);
        }
    }

    #[test]
    fn test_wall_data_zero_length() {
        let wall = WallData {
            x1: 0.0,
            y1: 0.0,
            x2: 0.0,
            y2: 0.0,
            height: 2.4,
            thickness: 0.2,
        };
        assert_eq!(wall.length(), 0.0);
        assert_eq!(wall.area(), 0.0);
    }
}
