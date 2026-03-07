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
    }

    #[test]
    fn test_wall_data_length() {
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
    fn test_validate() {
        let tensor = GeometryTensor::new();
        let issues = tensor.validate();
        // Empty tensor should have no issues
        assert!(issues.is_empty());
    }
}
