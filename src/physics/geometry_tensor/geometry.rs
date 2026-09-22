//! CTA geometry tensors
//!
//! Two parallel concerns share this file:
//!
//! 1. **CTA geometry tensors** ([`GeometryTensor`], [`WallData`]) — flat,
//!    copy-through tensors on the Python↔Rust boundary that carry zone coordinates,
//!    wall geometry, and inter-zone adjacency for the PDF/CAD ingestion pipeline
//!    (issues #448 / #453). These are intentionally kept as `Vec<f64>` so the
//!    Python bindings expose them with zero-copy semantics.

pub use fluxion_core::zone_count_policy::MAX_ZONES;

pub const MAX_WALLS: usize = 500;

pub const ZONE_COORDS_DIMS: (usize, usize) = (MAX_ZONES, 20);

pub const WALL_MATRIX_DIMS: (usize, usize) = (MAX_WALLS, 6);

pub const WINDOW_MATRIX_DIMS: (usize, usize) = (MAX_WALLS, 6);

pub const ADJACENCY_MATRIX_DIMS: (usize, usize) = (MAX_ZONES, MAX_ZONES);

pub const ZONE_PROPERTIES_DIMS: (usize, usize) = (MAX_ZONES, 5);

#[derive(Debug, Clone)]
pub struct GeometryTensor {
    pub zone_coords: Vec<f64>,
    pub wall_matrix: Vec<f64>,
    pub window_matrix: Vec<f64>,
    pub adjacency_matrix: Vec<f64>,
    pub zone_properties: Vec<f64>,
    pub summary: Vec<f64>,
}

impl GeometryTensor {
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

    #[cfg(feature = "python-bindings")]
    pub fn from_numpy_arrays(
        zone_coords: &[f64],
        wall_matrix: &[f64],
        window_matrix: &[f64],
        adjacency_matrix: &[f64],
        zone_properties: &[f64],
        summary: &[f64],
    ) -> Result<Self, String> {
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

    pub fn num_zones(&self) -> usize {
        self.summary[0] as usize
    }

    pub fn num_walls(&self) -> usize {
        self.summary[1] as usize
    }

    pub fn total_area(&self) -> f64 {
        self.summary[4]
    }

    pub fn total_volume(&self) -> f64 {
        self.summary[5]
    }

    pub fn get_zone_coords(&self, index: usize) -> Option<&[f64; 20]> {
        if index < MAX_ZONES {
            let start = index * 20;
            let slice = &self.zone_coords[start..start + 20];
            Some(unsafe { &*(slice.as_ptr() as *const [f64; 20]) })
        } else {
            None
        }
    }

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

    pub fn zones_adjacent(&self, i: usize, j: usize) -> bool {
        if i < MAX_ZONES && j < MAX_ZONES {
            let idx = i * MAX_ZONES + j;
            self.adjacency_matrix[idx] > 0.5
        } else {
            false
        }
    }

    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.zone_coords.iter().any(|x| x.is_nan()) {
            issues.push("zone_coords contains NaN".to_string());
        }
        if self.wall_matrix.iter().any(|x| x.is_nan()) {
            issues.push("wall_matrix contains NaN".to_string());
        }

        if self.zone_properties.iter().any(|&x| x.is_nan()) {
            for i in 0..MAX_ZONES {
                let area = self.zone_properties[i * 5];
                if area < 0.0 {
                    issues.push(format!("Zone {} has negative area: {}", i, area));
                }
            }
        }

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
    pub fn length(&self) -> f64 {
        let dx = self.x2 - self.x1;
        let dy = self.y2 - self.y1;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn area(&self) -> f64 {
        self.length() * self.height
    }
}

impl Default for GeometryTensor {
    fn default() -> Self {
        Self::new()
    }
}
