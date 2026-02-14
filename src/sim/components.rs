use crate::physics::continuous::ContinuousField;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul};

/// Orientation for surfaces and windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Orientation {
    North,
    East,
    South,
    West,
    Up,   // Roof
    Down, // Floor
}

impl Orientation {
    /// Returns the azimuth angle in degrees (0° = South, ASHRAE Standard).
    /// Note: ASHRAE 140 uses South=0, West=90, North=180, East=270.
    pub fn azimuth(&self) -> f64 {
        match self {
            Orientation::South => 0.0,
            Orientation::West => 90.0,
            Orientation::North => 180.0,
            Orientation::East => 270.0,
            Orientation::Up | Orientation::Down => 0.0, // Not applicable for horizontal
        }
    }

    /// Returns the cosine of the azimuth angle.
    pub fn cosine_azimuth(&self) -> f64 {
        (self.azimuth().to_radians()).cos()
    }

    /// Returns the sine of the azimuth angle.
    pub fn sine_azimuth(&self) -> f64 {
        (self.azimuth().to_radians()).sin()
    }

    /// Returns true if the orientation is a vertical wall (N/E/S/W).
    pub fn is_wall(&self) -> bool {
        matches!(
            self,
            Orientation::North | Orientation::East | Orientation::South | Orientation::West
        )
    }
}

/// Represents a wall surface in a thermal zone.
#[derive(Clone, Debug)]
pub struct WallSurface {
    /// Area of the surface in square meters (m²).
    pub area: f64,
    /// Thermal transmittance of the surface (W/m²K).
    pub u_value: f64,
    /// Orientation of the surface.
    pub orientation: Orientation,
}

impl WallSurface {
    /// Create a new WallSurface.
    pub fn new(area: f64, u_value: f64, orientation: Orientation) -> Self {
        WallSurface {
            area,
            u_value,
            orientation,
        }
    }

    /// Calculate heat gain for this surface given a continuous field representing
    /// the heat flux (W/m²) or similar potential over the surface.
    ///
    /// The field is assumed to be defined over the normalized domain [0, 1] x [0, 1].
    /// The integration result (total value over normalized domain) is scaled by the area.
    pub fn calculate_heat_gain<T>(&self, field: &impl ContinuousField<T>) -> T
    where
        T: Add<Output = T> + AddAssign + Mul<f64, Output = T> + Zero + Clone,
    {
        // Integrate the field over the normalized domain [0, 1] x [0, 1]
        let integral = field.integrate(0.0, 1.0, 0.0, 1.0);
        integral * self.area
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::continuous::ConstantField;

    #[test]
    fn test_heat_gain_constant() {
        let surface = WallSurface::new(10.0, 0.5, Orientation::South);
        let field = ConstantField { value: 2.0 }; // 2.0 W/m²

        // Total heat = 10.0 m² * 2.0 W/m² * 1.0 (integral over unit square) = 20.0 W
        let heat_gain = surface.calculate_heat_gain(&field);
        assert!((heat_gain - 20.0).abs() < 1e-6);
    }
}
