use crate::physics::continuous::ContinuousField;
use crate::sim::shading::{Overhang, ShadeFin};
use num_traits::Zero;
use std::ops::{Add, AddAssign, Mul};

use crate::validation::ashrae_140_cases::Orientation;

/// Represents a wall surface in a thermal zone.
#[derive(Clone, Debug)]
pub struct WallSurface {
    /// Area of the surface in square meters (m²).
    pub area: f64,
    /// Thermal transmittance of the surface (W/m²K).
    pub u_value: f64,
    /// Orientation of the surface.
    pub orientation: Orientation,
    /// Optional overhang shading device.
    pub overhang: Option<Overhang>,
    /// List of vertical shade fins.
    pub fins: Vec<ShadeFin>,
}

impl WallSurface {
    /// Create a new WallSurface.
    pub fn new(area: f64, u_value: f64, orientation: Orientation) -> Self {
        WallSurface {
            area,
            u_value,
            orientation,
            overhang: None,
            fins: Vec::new(),
        }
    }

    /// Set an overhang for this surface.
    pub fn with_overhang(mut self, overhang: Overhang) -> Self {
        self.overhang = Some(overhang);
        self
    }

    /// Add a shade fin to this surface.
    pub fn with_fin(mut self, fin: ShadeFin) -> Self {
        self.fins.push(fin);
        self
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
