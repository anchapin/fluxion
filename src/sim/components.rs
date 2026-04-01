use crate::physics::continuous::ContinuousField;
use crate::sim::shading::{Overhang, ShadeFin};
use num_traits::Zero;
use std::ops::{Add, AddAssign, Mul};

use crate::validation::ashrae_140_cases::Orientation;

/// Represents a wall surface in a thermal zone.
#[derive(Clone, Debug)]
pub struct WallSurface {
    /// Total area of the surface in square meters (m²).
    pub area: f64,
    /// Window area on this surface in square meters (m²).
    pub window_area: f64,
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
            window_area: 0.0,
            u_value,
            orientation,
            overhang: None,
            fins: Vec::new(),
        }
    }

    /// Create a new WallSurface with a window.
    pub fn with_window(mut self, window_area: f64) -> Self {
        self.window_area = window_area;
        self
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
        let integral = field.integrate(0.0, 1.0, 0.0, 1.0);
        integral * self.area
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::continuous::ConstantField;
    use crate::sim::shading::Side;

    #[test]
    fn test_heat_gain_constant() {
        let surface = WallSurface::new(10.0, 0.5, Orientation::South);
        let field = ConstantField { value: 2.0 };
        let heat_gain = surface.calculate_heat_gain(&field);
        assert!((heat_gain - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_wall_surface_new_defaults() {
        let surface = WallSurface::new(25.0, 1.5, Orientation::North);
        assert!((surface.area - 25.0).abs() < 1e-6);
        assert!((surface.window_area - 0.0).abs() < 1e-6);
        assert!((surface.u_value - 1.5).abs() < 1e-6);
        assert!(surface.overhang.is_none());
        assert!(surface.fins.is_empty());
    }

    #[test]
    fn test_wall_surface_with_window() {
        let surface = WallSurface::new(20.0, 0.8, Orientation::East).with_window(5.0);
        assert!((surface.window_area - 5.0).abs() < 1e-6);
        assert!((surface.area - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_wall_surface_with_overhang() {
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.5,
            extension: 0.0,
        };
        let surface = WallSurface::new(15.0, 1.2, Orientation::South).with_overhang(overhang);
        assert!(surface.overhang.is_some());
        assert!((surface.overhang.as_ref().unwrap().depth - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wall_surface_with_fin() {
        let fin = ShadeFin {
            depth: 0.5,
            distance_from_edge: 0.0,
            side: Side::Left,
        };
        let surface = WallSurface::new(12.0, 0.9, Orientation::West).with_fin(fin);
        assert_eq!(surface.fins.len(), 1);
        assert!((surface.fins[0].depth - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_wall_surface_with_multiple_fins() {
        let fin1 = ShadeFin {
            depth: 0.3,
            distance_from_edge: 0.0,
            side: Side::Left,
        };
        let fin2 = ShadeFin {
            depth: 0.6,
            distance_from_edge: 1.0,
            side: Side::Right,
        };
        let surface = WallSurface::new(10.0, 1.0, Orientation::South)
            .with_fin(fin1)
            .with_fin(fin2);
        assert_eq!(surface.fins.len(), 2);
    }

    #[test]
    fn test_wall_surface_builder_chain() {
        let overhang = Overhang {
            depth: 2.0,
            distance_above: 1.0,
            extension: 0.5,
        };
        let fin = ShadeFin {
            depth: 0.8,
            distance_from_edge: 0.5,
            side: Side::Left,
        };
        let surface = WallSurface::new(30.0, 2.0, Orientation::North)
            .with_window(8.0)
            .with_overhang(overhang)
            .with_fin(fin);
        assert!((surface.window_area - 8.0).abs() < 1e-6);
        assert!(surface.overhang.is_some());
        assert_eq!(surface.fins.len(), 1);
    }

    #[test]
    fn test_wall_surface_heat_gain_zero_field() {
        let surface = WallSurface::new(10.0, 0.5, Orientation::South);
        let field = ConstantField { value: 0.0 };
        let heat_gain = surface.calculate_heat_gain(&field);
        assert!((heat_gain - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_wall_surface_heat_gain_large_area() {
        let surface = WallSurface::new(100.0, 0.5, Orientation::South);
        let field = ConstantField { value: 1.0 };
        let heat_gain = surface.calculate_heat_gain(&field);
        assert!((heat_gain - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_wall_surface_clone_debug() {
        let surface = WallSurface::new(10.0, 0.5, Orientation::East);
        let cloned = surface.clone();
        assert!((cloned.area - surface.area).abs() < 1e-6);
        assert_eq!(cloned.orientation, surface.orientation);
        let debug_str = format!("{:?}", surface);
        assert!(debug_str.contains("WallSurface"));
    }
}
