#![allow(clippy::approx_constant)] // Allow spec constants like 0.318 (ASHRAE 140 values)

//! Multi-layer construction R-value calculator for building envelopes.
//!
//! # Crate split (Issue #2462 — Phase 2 of the crate split)
//!
//! As of #2462, the implementation of `ConstructionLayer`, `Construction`,
//! `MassClass`, `Materials`, `Assemblies`, `SurfaceType`, `interior_film_coeff`,
//! `exterior_film_coeff`, and the ASHRAE 140 film/air constants lives in the
//! workspace leaf crate `fluxion_core::construction` (where it was hoisted to
//! break the `physics ↔ sim` cycle documented in `docs/mutation_testing_crate_split.md`
//! §"Phase 2").
//!
//! This file re-exports those types at the historical
//! `fluxion::sim::construction::*` path so existing call sites — both inside
//! the workspace (`crate::sim::construction::ConstructionLayer`) and outside
//! (`fluxion::sim::construction::ConstructionLayer`) — keep working without edits.
//!
//! `WallSurface` (below) depends on `crate::sim::shading::{Overhang, ShadeFin}`
//! so it stays in the main crate rather than moving into `fluxion_core`.

use crate::sim::shading::{Overhang, ShadeFin};
// Issue #2462 (Phase 2 of the crate split): `ConstructionLayer`, `Construction`,
// `MassClass`, `Materials`, `Assemblies`, `SurfaceType`, the ASHRAE 140
// film/air constants, and the `interior_film_coeff`/`exterior_film_coeff`
// helpers all moved from this file into the `fluxion_core::construction`
// leaf module (see ARCHITECTURE.md §"Remaining cycles"). Re-export them at
// the historical `fluxion::sim::construction::*` path so existing call sites
// (e.g. `crate::sim::construction::ConstructionLayer`,
// `fluxion::sim::construction::Assemblies`) keep resolving unchanged.
use fluxion_core::ashrae_cases::Orientation;
#[doc(inline)]
pub use fluxion_core::construction::{
    exterior_film_coeff, interior_film_coeff, Assemblies, Construction, ConstructionLayer,
    MassClass, Materials, SurfaceType, AIR_DENSITY_SEA_LEVEL, AIR_SPECIFIC_HEAT,
    EXTERIOR_FILM_COEFF, EXTERIOR_FILM_COEFF_DEFAULT, INTERIOR_FILM_COEFF,
    INTERIOR_FILM_COEFF_CEILING, INTERIOR_FILM_COEFF_FLOOR, INTERIOR_FILM_COEFF_WALL,
};
use fluxion_core::tensor::ContinuousField;
use num_traits::Zero;
use std::ops::{Add, AddAssign, Mul};

/// Represents a wall surface in a thermal zone.
///
/// `WallSurface` lives in the main crate (not `fluxion_core`) because its fields
/// reference `sim::shading::{Overhang, ShadeFin}` — types that pull in additional
/// shading logic the leaf crate does not need. It composes `fluxion_core::construction`
/// types where applicable (e.g., its `orientation` is `fluxion_core::ashrae_cases::Orientation`).
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
    pub fn calculate_heat_gain<T>(&self, field: &impl ContinuousField<T>) -> T
    where
        T: Add<Output = T> + AddAssign + Mul<f64, Output = T> + Zero + Clone,
    {
        let integral = field.integrate(0.0, 1.0, 0.0, 1.0);
        integral * self.area
    }
}

#[cfg(test)]
mod wall_surface_tests {
    use super::*;
    use crate::sim::shading::Side;
    use fluxion_core::tensor::ConstantField;

    #[test]
    fn test_wall_surface_heat_gain() {
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
            height: 3.0, // Default height
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
            height: 3.0, // Default height
        };
        let fin2 = ShadeFin {
            depth: 0.6,
            distance_from_edge: 1.0,
            side: Side::Right,
            height: 3.0, // Default height
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
            height: 3.0, // Default height
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
    fn test_wall_surface_clone_debug() {
        let surface = WallSurface::new(10.0, 0.5, Orientation::East);
        let cloned = surface.clone();
        assert!((cloned.area - surface.area).abs() < 1e-6);
        assert_eq!(cloned.orientation, surface.orientation);
        let debug_str = format!("{:?}", surface);
        assert!(debug_str.contains("WallSurface"));
    }
}
