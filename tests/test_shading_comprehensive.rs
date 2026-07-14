//! Comprehensive shading tests for building energy simulation.
//!
//! This module provides extensive test coverage for the shading module,
//! including edge cases, boundary conditions, and physical validation.

use fluxion::sim::shading::{
    calculate_shaded_fraction, LocalSolarPosition, Overhang, ShadeFin, Side,
};
use fluxion::validation::ashrae_140_cases::WindowArea;
use std::f64::consts::PI;

#[cfg(test)]
mod basic_functionality {
    use super::*;

    #[test]
    fn test_overhang_zero_depth() {
        let window = WindowArea::with_dimensions(
            12.0,
            fluxion::validation::ashrae_140_cases::Orientation::South,
            2.0,
            6.0,
            0.2,
            0.5,
        );
        let overhang = Overhang {
            depth: 0.0,
            distance_above: 0.0,
            extension: 0.0,
        };
        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
        assert_eq!(shaded, 0.0);
    }

    #[test]
    fn test_fin_zero_depth() {
        let window = WindowArea::with_dimensions(
            12.0,
            fluxion::validation::ashrae_140_cases::Orientation::South,
            2.0,
            6.0,
            0.2,
            0.5,
        );
        let fins = vec![ShadeFin {
            depth: 0.0,
            distance_from_edge: 0.0,
            side: Side::Left,
            height: window.height, // Bounded by mounting_height
        }];
        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, None, &fins, &solar);
        assert_eq!(shaded, 0.0);
    }

    #[test]
    fn test_no_shading_devices() {
        let window = WindowArea::with_dimensions(
            12.0,
            fluxion::validation::ashrae_140_cases::Orientation::South,
            2.0,
            6.0,
            0.2,
            0.5,
        );
        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, None, &[], &solar);
        assert_eq!(shaded, 0.0);
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_sun_below_horizon() {
        let window = WindowArea::with_dimensions(
            12.0,
            fluxion::validation::ashrae_140_cases::Orientation::South,
            2.0,
            6.0,
            0.2,
            0.5,
        );
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.0,
            extension: 0.0,
        };
        let solar = LocalSolarPosition {
            altitude: -0.1,
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
        assert_eq!(shaded, 1.0);
    }

    #[test]
    fn test_overhang_no_shadow_when_sun_behind_surface() {
        let window = WindowArea::with_dimensions(
            12.0,
            fluxion::validation::ashrae_140_cases::Orientation::South,
            2.0,
            6.0,
            0.2,
            0.5,
        );
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.0,
            extension: 0.0,
        };
        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: PI * 0.6,
        }; // > 90 deg

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
        assert_eq!(shaded, 0.0);
    }

    #[test]
    fn test_overhang_distance_above_effect() {
        let window = WindowArea::with_dimensions(
            12.0,
            fluxion::validation::ashrae_140_cases::Orientation::South,
            2.0,
            6.0,
            0.2,
            0.5,
        );
        let overhang_close = Overhang {
            depth: 1.0,
            distance_above: 0.1,
            extension: 0.0,
        };
        let overhang_far = Overhang {
            depth: 1.0,
            distance_above: 1.0,
            extension: 0.0,
        };
        let solar = LocalSolarPosition {
            altitude: PI / 3.0,
            relative_azimuth: 0.0,
        };

        let shaded_close = calculate_shaded_fraction(&window, Some(&overhang_close), &[], &solar);
        let shaded_far = calculate_shaded_fraction(&window, Some(&overhang_far), &[], &solar);

        assert!(shaded_close > shaded_far);
    }

    #[test]
    fn test_fin_distance_from_edge_effect() {
        let window = WindowArea::with_dimensions(
            12.0,
            fluxion::validation::ashrae_140_cases::Orientation::South,
            2.0,
            6.0,
            0.2,
            0.5,
        );
        let fin_close = ShadeFin {
            depth: 1.0,
            distance_from_edge: 0.0,
            side: Side::Left,
            height: window.height, // Bounded by mounting_height
        };
        let fin_far = ShadeFin {
            depth: 1.0,
            distance_from_edge: 0.5,
            side: Side::Left,
            height: window.height, // Bounded by mounting_height
        };
        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: -PI / 6.0,
        };

        let shaded_close = calculate_shaded_fraction(&window, None, &[fin_close], &solar);
        let shaded_far = calculate_shaded_fraction(&window, None, &[fin_far], &solar);

        assert!(shaded_close > shaded_far);
    }
}

#[cfg(test)]
mod interactions {
    use super::*;

    #[test]
    fn test_overhang_and_fin_interaction() {
        let window = WindowArea::with_dimensions(
            12.0,
            fluxion::validation::ashrae_140_cases::Orientation::South,
            2.0,
            6.0,
            0.2,
            0.5,
        );
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.2,
            extension: 0.0,
        };
        let fins = vec![ShadeFin {
            depth: 1.0,
            distance_from_edge: 0.2,
            side: Side::Left,
            height: window.height, // Bounded by mounting_height
        }];
        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: 0.0,
        };

        let shaded_only_overhang = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
        let shaded_only_fins = calculate_shaded_fraction(&window, None, &fins, &solar);
        let shaded_both = calculate_shaded_fraction(&window, Some(&overhang), &fins, &solar);

        assert!(shaded_both >= shaded_only_overhang);
        assert!(shaded_both >= shaded_only_fins);
        assert!(shaded_both <= 1.0);
    }
}
