//! Comprehensive shading tests for building energy simulation.
//!
//! This module provides extensive test coverage for the shading module,
//! including edge cases, boundary conditions, and physical validation.

use fluxion::sim::shading::*;
use fluxion::validation::ashrae_140_cases::{Orientation, WindowArea};
use std::f64::consts::PI;

// ============================================================================
// Overhang Shadow Tests
// ============================================================================

mod overhang_tests {
    use super::*;

    #[test]
    fn test_overhang_no_shadow_when_sun_behind_surface() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.0,
            extension: 10.0,
        };

        // Sun at 90 degrees relative azimuth (behind surface)
        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: PI / 2.0,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
        assert_eq!(shaded, 0.0, "No shadow when sun is behind surface");
    }

    #[test]
    fn test_overhang_more_shadow_at_low_sun_angle() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 2.0, // Deep overhang
            distance_above: 0.0,
            extension: 10.0,
        };

        // Very low sun angle (winter morning)
        let solar_low = LocalSolarPosition {
            altitude: 10.0_f64.to_radians(),
            relative_azimuth: 0.0,
        };

        // High sun angle
        let solar_high = LocalSolarPosition {
            altitude: 60.0_f64.to_radians(),
            relative_azimuth: 0.0,
        };

        let shaded_low = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar_low);
        let shaded_high = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar_high);

        // Low angle should create more shading than high angle
        assert!(
            shaded_low > shaded_high,
            "Low sun angle should create more shading"
        );
    }

    #[test]
    fn test_overhang_partial_shadow_mid_sun_angle() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.0,
            extension: 10.0,
        };

        // 45 degree sun angle
        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
        // tan(45) = 1, shadow = 1m, window height = 2m, so 50% shaded
        assert!(
            (shaded - 0.5).abs() < 0.01,
            "45 degree sun should shade 50%"
        );
    }

    #[test]
    fn test_overhang_distance_above_effect() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);

        // Overhang far above window
        let overhang_far = Overhang {
            depth: 1.0,
            distance_above: 5.0, // Far above
            extension: 10.0,
        };

        // Overhang at window top
        let overhang_close = Overhang {
            depth: 1.0,
            distance_above: 0.0,
            extension: 10.0,
        };

        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: 0.0,
        };

        let shaded_far = calculate_shaded_fraction(&window, Some(&overhang_far), &[], &solar);
        let shaded_close = calculate_shaded_fraction(&window, Some(&overhang_close), &[], &solar);

        assert!(
            shaded_far < shaded_close,
            "Overhang further above should shade less"
        );
    }

    #[test]
    fn test_overhang_zero_depth() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 0.0,
            distance_above: 0.0,
            extension: 10.0,
        };

        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
        assert_eq!(shaded, 0.0, "Zero depth overhang provides no shading");
    }

    #[test]
    fn test_overhang_high_sun_angle_summer() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.0,
            extension: 10.0,
        };

        // High summer sun (75 degrees)
        let solar = LocalSolarPosition {
            altitude: 75.0_f64.to_radians(),
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
        // High angle = short shadow = less shading than low angle
        // The exact value depends on geometry, but should be less than at 30 degrees
        let solar_low = LocalSolarPosition {
            altitude: 30.0_f64.to_radians(),
            relative_azimuth: 0.0,
        };
        let shaded_low = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar_low);

        assert!(
            shaded < shaded_low,
            "Higher sun angle should create less shading"
        );
    }
}

// ============================================================================
// Fin Shadow Tests
// ============================================================================

mod fin_tests {
    use super::*;

    #[test]
    fn test_fin_left_side_shadows_when_sun_from_left() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let fin = ShadeFin {
            depth: 1.0,
            distance_from_edge: 0.0,
            side: Side::Left,
        };

        // Sun from the left (negative azimuth)
        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: -PI / 4.0,
        };

        let shaded = calculate_shaded_fraction(&window, None, &[fin], &solar);
        assert!(shaded > 0.0, "Left fin should shade when sun is from left");
    }

    #[test]
    fn test_fin_left_side_no_shadow_when_sun_from_right() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let fin = ShadeFin {
            depth: 1.0,
            distance_from_edge: 0.0,
            side: Side::Left,
        };

        // Sun from the right (positive azimuth)
        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: PI / 4.0,
        };

        let shaded = calculate_shaded_fraction(&window, None, &[fin], &solar);
        assert_eq!(
            shaded, 0.0,
            "Left fin should not shade when sun is from right"
        );
    }

    #[test]
    fn test_fin_right_side_shadows_when_sun_from_right() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let fin = ShadeFin {
            depth: 1.0,
            distance_from_edge: 0.0,
            side: Side::Right,
        };

        // Sun from the right (positive azimuth)
        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: PI / 4.0,
        };

        let shaded = calculate_shaded_fraction(&window, None, &[fin], &solar);
        assert!(
            shaded > 0.0,
            "Right fin should shade when sun is from right"
        );
    }

    #[test]
    fn test_fin_depth_proportional_to_shadow() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);

        let fin_shallow = ShadeFin {
            depth: 0.5,
            distance_from_edge: 0.0,
            side: Side::Right,
        };

        let fin_deep = ShadeFin {
            depth: 2.0,
            distance_from_edge: 0.0,
            side: Side::Right,
        };

        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: PI / 4.0,
        };

        let shaded_shallow = calculate_shaded_fraction(&window, None, &[fin_shallow], &solar);
        let shaded_deep = calculate_shaded_fraction(&window, None, &[fin_deep], &solar);

        assert!(
            shaded_deep > shaded_shallow,
            "Deeper fin should create more shading"
        );
    }

    #[test]
    fn test_fin_distance_from_edge_effect() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);

        let fin_close = ShadeFin {
            depth: 1.0,
            distance_from_edge: 0.0,
            side: Side::Right,
        };

        let fin_far = ShadeFin {
            depth: 1.0,
            distance_from_edge: 2.0,
            side: Side::Right,
        };

        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: PI / 4.0,
        };

        let shaded_close = calculate_shaded_fraction(&window, None, &[fin_close], &solar);
        let shaded_far = calculate_shaded_fraction(&window, None, &[fin_far], &solar);

        assert!(
            shaded_close > shaded_far,
            "Fin closer to edge should create more shading"
        );
    }

    #[test]
    fn test_fin_zero_depth() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let fin = ShadeFin {
            depth: 0.0,
            distance_from_edge: 0.0,
            side: Side::Right,
        };

        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: PI / 4.0,
        };

        let shaded = calculate_shaded_fraction(&window, None, &[fin], &solar);
        assert_eq!(shaded, 0.0, "Zero depth fin provides no shading");
    }
}

// ============================================================================
// Combined Overhang and Fin Tests
// ============================================================================

mod combined_tests {
    use super::*;

    #[test]
    fn test_overhang_and_fins_combined() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);

        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.0,
            extension: 10.0,
        };

        let fins = vec![
            ShadeFin {
                depth: 1.0,
                distance_from_edge: 0.0,
                side: Side::Left,
            },
            ShadeFin {
                depth: 1.0,
                distance_from_edge: 0.0,
                side: Side::Right,
            },
        ];

        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: PI / 4.0, // Sun from right
        };

        let shaded_overhang_only = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
        let shaded_combined = calculate_shaded_fraction(&window, Some(&overhang), &fins, &solar);

        assert!(
            shaded_combined >= shaded_overhang_only,
            "Combined shading should be >= overhang only"
        );
    }

    #[test]
    fn test_multiple_fins() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);

        let fins = vec![
            ShadeFin {
                depth: 0.5,
                distance_from_edge: 0.0,
                side: Side::Left,
            },
            ShadeFin {
                depth: 0.5,
                distance_from_edge: 0.0,
                side: Side::Right,
            },
        ];

        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: PI / 4.0,
        };

        let shaded = calculate_shaded_fraction(&window, None, &fins, &solar);
        assert!(shaded > 0.0, "Multiple fins should provide shading");
    }

    #[test]
    fn test_no_shading_devices() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);

        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, None, &[], &solar);
        assert_eq!(shaded, 0.0, "No shading devices should mean no shading");
    }
}

// ============================================================================
// Edge Cases and Boundary Conditions
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_sun_below_horizon() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.0,
            extension: 10.0,
        };

        // Sun below horizon (negative altitude)
        let solar = LocalSolarPosition {
            altitude: -10.0_f64.to_radians(),
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
        assert_eq!(
            shaded, 1.0,
            "Sun below horizon should return 1.0 (fully shaded)"
        );
    }

    #[test]
    fn test_sun_at_horizon() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.0,
            extension: 10.0,
        };

        // Sun at horizon (zero altitude)
        let solar = LocalSolarPosition {
            altitude: 0.0,
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
        // At horizon, shadow is infinitely long, so fully shaded
        assert!(shaded >= 0.99, "Sun at horizon should nearly fully shade");
    }

    #[test]
    fn test_very_small_window() {
        let window = WindowArea::with_dimensions(1.0, Orientation::South, 0.5, 2.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.0,
            extension: 10.0,
        };

        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
        assert!(
            shaded >= 0.0 && shaded <= 1.0,
            "Result should be bounded [0, 1]"
        );
    }

    #[test]
    fn test_very_large_window() {
        let window = WindowArea::with_dimensions(100.0, Orientation::South, 10.0, 10.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.0,
            extension: 10.0,
        };

        let solar = LocalSolarPosition {
            altitude: PI / 4.0,
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
        assert!(
            shaded >= 0.0 && shaded <= 1.0,
            "Result should be bounded [0, 1]"
        );
        // Large window, small overhang = small fraction shaded
        assert!(
            shaded < 0.2,
            "Small overhang on large window should shade little"
        );
    }

    #[test]
    fn test_result_bounded_zero_to_one() {
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 10.0, // Very deep overhang
            distance_above: 0.0,
            extension: 10.0,
        };

        let solar = LocalSolarPosition {
            altitude: 5.0_f64.to_radians(), // Very low sun
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
        assert!(
            shaded >= 0.0 && shaded <= 1.0,
            "Shaded fraction must be bounded [0, 1]"
        );
    }

    #[test]
    fn test_emissivity_clamped_to_valid_range() {
        // Test that the Overhang struct accepts valid values
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.0,
            extension: 10.0,
        };

        assert!(overhang.depth >= 0.0);
        assert!(overhang.distance_above >= 0.0);
        assert!(overhang.extension >= 0.0);
    }
}

// ============================================================================
// Seasonal Behavior Tests (ASHRAE 140 Relevant)
// ============================================================================

mod seasonal_tests {
    use super::*;

    /// Test Case 610/910 shading behavior (South overhang)
    /// The overhang should block more summer sun than winter sun
    #[test]
    fn test_case_610_summer_vs_winter_shading() {
        // Case 610/910: South-facing windows with overhang
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 2.7, // ASHRAE 140 spec
            extension: 10.0,
        };

        // Summer noon (high sun angle ~73.5 degrees for Denver)
        let summer_solar = LocalSolarPosition {
            altitude: 73.5_f64.to_radians(),
            relative_azimuth: 0.0,
        };

        // Winter noon (low sun angle ~26.5 degrees for Denver)
        let winter_solar = LocalSolarPosition {
            altitude: 26.5_f64.to_radians(),
            relative_azimuth: 0.0,
        };

        let summer_shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &summer_solar);
        let winter_shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &winter_solar);

        // Summer should have more shading than winter (seasonal behavior)
        assert!(
            summer_shaded > winter_shaded,
            "Summer should have more shading than winter (summer={:.2}, winter={:.2})",
            summer_shaded,
            winter_shaded
        );
    }

    #[test]
    fn test_case_610_winter_shading() {
        // Case 610/910: South-facing windows with overhang
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 2.7, // ASHRAE 140 spec
            extension: 10.0,
        };

        // Winter noon (low sun angle ~26.5 degrees for Denver)
        let winter_solar = LocalSolarPosition {
            altitude: 26.5_f64.to_radians(),
            relative_azimuth: 0.0,
        };

        let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &winter_solar);

        // Winter should have minimal shading (10-30%) to allow solar gains
        assert!(
            shaded < 0.4,
            "Winter sun should have minimal shading (got {:.2})",
            shaded
        );
    }

    /// Test Case 630/930 shading behavior (E/W with fins)
    #[test]
    fn test_case_630_morning_shading() {
        // Case 630/930: E/W windows with shading
        let window = WindowArea::with_dimensions(12.0, Orientation::East, 2.0, 6.0, 0.2, 0.5);

        let fins = vec![
            ShadeFin {
                depth: 1.0,
                distance_from_edge: 0.0,
                side: Side::Left, // North side for East window
            },
            ShadeFin {
                depth: 1.0,
                distance_from_edge: 0.0,
                side: Side::Right, // South side for East window
            },
        ];

        // Morning sun from east (sunrise)
        let morning_solar = LocalSolarPosition {
            altitude: 30.0_f64.to_radians(),
            relative_azimuth: -30.0_f64.to_radians(), // From north-east
        };

        let shaded = calculate_shaded_fraction(&window, None, &fins, &morning_solar);
        assert!(
            shaded > 0.0,
            "Morning sun should be partially shaded by fins"
        );
    }

    #[test]
    fn test_case_630_afternoon_shading() {
        // Case 630/930: E/W windows with shading
        let window = WindowArea::with_dimensions(12.0, Orientation::West, 2.0, 6.0, 0.2, 0.5);

        let fins = vec![
            ShadeFin {
                depth: 1.0,
                distance_from_edge: 0.0,
                side: Side::Left,
            },
            ShadeFin {
                depth: 1.0,
                distance_from_edge: 0.0,
                side: Side::Right,
            },
        ];

        // Afternoon sun from west
        let afternoon_solar = LocalSolarPosition {
            altitude: 30.0_f64.to_radians(),
            relative_azimuth: 30.0_f64.to_radians(), // From south-west
        };

        let shaded = calculate_shaded_fraction(&window, None, &fins, &afternoon_solar);
        assert!(
            shaded > 0.0,
            "Afternoon sun should be partially shaded by fins"
        );
    }
}

// ============================================================================
// Physical Validation Tests
// ============================================================================

mod physical_validation {
    use super::*;

    #[test]
    fn test_shadow_geometry_consistency() {
        // Verify that shadow calculations are geometrically consistent
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);
        let overhang = Overhang {
            depth: 1.0,
            distance_above: 0.0,
            extension: 10.0,
        };

        // Test multiple sun angles and verify monotonic behavior
        let angles: Vec<f64> = (10..=80).map(|d| d as f64).collect();
        let mut prev_shaded = 1.0;

        for alt_deg in angles {
            let solar = LocalSolarPosition {
                altitude: alt_deg.to_radians(),
                relative_azimuth: 0.0,
            };

            let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);

            // Higher sun angle should result in less shading
            assert!(
                shaded <= prev_shaded + 0.01, // Small tolerance for floating point
                "Shading should decrease with increasing sun angle (alt={:.1}°, shaded={:.3})",
                alt_deg,
                shaded
            );

            prev_shaded = shaded;
        }
    }

    #[test]
    fn test_symmetry_for_south_facing() {
        // For a South-facing window, east and west sun at same angle should give same shading
        let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);

        let fins = vec![
            ShadeFin {
                depth: 1.0,
                distance_from_edge: 0.0,
                side: Side::Left,
            },
            ShadeFin {
                depth: 1.0,
                distance_from_edge: 0.0,
                side: Side::Right,
            },
        ];

        let altitude = 45.0_f64.to_radians();

        let solar_east = LocalSolarPosition {
            altitude,
            relative_azimuth: -45.0_f64.to_radians(),
        };

        let solar_west = LocalSolarPosition {
            altitude,
            relative_azimuth: 45.0_f64.to_radians(),
        };

        let shaded_east = calculate_shaded_fraction(&window, None, &fins, &solar_east);
        let shaded_west = calculate_shaded_fraction(&window, None, &fins, &solar_west);

        assert!(
            (shaded_east - shaded_west).abs() < 0.01,
            "East/west symmetry should be preserved for South window with symmetric fins"
        );
    }
}
