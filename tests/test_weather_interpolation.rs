//! Tests for weather data interpolation functions.
//!
//! Covers all interpolation methods and field-specific method selection.

use fluxion::weather::interpolation::{
    cubic_spline_interpolate, interpolate_weather, linear_interpolate,
    piecewise_hermite_interpolate, select_method_for_field, step_interpolate, InterpolationMethod,
};

#[cfg(test)]
mod linear_interpolate_tests {
    use super::*;

    #[test]
    fn test_linear_midpoint() {
        let result = linear_interpolate(10.0, 20.0, 0.5);
        assert!((result - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_start_point() {
        let result = linear_interpolate(10.0, 20.0, 0.0);
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_end_point() {
        let result = linear_interpolate(10.0, 20.0, 1.0);
        assert!((result - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_quarter_point() {
        let result = linear_interpolate(0.0, 100.0, 0.25);
        assert!((result - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_negative_values() {
        let result = linear_interpolate(-10.0, 10.0, 0.5);
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_same_values() {
        let result = linear_interpolate(15.0, 15.0, 0.5);
        assert!((result - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_beyond_range() {
        // Extrapolation (fraction > 1.0)
        let result = linear_interpolate(10.0, 20.0, 1.5);
        assert!((result - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_before_range() {
        // Extrapolation (fraction < 0.0)
        let result = linear_interpolate(10.0, 20.0, -0.5);
        assert!((result - 5.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod piecewise_hermite_interpolate_tests {
    use super::*;

    #[test]
    fn test_hermite_start_point() {
        let result = piecewise_hermite_interpolate(10.0, 20.0, 0.0);
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_hermite_end_point() {
        let result = piecewise_hermite_interpolate(10.0, 20.0, 1.0);
        assert!((result - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_hermite_midpoint() {
        // With zero derivatives at boundaries, midpoint should be average
        let result = piecewise_hermite_interpolate(10.0, 20.0, 0.5);
        assert!((result - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_hermite_quarter_point() {
        let result = piecewise_hermite_interpolate(0.0, 100.0, 0.25);
        // h00(0.25) = 2*0.015625 - 3*0.0625 + 1 = 0.03125 - 0.1875 + 1 = 0.84375
        // h01(0.25) = -2*0.015625 + 3*0.0625 = -0.03125 + 0.1875 = 0.15625
        // result = 0.84375 * 0 + 0.15625 * 100 = 15.625
        assert!((result - 15.625).abs() < 1e-10);
    }

    #[test]
    fn test_hermite_same_values() {
        let result = piecewise_hermite_interpolate(15.0, 15.0, 0.5);
        assert!((result - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_hermite_smoothness() {
        // Verify C1 continuity - derivative should be zero at boundaries
        let eps = 0.001;
        let t1 = 10.0;
        let t2 = 20.0;

        let v_start = piecewise_hermite_interpolate(t1, t2, eps);
        let v_end = piecewise_hermite_interpolate(t1, t2, 1.0 - eps);

        // Near start, value should be close to t1
        assert!((v_start - t1).abs() < 1.0);
        // Near end, value should be close to t2
        assert!((v_end - t2).abs() < 1.0);
    }
}

#[cfg(test)]
mod step_interpolate_tests {
    use super::*;

    #[test]
    fn test_step_before_midpoint() {
        let result = step_interpolate(10.0, 20.0, 0.4);
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_after_midpoint() {
        let result = step_interpolate(10.0, 20.0, 0.6);
        assert!((result - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_at_midpoint() {
        let result = step_interpolate(10.0, 20.0, 0.5);
        assert!((result - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_at_start() {
        let result = step_interpolate(10.0, 20.0, 0.0);
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_at_end() {
        let result = step_interpolate(10.0, 20.0, 1.0);
        assert!((result - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_negative_values() {
        let result = step_interpolate(-5.0, 5.0, 0.75);
        assert!((result - 5.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod cubic_spline_interpolate_tests {
    use super::*;

    #[test]
    fn test_cubic_start_point() {
        let result = cubic_spline_interpolate(10.0, 20.0, 0.0);
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_end_point() {
        let result = cubic_spline_interpolate(10.0, 20.0, 1.0);
        assert!((result - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_midpoint() {
        // With zero derivatives, cubic should give same as linear at midpoint
        let result = cubic_spline_interpolate(10.0, 20.0, 0.5);
        assert!((result - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_same_values() {
        let result = cubic_spline_interpolate(15.0, 15.0, 0.5);
        assert!((result - 15.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod interpolate_weather_tests {
    use super::*;

    #[test]
    fn test_interpolate_weather_linear() {
        let result = interpolate_weather(
            "dry_bulb_temp",
            10.0,
            20.0,
            0.5,
            InterpolationMethod::Linear,
        );
        assert!((result - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_weather_hermite() {
        let result = interpolate_weather(
            "dni",
            100.0,
            200.0,
            0.5,
            InterpolationMethod::PiecewiseHermite,
        );
        assert!((result - 150.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_weather_step() {
        let result =
            interpolate_weather("present_weather", 0.0, 1.0, 0.3, InterpolationMethod::Step);
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_weather_cubic() {
        let result = interpolate_weather("ghi", 50.0, 150.0, 0.5, InterpolationMethod::CubicSpline);
        assert!((result - 100.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod select_method_for_field_tests {
    use super::*;

    #[test]
    fn test_temperature_fields_use_linear() {
        assert_eq!(
            select_method_for_field("dry_bulb_temp"),
            InterpolationMethod::Linear
        );
        assert_eq!(
            select_method_for_field("humidity"),
            InterpolationMethod::Linear
        );
        assert_eq!(
            select_method_for_field("dew_point"),
            InterpolationMethod::Linear
        );
    }

    #[test]
    fn test_solar_fields_use_hermite() {
        assert_eq!(
            select_method_for_field("dni"),
            InterpolationMethod::PiecewiseHermite
        );
        assert_eq!(
            select_method_for_field("dhi"),
            InterpolationMethod::PiecewiseHermite
        );
        assert_eq!(
            select_method_for_field("ghi"),
            InterpolationMethod::PiecewiseHermite
        );
        assert_eq!(
            select_method_for_field("horizontal_illuminance"),
            InterpolationMethod::PiecewiseHermite
        );
        assert_eq!(
            select_method_for_field("diffuse_illuminance"),
            InterpolationMethod::PiecewiseHermite
        );
    }

    #[test]
    fn test_discrete_fields_use_step() {
        assert_eq!(
            select_method_for_field("present_weather"),
            InterpolationMethod::Step
        );
        assert_eq!(
            select_method_for_field("present_weather_code"),
            InterpolationMethod::Step
        );
        assert_eq!(
            select_method_for_field("cloud_cover"),
            InterpolationMethod::Step
        );
        assert_eq!(
            select_method_for_field("snow_depth"),
            InterpolationMethod::Step
        );
        assert_eq!(
            select_method_for_field("snow_cover"),
            InterpolationMethod::Step
        );
    }

    #[test]
    fn test_wind_fields_use_linear() {
        assert_eq!(
            select_method_for_field("wind_speed"),
            InterpolationMethod::Linear
        );
        assert_eq!(
            select_method_for_field("wind_direction"),
            InterpolationMethod::Linear
        );
    }

    #[test]
    fn test_pressure_fields_use_linear() {
        assert_eq!(
            select_method_for_field("atmospheric_pressure"),
            InterpolationMethod::Linear
        );
    }

    #[test]
    fn test_ground_temperature_uses_linear() {
        assert_eq!(
            select_method_for_field("ground_temperature"),
            InterpolationMethod::Linear
        );
    }

    #[test]
    fn test_unknown_field_defaults_to_linear() {
        assert_eq!(
            select_method_for_field("unknown_field"),
            InterpolationMethod::Linear
        );
        assert_eq!(select_method_for_field(""), InterpolationMethod::Linear);
    }
}

#[cfg(test)]
mod interpolation_integration_tests {
    use super::*;

    #[test]
    fn test_temperature_interpolation_through_day() {
        // Simulate temperature interpolation through a day
        let temps = [10.0, 12.0, 15.0, 18.0, 20.0, 18.0, 15.0, 12.0];

        for i in 0..temps.len() - 1 {
            let t1 = temps[i];
            let t2 = temps[i + 1];

            // Test at quarter, half, and three-quarter points
            let q1 = linear_interpolate(t1, t2, 0.25);
            let mid = linear_interpolate(t1, t2, 0.5);
            let q3 = linear_interpolate(t1, t2, 0.75);

            // Values should be monotonically between t1 and t2
            let min_val = t1.min(t2);
            let max_val = t1.max(t2);
            assert!(q1 >= min_val - 1e-10 && q1 <= max_val + 1e-10);
            assert!(mid >= min_val - 1e-10 && mid <= max_val + 1e-10);
            assert!(q3 >= min_val - 1e-10 && q3 <= max_val + 1e-10);
        }
    }

    #[test]
    fn test_solar_radiation_interpolation() {
        // Solar radiation should use PiecewiseHermite
        let morning = 0.0;
        let noon = 800.0;

        let result = interpolate_weather("ghi", morning, noon, 0.5, select_method_for_field("ghi"));
        assert!(result > 0.0);
        assert!(result < noon);
    }

    #[test]
    fn test_cloud_cover_interpolation() {
        // Cloud cover should use step function
        let clear = 0.0;
        let overcast = 10.0;

        let morning = interpolate_weather(
            "cloud_cover",
            clear,
            overcast,
            0.3,
            select_method_for_field("cloud_cover"),
        );
        let afternoon = interpolate_weather(
            "cloud_cover",
            clear,
            overcast,
            0.7,
            select_method_for_field("cloud_cover"),
        );

        assert_eq!(morning, clear);
        assert_eq!(afternoon, overcast);
    }

    #[test]
    fn test_all_interpolation_methods_produce_valid_results() {
        let t1 = 10.0;
        let t2 = 20.0;
        let fraction = 0.3;

        let linear = interpolate_weather("temp", t1, t2, fraction, InterpolationMethod::Linear);
        let cubic = interpolate_weather("temp", t1, t2, fraction, InterpolationMethod::CubicSpline);
        let hermite = interpolate_weather(
            "temp",
            t1,
            t2,
            fraction,
            InterpolationMethod::PiecewiseHermite,
        );
        let step = interpolate_weather("temp", t1, t2, fraction, InterpolationMethod::Step);

        // All should produce finite values
        assert!(linear.is_finite());
        assert!(cubic.is_finite());
        assert!(hermite.is_finite());
        assert!(step.is_finite());

        // Step should return t1 (fraction < 0.5)
        assert_eq!(step, t1);

        // Linear, cubic, and hermite should all be between t1 and t2
        assert!(linear >= t1 && linear <= t2);
        assert!(cubic >= t1 && cubic <= t2);
        assert!(hermite >= t1 && hermite <= t2);
    }
}
