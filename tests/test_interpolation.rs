#[cfg(test)]
mod tests {
    use fluxion::weather::interpolation::{
        cubic_spline_interpolate, interpolate_weather, linear_interpolate,
        piecewise_hermite_interpolate, select_method_for_field, step_interpolate,
        InterpolationMethod,
    };

    #[test]
    fn test_linear_interpolation_midpoint() {
        let value = linear_interpolate(10.0, 20.0, 0.5);
        assert!((value - 15.0).abs() < 1e-10, "Linear interpolation failed");
    }

    #[test]
    fn test_linear_interpolation_boundaries() {
        assert_eq!(linear_interpolate(10.0, 20.0, 0.0), 10.0);
        assert_eq!(linear_interpolate(10.0, 20.0, 1.0), 20.0);
    }

    #[test]
    fn test_linear_interpolation_quarters() {
        assert!((linear_interpolate(0.0, 100.0, 0.25) - 25.0).abs() < 1e-10);
        assert!((linear_interpolate(0.0, 100.0, 0.75) - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_interpolation_negative() {
        assert!((linear_interpolate(-10.0, 10.0, 0.5) - 0.0).abs() < 1e-10);
        assert!((linear_interpolate(-20.0, -10.0, 0.5) - (-15.0)).abs() < 1e-10);
    }

    #[test]
    fn test_linear_interpolation_same_values() {
        assert_eq!(linear_interpolate(5.0, 5.0, 0.5), 5.0);
    }

    #[test]
    fn test_linear_interpolation_decreasing() {
        assert!((linear_interpolate(20.0, 10.0, 0.5) - 15.0).abs() < 1e-10);
        assert_eq!(linear_interpolate(100.0, 0.0, 0.5), 50.0);
    }

    #[test]
    fn test_step_interpolation_first_half() {
        assert_eq!(step_interpolate(10.0, 20.0, 0.0), 10.0);
        assert_eq!(step_interpolate(10.0, 20.0, 0.25), 10.0);
        assert_eq!(step_interpolate(10.0, 20.0, 0.49), 10.0);
    }

    #[test]
    fn test_step_interpolation_second_half() {
        assert_eq!(step_interpolate(10.0, 20.0, 0.5), 20.0);
        assert_eq!(step_interpolate(10.0, 20.0, 0.75), 20.0);
        assert_eq!(step_interpolate(10.0, 20.0, 1.0), 20.0);
    }

    #[test]
    fn test_step_interpolation_boundary() {
        assert_eq!(step_interpolate(10.0, 20.0, 0.5), 20.0);
    }

    #[test]
    fn test_piecewise_hermite_boundaries() {
        assert!((piecewise_hermite_interpolate(10.0, 20.0, 0.0) - 10.0).abs() < 1e-10);
        assert!((piecewise_hermite_interpolate(10.0, 20.0, 1.0) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_piecewise_hermite_midpoint() {
        let value = piecewise_hermite_interpolate(10.0, 20.0, 0.5);
        assert!(
            (10.0..=20.0).contains(&value),
            "Hermite interpolation out of range: {}",
            value
        );
    }

    #[test]
    fn test_piecewise_hermite_same_values() {
        assert_eq!(piecewise_hermite_interpolate(5.0, 5.0, 0.5), 5.0);
    }

    #[test]
    fn test_piecewise_hermite_negative() {
        let value = piecewise_hermite_interpolate(-10.0, 10.0, 0.5);
        assert!((-10.0..=10.0).contains(&value));
    }

    #[test]
    fn test_cubic_spline_boundaries() {
        assert!((cubic_spline_interpolate(10.0, 20.0, 0.0) - 10.0).abs() < 1e-10);
        assert!((cubic_spline_interpolate(10.0, 20.0, 1.0) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_spline_midpoint() {
        let value = cubic_spline_interpolate(10.0, 20.0, 0.5);
        assert!((10.0..=20.0).contains(&value));
    }

    #[test]
    fn test_cubic_spline_same_values() {
        assert_eq!(cubic_spline_interpolate(5.0, 5.0, 0.5), 5.0);
    }

    #[test]
    fn test_select_method_temperature() {
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
    fn test_select_method_solar() {
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
    }

    #[test]
    fn test_select_method_discrete() {
        assert_eq!(
            select_method_for_field("present_weather"),
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
    }

    #[test]
    fn test_select_method_wind() {
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
    fn test_select_method_pressure() {
        assert_eq!(
            select_method_for_field("atmospheric_pressure"),
            InterpolationMethod::Linear
        );
    }

    #[test]
    fn test_select_method_ground() {
        assert_eq!(
            select_method_for_field("ground_temperature"),
            InterpolationMethod::Linear
        );
    }

    #[test]
    fn test_select_method_unknown_defaults_to_linear() {
        assert_eq!(
            select_method_for_field("unknown_field"),
            InterpolationMethod::Linear
        );
    }

    #[test]
    fn test_interpolate_weather_linear() {
        let value = interpolate_weather(
            "dry_bulb_temp",
            10.0,
            20.0,
            0.5,
            InterpolationMethod::Linear,
        );
        assert!((value - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_weather_hermite() {
        let value = interpolate_weather(
            "dni",
            100.0,
            200.0,
            0.5,
            InterpolationMethod::PiecewiseHermite,
        );
        assert!((100.0..=200.0).contains(&value));
    }

    #[test]
    fn test_interpolate_weather_step() {
        let value = interpolate_weather("cloud_cover", 5.0, 8.0, 0.3, InterpolationMethod::Step);
        assert_eq!(value, 5.0);
    }

    #[test]
    fn test_interpolation_method_equality() {
        assert_eq!(InterpolationMethod::Linear, InterpolationMethod::Linear);
        assert_ne!(InterpolationMethod::Linear, InterpolationMethod::Step);
    }

    #[test]
    fn test_interpolation_method_clone_copy() {
        let method = InterpolationMethod::PiecewiseHermite;
        let cloned = method;
        assert_eq!(method, cloned);
    }

    #[test]
    fn test_interpolation_method_debug() {
        let method = InterpolationMethod::CubicSpline;
        let debug_str = format!("{:?}", method);
        assert!(debug_str.contains("CubicSpline"));
    }
}
