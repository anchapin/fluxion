#[cfg(test)]
mod tests {
    use fluxion::weather::interpolation::{
        interpolate_weather, linear_interpolate, piecewise_hermite_interpolate,
        select_method_for_field, step_interpolate, InterpolationMethod,
    };

    #[test]
    fn test_linear_interpolation() {
        // Test linear interpolation at midpoint
        let value = linear_interpolate(10.0, 20.0, 0.5);
        assert!((value - 15.0).abs() < 1e-10, "Linear interpolation failed");

        // Test at boundaries
        assert_eq!(linear_interpolate(10.0, 20.0, 0.0), 10.0);
        assert_eq!(linear_interpolate(10.0, 20.0, 1.0), 20.0);
    }

    #[test]
    fn test_step_interpolation() {
        // Test step interpolation (t1 for fraction < 0.5, t2 for >= 0.5)
        assert_eq!(step_interpolate(10.0, 20.0, 0.25), 10.0);
        assert_eq!(step_interpolate(10.0, 20.0, 0.5), 20.0);
        assert_eq!(step_interpolate(10.0, 20.0, 0.75), 20.0);
    }

    #[test]
    fn test_piecewise_hermite_interpolation() {
        // Test piecewise Hermite interpolation
        let value = piecewise_hermite_interpolate(10.0, 20.0, 0.5);
        // Should be smooth between 10.0 and 20.0
        assert!(
            value >= 10.0 && value <= 20.0,
            "Hermite interpolation out of range: {}",
            value
        );

        // Test at boundaries
        assert!((piecewise_hermite_interpolate(10.0, 20.0, 0.0) - 10.0).abs() < 1e-10);
        assert!((piecewise_hermite_interpolate(10.0, 20.0, 1.0) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_select_method_for_field() {
        // Test field-specific method selection
        assert_eq!(
            select_method_for_field("dry_bulb_temp"),
            InterpolationMethod::Linear
        );
        assert_eq!(
            select_method_for_field("dni"),
            InterpolationMethod::PiecewiseHermite
        );
        assert_eq!(
            select_method_for_field("present_weather"),
            InterpolationMethod::Step
        );
    }

    #[test]
    fn test_interpolate_weather_dispatch() {
        // Test interpolate_weather() dispatches to correct method
        let value = interpolate_weather(
            "dry_bulb_temp",
            10.0,
            20.0,
            0.5,
            InterpolationMethod::Linear,
        );
        assert!((value - 15.0).abs() < 1e-10);
    }
}
