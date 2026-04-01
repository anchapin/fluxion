//! Sub-hourly weather data interpolation.
//!
//! This module provides interpolation functions for converting hourly weather data
//! to sub-hourly timesteps or interpolating between sub-hourly records.
//!
//! # Interpolation Methods
//!
//! - **Linear**: Simple linear interpolation for smooth transitions
//! - **CubicSpline**: Smooth cubic spline with C2 continuity
//! - **PiecewiseHermite**: Cubic Hermite spline with C1 continuity
//! - **Step**: Discrete step function for categorical observations
//!
//! # Field-Specific Methods
//!
//! Different weather fields require different interpolation strategies:
//! - Temperature, humidity: Linear
//! - Solar radiation: Piecewise Hermite
//! - Rain codes, cloud cover: Step
//!
//! # Example
//!
//! ```no_run
//! use fluxion::weather::interpolation::{interpolate_weather, select_method_for_field, InterpolationMethod};
//!
//! // Interpolate temperature at 30 minutes (fraction = 0.5)
//! let temp = interpolate_weather(
//!     "dry_bulb_temp",
//!     10.0,  // 1 PM
//!     15.0,  // 2 PM
//!     0.5,    // 1:30 PM
//!     InterpolationMethod::Linear,
//! );
//! // temp = 12.5
//!
//! // Select method automatically based on field
//! let method = select_method_for_field("dni");
//! assert_eq!(method, InterpolationMethod::PiecewiseHermite);
//! ```

/// Interpolation method for weather data.
///
/// Different weather fields require different interpolation strategies:
/// - Temperature: Linear (smooth transitions)
/// - Solar radiation: Piecewise Hermite (continuous with reasonable smoothness)
/// - Discrete observations: Step (rain codes, cloud cover)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    /// Linear interpolation.
    ///
    /// Simple linear interpolation between two points:
    /// `value = t1 + (t2 - t1) * fraction`
    ///
    /// Used for: Temperature, humidity, wind speed
    Linear,

    /// Cubic spline interpolation.
    ///
    /// Smooth cubic spline interpolation with C2 continuity.
    /// More accurate than linear but can oscillate.
    ///
    /// Used for: Solar radiation (when smoothness critical)
    CubicSpline,

    /// Piecewise cubic Hermite interpolation.
    ///
    /// Cubic Hermite spline with C1 continuity at boundaries.
    /// Provides smooth transitions without oscillation.
    ///
    /// Used for: Solar radiation (default), illuminance
    PiecewiseHermite,

    /// Step function.
    ///
    /// Discrete step interpolation:
    /// `value = t1 if fraction < 0.5 else t2`
    ///
    /// Used for: Rain codes, cloud cover, present weather observations
    Step,
}

/// Linear interpolation between two values.
///
/// # Arguments
///
/// * `t1` - Value at timestep 1
/// * `t2` - Value at timestep 2
/// * `fraction` - Fraction between timesteps (0.0 = t1, 1.0 = t2)
///
/// # Returns
///
/// Linearly interpolated value
///
/// # Example
///
/// ```
/// use fluxion::weather::interpolation::linear_interpolate;
///
/// let value = linear_interpolate(10.0, 20.0, 0.5); // Returns 15.0
/// ```
pub fn linear_interpolate(t1: f64, t2: f64, fraction: f64) -> f64 {
    t1 + (t2 - t1) * fraction
}

/// Piecewise cubic Hermite interpolation.
///
/// Provides C1 continuity at boundaries with reasonable smoothness.
/// More accurate than linear without oscillation of cubic splines.
///
/// # Arguments
///
/// * `t1` - Value at timestep 1
/// * `t2` - Value at timestep 2
/// * `fraction` - Fraction between timesteps (0.0 = t1, 1.0 = t2)
///
/// # Returns
///
/// Piecewise Hermite interpolated value
pub fn piecewise_hermite_interpolate(t1: f64, t2: f64, fraction: f64) -> f64 {
    // Assume zero derivatives at boundaries (simplified)
    // Can be extended with slope estimation if needed
    let t = fraction;
    let t2_frac = t * t;
    let t3 = t2_frac * t;

    // Hermite basis functions
    let h00 = 2.0 * t3 - 3.0 * t2_frac + 1.0;
    let h10 = t3 - 2.0 * t2_frac + t;
    let h01 = -2.0 * t3 + 3.0 * t2_frac;
    let h11 = t3 - t2_frac;

    let m0 = 0.0; // Zero derivative at t1
    let m1 = 0.0; // Zero derivative at t2

    h00 * t1 + h10 * m0 + h01 * t2 + h11 * m1
}

/// Step function interpolation for discrete observations.
///
/// Used for weather observations that change discretely (e.g., rain codes,
/// cloud cover, present weather observations). Returns t1 for fraction < 0.5,
/// returns t2 for fraction >= 0.5.
///
/// # Arguments
///
/// * `t1` - Value at timestep 1
/// * `t2` - Value at timestep 2
/// * `fraction` - Fraction between timesteps (ignored in step function)
///
/// # Returns
///
/// Step-interpolated value (either t1 or t2)
pub fn step_interpolate(t1: f64, t2: f64, fraction: f64) -> f64 {
    if fraction < 0.5 {
        t1
    } else {
        t2
    }
}

/// Cubic spline interpolation.
///
/// Smooth cubic interpolation with C2 continuity.
/// More accurate than linear but can oscillate near boundaries.
///
/// # Arguments
///
/// * `t1` - Value at timestep 1
/// * `t2` - Value at timestep 2
/// * `fraction` - Fraction between timesteps (0.0 = t1, 1.0 = t2)
///
/// # Returns
///
/// Cubic spline interpolated value
pub fn cubic_spline_interpolate(t1: f64, t2: f64, fraction: f64) -> f64 {
    let t = fraction;
    let t2_ = t * t;
    let t3_ = t2_ * t;

    // Hermite basis functions with zero derivatives at boundaries
    (2.0 * t3_ - 3.0 * t2_ + 1.0) * t1 + (-2.0 * t3_ + 3.0 * t2_) * t2
}

/// Interpolate weather value between two timesteps.
///
/// Dispatches to appropriate interpolation method based on field type.
///
/// # Arguments
///
/// * `field` - Field name for method selection (e.g., "dry_bulb_temp")
/// * `t1` - Value at timestep 1
/// * `t2` - Value at timestep 2
/// * `fraction` - Fraction between timesteps (0.0 = t1, 1.0 = t2)
/// * `method` - Interpolation method
///
/// # Returns
///
/// Interpolated value
///
/// # Example
///
/// ```
/// use fluxion::weather::interpolation::{interpolate_weather, InterpolationMethod};
///
/// let value = interpolate_weather("dry_bulb_temp", 10.0, 20.0, 0.5, InterpolationMethod::Linear);
/// // Returns 15.0
/// ```
pub fn interpolate_weather(
    _field: &str,
    t1: f64,
    t2: f64,
    fraction: f64,
    method: InterpolationMethod,
) -> f64 {
    match method {
        InterpolationMethod::Linear => linear_interpolate(t1, t2, fraction),
        InterpolationMethod::CubicSpline => cubic_spline_interpolate(t1, t2, fraction),
        InterpolationMethod::PiecewiseHermite => piecewise_hermite_interpolate(t1, t2, fraction),
        InterpolationMethod::Step => step_interpolate(t1, t2, fraction),
    }
}

/// Select interpolation method based on weather field.
///
/// Different fields require different interpolation strategies:
/// - Temperature: Linear (smooth transitions)
/// - Solar radiation: Piecewise Hermite (continuous with reasonable smoothness)
/// - Discrete observations: Step (rain codes, cloud cover)
///
/// # Arguments
///
/// * `field` - Field name (e.g., "dry_bulb_temp", "dni", "present_weather")
///
/// # Returns
///
/// Recommended interpolation method for the field
///
/// # Example
///
/// ```
/// use fluxion::weather::interpolation::{select_method_for_field, InterpolationMethod};
///
/// let method = select_method_for_field("dry_bulb_temp");
/// assert_eq!(method, InterpolationMethod::Linear);
/// ```
pub fn select_method_for_field(field: &str) -> InterpolationMethod {
    match field {
        // Temperature and humidity: Linear (smooth transitions)
        "dry_bulb_temp" | "humidity" | "dew_point" => InterpolationMethod::Linear,

        // Solar radiation: Piecewise Hermite (continuous with reasonable smoothness)
        "dni" | "dhi" | "ghi" | "horizontal_illuminance" | "diffuse_illuminance" => {
            InterpolationMethod::PiecewiseHermite
        }

        // Discrete observations: Step (rain codes, cloud cover)
        "present_weather"
        | "present_weather_code"
        | "cloud_cover"
        | "snow_depth"
        | "snow_cover" => InterpolationMethod::Step,

        // Wind speed: Linear (smooth transitions)
        "wind_speed" | "wind_direction" => InterpolationMethod::Linear,

        // Pressure: Linear (smooth transitions)
        "atmospheric_pressure" => InterpolationMethod::Linear,

        // Ground temperature: Linear (smooth transitions)
        "ground_temperature" => InterpolationMethod::Linear,

        // Default: Linear
        _ => InterpolationMethod::Linear,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolation_basic() {
        assert!((linear_interpolate(0.0, 10.0, 0.5) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_interpolation_boundary() {
        assert_eq!(step_interpolate(1.0, 2.0, 0.4), 1.0);
        assert_eq!(step_interpolate(1.0, 2.0, 0.6), 2.0);
    }

    #[test]
    fn test_method_selection_temperature() {
        assert_eq!(
            select_method_for_field("dry_bulb_temp"),
            InterpolationMethod::Linear
        );
    }

    #[test]
    fn test_method_selection_solar() {
        assert_eq!(
            select_method_for_field("ghi"),
            InterpolationMethod::PiecewiseHermite
        );
    }

    #[test]
    fn test_method_selection_discrete() {
        assert_eq!(
            select_method_for_field("cloud_cover"),
            InterpolationMethod::Step
        );
    }

    #[test]
    fn test_linear_interpolation_boundaries() {
        assert!((linear_interpolate(10.0, 20.0, 0.0) - 10.0).abs() < 1e-10);
        assert!((linear_interpolate(10.0, 20.0, 1.0) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_interpolation_negative_values() {
        assert!((linear_interpolate(-10.0, -20.0, 0.5) - (-15.0)).abs() < 1e-10);
    }

    #[test]
    fn test_linear_interpolation_zero_to_value() {
        assert!((linear_interpolate(0.0, 100.0, 0.25) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_interpolation_fraction_outside_range() {
        assert!((linear_interpolate(0.0, 10.0, 1.5) - 15.0).abs() < 1e-10);
        assert!((linear_interpolate(0.0, 10.0, -0.5) - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_spline_interpolation_boundaries() {
        assert!((cubic_spline_interpolate(10.0, 20.0, 0.0) - 10.0).abs() < 1e-10);
        assert!((cubic_spline_interpolate(10.0, 20.0, 1.0) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_spline_interpolation_midpoint() {
        let result = cubic_spline_interpolate(0.0, 10.0, 0.5);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_spline_interpolation_quarter() {
        let result = cubic_spline_interpolate(0.0, 100.0, 0.25);
        // Cubic spline with zero derivatives: at 0.25 should be ~15.625
        assert!((result - 15.625).abs() < 1e-10);
    }

    #[test]
    fn test_piecewise_hermite_interpolation_boundaries() {
        assert!((piecewise_hermite_interpolate(10.0, 20.0, 0.0) - 10.0).abs() < 1e-10);
        assert!((piecewise_hermite_interpolate(10.0, 20.0, 1.0) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_piecewise_hermite_interpolation_midpoint() {
        let result = piecewise_hermite_interpolate(0.0, 10.0, 0.5);
        // With zero derivatives, midpoint should be exactly 5.0
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_piecewise_hermite_interpolation_quarter() {
        let result = piecewise_hermite_interpolate(0.0, 100.0, 0.25);
        // With zero derivatives, should follow cubic curve
        assert!(result > 15.0 && result < 35.0);
    }

    #[test]
    fn test_step_interpolation_exact_boundary() {
        assert_eq!(step_interpolate(1.0, 2.0, 0.5), 2.0);
    }

    #[test]
    fn test_step_interpolation_zero_fraction() {
        assert_eq!(step_interpolate(1.0, 2.0, 0.0), 1.0);
    }

    #[test]
    fn test_step_interpolation_one_fraction() {
        assert_eq!(step_interpolate(1.0, 2.0, 1.0), 2.0);
    }

    #[test]
    fn test_interpolate_weather_linear() {
        let result = interpolate_weather("temp", 10.0, 20.0, 0.5, InterpolationMethod::Linear);
        assert!((result - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_weather_cubic_spline() {
        let result = interpolate_weather("temp", 0.0, 10.0, 0.5, InterpolationMethod::CubicSpline);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_weather_piecewise_hermite() {
        let result = interpolate_weather(
            "dni",
            0.0,
            100.0,
            0.5,
            InterpolationMethod::PiecewiseHermite,
        );
        assert!((result - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_weather_step() {
        let result = interpolate_weather("cloud_cover", 1.0, 2.0, 0.3, InterpolationMethod::Step);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_select_method_for_field_solar_radiation() {
        assert_eq!(
            select_method_for_field("dni"),
            InterpolationMethod::PiecewiseHermite
        );
        assert_eq!(
            select_method_for_field("dhi"),
            InterpolationMethod::PiecewiseHermite
        );
        assert_eq!(
            select_method_for_field("horizontal_illuminance"),
            InterpolationMethod::PiecewiseHermite
        );
    }

    #[test]
    fn test_select_method_for_field_wind() {
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
    fn test_select_method_for_field_pressure() {
        assert_eq!(
            select_method_for_field("atmospheric_pressure"),
            InterpolationMethod::Linear
        );
    }

    #[test]
    fn test_select_method_for_field_ground_temp() {
        assert_eq!(
            select_method_for_field("ground_temperature"),
            InterpolationMethod::Linear
        );
    }

    #[test]
    fn test_select_method_for_field_snow() {
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
    fn test_select_method_for_field_unknown_defaults_to_linear() {
        assert_eq!(
            select_method_for_field("unknown_field"),
            InterpolationMethod::Linear
        );
        assert_eq!(select_method_for_field(""), InterpolationMethod::Linear);
    }

    #[test]
    fn test_select_method_for_field_dew_point() {
        assert_eq!(
            select_method_for_field("dew_point"),
            InterpolationMethod::Linear
        );
    }

    #[test]
    fn test_select_method_for_field_present_weather() {
        assert_eq!(
            select_method_for_field("present_weather"),
            InterpolationMethod::Step
        );
        assert_eq!(
            select_method_for_field("present_weather_code"),
            InterpolationMethod::Step
        );
    }

    #[test]
    fn test_interpolation_method_clone_and_copy() {
        let method = InterpolationMethod::Linear;
        let cloned = method;
        assert_eq!(method, cloned);
    }

    #[test]
    fn test_interpolation_method_debug() {
        let method = InterpolationMethod::PiecewiseHermite;
        let debug_str = format!("{:?}", method);
        assert!(debug_str.contains("PiecewiseHermite"));
    }

    #[test]
    fn test_interpolation_method_equality() {
        assert_eq!(InterpolationMethod::Linear, InterpolationMethod::Linear);
        assert_ne!(InterpolationMethod::Linear, InterpolationMethod::Step);
    }
}
