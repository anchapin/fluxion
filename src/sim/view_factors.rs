//! Radiative view factor calculations for inter-zone heat transfer.
//!
//! This module provides functions for computing geometric view factors between surfaces,
//! particularly for radiative exchange between zones in building energy simulations.
//!
//! For windows on a common wall (directly opposite each other), the view factor is 1.0
//! under the assumption that the windows are perfectly aligned and the wall thickness
//! is negligible. This is more accurate than area-weighted approximations for typical
//! configurations.
//!
//! For non-aligned windows or more complex geometries, additional geometric calculations
//! would be required (e.g., using shape factors or Monte Carlo integration).

/// Returns the view factor between two windows on a common wall.
///
/// For windows directly opposite each other with negligible wall thickness, the view factor
/// is effectively 1.0. This assumes:
/// - Windows are perfectly aligned and face each other across the opening
/// - Wall thickness is negligible compared to window dimensions
/// - No obstructions between the windows
///
/// This is more accurate than the previous area-weighted approximation
/// `(A_common / A_zone1) * (A_common / A_zone2)` for typical configurations.
///
/// # Arguments
/// * `window_area` - Area of the window through which radiation passes (m²)
///
/// # Returns
/// View factor (dimensionless, 0.0 to 1.0). Always returns 1.0 under current assumptions.
///
/// # Future Work
/// For non-aligned windows or configurations with significant wall thickness, a more
/// complex geometric calculation could be added, such as:
/// - Hottel's crossed-string method for rectangular surfaces
/// - Nusselt's analog for parallel surfaces
/// - Numerical integration or view factor libraries
pub fn window_to_window_view_factor(_window_area: f64) -> f64 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_to_window_view_factor() {
        let area = 10.8;
        let f = window_to_window_view_factor(area);
        assert_eq!(f, 1.0);
    }
}
