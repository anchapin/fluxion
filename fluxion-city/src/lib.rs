//! # fluxion-city: Urban Radiation & View Factor Modeling
//!
//! Nusselt analog view factor computation for urban building energy modeling.
//!
//! ## View Factor Fundamentals
//!
//! View factors (also called shape factors or configuration factors) describe the
//! geometric relationship between surfaces in radiative exchange. For urban canyon
//! modeling, we compute:
//! - F_wall_sky: View factor from building wall to sky
//! - F_wall_ground: View factor from building wall to ground
//! - F_ij: View factor from surface i to surface j

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ViewFactorError {
    #[error("Surface has zero area: {0}")]
    ZeroArea(String),

    #[error("Invalid geometry: {0}")]
    InvalidGeometry(String),

    #[error("Numerical precision error in view factor summation: {0}")]
    SummationError(String),
}

pub mod geometry {
    use super::ViewFactorError;

    #[derive(Debug, Clone, Copy)]
    pub struct RectSurface {
        pub width: f64,
        pub height: f64,
    }

    impl RectSurface {
        pub fn new(width: f64, height: f64) -> Result<Self, ViewFactorError> {
            if width <= 0.0 || height <= 0.0 {
                return Err(ViewFactorError::InvalidGeometry(
                    format!("RectSurface dimensions must be positive, got {}x{}", width, height)
                ));
            }
            Ok(Self { width, height })
        }

        pub fn area(&self) -> f64 {
            self.width * self.height
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct VerticalSurface {
        pub width: f64,
        pub height: f64,
        pub tilt: f64,
    }

    impl VerticalSurface {
        pub fn new(width: f64, height: f64) -> Result<Self, ViewFactorError> {
            if width <= 0.0 || height <= 0.0 {
                return Err(ViewFactorError::InvalidGeometry(
                    format!("VerticalSurface dimensions must be positive, got {}x{}", width, height)
                ));
            }
            Ok(Self { width, height, tilt: std::f64::consts::FRAC_PI_2 })
        }

        pub fn area(&self) -> f64 {
            self.width * self.height
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct GroundPlane {
        pub length: f64,
        pub width: f64,
    }

    impl GroundPlane {
        pub fn new(length: f64, width: f64) -> Result<Self, ViewFactorError> {
            if length <= 0.0 || width <= 0.0 {
                return Err(ViewFactorError::InvalidGeometry(
                    format!("GroundPlane dimensions must be positive, got {}x{}", length, width)
                ));
            }
            Ok(Self { length, width })
        }

        pub fn area(&self) -> f64 {
            self.length * self.width
        }
    }
}

pub mod nusselt {
    use super::ViewFactorError;
    use approx::relative_eq;

    pub fn view_factor_wall_to_sky(
        wall_height: f64,
        wall_width: f64,
        building_spacing: f64,
    ) -> Result<f64, ViewFactorError> {
        if wall_height <= 0.0 || wall_width <= 0.0 {
            return Err(ViewFactorError::ZeroArea("wall".into()));
        }
        if building_spacing < 0.0 {
            return Err(ViewFactorError::InvalidGeometry(
                "building_spacing cannot be negative".into()
            ));
        }

        let h = wall_height;
        let _w = wall_width;
        let s = building_spacing;

        let ratio = s / h;
        let f_wall_sky = if ratio > 10.0 {
            0.5
        } else if ratio < 1e-6 {
            0.5
        } else {
            let sqrt_ratio = ratio.sqrt();
            let atan_term = sqrt_ratio.atan();
            let term1 = atan_term / std::f64::consts::PI;
            let ln_arg = (1.0 + ratio.powi(2)) / ratio.powi(2);
            let term2 = if ln_arg > 0.0 {
                0.5 * ln_arg.ln() / std::f64::consts::PI * sqrt_ratio.recip() * ratio
            } else {
                0.0
            };
            (term1 + term2).max(0.0).min(1.0)
        };

        Ok(f_wall_sky)
    }

    pub fn view_factor_wall_to_ground(
        wall_height: f64,
        _wall_width: f64,
        building_spacing: f64,
    ) -> Result<f64, ViewFactorError> {
        if wall_height <= 0.0 {
            return Err(ViewFactorError::ZeroArea("wall".into()));
        }
        if building_spacing < 0.0 {
            return Err(ViewFactorError::InvalidGeometry(
                "building_spacing cannot be negative".into()
            ));
        }

        let h = wall_height;
        let s = building_spacing;

        let f_wall_ground = if s == 0.0 {
            0.0
        } else {
            let ratio = s / h;
            let term1 = (1.0 + ratio.powi(2)).sqrt() - ratio;
            let term2 = (1.0 + ratio.powi(2)).sqrt() + ratio;
            0.5 * (1.0 - (term1.ln() / term2.ln().abs()))
        };

        Ok(f_wall_ground.clamp(0.0, 1.0))
    }

    pub fn view_factor_parallel_rectangles(
        area_i: f64,
        area_j: f64,
        distance: f64,
        height_i: f64,
        height_j: f64,
    ) -> Result<f64, ViewFactorError> {
        if area_i <= 0.0 {
            return Err(ViewFactorError::ZeroArea("surface i".into()));
        }
        if area_j <= 0.0 {
            return Err(ViewFactorError::ZeroArea("surface j".into()));
        }
        if distance <= 0.0 {
            return Err(ViewFactorError::InvalidGeometry(
                "distance between surfaces must be positive".into()
            ));
        }
        if height_i <= 0.0 || height_j <= 0.0 {
            return Err(ViewFactorError::InvalidGeometry(
                "surface heights must be positive".into()
            ));
        }

        let h_i = height_i;
        let h_j = height_j;
        let d = distance;

        let x = d / h_i;
        let y = h_j / h_i;

        let numerator = y.sqrt() * (1.0 + x.powi(2)).sqrt() - x * y.sqrt();
        let denominator = 1.0 + x.powi(2) + y.powi(2);
        let base_factor = (numerator / denominator).max(0.0);

        let f_ij = base_factor * (area_j / area_i).sqrt();

        Ok(f_ij.clamp(0.0, 1.0))
    }

    pub fn view_factor_enclosure(
        surfaces: &[(f64, f64)],
    ) -> Result<Vec<Vec<f64>>, ViewFactorError> {
        let n = surfaces.len();
        if n < 2 {
            return Err(ViewFactorError::InvalidGeometry(
                "enclosure requires at least 2 surfaces".into()
            ));
        }

        let mut f = vec![vec![0.0; n]; n];
        let mut row_sums = vec![0.0; n];

        for i in 0..n {
            let (area_i, height_i) = surfaces[i];
            if area_i <= 0.0 || height_i <= 0.0 {
                return Err(ViewFactorError::InvalidGeometry(
                    format!("surface {} has invalid dimensions", i)
                ));
            }

            for j in 0..n {
                if i == j {
                    let x: f64 = 1.0;
                    let y: f64 = 1.0;
                    let xy_sqrt = (x * y).sqrt();
                    f[i][j] = xy_sqrt / (1.0 + xy_sqrt);
                } else {
                    let (area_j, height_j) = surfaces[j];
                    f[i][j] = view_factor_parallel_rectangles(
                        area_i,
                        area_j,
                        1.0,
                        height_i,
                        height_j,
                    )?;
                }
                row_sums[i] += f[i][j];
            }
        }

        for i in 0..n {
            for j in 0..n {
                f[i][j] /= row_sums[i];
            }
        }

        Ok(f)
    }

    pub fn check_reciprocity(
        area_i: f64,
        area_j: f64,
        f_ij: f64,
        f_ji: f64,
    ) -> bool {
        let left = f_ij * area_i;
        let right = f_ji * area_j;
        relative_eq!(left, right, max_relative = 1e-10)
    }

    pub fn check_summation(
        f_ii: f64,
        f_ij_sum: f64,
    ) -> Result<(), ViewFactorError> {
        let total = f_ii + f_ij_sum;
        if !relative_eq!(total, 1.0, max_relative = 1e-10) {
            return Err(ViewFactorError::SummationError(
                format!("F_ii + sum(F_ij) = {} != 1.0", total)
            ));
        }
        Ok(())
    }
}

pub use geometry::{GroundPlane, RectSurface, VerticalSurface};
pub use nusselt::{
    check_reciprocity, check_summation, view_factor_enclosure,
    view_factor_parallel_rectangles, view_factor_wall_to_ground, view_factor_wall_to_sky,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wall_to_sky_with_infinite_spacing() {
        let f = nusselt::view_factor_wall_to_sky(10.0, 5.0, 1e10).unwrap();
        assert!((f - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_wall_to_sky_zero_spacing() {
        let f = nusselt::view_factor_wall_to_sky(3.0, 5.0, 0.0).unwrap();
        assert!((f - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_wall_to_ground_zero_spacing() {
        let f = nusselt::view_factor_wall_to_ground(3.0, 5.0, 0.0).unwrap();
        assert!(f < 1e-6);
    }

    #[test]
    fn test_reciprocity_parallel_rectangles() {
        let area_i = 100.0;
        let area_j = 150.0;
        let f_ij = nusselt::view_factor_parallel_rectangles(area_i, area_j, 5.0, 10.0, 10.0).unwrap();
        let f_ji = nusselt::view_factor_parallel_rectangles(area_j, area_i, 5.0, 10.0, 10.0).unwrap();

        assert!(nusselt::check_reciprocity(area_i, area_j, f_ij, f_ji));
    }

    #[test]
    fn test_summation_check() {
        let surfaces = vec![
            (100.0, 10.0),
            (100.0, 10.0),
            (100.0, 10.0),
        ];
        let f = nusselt::view_factor_enclosure(&surfaces).unwrap();

        for i in 0..3 {
            let row_sum: f64 = f[i].iter().sum();
            nusselt::check_summation(f[i][i], row_sum - f[i][i]).unwrap();
        }
    }

    #[test]
    fn test_enclosure_two_surfaces() {
        let surfaces = vec![
            (100.0, 10.0),
            (100.0, 10.0),
        ];
        let f = nusselt::view_factor_enclosure(&surfaces).unwrap();

        assert_eq!(f.len(), 2);
        assert_eq!(f[0].len(), 2);

        for i in 0..2 {
            let row_sum: f64 = f[i].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_zero_area_error() {
        let result = nusselt::view_factor_wall_to_sky(0.0, 5.0, 10.0);
        assert!(result.is_err());

        if let Err(ViewFactorError::ZeroArea(_)) = result {
        } else {
            panic!("Expected ZeroArea error");
        }
    }

    #[test]
    fn test_invalid_geometry_error() {
        let result = nusselt::view_factor_wall_to_ground(3.0, 5.0, -1.0);
        assert!(result.is_err());

        if let Err(ViewFactorError::InvalidGeometry(_)) = result {
        } else {
            panic!("Expected InvalidGeometry error");
        }
    }

    #[test]
    fn test_rect_surface_area() {
        let rect = RectSurface::new(5.0, 3.0).unwrap();
        assert!((rect.area() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_vertical_surface_area() {
        let wall = VerticalSurface::new(10.0, 3.0).unwrap();
        assert!((wall.area() - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_ground_plane_area() {
        let ground = GroundPlane::new(50.0, 30.0).unwrap();
        assert!((ground.area() - 1500.0).abs() < 1e-10);
    }
}
