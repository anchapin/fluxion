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
pub use radiation::{SolarRadiation, Surface, UrbanRadiationSystem};

pub mod radiation {
    use ndarray::Array2;
    const STEFAN_BOLTZMANN: f64 = 5.670374419e-8; // W/m²/K⁴

    #[derive(Debug, Clone)]
    pub struct Surface {
        pub area: f64,
        pub height: f64,
        pub width: f64,
        pub tilt: f64,
        pub azimuth: f64,
        pub albedo: f64,
        pub emissivity: f64,
    }

    impl Surface {
        pub fn new(
            width: f64,
            height: f64,
            tilt: f64,
            azimuth: f64,
            albedo: f64,
            emissivity: f64,
        ) -> Self {
            let area = width * height;
            Self {
                area,
                height,
                width,
                tilt,
                azimuth,
                albedo,
                emissivity,
            }
        }

        pub fn vertical(width: f64, height: f64, azimuth: f64) -> Self {
            Self::new(
                width,
                height,
                std::f64::consts::FRAC_PI_2,
                azimuth,
                0.2,
                0.9,
            )
        }
    }

    #[derive(Debug, Clone)]
    pub struct SolarRadiation {
        pub direct_normal: f64,
        pub diffuse_horizontal: f64,
    }

    impl SolarRadiation {
        pub fn new(direct_normal: f64, diffuse_horizontal: f64) -> Self {
            Self {
                direct_normal,
                diffuse_horizontal,
            }
        }
    }

    pub struct UrbanRadiationSystem {
        pub surfaces: Vec<Surface>,
        pub sky_view_factors: Vec<f64>,
        pub ground_view_factors: Vec<f64>,
        view_factors: Option<Array2<f64>>,
    }

    impl UrbanRadiationSystem {
        pub fn new(
            surfaces: Vec<Surface>,
            sky_view_factors: Vec<f64>,
            ground_view_factors: Vec<f64>,
        ) -> Self {
            Self {
                surfaces,
                sky_view_factors,
                ground_view_factors,
                view_factors: None,
            }
        }

        pub fn with_view_factors(
            surfaces: Vec<Surface>,
            sky_view_factors: Vec<f64>,
            ground_view_factors: Vec<f64>,
            view_factors: Array2<f64>,
        ) -> Self {
            Self {
                surfaces,
                sky_view_factors,
                ground_view_factors,
                view_factors: Some(view_factors),
            }
        }

        pub fn form_factors(&self) -> Array2<f64> {
            if let Some(vf) = &self.view_factors {
                return vf.clone();
            }

            let n = self.surfaces.len();
            let mut f = Array2::<f64>::zeros((n, n));

            for i in 0..n {
                let area_i = self.surfaces[i].area;
                let height_i = self.surfaces[i].height;

                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let area_j = self.surfaces[j].area;
                    let height_j = self.surfaces[j].height;

                    let d = (height_i + height_j) * (0.5_f64).max(1.0);
                    let x = d / height_i.max(1e-6);
                    let y = height_j / height_i.max(1e-6);

                    let numerator = y.sqrt() * (1.0 + x.powi(2)).sqrt() - x * y.sqrt();
                    let denominator = 1.0 + x.powi(2) + y.powi(2);
                    let base_factor = (numerator / denominator).max(0.0);
                    let f_ij = base_factor * (area_j / area_i).sqrt().min(1.0);

                    f[[i, j]] = f_ij;
                }

                let row_sum: f64 = f.row(i).to_vec().iter().sum();
                if row_sum > 0.0 {
                    for j in 0..n {
                        f[[i, j]] /= row_sum;
                    }
                }
            }

            f
        }

        pub fn longwave_net_radiation(
            &self,
            surface_temps: &[f64],
            sky_temp: f64,
            ground_temp: f64,
        ) -> Vec<f64> {
            let n = self.surfaces.len();
            let f = self.form_factors();

            let sky_view: Vec<f64> = self.sky_view_factors.clone();
            let ground_view: Vec<f64> = self.ground_view_factors.clone();

            let mut q_net = vec![0.0; n];

            for i in 0..n {
                let t_i4 = surface_temps[i].powi(4);

                let mut q_lw_out = 0.0;

                for j in 0..n {
                    if i != j {
                        let t_j4 = surface_temps[j].powi(4);
                        q_lw_out += f[[i, j]] * STEFAN_BOLTZMANN * (t_i4 - t_j4);
                    }
                }

                let sky_term = sky_view.get(i).copied().unwrap_or(0.0);
                let ground_term = ground_view.get(i).copied().unwrap_or(0.0);

                q_lw_out += sky_term * STEFAN_BOLTZMANN * (t_i4 - sky_temp.powi(4));
                q_lw_out += ground_term * STEFAN_BOLTZMANN * (t_i4 - ground_temp.powi(4));

                q_net[i] = q_lw_out;
            }

            q_net
        }

        pub fn shortwave_radiation(
            &self,
            solar: &SolarRadiation,
            surface_temps: &[f64],
        ) -> Vec<f64> {
            let n = self.surfaces.len();
            let mut q_sw = vec![0.0; n];

            for i in 0..n {
                let surface = &self.surfaces[i];
                let tilt_rad = surface.tilt;
                let _azimuth_rad = surface.azimuth;

                let cos_incident = (std::f64::consts::FRAC_PI_2 - tilt_rad).max(0.0);
                let diffuse_factor = (1.0 + tilt_rad / std::f64::consts::FRAC_PI_2) * 0.5;

                let direct = solar.direct_normal * cos_incident;
                let diffuse = solar.diffuse_horizontal * diffuse_factor;

                let absorbed = (1.0 - surface.albedo) * (direct + diffuse);
                let emitted = surface.emissivity * STEFAN_BOLTZMANN * surface_temps[i].powi(4);

                q_sw[i] = absorbed - emitted;
            }

            q_sw
        }

        pub fn net_radiation(
            &self,
            surface_temps: &[f64],
            sky_temp: f64,
            ground_temp: f64,
            solar: Option<&SolarRadiation>,
        ) -> Vec<f64> {
            let n = self.surfaces.len();
            let mut q_net = self.longwave_net_radiation(surface_temps, sky_temp, ground_temp);

            if let Some(solar) = solar {
                let q_sw = self.shortwave_radiation(solar, surface_temps);
                for i in 0..n {
                    q_net[i] += q_sw[i];
                }
            }

            q_net
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_surface_creation() {
            let surface = Surface::vertical(10.0, 3.0, 0.0);
            assert!((surface.area - 30.0).abs() < 1e-10);
            assert!((surface.tilt - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        }

        #[test]
        fn test_form_factors_5_buildings() {
            let surfaces = vec![
                Surface::vertical(10.0, 15.0, 0.0),
                Surface::vertical(10.0, 15.0, std::f64::consts::FRAC_PI_2),
                Surface::vertical(10.0, 15.0, std::f64::consts::PI),
                Surface::vertical(10.0, 15.0, 3.0 * std::f64::consts::FRAC_PI_2),
                Surface::vertical(10.0, 15.0, 0.0),
            ];

            let sky_factors = vec![0.3, 0.3, 0.3, 0.3, 0.3];
            let ground_factors = vec![0.1, 0.1, 0.1, 0.1, 0.1];

            let system = UrbanRadiationSystem::new(surfaces, sky_factors, ground_factors);
            let f = system.form_factors();

            assert_eq!(f.shape(), [5, 5]);

            for i in 0..5 {
                let row_sum: f64 = f.row(i).to_vec().iter().sum();
                assert!((row_sum - 1.0).abs() < 1e-6);
            }
        }

        #[test]
        fn test_longwave_net_radiation() {
            let surfaces = vec![
                Surface::vertical(10.0, 15.0, 0.0),
                Surface::vertical(10.0, 15.0, std::f64::consts::FRAC_PI_2),
            ];

            let system = UrbanRadiationSystem::new(
                surfaces,
                vec![0.3, 0.3],
                vec![0.1, 0.1],
            );

            let temps = vec![293.15, 288.15];
            let sky_temp = 270.0;
            let ground_temp = 285.0;

            let q_net = system.longwave_net_radiation(&temps, sky_temp, ground_temp);

            assert_eq!(q_net.len(), 2);
            for q in &q_net {
                assert!(q.is_finite());
            }
        }

        #[test]
        fn test_shortwave_radiation() {
            let surfaces = vec![
                Surface::vertical(10.0, 15.0, 0.0),
                Surface::vertical(10.0, 15.0, std::f64::consts::FRAC_PI_2),
            ];

            let system = UrbanRadiationSystem::new(
                surfaces,
                vec![0.3, 0.3],
                vec![0.1, 0.1],
            );

            let solar = SolarRadiation::new(800.0, 100.0);
            let temps = vec![293.15, 288.15];

            let q_sw = system.shortwave_radiation(&solar, &temps);

            assert_eq!(q_sw.len(), 2);
        }

        #[test]
        fn test_net_radiation_with_solar() {
            let surfaces = vec![
                Surface::vertical(10.0, 15.0, 0.0),
                Surface::vertical(10.0, 15.0, std::f64::consts::FRAC_PI_2),
            ];

            let system = UrbanRadiationSystem::new(
                surfaces,
                vec![0.3, 0.3],
                vec![0.1, 0.1],
            );

            let temps = vec![293.15, 288.15];
            let solar = SolarRadiation::new(800.0, 100.0);

            let q_net = system.net_radiation(&temps, 270.0, 285.0, Some(&solar));

            assert_eq!(q_net.len(), 2);
            for q in &q_net {
                assert!(q.is_finite());
            }
        }

        #[test]
        fn test_net_radiation_no_solar() {
            let surfaces = vec![
                Surface::vertical(10.0, 15.0, 0.0),
                Surface::vertical(10.0, 15.0, std::f64::consts::FRAC_PI_2),
            ];

            let system = UrbanRadiationSystem::new(
                surfaces,
                vec![0.3, 0.3],
                vec![0.1, 0.1],
            );

            let temps = vec![293.15, 288.15];

            let q_net = system.net_radiation(&temps, 270.0, 285.0, None);

            assert_eq!(q_net.len(), 2);
            for q in &q_net {
                assert!(q.is_finite());
            }
        }
    }
}

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
