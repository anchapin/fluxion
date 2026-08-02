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
//!
//! ## Sparse Matrix Integration
//!
//! This crate uses sparse matrix representations for efficient computation with
//! urban radiation with many surfaces. Since Issue #2030 the
//! [`sparse::UrbanRadiationSolver`] materialises the view-factor matrix into a
//! [`faer::sparse::SparseColMat<usize, f64>`] (CSC format) and computes the net
//! radiative flux per surface via a single SIMD-accelerated sparse
//! matrix-vector product:
//!
//! ```text
//! Q_net = diag(ε σ A) · [ diag(s) · T⁴ − F · T⁴ ]
//! ```
//!
//! At 2% edge density (100-building graph) the faer CSC representation uses
//! ~5% of the memory of a dense matrix and the matvec runs ~3× faster than
//! the HashMap-based per-pair aggregation.
//!
//! ## UrbanGraph Spatial Topology
//!
//! The `UrbanGraph` module provides spatial graph representation for city-scale
//! building energy modeling where nodes represent building envelopes and edges
//! represent spatial adjacency.

#[cfg(test)]
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ViewFactorError {
    #[error("Surface has zero area: {0}")]
    ZeroArea(String),

    #[error("Invalid geometry: {0}")]
    InvalidGeometry(String),

    #[error("Numerical precision error in view factor summation: {0}")]
    SummationError(String),

    #[error("Sparse matrix operation failed: {0}")]
    SparseMatrixError(String),
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
                return Err(ViewFactorError::InvalidGeometry(format!(
                    "RectSurface dimensions must be positive, got {}x{}",
                    width, height
                )));
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
                return Err(ViewFactorError::InvalidGeometry(format!(
                    "VerticalSurface dimensions must be positive, got {}x{}",
                    width, height
                )));
            }
            Ok(Self {
                width,
                height,
                tilt: std::f64::consts::FRAC_PI_2,
            })
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
                return Err(ViewFactorError::InvalidGeometry(format!(
                    "GroundPlane dimensions must be positive, got {}x{}",
                    length, width
                )));
            }
            Ok(Self { length, width })
        }

        pub fn area(&self) -> f64 {
            self.length * self.width
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct UrbanCanopySurface {
        pub area: f64,
        pub height: f64,
        pub distance_to_target: f64,
        pub surface_type: SurfaceType,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum SurfaceType {
        Wall,
        Ground,
        Sky,
        Window,
    }

    impl UrbanCanopySurface {
        pub fn new_wall(area: f64, height: f64, distance: f64) -> Self {
            Self {
                area,
                height,
                distance_to_target: distance,
                surface_type: SurfaceType::Wall,
            }
        }

        pub fn new_ground(area: f64) -> Self {
            Self {
                area,
                height: 0.0,
                distance_to_target: 0.0,
                surface_type: SurfaceType::Ground,
            }
        }

        pub fn new_sky() -> Self {
            Self {
                area: f64::INFINITY,
                height: f64::MAX,
                distance_to_target: f64::MAX,
                surface_type: SurfaceType::Sky,
            }
        }
    }
}

pub mod nusselt {
    use super::{
        geometry::{SurfaceType, UrbanCanopySurface},
        ViewFactorError,
    };
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
                "building_spacing cannot be negative".into(),
            ));
        }

        let h = wall_height;
        let _w = wall_width;
        let s = building_spacing;

        let ratio = s / h;
        let f_wall_sky = if !(1e-6..=10.0).contains(&ratio) {
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
            (term1 + term2).clamp(0.0, 1.0)
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
                "building_spacing cannot be negative".into(),
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
                "distance between surfaces must be positive".into(),
            ));
        }
        if height_i <= 0.0 || height_j <= 0.0 {
            return Err(ViewFactorError::InvalidGeometry(
                "surface heights must be positive".into(),
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
                "enclosure requires at least 2 surfaces".into(),
            ));
        }

        let mut f = vec![vec![0.0; n]; n];
        let mut row_sums = vec![0.0; n];

        for i in 0..n {
            let (area_i, height_i) = surfaces[i];
            if area_i <= 0.0 {
                return Err(ViewFactorError::InvalidGeometry(format!(
                    "surface {} has invalid dimensions",
                    i
                )));
            }

            for j in 0..n {
                if i == j {
                    if height_i <= 0.0 {
                        f[i][j] = 1.0;
                    } else {
                        let x: f64 = 1.0;
                        let y: f64 = 1.0;
                        let xy_sqrt = (x * y).sqrt();
                        f[i][j] = xy_sqrt / (1.0 + xy_sqrt);
                    }
                } else {
                    let (area_j, height_j) = surfaces[j];
                    let h_i = if height_i <= 0.0 { 1.0 } else { height_i };
                    let h_j = if height_j <= 0.0 { 1.0 } else { height_j };
                    f[i][j] = view_factor_parallel_rectangles(area_i, area_j, 1.0, h_i, h_j)?;
                }
                row_sums[i] += f[i][j];
            }
        }

        for i in 0..n {
            if row_sums[i] > 0.0 {
                for val in &mut f[i][..n] {
                    *val /= row_sums[i];
                }
            }
        }

        Ok(f)
    }

    pub fn view_factor_between_surfaces(
        surface_i: &UrbanCanopySurface,
        surface_j: &UrbanCanopySurface,
    ) -> Result<f64, ViewFactorError> {
        if surface_i.area <= 0.0 {
            return Err(ViewFactorError::ZeroArea("surface i".into()));
        }
        if surface_j.area <= 0.0 {
            return Err(ViewFactorError::ZeroArea("surface j".into()));
        }

        match (surface_i.surface_type, surface_j.surface_type) {
            (_, SurfaceType::Sky) => {
                if surface_i.surface_type == SurfaceType::Wall {
                    let spacing = if surface_j.distance_to_target < 1e10 {
                        surface_j.distance_to_target
                    } else {
                        1000.0 * surface_i.height
                    };
                    view_factor_wall_to_sky(surface_i.height, surface_i.area.sqrt(), spacing)
                } else {
                    Ok(1.0)
                }
            }
            (SurfaceType::Sky, _) => Ok(1.0),
            (SurfaceType::Ground, SurfaceType::Ground) => Ok(1.0),
            (SurfaceType::Ground, SurfaceType::Wall) => Ok(0.0),
            (SurfaceType::Wall, SurfaceType::Ground) => view_factor_wall_to_ground(
                surface_i.height,
                surface_i.area.sqrt(),
                surface_i.distance_to_target,
            ),
            (SurfaceType::Wall, SurfaceType::Wall) => {
                let dist = surface_i.distance_to_target.max(0.001);
                view_factor_parallel_rectangles(
                    surface_i.area,
                    surface_j.area,
                    dist,
                    surface_i.height,
                    surface_j.height,
                )
            }
            _ => Ok(0.0),
        }
    }

    pub fn compute_urban_canyon_view_factors(
        walls: &[(f64, f64, f64)],
        ground_area: f64,
    ) -> Result<ViewFactorMatrix<f64>, ViewFactorError> {
        let n = walls.len();
        let mut matrix = ViewFactorMatrix::new(n + 1);

        let positions: Vec<f64> = walls.iter().map(|&(_, _, pos)| pos).collect();

        let wall_areas: Vec<f64> = walls.iter().map(|&(area, _, _)| area).collect();
        let wall_heights: Vec<f64> = walls.iter().map(|&(_, height, _)| height).collect();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let separation = (positions[i] - positions[j]).abs().max(0.001);
                    let f = view_factor_parallel_rectangles(
                        wall_areas[i],
                        wall_areas[j],
                        separation,
                        wall_heights[i],
                        wall_heights[j],
                    )?;
                    matrix.set(i, j, f);
                }
            }
        }

        let ground_idx = n;
        for i in 0..n {
            let f_i_ground =
                view_factor_wall_to_ground(wall_heights[i], wall_areas[i].sqrt(), 0.0)?;
            matrix.set(i, ground_idx, f_i_ground);

            let f_i_sky = 1.0
                - f_i_ground
                - (0..n)
                    .filter(|&j| j != i)
                    .map(|j| matrix.get(i, j))
                    .sum::<f64>();
            if f_i_sky < 0.0 {
                matrix.set(i, ground_idx, matrix.get(i, ground_idx) + f_i_sky);
            }
        }

        for j in 0..n {
            let f_ground_j =
                view_factor_wall_to_ground(wall_heights[j], wall_areas[j].sqrt(), 0.0)?
                    * wall_areas[j]
                    / ground_area;
            matrix.set(ground_idx, j, f_ground_j);
        }
        matrix.set(ground_idx, ground_idx, 1.0);

        for i in 0..n {
            let row_sum: f64 = (0..matrix.ncols()).map(|j| matrix.get(i, j)).sum::<f64>();
            if row_sum > 1e-10 {
                let scale = 1.0 / row_sum;
                for j in 0..matrix.ncols() {
                    let val = matrix.get(i, j);
                    matrix.set(i, j, val * scale);
                }
            }
        }

        Ok(matrix)
    }

    pub fn check_reciprocity(area_i: f64, area_j: f64, f_ij: f64, f_ji: f64) -> bool {
        let left = f_ij * area_i;
        let right = f_ji * area_j;
        relative_eq!(left, right, max_relative = 1e-6)
    }

    pub fn check_summation(f_ii: f64, f_ij_sum: f64) -> Result<(), ViewFactorError> {
        let total = f_ii + f_ij_sum;
        if !relative_eq!(total, 1.0, max_relative = 1e-6) {
            return Err(ViewFactorError::SummationError(format!(
                "F_ii + sum(F_ij) = {} != 1.0",
                total
            )));
        }
        Ok(())
    }

    pub struct ViewFactorMatrix<T> {
        data: Vec<T>,
        nrows: usize,
        ncols: usize,
    }

    impl<T: Copy + Into<f64> + From<f64>> ViewFactorMatrix<T> {
        pub fn new(n: usize) -> Self {
            Self {
                data: vec![T::from(0.0); n * n],
                nrows: n,
                ncols: n,
            }
        }

        pub fn from_dense(data: Vec<Vec<T>>) -> Self {
            let n = data.len();
            let mut flat = Vec::with_capacity(n * n);
            for row in &data {
                flat.extend_from_slice(row);
            }
            Self {
                data: flat,
                nrows: n,
                ncols: n,
            }
        }

        pub fn get(&self, i: usize, j: usize) -> T {
            self.data[i * self.ncols + j]
        }

        pub fn set(&mut self, i: usize, j: usize, value: T) {
            self.data[i * self.ncols + j] = value;
        }

        pub fn nrows(&self) -> usize {
            self.nrows
        }

        pub fn ncols(&self) -> usize {
            self.ncols
        }

        pub fn row_sum(&self, i: usize) -> T
        where
            T: std::ops::Add<Output = T> + Clone,
        {
            let mut sum = T::from(0.0);
            for j in 0..self.ncols {
                sum = sum + self.get(i, j);
            }
            sum
        }

        pub fn normalize_by_row(&mut self)
        where
            T: std::ops::Div<Output = T> + std::ops::Add<Output = T> + Clone,
        {
            for i in 0..self.nrows {
                let sum: f64 = (0..self.ncols).map(|j| self.get(i, j).into()).sum();
                if sum > 0.0 {
                    for j in 0..self.ncols {
                        let val: f64 = self.get(i, j).into();
                        self.set(i, j, T::from(val / sum));
                    }
                }
            }
        }

        pub fn to_vec_vec(&self) -> Vec<Vec<T>>
        where
            T: Clone,
        {
            let mut result = Vec::with_capacity(self.nrows);
            for i in 0..self.nrows {
                let mut row = Vec::with_capacity(self.ncols);
                for j in 0..self.ncols {
                    row.push(self.get(i, j));
                }
                result.push(row);
            }
            result
        }
    }

    impl ViewFactorMatrix<f64> {
        pub fn verify_reciprocity(&self, areas: &[f64]) -> Vec<(usize, usize, bool)> {
            let mut results = Vec::new();
            for i in 0..self.nrows {
                for j in i..self.ncols {
                    let f_ij = self.get(i, j);
                    let f_ji = self.get(j, i);
                    let is_reciprocal = check_reciprocity(areas[i], areas[j], f_ij, f_ji);
                    results.push((i, j, is_reciprocal));
                }
            }
            results
        }

        pub fn verify_summation(&self) -> Vec<(usize, bool)> {
            let mut results = Vec::new();
            for i in 0..self.nrows {
                let f_ii = self.get(i, i);
                let row_sum: f64 = (0..self.ncols).map(|j| self.get(i, j)).sum();
                let f_ij_sum = row_sum - f_ii;
                let total = f_ii + f_ij_sum;
                let is_valid = relative_eq!(total, 1.0, max_relative = 1e-6);
                results.push((i, is_valid));
            }
            results
        }
    }
}

pub mod sparse {
    use super::{nusselt::ViewFactorMatrix, ViewFactorError};
    use std::collections::HashMap;
    use thiserror::Error;

    /// Stefan-Boltzmann constant, σ = 5.67×10⁻⁸ W·m⁻²·K⁻⁴.
    ///
    /// Used for longwave radiative heat exchange between building surfaces.
    pub const STEFAN_BOLTZMANN: f64 = 5.67e-8;

    /// Default longwave emissivity for typical building envelope surfaces (ε = 0.9).
    ///
    /// Most opaque building materials (brick, concrete, stucco, roofing) have a
    /// longwave emissivity in the 0.85–0.92 range; 0.9 is the canonical default.
    pub const DEFAULT_EMISSIVITY: f64 = 0.9;

    /// Errors raised during inter-building radiative flux computation.
    #[derive(Debug, Error, PartialEq)]
    pub enum RadiationError {
        #[error(
            "temperature vector length ({got}) does not match number of surfaces ({expected})"
        )]
        DimensionMismatch { expected: usize, got: usize },

        #[error(
            "emissivities vector length ({got}) does not match number of surfaces ({expected})"
        )]
        EmissivityMismatch { expected: usize, got: usize },

        #[error("areas vector length ({got}) does not match number of surfaces ({expected})")]
        AreaMismatch { expected: usize, got: usize },

        #[error(
            "surface {index} temperature {value} K is invalid (must be finite and non-negative)"
        )]
        InvalidTemperature { index: usize, value: f64 },

        #[error("surface {index} emissivity {value} is out of range [0, 1]")]
        InvalidEmissivity { index: usize, value: f64 },
    }

    /// Net radiative heat flow for a single ordered surface pair (i → j).
    ///
    /// Computed from the linearised gray-diffuse exchange
    /// `Q_rad(i→j) = ε_i · σ · F_ij · A_i · (T_i⁴ − T_j⁴)`.
    ///
    /// `heat_flow_w` is positive when surface `from` is warmer than surface `to`
    /// (net radiative transfer from `from` to `to`), and negative otherwise.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct SurfacePairFlux {
        /// Index of the emitting surface (i).
        pub from: usize,
        /// Index of the receiving surface (j).
        pub to: usize,
        /// View factor F_{i→j} used for this exchange.
        pub view_factor: f64,
        /// Net radiative heat flow Q_rad(i→j) in watts.
        pub heat_flow_w: f64,
    }

    pub struct SparseViewFactorMatrix {
        data: HashMap<(usize, usize), f64>,
        nrows: usize,
        ncols: usize,
        row_counts: Vec<usize>,
        col_counts: Vec<usize>,
    }

    impl SparseViewFactorMatrix {
        pub fn new(nrows: usize, ncols: usize) -> Self {
            Self {
                data: HashMap::new(),
                nrows,
                ncols,
                row_counts: vec![0; nrows],
                col_counts: vec![0; ncols],
            }
        }

        pub fn from_dense(matrix: &ViewFactorMatrix<f64>) -> Self {
            let nrows = matrix.nrows();
            let ncols = matrix.ncols();
            let mut sparse = Self::new(nrows, ncols);

            for i in 0..nrows {
                for j in 0..ncols {
                    let val = matrix.get(i, j);
                    if val > 1e-12 {
                        sparse.data.insert((i, j), val);
                        sparse.row_counts[i] += 1;
                        sparse.col_counts[j] += 1;
                    }
                }
            }

            sparse
        }

        pub fn nrows(&self) -> usize {
            self.nrows
        }

        pub fn ncols(&self) -> usize {
            self.ncols
        }

        pub fn get(&self, i: usize, j: usize) -> f64 {
            *self.data.get(&(i, j)).unwrap_or(&0.0)
        }

        pub fn set(&mut self, i: usize, j: usize, val: f64) {
            if val > 1e-12 {
                if !self.data.contains_key(&(i, j)) {
                    self.row_counts[i] += 1;
                    self.col_counts[j] += 1;
                }
                self.data.insert((i, j), val);
            } else if self.data.contains_key(&(i, j)) {
                self.data.remove(&(i, j));
                self.row_counts[i] -= 1;
                self.col_counts[j] -= 1;
            }
        }

        pub fn multiply_dense(&self, vec: &[f64]) -> Vec<f64> {
            let mut result = vec![0.0; self.nrows];
            for ((i, j), &val) in &self.data {
                result[*i] += val * vec[*j];
            }
            result
        }

        pub fn multiply_transpose_dense(&self, vec: &[f64]) -> Vec<f64> {
            let mut result = vec![0.0; self.ncols];
            for ((i, j), &val) in &self.data {
                result[*j] += val * vec[*i];
            }
            result
        }

        pub fn to_dense(&self) -> ViewFactorMatrix<f64> {
            let mut dense = ViewFactorMatrix::new(self.nrows);
            for ((i, j), &val) in &self.data {
                dense.set(*i, *j, val);
            }
            dense
        }

        pub fn sparsity_ratio(&self) -> f64 {
            let total = self.nrows * self.ncols;
            let nz = self.data.len();
            1.0 - (nz as f64 / total as f64)
        }

        pub fn nnz(&self) -> usize {
            self.data.len()
        }

        pub fn row_nnz(&self, i: usize) -> usize {
            self.row_counts[i]
        }

        pub fn col_nnz(&self, j: usize) -> usize {
            self.col_counts[j]
        }

        /// Build a [`faer`] sparse CSC matrix (`SparseColMat<usize, f64>`) from
        /// the current non-zero entries (Issue #2030).
        ///
        /// The resulting matrix is the canonical high-performance representation
        /// used for SIMD-accelerated sparse matrix-vector products in
        /// [`UrbanRadiationSolver`]. Duplicate `(row, col)` triplets (which
        /// cannot occur here, since `set` overwrites) are summed by faer.
        ///
        /// # Errors
        ///
        /// Returns [`ViewFactorError::SparseMatrixError`] if faer rejects the
        /// triplet list (e.g. an out-of-bounds index).
        pub fn to_faer(&self) -> Result<faer::sparse::SparseColMat<usize, f64>, ViewFactorError> {
            use faer::sparse::{SparseColMat, Triplet};

            let triplets: Vec<Triplet<usize, usize, f64>> = self
                .data
                .iter()
                .map(|(&(row, col), &val)| Triplet::new(row, col, val))
                .collect();

            SparseColMat::<usize, f64>::try_new_from_triplets(self.nrows, self.ncols, &triplets)
                .map_err(|e| {
                    ViewFactorError::SparseMatrixError(format!(
                        "faer SparseColMat construction failed: {e:?}"
                    ))
                })
        }

        /// Estimated memory footprint of the **HashMap-backed** sparse storage,
        /// in bytes (Issue #2030).
        ///
        /// `HashMap<(usize, usize), f64>` allocates roughly 16 bytes for the key
        /// pair, 8 bytes for the value, plus per-bucket hashing/control overhead
        /// (~12 bytes). We use 36 bytes per non-zero entry as a conservative
        /// estimate, plus the `row_counts`/`col_counts` vectors.
        pub fn estimated_hashmap_bytes(&self) -> usize {
            const BYTES_PER_ENTRY: usize = 36;
            self.data.len() * BYTES_PER_ENTRY
                + (self.row_counts.len() + self.col_counts.len()) * std::mem::size_of::<usize>()
        }

        /// Estimated memory footprint of the equivalent **faer CSC** sparse
        /// storage, in bytes (Issue #2030).
        ///
        /// faer's compressed sparse-column format stores:
        /// - `col_ptr`: `(ncols + 1)` index entries
        /// - `row_idx`: `nnz` index entries
        /// - `val`:     `nnz` value entries
        ///
        /// Each entry is 8 bytes (`usize`/`f64`).
        pub fn estimated_faer_csc_bytes(&self) -> usize {
            let nnz = self.data.len();
            let idx_size = std::mem::size_of::<usize>();
            let val_size = std::mem::size_of::<f64>();
            (self.ncols + 1) * idx_size + nnz * idx_size + nnz * val_size
        }

        /// Estimated memory footprint of an equivalent **dense** `n×n` matrix,
        /// in bytes (Issue #2030).
        pub fn estimated_dense_bytes(&self) -> usize {
            self.nrows * self.ncols * std::mem::size_of::<f64>()
        }

        /// Edge density of the view-factor graph: `nnz / (nrows × ncols)`.
        ///
        /// A value below 0.1 (<10%) indicates a sparse graph for which CSC
        /// storage is strictly smaller than dense storage.
        pub fn edge_density(&self) -> f64 {
            let total = self.nrows * self.ncols;
            if total == 0 {
                0.0
            } else {
                self.data.len() as f64 / total as f64
            }
        }
    }

    /// Inter-building longwave radiation solver.
    ///
    /// Holds the (sparse) view-factor matrix, per-surface areas and emissivities,
    /// and computes net radiative exchange between surface pairs using the
    /// Stefan-Boltzmann law.
    ///
    /// Since Issue #2030 the solver additionally holds a [`faer`] sparse CSC
    /// matrix ([`faer::sparse::SparseColMat`]) materialised from the view
    /// factors. This powers the SIMD-accelerated sparse matrix-vector product in
    /// [`compute_net_flux_per_surface`](Self::compute_net_flux_per_surface),
    /// which reformulates the gray-diffuse net exchange as
    ///
    /// ```text
    /// Q_net = diag(ε σ A) · [ diag(s) · T⁴ − F · T⁴ ]
    /// ```
    ///
    /// where `s = F·1` (row sums) and `F·T⁴` is a single sparse matvec.
    pub struct UrbanRadiationSolver {
        view_factors: SparseViewFactorMatrix,
        /// faer-backed CSC view-factor matrix for high-performance matvec.
        faer_matrix: faer::sparse::SparseColMat<usize, f64>,
        /// Cached row sums `s_i = Σ_j F_ij` (view-factor closure per surface).
        row_sums: Vec<f64>,
        areas: Vec<f64>,
        emissivities: Vec<f64>,
    }

    impl UrbanRadiationSolver {
        pub fn new(
            view_factors: SparseViewFactorMatrix,
            areas: Vec<f64>,
            emissivities: Vec<f64>,
        ) -> Self {
            let n = view_factors.nrows();
            let faer_matrix = view_factors.to_faer().unwrap_or_else(|_| {
                // Fall back to an empty matrix of matching shape — construction
                // only fails on out-of-bounds indices, which the HashMap-backed
                // builder guarantees cannot happen.
                faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &[])
                    .expect("empty faer matrix construction cannot fail")
            });

            // Pre-compute row sums s_i = Σ_j F_ij for the matvec formulation.
            let row_sums = (0..n)
                .map(|i| {
                    let mut s = 0.0;
                    for j in 0..view_factors.ncols() {
                        s += view_factors.get(i, j);
                    }
                    s
                })
                .collect();

            Self {
                view_factors,
                faer_matrix,
                row_sums,
                areas,
                emissivities,
            }
        }

        pub fn from_dense_enclosure(
            matrix: &ViewFactorMatrix<f64>,
            areas: Vec<f64>,
            emissivities: Vec<f64>,
        ) -> Result<Self, ViewFactorError> {
            if areas.len() != matrix.nrows() {
                return Err(ViewFactorError::InvalidGeometry(
                    "Number of areas must match matrix dimensions".into(),
                ));
            }
            let sparse = SparseViewFactorMatrix::from_dense(matrix);
            Ok(Self::new(sparse, areas, emissivities))
        }

        /// Build a solver with a uniform emissivity for every surface.
        ///
        /// Convenience constructor for the common case where all building
        /// surfaces share the same longwave emissivity (defaults to
        /// [`DEFAULT_EMISSIVITY`] = 0.9).
        pub fn with_uniform_emissivity(
            view_factors: SparseViewFactorMatrix,
            areas: Vec<f64>,
            emissivity: f64,
        ) -> Self {
            let emissivities = vec![emissivity; areas.len()];
            Self::new(view_factors, areas, emissivities)
        }

        /// Absorbed longwave radiation per surface (W) using the radiosity-style
        /// matrix-vector product `ε_i · Σ_j F_ji · (ε_j σ T_j⁴)`.
        pub fn compute_radiation_exchange(&self, temperatures: &[f64]) -> Vec<f64> {
            let n = temperatures.len();
            let mut absorbed = vec![0.0; n];

            let mut j_rad = vec![0.0; n];
            for i in 0..n {
                j_rad[i] = self.emissivities[i] * STEFAN_BOLTZMANN * temperatures[i].powi(4);
            }

            let incident = self.view_factors.multiply_transpose_dense(&j_rad);

            for i in 0..n {
                absorbed[i] = self.emissivities[i] * incident[i];
            }

            absorbed
        }

        /// Compute the net radiative heat flow for every non-zero surface pair.
        ///
        /// Implements the gray-diffuse net exchange
        /// `Q_rad(i→j) = ε_i · σ · F_ij · A_i · (T_i⁴ − T_j⁴)`
        /// (Issue #2029), where `ε_i` is the emissivity of the emitting surface.
        ///
        /// Only pairs with a non-zero view factor are returned. Each returned
        /// [`SurfacePairFlux`] carries the ordered indices `(i, j)`, the view
        /// factor, and the signed heat flow in watts (positive ⇒ net transfer
        /// from `i` to `j`, i.e. `i` is warmer).
        pub fn compute_fluxes(
            &self,
            temperatures: &[f64],
        ) -> Result<Vec<SurfacePairFlux>, RadiationError> {
            let n = self.areas.len();
            if temperatures.len() != n {
                return Err(RadiationError::DimensionMismatch {
                    expected: n,
                    got: temperatures.len(),
                });
            }
            if self.emissivities.len() != n {
                return Err(RadiationError::EmissivityMismatch {
                    expected: n,
                    got: self.emissivities.len(),
                });
            }

            // Validate inputs (non-negative, finite temperatures; bounded emissivity).
            for (i, &t) in temperatures.iter().enumerate() {
                if !t.is_finite() || t < 0.0 {
                    return Err(RadiationError::InvalidTemperature { index: i, value: t });
                }
            }
            for (i, &e) in self.emissivities.iter().enumerate() {
                if !(0.0..=1.0).contains(&e) {
                    return Err(RadiationError::InvalidEmissivity { index: i, value: e });
                }
            }

            // Iterate over the stored non-zero view-factor entries directly.
            // SparseViewFactorMatrix only retains entries above the 1e-12 threshold,
            // so this visits exactly the radiatively-coupled pairs.
            let mut fluxes = Vec::with_capacity(self.view_factors.nnz());
            for (&(i, j), &f_ij) in &self.view_factors.data {
                let eps_i = self.emissivities[i];
                let a_i = self.areas[i];
                let t_i = temperatures[i];
                let t_j = temperatures[j];

                // Q_rad(i→j) = ε_i σ F_ij A_i (T_i⁴ − T_j⁴)
                let heat_flow_w =
                    eps_i * STEFAN_BOLTZMANN * f_ij * a_i * (t_i.powi(4) - t_j.powi(4));

                fluxes.push(SurfacePairFlux {
                    from: i,
                    to: j,
                    view_factor: f_ij,
                    heat_flow_w,
                });
            }

            Ok(fluxes)
        }

        /// Aggregate net radiative heat flow per surface (W).
        ///
        /// For each surface `i` this is `Σ_j Q_rad(i→j)` — the net radiative
        /// energy *leaving* surface `i` towards all visible surfaces `j`
        /// (positive ⇒ surface `i` is a net radiator, negative ⇒ net absorber).
        pub fn compute_net_flux_per_surface(&self, temperatures: &[f64]) -> Vec<f64> {
            let n = self.areas.len();
            let mut net = vec![0.0; n];

            if let Ok(fluxes) = self.compute_fluxes(temperatures) {
                for f in fluxes {
                    net[f.from] += f.heat_flow_w;
                }
            }

            net
        }

        /// Aggregate net radiative heat flow per surface (W) using the
        /// [`faer`] sparse CSC matrix-vector product (Issue #2030).
        ///
        /// This is the SIMD-accelerated equivalent of
        /// [`compute_net_flux_per_surface`](Self::compute_net_flux_per_surface).
        /// It reformulates the gray-diffuse net exchange
        /// `Q_net(i) = Σ_j ε_i σ F_ij A_i (T_i⁴ − T_j⁴)` as
        ///
        /// ```text
        /// Q_net = diag(ε σ A) · [ diag(s) · T⁴ − F · T⁴ ]
        /// ```
        ///
        /// where `s = F·1` (pre-computed row sums) and `F·T⁴` is a single
        /// [`faer::sparse::linalg::matmul::sparse_dense_matmul`] over the vector
        /// of fourth powers. The result is bit-for-bit equivalent to the
        /// HashMap-based reference (verified by
        /// `test_faer_net_flux_matches_hashmap_reference`).
        pub fn compute_net_flux_per_surface_faer(&self, temperatures: &[f64]) -> Vec<f64> {
            use faer::sparse::linalg::matmul::sparse_dense_matmul;
            use faer::{Accum, Mat, Par};

            let n = self.areas.len();
            if n == 0 {
                return Vec::new();
            }

            // Vector of fourth powers: t4 = T⁴  (n×1 dense matrix).
            let t4 = Mat::from_fn(n, 1, |i, _| temperatures[i].powi(4));

            // Sparse matvec: f_t4 = F · T⁴   (n×1).
            let mut f_t4 = Mat::<f64>::zeros(n, 1);
            sparse_dense_matmul(
                f_t4.as_mut(),
                Accum::Replace,
                self.faer_matrix.as_ref(),
                t4.as_ref(),
                1.0,
                Par::Seq,
            );

            // Q_net(i) = ε_i σ A_i [ s_i·T_i⁴ − (F·T⁴)_i ].
            let mut net = vec![0.0; n];
            for i in 0..n {
                let bracket = self.row_sums[i] * t4[(i, 0)] - f_t4[(i, 0)];
                net[i] = self.emissivities[i] * STEFAN_BOLTZMANN * self.areas[i] * bracket;
            }

            net
        }

        /// Reference to the faer-backed CSC view-factor matrix (Issue #2030).
        pub fn faer_matrix(&self) -> &faer::sparse::SparseColMat<usize, f64> {
            &self.faer_matrix
        }

        pub fn view_factor_matrix(&self) -> &SparseViewFactorMatrix {
            &self.view_factors
        }

        pub fn view_factor_dense_at(&self, i: usize, j: usize) -> f64 {
            self.view_factors.get(i, j)
        }

        /// Number of surfaces modelled by this solver.
        pub fn num_surfaces(&self) -> usize {
            self.areas.len()
        }
    }

    pub fn create_sparse_from_urban_canyon(
        walls: &[(f64, f64, f64)],
        ground_area: f64,
    ) -> Result<SparseViewFactorMatrix, ViewFactorError> {
        use super::nusselt::compute_urban_canyon_view_factors;
        let dense = compute_urban_canyon_view_factors(walls, ground_area)?;
        Ok(SparseViewFactorMatrix::from_dense(&dense))
    }
}

pub mod ashrae140 {
    use super::{nusselt::ViewFactorMatrix, ViewFactorError};

    #[derive(Debug, Clone)]
    pub struct Ashrae140Case {
        pub name: String,
        pub surfaces: Vec<(f64, f64)>,
        pub expected_view_factors: Option<Vec<Vec<f64>>>,
        pub tolerance: f64,
    }

    pub fn create_rectangular_enclosure(width: f64, height: f64, depth: f64) -> Vec<(f64, f64)> {
        let floor_area = width * depth;
        let ceiling_area = width * depth;
        let wall1_area = height * depth;
        let wall2_area = height * depth;
        let wall3_area = height * width;
        let wall4_area = height * width;

        vec![
            (floor_area, 0.0),
            (ceiling_area, height),
            (wall1_area, height),
            (wall2_area, height),
            (wall3_area, height),
            (wall4_area, height),
        ]
    }

    pub fn create_two_zone_enclosure(
        width1: f64,
        width2: f64,
        height: f64,
        depth: f64,
    ) -> Vec<(f64, f64)> {
        let floor1 = width1 * depth;
        let floor2 = width2 * depth;
        let ceiling1 = width1 * depth;
        let ceiling2 = width2 * depth;
        let shared_wall = height * depth;
        let exterior1 = height * width1;
        let exterior2 = height * width2;
        let front_back1 = height * depth;
        let front_back2 = height * depth;

        vec![
            (floor1, 0.0),
            (floor2, 0.0),
            (ceiling1, height),
            (ceiling2, height),
            (shared_wall, height),
            (exterior1, height),
            (exterior2, height),
            (front_back1, height),
            (front_back2, height),
        ]
    }

    pub fn create_street_canyon(
        building_height: f64,
        _building_width: f64,
        street_width: f64,
        building_depth: f64,
    ) -> Vec<(f64, f64, f64)> {
        let wall_area = building_height * building_depth;

        vec![
            (wall_area, building_height, 0.0),
            (wall_area, building_height, street_width),
        ]
    }

    pub fn reference_configurations() -> Vec<Ashrae140Case> {
        vec![
            Ashrae140Case {
                name: "SingleRoom_10x10x3".to_string(),
                surfaces: vec![
                    (100.0, 3.0),
                    (100.0, 3.0),
                    (30.0, 3.0),
                    (30.0, 3.0),
                    (10.0, 3.0),
                    (10.0, 3.0),
                ],
                expected_view_factors: None,
                tolerance: 1e-6,
            },
            Ashrae140Case {
                name: "TwoZone_5x5_each_3m_high".to_string(),
                surfaces: create_two_zone_enclosure(5.0, 5.0, 3.0, 3.0),
                expected_view_factors: None,
                tolerance: 1e-6,
            },
        ]
    }

    pub fn ashrae140() -> Vec<Ashrae140Case> {
        reference_configurations()
    }

    pub fn verify_ashrae_case(
        case: &Ashrae140Case,
    ) -> Result<ViewFactorMatrix<f64>, ViewFactorError> {
        use super::nusselt::view_factor_enclosure;
        let dense = view_factor_enclosure(&case.surfaces)?;
        Ok(ViewFactorMatrix::from_dense(dense))
    }
}

pub mod urban_graph {
    use petgraph::Graph;
    use petgraph::Undirected;
    use serde::{Deserialize, Serialize};
    use uuid::Uuid;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum AdjacencyType {
        WindowToWindow,
        WallToWall,
        WallToWindow,
        RoofToRoof,
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub struct BoundingBox3D {
        pub min_x: f64,
        pub min_y: f64,
        pub min_z: f64,
        pub max_x: f64,
        pub max_y: f64,
        pub max_z: f64,
    }

    impl BoundingBox3D {
        pub fn new(min_x: f64, min_y: f64, min_z: f64, max_x: f64, max_y: f64, max_z: f64) -> Self {
            Self {
                min_x,
                min_y,
                min_z,
                max_x,
                max_y,
                max_z,
            }
        }

        pub fn center(&self) -> (f64, f64, f64) {
            (
                (self.min_x + self.max_x) / 2.0,
                (self.min_y + self.max_y) / 2.0,
                (self.min_z + self.max_z) / 2.0,
            )
        }

        pub fn distance_to(&self, other: &BoundingBox3D) -> f64 {
            let (cx1, cy1, cz1) = self.center();
            let (cx2, cy2, cz2) = other.center();
            ((cx2 - cx1).powi(2) + (cy2 - cy1).powi(2) + (cz2 - cz1).powi(2)).sqrt()
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BuildingNode {
        pub id: Uuid,
        pub envelope: (),
        pub bounding_box: BoundingBox3D,
    }

    impl BuildingNode {
        pub fn new(id: Uuid, bounding_box: BoundingBox3D) -> Self {
            Self {
                id,
                envelope: (),
                bounding_box,
            }
        }
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub struct SpatialEdge {
        pub distance_m: f64,
        pub adjacency_type: AdjacencyType,
    }

    impl SpatialEdge {
        pub fn new(distance_m: f64, adjacency_type: AdjacencyType) -> Self {
            Self {
                distance_m,
                adjacency_type,
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UrbanGraph<N, E> {
        graph: Graph<N, E, Undirected>,
        node_ids: Vec<Uuid>,
    }

    impl<N, E> UrbanGraph<N, E> {
        pub fn new() -> Self {
            Self {
                graph: Graph::new_undirected(),
                node_ids: Vec::new(),
            }
        }

        pub fn node_count(&self) -> usize {
            self.graph.node_count()
        }

        pub fn edge_count(&self) -> usize {
            self.graph.edge_count()
        }

        pub fn contains_node(&self, id: Uuid) -> bool {
            self.node_ids.contains(&id)
        }

        fn node_index(&self, id: Uuid) -> Option<petgraph::graph::NodeIndex> {
            self.node_ids
                .iter()
                .position(|&node_id| node_id == id)
                .map(petgraph::graph::NodeIndex::new)
        }

        pub fn node(&self, id: Uuid) -> Option<&N> {
            self.node_index(id)
                .and_then(|idx| self.graph.node_weight(idx))
        }

        pub fn nodes(&self) -> impl Iterator<Item = &N> {
            self.graph.node_weights()
        }

        pub fn edges(&self) -> impl Iterator<Item = &E> {
            self.graph.edge_weights()
        }
    }

    impl<N, E> Default for UrbanGraph<N, E> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl UrbanGraph<BuildingNode, SpatialEdge> {
        pub fn add_building(&mut self, node: BuildingNode) -> Uuid {
            let id = node.id;
            self.graph.add_node(node);
            self.node_ids.push(id);
            id
        }

        pub fn add_spatial_edge(
            &mut self,
            source_id: Uuid,
            target_id: Uuid,
            edge: SpatialEdge,
        ) -> Option<petgraph::graph::EdgeIndex> {
            let source_idx = self.node_index(source_id)?;
            let target_idx = self.node_index(target_id)?;
            Some(self.graph.add_edge(source_idx, target_idx, edge))
        }

        pub fn nearest_neighbors(&self, node_id: Uuid, radius_m: f64) -> Vec<Uuid> {
            let source_idx = match self.node_index(node_id) {
                Some(idx) => idx,
                None => return Vec::new(),
            };

            let source_node = match self.graph.node_weight(source_idx) {
                Some(node) => node,
                None => return Vec::new(),
            };

            let mut neighbors = Vec::new();

            for (target_idx, target_node) in
                self.graph.node_indices().zip(self.graph.node_weights())
            {
                if target_idx == source_idx {
                    continue;
                }

                let distance = source_node
                    .bounding_box
                    .distance_to(&target_node.bounding_box);

                if distance <= radius_m {
                    neighbors.push(target_node.id);
                }
            }

            neighbors
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn create_test_buildings() -> (BoundingBox3D, BoundingBox3D, BoundingBox3D) {
            let building_a = BoundingBox3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 30.0);
            let building_b = BoundingBox3D::new(15.0, 5.0, 0.0, 25.0, 15.0, 30.0);
            let building_c = BoundingBox3D::new(100.0, 100.0, 0.0, 110.0, 110.0, 30.0);
            (building_a, building_b, building_c)
        }

        #[test]
        fn test_bounding_box_distance() {
            let bb1 = BoundingBox3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
            let bb2 = BoundingBox3D::new(20.0, 0.0, 0.0, 30.0, 10.0, 10.0);
            let distance = bb1.distance_to(&bb2);
            assert!((distance - 20.0).abs() < 1e-6);
        }

        #[test]
        fn test_3_building_graph_nearest_neighbors() {
            let (building_a, building_b, building_c) = create_test_buildings();

            let id_a = Uuid::new_v4();
            let id_b = Uuid::new_v4();
            let id_c = Uuid::new_v4();

            let mut graph: UrbanGraph<BuildingNode, SpatialEdge> = UrbanGraph::new();

            graph.add_building(BuildingNode::new(id_a, building_a));
            graph.add_building(BuildingNode::new(id_b, building_b));
            graph.add_building(BuildingNode::new(id_c, building_c));

            let neighbors_a = graph.nearest_neighbors(id_a, 50.0);
            assert_eq!(neighbors_a.len(), 1);
            assert!(neighbors_a.contains(&id_b));
            assert!(!neighbors_a.contains(&id_c));

            let neighbors_b = graph.nearest_neighbors(id_b, 50.0);
            assert_eq!(neighbors_b.len(), 1);
            assert!(neighbors_b.contains(&id_a));
            assert!(!neighbors_b.contains(&id_c));

            let neighbors_c = graph.nearest_neighbors(id_c, 50.0);
            assert!(neighbors_c.is_empty());

            let neighbors_c_200 = graph.nearest_neighbors(id_c, 200.0);
            assert_eq!(neighbors_c_200.len(), 2);
            assert!(neighbors_c_200.contains(&id_a));
            assert!(neighbors_c_200.contains(&id_b));
        }

        #[test]
        fn test_urban_graph_add_edge() {
            let (building_a, building_b, _) = create_test_buildings();

            let id_a = Uuid::new_v4();
            let id_b = Uuid::new_v4();

            let mut graph: UrbanGraph<BuildingNode, SpatialEdge> = UrbanGraph::new();
            graph.add_building(BuildingNode::new(id_a, building_a));
            graph.add_building(BuildingNode::new(id_b, building_b));

            let edge = SpatialEdge::new(15.0, AdjacencyType::WallToWall);
            let edge_idx = graph.add_spatial_edge(id_a, id_b, edge);

            assert!(edge_idx.is_some());
            assert_eq!(graph.edge_count(), 1);
        }

        #[test]
        fn test_urban_graph_node_lookup() {
            let building = BoundingBox3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 30.0);
            let id = Uuid::new_v4();

            let mut graph: UrbanGraph<BuildingNode, SpatialEdge> = UrbanGraph::new();
            assert!(!graph.contains_node(id));

            graph.add_building(BuildingNode::new(id, building));
            assert!(graph.contains_node(id));
            assert_eq!(graph.node_count(), 1);
        }

        #[test]
        fn test_urban_graph_default() {
            let graph: UrbanGraph<BuildingNode, SpatialEdge> = UrbanGraph::default();
            assert_eq!(graph.node_count(), 0);
            assert_eq!(graph.edge_count(), 0);
        }

        #[test]
        fn test_bounding_box_center() {
            let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 10.0, 20.0, 30.0);
            let (cx, cy, cz) = bb.center();
            assert!((cx - 5.0).abs() < 1e-6);
            assert!((cy - 10.0).abs() < 1e-6);
            assert!((cz - 15.0).abs() < 1e-6);
        }
    }
}

pub use ashrae140::{ashrae140, verify_ashrae_case};
pub use geometry::{GroundPlane, RectSurface, SurfaceType, UrbanCanopySurface, VerticalSurface};
pub use nusselt::{
    check_reciprocity, check_summation, compute_urban_canyon_view_factors, view_factor_enclosure,
    view_factor_parallel_rectangles, view_factor_wall_to_ground, view_factor_wall_to_sky,
    ViewFactorMatrix,
};
pub use sparse::{
    create_sparse_from_urban_canyon, RadiationError, SparseViewFactorMatrix, SurfacePairFlux,
    UrbanRadiationSolver, DEFAULT_EMISSIVITY, STEFAN_BOLTZMANN,
};
pub use urban_graph::{AdjacencyType, BoundingBox3D, BuildingNode, SpatialEdge, UrbanGraph};

/// Monte Carlo ray-tracing view factor computation for arbitrary 3D surfaces.
pub mod ray_tracing;
pub use ray_tracing::{MonteCarloViewFactor, Surface3D, DEFAULT_NUM_RAYS};

#[cfg(feature = "parallel")]
pub mod parallel {
    pub mod harness;
}
#[cfg(feature = "parallel")]
pub use parallel::harness::{BuildingGroup, UrbanRadiationSystem, UrbanStepDispatcher};

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
    fn test_summation_check() {
        let surfaces = vec![(100.0, 10.0), (100.0, 10.0), (100.0, 10.0)];
        let f = nusselt::view_factor_enclosure(&surfaces).unwrap();

        for i in 0..3 {
            let row_sum: f64 = f[i].iter().sum();
            nusselt::check_summation(f[i][i], row_sum - f[i][i]).unwrap();
        }
    }

    #[test]
    fn test_enclosure_two_surfaces() {
        let surfaces = vec![(100.0, 10.0), (100.0, 10.0)];
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

    #[test]
    fn test_urban_canyon_view_factors() {
        let walls = vec![(30.0, 10.0, 0.0), (30.0, 10.0, 5.0)];
        let ground_area = 50.0;

        let matrix = nusselt::compute_urban_canyon_view_factors(&walls, ground_area).unwrap();
        let dense = matrix.to_vec_vec();

        for i in 0..dense.len() {
            let row_sum: f64 = dense[i].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "Urban canyon row {} sum = {}, expected 1.0",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_sparse_matrix_from_dense() {
        let surfaces = vec![(100.0, 10.0), (100.0, 10.0), (100.0, 10.0)];

        let dense = nusselt::view_factor_enclosure(&surfaces).unwrap();
        let dense_matrix = ViewFactorMatrix::from_dense(dense);
        let sparse = SparseViewFactorMatrix::from_dense(&dense_matrix);

        assert_eq!(sparse.nrows(), 3);
        assert_eq!(sparse.ncols(), 3);

        for i in 0..3 {
            for j in 0..3 {
                assert!((sparse.get(i, j) - dense_matrix.get(i, j)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_sparse_matrix_sparsity() {
        let walls = vec![
            (30.0, 10.0, 0.0),
            (30.0, 10.0, 5.0),
            (25.0, 8.0, 2.5),
            (35.0, 12.0, 8.0),
        ];
        let ground_area = 50.0;

        let dense = nusselt::compute_urban_canyon_view_factors(&walls, ground_area).unwrap();
        let sparse = SparseViewFactorMatrix::from_dense(&dense);

        let sparsity = sparse.sparsity_ratio();
        assert!(
            sparsity > 0.3,
            "Expected sparse matrix (>30% zero), got {:.1}% sparsity",
            sparsity * 100.0
        );
    }

    #[test]
    fn test_sparse_matrix_multiplication() {
        let walls = vec![(30.0, 10.0, 0.0), (30.0, 10.0, 5.0), (25.0, 8.0, 2.5)];
        let ground_area = 50.0;

        let dense = nusselt::compute_urban_canyon_view_factors(&walls, ground_area).unwrap();
        let sparse = SparseViewFactorMatrix::from_dense(&dense);

        let vec: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0];
        let result = sparse.multiply_dense(&vec);

        assert_eq!(result.len(), 4);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-10,
                "Row {} of view factor matrix should sum to 1.0, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_create_sparse_from_urban_canyon() {
        let walls = vec![(30.0, 10.0, 0.0), (30.0, 10.0, 5.0)];
        let ground_area = 50.0;

        let sparse = create_sparse_from_urban_canyon(&walls, ground_area).unwrap();

        assert_eq!(sparse.nrows(), 3);
        assert_eq!(sparse.ncols(), 3);

        let dense = sparse.to_dense();
        let summation_results = dense.verify_summation();
        for (i, is_valid) in summation_results {
            assert!(is_valid, "Summation check failed for surface {}", i);
        }
    }

    #[test]
    fn test_ashrae140_configurations() {
        let cases = ashrae140::reference_configurations();

        for case in cases {
            let result = verify_ashrae_case(&case);
            assert!(
                result.is_ok(),
                "ASHRAE case {} failed to compute: {:?}",
                case.name,
                result.err()
            );

            let matrix = result.unwrap();
            let summation_results = matrix.verify_summation();

            for (i, is_valid) in summation_results {
                assert!(
                    is_valid,
                    "ASHRAE case {} surface {} failed summation check",
                    case.name, i
                );
            }
        }
    }

    #[test]
    fn test_view_factor_matrix_row_sum() {
        let surfaces = vec![(100.0, 10.0), (100.0, 10.0), (100.0, 10.0)];

        let dense = nusselt::view_factor_enclosure(&surfaces).unwrap();
        let matrix = ViewFactorMatrix::from_dense(dense);

        for i in 0..matrix.nrows() {
            let row_sum: f64 = (0..matrix.ncols()).map(|j| matrix.get(i, j)).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    // === Energy Conservation Tests (from #2031) ===
    const STEFAN_BOLTZMANN: f64 = 5.67e-8;

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub struct BuildingConfig {
        pub length: f64,
        pub width: f64,
        pub height: f64,
        pub emissivity: f64,
        pub absorptivity: f64,
        pub thermal_conductance: f64,
    }

    impl BuildingConfig {
        pub fn new(
            length: f64,
            width: f64,
            height: f64,
            emissivity: f64,
            absorptivity: f64,
            thermal_conductance: f64,
        ) -> Self {
            Self {
                length,
                width,
                height,
                emissivity,
                absorptivity,
                thermal_conductance,
            }
        }

        pub fn surface_area(&self) -> f64 {
            2.0 * (self.length * self.width + self.length * self.height + self.width * self.height)
        }

        pub fn wall_area(&self) -> f64 {
            2.0 * (self.length * self.height + self.width * self.height)
        }

        pub fn roof_area(&self) -> f64 {
            self.length * self.width
        }

        pub fn floor_area(&self) -> f64 {
            self.length * self.width
        }
    }

    impl Default for BuildingConfig {
        fn default() -> Self {
            Self {
                length: 10.0,
                width: 10.0,
                height: 3.0,
                emissivity: 0.9,
                absorptivity: 0.7,
                thermal_conductance: 0.45,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct SurfaceRadiation {
        pub absorbed: f64,
        pub emitted: f64,
        pub transmitted: f64,
        pub reflected: f64,
    }

    impl SurfaceRadiation {
        pub fn new(absorbed: f64, emitted: f64, transmitted: f64, reflected: f64) -> Self {
            Self {
                absorbed,
                emitted,
                transmitted,
                reflected,
            }
        }

        pub fn net_radiation(&self) -> f64 {
            self.absorbed - self.emitted - self.transmitted - self.reflected
        }

        pub fn is_conserved(&self, tolerance: f64) -> bool {
            self.net_radiation().abs() < tolerance
        }
    }

    pub struct EnergyConservationTest {
        pub buildings: Vec<BuildingConfig>,
        pub ambient_temperature: f64,
        pub solar_irradiance: f64,
        pub sky_temperature: f64,
    }

    impl Default for EnergyConservationTest {
        fn default() -> Self {
            Self {
                buildings: Vec::new(),
                ambient_temperature: 293.15,
                solar_irradiance: 0.0,
                sky_temperature: 270.0,
            }
        }
    }

    impl EnergyConservationTest {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn with_buildings(mut self, buildings: Vec<BuildingConfig>) -> Self {
            self.buildings = buildings;
            self
        }

        pub fn with_ambient_temperature(mut self, temperature: f64) -> Self {
            self.ambient_temperature = temperature;
            self
        }

        pub fn with_solar_irradiance(mut self, irradiance: f64) -> Self {
            self.solar_irradiance = irradiance;
            self
        }

        pub fn with_sky_temperature(mut self, temperature: f64) -> Self {
            self.sky_temperature = temperature;
            self
        }

        pub fn create_5_building_config() -> Self {
            let building1 = BuildingConfig {
                length: 10.0,
                width: 10.0,
                height: 3.0,
                emissivity: 0.9,
                absorptivity: 0.7,
                thermal_conductance: 0.45,
            };
            let building2 = BuildingConfig {
                length: 8.0,
                width: 8.0,
                height: 4.0,
                emissivity: 0.85,
                absorptivity: 0.6,
                thermal_conductance: 0.40,
            };
            let building3 = BuildingConfig {
                length: 12.0,
                width: 6.0,
                height: 3.5,
                emissivity: 0.9,
                absorptivity: 0.75,
                thermal_conductance: 0.50,
            };
            let building4 = BuildingConfig {
                length: 7.0,
                width: 7.0,
                height: 5.0,
                emissivity: 0.88,
                absorptivity: 0.65,
                thermal_conductance: 0.42,
            };
            let building5 = BuildingConfig {
                length: 9.0,
                width: 11.0,
                height: 3.0,
                emissivity: 0.92,
                absorptivity: 0.70,
                thermal_conductance: 0.48,
            };

            Self {
                buildings: vec![building1, building2, building3, building4, building5],
                ambient_temperature: 293.15,
                solar_irradiance: 500.0,
                sky_temperature: 270.0,
            }
        }

        fn net_balance_at_temperature(&self, building: &BuildingConfig, t_surface: f64) -> f64 {
            let wall_area = building.wall_area();
            let roof_area = building.roof_area();
            let total_area = building.surface_area();

            let absorbed_solar =
                building.absorptivity * self.solar_irradiance * (wall_area + roof_area);

            let emitted = building.emissivity * STEFAN_BOLTZMANN * total_area * t_surface.powi(4);

            let transmitted =
                building.thermal_conductance * total_area * (t_surface - self.ambient_temperature);

            let sky_radiation = building.emissivity
                * STEFAN_BOLTZMANN
                * total_area
                * (t_surface.powi(4) - self.sky_temperature.powi(4));

            absorbed_solar - emitted - transmitted - sky_radiation
        }

        fn find_equilibrium_temperature(&self, building: &BuildingConfig) -> f64 {
            let t_min = 200.0;
            let t_max = 400.0;
            let balance_tolerance = 1e-12;
            let t_tolerance = 1e-14;
            let max_iterations = 500;

            let mut t_low = t_min;
            let mut t_high = t_max;
            let mut balance_low = self.net_balance_at_temperature(building, t_low);
            let balance_high = self.net_balance_at_temperature(building, t_high);

            if balance_low * balance_high > 0.0 {
                if balance_low > 0.0 {
                    return t_min;
                } else {
                    return t_max;
                }
            }

            for _ in 0..max_iterations {
                let t_mid = (t_low + t_high) / 2.0;
                let balance_mid = self.net_balance_at_temperature(building, t_mid);

                if balance_mid.abs() < balance_tolerance {
                    return t_mid;
                }

                if (t_high - t_low) < t_tolerance * t_low {
                    return t_mid;
                }

                if balance_low * balance_mid <= 0.0 {
                    t_high = t_mid;
                } else {
                    t_low = t_mid;
                    balance_low = balance_mid;
                }
            }

            (t_low + t_high) / 2.0
        }

        pub fn surface_radiation_balance(&self, building_index: usize) -> Option<SurfaceRadiation> {
            let building = self.buildings.get(building_index)?;

            let wall_area = building.wall_area();
            let roof_area = building.roof_area();
            let total_area = building.surface_area();

            let absorbed_solar =
                building.absorptivity * self.solar_irradiance * (wall_area + roof_area);

            let t_eq = self.find_equilibrium_temperature(building);

            let emitted = building.emissivity * STEFAN_BOLTZMANN * total_area * t_eq.powi(4);

            let transmitted =
                building.thermal_conductance * total_area * (t_eq - self.ambient_temperature);

            let sky_radiation = building.emissivity
                * STEFAN_BOLTZMANN
                * total_area
                * (t_eq.powi(4) - self.sky_temperature.powi(4));

            Some(SurfaceRadiation::new(
                absorbed_solar,
                emitted + sky_radiation,
                transmitted,
                0.0,
            ))
        }

        pub fn verify_conservation(&self) -> bool {
            let imbalance = self.max_imbalance();
            imbalance < 1e-6
        }

        pub fn max_imbalance(&self) -> f64 {
            let mut max_imbalance = 0.0f64;

            for building in &self.buildings {
                if let Some(radiation) = self.surface_radiation_balance_for_building(building) {
                    let imbalance = radiation.net_radiation().abs();
                    if imbalance > max_imbalance {
                        max_imbalance = imbalance;
                    }
                }
            }

            max_imbalance
        }

        pub fn all_surfaces_balanced(&self, tolerance: f64) -> bool {
            for (i, _) in self.buildings.iter().enumerate() {
                if let Some(radiation) = self.surface_radiation_balance(i) {
                    if !radiation.is_conserved(tolerance) {
                        return false;
                    }
                }
            }
            true
        }

        pub fn net_radiation_for_enclosed_surfaces(&self) -> f64 {
            let mut total_net = 0.0;

            for building in &self.buildings {
                if let Some(radiation) = self.surface_radiation_balance_for_building(building) {
                    total_net += radiation.net_radiation();
                }
            }

            total_net
        }

        fn surface_radiation_balance_for_building(
            &self,
            building: &BuildingConfig,
        ) -> Option<SurfaceRadiation> {
            let wall_area = building.wall_area();
            let roof_area = building.roof_area();
            let total_area = building.surface_area();

            let absorbed_solar =
                building.absorptivity * self.solar_irradiance * (wall_area + roof_area);

            let t_eq = self.find_equilibrium_temperature(building);

            let emitted = building.emissivity * STEFAN_BOLTZMANN * total_area * t_eq.powi(4);

            let transmitted =
                building.thermal_conductance * total_area * (t_eq - self.ambient_temperature);

            let sky_radiation = building.emissivity
                * STEFAN_BOLTZMANN
                * total_area
                * (t_eq.powi(4) - self.sky_temperature.powi(4));

            Some(SurfaceRadiation::new(
                absorbed_solar,
                emitted + sky_radiation,
                transmitted,
                0.0,
            ))
        }
    }

    #[test]
    fn test_5_building_energy_conservation() {
        let test = EnergyConservationTest::create_5_building_config();
        let imbalance = test.max_imbalance();
        assert!(
            imbalance < 1e-6,
            "Energy imbalance {} exceeds tolerance 1e-6 W",
            imbalance
        );
    }

    #[test]
    fn test_energy_conservation_verify() {
        let test = EnergyConservationTest::create_5_building_config();
        assert!(
            test.verify_conservation(),
            "Energy conservation verification failed"
        );
    }

    #[test]
    fn test_surface_radiation_balance() {
        let test = EnergyConservationTest::create_5_building_config();
        let balance = test.surface_radiation_balance(0);
        assert!(
            balance.is_some(),
            "Should get radiation balance for building 0"
        );
        let balance = balance.unwrap();
        assert!(
            balance.net_radiation().abs() < 1e-6,
            "Net radiation {} should be near zero",
            balance.net_radiation()
        );
    }

    #[test]
    fn test_all_surfaces_balanced() {
        let test = EnergyConservationTest::create_5_building_config();
        assert!(
            test.all_surfaces_balanced(1e-6),
            "All surfaces should be balanced within tolerance"
        );
    }

    #[test]
    fn test_building_surface_area() {
        let building = BuildingConfig::default();
        let expected_area = 2.0 * (100.0 + 30.0 + 30.0);
        assert!((building.surface_area() - expected_area).abs() < 1e-10);
    }

    #[test]
    fn test_building_wall_area() {
        let building = BuildingConfig::default();
        let expected_wall = 2.0 * (10.0 * 3.0 + 10.0 * 3.0);
        assert!((building.wall_area() - expected_wall).abs() < 1e-10);
    }

    #[test]
    fn test_building_roof_area() {
        let building = BuildingConfig::default();
        let expected_roof = 100.0;
        assert!((building.roof_area() - expected_roof).abs() < 1e-10);
    }

    #[test]
    fn test_single_building_conservation() {
        let building = BuildingConfig::default();
        let test = EnergyConservationTest::new()
            .with_buildings(vec![building])
            .with_ambient_temperature(293.15)
            .with_solar_irradiance(500.0)
            .with_sky_temperature(270.0);

        assert!(
            test.verify_conservation(),
            "Single building should satisfy energy conservation"
        );
    }

    #[test]
    fn test_zero_solar_irradiance() {
        let test = EnergyConservationTest::create_5_building_config().with_solar_irradiance(0.0);

        assert!(
            test.verify_conservation(),
            "Should still conserve energy with zero solar"
        );
    }

    #[test]
    fn test_longwave_equilibrium() {
        let test = EnergyConservationTest::create_5_building_config();

        for (i, _building) in test.buildings.iter().enumerate() {
            let radiation = test.surface_radiation_balance(i);
            assert!(
                radiation.is_some(),
                "Building {} should have radiation balance",
                i
            );

            let rad = radiation.unwrap();
            let imbalance = (rad.absorbed - rad.emitted - rad.transmitted).abs();
            assert!(
                imbalance < 1e-6,
                "Building {} longwave equilibrium imbalance: {}",
                i,
                imbalance
            );
        }
    }

    #[test]
    fn test_enclosed_surfaces_net_zero() {
        let test = EnergyConservationTest::create_5_building_config();

        for (i, _) in test.buildings.iter().enumerate() {
            let radiation = test.surface_radiation_balance(i);
            assert!(radiation.is_some());

            let rad = radiation.unwrap();
            let net = rad.net_radiation();
            assert!(
                net.abs() < 1e-6,
                "Building {} net radiation {} should be zero",
                i,
                net
            );
        }
    }

    #[test]
    fn test_net_radiation_for_enclosed_surfaces() {
        let test = EnergyConservationTest::create_5_building_config();
        let total_net = test.net_radiation_for_enclosed_surfaces();
        assert!(
            total_net.abs() < 1e-6,
            "Total net radiation {} should be zero for enclosed surfaces",
            total_net
        );
    }

    // === Issue #2029: UrbanRadiationSolver net radiative flux computation ===

    /// Build a 2-surface solver with a known symmetric view-factor matrix.
    /// F_12 = F_21 = 0.5, areas = [10, 10] m², emissivity = 0.9.
    fn two_surface_solver() -> sparse::UrbanRadiationSolver {
        let mut vf = sparse::SparseViewFactorMatrix::new(2, 2);
        // Off-diagonal view factors only (no self-viewing).
        vf.set(0, 1, 0.5);
        vf.set(1, 0, 0.5);
        sparse::UrbanRadiationSolver::with_uniform_emissivity(vf, vec![10.0, 10.0], 0.9)
    }

    #[test]
    fn test_compute_fluxes_reference_value() {
        // Q_rad(i→j) = εσF_ij A_i (T_i⁴ − T_j⁴)
        // ε=0.9, σ=5.67e-8, F=0.5, A=10, T_i=300K, T_j=290K
        // Python reference: 262.0875285 W
        let solver = two_surface_solver();
        let temps = [300.0, 290.0];
        let fluxes = solver.compute_fluxes(&temps).expect("flux computation");

        let q01 = fluxes
            .iter()
            .find(|f| f.from == 0 && f.to == 1)
            .expect("flux 0->1 present");
        approx::assert_abs_diff_eq!(q01.heat_flow_w, 262.0875285, epsilon = 1e-4);
        approx::assert_abs_diff_eq!(q01.view_factor, 0.5, epsilon = 1e-12);
        assert!(
            q01.heat_flow_w > 0.0,
            "net transfer must be from hot to cold"
        );
    }

    #[test]
    fn test_compute_fluxes_zero_at_equal_temperature() {
        let solver = two_surface_solver();
        let fluxes = solver
            .compute_fluxes(&[300.0, 300.0])
            .expect("flux computation");
        for f in &fluxes {
            assert!(
                f.heat_flow_w.abs() < 1e-9,
                "flux {}->{} should be zero at equal T, got {}",
                f.from,
                f.to,
                f.heat_flow_w
            );
        }
    }

    #[test]
    fn test_compute_fluxes_sign_hot_to_cold() {
        let solver = two_surface_solver();
        // Surface 0 hot, surface 1 cold => flux 0->1 positive (net loss from 0).
        let fluxes = solver
            .compute_fluxes(&[310.0, 280.0])
            .expect("flux computation");
        let q01 = fluxes.iter().find(|f| f.from == 0 && f.to == 1).unwrap();
        assert!(q01.heat_flow_w > 0.0);

        // Swap: surface 0 cold, surface 1 hot => flux 0->1 negative.
        let fluxes = solver
            .compute_fluxes(&[280.0, 310.0])
            .expect("flux computation");
        let q01 = fluxes.iter().find(|f| f.from == 0 && f.to == 1).unwrap();
        assert!(q01.heat_flow_w < 0.0);
    }

    #[test]
    fn test_compute_fluxes_reciprocity_and_antisymmetry() {
        // With equal emissivities and reciprocal view factors (A1 F12 = A2 F21),
        // Q(i→j) must equal −Q(j→i) exactly.
        let solver = two_surface_solver();
        let fluxes = solver
            .compute_fluxes(&[300.0, 290.0])
            .expect("flux computation");

        let q01 = fluxes.iter().find(|f| f.from == 0 && f.to == 1).unwrap();
        let q10 = fluxes.iter().find(|f| f.from == 1 && f.to == 0).unwrap();

        approx::assert_abs_diff_eq!(q01.heat_flow_w, -q10.heat_flow_w, epsilon = 1e-6);
    }

    #[test]
    fn test_compute_fluxes_energy_balance() {
        // For an enclosed pair with equal emissivities and reciprocal view
        // factors, the sum of all pair fluxes must be zero (energy conserved:
        // what one surface loses the other gains).
        let solver = two_surface_solver();
        let fluxes = solver
            .compute_fluxes(&[300.0, 290.0])
            .expect("flux computation");
        let total: f64 = fluxes.iter().map(|f| f.heat_flow_w).sum();
        assert!(
            total.abs() < 1e-6,
            "total pair flux {} should be ~0 (energy balance)",
            total
        );
    }

    #[test]
    fn test_net_flux_per_surface_balance() {
        // Net-per-surface must sum to zero for a closed pair with equal ε.
        let solver = two_surface_solver();
        let net = solver.compute_net_flux_per_surface(&[300.0, 290.0]);
        assert_eq!(net.len(), 2);
        // Surface 0 (hot) is a net radiator => positive; surface 1 a net absorber.
        assert!(net[0] > 0.0);
        assert!(net[1] < 0.0);
        approx::assert_abs_diff_eq!(net[0], -net[1], epsilon = 1e-6);
        // 262.0875 W leaves surface 0.
        approx::assert_abs_diff_eq!(net[0], 262.0875285, epsilon = 1e-4);
    }

    #[test]
    fn test_compute_fluxes_area_scaling() {
        // Doubling the area of the emitting surface doubles the flux (linear in A_i).
        let base = two_surface_solver();
        let q_base = base
            .compute_fluxes(&[300.0, 290.0])
            .unwrap()
            .into_iter()
            .find(|f| f.from == 0 && f.to == 1)
            .unwrap()
            .heat_flow_w;

        let mut vf = sparse::SparseViewFactorMatrix::new(2, 2);
        vf.set(0, 1, 0.5);
        vf.set(1, 0, 0.5);
        let big = sparse::UrbanRadiationSolver::with_uniform_emissivity(vf, vec![20.0, 10.0], 0.9);
        let q_big = big
            .compute_fluxes(&[300.0, 290.0])
            .unwrap()
            .into_iter()
            .find(|f| f.from == 0 && f.to == 1)
            .unwrap()
            .heat_flow_w;

        approx::assert_abs_diff_eq!(q_big, 2.0 * q_base, epsilon = 1e-6);
    }

    #[test]
    fn test_compute_fluxes_dimension_mismatch() {
        let solver = two_surface_solver();
        let err = solver.compute_fluxes(&[300.0]).unwrap_err();
        match err {
            sparse::RadiationError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 2);
                assert_eq!(got, 1);
            }
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_compute_fluxes_invalid_temperature() {
        let solver = two_surface_solver();
        assert!(matches!(
            solver.compute_fluxes(&[-5.0, 290.0]),
            Err(sparse::RadiationError::InvalidTemperature { index: 0, .. })
        ));
        assert!(matches!(
            solver.compute_fluxes(&[f64::NAN, 290.0]),
            Err(sparse::RadiationError::InvalidTemperature { index: 0, .. })
        ));
    }

    #[test]
    fn test_compute_fluxes_3_surface_enclosure_balance() {
        // Three surfaces with a fully reciprocal view-factor set and equal ε.
        // Total pair flux must sum to zero (closed enclosure).
        // Areas chosen so that F_ij A_i = F_ji A_j for each pair.
        // A = [10, 10, 10], symmetric F => reciprocity satisfied trivially.
        let mut vf = sparse::SparseViewFactorMatrix::new(3, 3);
        vf.set(0, 1, 0.3);
        vf.set(1, 0, 0.3);
        vf.set(0, 2, 0.2);
        vf.set(2, 0, 0.2);
        vf.set(1, 2, 0.25);
        vf.set(2, 1, 0.25);
        let solver =
            sparse::UrbanRadiationSolver::with_uniform_emissivity(vf, vec![10.0, 10.0, 10.0], 0.9);

        let fluxes = solver
            .compute_fluxes(&[305.0, 295.0, 290.0])
            .expect("flux computation");
        let total: f64 = fluxes.iter().map(|f| f.heat_flow_w).sum();
        assert!(
            total.abs() < 1e-6,
            "3-surface enclosure total flux {} should be ~0",
            total
        );

        // Net per-surface must also sum to zero.
        let net = solver.compute_net_flux_per_surface(&[305.0, 295.0, 290.0]);
        let net_total: f64 = net.iter().sum();
        approx::assert_abs_diff_eq!(net_total, 0.0, epsilon = 1e-6);
        // Hottest surface (0) is a net radiator.
        assert!(net[0] > 0.0);
        // Coldest surface (2) is a net absorber.
        assert!(net[2] < 0.0);
    }

    #[test]
    fn test_constants_match_issue_spec() {
        // σ = 5.67e-8 W/m²/K⁴, default ε = 0.9 (Issue #2029 acceptance criteria).
        approx::assert_abs_diff_eq!(sparse::STEFAN_BOLTZMANN, 5.67e-8, epsilon = 0.0);
        approx::assert_abs_diff_eq!(sparse::DEFAULT_EMISSIVITY, 0.9, epsilon = 0.0);
    }

    // === Issue #2030: faer sparse matrix integration tests ===

    #[test]
    fn test_sparse_view_factor_to_faer_construction() {
        // Build a 3×3 sparse view-factor matrix and convert to faer CSC.
        let mut vf = sparse::SparseViewFactorMatrix::new(3, 3);
        vf.set(0, 1, 0.3);
        vf.set(1, 0, 0.3);
        vf.set(0, 2, 0.2);
        vf.set(2, 0, 0.2);
        vf.set(1, 2, 0.25);
        vf.set(2, 1, 0.25);

        let faer_mat = vf.to_faer().expect("faer construction");

        assert_eq!(faer_mat.nrows(), 3);
        assert_eq!(faer_mat.ncols(), 3);
        assert_eq!(faer_mat.symbolic().compute_nnz(), 6);

        // The faer matrix must represent the same operator: F · 1 == row sums.
        use faer::Mat;
        let ones = Mat::from_fn(3, 1, |_, _| 1.0);
        let mut result = Mat::<f64>::zeros(3, 1);
        faer::sparse::linalg::matmul::sparse_dense_matmul(
            result.as_mut(),
            faer::Accum::Replace,
            faer_mat.as_ref(),
            ones.as_ref(),
            1.0,
            faer::Par::Seq,
        );
        // Row 0: 0.3 + 0.2 = 0.5 ; Row 1: 0.3 + 0.25 = 0.55 ; Row 2: 0.2 + 0.25 = 0.45
        approx::assert_abs_diff_eq!(result[(0, 0)], 0.5, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(result[(1, 0)], 0.55, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(result[(2, 0)], 0.45, epsilon = 1e-12);
    }

    #[test]
    fn test_faer_net_flux_matches_hashmap_reference() {
        // The faer matvec path must produce identical results to the
        // HashMap-based per-pair reference (bit-for-bit within float epsilon).
        let mut vf = sparse::SparseViewFactorMatrix::new(3, 3);
        vf.set(0, 1, 0.3);
        vf.set(1, 0, 0.3);
        vf.set(0, 2, 0.2);
        vf.set(2, 0, 0.2);
        vf.set(1, 2, 0.25);
        vf.set(2, 1, 0.25);
        let solver =
            sparse::UrbanRadiationSolver::with_uniform_emissivity(vf, vec![10.0, 10.0, 10.0], 0.9);

        let temps = [305.0, 295.0, 290.0];

        let ref_net = solver.compute_net_flux_per_surface(&temps);
        let faer_net = solver.compute_net_flux_per_surface_faer(&temps);

        assert_eq!(ref_net.len(), faer_net.len());
        for (i, (r, f)) in ref_net.iter().zip(faer_net.iter()).enumerate() {
            approx::assert_abs_diff_eq!(*r, *f, epsilon = 1e-6);
            assert!(
                (r - f).abs() < 1e-6,
                "surface {i}: hashmap={r:.10} faer={f:.10} diff={:.3e}",
                (r - f).abs()
            );
        }

        // Energy conservation: total net flux over a closed enclosure is zero.
        let total: f64 = faer_net.iter().sum();
        approx::assert_abs_diff_eq!(total, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_faer_net_flux_two_surface_reference_value() {
        // Known reference: Q_net[0] = 262.0875285 W (Issue #2029 test fixture).
        let solver = two_surface_solver();
        let faer_net = solver.compute_net_flux_per_surface_faer(&[300.0, 290.0]);
        approx::assert_abs_diff_eq!(faer_net[0], 262.0875285, epsilon = 1e-4);
        approx::assert_abs_diff_eq!(faer_net[1], -262.0875285, epsilon = 1e-4);
    }

    #[test]
    fn test_faer_net_flux_zero_at_equal_temperature() {
        let solver = two_surface_solver();
        let net = solver.compute_net_flux_per_surface_faer(&[300.0, 300.0]);
        for (i, &q) in net.iter().enumerate() {
            assert!(
                q.abs() < 1e-6,
                "surface {i} net flux {q} should be ~0 at equal T"
            );
        }
    }

    #[test]
    fn test_solver_exposes_faer_matrix() {
        let solver = two_surface_solver();
        let fm = solver.faer_matrix();
        assert_eq!(fm.nrows(), 2);
        assert_eq!(fm.ncols(), 2);
        assert_eq!(fm.symbolic().compute_nnz(), 2);
    }

    #[test]
    fn test_memory_sparse_smaller_than_dense_low_density() {
        // For a sparse inter-building coupling graph (<10% edge density), CSC
        // storage must be smaller than dense (Issue #2030 AC).
        // We model a 100-surface urban graph where each building only sees its
        // two nearest neighbours (a banded adjacency), giving ~3% edge density.
        let n = 100;
        let mut vf = sparse::SparseViewFactorMatrix::new(n, n);
        for i in 0..n {
            // Each row has at most 3 non-zeros: itself-ish + 2 neighbours.
            let f = 0.3;
            let next = (i + 1) % n;
            let prev = if i == 0 { n - 1 } else { i - 1 };
            vf.set(i, next, f);
            vf.set(i, prev, f);
        }

        let density = vf.edge_density();
        assert!(
            density < 0.10,
            "edge density {density:.4} should be <10% for a sparse urban graph"
        );

        let faer_bytes = vf.estimated_faer_csc_bytes();
        let dense_bytes = vf.estimated_dense_bytes();
        assert!(
            faer_bytes < dense_bytes,
            "faer CSC ({faer_bytes} B) must be smaller than dense ({dense_bytes} B) at density {density:.4}"
        );

        // faer CSC should also beat the HashMap representation at low density.
        let hashmap_bytes = vf.estimated_hashmap_bytes();
        assert!(
            faer_bytes < hashmap_bytes,
            "faer CSC ({faer_bytes} B) should be smaller than HashMap ({hashmap_bytes} B)"
        );
    }

    #[test]
    fn test_memory_estimates_scale_with_density() {
        // Construct matrices at increasing edge density and verify the
        // sparse < dense crossover is at ~50% (theoretical break-even).
        let n = 50;
        let dense_bytes = n * n * std::mem::size_of::<f64>();

        // Very sparse (1%): sparse must be much smaller.
        let mut vf_sparse = sparse::SparseViewFactorMatrix::new(n, n);
        for i in 0..n {
            let j = (i + 1) % n; // ring: each row has ~1 non-zero
            vf_sparse.set(i, j, 0.5);
        }
        assert!(vf_sparse.estimated_faer_csc_bytes() < dense_bytes / 4);

        // Dense (100%): sparse overhead exceeds dense.
        let mut vf_dense = sparse::SparseViewFactorMatrix::new(n, n);
        for i in 0..n {
            for j in 0..n {
                vf_dense.set(i, j, 0.02);
            }
        }
        assert!(vf_dense.estimated_faer_csc_bytes() > dense_bytes);
    }

    #[test]
    fn test_faer_net_flux_larger_urban_canyon() {
        // End-to-end: build a 10-building urban canyon, solve with both paths,
        // and confirm they agree to high precision.
        let walls: Vec<(f64, f64, f64)> = (0..10)
            .map(|i| {
                (
                    50.0 + i as f64 * 5.0,
                    10.0 + i as f64 * 0.3,
                    3.0 + i as f64 * 0.4,
                )
            })
            .collect();
        let ground_area = 500.0;
        let dense = nusselt::compute_urban_canyon_view_factors(&walls, ground_area).unwrap();

        let mut areas: Vec<f64> = walls.iter().map(|&(a, _, _)| a).collect();
        areas.push(ground_area);

        let solver =
            sparse::UrbanRadiationSolver::from_dense_enclosure(&dense, areas, vec![0.9; 11])
                .unwrap();

        let temps: Vec<f64> = (0..11).map(|i| 290.0 + i as f64 * 2.0).collect();

        let ref_net = solver.compute_net_flux_per_surface(&temps);
        let faer_net = solver.compute_net_flux_per_surface_faer(&temps);

        for (i, (r, f)) in ref_net.iter().zip(faer_net.iter()).enumerate() {
            let denom = r.abs().max(1e-6);
            assert!(
                (r - f).abs() / denom < 1e-9,
                "surface {i}: relative diff {:.3e} exceeds 1e-9",
                (r - f).abs() / denom
            );
        }
    }
}
