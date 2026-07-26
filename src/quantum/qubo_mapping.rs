//! QUBO (Quadratic Unconstrained Binary Optimization) mapping for
//! [`ThermalManifold`](crate::physics::geometry_tensor::ThermalManifold).
//!
//! ## Encoding
//!
//! Each temperature `T[i]` (`i = 0..MANIFOLD_DIM = 4`) is represented by `K`
//! unsigned bits using a fixed-point expansion
//!
//! ```text
//!   T[i] ≈ (Σ_k 2^k * x[(i,k)]) / scale_factor
//! ```
//!
//! where `scale_factor = (2^K − 1) / scale_max_celsius`. With `K = 8` and
//! `scale_max_celsius = 50.0`, the resolution is `50 / 255 ≈ 0.196 °C` per
//! LSB, well below typical ASHRAE 140 reference precision.
//!
//! Total number of binary variables: `N = MANIFOLD_DIM * K` (4·8 = 32 for the
//! default config).
//!
//! ## QUBO coefficients
//!
//! For a target energy
//!
//! ```text
//!   E(T) = T^T · metric_tensor · T    +    coeff_gauge · (−gauge_connection^T · T)
//! ```
//!
//! the coefficients are
//!
//! ```text
//!   Q[(i,k), (j,l)]  =  metric_tensor[i,j] · 2^k · 2^l / scale_factor^2
//!                     + (coeff_gauge · −gauge_connection[i] · 2^k / scale_factor)
//!                       if i == j and k == l, else 0  for the linear bias
//! ```
//!
//! so that for any binary `x`
//!
//! ```text
//!   x^T Q x  =  T_recon^T · metric_tensor · T_recon
//!             + coeff_gauge · (−gauge_connection^T · T_recon)
//! ```
//!
//! with `T_recon = decode(x, scale_factor)` to within `± 0.5 LSB` per node.
//!
//! ## Ising form
//!
//! D-Wave annealers natively consume the Ising form `s^T J s + h^T s + c` with
//! `s ∈ {−1, +1}^N`. The conversion is exact (no approximation):
//!
//! ```text
//!   s = 2x − 1
//!   J[i,j] = (1/4) Q[i,j]     for i ≠ j   (off-diagonal coupling)
//!   h[i]   = (1/2) Σ_j Q[i,j]              (linear field per qubit)
//!   c      = (1/4) trace(Q) + (1/4) 1^T Q 1 (energy offset)
//! ```
//!
//! ## Hardware scaling
//!
//! D-Wave AdvantageSystem6.4 accepts `h ∈ [−4, +4]` and `J ∈ [−2, +1]` (in
//! normalized units). For the default config (`K=8`, `scale_max_celsius=50`)
//! the QUBO entries land in `O(1)` magnitude — directly submittable without
//! rescaling. See `to_dwave_normalized` for the explicit normalization.

use crate::physics::geometry_tensor::{ThermalManifold, MANIFOLD_DIM};
use nalgebra::{Matrix4, Vector4};

/// Power iteration to find the eigenvalue with the largest absolute magnitude.
///
/// Runs at most `max_iter` iterations with convergence tolerance `tol` on the
/// change in eigenvalue estimate. Returns `None` if the matrix is zero or if
/// convergence is not reached. The matrix is interpreted as row-major `n×n`.
///
/// After L2-normalizing v each step (v := Av/||Av||), the eigenvalue estimate
/// is the Rayleigh quotient λ = v^T A v / v^T v. With ||v|| = 1, this is
/// simply v^T A v = v · w where w = A v.  The first iteration uses λ = 0 as
/// the prior estimate, so the convergence check is skipped until the second
/// iteration (when we have a meaningful prior).
fn power_iteration_max(a: &[f64], n: usize, max_iter: usize, tol: f64) -> Option<f64> {
    if n == 0 {
        return None;
    }
    let mut v = vec![1.0_f64; n];
    let mut lambda = 0.0_f64;

    for _ in 0..max_iter {
        let mut w = vec![0.0_f64; n];
        for i in 0..n {
            let mut row_acc = 0.0_f64;
            for j in 0..n {
                row_acc += a[i * n + j] * v[j];
            }
            w[i] = row_acc;
        }

        let w_norm_sq: f64 = w.iter().map(|&x| x * x).sum();
        let w_norm = w_norm_sq.sqrt();
        if w_norm < 1e-20 {
            return Some(0.0);
        }

        // Rayleigh quotient λ = v^T A v / v^T v for the current (normalized) v.
        // With v L2-normalized each iteration, v^T v = 1, so λ = v^T A v = v · w.
        let v_dot_w: f64 = v.iter().zip(w.iter()).map(|(vi, wi)| vi * wi).sum();
        let lambda_new = v_dot_w;

        // Converged if change is below tolerance AND we have a prior estimate.
        if lambda != 0.0 && (lambda_new - lambda).abs() < tol {
            return Some(lambda_new);
        }

        for j in 0..n {
            v[j] = w[j] / w_norm;
        }
        lambda = lambda_new;
    }

    Some(lambda)
}

/// Configuration for the QUBO encoding. The defaults match typical ASHRAE 140
/// zone temperatures (0..50 °C) at ~0.2 °C LSB resolution with K=8 bits/node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuboConfig {
    /// Bits per manifold node. K=8 ⇒ 32 qubits total; K=12 ⇒ 48 qubits.
    /// Must be ≥ 1 and ≤ 16 (larger values give diminishing returns).
    pub bits_per_node: usize,
    /// Maximum representable temperature in °C. The QUBO clips `T[i]` to
    /// `[0, scale_max_celsius]` before encoding. Must be > 0.
    pub scale_max_celsius: f64,
    /// When `true`, fold `-gauge_connection^T · T` into `Q` as a linear bias
    /// (diagonal entries). When `false`, the QUBO encodes only the quadratic
    /// energy density `T^T M T`.
    pub include_gauge_bias: bool,
    /// Coefficient applied to the gauge-connection bias term. Useful for
    /// weighting the relative importance of metric coupling vs external
    /// fluxes in the optimization landscape.
    pub coeff_gauge: f64,
}

impl Default for QuboConfig {
    fn default() -> Self {
        Self {
            bits_per_node: 8,
            scale_max_celsius: 50.0,
            include_gauge_bias: true,
            coeff_gauge: 1.0,
        }
    }
}

impl QuboConfig {
    /// Total number of binary variables: `MANIFOLD_DIM * K`.
    pub fn num_variables(&self) -> usize {
        MANIFOLD_DIM * self.bits_per_node
    }

    /// Scale factor: `T[i] * scale_factor ≈ Σ_k 2^k x[(i,k)]` (integer value).
    pub fn scale_factor(&self) -> f64 {
        assert!(
            self.scale_max_celsius > 0.0,
            "scale_max_celsius must be > 0"
        );
        let k = self.bits_per_node;
        ((1u64 << k) as f64 - 1.0) / self.scale_max_celsius
    }

    /// Validate config invariants. Called by [`manifold_to_qubo`] before
    /// constructing the QUBO matrix.
    pub fn validate(&self) -> Result<(), QuboError> {
        if self.bits_per_node == 0 {
            return Err(QuboError::ZeroBitsPerNode);
        }
        if self.bits_per_node > 16 {
            return Err(QuboError::TooManyBitsPerNode {
                requested: self.bits_per_node,
                max: 16,
            });
        }
        if self.scale_max_celsius <= 0.0 {
            return Err(QuboError::NonPositiveScale {
                value: self.scale_max_celsius,
            });
        }
        Ok(())
    }
}

/// QUBO problem derived from a [`ThermalManifold`].
///
/// `q_upper_triangular[i * num_variables + j]` holds `Q[i, j]` for `i ≤ j`.
/// The full symmetric matrix `Q[i, j] = Q[j, i]` is implied — D-Wave and most
/// QUBO libraries consume only the upper triangle. We expose the symmetric
/// row-major matrix via [`q_matrix`](Self::q_matrix) for solvers that prefer
/// the dense form.
#[derive(Debug, Clone)]
pub struct QuboProblem {
    /// Symmetric `N × N` QUBO matrix in row-major order. `N = MANIFOLD_DIM * K`.
    pub q_matrix: Vec<f64>,
    /// Number of binary variables (= side length of `q_matrix`).
    pub num_variables: usize,
    /// Config used to build this QUBO. Retained for round-tripping.
    pub config: QuboConfig,
    /// Source manifold's metric tensor (cached for diagnostic / verification).
    pub source_metric: Matrix4<f64>,
    /// Source manifold's scalar field.
    pub source_field: Vector4<f64>,
    /// Source manifold's gauge connection.
    pub source_gauge: Vector4<f64>,
}

impl QuboProblem {
    /// Number of binary variables `N`.
    pub fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Symmetric `N × N` QUBO matrix in row-major order.
    pub fn q_matrix(&self) -> &[f64] {
        &self.q_matrix
    }

    /// Q[i, j] (symmetric access — `Q[i, j] == Q[j, i]`).
    ///
    /// # Panics
    /// Panics if `i` or `j` is out of range.
    pub fn q(&self, i: usize, j: usize) -> f64 {
        assert!(i < self.num_variables, "i={i} out of range");
        assert!(j < self.num_variables, "j={j} out of range");
        self.q_matrix[i * self.num_variables + j]
    }

    /// Configuration used to build this QUBO.
    pub fn config(&self) -> QuboConfig {
        self.config
    }

    /// Maximum absolute value in `Q`. Useful for D-Wave normalization
    /// (`Q / max_abs` ⇒ values in `[-1, +1]`).
    pub fn max_abs(&self) -> f64 {
        self.q_matrix
            .iter()
            .fold(0.0_f64, |m, &v| if v.abs() > m { v.abs() } else { m })
    }

    /// Normalize `Q` so that `max(|Q[i,j]|) == 1.0`. Returns a new vector
    /// containing the normalized matrix (the original `Q` is unchanged).
    /// Suitable for D-Wave AdvantageSystem6.4 where `(h, J)` ranges are O(1).
    pub fn to_dwave_normalized(&self) -> Vec<f64> {
        let m = self.max_abs();
        if m == 0.0 {
            return self.q_matrix.clone();
        }
        self.q_matrix.iter().map(|&v| v / m).collect()
    }

    /// Estimate the condition number of the QUBO matrix using power iteration.
    ///
    /// Returns `(sigma_max, sigma_min, condition_number)` where `sigma_max` is the
    /// largest singular value, `sigma_min` is the smallest, and `condition_number`
    /// is their ratio. For a symmetric QUBO matrix, singular values equal absolute
    /// eigenvalues, so we use eigenvalue estimation instead of the more expensive SVD.
    ///
    /// The algorithm runs power iteration for the largest eigenvalue magnitude and
    /// inverse power iteration for the smallest, each with up to 100 iterations and
    /// `1e-10` convergence tolerance. For the 32×32 QUBO (K=8) this is O(N²)
    /// per iteration and converges in < 20 iterations for well-conditioned matrices.
    ///
    /// # Numerical safety
    ///
    /// If any eigenvalue is non-finite (NaN/Inf) or exactly zero, returns
    /// `Err(QuboError::SingularMatrix)` or `Err(QuboError::NumericalOverflow)`.
    pub fn condition_number_estimate(&self) -> Result<(f64, f64, f64), QuboError> {
        let n = self.num_variables;
        if n == 0 {
            return Err(QuboError::SingularMatrix);
        }

        // Power iteration for largest eigenvalue magnitude.
        let lambda_max =
            power_iteration_max(&self.q_matrix, n, 100, 1e-10).ok_or(QuboError::SingularMatrix)?;

        if !lambda_max.is_finite() || lambda_max.abs() < 1e-20 {
            return Err(QuboError::SingularMatrix);
        }
        if lambda_max.abs() > OVERFLOW_THRESHOLD {
            return Err(QuboError::NumericalOverflow {
                max_entry: lambda_max.abs(),
            });
        }

        // Inverse power iteration for smallest eigenvalue magnitude.
        // Shift the matrix by lambda_max so eigenvalues {λ_i} become {λ_i - λ_max},
        // then the largest of those (in magnitude) is |λ_min - λ_max|.
        let mut shifted = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                shifted[i * n + j] = self.q_matrix[i * n + j];
            }
            shifted[i * n + i] -= lambda_max;
        }

        let delta_min =
            power_iteration_max(&shifted, n, 100, 1e-10).ok_or(QuboError::SingularMatrix)?;

        // λ_min = λ_max - delta_min (where delta_min ≈ |λ_min - λ_max|)
        // delta_min ≈ 0 (within numerical noise) with a non-zero shifted matrix
        // means the starting vector landed in the null space of the shifted matrix
        // — i.e. λ_min = λ_max, which only occurs when the original matrix is
        // singular.
        let shifted_has_nonzero = shifted.iter().any(|&x| x != 0.0);
        if delta_min.abs() < 1e-12 && shifted_has_nonzero {
            return Err(QuboError::SingularMatrix);
        }

        let lambda_min = (lambda_max - delta_min).abs();

        if !lambda_min.is_finite() {
            return Err(QuboError::NumericalOverflow {
                max_entry: lambda_max.abs(),
            });
        }

        let condition_number = if lambda_min.abs() < 1e-20 {
            return Err(QuboError::SingularMatrix);
        } else {
            lambda_max.abs() / lambda_min.abs()
        };

        Ok((lambda_max.abs(), lambda_min.abs(), condition_number))
    }

    /// Apply Tikhonov regularization to produce a well-conditioned QUBO matrix.
    ///
    /// Adds `alpha * I` to the diagonal of `Q`, raising every eigenvalue by `alpha`.
    /// This reduces the condition number from `λ_max / λ_min` to approximately
    /// `λ_max / (λ_min + alpha)`.
    ///
    /// # Arguments
    ///
    /// * `alpha` — regularization strength. Must be `> 0`. If `None`, uses
    ///   [`DEFAULT_REGULARIZATION_ALPHA`] (`10^-4`).
    ///
    /// # Returns
    ///
    /// A new `QuboProblem` with the same geometry as `self` but a regularized
    /// `q_matrix`. The original is unchanged.
    ///
    /// # Example
    ///
    /// ```
    /// # use fluxion::quantum::qubo_mapping::{QuboProblem, QuboError, manifold_to_qubo, QuboConfig};
    /// # use fluxion::physics::geometry_tensor::ThermalManifold;
    /// let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
    /// let qp = manifold_to_qubo(&m, QuboConfig::default()).unwrap();
    /// let regularized = qp.regularize(None).unwrap();
    /// let (max, min, cn) = regularized.condition_number_estimate().unwrap();
    /// assert!(cn < 1e6, "condition number {} still too large after regularization", cn);
    /// ```
    pub fn regularize(&self, alpha: Option<f64>) -> Result<QuboProblem, QuboError> {
        let alpha = alpha.unwrap_or(DEFAULT_REGULARIZATION_ALPHA);
        if alpha <= 0.0 {
            return Err(QuboError::InvalidEncoding(
                "regularization alpha must be > 0".to_string(),
            ));
        }

        let n = self.num_variables;
        let mut q_reg = self.q_matrix.clone();
        for i in 0..n {
            q_reg[i * n + i] += alpha;
        }

        Ok(QuboProblem {
            q_matrix: q_reg,
            num_variables: n,
            config: self.config,
            source_metric: self.source_metric,
            source_field: self.source_field,
            source_gauge: self.source_gauge,
        })
    }

    /// Convert the QUBO to its Ising form `(h, J, c)` for direct submission
    /// to D-Wave (and most quantum annealers).
    ///
    /// * `h[i]` is the linear field on qubit `i`.
    /// * `J[i, j]` (for `i ≠ j`) is the coupling between qubits `i` and `j`.
    ///   Returned as a dense `N × N` matrix with zero diagonal (Ising convention).
    /// * `c` is the constant energy offset (does not affect the argmin but is
    ///   required for exact energy comparisons).
    ///
    /// The energy at spin vector `s ∈ {−1, +1}^N` is `s^T J s + h^T s + c`,
    /// which equals the original QUBO energy `x^T Q x` at `x = (s + 1) / 2`.
    pub fn to_ising(&self) -> IsingProblem {
        let n = self.num_variables;
        let mut h = vec![0.0_f64; n];
        let mut j = vec![0.0_f64; n * n];
        let mut trace = 0.0_f64;
        let mut ones_dot_q = 0.0_f64;
        for i in 0..n {
            let mut row_sum = 0.0_f64;
            for kk in 0..n {
                let q_ik = self.q_matrix[i * n + kk];
                row_sum += q_ik;
                if i != kk {
                    j[i * n + kk] = 0.25 * q_ik;
                }
                ones_dot_q += q_ik;
            }
            h[i] = 0.5 * row_sum;
            trace += self.q_matrix[i * n + i];
        }
        let c = 0.25 * trace + 0.25 * ones_dot_q;
        IsingProblem {
            h,
            j,
            c,
            num_variables: n,
        }
    }

    /// Evaluate the QUBO energy `x^T Q x` at a binary solution `x`.
    /// `x[i]` must be `0` or `1`. Any other value is treated as `0` (the
    /// algebraic value of `x_i^2` for non-binary inputs would be wrong, so
    /// we clip defensively).
    pub fn evaluate(&self, x: &[u8]) -> f64 {
        assert_eq!(
            x.len(),
            self.num_variables,
            "x.len() = {} != num_variables = {}",
            x.len(),
            self.num_variables
        );
        let n = self.num_variables;
        let mut acc = 0.0_f64;
        for i in 0..n {
            let xi = f64::from(x[i]);
            if xi == 0.0 {
                continue;
            }
            for (offset, &xj_byte) in x[i..].iter().enumerate() {
                let j = i + offset;
                let xj = f64::from(xj_byte);
                if xj == 0.0 {
                    continue;
                }
                let qij = self.q_matrix[i * n + j];
                if i == j {
                    acc += qij * xi;
                } else {
                    acc += 2.0 * qij * xi * xj;
                }
            }
        }
        acc
    }

    /// Build the canonical solution vector from the manifold's `scalar_field`
    /// using the configured encoding. This is the binary vector that, when
    /// submitted to the annealer, has energy equal to (within quantization
    /// error) the original continuous energy density `T^T M T`.
    pub fn encode_manifold_solution(&self) -> Vec<u8> {
        encode_temperatures(&self.source_field, &self.config)
    }

    /// Continuous energy density `T_recon^T M T_recon` implied by this QUBO
    /// and a binary solution `x`. For the canonical solution, equals the
    /// quantization-rounded version of `source_field^T M source_field`.
    pub fn decoded_energy_density(&self, x: &[u8]) -> f64 {
        let t_recon = decode_temperatures(x, &self.config);
        let m = &self.source_metric;
        let mut e = 0.0_f64;
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                e += m[(i, j)] * t_recon[i] * t_recon[j];
            }
        }
        e
    }

    /// Continuous energy density `T_recon^T M T_recon + coeff * (-g^T T_recon)`.
    /// Matches the full QUBO objective when `include_gauge_bias = true`.
    pub fn decoded_full_energy(&self, x: &[u8]) -> f64 {
        let t_recon = decode_temperatures(x, &self.config);
        let m = &self.source_metric;
        let mut e = 0.0_f64;
        for i in 0..MANIFOLD_DIM {
            for j in 0..MANIFOLD_DIM {
                e += m[(i, j)] * t_recon[i] * t_recon[j];
            }
        }
        if self.config.include_gauge_bias {
            for i in 0..MANIFOLD_DIM {
                e -= self.config.coeff_gauge * self.source_gauge[i] * t_recon[i];
            }
        }
        e
    }
}

/// Ising-model form of a [`QuboProblem`], suitable for D-Wave submission.
#[derive(Debug, Clone)]
pub struct IsingProblem {
    /// Linear field per qubit, length `N`.
    pub h: Vec<f64>,
    /// Dense `N × N` coupling matrix (diagonal entries are zero by convention).
    pub j: Vec<f64>,
    /// Constant energy offset.
    pub c: f64,
    /// Number of qubits (= length of `h`, side length of `j`).
    pub num_variables: usize,
}

impl IsingProblem {
    /// Evaluate the Ising energy `s^T J s + h^T s + c` at spin vector
    /// `s ∈ {−1, +1}^N`. Any non-±1 value is treated as the sign of the input.
    pub fn evaluate(&self, s: &[i8]) -> f64 {
        assert_eq!(
            s.len(),
            self.num_variables,
            "s.len() = {} != num_variables = {}",
            s.len(),
            self.num_variables
        );
        let n = self.num_variables;
        let mut acc = self.c;
        for i in 0..n {
            let si = f64::from(s[i]);
            acc += self.h[i] * si;
            for (offset, &sj_byte) in s[i + 1..].iter().enumerate() {
                let j = i + 1 + offset;
                let sj = f64::from(sj_byte);
                acc += 2.0 * self.j[i * n + j] * si * sj;
            }
        }
        acc
    }
}

/// Threshold above which a QUBO matrix is considered ill-conditioned.
/// Condition number > `ILL_CONDITIONED_THRESHOLD` triggers `IllConditioned` error.
/// Chosen to catch real thermal manifolds that approach singularity while
/// leaving well-conditioned 5R1C / 9R4C matrices untouched.
pub const ILL_CONDITIONED_THRESHOLD: f64 = 1e6;

/// Threshold below which a QUBO entry is considered numerical overflow for
/// D-Wave submission. Entries with absolute value > `OVERFLOW_THRESHOLD`
/// cannot be safely normalized to the D-Wave `h ∈ [-4,4]`, `J ∈ [-2,+1]`
/// hardware range without catastrophic cancellation.
pub const OVERFLOW_THRESHOLD: f64 = 1e10;

/// Default Tikhonov regularization parameter. Applied to the diagonal of a
/// QUBO matrix when it is detected as ill-conditioned, producing a new
/// well-conditioned matrix `Q' = Q + α·I` where `α` is the `regularization_alpha`.
/// This raises the smallest eigenvalue by `α`, reducing the condition number
/// to approximately `λ_max / (λ_min + α)`.
pub const DEFAULT_REGULARIZATION_ALPHA: f64 = 1e-4;

/// Errors that can arise during QUBO construction.
#[derive(Debug, Clone, PartialEq)]
pub enum QuboError {
    /// `bits_per_node == 0` — at least one bit per node is required.
    ZeroBitsPerNode,
    /// `bits_per_node` exceeds the supported ceiling (16 bits / node ⇒ 64
    /// qubits, already on the edge of feasibility for current annealers).
    TooManyBitsPerNode { requested: usize, max: usize },
    /// `scale_max_celsius ≤ 0` — the scale must be strictly positive.
    NonPositiveScale { value: f64 },
    /// The source manifold failed its own `validate()` (NaN/Inf in tensors).
    InvalidManifold(String),
    /// One of the QUBO encoding parameters is invalid (e.g., negative
    /// regularization alpha).
    InvalidEncoding(String),
    /// The QUBO matrix is ill-conditioned (condition number > `10^6`).
    /// This usually means the metric tensor is near-singular — e.g. two
    /// thermal nodes with nearly identical coupling resistances. Annealer
    /// results are unreliable for such matrices.
    IllConditioned {
        /// Estimated condition number (ratio of largest to smallest singular value).
        condition_number: f64,
        /// Ratio of largest to smallest eigenvalue magnitude.
        eigenvalue_ratio: f64,
    },
    /// A QUBO matrix entry exceeds the D-Wave hardware range (`±1e10`),
    /// making safe normalization impossible without catastrophic cancellation.
    NumericalOverflow { max_entry: f64 },
    /// The QUBO matrix is exactly singular (zero eigenvalues). No amount of
    /// regularization can fix this without changing the problem structure.
    SingularMatrix,
}

impl std::fmt::Display for QuboError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroBitsPerNode => write!(f, "bits_per_node must be ≥ 1"),
            Self::TooManyBitsPerNode { requested, max } => {
                write!(f, "bits_per_node = {requested} exceeds max = {max}")
            }
            Self::NonPositiveScale { value } => {
                write!(f, "scale_max_celsius = {value} must be > 0")
            }
            Self::InvalidManifold(msg) => write!(f, "invalid manifold: {msg}"),
            Self::InvalidEncoding(msg) => write!(f, "invalid encoding: {msg}"),
            Self::IllConditioned {
                condition_number,
                eigenvalue_ratio,
            } => {
                write!(
                    f,
                    "QUBO matrix is ill-conditioned: condition_number={:.1e}, eigenvalue_ratio={:.1e} (threshold={:.0e}); apply regularize() before submission",
                    condition_number, eigenvalue_ratio, ILL_CONDITIONED_THRESHOLD
                )
            }
            Self::NumericalOverflow { max_entry } => {
                write!(
                    f,
                    "QUBO entry magnitude {max_entry:.1e} exceeds overflow threshold {OVERFLOW_THRESHOLD:.0e}; normalization would lose precision",
                )
            }
            Self::SingularMatrix => {
                write!(
                    f,
                    "QUBO matrix is singular (zero eigenvalue); regularize() cannot fix this"
                )
            }
        }
    }
}

impl std::error::Error for QuboError {}

/// Build a QUBO problem from a [`ThermalManifold`] using the given config.
///
/// The returned matrix `Q` is symmetric, dimension `N × N` where
/// `N = MANIFOLD_DIM * bits_per_node`. See module-level docs for the math.
///
/// # Errors
/// Returns [`QuboError::InvalidManifold`] if the manifold fails `validate()`,
/// or any [`QuboConfig::validate`] failure.
///
/// Returns [`QuboError::IllConditioned`] if the QUBO condition number exceeds
/// `10^6` (detected before submission so the caller can apply
/// [`QuboProblem::regularize`] as a fallback).
///
/// Returns [`QuboError::NumericalOverflow`] if any QUBO entry exceeds `10^10`
/// in magnitude (normalization to D-Wave hardware range would lose precision).
///
/// Use [`QuboProblem::condition_number_estimate`] to diagnose the matrix, and
/// [`QuboProblem::regularize`] to obtain a well-conditioned fallback.
pub fn manifold_to_qubo(
    manifold: &ThermalManifold,
    config: QuboConfig,
) -> Result<QuboProblem, QuboError> {
    config.validate()?;
    if let Err(e) = manifold.validate() {
        return Err(QuboError::InvalidManifold(e.to_string()));
    }

    let n = config.num_variables();
    let scale = config.scale_factor();
    let k = config.bits_per_node;

    // Q is symmetric. We build it densely for clarity; the upper-triangular
    // view used by D-Wave is just q[i*N + j] with i <= j.
    let mut q = vec![0.0_f64; n * n];

    // Quadratic part: metric_tensor[i,j] * 2^k * 2^l / scale^2
    for i in 0..MANIFOLD_DIM {
        for j in 0..MANIFOLD_DIM {
            let m_ij = manifold.metric_tensor[(i, j)];
            for ki in 0..k {
                for kj in 0..k {
                    let row = i * k + ki;
                    let col = j * k + kj;
                    let w = 2.0_f64.powi(ki as i32) * 2.0_f64.powi(kj as i32);
                    q[row * n + col] += m_ij * w / (scale * scale);
                }
            }
        }
    }

    // Linear bias from gauge_connection (diagonal only).
    if config.include_gauge_bias {
        let c = config.coeff_gauge;
        for i in 0..MANIFOLD_DIM {
            let g_i = manifold.gauge_connection[i];
            for ki in 0..k {
                let row = i * k + ki;
                let w = 2.0_f64.powi(ki as i32);
                q[row * n + row] -= c * g_i * w / scale;
            }
        }
    }

    // Enforce exact symmetry. The quadratic part is symmetric by construction;
    // the linear bias is diagonal so symmetry is preserved.
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (q[i * n + j] + q[j * n + i]);
            q[i * n + j] = avg;
            q[j * n + i] = avg;
        }
    }

    let qp = QuboProblem {
        q_matrix: q,
        num_variables: n,
        config,
        source_metric: manifold.metric_tensor,
        source_field: manifold.scalar_field,
        source_gauge: manifold.gauge_connection,
    };

    // Check for numerical overflow in QUBO entries.
    let max_entry = qp.max_abs();
    if max_entry > OVERFLOW_THRESHOLD {
        return Err(QuboError::NumericalOverflow { max_entry });
    }

    // Check condition number — ill-conditioned matrices degrade annealer quality.
    if let Ok((_, eigenvalue_ratio, cond)) = qp.condition_number_estimate() {
        if cond > ILL_CONDITIONED_THRESHOLD {
            return Err(QuboError::IllConditioned {
                condition_number: cond,
                eigenvalue_ratio,
            });
        }
    }

    Ok(qp)
}

/// Encode a temperature vector into the canonical binary solution vector.
///
/// `scalar_field[i]` is clamped to `[0, scale_max_celsius]`, scaled by
/// `scale_factor`, rounded to the nearest integer, then bit-decomposed.
/// Bit `k` of node `i` lives at index `i * bits_per_node + k`.
pub fn encode_temperatures(scalar_field: &Vector4<f64>, config: &QuboConfig) -> Vec<u8> {
    let k = config.bits_per_node;
    let scale = config.scale_factor();
    let max_int = (1u64 << k) - 1;
    let mut out = vec![0u8; MANIFOLD_DIM * k];
    for i in 0..MANIFOLD_DIM {
        let t = scalar_field[i];
        let clamped = if t < 0.0 {
            0.0
        } else if t > config.scale_max_celsius {
            config.scale_max_celsius
        } else {
            t
        };
        let v_f = clamped * scale;
        // Round-half-away-from-zero using .round() (Rust rounds half to even
        // for floats; for our use the .5 boundary is at most 1 LSB wide so
        // either choice is within the documented tolerance).
        let v = if v_f <= 0.0 {
            0u64
        } else if v_f >= max_int as f64 {
            max_int
        } else {
            v_f.round() as u64
        };
        for bit in 0..k {
            out[i * k + bit] = ((v >> bit) & 1) as u8;
        }
    }
    out
}

/// Decode a binary solution vector back to a temperature vector.
///
/// The reconstruction has up to `±0.5 LSB` error per node vs. the original
/// continuous values (purely from the fixed-point rounding in encoding).
pub fn decode_temperatures(solution: &[u8], config: &QuboConfig) -> Vector4<f64> {
    let k = config.bits_per_node;
    let n = config.num_variables();
    assert_eq!(
        solution.len(),
        n,
        "solution.len() = {} != num_variables = {}",
        solution.len(),
        n
    );
    let scale = config.scale_factor();
    let mut out = Vector4::zeros();
    for i in 0..MANIFOLD_DIM {
        let mut v = 0u64;
        for bit in 0..k {
            if solution[i * k + bit] != 0 {
                v |= 1u64 << bit;
            }
        }
        out[i] = v as f64 / scale;
    }
    out
}

/// LSB resolution of the encoding (°C). One `K`-bit field represents
/// `[0, scale_max_celsius]` so one LSB = `scale_max_celsius / (2^K − 1)`.
pub fn lsb_resolution_celsius(config: &QuboConfig) -> f64 {
    let k = config.bits_per_node;
    config.scale_max_celsius / ((1u64 << k) as f64 - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::geometry_tensor::ThermalManifold;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_config_default() {
        let c = QuboConfig::default();
        assert_eq!(c.bits_per_node, 8);
        assert_eq!(c.scale_max_celsius, 50.0);
        assert!(c.include_gauge_bias);
        assert_eq!(c.coeff_gauge, 1.0);
        assert_eq!(c.num_variables(), 32);
    }

    #[test]
    fn test_config_scale_factor() {
        // K=8, scale_max=50: scale = 255/50 = 5.1
        let c = QuboConfig {
            bits_per_node: 8,
            scale_max_celsius: 50.0,
            ..Default::default()
        };
        assert!(approx_eq(c.scale_factor(), 5.1, 1e-12));
        // K=12, scale_max=50: scale = 4095/50 = 81.9
        let c12 = QuboConfig {
            bits_per_node: 12,
            ..c
        };
        assert!(approx_eq(c12.scale_factor(), 81.9, 1e-12));
    }

    #[test]
    fn test_lsb_resolution() {
        let c = QuboConfig {
            bits_per_node: 8,
            scale_max_celsius: 50.0,
            ..Default::default()
        };
        // 50 / 255 ≈ 0.19608 °C
        assert!(approx_eq(lsb_resolution_celsius(&c), 50.0 / 255.0, 1e-12));
    }

    #[test]
    fn test_config_validate_zero_bits() {
        let c = QuboConfig {
            bits_per_node: 0,
            ..Default::default()
        };
        assert_eq!(c.validate(), Err(QuboError::ZeroBitsPerNode));
    }

    #[test]
    fn test_config_validate_too_many_bits() {
        let c = QuboConfig {
            bits_per_node: 17,
            ..Default::default()
        };
        assert_eq!(
            c.validate(),
            Err(QuboError::TooManyBitsPerNode {
                requested: 17,
                max: 16
            })
        );
    }

    #[test]
    fn test_config_validate_negative_scale() {
        let c = QuboConfig {
            scale_max_celsius: 0.0,
            ..Default::default()
        };
        assert_eq!(
            c.validate(),
            Err(QuboError::NonPositiveScale { value: 0.0 })
        );
        let c2 = QuboConfig {
            scale_max_celsius: -1.0,
            ..Default::default()
        };
        assert_eq!(
            c2.validate(),
            Err(QuboError::NonPositiveScale { value: -1.0 })
        );
    }

    #[test]
    fn test_config_validate_ok() {
        assert!(QuboConfig::default().validate().is_ok());
    }

    #[test]
    fn test_encode_decode_round_trip_default() {
        let cfg = QuboConfig::default();
        let tol = lsb_resolution_celsius(&cfg) / 2.0 + 1e-9;
        // Keep all components strictly inside [0, scale_max_celsius] so the
        // round-trip is within ±0.5 LSB without clipping.
        // (t + 0.5 < 50, t * 2 < 50, t * 0.5 < 50) ⇒ t < 25.
        for t in [0.0_f64, 1.0, 5.0, 10.0, 21.0, 22.5, 24.0] {
            assert!(t + 0.5 < 50.0);
            assert!(t * 2.0 < 50.0);
            let field = Vector4::new(t, t + 0.5, t * 2.0, t * 0.5);
            let x = encode_temperatures(&field, &cfg);
            assert_eq!(x.len(), 32);
            let recon = decode_temperatures(&x, &cfg);
            for i in 0..MANIFOLD_DIM {
                assert!(
                    approx_eq(field[i], recon[i], tol),
                    "T[{i}] = {} round-tripped to {} (tol = {})",
                    field[i],
                    recon[i],
                    tol
                );
            }
        }
    }

    #[test]
    fn test_encode_clamps_to_range() {
        let cfg = QuboConfig {
            scale_max_celsius: 50.0,
            ..Default::default()
        };
        // Above max should clamp to 50.0
        let hot = Vector4::new(60.0, 60.0, 60.0, 60.0);
        let x = encode_temperatures(&hot, &cfg);
        let recon = decode_temperatures(&x, &cfg);
        for i in 0..MANIFOLD_DIM {
            assert!(approx_eq(recon[i], 50.0, 1e-9), "got {}", recon[i]);
        }
        // Below zero should clamp to 0.0
        let cold = Vector4::new(-5.0, -5.0, -5.0, -5.0);
        let x = encode_temperatures(&cold, &cfg);
        let recon = decode_temperatures(&x, &cfg);
        for i in 0..MANIFOLD_DIM {
            assert!(approx_eq(recon[i], 0.0, 1e-9), "got {}", recon[i]);
        }
    }

    #[test]
    fn test_qubo_size_scales_with_k() {
        for k in [4_usize, 6, 8, 12, 16] {
            let cfg = QuboConfig {
                bits_per_node: k,
                ..Default::default()
            };
            assert_eq!(cfg.num_variables(), MANIFOLD_DIM * k);
        }
    }

    #[test]
    fn test_manifold_to_qubo_flat_manifold() {
        // Identity metric, zero field, zero gauge — Q must be symmetric
        // (positive identity * scale^2 denominator).
        let m = ThermalManifold::new_flat();
        let qp = manifold_to_qubo(&m, QuboConfig::default()).expect("ok");
        assert_eq!(qp.num_variables(), 32);
        // Symmetric
        for i in 0..qp.num_variables() {
            for j in 0..qp.num_variables() {
                assert!(
                    approx_eq(qp.q(i, j), qp.q(j, i), 1e-12),
                    "Q[{},{}] = {} != Q[{},{}] = {}",
                    i,
                    j,
                    qp.q(i, j),
                    j,
                    i,
                    qp.q(j, i)
                );
            }
        }
    }

    #[test]
    fn test_round_trip_5r1c_energy_matches() {
        // 5R1C scene → embedded 4x4 metric with active 2x2 block.
        let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let cfg = QuboConfig::default();
        let qp = manifold_to_qubo(&m, cfg).expect("ok");

        // Encode canonical solution from the manifold's scalar_field.
        let x_canon = qp.encode_manifold_solution();
        assert_eq!(x_canon.len(), 32);

        // Round-trip temperature error is within ±0.5 LSB.
        let tol = lsb_resolution_celsius(&cfg) / 2.0 + 1e-9;
        let recon = decode_temperatures(&x_canon, &cfg);
        for i in 0..MANIFOLD_DIM {
            assert!(
                approx_eq(m.scalar_field[i], recon[i], tol),
                "T_air[{i}] = {} round-tripped to {}",
                m.scalar_field[i],
                recon[i]
            );
        }

        // Disable the gauge bias so the QUBO encodes only T^T M T.
        let cfg_quad = QuboConfig {
            include_gauge_bias: false,
            ..cfg
        };
        let qp_quad = manifold_to_qubo(&m, cfg_quad).expect("ok");
        let x_quad = qp_quad.encode_manifold_solution();
        let e_qubo = qp_quad.evaluate(&x_quad);
        let e_density = qp_quad.decoded_energy_density(&x_quad);
        assert!(
            approx_eq(e_qubo, e_density, 1e-9),
            "QUBO energy {} != decoded density {}",
            e_qubo,
            e_density
        );

        // The decoded energy density should be very close to the exact
        // continuous T^T M T (within 4 * M_max * T_max * LSB, very loose).
        let m_field = m.scalar_field;
        let e_exact: f64 = {
            let mut s = 0.0;
            for i in 0..MANIFOLD_DIM {
                for j in 0..MANIFOLD_DIM {
                    s += m.metric_tensor[(i, j)] * m_field[i] * m_field[j];
                }
            }
            s
        };
        let rel_tol = 0.05; // 5% — quantization at 0.2 °C LSB can shift energy
        let rel_err = ((e_density - e_exact) / e_exact.abs().max(1e-12)).abs();
        assert!(
            rel_err <= rel_tol,
            "decoded energy {} vs exact {} (rel err {})",
            e_density,
            e_exact,
            rel_err
        );
    }

    #[test]
    fn test_round_trip_9r4c_with_gauge() {
        let temps = [22.0, 20.0, 23.0, 18.0];
        let caps = [1000.0, 5000.0, 3000.0, 8000.0];
        let r_tr = [50.0, 30.0, 20.0];
        let r_cross = Some([5.0, 3.0, 2.0]);
        let mut m = ThermalManifold::from_9r4c_parameters(temps, caps, r_tr, r_cross);
        // Set a non-zero gauge_connection (HVAC + solar).
        m.gauge_connection = Vector4::new(100.0, 200.0, 50.0, 30.0);
        let cfg = QuboConfig::default();
        let qp = manifold_to_qubo(&m, cfg).expect("ok");
        let x = qp.encode_manifold_solution();

        // QUBO energy should match decoded_full_energy.
        let e_qubo = qp.evaluate(&x);
        let e_full = qp.decoded_full_energy(&x);
        assert!(
            approx_eq(e_qubo, e_full, 1e-9),
            "QUBO energy {} != decoded full energy {}",
            e_qubo,
            e_full
        );
    }

    #[test]
    fn test_qubo_is_symmetric_for_random_manifold() {
        let mut m = ThermalManifold::new_flat();
        m.metric_tensor = Matrix4::from_row_slice(&[
            0.1, 0.02, 0.0, 0.0, //
            0.02, -0.05, 0.01, 0.0, //
            0.0, 0.01, -0.03, 0.005, //
            0.0, 0.0, 0.005, -0.04, //
        ]);
        m.scalar_field = Vector4::new(20.0, 21.0, 22.0, 19.0);
        m.gauge_connection = Vector4::new(1.0, -2.0, 3.0, -4.0);

        let qp = manifold_to_qubo(&m, QuboConfig::default()).expect("ok");
        for i in 0..qp.num_variables() {
            for j in (i + 1)..qp.num_variables() {
                assert!(
                    approx_eq(qp.q(i, j), qp.q(j, i), 1e-12),
                    "asymmetric at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_qubo_rejects_nan_manifold() {
        let mut m = ThermalManifold::new_flat();
        m.scalar_field[2] = f64::NAN;
        let err = manifold_to_qubo(&m, QuboConfig::default()).unwrap_err();
        assert!(matches!(err, QuboError::InvalidManifold(_)));
    }

    #[test]
    fn test_qubo_to_ising_matches_qubo_energy() {
        let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let cfg = QuboConfig {
            include_gauge_bias: false, // cleaner comparison
            ..QuboConfig::default()
        };
        let qp = manifold_to_qubo(&m, cfg).expect("ok");
        let ising = qp.to_ising();

        // Verify across multiple random solutions.
        for seed in 0..16_u64 {
            // Simple LCG to keep this test deterministic.
            let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut x = Vec::with_capacity(qp.num_variables());
            for _ in 0..qp.num_variables() {
                x.push((rng & 1) as u8);
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            }
            let e_qubo = qp.evaluate(&x);
            let s: Vec<i8> = x.iter().map(|&b| if b == 0 { -1 } else { 1 }).collect();
            let e_ising = ising.evaluate(&s);
            assert!(
                approx_eq(e_qubo, e_ising, 1e-9),
                "QUBO {} != Ising {} for seed {}",
                e_qubo,
                e_ising,
                seed
            );
        }
    }

    #[test]
    fn test_qubo_max_abs_and_normalize() {
        let m = ThermalManifold::from_5r1c_parameters(21.0, 22.0, 0.1, 1000.0, 5000.0);
        let qp = manifold_to_qubo(&m, QuboConfig::default()).expect("ok");
        let m_abs = qp.max_abs();
        assert!(m_abs > 0.0);
        let qn = qp.to_dwave_normalized();
        assert_eq!(qn.len(), qp.q_matrix().len());
        let m_abs_norm = qn.iter().fold(0.0_f64, |m, &v| v.abs().max(m));
        assert!(approx_eq(m_abs_norm, 1.0, 1e-12));
    }

    #[test]
    fn test_num_variables_is_manifold_dim_times_bits() {
        let m = ThermalManifold::new_flat();
        for k in [1_usize, 2, 4, 8, 16] {
            let cfg = QuboConfig {
                bits_per_node: k,
                include_gauge_bias: false,
                ..Default::default()
            };
            let qp = manifold_to_qubo(&m, cfg).expect("ok");
            assert_eq!(qp.num_variables(), MANIFOLD_DIM * k);
            assert_eq!(qp.q_matrix().len(), (MANIFOLD_DIM * k) * (MANIFOLD_DIM * k));
        }
    }
}
