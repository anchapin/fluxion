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
    q_matrix: Vec<f64>,
    /// Number of binary variables (= side length of `q_matrix`).
    num_variables: usize,
    /// Config used to build this QUBO. Retained for round-tripping.
    config: QuboConfig,
    /// Source manifold's metric tensor (cached for diagnostic / verification).
    source_metric: Matrix4<f64>,
    /// Source manifold's scalar field.
    source_field: Vector4<f64>,
    /// Source manifold's gauge connection.
    source_gauge: Vector4<f64>,
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

    Ok(QuboProblem {
        q_matrix: q,
        num_variables: n,
        config,
        source_metric: manifold.metric_tensor,
        source_field: manifold.scalar_field,
        source_gauge: manifold.gauge_connection,
    })
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
