// =============================================================================
// QUBO translation utility (issue #1772)
// =============================================================================
//
// Standardized, stable translation between the continuous Riemannian metric
// tensor of [`ThermalManifold`] and a Quadratic Unconstrained Binary
// Optimization (QUBO) matrix. This replaces the fragile, research-grade
// intermediate structs (the `QuboProblem` in `crate::quantum::qubo_mapping`,
// which caches source tensors and mixes the matrix with its inputs) with a
// self-contained [`QuboMatrix`] that carries only the QUBO data and its
// encoding.
//
// The translation is intentionally narrow and lives next to the
// [`ThermalManifold`] it operates on, so the metric-tensor → QUBO surface is
// discoverable from the type that owns the tensor. The research bridge in
// `crate::quantum` may delegate to this utility; it is not altered here.
//
// [`QuboMatrix`]: super::manifold::QuboMatrix

/// Fixed-point encoding parameters for the metric-tensor → QUBO translation.
///
/// Each manifold node `i` is represented by `bits_per_node` unsigned bits so
/// that `T[i] ≈ (Σ_k 2^k · x[(i,k)]) / scale_factor`, where
/// `scale_factor = (2^K − 1) / scale_max_celsius`. With the default
/// `K = 8`, `scale_max_celsius = 50.0`, the resolution is `50 / 255 ≈ 0.196 °C`
/// per LSB — well below typical ASHRAE 140 reference precision.
///
/// This is the *stable* encoding dial: it deliberately carries only the
/// precision/scale knobs, not the gauge-bias coefficient, so the
/// [`QuboMatrix`] it produces is a pure function of the metric tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuboEncoding {
    /// Bits per manifold node. `K=8 ⇒ 32` binary variables total; must be in
    /// `1..=16` (larger values give diminishing returns and exceed current
    /// annealer qubit budgets).
    pub bits_per_node: usize,
    /// Maximum representable temperature in °C. Must be `> 0`.
    pub scale_max_celsius: f64,
}

impl Default for QuboEncoding {
    /// `K = 8` bits/node, `scale_max = 50 °C` — matches typical ASHRAE 140
    /// zone temperatures at ~0.2 °C LSB resolution.
    fn default() -> Self {
        Self {
            bits_per_node: 8,
            scale_max_celsius: 50.0,
        }
    }
}

impl QuboEncoding {
    /// Total number of binary variables: `MANIFOLD_DIM * bits_per_node`.
    ///
    /// [`MANIFOLD_DIM`]: super::manifold::MANIFOLD_DIM
    pub fn num_variables(&self) -> usize {
        super::manifold::MANIFOLD_DIM * self.bits_per_node
    }

    /// Scale factor: `T[i] * scale_factor ≈ Σ_k 2^k x[(i,k)]`.
    ///
    /// # Panics
    /// Panics if `scale_max_celsius <= 0` (call [`validate`](Self::validate)
    /// first in fallible contexts).
    pub fn scale_factor(&self) -> f64 {
        assert!(
            self.scale_max_celsius > 0.0,
            "scale_max_celsius must be > 0"
        );
        let k = self.bits_per_node;
        ((1u64 << k) as f64 - 1.0) / self.scale_max_celsius
    }

    /// LSB resolution of the encoding in °C:
    /// `scale_max_celsius / (2^K − 1)`.
    pub fn lsb_resolution_celsius(&self) -> f64 {
        let k = self.bits_per_node;
        self.scale_max_celsius / ((1u64 << k) as f64 - 1.0)
    }

    /// Validate the encoding invariants. Called by
    /// [`ThermalManifold::to_qubo_matrix`] before constructing the matrix.
    pub fn validate(&self) -> Result<(), QuboTranslateError> {
        if self.bits_per_node == 0 {
            return Err(QuboTranslateError::InvalidEncoding(
                "bits_per_node must be ≥ 1".to_string(),
            ));
        }
        if self.bits_per_node > 16 {
            return Err(QuboTranslateError::InvalidEncoding(format!(
                "bits_per_node = {} exceeds the supported maximum of 16",
                self.bits_per_node
            )));
        }
        if self.scale_max_celsius <= 0.0 {
            return Err(QuboTranslateError::InvalidEncoding(format!(
                "scale_max_celsius = {} must be > 0",
                self.scale_max_celsius
            )));
        }
        Ok(())
    }
}

/// Errors that can arise during the standardized QUBO translation.
#[derive(Debug, Clone, PartialEq)]
pub enum QuboTranslateError {
    /// The [`QuboEncoding`] failed validation (zero/too-many bits, or a
    /// non-positive scale).
    InvalidEncoding(String),
    /// The source [`ThermalManifold`] failed its own `validate()` (NaN/Inf in
    /// the metric, field, or connection).
    InvalidManifold(String),
}

impl std::fmt::Display for QuboTranslateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidEncoding(msg) => write!(f, "invalid QUBO encoding: {msg}"),
            Self::InvalidManifold(msg) => write!(f, "invalid manifold: {msg}"),
        }
    }
}

impl std::error::Error for QuboTranslateError {}
