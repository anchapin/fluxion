//! Error types for the Unscented Kalman Filter.

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum KalmanError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("matrix is not positive semi-definite")]
    NonPositiveDefiniteMatrix,

    #[error("matrix is singular or near-singular (determinant = 0)")]
    SingularMatrix,

    #[error("Cholesky decomposition failed")]
    CholeskyFailed,

    #[error("sigma point generation failed")]
    SigmaPointGenerationFailed,

    #[error("prediction step failed")]
    PredictionFailed,

    #[error("update step failed")]
    UpdateFailed,
}
