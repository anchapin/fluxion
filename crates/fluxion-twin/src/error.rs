use thiserror::Error;

#[derive(Debug, Error)]
pub enum KalmanError {
    #[error("Covariance matrix is not positive semi-definite: {0}")]
    NotPositiveSemiDefinite(String),

    #[error("Covariance matrix is symmetric: {0}")]
    NotSymmetric(String),

    #[error("Matrix inversion failed: {0}")]
    MatrixInversionFailed(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Singular matrix encountered during computation")]
    SingularMatrix,

    #[error("Numerical instability: values diverged to NaN or Inf")]
    NumericalInstability,

    #[error("Sigma point generation failed: {0}")]
    SigmaPointGenerationFailed(String),
}

pub type KalmanResult<T> = Result<T, KalmanError>;
