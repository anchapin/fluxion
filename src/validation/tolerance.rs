// Validation tolerance module
// This module provides tolerance definitions and validation

#[derive(Default, Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ValidationTolerance {
    /// NMBE (Normalized Mean Bias Error) limit
    pub nmbe_limit: f64,
    /// MAE (Mean Absolute Error) limit
    pub mae_limit: f64,
    /// CV(RMSE) (Coefficient of Variation of RMSE) limit
    pub cv_rmse_limit: f64,
}

impl ValidationTolerance {
    pub fn new() -> Self {
        Self {
            nmbe_limit: 5.0,
            mae_limit: 0.1,
            cv_rmse_limit: 10.0,
        }
    }
}
