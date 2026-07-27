use thiserror::Error;

#[derive(Debug, Error)]
pub enum GridModelError {
    #[error("voltage {voltage:.3} pu is outside valid range [0.5, 1.5]")]
    VoltageOutOfRange { voltage: f64 },

    #[error("coupler thermal mass is zero — cannot apply voltage adjustment")]
    ZeroThermalMass,

    #[error("negative COP adjustment factor {factor:.4} would increase COP")]
    NegativeAdjustment { factor: f64 },
}
