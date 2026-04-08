//! High-mass thermal validation types.
//!
//! This module provides data structures and validation logic for ASHRAE 140-2017
//! Addendum B high-mass thermal validation.

pub mod types;
pub mod validator;

pub use types::{ConstructionType, HighMassCase, ValidationResult};
pub use validator::{calculate_cv_rmse, calculate_nmbe, ThermalMassValidator, ToleranceBands};
