//! ASHRAE 140 thermal constants.
//!
//! This module provides ASHRAE Standard 140 constants for surface heat transfer
//! coefficients and building surface properties. Constants are versioned to
//! support different ASHRAE 140 editions (2021, 2023, etc.).

pub mod v2021;
pub mod v2023;

// Select version via feature flag (if needed in future)
// Default to latest (v2023)

#[cfg(feature = "ashrae_140_v2021")]
pub use v2021::*;

#[cfg(not(feature = "ashrae_140_v2021"))]
pub use v2023::*;
