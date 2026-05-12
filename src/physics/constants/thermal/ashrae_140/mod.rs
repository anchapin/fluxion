//! ASHRAE 140 thermal constants.
//!
//! [`materials`] is the single source of truth for all material properties.
//! Import from there, not from any other location in the codebase.

pub mod materials;
pub mod v2021;
pub mod v2023;

/// Re-export material constants at module level for convenience.
pub use materials::*;

#[cfg(feature = "ashrae_140_v2021")]
pub use v2021::*;

#[cfg(not(feature = "ashrae_140_v2021"))]
pub use v2023::*;
