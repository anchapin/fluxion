//! Cross-Validation Adapters Module
//!
//! This module provides implementations for comparing Fluxion results
//! against external building energy modeling tools.

pub mod energyplus;
pub mod trnsys;

pub use energyplus::EnergyPlusAdapter;
pub use trnsys::TRNSYSAdapter;