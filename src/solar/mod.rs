//! Pure solar position and irradiance calculations.
//!
//! This module is isolated from building-level concerns — it has ZERO imports
//! from `sim::` or `validation::` modules. It can be tested with just a CSV file
//! and a function call.
//!
//! # Module Structure
//! - `solar_position` — NOAA solar calculator for sun altitude/azimuth
//! - `surface_irradiance` — Beam, diffuse (Perez), and ground-reflected irradiance on tilted surfaces
//!
//! # Validation
//! Tests compare against EnergyPlus 25.2 reference output for Denver TMY3:
//! - Solar position: 0.5° tolerance (altitude and azimuth)
//! - Surface irradiance: 1% tolerance (beam and diffuse)

pub mod solar_position;
pub mod surface_irradiance;

// Re-export primary types and functions for convenient access
pub use solar_position::{calculate_day_of_year, calculate_solar_position, SolarPosition};
pub use surface_irradiance::{
    calculate_surface_irradiance, extraterrestrial_irradiance, orientation_to_angles,
    relative_airmass, PerezSkyModel, SurfaceIrradiance,
};
