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
//! PV panel/inverter types (`PvPanel`, `PvSystem`, `SimpleInverter`) live in the
//! `fluxion-grid` crate and are re-exported here when the `grid` feature is enabled
//! (Issue #4005) — the former local `pv.rs` was a near-verbatim duplicate and was
//! deleted. Solar *thermal* collectors are intentionally not in this module; they
//! exchange heat with the thermal model directly and belong here as future
//! thermal-side work, not in the electrical grid integration.
//!
//! # Validation
//! Tests compare against EnergyPlus 25.2 reference output for Denver TMY3:
//! - Solar position: 0.5° tolerance (altitude and azimuth)
//! - Surface irradiance: 1% tolerance (beam and diffuse)

pub mod solar_position;
pub mod surface_irradiance;

// Re-export primary types and functions for convenient access.
//
// Issue #4005: PV types come from `fluxion-grid` (feature-gated) — the crate is an
// optional dependency, so the re-export only exists with `--features grid`.
#[cfg(feature = "grid")]
pub use fluxion_grid::{PvPanel, PvSystem, SimpleInverter};
pub use solar_position::{calculate_day_of_year, calculate_solar_position, SolarPosition};
pub use surface_irradiance::{
    calculate_surface_irradiance, extraterrestrial_irradiance, orientation_to_angles,
    relative_airmass, PerezSkyModel, SurfaceIrradiance,
};
