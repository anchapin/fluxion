//! # fluxion-behavior
//!
//! Occupant behavior modeling, comfort triggers, and adaptive actions for Fluxion.
//!
//! ## Modules
//!
//! - [`occupancy`] — Occupant presence schedules and density models
//! - [`internal_gains`] — Metabolic gains, equipment, and lighting heat generation
//! - [`comfort`] — Thermal comfort models (PMV/PPC, Adaptive ASHRAE 55)
//! - [`triggers`] — Occupant comfort trigger events and handlers
//! - [`markov_occupancy`] — Markov chain-based occupancy state generation
//! - [`lighting`] — Lighting power and daylighting response
//! - [`plug_loads`] — Plug and process load models
//! - [`moisture`] — Moisture generation from occupant respiration
//! - [`dynamic_adapter`] — Dynamic internal gain adapter combining all sources
//!
//! ## References
//!
//! - ASHRAE 55 (2020): Thermal Environmental Conditions for Human Occupancy
//! - ISO 7730:2005 — Ergonomics of the thermal environment
//! - ASHRAE 90.1 (2019): Energy Standard for Buildings

pub mod comfort;
pub mod dynamic_adapter;
pub mod internal_gains;
pub mod lighting;
pub mod markov_occupancy;
pub mod moisture;
pub mod occupancy;
pub mod plug_loads;
pub mod triggers;

pub use comfort::{AdaptiveComfort, AdaptiveComfortStatus, PmvComfort, PmvComfortStatus};
pub use dynamic_adapter::{DynamicInternalGainAdapter, InternalGains};
pub use internal_gains::{Co2Generation, MetabolicRate};
pub use lighting::LightingModel;
pub use markov_occupancy::{MarkovOccupancyGenerator, OccupancyState};
pub use moisture::{ActivityLevel, MoistureGeneration};
pub use occupancy::OccupancySchedule;
pub use plug_loads::MockPlugLoad;
pub use triggers::{ComfortViolation, ComfortViolationType, OccupantComfortTriggers};
