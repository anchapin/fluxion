//! Monte Carlo parameter sweeps for Phase 2 ML data generation (Issue #1776, Task T5.1).
//!
//! This module provides the infrastructure for generating diverse, reproducible
//! parameter combinations that can be fed to the physics solver to produce
//! training data for the ML surrogate models.
//!
//! # Architecture
//!
//! ```text
//!  SurrogateDomain ──► SweepConfig ──► SamplingStrategy ──► SweepSample ──► Physics Solver
//!       │                  │                  │                   │
//!       │                  ▼                  ▼                   ▼
//!       │           Distributions       Unit-cube draws      ParameterManifest
//!       │           (Uniform, Normal,   (Random, LHS,         (seeded, reproducible)
//!       │            LogNormal, Choice)  Sobol)
//!       │
//!       └─── bounds for weather, occupancy, zone temp
//!            + climate zones + building types
//! ```
//!
//! # Quick start
//!
//! ```no_run
//! use fluxion::ai::sweeps::config::SweepConfig;
//! use fluxion::ai::surrogate::SurrogateDomain;
//! use fluxion::ai::sweeps::manifest::generate_samples;
//!
//! let domain = SurrogateDomain::default_residential();
//! let config = SweepConfig::from_domain(&domain);
//! let result = generate_samples(&config);
//!
//! println!("Generated {} samples", result.len());
//! println!("Manifest: seed={}, strategy={}",
//!     result.manifest.seed, result.manifest.strategy);
//! ```

pub mod config;
pub mod distributions;
pub mod manifest;
pub mod sampling;
pub mod weather;

pub use config::{
    BuildingGeometryParams, InsulationParams, SweepConfig, SweepConfigBuilder,
    WeatherSamplingParams, NUM_CONTINUOUS_DIMENSIONS,
};
pub use distributions::{Choice, ParameterDistribution};
pub use manifest::{generate_samples, ParameterManifest, SweepResult, SweepSample};
pub use sampling::{generate_unit_samples, SamplingStrategy};
pub use weather::{WeatherFileEntry, WeatherFileRegistry};
