//! Integration testing framework for Fluxion
//!
//! Provides reusable fixtures, wiring validation, and E2E test infrastructure.
//! This module is test-only and compiled with `#[cfg(test)]`.

#[cfg(test)]
pub mod fixtures;

#[cfg(test)]
pub mod wiring;

#[cfg(test)]
pub mod scenarios;
