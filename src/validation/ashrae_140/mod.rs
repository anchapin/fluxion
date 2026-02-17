//! ASHRAE Standard 140 validation test cases.
//!
//! This module contains implementations of ASHRAE Standard 140 test cases
//! for validating building energy simulation accuracy.

pub mod case_600;

pub use case_600::{Case600Model, SimulationResult};
