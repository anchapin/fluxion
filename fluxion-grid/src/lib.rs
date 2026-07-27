//! # fluxion-grid
//!
//! Grid-edge electrical network components for Fluxion building energy modeling.
//!
//! This crate provides battery storage node models with state-of-charge (SoC) tracking
//! and electrical characteristics for integration with building energy simulations.
//!
//! ## Contents
//!
//! | Module | Description |
//! |--------|-------------|
//! | `battery_storage_node` | `BatteryStorageNode` — single-cell battery model with SoC, terminal voltage, and C-rate dynamics |

#![allow(nonstandard_style)]
#![allow(clippy::all)]

pub mod battery_storage_node;

pub use battery_storage_node::BatteryStorageNode;
