//! fluxion-grid: Electrical grid modeling with bus node types for power flow analysis.
//!
//! This crate provides foundational types for electrical bus modeling including:
//! - Bus node types (Slack, PV, PQ)
//! - Electrical bus structures with voltage, angle, and power attributes
//! - Battery bus with State of Charge (SoC) tracking
//!
//! # Example
//!
//! ```
//! use fluxion_grid::{BusNodeType, ElectricalBus, BatteryBus};
//!
//! // Create a PQ bus
//! let bus = ElectricalBus::new_pq(1, 0.5, 0.2);
//! assert!(matches!(bus.node_type, BusNodeType::PQ));
//! ```

pub mod bus;
pub mod battery;
pub mod power_flow;

pub use bus::{BusNodeType, ElectricalBus};
pub use battery::BatteryBus;
pub use power_flow::PowerFlowState;
