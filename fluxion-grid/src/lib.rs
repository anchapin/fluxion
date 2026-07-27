//! fluxion-grid: Electrical grid modeling with bus node types for power flow analysis.
//!
//! This crate provides foundational types for electrical bus modeling including:
//! - Bus node types (Slack, PV, PQ)
//! - Electrical bus structures with voltage, angle, and power attributes
//! - Battery bus with State of Charge (SoC) tracking
//! - Thermal-electrical coupler for heat pump modeling
//!
//! # Example
//!
//! ```
//! use fluxion_grid::{BusNodeType, ElectricalBus, BatteryBus, ThermalElectricalCoupler};
//!
//! // Create a PQ bus
//! let bus = ElectricalBus::new_pq(1, 0.5, 0.2);
//! assert!(matches!(bus.node_type, BusNodeType::PQ));
//!
//! // Create a thermal-electrical coupler for heat pump modeling
//! let coupler = ThermalElectricalCoupler::new(3.0, 280.15, 293.15);
//! assert!((coupler.cop() - 3.0).abs() < 0.01);
//! ```

pub mod bus;
pub mod battery;
pub mod power_flow;
pub mod thermal_electrical_coupler;

pub use bus::{BusNodeType, ElectricalBus};
pub use battery::BatteryBus;
pub use power_flow::PowerFlowState;
pub use thermal_electrical_coupler::ThermalElectricalCoupler;
