//! Electrical bus node types and structures.

use serde::{Deserialize, Serialize};

/// Bus node types in power flow analysis.
///
/// - **Slack (or Swing)**: Slack bus balances total generation and load, setting
///   the system frequency reference. Only one slack bus per network.
/// - **PV**: Generator bus with fixed active power and voltage magnitude.
///   Controls voltage magnitude at its bus.
/// - **PQ**: Load bus with fixed active and reactive power.
///   Most common bus type in power flow studies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusNodeType {
    /// Slack (Swing) bus - voltage magnitude and angle specified, P and Q computed
    Slack,
    /// PV bus - active power P and voltage magnitude V specified, Q computed
    PV,
    /// PQ bus - active power P and reactive power Q specified, V and angle computed
    PQ,
}

/// Electrical bus representation for power flow analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectricalBus {
    /// Bus identifier
    pub id: u32,
    /// Node type classification
    pub node_type: BusNodeType,
    /// Voltage magnitude in per-unit (pu)
    pub voltage_magnitude: f64,
    /// Voltage angle in radians
    pub voltage_angle: f64,
    /// Active power in per-unit (pu)
    pub active_power: f64,
    /// Reactive power in per-unit (pu)
    pub reactive_power: f64,
}

impl ElectricalBus {
    /// Create a new Slack bus with specified voltage magnitude and angle.
    pub fn new_slack(id: u32, voltage_magnitude: f64, voltage_angle: f64) -> Self {
        Self {
            id,
            node_type: BusNodeType::Slack,
            voltage_magnitude,
            voltage_angle,
            active_power: 0.0,
            reactive_power: 0.0,
        }
    }

    /// Create a new PV bus with specified active power and voltage magnitude.
    pub fn new_pv(id: u32, active_power: f64, voltage_magnitude: f64) -> Self {
        Self {
            id,
            node_type: BusNodeType::PV,
            voltage_magnitude,
            voltage_angle: 0.0,
            active_power,
            reactive_power: 0.0,
        }
    }

    /// Create a new PQ bus with specified active and reactive power.
    pub fn new_pq(id: u32, active_power: f64, reactive_power: f64) -> Self {
        Self {
            id,
            node_type: BusNodeType::PQ,
            voltage_magnitude: 1.0,
            voltage_angle: 0.0,
            active_power,
            reactive_power,
        }
    }

    /// Update the voltage magnitude and angle (for solution iteration).
    pub fn update_voltage(&mut self, magnitude: f64, angle: f64) {
        self.voltage_magnitude = magnitude;
        self.voltage_angle = angle;
    }

    /// Update the active and reactive power injections.
    pub fn update_power(&mut self, active: f64, reactive: f64) {
        self.active_power = active;
        self.reactive_power = reactive;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slack_bus() {
        let bus = ElectricalBus::new_slack(1, 1.05, 0.0);
        assert!(matches!(bus.node_type, BusNodeType::Slack));
        assert_eq!(bus.voltage_magnitude, 1.05);
    }

    #[test]
    fn test_pv_bus() {
        let bus = ElectricalBus::new_pv(2, 1.0, 1.02);
        assert!(matches!(bus.node_type, BusNodeType::PV));
        assert_eq!(bus.active_power, 1.0);
    }

    #[test]
    fn test_pq_bus() {
        let bus = ElectricalBus::new_pq(3, 0.5, 0.2);
        assert!(matches!(bus.node_type, BusNodeType::PQ));
        assert_eq!(bus.reactive_power, 0.2);
    }

    #[test]
    fn test_update_voltage() {
        let mut bus = ElectricalBus::new_pq(1, 0.0, 0.0);
        bus.update_voltage(1.02, 0.1);
        assert_eq!(bus.voltage_magnitude, 1.02);
        assert_eq!(bus.voltage_angle, 0.1);
    }
}
