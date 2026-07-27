//! Battery storage node model with state-of-charge and electrical dynamics.
//!
//! Models a single battery cell/storage unit with:
//! - State of charge (SoC) tracking: 0.0 (empty) to 1.0 (full)
//! - C-rate dependent discharge behavior
//! - Internal resistance for terminal voltage calculation
//!
//! ## Physics
//!
//! SoC update (conservation of charge):
//! ```text
//! dSOC/dt = -I / (capacity_ah * 3600)
//! ```
//!
//! Terminal voltage (internal resistance model):
//! ```text
//! V_terminal = V_oc - I * R_internal
//! ```
//!
//! Where V_oc (open-circuit voltage) is approximated as V_nominal for the simplified model.

use uuid::Uuid;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BatteryStorageNode {
    pub bus_id: Uuid,
    pub soc: f64,
    pub c_rate: f64,
    pub capacity_ah: f64,
    pub r_internal_ohm: f64,
    pub v_nominal: f64,
}

impl BatteryStorageNode {
    pub fn new(
        bus_id: Uuid,
        soc: f64,
        c_rate: f64,
        capacity_ah: f64,
        r_internal_ohm: f64,
        v_nominal: f64,
    ) -> Self {
        Self {
            bus_id,
            soc: soc.clamp(0.0, 1.0),
            c_rate,
            capacity_ah,
            r_internal_ohm,
            v_nominal,
        }
    }

    pub fn step(&mut self, dt: std::time::Duration, current_amps: f64) -> (f64, f64) {
        let dt_hours = dt.as_secs_f64() / 3600.0;
        let d_soc = -(current_amps * dt_hours) / self.capacity_ah;
        self.soc = (self.soc + d_soc).clamp(0.0, 1.0);
        let terminal_v = self.terminal_voltage(current_amps);
        (self.soc, terminal_v)
    }

    pub fn terminal_voltage(&self, current: f64) -> f64 {
        self.v_nominal - current * self.r_internal_ohm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_1c_discharge_1_hour() {
        let bus_id = Uuid::new_v4();
        let mut battery = BatteryStorageNode::new(bus_id, 1.0, 1.0, 100.0, 0.01, 400.0);

        let dt = Duration::from_secs(3600);
        let current = 100.0;

        let (final_soc, _terminal_v) = battery.step(dt, current);

        assert!(
            final_soc < 0.01,
            "After 1C discharge for 1 hour, SoC should be near 0, got {}",
            final_soc
        );
    }

    #[test]
    fn test_soc_bounds() {
        let bus_id = Uuid::new_v4();
        let mut battery = BatteryStorageNode::new(bus_id, 0.5, 1.0, 100.0, 0.01, 400.0);

        let dt = Duration::from_secs(3600 * 10);
        battery.step(dt, 1000.0);

        assert!(battery.soc >= 0.0, "SoC should not go below 0");
        assert!(battery.soc <= 1.0, "SoC should not exceed 1");
    }

    #[test]
    fn test_terminal_voltage_drop() {
        let bus_id = Uuid::new_v4();
        let battery = BatteryStorageNode::new(bus_id, 1.0, 1.0, 100.0, 0.01, 400.0);

        let v_no_load = battery.terminal_voltage(0.0);
        let v_with_load = battery.terminal_voltage(100.0);

        assert_eq!(v_no_load, 400.0);
        assert_eq!(v_with_load, 399.0);
    }
}
