//! Equipment Load Modeling
//!
//! This module provides equipment load modeling with trait-based abstraction
//! for different equipment types (computers, servers, generic equipment).

use crate::sim::schedule::DailySchedule;
use serde::{Deserialize, Serialize};

/// Trait for equipment load modeling
///
/// Provides consistent API for different equipment types with
/// time-varying schedules and thermal characteristics.
pub trait Equipment {
    /// Equipment identifier
    fn id(&self) -> &str;

    /// Calculate equipment power at a specific hour of year
    fn power_at_hour(&self, hour_of_year: usize) -> f64;

    /// Calculate convective heat gains (Watts)
    /// Convective heat instantly warms zone air
    fn convective_gains(&self, hour_of_year: usize) -> f64;

    /// Calculate radiative heat gains (Watts)
    /// Radiative heat is absorbed by thermal mass
    fn radiative_gains(&self, hour_of_year: usize) -> f64;

    /// Mass coupling factor (0-1)
    /// Fraction of radiative heat absorbed by thermal mass
    /// Remaining fraction (1.0 - factor) goes to air
    fn mass_coupling_factor(&self) -> f64;

    /// Returns self as Any for downcasting (needed for cloning)
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Computer equipment (desktops, laptops, monitors)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputerEquipment {
    pub id: String,
    pub rated_power_w: f64,
    pub count: usize,
    pub schedule: DailySchedule,
    pub radiative_fraction: f64,
    pub convective_fraction: f64,
    pub mass_coupling_factor: f64,
}

impl ComputerEquipment {
    pub fn new(id: String, rated_power_w: f64, count: usize) -> Self {
        Self {
            id,
            rated_power_w,
            count,
            schedule: DailySchedule::new(), // Off by default
            radiative_fraction: 0.3,
            convective_fraction: 0.7,
            mass_coupling_factor: 0.2,
        }
    }

    pub fn with_schedule(mut self, schedule: DailySchedule) -> Self {
        self.schedule = schedule;
        self
    }

    pub fn validate(&self) -> Result<(), String> {
        let total = self.radiative_fraction + self.convective_fraction;
        if (total - 1.0).abs() > 1e-10 {
            return Err(format!(
                "Radiative + convective fractions must sum to 1.0, got {}",
                total
            ));
        }
        if !(0.0..=1.0).contains(&self.mass_coupling_factor) {
            return Err(format!(
                "Mass coupling factor must be in [0, 1], got {}",
                self.mass_coupling_factor
            ));
        }
        Ok(())
    }
}

impl Equipment for ComputerEquipment {
    fn id(&self) -> &str {
        &self.id
    }

    fn power_at_hour(&self, hour_of_year: usize) -> f64 {
        self.rated_power_w * self.count as f64 * self.schedule.value(hour_of_year % 24)
    }

    fn convective_gains(&self, hour_of_year: usize) -> f64 {
        self.power_at_hour(hour_of_year) * self.convective_fraction
    }

    fn radiative_gains(&self, hour_of_year: usize) -> f64 {
        self.power_at_hour(hour_of_year) * self.radiative_fraction
    }

    fn mass_coupling_factor(&self) -> f64 {
        self.mass_coupling_factor
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Server rack equipment (data center servers)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerRack {
    pub id: String,
    pub rated_power_w: f64,
    pub count: usize,
    pub schedule: DailySchedule,
    pub radiative_fraction: f64,
    pub convective_fraction: f64,
    pub mass_coupling_factor: f64,
}

impl ServerRack {
    pub fn new(id: String, rated_power_w: f64, count: usize) -> Self {
        Self {
            id,
            rated_power_w,
            count,
            schedule: DailySchedule::constant(1.0), // 24/7 by default
            radiative_fraction: 0.5,
            convective_fraction: 0.5,
            mass_coupling_factor: 0.8,
        }
    }

    pub fn with_schedule(mut self, schedule: DailySchedule) -> Self {
        self.schedule = schedule;
        self
    }
}

impl Equipment for ServerRack {
    fn id(&self) -> &str {
        &self.id
    }

    fn power_at_hour(&self, hour_of_year: usize) -> f64 {
        self.rated_power_w * self.count as f64 * self.schedule.value(hour_of_year % 24)
    }

    fn convective_gains(&self, hour_of_year: usize) -> f64 {
        self.power_at_hour(hour_of_year) * self.convective_fraction
    }

    fn radiative_gains(&self, hour_of_year: usize) -> f64 {
        self.power_at_hour(hour_of_year) * self.radiative_fraction
    }

    fn mass_coupling_factor(&self) -> f64 {
        self.mass_coupling_factor
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Generic equipment (any other equipment type)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericEquipment {
    pub id: String,
    pub rated_power_w: f64,
    pub count: usize,
    pub schedule: DailySchedule,
    pub radiative_fraction: f64,
    pub convective_fraction: f64,
    pub mass_coupling_factor: f64,
}

impl GenericEquipment {
    pub fn new(id: String, rated_power_w: f64, count: usize) -> Self {
        Self {
            id,
            rated_power_w,
            count,
            schedule: DailySchedule::new(),
            radiative_fraction: 0.5,
            convective_fraction: 0.5,
            mass_coupling_factor: 0.5,
        }
    }

    pub fn with_schedule(mut self, schedule: DailySchedule) -> Self {
        self.schedule = schedule;
        self
    }
}

impl Equipment for GenericEquipment {
    fn id(&self) -> &str {
        &self.id
    }

    fn power_at_hour(&self, hour_of_year: usize) -> f64 {
        self.rated_power_w * self.count as f64 * self.schedule.value(hour_of_year % 24)
    }

    fn convective_gains(&self, hour_of_year: usize) -> f64 {
        self.power_at_hour(hour_of_year) * self.convective_fraction
    }

    fn radiative_gains(&self, hour_of_year: usize) -> f64 {
        self.power_at_hour(hour_of_year) * self.radiative_fraction
    }

    fn mass_coupling_factor(&self) -> f64 {
        self.mass_coupling_factor
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equipment_trait() {
        let computers = ComputerEquipment::new("Computers".to_string(), 150.0, 10);
        let servers = ServerRack::new("Servers".to_string(), 500.0, 5);
        let generic = GenericEquipment::new("Generic".to_string(), 200.0, 8);

        let equipment: Vec<&dyn Equipment> = vec![&computers, &servers, &generic];

        for eq in equipment {
            assert!(!eq.id().is_empty());
            // Each equipment type has its own mass coupling factor
            assert!((0.0..=1.0).contains(&eq.mass_coupling_factor()));
        }
    }

    #[test]
    fn test_equipment_power_at_hour() {
        let mut computers = ComputerEquipment::new("Computers".to_string(), 150.0, 10);
        computers.schedule = DailySchedule::constant(0.5);

        let hour = 100;
        let power = computers.power_at_hour(hour);

        // 150W * 10 computers * 0.5 schedule = 750W
        assert!((power - 750.0).abs() < 1e-10);

        let convective = computers.convective_gains(hour);
        let radiative = computers.radiative_gains(hour);

        // 750W * 0.7 convective = 525W
        assert!((convective - 525.0).abs() < 1e-10);

        // 750W * 0.3 radiative = 225W
        assert!((radiative - 225.0).abs() < 1e-10);

        // Total power = convective + radiative
        assert!((power - (convective + radiative)).abs() < 1e-10);
    }

    #[test]
    fn test_mass_coupled_radiative() {
        let mut computers = ComputerEquipment::new("Computers".to_string(), 150.0, 10);
        computers.schedule = DailySchedule::constant(0.5);

        let hour = 100;
        let radiative = computers.radiative_gains(hour);
        let coupling_factor = computers.mass_coupling_factor();

        let radiative_to_mass = radiative * coupling_factor;
        let radiative_to_air = radiative * (1.0 - coupling_factor);

        // 225W radiative * 0.2 coupling = 45W to mass
        assert!((radiative_to_mass - 45.0).abs() < 1e-10);

        // 225W radiative * 0.8 (1 - 0.2) = 180W to air
        assert!((radiative_to_air - 180.0).abs() < 1e-10);

        // Total radiative = to_mass + to_air
        assert!((radiative - (radiative_to_mass + radiative_to_air)).abs() < 1e-10);
    }

    #[test]
    fn test_server_rack_24_7() {
        let servers = ServerRack::new("Servers".to_string(), 500.0, 5);

        // Servers should be 24/7 with constant 1.0 schedule
        let power_day = servers.power_at_hour(0);
        let power_night = servers.power_at_hour(2359);

        // 500W * 5 servers * 1.0 = 2500W constant
        assert!((power_day - 2500.0).abs() < 1e-10);
        assert!((power_night - 2500.0).abs() < 1e-10);
    }
}
