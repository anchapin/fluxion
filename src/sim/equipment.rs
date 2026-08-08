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

/// IT Equipment Load for data centers with UPS loss modeling
///
/// Data center IT equipment differs from generic equipment:
/// - 100% sensible heat (no latent heat)
/// - UPS overhead losses that generate additional heat
/// - Constant 24/7 load profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ITEquipmentLoad {
    pub id: String,
    pub rated_power_w: f64,
    pub ups_efficiency: f64,
    pub standby_loss_w: f64,
    pub count: usize,
    pub schedule: DailySchedule,
    pub radiative_fraction: f64,
    pub convective_fraction: f64,
    pub mass_coupling_factor: f64,
}

impl ITEquipmentLoad {
    pub fn new(
        id: String,
        rated_power_w: f64,
        ups_efficiency: f64,
        standby_loss_w: f64,
        count: usize,
    ) -> Self {
        Self {
            id,
            rated_power_w,
            ups_efficiency,
            standby_loss_w,
            count,
            schedule: DailySchedule::constant(1.0),
            radiative_fraction: 0.0,
            convective_fraction: 1.0,
            mass_coupling_factor: 0.0,
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
        if !(0.0..=1.0).contains(&self.ups_efficiency) {
            return Err(format!(
                "UPS efficiency must be in [0, 1], got {}",
                self.ups_efficiency
            ));
        }
        if self.rated_power_w < 0.0 {
            return Err(format!(
                "Rated power must be non-negative, got {}",
                self.rated_power_w
            ));
        }
        if self.standby_loss_w < 0.0 {
            return Err(format!(
                "Standby loss must be non-negative, got {}",
                self.standby_loss_w
            ));
        }
        Ok(())
    }

    fn ups_loss_at_hour(&self, hour_of_year: usize) -> f64 {
        let it_power =
            self.rated_power_w * self.count as f64 * self.schedule.value(hour_of_year % 24);
        let total_power = it_power / self.ups_efficiency;
        total_power - it_power
    }

    fn total_sensible_gain(&self, hour_of_year: usize) -> f64 {
        let it_power =
            self.rated_power_w * self.count as f64 * self.schedule.value(hour_of_year % 24);
        let ups_loss = self.ups_loss_at_hour(hour_of_year);
        let standby_loss = self.standby_loss_w * self.count as f64;
        it_power + ups_loss + standby_loss
    }
}

impl Equipment for ITEquipmentLoad {
    fn id(&self) -> &str {
        &self.id
    }

    fn power_at_hour(&self, hour_of_year: usize) -> f64 {
        self.total_sensible_gain(hour_of_year)
    }

    fn convective_gains(&self, hour_of_year: usize) -> f64 {
        self.total_sensible_gain(hour_of_year) * self.convective_fraction
    }

    fn radiative_gains(&self, hour_of_year: usize) -> f64 {
        self.total_sensible_gain(hour_of_year) * self.radiative_fraction
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

    #[test]
    fn test_computer_equipment_validation_valid() {
        let computers = ComputerEquipment::new("Computers".to_string(), 150.0, 10);
        assert!(computers.validate().is_ok());
    }

    #[test]
    fn test_computer_equipment_validation_invalid_fractions() {
        let mut computers = ComputerEquipment::new("Computers".to_string(), 150.0, 10);
        computers.radiative_fraction = 0.8;
        computers.convective_fraction = 0.8; // Sum = 1.6
        assert!(computers.validate().is_err());
    }

    #[test]
    fn test_computer_equipment_validation_invalid_coupling() {
        let mut computers = ComputerEquipment::new("Computers".to_string(), 150.0, 10);
        computers.mass_coupling_factor = 1.5;
        assert!(computers.validate().is_err());
    }

    #[test]
    fn test_computer_equipment_with_schedule() {
        let schedule = DailySchedule::constant(0.8);
        let computers =
            ComputerEquipment::new("Computers".to_string(), 100.0, 5).with_schedule(schedule);
        let power = computers.power_at_hour(100);
        assert!((power - 400.0).abs() < 1e-10); // 100 * 5 * 0.8
    }

    #[test]
    fn test_server_rack_with_schedule() {
        let schedule = DailySchedule::constant(0.5);
        let servers = ServerRack::new("Servers".to_string(), 200.0, 10).with_schedule(schedule);
        let power = servers.power_at_hour(500);
        assert!((power - 1000.0).abs() < 1e-10); // 200 * 10 * 0.5
    }

    #[test]
    fn test_generic_equipment_default_values() {
        let generic = GenericEquipment::new("Generic".to_string(), 100.0, 1);
        assert_eq!(generic.radiative_fraction, 0.5);
        assert_eq!(generic.convective_fraction, 0.5);
        assert_eq!(generic.mass_coupling_factor, 0.5);
    }

    #[test]
    fn test_generic_equipment_with_schedule() {
        let schedule = DailySchedule::constant(0.75);
        let generic =
            GenericEquipment::new("Generic".to_string(), 200.0, 4).with_schedule(schedule);
        let power = generic.power_at_hour(1000);
        assert!((power - 600.0).abs() < 1e-10); // 200 * 4 * 0.75
    }

    #[test]
    fn test_equipment_power_zero_count() {
        let computers = ComputerEquipment::new("Computers".to_string(), 150.0, 0);
        assert_eq!(computers.power_at_hour(100), 0.0);
    }

    #[test]
    fn test_equipment_power_zero_rated() {
        let computers = ComputerEquipment::new("Computers".to_string(), 0.0, 10);
        assert_eq!(computers.power_at_hour(100), 0.0);
    }

    #[test]
    fn test_equipment_power_off_schedule() {
        let mut computers = ComputerEquipment::new("Computers".to_string(), 150.0, 10);
        computers.schedule = DailySchedule::constant(0.0);
        assert_eq!(computers.power_at_hour(100), 0.0);
    }

    #[test]
    fn test_equipment_hour_wraparound() {
        let mut computers = ComputerEquipment::new("Computers".to_string(), 100.0, 1);
        computers.schedule = DailySchedule::constant(1.0);
        assert_eq!(computers.power_at_hour(24), computers.power_at_hour(0));
        assert_eq!(computers.power_at_hour(48), computers.power_at_hour(0));
    }

    #[test]
    fn test_equipment_as_any_downcast() {
        let computers = ComputerEquipment::new("Computers".to_string(), 150.0, 10);
        let any_ref = computers.as_any();
        let downcasted = any_ref.downcast_ref::<ComputerEquipment>();
        assert!(downcasted.is_some());
        assert_eq!(downcasted.unwrap().id, "Computers");
    }

    #[test]
    fn test_server_rack_as_any_downcast() {
        let servers = ServerRack::new("Servers".to_string(), 500.0, 5);
        let any_ref = servers.as_any();
        let downcasted = any_ref.downcast_ref::<ServerRack>();
        assert!(downcasted.is_some());
        assert_eq!(downcasted.unwrap().id, "Servers");
    }

    #[test]
    fn test_equipment_trait_polymorphism() {
        let computers: Box<dyn Equipment> =
            Box::new(ComputerEquipment::new("C1".to_string(), 100.0, 2));
        let servers: Box<dyn Equipment> = Box::new(ServerRack::new("S1".to_string(), 200.0, 3));
        let generic: Box<dyn Equipment> =
            Box::new(GenericEquipment::new("G1".to_string(), 150.0, 1));

        let equipment_list: Vec<Box<dyn Equipment>> = vec![computers, servers, generic];

        for eq in &equipment_list {
            assert!(!eq.id().is_empty());
            assert!(eq.power_at_hour(100) >= 0.0);
            assert!(eq.convective_gains(100) >= 0.0);
            assert!(eq.radiative_gains(100) >= 0.0);
            let factor = eq.mass_coupling_factor();
            assert!((0.0..=1.0).contains(&factor));
        }
    }

    #[test]
    fn test_equipment_gains_sum_to_power() {
        let mut computers = ComputerEquipment::new("Computers".to_string(), 100.0, 5);
        computers.schedule = DailySchedule::constant(1.0);

        for hour in 0..24 {
            let power = computers.power_at_hour(hour);
            let convective = computers.convective_gains(hour);
            let radiative = computers.radiative_gains(hour);
            assert!((power - (convective + radiative)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_server_rack_default_schedule_is_24_7() {
        let servers = ServerRack::new("Servers".to_string(), 100.0, 1);
        // Check all hours have same power
        let first_power = servers.power_at_hour(0);
        for hour in 0..24 {
            assert_eq!(servers.power_at_hour(hour), first_power);
        }
    }

    #[test]
    fn test_computer_equipment_default_fractions() {
        let computers = ComputerEquipment::new("Computers".to_string(), 100.0, 1);
        assert_eq!(computers.radiative_fraction, 0.3);
        assert_eq!(computers.convective_fraction, 0.7);
        assert_eq!(computers.mass_coupling_factor, 0.2);
    }

    #[test]
    fn test_server_rack_default_fractions() {
        let servers = ServerRack::new("Servers".to_string(), 100.0, 1);
        assert_eq!(servers.radiative_fraction, 0.5);
        assert_eq!(servers.convective_fraction, 0.5);
        assert_eq!(servers.mass_coupling_factor, 0.8);
    }

    #[test]
    fn test_it_equipment_load_default_fractions() {
        let it = ITEquipmentLoad::new("IT".to_string(), 1000.0, 0.95, 10.0, 1);
        assert_eq!(it.radiative_fraction, 0.0);
        assert_eq!(it.convective_fraction, 1.0);
        assert_eq!(it.mass_coupling_factor, 0.0);
    }

    #[test]
    fn test_it_equipment_load_ups_loss_90_percent_efficient() {
        let it = ITEquipmentLoad::new("IT".to_string(), 10000.0, 0.90, 0.0, 1);
        let hour = 100;
        let ups_loss = it.ups_loss_at_hour(hour);
        assert!((ups_loss - 1111.11).abs() < 1.0);
    }

    #[test]
    fn test_it_equipment_load_total_gain_11kw_with_90_percent_ups() {
        let it = ITEquipmentLoad::new("IT".to_string(), 10000.0, 0.90, 0.0, 1);
        let hour = 100;
        let total_gain = it.total_sensible_gain(hour);
        assert!((total_gain - 11111.11).abs() < 1.0);
    }

    #[test]
    fn test_it_equipment_load_100_percent_sensible() {
        let it = ITEquipmentLoad::new("IT".to_string(), 5000.0, 0.95, 50.0, 1);
        let hour = 100;
        let convective = it.convective_gains(hour);
        let radiative = it.radiative_gains(hour);
        assert!((convective - it.total_sensible_gain(hour)).abs() < 1e-10);
        assert!((radiative - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_it_equipment_load_standby_loss_when_server_off() {
        let it = ITEquipmentLoad::new("IT".to_string(), 10000.0, 0.90, 5.0, 1);
        let hour_off = 100;
        let mut schedule_off = DailySchedule::constant(0.0);
        let it_with_off_schedule = it.with_schedule(schedule_off);
        let gain_when_off = it_with_off_schedule.total_sensible_gain(hour_off);
        assert!((gain_when_off - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_it_equipment_load_standby_plus_ups_loss_when_server_off() {
        let it = ITEquipmentLoad::new("IT".to_string(), 10000.0, 0.90, 5.0, 1);
        let mut schedule_off = DailySchedule::constant(0.0);
        let it_off = it.with_schedule(schedule_off);
        let gain = it_off.total_sensible_gain(0);
        let ups_loss_when_off = it_off.ups_loss_at_hour(0);
        assert!((ups_loss_when_off - 0.0).abs() < 1e-10);
        assert!((gain - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_it_equipment_load_constant_24_7_schedule() {
        let it = ITEquipmentLoad::new("IT".to_string(), 1000.0, 0.95, 10.0, 1);
        let power_day = it.power_at_hour(0);
        let power_night = it.power_at_hour(2359);
        assert_eq!(power_day, power_night);
    }

    #[test]
    fn test_it_equipment_load_multiple_count() {
        let it = ITEquipmentLoad::new("IT".to_string(), 1000.0, 0.95, 10.0, 5);
        let hour = 100;
        let ups_loss = it.ups_loss_at_hour(hour);
        let expected_ups_loss = (1000.0 * 5.0 * 1.0) * (1.0 / 0.95 - 1.0);
        assert!((ups_loss - expected_ups_loss).abs() < 1.0);
    }

    #[test]
    fn test_it_equipment_load_validate_valid() {
        let it = ITEquipmentLoad::new("IT".to_string(), 1000.0, 0.95, 10.0, 1);
        assert!(it.validate().is_ok());
    }

    #[test]
    fn test_it_equipment_load_validate_invalid_ups_efficiency() {
        let it = ITEquipmentLoad::new("IT".to_string(), 1000.0, 1.5, 10.0, 1);
        assert!(it.validate().is_err());
    }

    #[test]
    fn test_it_equipment_load_validate_negative_rated_power() {
        let it = ITEquipmentLoad::new("IT".to_string(), -100.0, 0.95, 10.0, 1);
        assert!(it.validate().is_err());
    }

    #[test]
    fn test_it_equipment_load_validate_negative_standby_loss() {
        let it = ITEquipmentLoad::new("IT".to_string(), 1000.0, 0.95, -10.0, 1);
        assert!(it.validate().is_err());
    }

    #[test]
    fn test_it_equipment_load_as_any_downcast() {
        let it = ITEquipmentLoad::new("IT".to_string(), 1000.0, 0.95, 10.0, 1);
        let any_ref = it.as_any();
        let downcasted = any_ref.downcast_ref::<ITEquipmentLoad>();
        assert!(downcasted.is_some());
        assert_eq!(downcasted.unwrap().id, "IT");
    }

    #[test]
    fn test_it_equipment_load_gains_sum_to_power() {
        let it = ITEquipmentLoad::new("IT".to_string(), 1000.0, 0.95, 10.0, 1);
        for hour in 0..24 {
            let power = it.power_at_hour(hour);
            let convective = it.convective_gains(hour);
            let radiative = it.radiative_gains(hour);
            assert!((power - (convective + radiative)).abs() < 1e-10);
        }
    }
}
