//! Occupancy Modeling and Schedules
//!
//! This module provides realistic occupancy modeling and schedule generation
//! for commercial building energy simulations.

use serde::{Deserialize, Serialize};

/// Building types for occupancy modeling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildingType {
    /// Office building
    Office,
    /// Retail store
    Retail,
    /// School/educational
    School,
    /// Hospital/healthcare
    Hospital,
    /// Hotel/motel
    Hotel,
    /// Restaurant
    Restaurant,
    /// Warehouse
    Warehouse,
}

/// Occupancy schedule type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OccupancyScheduleType {
    /// 24/7 continuous occupancy
    Continuous,
    /// Standard business hours (8am-6pm weekdays)
    StandardOffice,
    /// Extended hours (6am-10pm)
    Extended,
    /// Shift-based (two shifts)
    TwoShift,
    /// Weekend occupancy only
    WeekendOnly,
}

/// Represents an occupancy profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OccupancyProfile {
    /// Profile identifier
    pub id: String,
    /// Building type
    pub building_type: BuildingType,
    /// Schedule type
    pub schedule_type: OccupancyScheduleType,
    /// Maximum occupancy count
    pub max_occupancy: f64,
    /// Hourly occupancy fraction (0-1) for each hour of the week
    /// 168 values (7 days x 24 hours)
    pub hourly_schedule: Vec<f64>,
    /// Sensible heat gain per person (W)
    pub sensible_heat_per_person: f64,
    /// Latent heat gain per person (W)
    pub latent_heat_per_person: f64,
}

impl OccupancyProfile {
    /// Create a new occupancy profile
    pub fn new(id: String, building_type: BuildingType, max_occupancy: f64) -> Self {
        let schedule_type = match building_type {
            BuildingType::Office => OccupancyScheduleType::StandardOffice,
            BuildingType::Retail => OccupancyScheduleType::Extended,
            BuildingType::School => OccupancyScheduleType::StandardOffice,
            BuildingType::Hospital => OccupancyScheduleType::Continuous,
            BuildingType::Hotel => OccupancyScheduleType::Continuous,
            BuildingType::Restaurant => OccupancyScheduleType::Extended,
            BuildingType::Warehouse => OccupancyScheduleType::WeekendOnly,
        };

        let (sensible, latent) = heat_gains(building_type);

        Self {
            id,
            building_type,
            schedule_type,
            max_occupancy,
            hourly_schedule: Vec::with_capacity(168),
            sensible_heat_per_person: sensible,
            latent_heat_per_person: latent,
        }
    }

    /// Generate standard office occupancy schedule
    pub fn office_schedule(mut self) -> Self {
        self.hourly_schedule = vec![0.0; 168];
        
        for day in 0..5 {
            // Weekdays: low at night, ramp up morning, peak mid-day, evening
            for hour in 0..24 {
                let idx = day * 24 + hour;
                let fraction = match hour {
                    0..=6 => 0.05,   // Night: 5%
                    7 => 0.20,       // 7am: 20%
                    8 => 0.50,       // 8am: 50%
                    9..=11 => 0.90,  // Morning peak: 90%
                    12 => 0.80,      // Lunch: 80%
                    13..=16 => 0.90, // Afternoon: 90%
                    17 => 0.70,      // 5pm: 70%
                    18 => 0.40,      // 6pm: 40%
                    19 => 0.20,      // 7pm: 20%
                    20..=23 => 0.10, // Evening: 10%
                    _ => 0.05,
                };
                self.hourly_schedule[idx] = fraction;
            }
        }

        // Weekend: minimal occupancy
        for day in 5..7 {
            for hour in 0..24 {
                let idx = day * 24 + hour;
                self.hourly_schedule[idx] = 0.05;
            }
        }

        self
    }

    /// Generate retail occupancy schedule
    pub fn retail_schedule(mut self) -> Self {
        self.hourly_schedule = vec![0.0; 168];
        
        for day in 0..7 {
            for hour in 0..24 {
                let idx = day * 24 + hour;
                let fraction = match hour {
                    0..=8 => 0.0,
                    9 => 0.20,
                    10 => 0.40,
                    11..=14 => 0.70,
                    15..=18 => 0.80,
                    19 => 0.60,
                    20 => 0.30,
                    21 => 0.10,
                    22..=23 => 0.0,
                    _ => 0.0,
                };
                self.hourly_schedule[idx] = fraction;
            }
        }

        self
    }

    /// Generate school occupancy schedule
    pub fn school_schedule(mut self) -> Self {
        self.hourly_schedule = vec![0.0; 168];
        
        // Weekdays only
        for day in 0..5 {
            for hour in 0..24 {
                let idx = day * 24 + hour;
                let fraction = match hour {
                    0..=6 => 0.10,
                    7 => 0.30,
                    8 => 0.80,
                    9..=14 => 0.95, // Class time
                    15 => 0.70,     // After school
                    16 => 0.40,
                    17..=23 => 0.10,
                    _ => 0.10,
                };
                self.hourly_schedule[idx] = fraction;
            }
        }

        // Weekend
        for day in 5..7 {
            for hour in 0..24 {
                let idx = day * 24 + hour;
                self.hourly_schedule[idx] = 0.10;
            }
        }

        self
    }

    /// Get occupancy count for a specific hour of the week
    pub fn occupancy_at(&self, hour_of_week: usize) -> f64 {
        let idx = hour_of_week % 168;
        self.max_occupancy * self.hourly_schedule[idx]
    }

    /// Get occupancy count for a specific day and hour
    pub fn occupancy_at_time(&self, day: usize, hour: usize) -> f64 {
        let idx = (day % 7) * 24 + (hour % 24);
        self.max_occupancy * self.hourly_schedule[idx]
    }

    /// Calculate total internal heat gains from occupancy
    pub fn internal_gains(&self, hour_of_week: usize) -> f64 {
        let occupancy = self.occupancy_at(hour_of_week);
        occupancy * (self.sensible_heat_per_person + self.latent_heat_per_person)
    }
}

/// Heat gains per person by building type
fn heat_gains(building_type: BuildingType) -> (f64, f64) {
    match building_type {
        BuildingType::Office => (75.0, 55.0),     // Seated office work
        BuildingType::Retail => (120.0, 80.0),     // Light work
        BuildingType::School => (80.0, 60.0),      // Classroom
        BuildingType::Hospital => (100.0, 100.0),  // Patient care
        BuildingType::Hotel => (90.0, 70.0),       // Hotel room
        BuildingType::Restaurant => (130.0, 100.0), // Restaurant
        BuildingType::Warehouse => (200.0, 50.0),  // Heavy work
    }
}

/// Demand-controlled ventilation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemandControlledVentilation {
    /// Whether DCV is enabled
    pub enabled: bool,
    /// Minimum ventilation rate when unoccupied (ACH)
    pub min_ach_unoccupied: f64,
    /// Maximum ventilation rate when fully occupied (ACH)
    pub max_ach_occupied: f64,
    /// CO2 setpoint for demand control (ppm)
    pub co2_setpoint: f64,
    /// Occupancy threshold to trigger increased ventilation (fraction)
    pub occupancy_threshold: f64,
}

impl DemandControlledVentilation {
    /// Create new DCV settings
    pub fn new() -> Self {
        Self {
            enabled: false,
            min_ach_unoccupied: 0.5,
            max_ach_occupied: 2.0,
            co2_setpoint: 1000.0,
            occupancy_threshold: 0.5,
        }
    }

    /// Calculate ventilation rate based on occupancy
    pub fn ventilation_rate(&self, occupancy_fraction: f64) -> f64 {
        if !self.enabled {
            return self.max_ach_occupied;
        }

        if occupancy_fraction < self.occupancy_threshold {
            self.min_ach_unoccupied
        } else {
            self.min_ach_unoccupied
                + (self.max_ach_occupied - self.min_ach_unoccupied) * occupancy_fraction
        }
    }
}

impl Default for DemandControlledVentilation {
    fn default() -> Self {
        Self::new()
    }
}

/// Occupancy-based controls for lighting and equipment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OccupancyControls {
    /// Lighting control enabled
    pub lighting_control: bool,
    /// Equipment control enabled
    pub equipment_control: bool,
    /// Delay before turning off after vacancy (minutes)
    pub off_delay_minutes: u32,
    /// Partial ON fraction for lighting (0-1)
    pub partial_on_fraction: f64,
}

impl OccupancyControls {
    /// Create new occupancy controls
    pub fn new() -> Self {
        Self {
            lighting_control: true,
            equipment_control: true,
            off_delay_minutes: 15,
            partial_on_fraction: 0.5,
        }
    }

    /// Determine if lights should be on based on occupancy
    pub fn should_lights_on(&self, occupied: bool, time_since_occupancy: u32) -> bool {
        if !self.lighting_control {
            return true;
        }

        if occupied {
            true
        } else {
            time_since_occupancy < self.off_delay_minutes
        }
    }

    /// Determine lighting level based on occupancy
    pub fn lighting_level(&self, occupied: bool, time_since_occupancy: u32) -> f64 {
        if !self.lighting_control {
            return 1.0;
        }

        if occupied {
            1.0
        } else if time_since_occupancy < self.off_delay_minutes {
            self.partial_on_fraction
        } else {
            0.0
        }
    }
}

impl Default for OccupancyControls {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_occupancy_profile_office() {
        let profile = OccupancyProfile::new(
            "Office-1".to_string(),
            BuildingType::Office,
            100.0,
        ).office_schedule();

        // Peak occupancy during work hours
        let occupancy = profile.occupancy_at_time(2, 10); // Wednesday 10am
        assert!(occupancy > 80.0);

        // Low occupancy at night
        let night_occupancy = profile.occupancy_at_time(0, 2); // Sunday 2am
        assert!(night_occupancy < 10.0);
    }

    #[test]
    fn test_occupancy_profile_retail() {
        let profile = OccupancyProfile::new(
            "Retail-1".to_string(),
            BuildingType::Retail,
            50.0,
        ).retail_schedule();

        let occupancy = profile.occupancy_at_time(3, 14); // Saturday 2pm
        assert!(occupancy > 0.0);
    }

    #[test]
    fn test_internal_gains() {
        let profile = OccupancyProfile::new(
            "Office-1".to_string(),
            BuildingType::Office,
            100.0,
        ).office_schedule();

        let gains = profile.internal_gains(50); // Tuesday 2am
        assert!(gains > 0.0);
    }

    #[test]
    fn test_demand_controlled_ventilation() {
        let mut dcv = DemandControlledVentilation::new();
        dcv.enabled = true;

        // Unoccupied
        let ach = dcv.ventilation_rate(0.1);
        assert!(ach < 1.0);

        // Fully occupied
        let ach_occupied = dcv.ventilation_rate(1.0);
        assert!(ach_occupied > ach);
    }

    #[test]
    fn test_occupancy_controls() {
        let controls = OccupancyControls::new();

        // Occupied
        assert!(controls.should_lights_on(true, 0));

        // Just left - within delay
        assert!(controls.should_lights_on(false, 5));

        // Left - beyond delay
        assert!(!controls.should_lights_on(false, 20));
    }
}
