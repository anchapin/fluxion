//! Time-based scheduling for building systems.
//!
//! This module provides the `DailySchedule` struct for defining hourly values,
//! supporting various schedule types from constant values to complex daily cycles.

use serde::{Deserialize, Serialize};

/// Type of schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScheduleType {
    /// Constant value for all hours.
    Constant,
    /// 24-hour repeating daily cycle.
    DailyCycle,
    /// 7-day weekly cycle (future).
    Weekly,
    /// Arbitrary hourly data (future).
    Custom,
}

/// A schedule with hourly resolution for a 24-hour period.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DailySchedule {
    /// Schedule name or identifier.
    pub name: String,
    /// Schedule type.
    pub schedule_type: ScheduleType,
    /// Hourly values.
    pub values: [f64; 24],
}

impl DailySchedule {
    /// Creates a new, empty schedule with all values at zero.
    pub fn new() -> Self {
        Self {
            name: "Default Schedule".to_string(),
            schedule_type: ScheduleType::DailyCycle,
            values: [0.0; 24],
        }
    }

    /// Sets the value for a specific hour.
    pub fn set_hour(&mut self, hour: usize, value: f64) {
        if hour < 24 {
            self.values[hour] = value;
        }
    }

    /// Fills a range of hours with a specific value.
    ///
    /// Range is [start_hour, end_hour), wrapping around midnight if start > end.
    /// If start_hour == end_hour, no hours are filled.
    pub fn fill_range(&mut self, start_hour: usize, end_hour: usize, value: f64) {
        if start_hour == end_hour {
            return;
        }
        if start_hour < end_hour {
            for i in start_hour..end_hour {
                self.set_hour(i, value);
            }
        } else {
            // Wraps midnight
            for i in start_hour..24 {
                self.set_hour(i, value);
            }
            for i in 0..end_hour {
                self.set_hour(i, value);
            }
        }
    }

    /// Creates a constant schedule for all 24 hours.
    pub fn constant(value: f64) -> Self {
        let mut schedule = Self::new();
        schedule.schedule_type = ScheduleType::Constant;
        schedule.fill_range(0, 24, value);
        schedule
    }

    /// Returns the value for a given hour.
    pub fn value(&self, hour: usize) -> f64 {
        self.values[hour % 24]
    }
}

impl Default for DailySchedule {
    fn default() -> Self {
        Self::new()
    }
}

/// A combined HVAC schedule for heating and cooling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HVACSchedule {
    pub heating: DailySchedule,
    pub cooling: DailySchedule,
}

impl HVACSchedule {
    /// Creates a new HVAC schedule with default (zero) setpoints.
    pub fn new() -> Self {
        Self {
            heating: DailySchedule::new(),
            cooling: DailySchedule::new(),
        }
    }

    /// Creates a constant HVAC schedule.
    pub fn constant_schedule(heating_sp: f64, cooling_sp: f64) -> Self {
        Self {
            heating: DailySchedule::constant(heating_sp),
            cooling: DailySchedule::constant(cooling_sp),
        }
    }

    /// Creates a setback schedule.
    pub fn setback_schedule(
        day_heat: f64,
        night_heat: f64,
        cool_sp: f64,
        night_start: usize,
        night_end: usize,
    ) -> Self {
        let mut heating = DailySchedule::constant(day_heat);
        heating.fill_range(night_start, night_end, night_heat);
        Self {
            heating,
            cooling: DailySchedule::constant(cool_sp),
        }
    }

    /// Creates a schedule with operating hours.
    pub fn with_operating_hours(
        heating_sp: f64,
        cooling_sp: f64,
        start_hour: usize,
        end_hour: usize,
    ) -> Self {
        let mut heating = DailySchedule::constant(-100.0);
        let mut cooling = DailySchedule::constant(100.0);
        heating.fill_range(start_hour, end_hour, heating_sp);
        cooling.fill_range(start_hour, end_hour, cooling_sp);
        Self { heating, cooling }
    }

    /// Creates a free-floating schedule.
    pub fn free_floating() -> Self {
        Self::with_operating_hours(0.0, 0.0, 0, 0)
    }

    /// Returns true if this schedule represents a free-floating state (no HVAC control).
    pub fn is_free_floating(&self) -> bool {
        self.heating.values.iter().all(|&s| s <= -100.0) && 
        self.cooling.values.iter().all(|&s| s >= 100.0)
    }

    /// Returns the heating setpoint for a given hour.
    pub fn heating_setpoint(&self, hour: usize) -> f64 {
        self.heating.value(hour)
    }

    /// Returns the cooling setpoint for a given hour.
    pub fn cooling_setpoint(&self, hour: usize) -> f64 {
        self.cooling.value(hour)
    }
}

impl Default for HVACSchedule {
    fn default() -> Self {
        Self::new()
    }
}
