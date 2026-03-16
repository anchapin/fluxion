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
    /// 7-day weekly cycle.
    Weekly,
    /// Arbitrary hourly data (future).
    Custom,
}

/// Day type for weekly schedules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DayType {
    /// Aggregate type for Monday-Friday
    Weekday,
    /// Aggregate type for Saturday-Sunday
    Weekend,
    /// Holiday (falls back to weekday schedule)
    Holiday,
    /// Specific days
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

/// Schedule values storage based on schedule type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScheduleValues {
    /// 24-hour daily values.
    Daily([f64; 24]),
    /// 7-day weekly values (7 days × 24 hours = 168 values).
    Weekly([[f64; 24]; 7]),
}

/// A schedule with hourly resolution for a 24-hour period or 7-day weekly cycle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DailySchedule {
    /// Schedule name or identifier.
    pub name: String,
    /// Schedule type.
    pub schedule_type: ScheduleType,
    /// Schedule values (storage varies by type).
    pub values: ScheduleValues,
}

impl DailySchedule {
    /// Creates a new, empty schedule with all values at zero.
    pub fn new() -> Self {
        Self {
            name: "Default Schedule".to_string(),
            schedule_type: ScheduleType::DailyCycle,
            values: ScheduleValues::Daily([0.0; 24]),
        }
    }

    /// Creates a new weekly schedule with 168 zero values (7 days × 24 hours).
    pub fn weekly(name: String) -> Self {
        Self {
            name,
            schedule_type: ScheduleType::Weekly,
            values: ScheduleValues::Weekly([[0.0; 24]; 7]),
        }
    }

    /// Sets the value for a specific hour (for daily schedules).
    pub fn set_hour(&mut self, hour: usize, value: f64) {
        if hour < 24 {
            match &mut self.values {
                ScheduleValues::Daily(arr) => arr[hour] = value,
                ScheduleValues::Weekly(_) => panic!("Use set_hour_for_day for weekly schedules"),
            }
        }
    }

    /// Sets the value for a specific hour on a specific day (for weekly schedules).
    pub fn set_hour_for_day(&mut self, day: usize, hour: usize, value: f64) {
        if day < 7 && hour < 24 {
            if let ScheduleValues::Weekly(weekly) = &mut self.values {
                weekly[day][hour] = value;
            }
        }
    }

    /// Fills a range of hours with a specific value (for daily schedules).
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

    /// Fills a range of hours for a specific day with a specific value (for weekly schedules).
    ///
    /// Range is [start_hour, end_hour), wrapping around midnight if start > end.
    /// If start_hour == end_hour, no hours are filled.
    pub fn fill_range_for_day(
        &mut self,
        day: usize,
        start_hour: usize,
        end_hour: usize,
        value: f64,
    ) {
        if day >= 7 || start_hour == end_hour {
            return;
        }
        if start_hour < end_hour {
            for i in start_hour..end_hour {
                self.set_hour_for_day(day, i, value);
            }
        } else {
            // Wraps midnight
            for i in start_hour..24 {
                self.set_hour_for_day(day, i, value);
            }
            for i in 0..end_hour {
                self.set_hour_for_day(day, i, value);
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

    /// Returns the value for a given hour (for daily schedules).
    pub fn value(&self, hour: usize) -> f64 {
        match &self.values {
            ScheduleValues::Daily(arr) => arr[hour % 24],
            ScheduleValues::Weekly(weekly) => weekly[0][hour % 24], // Fallback to Monday
        }
    }

    /// Returns the value for a given day type and hour.
    ///
    /// For DayType::Weekday, returns Monday's values (index 0).
    /// For DayType::Weekend, returns Saturday's values (index 5).
    /// For DayType::Holiday, returns Monday's values (index 0).
    pub fn value_for_day(&self, day_type: DayType, hour: usize) -> f64 {
        let day_idx = match day_type {
            DayType::Weekday => 0,
            DayType::Weekend => 5,
            DayType::Holiday => 0,
            DayType::Monday => 0,
            DayType::Tuesday => 1,
            DayType::Wednesday => 2,
            DayType::Thursday => 3,
            DayType::Friday => 4,
            DayType::Saturday => 5,
            DayType::Sunday => 6,
        };
        match &self.values {
            ScheduleValues::Daily(arr) => arr[hour % 24],
            ScheduleValues::Weekly(weekly) => weekly[day_idx][hour % 24],
        }
    }

    /// Fills weekday hours (Monday-Friday) with a specific value.
    ///
    /// # Arguments
    /// * `start_hour` - Start hour (0-23)
    /// * `end_hour` - End hour (0-23, exclusive)
    /// * `value` - Value to set
    pub fn fill_weekday(&mut self, start_hour: usize, end_hour: usize, value: f64) {
        if let ScheduleValues::Weekly(ref mut weekly) = self.values {
            for day in 0..5 {
                for hour in start_hour..end_hour {
                    if hour < 24 {
                        weekly[day][hour] = value;
                    }
                }
            }
        }
    }

    /// Fills weekend hours (Saturday-Sunday) with a specific value.
    ///
    /// # Arguments
    /// * `start_hour` - Start hour (0-23)
    /// * `end_hour` - End hour (0-23, exclusive)
    /// * `value` - Value to set
    pub fn fill_weekend(&mut self, start_hour: usize, end_hour: usize, value: f64) {
        if let ScheduleValues::Weekly(ref mut weekly) = self.values {
            for day in 5..7 {
                for hour in start_hour..end_hour {
                    if hour < 24 {
                        weekly[day][hour] = value;
                    }
                }
            }
        }
    }

    /// Fills holiday hours (all days) with a specific value.
    ///
    /// # Arguments
    /// * `start_hour` - Start hour (0-23)
    /// * `end_hour` - End hour (0-23, exclusive)
    /// * `value` - Value to set
    pub fn fill_holiday(&mut self, start_hour: usize, end_hour: usize, value: f64) {
        if let ScheduleValues::Weekly(ref mut weekly) = self.values {
            for day in 0..7 {
                for hour in start_hour..end_hour {
                    if hour < 24 {
                        weekly[day][hour] = value;
                    }
                }
            }
        }
    }
}

impl DailySchedule {
    /// Creates an office hours pattern for a weekly schedule (8am-6pm Monday-Friday).
    ///
    /// This method returns self for builder-style chaining:
    /// ```ignore
    /// let schedule = DailySchedule::weekly("Office".to_string()).office_hours();
    /// ```
    pub fn office_hours(mut self) -> Self {
        if let ScheduleValues::Weekly(ref mut weekly) = self.values {
            for day in 0..5 {
                // Monday-Friday
                for hour in 8..=17 {
                    // 8am-6pm
                    weekly[day][hour] = 1.0;
                }
            }
        }
        self
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
        fn is_heating_off(schedule: &DailySchedule) -> bool {
            match &schedule.values {
                ScheduleValues::Daily(arr) => arr.iter().all(|&s| s <= -100.0),
                ScheduleValues::Weekly(weekly) => {
                    weekly.iter().all(|day| day.iter().all(|&s| s <= -100.0))
                }
            }
        }

        fn is_cooling_off(schedule: &DailySchedule) -> bool {
            match &schedule.values {
                ScheduleValues::Daily(arr) => arr.iter().all(|&s| s >= 100.0),
                ScheduleValues::Weekly(weekly) => {
                    weekly.iter().all(|day| day.iter().all(|&s| s >= 100.0))
                }
            }
        }

        is_heating_off(&self.heating) && is_cooling_off(&self.cooling)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weekly_schedule_factory() {
        let schedule = DailySchedule::weekly("Test".to_string());

        // Verify schedule type is Weekly
        assert_eq!(schedule.schedule_type, ScheduleType::Weekly);

        // Verify 7 days of 24 hours each
        match &schedule.values {
            ScheduleValues::Weekly(weekly) => {
                assert_eq!(weekly.len(), 7);
                for day in weekly {
                    assert_eq!(day.len(), 24);
                }
            }
            ScheduleValues::Daily(_) => panic!("Expected Weekly schedule"),
        }
    }

    #[test]
    fn test_value_for_day_specific_days() {
        let mut schedule = DailySchedule::weekly("Test".to_string());

        // Set Monday 8am to 1.0
        schedule.set_hour_for_day(0, 8, 1.0);
        // Set Tuesday 8am to 2.0
        schedule.set_hour_for_day(1, 8, 2.0);
        // Set Saturday 8am to 3.0
        schedule.set_hour_for_day(5, 8, 3.0);

        assert_eq!(schedule.value_for_day(DayType::Monday, 8), 1.0);
        assert_eq!(schedule.value_for_day(DayType::Tuesday, 8), 2.0);
        assert_eq!(schedule.value_for_day(DayType::Saturday, 8), 3.0);
    }

    #[test]
    fn test_value_for_day_aggregate_types() {
        let mut schedule = DailySchedule::weekly("Test".to_string());

        // Set Monday 8am to 1.0
        schedule.set_hour_for_day(0, 8, 1.0);
        // Set Saturday 8am to 2.0
        schedule.set_hour_for_day(5, 8, 2.0);

        // Weekday uses Monday (index 0)
        assert_eq!(schedule.value_for_day(DayType::Weekday, 8), 1.0);
        // Weekend uses Saturday (index 5)
        assert_eq!(schedule.value_for_day(DayType::Weekend, 8), 2.0);
        // Holiday uses Monday (index 0)
        assert_eq!(schedule.value_for_day(DayType::Holiday, 8), 1.0);
    }

    #[test]
    fn test_fill_range_for_day() {
        let mut schedule = DailySchedule::weekly("Test".to_string());

        schedule.fill_range_for_day(0, 8, 18, 1.0);
        schedule.fill_range_for_day(5, 10, 16, 0.5);

        // Check Monday 8am-5pm
        for hour in 8..18 {
            assert_eq!(schedule.value_for_day(DayType::Monday, hour), 1.0);
        }

        // Check Saturday 10am-4pm
        for hour in 10..16 {
            assert_eq!(schedule.value_for_day(DayType::Saturday, hour), 0.5);
        }
    }

    #[test]
    fn test_office_hours() {
        let schedule = DailySchedule::weekly("Office".to_string()).office_hours();

        // Verify Monday-Friday 8am-6pm are filled with 1.0
        for day in 0..5 {
            for hour in 8..=17 {
                assert_eq!(
                    schedule.value_for_day(
                        match day {
                            0 => DayType::Monday,
                            1 => DayType::Tuesday,
                            2 => DayType::Wednesday,
                            3 => DayType::Thursday,
                            4 => DayType::Friday,
                            _ => panic!("Invalid day"),
                        },
                        hour
                    ),
                    1.0
                );
            }
        }

        // Verify weekend hours are zero
        for day in 5..7 {
            for hour in 0..24 {
                assert_eq!(
                    schedule.value_for_day(
                        match day {
                            5 => DayType::Saturday,
                            6 => DayType::Sunday,
                            _ => panic!("Invalid day"),
                        },
                        hour
                    ),
                    0.0
                );
            }
        }

        // Verify weekday hours outside 8am-6pm are zero
        for day in 0..5 {
            for hour in 0..8 {
                assert_eq!(schedule.value_for_day(DayType::Weekday, hour), 0.0);
            }
            for hour in 18..24 {
                assert_eq!(schedule.value_for_day(DayType::Weekday, hour), 0.0);
            }
        }
    }

    #[test]
    fn test_fill_weekday() {
        let mut schedule = DailySchedule::weekly("Test".to_string());

        schedule.fill_weekday(9, 17, 0.5);

        // Verify weekday hours filled
        for day in 0..5 {
            for hour in 9..17 {
                assert_eq!(schedule.value_for_day(DayType::Weekday, hour), 0.5);
            }
        }

        // Verify weekend hours not filled
        for day in 5..7 {
            for hour in 9..17 {
                assert_eq!(
                    schedule.value_for_day(
                        match day {
                            5 => DayType::Saturday,
                            6 => DayType::Sunday,
                            _ => panic!("Invalid day"),
                        },
                        hour
                    ),
                    0.0
                );
            }
        }
    }

    #[test]
    fn test_fill_weekend() {
        let mut schedule = DailySchedule::weekly("Test".to_string());

        schedule.fill_weekend(10, 18, 0.3);

        // Verify weekend hours filled
        for day in 5..7 {
            for hour in 10..18 {
                assert_eq!(
                    schedule.value_for_day(
                        match day {
                            5 => DayType::Saturday,
                            6 => DayType::Sunday,
                            _ => panic!("Invalid day"),
                        },
                        hour
                    ),
                    0.3
                );
            }
        }

        // Verify weekday hours not filled
        for day in 0..5 {
            for hour in 10..18 {
                assert_eq!(schedule.value_for_day(DayType::Weekday, hour), 0.0);
            }
        }
    }

    #[test]
    fn test_fill_holiday() {
        let mut schedule = DailySchedule::weekly("Test".to_string());

        schedule.fill_holiday(12, 14, 0.8);

        // Verify all days filled
        for day in 0..7 {
            for hour in 12..14 {
                assert_eq!(
                    schedule.value_for_day(
                        match day {
                            0 => DayType::Monday,
                            1 => DayType::Tuesday,
                            2 => DayType::Wednesday,
                            3 => DayType::Thursday,
                            4 => DayType::Friday,
                            5 => DayType::Saturday,
                            6 => DayType::Sunday,
                            _ => panic!("Invalid day"),
                        },
                        hour
                    ),
                    0.8
                );
            }
        }
    }
}
