//! Zone-level HVAC control module
//!
//! This module provides zone-specific HVAC functionality including
//! setpoints, scheduling, and layered control strategies.

pub mod schedule;
pub mod zone_control;
pub mod zone_setpoints;

pub use schedule::{DailySchedule, DayType, HVACSchedule, ScheduleType, ScheduleValues};

pub use zone_control::{
    ControlStrategy, HVACStatus, LayeredController, LayeredControllerConfig, ZoneControl,
};
