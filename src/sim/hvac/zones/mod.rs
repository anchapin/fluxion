//! HVAC module for zone-level heating, ventilation, and air conditioning control.
//!
//! This module provides the core HVAC functionality for multi-zone building energy modeling.

pub mod schedule;
pub mod zone_control;
pub mod zone_setpoints;

pub use schedule::{DailySchedule, DayType, HVACSchedule, ScheduleType, ScheduleValues};

pub use zone_control::{
    ControlStrategy, HVACStatus, LayeredController, LayeredControllerConfig, ZoneControl,
};

pub use zone_setpoints::OccupancyMode;
