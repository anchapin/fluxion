//! Ventilation and infiltration modeling.
//!
//! This module provides tools for defining ventilation schedules and calculating
//! time-varying air change rates.

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Trait for defining air change rate (ACH) schedules.
pub trait VentilationSchedule: Debug + Send + Sync {
    /// Returns the air change rate (ACH) for a given hour.
    fn get_ach(&self, hour: usize) -> f64;
    /// Clones the schedule into a boxed trait object.
    fn clone_box(&self) -> Box<dyn VentilationSchedule>;
}

/// A constant ventilation schedule.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConstantVentilation {
    pub ach: f64,
}

impl ConstantVentilation {
    pub fn new(ach: f64) -> Self {
        Self { ach }
    }
}

impl VentilationSchedule for ConstantVentilation {
    fn get_ach(&self, _hour: usize) -> f64 {
        self.ach
    }
    fn clone_box(&self) -> Box<dyn VentilationSchedule> {
        Box::new(*self)
    }
}

/// A scheduled ventilation system with base infiltration and a timed fan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledVentilation {
    /// Base infiltration rate (ACH) always present.
    pub base_ach: f64,
    /// Additional ACH when fan is ON.
    pub fan_ach: f64,
    /// 24-hour binary schedule (true = fan ON, false = fan OFF).
    pub schedule: [bool; 24],
}

impl ScheduledVentilation {
    /// Creates a new scheduled ventilation.
    pub fn new(base_ach: f64, fan_ach: f64) -> Self {
        Self {
            base_ach,
            fan_ach,
            schedule: [false; 24],
        }
    }

    /// Creates a night ventilation schedule (ON during specified range).
    pub fn night_ventilation(
        base_ach: f64,
        fan_ach: f64,
        start_hour: usize,
        end_hour: usize,
    ) -> Self {
        let mut vent = Self::new(base_ach, fan_ach);
        if start_hour == end_hour {
            vent.schedule = [true; 24];
        } else if start_hour < end_hour {
            for i in start_hour..end_hour {
                vent.schedule[i] = true;
            }
        } else {
            for i in start_hour..24 {
                vent.schedule[i] = true;
            }
            for i in 0..end_hour {
                vent.schedule[i] = true;
            }
        }
        vent
    }
}

impl VentilationSchedule for ScheduledVentilation {
    fn get_ach(&self, hour: usize) -> f64 {
        if self.schedule[hour] {
            self.base_ach + self.fan_ach
        } else {
            self.base_ach
        }
    }
    fn clone_box(&self) -> Box<dyn VentilationSchedule> {
        Box::new(self.clone())
    }
}

/// Utility to calculate thermal conductance (W/K) from air change rate (ACH).
///
/// # Arguments
/// * `ach` - Air changes per hour (1/h)
/// * `volume` - Zone volume (m³)
/// * `rho` - Air density (kg/m³), typically 1.2
/// * `cp` - Specific heat capacity of air (J/kg·K), typically 1005
pub fn ach_to_conductance(ach: f64, volume: f64, rho: f64, cp: f64) -> f64 {
    (ach * volume * rho * cp) / 3600.0
}

