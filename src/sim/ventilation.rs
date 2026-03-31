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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_ventilation() {
        let vent = ConstantVentilation::new(0.5);
        assert_eq!(vent.ach, 0.5);
        assert_eq!(vent.get_ach(0), 0.5);
        assert_eq!(vent.get_ach(12), 0.5);
        assert_eq!(vent.get_ach(23), 0.5);
    }

    #[test]
    fn test_constant_ventilation_clone() {
        let vent = ConstantVentilation::new(1.0);
        let cloned = vent.clone_box();
        assert_eq!(cloned.get_ach(5), 1.0);
    }

    #[test]
    fn test_scheduled_ventilation_default() {
        let vent = ScheduledVentilation::new(0.3, 2.0);
        assert_eq!(vent.base_ach, 0.3);
        assert_eq!(vent.fan_ach, 2.0);
        assert!(!vent.schedule.iter().any(|&x| x)); // all false
                                                    // Should return base_ach for all hours
        for hour in 0..24 {
            assert_eq!(vent.get_ach(hour), 0.3);
        }
    }

    #[test]
    fn test_night_ventilation_normal_range() {
        let vent = ScheduledVentilation::night_ventilation(0.3, 2.0, 22, 6);
        // Fan ON from hour 22 to 23, 0 to 5
        assert_eq!(vent.get_ach(21), 0.3); // before start
        assert_eq!(vent.get_ach(22), 2.3); // fan on
        assert_eq!(vent.get_ach(23), 2.3); // fan on
        assert_eq!(vent.get_ach(0), 2.3); // fan on (next day)
        assert_eq!(vent.get_ach(5), 2.3); // fan on
        assert_eq!(vent.get_ach(6), 0.3); // fan off
        assert_eq!(vent.get_ach(12), 0.3); // fan off
    }

    #[test]
    fn test_night_ventilation_same_start_end() {
        let vent = ScheduledVentilation::night_ventilation(0.3, 2.0, 10, 10);
        // When start == end, fan is on all 24 hours
        for hour in 0..24 {
            assert_eq!(vent.get_ach(hour), 2.3);
        }
    }

    #[test]
    fn test_night_ventilation_single_hour() {
        let vent = ScheduledVentilation::night_ventilation(0.5, 1.5, 14, 15);
        assert_eq!(vent.get_ach(13), 0.5);
        assert_eq!(vent.get_ach(14), 2.0); // fan on
        assert_eq!(vent.get_ach(15), 0.5); // fan off
    }

    #[test]
    fn test_scheduled_ventilation_clone() {
        let vent = ScheduledVentilation::night_ventilation(0.3, 2.0, 20, 8);
        let cloned = vent.clone_box();
        assert_eq!(cloned.get_ach(21), 2.3);
        assert_eq!(cloned.get_ach(10), 0.3);
    }

    #[test]
    fn test_ach_to_conductance() {
        // Standard values: ach=1.0, volume=100m³, rho=1.2, cp=1005
        let conductance = ach_to_conductance(1.0, 100.0, 1.2, 1005.0);
        assert!((conductance - 33.5).abs() < 0.01); // (1*100*1.2*1005)/3600 = 33.5
    }

    #[test]
    fn test_ach_to_conductance_zero() {
        assert_eq!(ach_to_conductance(0.0, 100.0, 1.2, 1005.0), 0.0);
    }

    #[test]
    fn test_ach_to_conductance_scaling() {
        // Doubling ACH should double conductance
        let c1 = ach_to_conductance(0.5, 100.0, 1.2, 1005.0);
        let c2 = ach_to_conductance(1.0, 100.0, 1.2, 1005.0);
        assert!((c2 - 2.0 * c1).abs() < 0.001);
    }

    #[test]
    fn test_ventilation_schedule_trait_object() {
        let vent1: Box<dyn VentilationSchedule> = Box::new(ConstantVentilation::new(0.5));
        let vent2: Box<dyn VentilationSchedule> =
            Box::new(ScheduledVentilation::night_ventilation(0.3, 2.0, 22, 6));

        assert_eq!(vent1.get_ach(10), 0.5);
        assert_eq!(vent2.get_ach(23), 2.3);
    }
}
