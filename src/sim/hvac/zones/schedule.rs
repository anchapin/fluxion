//! Unified schedule and setback engine for building HVAC systems.
//!
//! This module provides a shared contract for schedules across Rust simulation,
//! CLI configuration, and Python bindings. It re-exports the core schedule types
//! from `sim::schedule` and provides conversion utilities from validation specs.

pub use crate::sim::schedule::DailySchedule;
pub use crate::sim::schedule::DayType;
pub use crate::sim::schedule::HVACSchedule;
pub use crate::sim::schedule::ScheduleType;
pub use crate::sim::schedule::ScheduleValues;

impl From<&crate::validation::ashrae_140_cases::HvacSchedule> for HVACSchedule {
    fn from(spec: &crate::validation::ashrae_140_cases::HvacSchedule) -> Self {
        if spec.is_free_floating() {
            return HVACSchedule::free_floating();
        }

        let mut heating = DailySchedule::new();
        let mut cooling = DailySchedule::new();

        heating.fill_range(0, 24, spec.heating_setpoint);
        cooling.fill_range(0, 24, spec.cooling_setpoint);

        if let (Some(setback_setpoint), Some((setback_start, setback_end))) =
            (spec.setback_setpoint, spec.setback_hours)
        {
            heating.fill_range(
                setback_start as usize,
                setback_end as usize,
                setback_setpoint,
            );
        }

        let (op_start, op_end) = spec.operating_hours;
        if op_start != op_end {
            let disabled_heating = -100.0;
            let disabled_cooling = 100.0;

            if op_end > op_start {
                heating.fill_range(0, op_start as usize, disabled_heating);
                heating.fill_range(op_end as usize, 24, disabled_heating);
                cooling.fill_range(0, op_start as usize, disabled_cooling);
                cooling.fill_range(op_end as usize, 24, disabled_cooling);
            } else {
                heating.fill_range(op_end as usize, op_start as usize, disabled_heating);
                cooling.fill_range(op_end as usize, op_start as usize, disabled_cooling);
            }
        }

        HVACSchedule { heating, cooling }
    }
}

impl From<crate::validation::ashrae_140_cases::HvacSchedule> for HVACSchedule {
    fn from(spec: crate::validation::ashrae_140_cases::HvacSchedule) -> Self {
        HVACSchedule::from(&spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_from_validation_hvac_schedule() {
        let validation_spec = crate::validation::ashrae_140_cases::HvacSchedule::with_setback(
            20.0, 25.0, 15.0, 22, 6,
        );

        let unified: HVACSchedule = (&validation_spec).into();

        assert_eq!(unified.heating_setpoint(10), 20.0);
        assert_eq!(unified.heating_setpoint(23), 15.0);
        assert_eq!(unified.heating_setpoint(2), 15.0);
        assert_eq!(unified.cooling_setpoint(10), 25.0);
    }

    #[test]
    fn test_convert_free_floating() {
        let validation_spec = crate::validation::ashrae_140_cases::HvacSchedule::free_floating();

        let unified: HVACSchedule = (&validation_spec).into();

        assert!(unified.is_free_floating());
    }

    #[test]
    fn test_convert_with_operating_hours() {
        let validation_spec =
            crate::validation::ashrae_140_cases::HvacSchedule::with_operating_hours(
                20.0, 25.0, 8, 18,
            );

        let unified: HVACSchedule = (&validation_spec).into();

        assert_eq!(unified.heating_setpoint(10), 20.0);
        assert_eq!(unified.heating_setpoint(2), -100.0);
    }

    #[test]
    fn test_roundtrip_conversion() {
        let validation_spec = crate::validation::ashrae_140_cases::HvacSchedule::with_setback(
            20.0, 25.0, 15.0, 22, 6,
        );
        let converted: HVACSchedule = (&validation_spec).into();

        assert_eq!(converted.heating_setpoint(10), 20.0);
        assert_eq!(converted.heating_setpoint(23), 15.0);
        assert_eq!(converted.cooling_setpoint(10), 25.0);
    }
}
