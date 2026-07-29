//! HVAC Sizing from Design Day Loads
//!
//! Computes HVAC equipment capacity by running simulations on extreme
//! heating and cooling design days and applying ASHRAE-recommended safety factors.

use crate::physics::cta::VectorField;
use crate::sim::thermal_model_core::ThermalModel;
use fluxion_core::weather::{generate_design_day_hours, DailySummary, DesignDaySelector};

pub struct HvacSizingResult {
    pub heating_capacity_w: f64,
    pub cooling_capacity_w: f64,
    pub heating_design_day: DailySummary,
    pub cooling_design_day: DailySummary,
}

pub struct HvacSizer {
    safety_factor: f64,
}

impl HvacSizer {
    pub fn new(safety_factor: f64) -> Self {
        Self { safety_factor }
    }

    pub fn size_from_thermal_model(
        &self,
        model: &mut ThermalModel<VectorField>,
        hourly_weather: &[fluxion_core::weather::HourlyWeatherData],
    ) -> Option<HvacSizingResult> {
        let mut selector = DesignDaySelector::new();
        selector.select_from_hourly(hourly_weather);

        let heating_spec = selector.heating_design()?.clone();
        let cooling_spec = selector.cooling_design()?.clone();

        let heating_hours = generate_design_day_hours(&fluxion_core::weather::DesignDaySpec {
            name: "Heating Design Day".to_string(),
            month: heating_spec.month,
            day_of_month: heating_spec.day_of_month,
            max_temp: heating_spec.min_temp,
            temp_range: heating_spec.temp_range,
            day_type: "WinterDesignDay".to_string(),
            wetbulb: Some(heating_spec.min_temp),
            humidity_type: Some("Wetbulb".to_string()),
            humidity_ratio: None,
            enthalpy: None,
        });

        let cooling_hours = generate_design_day_hours(&fluxion_core::weather::DesignDaySpec {
            name: "Cooling Design Day".to_string(),
            month: cooling_spec.month,
            day_of_month: cooling_spec.day_of_month,
            max_temp: cooling_spec.max_temp,
            temp_range: cooling_spec.temp_range,
            day_type: "SummerDesignDay".to_string(),
            wetbulb: Some(cooling_spec.max_temp - 5.0),
            humidity_type: Some("Wetbulb".to_string()),
            humidity_ratio: None,
            enthalpy: None,
        });

        let (heating_capacity_w, cooling_capacity_w) = model
            .calculate_hvac_capacity_from_design_day(
                &heating_hours,
                &cooling_hours,
                self.safety_factor,
            );

        Some(HvacSizingResult {
            heating_capacity_w,
            cooling_capacity_w,
            heating_design_day: heating_spec,
            cooling_design_day: cooling_spec,
        })
    }
}

impl Default for HvacSizer {
    fn default() -> Self {
        Self::new(1.15)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hvac_sizer_default_safety_factor() {
        let sizer = HvacSizer::default();
        assert!((sizer.safety_factor - 1.15).abs() < 1e-10);
    }

    #[test]
    fn test_hvac_sizer_custom_safety_factor() {
        let sizer = HvacSizer::new(1.20);
        assert!((sizer.safety_factor - 1.20).abs() < 1e-10);
    }
}
