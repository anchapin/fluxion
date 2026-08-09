use chrono::{DateTime, Timelike};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OccupantState {
    Absent,
    PresentActive,
    Sleeping,
}

impl Default for OccupantState {
    fn default() -> Self {
        OccupantState::Absent
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightingModel {
    pub installed_watts_per_area: f64,
    pub luminaire_efficacy: f64,
    pub daylight_fraction: f64,
}

impl Default for LightingModel {
    fn default() -> Self {
        Self::office()
    }
}

impl LightingModel {
    pub fn office() -> Self {
        Self {
            installed_watts_per_area: 10.0,
            luminaire_efficacy: 80.0,
            daylight_fraction: 0.5,
        }
    }

    pub fn with_watts_per_area(mut self, watts_per_area: f64) -> Self {
        self.installed_watts_per_area = watts_per_area;
        self
    }

    pub fn with_luminaire_efficacy(mut self, efficacy: f64) -> Self {
        self.luminaire_efficacy = efficacy;
        self
    }

    pub fn with_daylight_fraction(mut self, fraction: f64) -> Self {
        self.daylight_fraction = fraction;
        self
    }

    pub fn compute(&self, t: DateTime<chrono::Utc>, occupancy: OccupantState) -> f64 {
        let fraction_on = match occupancy {
            OccupantState::Absent => 0.0,
            OccupantState::PresentActive => 1.0,
            OccupantState::Sleeping => 0.1,
        };

        // Lighting demand is reduced by the daylight control response. The
        // `daylight_availability` term is a binary indicator (1.0 during
        // daytime hours, 0.0 at night) and `daylight_fraction` is the
        // daylight factor — the fraction of the lighting load that can be
        // offset by available daylight when controls are active.
        let daylight_availability = self.estimate_daylight_fraction(t);
        let lighting_demand = fraction_on * (1.0 - daylight_availability * self.daylight_fraction);

        let watts = self.installed_watts_per_area * lighting_demand;
        watts.max(0.0)
    }

    /// Returns the daylight availability indicator: 1.0 when daylight is
    /// available (daytime hours 06:00–18:00) and 0.0 otherwise.
    ///
    /// This is a coarse time-of-day indicator used by `compute()`. The
    /// magnitude of the daylight savings is controlled separately by
    /// `daylight_fraction` (the daylight factor).
    fn estimate_daylight_fraction(&self, t: DateTime<chrono::Utc>) -> f64 {
        let hour = t.hour();
        if hour >= 6 && hour <= 18 {
            1.0
        } else {
            0.0
        }
    }

    pub fn radiative_fraction(&self) -> f64 {
        0.7
    }

    pub fn convective_fraction(&self) -> f64 {
        0.3
    }

    pub fn radiative_gain(&self, t: DateTime<chrono::Utc>, occupancy: OccupantState) -> f64 {
        self.compute(t, occupancy) * self.radiative_fraction()
    }

    pub fn convective_gain(&self, t: DateTime<chrono::Utc>, occupancy: OccupantState) -> f64 {
        self.compute(t, occupancy) * self.convective_fraction()
    }
}

#[cfg(test)]
mod behavior_tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_lighting_default() {
        let lighting = LightingModel::default();
        assert!((lighting.installed_watts_per_area - 10.0).abs() < 1e-10);
        assert!((lighting.luminaire_efficacy - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_occupied_full_demand() {
        let lighting = LightingModel::office();
        let t = chrono::Utc.with_ymd_and_hms(2024, 7, 15, 12, 0, 0).unwrap();
        let power = lighting.compute(t, OccupantState::PresentActive);
        assert!((power - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_absent_no_demand() {
        let lighting = LightingModel::office();
        let t = chrono::Utc.with_ymd_and_hms(2024, 7, 15, 12, 0, 0).unwrap();
        let power = lighting.compute(t, OccupantState::Absent);
        assert!((power - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sleeping_night_light() {
        let lighting = LightingModel::office();
        let t = chrono::Utc.with_ymd_and_hms(2024, 7, 15, 12, 0, 0).unwrap();
        let power = lighting.compute(t, OccupantState::Sleeping);
        assert!((power - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_nighttime_no_daylight() {
        let lighting = LightingModel::office();
        let t = chrono::Utc.with_ymd_and_hms(2024, 7, 15, 22, 0, 0).unwrap();
        let power_night = lighting.compute(t, OccupantState::PresentActive);
        let t_day = chrono::Utc.with_ymd_and_hms(2024, 7, 15, 12, 0, 0).unwrap();
        let power_day = lighting.compute(t_day, OccupantState::PresentActive);
        assert!(power_night > power_day);
    }

    #[test]
    fn test_daylight_dimming_reduces_demand() {
        let lighting = LightingModel::office();
        let t = chrono::Utc.with_ymd_and_hms(2024, 7, 15, 12, 0, 0).unwrap();
        let power = lighting.compute(t, OccupantState::PresentActive);
        assert!(power < 10.0);
        assert!(power >= 5.0);
    }

    #[test]
    fn test_radiative_convective_fractions() {
        let lighting = LightingModel::office();
        assert!((lighting.radiative_fraction() - 0.7).abs() < 1e-10);
        assert!((lighting.convective_fraction() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_gain_methods() {
        let lighting = LightingModel::office();
        let t = chrono::Utc.with_ymd_and_hms(2024, 7, 15, 12, 0, 0).unwrap();
        let total = lighting.compute(t, OccupantState::PresentActive);
        let radiative = lighting.radiative_gain(t, OccupantState::PresentActive);
        let convective = lighting.convective_gain(t, OccupantState::PresentActive);
        assert!((radiative + convective - total).abs() < 1e-10);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleLightingModel {
    pub power_density: f64,
    pub daylighting_factor: f64,
    pub schedule: Vec<f64>,
}

impl Default for ScheduleLightingModel {
    fn default() -> Self {
        Self::office()
    }
}

impl ScheduleLightingModel {
    pub fn office() -> Self {
        Self {
            power_density: 10.0,
            daylighting_factor: 0.3,
            schedule: Self::default_schedule(),
        }
    }

    fn default_schedule() -> Vec<f64> {
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8,
            0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0,
        ]
    }

    pub fn with_power_density(mut self, power_density: f64) -> Self {
        self.power_density = power_density;
        self
    }

    pub fn with_daylighting_factor(mut self, factor: f64) -> Self {
        self.daylighting_factor = factor;
        self
    }

    pub fn lighting_power(&self, hour: f64, zone_area: f64, daylight_illuminance: f64) -> f64 {
        let schedule_fraction = self.schedule_fraction(hour);
        let base_power = self.power_density * zone_area;

        let daylight_reduction = (daylight_illuminance / 1000.0).min(1.0) * self.daylighting_factor;

        base_power * schedule_fraction * (1.0 - daylight_reduction)
    }

    fn schedule_fraction(&self, hour: f64) -> f64 {
        let index = hour as usize % 24;
        self.schedule.get(index).copied().unwrap_or(0.0)
    }

    pub fn radiative_fraction(&self) -> f64 {
        0.7
    }

    pub fn convective_fraction(&self) -> f64 {
        0.3
    }

    pub fn radiative_gain(&self, hour: f64, zone_area: f64, daylight_illuminance: f64) -> f64 {
        self.lighting_power(hour, zone_area, daylight_illuminance) * self.radiative_fraction()
    }

    pub fn convective_gain(&self, hour: f64, zone_area: f64, daylight_illuminance: f64) -> f64 {
        self.lighting_power(hour, zone_area, daylight_illuminance) * self.convective_fraction()
    }
}

#[cfg(test)]
mod schedule_tests {
    use super::*;

    #[test]
    fn test_lighting_default() {
        let lighting = ScheduleLightingModel::default();
        assert!((lighting.power_density - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_schedule_fraction() {
        let lighting = ScheduleLightingModel::default();
        assert!((lighting.schedule_fraction(10.0) - 1.0).abs() < 1e-10);
        assert!((lighting.schedule_fraction(1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_daylighting_reduction() {
        let lighting = ScheduleLightingModel::default();
        let power_no_daylight = lighting.lighting_power(10.0, 100.0, 0.0);
        let power_with_daylight = lighting.lighting_power(10.0, 100.0, 1000.0);
        assert!(power_with_daylight < power_no_daylight);
    }
}
