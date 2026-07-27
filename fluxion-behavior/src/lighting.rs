use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightingModel {
    pub power_density: f64,
    pub daylighting_factor: f64,
    pub schedule: Vec<f64>,
}

impl Default for LightingModel {
    fn default() -> Self {
        Self::office()
    }
}

impl LightingModel {
    pub fn office() -> Self {
        Self {
            power_density: 10.0,
            daylighting_factor: 0.3,
            schedule: Self::default_schedule(),
        }
    }

    fn default_schedule() -> Vec<f64> {
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.8, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0,
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

    pub fn lighting_power(
        &self,
        hour: f64,
        zone_area: f64,
        daylight_illuminance: f64,
    ) -> f64 {
        let schedule_fraction = self.schedule_fraction(hour);
        let base_power = self.power_density * zone_area;

        let daylight_reduction =
            (daylight_illuminance / 1000.0).min(1.0) * self.daylighting_factor;

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

    pub fn radiative_gain(
        &self,
        hour: f64,
        zone_area: f64,
        daylight_illuminance: f64,
    ) -> f64 {
        self.lighting_power(hour, zone_area, daylight_illuminance) * self.radiative_fraction()
    }

    pub fn convective_gain(
        &self,
        hour: f64,
        zone_area: f64,
        daylight_illuminance: f64,
    ) -> f64 {
        self.lighting_power(hour, zone_area, daylight_illuminance) * self.convective_fraction()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lighting_default() {
        let lighting = LightingModel::default();
        assert!((lighting.power_density - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_schedule_fraction() {
        let lighting = LightingModel::default();
        assert!((lighting.schedule_fraction(10.0) - 1.0).abs() < 1e-10);
        assert!((lighting.schedule_fraction(1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_daylighting_reduction() {
        let lighting = LightingModel::default();
        let power_no_daylight = lighting.lighting_power(10.0, 100.0, 0.0);
        let power_with_daylight = lighting.lighting_power(10.0, 100.0, 1000.0);
        assert!(power_with_daylight < power_no_daylight);
    }
}
