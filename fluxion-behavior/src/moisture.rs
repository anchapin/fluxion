use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum ActivityLevel {
    Sleeping,
    #[default]
    SeatedQuiet,
    OfficeWork,
    LightActivity,
    Standing,
    Walking,
}

impl ActivityLevel {
    pub fn metabolic_rate_w(&self) -> f64 {
        match self {
            ActivityLevel::Sleeping => 80.0,
            ActivityLevel::SeatedQuiet => 100.0,
            ActivityLevel::OfficeWork => 120.0,
            ActivityLevel::LightActivity => 160.0,
            ActivityLevel::Standing => 140.0,
            ActivityLevel::Walking => 200.0,
        }
    }

    pub fn latent_heat_fraction(&self) -> f64 {
        match self {
            ActivityLevel::Sleeping => 0.5,
            ActivityLevel::SeatedQuiet => 0.4,
            ActivityLevel::OfficeWork => 0.3,
            ActivityLevel::LightActivity => 0.25,
            ActivityLevel::Standing => 0.35,
            ActivityLevel::Walking => 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoistureGeneration {
    pub metabolic_rate_w: f64,
    pub activity_level: ActivityLevel,
}

impl Default for MoistureGeneration {
    fn default() -> Self {
        Self::office()
    }
}

impl MoistureGeneration {
    pub fn office() -> Self {
        Self {
            metabolic_rate_w: 120.0,
            activity_level: ActivityLevel::OfficeWork,
        }
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_metabolic_rate(mut self, rate: f64) -> Self {
        self.metabolic_rate_w = rate;
        self
    }

    pub fn with_activity_level(mut self, level: ActivityLevel) -> Self {
        self.activity_level = level;
        self
    }

    pub fn moisture_generation_rate(&self, occupants: f64) -> f64 {
        let _sensible_heat = self.metabolic_rate_w * (1.0 - self.latent_heat_fraction());
        let latent_heat_fraction = self.latent_heat_fraction();

        if latent_heat_fraction <= 0.0 {
            return 0.0;
        }

        let latent_heat_per_person = self.metabolic_rate_w * latent_heat_fraction;
        latent_heat_per_person * occupants / 2.5e6
    }

    pub fn latent_heat_fraction(&self) -> f64 {
        self.activity_level.latent_heat_fraction()
    }

    pub fn latent_heat_gain(&self, occupants: f64) -> f64 {
        let moisture_rate = self.moisture_generation_rate(occupants);
        moisture_rate * 2.5e6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moisture_default() {
        let moisture = MoistureGeneration::default();
        assert!((moisture.metabolic_rate_w - 120.0).abs() < 1e-10);
    }

    #[test]
    fn test_moisture_generation_rate() {
        let moisture = MoistureGeneration::default();
        let rate = moisture.moisture_generation_rate(1.0);
        assert!(rate > 0.0);
    }

    #[test]
    fn test_latent_heat_fraction() {
        let moisture = MoistureGeneration::default();
        let fraction = moisture.latent_heat_fraction();
        assert!(fraction > 0.0 && fraction < 1.0);
    }
}
