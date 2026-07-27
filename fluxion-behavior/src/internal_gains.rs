use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MetabolicRate {
    Sleeping,
    SeatedQuiet,
    OfficeWork,
    LightActivity,
    Standing,
    Walking,
}

impl MetabolicRate {
    pub fn watts_per_kg(&self) -> f64 {
        match self {
            MetabolicRate::Sleeping => 1.0,
            MetabolicRate::SeatedQuiet => 1.2,
            MetabolicRate::OfficeWork => 1.4,
            MetabolicRate::LightActivity => 2.0,
            MetabolicRate::Standing => 1.7,
            MetabolicRate::Walking => 2.8,
        }
    }

    pub fn met(&self) -> f64 {
        self.watts_per_kg()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Co2Generation {
    pub co2_generation_rate_per_person: f64,
}

impl Default for Co2Generation {
    fn default() -> Self {
        Self::new()
    }
}

impl Co2Generation {
    pub fn new() -> Self {
        Self {
            co2_generation_rate_per_person: 0.005,
        }
    }

    pub fn with_rate(mut self, rate: f64) -> Self {
        self.co2_generation_rate_per_person = rate;
        self
    }

    pub fn calculate_co2_generation(
        &self,
        occupant_count: f64,
        metabolic_rate: MetabolicRate,
    ) -> f64 {
        let activity_factor = match metabolic_rate {
            MetabolicRate::Sleeping => 0.8,
            MetabolicRate::SeatedQuiet => 1.0,
            MetabolicRate::OfficeWork => 1.2,
            MetabolicRate::LightActivity => 1.6,
            MetabolicRate::Standing => 1.4,
            MetabolicRate::Walking => 2.0,
        };
        occupant_count * self.co2_generation_rate_per_person * activity_factor
    }
}

impl Default for MetabolicRate {
    fn default() -> Self {
        MetabolicRate::SeatedQuiet
    }
}
