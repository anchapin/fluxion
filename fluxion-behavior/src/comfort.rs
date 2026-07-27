use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PmvComfortStatus {
    Comfortable,
    SlightlyWarm,
    SlightlyCool,
    Warm,
    Cool,
    Hot,
    Cold,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PmvComfort {
    pub ppd_threshold: f64,
}

impl Default for PmvComfort {
    fn default() -> Self {
        Self::new()
    }
}

impl PmvComfort {
    pub fn new() -> Self {
        Self { ppd_threshold: 10.0 }
    }

    pub fn with_ppd_threshold(mut self, threshold: f64) -> Self {
        self.ppd_threshold = threshold;
        self
    }

    pub fn calculate_pmv(
        &self,
        air_temp: f64,
        radiant_temp: f64,
        rel_humidity: f64,
        air_velocity: f64,
        metabolic_rate: f64,
        clothing_level: f64,
    ) -> f64 {
        let ta = air_temp;
        let tr = radiant_temp;
        let vel = air_velocity.max(0.0);

        let pmv = 0.303_f64 * (-0.5_f64).exp() * metabolic_rate
            + 0.0275 * (ta * 9.0 / 5.0 + 32.0 - 75.0)
            - 0.0014 * (ta * 9.0 / 5.0 + 32.0) * (rel_humidity * 100.0 - 50.0)
            - 0.00034 * (ta * 9.0 / 5.0 + 32.0) * vel.max(0.1)
            + 0.00028 * (tr * 9.0 / 5.0 + 32.0 - 65.0)
            - clothing_level * 0.5;

        pmv.max(-4.0).min(4.0)
    }

    pub fn calculate_ppd(&self, pmv: f64) -> f64 {
        let p = 1.0 + 0.5 * (4.0 - pmv.abs()).exp();
        100.0 * (1.0 - (-p * pmv * pmv).exp())
    }

    pub fn evaluate_status(&self, pmv: f64) -> PmvComfortStatus {
        let ppd = self.calculate_ppd(pmv);
        if ppd <= self.ppd_threshold {
            PmvComfortStatus::Comfortable
        } else if pmv > 0.5 && pmv <= 1.0 {
            PmvComfortStatus::SlightlyWarm
        } else if pmv >= -1.0 && pmv < -0.5 {
            PmvComfortStatus::SlightlyCool
        } else if pmv > 1.0 && pmv <= 2.0 {
            PmvComfortStatus::Warm
        } else if pmv >= -2.0 && pmv <= -1.0 {
            PmvComfortStatus::Cool
        } else if pmv > 2.0 {
            PmvComfortStatus::Hot
        } else {
            PmvComfortStatus::Cold
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptiveComfortStatus {
    Comfortable,
    NoAssignmentPossible,
    SlightlyWarm,
    SlightlyCool,
    Warm,
    Cool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveComfort {
    pub fix_temperature: Option<f64>,
    pub neutrality_temperature: Option<f64>,
}

impl Default for AdaptiveComfort {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveComfort {
    pub fn new() -> Self {
        Self {
            fix_temperature: None,
            neutrality_temperature: None,
        }
    }

    pub fn with_fix_temperature(mut self, temp: f64) -> Self {
        self.fix_temperature = Some(temp);
        self
    }

    pub fn calculate_running_mean(&self, daily_temps: &[f64], alpha: f64) -> f64 {
        if daily_temps.is_empty() {
            return 20.0;
        }
        let n = daily_temps.len();
        let mut rtm = daily_temps[0];
        for i in 1..n.min(7) {
            rtm = alpha * daily_temps[i] + (1.0 - alpha) * rtm;
        }
        rtm
    }

    pub fn calculate_comfort_band(&self, running_mean_temp: f64, category: u8) -> (f64, f64) {
        let (upper_offset, lower_offset) = match category {
            1 => (2.5, 2.5),
            2 => (3.5, 2.0),
            3 => (4.5, 1.5),
            _ => (3.5, 2.0),
        };
        let centre = 0.33 * running_mean_temp + 18.83;
        (centre + upper_offset, centre - lower_offset)
    }

    pub fn evaluate_status(
        &self,
        operative_temp: f64,
        running_mean_temp: f64,
        category: u8,
    ) -> AdaptiveComfortStatus {
        let (upper_limit, lower_limit) = self.calculate_comfort_band(running_mean_temp, category);
        if operative_temp >= lower_limit && operative_temp <= upper_limit {
            AdaptiveComfortStatus::Comfortable
        } else if operative_temp > upper_limit && operative_temp <= upper_limit + 2.5 {
            AdaptiveComfortStatus::SlightlyWarm
        } else if operative_temp < lower_limit && operative_temp >= lower_limit - 2.5 {
            AdaptiveComfortStatus::SlightlyCool
        } else if operative_temp > upper_limit + 2.5 {
            AdaptiveComfortStatus::Warm
        } else if operative_temp < lower_limit - 2.5 {
            AdaptiveComfortStatus::Cool
        } else {
            AdaptiveComfortStatus::NoAssignmentPossible
        }
    }
}
