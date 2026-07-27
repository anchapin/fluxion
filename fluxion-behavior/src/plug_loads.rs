use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MockPlugLoad {
    pub power_w: f64,
}

impl MockPlugLoad {
    pub fn new(power_w: f64) -> Self {
        Self { power_w }
    }

    pub fn total_load(&self) -> f64 {
        self.power_w
    }

    pub fn radiative_gain(&self, _hour: f64, _zone_area: f64, _occupancy_fraction: f64) -> f64 {
        self.power_w * 0.6
    }

    pub fn convective_gain(&self, _hour: f64, _zone_area: f64, _occupancy_fraction: f64) -> f64 {
        self.power_w * 0.4
    }
}

impl Default for MockPlugLoad {
    fn default() -> Self {
        Self::new(50.0)
    }
}
