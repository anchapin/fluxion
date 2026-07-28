use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct MockPlugLoad {
    pub power_w: f64,
}

#[allow(dead_code)]
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

#[derive(Debug, Clone)]
pub struct MockPlugLoadGenerator {
    base_watts: f64,
    diurnal_amplitude: f64,
    noise_std: f64,
    rng: SmallRng,
}

impl MockPlugLoadGenerator {
    pub fn new() -> Self {
        Self {
            base_watts: 200.0,
            diurnal_amplitude: 150.0,
            noise_std: 30.0,
            rng: SmallRng::from_entropy(),
        }
    }

    pub fn with_params(base_watts: f64, diurnal_amplitude: f64, noise_std: f64) -> Self {
        Self {
            base_watts,
            diurnal_amplitude,
            noise_std,
            rng: SmallRng::from_entropy(),
        }
    }

    pub fn generate_24hr(&mut self, seed: u64) -> Vec<f64> {
        self.rng = SmallRng::seed_from_u64(seed);
        (0..24)
            .map(|hour| {
                let diurnal = self.diurnal_amplitude * Self::diurnal_factor(hour);
                let normal = Normal::new(0.0, self.noise_std).unwrap();
                let noise = normal.sample(&mut self.rng);
                (self.base_watts + diurnal + noise).max(0.0)
            })
            .collect()
    }

    pub fn generate_at_hour(&mut self, hour: u8, seed: u64) -> f64 {
        self.rng = SmallRng::seed_from_u64(seed);
        let diurnal = self.diurnal_amplitude * Self::diurnal_factor(hour);
        let normal = Normal::new(0.0, self.noise_std).unwrap();
        let noise = normal.sample(&mut self.rng);
        (self.base_watts + diurnal + noise).max(0.0)
    }

    fn diurnal_factor(hour: u8) -> f64 {
        let peak_hour = 14.0;
        let phase = (hour as f64 - peak_hour) * PI / 12.0;
        phase.cos()
    }

    pub fn base_watts(&self) -> f64 {
        self.base_watts
    }

    pub fn diurnal_amplitude(&self) -> f64 {
        self.diurnal_amplitude
    }

    pub fn noise_std(&self) -> f64 {
        self.noise_std
    }
}

impl Default for MockPlugLoadGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diurnal_factor_2pm_peak() {
        let peak = MockPlugLoadGenerator::diurnal_factor(14);
        assert!((peak - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diurnal_factor_4am() {
        let val = MockPlugLoadGenerator::diurnal_factor(4);
        let expected = ((4.0 - 14.0) * PI / 12.0).cos();
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_diurnal_factor_symmetry() {
        let peak = MockPlugLoadGenerator::diurnal_factor(14);
        assert!((peak - 1.0).abs() < 1e-10);
        let val_at_4 = MockPlugLoadGenerator::diurnal_factor(4);
        let val_at_24 = MockPlugLoadGenerator::diurnal_factor(24);
        assert!((val_at_4 - val_at_24).abs() < 1e-10);
    }

    #[test]
    fn test_generate_24hr_length() {
        let mut generator = MockPlugLoadGenerator::new();
        let values = generator.generate_24hr(42);
        assert_eq!(values.len(), 24);
    }

    #[test]
    fn test_generate_24hr_positive() {
        let mut generator = MockPlugLoadGenerator::new();
        let values = generator.generate_24hr(42);
        assert!(values.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_generate_24hr_reproducible() {
        let mut generator1 = MockPlugLoadGenerator::new();
        let mut generator2 = MockPlugLoadGenerator::new();
        let values1 = generator1.generate_24hr(12345);
        let values2 = generator2.generate_24hr(12345);
        assert_eq!(values1, values2);
    }

    #[test]
    fn test_generate_24hr_different_seeds_different_results() {
        let mut generator1 = MockPlugLoadGenerator::new();
        let mut generator2 = MockPlugLoadGenerator::new();
        let values1 = generator1.generate_24hr(111);
        let values2 = generator2.generate_24hr(222);
        assert_ne!(values1, values2);
    }

    #[test]
    fn test_generate_at_hour() {
        let mut generator = MockPlugLoadGenerator::new();
        let power = generator.generate_at_hour(14, 42);
        assert!(power >= 0.0);
    }

    #[test]
    fn test_with_params() {
        let generator = MockPlugLoadGenerator::with_params(300.0, 100.0, 20.0);
        assert_eq!(generator.base_watts(), 300.0);
        assert_eq!(generator.diurnal_amplitude(), 100.0);
        assert_eq!(generator.noise_std(), 20.0);
    }
}
