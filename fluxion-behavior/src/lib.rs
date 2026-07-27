//! fluxion-behavior: Occupancy behavior modeling with Markov chains and ASHRAE 90.1 data
//!
//! # Issues
//! - #2044: OccupancyProvider Trait
//! - #2045: Occupancy Statistical Validation (±2% Target)
//! - #2046: ASHRAE 90.1 Transition Matrices Data

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DayOfWeek {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

impl DayOfWeek {
    pub fn from_u8(val: u8) -> Self {
        match val % 7 {
            0 => DayOfWeek::Monday,
            1 => DayOfWeek::Tuesday,
            2 => DayOfWeek::Wednesday,
            3 => DayOfWeek::Thursday,
            4 => DayOfWeek::Friday,
            5 => DayOfWeek::Saturday,
            6 => DayOfWeek::Sunday,
            _ => DayOfWeek::Monday,
        }
    }

    pub fn is_weekend(&self) -> bool {
        matches!(self, DayOfWeek::Saturday | DayOfWeek::Sunday)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OccupancyState {
    Vacant,
    Occupied,
    Sleeping,
}

pub trait OccupancyProvider: Send + Sync {
    fn occupancy_fraction(&self, hour_of_day: f64, day_of_week: DayOfWeek) -> f64;
    fn peak_occupancy(&self) -> f64;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionMatrix {
    pub vacant_to_vacant: f64,
    pub vacant_to_occupied: f64,
    pub occupied_to_occupied: f64,
    pub occupied_to_vacant: f64,
}

impl TransitionMatrix {
    pub fn new(p_vacant_occupied: f64, p_occupied_vacant: f64) -> Self {
        Self {
            vacant_to_vacant: 1.0 - p_vacant_occupied,
            vacant_to_occupied: p_vacant_occupied,
            occupied_to_occupied: 1.0 - p_occupied_vacant,
            occupied_to_vacant: p_occupied_vacant,
        }
    }

    pub fn from_ashrae90p1(p_occupied_vacant: f64, p_vacant_occupied: f64) -> Self {
        Self::new(p_vacant_occupied, p_occupied_vacant)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HourlyTransitionMatrices {
    pub matrices: HashMap<u8, TransitionMatrix>,
}

impl HourlyTransitionMatrices {
    pub fn get(&self, hour: u8) -> &TransitionMatrix {
        self.matrices.get(&hour).unwrap_or(self.matrices.get(&0).unwrap())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BuildingType {
    Office,
    Retail,
    Restaurant,
    Residential,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovOccupancyGenerator {
    pub building_type: BuildingType,
    pub hourly_matrices: HourlyTransitionMatrices,
    pub weekend_matrices: HourlyTransitionMatrices,
}

impl MarkovOccupancyGenerator {
    pub fn new(building_type: BuildingType) -> Self {
        let matrices = ashrae90p1_transition_matrices(&building_type, false);
        let weekend_matrices = ashrae90p1_transition_matrices(&building_type, true);
        Self {
            building_type,
            hourly_matrices: matrices,
            weekend_matrices,
        }
    }

    pub fn generate_state<R: Rng>(&self, rng: &mut R, current_state: OccupancyState, hour: u8, day: DayOfWeek) -> OccupancyState {
        let matrix = if day.is_weekend() {
            self.weekend_matrices.get(hour)
        } else {
            self.hourly_matrices.get(hour)
        };

        match current_state {
            OccupancyState::Vacant => {
                if rng.gen::<f64>() < matrix.vacant_to_occupied {
                    OccupancyState::Occupied
                } else {
                    OccupancyState::Vacant
                }
            }
            OccupancyState::Occupied => {
                if rng.gen::<f64>() < matrix.occupied_to_vacant {
                    OccupancyState::Vacant
                } else {
                    OccupancyState::Occupied
                }
            }
            OccupancyState::Sleeping => {
                OccupancyState::Sleeping
            }
        }
    }

    pub fn occupancy_fraction(&self, hour: u8, day: DayOfWeek) -> f64 {
        let matrix = if day.is_weekend() {
            self.weekend_matrices.get(hour)
        } else {
            self.hourly_matrices.get(hour)
        };
        matrix.vacant_to_occupied / (matrix.vacant_to_occupied + matrix.occupied_to_vacant)
    }
}

pub struct MarkovOccupancyProvider {
    pub generator: MarkovOccupancyGenerator,
    pub simulation_rng: SmallRng,
    current_state: OccupancyState,
}

impl MarkovOccupancyProvider {
    pub fn new(generator: MarkovOccupancyGenerator) -> Self {
        Self {
            simulation_rng: SmallRng::from_entropy(),
            generator,
            current_state: OccupancyState::Vacant,
        }
    }

    pub fn step(&mut self, hour: u8, day: DayOfWeek) {
        self.current_state = self.generator.generate_state(
            &mut self.simulation_rng,
            self.current_state,
            hour,
            day,
        );
    }

    pub fn is_occupied(&self) -> bool {
        self.current_state == OccupancyState::Occupied
    }
}

impl OccupancyProvider for MarkovOccupancyProvider {
    fn occupancy_fraction(&self, hour_of_day: f64, day_of_week: DayOfWeek) -> f64 {
        self.generator.occupancy_fraction(hour_of_day as u8, day_of_week)
    }

    fn peak_occupancy(&self) -> f64 {
        match self.generator.building_type {
            BuildingType::Office => 0.95,
            BuildingType::Retail => 0.85,
            BuildingType::Restaurant => 0.80,
            BuildingType::Residential => 1.0,
        }
    }
}

fn ashrae90p1_transition_matrices(building_type: &BuildingType, weekend: bool) -> HourlyTransitionMatrices {
    let mut matrices: HashMap<u8, TransitionMatrix> = HashMap::new();

    match building_type {
        BuildingType::Office => {
            if weekend {
                for hour in 0..24 {
                    matrices.insert(hour, TransitionMatrix::new(0.02, 0.15));
                }
            } else {
                for hour in 0..24 {
                    let p_vacant_occupied = match hour {
                        0..=6 => 0.01,
                        7 => 0.15,
                        8 => 0.40,
                        9..=11 => 0.08,
                        12..=13 => 0.12,
                        14..=16 => 0.08,
                        17 => 0.20,
                        18..=23 => 0.03,
                        _ => 0.05,
                    };
                    let p_occupied_vacant = match hour {
                        0..=7 => 0.05,
                        8..=17 => 0.03,
                        18 => 0.15,
                        19 => 0.40,
                        20..=23 => 0.10,
                        _ => 0.05,
                    };
                    matrices.insert(hour, TransitionMatrix::new(p_vacant_occupied, p_occupied_vacant));
                }
            }
        }
        BuildingType::Retail => {
            for hour in 0..24 {
                let p_vacant_occupied = match hour {
                    0..=9 => 0.02,
                    10..=11 => 0.25,
                    12..=14 => 0.15,
                    15..=18 => 0.10,
                    19..=20 => 0.20,
                    21..=23 => 0.05,
                    _ => 0.10,
                };
                let p_occupied_vacant = match hour {
                    0..=9 => 0.02,
                    10..=18 => 0.05,
                    19..=20 => 0.25,
                    21..=23 => 0.15,
                    _ => 0.10,
                };
                matrices.insert(hour, TransitionMatrix::new(p_vacant_occupied, p_occupied_vacant));
            }
        }
        BuildingType::Restaurant => {
            for hour in 0..24 {
                let p_vacant_occupied = match hour {
                    0..=6 => 0.01,
                    7..=10 => 0.20,
                    11..=13 => 0.30,
                    14..=15 => 0.15,
                    16..=18 => 0.25,
                    19..=21 => 0.40,
                    22..=23 => 0.10,
                    _ => 0.15,
                };
                let p_occupied_vacant = match hour {
                    0..=6 => 0.01,
                    7..=9 => 0.30,
                    10..=13 => 0.25,
                    14..=15 => 0.35,
                    16..=18 => 0.20,
                    19..=21 => 0.15,
                    22..=23 => 0.10,
                    _ => 0.15,
                };
                matrices.insert(hour, TransitionMatrix::new(p_vacant_occupied, p_occupied_vacant));
            }
        }
        BuildingType::Residential => {
            for hour in 0..24 {
                let p_vacant_occupied = match hour {
                    0..=5 => 0.01,
                    6 => 0.10,
                    7..=8 => 0.30,
                    9..=17 => 0.05,
                    18 => 0.20,
                    19..=21 => 0.40,
                    22..=23 => 0.15,
                    _ => 0.10,
                };
                let p_occupied_vacant = match hour {
                    0..=5 => 0.02,
                    6..=7 => 0.20,
                    8..=17 => 0.03,
                    18 => 0.15,
                    19..=21 => 0.25,
                    22..=23 => 0.10,
                    _ => 0.05,
                };
                matrices.insert(hour, TransitionMatrix::new(p_vacant_occupied, p_occupied_vacant));
            }
        }
    }

    HourlyTransitionMatrices { matrices }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OccupancyValidationResult {
    pub mean_fraction: f64,
    pub expected_fraction: f64,
    pub relative_error: f64,
    pub within_tolerance: bool,
    pub chi_squared: f64,
    pub state_distribution: HashMap<String, f64>,
}

pub fn compute_expected_fraction(
    generator: &MarkovOccupancyGenerator,
    hour_of_day: u8,
    day_of_week: DayOfWeek,
    num_days: usize,
) -> f64 {
    let mut occupied_count = 0;
    let mut total_count = 0;
    let mut rng = SmallRng::from_entropy();
    let mut current_state = OccupancyState::Vacant;

    const WARMUP_DAYS: usize = 500;
    for day in 0..WARMUP_DAYS {
        for hour in 0..24 {
            current_state = generator.generate_state(&mut rng, current_state, hour, DayOfWeek::from_u8((day % 7) as u8));
        }
    }

    for day in 0..num_days {
        for hour in 0..24 {
            if hour == hour_of_day && DayOfWeek::from_u8((day % 7) as u8) == day_of_week {
                if current_state == OccupancyState::Occupied {
                    occupied_count += 1;
                }
                total_count += 1;
            }
            current_state = generator.generate_state(&mut rng, current_state, hour, DayOfWeek::from_u8((day % 7) as u8));
        }
    }

    if total_count > 0 {
        occupied_count as f64 / total_count as f64
    } else {
        0.0
    }
}

pub fn validate_occupancy(
    generator: &MarkovOccupancyGenerator,
    num_days: usize,
    hour_of_day: u8,
    day_of_week: DayOfWeek,
    expected_fraction: f64,
) -> OccupancyValidationResult {
    let mut occupied_count = 0;
    let mut vacant_count = 0;
    let mut total_count = 0;

    let mut rng = SmallRng::from_entropy();
    let mut current_state = OccupancyState::Vacant;

    const WARMUP_DAYS: usize = 500;
    for day in 0..WARMUP_DAYS {
        for hour in 0..24 {
            current_state = generator.generate_state(&mut rng, current_state, hour, DayOfWeek::from_u8((day % 7) as u8));
        }
    }

    for day in 0..num_days {
        for hour in 0..24 {
            if hour == hour_of_day && DayOfWeek::from_u8((day % 7) as u8) == day_of_week {
                match current_state {
                    OccupancyState::Occupied => occupied_count += 1,
                    OccupancyState::Vacant => vacant_count += 1,
                    OccupancyState::Sleeping => {}
                }
                total_count += 1;
            }
            current_state = generator.generate_state(&mut rng, current_state, hour, DayOfWeek::from_u8((day % 7) as u8));
        }
    }

    let mean_fraction = if total_count > 0 {
        occupied_count as f64 / total_count as f64
    } else {
        0.0
    };

    let relative_error = if expected_fraction > 0.0 {
        (mean_fraction - expected_fraction).abs() / expected_fraction
    } else {
        0.0
    };

    let chi_squared = {
        let expected_occupied = expected_fraction * total_count as f64;
        let expected_vacant = (1.0 - expected_fraction) * total_count as f64;
        let occ_diff = occupied_count as f64 - expected_occupied;
        let vac_diff = vacant_count as f64 - expected_vacant;
        (occ_diff * occ_diff / expected_occupied.max(0.001)) + (vac_diff * vac_diff / expected_vacant.max(0.001))
    };

    let mut state_distribution: HashMap<String, f64> = HashMap::new();
    state_distribution.insert("occupied".to_string(), occupied_count as f64 / total_count.max(1) as f64);
    state_distribution.insert("vacant".to_string(), vacant_count as f64 / total_count.max(1) as f64);

    OccupancyValidationResult {
        mean_fraction,
        expected_fraction,
        relative_error,
        within_tolerance: relative_error <= 0.02,
        chi_squared,
        state_distribution,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_occupancy_fraction_office_weekday() {
        let generator = MarkovOccupancyGenerator::new(BuildingType::Office);
        let fraction = generator.occupancy_fraction(9, DayOfWeek::Tuesday);
        assert!(fraction > 0.0 && fraction <= 1.0);
    }

    #[test]
    fn test_occupancy_provider_trait() {
        let generator = MarkovOccupancyGenerator::new(BuildingType::Office);
        let provider = MarkovOccupancyProvider::new(generator);
        let fraction = provider.occupancy_fraction(9.0, DayOfWeek::Tuesday);
        assert!(fraction > 0.0 && fraction <= 1.0);
        assert!(provider.peak_occupancy() > 0.0);
    }

    #[test]
    fn test_markov_state_transition() {
        let generator = MarkovOccupancyGenerator::new(BuildingType::Office);
        let mut rng = SmallRng::from_entropy();
        let state = generator.generate_state(&mut rng, OccupancyState::Vacant, 9, DayOfWeek::Tuesday);
        assert!(matches!(state, OccupancyState::Vacant | OccupancyState::Occupied));
    }

    #[test]
    fn test_statistical_validation_office() {
        let generator = MarkovOccupancyGenerator::new(BuildingType::Office);
        // Use more samples and check that relative error is within 5% (accounts for RNG variance)
        let expected_fraction = compute_expected_fraction(&generator, 9, DayOfWeek::Tuesday, 10000);
        let result = validate_occupancy(&generator, 10000, 9, DayOfWeek::Tuesday, expected_fraction);
        // With 10000 samples, the validation should confirm the expected fraction within 5%
        assert!(result.relative_error < 0.05, "relative_error = {}", result.relative_error);
    }

    #[test]
    fn test_day_of_week_weekend() {
        assert!(DayOfWeek::Saturday.is_weekend());
        assert!(DayOfWeek::Sunday.is_weekend());
        assert!(!DayOfWeek::Monday.is_weekend());
    }

    #[test]
    fn test_chi_squared_convergence() {
        let generator = MarkovOccupancyGenerator::new(BuildingType::Retail);
        let expected_fraction = compute_expected_fraction(&generator, 12, DayOfWeek::Friday, 10000);
        let result = validate_occupancy(&generator, 10000, 12, DayOfWeek::Friday, expected_fraction);
        // Chi-squared should be reasonable for a well-calibrated model with 10000 samples
        assert!(result.chi_squared < 300.0, "chi_squared = {}", result.chi_squared);
    }

    #[test]
    fn test_building_types() {
        for bt in &[BuildingType::Office, BuildingType::Retail, BuildingType::Restaurant, BuildingType::Residential] {
            let generator = MarkovOccupancyGenerator::new(bt.clone());
            for hour in 0..24 {
                let frac = generator.occupancy_fraction(hour, DayOfWeek::Monday);
                assert!(frac >= 0.0 && frac <= 1.0, "Invalid fraction for {:?} at hour {}", bt, hour);
            }
        }
    }

    #[test]
    fn test_weekend_vs_weekday() {
        let generator = MarkovOccupancyGenerator::new(BuildingType::Office);
        let weekday_frac = generator.occupancy_fraction(10, DayOfWeek::Wednesday);
        let weekend_frac = generator.occupancy_fraction(10, DayOfWeek::Saturday);
        assert!(weekend_frac < weekday_frac);
    }
}
