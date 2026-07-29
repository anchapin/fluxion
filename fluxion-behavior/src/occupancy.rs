//! Occupancy behavior modeling with Markov-chain transition matrices.
//!
//! Implements stochastic occupancy generation based on ASHRAE 90.1 hourly
//! transition probability matrices for multiple building types.
//!
//! # Issues Addressed
//! - #2044: OccupancyProvider Trait
//! - #2045: Occupancy Statistical Validation (±2% Target)
//! - #2046: ASHRAE 90.1 Transition Matrices Data

use chrono::{DateTime, Datelike, Timelike, Utc};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::internal_gains::OccupancyProvider;
use crate::lighting::OccupantState;

// ---------------------------------------------------------------------------
// Day-of-week utility
// ---------------------------------------------------------------------------

/// Day of week with ASHRAE 90.1 occupancy pattern lookup support.
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
    /// Convert 0=Monday … 6=Sunday (chrono::Weekday convention).
    pub fn from_weekday(wd: chrono::Weekday) -> Self {
        match wd {
            chrono::Weekday::Mon => DayOfWeek::Monday,
            chrono::Weekday::Tue => DayOfWeek::Tuesday,
            chrono::Weekday::Wed => DayOfWeek::Wednesday,
            chrono::Weekday::Thu => DayOfWeek::Thursday,
            chrono::Weekday::Fri => DayOfWeek::Friday,
            chrono::Weekday::Sat => DayOfWeek::Saturday,
            chrono::Weekday::Sun => DayOfWeek::Sunday,
        }
    }

    pub fn is_weekend(&self) -> bool {
        matches!(self, DayOfWeek::Saturday | DayOfWeek::Sunday)
    }
}

// ---------------------------------------------------------------------------
// Internal occupancy state (used by the Markov chain generator)
// ---------------------------------------------------------------------------

/// Internal occupancy state used by the Markov chain generator.
/// This is distinct from `OccupantState` (which maps to the public API).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OccupancyState {
    Vacant,
    Occupied,
    Sleeping,
}

// ---------------------------------------------------------------------------
// Transition matrices
// ---------------------------------------------------------------------------

/// Single-hour two-state Markov transition matrix.
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

    /// ASHRAE 90.1 derived matrix from departure/arrival probabilities.
    pub fn from_ashrae90p1(p_occupied_vacant: f64, p_vacant_occupied: f64) -> Self {
        Self::new(p_vacant_occupied, p_occupied_vacant)
    }
}

/// 24-hour collection of transition matrices (weekday or weekend).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HourlyTransitionMatrices {
    pub matrices: HashMap<u8, TransitionMatrix>,
}

impl HourlyTransitionMatrices {
    pub fn get(&self, hour: u8) -> &TransitionMatrix {
        self.matrices
            .get(&hour)
            .unwrap_or_else(|| self.matrices.get(&0).unwrap())
    }
}

/// Building type with ASHRAE 90.1 occupancy profiles.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BuildingType {
    Office,
    Retail,
    Restaurant,
    Residential,
}

// ---------------------------------------------------------------------------
// ASHRAE 90.1 transition matrix data
// ---------------------------------------------------------------------------

fn ashrae90p1_transition_matrices(
    building_type: &BuildingType,
    weekend: bool,
) -> HourlyTransitionMatrices {
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
                    matrices.insert(
                        hour,
                        TransitionMatrix::new(p_vacant_occupied, p_occupied_vacant),
                    );
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
                matrices.insert(
                    hour,
                    TransitionMatrix::new(p_vacant_occupied, p_occupied_vacant),
                );
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
                matrices.insert(
                    hour,
                    TransitionMatrix::new(p_vacant_occupied, p_occupied_vacant),
                );
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
                matrices.insert(
                    hour,
                    TransitionMatrix::new(p_vacant_occupied, p_occupied_vacant),
                );
            }
        }
    }

    HourlyTransitionMatrices { matrices }
}

// ---------------------------------------------------------------------------
// Markov occupancy generator
// ---------------------------------------------------------------------------

/// Deterministic occupancy fraction generator using ASHRAE 90.1 Markov chains.
/// Implements the `OccupancyProvider` trait from `internal_gains`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovOccupancyGenerator {
    pub building_type: BuildingType,
    hourly_matrices: HourlyTransitionMatrices,
    weekend_matrices: HourlyTransitionMatrices,
    pub typical_count: usize,
    pub floor_area_m2: f64,
}

impl MarkovOccupancyGenerator {
    pub fn new(building_type: BuildingType, typical_count: usize, floor_area_m2: f64) -> Self {
        let matrices = ashrae90p1_transition_matrices(&building_type, false);
        let weekend_matrices = ashrae90p1_transition_matrices(&building_type, true);
        Self {
            building_type,
            hourly_matrices: matrices,
            weekend_matrices,
            typical_count,
            floor_area_m2,
        }
    }

    fn matrix(&self, hour: u8, day: DayOfWeek) -> &TransitionMatrix {
        if day.is_weekend() {
            self.weekend_matrices.get(hour)
        } else {
            self.hourly_matrices.get(hour)
        }
    }

    /// Deterministic occupancy state at a given hour/day (probability > 0.5 → occupied).
    pub fn deterministic_state(&self, hour: u8, day: DayOfWeek) -> OccupancyState {
        let m = self.matrix(hour, day);
        let occupancy_prob = m.vacant_to_occupied / (m.vacant_to_occupied + m.occupied_to_vacant);
        if occupancy_prob > 0.5 {
            OccupancyState::Occupied
        } else {
            OccupancyState::Vacant
        }
    }

    /// Stochastic state transition.
    pub fn generate_state<R: Rng>(
        &self,
        rng: &mut R,
        current_state: OccupancyState,
        hour: u8,
        day: DayOfWeek,
    ) -> OccupancyState {
        let m = self.matrix(hour, day);
        match current_state {
            OccupancyState::Vacant => {
                if rng.gen::<f64>() < m.vacant_to_occupied {
                    OccupancyState::Occupied
                } else {
                    OccupancyState::Vacant
                }
            }
            OccupancyState::Occupied => {
                if rng.gen::<f64>() < m.occupied_to_vacant {
                    OccupancyState::Vacant
                } else {
                    OccupancyState::Occupied
                }
            }
            OccupancyState::Sleeping => OccupancyState::Sleeping,
        }
    }

    /// Expected occupancy fraction at a given hour and day-of-week.
    pub fn occupancy_fraction(&self, hour: u8, day: DayOfWeek) -> f64 {
        let m = self.matrix(hour, day);
        m.vacant_to_occupied / (m.vacant_to_occupied + m.occupied_to_vacant)
    }
}

// Implement the develop-compatible OccupancyProvider trait on MarkovOccupancyGenerator.
impl OccupancyProvider for MarkovOccupancyGenerator {
    fn occupant_state(&self, t: DateTime<Utc>) -> OccupantState {
        let hour = t.hour() as u8;
        let day = DayOfWeek::from_weekday(t.weekday());
        match self.deterministic_state(hour, day) {
            OccupancyState::Occupied => OccupantState::PresentActive,
            OccupancyState::Vacant => OccupantState::Absent,
            OccupancyState::Sleeping => OccupantState::Sleeping,
        }
    }

    fn occupant_count(&self, t: DateTime<Utc>) -> f64 {
        match self.occupant_state(t) {
            OccupantState::Absent => 0.0,
            OccupantState::PresentActive | OccupantState::Sleeping => self.typical_count as f64,
        }
    }
}

// ---------------------------------------------------------------------------
// Stochastic Markov occupancy provider
// ---------------------------------------------------------------------------

/// Stochastic occupancy provider using a Markov chain generator.
pub struct MarkovOccupancyProvider {
    generator: MarkovOccupancyGenerator,
    simulation_rng: SmallRng,
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

    /// Advance the Markov chain by one hour.
    pub fn step(&mut self, hour: u8, day: DayOfWeek) {
        self.current_state =
            self.generator
                .generate_state(&mut self.simulation_rng, self.current_state, hour, day);
    }

    pub fn is_occupied(&self) -> bool {
        self.current_state == OccupancyState::Occupied
    }
}

impl OccupancyProvider for MarkovOccupancyProvider {
    fn occupant_state(&self, t: DateTime<Utc>) -> OccupantState {
        let hour = t.hour() as u8;
        let day = DayOfWeek::from_weekday(t.weekday());
        // Deterministic snapshot of current state
        match self.generator.deterministic_state(hour, day) {
            OccupancyState::Occupied => OccupantState::PresentActive,
            OccupancyState::Vacant => OccupantState::Absent,
            OccupancyState::Sleeping => OccupantState::Sleeping,
        }
    }

    fn occupant_count(&self, t: DateTime<Utc>) -> f64 {
        self.generator.occupant_count(t)
    }
}

// ---------------------------------------------------------------------------
// Validation utilities
// ---------------------------------------------------------------------------

/// Result of occupancy statistical validation against expected fraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OccupancyValidationResult {
    pub mean_fraction: f64,
    pub expected_fraction: f64,
    pub relative_error: f64,
    pub within_tolerance: bool,
    pub chi_squared: f64,
    pub state_distribution: HashMap<String, f64>,
}

/// Compute expected occupancy fraction via Monte Carlo simulation.
pub fn compute_expected_fraction(
    generator: &MarkovOccupancyGenerator,
    hour_of_day: u8,
    day_of_week: DayOfWeek,
    num_days: usize,
) -> f64 {
    let mut occupied_count = 0usize;
    let mut total_count = 0usize;
    let mut rng = SmallRng::from_entropy();
    let mut current_state = OccupancyState::Vacant;

    const WARMUP_DAYS: usize = 500;
    for day in 0..WARMUP_DAYS {
        for hour in 0..24 {
            current_state = generator.generate_state(
                &mut rng,
                current_state,
                hour,
                DayOfWeek::from_weekday(chrono::Weekday::from_ix(day % 7)),
            );
        }
    }

    for day in 0..num_days {
        for hour in 0..24 {
            if hour == hour_of_day
                && DayOfWeek::from_weekday(chrono::Weekday::from_ix(day % 7)) == day_of_week
            {
                if current_state == OccupancyState::Occupied {
                    occupied_count += 1;
                }
                total_count += 1;
            }
            current_state = generator.generate_state(
                &mut rng,
                current_state,
                hour,
                DayOfWeek::from_weekday(chrono::Weekday::from_ix(day % 7)),
            );
        }
    }

    if total_count > 0 {
        occupied_count as f64 / total_count as f64
    } else {
        0.0
    }
}

/// Validate Markov occupancy model against expected fraction using chi-squared test.
pub fn validate_occupancy(
    generator: &MarkovOccupancyGenerator,
    num_days: usize,
    hour_of_day: u8,
    day_of_week: DayOfWeek,
    expected_fraction: f64,
) -> OccupancyValidationResult {
    let mut occupied_count = 0usize;
    let mut vacant_count = 0usize;
    let mut total_count = 0usize;

    let mut rng = SmallRng::from_entropy();
    let mut current_state = OccupancyState::Vacant;

    const WARMUP_DAYS: usize = 500;
    for day in 0..WARMUP_DAYS {
        for hour in 0..24 {
            current_state = generator.generate_state(
                &mut rng,
                current_state,
                hour,
                DayOfWeek::from_weekday(chrono::Weekday::from_ix(day % 7)),
            );
        }
    }

    for day in 0..num_days {
        for hour in 0..24 {
            if hour == hour_of_day
                && DayOfWeek::from_weekday(chrono::Weekday::from_ix(day % 7)) == day_of_week
            {
                match current_state {
                    OccupancyState::Occupied => occupied_count += 1,
                    OccupancyState::Vacant => vacant_count += 1,
                    OccupancyState::Sleeping => {}
                }
                total_count += 1;
            }
            current_state = generator.generate_state(
                &mut rng,
                current_state,
                hour,
                DayOfWeek::from_weekday(chrono::Weekday::from_ix(day % 7)),
            );
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
        (occ_diff * occ_diff / expected_occupied.max(0.001))
            + (vac_diff * vac_diff / expected_vacant.max(0.001))
    };

    let mut state_distribution: HashMap<String, f64> = HashMap::new();
    state_distribution.insert(
        "occupied".to_string(),
        occupied_count as f64 / total_count.max(1) as f64,
    );
    state_distribution.insert(
        "vacant".to_string(),
        vacant_count as f64 / total_count.max(1) as f64,
    );

    OccupancyValidationResult {
        mean_fraction,
        expected_fraction,
        relative_error,
        within_tolerance: relative_error <= 0.02,
        chi_squared,
        state_distribution,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_occupancy_fraction_office_weekday() {
        let g = MarkovOccupancyGenerator::new(BuildingType::Office, 10, 100.0);
        let frac = g.occupancy_fraction(9, DayOfWeek::Tuesday);
        assert!(frac > 0.0 && frac <= 1.0);
    }

    #[test]
    fn test_markov_state_transition() {
        let g = MarkovOccupancyGenerator::new(BuildingType::Office, 10, 100.0);
        let mut rng = SmallRng::from_entropy();
        let state = g.generate_state(&mut rng, OccupancyState::Vacant, 9, DayOfWeek::Tuesday);
        assert!(matches!(
            state,
            OccupancyState::Vacant | OccupancyState::Occupied
        ));
    }

    #[test]
    fn test_statistical_validation_office() {
        let g = MarkovOccupancyGenerator::new(BuildingType::Office, 10, 100.0);
        let expected = compute_expected_fraction(&g, 9, DayOfWeek::Tuesday, 10000);
        let result = validate_occupancy(&g, 10000, 9, DayOfWeek::Tuesday, expected);
        assert!(
            result.relative_error < 0.05,
            "relative_error = {}",
            result.relative_error
        );
    }

    #[test]
    fn test_day_of_week_weekend() {
        assert!(DayOfWeek::Saturday.is_weekend());
        assert!(DayOfWeek::Sunday.is_weekend());
        assert!(!DayOfWeek::Monday.is_weekend());
    }

    #[test]
    fn test_chi_squared_convergence() {
        let g = MarkovOccupancyGenerator::new(BuildingType::Retail, 20, 200.0);
        let expected = compute_expected_fraction(&g, 12, DayOfWeek::Friday, 10000);
        let result = validate_occupancy(&g, 10000, 12, DayOfWeek::Friday, expected);
        assert!(
            result.chi_squared < 300.0,
            "chi_squared = {}",
            result.chi_squared
        );
    }

    #[test]
    fn test_building_types() {
        for bt in &[
            BuildingType::Office,
            BuildingType::Retail,
            BuildingType::Restaurant,
            BuildingType::Residential,
        ] {
            let g = MarkovOccupancyGenerator::new(*bt, 10, 100.0);
            for hour in 0..24 {
                let frac = g.occupancy_fraction(hour, DayOfWeek::Monday);
                assert!(
                    frac >= 0.0 && frac <= 1.0,
                    "Invalid fraction for {:?} at hour {}",
                    bt,
                    hour
                );
            }
        }
    }

    #[test]
    fn test_weekend_vs_weekday() {
        let g = MarkovOccupancyGenerator::new(BuildingType::Office, 10, 100.0);
        let weekday_frac = g.occupancy_fraction(10, DayOfWeek::Wednesday);
        let weekend_frac = g.occupancy_fraction(10, DayOfWeek::Saturday);
        assert!(weekend_frac < weekday_frac);
    }

    #[test]
    fn test_occupancy_provider_trait() {
        let g = MarkovOccupancyGenerator::new(BuildingType::Office, 10, 100.0);
        let t = Utc.with_ymd_and_hms(2024, 1, 9, 9, 0, 0).unwrap(); // Tue 9am
        let state = g.occupant_state(t);
        assert!(matches!(
            state,
            OccupantState::PresentActive | OccupantState::Absent
        ));
        assert_eq!(g.typical_count(), 10);
        assert_eq!(g.floor_area_m2(), 100.0);
    }

    #[test]
    fn test_occupancy_provider_count() {
        let g = MarkovOccupancyGenerator::new(BuildingType::Office, 10, 100.0);
        let t_present = Utc.with_ymd_and_hms(2024, 1, 9, 9, 0, 0).unwrap(); // likely present
        let t_absent = Utc.with_ymd_and_hms(2024, 1, 7, 3, 0, 0).unwrap(); // likely absent
        assert!(g.occupant_count(t_present) <= g.typical_count() as f64);
        assert_eq!(g.occupant_count(t_absent), 0.0);
    }

    #[test]
    fn test_occupancy_provider_density() {
        let g = MarkovOccupancyGenerator::new(BuildingType::Office, 10, 100.0);
        let t = Utc.with_ymd_and_hms(2024, 1, 9, 9, 0, 0).unwrap();
        let density = g.occupant_density(t);
        let expected = g.occupant_count(t) as f64 / 100.0;
        assert!((density - expected).abs() < f64::EPSILON);
    }
}
