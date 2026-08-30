use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OccupancyState {
    Vacant,
    Occupied,
}

impl Default for OccupancyState {
    fn default() -> Self {
        OccupancyState::Vacant
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovOccupancyGenerator {
    pub transition_matrix: Vec<Vec<f64>>,
    pub initial_state: OccupancyState,
}

impl MarkovOccupancyGenerator {
    pub fn new(transition_matrix: Vec<Vec<f64>>, initial_state: OccupancyState) -> Self {
        Self {
            transition_matrix,
            initial_state,
        }
    }

    pub fn from_ashrae90_1() -> Self {
        let p_occupied_given_vacant = 0.007;
        let p_vacant_given_occupied = 0.020;
        let transition_matrix = vec![
            vec![1.0 - p_occupied_given_vacant, p_occupied_given_vacant],
            vec![p_vacant_given_occupied, 1.0 - p_vacant_given_occupied],
        ];
        Self {
            transition_matrix,
            initial_state: OccupancyState::Vacant,
        }
    }

    pub fn next_state(&self, current: OccupancyState, rng: &mut impl Rng) -> OccupancyState {
        let row = match current {
            OccupancyState::Vacant => 0,
            OccupancyState::Occupied => 1,
        };
        let probabilities = &self.transition_matrix[row];
        let p_occupied = probabilities[1];
        if rng.random::<f64>() < p_occupied {
            OccupancyState::Occupied
        } else {
            OccupancyState::Vacant
        }
    }

    pub fn generate_day(&self, steps_per_hour: usize, rng: &mut impl Rng) -> Vec<OccupancyState> {
        let total_steps = 24 * steps_per_hour;
        let mut states = Vec::with_capacity(total_steps);
        let mut current = self.initial_state;
        states.push(current);
        for _ in 1..total_steps {
            current = self.next_state(current, rng);
            states.push(current);
        }
        states
    }

    pub fn validate_matrix(&self) -> Result<(), String> {
        if self.transition_matrix.len() != 2 {
            return Err("Transition matrix must have 2 rows".to_string());
        }
        for (i, row) in self.transition_matrix.iter().enumerate() {
            if row.len() != 2 {
                return Err(format!("Row {} must have 2 elements", i));
            }
            let sum = row.iter().sum::<f64>();
            if (sum - 1.0).abs() > 1e-10 {
                return Err(format!("Row {} sums to {}, not 1.0", i, sum));
            }
            for &p in row {
                if p < 0.0 || p > 1.0 {
                    return Err(format!("Probability {} out of [0,1] range", p));
                }
            }
        }
        Ok(())
    }
}

impl Default for MarkovOccupancyGenerator {
    fn default() -> Self {
        Self::from_ashrae90_1()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_ashrae90_1_matrix_properties() {
        let generator = MarkovOccupancyGenerator::from_ashrae90_1();
        assert!(generator.validate_matrix().is_ok());
        let matrix = &generator.transition_matrix;
        for row in matrix {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "Row sums to {}, expected 1.0", sum);
        }
    }

    #[test]
    fn test_custom_matrix_validation() {
        let valid_matrix = vec![vec![0.99, 0.01], vec![0.05, 0.95]];
        let generator = MarkovOccupancyGenerator::new(valid_matrix, OccupancyState::Vacant);
        assert!(generator.validate_matrix().is_ok());

        let invalid_matrix = vec![vec![0.99, 0.02]];
        let invalid_generator =
            MarkovOccupancyGenerator::new(invalid_matrix, OccupancyState::Vacant);
        assert!(invalid_generator.validate_matrix().is_err());
    }

    #[test]
    fn test_next_state_deterministic() {
        let matrix = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let generator = MarkovOccupancyGenerator::new(matrix, OccupancyState::Vacant);
        let mut rng = StdRng::seed_from_u64(42);
        assert_eq!(
            generator.next_state(OccupancyState::Vacant, &mut rng),
            OccupancyState::Vacant
        );
        assert_eq!(
            generator.next_state(OccupancyState::Occupied, &mut rng),
            OccupancyState::Occupied
        );
    }

    #[test]
    fn test_generate_day_length() {
        let generator = MarkovOccupancyGenerator::from_ashrae90_1();
        let mut rng = StdRng::seed_from_u64(42);
        let states = generator.generate_day(4, &mut rng);
        assert_eq!(states.len(), 24 * 4);
    }

    #[test]
    fn test_generate_day_stochastic() {
        let high_prob_matrix = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let generator = MarkovOccupancyGenerator::new(high_prob_matrix, OccupancyState::Vacant);
        let mut rng1 = StdRng::seed_from_u64(123);
        let mut rng2 = StdRng::seed_from_u64(456);
        let states1 = generator.generate_day(4, &mut rng1);
        let states2 = generator.generate_day(4, &mut rng2);
        assert_ne!(
            states1, states2,
            "Different seeds should produce different sequences"
        );
    }

    #[test]
    fn test_probability_bounds() {
        let matrix = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let generator = MarkovOccupancyGenerator::new(matrix, OccupancyState::Vacant);
        let mut rng = StdRng::seed_from_u64(42);
        let mut occupied_count = 0;
        for _ in 0..1000 {
            let next = generator.next_state(OccupancyState::Vacant, &mut rng);
            if matches!(next, OccupancyState::Occupied) {
                occupied_count += 1;
            }
        }
        let occupied_ratio = occupied_count as f64 / 1000.0;
        assert!(
            (occupied_ratio - 0.5).abs() < 0.1,
            "Occupied ratio {} far from expected 0.5",
            occupied_ratio
        );
    }
}
