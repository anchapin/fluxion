//! TOPSIS multi-criteria decision making for Pareto optimization.
//!
//! TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) selects
//! optimal designs from Pareto frontiers by calculating geometric distance from
//! ideal positive/negative solutions.
//!
//! # References
//! - Hwang & Yoon (1981) "Multiple Attribute Decision Making"
//! - Research: "Advancements in Building Energy Simulation Engines" - SPEA-2/TOPSIS section

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ParetoPoint {
    pub id: usize,
    pub ec: f64,
    pub tdhp: f64,
    pub lcc: f64,
    pub lcco2: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TopsisWeights {
    pub ec: f64,
    pub tdhp: f64,
    pub lcc: f64,
    pub lcco2: f64,
}

impl Default for TopsisWeights {
    fn default() -> Self {
        Self {
            ec: 0.30,
            tdhp: 0.20,
            lcc: 0.25,
            lcco2: 0.25,
        }
    }
}

impl TopsisWeights {
    pub fn new(ec: f64, tdhp: f64, lcc: f64, lcco2: f64) -> Self {
        Self {
            ec,
            tdhp,
            lcc,
            lcco2,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        let sum = self.ec + self.tdhp + self.lcc + self.lcco2;
        if (sum - 1.0).abs() > 1e-6 {
            return Err("Weights must sum to 1.0");
        }
        if self.ec < 0.0 || self.tdhp < 0.0 || self.lcc < 0.0 || self.lcco2 < 0.0 {
            return Err("All weights must be non-negative");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TopsisResult {
    pub selected_index: usize,
    pub selected_point: ParetoPoint,
    pub closeness_scores: BTreeMap<usize, f64>,
    pub ideal_positive: [f64; 4],
    pub ideal_negative: [f64; 4],
}

pub struct Topsis;

impl Topsis {
    pub fn select_optimal(
        pareto_front: &[ParetoPoint],
        weights: &TopsisWeights,
    ) -> Result<TopsisResult, &'static str> {
        if pareto_front.is_empty() {
            return Err("Pareto front is empty");
        }
        weights.validate()?;

        let criteria: Vec<[f64; 4]> = pareto_front
            .iter()
            .map(|p| [p.ec, p.tdhp, p.lcc, p.lcco2])
            .collect();

        let (normalized, ideal_pos, ideal_neg) = Self::normalize_and_find_ideals(&criteria);

        let weighted = Self::apply_weights(&normalized, weights);

        let distances = Self::calculate_distances(&weighted, &ideal_pos, &ideal_neg);

        let closeness_scores: BTreeMap<usize, f64> = distances
            .iter()
            .enumerate()
            .map(|(i, &(d_pos, d_neg))| {
                let closeness = d_neg / (d_pos + d_neg);
                (pareto_front[i].id, closeness)
            })
            .collect();

        let selected_id = *closeness_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(id, _)| id)
            .unwrap();

        let selected_idx = pareto_front
            .iter()
            .position(|p| p.id == selected_id)
            .unwrap();

        Ok(TopsisResult {
            selected_index: selected_idx,
            selected_point: pareto_front[selected_idx],
            closeness_scores,
            ideal_positive: ideal_pos,
            ideal_negative: ideal_neg,
        })
    }

    fn normalize_and_find_ideals(criteria: &[[f64; 4]]) -> (Vec<[f64; 4]>, [f64; 4], [f64; 4]) {
        if criteria.is_empty() {
            return (Vec::new(), [0.0; 4], [0.0; 4]);
        }

        let sum_squares: [f64; 4] = [
            criteria.iter().map(|c| c[0].powi(2)).sum(),
            criteria.iter().map(|c| c[1].powi(2)).sum(),
            criteria.iter().map(|c| c[2].powi(2)).sum(),
            criteria.iter().map(|c| c[3].powi(2)).sum(),
        ];

        let sqrt_sums = [
            sum_squares[0].sqrt(),
            sum_squares[1].sqrt(),
            sum_squares[2].sqrt(),
            sum_squares[3].sqrt(),
        ];

        let normalized: Vec<[f64; 4]> = criteria
            .iter()
            .map(|c| {
                [
                    c[0] / sqrt_sums[0],
                    c[1] / sqrt_sums[1],
                    c[2] / sqrt_sums[2],
                    c[3] / sqrt_sums[3],
                ]
            })
            .collect();

        let ideal_pos = if criteria.len() > 1 {
            [
                normalized
                    .iter()
                    .map(|c| c[0])
                    .fold(f64::INFINITY, f64::min),
                normalized
                    .iter()
                    .map(|c| c[1])
                    .fold(f64::INFINITY, f64::min),
                normalized
                    .iter()
                    .map(|c| c[2])
                    .fold(f64::INFINITY, f64::min),
                normalized
                    .iter()
                    .map(|c| c[3])
                    .fold(f64::INFINITY, f64::min),
            ]
        } else {
            normalized[0]
        };

        let ideal_neg = if criteria.len() > 1 {
            [
                normalized
                    .iter()
                    .map(|c| c[0])
                    .fold(f64::NEG_INFINITY, f64::max),
                normalized
                    .iter()
                    .map(|c| c[1])
                    .fold(f64::NEG_INFINITY, f64::max),
                normalized
                    .iter()
                    .map(|c| c[2])
                    .fold(f64::NEG_INFINITY, f64::max),
                normalized
                    .iter()
                    .map(|c| c[3])
                    .fold(f64::NEG_INFINITY, f64::max),
            ]
        } else {
            normalized[0]
        };

        (normalized, ideal_pos, ideal_neg)
    }

    fn apply_weights(normalized: &[[f64; 4]], weights: &TopsisWeights) -> Vec<[f64; 4]> {
        normalized
            .iter()
            .map(|c| {
                [
                    c[0] * weights.ec,
                    c[1] * weights.tdhp,
                    c[2] * weights.lcc,
                    c[3] * weights.lcco2,
                ]
            })
            .collect()
    }

    fn calculate_distances(
        weighted: &[[f64; 4]],
        ideal_pos: &[f64; 4],
        ideal_neg: &[f64; 4],
    ) -> Vec<(f64, f64)> {
        weighted
            .iter()
            .map(|c| {
                let d_pos = ((c[0] - ideal_pos[0]).powi(2)
                    + (c[1] - ideal_pos[1]).powi(2)
                    + (c[2] - ideal_pos[2]).powi(2)
                    + (c[3] - ideal_pos[3]).powi(2))
                .sqrt();
                let d_neg = ((c[0] - ideal_neg[0]).powi(2)
                    + (c[1] - ideal_neg[1]).powi(2)
                    + (c[2] - ideal_neg[2]).powi(2)
                    + (c[3] - ideal_neg[3]).powi(2))
                .sqrt();
                (d_pos, d_neg)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topsis_basic() {
        let pareto_front = vec![
            ParetoPoint {
                id: 0,
                ec: 50.0,
                tdhp: 5.0,
                lcc: 100.0,
                lcco2: 20.0,
            },
            ParetoPoint {
                id: 1,
                ec: 30.0,
                tdhp: 8.0,
                lcc: 150.0,
                lcco2: 15.0,
            },
            ParetoPoint {
                id: 2,
                ec: 40.0,
                tdhp: 3.0,
                lcc: 120.0,
                lcco2: 25.0,
            },
        ];

        let weights = TopsisWeights::default();
        let result = Topsis::select_optimal(&pareto_front, &weights).unwrap();

        assert!(result
            .closeness_scores
            .values()
            .all(|&c| c >= 0.0 && c <= 1.0));
        assert!(result.selected_point.id <= 2);
    }

    #[test]
    fn test_weights_validation() {
        let valid = TopsisWeights::new(0.3, 0.2, 0.25, 0.25);
        assert!(valid.validate().is_ok());

        let invalid_sum = TopsisWeights::new(0.5, 0.5, 0.5, 0.5);
        assert!(invalid_sum.validate().is_err());

        let negative = TopsisWeights::new(-0.1, 0.4, 0.4, 0.3);
        assert!(negative.validate().is_err());
    }

    #[test]
    fn test_empty_pareto_front() {
        let empty: Vec<ParetoPoint> = vec![];
        let weights = TopsisWeights::default();
        let result = Topsis::select_optimal(&empty, &weights);
        assert!(result.is_err());
    }
}
