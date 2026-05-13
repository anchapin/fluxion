//! Temporal Decision Quality Score (TDQS) — Formal Implementation
//!
//! # Formula
//!
//! ```text
//! TDQS = Σᵢ [ correct(dᵢ) × w(dᵢ) × cost_avoided(dᵢ) ]
//!        ───────────────────────────────────────────────
//!        Σᵢ [ w(dᵢ) × cost_available(dᵢ) ]
//! ```
//!
//! Range: TDQS ∈ [0.0, 1.0]
//!
//! | TDQS  | Interpretation                                 |
//! |-------|------------------------------------------------|
//! | 1.0   | All decisions correct, maximum savings captured |
//! | ~0.75 | Expected rule-based system baseline            |
//! | 0.5   | Chance performance                             |
//! | < 0.5 | Systematic bias present                        |
//!
//! # Decision Type Weights
//!
//! | Type               | Weight | Max cost saved (s) |
//! |--------------------|--------|--------------------|
//! | Solver selection   | 3.0    | 300 (5 min FD run) |
//! | Adaptive timestep  | 1.5    | 45                 |
//! | Surrogate routing  | 2.0    | 2 per query        |
//! | Constraint warning | 1.0    | 30                 |
//! | HVAC horizon       | 1.5    | 10                 |
//!
//! # Building Scientist integration note
//!
//! Decision types are currently implicit in the Rust engine.
//! Once `zgmkhirakpnot032` adds the formal `OrchestrationDecisionKind` enum and
//! `OrchestrationDecision` trait to `src/orchestration/`, the mock implementations
//! in `benchmark_runner.rs` will be replaced with real hooks. The TDQS formula
//! and serialization in this file are engine-agnostic.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Decision type catalogue
// ---------------------------------------------------------------------------

/// The five formal orchestration decision types in the fluxion simulation engine.
///
/// NOTE: The Rust engine currently makes these decisions implicitly.  A formal
/// `OrchestrationDecisionKind` enum is being added to `src/orchestration/` by
/// the Building Scientist so each decision site can emit structured records
/// compatible with this harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecisionType {
    /// Choose CTF or Finite Difference solver for a given construction layer.
    /// High-mass (concrete ≥ 200 mm): FD required.  Lightweight: CTF.
    SolverSelection,
    /// Trigger adaptive timestep reduction (1 h → 6 min) on rapid transients.
    AdaptiveTimestep,
    /// Route query to BatchOracle surrogate or full physics solver.
    /// (Not yet wired — v2.1+ feature; stub present for baseline capture.)
    SurrogateRouting,
    /// Predict a constraint violation before the simulation runs.
    ConstraintWarning,
    /// Select HVAC optimal-control horizon: 6 h / 24 h / 48 h / 72 h.
    HvacHorizon,
}

impl DecisionType {
    /// All decision types in canonical order.
    pub const ALL: [Self; 5] = [
        Self::SolverSelection,
        Self::AdaptiveTimestep,
        Self::SurrogateRouting,
        Self::ConstraintWarning,
        Self::HvacHorizon,
    ];

    /// Importance weight for this decision type (1.0 – 3.0).
    pub const fn weight(self) -> f64 {
        match self {
            Self::SolverSelection   => 3.0,
            Self::AdaptiveTimestep  => 1.5,
            Self::SurrogateRouting  => 2.0,
            Self::ConstraintWarning => 1.0,
            Self::HvacHorizon       => 1.5,
        }
    }

    /// Maximum compute cost saved (seconds) per perfectly-correct decision.
    /// Used as `cost_available` in the TDQS denominator.
    pub const fn max_cost_available_s(self) -> f64 {
        match self {
            Self::SolverSelection   => 300.0, // 5 min FD run avoided on lightweight case
            Self::AdaptiveTimestep  => 45.0,  // fine-timestep step avoided on stable period
            Self::SurrogateRouting  => 2.0,   // physics solver avoided per query
            Self::ConstraintWarning => 30.0,  // failed simulation avoided
            Self::HvacHorizon       => 10.0,  // ~1 kWh energy delta mapped to ≈10 s equivalent
        }
    }

    /// Short string key used in JSON serialisation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SolverSelection   => "solver_selection",
            Self::AdaptiveTimestep  => "adaptive_timestep",
            Self::SurrogateRouting  => "surrogate_routing",
            Self::ConstraintWarning => "constraint_warning",
            Self::HvacHorizon       => "hvac_horizon",
        }
    }
}

impl fmt::Display for DecisionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Decision instance
// ---------------------------------------------------------------------------

/// A single labeled decision instance for TDQS computation.
#[derive(Debug, Clone)]
pub struct DecisionInstance {
    /// Which of the 5 decision types this is.
    pub decision_type: DecisionType,
    /// Whether the system's decision matched the ground truth label.
    pub correct: bool,
    /// Compute time actually avoided by this decision (0.0 when `correct = false`).
    pub cost_avoided_s: f64,
    /// Source simulation case identifier (e.g. `"ASHRAE140_Case900"`).
    pub source_case: Option<String>,
    /// Timestep index within the simulation (if applicable).
    pub timestep_index: Option<usize>,
}

impl DecisionInstance {
    /// Correct decision with the given cost savings realised.
    pub fn correct(decision_type: DecisionType, cost_avoided_s: f64) -> Self {
        Self {
            decision_type,
            correct: true,
            cost_avoided_s,
            source_case: None,
            timestep_index: None,
        }
    }

    /// Incorrect decision — no cost savings realised.
    pub fn incorrect(decision_type: DecisionType) -> Self {
        Self {
            decision_type,
            correct: false,
            cost_avoided_s: 0.0,
            source_case: None,
            timestep_index: None,
        }
    }

    /// Builder: attach a source case label.
    pub fn with_source(mut self, case: impl Into<String>) -> Self {
        self.source_case = Some(case.into());
        self
    }

    /// Builder: attach a timestep index.
    pub fn with_timestep(mut self, idx: usize) -> Self {
        self.timestep_index = Some(idx);
        self
    }

    // --- TDQS formula contributions ------------------------------------------

    /// Numerator contribution: `correct(d) × w(d) × cost_avoided(d)`
    #[inline]
    pub fn numerator_contribution(&self) -> f64 {
        if self.correct {
            self.decision_type.weight() * self.cost_avoided_s
        } else {
            0.0
        }
    }

    /// Denominator contribution: `w(d) × cost_available(d)`
    #[inline]
    pub fn denominator_contribution(&self) -> f64 {
        self.decision_type.weight() * self.decision_type.max_cost_available_s()
    }
}

// ---------------------------------------------------------------------------
// TDQS computation
// ---------------------------------------------------------------------------

/// Compute the Temporal Decision Quality Score over a slice of decision instances.
///
/// Returns `0.0` for an empty dataset or a zero-denominator (all weights × costs = 0).
pub fn compute_tdqs(decisions: &[DecisionInstance]) -> f64 {
    if decisions.is_empty() {
        return 0.0;
    }
    let numerator: f64 = decisions.iter().map(|d| d.numerator_contribution()).sum();
    let denominator: f64 = decisions.iter().map(|d| d.denominator_contribution()).sum();
    if denominator == 0.0 {
        return 0.0;
    }
    (numerator / denominator).clamp(0.0, 1.0)
}

/// Per-decision-type breakdown produced by [`compute_tdqs_breakdown`].
#[derive(Debug, Clone)]
pub struct TdqsBreakdown {
    /// Overall TDQS across all decision types.
    pub overall: f64,
    /// Per-type: `(type, tdqs, n_correct, n_total)`.
    pub per_type: Vec<(DecisionType, f64, usize, usize)>,
}

impl TdqsBreakdown {
    /// Returns the TDQS for a specific decision type, or `None` if no decisions
    /// of that type appear in the dataset.
    pub fn tdqs_for(&self, dt: DecisionType) -> Option<f64> {
        self.per_type
            .iter()
            .find(|(t, _, _, _)| *t == dt)
            .map(|(_, score, _, _)| *score)
    }

    /// Accuracy (fraction correct) for a specific decision type.
    pub fn accuracy_for(&self, dt: DecisionType) -> Option<f64> {
        self.per_type
            .iter()
            .find(|(t, _, _, _)| *t == dt)
            .and_then(|(_, _, correct, total)| {
                if *total == 0 {
                    None
                } else {
                    Some(*correct as f64 / *total as f64)
                }
            })
    }
}

/// Compute TDQS with a full per-decision-type breakdown.
pub fn compute_tdqs_breakdown(decisions: &[DecisionInstance]) -> TdqsBreakdown {
    let overall = compute_tdqs(decisions);

    let per_type = DecisionType::ALL
        .iter()
        .map(|&dt| {
            let subset: Vec<_> = decisions
                .iter()
                .filter(|d| d.decision_type == dt)
                .cloned()
                .collect();
            let n_correct = subset.iter().filter(|d| d.correct).count();
            let n_total = subset.len();
            let score = compute_tdqs(&subset);
            (dt, score, n_correct, n_total)
        })
        .collect();

    TdqsBreakdown { overall, per_type }
}

/// Compute per-case TDQS scores, keyed by `source_case`.
pub fn compute_tdqs_per_case(
    decisions: &[DecisionInstance],
) -> HashMap<String, f64> {
    let mut case_decisions: HashMap<String, Vec<DecisionInstance>> = HashMap::new();

    for d in decisions {
        let key = d
            .source_case
            .clone()
            .unwrap_or_else(|| "unknown".into());
        case_decisions.entry(key).or_default().push(d.clone());
    }

    case_decisions
        .into_iter()
        .map(|(case, decisions)| (case, compute_tdqs(&decisions)))
        .collect()
}

// ---------------------------------------------------------------------------
// Regression gate
// ---------------------------------------------------------------------------

/// Returns `true` if the new TDQS represents a regression vs. `baseline_tdqs`.
///
/// A regression is defined as: `baseline - new > threshold`.
/// For CI: `regression_detected(new, baseline, 0.05)` → fail if TDQS drops > 5 pp.
pub fn regression_detected(new_tdqs: f64, baseline_tdqs: f64, threshold: f64) -> bool {
    baseline_tdqs - new_tdqs > threshold
}

/// Per-type regression check.  Returns decision types that regressed.
pub fn regression_by_type(
    new: &TdqsBreakdown,
    baseline: &TdqsBreakdown,
    threshold: f64,
) -> Vec<DecisionType> {
    DecisionType::ALL
        .iter()
        .filter(|&&dt| {
            let new_score = new.tdqs_for(dt).unwrap_or(0.0);
            let base_score = baseline.tdqs_for(dt).unwrap_or(0.0);
            regression_detected(new_score, base_score, threshold)
        })
        .copied()
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn all_correct_full_savings() -> Vec<DecisionInstance> {
        DecisionType::ALL
            .iter()
            .map(|&dt| DecisionInstance::correct(dt, dt.max_cost_available_s()))
            .collect()
    }

    #[test]
    fn tdqs_all_correct_is_one() {
        let d = all_correct_full_savings();
        let score = compute_tdqs(&d);
        assert!((score - 1.0).abs() < 1e-10, "Expected 1.0, got {score}");
    }

    #[test]
    fn tdqs_all_incorrect_is_zero() {
        let d: Vec<_> = DecisionType::ALL
            .iter()
            .map(|&dt| DecisionInstance::incorrect(dt))
            .collect();
        assert_eq!(compute_tdqs(&d), 0.0);
    }

    #[test]
    fn tdqs_empty_is_zero() {
        assert_eq!(compute_tdqs(&[]), 0.0);
    }

    #[test]
    fn tdqs_in_range_for_partial_dataset() {
        let d = vec![
            DecisionInstance::correct(DecisionType::SolverSelection, 150.0),
            DecisionInstance::incorrect(DecisionType::AdaptiveTimestep),
            DecisionInstance::correct(DecisionType::SurrogateRouting, 1.5),
        ];
        let score = compute_tdqs(&d);
        assert!((0.0..=1.0).contains(&score), "Out of range: {score}");
    }

    #[test]
    fn regression_gate_fires_above_threshold() {
        assert!(regression_detected(0.65, 0.72, 0.05));  // 0.07 drop > 0.05
        assert!(!regression_detected(0.68, 0.72, 0.05)); // 0.04 drop < 0.05
        assert!(!regression_detected(0.72, 0.72, 0.05)); // no change
    }

    #[test]
    fn weight_table_matches_spec() {
        assert_eq!(DecisionType::SolverSelection.weight(), 3.0);
        assert_eq!(DecisionType::SurrogateRouting.weight(), 2.0);
        assert_eq!(DecisionType::AdaptiveTimestep.weight(), 1.5);
        assert_eq!(DecisionType::HvacHorizon.weight(), 1.5);
        assert_eq!(DecisionType::ConstraintWarning.weight(), 1.0);
    }

    #[test]
    fn breakdown_per_type_accuracy() {
        let d = vec![
            DecisionInstance::correct(DecisionType::SolverSelection, 300.0).with_source("case_600"),
            DecisionInstance::correct(DecisionType::SolverSelection, 300.0).with_source("case_610"),
            DecisionInstance::incorrect(DecisionType::SolverSelection).with_source("case_900"),
        ];
        let b = compute_tdqs_breakdown(&d);
        let acc = b.accuracy_for(DecisionType::SolverSelection).unwrap();
        assert!((acc - 2.0 / 3.0).abs() < 1e-10, "Accuracy: {acc}");
    }

    #[test]
    fn per_case_tdqs() {
        let d = vec![
            DecisionInstance::correct(DecisionType::SolverSelection, 300.0)
                .with_source("case_600"),
            DecisionInstance::incorrect(DecisionType::SolverSelection)
                .with_source("case_900"),
        ];
        let per_case = compute_tdqs_per_case(&d);
        assert_eq!(per_case["case_600"], 1.0);
        assert_eq!(per_case["case_900"], 0.0);
    }
}
