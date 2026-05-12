//! Decision recording middleware.
//!
//! Wraps each orchestration decision call site to:
//! 1. Capture the decision and its ground-truth label.
//! 2. Measure the wall-clock time to *make* the decision (separate from simulation time).
//! 3. Accumulate records for TDQS computation at end-of-run.
//!
//! # Integration plan
//!
//! Once the Building Scientist adds `OrchestrationDecisionKind` to the Rust
//! engine (`src/orchestration/`), call sites will look like:
//!
//! ```rust,ignore
//! let solver = recorder.timed_record(
//!     DecisionType::SolverSelection,
//!     "ASHRAE140_Case900",
//!     None,
//!     || {
//!         let solver = engine.select_solver(&construction);
//!         let correct = ground_truth::is_solver_correct(&construction, solver);
//!         let cost = if !correct { 0.0 } else { 300.0 };
//!         (solver, correct, cost)
//!     },
//! );
//! ```

use std::time::{Duration, Instant};

#[path = "tdqs.rs"]
mod tdqs_mod;
pub use tdqs_mod::{DecisionInstance, DecisionType};

// ---------------------------------------------------------------------------
// Recorded decision
// ---------------------------------------------------------------------------

/// A single decision captured during a simulation run, including timing.
#[derive(Debug, Clone)]
pub struct RecordedDecision {
    /// The TDQS-compatible decision instance.
    pub instance: DecisionInstance,
    /// Wall-clock time to execute the decision logic itself.
    pub decision_latency: Duration,
    /// Source simulation case identifier.
    pub sim_case: String,
    /// Timestep index within the simulation (if applicable).
    pub timestep_index: Option<usize>,
}

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------

/// Accumulates orchestration decisions during a simulation run or replay.
///
/// Create one per simulation run, call [`record`] / [`timed_record`] at each
/// decision point, then call [`into_instances`] to feed TDQS computation.
#[derive(Debug, Default)]
pub struct DecisionRecorder {
    records: Vec<RecordedDecision>,
}

impl DecisionRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a decision that was made externally (latency not measured here).
    pub fn record(
        &mut self,
        decision_type: DecisionType,
        correct: bool,
        cost_avoided_s: f64,
        sim_case: impl Into<String>,
        timestep_index: Option<usize>,
    ) {
        let instance = if correct {
            DecisionInstance::correct(decision_type, cost_avoided_s)
                .with_source(sim_case.into())
        } else {
            DecisionInstance::incorrect(decision_type)
                .with_source(sim_case.into())
        };
        self.records.push(RecordedDecision {
            instance,
            decision_latency: Duration::ZERO,
            sim_case: String::new(), // already in instance.source_case
            timestep_index,
        });
    }

    /// Execute `decision_fn`, measure its wall-clock latency, and record the result.
    ///
    /// `decision_fn` must return `(result, correct, cost_avoided_s)`.
    pub fn timed_record<F, R>(
        &mut self,
        decision_type: DecisionType,
        sim_case: impl Into<String> + Clone,
        timestep_index: Option<usize>,
        decision_fn: F,
    ) -> R
    where
        F: FnOnce() -> (R, bool, f64),
    {
        let t0 = Instant::now();
        let (result, correct, cost_avoided_s) = decision_fn();
        let latency = t0.elapsed();

        let case_str: String = sim_case.into();
        let instance = if correct {
            DecisionInstance::correct(decision_type, cost_avoided_s)
                .with_source(case_str.clone())
        } else {
            DecisionInstance::incorrect(decision_type)
                .with_source(case_str.clone())
        };
        let instance = if let Some(ts) = timestep_index {
            instance.with_timestep(ts)
        } else {
            instance
        };

        self.records.push(RecordedDecision {
            instance,
            decision_latency: latency,
            sim_case: case_str,
            timestep_index,
        });
        result
    }

    // --- Accessors -----------------------------------------------------------

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Consume the recorder and return all `DecisionInstance`s.
    pub fn into_instances(self) -> Vec<DecisionInstance> {
        self.records.into_iter().map(|r| r.instance).collect()
    }

    /// Borrow all instances without consuming.
    pub fn instances(&self) -> Vec<DecisionInstance> {
        self.records.iter().map(|r| r.instance.clone()).collect()
    }

    /// All records (with latency data).
    pub fn records(&self) -> &[RecordedDecision] {
        &self.records
    }

    /// Average decision latency across all records.
    pub fn avg_decision_latency(&self) -> Duration {
        if self.records.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.records.iter().map(|r| r.decision_latency).sum();
        total / self.records.len() as u32
    }

    /// Average decision latency for a specific type.
    pub fn avg_latency_for(&self, dt: DecisionType) -> Duration {
        let relevant: Vec<_> = self
            .records
            .iter()
            .filter(|r| r.instance.decision_type == dt)
            .collect();
        if relevant.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = relevant.iter().map(|r| r.decision_latency).sum();
        total / relevant.len() as u32
    }
}

// ---------------------------------------------------------------------------
// Mock decision functions (replace with real engine calls once #726 / v2.1 land)
// ---------------------------------------------------------------------------

/// Ground-truth: should this construction use FD (true) or CTF (false)?
///
/// Rule: FD required when density ≥ 1800 kg/m³ AND thickness ≥ 0.200 m.
/// This matches ASHRAE 140 Case 900-series concrete walls.
pub fn ground_truth_solver_is_fd(density_kg_m3: f64, thickness_m: f64) -> bool {
    density_kg_m3 >= 1800.0 && thickness_m >= 0.200
}

/// Current (rule-based) solver selection — all constructions use CTF.
/// Returns `true` for FD, `false` for CTF.
pub fn current_solver_decision(_density_kg_m3: f64, _thickness_m: f64) -> bool {
    false // CTF everywhere — the known limitation causing 900-series errors
}

/// Ground-truth: should adaptive timestep trigger at this moment?
pub fn ground_truth_adaptive_timestep(
    t_zone_slope_k_per_h: f64,
    solar_delta_w_per_m2: f64,
) -> bool {
    t_zone_slope_k_per_h.abs() > 3.0 || solar_delta_w_per_m2.abs() > 150.0
}

/// Current adaptive timestep decision (rule-based, same as ground truth for now).
pub fn current_adaptive_timestep_decision(
    t_zone_slope_k_per_h: f64,
    solar_delta_w_per_m2: f64,
) -> bool {
    t_zone_slope_k_per_h.abs() > 3.0 || solar_delta_w_per_m2.abs() > 150.0
}

/// Ground-truth: should this query go to surrogate (true) or physics (false)?
/// Currently always false (surrogate not yet deployed).
pub fn ground_truth_surrogate_routing(
    mahalanobis_dist: f64,
    _required_accuracy_rmse: f64,
) -> bool {
    mahalanobis_dist < 2.0 // within training distribution
}

/// Current surrogate routing — always physics (feature not yet implemented).
pub fn current_surrogate_routing_decision(
    _mahalanobis_dist: f64,
    _required_accuracy_rmse: f64,
) -> bool {
    false // always physics
}

/// Ground-truth: should a constraint violation warning be raised?
pub fn ground_truth_constraint_warning(
    t_zone_min_c: f64,
    t_zone_max_c: f64,
    energy_balance_error: f64,
) -> bool {
    t_zone_min_c < -50.0
        || t_zone_max_c > 100.0
        || energy_balance_error > 0.01
}

/// Current constraint check (post-hoc only — returns false pre-sim).
pub fn current_constraint_warning_decision(
    _t_zone_min_c: f64,
    _t_zone_max_c: f64,
    _energy_balance_error: f64,
) -> bool {
    false // no pre-flight check yet
}

/// Ground-truth: optimal HVAC horizon in hours.
pub fn ground_truth_hvac_horizon(
    weather_forecast_confidence: f64,
    dr_event_probability_72h: f64,
) -> u32 {
    if dr_event_probability_72h > 0.5 {
        6
    } else if weather_forecast_confidence > 0.70 {
        72
    } else {
        24
    }
}

/// Current HVAC horizon selection (fixed 24 h).
pub fn current_hvac_horizon_decision(
    _weather_forecast_confidence: f64,
    _dr_event_probability_72h: f64,
) -> u32 {
    24
}
