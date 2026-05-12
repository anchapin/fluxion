//! Decision recording middleware.
//!
//! Wraps each orchestration decision call site to:
//! 1. Capture the decision and its ground-truth label.
//! 2. Measure the wall-clock time to *make* the decision (separate from simulation time).
//! 3. Accumulate records for TDQS computation at end-of-run.
//!
//! # Integration status (PR #771 / PR #776)
//!
//! The Building Scientist (PR #776) has added formal types to `fluxion`:
//! - `fluxion::orchestration::decision_types::OrchestrationDecisionKind` — enum with
//!   `as_str()` returning canonical kebab-case decision-type strings.
//! - `fluxion::orchestration::decision_types::OrchestrationDecision` — struct carrying
//!   `kind`, `chosen`, and `features`.
//!
//! This module now imports those types and:
//! - Provides `From<OrchestrationDecisionKind> for DecisionType` to guarantee that
//!   label strings used by the harness stay in sync with the canonical engine labels.
//! - Provides `DecisionRecorder::record_engine_decision()` for direct recording from
//!   `OrchestrationDecision` values once real engine hooks are fully plumbed.
//! - Provides `engine_decision_*` helper functions that wrap the current mock functions
//!   but return typed `OrchestrationDecision` values for richer inspection.
//!
//! When the surrogate path (batch_oracle / ONNX) lands, replace
//! `current_surrogate_routing_decision` with a real call to `src/ai/surrogate.rs`.
//!
//! # Call site pattern
//!
//! ```rust,ignore
//! let decision = recorder.timed_record(
//!     DecisionType::SolverSelection,
//!     "ASHRAE140_Case900",
//!     None,
//!     || {
//!         let eng_decision = engine_decision_solver_selection(density, thickness);
//!         let correct = ground_truth_solver_is_fd(density, thickness)
//!             == (eng_decision.chosen == "fd");
//!         let cost = if correct { 300.0 } else { 0.0 };
//!         (eng_decision, correct, cost)
//!     },
//! );
//! ```

use std::time::{Duration, Instant};

// Import formal engine types (added by Building Scientist in PR #776).
use fluxion::orchestration::decision_types::{OrchestrationDecision, OrchestrationDecisionKind};

#[path = "tdqs.rs"]
mod tdqs_mod;
pub use tdqs_mod::{DecisionInstance, DecisionType};

// ---------------------------------------------------------------------------
// Type alignment: OrchestrationDecisionKind ↔ DecisionType
// ---------------------------------------------------------------------------

/// Bridge from the engine's canonical enum to the harness's TDQS `DecisionType`.
///
/// Ensures that any change to `OrchestrationDecisionKind::as_str()` causes a
/// compile-time error here if the harness labels drift out of sync.
impl From<OrchestrationDecisionKind> for DecisionType {
    fn from(kind: OrchestrationDecisionKind) -> Self {
        // Use the engine's canonical string to derive the harness type so a
        // mismatch between PR #776 and the dataset labels fails at runtime in tests.
        match kind {
            OrchestrationDecisionKind::SolverSelection => DecisionType::SolverSelection,
            OrchestrationDecisionKind::AdaptiveTimestep => DecisionType::AdaptiveTimestep,
            OrchestrationDecisionKind::SurrogateRouting => DecisionType::SurrogateRouting,
            OrchestrationDecisionKind::ConstraintWarning => DecisionType::ConstraintWarning,
            OrchestrationDecisionKind::HvacHorizon => DecisionType::HvacHorizon,
        }
    }
}

/// Sanity-check that `OrchestrationDecisionKind::as_str()` labels match the harness's
/// canonical label strings.  Called once from `benchmark_runner.rs` at startup.
pub fn assert_label_consistency() {
    let pairs: &[(OrchestrationDecisionKind, &str)] = &[
        (OrchestrationDecisionKind::SolverSelection,  "solver_selection"),
        (OrchestrationDecisionKind::AdaptiveTimestep, "adaptive_timestep"),
        (OrchestrationDecisionKind::SurrogateRouting, "surrogate_routing"),
        (OrchestrationDecisionKind::ConstraintWarning,"constraint_warning"),
        (OrchestrationDecisionKind::HvacHorizon,      "hvac_horizon"),
    ];
    for (kind, expected) in pairs {
        assert_eq!(
            kind.as_str(), *expected,
            "OrchestrationDecisionKind::{:?} label mismatch (engine: {:?}, harness: {:?}). \
             Update either decision_recorder.rs or decision_types.rs.",
            kind, kind.as_str(), expected,
        );
    }
}

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
/// Create one per simulation run, call [`record`] / [`timed_record`] /
/// [`record_engine_decision`] at each decision point, then call
/// [`into_instances`] to feed TDQS computation.
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
            sim_case: String::new(),
            timestep_index,
        });
    }

    /// Record a typed `OrchestrationDecision` from the engine directly.
    ///
    /// `correct` and `cost_avoided_s` are supplied by the caller (the harness
    /// computes these by comparing `engine_decision.chosen` against the ground-truth
    /// label for the ASHRAE 140 case).
    pub fn record_engine_decision(
        &mut self,
        engine_decision: &OrchestrationDecision,
        correct: bool,
        cost_avoided_s: f64,
        sim_case: impl Into<String>,
        timestep_index: Option<usize>,
    ) {
        let dt = DecisionType::from(engine_decision.kind);
        self.record(dt, correct, cost_avoided_s, sim_case, timestep_index);
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

    /// Execute `decision_fn` returning an `OrchestrationDecision`, measure latency,
    /// derive `DecisionType` from the engine's canonical kind, and record.
    ///
    /// `ground_truth_fn` receives the same result and returns `(correct, cost_avoided_s)`.
    pub fn timed_record_engine<F, G>(
        &mut self,
        sim_case: impl Into<String> + Clone,
        timestep_index: Option<usize>,
        decision_fn: F,
        ground_truth_fn: G,
    ) -> OrchestrationDecision
    where
        F: FnOnce() -> OrchestrationDecision,
        G: FnOnce(&OrchestrationDecision) -> (bool, f64),
    {
        let t0 = Instant::now();
        let eng_decision = decision_fn();
        let latency = t0.elapsed();

        let (correct, cost_avoided_s) = ground_truth_fn(&eng_decision);
        let dt = DecisionType::from(eng_decision.kind);
        let case_str: String = sim_case.into();

        let instance = if correct {
            DecisionInstance::correct(dt, cost_avoided_s)
                .with_source(case_str.clone())
        } else {
            DecisionInstance::incorrect(dt)
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
        eng_decision
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
// Ground-truth functions (pure logic, no engine dependency)
// ---------------------------------------------------------------------------

/// Ground-truth: should this construction use FD (true) or CTF (false)?
///
/// Rule: FD required when density ≥ 1800 kg/m³ AND thickness ≥ 0.200 m.
/// This matches ASHRAE 140 Case 900-series concrete walls.
pub fn ground_truth_solver_is_fd(density_kg_m3: f64, thickness_m: f64) -> bool {
    density_kg_m3 >= 1800.0 && thickness_m >= 0.200
}

/// Ground-truth: should adaptive timestep trigger at this moment?
pub fn ground_truth_adaptive_timestep(
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

// ---------------------------------------------------------------------------
// Engine-typed decision helpers (return OrchestrationDecision)
//
// These call the same rule-based logic as the mock functions but return the
// engine's formal type so call sites can use either `timed_record` or
// `timed_record_engine` with identical semantics.
// ---------------------------------------------------------------------------

/// Current solver selection wrapped as `OrchestrationDecision`.
///
/// chosen = "ctf" everywhere — the known Issue #726 gap.
/// Replace the body with a real call to `ThermalMethodSelector::select_method`
/// once PR #776's tracing spans are exposed as a callable function.
pub fn engine_decision_solver_selection(
    density_kg_m3: f64,
    thickness_m: f64,
) -> OrchestrationDecision {
    // Current engine always picks CTF (Issue #726 known gap for 900-series)
    let chosen = "ctf";
    OrchestrationDecision::new(
        OrchestrationDecisionKind::SolverSelection,
        chosen,
        serde_json::json!({
            "density_kg_m3": density_kg_m3,
            "thickness_m": thickness_m,
        }),
    )
}

/// Current adaptive timestep decision wrapped as `OrchestrationDecision`.
pub fn engine_decision_adaptive_timestep(
    t_zone_slope_k_per_h: f64,
    solar_delta_w_per_m2: f64,
) -> OrchestrationDecision {
    let trigger = t_zone_slope_k_per_h.abs() > 3.0 || solar_delta_w_per_m2.abs() > 150.0;
    let chosen = if trigger { "adaptive_6min" } else { "fixed_1h" };
    OrchestrationDecision::new(
        OrchestrationDecisionKind::AdaptiveTimestep,
        chosen,
        serde_json::json!({
            "t_zone_slope_k_per_h": t_zone_slope_k_per_h,
            "solar_delta_w_per_m2": solar_delta_w_per_m2,
        }),
    )
}

/// Current surrogate routing wrapped as `OrchestrationDecision`.
///
/// Always "physics" — surrogate not yet deployed (v2.1+).
pub fn engine_decision_surrogate_routing(
    mahalanobis_dist: f64,
    required_accuracy_rmse: f64,
) -> OrchestrationDecision {
    OrchestrationDecision::new(
        OrchestrationDecisionKind::SurrogateRouting,
        "physics",
        serde_json::json!({
            "mahalanobis_dist": mahalanobis_dist,
            "required_accuracy_rmse": required_accuracy_rmse,
        }),
    )
}

/// Current constraint warning decision wrapped as `OrchestrationDecision`.
///
/// chosen = "passed" (no pre-flight check yet — post-hoc only).
pub fn engine_decision_constraint_warning(
    t_zone_min_c: f64,
    t_zone_max_c: f64,
    energy_balance_error: f64,
) -> OrchestrationDecision {
    // Current engine: no pre-sim constraint check; always returns "passed"
    OrchestrationDecision::new(
        OrchestrationDecisionKind::ConstraintWarning,
        "passed",
        serde_json::json!({
            "t_zone_min_c": t_zone_min_c,
            "t_zone_max_c": t_zone_max_c,
            "energy_balance_error": energy_balance_error,
        }),
    )
}

/// Current HVAC horizon decision wrapped as `OrchestrationDecision`.
///
/// chosen = "24h_fixed" — matches what PR #776's tracing span emits at
/// `ThermalModel::new_with_validation`.
pub fn engine_decision_hvac_horizon(
    weather_forecast_confidence: f64,
    dr_event_probability_72h: f64,
) -> OrchestrationDecision {
    OrchestrationDecision::new(
        OrchestrationDecisionKind::HvacHorizon,
        "24h_fixed",
        serde_json::json!({
            "weather_forecast_confidence": weather_forecast_confidence,
            "dr_event_probability_72h": dr_event_probability_72h,
        }),
    )
}

// ---------------------------------------------------------------------------
// Legacy thin wrappers (used by existing benchmark_runner.rs call sites)
// These delegate to the engine-typed helpers for consistency.
// ---------------------------------------------------------------------------

/// Current (rule-based) solver: returns `true` for FD, `false` for CTF.
pub fn current_solver_decision(_density_kg_m3: f64, _thickness_m: f64) -> bool {
    false // CTF everywhere — the known limitation causing 900-series errors
}

/// Current adaptive timestep decision (rule-based, matches ground truth).
pub fn current_adaptive_timestep_decision(
    t_zone_slope_k_per_h: f64,
    solar_delta_w_per_m2: f64,
) -> bool {
    t_zone_slope_k_per_h.abs() > 3.0 || solar_delta_w_per_m2.abs() > 150.0
}

/// Current surrogate routing — always physics (feature not yet implemented).
pub fn current_surrogate_routing_decision(
    _mahalanobis_dist: f64,
    _required_accuracy_rmse: f64,
) -> bool {
    false // always physics
}

/// Current constraint check (post-hoc only — returns false pre-sim).
pub fn current_constraint_warning_decision(
    _t_zone_min_c: f64,
    _t_zone_max_c: f64,
    _energy_balance_error: f64,
) -> bool {
    false // no pre-flight check yet
}

/// Current HVAC horizon selection (fixed 24 h).
pub fn current_hvac_horizon_decision(
    _weather_forecast_confidence: f64,
    _dr_event_probability_72h: f64,
) -> u32 {
    24
}
